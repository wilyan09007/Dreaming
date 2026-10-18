"""The `hdc` candidate offline: fit on rows labelled by a keyword rule, load, read back, refuse a foreign encoder."""

from __future__ import annotations

import pytest
from conftest import LABELS, CharEncoder
from dreaming.hashing import sha_bytes
from dreaming.predictors import profile_sha
from dreaming.predictors.candidates import CANDIDATES, for_site
from dreaming.predictors.hdc import HDC, D
from test_memory import SITE, make_rows


def test_hdc_bundles_its_rows_byte_stably_and_refuses_a_foreign_encoder():
    rows, enc = make_rows(80), CharEncoder()
    assert "hdc" not in [c.name for c in for_site(SITE.model_copy(update={"labels": LABELS}))]  # opt-in only
    train, hold = rows[:60], rows[60:]

    f = CANDIDATES["hdc"].fit(SITE, train, enc)
    assert f.n_train == 60 and f.holdout_accuracy >= 0.9 and f.weights_sha == sha_bytes(f.data)
    assert CANDIDATES["hdc"].fit(SITE, train[::-1], enc).data == f.data  # row order cannot change the bytes
    m = CANDIDATES["hdc"].load(SITE, f.data, enc)
    assert isinstance(m, HDC) and m.weights_sha == f.weights_sha
    assert m.env_sha == profile_sha(encoder=enc.env_sha, candidate="hdc", dim=D)

    preds = [m.predict(str(r.request)) for r in hold]
    assert sum(p.label == r.teacher.label for p, r in zip(preds, hold, strict=True) if r.teacher) >= 18  # >= 90% of 20
    again = [m.predict(str(r.request)) for r in hold]
    assert preds == again == [m.predict(str(r.request), fresh=True) for r in hold]  # same bytes every time
    assert all(p.scores["Other"] == 0.0 and p.label != "Other" and 0.0 <= p.confidence <= 1.0 for p in preds)

    other = CharEncoder()
    other.model_sha = "charenc-v2"
    with pytest.raises(ValueError, match="charenc-v1.*charenc-v2|charenc-v2.*charenc-v1"):
        CANDIDATES["hdc"].load(SITE, f.data, other)
