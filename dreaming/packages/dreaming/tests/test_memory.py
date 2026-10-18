"""The `memory` candidate offline: fit on rows labelled by a keyword rule, load, read back, refuse a foreign encoder."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from conftest import LABELS, CharEncoder, KeywordTeacher, transactions
from dreaming.contracts import CallSite, Decision, Kind, TraceRow
from dreaming.hashing import sha, sha_bytes
from dreaming.predictors import profile_sha
from dreaming.predictors.candidates import CANDIDATES, for_site
from dreaming.predictors.memory import Memory

SITE = CallSite(id="books.categorize", workflow="books", name="categorize", kind=Kind.model,
                labels=[*LABELS, "Other"])  # "Other" has no rows


def make_rows(n: int) -> list[TraceRow]:
    teacher, rows = KeywordTeacher(), list[TraceRow]()
    for i, txn in enumerate(transactions(n), 1):
        text = f"Transaction: {txn['d']}, ${txn['amt']}"
        ans = teacher.ask(SITE, text)
        rows.append(TraceRow(
            id=f"r{i}", run_id="run", i=i, call_site=SITE.id, kind=Kind.model, request=text, request_sha=sha(text),
            response=ans.text, decision=Decision.called, teacher=ans, latency_ms=800, cost_usd=0.01,
            ts=datetime(2026, 10, 17, tzinfo=UTC),
        ))
    return rows


def test_memory_reads_back_its_rows_byte_stably_and_refuses_a_foreign_encoder():
    rows, enc = make_rows(60), CharEncoder()
    assert [c.name for c in for_site(SITE.model_copy(update={"labels": LABELS}))] == ["head", "memory"]

    f = CANDIDATES["memory"].fit(SITE, rows, enc)
    assert f.n_train == 60 and f.holdout_accuracy >= 0.9 and f.weights_sha == sha_bytes(f.data)
    m = CANDIDATES["memory"].load(SITE, f.data, enc)
    assert isinstance(m, Memory) and m.weights_sha == f.weights_sha
    assert m.env_sha == profile_sha(encoder=enc.env_sha, candidate="memory", beta=m.beta)

    preds = [m.predict(str(r.request)) for r in rows]
    assert sum(p.label == r.teacher.label for p, r in zip(preds, rows, strict=True) if r.teacher) >= 54  # >= 90% of 60
    again = [m.predict(str(r.request)) for r in rows]
    assert preds == again == [m.predict(str(r.request), fresh=True) for r in rows]  # same bytes every time
    assert all(p.scores["Other"] == 0.0 and 0.0 <= p.confidence <= 1.0 for p in preds)  # no rows, probability 0

    other = CharEncoder()
    other.model_sha = "charenc-v2"
    with pytest.raises(ValueError, match="charenc-v1.*charenc-v2|charenc-v2.*charenc-v1"):
        CANDIDATES["memory"].load(SITE, f.data, other)
