"""Two candidates on one site: both train, both shadow, each certifies at delta/N, the best coverage answers, and when the
winner's audits fail the spare takes over only on the strength of its own shadow answers on the audited rows."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from conftest import LABELS, WORDS, CharEncoder, KeywordTeacher, transactions
from dreaming import Runtime, registry
from dreaming.contracts import CallSite, EventKind, Hashes, Json, Phase, Prediction, Thresholds, TraceRow
from dreaming.hashing import sha_bytes
from dreaming.predictors import Loader
from dreaming.predictors.candidates import CANDIDATES, Candidate, Fitted, register
from dreaming.predictors.train import LocalTrainer, labelled
from dreaming.store import MemoryStore

BROKEN = {"merchant": False}  # flips the merchant candidate's answers after promotion: the drift its audits must catch


def _fit_merchant(site: CallSite, rows: Sequence[TraceRow], encoder: Any) -> Fitted:
    """A lookup from the merchant word to the label: 100% coverage on repeated merchants, nothing on new ones."""
    t0 = time.perf_counter()
    table = {text.split()[1]: label for text, label in labelled(rows)}
    return Fitted(json.dumps(dict(sorted(table.items()))).encode(), len(table), 1.0, time.perf_counter() - t0)


class Merchant:
    env_sha = "test-env"

    def __init__(self, site: CallSite, data: bytes) -> None:
        self.table, self.weights_sha, self.labels = dict(json.loads(data)), sha_bytes(data), list(site.labels or [])

    def predict(self, rendered: str, *, fresh: bool = False) -> Prediction:
        label = self.table.get(rendered.split()[1])
        if label is not None and BROKEN["merchant"]:
            label = self.labels[(self.labels.index(label) + 1) % len(self.labels)]  # confidently wrong, like a merchant that changed hands
        return Prediction(label=label or self.labels[0], confidence=1.0 if label else 0.0,
                          hashes=Hashes(prompt=sha_bytes(rendered.encode()), weights=self.weights_sha, env=self.env_sha))


if "merchant" not in CANDIDATES:
    register(Candidate("merchant", _fit_merchant, lambda site, data, enc: Merchant(site, data), needs_encoder=False))


def build(store: MemoryStore) -> Runtime:
    BROKEN["merchant"] = False
    loader = Loader(encoder=CharEncoder())
    rt = Runtime(store, KeywordTeacher(), LocalTrainer(store, loader), loader, budget=0.05)

    @registry.step("categorize", workflow="books", labels=LABELS, thresholds=Thresholds(watch_after=60, calibrate_after=200),
                   candidates=["merchant", "head"])  # the first weights back take the slot; ties keep the incumbent
    def categorize(txn: dict[str, object]) -> str:
        return f"Transaction: {txn['d']}, ${txn['amt']}"

    def workflow(txn: Json) -> None:
        assert isinstance(txn, dict)
        categorize(txn)

    rt.bind()
    rt.workflow = workflow
    return rt


def test_every_candidate_trains_shadows_and_the_best_coverage_answers():
    store = MemoryStore()
    rt = build(store)
    assert rt.workflow
    rt.fast_forward("books", transactions(600), rt.workflow)
    p = rt.policies["books.categorize"]
    kinds = [e.kind for e in store.events()]
    assert kinds.count(EventKind.train_started) == 2 and kinds.count(EventKind.train_finished) == 2
    assert p.phase == Phase.dreaming and p.candidate == "merchant", [e.message for e in store.events()]  # both certify at 100%; ties keep the incumbent
    assert set(p.shadow) == {"head"} and "head" in rt.spares["books.categorize"] and EventKind.candidate_promoted in kinds
    cert = store.certificate(p.cert_id or "")
    assert cert.candidate == "merchant" and cert.delta == 0.025  # delta/N with N = 2
    spare = p.shadow["head"]
    assert spare.cert_id and store.certificate(spare.cert_id).candidate == "head" and spare.coverage is not None
    shadowed = [r for r in store.rows("books.categorize", shadow_only=True)]
    assert shadowed and all(set(r.shadow) <= {"head", "merchant"} and len(r.shadow) == 1 for r in shadowed)  # the other candidate shadowed every calibration row
    assert all(r.prediction and r.prediction.candidate for r in shadowed)
    np.testing.assert_array_less(0, cert.coverage)


def test_a_revoked_winner_fails_over_to_the_spare_only_on_the_audited_rows():
    store = MemoryStore()
    rt = build(store)
    assert rt.workflow
    rt.fast_forward("books", transactions(600), rt.workflow)
    p = rt.policies["books.categorize"]
    assert p.candidate == "merchant" and p.phase == Phase.dreaming
    BROKEN["merchant"] = True  # every confident answer is now wrong; 1 in 20 is audited, so ~1,200 runs make the bound certain
    rt.fast_forward("books", transactions(1200, seed=3), rt.workflow)
    events = [e for e in store.events() if e.kind != EventKind.batch_progress]
    kinds = [e.kind for e in events]
    assert EventKind.cert_revoked in kinds
    assert p.candidate == "head" and p.phase == Phase.dreaming, [e.message for e in events][-8:]
    cert = store.certificate(p.cert_id or "")
    assert cert.candidate == "head" and cert.status == "active"
    # the handover is one of two paths, both certified against the teacher: the spare's own answers on the audited rows at the budget's
    # delta (0.05), or, when the ~59 audits that revoke are too few for the grid gate, a re-certification in testing at delta/N (0.025)
    assert cert.delta in (0.05, 0.025)
    assert kinds.index(EventKind.cert_revoked) < len(kinds) - 1 - kinds[::-1].index(EventKind.cert_issued)  # revoked, then re-issued
    revoked = [e for e in events if e.kind == EventKind.cert_revoked]
    assert all(store.certificate(str(e.data["cert_id"])).status == "revoked" for e in revoked if e.data.get("cert_id"))
    assert "merchant" in rt.spares["books.categorize"] and rt.predictors["books.categorize"].weights_sha == cert.weights_sha
    assert sum(1 for e in events if e.kind == EventKind.train_started) == 2  # nothing retrained: the spare was already loaded


def test_merchant_words_cover_the_fixture():  # the fixture's merchants are keyword-labelled, so the lookup candidate is exact
    assert all(any(w in WORDS[lab] for lab in LABELS) for w in [w for ws in WORDS.values() for w in ws])
