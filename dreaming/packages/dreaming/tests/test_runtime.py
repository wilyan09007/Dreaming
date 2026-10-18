"""The whole loop offline: log -> shadow -> certify -> promote -> budget change -> outage -> replay."""

from __future__ import annotations

from conftest import LABELS, CharEncoder, KeywordTeacher, transactions
from dreaming import Runtime, registry
from dreaming.contracts import Decision, EventKind, Json, Phase, Thresholds
from dreaming.predictors import Loader
from dreaming.predictors.train import LocalTrainer
from dreaming.replay import replay
from dreaming.store import MemoryStore, Store

DOWN = {"vendor": False}


def build(store: Store, fallback: bool = True) -> Runtime:
    DOWN["vendor"] = False
    loader = Loader(encoder=CharEncoder())
    rt = Runtime(store, KeywordTeacher(), LocalTrainer(store, loader), loader, budget=0.05)

    @registry.step("categorize", workflow="books", labels=LABELS, thresholds=Thresholds(watch_after=60, calibrate_after=200))
    def categorize(txn: dict[str, object]) -> str:
        return f"Transaction: {txn['d']}, ${txn['amt']}"

    def snapshot_lookup(d: str) -> str:  # the registered fallback: last night's snapshot
        return {"ADOBE": "Adobe Inc."}.get(d.split()[0], "unknown")

    @registry.tool("vendor_lookup", workflow="books", fallback=snapshot_lookup if fallback else None,
                   thresholds=Thresholds(watch_after=60, calibrate_after=200))
    def vendor_lookup(descriptor: str) -> str:
        if DOWN["vendor"]:  # fault injection into the mock; the runtime detects the outage from the failures
            raise ConnectionError("vendor API down")
        return descriptor.split()[0].title() + " Inc."

    def workflow(txn: Json) -> None:
        assert isinstance(txn, dict)
        categorize(txn)
        vendor_lookup(str(txn["d"]).split(" *")[0])

    rt.bind()
    rt.workflow = workflow
    return rt


def test_loop_promotes_certifies_and_replays():
    store = MemoryStore()
    rt = build(store)
    assert rt.workflow
    m = rt.fast_forward("books", transactions(900), rt.workflow)
    cat, ven = rt.policies["books.categorize"], rt.policies["books.vendor_lookup"]
    assert cat.phase == Phase.dreaming and ven.phase == Phase.dreaming, [e.message for e in store.events()]
    assert cat.cert_id and store.certificate(cat.cert_id).bound <= 0.05
    kinds = [e.kind for e in store.events()]
    assert kinds.count(EventKind.cert_issued) >= 2 and EventKind.train_finished in kinds and EventKind.component_error not in kinds
    assert m.runs > 0 and m.replayable_pct > 30 and m.saved_usd > 0 and m.cost_per_run < 0.02

    # a predicted decision re-executes byte for byte
    run = next(r for r in reversed(store.runs()) if r.replayable)
    rep = replay(store, run.id, rt.predictors, rt.predictor_for)
    assert rep.re_executed == run.replayable and rep.same == rep.re_executed

    # tightening the budget re-certifies without retraining; loosening restores
    weights_before = cat.weights_sha
    rt.set_budget(0.001)
    assert cat.cutoff is None or cat.cutoff > 0
    rt.set_budget(0.05)
    assert cat.phase == Phase.dreaming and cat.weights_sha == weights_before

    # an outage: certified inputs still answered by the dream, unknown ones by the fallback, nothing is stuck
    # the dream answers every known merchant, so the outage only shows through the 1-in-20 audits: 200 runs make it certain
    DOWN["vendor"] = True
    before = {run.id for run in store.runs(limit=10_000)}
    rt.fast_forward("books", transactions(200, seed=7), rt.workflow)
    new = [run for run in store.runs(limit=10_000) if run.id not in before]
    rows = [r for run in new for r in store.run_rows(run.id) if r.call_site == "books.vendor_lookup"]
    assert len(new) == 200 and rows and all(r.decision in (Decision.predicted, Decision.fallback) for r in rows)
    assert not any(run.failed for run in new)
    assert ven.outage and EventKind.tool_outage in [e.kind for e in store.events()]  # detected from the failures, never declared


def test_untrained_tool_without_fallback_gets_stuck():
    store = MemoryStore()
    rt = build(store, fallback=False)
    DOWN["vendor"] = True
    assert rt.workflow
    rt.fast_forward("books", transactions(8), rt.workflow)
    runs = store.runs()
    assert all(r.failed for r in runs)  # the first failures raise the tool's error; from the third on the run is Stuck
    assert any(r.stuck_at == "books.vendor_lookup" for r in runs) and rt.policies["books.vendor_lookup"].outage


def test_audits_revoke_only_when_their_bound_can_clear_the_budget_and_a_restart_keeps_the_promise():
    """50 clean audits bound the error at 5.8%: the old rule revoked a perfect predictor there. Audits may revoke only once
    enough of them could have cleared the budget (59 at 5%/5%), and then only when their bound exceeds it (B3). A second process
    on the same store reloads the weights by hash and keeps the certificate (B8)."""
    store = MemoryStore()
    rt = build(store)
    assert rt.workflow
    rt.fast_forward("books", transactions(900), rt.workflow)
    cat = rt.policies["books.categorize"]
    assert cat.phase == Phase.dreaming
    for audits, errors, revoked in ((50, 0, False), (58, 0, False), (59, 0, False), (200, 2, False), (100, 10, True)):
        cat.audit_rows, cat.audit_errors = audits, errors
        rt.tick()
        assert (cat.phase == Phase.testing) is revoked, (audits, errors, cat.phase)
        if revoked:
            assert EventKind.cert_revoked in [e.kind for e in store.events(limit=100_000)]
    rt.set_budget(0.05)  # back through the gate on the frozen calibration rows: the promise returns
    assert cat.phase == Phase.dreaming

    registry._SITES.clear()  # pyright: ignore[reportPrivateUsage]
    registry.bind(None)
    again = build(store)
    p = again.policies["books.categorize"]
    assert p.phase == Phase.dreaming and p.cert_id == cat.cert_id
    assert again.predictors["books.categorize"].weights_sha == p.weights_sha
    assert again.predictors["books.vendor_lookup"].weights_sha == again.policies["books.vendor_lookup"].weights_sha


def test_calibration_set_survives_a_long_dreaming_stream():
    """The store's shadow_only window holds calibration rows only, so 10k dreaming-phase rows cannot page them out (B1)."""
    store = MemoryStore()
    rt = build(store)
    assert rt.workflow
    rt.fast_forward("books", transactions(600), rt.workflow)
    cat = rt.policies["books.categorize"]
    cert = store.certificate(cat.cert_id or "")
    sample = next(r for r in store.rows("books.categorize") if r.phase == Phase.dreaming and r.agree is not None)
    store.write_rows([sample.model_copy(update={"id": f"row_x{i}"}) for i in range(10_500)])
    rt.set_budget(0.001)
    rt.set_budget(0.05)
    assert cat.phase == Phase.dreaming and store.certificate(cat.cert_id or "").calib_sha == cert.calib_sha
    assert store.certificate(cert.id).status == "revoked"


def test_run_accounting_survives_a_flush_and_a_concurrent_live_run():
    store = MemoryStore()
    rt = build(store)
    rt.flush_every = 3
    assert rt.workflow
    with rt.run("books", {"d": "ADOBE *1", "amt": 1.0}, "live") as outer:
        rt.workflow({"d": "ADOBE *1", "amt": 1.0})  # 2 calls
        with rt.run("books", {"d": "UBER *2", "amt": 2.0}, "live") as inner:  # joins the open run (nested boundary)
            rt.workflow({"d": "UBER *2", "amt": 2.0})
    assert outer.calls == 4 and outer.decisions == 4 and inner.id == outer.id
    assert len(store.run_rows(outer.id)) == 4 and store.run_count() == 1
