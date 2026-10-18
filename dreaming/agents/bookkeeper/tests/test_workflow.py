"""The bookkeeper end to end, offline: mock world, rule teacher, hashed-trigram encoder, in-memory store."""

from __future__ import annotations

import json
from collections import Counter

import pytest
from bookkeeper import fast_forward, offline, record, tools, workflow
from dreaming.contracts import Decision, EventKind, Metrics, Phase, Policy
from dreaming.store import MemoryStore

VENDOR = "bookkeeper.vendor_lookup"


def _ticket(descriptor: str) -> str:
    return descriptor.split(" *")[0]


def test_ledger_is_a_seeded_world_with_a_long_tail() -> None:
    rows = list(tools.ledger(2000, seed=0))
    assert rows == list(tools.ledger(2000, seed=0)) and rows != list(tools.ledger(2000, seed=1))
    assert all(set(r) == {"id", "descriptor", "amount", "date", "merchant"} for r in rows) and len({str(r["id"]) for r in rows}) == 2000
    assert len(tools.MERCHANTS) == 120 and {m.category for m in tools.MERCHANTS.values()} == set(workflow.CATEGORIES)
    names = {_ticket(str(r["descriptor"])): r["merchant"] for r in rows}  # the split key is stable: one merchant per key
    assert all(tools.MERCHANTS[k].name == v for k, v in names.items())
    seen = {_ticket(str(r["descriptor"])) for r in rows[:500]}
    late_new = [r for r in rows[500:] if _ticket(str(r["descriptor"])) not in seen]
    assert 0.01 < len(late_new) / 1500 < 0.16  # merchants never seen in the first 500 rows (the rare tail is drawn from row one)
    ratio = [float(str(r["amount"])) / tools.MERCHANTS[_ticket(str(r["descriptor"]))].usual for r in rows]
    assert 0.01 < sum(x > 10 for x in ratio) / 2000 < 0.035 and all(x < 5 or 19.9 < x < 100.1 for x in ratio)  # planted: 20-100x usual
    top = Counter(_ticket(str(r["descriptor"])) for r in rows).most_common(1)[0][1]
    assert top > 10 * 2000 / 120  # repeat merchants dominate


def test_externals() -> None:
    assert tools.vendor_api("ADOBE") == "Adobe Inc." and tools.vendor_api("NOPE") == "unknown"
    first = {_ticket(str(r["descriptor"])) for r in tools.ledger(500, seed=0)}
    never = next(t for t in tools.MERCHANTS if t not in first)
    assert tools.vendor_snapshot(min(first)) == tools.vendor_api(min(first)) and tools.vendor_snapshot(never) == "unknown"
    tools.DOWN["vendor_api"] = True
    with pytest.raises(ConnectionError):
        tools.vendor_api("ADOBE")
    assert tools.vendor_snapshot(min(first)) != "unknown"  # the snapshot is what survives the outage
    assert all(0 <= len(tools.open_invoices(v)) <= 3 and 0 <= len(tools.history(v)) <= 5 for v in (m.name for m in tools.MERCHANTS.values()))
    assert tools.open_invoices("Adobe Inc.") == tools.open_invoices("Adobe Inc.") and tools.history("Adobe Inc.") == tools.history("Adobe Inc.")


def test_fast_forward_learns_offline_and_survives_an_unplugged_api() -> None:
    store = MemoryStore()
    rt = offline.build_runtime(store)
    m = rt.fast_forward(workflow.W, tools.ledger(1500, seed=0), workflow.run_transaction)
    phase = {sid.split(".")[1]: p.phase for sid, p in rt.policies.items()}
    errors = [(e.kind, e.error) for e in store.events(limit=100_000) if e.error or e.kind == EventKind.component_error]
    assert not errors, errors[:3]
    assert phase["categorize"] == phase["vendor_lookup"] == phase["match_invoice"] == Phase.dreaming, phase
    assert phase["anything_odd"] == Phase.dreaming, phase  # the router site: thinking off certifies at 5% in this world
    assert m.replayable_pct > 20 and m.failed_pct == 0 and m.saved_usd > 0, m
    for sid in (s for s in rt.policies if s.split(".")[1] in ("categorize", "vendor_lookup", "match_invoice")):
        assert store.certificate(rt.policies[sid].cert_id or "").bound <= rt.budget
    assert not {r.teacher.provider for r in store.rows("bookkeeper.categorize") if r.teacher} - {"rule"}

    # unplug the vendor API after promotion: the memory answers what it knows, the snapshot the rest, no run fails
    tools.DOWN["vendor_api"] = True  # fault injection; the runtime detects the outage from the failures
    before = {r.id for r in store.runs(limit=10_000)}
    rt.fast_forward(workflow.W, tools.ledger(100, seed=7), workflow.run_transaction)
    new = [r for r in store.runs(limit=10_000) if r.id not in before]
    rows = [row for r in new for row in store.run_rows(r.id) if row.call_site == VENDOR]
    assert len(new) == 100 and len(rows) == 100 and not any(r.failed for r in new)
    decisions = Counter(row.decision for row in rows)
    assert set(decisions) <= {Decision.predicted, Decision.fallback} and decisions[Decision.predicted] > 0, decisions
    assert all(row.response in {m.name for m in tools.MERCHANTS.values()} | {"unknown"} for row in rows)
    assert any(e.kind == EventKind.tool_outage for e in store.events(limit=100_000))


def test_unplugged_before_learning_the_fallback_answers() -> None:
    store = MemoryStore()
    rt = offline.build_runtime(store)
    tools.DOWN["vendor_api"] = True
    rt.fast_forward(workflow.W, tools.ledger(30, seed=3), workflow.run_transaction)
    rows = [row for r in store.runs() for row in store.run_rows(r.id) if row.call_site == VENDOR]
    assert len(rows) == 30 and all(row.decision == Decision.fallback for row in rows)
    assert not any(r.failed for r in store.runs())
    assert {str(row.response) for row in rows} <= {m.name for m in tools.MERCHANTS.values()} | {"unknown"}


def test_a_raising_vendor_api_is_one_fallback_row_not_a_failed_run() -> None:
    store = MemoryStore()
    rt = offline.build_runtime(store)
    tools.DOWN["vendor_api"] = True  # nobody told the runtime: the real call raises and the router falls back
    rt.fast_forward(workflow.W, tools.ledger(10, seed=3), workflow.run_transaction)
    rows = [row for r in store.runs() for row in store.run_rows(r.id) if row.call_site == VENDOR]
    assert len(rows) == 10 and all(row.decision == Decision.fallback for row in rows)
    assert all("ConnectionError" in (row.error or "") for row in rows[:3])  # the first failures are tried and logged
    assert rt.policies[VENDOR].outage and any(e.kind == EventKind.tool_outage for e in store.events(limit=100_000))  # then detected
    assert not any(r.failed for r in store.runs())


def test_commands(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    assert fast_forward.main(["120", "--seed", "2"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert Metrics.model_validate(lines[0]).runs == 120
    assert [Policy.model_validate(d).call_site for d in lines[1:]] == [s for s in (f"bookkeeper.{n}" for n in (
        "categorize", "vendor_lookup", "match_invoice", "anything_odd", "post_to_books", "wake_a_human"))]
    monkeypatch.delenv("OPENROUTERKEY", raising=False)
    with pytest.raises(SystemExit, match="OPENROUTERKEY"):  # no key, no recording: it refuses before touching any file
        record.main(["10"])


def test_replay_serves_router_rows_from_the_log() -> None:
    """A router's prediction is a frontier call: replay never re-executes it (B4), and the receipt still adds up."""
    from dreaming.replay import replay

    store = MemoryStore()
    rt = offline.build_runtime(store)
    rt.fast_forward(workflow.W, tools.ledger(1500, seed=0), workflow.run_transaction)
    assert rt.policies["bookkeeper.anything_odd"].phase == Phase.dreaming
    run = next(r for r in reversed(store.runs(limit=10_000))
               if any(row.call_site.endswith("anything_odd") and row.decision == Decision.predicted for row in store.run_rows(r.id)))
    rep = replay(store, run.id, rt.predictors, rt.predictor_for)
    odd = next(s for s in rep.steps if s.call_site.endswith("anything_odd"))
    assert not odd.re_executed and rep.re_executed == run.replayable and rep.same == rep.re_executed
