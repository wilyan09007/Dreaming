"""Speculation offline (docs/speculation.md §3): calls overlap, a dependent starts on a confident guess, nothing seen is a guess.

categorize (the keyword teacher, 800 ms) and a 1.2 s mock vendor lookup run side by side; a certified student downstream reads
both. Latencies are virtual (context.charge): the numbers below are exact.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from conftest import LABELS, CharEncoder, KeywordTeacher
from dreaming import Runtime, registry
from dreaming.context import charge
from dreaming.contracts import Decision, Hashes, Json, Phase, Prediction, Run, Thresholds, TraceRow
from dreaming.hashing import sha
from dreaming.predictors import Loader
from dreaming.predictors.train import LocalTrainer
from dreaming.replay import replay
from dreaming.speculate import Pending, result
from dreaming.store import MemoryStore

TXN: dict[str, Json] = {"d": "ADOBE *1234", "amt": 52.0}


class Fixed:
    """A predictor that always answers `label` at `confidence`, in 5 ms."""

    env_sha = "test-env"

    def __init__(self, label: str, confidence: float) -> None:
        self.label, self.confidence, self.weights_sha = label, confidence, sha([label, confidence])

    def predict(self, rendered: str, *, fresh: bool = False) -> Prediction:
        return Prediction(label=self.label, confidence=self.confidence, latency_ms=5,
                          hashes=Hashes(prompt=sha(rendered), weights=self.weights_sha, env=self.env_sha))


def build(speculate: bool, vendor_guess: str, speculate_at: float | None = 0.5) -> tuple[Runtime, MemoryStore]:
    store = MemoryStore()
    loader = Loader(encoder=CharEncoder())
    rt = Runtime(store, KeywordTeacher(), LocalTrainer(store, loader), loader, speculate=speculate)

    @registry.step("categorize", workflow="t", labels=LABELS)
    def categorize(txn: dict[str, Json]) -> str:
        return f"Transaction: {txn['d']}"

    @registry.tool("vendor_lookup", workflow="t", thresholds=Thresholds(speculate_at=speculate_at))
    def vendor_lookup(token: str) -> str:
        charge(1200)
        return token.title() + " Inc."

    @registry.step("check", workflow="t", labels=["ok", "odd"], thresholds=Thresholds(audit_rate=0.0))
    def check(category: str, vendor: str) -> str:
        return f"{vendor} filed under {category}"

    @registry.effect("post", workflow="t")
    def post(category: str, vendor: str, verdict: str) -> str:
        return f"{vendor}/{category}/{verdict}"

    def flow(txn: Json) -> str:
        assert isinstance(txn, dict)
        category = categorize(txn)
        vendor = vendor_lookup(str(txn["d"]).split(" *")[0])
        return post(category, vendor, check(category, vendor))

    rt.bind()
    rt.workflow = flow
    rt.predictors["t.vendor_lookup"] = Fixed(vendor_guess, 0.7)  # shadowing: a guess, not an answer
    rt.policies["t.vendor_lookup"].phase = Phase.watching
    rt.predictors["t.check"] = student = Fixed("ok", 0.9)  # certified: answers in 5 ms, never audited
    p = rt.policies["t.check"]
    p.phase, p.cutoff, p.cert_id, p.weights_sha = Phase.dreaming, 0.5, "cert_t", student.weights_sha
    return rt, store


def one_run(rt: Runtime, store: MemoryStore) -> tuple[Run, list[TraceRow]]:
    assert rt.workflow
    with rt.run("t", TXN, "fast_forward") as run:
        out = rt.workflow(TXN)
    rt.flush()
    assert out == "Adobe Inc./Software/ok" and not run.failed
    return run, store.run_rows(run.id)


def test_a_dependent_starts_on_the_guess_and_the_run_takes_the_slowest_call():
    rt, store = build(True, '"Adobe Inc."')
    run, rows = one_run(rt, store)
    check = [r for r in rows if r.call_site == "t.check"]
    assert len(check) == 1 and check[0].speculated and check[0].kept and check[0].decision == Decision.predicted
    # categorize 800 ms alongside the lookup's 1.2 s; check started at 800 ms on the lookup's guess and was kept at 1.2 s
    assert run.critical_ms == run.latency_ms == 1200 < sum(r.latency_ms for r in rows)
    assert [r.i for r in rows] == [1, 2, 3, 4]


def test_dag_alone_overlaps_independent_calls_and_runs_nothing_on_a_guess():
    rt, store = build(True, '"Adobe Inc."', speculate_at=None)
    run, rows = one_run(rt, store)
    assert not any(r.speculated for r in rows)
    assert run.critical_ms == 1200 + 5 < sum(r.latency_ms for r in rows)  # check waited for the verified lookup


def test_a_wrong_guess_is_redone_on_the_real_value_and_replay_skips_the_discarded_row():
    rt, store = build(True, '"Wrong Inc."')
    run, rows = one_run(rt, store)
    first, again = [r for r in rows if r.call_site == "t.check"]
    assert first.speculated and first.kept is False and "Wrong Inc." in str(first.request)
    assert not again.speculated and again.kept is None and "Adobe Inc." in str(again.request) and first.i == again.i
    assert run.critical_ms == 1200 + 5 and run.replayable == 1  # redone after the lookup verified; the discard decided nothing
    rep = replay(store, run.id, rt.predictors)
    assert [s.i for s in rep.steps].count(first.i) == 1 and rep.re_executed == 1 and rep.same == 1


def test_speculate_off_returns_plain_values_and_the_critical_path_is_the_sum():
    rt, store = build(False, '"Adobe Inc."')
    run, rows = one_run(rt, store)
    assert not any(r.speculated for r in rows)
    assert run.critical_ms == run.latency_ms == 800 + 1200 + 5 == sum(r.latency_ms for r in rows)


def test_a_pending_in_plain_python_is_a_type_error():
    rt, _ = build(True, '"Adobe Inc."')
    assert rt.spec
    with rt.run("t", TXN, "fast_forward"):
        p = rt.spec.submit(registry.get("t.vendor_lookup"), ("ADOBE",), {})
        assert isinstance(p, Pending)
        q: Any = p
        for plain in (lambda: f"{q}", lambda: q == "Adobe Inc.", lambda: q.upper(), lambda: ", ".join(q)):
            with pytest.raises(TypeError):
                plain()
        assert result(p) == p.result() == "Adobe Inc." and result("plain") == "plain"


def test_an_impure_dependent_waits_for_verified_inputs():
    rt, store = build(True, '"Wrong Inc."')  # a pure dependent would start on this wrong guess and be redone
    assert rt.spec
    with rt.run("t", TXN, "fast_forward") as run:
        category = rt.spec.submit(registry.get("t.categorize"), (TXN,), {})
        vendor = rt.spec.submit(registry.get("t.vendor_lookup"), ("ADOBE",), {})
        assert result(rt.spec.submit(dataclasses.replace(registry.get("t.check"), pure=False), (category, vendor), {})) == "ok"
    rt.flush()
    rows = store.run_rows(run.id)
    assert len(rows) == 3 and not any(r.speculated for r in rows) and run.critical_ms == 1200 + 5


@pytest.mark.parametrize("speculate", [False, True])
def test_a_two_argument_route_wrapper_still_routes(speculate: bool):
    rt, store = build(speculate, '"Adobe Inc."')
    inner = rt._route  # pyright: ignore[reportPrivateUsage]
    rt._route = lambda reg, call: inner(reg, call)  # telemetry.instrument wraps it exactly so  # pyright: ignore
    _, rows = one_run(rt, store)
    assert len(rows) == 4
