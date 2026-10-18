"""The benchmark report, offline: MemoryStore holding two candidates on one site, the scores hand-made so the right coverage is known.

`q.invoice` (labels a/b, every true label "a"). Calibration is 400 rows: the top 200 by confidence all correct, the bottom 200 alternately
wrong and right. With delta 0.05 over 40 grid levels a zero-error prefix needs 331 rows to clear a 2% budget (so nothing certifies at 2%) and
200 rows clear 5%, while the next level up, 210 rows with 5 errors, does not: coverage 0.5 at 5%. The test split mirrors it (100 above the cutoff
with one error, 100 below all wrong), so the cutoff keeps exactly half the test rows at 1% error. `memory` is right everywhere: coverage 1.0.
The forward slice (400 rows in time order, repeat traffic) is certified on its own first 200 rows and measured on its last 200. By default it is
right everywhere with the same confidences in both halves (coverage 1.0, so `memory`'s two columns differ only by being different slices); in
`half_certifiable` the cutoff certified on the first half (200 clean rows, confidences 0.9 down to 0.6) keeps 75 of the last 200 rows, 3 of them
wrong: forward coverage 0.375 at error 0.04, not the test slice's 0.5 at 0.01.
The scores are in the npz's real layout (apps/trainer campaign.py): idx, split codes 0/1/2, scores, conf, pred, conf2, pred2.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import modal
import numpy as np
import pytest
from dreaming.contracts import (
    CallSite,
    Checkpoint,
    Decision,
    EvalSet,
    Kind,
    TeacherAnswer,
    TeacherSpec,
    TraceRow,
    TrainRun,
)
from dreaming.hashing import sha
from dreaming.store import MemoryStore
from dreaming_api.benchmark import (
    BUDGETS,
    Benchmark,
    Row,
    benchmark_router,
    markdown,
    quickbooks_baselines,
    table,
    volume_scores,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient

SITE = "q.invoice"
N = (400, 200, 400)  # calib, test, forward rows
T0 = datetime(2026, 10, 18, 4, 0, tzinfo=UTC)
Arrays = dict[str, Any]


def arrays(calib_ok: Any, calib_conf: Any, test_ok: Any, test_conf: Any, *, forward_ok: Any = None, forward_conf: Any = None, with_ok: bool = False) -> Arrays:
    fok = np.ones(N[2], bool) if forward_ok is None else forward_ok
    fconf = np.tile(np.linspace(0.9, 0.6, N[2] // 2), 2) if forward_conf is None else forward_conf
    ok = np.concatenate([calib_ok, test_ok, fok]).astype(bool)
    conf = np.concatenate([calib_conf, test_conf, fconf]).astype(np.float32)
    pred = np.where(ok, 0, 1).astype(np.int32)  # the true label is always "a" = index 0
    out: Arrays = {"idx": np.arange(len(ok), dtype=np.int32), "split": np.repeat(np.array([0, 1, 2], np.int8), N), "scores": np.zeros((len(ok), 2), np.float16),
                   "conf": conf, "pred": pred, "conf2": conf, "pred2": pred}
    return out | {"ok": ok} if with_ok else out


def half_certifiable() -> Arrays:
    """Coverage 0.5 at 5% (see the module docstring)."""
    alternating = np.arange(200) % 2 == 1  # wrong, right, wrong, ...
    test_top = np.ones(100, bool)
    test_top[40] = False
    late = np.zeros(200, bool)  # forward's second half: 75 rows clear the first half's cutoff (conf >= 0.6), 3 of them wrong; the rest are wrong
    late[[i for i in range(75) if i not in (10, 40, 70)]] = True
    return arrays(np.concatenate([np.ones(200, bool), alternating]), np.concatenate([np.linspace(0.9, 0.5, 200), np.linspace(0.4, 0.0, 200)]),
                  np.concatenate([test_top, np.zeros(100, bool)]), np.concatenate([np.linspace(0.9, 0.5, 100), np.linspace(0.4, 0.0, 100)]),
                  forward_ok=np.concatenate([np.ones(200, bool), late]), forward_conf=np.concatenate([np.linspace(0.9, 0.6, 200), np.linspace(0.9, 0.1, 200)]))


def always_right() -> Arrays:
    return arrays(np.ones(N[0], bool), np.linspace(0.9, 0.1, N[0]), np.ones(N[1], bool), np.linspace(0.9, 0.1, N[1]))


def never_right() -> Arrays:
    return arrays(np.zeros(N[0], bool), np.linspace(0.9, 0.1, N[0]), np.zeros(N[1], bool), np.linspace(0.9, 0.1, N[1]))


def eval_sets(site: str = SITE) -> list[EvalSet]:
    out: list[EvalSet] = []
    for split, n in zip(("calib", "test", "forward"), N, strict=True):
        ids = [str(i) for i in range(n)]
        out.append(EvalSet(sha=sha([site, split, ids]), call_site=site, split=split, row_ids=ids, labels=["a"] * n, groups=[f"g{i % 10}" for i in range(n)]))  # type: ignore[arg-type]
    return out


class World:
    """A MemoryStore plus the loader a Volume would be: uri -> arrays."""

    def __init__(self, site: CallSite | None = None) -> None:
        self.store = MemoryStore()
        self.files: dict[str, Arrays] = {}
        self.loads: list[str] = []
        self.store.upsert_sites([site or CallSite(id=SITE, workflow="q", name="invoice", kind=Kind.model, labels=["a", "b"],
                                                   teacher=TeacherSpec(provider="stepfun", model="step-5-preview", floor=0.035))])
        for es in eval_sets():
            self.store.put_eval_set(es)

    def cell(self, candidate: str, seed: int, files: dict[int, Arrays], *, variant: str = "seed+real", seconds: float = 1.0, embed_ms: float = 4.0) -> None:
        run = TrainRun(id=f"night:{SITE}:{candidate}:{variant}:s{seed}", campaign="night", call_site=SITE, candidate=candidate, variant=variant,  # type: ignore[arg-type]
                       seed=seed, where="modal:cpu", status="done", started_at=T0)
        self.store.put_train_run(run)
        for rung, a in files.items():
            uri = f"modal://dreaming-weights/campaign/{run.id}/{rung}.npz"
            self.files[uri] = a
            self.store.put_checkpoint(Checkpoint(run_id=run.id, rung=rung, weights_sha="w" * 64, eval_sha=sha([e.sha for e in eval_sets()]), scores_uri=uri, seconds=seconds,
                                                 embed_ms=embed_ms, head_ms=0.3, bytes=1, stable=True, summary={}, ts=T0))

    def load(self, uri: str) -> dict[str, Any]:
        self.loads.append(uri)
        return self.files[uri]  # a KeyError here is a checkpoint the table should not have needed


def rows_of(b: Benchmark, candidate: str, variant: str = "seed+real") -> Row:
    return next(r for r in b.rows if r.candidate == candidate and r.variant == variant)


def no_baselines(site: str) -> list[dict[str, Any]]:
    return []


@pytest.fixture
def w() -> World:
    """head: seeds 0 and 1 at the top rung 200 (and a rung 60 that has no file), memory: seed 0."""
    w = World()
    w.cell("head", 0, {200: half_certifiable()})
    w.cell("head", 1, {200: always_right()})
    w.store.put_checkpoint(w.store.checkpoints("night")[0].model_copy(update={"rung": 60, "scores_uri": "modal://dreaming-weights/campaign/missing/60.npz"}))
    w.cell("memory", 0, {200: always_right()})
    return w


def test_coverage_is_recomputed_from_the_stored_scores(w: World) -> None:
    b = table(w.store, w.load, no_baselines)
    head = rows_of(b, "head")
    assert (head.seeds, head.rung, head.source, head.stable) == (2, 200, "scores", True)  # the rung-60 checkpoint is never scored
    two, five = head.cells[:2]
    assert [c.budget for c in head.cells] == list(BUDGETS)
    assert two.test_coverage == pytest.approx(0.5) and two.coverage == pytest.approx(0.5)  # 2%: seed 0 certifies nothing, seed 1 everything
    assert five.test_coverage == pytest.approx(0.75) and five.coverage == pytest.approx(0.75)  # 5%: (0.5 + 1.0) / 2
    assert five.test_error == pytest.approx(0.005)  # seed 0: 1 of the 100 test rows above the cutoff is wrong, seed 1: none
    assert (two.forward_coverage, two.forward_error) == (0.0, None)  # 200 clean rows do not clear 2% on the forward slice either
    assert five.forward_coverage == pytest.approx((0.375 + 1.0) / 2) and five.forward_error == pytest.approx(0.02)  # seed 0: 75 rows above its cutoff, 3 wrong (0.04); seed 1: all, none wrong
    mem = rows_of(b, "memory")
    assert [c.test_coverage for c in mem.cells] == [1.0] * 4 and [c.test_error for c in mem.cells] == [0.0] * 4 and mem.seeds == 1
    assert [c.forward_coverage for c in mem.cells] == [0.0, 1.0, 1.0, 1.0] and [c.forward_error for c in mem.cells] == [None, 0.0, 0.0, 0.0]
    assert not any("repeat-traffic" in n for n in b.notes)
    assert w.loads.count("modal://dreaming-weights/campaign/night:q.invoice:head:seed+real:s0/200.npz") == 1
    assert not [u for u in w.loads if u.endswith("/60.npz")]


def test_one_candidate_alone_has_the_coverage_worked_out_in_the_docstring() -> None:
    w = World()
    w.cell("head", 0, {200: half_certifiable()})
    head = rows_of(table(w.store, w.load, no_baselines), "head")
    two, five, ten, twenty = head.cells
    assert (two.coverage, two.test_coverage, two.test_error) == (0.0, 0.0, None)  # a 2% budget needs 331 clean rows and only 200 are
    assert (five.coverage, five.test_coverage, five.test_error) == (0.5, 0.5, 0.01)
    assert (five.forward_coverage, five.forward_error) == (pytest.approx(0.375), pytest.approx(0.04))  # the forward cutoff is its own: certified on the first half, measured on the second
    assert 0.5 <= ten.coverage <= twenty.coverage and ten.test_error is not None and twenty.test_error is not None  # type: ignore[operator]  # a larger budget never covers less


def test_rows_sort_by_site_then_coverage_at_5_and_baselines_follow(w: World) -> None:
    b = table(w.store, w.load, lambda site: [{"name": "rule", "coverage": 0.9, "error": 0.02}])
    assert [(r.candidate, r.source) for r in b.rows] == [("memory", "scores"), ("head", "scores"), ("teacher alone", "baseline"), ("rule", "baseline")]
    teacher, rule = b.rows[2], b.rows[3]
    assert [c.test_coverage for c in rule.cells] == [0.9, 0.9, 0.9, 0.9]  # its error, 2%, is within every budget
    assert [c.test_coverage for c in teacher.cells] == [0.0, 1.0, 1.0, 1.0]  # a 3.5% floor cannot be promised below 3.5%
    assert teacher.cells[1].test_error == 0.035 and (teacher.provider, teacher.floor) == ("stepfun/step-5-preview", 0.035)


def test_missing_baselines_are_skipped_with_a_note(w: World) -> None:
    def absent(site: str) -> list[dict[str, Any]]:
        raise ImportError("No module named 'quickbooks.baselines'")

    b = table(w.store, w.load, absent)
    assert {r.candidate for r in b.rows if r.source == "baseline"} == {"teacher alone"}
    assert any("rule and exact rows skipped" in n and "quickbooks.baselines" in n for n in b.notes)
    assert quickbooks_baselines("bookkeeper.categorize") == []  # only the Intuit sites have those rows


def test_unreadable_scores_fall_back_to_the_checkpoint_summary(w: World) -> None:
    for ck in w.store.checkpoints("night"):
        w.store.put_checkpoint(ck.model_copy(update={"summary": {"coverage_02": 0.0, "err_02": 0.0, "coverage_05": 0.4, "err_05": 0.0, "coverage_10": 0.6, "err_10": 0.02}}))

    def gone(uri: str) -> dict[str, Any]:
        raise OSError(f"{uri}: volume unavailable")

    b = table(w.store, gone, no_baselines)
    head = rows_of(b, "head")
    assert head.source == "summary" and [c.coverage for c in head.cells] == [None] * 4 and [c.forward_coverage for c in head.cells] == [None] * 4
    assert any("head seed+real: no repeat-traffic" in n and "acc_forward" in n for n in b.notes)
    assert [c.test_coverage for c in head.cells] == [0.0, 0.4, 0.6, None] and [c.test_error for c in head.cells] == [None, 0.0, 0.02, None]
    assert any("volume unavailable" in n for n in b.notes) and "†" in markdown(b)


def test_an_ok_array_scores_a_site_with_no_label_list() -> None:
    w = World(CallSite(id=SITE, workflow="q", name="invoice", kind=Kind.tool, teacher=TeacherSpec(provider="tool")))  # a tool site: pred indexes a label list the store does not have
    w.cell("exact", 0, {500: arrays(np.ones(N[0], bool), np.linspace(0.9, 0.1, N[0]), np.ones(N[1], bool), np.linspace(0.9, 0.1, N[1]), with_ok=True)})
    b = table(w.store, w.load, no_baselines)
    exact = rows_of(b, "exact")
    assert exact.source == "scores" and exact.cells[1].test_coverage == 1.0 and exact.provider == "tool" and exact.floor is None
    del w.files[next(iter(w.files))]["ok"]  # without it the same scores cannot be scored: summary fallback, with the reason
    b = table(w.store, w.load, no_baselines)
    assert rows_of(b, "exact").source == "summary" and any("no label list" in n for n in b.notes)


def test_seconds_to_the_first_certificate_with_seed_rows_only() -> None:
    w = World()
    w.cell("head", 0, {10: never_right(), 30: always_right()}, variant="seed", seconds=1.5)
    w.cell("head", 0, {200: always_right()})
    b = table(w.store, w.load, no_baselines)
    assert rows_of(b, "head", "seed").seed_cert_s == 1.5 and rows_of(b, "head", "seed").rung == 30  # rung 10 certifies nothing, rung 30 does
    assert rows_of(b, "head").seed_cert_s == 1.5  # the column belongs to the candidate: it sits on its seed+real row too
    md = markdown(b)
    assert any(ln.startswith("| q.invoice") and " head (seed) " in ln for ln in md.splitlines()) and not any(" head (seed+real) " in ln for ln in md.splitlines())
    assert not any("seed-only" in n for n in b.notes)
    w2 = World()
    w2.cell("head", 0, {200: always_right()})
    assert any("no seed-only" in n for n in table(w2.store, w2.load, no_baselines).notes)


def test_cost_and_latency_per_1000_calls_from_the_sites_teacher_calls() -> None:
    w = World()
    w.cell("head", 0, {200: half_certifiable()}, embed_ms=0.0)  # a cell on precomputed vectors reports no embedding time
    answer = TeacherAnswer(text="a", label="a", provider="stepfun", model="step-5-preview", cost_usd=0.01, latency_ms=800)
    w.store.write_rows([TraceRow(id=f"r{i}", run_id="x", i=i + 1, call_site=SITE, kind=Kind.model, request="t", request_sha="s", response="a", decision=Decision.called,
                                 teacher=answer, latency_ms=800, cost_usd=0.01, ts=T0) for i in range(20)])
    b = table(w.store, w.load, no_baselines)
    head = rows_of(b, "head")
    assert head.without_dreaming is not None and head.with_dreaming is not None
    assert (head.without_dreaming.usd, head.without_dreaming.p50_ms, head.without_dreaming.p95_ms) == pytest.approx((10.0, 800.0, 800.0))
    assert head.with_dreaming.usd == pytest.approx(1000 * (0.5 + 0.5 * 0.05) * 0.01, rel=1e-3)  # half answered, 5% of those still audited, plus microdollars of CPU
    assert head.with_dreaming.p50_ms == pytest.approx(5.3)  # est. 5.0 embed + 0.3 head: half the calls are that fast, so the median is
    assert head.with_dreaming.p95_ms == pytest.approx(800.0)  # the 95th is still a teacher call
    assert any("est. 5.0 ms" in n and "q.invoice head" in n for n in b.notes)
    assert next(r for r in b.rows if r.candidate == "teacher alone").without_dreaming is not None


def test_the_routes_serve_json_and_markdown_and_read_the_volume_once_a_minute(w: World) -> None:
    app = FastAPI()
    app.include_router(benchmark_router(w.store, w.load, no_baselines))
    c = TestClient(app)
    j = c.get("/benchmark").json()
    assert [r["candidate"] for r in j["rows"]] == ["memory", "head", "teacher alone"] and j["campaign"] == "night"
    md = c.get("/benchmark.md")
    assert md.headers["content-type"].startswith("text/markdown")
    lines = [ln for ln in md.text.splitlines() if ln.startswith("| q.invoice")]
    assert [ln.split(" | ")[3] for ln in lines] == ["memory", "head", "teacher alone"]
    assert "| 75.0% |" in lines[1] and "campaign `night`" in md.text
    assert len(w.loads) == 3  # head s0, head s1, memory s0: one read each across both requests
    assert c.get("/benchmark", params={"campaign": "other"}).json()["rows"] == []
    assert len(w.loads) == 3  # an unknown campaign reads nothing


def test_the_volume_loader_reads_an_npz_through_modal_and_turns_its_failures_into_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    buf = io.BytesIO()
    np.savez(buf, conf=np.array([0.5], np.float32), split=np.array([0], np.int8))
    seen: list[str] = []

    class FakeVolume:
        def read_file(self, path: str) -> Iterator[bytes]:
            seen.append(path)
            yield buf.getvalue()[:10]
            yield buf.getvalue()[10:]

    def from_name(name: str) -> FakeVolume:
        seen.append(name)
        return FakeVolume()

    monkeypatch.setattr(modal.Volume, "from_name", from_name)
    z = volume_scores("modal://dreaming-weights/campaign/r1/200.npz")
    assert seen == ["dreaming-weights", "campaign/r1/200.npz"] and z["conf"].tolist() == [0.5]

    def broken(name: str) -> FakeVolume:
        raise RuntimeError("no token")

    monkeypatch.setattr(modal.Volume, "from_name", broken)
    with pytest.raises(OSError, match="no token"):
        volume_scores("modal://dreaming-weights/x.npz")



def test_markdown_marks_budgets_below_the_floor_and_names_the_variant(w: World) -> None:
    md = markdown(table(w.store, w.load, no_baselines))
    head = next(ln for ln in md.splitlines() if " | head | " in ln)
    cols = [c.strip() for c in head.strip("|").split("|")]
    assert cols[8].endswith("*") and not cols[9].endswith("*")  # floor 3.5%: the unseen 2% column is not measurable, the 5% one is
    assert cols[6:8] == ["68.8%", "2.0%"] and cols[9] == "75.0%"  # repeat traffic first (forward: 0.6875, 0.02), then unseen merchants (test: 0.75)
    assert "repeat cov 5%" in md and "unseen cov 20%" in md and "repeat = the forward slice" in md and "unseen = group-held-out" in md
    assert "* a budget below the provider's floor" in md
    teacher = next(ln for ln in md.splitlines() if " | teacher alone | " in ln)
    assert "*" not in teacher  # a baseline sits at its own error, not under a floor


def test_an_unstable_checkpoint_is_called_out_not_hidden(w: World) -> None:
    for ck in w.store.checkpoints("night"):
        w.store.put_checkpoint(ck.model_copy(update={"stable": ck.run_id.endswith("head:seed+real:s1")}))
    b = table(w.store, w.load, no_baselines)
    assert not rows_of(b, "head").stable and rows_of(b, "memory").stable is False
    assert any("q.invoice head seed+real" in n and "stable is false" in n for n in b.notes)


def test_a_failing_baseline_source_costs_its_rows_not_the_report(w: World) -> None:
    def broken(site: str) -> list[dict[str, Any]]:
        raise FileNotFoundError("no QuickBooks data on this host")

    b = table(w.store, w.load, broken)
    assert {r.candidate for r in b.rows if r.source == "baseline"} == {"teacher alone"}
    assert any("FileNotFoundError" in n and "no QuickBooks data" in n for n in b.notes)


def test_intuit_sites_say_which_baselines_are_pending() -> None:
    site = "quickbooks.invoice"
    w = World(CallSite(id=site, workflow="quickbooks", name="invoice", kind=Kind.model, labels=["a", "b"], teacher=TeacherSpec(provider="tool")))
    run = TrainRun(id="night:qb:head:real:s0", campaign="night", call_site=site, candidate="head", variant="real", seed=0, where="modal:cpu", status="done", started_at=T0)
    w.store.put_train_run(run)
    w.files["modal://v/p.npz"] = always_right()
    w.store.put_checkpoint(Checkpoint(run_id=run.id, rung=60, weights_sha="w" * 64, eval_sha="x" * 64, scores_uri="modal://v/p.npz", seconds=1, embed_ms=0, head_ms=0.1, bytes=1,
                                      stable=True, summary={}, ts=T0))
    b = table(w.store, w.load, lambda s: [{"name": "rule", "coverage": 0.5, "error": 0.01}])
    assert any("LLM-alone and Jev baselines pending" in n for n in b.notes)
    assert any("no calib, test and forward EvalSets" in n for n in b.notes) and rows_of(b, "head", "real").source == "summary"  # eval sets never written: the reason is named
    b = table(w.store, w.load, lambda s: [{"name": "llm alone", "coverage": 1.0, "error": 0.1}, {"name": "jev", "coverage": 1.0, "error": 0.08}])
    assert not any("pending" in n for n in b.notes)


def test_scores_whose_arrays_disagree_in_length_are_not_scored() -> None:
    w = World()
    bad = always_right() | {"ok": np.ones(5, bool)}
    w.cell("head", 0, {200: bad})
    b = table(w.store, w.load, no_baselines)
    assert rows_of(b, "head").source == "summary" and any("differ in length" in n for n in b.notes)


def test_a_truncated_npz_is_an_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeVolume:
        def read_file(self, path: str) -> Iterator[bytes]:
            yield b"PK not really a zip"

    def from_name(name: str) -> FakeVolume:
        return FakeVolume()

    monkeypatch.setattr(modal.Volume, "from_name", from_name)
    with pytest.raises(OSError, match="r1/200.npz"):
        volume_scores("modal://dreaming-weights/campaign/r1/200.npz")


def test_scores_without_forward_rows_leave_the_repeat_columns_empty_and_say_so() -> None:
    w = World(CallSite(id=SITE, workflow="q", name="invoice", kind=Kind.tool, teacher=TeacherSpec(provider="tool")))
    a = arrays(np.ones(N[0], bool), np.linspace(0.9, 0.1, N[0]), np.ones(N[1], bool), np.linspace(0.9, 0.1, N[1]), with_ok=True)
    w.cell("exact", 0, {500: {k: v[a["split"] != 2] for k, v in a.items()}})  # a cell that kept no forward rows: conf, split and ok still agree in length
    b = table(w.store, w.load, no_baselines)
    exact = rows_of(b, "exact")
    assert exact.source == "scores" and exact.cells[1].test_coverage == 1.0 and [c.forward_coverage for c in exact.cells] == [None] * 4
    assert any("exact seed+real: no repeat-traffic" in n and "fewer than 2 forward rows" in n for n in b.notes)
    row = next(ln for ln in markdown(b).splitlines() if " | exact | " in ln)
    assert [c.strip() for c in row.strip("|").split("|")][6:8] == ["-", "-"]


def test_the_cached_table_expires_after_the_ttl(w: World, monkeypatch: pytest.MonkeyPatch) -> None:
    now = [1000.0]
    monkeypatch.setattr("dreaming_api.benchmark.time", SimpleNamespace(monotonic=lambda: now[0]))
    app = FastAPI()
    app.include_router(benchmark_router(w.store, w.load, no_baselines))
    c = TestClient(app)
    c.get("/benchmark")
    now[0] += 30
    c.get("/benchmark.md")
    assert len(w.loads) == 3  # inside the minute: served from the cache
    now[0] += 31
    c.get("/benchmark")
    assert len(w.loads) == 6  # past it: recomputed from the Volume
