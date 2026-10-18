"""The API contract, offline: every route of main.py against a real Runtime on MemoryStore with fakes.

The loop is the one in packages/dreaming/tests/test_runtime.py (two sites, `books.categorize` and `books.vendor_lookup`),
driven through HTTP. A batch runs in a background thread of the app, so tests either hold its first run inside the
workflow (a batch that is provably mid-run) or poll until the thread has finished.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import NoReturn

import pytest
from dreaming.contracts import (
    Batch,
    Component,
    Decision,
    Event,
    EventKind,
    Health,
    Json,
    Kind,
    Phase,
    Policy,
    ReplayReport,
    RunView,
    SiteView,
    State,
)
from dreaming.predictors.candidates import DEFAULT
from dreaming.runtime import Runtime
from dreaming_api import main
from fastapi.testclient import TestClient
from pydantic import TypeAdapter
from starlette.routing import Mount

CAT, VEN = "books.categorize", "books.vendor_lookup"
Clients = Callable[..., TestClient]  # make_client(inputs, *, faults=True, raise_server_exceptions=True)
Txns = Callable[..., list[dict[str, Json]]]


# ------------------------------------------------------------------ helpers


def wait_until(cond: Callable[[], bool], what: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out after {timeout:.0f}s waiting for {what}")


def batch_running() -> bool:
    return any(t.name == "fast-forward" and t.is_alive() for t in threading.enumerate())


def join_batch() -> None:
    """Join the background thread by polling (it is a daemon thread of the app, not ours to join)."""
    wait_until(lambda: not batch_running(), "the fast-forward thread to finish")


def state(client: TestClient) -> State:
    r = client.get("/state")
    assert r.status_code == 200, r.text
    return State.model_validate(r.json())


def site(s: State, site_id: str) -> SiteView:
    return next(v for v in s.sites if v.site.id == site_id)


def events(client: TestClient, **params: str | int) -> list[Event]:
    r = client.get("/events", params=params)
    assert r.status_code == 200, r.text
    return TypeAdapter(list[Event]).validate_python(r.json())


def kinds(client: TestClient) -> list[EventKind]:
    return [e.kind for e in events(client, limit=1000)]


def promoted(s: State) -> bool:
    return bool(s.series) and site(s, CAT).policy.phase == Phase.dreaming


def batch(client: TestClient, n: int) -> State:
    """POST /batch, wait for the thread to finish."""
    r = client.post("/batch", json={"n": n})
    assert r.status_code == 200 and r.json() == {"status": "started"}, r.text
    join_batch()
    return state(client)


def train(client: TestClient, n: int = 600) -> State:
    """POST /batch, poll /state until the series has points and categorize is dreaming, then let the batch finish."""
    r = client.post("/batch", json={"n": n})
    assert r.status_code == 200 and r.json() == {"status": "started"}, r.text
    wait_until(lambda: promoted(state(client)), "a series and a dreaming categorize")
    join_batch()
    return state(client)


def mounts(app: object) -> list[str | None]:
    return [r.name for r in getattr(app, "routes", []) if isinstance(r, Mount)]


class Hold:
    """Blocks the workflow inside run 1 until `gate` is set: a batch that is provably mid-run."""

    def __init__(self, rt: Runtime) -> None:
        assert rt.workflow is not None
        inner = rt.workflow
        self.entered, self.gate = threading.Event(), threading.Event()

        def held(txn: Json) -> object:
            self.entered.set()
            assert self.gate.wait(10), "the test never released the batch"
            return inner(txn)

        rt.workflow = held


# ------------------------------------------------------------------ /healthz


def test_healthz_shape(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    h = Health.model_validate(r.json())  # extra="forbid": no field the model does not know
    assert set(r.json()) == set(Health.model_fields)  # and every field is there
    assert h.component == Component.api and h.ok is True
    assert h.version == "dev" and h.env_sha == "dev"
    assert set(h.checks) == {"db", "trainer", "teacher", "realtime"}
    assert h.checks["db"] == "ok"
    # Health.checks: "ok" or "ok (detail)" is green, anything else is the error the page shows in red
    assert all(v == "ok" or v.startswith(("ok (", "configured (")) for v in h.checks.values()), h.checks
    assert h.profile["image"] == "dev" and {"python", "device"} <= set(h.profile)  # the execution profile behind env_sha


def test_healthz_names_the_red_dependency(rt: Runtime, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def down() -> NoReturn:
        raise ConnectionError("pg gone")

    monkeypatch.setattr(rt.store, "sites", down)
    h = Health.model_validate(client.get("/healthz").json())
    assert h.ok is False
    assert h.checks["db"] == "ConnectionError: pg gone"
    assert h.checks["trainer"].startswith("configured") and h.checks["teacher"].startswith("configured")


def test_healthz_reports_deployment_environment(make_client: Clients, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DREAMING_VERSION", "1.2.3")
    monkeypatch.setenv("MODAL_IMAGE_ID", "im-abc")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    h = Health.model_validate(make_client(None).get("/healthz").json())
    assert h.ok and h.version == "1.2.3" and h.env_sha == "im-abc" and h.profile["image"] == "im-abc"
    assert h.checks["realtime"] == "configured (supabase)"


# ------------------------------------------------------------------ /state


def test_state_before_any_run(client: TestClient) -> None:
    s = state(client)
    assert {v.site.id for v in s.sites} == {CAT, VEN}
    for v in s.sites:
        assert v.policy.phase == Phase.awake and v.policy.rows == 0 and v.policy.shadow_rows == 0 and v.policy.attempt is None
        assert v.policy.cert_id is None and v.policy.cutoff is None and v.certificate is None and not v.policy.outage
        assert (v.cost_awake_usd, v.cost_dream_usd, v.sec_awake, v.sec_dream) == (0.0, 0.0, 0.0, 0.0)
    assert s.budget == 0.05
    assert s.metrics.model_dump() == {"runs": 0, "cost_per_run": 0.0, "sec_per_run": 0.0, "failed_pct": 0.0, "replayable_pct": 0.0, "saved_usd": 0.0}
    assert s.series == [] and s.runs == []
    assert s.batch == Batch() and s.config == {}  # idle, nothing recorded, no Realtime settings
    assert events(client) == []


def test_state_config_hands_the_page_its_realtime_settings(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    assert state(client).config == {"supabase_url": "https://example.supabase.co"}
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "sb_publishable_x")
    assert state(client).config == {"supabase_url": "https://example.supabase.co", "supabase_publishable_key": "sb_publishable_x"}


def test_batch_learns_and_state_tells_the_story(client: TestClient) -> None:
    s = train(client)

    cat = site(s, CAT)
    cert = cat.certificate
    assert cat.policy.phase == Phase.dreaming and cat.policy.rows == 600 and cat.policy.predicted_rows > 0
    assert cert is not None and cert.id == cat.policy.cert_id and cert.call_site == CAT and cert.status == "active"
    assert cert.bound <= cert.budget == s.budget
    assert cat.policy.cutoff == cert.cutoff and cat.policy.weights_sha == cert.weights_sha
    # "2.1¢ -> 0.33¢": what a call costs awake vs dreaming, from the logged rows
    assert cat.cost_awake_usd == pytest.approx(0.01) and cat.sec_awake == pytest.approx(0.8)
    assert cat.cost_dream_usd < cat.cost_awake_usd and cat.sec_dream < cat.sec_awake

    # "the more it runs": one point per 20 runs (x = runs so far), cumulative savings only grow, cost per run falls,
    # and the last point is the headline
    assert [m.runs for m in s.series] == list(range(20, 601, 20))
    assert [m.saved_usd for m in s.series] == sorted(m.saved_usd for m in s.series)
    assert s.series[-1].cost_per_run < s.series[0].cost_per_run
    assert s.metrics == s.series[-1] and s.metrics.runs == 600
    assert s.metrics.replayable_pct > 30 and s.metrics.saved_usd > 0 and s.metrics.cost_per_run < 0.02

    assert [r.n for r in s.runs] == list(range(551, 601))  # the newest 50, newest last
    assert all(r.mode == "fast_forward" and r.finished_at for r in s.runs)
    assert s.batch == Batch(running=False, paused=False, target=600, done=600, recorded=True)  # recorded: the page says so

    ks = kinds(client)
    assert ks[0] == EventKind.batch_started and ks[-1] == EventKind.batch_finished
    assert ks.count(EventKind.batch_progress) == len(s.series)
    assert EventKind.cert_issued in ks and EventKind.component_error not in ks


@pytest.mark.parametrize("recorded", [False])
def test_batch_is_not_labelled_recorded_when_the_teacher_is_live(client: TestClient) -> None:
    s = train(client)
    assert site(s, CAT).policy.phase == Phase.dreaming
    assert s.batch.recorded is False and s.batch.done == 600


@pytest.mark.parametrize(("recorded", "answers"), [(True, None), (False, None), (False, "offline rule teacher answers")])
def test_batch_started_says_whose_answers(rt: Runtime, client: TestClient, recorded: bool, answers: str | None) -> None:
    if answers:
        rt.teacher.answers = answers  # type: ignore[attr-defined]  # what offline.RuleTeacher declares; a live teacher declares nothing
    batch(client, 20)
    want = "pre-recorded teacher answers" if recorded else answers or "live teacher answers"
    assert [e.message for e in events(client, limit=1000) if e.kind == EventKind.batch_started] == [f"fast-forward bookkeeper: {want}, virtual clock"]


# ------------------------------------------------------------------ POST /batch, /batch/pause, /batch/resume


def test_second_batch_while_one_runs_is_409(rt: Runtime, make_client: Clients, txns: Txns) -> None:
    hold = Hold(rt)
    client = make_client(txns(60))
    assert client.post("/batch", json={"n": 30}).status_code == 200
    assert hold.entered.wait(10)

    busy = client.post("/batch", json={"n": 30})
    assert busy.status_code == 409 and "already running" in busy.json()["detail"]
    assert state(client).batch == Batch(running=True, target=30, done=0, recorded=True)  # reads are never blocked by the batch

    hold.gate.set()
    join_batch()
    assert client.post("/batch", json={"n": 30}).status_code == 200  # the lock was released: the next 30 of the stream
    join_batch()
    finished = [e.message for e in events(client, limit=1000) if e.kind == EventKind.batch_finished]
    assert len(finished) == 2 and all("30 runs" in m for m in finished)


def test_pause_holds_the_batch_between_runs_until_resume(rt: Runtime, make_client: Clients, txns: Txns) -> None:
    hold = Hold(rt)
    client = make_client(txns(40))
    assert client.post("/batch", json={"n": 40}).status_code == 200
    assert hold.entered.wait(10)

    assert client.post("/batch/pause").json() == {"paused": True}
    assert state(client).batch.paused is True
    hold.gate.set()  # run 1 finishes; the loop then waits for resume before run 2
    wait_until(lambda: state(client).batch.done == 1, "run 1 to finish")
    time.sleep(0.2)
    assert state(client).batch == Batch(running=True, paused=True, target=40, done=1, recorded=True)

    assert client.post("/batch/resume").json() == {"paused": False}
    join_batch()
    assert state(client).batch == Batch(running=False, paused=False, target=40, done=40, recorded=True)


def test_batch_without_inputs_is_503_and_keeps_no_lock(make_client: Clients) -> None:
    client = make_client(None)
    for _ in range(2):  # a leaked lock would turn the second answer into a 409
        r = client.post("/batch", json={"n": 10})
        assert r.status_code == 503 and "no workflow inputs" in r.json()["detail"]
    assert not batch_running()


def test_batch_without_a_workflow_is_503(rt: Runtime, client: TestClient) -> None:
    rt.workflow = None
    assert client.post("/batch", json={"n": 10}).status_code == 503
    assert not batch_running() and kinds(client) == []


def test_batch_larger_than_the_inputs_still_finishes(make_client: Clients, txns: Txns) -> None:
    client = make_client(txns(30))
    s = batch(client, 50)
    finished = [e for e in events(client, limit=1000) if e.kind == EventKind.batch_finished]
    assert len(finished) == 1 and "30 runs" in finished[0].message
    assert s.batch == Batch(running=False, target=30, done=30, recorded=True) and len(s.runs) == 30
    assert all(e.error is None for e in events(client, limit=1000))


def test_a_broken_input_stream_leaves_a_component_error_and_no_lock(make_client: Clients, txns: Txns) -> None:
    def feed() -> Iterator[Json]:
        yield from txns(25)
        raise ValueError("ledger exhausted")

    client = make_client(feed())
    assert client.post("/batch", json={"n": 100}).status_code == 200
    join_batch()
    errors = [e for e in events(client, limit=1000) if e.kind == EventKind.component_error]
    assert len(errors) == 1 and errors[0].component == Component.api
    assert errors[0].error == "ValueError: ledger exhausted" and errors[0].data == {"dependency": "fast-forward"}
    assert client.post("/batch", json={"n": 10}).status_code == 200  # the lock was released
    join_batch()


def test_batch_body_is_validated(client: TestClient) -> None:
    assert client.post("/batch", json={"n": "many"}).status_code == 422
    assert not batch_running()


# ------------------------------------------------------------------ POST /budget


def test_budget_recertifies_and_restores(client: TestClient) -> None:
    s0 = train(client)
    before = site(s0, CAT)
    assert before.certificate is not None

    r = client.post("/budget", json={"budget": 0.001})
    assert r.status_code == 200 and r.json() == {"budget": 0.001}
    s1 = state(client)
    tight = site(s1, CAT)
    assert s1.budget == 0.001
    assert (tight.policy.phase, tight.policy.cert_id) != (before.policy.phase, before.policy.cert_id)
    # a promise of 0.1% needs thousands of clean calibration rows: the certificate is revoked, the real call answers again
    assert tight.policy.phase == Phase.testing and tight.policy.cert_id is None and tight.certificate is None
    assert tight.policy.weights_sha == before.policy.weights_sha  # the gate re-ran, nothing retrained
    att = tight.policy.attempt  # the page's "no promise at 0.1%" numbers
    assert att is not None and att.budget == 0.001 and att.n > 0 and att.bound > 0.001
    assert att.min_budget is not None and 0.001 < att.min_budget <= 0.05

    r = client.post("/budget", json={"budget": 0.05})
    assert r.status_code == 200 and r.json() == {"budget": 0.05}
    s2 = state(client)
    back = site(s2, CAT)
    assert s2.budget == 0.05 and back.policy.phase == Phase.dreaming and back.certificate is not None and back.policy.attempt is None
    assert back.policy.cert_id not in (None, before.policy.cert_id)  # a fresh promise ...
    assert back.certificate.calib_sha == before.certificate.calib_sha and back.certificate.bound <= 0.05  # ... on the same stored rows
    assert back.policy.weights_sha == before.policy.weights_sha

    evs = events(client, limit=1000)
    assert [e.data["budget"] for e in evs if e.kind == EventKind.budget_changed] == [0.001, 0.05]
    assert EventKind.cert_revoked in [e.kind for e in evs]


def test_budget_body_is_validated(client: TestClient) -> None:
    assert client.post("/budget", json={"budget": "lots"}).status_code == 422
    assert client.post("/budget", json={}).status_code == 422
    assert state(client).budget == 0.05


# ------------------------------------------------------------------ POST /tools/{site}/outage


def test_unplugging_the_tool_is_noticed_from_the_failures_and_plugging_it_back_from_a_probe(client: TestClient, down: dict[str, bool]) -> None:
    r = client.post(f"/tools/{VEN}/outage", json={"down": True})
    assert r.status_code == 200
    p = Policy.model_validate(r.json())
    assert p.call_site == VEN and down["vendor"] is True
    assert p.outage is False  # the switch is fault injection: the runtime has not seen a failure yet

    s = batch(client, 10)  # awake, snapshot fallback registered: the runs go on, three failures in a row make it an outage
    assert site(s, VEN).policy.outage is True and site(s, CAT).policy.outage is False
    assert not any(r.failed for r in s.runs) and s.runs[-1].down == [VEN]
    outages = [e for e in events(client, limit=1000) if e.kind == EventKind.tool_outage]
    assert len(outages) == 1 and outages[0].call_site == VEN

    assert client.post(f"/tools/{VEN}/outage", json={"down": False}).status_code == 200 and down["vendor"] is False
    for _ in range(6):  # while out, 1 call in 20 probes the real call; a success clears the outage
        s = batch(client, 50)
        if not site(s, VEN).policy.outage:
            break
    assert site(s, VEN).policy.outage is False and EventKind.tool_recovered in kinds(client)


def test_outage_while_dreaming_keeps_the_runs_alive(client: TestClient) -> None:
    s0 = train(client)
    assert site(s0, VEN).policy.phase == Phase.dreaming
    assert client.post(f"/tools/{VEN}/outage", json={"down": True}).status_code == 200

    new = batch(client, 50).runs  # /state shows the newest 50: exactly this batch
    assert [r.n for r in new] == list(range(601, 651)) and not any(r.failed or r.stuck_at for r in new)
    rows = [row for r in new for row in RunView.model_validate(client.get(f"/runs/{r.id}").json()).rows if row.call_site == VEN]
    decisions = {row.decision for row in rows}
    assert decisions <= {Decision.predicted, Decision.fallback} and Decision.predicted in decisions  # the memory answers what it knows


def test_outage_without_a_fault_switch_is_503(make_client: Clients, txns: Txns) -> None:
    client = make_client(txns(10), faults=False)
    r = client.post(f"/tools/{VEN}/outage", json={"down": True})
    assert r.status_code == 503 and "fault switch" in r.json()["detail"]
    assert client.post("/tools/books.nope/outage", json={"down": True}).status_code == 404  # an unknown site is still a 404


def test_outage_body_is_validated(client: TestClient, down: dict[str, bool]) -> None:
    assert client.post(f"/tools/{VEN}/outage", json={"down": "maybe"}).status_code == 422
    assert client.post(f"/tools/{VEN}/outage", json={}).status_code == 422
    assert down["vendor"] is False


def test_unknown_site_is_404(client: TestClient, down: dict[str, bool]) -> None:
    r = client.post("/tools/books.nope/outage", json={"down": True})
    assert r.status_code == 404 and "books.nope" in r.json()["detail"]
    assert events(client) == [] and down["vendor"] is False  # nothing was unplugged


# ------------------------------------------------------------------ POST /run, GET /runs, POST /replay


def test_run_executes_one_live_run_and_returns_its_receipt(client: TestClient, txns: Txns) -> None:
    txn = txns(1)[0]
    r = client.post("/run", json={"input": txn})
    assert r.status_code == 200
    rv = RunView.model_validate(r.json())
    assert rv.run.mode == "live" and rv.run.input == txn and rv.run.finished_at is not None and not rv.run.failed
    assert rv.run.n == 1 and rv.run.calls == 2 == len(rv.rows) and rv.run.decisions == 2
    assert [(row.i, row.call_site) for row in rv.rows] == [(1, CAT), (2, VEN)] and {row.run_id for row in rv.rows} == {rv.run.id}
    assert all(row.decision == Decision.called and row.teacher is not None for row in rv.rows)  # awake: the real calls answered
    # a live run is flushed at once: the receipt is readable from the store, and the next run is number 2
    assert RunView.model_validate(client.get(f"/runs/{rv.run.id}").json()) == rv
    assert state(client).runs[-1].id == rv.run.id
    assert client.post("/run", json={"input": txn}).json()["run"]["n"] == 2


def test_run_without_a_workflow_is_503_and_needs_an_input(rt: Runtime, client: TestClient, txns: Txns) -> None:
    assert client.post("/run", json={}).status_code == 422
    rt.workflow = None
    r = client.post("/run", json={"input": txns(1)[0]})
    assert r.status_code == 503 and state(client).runs == []


def test_run_detail_is_the_run_and_its_rows_in_call_order(client: TestClient) -> None:
    run = train(client).runs[-1]
    r = client.get(f"/runs/{run.id}")
    assert r.status_code == 200
    rv = RunView.model_validate(r.json())
    assert rv.run == run and run.n == 600
    assert run.calls == 2 and [row.i for row in rv.rows] == [1, 2]  # 1-based, in order
    assert {row.run_id for row in rv.rows} == {run.id} and [row.call_site for row in rv.rows] == [CAT, VEN]


def test_run_detail_names_the_rows_that_voted_for_a_memory_answer(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(DEFAULT, Kind.model, ["memory"])  # the fakes promote the head; make the memory the one that answers
    run = next(r for r in reversed(train(client).runs) if r.replayable)
    rv = RunView.model_validate(client.get(f"/runs/{run.id}").json())
    p = next(row.prediction for row in rv.rows if row.call_site == CAT and row.decision == Decision.predicted)
    assert p is not None and p.candidate == "memory" and p.evidence and len(p.evidence) == 3
    assert {e.row_id for e in p.evidence} == set(rv.evidence_texts)  # every voter is a logged call here (no seed rows in the fakes)
    assert all(t.startswith("Transaction: ") for t in rv.evidence_texts.values())


def test_replay_reexecutes_predicted_rows_byte_for_byte(client: TestClient) -> None:
    run = next(r for r in reversed(train(client).runs) if r.replayable)
    r = client.post(f"/replay/{run.id}")
    assert r.status_code == 200
    rep = ReplayReport.model_validate(r.json())
    assert rep.run_id == run.id
    assert rep.re_executed == run.replayable > 0 and rep.same == rep.re_executed
    assert rep.from_log == run.calls - run.replayable and [st.i for st in rep.steps] == list(range(1, run.calls + 1))
    for st in rep.steps:
        assert (st.same_bytes, st.output_sha is not None) == ((True, True) if st.re_executed else (None, False))
        assert st.re_executed == (st.decision == Decision.predicted)


def test_replay_after_a_restart_loads_the_weights_by_hash(rt: Runtime, client: TestClient) -> None:
    run = next(r for r in reversed(train(client).runs) if r.replayable)
    rt.predictors.clear()  # a process that has not loaded any predictor yet
    rep = ReplayReport.model_validate(client.post(f"/replay/{run.id}").json())
    assert rep.re_executed == run.replayable > 0 and rep.same == rep.re_executed


def test_replay_without_the_weights_is_409_not_500(rt: Runtime, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    run = next(r for r in reversed(train(client).runs) if r.replayable)
    rt.predictors.clear()

    def gone(weights_sha: str) -> bytes:
        raise KeyError(weights_sha)

    monkeypatch.setattr(rt.store, "weights", gone)  # the weights a row names are nowhere to be found
    r = client.post(f"/replay/{run.id}")
    assert r.status_code == 409 and r.json()["detail"]


def test_first_run_stays_readable_past_the_runs_window(rt: Runtime, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    first = batch(client, 40).first_run
    assert first is not None
    newest = rt.store.runs(limit=5)
    monkeypatch.setattr(rt.store, "runs", lambda *, limit=50: newest[-limit:])  # as after 10,000+ runs: run 1 is out of the window
    assert client.get(f"/runs/{first.id}").status_code == 200  # /state still lists it on the replay tab


def test_unknown_run_is_404(client: TestClient) -> None:
    batch(client, 40)
    assert client.get("/runs/run_nope").status_code == 404
    assert client.post("/replay/run_nope").status_code == 404  # not a vacuously perfect "0 of 0 re-executed" report


def test_rows_without_a_run_row_are_404_not_500(rt: Runtime, client: TestClient) -> None:
    run = batch(client, 40).runs[-1]
    orphan = rt.store.run_rows(run.id)[0].model_copy(update={"run_id": "run_orphan"})
    rt.store.write_rows([orphan])  # the runtime swallows a failed write_runs (fail open): rows can outlive their Run
    assert client.get("/runs/run_orphan").status_code == 404


# ------------------------------------------------------------------ GET /events


def test_events_page_by_cursor(client: TestClient) -> None:
    train(client)
    everything = events(client, limit=1000)
    assert len(everything) > 40 and len({e.id for e in everything}) == len(everything)
    assert [e.ts for e in everything] == sorted(e.ts for e in everything)

    assert len(events(client)) == min(len(everything), 200)  # default limit
    assert events(client, limit=3) == everything[-3:]  # no cursor: the newest, oldest first
    first = everything[0]
    assert events(client, after=first.id, limit=4) == everything[1:5]  # a cursor: the page right after it
    assert events(client, after=everything[-1].id) == []  # caught up

    walked = [first]  # the polling fallback: walk the whole log, 7 at a time
    while page := events(client, after=walked[-1].id, limit=7):
        assert len(page) <= 7
        walked += page
    assert walked == everything


def test_events_limit_must_be_positive(client: TestClient) -> None:
    batch(client, 40)
    for bad in (0, -1):  # 0 would otherwise page the whole log
        assert client.get("/events", params={"limit": bad}).status_code == 422


# ------------------------------------------------------------------ the page


def fake_repo(tmp_path: Path, *, index: bool) -> str:
    """A repo layout whose apps/web exists (today: just a .gitkeep) and may hold the page. Returns the fake path of main.py."""
    web = tmp_path / "apps" / "web"
    web.mkdir(parents=True)
    (web / ".gitkeep").touch()
    if index:
        (web / "index.html").write_text("<h1>dreaming</h1>", encoding="utf-8")
    return str(tmp_path / "apps" / "api" / "src" / "dreaming_api" / "main.py")


def test_no_static_mount_without_index_html(rt: Runtime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "__file__", fake_repo(tmp_path, index=False))  # create_app finds apps/web from its own location
    app = main.create_app(runtime=rt)
    assert mounts(app) == []
    c = TestClient(app)
    assert c.get("/").status_code == 404
    assert c.post("/nope").status_code == 404  # a "/" mount would answer 405 here


def test_static_mount_serves_index_html_and_loses_to_the_api(rt: Runtime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "__file__", fake_repo(tmp_path, index=True))
    app = main.create_app(runtime=rt)
    c = TestClient(app)
    assert mounts(app) == ["web"] and "<h1>dreaming</h1>" in c.get("/").text
    assert c.get("/healthz").json()["component"] == "api" and c.get("/state").status_code == 200  # mounted last


def test_the_static_mount_tracks_the_page_file(rt: Runtime) -> None:
    """The page is mounted exactly when apps/web/index.html exists (it does now: the developer view in apps/web)."""
    web = Path(main.__file__).resolve().parents[3] / "web"  # apps/web
    assert web.parent.name == "apps"
    assert (mounts(main.create_app(runtime=rt)) == ["web"]) == (web / "index.html").is_file()


# ------------------------------------------------------------------ the contract


def test_openapi_lists_the_documented_routes(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    served = {(path, method) for path, ops in paths.items() for method in ops}
    documented = {
        ("/healthz", "get"), ("/state", "get"), ("/batch", "post"), ("/batch/pause", "post"), ("/batch/resume", "post"),
        ("/budget", "post"), ("/tools/{site}/outage", "post"), ("/run", "post"), ("/replay/{run_id}", "post"), ("/runs/{run_id}", "get"),
        ("/events", "get"),
    }  # /v1/chat/completions is in the module docstring too, but nothing mounts proxy_router (see the report)
    assert documented <= served
