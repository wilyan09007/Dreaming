"""The API: control plane for the dashboard, proxy for connector agents, host of fast-forward mode, and the page itself.

Routes (bodies and replies are contracts.py models, so the OpenAPI schema is the browser's contract):

    GET  /healthz                      Health: db probed ("ok" or the error); trainer, teacher, realtime reported as configured
    GET  /state                        State: sites (policy, certificate, cost awake vs dreaming), budget, metrics, series, runs, batch, config
    POST /batch {n}                    start fast-forward of n transactions (409 while one runs)
    POST /batch/pause · /batch/resume  the run button's pause and resume
    POST /budget {budget}              re-certify every site at the new budget (pure gate, milliseconds); 422 below the teacher's measured floor
    POST /tools/{site}/outage {down}   unplug / plug the mock tool (fault injection; the runtime detects it, the page sees tool.outage)
    POST /run {input}                  one live run of the workflow (the uAgent relays ASI:One chats here); returns RunView
    POST /replay/{run_id}              ReplayReport
    GET  /runs/{run_id}                RunView: the run and its TraceRows (the replay tab's receipt)
    GET  /events?after=&limit=         polling fallback for Supabase Realtime
    POST /v1/systemone                 TypeSafe System One wire format over the trained students (systemone.py)
    POST /campaign/run · GET /campaign/{name}   the Modal campaign's launch and progress table (campaign_routes.py)
    POST /v1/chat/completions          OpenAI-compatible proxy (connectors/proxy.py), mounted by wiring when a key exists
    GET  /                             the dashboard (apps/web, StaticFiles, mounted last)

One runtime owner: this process holds the policies; nothing else writes them. shortcut: one fast-forward at a time, in a
background thread; a second API machine would need the batch lock in Postgres.
"""

from __future__ import annotations

import itertools
import os
import threading
import uuid
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path

from dreaming.contracts import (
    Component,
    Event,
    EventKind,
    Health,
    Json,
    Metrics,
    Policy,
    ReplayReport,
    RunView,
    SiteView,
    State,
)
from dreaming.predictors import env_profile
from dreaming.replay import replay
from dreaming.runtime import Runtime
from dreaming.telemetry import TRACE, emit
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .campaign_routes import router as campaign_router
from .systemone import systemone_router
from .wiring import build_runtime, checks


class BatchIn(BaseModel):
    n: int = 2000


class BudgetIn(BaseModel):
    budget: float  # 0.01 … 0.10


class OutageIn(BaseModel):
    down: bool


class RunIn(BaseModel):
    input: Json


def create_app(runtime: Runtime | None = None, inputs: Iterable[Json] | None = None, inject: Callable[[str, bool], None] | None = None) -> FastAPI:
    """`inputs`: the stream fast-forward draws from. `inject(site, down)`: flips the mock tool's fault switch for that site."""
    rt = runtime or build_runtime(Component.api)
    stream = iter(inputs) if inputs is not None else None
    app = FastAPI(title="dreaming", version=os.environ.get("DREAMING_VERSION", "dev"))
    lock = threading.Lock()

    @app.middleware("http")
    async def trace(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """Every event emitted while handling this request carries the W3C traceparent trace id (or a fresh one)."""
        header = request.headers.get("traceparent", "")
        token = TRACE.set(header.split("-")[1] if header.count("-") == 3 else uuid.uuid4().hex)
        try:
            return await call_next(request)
        finally:
            TRACE.reset(token)

    @app.get("/healthz")
    def healthz() -> Health:
        c = checks(rt)
        return Health(component=Component.api, ok=c.get("db") == "ok", checks=c, version=app.version,  # only the db is probed
                      env_sha=os.environ.get("MODAL_IMAGE_ID") or os.environ.get("DREAMING_IMAGE_DIGEST", "dev"),
                      profile={k: str(v) for k, v in env_profile().items()})

    @app.get("/state")
    def state() -> State:
        views: list[SiteView] = []
        for s in rt.sites.values():
            p = rt.policies[s.id]
            ca, cd, sa, sd = rt.costs(s.id)
            views.append(SiteView(site=s, policy=p, certificate=rt.store.certificate(p.cert_id) if p.cert_id else None,
                                  cost_awake_usd=ca, cost_dream_usd=cd, sec_awake=sa, sec_dream=sd))
        series = [Metrics.model_validate(e.data["metrics"]) for e in rt.store.events(limit=2000) if e.kind == "batch.progress"]
        config = {k: os.environ[v] for k, v in (("supabase_url", "SUPABASE_URL"), ("supabase_publishable_key", "SUPABASE_PUBLISHABLE_KEY")) if os.environ.get(v)}
        return State(sites=views, budget=rt.budget, metrics=rt.metrics(), series=series, runs=rt.store.runs(limit=50),
                     first_run=rt.store.first_run(), batch=rt.batch, config=config)

    @app.post("/batch")
    def batch(body: BatchIn) -> dict[str, str]:
        if stream is None or rt.workflow is None:
            raise HTTPException(503, "no workflow inputs configured on this API")
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "a batch is already running")
        fn, chunk = rt.workflow, itertools.islice(stream, body.n)  # drawn inside the thread: the request returns at once

        def work() -> None:
            try:
                rt.fast_forward("bookkeeper", chunk, fn)
            except Exception:  # noqa: BLE001 - the runtime already emitted the component.error; the thread just ends
                return
            finally:
                lock.release()

        threading.Thread(target=work, name="fast-forward", daemon=True).start()
        return {"status": "started"}

    @app.post("/batch/pause")
    def pause() -> dict[str, bool]:
        rt.pause()
        return {"paused": True}

    @app.post("/batch/resume")
    def resume() -> dict[str, bool]:
        rt.resume()
        return {"paused": False}

    @app.post("/budget")
    def budget(body: BudgetIn) -> dict[str, float]:
        floor = max((s.teacher.floor or 0.0 for s in rt.sites.values()), default=0.0)  # the teacher's self-disagreement bounds every certificate
        if body.budget < floor:
            raise HTTPException(422, f"budget {body.budget:.0%} is below the teacher's measured self-disagreement {floor:.1%}: no certificate at that level is measurable")
        rt.set_budget(body.budget)
        return {"budget": rt.budget}

    @app.post("/tools/{site}/outage")
    def outage(site: str, body: OutageIn) -> Policy:
        if site not in rt.sites:
            raise HTTPException(404, f"unknown call site {site}")
        if inject is None:
            raise HTTPException(503, "this API has no fault switch for its tools")
        inject(site, body.down)
        return rt.policies[site]

    @app.post("/run")
    def run_once(body: RunIn) -> RunView:
        if rt.workflow is None:
            raise HTTPException(503, "no workflow configured on this API")
        with rt.run("bookkeeper", body.input, "live") as r:
            rt.workflow(body.input)
        return RunView(run=r, rows=rt.store.run_rows(r.id))

    @app.post("/replay/{run_id}")
    def do_replay(run_id: str) -> ReplayReport:
        if not rt.store.run_rows(run_id):
            raise HTTPException(404, run_id)
        try:
            report = replay(rt.store, run_id, rt.predictors, rt.predictor_for)
        except LookupError as e:  # the weights or environment a row names are not available here: say which, not 500
            raise HTTPException(409, str(e)) from e
        emit(rt.store, Component.api, EventKind.replay_finished, f"replayed run {run_id}: {report.same} of {report.re_executed} same bytes",
             run_id=run_id, data={"report": report.model_dump()})
        return report

    @app.get("/runs/{run_id}")
    def run(run_id: str) -> RunView:
        rows = rt.store.run_rows(run_id)
        match = [r for r in rt.store.runs(limit=10_000) if r.id == run_id]
        if not rows or not match:
            raise HTTPException(404, run_id)
        return RunView(run=match[0], rows=rows)

    @app.get("/events")
    def events(after: str | None = None, limit: int = Query(200, ge=1, le=1000)) -> list[Event]:
        return rt.store.events(after_id=after, limit=limit)

    app.include_router(systemone_router(rt))
    app.include_router(campaign_router)

    web = Path(__file__).resolve().parents[3] / "web"
    if (web / "index.html").is_file():
        app.mount("/", StaticFiles(directory=web, html=True), name="web")  # last: everything above wins
    return app
