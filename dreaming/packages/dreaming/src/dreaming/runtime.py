"""The native step runner: dispatch for decorated call sites, runs, fast-forward, metrics. Learning lives in learning.py.

One Runtime per process. It binds the registry, keeps one Policy per site (the Learner is the only thing that changes
one; the runtime writes the changed ones to the store on every flush), buffers trace rows and runs and writes them in
batches. Fail open everywhere: a storage failure is one component.error event and never blocks a call.
With speculate=True every step and tool call returns a Pending and runs in a thread pool (speculate.py); the default
returns plain values, exactly as before.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections import deque
from collections.abc import Callable, Generator, Iterable
from datetime import UTC, datetime
from typing import Any

from pydantic_core import to_jsonable_python

from . import registry
from .context import RunCtx, current, run_boundary
from .contracts import (
    Batch,
    CallSite,
    Component,
    Decision,
    EventKind,
    Json,
    Kind,
    Metrics,
    Phase,
    Policy,
    Run,
    TeacherAnswer,
    TraceRow,
)
from .hashing import canonical
from .learning import Learner
from .predictors import Loader, Predictor
from .predictors.train import Trainer
from .router import Answer, Call, Stuck, route
from .speculate import Speculator, hooks, result
from .store import Store
from .teachers import Teacher
from .telemetry import boundary, emit


class Runtime:
    def __init__(self, store: Store, teacher: Teacher, trainer: Trainer, loader: Loader, *,
                 component: Component = Component.agent, budget: float = 0.05, flush_every: int = 200, speculate: bool = False) -> None:
        self.store, self.teacher, self.trainer, self.loader = store, teacher, trainer, loader
        self.component, self.budget, self.flush_every = component, budget, flush_every
        self.policies: dict[str, Policy] = {}
        self.predictors: dict[str, Predictor] = {}  # the promoted candidate per site: the one that answers
        self.spares: dict[str, dict[str, Predictor]] = {}  # the other loaded candidates per site, by name: they shadow every call
        self.sites: dict[str, CallSite] = {}
        self.workflow: Callable[[Json], Any] | None = None  # the function one run executes; set by the app that owns the workflow
        self.learner = Learner(self)
        self.dirty: set[str] = set()  # policies changed since the last flush
        self.batch = Batch()
        self.run_count = 0
        self.saved_usd = 0.0
        self._rows: list[TraceRow] = []  # buffered, written in batches
        self._runs: list[Run] = []
        self._lock = threading.Lock()  # the buffers and run_count: a live /run may land during a batch
        self._recent: deque[Run] = deque(maxlen=100)  # the metrics window
        self._baseline: dict[str, tuple[float, int]] = {}  # site -> (sum of teacher cost, n): what a call costs when awake
        self._by_hash: dict[str, Predictor] = {}  # weights_sha -> predictor, for replay of rows answered by earlier weights
        self._reported: set[str] = set()  # predictor failures already turned into an event
        self._resume = threading.Event()  # cleared by pause(), set by resume(); fast_forward waits on it between runs
        self._resume.set()
        self.spec = Speculator(self._call, lambda row: self._record(row, self.policies[row.call_site])) if speculate else None

    # ------------------------------------------------------------------ binding
    def bind(self) -> None:
        """Declare the registry's sites to the store, load policies and the weights they name, install the dispatch."""
        sites = registry.sites()
        self.store.upsert_sites(sites)
        self.run_count = self.store.run_count()
        self._recent.extend(self.store.runs(limit=100))
        for s in sites:
            self.sites[s.id] = s
            self.policies[s.id] = p = self.store.policy(s.id)
            if p.weights_sha:
                with boundary(self.store, self.component, "weights", swallow=True):
                    self.predictors[s.id] = self.loader.load(s, self.store.weights(p.weights_sha), p.candidate or None)
            for name, spare in p.shadow.items():
                with boundary(self.store, self.component, "weights", swallow=True):
                    self.spares.setdefault(s.id, {})[name] = self.loader.load(s, self.store.weights(spare.weights_sha), name)
        self.learner.check_certificates()
        registry.bind(self._dispatch)

    def declare(self, site: CallSite) -> None:
        """A connector's call site, discovered at runtime (untagged sites are logged only until they get labels)."""
        if site.id not in self.sites:
            self.store.upsert_sites([site])
            self.sites[site.id], self.policies[site.id] = site, self.store.policy(site.id)

    def predictor_for(self, site_id: str, weights_sha: str) -> Predictor:
        """The predictor that answered a logged row: the current one, or earlier weights loaded from the store by hash."""
        for cur in [self.predictors.get(site_id), *self.spares.get(site_id, {}).values()]:
            if cur is not None and cur.weights_sha == weights_sha:
                return cur
        if weights_sha not in self._by_hash:
            self._by_hash[weights_sha] = self.loader.load(self.sites[site_id], self.store.weights(weights_sha))
        return self._by_hash[weights_sha]

    def persist(self, p: Policy) -> None:
        p.updated_at = datetime.now(UTC)
        self.dirty.discard(p.call_site)
        with boundary(self.store, self.component, "store.set_policy", swallow=True):
            self.store.set_policy(p)

    # ------------------------------------------------------------------ one call
    def _dispatch(self, reg: registry.Registered, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if self.spec is not None:
            if reg.site.kind != Kind.effect:
                return self.spec.submit(reg, args, kwargs)
            args, kwargs = tuple(map(result, args)), {k: result(v) for k, v in kwargs.items()}  # effects wait for verified inputs
        return self._call(reg, args, kwargs)

    def _call(self, reg: registry.Registered, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """One call, synchronously: the whole path when speculate=False, one worker's job when it is on."""
        site = reg.site
        if site.kind == Kind.effect:
            return self._route(reg, Call(reg, to_jsonable_python({"args": args, "kwargs": kwargs}, fallback=str), "",
                                         real=lambda: _timed(lambda: reg.fn(*args, **kwargs), site)))
        if site.kind == Kind.tool:
            request = to_jsonable_python({"args": args, "kwargs": kwargs}, fallback=str)
            return self._route(reg, Call(
                reg, request, canonical(request).decode(),
                real=lambda: _timed(lambda: reg.fn(*args, **kwargs), site, label=lambda v: canonical(to_jsonable_python(v, fallback=str)).decode()),
                fallback=(lambda: reg.fallback(*args, **kwargs)) if reg.fallback else None,  # type: ignore[misc]
                decode=json.loads,
            ))
        rendered: str = reg.fn(*args, **kwargs)  # a step renders its input; the library makes the call
        return self._route(reg, Call(reg, rendered, rendered, real=lambda: self._ask(site, rendered)))

    def _ask(self, site: CallSite, rendered: str) -> Answer:
        ans = self.teacher.ask(site, rendered)
        return Answer(value=ans.label or ans.text, label=ans.label or "", latency_ms=ans.latency_ms, cost_usd=ans.cost_usd, teacher=ans)

    def _route(self, reg: registry.Registered, call: Call) -> Any:
        policy = self.policies[reg.site.id]
        predictor, record = hooks(self.predictors.get(reg.site.id), lambda row: self._record(row, policy))  # speculated: tapped, rows held
        return route(call, policy, predictor, current(), record, lambda e: self._predictor_failed(reg.site, e), self.spares.get(reg.site.id))

    def _predictor_failed(self, site: CallSite, e: Exception) -> None:
        """A predictor that raises is a miss (the real call answers) but never a silent one: one event per weights."""
        key = f"{site.id}:{self.policies[site.id].weights_sha}"
        if key not in self._reported:
            self._reported.add(key)
            emit(self.store, self.component, EventKind.component_error, f"the {site.name} predictor raised; the real call answers",
                 call_site=site.id, error=f"{type(e).__name__}: {e}")

    def external(self, site_id: str, request: Json, rendered: str, *, real: Callable[[], Any], label_of: Callable[[Any], str],
                 decode: Callable[[str], Any] = lambda s: s, kind: Kind = Kind.model) -> Any:
        """Connectors enter here. The site is auto-declared with no labels, so it is logged until someone declares them."""
        if site_id not in self.sites:
            workflow, _, name = site_id.partition(".")
            self.declare(CallSite(id=site_id, workflow=workflow or "external", name=name or site_id, kind=kind, labels=None))
        reg = registry.Registered(self.sites[site_id], real)
        return self._route(reg, Call(reg, request, rendered, real=lambda: _timed(real, reg.site, label=label_of), decode=decode))

    def _record(self, row: TraceRow, policy: Policy) -> None:
        self.learner.observe(row, policy)  # every Policy counter, outage detection
        if row.teacher is not None:
            s, n = self._baseline.get(row.call_site, (0.0, 0))
            self._baseline[row.call_site] = (s + row.teacher.cost_usd, n + 1)
            if self.batch.running and not row.teacher.recorded and row.kind in (Kind.model, Kind.router):
                self.batch.live_calls += 1  # a frontier answer that was not pre-recorded; tool calls are real by nature
        if row.decision == Decision.predicted and row.kept is not False:  # a discarded speculative answer saved nothing
            self.saved_usd += self.baseline_cost(row.call_site) - row.cost_usd
        ctx = current()
        if ctx is not None:
            ctx.rows.append(row)
        with self._lock:
            self._rows.append(row)
            full = len(self._rows) >= self.flush_every
        if full:
            self.flush()

    def baseline_cost(self, site_id: str) -> float:
        s, n = self._baseline.get(site_id, (0.0, 0))
        return s / n if n else 0.0

    def flush(self) -> None:
        """Write buffered rows and runs, and the policies that changed (the page reads counters from the store)."""
        with self._lock:
            rows, self._rows = self._rows, []
            runs, self._runs = self._runs, []
            dirty, self.dirty = self.dirty, set()
        if rows:
            with boundary(self.store, self.component, "store.write_rows", swallow=True):  # fail open: a lost batch never blocks a call
                self.store.write_rows(rows)
        if runs:
            with boundary(self.store, self.component, "store.write_runs", swallow=True):
                self.store.write_runs(runs)
        with boundary(self.store, self.component, "store.set_policy", swallow=True):
            for site_id in dirty:
                p = self.policies[site_id]
                p.updated_at = datetime.now(UTC)
                self.store.set_policy(p)

    # ------------------------------------------------------------------ runs
    @contextlib.contextmanager
    def run(self, workflow: str, input: Json, mode: str = "live", seed: int | None = None) -> Generator[Run, None, None]:
        """Open a run around a workflow invocation; on exit the Run row summarizes its trace rows."""
        with self._lock:
            self.run_count += 1
            n = self.run_count
        with run_boundary(workflow, mode, seed=n if seed is None else seed) as ctx:  # type: ignore[arg-type]
            run = Run(id=ctx.id, n=n, workflow=workflow, mode=mode, input=input, started_at=datetime.now(UTC),  # type: ignore[arg-type]
                      down=[s for s, p in self.policies.items() if p.outage])
            try:
                yield run
                if self.spec is not None and (err := self.spec.drain(ctx)) is not None:
                    raise err  # a call nobody waited for failed: the run failed, as it would have at the call
            except Stuck as e:
                run.failed, run.stuck_at = True, str(e).split(" ", 1)[0]
            except Exception as e:  # noqa: BLE001 - a failed run is a row, not a crash of the loop
                run.failed = True
                emit(self.store, self.component, EventKind.component_error, f"run {ctx.id} failed", error=f"{type(e).__name__}: {e}")
            finally:
                if self.spec is not None:
                    self.spec.drain(ctx)  # a run that raised still waits for its calls in flight
                self._close(run, ctx)

    def _close(self, run: Run, ctx: RunCtx) -> None:
        rows = ctx.rows
        run.finished_at = datetime.now(UTC)
        run.calls, run.cost_usd, run.latency_ms = len(rows), sum(r.cost_usd for r in rows), ctx.elapsed_ms()
        run.critical_ms = run.latency_ms  # the virtual clock is the critical path: the sum when calls run one at a time
        used = [r for r in rows if r.kept is not False]  # a discarded speculative row decided nothing
        run.decisions = sum(1 for r in used if r.kind != Kind.effect)
        run.replayable = sum(1 for r in used if r.decision == Decision.predicted and self.sites[r.call_site].replayable)
        with self._lock:
            self._runs.append(run)
            self._recent.append(run)
        if run.mode == "live":
            self.flush()

    def fast_forward(self, workflow: str, inputs: Iterable[Json], fn: Callable[[Json], Any], *, every: int = 20,
                     on_progress: Callable[[Metrics], None] | None = None) -> Metrics:
        """Run many inputs against recorded teachers and in-process tools; learning becomes visible in seconds."""
        self.batch = Batch(running=True, recorded=self.recorded)
        emit(self.store, self.component, EventKind.batch_started,
             f"fast-forward {workflow}: {'pre-recorded' if self.recorded else 'live'} teacher answers, virtual clock")
        n = 0
        try:
            items = list(inputs)
            self.batch.target = len(items)
            for n, inp in enumerate(items, 1):
                self._resume.wait()
                with self.run(workflow, inp, "fast_forward", seed=n):
                    fn(inp)
                self.batch.done = n
                if n % every == 0:
                    self.flush()
                    self.tick()
                    m = self.metrics()
                    emit(self.store, self.component, EventKind.batch_progress, f"{n} runs", data={"metrics": m.model_dump()})
                    if on_progress:
                        on_progress(m)
            self.flush()
            self.tick()
        except Exception as e:
            emit(self.store, self.component, EventKind.component_error, f"fast-forward stopped after {n} runs", data={"dependency": "fast-forward"},
                 error=f"{type(e).__name__}: {e}")
            raise
        finally:
            self.batch = Batch(running=False, target=self.batch.target, done=n, recorded=self.recorded, live_calls=self.batch.live_calls)
            save = getattr(self.loader.encoder, "save_cache", None)  # the embedding cache outlives the process
            if callable(save):
                with boundary(self.store, self.component, "encoder.save_cache", swallow=True):
                    save()
        m = self.metrics()
        emit(self.store, self.component, EventKind.batch_finished, f"done · {n} runs · cost per run {m.cost_per_run * 100:.1f}¢",
             data={"metrics": m.model_dump()})
        return m

    def pause(self) -> None:
        self._resume.clear()
        self.batch.paused = True

    def resume(self) -> None:
        self.batch.paused = False
        self._resume.set()

    @property
    def recorded(self) -> bool:
        return bool(getattr(self.teacher, "recorded", False))  # RecordedTeacher says so; the page labels such batches

    def tick(self) -> None:
        self.learner.tick()

    def set_budget(self, budget: float) -> None:
        self.learner.set_budget(budget)

    # ------------------------------------------------------------------ what the page shows
    def metrics(self) -> Metrics:
        runs = [r for r in list(self._recent) if r.finished_at]
        if not runs:
            return Metrics(runs=self.run_count, cost_per_run=0.0, sec_per_run=0.0, failed_pct=0.0, replayable_pct=0.0, saved_usd=self.saved_usd)
        decisions = sum(r.decisions for r in runs) or 1
        return Metrics(
            runs=self.run_count, cost_per_run=sum(r.cost_usd for r in runs) / len(runs), sec_per_run=sum(r.latency_ms for r in runs) / len(runs) / 1000,
            failed_pct=100 * sum(r.failed for r in runs) / len(runs), replayable_pct=100 * sum(r.replayable for r in runs) / decisions,
            saved_usd=self.saved_usd,
        )

    def costs(self, site_id: str, window: int = 500) -> tuple[float, float, float, float]:
        """(cost awake, cost dreaming, seconds awake, seconds dreaming) per call, from the last rows: the page's "2.1¢ → 0.33¢"."""
        rows = self.store.rows(site_id, limit=window)
        real = [r for r in rows if r.decision in (Decision.called, Decision.audited)]
        # "dreaming" is the blended price of a call once promoted: predicted rows (free), the real calls the dream was unsure
        # about and the audits; predicted rows alone would print "-100%" for every student site (web review, 2026-10-18)
        dream = [r for r in rows if r.phase == Phase.dreaming] or [r for r in rows if r.decision == Decision.predicted]
        return (_mean([r.cost_usd for r in real]), _mean([r.cost_usd for r in dream]),
                _mean([r.latency_ms for r in real]) / 1000, _mean([r.latency_ms for r in dream]) / 1000)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _timed(fn: Callable[[], Any], site: CallSite, label: Callable[[Any], str] = lambda v: str(v)) -> Answer:
    """Run a real call and record it as what it is: the teacher. Reality is the teacher for tools; a connector's
    frontier call is the teacher for its site. Cost is the site's per-call price (tools) or unknown (connectors)."""
    t0, ctx = time.perf_counter(), current()
    try:
        v = fn()
    except BaseException:
        if ctx is not None:  # a charged latency belongs to the call that failed, not to the next row
            ctx.spend(ctx.pending_ms)
            ctx.pending_ms = 0
        raise
    ms, lab = round((time.perf_counter() - t0) * 1000), label(v)
    if ctx is not None:  # a mock tool charged virtual latency (context.charge): it belongs to this row
        ms, ctx.pending_ms = ms + ctx.pending_ms, 0
    t = TeacherAnswer(text=lab, label=lab, provider=site.teacher.provider, model=site.teacher.model or site.id, cost_usd=site.teacher.price_call, latency_ms=ms)
    return Answer(value=v, label=lab, latency_ms=ms, cost_usd=t.cost_usd, teacher=t)


__all__ = ["Runtime", "Trainer"]
