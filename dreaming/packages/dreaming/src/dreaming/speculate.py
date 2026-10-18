"""Lazy futures for the native runtime: with Runtime(speculate=True) a call returns a Pending at once (docs/speculation.md §3).

The work runs in a thread pool, so independent calls overlap with nothing declared: the data flow is the graph. A dependent
call renders on an input's guess (the predictor's top label, top-1 only) when the guess's confidence clears that input site's
Thresholds.speculate_at and the dependent is a pure step or tool. Its value is published only after every guessed input
verified equal; otherwise it is re-run on the real inputs at the same position `i`, and the first row stays in the log with
kept=false. Effects and plain Python wait for verified values; handing a Pending to plain Python is a TypeError, never a silent
fallback. Nothing a user sees is a guess.

Virtual time: every Pending carries when its guess and its verified value became available on the run's clock, so a
fast-forward run's latency is the critical path over its calls, not their sum. speculate=False never enters this module.
"""

from __future__ import annotations

import contextvars
import json
import random
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NoReturn

from .context import RUN, RunCtx, current
from .contracts import CallSite, Decision, Kind, Prediction, TraceRow
from .predictors import Predictor
from .registry import Registered

CallFn = Callable[[Registered, tuple[Any, ...], dict[str, Any]], Any]  # the runtime's sync path for one call: (reg, args, kwargs) -> value
# set by the speculator around one call, in that call's own context: the Pending to tell its guess, where the call's rows go.
# A contextvar, not parameters, so Runtime._call/_route keep their signatures (telemetry.instrument wraps _route).
HOOK: contextvars.ContextVar[tuple[Pending, Callable[[TraceRow], None]] | None] = contextvars.ContextVar("dreaming_spec", default=None)


class Pending:
    """A call in flight. `.guess`: the predictor's top answer once the router has it (None: no predictor ran);
    `.result()`: the verified value, blocking; `.verified`: set when it is known. Resolved only by the runtime."""

    def __init__(self, site: CallSite) -> None:
        self.site = site
        self.guess: Prediction | None = None
        self.guessed, self.verified = threading.Event(), threading.Event()
        self.start = self.guess_at = self.done_at = 0  # virtual ms on the run's clock
        self.predicted = False  # a certified prediction answered: its guess was the answer, there was nothing to verify
        self.value: Any = None  # set by the runtime before `verified`; read it through .result()
        self.error: BaseException | None = None

    def result(self) -> Any:
        self.verified.wait()
        ctx = current()
        if ctx is not None:  # whoever waits for a value is at least as late as the value
            ctx.virtual_ms = max(ctx.virtual_ms, self.done_at)
        if self.error is not None:
            raise self.error
        return self.value

    def decode(self, label: str) -> Any:
        return json.loads(label) if self.site.kind == Kind.tool else label  # what the caller would receive (runtime's Call.decode)

    def __repr__(self) -> str:
        return f"Pending({self.site.id})"

    def __getattr__(self, name: str) -> NoReturn:  # `vendor.encode()` on a Pending: a type error, not an AttributeError
        if name.startswith("__"):
            raise AttributeError(name)
        _plain(self)


def _plain(p: Pending, *_: Any) -> NoReturn:
    raise TypeError(f"{p.site.id} returned a Pending: pass it to a call site as is, or use .result() for its verified value")


for _op in ("__str__", "__format__", "__bool__", "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__", "__iter__",
            "__len__", "__getitem__", "__contains__", "__add__", "__radd__", "__float__", "__int__", "__index__"):
    setattr(Pending, _op, _plain)  # no magic proxy (docs/speculation.md §3)


def result(x: Any) -> Any:
    """The verified value of whatever a call returned: a Pending waits for it; a plain value (speculate=False) is returned as is."""
    return x.result() if isinstance(x, Pending) else x


class _Tap:
    """The site's predictor, wrapped so the Pending learns its guess the moment the router has it, before the real call returns."""

    def __init__(self, inner: Predictor, p: Pending) -> None:
        self.inner, self.p = inner, p
        self.weights_sha, self.env_sha = inner.weights_sha, inner.env_sha

    def predict(self, rendered: str, *, fresh: bool = False) -> Prediction:
        g = self.inner.predict(rendered, fresh=fresh)
        if not self.p.guessed.is_set():  # the first rendering's guess is the one dependents may have started on
            self.p.guess, self.p.guess_at = g, self.p.start + g.latency_ms
            self.p.guessed.set()
        return g


def hooks(predictor: Predictor | None, record: Callable[[TraceRow], None]) -> tuple[Predictor | None, Callable[[TraceRow], None]]:
    """What Runtime._route hands the router: unchanged, or inside a speculated call the predictor tapped and the call's row buffer."""
    if (h := HOOK.get()) is None:
        return predictor, record
    HOOK.set(None)  # this call's own route only: a connector call made inside it (Runtime.external) records as usual
    return (_Tap(predictor, h[0]) if predictor is not None else None), h[1]


class Speculator:
    """The runtime's pool. submit() returns a Pending; drain() waits for a run's calls before the run closes."""

    def __init__(self, call: CallFn, record: Callable[[TraceRow], None], workers: int = 32) -> None:
        self.call, self.record = call, record
        self.pool = ThreadPoolExecutor(workers, thread_name_prefix="dreaming-call")  # FIFO: an input always starts before its dependents
        self.inflight: dict[str, list[Pending]] = {}  # run id -> the Pendings it submitted
        self.lock = threading.Lock()  # ponytail: one lock for every speculated row; Runtime._record's counters assume one writer at a time

    def submit(self, reg: Registered, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Pending:
        p, parent, child = Pending(reg.site), current(), None
        if parent is not None:  # the call's own counter, clock and draws, all fixed at submission: `i` and audits reproduce
            child = RunCtx(parent.id, parent.workflow, parent.mode, parent.seed, calls=parent.next_i() - 1,
                           virtual_ms=parent.virtual_ms, rows=parent.rows)
            child.rng = random.Random(parent.rng.random())
            self.inflight.setdefault(parent.id, []).append(p)
        self.pool.submit(contextvars.copy_context().run, self._work, p, reg, args, kwargs, child)
        return p

    def drain(self, ctx: RunCtx) -> BaseException | None:
        """Wait for every call the run submitted; the run's clock ends at the last one. Returns the first error among them."""
        err: BaseException | None = None
        for p in self.inflight.pop(ctx.id, []):
            p.verified.wait()
            ctx.virtual_ms = max(ctx.virtual_ms, p.done_at)
            err = err or p.error
        return err

    def _run(self, reg: Registered, args: tuple[Any, ...], kwargs: dict[str, Any], p: Pending,
             ctx: RunCtx | None) -> tuple[Any, BaseException | None, list[TraceRow]]:
        rows: list[TraceRow] = []
        p.start = ctx.virtual_ms if ctx else 0
        HOOK.set((p, rows.append))  # this worker's copied context only
        try:
            return self.call(reg, args, kwargs), None, rows
        except Exception as e:  # noqa: BLE001 - published through .result(), where the default path would have raised
            return None, e, rows

    def _record(self, rows: list[TraceRow]) -> None:
        with self.lock:
            for row in rows:
                self.record(row)

    def _guesses(self, reg: Registered, inputs: list[Pending], ctx: RunCtx | None) -> dict[int, Any]:
        """The inputs to render on their guess (by id): confident enough for their own site's speculate_at, and not verified by
        the time this call could start. Live, "not yet" is the wall clock; on the virtual clock it is read from the inputs' times,
        so a fast-forward run is the same run every time. Every other input is waited for, which advances this call's clock."""

        def confident(x: Pending) -> Prediction | None:
            x.guessed.wait()
            at = x.site.thresholds.speculate_at
            return x.guess if reg.pure and x.guess is not None and at is not None and x.guess.confidence >= at else None

        cands = [(x, confident(x)) for x in inputs]
        for x, g in cands:
            if g is None:
                x.result()
        out: dict[int, Any] = {}
        for x, g in cands:
            if g is None:
                continue
            if ctx is not None and ctx.mode != "live":
                late = x.verified.wait() and x.done_at > ctx.virtual_ms
            else:
                late = not x.verified.is_set()
            if not late:
                x.result()
                continue
            out[id(x)] = x.decode(g.label)
            if ctx is not None:
                ctx.virtual_ms = max(ctx.virtual_ms, x.guess_at)
        return out

    def _work(self, p: Pending, reg: Registered, args: tuple[Any, ...], kwargs: dict[str, Any], ctx: RunCtx | None) -> None:
        if ctx is not None:
            RUN.set(ctx)  # this call's copied context only
        i0 = ctx.calls if ctx else 0
        try:
            inputs = [x for x in (*args, *kwargs.values()) if isinstance(x, Pending)]
            guesses = self._guesses(reg, inputs, ctx)
            guessed = [x for x in inputs if id(x) in guesses]

            def take(x: Any) -> Any:
                return guesses[id(x)] if id(x) in guesses else result(x)

            value, error, rows = self._run(reg, tuple(map(take, args)), {k: take(v) for k, v in kwargs.items()}, p, ctx)
            if guessed:  # publish only once every guess is verified equal
                for x in guessed:
                    x.verified.wait()  # every one: .predicted is read below
                ok = all(x.error is None and x.value == guesses[id(x)] for x in guessed)
                speculated = not ok or any(not x.predicted for x in guessed)  # a certified input was the answer, not a guess
                for row in rows:
                    row.speculated, row.kept = speculated, ok if speculated else None
                if ok and ctx is not None:
                    ctx.virtual_ms = max(ctx.virtual_ms, *(x.done_at for x in guessed))
                elif not ok:  # redo on the real inputs at the same i; the discarded row stays, kept=false (replay skips it)
                    self._record(rows)
                    if ctx is not None:
                        ctx.calls = i0
                    value, error, rows = self._run(reg, tuple(map(result, args)), {k: result(v) for k, v in kwargs.items()}, p, ctx)
            self._record(rows)
            p.value, p.error, p.predicted = value, error, any(r.decision == Decision.predicted for r in rows)
        except BaseException as e:  # noqa: BLE001 - an input's error is this call's error, surfaced by .result()
            p.error = e
        finally:
            p.done_at = ctx.virtual_ms if ctx else 0
            p.guessed.set()
            p.verified.set()
