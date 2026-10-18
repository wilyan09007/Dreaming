"""Per-run state: run id, call counter, mode, clock and random generator.

A run boundary opens one Run and carries it in a ContextVar, so every call made inside lands under it
(S12: the id survives LangGraph thread pools, asyncio tasks and to_thread; raw user threads need copy_context).
The clock is virtual in fast-forward and replay: latencies are numbers that are added, never slept.
"""

from __future__ import annotations

import contextlib
import contextvars
import random
import time
import uuid
from collections.abc import Generator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from .contracts import TraceRow

Mode = Literal["live", "fast_forward"]


def _no_rows() -> list[TraceRow]:
    return []


@dataclass
class RunCtx:
    id: str
    workflow: str
    mode: Mode
    seed: int = 0
    calls: int = 0  # the next call gets i = calls + 1
    virtual_ms: int = 0  # elapsed virtual time in fast-forward/replay
    pending_ms: int = 0  # latency charged by the call in flight (charge()); the runtime adds it to that call's row
    started: float = field(default_factory=time.time)
    rows: list[TraceRow] = field(default_factory=_no_rows)  # this run's rows, whatever else the process is doing concurrently
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(f"{self.workflow}:{self.seed}")  # audit and probe draws reproduce for the same workflow and seed

    def next_i(self) -> int:
        self.calls += 1
        return self.calls

    def now(self) -> datetime:
        return datetime.now(UTC)

    def elapsed_ms(self) -> int:
        return self.virtual_ms if self.mode != "live" else round((time.time() - self.started) * 1000)

    def spend(self, latency_ms: int) -> None:
        """Account for a call's latency. Live runs really waited; the other modes only count."""
        if self.mode != "live":
            self.virtual_ms += latency_ms


RUN: contextvars.ContextVar[RunCtx | None] = contextvars.ContextVar("dreaming_run", default=None)


def current() -> RunCtx | None:
    return RUN.get()


def charge(latency_ms: int) -> None:
    """A mock tool's latency, as a number: it lands on the current call's row and the run's virtual clock, never slept."""
    ctx = RUN.get()
    if ctx is not None:
        ctx.pending_ms += latency_ms


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@contextlib.contextmanager
def run_boundary(workflow: str, mode: Mode = "live", run_id: str | None = None, seed: int = 0) -> Generator[RunCtx, None, None]:
    """Open a run (or join the one already open: nested boundaries do not start a new run)."""
    outer = RUN.get()
    if outer is not None:
        yield outer
        return
    ctx = RunCtx(run_id or new_id("run"), workflow, mode, seed)
    token = RUN.set(ctx)
    try:
        yield ctx
    finally:
        RUN.reset(token)
