"""One way to say what happened: an Event in the store (the dashboard's feed and the audit trail) plus one JSON
log line on stdout (the host's log viewer: fly logs, modal app logs). Optional: the same line to Logfire. Optional Sentry
(SENTRY_DSN): the same line as a Sentry Log, every event with an error as one tagged issue (component, kind, call_site,
dependency), and instrument() adds the agent spans (a gen_ai.invoke_agent per run, a gen_ai.execute_tool or gen_ai.chat per
routed call) that Sentry's Tracing and AI-agent views read. Every Sentry path is a no-op without the DSN and swallows its own errors.

Root cause in three fields: Event.component (who), Event.kind (what), Event.error (why). Boundaries wrap every
place a component talks to another (store, trainer, teacher, HTTP handler): an exception there becomes a
component.error event with the dependency named, then is re-raised or swallowed per the caller's fail-open rule.
"""

from __future__ import annotations

import contextlib
import contextvars
import importlib
import json
import logging
import os
import sys
from collections.abc import Generator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .context import current, new_id
from .contracts import Component, Event, EventKind, Json, Kind, TraceRow
from .store import Store

if TYPE_CHECKING:
    from .registry import Registered
    from .router import Call
    from .runtime import Runtime

log = logging.getLogger("dreaming")
TRACE: contextvars.ContextVar[str | None] = contextvars.ContextVar("dreaming_trace", default=None)  # set per HTTP request by the API


class _JsonLine(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, Any] = {"ts": datetime.now(UTC).isoformat(), "level": record.levelname, "msg": record.getMessage()}
        line.update(getattr(record, "_fields", {}))
        if record.exc_info:
            line["exc"] = self.formatException(record.exc_info)
        return json.dumps(line, default=str)


def _line(fields: dict[str, Any]) -> dict[str, Any]:
    """`extra` for one JSON log line. `_fields` is what the stdout formatter prints (Sentry's log handler skips `_` keys); the scalar
    leaves, `data` one level down as data.<key>, ride along as plain record attributes: Sentry Log attributes one can query."""
    scalar = (str, int, float, bool)
    data: dict[str, Any] = fields.get("data") or {}
    flat = {k: v for k, v in fields.items() if isinstance(v, scalar)}
    flat.update({f"data.{k}": v for k, v in data.items() if isinstance(v, scalar)})
    return {"_fields": fields, **flat}


def configure(component: Component, *, level: str = "INFO") -> None:
    """Call once per process. LOGFIRE_TOKEN turns on Logfire export, SENTRY_DSN turns on Sentry (SENTRY_ENV, SENTRY_RELEASE,
    SENTRY_TRACES tune it); nothing else changes."""
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(_JsonLine())
    root = logging.getLogger()
    root.handlers[:] = [h]
    root.setLevel(level)
    if dsn := os.environ.get("SENTRY_DSN"):  # before Logfire: its missing-package path returns
        try:
            sentry: Any = importlib.import_module("sentry_sdk")  # optional: sentry-sdk[fastapi]>=2 (the API image installs it)
            sentry.init(dsn=dsn, environment=os.environ.get("SENTRY_ENV") or os.environ.get("SENTRY_ENVIRONMENT", "dev"),
                        release=os.environ.get("SENTRY_RELEASE", "dreaming@dev"), enable_logs=True,  # no PII: the SDK default
                        traces_sample_rate=float(os.environ.get("SENTRY_TRACES", "1.0")))  # fastapi/starlette/httpx integrations switch on by themselves
            importlib.import_module("sentry_sdk.integrations.logging").ignore_logger("dreaming")  # events only: emit() sends the tagged issue; the Logs still flow
        except Exception as e:  # noqa: BLE001 - fail open, like logfire: no package, a malformed DSN or rate is a warning, never a dead process
            log.warning("SENTRY_DSN is set but Sentry did not start (%s); no Sentry", type(e).__name__)
    log.info("configured", extra=_line({"component": component}))  # after the Sentry init, so it is the first Sentry Log
    if os.environ.get("LOGFIRE_TOKEN"):
        try:
            logfire: Any = importlib.import_module("logfire")  # optional: logfire>=5.2,<6 (the API image installs it)
        except ModuleNotFoundError:  # fail open: a token without the package is a warning, never a dead process
            log.warning("LOGFIRE_TOKEN is set but the logfire package is not installed; logging to stdout only")
            return

        # the default scrubber replaces any value containing "auth", "session" … with a placeholder, wiping root causes
        def keep(match: Any) -> Any:
            return match.value

        logfire.configure(service_name=component, send_to_logfire="if-token-present", scrubbing=logfire.ScrubbingOptions(callback=keep))
        root.addHandler(logfire.LogfireLoggingHandler())


def _sentry(message: str, tags: dict[str, Any], extras: dict[str, Any]) -> None:
    """One error issue to Sentry if it is initialized. Fail open like the store: Sentry can never break emit.
    Grouped by where it happened (component, kind, call site or dependency), not by message: messages carry run ids."""
    with contextlib.suppress(Exception):
        sentry: Any = sys.modules.get("sentry_sdk")
        if sentry is not None and sentry.is_initialized():
            sentry.capture_message(message, level="error", tags={k: str(v) for k, v in tags.items() if v}, extras=extras,
                                   fingerprint=[str(tags.get(k) or "") for k in ("component", "kind", "call_site", "dependency")])


def _set(sp: Any, **data: Any) -> None:
    """Span data, minus the empty values. Never raises."""
    if sp is not None:
        with contextlib.suppress(Exception):
            for k, v in data.items():
                if v not in (None, ""):
                    sp.set_data(k, v)


@contextlib.contextmanager
def span(op: str, name: str, **data: Any) -> Generator[Any, None, None]:
    """One Sentry span around a block; a transaction when none is open (a span outside a transaction is dropped). A thread
    started by a request, like the fast-forward batch, inherits that request's transaction after it was sent: that is none too.
    Yields the span, or None when Sentry is off. Sentry's errors stay here; the block's pass through."""
    sentry: Any = sys.modules.get("sentry_sdk")
    if sentry is None or not sentry.is_initialized():
        yield None
        return
    txn = getattr(sentry.get_current_span(), "containing_transaction", None)
    with (sentry.start_transaction if txn is None or txn.timestamp is not None else sentry.start_span)(op=op, name=name) as sp:
        _set(sp, **data)
        yield sp


def instrument(rt: Runtime) -> None:
    """Sentry's agent spans, added by wrapping two methods of this Runtime instance so no other module knows Sentry:
    `run` opens a gen_ai.invoke_agent span per run; `_route` opens a gen_ai.execute_tool (tool, effect, a2a) or gen_ai.chat
    (model, router) span per routed call and stamps the router's decision, candidate and certificate from the call's TraceRow.
    No-op unless configure() initialized Sentry."""
    sentry: Any = sys.modules.get("sentry_sdk")
    if sentry is None or not sentry.is_initialized():
        return
    run, route = rt.run, rt._route  # pyright: ignore[reportPrivateUsage]

    @contextlib.contextmanager
    def traced_run(workflow: str, input: Json, mode: str = "live", seed: int | None = None) -> Generator[Any, None, None]:
        with span("gen_ai.invoke_agent", f"invoke_agent {workflow}", **{"gen_ai.operation.name": "invoke_agent",
                  "gen_ai.agent.name": workflow, "dreaming.mode": mode}) as sp:
            with run(workflow, input, mode, seed) as r:
                _set(sp, **{"dreaming.run_id": r.id})
                yield r
            if sp is not None and r.failed:  # rt.run swallows the failure into the Run row, so the span has to be told
                sp.set_status("internal_error")
                _set(sp, **{"dreaming.stuck_at": r.stuck_at})

    def traced_route(reg: Registered, call: Call) -> Any:
        site = reg.site
        ctx, op = current(), "chat" if site.kind in (Kind.model, Kind.router) else "execute_tool"
        before = len(ctx.rows) if ctx else 0
        with span(f"gen_ai.{op}", f"{op} {site.name}", **{"gen_ai.operation.name": op, "dreaming.site": site.id,
                  "gen_ai.request.model" if op == "chat" else "gen_ai.tool.name": site.teacher.model if op == "chat" else site.name}) as sp:
            try:
                return route(reg, call)
            finally:  # the router records the row before it returns or re-raises, so the last row is this call's (ponytail: unless a run's calls overlap, as speculation's do; then a decision may land on the neighbour's span)
                if ctx and len(ctx.rows) > before:
                    _stamp(sp, ctx.rows[-1])

    rt.run, rt._route = traced_run, traced_route  # type: ignore[method-assign,assignment]  # pyright: ignore


def _stamp(sp: Any, row: TraceRow) -> None:
    p, t = row.prediction, row.teacher
    u = t.usage if t and t.recorded else {}  # a live call's tokens are counted by Sentry's own OpenAI integration (when active): never twice
    _set(sp, **{"gen_ai.provider.name": t and t.provider, "gen_ai.response.model": t and t.model, "gen_ai.usage.input_tokens": u.get("prompt_tokens"),
                "gen_ai.usage.output_tokens": u.get("completion_tokens"), "gen_ai.cost.total_tokens": row.cost_usd if u else None,
                "dreaming.decision": row.decision.value, "dreaming.candidate": p.candidate if p else "", "dreaming.cert_id": p.cert_id if p else None,
                "dreaming.agree": row.agree, "dreaming.cost_usd": row.cost_usd, "dreaming.latency_ms": row.latency_ms, "dreaming.error": row.error})
    if sp is not None and row.error:
        sp.set_status("internal_error")


def emit(
    store: Store | None,
    component: Component,
    kind: EventKind,
    message: str,
    *,
    call_site: str | None = None,
    data: dict[str, Json] | None = None,
    error: str | None = None,
    run_id: str | None = None,
) -> Event:
    """Log and store one event. Never raises: a store that is down is itself logged, not propagated."""
    ctx = current()
    ev = Event(
        id=new_id("evt"), ts=datetime.now(UTC), component=component, kind=kind, message=message,
        run_id=run_id or (ctx.id if ctx else None), call_site=call_site, data=data or {}, error=error, trace_id=TRACE.get(),
    )
    log.log(logging.ERROR if error else logging.INFO, message, extra=_line(ev.model_dump(mode="json", exclude={"message"})))
    if store is not None:
        try:
            store.emit(ev)
        except Exception as e:
            log.exception("store.emit failed", extra=_line({"component": component, "kind": kind}))
            _sentry("store.emit failed", {"component": component, "kind": kind, "dependency": "store"}, {"error": repr(e)})
    if error is not None:
        _sentry(message, {"component": component, "kind": kind, "call_site": call_site, "dependency": (data or {}).get("dependency")},
                {"error": error, "event_id": ev.id})
    return ev


@contextlib.contextmanager
def boundary(store: Store | None, component: Component, dependency: str, *, swallow: bool = False) -> Generator[None, None, None]:
    """Wrap a call into another component. On failure: one component.error event naming the dependency."""
    try:
        yield
    except Exception as e:
        emit(store, component, EventKind.component_error, f"{component}: {dependency} failed", data={"dependency": dependency},
             error=f"{type(e).__name__}: {e}")
        if not swallow:
            raise
