"""One way to say what happened: an Event in the store (the dashboard's feed and the audit trail) plus one JSON
log line on stdout (the host's log viewer: fly logs, modal app logs). Optional: the same line to Logfire; every event with an
error also goes to Sentry as one tagged event (component, kind, call_site, dependency).

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
from typing import Any

from .context import current, new_id
from .contracts import Component, Event, EventKind, Json
from .store import Store

log = logging.getLogger("dreaming")
TRACE: contextvars.ContextVar[str | None] = contextvars.ContextVar("dreaming_trace", default=None)  # set per HTTP request by the API


class _JsonLine(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, Any] = {"ts": datetime.now(UTC).isoformat(), "level": record.levelname, "msg": record.getMessage()}
        line.update(getattr(record, "fields", {}))
        if record.exc_info:
            line["exc"] = self.formatException(record.exc_info)
        return json.dumps(line, default=str)


def configure(component: Component, *, level: str = "INFO") -> None:
    """Call once per process. LOGFIRE_TOKEN turns on Logfire export, SENTRY_DSN turns on Sentry (SENTRY_ENV, SENTRY_RELEASE,
    SENTRY_TRACES tune it); nothing else changes."""
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(_JsonLine())
    root = logging.getLogger()
    root.handlers[:] = [h]
    root.setLevel(level)
    log.info("configured", extra={"fields": {"component": component}})
    if dsn := os.environ.get("SENTRY_DSN"):  # before Logfire: its missing-package path returns
        try:
            sentry: Any = importlib.import_module("sentry_sdk")  # optional: sentry-sdk[fastapi]>=2 (the API image installs it)
        except ModuleNotFoundError:  # fail open, like logfire
            log.warning("SENTRY_DSN is set but the sentry-sdk package is not installed; no Sentry")
        else:
            sentry.init(dsn=dsn, environment=os.environ.get("SENTRY_ENV") or os.environ.get("SENTRY_ENVIRONMENT", "dev"),
                        release=os.environ.get("SENTRY_RELEASE", "dreaming-mock@0"), send_default_pii=False,
                        traces_sample_rate=float(os.environ.get("SENTRY_TRACES", "0.1")))  # fastapi/starlette/httpx integrations switch on by themselves
            importlib.import_module("sentry_sdk.integrations.logging").ignore_logger("dreaming")  # emit() sends the tagged event; no double
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
    """One error event to Sentry if it is initialized. Fail open like the store: Sentry can never break emit."""
    with contextlib.suppress(Exception):
        sentry: Any = sys.modules.get("sentry_sdk")
        if sentry is not None and sentry.is_initialized():
            sentry.capture_message(message, level="error", tags={k: str(v) for k, v in tags.items() if v}, extras=extras)


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
    log.log(logging.ERROR if error else logging.INFO, message, extra={"fields": ev.model_dump(mode="json", exclude={"message"})})
    if store is not None:
        try:
            store.emit(ev)
        except Exception as e:
            log.exception("store.emit failed", extra={"fields": {"component": component, "kind": kind}})
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
