"""Step and tool declarations. A workflow is the CallSites declared here plus the Python functions behind them.

    @step("categorize", workflow="bookkeeper", labels=CATS, descriptions=DESC)
    def categorize(txn: Txn) -> str:            # returns the rendered input; the library makes the call
        return f"Transaction: {txn.descriptor}, ${txn.amount:.2f}"

    @tool("vendor_lookup", workflow="bookkeeper", fallback=snapshot_lookup)
    def vendor_lookup(descriptor: str) -> Vendor:   # the real call; the library may skip it
        ...

Calling `categorize(txn)` returns the label (str) the way the agent's code expects; who answered is in the trace.
The student's day-zero prompt is rendered from `labels` and `descriptions`, never from the teacher's prompt.
The functions are bound to a Runtime by `runtime.bind()`; until then they raise, so a missing runtime is loud.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .contracts import CallSite, Kind, TeacherSpec, Thresholds


class Dispatch(Protocol):
    """What the runtime installs: route one call of one site. Returns what the caller receives."""

    def __call__(self, reg: Registered, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any: ...


@dataclass(frozen=True)
class Registered:
    site: CallSite
    fn: Callable[..., Any]  # steps: renders the input text; tools: the real call
    fallback: Callable[..., Any] | None = None  # tools: answers when the real call fails or is down
    speculate_next: str | None = None  # tools: name of the read-only step to prefetch on the likely return (opt-in)


_SITES: dict[str, Registered] = {}
_dispatch_fn: Dispatch | None = None


def _unbound(reg: Registered, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    raise RuntimeError(f"{reg.site.id}: no Runtime bound; call runtime.bind() first")


def bind(dispatch: Dispatch | None) -> None:
    """Install the runtime's router. `None` unbinds (tests)."""
    global _dispatch_fn
    _dispatch_fn = dispatch


def sites() -> list[CallSite]:
    return [r.site for r in _SITES.values()]


def get(site_id: str) -> Registered:
    return _SITES[site_id]


def _register(reg: Registered) -> Callable[..., Any]:
    if reg.site.id in _SITES:
        raise ValueError(f"call site {reg.site.id} declared twice")
    _SITES[reg.site.id] = reg

    def call(*args: Any, **kwargs: Any) -> Any:
        return (_dispatch_fn or _unbound)(reg, args, kwargs)

    call.__name__ = reg.fn.__name__
    call.__doc__ = reg.fn.__doc__
    return call


def step(
    name: str,
    *,
    workflow: str,
    labels: list[str],
    descriptions: dict[str, str] | None = None,
    teacher: TeacherSpec | None = None,
    cheap: TeacherSpec | None = None,
    thresholds: Thresholds | None = None,
    candidates: list[str] | None = None,
) -> Callable[[Callable[..., str]], Callable[..., str]]:
    """Declare a typed model step. The decorated function renders the input text and returns it.
    `cheap` makes it a router site (mission 1): the cheaper configuration is the predictor on trial.
    `candidates` names the predictor families to train in parallel (predictors.CANDIDATES); None = the kind's default."""

    def deco(fn: Callable[..., str]) -> Callable[..., str]:
        site = CallSite(
            id=f"{workflow}.{name}", workflow=workflow, name=name, kind=Kind.router if cheap else Kind.model, labels=labels,
            descriptions=descriptions or {}, teacher=teacher or TeacherSpec(), cheap=cheap, replayable=cheap is None,
            thresholds=thresholds or Thresholds(), candidates=candidates or [],
        )
        return _register(Registered(site, fn))

    return deco


def tool(
    name: str,
    *,
    workflow: str,
    fallback: Callable[..., Any] | None = None,
    speculate_next: str | None = None,
    thresholds: Thresholds | None = None,
    price_call: float = 0.0,
    candidates: list[str] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Declare a tool call site around the real function. Its returns are the labels; reality is the teacher."""

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        site = CallSite(
            id=f"{workflow}.{name}", workflow=workflow, name=name, kind=Kind.tool, teacher=TeacherSpec(provider="tool", price_call=price_call),
            fallback=fallback.__name__ if fallback else None, thresholds=thresholds or Thresholds(), candidates=candidates or [],
        )
        return _register(Registered(site, fn, fallback, speculate_next))

    return deco


def effect(name: str, *, workflow: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Declare a call that changes the world (post to books, send a message). Logged only, never predicted."""

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        site = CallSite(id=f"{workflow}.{name}", workflow=workflow, name=name, kind=Kind.effect, replayable=False,
                        teacher=TeacherSpec(provider="tool"))
        return _register(Registered(site, fn))

    return deco
