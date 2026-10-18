"""The one predict-or-call function. The native runtime and every connector go through it.

Decision table (phase from the site's Policy; `confident` = a prediction exists and confidence >= cutoff):

    phase            confident   audit draw   real call   decision     notes
    awake            -           -            runs        called       no predictor yet
    watching/testing -           -            runs        called       prediction logged in shadow; agree recorded
    dreaming         yes         no           skipped     predicted    cert_id stamped on the prediction
    dreaming         yes         yes (1 in N) runs        audited      both logged: keeps the certificate current
    dreaming         no          -            runs        called
    outage           yes         -            skipped     predicted    Policy.outage (detected, learning.py): the dream still answers
    outage           no          -            skipped     fallback     registered fallback; no fallback => Stuck; 1 in 20 calls probes the real call
    any              -           -            raises      fallback     if a fallback is registered, else the error propagates (the row keeps the error)

Fail open: a predictor that raises is a miss; nothing here touches the network or the store. The router returns
the value and the TraceRow; the runtime writes rows in batches and never lets a storage error block a call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from pydantic_core import to_jsonable_python

from .context import RunCtx, new_id
from .contracts import Decision, Json, Kind, Phase, Policy, Prediction, TeacherAnswer, TraceRow
from .hashing import sha
from .predictors import Predictor
from .registry import Registered


@dataclass(frozen=True)
class Answer:
    """What a real call produced. For steps `teacher` is set; for tools it is None and `value` is the return."""

    value: Any
    label: str  # the typed output: a step's label, or a tool return serialized by the runtime
    latency_ms: int
    cost_usd: float = 0.0
    teacher: TeacherAnswer | None = None


@dataclass(frozen=True)
class Call:
    reg: Registered
    request: Json  # what is logged: the rendered text for steps, the arguments for tools
    rendered: str  # what the predictor and the teacher read
    real: Callable[[], Answer]  # the real call
    fallback: Callable[[], Any] | None = None  # bound to this call's arguments
    decode: Callable[[str], Any] = lambda label: label  # label -> the value the caller receives


PROBE = 0.05  # share of calls that still try a tool that is marked down, so recovery is noticed


class Stuck(RuntimeError):
    """A tool is down, nothing has learned it, and no fallback is registered. The run cannot continue."""


def _shadow(predictor: Predictor | None, rendered: str, on_error: Callable[[Exception], None] | None) -> Prediction | None:
    if predictor is None:
        return None
    try:
        return predictor.predict(rendered)
    except Exception as e:  # noqa: BLE001 - fail open: a broken predictor is a miss, never an outage; but never a silent one
        if on_error:
            on_error(e)
        return None


def route(call: Call, policy: Policy, predictor: Predictor | None, ctx: RunCtx | None, record: Callable[[TraceRow], None],
          on_predictor_error: Callable[[Exception], None] | None = None, spares: Mapping[str, Predictor] | None = None) -> Any:
    """Route one call; `record` receives its TraceRow before the value is returned or the real call's error re-raised.
    `spares` (the site's other candidates) predict too, into TraceRow.shadow: free calibration data, never an answer."""
    t0 = perf_counter()
    site = call.reg.site
    i = ctx.next_i() if ctx else 0
    shadow: dict[str, Prediction] = {}
    prediction = _shadow(predictor, call.rendered, on_predictor_error) if policy.phase != Phase.awake else None
    if prediction is not None:
        prediction = prediction.model_copy(update={"candidate": policy.candidate})
        for name, spare in (spares or {}).items():
            if (q := _shadow(spare, call.rendered, on_predictor_error)) is not None:
                shadow[name] = q.model_copy(update={"candidate": name})
    confident = (
        prediction is not None and policy.cutoff is not None and prediction.confidence >= policy.cutoff
        and policy.phase == Phase.dreaming
    )
    if confident and prediction is not None:
        prediction = prediction.model_copy(update={"cert_id": policy.cert_id})  # the certificate in force, on predicted and audited rows alike
    probe = policy.outage and ctx is not None and ctx.rng.random() < PROBE  # drawn first: a probe outranks a confident prediction
    audit = probe or (confident and not policy.outage and ctx is not None and ctx.rng.random() < site.thresholds.audit_rate)
    value: Any = None
    teacher: TeacherAnswer | None = None
    agree: bool | None = None
    error: str | None = None
    failed: BaseException | None = None
    cost = 0.0
    latency = 0

    if confident and not audit and prediction is not None:
        value, decision, cost, latency = call.decode(prediction.label), Decision.predicted, prediction.cost_usd, prediction.latency_ms
    elif policy.outage and not probe and site.kind in (Kind.tool, Kind.a2a):
        # only tools are short-circuited while down: a model step has no fallback, so skipping its teacher would only turn a
        # transient failure (a rate limit, seen live) into a run that is stuck for ~20 calls; the call is made and may fail
        if call.fallback is None:
            failed = Stuck(f"{site.id} is down, nothing certified for this input, no fallback registered")
            decision, error = Decision.called, f"Stuck: {failed}"
        else:
            value, decision = call.fallback(), Decision.fallback
    else:
        try:
            ans = call.real()
            value, teacher, cost, latency = ans.value, ans.teacher, ans.cost_usd, ans.latency_ms
            decision = Decision.audited if confident else Decision.called
            if prediction is not None:
                agree = prediction.label == ans.label
        except Exception as e:  # noqa: BLE001 - any failure of the real call is one row, then the fallback or the error itself
            error = f"{type(e).__name__}: {e}"
            if call.fallback is None:
                failed, decision = e, Decision.called
            else:
                value, decision = call.fallback(), Decision.fallback

    if ctx is not None:
        ctx.spend(latency)
    wall = round((perf_counter() - t0) * 1000)
    row = TraceRow(
        id=new_id("row"), run_id=ctx.id if ctx else "", i=i, call_site=site.id, kind=site.kind,
        phase=Phase.down if policy.outage else policy.phase, request=call.request, request_sha=sha(call.request), response=to_jsonable_python(value, fallback=str),
        decision=decision, prediction=prediction, shadow=shadow, teacher=teacher, agree=agree,
        latency_ms=latency if latency else wall, cost_usd=cost + (prediction.cost_usd if prediction and decision != Decision.predicted else 0.0),
        error=error, ts=datetime.now(UTC),
    )
    record(row)
    if failed is not None:
        raise failed
    return value
