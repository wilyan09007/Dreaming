# Interface contracts

Frozen early so four people can build in parallel. Types are illustrative; the shapes and names are what matter. Anything here changes only by agreement of all workstreams (see `PLAN.md`).

## 1. Call type declaration

```python
@dataclass(frozen=True)
class CallTypeSpec:
    name: str                       # stable id, e.g. "categorize_transaction"
    version: int                    # bumps on prompt/schema/model change
    kind: Literal["model", "tool", "reasoning"]
    effect: Literal["pure", "read", "write"]
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    ttl_s: int | None = None        # reads only: staleness bound
    alternates: list[str] = ()      # tool names with a compatible schema
    ignore_for_agreement: set[str] = ()
    idempotent: bool = False        # writes: may be retried
```

Invariant enforced by the gateway: `effect == "write"` is never served by a predictor.

## 2. Trace record

One JSON object per call (real or predicted). Stored in the trace store; also emitted as an OpenTelemetry span.

```json
{
  "trace_id": "uuid", "run_id": "uuid", "span_id": "uuid", "parent_span_id": "uuid|null",
  "call_type": "categorize_transaction", "call_version": 1,
  "source": "real | predictor | real_after_lowconf | audit",
  "input": {}, "input_digest": "sha256-hex",
  "output": {}, "error": {"class": "timeout", "http_status": null, "retryable": true},
  "latency_ms": 0.0, "cost_usd": 0.0,
  "tokens": {"in": 0, "out": 0, "reasoning": 0},
  "predictor_version": null, "confidence": null, "agreement": null,
  "context": {"endpoint": "bank_feed_primary", "step_idx": 1, "hour_bucket": 0},
  "model_id": null, "ts": "2026-10-17T00:00:00Z"
}
```

Rule: **training labels come only from `source in {real, real_after_lowconf, audit}`**. The Dreamer query enforces this.

## 3. Gateway API (what the agent calls)

```python
result = await gateway.call(spec: CallTypeSpec, inp: BaseModel, *, run_id: str) -> CallResult

@dataclass
class CallResult:
    output: BaseModel | None
    source: Literal["real", "predictor", "real_after_lowconf", "routed", "error"]
    confidence: float | None
    cost_usd: float
    latency_ms: float
    endpoint: str | None            # which tool endpoint served it
```

## 4. Predictor interface (workstream B implements, gateway consumes)

```python
class Predictor(Protocol):
    call_type: str
    version: int
    def predict(self, inp: BaseModel) -> tuple[BaseModel, float]: ...   # output, calibrated confidence
    def tau_serve(self) -> float: ...
```

Trainers (local tier 0 to 2, Veri SFT) all produce an artifact plus metadata the registry stores.

## 5. Registry

Per call type: `state` in `{OBSERVING, SHADOW, PROMOTED, DEMOTED}`, current `predictor_version`, `tau_serve`, `audit_rate`, `bound` (the promotion statistic), `n_shadow`, `coverage`, `cooldown_until`. The gateway reads state on every call; changes take effect on the next call.

## 6. Tool model interface (workstream C)

```python
class ToolModel(Protocol):
    def predict(self, endpoint: str, inp: BaseModel, ctx: dict) -> ToolForecast: ...
    def observe(self, endpoint: str, ok: bool, latency_ms: float, ctx: dict) -> None: ...

@dataclass
class ToolForecast:
    p_fail: float
    failure_class_probs: dict[str, float]
    latency_p50_ms: float
    latency_p95_ms: float
```

Router: `choose(spec, inp) -> endpoint`. Recovery is the call type `recover_from_tool_failure` (output: one of `retry`, `use:<endpoint>`, `use_cache`, `skip_step`, `escalate`).

## 7. Dashboard event stream (workstream D consumes)

Server-sent or WebSocket JSON events:

| Event | Fields |
|---|---|
| `run_started` | `run_id`, `ts` |
| `step` | `run_id`, `call_type`, `source`, `cost_usd`, `latency_ms`, `confidence`, `endpoint` |
| `run_finished` | `run_id`, `cost_usd`, `latency_ms`, `success`, `frontier_calls`, `predicted_steps`, `routed_steps` |
| `state_change` | `call_type`, `from`, `to`, `bound`, `n`, `coverage` |
| `audit_result` | `call_type`, `agree` (bool), `rolling_agreement` |
| `tool_health` | `endpoint`, `p_fail`, `status` |
| `tool_killed` | `endpoint`, `ts` |
| `recovery` | `run_id`, `endpoint_from`, `action`, `frontier_calls` (expected 0) |

## 8. Cost accounting

`cost_usd = tokens_in * price_in + tokens_out * price_out (+ reasoning tokens at output price) + api_fee`. Prices live in one table (`prices.yaml`). Predictor inference cost is charged at an amortized per-call figure (document the number). Audit cost is attributed to the run that triggered it.

## 9. Mock tool services (shared by C and D)

HTTP services with a control endpoint:

- `POST /control/kill` and `POST /control/revive`
- `POST /control/flakiness` with a profile name and seed
- every response includes `X-Mock-Endpoint` for tracing

Seeds make runs reproducible; the same seed must produce the same failure schedule for the baseline and Dreaming arms.
