# Dreaming: Design Doc

> A runtime that learns a model of every call your agent makes, then stops making the ones it can predict. The agent gets cheaper and more reliable the more it runs.

| | |
|---|---|
| **Status** | Draft v0.1 |
| **Scope** | Architecture, planning, and the stage demo |
| **Last updated** | 2026-10-17 |

---

## 1. Summary

Agents keep paying for the same work. A support agent classifies its ten-thousandth "where is my order" ticket with a frontier model and gets the same answer it got the other 9,999 times. It calls a flaky shipping API that fails the same way every Tuesday night, and then pays a frontier model to reason its way around the failure again.

**Dreaming** is a runtime layer that wraps an agent's expensive calls: frontier-model calls, chain-of-thought (CoT) reasoning steps, and external API/tool calls. It does four things:

1. **Records** every expensive call as a structured, typed trace.
2. **Trains** a small, typed predictor for each *call type* (a call site with a fixed input/output schema).
3. **Shadows**: the predictor runs next to the real call, and we measure agreement without affecting behavior.
4. **Promotes** a predictor once its agreement clears a statistically safe threshold. After that, the real call runs only when the predictor has low confidence, or on a small sampled fraction kept for ongoing monitoring (audit).

There are two kinds of predictor, and they drive the two falling lines in the demo:

- **Replacement predictors** (distillation) predict a call's *output*, so the call can be skipped. This brings **cost** down.
- **Outcome predictors** (the tool model) predict whether a call will *succeed*, how slow it will be, and how it will fail. They let the agent route around bad endpoints and handle failures without asking a frontier model. This brings the **failure rate** down.

The name comes from the offline training loop. The agent "dreams" over the day's traces and wakes up needing fewer expensive calls.

---

## 2. Understanding the idea (restated precisely)

These are the exact claims the system has to make true, and where the hard parts are.

| Claim | What it really requires | Hard part |
|---|---|---|
| "Learns a model of every call" | Each expensive call site is identified, typed, and logged with enough context to reproduce its decision. | Call *identity* has to stay stable across runs and code changes, and inputs have to be canonicalized into features. |
| "Stops making the ones it can predict" | A per-call decision rule: predict, or run the real call. | Knowing *when* you can predict means **calibrated confidence**, not just accuracy. |
| "Shadow until agreement clears a threshold" | Side-by-side evaluation with a type-aware agreement metric and a statistical promotion test. | Agreement on free text is ill-defined. The frontier model doesn't fully agree with *itself* either. |
| "Real call runs on low confidence or a sampled fraction" | Selective prediction plus an audit sampler. | Picking a threshold that holds a target error rate. Detecting drift quickly. |
| "Cheaper the more it runs" | More traces → more promoted call types, and higher coverage within each. | Long-tail inputs never reach coverage, so there is a floor (audit + tail + non-distillable steps). |
| "More reliable the more it runs" | Tool outcomes are predictable, and the agent has *alternatives* to route to. | You can't fake a side-effecting call. Reliability comes from **routing and recovery**, not from replacing writes. |
| Live tool kill → reroute without a frontier call | The *recovery decision* was itself distilled from earlier frontier-handled failures, and the tool model has a learned failure prior. | A hard kill is only "expected" in the sense of a prior plus fast detection. See §8.4 for what is honest to claim on stage. |

### 2.1 What gets replaced, and what never does

Each call type declares an **effect class**:

| Effect class | Examples | Can a predictor *replace* it? | Can the tool model *route* it? |
|---|---|---|---|
| `pure` | LLM classification, extraction, CoT planning that outputs a decision, scoring | **Yes** | n/a |
| `read` | GET order status, search, policy lookup | **Yes, with a TTL / staleness bound** (it is a learned cache) | **Yes** |
| `write` | Issue refund, send email, create ticket | **Never** | **Yes** (choose endpoint, pre-check, retry policy) |

This is a hard safety invariant. A predictor can never stand in for a side effect.

---

## 3. Goals and non-goals

### Goals
- **G1.** Drop-in SDK: wrapping an existing agent takes under 30 lines of changes.
- **G2.** Per-call-type predictors that are typed. The output is a valid instance of the call's output schema.
- **G3.** Promotion is safe by construction. A promoted call type stays within a configured error budget (e.g. ≤1% disagreement on audited calls, measured relative to the frontier model's own self-agreement).
- **G4.** Automatic demotion on drift (input distribution shift, audit disagreement spike, upstream model/API version change).
- **G5.** Tool reliability model + router that measurably cuts end-to-end failure rate when alternatives exist.
- **G6.** A stage demo that shows both curves falling over a few hundred runs, plus the live tool-kill moment.

### Non-goals (v1)
- Replacing **free-form text generation** (e.g. a customer-facing reply). We still trace and route it, but don't replace it. (Possible v2: template + slot predictors.)
- Replacing `write` calls, ever.
- Training large models. Predictors are small and cheap: trees, linear models, kNN, and at most a small fine-tuned LM.
- Multi-tenant hosted service. v1 is a library plus a local worker.
- Learning tool *equivalence* automatically. Alternates are declared by the developer in v1.

---

## 4. Core concepts

**Call type.** The unit of learning. Identified by `(name, version, input_schema_hash, output_schema_hash)`. Declared via decorator, or inferred by an auto-wrapper from the call site plus the prompt template hash. A change to the prompt template, schema, or underlying model version produces a **new version**, and the predictor resets to `SHADOW` (warm-started from the old one).

**Trace.** One record per real or predicted call. See §6.2.

**Predictor.** A model `f(features(input)) → (output, confidence)` for one call type. Confidence is calibrated, meaning P(output agrees with the real call | confidence = c) ≈ c.

**Agreement.** A type-aware comparison between predicted and real outputs (§7.3). Not string equality.

**Coverage.** The fraction of calls of a type where the predictor is confident enough to answer. **Savings ≈ coverage × (cost_real − cost_pred) − audit_rate × cost_real.**

**Lifecycle state** per call type:

```
          enough traces               LCB(agreement) ≥ τ_promote
OBSERVING ────────────────▶ SHADOW ─────────────────────────────▶ PROMOTED
    ▲                         ▲                                      │
    │  schema/model change    │   audit disagreement / drift         │
    └─────────────────────────┴──────────────── DEMOTED ◀────────────┘
```

- `OBSERVING`: real call only, collecting traces. No predictor yet.
- `SHADOW`: real call is authoritative. Predictor runs in parallel and is scored.
- `PROMOTED`: predictor answers when `confidence ≥ τ_serve`. Otherwise the real call runs. A fraction `ρ` of confident calls are also audited.
- `DEMOTED`: back to shadow behavior, with a cooldown before re-promotion is allowed.

---

## 5. Architecture

### 5.1 Component overview

```
┌──────────────────────────────── Agent process ─────────────────────────────────┐
│                                                                                │
│   agent code ──▶ @dream.call / auto-wrapped clients (LLM, HTTP, MCP tools)     │
│                         │                                                      │
│                         ▼                                                      │
│               ┌───────────────────┐   reads      ┌──────────────────────────┐  │
│               │   Call Gateway    │◀────────────▶│ Predictor Cache          │  │
│               │  (decision rule)  │              │ (in-process, hot models) │  │
│               └───┬──────┬────────┘              └──────────────────────────┘  │
│        real call  │      │  predicted output                                   │
│                   ▼      ▼                                                     │
│          ┌──────────┐  ┌──────────────┐   ┌───────────────────────────────┐    │
│          │ Executor │  │  Comparator  │   │ Tool Model + Router           │    │
│          │ (real)   │  │ (shadow/audit│   │ (P(fail), latency, alternates,│    │
│          └────┬─────┘  │  agreement)  │   │  distilled recovery policy)   │    │
│               │        └──────┬───────┘   └───────────────────────────────┘    │
│               └───────┬───────┘                                                │
│                       ▼                                                        │
│                 Trace Emitter (async, non-blocking)                            │
└───────────────────────┼────────────────────────────────────────────────────────┘
                        ▼
              ┌───────────────────┐        ┌──────────────────────────────┐
              │   Trace Store     │───────▶│  Dreamer (training worker)   │
              │ (DuckDB/Parquet)  │        │  featurize → train → calibrate│
              └─────────┬─────────┘        │  → evaluate → register        │
                        │                  └──────────────┬───────────────┘
                        │                                 ▼
                        │                  ┌──────────────────────────────┐
                        │                  │ Registry                     │
                        │                  │ call types, predictor versions│
                        │                  │ lifecycle state, thresholds   │
                        │                  └──────────────┬───────────────┘
                        ▼                                 ▼
              ┌──────────────────────────────────────────────────────────────┐
              │ Policy Engine: promotion / demotion / audit rate / drift     │
              └──────────────────────────────────────────────────────────────┘
                        │
                        ▼
              ┌──────────────────────────────────────────────────────────────┐
              │ Live dashboard (cost/run, failure rate, per-call-type states)│
              └──────────────────────────────────────────────────────────────┘
```

### 5.2 Components

| Component | Responsibility | v1 implementation |
|---|---|---|
| **SDK / Interceptor** | Declares call types and wraps LLM clients, HTTP, and MCP tools. Captures input/output/cost/latency/errors. | Python library. Pydantic for schemas. Decorator + client wrappers. |
| **Call Gateway** | Per-call decision: run real / predict / predict+audit / route. Enforces effect-class invariants. | In-process. Adds < 1 ms overhead excluding model inference. |
| **Executor** | Runs the real call with timeout/retry policy from the router. | Thin wrapper around the original function. |
| **Predictor Cache** | Keeps current predictor versions loaded in memory. Hot-swaps on registry update. | Pickled sklearn/LightGBM models + embedding model. Polls the registry. |
| **Comparator** | Computes type-aware agreement for shadow and audit pairs (§7.3). | Pure Python. Optional LLM judge for semantic fields (sampled, cached). |
| **Trace Emitter / Store** | Append-only structured traces. | DuckDB file (demo); Parquet on object storage (later). |
| **Dreamer** | Background training: featurize, train, calibrate, evaluate, register. | Separate process. Triggered every N new traces per call type or every T seconds. |
| **Registry** | Source of truth for call types, versions, predictor artifacts, lifecycle state, thresholds. | SQLite table + artifact directory. |
| **Policy Engine** | Promotion tests, demotion triggers, adaptive audit rate, drift detection. | Runs inside the Dreamer loop plus on every audit result. |
| **Tool Model + Router** | Predicts tool outcome and latency. Picks among declared alternates. Serves the distilled recovery policy. | Per-endpoint classifier + online health state (Bayesian/EWMA). |
| **Dashboard** | Live charts and per-call-type state for the demo. | FastAPI + WebSocket, small React page. |

### 5.3 Request path (per call)

```python
def gateway(call_type, inp):
    ct = registry[call_type]
    if ct.effect == "write":
        return router.execute(ct, inp)                 # never predicted

    if ct.state in (OBSERVING,):
        return run_real_and_trace(ct, inp)

    pred, conf = predictors[ct].predict(inp)          # microseconds to ms

    if ct.state in (SHADOW, DEMOTED):
        real = run_real(ct, inp)
        comparator.record(ct, inp, pred, conf, real)   # async
        return real

    # PROMOTED
    if conf >= ct.tau_serve:
        if sampler.audit(ct, conf):                    # rate ρ, adaptive
            spawn(audit, ct, inp, pred)                # async real call, never blocks
        trace(ct, inp, pred, source="predictor")
        return pred
    return run_real_and_trace(ct, inp)                 # low confidence → real call
```

Audits are **asynchronous by default**. The agent keeps the predicted value and the real result is used only for scoring. Sync audit mode exists for high-stakes call types, and in that mode the real result wins.

---

## 6. SDK and data model

### 6.1 Developer API

```python
from dreaming import dream, Effect
from pydantic import BaseModel
from typing import Literal

class Ticket(BaseModel):
    subject: str
    body: str
    customer_tier: Literal["free", "pro", "enterprise"]

class Triage(BaseModel):
    intent: Literal["where_is_order", "refund", "damaged", "cancel", "other"]
    urgency: Literal["low", "med", "high"]
    order_id: str | None

@dream.call("triage_ticket", effect=Effect.PURE, cost_model="llm")
def triage(t: Ticket) -> Triage:
    return llm.structured(prompt=TRIAGE_PROMPT, input=t, output=Triage)

@dream.tool("shipping_status", effect=Effect.READ, ttl="10m",
            alternates=["shipping_status_backup"])
def shipping_status(tracking: str) -> ShipStatus: ...

@dream.tool("issue_refund", effect=Effect.WRITE)
def issue_refund(order_id: str, amount: float) -> RefundResult: ...
```

Auto-wrapping (zero-annotation mode): `dream.instrument(anthropic_client)` and `dream.instrument_mcp(session)` create call types from `(call site, prompt template hash, tool name)`. Outputs without a schema are traced but stay `OBSERVING` forever, and the dashboard flags them as "add a schema to unlock savings".

### 6.2 Trace schema

```text
trace_id          uuid
run_id            uuid         -- one agent run (workflow execution)
span_id/parent    uuid         -- OpenTelemetry compatible; step ordering
call_type         str
call_version      int
source            enum(real, predictor, real_after_lowconf, audit)
input             json         -- canonicalized, validated against input schema
input_digest      bytes        -- hash for dedupe / exact-match tier
output            json|null    -- validated against output schema
error             {class, message, http_status, retryable}|null
latency_ms        float
cost_usd          float        -- from tokens × price table, or API price
tokens            {in, out, reasoning}
predictor_version int|null
confidence        float|null
agreement         float|null   -- filled by comparator for shadow/audit pairs
context           json         -- endpoint, region, hour, recent error counts, run step idx
model_id          str|null     -- upstream model used for real LLM calls
ts                timestamp
```

**Labels come only from real calls.** Predictor outputs are never used as training targets. This keeps the system from training on its own guesses and collapsing.

---

## 7. Predictors

### 7.1 Predictor ladder

Each call type gets the cheapest predictor that clears the bar. The Dreamer trains candidates in order and keeps the best one on the calibrated risk-coverage curve (§7.4).

| Tier | Predictor | Good for | Cost/latency |
|---|---|---|---|
| 0 | **Exact memo** (input digest → output, with TTL for `read`) | Repeated identical inputs | ~0 |
| 1 | **kNN over embeddings** (answer = consensus of the k nearest past real outputs; confidence = agreement among neighbors × distance kernel) | Near-duplicate inputs, small data | < 5 ms |
| 2 | **Gradient-boosted trees / logistic regression** over [embedding ‖ structured features] | Enum / bool / small structured outputs | < 5 ms |
| 3 | **Small fine-tuned LM** with constrained (schema-guided) decoding | Structured extraction with copy spans, multi-field JSON | 20–200 ms, local GPU/CPU |

For the demo, tiers 0–2 are enough and train in seconds. Tier 3 is post-demo.

### 7.2 Typed outputs → typed heads

A predictor's output is built field by field from the output schema:

| Field type | Head | Field confidence |
|---|---|---|
| `Literal` / enum / bool | classifier | calibrated class probability |
| `int` / `float` | quantile regressor | interval width vs tolerance |
| `str` copied from input (ids, names) | span extractor (regex/learned pointer) | extractor score |
| `str` free-form | **not predicted in v1** | n/a: call type ineligible unless field is marked `ignore_for_agreement` |
| nested object / list | recursive heads; lists via set prediction | min over children |

**Call-level confidence = minimum over field confidences**, then recalibrated on held-out shadow data.

**CoT steps.** We never predict the reasoning text. A CoT call is eligible when its *final decision* is typed (e.g. `plan: list[ToolName]`, `next_action: Literal[...]`). The reasoning tokens are where most of the cost goes, and skipping them is the point.

### 7.3 Agreement metric

Agreement is defined per field, using the schema:

- enum/bool/int: exact match.
- float: `|a − b| ≤ tol` (declared, or a default relative tolerance).
- copied string: normalized exact match.
- list: set equality (or Jaccard ≥ declared threshold if order and completeness don't matter).
- fields marked `ignore_for_agreement` are excluded.

**Downstream agreement (optional, stronger).** Two outputs agree if the agent's *next action* is the same under both. This is measured cheaply when the next step is itself a traced call type. It catches harmless differences (e.g. `urgency: med` vs `high` when both lead to the same route).

**Self-agreement baseline.** For LLM call types, a small sample is run twice against the real model to estimate its self-agreement `s`. The promotion threshold is relative: the predictor must agree with the model at least as often as `s − ε`. This prevents demanding 99% from a predictor when the frontier model only agrees with itself 94% of the time at temperature > 0.

### 7.4 Calibration and serving threshold

This is selective prediction: the predictor answers only above a confidence threshold `τ_serve`.

1. Hold out a calibration set of recent shadow pairs.
2. Fit isotonic/temperature calibration on confidence → agreement.
3. Pick `τ_serve` as the lowest threshold where the **upper confidence bound on disagreement** for the covered set stays ≤ the error budget `δ` (e.g. δ = 1%). This is a split-conformal / Clopper–Pearson style guarantee.
4. Coverage at `τ_serve` is the call type's savings potential, and it shows on the dashboard.

### 7.5 Promotion test

Promote a call type `SHADOW → PROMOTED` when **all** of the following hold:

- `n_shadow ≥ N_min` (default 50 for demo, 500 for prod).
- Wilson lower bound of agreement on the covered set at `τ_serve` ≥ `max(τ_promote, s − ε)`.
- Coverage ≥ `c_min` (default 20%). Promoting a predictor that never answers is pointless.
- No slice is failing: agreement holds per major slice (e.g. per customer tier, per intent class) with a minimum per-slice n.
- Cooldown satisfied if previously demoted.

### 7.6 Audit sampling and demotion

- **Audit rate `ρ`** starts high after promotion (e.g. 20%) and decays toward a floor (e.g. 2–5%) as audited agreement holds. It jumps back up on any drift signal. Low-confidence-but-served calls (just above `τ_serve`) are sampled at a higher rate.
- **Demotion triggers** (any one):
  - Sequential test on audit disagreement (CUSUM / SPRT) crosses the bound.
  - Input drift: embedding-space OOD score or population-stability index on features exceeds a threshold.
  - Upstream change: model id, prompt template hash, tool schema, or API version changes.
  - Manual kill switch per call type, or globally.
- Demotion is instant and local (Gateway reads state on every call). Retraining happens in the background.

---

## 8. Tool model and router (the reliability half)

### 8.1 What it predicts

For each tool endpoint `e` and call input `x`, given context `z` (time of day, recent error/latency history, region, payload features):

- `P(fail | e, x, z)` and the failure class (timeout, 5xx, 4xx-semantic, malformed response)
- latency distribution (p50/p95)
- for `read` tools: optionally the *output* itself (tier 0/1 learned cache, with TTL)

Model = **learned prior + online state**:
- *Learned prior:* a GBDT over `(x, z)` trained on traces. It captures structured failure patterns, e.g. "provider A times out on international tracking numbers" or "A degrades under burst load".
- *Online state:* per-endpoint EWMA error rate and latency, plus a Bayesian (Beta) health posterior updated on every call and on optional lightweight health probes. This reacts within one or two calls.

`P(fail) = combine(prior, online)`, e.g. logistic combination with online state as features.

### 8.2 Router

Given a tool call with declared alternates `{e1, e2, …}` (same input/output schema, or an adapter):

```
choose e* = argmin_e  E[cost_e] + λ·P(fail_e) + μ·E[latency_e]
```

with exploration (Thompson sampling on the health posterior) so a recovered endpoint gets traffic again. For a `write` tool, routing only happens *before* the call. After a failure, retry is allowed only when the tool is declared idempotent or provides an idempotency key.

### 8.3 Distilled recovery policy

When a tool fails without a declared alternate, the agent normally asks the frontier model what to do: retry, use a different tool, degrade gracefully, or escalate. **That recovery decision is just another call type** (`recover_from_tool_failure`, typed output `Literal["retry", "use:<tool>", "skip_step", "escalate"]` + args). It gets traced, shadowed, and promoted like any other.

During training runs, flaky endpoints produce many frontier-handled recoveries. Once the recovery call type is promoted, failures are handled by the predictor, with no frontier call.

### 8.4 The tool-kill moment: what is true and what to claim

When we kill a tool live:

1. The next call to it fails (or a health probe fails first, if probes are enabled at ~1 s intervals). The online health posterior collapses within that one observation, because the learned prior already gave this endpoint a meaningful failure probability: it has been flaky all session.
2. The Gateway invokes `recover_from_tool_failure`. It is `PROMOTED`, with high confidence for `(tool=shipping_status, error=connection_refused)`, because it has seen dozens of frontier-handled examples.
3. The predicted recovery says `use:shipping_status_backup`. The router executes it, and every subsequent run routes to the backup **pre-emptively**: `P(fail)` for the dead endpoint is now ~1, so no call is wasted on it.
4. The dashboard shows: tool killed → 0 frontier calls for recovery → run succeeds → next runs skip the dead endpoint entirely.

**Honest framing for stage:** "The agent had already learned how to recover from this tool failing, so it didn't need to ask the frontier model what to do. After the first failure, it stopped calling the dead tool altogether." To make the *first* call pre-emptive too, run health probes and kill the tool between runs. The probe catches it, and the router avoids it before any run touches it. Both variants are worth rehearsing. Pick based on how it plays.

---

## 9. The Dreamer (training loop)

Per call type, triggered when `new_traces ≥ K` (demo: K = 10) or every T seconds:

```
1. load traces (real labels only) for current call_version, recency-weighted
2. featurize: canonicalize input → structured features + text embedding (cached by digest)
3. train candidates on the predictor ladder (tiers allowed by config)
4. calibrate on held-out shadow pairs; compute risk-coverage curve; pick τ_serve
5. evaluate promotion test (§7.5); compare against current predictor version
6. register new version if it dominates on (coverage at δ); hot-swap via registry
7. emit metrics to dashboard
```

- **Featurization** is shared across call types: one embedding model (small, local, e.g. a ~30M-parameter sentence embedder) plus schema-driven structured features (enums one-hot, numbers scaled, string-length/regex features).
- **Speed budget (demo):** tiers 0–2 retrain in under 2 s per call type on ≤ 5k traces. Retraining fast enough is what makes the curve visibly fall during a live run.
- **Warm start:** when a call type version changes, the previous version's traces are used only after re-validation in shadow (they may be stale).

---

## 10. Cost and reliability accounting

- **Cost per run** = Σ over calls in the run: real-call cost (tokens × price table or API price) + predictor inference cost (amortized compute) + audit cost (async audits attributed to the run that triggered them).
- **Failure rate** = fraction of runs that end in an unrecovered error or fail the workflow's success check.
- **Correctness** = end-to-end task success measured against a fixed evaluation set. It is re-run periodically against the full real-call path to confirm distilled runs aren't silently worse.

**Expected shape (illustrative targets, not measurements):**

| Phase | Runs | Cost / run | Failure rate | What's happening |
|---|---|---|---|---|
| Cold | 0–30 | 100% (baseline) | ~15% | Everything real. Flaky endpoint fails often. |
| Shadow | 30–80 | ~100% | falling | Predictors training. Tool model starts routing. |
| Promotions | 80–200 | stepping down to ~30% | ~3% | Triage, extraction, policy, and recovery promoted one by one (visible steps in the curve). |
| Steady | 200+ | ~10–20% | ~1% | Floor = audits + long-tail low-confidence calls + non-distillable reply step. |

---

## 11. Demo plan

### 11.1 Workflow: support ticket resolver

A realistic, mostly distillable agent. Per run, one synthetic ticket flows through:

| Step | Call type | Kind | Effect | Distillable? |
|---|---|---|---|---|
| 1 | `triage_ticket` | frontier LLM, structured | pure | ✅ enum + id extraction |
| 2 | `lookup_order` | API (primary + replica) | read | ✅ learned cache / routing |
| 3 | `shipping_status` | API (**provider A flaky**, provider B backup) | read | routing via tool model |
| 4 | `plan_resolution` | frontier CoT → typed plan | pure | ✅ plan is `list[Action]` |
| 5 | `check_refund_policy` | frontier LLM, structured | pure | ✅ bool + reason code |
| 6 | `issue_refund` / `create_return` | API | write | ❌ never replaced (routed only) |
| 7 | `draft_reply` | cheaper LLM, free text | pure | ❌ in v1 (stays real; sets the cost floor) |
| — | `recover_from_tool_failure` | frontier LLM, structured | pure | ✅ the kill-moment hero |

**Ticket generator.** It samples from a realistic, long-tailed distribution: ~70% from ~15 common archetypes with paraphrase variation, ~30% long-tail. This makes the curve fall convincingly *and* leaves a visible floor. A pure-repetition dataset would look fake.

**Flakiness model for provider A.** Structured, not random, so the tool model can learn something real:
- times out on international tracking numbers (prefix pattern) ~60% of the time
- 5xx bursts during simulated "peak" windows
- ~3% background noise

### 11.2 Stage layout

```
┌──────────────────────────────────────────────┬──────────────────────────┐
│  Cost per run ($)            ── falling      │  Call types              │
│  Failure rate (%)            ── falling      │  triage_ticket  PROMOTED │
│  (rolling mean, x = run #, promotion events  │  plan_resolution SHADOW  │
│   marked as vertical ticks on the chart)     │  refund_policy  PROMOTED │
│                                              │  recover_…      PROMOTED │
├──────────────────────────────────────────────┤  shipping A  ⚠ p(fail).41│
│  Live run feed: step → real / predicted /    │  shipping B  ✓ p(fail).02│
│  audit / routed, with $ per step             │  [ KILL provider B ]     │
└──────────────────────────────────────────────┴──────────────────────────┘
```

(Kill the endpoint the agent is currently *relying on*, i.e. whichever is healthier at that moment, so the reroute is meaningful.)

### 11.3 Run-of-show (~6 minutes)

1. **0:00** Explain the agent. Show one run with every step real. Show the cost.
2. **0:45** Start the loop: 300 runs at concurrency ~8. Both lines start high.
3. **1:30–3:30** Narrate promotions as they land: "triage just got promoted, so that step costs zero now unless it's unsure"; "the agent learned provider A fails on international numbers and stopped sending them there."
4. **3:30** **Kill the tool** from the dashboard button. Live feed shows: failure → `recover_from_tool_failure` served by predictor (0 frontier calls) → backup used → run succeeds. Subsequent runs route around it.
5. **4:30** Show the audit trail: "it still double-checks 3% of its guesses against the real model, and here's the agreement rate."
6. **5:30** Close on the two curves.

### 11.4 Demo engineering requirements

- **Throughput:** 300 runs in ~3 minutes means ~1.7 runs/s. Frontier calls happen mostly early, so use concurrency 8–16 and request-rate headroom with the provider.
- **Determinism / risk:** seedable ticket generator and flakiness. A **replay mode** re-runs a recorded session with real timings, as a fallback if the network fails on stage. The live tool kill must work in both modes. In replay mode, the kill injects a real failure into the local mock tool.
- **Mock tools:** the shipping/order APIs are local FastAPI services we control. That gives deterministic flakiness and a real kill (process stop / port close).
- **Accelerated retraining:** Dreamer K = 10 traces and a < 2 s train budget, so promotions happen in the window.
- **Pre-flight:** a rehearsal script asserts that promotions land between runs 60–200 and that the final cost is ≤ 25% of baseline.

---

## 12. Tech stack (v1)

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.12 | Agent ecosystem, ML libs |
| Schemas | Pydantic v2 | Typed I/O, JSON schema for constrained decoding |
| LLM provider | Claude API (frontier model for real calls; a smaller model for `draft_reply`) | Structured outputs, tool use |
| Embeddings | Small local sentence-embedding model | Fast, free, offline |
| Predictors | scikit-learn, LightGBM | Train in seconds, calibrate easily |
| Trace store | DuckDB (+ Parquet export) | Zero-ops, fast analytics |
| Registry | SQLite + artifacts dir | Simple, atomic swaps |
| Mock tools | FastAPI | Controllable flakiness, killable |
| Dashboard | FastAPI WebSocket + React + a charting lib | Live charts |
| Telemetry | OpenTelemetry span ids in traces | Interop later |

---

## 13. Repository layout (proposed)

```
dreaming/
  sdk/            decorators, client instrumentation, gateway, effect invariants
  traces/         trace schema, emitter, DuckDB store
  predictors/     ladder (memo, knn, gbdt), typed heads, calibration
  dreamer/        training worker, scheduler
  policy/         promotion/demotion tests, audit sampler, drift detection
  tools/          tool model, health posterior, router, recovery call type
  registry/       call-type + predictor version registry
demo/
  agent/          support-ticket resolver workflow
  mocks/          order + shipping services (flaky A, backup B), kill endpoint
  generator/      ticket distribution
  replay/         record / replay harness
UI/               live dashboard
design/           this doc
research/         notes: selective prediction, distillation, conformal bounds
video/            demo recordings
```

---

## 14. Plan and milestones

The demo-critical path is marked ★.

| # | Milestone | Deliverables | Exit criteria |
|---|---|---|---|
| M0 ★ | **Baseline agent + mocks** | Support workflow on real frontier calls. Mock order/shipping services with configurable flakiness and kill. Ticket generator. | 300 runs complete. Baseline cost/run and failure rate recorded. |
| M1 ★ | **Tracing** | SDK decorators, trace schema, DuckDB store, cost accounting. | Every call in a run is traced with cost. Run-level rollups correct. |
| M2 ★ | **Shadow predictors** | Ladder tiers 0–2, typed heads for enum/bool/id fields, comparator, Dreamer loop. | Shadow agreement visible per call type. Retrain < 2 s. |
| M3 ★ | **Promotion + audit** | Calibration, `τ_serve` selection, promotion test, audit sampler, demotion on disagreement. | Cost/run falls ≥ 60% over 300 runs with end-to-end task success within 1 pt of baseline. |
| M4 ★ | **Tool model + router + recovery** | Health posterior, GBDT failure prior, router over alternates, `recover_from_tool_failure` call type. | Failure rate falls ≥ 70%. Kill-tool recovers with 0 frontier calls. |
| M5 ★ | **Dashboard + replay** | Live charts, call-type state panel, kill button, record/replay. | Full run-of-show rehearsed 3× clean, including replay fallback. |
| M6 | **Hardening** | Drift detection (OOD/PSI), slice checks, self-agreement baseline, version resets, kill switches. | Injected drift triggers demotion within 50 calls. |
| M7 | **Generalize** | Auto-instrumentation for LLM clients + MCP, tier-3 small-LM predictors, Parquet store. | Wrap a third-party agent with < 30 lines changed. |

Suggested sequencing: M0 → M1 → (M2 ‖ M4 tool-model prior) → M3 → M4 recovery → M5. M6–M7 come after the demo.

---

## 15. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **Error compounding across steps** (5 promoted steps × 1% each ≈ 5% run-level divergence) | Silent quality loss | Per-run error budget split across steps. Downstream agreement metric. Periodic full-real eval runs comparing end-to-end success. |
| **Frontier non-determinism** makes "agreement" noisy | Over-strict or meaningless thresholds | Self-agreement baseline (§7.3). Temperature 0 where possible. Relative thresholds. |
| **Free-text outputs** dominate cost in some agents | Smaller savings | Push typed decisions upstream of text generation. v2: template + slot predictors. Be upfront that the floor exists. |
| **Long-context inputs** (big documents, long histories) are hard to featurize | Low coverage | Featurize on salient extracted fields + embeddings of a summary. Mark as low-coverage candidates. |
| **Stale `read` predictions** (cached order status changed) | Wrong facts | TTL per tool. Never predict reads that feed a `write` without a fresh real read (configurable invariant). |
| **Drift** (new product, new ticket types) | Degraded predictions | OOD gating per call (OOD inputs → real call), audit-rate bump, auto-demotion. |
| **Feedback loops** (training on predictions) | Model collapse | Labels only from real calls (§6.2 invariant, enforced in the Dreamer query). |
| **Audit selection bias** (audits only where confident) | Overestimated agreement | Low-confidence fall-throughs also produce labels. Random audits include a small slice below `τ_serve`. |
| **Privacy:** traces contain user data | Compliance | PII redaction hooks at emit time. Per-field `sensitive` flag. Local-only storage by default. |
| **Demo:** network or rate limits on stage | Failed demo | Replay mode. Local mock tools. Pre-warmed connection pool. Rehearsal assertions. |
| **Demo looks rigged** (too-perfect curves) | Credibility | Long-tail tickets, visible audit disagreements, honest floor. Show the trace table live. |

---

## 16. Metrics

**Per call type:** state, n traces, shadow/audit agreement (with CI), coverage at `τ_serve`, audit rate, $ saved, p50 latency real vs predicted.

**Per run:** cost, latency, frontier-call count, predicted-step count, routed-step count, success/failure.

**System:** cost/run trend, failure rate trend, end-to-end success vs the full-real baseline, demotions per day, drift alarms.

---

## 17. Open questions

1. **Call identity in auto-instrumented mode.** Is prompt-template hash + call site stable enough, or do we need explicit names? (Leaning: explicit names required for promotion, auto mode only for observation.)
2. **Threshold policy owner.** Is `δ` (error budget) set per call type by developers, or globally with per-effect defaults?
3. **Sync vs async audits.** Should high-stakes call types (e.g. `check_refund_policy`, which gates a `write`) always audit synchronously?
4. **Tool equivalence.** v1 needs declared alternates. Can we learn equivalence from schema + output agreement in v2?
5. **Where predictors run** for multi-process / multi-host agents. Shared predictor service vs in-process copies?
6. **Downstream-agreement scope.** One step lookahead, or full-run outcome equivalence on sampled runs?
7. **Demo kill variant.** Kill mid-run (first call fails, recovery is distilled) or between runs with probes (fully pre-emptive)? Rehearse both, decide on stage feel.

---

## Appendix A: Glossary

- **Call type:** a named, versioned, schema-typed call site.
- **Shadow:** the predictor runs alongside the real call, which stays authoritative.
- **Promotion:** the predictor becomes authoritative above a confidence threshold.
- **Coverage:** the fraction of calls the predictor answers.
- **Audit:** a sampled real call made for a promoted call type, used to monitor agreement.
- **Tool model:** the predictor of tool outcomes (failure, latency), used for routing.
- **Recovery policy:** the distilled decision of what to do after a tool failure.
- **Dreamer:** the background training worker.
