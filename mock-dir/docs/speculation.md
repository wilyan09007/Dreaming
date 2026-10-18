# Speculation: running workflow nodes in parallel on predicted values

The question asked on 2026-10-17: can the runtime run a workflow's nodes in parallel and use tool and step predictions to start
downstream work before upstream calls return, for a large speed-up? Short answer: **yes, it is buildable in the runtime we
have, and it is lossless (the user never sees a guessed value), but the gain is bounded by the few real calls that remain:
7–14% of mean latency on the bookkeeper at realistic hit rates, on top of what plain DAG parallelism gives.** The big latency
win is coverage (a dreaming site answers in milliseconds instead of 0.3–1.2 s); speculation only shortens the chain of calls
that still go out for real. Numbers below are from `experiments/speculation/sim.py` (run it) and the 2026-10-17 smokes; the
literature numbers are in `../../smokes/research/tool-prediction-speculation.md`. It is not in tonight's build order (§5).

## 1. Three sources of speed, and what each needs

| Source | What it does | Needs | Gain on the bookkeeper (simulated from measured inputs) |
|---|---|---|---|
| **Coverage** (dreaming) | a certified predictor answers instead of the call | the loop we have | a predicted call is ~5 ms on a free core (300 ms on Fly's shared CPU) vs 300–1,200 ms real; at the measured 40–67% coverage the mean run goes from 2.5 s (all awake, sequential) to 1.2 s |
| **DAG parallelism** | independent nodes run at the same time | the runtime must know the graph, or get lazy futures (§3) | 1.21 → 1.10 s mean (9%), p95 2.5 → 2.0 s: `categorize` and `vendor_lookup` overlap; on Fly's CPU 1.70 → 1.36 s (20%) |
| **Speculation** | a node starts on the predictor's top guess for an input still in flight; kept if the real value matches, redone if not | a confident guess (any candidate's `Prediction`), pure nodes, effects that wait | 1.10 → 1.02 / 0.95 s mean at hit rates 0.3 / 0.6 (7% / 14%); p95 2.0 → 2.0 / 1.7 s; wasted teacher spend ≤ $0.0007 per run |

Simulated with the per-site coverage measured on the deployed stack with the rule teacher (categorize 40%, vendor_lookup 67%,
match_invoice 60%; the router site always a real 300 ms call), asi1-mini latency 0.5 s per typed step, the mock vendor API 1.2 s,
a wasted guess into a teacher call at $0.0014 (the measured $0.0042 per 3-call live run), 20,000 runs:

```
predicted call = 5 ms (a free core)          predicted call = 300 ms (Fly shared-cpu-1x, measured)
executor      hit   mean s   p95 s  $/run     executor      hit   mean s   p95 s  $/run
sequential      -     1.21    2.50  0.00000   sequential      -     1.70    2.50  0.00000
dag             -     1.10    2.00  0.00000   dag             -     1.36    2.00  0.00000
speculate     0.3     1.02    2.00  0.00068   speculate     0.3     1.26    2.00  0.00068
speculate     0.6     0.95    1.70  0.00039   speculate     0.6     1.18    1.80  0.00039
speculate     0.9     0.86    1.50  0.00010   speculate     0.9     1.09    1.50  0.00010
```

Why the ceiling is where it is: a run's latency with speculation is bounded below by its slowest real call (the vendor API at
1.2 s when the lookup is not certified), and every node that is already predicted costs the same whether speculated or not. The
published top-1 hit rates for guessing a tool's return or the next call are 22–29% (Speculative Actions on tau-bench, PASTE,
AOSpec) and 44–46% at top-3; our own S4 measured 0.63 on 71% of calls for a hidden-state tool in a synthetic env. So plan on
the 0.3–0.6 rows: 7–14% of mean latency and 0–15% of p95. Speculation pays most in workflows with long chains of slow calls
(coding and research agents: 17% of calls hold 97% of tool time); the bookkeeper is not that workflow.

## 2. What is lossless and what is not

- A **guess is never returned to the user and never written anywhere**. A speculated node's output is held until every input it
  used is verified (the real value arrived and equals the guess); on mismatch the node is re-run on the real inputs and the
  speculative work is discarded. Effects (`@effect`) and anything that leaves the process wait for verified inputs. This is the
  acceptance rule of AOSpec and Speculative Actions, and the same rule that keeps replay byte-identical.
- A **certified** answer (above the cutoff) is not a guess: the real call is skipped, there is nothing to verify, and the
  dependent runs at once. Speculation is for the band *below* the certified cutoff and above a lower **speculation threshold**
  (`Thresholds.speculate_at`, default None: never; a sensible value is the confidence at which the predictor is right half the
  time on calibration rows).
- **Hidden-state tools** (a refund's status): the guess is the predictor's top return; the real call always runs; the next
  read-only node starts on the guess. Same mechanism.
- **Cost**: a wrong guess into a node whose real call is a frontier model wastes one teacher call (~$0.0014); into a predicted
  node it wastes milliseconds. A cap of one speculative branch per in-flight input (top-1 only) and a per-run budget of wasted
  calls (default 2) bound the spend. The row records it: `TraceRow.speculated=true`, `kept=false`, so the dashboard can show the waste.

## 3. Design: lazy futures, no DAG declaration

The customer's code stays sequential Python. The runtime already intercepts every call site; the change is that a call returns at
once with a **`Pending`** handle and the real work runs in a thread pool. `run_transaction` keeps its shape:

```python
category = categorize(txn)                      # Pending: submitted, not waited for
vendor = vendor_lookup(descriptor)              # Pending: runs in parallel with categorize
invoice = match_invoice(txn, vendor, category, open_invoices(vendor.result()))   # a plain helper needs the verified value
odd = anything_odd(txn, vendor, category, invoice, history(vendor.result()))
return wake_a_human(...) if odd.result() == "yes" else post_to_books(...)         # the branch forces the verified value
```

- **`Pending`** (`speculate.py`): `.guess` (the best available `Prediction` for this call, set as soon as the shared embedding and
  the shadow heads have run), `.result()` (blocks for the verified value), `.verified` (an `Event`). A `Pending` is resolved only
  by the runtime, when a registered site renders it, or explicitly by `.result()`. There is no magic proxy: handing one to plain
  Python (string formatting, a comparison, a helper) is a type error, not a silent fallback, which is why the two helper calls
  above gain `.result()` or become `@tool`s. A runtime flag (`speculate=False`, the default today) makes every call return plain
  values, which is exactly the current behaviour.
- **Submitting a call whose inputs are `Pending`**: `router.route` is unchanged; the dispatcher in `runtime.py` decides, per
  input, whether to render on the guess (confidence ≥ the input site's `speculate_at` and the input's site is not an effect) or
  wait. If it rendered on any guess, the row is `speculated=true` and the result is published only after each guessed input's
  `.verified` fires with an equal value; otherwise the call is re-submitted with the real inputs and the first row is kept with
  `kept=false`.
- **Effects** always `.result()` their inputs. So do tools that are not pure: `@tool(pure=False)` opts out of being speculated
  *into*; every tool is still a legal *source* of guesses.
- **Replay** is unchanged: a speculated row that was kept re-executes on its logged rendered prompt exactly as any predicted or
  called row; a discarded row is skipped (`kept=false`). Run order in the log is submission order, so `i` stays deterministic.
- **Fast-forward** keeps the virtual clock: a run's latency becomes the critical path over its rows' latencies (`Run.critical_ms`),
  computed from the `Pending` graph, not the sum; the page's "sec per run" line therefore moves on its own when speculation lands.
- **Connectors**: LangGraph already runs independent branches in parallel; the OpenAI-client wrapper and the MCP proxy cannot
  start a node early (they do not own the graph), so they get coverage and DAG parallelism from the framework and no speculation.
  Our own runtime and the uAgents handler get all three.

What is already in the contracts (migration 0002, they cost nothing): `TraceRow.speculated`, `TraceRow.kept`,
`Thresholds.speculate_at`, `Run.critical_ms`. What remains: one module (`speculate.py`, replacing the stub), one dispatcher
change, one test: a run with a 1.2 s mock tool and a certified student downstream completes in max(1.2 s, ·) virtual time, and a
forced mismatch re-runs the dependent and marks the first row `kept=false`.

## 4. What would make it "blazing"

- Every site dreaming: a 4-step run is four embeddings and four heads, ~20 ms on a free core, two orders of magnitude under
  the 2.5 s all-awake run. That is coverage, already in the loop; the campaign's job is to raise it.
- Rollout (mission 7): imagine the whole trajectory from the first input, then run only the calls the imagination was unsure
  about. This is speculation chained through every node at once; it needs the same `Pending` machinery and adds nothing new to the
  contracts. S9 measured 76% of 4-step trajectories imagined correctly end to end.
- The shared embedding cache: with the ledger's texts pre-embedded (`just record`), the ~300 ms per prediction seen on Fly's
  shared CPU disappears (the 1.70 → 1.21 s column above), which matters more for the demo's throughput than speculation does.

## 5. Verdict

Buildable, lossless, about a day of work, no new service or table, and worth 7–14% of the residual latency on this workflow.
That is not a tonight item: the campaign and the sponsor evidence come first, and the honest claim on the page is "nodes run as
soon as their inputs are predictable; nothing a user sees is ever a guess", shown next to the coverage number that does the
heavy lifting. Build it after the hackathon, or on Sunday morning only if everything else is done.
