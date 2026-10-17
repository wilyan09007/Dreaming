# Demo

One demo, two falling lines, one live moment. Source: [`../research/00-unified-runtime-brief.md`](../research/00-unified-runtime-brief.md) section 5.

## 1. What the audience sees

Run the same agent workflow a few hundred times on stage with a chart of two falling lines:

1. **Cost per run**, dropping as steps get distilled and promoted.
2. **Failure rate**, dropping as the tool model lets the agent route around flaky endpoints.

Then **kill a tool mid-run**. The agent reroutes without a frontier call because its predictor already expected the failure.

## 2. Workflow: QuickBooks-style transaction processing

Uses Intuit's synthetic QuickBooks data (schema and labels to be confirmed; see `../research/05-sponsors-and-track-fit.md`). One run processes one transaction end to end.

| Step | Call type | Kind | Effect | Plan |
|---|---|---|---|---|
| 1 | `fetch_bank_feed` | tool (bank-feed API, flaky) | read | Routed; learned fallback on failure |
| 2 | `lookup_vendor` | tool (vendor-lookup API, flaky) | read | Routed; cached vendor directory as alternate |
| 3 | `categorize_transaction` | frontier LLM, typed | pure | **Distill target.** Output: category enum (plus confidence) |
| 4 | `match_invoice` | LLM or retrieval, typed | pure | Stays real in the baseline plan. Candidate to distill if time allows |
| 5 | `flag_anomaly` | frontier LLM, typed | pure | **Distill target.** Output: bool plus reason code |
| 6 | `draft_note` | LLM, free text | pure | Not distilled. Sets the cost floor |
| - | `recover_from_tool_failure` | frontier LLM, typed | pure | Distilled from frontier-handled failures. The kill-moment hero |

Categorize and flag are the distill targets (brief). Tools are a bank-feed API and a vendor-lookup API with injected flakiness. Whether step 4 is distilled is decided after the offline predictability check.

> Step ordering and the exact tool/field names are a proposal. Adjust to the real dataset.

## 3. Mock tools and flakiness

Local services, killable, seeded (see `../design/CONTRACTS.md` section 9).

| Service | Endpoints | Flakiness profile (structured, so a model can learn it) |
|---|---|---|
| Bank feed | primary, replica | Primary: timeouts clustered in "batch window" periods (a simulated time-of-day feature); replica healthier but slower |
| Vendor lookup | primary only, plus a learned/cached vendor directory as fallback | Primary: fails on a pattern in the input (for example vendor-name prefix class) at a high rate, plus ~3% background noise |

The flakiness must depend on **observable features** (time bucket, input pattern, recent error counts). Purely random failures cannot be predicted, so they would not show learning, and showing learning on them would be rigged.

## 4. Data generator

- Seeded. About 70% of rows from common patterns with variation, about 30% long tail (rows unlike the common patterns).
- Long tail guarantees a visible cost floor and keeps the curve honest.
- Same seed and same flakiness schedule for baseline and Dreaming arms.

## 5. Run of show (about 6 minutes)

| Time | Beat | What the audience sees |
|---|---|---|
| 0:00 | One run, everything real | Step feed shows each frontier call and its cost |
| 0:45 | Start 300 runs, concurrency about 8 | Both lines start high |
| 1:30 to 3:30 | Narrate promotions | Vertical ticks on the chart as `categorize` then `flag_anomaly` promote; tool panel shows `p(fail)` rising on the flaky endpoint and routing shifting |
| 3:30 | **Kill a tool** (button) | Feed: failure, then `recover_from_tool_failure` served by predictor (0 frontier calls), fallback used, run succeeds; later runs skip the dead endpoint |
| 4:30 | Audit trail | "It still double-checks a few percent of its guesses against the real model; here is the live agreement rate" |
| 5:30 | Close | The two curves, baseline vs Dreaming table (latency, cost, accuracy on held-out rows) |

Kill variants (decide at rehearsal, see `../design/DESIGN.md` section 8.4):
- **Mid-run kill:** first call fails, recovery is the distilled predictor. Honest framing: "it already knew how to recover."
- **Between-run kill with health probes:** fully pre-emptive. More impressive, needs probes working.

## 6. Engineering requirements

- Throughput: 300 runs in about 3 minutes is about 1.7 runs/s; use concurrency 8 to 16 and check provider rate limits early.
- Fast retraining: Dreamer triggers every ~10 new traces per call type with a train budget under 2 s (local tiers 0 to 2) so promotions land inside the window.
- **Demo error budget:** use `delta = 5%` (promotion needs about 60 clean shadow samples; 1% would need about 300). Say it on stage. See `../research/03-statistics-of-promotion.md`.
- **Replay mode:** record a full session (inputs, frontier outputs, timings, failure schedule) and replay it deterministically if the network fails. The kill must still be a real failure injected into the local mock.
- Pre-flight script asserts: promotions land between runs about 60 and 200, final cost at most about 25% of baseline, kill recovery shows 0 frontier calls, no unhandled errors.

## 7. Fallbacks

| Failure | Fallback |
|---|---|
| No network or API rate limit | Replay mode |
| Promotion lands too late or too early | Pre-warmed registry snapshot at a chosen run number |
| Kill moment misfires | Rehearsed second variant; recorded clip in the video |
| Dashboard crash | Terminal view of the same event stream |

## 8. Honesty checklist

- Long-tail rows shown, floor explained.
- Audit disagreements visible, not hidden.
- Baseline and Dreaming use the same seed.
- Illustrative numbers never presented as measured.
