# Dreaming: a runtime that learns a model of every call your agent makes

Recorded 2026-10-16. Fusion of the distill-as-you-go runtime (F1, current rank 1) and the imagination-loop agents (F5, rank 9) into one mechanism. Name: Dreaming. The runtime dreams each call before making it, and the name fits the Dreamer lineage of learning a model and planning inside it.

---

## 1. One-sentence pitch

A runtime that learns a model of every call your agent makes, then stops making the ones it can predict. The agent gets cheaper and more reliable the more it runs.

## 2. Why merge, and the condition

The two ideas share everything that matters: the same data (traces), the same gate (shadow run until agreement with reality crosses a threshold), the same promotion logic (swap the cheap predictor in, keep the expensive call as fallback). In RL terms the distill half is policy distillation and the environment half is a dynamics model.

The condition: it must be one mechanism applied to several call types, not two projects stapled together. The fusion judge in the Devpost sweep struck every proposal that read as a kitchen sink with no single demo moment, and both adversarial refuters hit F5 because a world model trained on one day of traces is a lookup table. The merge wins only if it fixes that rather than inheriting it.

## 3. The unifying principle, from fundamentals

Every expensive thing an agent does is a call:
- a call to the frontier model to perform a step,
- a call to a tool or API,
- a decision to spend chain-of-thought on a step.

The runtime logs all of them as structured traces. For each call type it trains a small typed predictor, runs the predictor in shadow next to the real call, measures agreement, and promotes the predictor when agreement clears a threshold. Promoted predictors answer first; the real call runs only on low confidence or on a sampled fraction for continued monitoring. Every component is a known pattern: request logging, distillation (teacher outputs become student labels), canary with automatic diff, semantic cache keyed on meaning, fast path and slow path.

## 4. What to model, in priority order

| Priority | Thing modeled | Replaces | Learned from | Verified at the judging table by |
|---|---|---|---|---|
| 1 | Narrow policy steps (classify, extract, decide) | Frontier model calls | Step inputs and teacher outputs | Agreement rate on held-out rows with the cache disabled |
| 2 | Tool outcomes: will this call fail, how long will it take, what will it cost | Blind tool calls and retry loops | Tool request and response logs, including failures | Predicted versus actual failure rate on live flaky endpoints |
| 3 | Thinking budget | Chain-of-thought on easy steps | Which steps changed their answer when reasoning was added | Accuracy with and without the router |
| 4 | Full environment rollouts (imagine a plan before executing) | Trial and error | All of the above | Success rate versus a baseline agent. Only if 1 to 3 land |

Priority 2 is what rescues the environment half. A typed predictor of tool reliability and latency is small, honest, and feeds a planner directly: if the tool is likely to fail, pre-plan the fallback before calling it. That earns Fetch's bonus for recovery from failed tool calls without claiming more than can be shown.

Build order is also the fallback plan. If only priority 1 ships, you still have the current number one idea. If only priority 4 ships, you have the idea the refuters rejected.

## 5. The one demo

Run the same agent workflow a few hundred times on stage with a chart that has two falling lines:
- cost per run, which drops as steps get distilled and promoted,
- failure rate, which drops as the tool model lets the agent route around flaky endpoints.

Then one live moment: kill a tool mid-run. The agent reroutes without a frontier call because its predictor already expected the failure.

Workflow to use: Intuit's synthetic QuickBooks data. Steps are categorize transaction, match to invoice, flag anomaly, draft note. Categorize and flag are the distill targets. Tools are a bank-feed API and a vendor-lookup API with injected flakiness. Report latency, cost and accuracy on held-out rows against the frontier-only baseline.

## 6. Worked example outside the demo

An e-commerce refund agent handles a ticket in four steps: classify intent, look up the order, decide eligibility, draft the reply. After a couple of thousand runs:
- Dreaming has promoted a small classifier for intent and a typed yes-or-no model for eligibility. Three of four frontier calls per ticket become one.
- The tool model has learned that the order-lookup API fails about one call in twenty during a nightly batch window, so the planner schedules a cached-order fallback before calling it in that window. Retries and stuck tickets drop.
- The router has learned that eligibility rarely changes with chain-of-thought, so that step runs without it.

Illustrative result: ticket latency from about nine seconds to three, cost per ticket down by roughly two thirds, stuck-ticket rate down by half. The shape is what you would measure, the numbers are placeholders.

## 7. Scoring against the event rubric

| Dimension | Estimate | Reason |
|---|---|---|
| Technical Depth | 5 | Training loop, shadow gating, tool-outcome model, router, planner, all with measured results |
| Product Thinking & Impact | 4 | Cost and reliability are the two things every agent team is paid to fix |
| Execution & Polish | 4 | Two demo moments instead of a dashboard. Risk is that too little ships |
| Originality & Insight | 4.5 | Nobody at these events has shipped the unified version. Distillation alone is OpenPipe, world models alone got refuted |
| Overall | about 4.4 | |
| Coverage | 5 | See section 8 |
| Combined | about 4.6 | Above the current top score of 4.34 |

Flags: execution risk, kitchen-sink risk if the pitch lists mechanisms instead of stating the one principle.

## 8. Track and sponsor stack

- Track: Model Learning. Thin, and the project is literally a system that improves from data.
- veri.studio: two training jobs (distillation SFT, tool-outcome predictor) and optionally GRPO on the policy against the tool model as reward. Their own example idea.
- Intuit: typed System One predictors with a realistic evaluation on their data. Their rubric nearly verbatim.
- Fetch.ai: register the runtime as an Agentverse agent, run the workflow in ASI:One. The reliability and recovery bonus is earned by priority 2.
- Sentry: tracing and agent monitoring are the trace source. Two products beyond errors.
- Tiger Data: trace store as time series.
- Cognition: build with Devin, say so in the README.
- Bolt-ons: Paper for the dashboard, Photon for a text interface.

## 9. On "everyone wants to train models"

If the Model Learning track is fuller than assumed because Veri is handing out GPUs, most of those teams will fine-tune a domain chatbot. Dreaming trains models of the agent system itself, which is a different pitch in the same room. If the point is to lean into the trend, this is the way: not one model, a runtime that manufactures them.

## 10. Team split for 24 hours

- Harness and trace store, Sentry and Tiger Data wiring.
- Training pipeline on Veri: distillation, shadow gate, promotion.
- Tool-outcome predictor and the planner fallback, plus the kill-a-tool moment.
- Evaluation, dashboard, demo video, Devpost page, sponsor registrations (Agentverse, Devin, Intuit dataset).

Reserve the last four hours for the demo. Every winner report from 2026 says the video took longer than expected.
