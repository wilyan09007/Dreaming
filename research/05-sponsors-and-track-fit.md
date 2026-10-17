# Track and sponsor fit

Source: brief section 8. Everything about what a sponsor offers or judges on comes from the brief and is **not independently verified**. The checklist in section 3 lists what to confirm.

## 1. Mapping

| Sponsor / track | Role in the build | What the brief says we earn |
|---|---|---|
| Track: Model Learning | Primary track. The system improves from data by construction. | "Thin" track, good fit. |
| veri.studio | Runs training jobs: distillation SFT and the tool-outcome predictor; optionally GRPO on the policy using the tool model as the reward signal. | Their own example idea. |
| Intuit | Synthetic QuickBooks data as the workflow; typed System One predictors; evaluation on held-out rows against the frontier-only baseline. | "Their rubric nearly verbatim." |
| Fetch.ai | Register the runtime as an Agentverse agent; run the workflow in ASI:One. | Reliability and recovery bonus, earned by the tool model (priority 2). |
| Sentry | Tracing and agent monitoring as the trace source. | Uses two products beyond errors. |
| Tiger Data | Trace store as time series. | Natural fit for per-run cost/failure series. |
| Cognition | Build with Devin and say so in the README. | Attribution. |
| Paper, Photon | Bolt-ons: Paper for the dashboard, Photon for a text interface. | Optional. |

## 2. How it affects the design

- **Training on Veri.** The in-demo online predictors (tiers 0 to 2) are small and train in seconds locally. The Veri jobs are the "heavy dreaming" path: SFT distillation of a small language model for steps that need more than a tree model (tier 3), and the tool-outcome model. Design keeps both paths behind the same Dreamer interface so a Veri job is just another trainer.
- **Trace source.** The trace schema is OpenTelemetry-compatible so Sentry tracing can be the emitter or a mirror. Keep our own store as the source of truth for training; Sentry is for monitoring and the judge-facing view.
- **Store.** The trace store sits behind an interface. DuckDB is the local default; Tiger Data (time series) is an alternate backend and a good home for the dashboard's series. Do not let the demo depend on a remote store being reachable (replay mode must work offline).
- **Agentverse.** Wrap the runtime's entry point as an agent message handler so the same workflow can be triggered from ASI:One. Keep this a thin adapter around the same Gateway.
- **GRPO (optional).** Only if priorities 1 and 2 are done. It uses the tool model as a learned reward environment, which is the first real use of the "environment model" beyond routing.

## 3. Verify at the event (checklist)

- [ ] Exact judging rubric and weights per sponsor prize.
- [ ] What compute and API access veri.studio provides, job size limits, and whether an SFT job fits the 24-hour window.
- [ ] Intuit dataset: schema, size, whether category and anomaly labels exist, license and sharing terms.
- [ ] Fetch.ai: what "recovery from failed tool calls" bonus requires concretely and how it is demonstrated.
- [ ] Sentry: which SDK features are needed for agent tracing and whether a free tier covers the demo volume.
- [ ] Tiger Data: connection details and whether the demo can rely on network access.
- [ ] Submission requirements: Devpost fields, demo video length, repo visibility, required sponsor mentions.
- [ ] Rules on pre-existing work. This repo already contains planning documents only; confirm that is acceptable.

## 4. Pitch line for the crowded track

"Most teams will fine-tune a domain chatbot. Dreaming trains models of the agent system itself: not one model, a runtime that manufactures them."
