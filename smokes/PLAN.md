# Smokes plan (2026-10-17)

Goal: for every mission in `missions.md`, answer (a) does data/benchmark exist or can we synthesize it, (b) is the method feasible at high accuracy, (c) how long/costly is training, (d) what breaks the plan.

## Resources (checked 2026-10-17)
| Resource | State |
|---|---|
| Local | Windows, CPU only, no torch GPU. uv venv at `.venv` (torch-cpu, sentence-transformers, sklearn, datasets) |
| OpenRouter | key limit $50, ~$20 account credit remaining |
| RunPod | $19.99 balance, spend limit $80 |
| AWS | us-east-1, S3 works, **GPU quota = 0** (no G/VT on-demand or spot) → AWS is S3-only for us |
| Hard cap | $100 total; target well under. Running tally in `smokes/COSTS.md` |

## Experiment matrix
| ID | Mission | Smoke | Where | Est. cost |
|---|---|---|---|---|
| R | all | Web research sweep: datasets, benchmarks, reusable components, pricing, precedents | sonnet agents | tokens only |
| S1 | 2a student + certificate | Banking77 / CLINC150: frozen-embedding + linear student, learning curve (rows→acc), top-2-gap confidence, certificate cutoff (Clopper-Pearson), coverage@budget | CPU | $0 |
| S2 | 2d weight generator | Hypernetwork: schema+label descriptions (+k examples) → linear classifier weights; vs label-embedding zero-shot and prototype baselines on unseen tasks (CLINC150 domains held out) | CPU | $0 |
| S3 | 4 human review deferral | Reviewer-change = student≠gold; confidence head on (scores,input) → P(change); caught-vs-reviewed curve vs raw gap | CPU | $0 |
| S4 | 3 tool return predictor | Mock bookkeeping tools (vendor_lookup fn-of-input, stateful order status, side-effecting refund): learn mapping, latency, error; speculation hit-rate; outage test vs circuit breaker | CPU | $0 |
| S5 | 1 tier/thinking router | (a) RouterBench (public multi-model answers) match rates between tiers, router classifier; (b) live: 200 inputs through expensive vs cheap config on OpenRouter, thinking on/off | CPU + OpenRouter | ~$2 |
| S6 | 5 context pruner | Typed step with multi-span prompt: ablation labels via cheap model, span scorer, agreement of pruned vs full prompt | OpenRouter | ~$2 |
| S7 | replayability | Determinism: 1000 runs under concurrency, distinct outputs; CPU and GPU (batch-1 fixed shape vs dynamic batching) | CPU + RunPod | ~$0.5 |
| S8 | 2a/2b/2c + cost | GPU LoRA fine-tune of a 0.5B LM as label scorer: time per 1k rows, acc vs rows, shared-workflow (2b) vs per-step (2a) rows-to-certify on a new step | RunPod | ~$1-3 |
| S9 | 7 full rollout | Chain S1 student + S4 tool predictor on mock workflow; steps-ahead match | CPU | $0 |
| S10 | 6 connectors | Minimal OpenAI-compatible proxy; LangGraph + one other framework pointed at it with call-site header; MCP proxy feasibility | CPU | ~$0.2 |
| S11 | ASI:One | Research only + hello-world uAgent if no sign-in needed | research | $0 |

Order: R and S1 first (S1 validates the shared pipeline), then fan out S2-S4, S9; S5-S6 after cheap-model selection from R; S7/S8 on RunPod once the GPU script is validated on CPU with a tiny slice.

Findings roll up into `smokes/FINDINGS.md`. Each smoke keeps `run.py` + `RESULTS.md` in its own folder.
