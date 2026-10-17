# Research

Background, justification, and methodology behind Dreaming. The architecture itself lives in [`../design/DESIGN.md`](../design/DESIGN.md); the build plan in [`../design/PLAN.md`](../design/PLAN.md).

| Doc | What it answers |
|---|---|
| [`00-unified-runtime-brief.md`](00-unified-runtime-brief.md) | The source planning brief (verbatim, 2026-10-16): the fusion of distill-as-you-go and imagination-loop agents into one mechanism. |
| [`01-prior-art.md`](01-prior-art.md) | What this builds on, what is genuinely new, and how we differ from distillation services, semantic caches, and LLM routers. |
| [`02-fusion-rationale-and-critique.md`](02-fusion-rationale-and-critique.md) | Why one mechanism instead of two projects, the "world model is a lookup table" critique, and the claims we will and will not make. |
| [`03-statistics-of-promotion.md`](03-statistics-of-promotion.md) | The math behind shadow gating, calibration, audit sampling, demotion, and per-run error budgets, with worked numbers. |
| [`04-evaluation-methodology.md`](04-evaluation-methodology.md) | Baselines, data splits, the cache-disabled held-out test, ablations, and how results get reported honestly. |
| [`05-sponsors-and-track-fit.md`](05-sponsors-and-track-fit.md) | How the build maps to the track and sponsor stack, and a checklist of facts to verify at the event. |

## Status of citations

Papers and products referenced in `01-prior-art.md` are cited **from memory** and have not been re-checked against sources in this session. Verify titles, authors, and years before they go on a slide or the Devpost page.
