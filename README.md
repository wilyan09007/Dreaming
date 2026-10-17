# Dreaming

A runtime that learns a model of every call your agent makes, then stops making the ones it can predict. The agent gets cheaper and more reliable the more it runs.

Every expensive call (frontier model, chain-of-thought, tool or API) is logged as a structured trace. Each call type trains its own small typed predictor, which runs in shadow beside the real call until agreement clears a threshold and it is promoted. After that the real call runs only on low confidence or on a sampled fraction for monitoring.

**The demo:** run one agent workflow a few hundred times with two falling lines, cost per run and failure rate, then kill a tool mid-run and watch the agent reroute without a frontier call.

## Map

| Folder | Contents |
|---|---|
| [`design/`](design/DESIGN.md) | Architecture ([`DESIGN.md`](design/DESIGN.md)), 24-hour build plan ([`PLAN.md`](design/PLAN.md)), interface contracts ([`CONTRACTS.md`](design/CONTRACTS.md)) |
| [`research/`](research/README.md) | Source brief, prior art, fusion rationale and critique, statistics of promotion, evaluation methodology, sponsor fit |
| [`demo/`](demo/README.md) | Workflow, mock tools and flakiness, run of show, fallbacks |
| [`UI/`](UI/README.md) | Dashboard spec |
| [`video/`](video/README.md) | Video script, shot list, Devpost outline |

## Status

Planning only; no code yet. Numbers in the docs marked "target" or "illustrative" are not measurements.
