# Prior art and positioning

> Citations are from memory and unverified. Check before publishing.

Dreaming is a composition of known patterns (the brief says as much: "every component is a known pattern"). The novelty claim is the **unification and the gating discipline**, not any single component.

## 1. Building blocks

| Pattern | Reference (verify) | What we borrow | Where it appears in Dreaming |
|---|---|---|---|
| Knowledge distillation | Hinton, Vinyals, Dean, *Distilling the Knowledge in a Neural Network* (2015) | Teacher outputs become student labels | Replacement predictors trained on real-call outputs |
| Policy distillation | Rusu et al., *Policy Distillation* (2015) | Compress a policy into a smaller one by imitation | Distilling narrow policy steps (classify, extract, decide) |
| World models / Dreamer | Ha & Schmidhuber, *World Models* (2018); Hafner et al., Dreamer / DreamerV3 | Learn a dynamics model, plan inside it | Tool-outcome model (priority 2); full rollouts (priority 4, stretch). The name Dreaming nods to this lineage |
| Reject option / selective classification | Chow (1970); Geifman & El-Yaniv, *Selective Classification for Deep Neural Networks* (2017) | Answer only above a confidence threshold, trade coverage for risk | `tau_serve` and the risk-coverage curve |
| Conformal prediction / risk control | Vovk et al.; Angelopoulos & Bates tutorial | Distribution-free coverage guarantees from held-out data | Choosing `tau_serve` with a bound on disagreement |
| Binomial confidence bounds | Wilson (1927); Clopper & Pearson (1934) | Lower bound on agreement from `k` of `n` | Promotion test |
| Sequential change detection | Wald SPRT (1945); Page CUSUM (1954) | Detect disagreement spikes with few samples | Demotion on audit results |
| Shadow / canary deployment | SRE practice (e.g. Google SRE book) | Run new path beside old, diff automatically, promote on agreement | The whole lifecycle |
| Circuit breakers, health tracking | Nygard, *Release It!* | Stop calling a failing dependency | Router and health posterior |
| Thompson sampling | Thompson (1933) | Explore recovered endpoints without hammering dead ones | Router exploration |
| Drift metrics | Population stability index; embedding OOD scores | Detect that inputs no longer resemble training data | Drift demotion, OOD gating |
| Cascades and routing between models | Chen, Zaharia, Zou, *FrugalGPT* (2023); Ong et al., *RouteLLM* (2024); Ding et al., *Hybrid LLM* (2024) | Send easy queries to cheap models, hard ones to expensive | Confidence-gated fallback to the real call |
| Semantic caching | GPTCache and similar | Reuse answers for similar queries | Tier 0/1 predictors, with guarantees added |
| LLM-log distillation services | OpenPipe (named in the brief) | Fine-tune a small model on production LLM logs | Tier 3 SFT path |

## 2. How Dreaming differs

| System type | What it does | What it lacks that Dreaming adds |
|---|---|---|
| Distillation services (OpenPipe-like) | Fine-tune a cheaper model on logged LLM calls | Usually offline and human-promoted. LLM calls only, not tools or thinking budget. No continuous audit or automatic demotion |
| Semantic caches | Return a stored answer for a similar prompt | Similarity threshold is a heuristic with no measured agreement. No typed outputs, no effect classes, no demotion |
| LLM routers (RouteLLM, FrugalGPT) | Pick which model answers | Route between models, not "no model at all". No tool-outcome modeling. No per-call-type lifecycle |
| Retry and circuit-breaker libraries | React to failures after they happen | No learned prior over failures. No distilled recovery decision |
| Observability (Sentry, OTel) | Record what happened | Passive. Nothing closes the loop into behavior |
| World-model agents (Dreamer-style) | Plan in a learned simulator | Require large data and are brittle on sparse traces. Dreaming starts with small typed models and only goes further if earlier layers land |

## 3. The one-line differentiator

Distillation alone, caching alone, routing alone, and world models alone each exist. What does not (per the brief, "nobody at these events has shipped the unified version") is **one lifecycle (observe, shadow, promote, audit, demote) applied uniformly to three kinds of call**: model calls, tool calls, and reasoning-budget decisions, with measured agreement as the only license to stop making a call.

## 4. Research gaps to close

1. Evidence on how often LLM-driven structured steps are actually predictable by small models on the QuickBooks-style workflow. The demo depends on this; measure early (see `04-evaluation-methodology.md`).
2. Literature on adaptive reasoning effort, to cite for the thinking-budget router (priority 3). Not yet surveyed.
3. Current state of any competing "learn from traces and skip calls" products at the event.
