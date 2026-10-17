# Fusion rationale and the strongest critique

Source: [`00-unified-runtime-brief.md`](00-unified-runtime-brief.md), sections 2 and 4.

## 1. Why one mechanism

The two original ideas (distill-as-you-go; imagination-loop agents) share everything that matters:

| Shared piece | Distill half | Environment-model half |
|---|---|---|
| Data | Traces of model calls | Traces of tool calls |
| Gate | Shadow until agreement with the real call crosses a threshold | Same |
| Promotion | Swap in the cheap predictor, keep the real call as fallback | Same |
| RL analogue | Policy distillation | Dynamics model |

So the runtime is **one loop** (observe, train, shadow, promote, audit, demote) applied to **three call kinds** (model, tool, reasoning budget), with a fourth (full rollouts) as a stretch.

The brief's condition: it must read as one mechanism, not two projects stapled together. The pitch therefore states the principle ("learn a model of every call; stop making the ones you can predict") and never lists mechanisms.

## 2. The critique we have to survive

Both adversarial reviewers in the brief's sweep hit the environment-model idea with the same point:

> A world model trained on one day of traces is a lookup table.

The merge only wins if it fixes this rather than inheriting it. Our answers:

| Concern | Response in the design |
|---|---|
| "It memorizes, it doesn't generalize" | Report agreement on **held-out** rows with the exact-match tier **disabled** (see `04-evaluation-methodology.md`). A lookup table scores near zero there; a predictor that generalizes does not. |
| "A world model of everything is too ambitious" | Model only **narrow, typed** things: an enum, a bool, a failure probability, a latency. Priority 4 (full rollouts) is explicitly gated on 1 to 3 landing. |
| "Sparse traces, brittle model" | Start with small models whose confidence is calibrated; low confidence falls through to the real call, so brittleness costs savings, not correctness. |
| "How do you know it's safe to stop calling?" | Promotion requires a statistical lower bound on agreement (`03-statistics-of-promotion.md`), audits continue after promotion, and drift or disagreement demotes automatically. |
| "Kitchen sink" | Single principle, one demo with two lines and one live moment. Build order doubles as the cut order. |

## 3. Build order is the fallback plan

| If only this ships | You still have |
|---|---|
| Priority 1 (distill narrow steps) | The original number-one idea: cost falls, accuracy holds |
| Priorities 1 and 2 | Cost line **and** failure line, plus the kill-a-tool moment (the demo as written) |
| Priorities 1 to 3 | Adds the reasoning-budget router: "it stopped thinking on easy steps" |
| Priority 4 only | The idea the reviewers rejected. Never build this first |

## 4. Claims we will and will not make

**Will claim (and can show):**
- Cost per run falls as call types are promoted, with end-to-end accuracy within a stated margin of the frontier-only baseline.
- Failure rate falls because a typed tool-outcome model lets the router avoid endpoints it expects to fail.
- On kill, the recovery was decided by a promoted predictor with zero frontier calls.
- Agreement numbers on held-out data with the exact-match tier off.

**Will not claim:**
- General world modeling, or that the agent "understands" its tools.
- Replacement of free-text generation (the draft step stays real and sets the cost floor).
- That the first-ever call to a dead endpoint is avoided, unless health probes are on (see `DESIGN.md` section 8.4).
- Any guarantee beyond the stated statistical bound and error budget.
- Illustrative numbers (for example the refund-agent example in the brief: 9s to 3s, about two thirds cheaper) as measurements. Only measured demo results get presented as results.

## 5. What would falsify the idea

Worth stating up front, so the evaluation can actually say no:

1. On held-out rows with the exact-match tier off, agreement for `categorize` stays below the self-agreement baseline of the frontier model.
2. Coverage at the serving threshold stays under about 20%, so promotion saves little.
3. The tool-outcome model does no better than a plain EWMA health tracker (then the learned prior is not earning its complexity).
4. Reasoning-budget routing changes accuracy by more than the stated margin.
