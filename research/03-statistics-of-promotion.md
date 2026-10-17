# Statistics of promotion, audit, and demotion

Everything here supports `DESIGN.md` sections 7 and 8. Numbers in the tables were computed in this session (Wilson score interval at z = 1.96; one-sided 95% Clopper–Pearson).

## 1. What "agreement" has to mean

For call type `c`, a shadow pair is `(predictor output, real output)`. Agreement is per field, type-aware (see `DESIGN.md` 7.3). Two baselines matter:

- **Agreement with the teacher** (the real call). This is what the gate uses.
- **Accuracy against ground truth.** Available for the QuickBooks synthetic data if labels exist (confirm). Reported separately, because a predictor can match a teacher that is itself wrong.

For LLM call types, estimate the teacher's **self-agreement** `s` by running a sample twice. Demanding more agreement than `s` is meaningless. Gate on `agreement_LCB >= s - epsilon`.

## 2. The promotion test

Let `k` of `n` shadow pairs on the **covered set** (confidence >= `tau_serve`) agree. Promote when the Wilson lower bound on agreement, `L(k, n)`, clears the bar, plus coverage and slice checks.

### Worked numbers (95% confidence)

| n | disagreements | Wilson lower bound on agreement | Upper bound on disagreement (Clopper–Pearson) |
|---:|---:|---:|---:|
| 50 | 0 | 0.929 | 5.8% |
| 50 | 1 | 0.895 | 9.1% |
| 100 | 0 | 0.963 | 3.0% |
| 100 | 2 | 0.930 | 6.2% |
| 300 | 0 | 0.987 | 1.0% |
| 300 | 3 | 0.971 | 2.6% |
| 500 | 0 | 0.992 | 0.6% |
| 500 | 5 | 0.977 | 2.1% |
| 1000 | 5 | 0.988 | 1.05% |

### Samples needed with zero disagreements

Smallest `n` such that `(1 - delta)^n <= 0.05` ("rule of three" regime):

| Error budget `delta` | Clean shadow samples needed |
|---:|---:|
| 5% | 59 |
| 2% | 149 |
| 1% | 299 |

**Consequence for the demo.** A few hundred runs with one call per type per run cannot certify 1%. The demo should set `delta = 5%` (promotion lands after roughly 60 to 120 runs, earlier if the workflow makes several calls of a type per run), and say so on stage. Production defaults stay at 1% with `N_min` around 300 to 500. The DESIGN doc is updated to match.

## 3. Calibration and choosing `tau_serve`

1. Hold out recent shadow pairs, never used for fitting the predictor.
2. Fit isotonic regression (or temperature scaling when data is thin) from raw confidence to P(agree).
3. Sweep thresholds to get a **risk-coverage curve**: for each `tau`, coverage and the Clopper–Pearson upper bound on disagreement over covered held-out pairs.
4. Choose the lowest `tau` whose bound is within `delta`. That `tau` is `tau_serve`; the coverage there is the savings potential.

Caveat: the bound holds if calibration data is exchangeable with live traffic. Drift breaks that, which is why audits and demotion exist.

## 4. Audits after promotion

- Audit rate `rho` starts high (for example 20%) and decays toward a floor (2 to 5%) while audited agreement holds, and jumps back up on any drift signal.
- Sample a small slice **below** `tau_serve` too. Auditing only confident calls overestimates agreement (selection bias).
- Cost of audits is part of the accounting: `savings = coverage * (cost_real - cost_pred) - rho * coverage * cost_real`. The audit term is the main reason cost never reaches zero.

## 5. Demotion

Run a sequential test on audit outcomes. A simple one-sided SPRT/CUSUM on the Bernoulli disagreement rate, with null rate `delta0` (accepted) against alternative `delta1` (unacceptable, for example 3x), flags a spike within a few dozen audits without waiting for a fixed window.

Additional triggers: input drift (PSI on features or embedding OOD score), upstream change (model id, prompt hash, schema, API version), manual kill switch.

## 6. Error compounding across steps

If `m` steps are promoted with budgets `delta_i`, the chance a run differs from the all-real run at some step is at most `sum(delta_i)` (union bound), pessimistically. With three promoted steps at 5%, that is up to 15%, which is too loose to claim "no quality loss".

Mitigations:
- Many step disagreements do not change the end result. Measure **end-to-end** outcome agreement on sampled full-real runs; that is the number that gets reported.
- Allocate a run-level budget `delta_run` and split it: `delta_i = delta_run / m`.
- Use the downstream-agreement metric (same next action) for steps where small differences are harmless.

## 7. Tool-outcome model

- Online health: Beta posterior per endpoint, updated per call; a failure moves it immediately.
- Learned prior: gradient-boosted classifier over `(input features, context)` for `P(fail)` and a latency quantile model.
- Evaluate with proper scoring rules (log loss, Brier) and calibration plots, **not** accuracy, since failures are the minority class.
- Required comparison: against a plain EWMA tracker. If the learned prior does not beat it on held-out failure prediction, drop it from the claims (see falsifiers in `02-fusion-rationale-and-critique.md`).

## 8. Thinking-budget router

Label construction: on a sample of steps, run both with reasoning and without. The label for "reasoning helps" is that the final typed answer **differs**, or is correct only with reasoning if ground truth exists. Train a predictor of `P(answer changes with reasoning | input)`. Route to no-reasoning when this probability is below a bound chosen the same way as `tau_serve`. Report accuracy with and without the router.

## 9. Open statistical questions

- Slice definition for per-slice agreement when slices are small. Fall back to pooled with a hierarchical shrinkage estimate?
- Whether to use conformal risk control instead of the Clopper–Pearson sweep to get a cleaner finite-sample statement under the selection of `tau`. (Selecting `tau` on the same data used to bound its error needs a held-out split or correction.)
