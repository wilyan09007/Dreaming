# Evaluation methodology

Goal: produce numbers that survive a skeptical judge. The brief's verification column is the spec: each priority has a named check "at the judging table".

## 1. Systems compared

| Arm | Description |
|---|---|
| **Baseline** | Frontier-only: every call real, no Dreaming. |
| **Dreaming (full)** | Promotion, audits, tool model, router, and (if built) reasoning-budget router. |
| Ablation A | No shadow gate: promote after fixed N traces regardless of agreement. Shows the gate matters. |
| Ablation B | No tool model: EWMA health only. Shows the learned prior earns its keep. |
| Ablation C | No recovery distillation: failures handled by a frontier call each time. |
| Ablation D | Exact-match cache only (tier 0). Shows generalization beyond a lookup table. |

## 2. Data splits

For the QuickBooks-style synthetic data (schema and label availability to be confirmed at the event):

| Split | Use |
|---|---|
| Train stream | Rows flowing through the online loop; become traces. Order is shuffled with a fixed seed. |
| Shadow/calibration | Recent shadow pairs held out from predictor fitting. |
| **Held-out test** | Never seen by any predictor. Used for the headline agreement and accuracy numbers. |

The generator also emits a **long tail** (rows unlike the common patterns) so coverage plateaus below 100% and the floor is visible and honest.

## 3. The check that answers the "lookup table" critique

For `categorize` and `flag_anomaly`: agreement of the promoted predictor with the real call on the held-out set, **with tier 0 (exact-match) and tier 1 (near-duplicate) disabled**, so the predictor must rely on learned features. Report next to the same measurement for the exact-match-only arm. The gap is the evidence of generalization.

## 4. Metrics

**Cost and latency**
- Cost per run (tokens x price table, plus API cost, plus predictor inference, plus audit cost).
- p50 and p95 latency per run and per step.
- Frontier calls per run.

**Quality**
- Per-call-type agreement with the real call (with CI).
- Accuracy against ground truth where labels exist.
- **End-to-end outcome agreement** on sampled full-real runs (the number that matters for error compounding).

**Reliability**
- Run failure rate (unrecovered error, or fails the success check).
- Predicted vs actual failure rate for the tool model, calibration plot, Brier score.
- Frontier calls spent on recovery.
- Wasted calls to endpoints that were already dead.

**Reasoning-budget router (if built)**
- Accuracy with vs without router, and reasoning tokens saved.

## 5. Protocol

1. Run the baseline on the full run schedule to establish cost, latency, failure rate. Same seed, same flakiness schedule.
2. Run Dreaming from a cold registry on the same schedule. Log every promotion event with its run number and bound.
3. At fixed checkpoints (for example runs 50, 100, 200, 300) run the held-out evaluation, cache tiers off.
4. Run ablations on the same schedule and seed.
5. Repeat the main arm across several seeds. Report mean and range, not the best seed.

## 6. Reporting rules

- Plot rolling means with the window size stated; show raw per-run points faintly.
- Mark promotion and demotion events on the x-axis.
- Report the error budget and confidence level used for promotion next to every savings number.
- Show the floor: what fraction of cost remains and why (audits, low-confidence tail, the free-text draft step).
- Anything not measured is labeled illustrative. The brief's refund example numbers are in that category.

## 7. What we need to learn early (first hours)

1. Is the synthetic data labeled well enough to measure accuracy, or only agreement with the teacher?
2. How predictable are `categorize` and `flag_anomaly` for a small model on held-out rows? A quick offline test with logged frontier outputs and a gradient-boosted model over embeddings answers this before building the runtime.
3. Frontier-model self-agreement on these steps, to set the promotion bar.

If (2) comes back poor, the demo's cost line will not fall. Better to know in hour 3 than hour 20.
