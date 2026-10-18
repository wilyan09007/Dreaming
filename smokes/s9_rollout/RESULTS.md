# S9: Full rollout over a 4-step workflow (Mission 7)

**The environment is synthetic.** The workflow runs on the same generator as S4 (`../s4_tool_predictor/env.py`): merchants, descriptors, invoices and the anomaly rule are all invented by me. The results therefore only test the plumbing (chain predictors from the raw transaction, compare with the real execution, route on the imagined outcome, certify each step or the whole trajectory, skip when the trajectory is confident, replay the side effects of skipping), not how predictable a real bookkeeping workflow is. Every accuracy, precision and saving below is a property of this generator, not an estimate for real data.

## Revision

A reviewer found that the first `RESULTS.md` (written before `run.py` was revised) no longer matched `run.py` and `log.txt`. Every table below is now copied from the current `log.txt` (the unchanged-code re-run reproduces the previous log line for line; the only diffs are the relabelled routing row and the wall seconds). The first version's log is kept as `log_v1.txt`.

**1. Certifier.** The first version used `common.certify`, which is path dependent (documented in the Revision of `../s4_tool_predictor/RESULTS.md`: its binary search starts at k = n/2 and can step over the feasible region, because the Clopper-Pearson bound is U-shaped in coverage). `run.py` now uses S4's `certify_grid` (20 fixed coverage levels, delta/20 each) for every certificate; `common.certify` is kept only in the "original" comparison tables. `common.py` itself was not modified.

**2. How the main-world numbers moved** (per-step budget b, delta 0.05 per step, sequential gating; test = 1200 transactions):

| b | first version (shipped `certify`) | revised (grid certifier) |
|---|---|---|
| 2% | nothing certifiable (cut 3 and 4 = inf), 0 confident | 34.2% confident (410), trajectory error 5/410 = 1.22% |
| 3% | 43.6% confident, 10/523 = 1.91% | 43.6% confident, 8/523 = 1.53% |
| 5% | 57.3% confident, 31/688 = 4.51% | 65.5% confident, 35/786 = 4.45% |
| 10% | 86.5% confident, 131/1038 = 12.62% | 84.1% confident, 103/1009 = 10.21% |

The first version blamed the 2% failure on invoice-step label noise. That was wrong: at 2% the shipped certifier returned `inf` for steps 3 and 4 because of the search path, while the grid certifier finds a 34.2% confident set with 1.22% error. Under the grid certifier 3 of the 6 worlds are certifiable at 2% and all 6 at 3% and 5% (the shipped certifier: 0, 3 and 5 of 6; see the six-world tables below).

**3. Calls avoided were overstated by 1.7-1.9x.** The first version counted `match_invoice` as skipped for every confident trajectory (all four calls). The rollout cannot supply an invoice id, so `match_invoice` is skippable only when the imagined output is NONE, and 93% of confident trajectories at 3% imagine `found`. Tool calls avoided at 3% are 23.3% (not 43.6%) and at 5% 38.1% (not 65.5%). Teacher calls (categorize, flag_anomaly) avoided still equal the confident share; latency avoided is 41.2% vs 43.0% at 3%.

**4. Conclusions rewritten.** The first version said "per-step budgets do not give a trajectory budget" from 7.5-9.2% trajectory error in four worlds at b = 5% and "nothing certifiable at 3% in three of five worlds". Both came from the shipped certifier. With the grid certifier all five other worlds have a confident set at 3% and 5%, and the realized trajectory error is within the per-step budget in 5 of 6 worlds (details below). The direct trajectory certificate (new, not in the first version) held its budget in every non-empty world and budget cell.

**5. Smaller fixes.** The routing row "the true rule applied to imagined upstream outputs" is relabelled "approximate rule": it uses the vendor mean estimated from training traces (`W['vmean']`), not the env's hidden vendor mean. Numbers unchanged (precision 0.809, recall 0.862). The time line below is corrected (the first version said "roughly 2 minutes"; the run takes 51-65 s). The revised `run.py` also adds the state-divergence replay of skipping `match_invoice`, counts all `vendor_lookup` retry attempts in the latency accounting, and evaluates the direct trajectory certificate. `run.py` (230 lines) is still denser than the "ponytail minimal" brief asks for; I did not trim it.

## Setup

- **Workflow** (all four steps executed for real on every transaction to produce the ground-truth trajectory):
  1. `categorize(descriptor) -> category` (12 classes). Teacher = the env's hidden merchant category (stands in for a frontier model; 900 ms nominal). Student = char 2-4gram TF-IDF + LR (or NN, kept by certified coverage on the calibration slice at a 2% budget).
  2. `vendor_lookup(descriptor) -> vendor_id` (real tool, retried on its 2% timeouts; every attempt counts toward latency). Predictor = the S4 vendor predictor (`fit_text` / `pick_text` imported from `../s4_tool_predictor/run.py`, retrained on this world's traces).
  3. `match_invoice(vendor_id, amount) -> invoice id or NONE`. Hidden invoice table: whether a transaction has an invoice depends on its vendor (about 99% for "invoiced" vendors, 1% for others; for invoiced vendors only 15% when the amount is under $25); the invoice amount is within 1.9% of the transaction amount. The tool returns the closest invoice of that vendor within 2% of the amount and consumes it. Distractor invoices (same vendor, amount at least 10% away) also exist; a later transaction can coincidentally land within 2% of a stale distractor, which is label noise. Realized "has a matching invoice" rate: 74.8% in train, 71.4% in test (target was 70%). Predictor = HGB on [vendor id, trace-derived per-vendor invoice rate (5-fold out-of-fold on training rows), category, amount]; the per-vendor rate goes beyond the spec's [vendor id, amount, category]. **The invoice id is arbitrary, so the predicted output is "found / NONE" and a step-3 match means the imagined and real outputs agree on found versus NONE.**
  4. `flag_anomaly(vendor, amount, invoice) -> yes/no`. Rule: no invoice OR amount > 3x the vendor's mean amount (hidden master data; 4% of transactions are amount spikes of 3.5-8x). Predictor = HGB on [invoice found, amount, amount / vendor mean estimated from training traces (leave-one-out on train rows), category]. Nominal latency 600 ms (teacher-like).
- **Rollout**: from the raw transaction (descriptor, amount) alone, step 1 predicts the category, step 2 the vendor, step 3 consumes the predicted vendor and category, step 4 consumes the predicted invoice outcome, vendor and category. The imagined trajectory is compared with the real one. "Steps ahead matched" = length of the matching prefix (0-4). The tabular predictors are trained on real upstream outputs (teacher-forced) and run on predicted ones at rollout time; I did not train them on predicted upstream outputs.
- **Data**: env seed 9, 6000 transactions, chronological split train 3000 / calibration 1800 / test 1200. Train uses only seen merchants; calibration and test are 80% seen / 20% merchants never in training (disjoint unseen sets for calibration and test). In test, 19.5% of transactions have a vendor absent from training ("novel").
- **Certification**: per-step cutoffs with `certify_grid` (delta 0.05 per step) on the calibration slice, using the chained rollout outputs (step i imagined vs step i real; confidence = top-2 gap). Steps are gated sequentially: step i is certified on the calibration rows where steps 1..i-1 were already above their cutoffs. A trajectory is "confident" when all four steps are above their cutoffs. There is no joint delta over the four steps. The **direct trajectory certificate** instead certifies one cutoff on min(step confidences) with "whole trajectory matched" as the correctness label, so the budget applies to the trajectory.
- Students kept (grid certifier, 2% budget): categorize = LR (cal acc LR 0.954, NN 0.857); vendor_lookup = LR (cal acc LR 0.792, NN 0.774). LR/LR in all six worlds, and the shipped certifier would have kept the same students in all six, so the grid-certified alternative-student table is empty.
- Seeds: main world 9; robustness worlds 21-25; HGB `random_state=0`. Run: `python run.py > log.txt 2> err.txt`.

## Results (main world, seed 9, 1200 test transactions)

### Steps ahead matched

| test slice | n | k=0 | k=1 | k=2 | k=3 | k=4 (whole trajectory) | mean k |
|---|---|---|---|---|---|---|---|
| all test | 1200 | 0.057 | 0.142 | 0.040 | 0.005 | 0.757 | 3.263 |
| known vendors | 966 | 0.002 | 0.002 | 0.050 | 0.006 | 0.940 | 3.880 |
| novel vendors | 234 | 0.282 | 0.718 | 0 | 0 | 0 | 0.718 |

Per-step accuracy, chained (consumes predicted upstream) versus teacher-forced (consumes real upstream; steps 1-2 have no upstream):

| step | chained acc | teacher-forced acc | chained, known vendors | chained, novel vendors |
|---|---|---|---|---|
| categorize | 0.943 | 0.943 | 0.998 | 0.718 |
| vendor_lookup | 0.803 | 0.803 | 0.998 | 0 |
| match_invoice | 0.896 | 0.875 | 0.950 | 0.671 |
| flag_anomaly | 0.898 | 0.991 | 0.943 | 0.709 |

Majority-class baselines: invoice found 0.714, anomaly 0.686. Teacher-forced is an upper bound only for step 4; for step 3 it can be lower than chained because real vendors absent from training reach the HGB as arbitrary vendor numbers, whereas a predicted vendor is always a training vendor.

### Early routing: send to review immediately when the imagined anomaly is "yes"

Real anomaly=yes base rate in test: 0.314 (377 of 1200).

| policy | share routed | precision | recall | true anomalies routed | false routes |
|---|---|---|---|---|---|
| send everything to review | 1.000 | 0.314 | 1.000 | 377 | 823 |
| **imagined anomaly = yes (chained rollout)** | 0.337 | **0.814** | **0.873** | 329 | 75 |
| imagined P(anomaly) > 0.8 | 0.331 | 0.819 | 0.862 | 325 | 72 |
| imagined P(anomaly) > 0.95 | 0.323 | 0.822 | 0.844 | 318 | 69 |
| step-4 predictor on the REAL upstream outputs (reference) | 0.312 | 0.989 | 0.981 | 370 | 4 |
| approximate rule (no invoice OR amount > 3x train-estimated vendor mean) on imagined upstream outputs | 0.335 | 0.809 | 0.862 | 325 | 77 |

| slice | n | real anomaly rate | precision (chained) | recall (chained) |
|---|---|---|---|---|
| known vendors | 966 | 0.256 | 0.843 | 0.956 |
| novel vendors | 234 | 0.556 | 0.750 | 0.715 |

(The 55.6% anomaly rate among novel vendors is most likely an artifact of which 30 vendors happened to be in the unseen test set; invoice behaviour is a per-vendor draw in the env. I did not investigate it further and it is not a general property.)

### Confident trajectories, grid certifier

Every step above its certified cutoff; per-step budget b; steps gated sequentially on the calibration slice.

| b | cut 1 | cut 2 | cut 3 | cut 4 | share of test with whole trajectory confident | n | trajectory error (any step wrong) | mean confident prefix (steps skippable) |
|---|---|---|---|---|---|---|---|---|
| 2% | 0.726 | 0.695 | 0.971 | 0.998 | 0.342 | 410 | 5/410 = 1.22% | 2.447 |
| 3% | 0.454 | 0.682 | 0.933 | 0.998 | 0.436 | 523 | 8/523 = 1.53% | 2.703 |
| 5% | 0.185 | 0.383 | 0.891 | 0.971 | 0.655 | 786 | 35/786 = 4.45% | 3.115 |
| 10% | 0.001 | 0.233 | 0.064 | 0.024 | 0.841 | 1009 | 103/1009 = 10.21% | 3.525 |

Same, with the shipped `common.certify` (`inf` = it found nothing certifiable; path dependent, kept for comparison only):

| b | cut 1 | cut 2 | cut 3 | cut 4 | share confident | n | trajectory error |
|---|---|---|---|---|---|---|---|
| 2% | 0.454 | 0.513 | inf | inf | 0 | 0 | 0/0 |
| 3% | 0.231 | 0.490 | 0.942 | 0.998 | 0.436 | 523 | 10/523 = 1.91% |
| 5% | 0.028 | 0.376 | 0.863 | 0.997 | 0.573 | 688 | 31/688 = 4.51% |
| 10% | 0.001 | 0.166 | 0.064 | 0.024 | 0.865 | 1038 | 131/1038 = 12.62% |

Error structure among the confident trajectories (grid certifier):

| b | n confident | err step 1 | err step 2 (vendor) | err step 3 (invoice) | err step 4 | trajectory error | trajectory error / b | novel-vendor share among confident | mean k among confident |
|---|---|---|---|---|---|---|---|---|---|
| 2% | 410 | 0 | 0.0049 | 0.0073 | 0.0073 | 0.0122 | 0.61 | 0.0049 | 3.971 |
| 3% | 523 | 0 | 0.0038 | 0.0115 | 0.0115 | 0.0153 | 0.51 | 0.0038 | 3.966 |
| 5% | 786 | 0.0013 | 0.0204 | 0.0229 | 0.0229 | 0.0445 | 0.89 | 0.0204 | 3.888 |
| 10% | 1009 | 0.0059 | 0.0496 | 0.0525 | 0.0595 | 0.1021 | 1.02 | 0.0476 | 3.745 |

At 2-5% the vendor errors and invoice errors are on different transactions (P(vendor and invoice both wrong) = 0), so the trajectory error is the sum of the two (vendor errors are novel-vendor confident mistakes; the step-4 error just inherits the invoice error). Step 4 wrong while step 3 right: 0 up to 5%, 0.7% at 10%.

### Calls avoided when the whole imagined trajectory is confident

categorize and flag_anomaly (teacher calls) and vendor_lookup have fully specified outputs (a category, a vendor id, yes/no), so all three are avoided whenever the trajectory is confident. `match_invoice` returns an invoice id, which the rollout cannot supply: it can be skipped only when the imagined output is NONE. Each test transaction has 2 teacher calls and 2 tool calls. "Old accounting" is the first version's (match_invoice skipped for every confident trajectory).

| b | confident share = teacher calls avoided = vendor_lookup avoided | confident trajectories whose imagined invoice is `found` | match_invoice avoided (share of test) | **tool calls avoided** (fully specified) | tool calls avoided, old accounting | **step latency avoided** (fully specified) | step latency avoided, old accounting |
|---|---|---|---|---|---|---|---|
| 2% | 0.342 | 1.000 | 0 | **0.171** | 0.342 | **0.320** | 0.335 |
| 3% | 0.436 | 0.933 | 0.029 | **0.233** | 0.436 | **0.412** | 0.430 |
| 5% | 0.655 | 0.837 | 0.107 | **0.381** | 0.655 | **0.629** | 0.653 |
| 10% | 0.841 | 0.740 | 0.218 | **0.530** | 0.841 | **0.809** | 0.837 |

At b = 3% that is about 558 of 2400 tool calls avoided (not the 1046 the first version claimed), 1046 of 2400 teacher calls, and 41.2% of step latency.

### State divergence from skipping match_invoice (replayed on the invoice table; an executed match consumes its invoice)

Policy A = skip `match_invoice` only when confident and imagined NONE; the executed matches in confident trajectories use the predicted (not the real) vendor id. Policy B = the old accounting (skip it for every confident trajectory; executed matches use the real vendor). Executing everything reproduces the logged outputs (asserted in `run.py`).

| b | policy | match_invoice calls still executed | executed results that differ from the all-real run | of which found/NONE flips | confident trajectories with an error |
|---|---|---|---|---|---|
| 2% | A | 1200 | 2/1200 = 0.17% | 2/1200 = 0.17% | 5 |
| 2% | B | 790 | 0/790 | 0/790 | 5 |
| 3% | A | 1165 | 2/1165 = 0.17% | 2/1165 = 0.17% | 8 |
| 3% | B | 677 | 0/677 | 0/677 | 8 |
| 5% | A | 1072 | 16/1072 = 1.49% | 15/1072 = 1.40% | 35 |
| 5% | B | 414 | 0/414 | 0/414 | 35 |
| 10% | A | 938 | 33/938 = 3.52% | 31/938 = 3.30% | 103 |
| 10% | B | 191 | 0/191 | 0/191 | 103 |

Skipping a call leaves the invoice table in a different state, and an executed match with a wrongly predicted vendor can consume the wrong invoice. Policy B shows no divergence only because it never executes the match for confident trajectories, which leaves the downstream real system without an invoice id.

### Direct trajectory certificate vs per-step budgets of the same size (main world)

Direct: confidence = min of the four step gaps, correct = whole trajectory matched, budget applies to the trajectory.

| budget | direct: cutoff | direct: cal coverage | direct: share of test confident | direct: trajectory error | per-step b: share confident | per-step b: trajectory error |
|---|---|---|---|---|---|---|
| 2% | inf | 0 | 0 | 0/0 | 0.342 | 5/410 = 1.22% |
| 3% | 0.942 | 0.350 | 0.384 | 8/461 = 1.74% | 0.436 | 8/523 = 1.53% |
| 5% | 0.901 | 0.501 | 0.526 | 16/631 = 2.54% | 0.655 | 35/786 = 4.45% |

## Results (six worlds: seed 9 plus robustness seeds 21-25)

### Rollout quality

| env seed | novel-vendor share | mean k | whole trajectory matched | anomaly base rate | routing precision | routing recall | students (cat/vendor) |
|---|---|---|---|---|---|---|---|
| 9 | 0.195 | 3.263 | 0.757 | 0.314 | 0.814 | 0.873 | LR/LR |
| 21 | 0.209 | 3.173 | 0.721 | 0.302 | 0.743 | 0.856 | LR/LR |
| 22 | 0.193 | 3.196 | 0.748 | 0.260 | 0.666 | 0.933 | LR/LR |
| 23 | 0.204 | 3.212 | 0.730 | 0.292 | 0.719 | 0.877 | LR/LR |
| 24 | 0.216 | 3.217 | 0.748 | 0.324 | 0.773 | 0.812 | LR/LR |
| 25 | 0.201 | 3.237 | 0.749 | 0.274 | 0.691 | 0.878 | LR/LR |
| mean | 0.203 | 3.216 | 0.742 | 0.294 | 0.734 | 0.872 | |

### Per-step gating (grid certifier): share of test with a confident trajectory, trajectory error

| env seed | b=2%: share | b=2%: error | b=3%: share | b=3%: error | b=5%: share | b=5%: error |
|---|---|---|---|---|---|---|
| 9 | 0.342 | 5/410 = 1.22% | 0.436 | 8/523 = 1.53% | 0.655 | 35/786 = 4.45% |
| 21 | 0 | nothing certifiable | 0.391 | 9/469 = 1.92% | 0.507 | 26/608 = 4.28% |
| 22 | 0 | nothing certifiable | 0.488 | 3/586 = 0.51% | 0.598 | 33/718 = 4.60% |
| 23 | 0 | nothing certifiable | 0.284 | **11/341 = 3.23%** | 0.600 | **57/720 = 7.92%** |
| 24 | 0.434 | 6/521 = 1.15% | 0.539 | 11/647 = 1.70% | 0.693 | 26/831 = 3.13% |
| 25 | 0.391 | 3/469 = 0.64% | 0.447 | 7/536 = 1.31% | 0.593 | 23/711 = 3.23% |
| pooled (errors / confident) | 14/1400 = 1.00% (3 worlds) | | 49/3102 = 1.58% | | 200/4374 = 4.57% | |

Bold = above the per-step budget. Seed 23 is the only world that breaches. In that world the vendor step alone has 2.64% error among the confident trajectories at b = 3% and 5.28% at b = 5% (invoice step 0.9% and 3.2%), so the vendor step's own certificate is what failed (a cluster of novel-vendor confident mistakes, the S4 failure shape), and the invoice errors add on top.

Same, with the shipped `common.certify` (the first version's numbers; nothing certifiable in several worlds, and 7.5-9.3% trajectory error in worlds 22-25 at b = 5%):

| env seed | b=2% | b=3%: share | b=3%: error | b=5%: share | b=5%: error |
|---|---|---|---|---|---|
| 9 | nothing certifiable | 0.436 | 10/523 = 1.91% | 0.573 | 31/688 = 4.51% |
| 21 | nothing certifiable | 0 | nothing certifiable | 0 | nothing certifiable |
| 22 | nothing certifiable | 0 | nothing certifiable | 0.667 | 60/800 = 7.50% |
| 23 | nothing certifiable | 0 | nothing certifiable | 0.433 | 46/520 = 8.85% |
| 24 | nothing certifiable | 0.479 | 33/575 = 5.74% | 0.577 | 64/692 = 9.25% |
| 25 | nothing certifiable | 0.480 | 11/576 = 1.91% | 0.598 | 60/717 = 8.37% |

### Direct trajectory certificate vs per-step budgets of the same size

| env seed | budget | direct: share confident | direct: trajectory error | per-step: share confident | per-step: trajectory error |
|---|---|---|---|---|---|
| 9 | 2% | 0 | 0/0 | 0.342 | 5/410 = 1.22% |
| 21 | 2% | 0 | 0/0 | 0 | 0/0 |
| 22 | 2% | 0 | 0/0 | 0 | 0/0 |
| 23 | 2% | 0 | 0/0 | 0 | 0/0 |
| 24 | 2% | 0.258 | 3/310 = 0.97% | 0.434 | 6/521 = 1.15% |
| 25 | 2% | 0.228 | 1/274 = 0.36% | 0.391 | 3/469 = 0.64% |
| 9 | 3% | 0.384 | 8/461 = 1.74% | 0.436 | 8/523 = 1.53% |
| 21 | 3% | 0.447 | 12/536 = 2.24% | 0.391 | 9/469 = 1.92% |
| 22 | 3% | 0.389 | 2/467 = 0.43% | 0.488 | 3/586 = 0.51% |
| 23 | 3% | 0 | 0/0 | 0.284 | **11/341 = 3.23%** |
| 24 | 3% | 0.368 | 6/442 = 1.36% | 0.539 | 11/647 = 1.70% |
| 25 | 3% | 0.288 | 2/345 = 0.58% | 0.447 | 7/536 = 1.31% |
| 9 | 5% | 0.526 | 16/631 = 2.54% | 0.655 | 35/786 = 4.45% |
| 21 | 5% | 0.603 | 29/724 = 4.01% | 0.507 | 26/608 = 4.28% |
| 22 | 5% | 0.586 | 29/703 = 4.13% | 0.598 | 33/718 = 4.60% |
| 23 | 5% | 0.204 | 4/245 = 1.63% | 0.600 | **57/720 = 7.92%** |
| 24 | 5% | 0.697 | 28/836 = 3.35% | 0.693 | 26/831 = 3.13% |
| 25 | 5% | 0.488 | 12/585 = 2.05% | 0.593 | 23/711 = 3.23% |
| pooled | 2% / 3% / 5% | | 4/584 = 0.68% / 30/2251 = 1.33% / 118/3724 = 3.17% | | 14/1400 = 1.00% / 49/3102 = 1.58% / 200/4374 = 4.57% |

### Calls avoided and state divergence by world (grid certifier, per-step gating; fully specified outputs only)

| env seed | b | confident share (teacher calls and vendor_lookup avoided) | confident trajectories with imagined `found` | tool calls avoided | tool calls avoided, old accounting | latency avoided | latency avoided, old accounting | policy A: executed matches changed |
|---|---|---|---|---|---|---|---|---|
| 9 | 3% | 0.436 | 0.933 | 0.233 | 0.436 | 0.412 | 0.430 | 2/1165 = 0.17% |
| 21 | 3% | 0.391 | 1.000 | 0.195 | 0.391 | 0.380 | 0.398 | 4/1200 = 0.33% |
| 22 | 3% | 0.488 | 0.993 | 0.246 | 0.488 | 0.463 | 0.484 | 0/1196 = 0.00% |
| 23 | 3% | 0.284 | 0.997 | 0.143 | 0.284 | 0.269 | 0.281 | 8/1199 = 0.67% |
| 24 | 3% | 0.539 | 0.921 | 0.291 | 0.539 | 0.517 | 0.539 | 0/1149 = 0.00% |
| 25 | 3% | 0.447 | 0.935 | 0.238 | 0.447 | 0.415 | 0.433 | 0/1165 = 0.00% |
| mean | 3% | 0.431 | | 0.224 | 0.431 | 0.409 | | |
| 9 | 5% | 0.655 | 0.837 | 0.381 | 0.655 | 0.629 | 0.653 | 16/1072 = 1.49% |
| 21 | 5% | 0.507 | 0.824 | 0.298 | 0.507 | 0.490 | 0.509 | 4/1093 = 0.37% |
| 22 | 5% | 0.598 | 0.798 | 0.360 | 0.598 | 0.576 | 0.597 | 0/1055 = 0.00% |
| 23 | 5% | 0.600 | 0.853 | 0.344 | 0.600 | 0.582 | 0.605 | 33/1094 = 3.02% |
| 24 | 5% | 0.693 | 0.758 | 0.430 | 0.693 | 0.670 | 0.694 | 0/999 = 0.00% |
| 25 | 5% | 0.593 | 0.814 | 0.351 | 0.593 | 0.562 | 0.582 | 1/1068 = 0.09% |
| mean | 5% | 0.608 | | 0.361 | 0.608 | 0.585 | | |

## Interpretation for the mission

- **The rollout plumbing works and the imagined trajectory is right a lot of the time, in this env.** The whole 4-step trajectory was reproduced for 75.7% of test transactions in the main world (72.1-75.0% in the other five; mean 74.2%) with a mean matching prefix of 3.26 steps. For known vendors it is 94%; for novel vendors it is 0% by construction (a closed-set vendor predictor cannot name an unseen vendor), while categorize is still right 72% of the time because the category words in the name generalise. Novel vendors therefore have to be caught by the vendor step's confidence gate and not imagined past.
- **Early routing is a real lift, but the imagined route is much worse than the same predictor on real inputs.** Routing on imagined anomaly=yes gives precision 0.81 / recall 0.87 in the main world (0.67-0.81 and 0.81-0.93 across the six worlds; base rate 0.26-0.32), with 75 false routes instead of 823. The step-4 predictor on real upstream outputs gets 0.99 / 0.98, so the loss is upstream (the invoice-found step at 0.90 accuracy and unseen vendors). The HGB step-4 model adds nothing over applying the approximate rule to the imagined upstream outputs (0.809 / 0.862 vs 0.814 / 0.873); thresholds on P(anomaly) trade a little recall for a little precision.
- **Certified skipping of whole trajectories is feasible, and in this env more than a third of transactions can be skipped at 2-3% error.** Pooled over six worlds the per-step grid-certified gating gave: b = 3%, mean 43.1% confident at 1.58% trajectory error (49/3102); b = 5%, mean 60.8% confident at 4.57% (200/4374); b = 2%, certifiable in 3 of 6 worlds (34.2%, 43.4%, 39.1% confident, 0.64-1.22% error). Main world: 34.2% / 1.22% at 2%, 43.6% / 1.53% at 3%, 65.5% / 4.45% at 5%, 84.1% / 10.2% at 10% (at 10% the trajectory error reaches the budget: 1.02x).
- **The value of full rollout is smaller than the confident share suggests, because the invoice id cannot be imagined.** Teacher calls avoided equal the confident share (43.6% at 3%, 65.5% at 5%), but tool calls avoided are 23.3% and 38.1% (six-world means 22.4% and 36.1%), since `match_invoice` can be skipped only when the imagined output is NONE (2.9% of test transactions at 3%, 10.7% at 5%). Step latency avoided is 41.2% at 3% and 62.9% at 5% (six-world means 40.9% and 58.5%), close to the old accounting because the invoice call is only ~80 ms. Executing `match_invoice` with a predicted vendor changes 0.17% of the executed results at 2-3% and 1.5% at 5% (3.0% in seed 23), because a wrong vendor can consume the wrong invoice.
- **Per-step budgets did bound the trajectory error in 5 of 6 worlds, with one breach; a direct trajectory certificate had no breach.** With the grid certifier the realized trajectory error was within b in every world except seed 23 (3.23% at b = 3%, 7.92% at b = 5%): 1 of 6 worlds, the same shape as S4's one seed-11 breach, driven by a novel-vendor cluster at the vendor step. Seeds 21-25 at b = 3%: 1.92, 0.51, 3.23, 1.70, 1.31%; at b = 5%: 4.28, 4.60, 7.92, 3.13, 3.23% (seed 9: 1.53 / 4.45%). Per-step budgets carry no joint guarantee (errors from steps 2 and 3 add: trajectory error/b is 0.5-0.9 at 2-5% in the main world and 1.0 at 10%, 1.6 in seed 23 at 5%). The direct certificate (min step gap, trajectory-level label) held in all 13 non-empty world/budget cells (max 4.13% at b = 5%, 2.24% at 3%, 0.97% at 2%), and it declined to certify seed 23 at 3% while certifying only 20.4% there at 5% (1.63% error). The price is coverage in the main world (52.6% vs 65.5% at 5%, 38.4% vs 43.6% at 3%; the direct certificate covers more than the per-step one in seed 21 at 3% and 5%, and in seed 24 at 5% by 0.4 points). This supports certifying the trajectory outcome directly. With 6 worlds and 1 breach this does not show per-step budgets are unsafe in general; it shows they are not guaranteed.
- **What breaks: novel vendors and the unmodelled side effects.** In the main world at b <= 5% every vendor-step error among confident trajectories is a novel-vendor mistake (the novel-vendor share among confident equals the step-2 error); the cutoffs are fitted on a calibration slice with the same ~20% novelty mix as the test slice (by construction), which is the case that makes these certificates hold. The sequential per-step certificates use delta 0.05 each with no correction across the four steps, the direct certificate uses delta 0.05 for the trajectory. Audit sampling (S4 step 5) is needed to keep any of this current and is not simulated.

## Caveats

- Synthetic: the invoice process, the anomaly rule (deterministic given the invoice and the vendor mean, hence the 0.99 teacher-forced accuracy), the vendor-level invoice types and the noise floor are all my design choices. In real bookkeeping, invoice matching is far noisier, the anomaly judgement is a frontier-model call that is not a clean rule, and teacher outputs for categorize are not a hidden table lookup. Do not read the 75% / 94% / 0.81 / 40% figures as predictions about real workflows.
- Leakage and optimism: tabular steps use trace-derived vendor statistics fit on the training slice only (out-of-fold for the invoice rate; leave-one-out for the vendor mean); the LR-vs-NN choice and the certification share the same calibration slice; test is only used for evaluation. The step-3 and step-4 models are trained on real upstream outputs and run on imagined ones. The step-4 vendor mean is the train-estimated one (so the "approximate rule" row is not the env's true rule). The step-3 predictor uses a trace-derived per-vendor invoice rate beyond the spec's [vendor id, amount, category].
- The grid certifier's cutoffs are data-dependent quantiles of the calibration scores, so its union bound over 20 levels is not strictly valid either; treat the "delta" as nominal. The per-step gating has no joint delta over four steps.
- Step 3 compares found vs NONE, not the invoice id; a real step would have to predict or look up the id too. That is why tool calls avoided are roughly half the confident share.
- Teacher/tool "calls avoided" are counts, and latency uses the env's nominal latencies (teacher 900 ms, anomaly 600 ms, vendor lookup lognormal median 120 ms with a 3% tail plus retried timeouts, invoice match median 80 ms); no dollars. Audit sampling to keep certificates current is not simulated.
- Single main world plus 5 robustness worlds; per-world test sets have 1200 transactions and a few dozen confident errors at most, so the trajectory error rates and "1 of 6 breaches" have wide sampling error. Calibration and test novelty are matched (about 20% novel vendors in both); a different novelty rate was not run for S9 (see the novelty-shift table in S4's RESULTS).
- Real deployment changes: the vendor step would use live tool outputs when the predictor is uncertain (so downstream steps would see real, not imagined, inputs), step students would be trained on frontier-model outputs rather than a hidden table, and certificates would be refreshed continuously.
- Process notes: `common.certify` raises on empty input (`np.log2(0)`), so S9 guards the empty case (in `gate`, via `mask.sum() > 1`, and `certify_grid` returns `inf` on empty input); HistGradientBoosting was 100x slower than expected under CPU contention until `OMP_NUM_THREADS=1` was set (set at the top of `run.py`).

## Time and cost

- Wall time of the final `python run.py` (six worlds): 65 s on a shared CPU (earlier runs of the same code: 51 s, a reviewer re-run 60 s; most of it is the 6 x vendor LR/NN student fits).
- Cost: $0 (no API calls).
- Files: `run.py`, `log.txt` (current run), `log_v1.txt` (first version's log), `err.txt` (progress timing). Depends on `../s4_tool_predictor/env.py` and `../s4_tool_predictor/run.py`; `common.py` is unchanged.
