# S24 other families: memory students, Bayesian heads, LLM-written rules (partial: PFN not evaluated)

Status: salvaged from a pod that was cut off. 450 of the planned cells for arms 0, 1, 2 and 4 are complete (10 arms x 5 tasks x 3 k x 3 draws, no gaps). The prior-fitted-network arm (3) was still being meta-trained when the pod stopped and was never scored on the tasks; the optional laplace-torch arm (2b') was never reached. No critic pass was done. Every number below is recomputed from `salvaged_results.json` by `report_salvaged.py` (run `python3 -I report_salvaged.py`).

## In plain words

The question was whether a student that is not "frozen embedding plus logistic regression (LR)" can reach a certified 5% error budget with fewer labelled rows. Three families were tried on the same five new-step tasks as S17. Memory students (look up the nearest stored rows) did not help on their own: a plain nearest-neighbour vote was worse than LR, and a "Hopfield" read-out (a soft, similarity-weighted vote over the stored rows) tied LR on coverage but was about 1 point lower in accuracy at every k. The one real gain came from filling the Hopfield memory with the 10 LLM-written examples per label together with the real rows: with 1 real row per label it answers 50% of traffic within the budget (LR: 14%) and is 8 points more accurate, with no training at all. That gain is gone by 3 rows per label. Bayesian heads (a Gaussian process, a Laplace approximation of the last layer, temperature scaling) gave the same accuracy and the same certified coverage as plain LR, so a posterior uncertainty did not certify more than the top-2 gap of the softmax; the Gaussian process with its kernel fitted by marginal likelihood collapsed at 1 row per label. LLM-written symbolic rules were a clear failure on 12 to 15 intents: 24 to 33% accuracy averaged over all cells, and only 1 of 40 usable rule sets certified anything. The prior-fitted network could not be judged: only its training curve exists.

## Setup

Same protocol as S17 (`smokes/s17_warm_start/RESULTS.md`): the baseline row reproduces S17's baseline to the digit (0.802 / 0.143 at k=1).

- **Tasks (5, new-step):** CLINC150 `travel` (15 labels) and `work` (15 labels); three 12-label Banking77 sub-tasks (`sub0`, `sub1`, `sub2`). Calibration = 300 rows, test = 450 / 450 / 480 / 480 / 479 rows. Rows per task: `meta.tasks` in the JSON.
- **Rows:** k real rows per label, k in {1, 3, 10}, 3 draws (seeds 0-2).
- **Metrics:** accuracy on the test rows; certified coverage at a 5% error budget via `common.certify` (fixed grid, Clopper-Pearson, delta 0.05) on the 300 calibration rows with confidence = top-2 gap of the arm's class scores, cutoff applied to the test rows. For the symbolic arm confidence is 1 if a rule fired and 0 if it abstained. "+-" is the s.d. over the 3 draws of the task-averaged value.
- **Embeddings:** frozen bge-small for every arm except the symbolic one, which reads raw text.
- **Arms (all 10 ran in full):**
  - 0 baseline: LR (C=10) on the k rows.
  - 1a kNN vote (cosine, min(5, k) neighbours; confidence = vote margin). 1b Hopfield read-out (softmax(beta x cosine) weighted label vote, beta picked on the calibration rows from 5 to 640). 1c: the same two with the 10 LLM-written examples per label (S17's cached rows) added to the memory next to the k real rows.
  - 2a GP classifier (RBF kernel, one-vs-rest, kernel hyper-parameters by marginal likelihood). 2a' GP with a fixed RBF whose length-scale is picked on the calibration rows. 2b Laplace last layer (own block-diagonal code, prior precision by evidence from {0.01 ... 100}). 2c LR plus a temperature fitted on the calibration rows (grid 0.25 to 4).
  - 4 symbolic rules: a cheap LLM (asi1-mini) writes a regex-only `classify(text)` from the label descriptions and the k examples per label; run in a sandboxed subprocess; abstentions count as wrong in accuracy.
- **Not run:** arm 3, prior-fitted network (3.3M-parameter transformer meta-trained for 840 s, scored in-context with 1 pass and with 4 slot permutations): trained for 333 s, never scored on the five tasks (see Results). Arm 2b', laplace-torch (Kronecker, optional): not reached. Nothing else of the plan in `run.py` is missing; the cell grid is complete for the other arms.
- Run: one RTX 4090 pod; prep 93 s, memory and Bayesian arms done by 261 s, symbolic arm done by 175 s (LLM spend $0.055 logged by the run, of which $0.053 is the 45 symbolic cells).

## Results

All numbers: mean over 5 tasks x 3 draws (15 cells per arm and k).

### Main table

| arm | acc k=1 | acc k=3 | acc k=10 | cert. cov k=1 | k=3 | k=10 | rows per label to 50% coverage |
|---|---|---|---|---|---|---|---|
| 0 baseline LR | 0.802 +- 0.015 | 0.925 +- 0.003 | 0.968 +- 0.002 | 0.143 +- 0.063 | 0.597 +- 0.101 | 0.906 +- 0.041 | 2.4 |
| 1a kNN vote | 0.794 +- 0.011 | 0.893 +- 0.004 | 0.949 +- 0.008 | 0.000 +- 0.000 | 0.242 +- 0.051 | 0.679 +- 0.271 | 6.1 |
| 1b Hopfield read-out | 0.794 +- 0.011 | 0.918 +- 0.005 | 0.958 +- 0.006 | 0.099 +- 0.098 | 0.605 +- 0.086 | 0.878 +- 0.059 | 2.4 |
| 1c kNN + LLM-written rows | 0.863 +- 0.005 | 0.911 +- 0.009 | 0.954 +- 0.005 | 0.281 +- 0.118 | 0.495 +- 0.023 | 0.742 +- 0.194 | 3.1 |
| 1c Hopfield + LLM-written rows | 0.886 +- 0.013 | 0.926 +- 0.005 | 0.960 +- 0.007 | 0.501 +- 0.157 | 0.627 +- 0.036 | 0.878 +- 0.037 | <=1 |
| 2a GP, learned kernel | 0.142 +- 0.026 | 0.914 +- 0.013 | 0.968 +- 0.002 | 0.000 +- 0.000 | 0.519 +- 0.003 | 0.913 +- 0.036 | 2.9 |
| 2a' GP, fixed RBF from calibration | 0.804 +- 0.014 | 0.923 +- 0.003 | 0.965 +- 0.001 | 0.180 +- 0.077 | 0.596 +- 0.098 | 0.898 +- 0.046 | 2.3 |
| 2b Laplace last layer | 0.802 +- 0.015 | 0.925 +- 0.003 | 0.968 +- 0.002 | 0.144 +- 0.063 | 0.598 +- 0.103 | 0.914 +- 0.034 | 2.4 |
| 2c LR + temperature | 0.802 +- 0.015 | 0.925 +- 0.003 | 0.968 +- 0.002 | 0.139 +- 0.060 | 0.596 +- 0.099 | 0.908 +- 0.034 | 2.4 |
| 4 symbolic rules (all 45 cells, failed rule sets = 0) | 0.235 +- 0.069 | 0.313 +- 0.015 | 0.328 +- 0.045 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.035 +- 0.060 | >10 |
| 3 prior-fitted network | not scored | | | | | | |

"Rows per label to 50% coverage": log-k interpolation of the all-task mean; "<=1" = already above 50% at k=1, the bottom of the grid.

### Paired difference to the baseline (same task, k, draw; mean +- standard error over 15 cells)

| arm | d-coverage k=1 | k=3 | k=10 | d-accuracy k=1 | k=3 | k=10 |
|---|---|---|---|---|---|---|
| 1a kNN vote | -0.143 +- 0.064 | -0.356 +- 0.099 | -0.228 +- 0.080 | -0.009 +- 0.004 | -0.033 +- 0.006 | -0.019 +- 0.004 |
| 1b Hopfield read-out | -0.044 +- 0.040 | +0.008 +- 0.043 | -0.028 +- 0.017 | -0.009 +- 0.004 | -0.007 +- 0.004 | -0.011 +- 0.004 |
| 1c kNN + LLM rows | +0.138 +- 0.084 | -0.102 +- 0.049 | -0.164 +- 0.062 | +0.061 +- 0.014 | -0.014 +- 0.007 | -0.014 +- 0.004 |
| 1c Hopfield + LLM rows | +0.358 +- 0.079 | +0.029 +- 0.069 | -0.028 +- 0.017 | +0.084 +- 0.013 | +0.001 +- 0.007 | -0.008 +- 0.004 |
| 2a GP, learned kernel | -0.143 +- 0.064 | -0.078 +- 0.048 | +0.007 +- 0.014 | -0.660 +- 0.019 | -0.012 +- 0.006 | +0.000 +- 0.001 |
| 2a' GP, fixed RBF | +0.037 +- 0.036 | -0.001 +- 0.006 | -0.008 +- 0.013 | +0.001 +- 0.001 | -0.002 +- 0.002 | -0.003 +- 0.001 |
| 2b Laplace | +0.001 +- 0.003 | +0.001 +- 0.002 | +0.008 +- 0.013 | +0.000 | -0.000 | +0.000 |
| 2c temperature | -0.004 +- 0.003 | -0.001 +- 0.001 | +0.002 +- 0.008 | +0.000 | +0.000 | +0.000 |
| 4 symbolic rules | -0.143 +- 0.064 | -0.597 +- 0.115 | -0.872 +- 0.040 | -0.568 +- 0.043 | -0.612 +- 0.031 | -0.640 +- 0.043 |

### Per family (accuracy / coverage, CLINC tasks ; Banking77 tasks)

| arm | k=1 | k=3 | k=10 |
|---|---|---|---|
| 0 baseline LR | 0.81/0.20 ; 0.80/0.11 | 0.92/0.49 ; 0.93/0.67 | 0.97/0.90 ; 0.97/0.91 |
| 1b Hopfield read-out | 0.80/0.08 ; 0.79/0.11 | 0.91/0.45 ; 0.92/0.71 | 0.95/0.82 ; 0.96/0.91 |
| 1c Hopfield + LLM rows | 0.88/0.43 ; 0.89/0.55 | 0.92/0.41 ; 0.93/0.77 | 0.96/0.82 ; 0.96/0.92 |
| 1c kNN + LLM rows | 0.86/0.42 ; 0.86/0.19 | 0.91/0.47 ; 0.91/0.52 | 0.96/0.72 ; 0.95/0.76 |
| 2b Laplace | 0.81/0.20 ; 0.80/0.10 | 0.92/0.49 ; 0.93/0.67 | 0.97/0.93 ; 0.97/0.90 |
| 4 symbolic rules | 0.28/0.00 ; 0.20/0.00 | 0.37/0.00 ; 0.27/0.00 | 0.34/0.09 ; 0.32/0.00 |

Per-task tables are in the script output. The seeded-Hopfield coverage at k=1 is uneven across tasks: banking/sub1 0.88, clinc/travel 0.70, banking/sub0 0.54, banking/sub2 0.23, clinc/work 0.15 (LR: 0.32, 0.40, 0.00, 0.00, 0.00). clinc/work stays uncertified for most arms at k=1 and k=3.

### AUROC of the confidence for "prediction is correct" (k = 1 / 3 / 10)

Baseline LR 0.863 / 0.933 / 0.948; Laplace 0.863 / 0.933 / 0.948; temperature 0.864 / 0.934 / 0.948; fixed-RBF GP 0.860 / 0.930 / 0.947; learned-kernel GP 0.844 / 0.930 / 0.952; Hopfield 0.859 / 0.917 / 0.938; Hopfield + LLM rows 0.890 / 0.914 / 0.938; kNN 0.500 / 0.787 / 0.893. The Bayesian heads do not rank errors better than the softmax gap.

### Symbolic rules

- Rule sets: 40 of 45 usable ("ok"), 5 syntax errors from the LLM (3 at k=1, 2 at k=10; the main-table row scores those cells as accuracy 0). Accuracy over the 40 usable cells: 0.293 (k=1), 0.313 (k=3), 0.379 (k=10); range 0.12 to 0.58; overall 0.328.
- The rules abstain on 63% of texts on average (41 to 77% by task). When a rule does fire it is right 90% of the time on average, but that is still above the 5% budget, so almost nothing certifies: 1 of 40 usable cells had a certified cutoff (one draw of clinc/travel at k=10, coverage 0.52; 0.17 as the mean of that task's three draws). Best task: clinc/travel at about 0.5 accuracy.
- Per task and k (usable cells): banking/sub0 0.26 / 0.21 / 0.28, banking/sub1 0.29 / 0.29 / 0.32, banking/sub2 0.22 / 0.31 / 0.35, clinc/travel 0.49 / 0.49 / 0.51, clinc/work 0.24 / 0.26 / 0.52 (k = 1 / 3 / 10).
- The high AUROC of "rule fired" (0.96 to 0.97) is real but irrelevant: the accuracy is dominated by abstentions. Programs average 5,700 characters; the LLM call averages 7.9 s per cell; 4 cells ran in parallel.

### Cost and latency (k=10)

| arm | trained parameters | fit s | predict ms per 1000 rows |
|---|---|---|---|
| 0 baseline LR | 4,620* | 0.115 | 0.7 |
| 1a / 1b / 1c memory | 0 | 0.00 to 0.03 | 1.7 to 25.6 |
| 2a GP (learned kernel) | 24* | 0.45 | 1,137 |
| 2a' GP (fixed RBF) | 0 | 1.99 | 1,096 |
| 2b Laplace | 4,620* | 0.93 | 120.7 |
| 2c temperature | 4,621* | 0.068 | 7.3 |
| 4 symbolic rules | 0 | 7.9 (LLM call) | 120.8 |

*Parameter counts are for the 12-label tasks; the two 15-label CLINC tasks have 5,775 / 5,776 / 30. The task-mean at k=10 is 5,082 / 5,083 / 26.4. The other columns are means over all tasks.

### Certificate check

Cells with a certified cutoff whose realized test error exceeded 5%: baseline 1 of 29, Laplace 1 of 29, temperature 1 of 29, kNN + LLM rows 1 of 28, every other arm 0. The worst is 0.052 (kNN + LLM rows), the others 0.050, so these are marginal overshoots of a 5% target, not large violations; mean realized error of certified cells is 0.014 to 0.017 (symbolic: 1 cell, 0.004).

### Prior-fitted network (training curve only)

3,332,368 parameters, trained on synthetic tasks (35% of batches) plus tasks from ag_news, banking, CLINC, emotion and TREC pool labels, evaluated during training on a held-out CLINC "meta" domain (log label "val(meta domain)"; LR reference on the same episodes: 0.773 / 0.883 / 0.925 at k = 1 / 3 / 10). The log stops at step 27,500 (333 s of the 840 s budget). Validation accuracy rose quickly to about 0.78 and then sat flat: best-mean checkpoint (step 25,500) 0.675 / 0.825 / 0.898 (mean 0.799; the per-k peaks in the log are 0.690 at step 11,500, 0.833 at step 6,500 and 0.898), mean of the last 20 checks 0.656 / 0.807 / 0.877 (mean 0.780). It never reached the LR reference at any k (-0.10 at k=1, -0.06 at k=3, -0.03 at k=10 for the best-mean checkpoint). This is a hint, not a result: no test-task evaluation, no coverage, 40% of the training budget used.

## Interpretation

- **Memory students alone do not beat LR.** The kNN vote gives few distinct confidence values (a vote margin out of at most 5), so it certifies nothing at k=1 and loses 14 to 36 points of coverage against LR at every k. The Hopfield read-out has a smooth confidence and ties LR on coverage (differences -0.044 +- 0.040, +0.008 +- 0.043, -0.028 +- 0.017 are inside noise) but is about 1 point lower in accuracy at every k (-0.009, -0.007, -0.011, each +- 0.004, 1.8 to 2.8 standard errors).
- **Seeding the memory with the LLM-written rows is the only mechanism that helped, and only at k=1.** Hopfield + LLM rows: 0.886 accuracy and 0.50 coverage at k=1 against 0.802 and 0.14 (+0.36 +- 0.08 coverage), with zero training. By k=3 it equals LR (+0.03 +- 0.07) and at k=10 it is marginally behind (-0.03 coverage, -0.008 accuracy). Seeded kNN is a weaker version of the same thing and falls below LR from k=3. Compared with S17's synthetic-prior arm (LR trained on the same LLM rows plus the k rows: 0.914 accuracy and 0.80 coverage at k=1, 0.87 at k=3), the training-free memory is clearly worse at k=1 (0.50 vs 0.80 coverage) and at k=3 (0.63 vs 0.87); that comparison is across two result files with the same tasks, draws and baseline row, not a paired run in this folder. The day-zero no-training alternative therefore costs coverage compared with simply fitting LR on the LLM rows; its attraction is byte-exact replay and no fit.
- **Bayesian heads add nothing.** The Laplace last layer and the temperature-scaled LR track LR to within 0.01 in both columns and in AUROC. This is expected: the two do not change the ranking of rows much, and the temperature is at the bottom of its grid; the Laplace prior precision (0.1) and the GP length-scale (0.8) are interior but never vary (see Caveats). The GP with a fixed kernel is within noise of LR; the GP with a learned kernel fits the kernel hyper-parameters on 12 to 15 rows and the one-vs-rest decision collapses at k=1 (accuracy 0.14 on 12 to 15 labels), recovering to LR level by k=3 to k=10 (coverage 0.52 at k=3 is below LR's 0.60, 0.91 at k=10 is equal). Posterior uncertainty did not certify more than the softmax gap.
- **LLM-written rules fail** on 12 to 15 overlapping intents (accuracy 0.29 to 0.38 where the program is usable, 0.24 to 0.33 counting failed programs), and more examples barely help (0.29 to 0.38 from k=1 to k=10). Certification needs under 5% error on the answered share, and a rule that fires wrongly 10% of the time cannot give that.
- **Recommendation.** Keep the S17 warm start (schema, LLM-written examples as training rows, ordinary LR student). Use the Hopfield-over-trace-store variant seeded with the LLM rows only if a no-training, exactly replayable memory is worth giving up about 30 points of day-one coverage. Do not build kNN, Bayesian-head or rule students for certification. The prior-fitted network is undecided from this run; its training curve was below LR on a held-out domain and a rerun would need to score it on the five tasks.

## Caveats

- **Partial run, no critic pass.** The pod stopped before the PFN evaluation and laplace-torch. The PFN row in the tables is empty on purpose; the training-curve paragraph above is the only PFN evidence.
- **Agreement with FINDINGS.md (round 3).** Recomputed values match the bullet: seeded Hopfield 0.89 / 0.50 at k=1 (0.886 / 0.501), LR 0.80 / 0.14, kNN worse coverage, Hopfield ~ LR, Bayesian heads ~ LR, GP learned kernel collapses at k=1, and 450 cells. One wording difference: the bullet's symbolic range "0.29 to 0.38" holds only for the 40 usable rule sets; averaged over all 45 cells (5 syntax errors scored 0) it is 0.24 / 0.31 / 0.33. The bullet's "GP with calibrated kernel" corresponds to arm 2a' here; arm 2a ("learned kernel") is the one that collapsed. The bullet says the Hopfield memory is "credible" as an alternative; the cross-file comparison above shows it is well behind LR-on-LLM-rows at k=1 and k=3, which the bullet does not say. The bullet's "Prior-fitted-network arm not reached" is slightly off: the arm was reached and meta-trained for 333 s (27,500 steps), then the pod stopped before it was scored on the five tasks. No number in the bullet was found to be wrong.
- **Small sample.** 5 tasks, 3 draws; standard errors of coverage differences at k=1 and k=3 are 0.04 to 0.10, so only the seeded-Hopfield gain at k=1 (about 4.5 s.e.), the kNN losses, the GP collapse and the symbolic failure are clear. The seeded-Hopfield gain is mostly from three tasks (banking/sub1, clinc/travel, banking/sub0); clinc/work and banking/sub2 stay low.
- **Hyper-parameters at the grid edge or never varying.** The temperature grid minimum (0.25) was selected in 89% of cells (the rest 0.35), so the calibrated LR is probably under-sharpened relative to what a wider grid would pick; ranking by gap is almost unaffected by temperature, so the conclusion "no gain" is unlikely to move. The Laplace prior precision was 0.1 in all 45 cells and the predictive ranking barely changed; why was not investigated. The fixed-RBF GP length-scale was 0.8 in all 45 cells (interior to 0.3 to 1.2). Hopfield beta: arm 1c median 80 (40 to 160); arm 1b median 40 (20 to 80); pooled over both arms median 80 (20 to 160).
- **Certificate.** A handful of certified cells overshoot 5% realized error by up to 0.2 points (baseline too); the certificate is a per-cell high-probability bound, not a guarantee on each finite test draw.
- **Symbolic arm.** One LLM (asi1-mini) at temperature 0.2, one sample per cell, no repair loop for syntax errors (5 of 45 programs lost) and no retry; a stronger model or an agentic loop could do better on accuracy but the abstention-heavy structure makes a 5% certificate hard. It also only used regexes by design.
- **Cross-smoke comparisons** (S17 arms) rely on the same task, label and draw setup and an identical baseline row, but the LLM rows are S17's cached rows and no paired test across folders was run.

## Verdict

Memory, Bayesian and rule-based students do not improve on a frozen-embedding LR student for certified coverage, except the LLM-seeded Hopfield memory at k=1 (+0.358 +- 0.079 coverage, about 4.5 standard errors). That single useful effect is the LLM-written seed rows (also visible here as a no-training Hopfield memory at k=1), and that is dominated by S17's simpler recipe of training LR on the same rows. The PFN remains untested on the tasks.

## Revision

Corrections after a critique, each re-checked against `salvaged_results.json` and the pod log:

- PFN gap to the LR reference corrected to -0.10 / -0.06 / -0.03 (was -0.1 and -0.07 to -0.05); 0.675 / 0.825 / 0.898 is labelled the best-mean checkpoint, with the per-k peaks (0.690, 0.833, 0.898) given separately.
- Grid-edge claim narrowed: only the temperature is at its grid edge; Laplace precision and GP length-scale are interior but constant.
- Hopfield read-out "ties LR" restricted to coverage; accuracy is about 1 point lower at every k (plain-words section and Interpretation).
- Verdict's first sentence now excepts the seeded Hopfield memory at k=1.
- Hopfield beta range now given per arm and pooled.
- Parameter counts in the cost table marked as 12-label values, with the 15-label values.
- Realized-error range corrected to 0.014 to 0.017 (symbolic: 1 cell).
- Added the PFN "not reached" wording difference to the FINDINGS comparison.
- banking/sub2 symbolic per-task line rounded correctly (0.22 / 0.31 / 0.35).
