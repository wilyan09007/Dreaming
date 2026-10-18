# S25 exotic model families on the new-step tasks (13 of 16 arm rows measured; Tsetlin machine and the two SAE-concept arms produced no result)

Status: one complete pod run (the first launch). 585 scored cells = 13 method rows x 5 tasks x 3 draws x k in {1, 3, 10}, no gaps. Three planned arms have no rows: the Tsetlin machine (its process ran about 1,970 s, until the deadline kill at 2,040 s after T0, and wrote nothing) and the two SAE-concept arms (they failed on a sparse-array bug, 8 error rows). One critic pass has been applied (see Revision at the end). Every number below is recomputed from the raw rows in `results.json` by `report_s25.py` (run `python -I report_s25.py`; it does not read `tables.md`).

How to read the statistics: the 15 cells per method and k (5 tasks x 3 draws) are not 15 independent observations. Draws within a task share the pool, the calibration set and the test set, and for the JEPA arm they share the identical network (its pretraining is seeded per task and does not change with the draw or k). The unit used in the text below is therefore the task: per task the 3 draws' paired differences are averaged, and the test is a one-sample t on the 5 task means (4 degrees of freedom; "task-level t"). With 5 tasks and no correction for the many comparisons (12 rows x 3 k x 2 measures), nothing here is more than suggestive; the cell-level s.e. table is kept for reference and overstates precision.

## In plain words

The question was whether any "exotic" model family reaches the certified 5% error budget on a new step with fewer labelled rows than the plain student: a frozen 33M-parameter encoder (bge-small) plus logistic regression (LR). Answer: none does. The ones that did not lose, in order of closeness:

- **A kernel head** (the neural tangent kernel, NTK: a closed-form similarity between two embeddings, used in kernel ridge regression or an SVM) ties LR. Its paired difference in certified coverage is +0.037 / +0.044 / +0.017 at 1 / 3 / 10 rows per label (task-level t +1.0 / +1.2 / +1.6), accuracy within 0.2 points.
- **LR on the embedding plus a hidden layer from a small label-free "JEPA-style" network** (trained on the step's unlabelled pool texts to predict the clean-text embedding from a word-dropped copy) is the only change that raises accuracy at every k, and the gain is small: +2.0 / +0.5 / +0.4 points, positive in 5 / 4 / 5 of 5 tasks (the same sign in 14 of 15 task x k means; clinc/travel is -0.001 at k=3), task-level t +3.2 / +2.4 / +3.0 (p 0.03 / 0.07 / 0.04, uncorrected). Its coverage gain (+0.057 / +0.045 / +0.018, task-level t +1.0 / +0.8 / +1.7) is not distinguishable from zero and is positive in only 2 / 3 / 3 of 5 tasks; clinc/travel is negative at every k (-0.084 / -0.019 / -0.013). It adds 394,000 inference parameters and a pretraining pass per task.
- **A binary hypervector head** (HDC "bundle": each class is the sum of its rows' 10,000-bit codes) ties at 1 and 3 rows per label. At 10 its coverage is -0.107 (pool-centered -0.139), task-level t -1.3 for both (p 0.26 / 0.27): not distinguishable from zero with 5 tasks, and the one-shot loss is mostly one task (clinc/work -0.432; the other four -0.004 to -0.044). The one-shot bundle loses about 1 point of accuracy at every k (-1.7 / -0.7 / -0.8; the pool-centered variant is +0.3 at k=1 and -1.2 / -1.0 after). Its one-pass "OnlineHD" variant loses 0.233 of coverage at k=3 (task-level t -3.5, p 0.03, negative in 5 of 5 tasks), the only coverage loss among the arms that do not fail outright that holds up at task level.
- **A spiking head** (two layers of leaky integrate-and-fire neurons) loses accuracy: -3.3 / -1.2 / -0.3 points (task-level t -3.0 / -4.7 / -2.4). Its coverage depends on which of its two confidences is used. With the membrane gap it is within noise at every k (-0.064 / -0.051 / -0.034, task-level t -1.1 / -0.5 / -1.5; k=3 is positive in 2 of 5 tasks). With the spike-count gap it is -0.083 / -0.176 / -0.112 (task-level t -1.5 / -2.0 / -1.8; the cell-level t of -2.8 at k=3 does not survive the task unit).
- **gzip compression distance with 3-nearest-neighbours, a product-key memory head, and the GPT-2 residual control** answer almost nothing within the budget: 3 certified cells out of 180 (4 rows x 45 cells, counting both gzip rows; one gzip NCD-gap cell and two product-key memory cells, all at k=10). Their low accuracy (0.25 to 0.74) is the likelier reason, not the resolution of their confidences (see Caveats).

The replay value that motivated these families (integer-exact heads) is real only for the HDC bundle, and was not measured here: the determinism check was a same-process rerun, which all 13 rows pass, so it cannot tell an integer head from a float one. Every head except gzip (raw text) and the GPT-2 control (GPT-2 small activations) reads the same float bge-small embedding, so the pinned-environment hash is still needed.

## Setup

- **Protocol (identical to S23 and S27):** 5 new-step tasks: CLINC150 `travel` and `work` (15 labels each) and three disjoint 12-label Banking77 sub-tasks (`sub0`, `sub1`, `sub2`; `default_rng(0).permutation(77)` blocks 0:12, 12:24, 24:36). Per task 300 random train rows are the fixed calibration set (never labelled rows, never in the unlabelled pool), the rest of the train rows are the pool (1,200 / 1,200 / 1,415 / 1,264 / 1,285 rows), the official test split is the evaluation set (450 / 450 / 480 / 480 / 479 rows). k real labelled rows per label, k in {1, 3, 10}, 3 draws (seeds 0 to 2), the same rows for every arm so arms are paired.
- **Metrics:** test accuracy; certified coverage at a 5% error budget via `common.certify` (fixed 40-level grid, Clopper-Pearson, delta 0.05 split over the grid) fitted on the 300 calibration rows with each arm's own confidence, cutoff applied to the test rows; realized error among the covered test rows. "+-" is the sample s.d. over the 3 draws of the task-averaged value. Paired differences are method minus baseline on the same (task, draw, k). "Task-level t" (the statistic quoted in the text) averages the 3 draws' differences per task and tests the 5 task means against zero (t = mean / (s.d. / sqrt(5)), 4 d.f., two-sided p, uncorrected); "+ of 5" counts tasks with a positive mean difference. "Cell-level s.e." is the s.d. of the 15 cell differences over sqrt(15), which treats the cells as independent and overstates precision (draws within a task share pool, calibration set, test set and, for JEPA, the network). W/L/T counts cells where the method has higher / lower / equal coverage than the baseline.
- **Baseline:** frozen BAAI/bge-small-en-v1.5 (33M parameters, 384-d, unit-normalised) + `LogisticRegression(C=10)`, confidence = top-2 probability gap. It reproduces S27's baseline row to the digit (0.811 / 0.925 / 0.962 accuracy, 0.128 / 0.635 / 0.869 coverage).
- **Arms (all hyper-parameters are fixed, untuned defaults; nothing uses the calibration or test rows except to predict):**
  - HDC: embedding -> `sign(random 384 x 10,000 projection)` (projection regenerated from seed 0). "bundle" = per-class sum of the k hypervectors (one shot, order independent); "pool-centered" = subtract the unlabelled pool mean before the sign; "online" = one OnlineHD pass (each row adds only its novel part, subtracts from a wrongly winning class). Confidence = top-2 cosine gap between query and class vectors.
  - NTK: infinite-width ReLU NTK of depth 3 on the unit-norm embeddings; kernel ridge (lambda 0.1, one-hot targets) and one-vs-rest SVM (C=10) on the precomputed kernel. Confidence = top-2 gap of the scores.
  - gzip NCD + 3-NN: gzip (level 9) normalised compression distance on the raw text to every stored row, 3 nearest rows. Two confidences: "vote margin" = (top votes - runner-up votes) / 3; "NCD gap" = 1-NN class with the gap between the nearest and second-nearest class distance.
  - Product-key memory head: embedding -> product-key memory (32 x 32 = 1,024 slots, 8 active) -> linear; 300 full-batch Adam steps, lr 1e-2; softmax top-2 gap.
  - Spiking head (snnTorch): z-scored embedding -> rate code over T=16 steps -> 256 LIF -> K LIF, beta 0.9, surrogate gradient, 300 full-batch Adam steps, lr 1e-3. Same predictions, two confidences: "spike-count gap" = (top output spikes - runner-up) / 16 (17 possible values) and "membrane gap" = top-2 gap of softmax(time-mean membrane).
  - JEPA-lite: MLP 384 -> 1,024 -> 384 trained on the task's pool texts only (no labels, no calibration or test rows) to predict the clean-text embedding from a 30%-word-dropout view (4 views per text, cosine loss, 2,000 AdamW steps); features = [embedding, L2-normalised hidden layer]; LR C=10 on the k rows.
  - GPT-2 control: GPT-2 small, block-8 residual at the last token, pool-standardised and L2-normalised, LR C=10 (same-LM control for the SAE arms; see below).
  - Not measured: Tsetlin machine on bag-of-words (tmu), SAE concepts on GPT-2 small, last token and mean over tokens.
- **Run:** one RTX 4090 pod, five arm groups as parallel processes (single-thread numpy / CPU torch, GPU for JEPA and GPT-2), then a single-thread timing pass ("bench"). The first launcher did not die with the cloud session. It kept polling the pod until 2,439 s, then logged "pod gone", a POD_NOT_FOUND on terminate, and "wall 41.2 min, est cost $0.611, ok=False" (`pod_launch_s25.txt`). The pod had vanished because the laptop re-attach to the same pod (`v6afea12hinimh`, `pod_launch_s25_attach.txt`) had already extracted the results and called `podTerminate` at its wall 19.5 min; that window lies entirely inside the launcher's 41.2 minutes. The "ok=False" and the conclusion that the first run was lost when the pod vanished are therefore an artifact of the attach finishing first: the results were already safe on the laptop, and the relaunch (`pod_launch_s25b.txt`, pod `lzpc3witb7kgsq`), started on the belief that they were lost, was redundant and was terminated after about 2 minutes.

## Results

Mean over 5 tasks x 3 draws (15 cells per method and k).

### Main table

| method | acc k=1 | acc k=3 | acc k=10 | cert. cov k=1 | cov k=3 | cov k=10 | rows per label to 50% coverage |
|---|---|---|---|---|---|---|---|
| Baseline: bge-small + LR | 0.811 +- 0.026 | 0.925 +- 0.001 | 0.962 +- 0.004 | 0.128 +- 0.035 | 0.635 +- 0.055 | 0.869 +- 0.015 | 2.2 |
| HDC bundle (one-shot) | 0.794 +- 0.030 | 0.918 +- 0.003 | 0.955 +- 0.004 | 0.124 +- 0.086 | 0.643 +- 0.065 | 0.762 +- 0.056 | 2.2 |
| HDC bundle, pool-centered | 0.814 +- 0.024 | 0.913 +- 0.003 | 0.953 +- 0.006 | 0.129 +- 0.024 | 0.582 +- 0.132 | 0.730 +- 0.058 | 2.5 |
| HDC online (OnlineHD, 1 pass) | 0.794 +- 0.030 | 0.876 +- 0.009 | 0.954 +- 0.006 | 0.124 +- 0.086 | 0.402 +- 0.084 | 0.752 +- 0.069 | 4.2 |
| NTK kernel ridge | 0.811 +- 0.025 | 0.924 +- 0.004 | 0.964 +- 0.004 | 0.166 +- 0.078 | 0.679 +- 0.084 | 0.886 +- 0.013 | 2.0 |
| NTK kernel SVM (OvR) | 0.812 +- 0.025 | 0.924 +- 0.001 | 0.965 +- 0.004 | 0.165 +- 0.077 | 0.647 +- 0.072 | 0.882 +- 0.015 | 2.1 |
| gzip NCD + 3-NN [vote margin] | 0.363 +- 0.018 | 0.536 +- 0.027 | 0.721 +- 0.008 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 | >10 |
| gzip NCD + 3-NN [NCD gap] | 0.363 +- 0.018 | 0.542 +- 0.025 | 0.721 +- 0.008 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.024 +- 0.042 | >10 |
| Product-key memory head | 0.421 +- 0.046 | 0.611 +- 0.038 | 0.743 +- 0.013 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.111 +- 0.098 | >10 |
| Spiking head [spike-count gap] | 0.778 +- 0.034 | 0.913 +- 0.007 | 0.959 +- 0.004 | 0.045 +- 0.078 | 0.459 +- 0.122 | 0.757 +- 0.095 | 3.5 |
| Spiking head [membrane gap] | 0.778 +- 0.034 | 0.913 +- 0.007 | 0.959 +- 0.004 | 0.064 +- 0.058 | 0.584 +- 0.075 | 0.835 +- 0.042 | 2.5 |
| LR on [embedding + JEPA hidden] | 0.831 +- 0.023 | 0.930 +- 0.001 | 0.966 +- 0.003 | 0.186 +- 0.063 | 0.680 +- 0.156 | 0.887 +- 0.005 | 2.0 |
| GPT-2 L8 dense residual, last token (control) | 0.253 +- 0.006 | 0.404 +- 0.006 | 0.619 +- 0.023 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 | >10 |

"Rows per label to 50% coverage" is a log-linear interpolation of the macro coverage between grid points (3 grid points, so read it as a rough ordering). The two spiking rows are one model with two confidences; any single "spiking head" figure for this column picks one after the fact (membrane gap 2.5, spike-count gap 3.5). At k=1 OnlineHD equals the bundle by construction: with one row per label no class is ever seen twice, so it never subtracts anything.

### Paired difference to the baseline

Certified coverage, mean +- s.d. over draws [cells better / worse / equal of 15]:

| method | d-cov k=1 | d-cov k=3 | d-cov k=10 |
|---|---|---|---|
| HDC bundle (one-shot) | -0.004 +- 0.074 [1/3/11] | +0.008 +- 0.040 [6/8/1] | -0.107 +- 0.070 [3/12/0] |
| HDC bundle, pool-centered | +0.000 +- 0.011 [1/2/12] | -0.053 +- 0.126 [3/10/2] | -0.139 +- 0.073 [3/12/0] |
| HDC online | -0.004 +- 0.074 [1/3/11] | -0.233 +- 0.060 [2/11/2] | -0.117 +- 0.083 [5/10/0] |
| NTK kernel ridge | +0.037 +- 0.052 [3/1/11] | +0.044 +- 0.065 [8/6/1] | +0.017 +- 0.010 [11/4/0] |
| NTK kernel SVM | +0.037 +- 0.052 [3/1/11] | +0.012 +- 0.019 [6/7/2] | +0.013 +- 0.011 [9/6/0] |
| gzip NCD [vote margin] | -0.128 +- 0.035 [0/3/12] | -0.635 +- 0.055 [0/13/2] | -0.869 +- 0.015 [0/15/0] |
| gzip NCD [NCD gap] | -0.128 +- 0.035 [0/3/12] | -0.635 +- 0.055 [0/13/2] | -0.845 +- 0.032 [0/15/0] |
| Product-key memory head | -0.128 +- 0.035 [0/3/12] | -0.635 +- 0.055 [0/13/2] | -0.758 +- 0.108 [0/15/0] |
| Spiking head [spike-count gap] | -0.083 +- 0.070 [0/3/12] | -0.176 +- 0.110 [2/11/2] | -0.112 +- 0.089 [6/9/0] |
| Spiking head [membrane gap] | -0.064 +- 0.025 [1/3/11] | -0.051 +- 0.020 [5/10/0] | -0.034 +- 0.038 [2/11/2] |
| LR on [embedding + JEPA hidden] | +0.057 +- 0.035 [4/1/10] | +0.045 +- 0.121 [10/4/1] | +0.018 +- 0.019 [11/3/1] |
| GPT-2 L8 dense residual (control) | -0.128 +- 0.035 [0/3/12] | -0.635 +- 0.055 [0/13/2] | -0.869 +- 0.015 [0/15/0] |

Accuracy, same layout:

| method | d-acc k=1 | d-acc k=3 | d-acc k=10 |
|---|---|---|---|
| HDC bundle (one-shot) | -0.017 +- 0.007 [2/13/0] | -0.007 +- 0.003 [3/11/1] | -0.008 +- 0.004 [2/12/1] |
| HDC bundle, pool-centered | +0.003 +- 0.004 [8/7/0] | -0.012 +- 0.004 [2/13/0] | -0.010 +- 0.004 [3/10/2] |
| HDC online | -0.017 +- 0.007 [2/13/0] | -0.049 +- 0.009 [0/15/0] | -0.009 +- 0.006 [2/12/1] |
| NTK kernel ridge | -0.000 +- 0.002 [7/6/2] | -0.001 +- 0.004 [6/7/2] | +0.002 +- 0.001 [7/4/4] |
| NTK kernel SVM | +0.001 +- 0.002 [8/6/1] | -0.001 +- 0.001 [5/7/3] | +0.002 +- 0.001 [8/3/4] |
| gzip NCD (both) | -0.447 +- 0.010 [0/15/0] | -0.389 +- 0.027 / -0.383 +- 0.025 [0/15/0] | -0.241 +- 0.012 [0/15/0] |
| Product-key memory head | -0.390 +- 0.036 [0/15/0] | -0.314 +- 0.038 [0/15/0] | -0.220 +- 0.016 [0/15/0] |
| Spiking head (both) | -0.033 +- 0.012 [1/14/0] | -0.012 +- 0.007 [2/13/0] | -0.003 +- 0.002 [3/8/4] |
| LR on [embedding + JEPA hidden] | +0.020 +- 0.005 [12/3/0] | +0.005 +- 0.001 [10/2/3] | +0.004 +- 0.001 [13/2/0] |
| GPT-2 L8 dense residual (control) | -0.558 +- 0.024 [0/15/0] | -0.520 +- 0.006 [0/15/0] | -0.343 +- 0.021 [0/15/0] |

Task-level test (the one quoted in the text): per task the 3 draws' paired differences are averaged; mean of the 5 task means (t, two-sided p on 4 d.f., uncorrected, tasks with a positive difference), for the arms that matter:

| method | measure | k=1 | k=3 | k=10 |
|---|---|---|---|---|
| NTK kernel ridge | coverage | +0.037 (t +1.0, p 0.36, 2 of 5) | +0.044 (t +1.2, p 0.31, 4 of 5) | +0.017 (t +1.6, p 0.18, 4 of 5) |
| NTK kernel SVM | coverage | +0.037 (t +1.0, p 0.37, 2 of 5) | +0.012 (t +0.9, p 0.40, 3 of 5) | +0.013 (t +1.1, p 0.35, 3 of 5) |
| LR on [embedding + JEPA hidden] | coverage | +0.057 (t +1.0, p 0.36, 2 of 5) | +0.045 (t +0.8, p 0.45, 3 of 5) | +0.018 (t +1.7, p 0.17, 3 of 5) |
| LR on [embedding + JEPA hidden] | accuracy | +0.020 (t +3.2, p 0.03, 5 of 5) | +0.005 (t +2.4, p 0.07, 4 of 5) | +0.004 (t +3.0, p 0.04, 5 of 5) |
| HDC bundle (one-shot) | coverage | -0.004 (t -0.1, p 0.93, 1 of 5) | +0.008 (t +0.2, p 0.85, 1 of 5) | -0.107 (t -1.3, p 0.26, 0 of 5) |
| HDC bundle (one-shot) | accuracy | -0.017 (t -3.2, p 0.03, 0 of 5) | -0.007 (t -1.8, p 0.15, 1 of 5) | -0.008 (t -2.8, p 0.05, 0 of 5) |
| HDC bundle, pool-centered | coverage | +0.000 (t +0.1, p 0.94, 1 of 5) | -0.053 (t -0.7, p 0.53, 2 of 5) | -0.139 (t -1.3, p 0.27, 1 of 5) |
| HDC bundle, pool-centered | accuracy | +0.003 (t +0.6, p 0.60, 3 of 5) | -0.012 (t -2.6, p 0.06, 1 of 5) | -0.010 (t -3.7, p 0.02, 0 of 5) |
| HDC online | coverage | -0.004 (t -0.1, p 0.93, 1 of 5) | -0.233 (t -3.5, p 0.03, 0 of 5) | -0.117 (t -1.1, p 0.32, 0 of 5) |
| Spiking head [membrane gap] | coverage | -0.064 (t -1.1, p 0.35, 0 of 5) | -0.051 (t -0.5, p 0.63, 2 of 5) | -0.034 (t -1.5, p 0.20, 0 of 5) |
| Spiking head [spike-count gap] | coverage | -0.083 (t -1.5, p 0.20, 0 of 5) | -0.176 (t -2.0, p 0.12, 1 of 5) | -0.112 (t -1.8, p 0.14, 1 of 5) |
| Spiking head (both) | accuracy | -0.033 (t -3.0, p 0.04, 1 of 5) | -0.012 (t -4.7, p 0.01, 0 of 5) | -0.003 (t -2.4, p 0.07, 1 of 5) |

The task means behind the headline statements: JEPA accuracy +0.001 / +0.013 / +0.029 / +0.037 / +0.022 at k=1 (travel, work, sub0, sub1, sub2); JEPA coverage at travel -0.084 / -0.019 / -0.013 at k=1 / 3 / 10; HDC bundle coverage at k=10 -0.004 / -0.432 / -0.037 / -0.020 / -0.044; OnlineHD coverage at k=3 -0.145 / -0.489 / -0.215 / -0.110 / -0.207. gzip, product-key memory and the GPT-2 control lose on every task at k=3 and 10 (task-level coverage t -6 to -24); the full 12-row task-level table is in the script output. Of the 72 task-level comparisons (12 rows x 3 k x 2 measures) a few would fall below p 0.05 by chance if nothing were real, so the p 0.03 to 0.07 entries above are suggestive only.

Cell-level, for reference only (mean +- s.e. over the 15 cells, t = mean / s.e.; this treats the 15 cells as independent and overstates precision: draws within a task share pool, calibration set, test set and the JEPA network):

| method | measure | k=1 | k=3 | k=10 |
|---|---|---|---|---|
| NTK kernel ridge | coverage | +0.037 +- 0.034 (t +1.1) | +0.044 +- 0.033 (t +1.3) | +0.017 +- 0.011 (t +1.6) |
| NTK kernel SVM | coverage | +0.037 +- 0.034 (t +1.1) | +0.012 +- 0.011 (t +1.2) | +0.013 +- 0.011 (t +1.1) |
| LR on [embedding + JEPA hidden] | coverage | +0.057 +- 0.050 (t +1.1) | +0.045 +- 0.051 (t +0.9) | +0.018 +- 0.011 (t +1.6) |
| LR on [embedding + JEPA hidden] | accuracy | +0.020 +- 0.005 (t +4.2) | +0.005 +- 0.002 (t +2.9) | +0.004 +- 0.001 (t +4.0) |
| HDC bundle (one-shot) | coverage | -0.004 +- 0.042 (t -0.1) | +0.008 +- 0.036 (t +0.2) | -0.107 +- 0.054 (t -2.0) |
| HDC bundle (one-shot) | accuracy | -0.017 +- 0.004 (t -4.5) | -0.007 +- 0.003 (t -2.6) | -0.008 +- 0.003 (t -3.1) |
| HDC online | coverage | -0.004 +- 0.042 (t -0.1) | -0.233 +- 0.064 (t -3.6) | -0.117 +- 0.069 (t -1.7) |
| Spiking head [membrane gap] | coverage | -0.064 +- 0.064 (t -1.0) | -0.051 +- 0.077 (t -0.7) | -0.034 +- 0.023 (t -1.5) |
| Spiking head [spike-count gap] | coverage | -0.083 +- 0.056 (t -1.5) | -0.176 +- 0.063 (t -2.8) | -0.112 +- 0.072 (t -1.5) |
| Spiking head (both) | accuracy | -0.033 +- 0.009 (t -3.4) | -0.012 +- 0.003 (t -4.3) | -0.003 +- 0.002 (t -1.7) |

### Per task: certified coverage at k = 1 / 3 / 10 (mean of 3 draws)

| method | clinc/travel | clinc/work | banking/sub0 | banking/sub1 | banking/sub2 |
|---|---|---|---|---|---|
| Baseline LR | 0.49 / 0.96 / 0.98 | 0.00 / 0.49 / 0.75 | 0.00 / 0.45 / 0.81 | 0.15 / 0.80 / 0.97 | 0.00 / 0.47 / 0.83 |
| HDC bundle (one-shot) | 0.34 / 0.95 / 0.98 | 0.00 / 0.43 / 0.32 | 0.00 / 0.61 / 0.78 | 0.28 / 0.75 / 0.95 | 0.00 / 0.47 / 0.79 |
| NTK kernel ridge | 0.50 / 0.97 / 0.98 | 0.00 / 0.50 / 0.76 | 0.00 / 0.63 / 0.87 | 0.33 / 0.77 / 0.97 | 0.00 / 0.53 / 0.85 |
| NTK kernel SVM | 0.50 / 0.97 / 0.97 | 0.00 / 0.48 / 0.75 | 0.00 / 0.47 / 0.87 | 0.33 / 0.78 / 0.97 | 0.00 / 0.53 / 0.85 |
| Spiking head [membrane gap] | 0.19 / 0.87 / 0.98 | 0.00 / 0.14 / 0.63 | 0.00 / 0.59 / 0.80 | 0.13 / 0.67 / 0.95 | 0.00 / 0.66 / 0.81 |
| LR on [embedding + JEPA hidden] | 0.41 / 0.94 / 0.97 | 0.00 / 0.41 / 0.75 | 0.15 / 0.68 / 0.85 | 0.37 / 0.84 / 1.00 | 0.00 / 0.52 / 0.86 |

The k=1 gains of NTK and JEPA both come from banking/sub1 (+0.18 and +0.22) and, for JEPA, banking/sub0 (+0.15); JEPA loses on travel at k=1 (0.41 vs 0.49). HDC bundle's k=10 loss is concentrated in clinc/work (0.32 vs 0.75). clinc/work is the hardest task for the baseline's accuracy (lowest at every k) and for the three HDC rows, but not for every arm: GPT-2, gzip and product-key memory score their best or near-best accuracy on work (GPT-2 at k=1: 0.396 against 0.20 to 0.24 on the other tasks), and at k=3 banking/sub0 has lower coverage than work for the baseline (0.45 vs 0.49), the NTK SVM (0.47 vs 0.48) and the spike-count confidence (0.00 vs 0.17). Accuracy and coverage for the other rows are in the script output.

### Realized error against the 5% budget

Cells that certified anything (of 15) at k = 1 / 3 / 10, with the pooled realized error among covered test rows (covered rows summed over cells) and the worst cell in brackets:

| method | k=1 | k=3 | k=10 |
|---|---|---|---|
| Baseline LR | 3: 0.0045 (0.009) | 13: 0.0110 (0.020) | 15: 0.0089 (0.027) |
| HDC bundle | 4: 0.0081 (0.012) | 14: 0.0107 (0.022) | 14: 0.0104 (0.030) |
| HDC bundle, pool-centered | 3: 0.0068 (0.021) | 12: 0.0127 (0.030) | 13: 0.0144 (0.034) |
| HDC online | 4: 0.0081 (0.012) | 8: 0.0103 (0.021) | 13: 0.0109 (0.035) |
| NTK kernel ridge | 4: 0.0096 (0.016) | 14: 0.0114 (0.027) | 15: 0.0076 (0.019) |
| NTK kernel SVM | 4: 0.0087 (0.012) | 13: 0.0113 (0.027) | 15: 0.0073 (0.023) |
| gzip NCD [NCD gap] | 0 | 0 | 1: 0.0061 (0.006) |
| Product-key memory head | 0 | 0 | 2: 0.0255 (0.026) |
| Spiking head [spike-count gap] | 1: 0.0066 (0.007) | 9: 0.0084 (0.020) | 13: 0.0092 (0.020) |
| Spiking head [membrane gap] | 2: 0.0067 (0.012) | 13: 0.0114 (0.027) | 15: 0.0116 (0.024) |
| LR on [embedding + JEPA hidden] | 5: 0.0208 (0.039) | 13: 0.0126 (0.029) | 15: 0.0100 (0.040) |

gzip NCD [vote margin] and the GPT-2 control certified nothing in any cell. Over all 13 rows 270 of 585 cells certified something, and in none of those 270 did the realized test error exceed 0.05 (worst cell 0.040, JEPA at k=10; next 0.039, JEPA at k=1). These counts are by row, not by distinct predictor: the two spiking rows share predictions and contribute 23 + 30 of the 270 (32 distinct cells where at least one of the two confidences certified, so 249 of 540 cells counting the spiking head once), and the "3 of 180" for gzip, product-key memory and the GPT-2 control counts both gzip rows (4 rows x 45 cells; the two gzip rows are different predictors, a 3-NN vote and the 1-NN class). The zero-exceedance result is unchanged under either count. For the baseline the coverage the certificate promised on the calibration rows (0.547) matches the coverage realized on test (0.544). JEPA is the arm with the highest realized error at k=1 (0.021 pooled over 5 cells) while still under budget. At k=1 no arm certifies anything in more than 5 of its 15 cells, so k=1 coverage means are dominated by clinc/travel and banking/sub1.

### Cost, parameters, latency

`pred_ms_per_1000` in the code and in `tables.md` is `seconds / n_test_rows * 1000`, which is milliseconds per row (equivalently seconds per 1,000 rows), not "ms per 1,000 rows". The table uses ms per row.

Quiet single-thread timing pass (clinc/travel, k=3, draw 0, min of 3 fits / 3 predicts; predict includes featurising from the cached embedding: the HDC sign-projection, the NCD compressions; the bge encoder itself is excluded for every row):

| method | params at inference (15 labels) | fit s | predict ms per row | one-off prep s |
|---|---|---|---|---|
| Baseline LR | 5,775 | 0.009 | 0.001 | 0 |
| HDC bundle / centered / online | 150,000 integers or floats (class memory; the 384 x 10,000 projection is regenerated from seed 0) | 0.0003 / 0.0003 / 0.004 | 0.17 | 0.34 to 0.39 |
| NTK kernel ridge | 675 dual coefficients | 0.0003 | 0.004 | 0 |
| NTK kernel SVM | 675 dual coefficients | 0.014 | 0.013 | 0 |
| gzip NCD + 3-NN | 0 (stores the 45 raw texts) | 0.00001 | 0.455 (about 10 microseconds per stored row, grows with the number of stored rows) | 0.013 |

The bench pass was killed at its deadline while it was on the Tsetlin arm, which comes before the torch and GPU groups in the arm list, so product-key memory, spiking, JEPA and the GPT-2 control have no quiet timing. Their numbers from the main run (five groups in parallel on a 128-CPU pod, so contended and indicative only; predict excludes featurisation because features were cached): fit 0.58 s (product-key memory, 300 Adam steps), 5.8 s (spiking head, 300 Adam steps), 0.026 s plus a per-task pretraining pass of about 5.8 s on the GPU (JEPA, mean over tasks at draw 0), 0.016 s (GPT-2 control; its feature-extraction time is not meaningful because the failed SAE arms had already cached part of the GPT-2 features); predict 0.008 / 0.18 / 0.002 / 0.006 ms per row. Parameters at inference, mean over the five tasks (12- and 15-label): LR 5,082; HDC 132,000; NTK 529 (plus the k x K stored support embeddings, 13.8k floats at k=3 on 12 labels, which the count leaves out); product-key memory 119,770; spiking head 101,952; JEPA 412,839 (394,240 predictor weights needed at inference plus the LR); GPT-2 control 10,151. For scale, S27 measured bge-small encoding at 4.8 ms per text on a contended CPU, so every head here costs less than the encoder; the HDC projection adds about 0.17 ms (3.5%).

### Determinism

All 13 rows are bit-identical (predictions and confidences) on 5 of 5 tasks when features and any pretraining (JEPA, spiking, product-key memory, all trained from the same seed) are recomputed in the same process on the same pod at k=3, draw 0. This is a same-machine, same-thread-count rerun. The scout's plan for the HDC arm (score bytes across CPU vs GPU, 1 vs 8 threads, 1,000 repeated calls) was not run, and the check passes for float heads too, so it says nothing about integer-exactness.

## The arms with no result

### Tsetlin machine: not measured

What the pod log establishes, and what it does not:

- Installed: `pip install tmu` rc=0 (1 s) at 22:20:33. The tm process printed tmu's import warnings at 22:21:41 (pycuda missing, so it fell back to CPU clause banks, as logged). Those warnings come from `tmu.clause_bank.clause_bank_cuda`, which is imported partway through tmu's import, so the line establishes that the import had started and had reached that module, not that it finished.
- Then silence for 32 minutes 47 seconds. The log has no `SKIP` line, no `ERROR` line, no "draw 0 done", and `rows_tm.jsonl` was never created (`run_arm.py` deletes it at start and only creates it on the first emitted row). A missing `SKIP` line only means no exception was raised; an import or a constructor that never returned would leave the same log, so "construction succeeded" and "the import ran to the end" are not established. The stall could be anywhere from the end of the import to the first predict. The group was terminated by SIGTERM at the deadline (`tm past deadline, terminating`, rc -15, at 22:54:28), so no traceback exists. The tm process itself ran about 1,970 s (22:21:39 to 22:54:28); the 2,040 s in `timing.json` is the time since T0, which includes the 69 s task build.
- The first cell was clinc/travel, draw 0, k=1: 15 rows, 100 epochs = 1,500 sample updates, then prediction on 450 test and 300 calibration rows (750 predictions). No row was written for it in about 1,970 s.
- The bench pass hit the same arm: at 22:54:34 the same tmu warnings appear, and nothing follows until the bench was killed at 22:58:28. The bench does one prep, three fits at k=3 (45 rows x 33 epochs = 1,485 updates each) and three predicts, so even that unit of work did not finish in 234 s, while every other CPU arm fits in 0.0003 to 6 s.
- Ruled out by the log: install failure (rc 0), CUDA (CPU fallback logged), the sparse-array bug (the Tsetlin arm feeds dense `uint32` matrices from `CountVectorizer.toarray()` and does not call `full_proba`), a logged exception (none appears).
- Not established: whether tmu hung or was only slow, and where. If it was not hung, that one cell (1,500 updates plus 750 predictions) took at least about 1,970 s, which is at least about 340 times the slowest other CPU fit (the spiking head, 5.8 s). Candidates I could not check: another numpy-2 incompatibility in tmu 0.8.3 beyond the one patched expression; the cost of 15 classes x 200 weighted clauses x 4,000 literals with `max_included_literals=16`. The tmu version on the pod (0.8.3) is only known from a comment in `arms.py`; the version list that pip offers on the laptop stops at 0.7.1, so I could not reproduce it locally.
- Consequence for a rerun: time one fit on the pod with a printed per-epoch time and put a wall-clock cap on each cell before running the grid. In this run the Tsetlin arm held the pod for 30 minutes after everything else had finished (see Time and cost).

Evidence about readable clauses, integer-only scores and Boolean-feature students is therefore absent from this run.

### SAE concept arms: not measured, and what the sparse-array bug touched

The one-line fix changes `lib.full_proba` from `np.zeros((len(X), K))` to `np.zeros((X.shape[0], K))`. `len()` of a scipy sparse matrix raises `TypeError: sparse array length is ambiguous; use getnnz() or shape[0]`.

- **Affected:** only the two SAE-concept arms ("last token" and "mean over tokens"), the only ones that pass a sparse matrix (L2-normalised CSR codes over 24,576 features) to `full_proba`. Eight error rows in `results.json`, four per arm: clinc/travel k=1, 3, 10 and clinc/work k=1, draw 0. After those four errors `run_arm.py`'s `errs >= 4` guard skipped every remaining cell of both arms without writing another row (the log prints "draw 0/1/2 done, finished" within the same second). Both arms produced zero scored cells; no `explain_*.json` concept audit trail was written.
- **Not affected:** the other 13 rows. Their inputs are dense arrays, for which `len(X) == X.shape[0]`, so the fix cannot change their behaviour. The GPT-2 dense-residual control shares the SAE class but takes the dense path and is valid. A comparison of `arms.py`, `run.py`, `run_arm.py`, `lib.py` and `report.py` against the code the pod ran shows that line as the only change, so the on-disk tree should reproduce the 585 rows even though the pod ran the pre-fix `lib.py` (I did not re-run anything).
- The fixed relaunch (`pod_s25b`) was terminated, so the SAE arms stay unmeasured.
- What the control does say: the GPT-2 layer-8 last-token residual alone gives 0.253 / 0.404 / 0.619 accuracy and certifies nothing, so a decoder residual is a far weaker sentence feature than bge-small. A sparse code of those same activations would start from that weak base; the public SAE-probe result the scout cites (sparse codes underperform dense probes at these label counts) points the same way, but this run did not test it.

## Replay and auditability: what each family buys

Replay in the mission means the same input gives the same output byte for byte; audit means a decision can be traced to the rows or concepts behind it. All measured heads except gzip (raw text) and the GPT-2 control (GPT-2 small activations) read the same float bge-small embedding, so the encoder's float arithmetic is common to all of them and the pinned-environment hash is still required. What differs is the head. The table separates what follows from the construction (not measured here) from what was measured.

| family | head arithmetic | what it buys for replay / audit | what it costs here |
|---|---|---|---|
| HDC bundle | class memory = integer sums of +-1 codes (each entry in [-k, k]); a score is an integer dot product of at most 10,000 x k < 2^24, so float32 sums are exact in any order. Only the encoder and the sign step are float; the cosine normalisation is deterministic IEEE arithmetic on integer-valued data. | The head's own arithmetic is integer, so it adds no float rounding of its own (by construction; not measured across threads or hardware). This does not remove the encoder's jitter. S7's top-2 gap spread of 5.66e-7 across batch variants (argmax unchanged) came from the embedding itself changing with batch composition (max embedding difference 8.94e-8 under dynamic padding), passed through LR; S7 did not measure the LR head on a fixed input, so it does not show that the LR head adds jitter. An HDC head fed the same jittered embedding can turn it into an integer jump instead, because the sign step can flip a near-zero projection. The bundle is an order-independent sum, so one trace row can be added or subtracted exactly. The projection is "regenerable from seed 0", but its bytes depend on the numpy generator version, so pin numpy or hash the 15 MB matrix. The online variant is not integer (fractional updates). | Accuracy -1.7 / -0.7 / -0.8 points, coverage -0.107 at k=10 (task-level t -1.3, p 0.26: not distinguishable from zero); 150,000 stored numbers against LR's 5,775; 0.17 ms per row for the projection. |
| NTK kernel head | float64 (arccos, sin, a linear solve for the ridge) | A prediction is a weighted sum over the k x K stored rows, so "which stored rows decided" is readable. LR's weights are equally explicit, so this is retrieval-style audit rather than a replay gain. | Ties LR; needs the stored support embeddings. |
| gzip NCD | integer compressed lengths, float division; no embedding, no training | Only arm with no encoder and no learned weights: the whole student is the stored texts. Byte-exactness would depend on the zlib build (not tested). | Accuracy 0.36 / 0.54 / 0.72, no certificate; cost grows with the stored rows. |
| Spiking head | integer output spike counts, but 101,952 float weights trained by surrogate gradient | The only integer-valued confidence besides gzip's vote margin. | Using it costs coverage: 0.045 / 0.459 / 0.757 against 0.064 / 0.584 / 0.835 for the membrane gap of the same model (same predictions), -0.12 at k=3. A 17-level confidence leaves large tie groups, and `certify` closes ties before testing the bound. The spiking pair is the only same-prediction comparison in this run (spike-count minus membrane gap: -0.019 / -0.125 / -0.078 at k = 1 / 3 / 10). gzip is not evidence for the same effect: its two rows are different predictors (a 3-NN vote and the 1-NN class, equal accuracy in only 17 of 45 cells, all 15 at k=1 plus 2 others), and the continuous NCD gap also certifies nothing at k=1 and k=3 and in 1 cell at k=10. Its accuracy (0.36 to 0.72), not confidence resolution, explains it. |
| Product-key memory, JEPA-lite | float nets | Nothing. JEPA adds a GPU-trained float predictor (394,240 weights) to hash and replay; its retraining was bit-identical on the same pod (5/5). | JEPA: +2.0 / +0.5 / +0.4 points of accuracy; product-key memory: fails. |
| Tsetlin machine, SAE concepts | integer clause votes; named concept features | The two families the scout expected to give readable, integer-only or concept-named decisions. | Not measured. |

So an integer-exact head costs something in every measured case, and the only fully encoder-free students (gzip, and the unmeasured Tsetlin machine on bag-of-words) are the ones that lost the embedding.

## Against the research scout's predictions (`research/exotic-families.md`)

| scout said | S25 found |
|---|---|
| Headline: no exotic family has a published number showing fewer labels than frozen embeddings + LR; value is replay and audit, not label savings. | Confirmed on labels: rows per label to 50% certified coverage are 2.0 (NTK ridge, JEPA), 2.1 (NTK SVM), 2.2 (HDC bundle), 2.5 (spiking, membrane gap; the better of its two confidences, picked after the fact: spike-count gap 3.5) against 2.2 for the baseline, i.e. at best 0.2 of a row per label on a 3-point grid; no family saves a row. The replay half is untested (see above). |
| HDC (item 4): pass if within 2 points of LR at k >= 3 and bit-identical across profiles; value is replay, not labels. | Accuracy half: bundle and centered are 0.7 to 1.2 points below at k=3 and 0.8 to 1.0 at k=10 (pass); OnlineHD is 4.9 points below at k=3 (fail). Bit-identical across profiles: not run. Coverage is 0.11 to 0.14 lower at k=10 (task-level t -1.3 for both, not distinguishable from zero). |
| Spiking nets: skip, an energy argument only. | Run anyway: trails LR on accuracy at every k (-3.3 / -1.2 / -0.3 points, task-level t -3.0 / -4.7 / -2.4). Coverage is lower in the means, but within noise at task level for the membrane-gap confidence (t -1.1 / -0.5 / -1.5) and for the spike-count gap (t -1.5 / -2.0 / -1.8). |
| Product-key memory, gzip/neural NCD: skip. | Run anyway: both fail (0.42 to 0.74 and 0.36 to 0.72 accuracy, almost no certificate). |
| NTK: skip, gradient-feature kernels too heavy at 77 labels, GP heads cover the idea. | The infinite-width closed form on 12 to 15 labels is cheap (0.3 ms fit) and ties LR, as S24's fixed-kernel GP did (S24: +0.037 / -0.001 / -0.008 coverage). |
| Standalone JEPA: no, it needs paired views; the useful kernel is teacher side-information (item 3). | A label-free, pool-only JEPA-lite gave a small accuracy gain (same sign in 14 of 15 task x k means) and a coverage tie. This is not the scout's item 3 (no teacher tags), which was not run. |
| Tsetlin: possible but expected to lose at k < 10; skip unless an interpretable integer-only student is wanted. SAE route (item 5): prior is that it loses. | Neither measured. |

## Transfer to Dreaming

- **Keep the ordinary student** (frozen bge-small + LR). Across all measured families the best coverage gain is +0.06 at k=1 and it is within noise. S17's LLM-written-example prior took k=1 coverage from 0.14 to 0.80 (FINDINGS round 3, same tasks; not re-run here), more than ten times the best number in this table.
- **NTK ridge** is a drop-in head with no gain and no loss. There is no reason to adopt it.
- **JEPA-lite features** are the only accuracy gain (positive in 5 / 4 / 5 of 5 tasks at k = 1 / 3 / 10, task-level t +3.2 / +2.4 / +3.0; 12 of 15 cells at k=1), and they use the unlabelled stream. S23's semi-supervised probes on the same stream (TK-KNN self-training, uniform-prior Sinkhorn) report 0.89 to 0.91 accuracy and 0.35 to 0.56 coverage at k=1 on the same baseline row (0.81 / 0.13), 8 to 10 points of accuracy against 2 here. If the stream is to be used, those are the better lever. This is a comparison across result files, not a paired run.
- **If integer-exact audit is wanted**, the one-shot HDC bundle is the only integer-scored candidate and stays within 1.7 points of LR accuracy at every k (-1.7 / -0.7 / -0.8). Before relying on it, run the cross-profile byte test (threads, CPU vs GPU, batch shape) that this run did not. It makes the head's own arithmetic integer; it does not remove the encoder's jitter (S7's gap spread came from the embedding changing with batch composition, and an HDC head fed that embedding can turn a near-zero projection's sign flip into an integer jump), so the encoder still needs the pinned environment (S7: batch size 1 and a fixed thread count gave one distinct embedding in 1,000 calls from 8 threads).
- **A float-free student** would have to be gzip (fails) or the unmeasured Tsetlin machine. That arm needs its own run with the hang diagnosed first.

## Caveats

- **Partial and single run.** Three of 16 planned arm rows have no data (Tsetlin, SAE x2), first launch only, one critic pass (see Revision). Three draws per cell: the s.d. over draws of the task-mean coverage is 0.005 (JEPA at k=10) to 0.156 (JEPA at k=3) for the rows that certify anything (NTK ridge at k=10: 0.013), and at k=1 no arm certifies anything in more than 5 of 15 cells.
- **The effective sample is 5 tasks, not 15 cells.** Draws within a task share pool, calibration and test sets, and JEPA's network is identical across draws and k. The cell-level s.e. and t in the reference table therefore overstate precision (JEPA accuracy: cell-level t +4.2 / +2.9 / +4.0, task-level +3.2 / +2.4 / +3.0; HDC bundle k=10 coverage: -2.0 against -1.3; spike-count coverage at k=3: -2.8 against -2.0). Task-level p values are uncorrected for the 72 comparisons.
- **Untuned defaults, no per-arm tuning.** HDC dimension, NTK depth and lambda, SVM C, spiking T / width / beta / steps, product-key memory size and steps are single fixed choices; the spiking head's loss and learning rate were fixed by a 5-configuration check on pool rows of a tiny travel task (stated in `arms.py`). "Never test or calibration rows" holds only for the tiny task's own splits: its 240 pool rows are drawn from the same CLINC travel train split that supplies the full task's 300 calibration rows, so about 48 of them are expected to coincide with full-task calibration rows. The effect on a 5-configuration choice is negligible. Unlike S24 and S27, no hyper-parameter was chosen on the calibration rows, so there is no selection optimism, but also no tuning. The product-key memory result (0.42 accuracy at k=1 on a frozen embedding where LR reaches 0.81; banking/sub1 falls from 0.704 at k=3 to 0.687 at k=10) most likely reflects an under-trained or unstable 300-step full-batch fit, not a property of memory layers; no training loss or training accuracy was logged to separate the two. Same caution for the spiking head and gzip.
- **gzip at k=1 is degenerate for the vote margin.** With one stored row per label the three nearest rows carry three different labels, so every vote ties and the margin is 0 for all queries; the vote-margin variant cannot certify at k=1 by construction. Tie-breaking uses the nearest distance, not the optimistic tie-break that inflated the original gzip result.
- **Confidence resolution is separable from head quality only for the spiking head.** `certify` evaluates 40 coverage levels on 300 calibration rows with delta/40 each and closes ties, so a 17-valued confidence (spike counts) can lose coverage that a continuous gap of the same predictions keeps. The spiking pair is the only same-prediction comparison here (spike-count minus membrane gap: -0.019 / -0.125 / -0.078 at k = 1 / 3 / 10). The two gzip rows are different predictors (a 3-NN vote and the 1-NN class: equal accuracy in 17 of 45 cells, all 15 at k=1 plus 2 others), not a coarse and a continuous confidence on the same predictions; the continuous NCD gap also certifies nothing at k=1 and k=3 and in 1 cell at k=10. Low accuracy (0.36 to 0.72) explains gzip; resolution does not.
- **JEPA-lite is not a like-for-like head swap.** It reads all unlabelled pool texts (including the unlabelled rows of the pool); the baseline reads only the k rows.
- **Timing.** The quiet bench covers 7 of 13 rows. The rest are contended main-run numbers. The `pred_ms_per_1000` field is ms per row (see Cost). The cost table in `tables.md` combines task-mean parameters with travel-only timings; parameter counts in the stored-row heads leave out the stored support rows.
- **Determinism check is weak evidence** (above). It is a same-process rerun on one machine, which every arm passes.
- **Cross-file comparisons.** S23 and S27 use the same tasks, splits and draws as this run (the baseline row here equals theirs). S17 and S24 do not: they use the CLINC validation split (20 rows per label) as calibration with the full 1,500-row train pool and keep the Banking77 pool in permuted order, so `draw()` picks different rows; the baseline differs from S24's (0.802 / 0.143 at k=1) for that reason (explained in S27's RESULTS). No paired test across folders.
- **Run-level checks:** the certificate held in all 270 certified cells (worst realized error 0.040 against a 0.05 budget). That is consistent with the bound, not a test of it.

## Verdict

No exotic family beats the 33M frozen encoder + LR on certified coverage at a 5% budget. The unit below is the task (5 tasks, 4 degrees of freedom): the 15 cells share pools, calibration and test sets, and the JEPA network. Two arms tie: the NTK kernel heads (coverage +0.037 / +0.044 / +0.017 for ridge, task-level t +1.0 / +1.2 / +1.6; accuracy within 0.2 points) and LR on [embedding + JEPA hidden]. For JEPA, coverage is +0.057 / +0.045 / +0.018 (task-level t +1.0 / +0.8 / +1.7), positive in only 2 / 3 / 3 of 5 tasks and negative on clinc/travel at every k; accuracy is +2.0 / +0.5 / +0.4 points, a small gain with the same sign in 14 of 15 task x k means (task-level t +3.2 / +2.4 / +3.0, p 0.03 / 0.07 / 0.04, uncorrected), the only consistent accuracy gain, at 394,000 extra inference parameters and a pretraining pass per task. The HDC bundle ties at 1 and 3 rows per label; its k=10 coverage is -0.107 (pool-centered -0.139), task-level t -1.3 for both, and the one-shot loss is mostly one task (clinc/work -0.432, the other four -0.004 to -0.044). Its one-shot accuracy is about 1 point lower (-1.7 / -0.7 / -0.8). OnlineHD loses 0.233 of coverage at k=3 (task-level t -3.5, 5 of 5 tasks negative), the only coverage loss among the arms that do not fail outright that is robust at task level. The spiking head loses accuracy (-3.3 / -1.2 / -0.3 points, task-level t -3.0 / -4.7 / -2.4) but its coverage is within noise at task level: membrane gap t -1.1 / -0.5 / -1.5, spike-count gap t -1.5 / -2.0 / -1.8. gzip NCD, the product-key memory head and the GPT-2 control lose by almost everything (accuracy 0.25 to 0.74, 3 certified cells of 180). The Tsetlin machine and the SAE-concept arms are not measured: the Tsetlin process ran about 1,970 s with no output for a cause the log does not show (hung or very slow; where it stalled is unknown), and the SAE arms hit a sparse-array bug that did not touch the other 13 rows. This matches the scout: no family saves labels, and whatever they offer must be argued on replay and audit. On that side the run gives only a by-construction argument for the HDC bundle (integer head scores, exact add and remove of a row), measured to cost 0.7 to 1.7 points of accuracy and, at k=10, 0.107 of coverage (not distinguishable from zero), with the float encoder and its batch-composition jitter still in the path; the cross-profile byte test that would settle it was not run.

## Time and cost

- RTX 4090 pod (community pool unavailable, fell back to the ALL pool), $0.89/h. Timeline from `timing.json`: task build 69 s; fast numpy, gzip and GPU groups done by 170 s; the spiking and product-key group by 448 s (spiking about 104 s per draw); Tsetlin group cut at 2,040 s; bench cut at 2,280 s (38 minutes after T0). Everything that produced data was finished 7.5 minutes after T0. The remaining 30.5 minutes of waiting on the Tsetlin arm and the bench cost about $0.45 of the pod bill (1,832 s at $0.89/h) and bought the 7 bench rows.
- Ledger in `COSTS.md`: `pod_s25` $0.611 (41 min, first launcher) and `pod_s25` $0.290 (20 min, the laptop re-attach), plus `pod_s25b` $0.03. The 41-minute row at $0.89/h is the whole life of pod `v6afea12hinimh` (about 3 minutes boot and upload plus the 38-minute run), and the re-attach (19.5 minutes) lies entirely inside that window and bills the same pod, so the $0.290 row is a double count. The S25 total is $0.611 + $0.03 = about $0.64 (including the terminated relaunch), not $0.93. `COSTS.md` is not edited here; its $0.290 `pod_s25` row should be removed or zeroed with a note.
- No LLM or API calls. The analysis in this write-up is CPU only and took seconds.

## Files

`run.py`, `run_arm.py`, `arms.py`, `lib.py` (the pod code; the only change since the pod run is the `full_proba` fix), `report.py` (the pod's own aggregation, writes `tables.md`), `report_s25.py` (recomputation used here; numpy and scipy, includes the task-level test and the grouping counts quoted above), `results.json` (600 rows: 585 scored, 8 SAE error rows, 7 bench rows), `rows_*.jsonl` (raw), `timing.json`, `tables.md`, and the logs `smokes/pod_log_s25.txt`, `smokes/pod_launch_s25.txt`, `smokes/pod_launch_s25_attach.txt`, `smokes/pod_launch_s25b.txt`. `tables.md` agrees with this document on every accuracy and coverage mean; its cost table mixes task-mean parameters with travel-only timings and labels ms per row as "ms / 1000 rows".

## Revision

One critic pass was applied after the first draft. Every table number was re-derived from `results.json` (equal to the `rows_*.jsonl` files) and none changed; what changed is the statistical unit, the run narrative, the cost, and several attributions. `report_s25.py` was extended (task-level test, grouping counts); its earlier output is unchanged.

1. **Statistics: the task, not the cell, is the unit.** The first draft quoted cell-level standard errors ("2.9 to 4.2 standard errors", "about 2 standard errors") as if the 15 cells were independent; the only caveat sat in a table header. Draws within a task share pool, calibration and test sets, and JEPA shares the identical network (pretraining is seeded per task). Replaced by task-level t (n=5, 4 d.f.) in "In plain words", the arm table notes, Transfer, Caveats and Verdict, with the cell-level table kept as reference. Changed claims: JEPA accuracy is "small, same sign in 14 of 15 task x k means, task-level t +3.2 / +2.4 / +3.0" (was 2.9 to 4.2 standard errors); JEPA coverage is positive in only 2 / 3 / 3 of 5 tasks and negative on clinc/travel at every k; HDC bundle k=10 coverage -0.107 has task-level t -1.3 (p 0.26), not about 2 standard errors, and the loss is mostly clinc/work (-0.432), pool-centered is also t -1.3; the spiking head loses on accuracy only (task-level t -3.0 / -4.7 / -2.4), its membrane-gap coverage is within noise at every k (t -1.1 / -0.5 / -1.5) and its spike-count coverage at k=3 is t -2.0, not -2.8; "about 1 point of accuracy at every k" now says one-shot bundle (pool-centered is +0.3 at k=1), and the Transfer line says "within 1.7 points"; OnlineHD k=3 (-0.233, t -3.5, 5 of 5 tasks negative) is the only coverage loss among the non-failing arms that is robust at task level.
2. **Run narrative and cost.** "The first launcher died with the cloud session" was wrong. `pod_launch_s25.txt` shows it kept polling to 2,439 s, then logged "pod gone", POD_NOT_FOUND on terminate and "wall 41.2 min, est cost $0.611, ok=False": the laptop re-attach on the same pod (`v6afea12hinimh`) had extracted the results and called `podTerminate` at its wall 19.5 min, a window entirely inside the launcher's 41.2 minutes. That is why the first run was taken as lost and why `pod_s25b` was started. The `COSTS.md` double count is stated as fact ("appears" and "If so" dropped): S25 total is $0.611 + $0.03 = about $0.64, not $0.93. `COSTS.md` itself was not edited.
3. **"Every head except gzip reads bge-small" was false** for the GPT-2 control (GPT-2 small activations). Fixed in "In plain words" and in the Replay section.
4. **S7 citation in the HDC replay row.** S7's top-2 gap spread of 5.66e-7 came from the embedding changing with batch composition (max embedding difference 8.94e-8 under dynamic padding) passed through LR; S7 did not measure the LR head on a fixed input. "No head-level float jitter (S7 measured ... for the LR head)" and the Transfer line "replaces the LR head's float jitter" credited HDC with removing something S7 never attributed to the head. Reworded: the head's own arithmetic is integer (by construction), the encoder's jitter remains, and an HDC head can turn a near-zero sign flip into an integer jump.
5. **Confidence-resolution claim for gzip removed.** The two gzip rows are different predictors (3-NN vote vs 1-NN class), with equal accuracy in 17 of 45 cells (all 15 at k=1 plus 2 others), so they are not a coarse and a continuous confidence on the same predictions. The continuous NCD gap also certifies nothing at k=1 and k=3 and 1 cell at k=10; low accuracy (0.36 to 0.72), not resolution, explains it. The only same-prediction comparison is the spiking pair (-0.019 / -0.125 / -0.078).
6. **Variants in pooled counts and ranking.** Stated where they appear: the 270 of 585 certified cells include both spiking confidences (23 + 30; 32 distinct cells, 249 of 540 with the spiking head counted once), the "3 of 180" counts both gzip rows (4 rows x 45 cells), and the 50%-coverage comparison for the spiking head picks the better confidence after the fact (membrane gap 2.5, spike-count gap 3.5). The zero-exceedance result is unchanged.
7. **Tsetlin description trimmed to what the log supports.** The pycuda warnings come from partway through tmu's import; a missing `SKIP` line only means no exception was raised, so "construction succeeded" and "ran to the end of the import" are dropped and the stall location is stated as unknown (end of import to first predict). "Roughly a thousand times slower" was a guess; the log supports one cell taking at least about 1,970 s, at least about 340 times the slowest other CPU fit (spiking, 5.8 s), if it was not hung. The Status line's 2,040 s is the time since T0; the tm process ran about 1,970 s (22:21:39 to 22:54:28). Verified and left as is: install rc 0 and the CPU fallback, the SIGTERM at 22:54:28, the bench stall of 234 s, the 8 sparse-array error rows followed by the `errs>=4` skip, and the `len(X)` to `X.shape[0]` fix being the only code change since the original S25 code (no other arm passes sparse input).
8. **Small factual slips.** "clinc/work is the hardest task for every arm" replaced (it is the lowest-accuracy task for the baseline and the HDC rows, but GPT-2, gzip and product-key memory score best or near-best on work, and at k=3 banking/sub0 has lower coverage than work for the baseline, the NTK SVM and the spike-count confidence). Coverage s.d. over draws is 0.005 (JEPA k=10) to 0.156 (JEPA k=3), not 0.02 to 0.16 (NTK ridge k=10: 0.013). "Cross-file comparisons rely on the same tasks and draws" is true for S23 and S27 only; S17 and S24 use CLINC validation as calibration and a 1,500-row pool, so `draw()` picks different rows. The spiking head's "never test or calibration rows" holds only for the tiny task's own splits; its pool rows come from the same travel train split as the full task's 300 calibration rows (about 48 expected overlaps, negligible).

Checked and unchanged: the main, paired, per-task, realized-error, rows-per-label, determinism, bench and parameter tables; realized error (270 certified cells, 0 above 0.05, worst 0.040 JEPA sub1 k=10 draw 2, next 0.039); calibration-promised vs realized baseline coverage (0.547 vs 0.544); the protocol being identical to S23's `lib.py` apart from `score_pc` and the fix; the baseline equalling S27's row; the cited S23, S24 and S27 numbers and the scout's predictions.
