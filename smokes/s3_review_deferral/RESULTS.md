# S3 — Human review deferral (Mission 4): can a confidence head route a review budget better than raw student confidence?

Short answer: ranking by the student's raw top-2 gap already does most of the work (reviewing 20% of outputs catches 66% of changes for an 82%-accurate student; random catches 20%). A head trained on 1000 reviewer-labelled rows and fed only student scores (`LR feat`) beats the gap for the weak student on every ranking metric in 5/5 splits (share for 80%: 28.8% to 25.5%, AP 0.58 to 0.69, caught at 10% review 35.4% to 42.4%). For the stronger student it is mixed: better at the top of the queue (caught at 10% review +2.6 points, 5/5 splits) and on share-for-90% (-1.9 points, 4/5), a wash on share-for-80% (-0.2 points, better in 3/5 splits), and lower on AP (-0.017, worse in 5/5). The naive embedding head (`LR feat+emb`, one shared C over scores, one-hots and 384 standardized embedding dims) is worse than `LR feat`, and against the raw gap it is mixed for the weak student (higher AP and caught@10% in 5/5 splits, worse share-for-80% in 5/5) and worse on AP and share for the strong student. Certified auto-pass coverage depends strongly on calibration-set size (weak student, gap, 5% budget: 0.00 / 0.35 / 0.54 / 0.57 at 250 / 500 / 1000 / 1500 calibration rows), and the ranker also matters when it puts an early error into the top rows, which makes the certificate infeasible. Recommendation unchanged: start on the raw gap, use `LR feat` once there are a few hundred corrections.

## Revision

An independent reviewer found problems in the first version of this file. What changed, and how the numbers moved:

1. **Embedding-head claim corrected.** The first version said `LR feat+emb` was "worse than the gap on share-for-80% and AP for both students" and that "adding the embedding hurt". That was false for AP on the 770 student: AP 0.663 (seed 0) and 0.632 ± 0.021 over 5 splits vs gap 0.578 / 0.583, and caught@10% 0.381 vs 0.343 / 0.381 vs 0.354. Paired over the 5 splits (new table below) the embedding head is above the gap on AP (5/5, +0.000 to +0.085) and caught@10% (5/5) and worse on share-for-80% (5/5, +0.4 to +3.8 points). For the 3080 student it is worse on AP and share-for-80% (5/5 each) but above the gap on caught@10% (5/5). The supported statement is: worse than `LR feat` (all metrics for the weak student), mixed against the gap for the weak student, worse for the strong student. The recommendation (use the gap or `LR feat`) does not change. The "embeddings overfit" explanation was also over-claimed: the head has one C for 8 numeric features, 77 one-hots and 384 unit-variance standardized dims, so the result is about this naive head, not about embeddings.
2. **"Limited by calibration size, not by the ranker" was untested; it is now measured.** `run.py` has a new calibration-size sweep (fixed eval set, 20 random calibration draws per size, all five rankers, both budgets, both students). Result for the weak student at a 5% budget: the gap goes 0.000 / 0.353 / 0.539 / 0.568 over 250 / 500 / 1000 / 1500 rows, so calibration size is a first-order effect. But the ranker is not irrelevant: on the same fixed head and eval set, `LR feat` is 0.08 to 0.14 above the gap at each size, and `GBC feat` / `LR feat+emb` are infeasible in 12 of 20 draws at 500 rows (gap 3 of 20). The first version's own tables already showed the ranker moving coverage (GBC 0.147, embedding head 0.172 vs gap 0.447 at 5%). The claim holds only for gap vs `LR feat` across the 5 splits (paired delta +0.011, range -0.33 to +0.26, which is larger than the mean). For the strong student the ranker barely matters at 500 cal rows or more (all within 0.035 of the gap, except the embedding head, which is 0.04 to 0.08 below).
3. **Omitted result added.** At the 10% budget `LR feat` gives the 770 student consistently higher certified coverage than the gap: 0.788 vs 0.751, with equal realized change rate (0.062 vs 0.063); paired +0.025 to +0.054 in 5/5 splits.
4. **Diagnostics now come from `run.py`** (they were quoted from an ad-hoc recompute). Corrected numbers for the infeasible GBC 5% certificate on the 770 student, seed-0 cal rows: m = ceil(log2 500) + 1 = 10 search steps, so per-step delta = 0.05/10 = 0.005 (the first version said delta/11); zero errors would need 104 top rows (first version: about 106); the first error is at rank 81 in stable order (first version: 80); 313 unique scores among 500 rows, so 187 is 500 minus 313, and the largest tie block is 76 rows. A tie-aware scan over all 313 candidate cutoffs finds none that satisfies the bound. So the cause is an error inside the top 81 rows when 104 error-free rows are required; ties are not shown to be the cause.
5. **Robustness table no longer hides infeasible splits.** It now has infeasible-split counts, the max (not only mean) realized change rate and per-split coverage. On the 770 student at 5%, GBC is infeasible in 3 of 5 splits (0, 0, 0.286, 0.449, 0) and `LR feat+emb` in 3 of 5 (0.457, 0, 0.404, 0, 0); the first version mentioned only GBC, and also missed that `max prob` is infeasible in 1 of 5. The change-rate means in the old table were over feasible splits only (GBC: 2 splits).
6. **Chosen C corrected.** `LR feat` chose C = 1.0 on the seed-0 split for both students. `LR feat+emb` chose C = 0.01 for the 770 student (second-lowest grid value; across the 5 splits 0.01, 0.01, 0.01, 0.003, 0.003) and 0.003 for the 3080 student (grid minimum; across the 5 splits 0.003 in 4 of 5, 0.03 in one). The first version said it sat at the grid minimum for both.
7. **Head-size conclusion re-derived per metric.** The first version's rule "switch to a head after about 1000 corrections" rested on share-for-80% for three sizes. The head-size tables now count, per metric, in how many of the 5 subsets the head beats the gap. For the 770 student `LR feat` already beats the gap on AP and caught@10% at 300 rows (AP 0.628 ± 0.038 vs 0.578, 5/5; caught@10% 0.376 vs 0.343, 5/5), and at 100 rows in 4/5 subsets, while share-for-80% only improves reliably at 1000. For the 3080 student `LR feat` has lower AP than the gap at every size (0/5 at 100 and 300; 0.449 vs 0.459 at 1000, worse in 5/5 splits) while caught@10% is higher in 5/5 splits at 1000. The threshold is metric-dependent.
8. **Per-split maxima of the realized change rate** are now tabulated (below). At a 5% budget the max over 5 splits is 3.0% (gap) and 2.7% (`LR feat`) for the 770 student and 2.8% (gap), 2.9% (`LR feat`), 3.6% (`LR feat+emb`) for the 3080 student; at 10% it is at most 8.2%. No split exceeded its budget. The 5 holdouts overlap heavily, so the sd understates variation.
9. **Smaller fixes.** `log.txt` is now UTF-8 (`sys.stdout.reconfigure`), so the plus-minus signs are no longer mojibake. The global `warnings.filterwarnings("ignore")` is replaced by three message-specific filters (the CV-fold warning and two sklearn 1.10 FutureWarnings about `LogisticRegressionCV`), so other warnings now show in the log. This exposed 14 sklearn `UserWarning: No positive class found in y_true` lines, all during the 100-row head fits of the 3080 student: some CV folds had zero positives, so C was chosen on a fold score that is undefined there. No ConvergenceWarning appeared. `run.py` also counts certificates whose tie-inclusive cutoff set breaks the bound (see Caveats).

10. **Verification pass.** `run.py` was re-run unchanged: `log.txt` and `tables.md` reproduce the previous run exactly except the wall-time lines (same seeds). Every Revision item above was re-checked against `log.txt`. Three text fixes: the strong-student description now includes the share-for-90% gain (-1.9 points, 4/5 splits) instead of saying the head helps "only at the top of the queue"; the "little reason to deploy a head" sentence is replaced by the metric-dependent statement (caught@10% above the gap in 5/5 splits at 1000 rows, AP below it); and "the embedding head overfits or under-fits" in "What breaks" is replaced by the measured gaps, since the cause was not isolated. No new spend.

Numbers that did not move: every number in the first version's headline, certificate, head-size and robustness tables reproduces exactly (same seeds); only columns were added and the interpretation text was rewritten.

## Setup

- Data: Banking77 (mteb/banking77), train 9993 rows, test 3076 rows, 77 intents. Embeddings: BAAI/bge-small-en-v1.5 (384-d, normalized), loaded from the shared cache `smokes/.cache` that S1 already filled (train and test files were present at start, so nothing was re-embedded). S1's RESULTS.md never appeared during my 10-minute poll, which did not matter because the Banking77 cache files already existed.
- Student: `LogisticRegression(C=10, max_iter=2000)` on frozen embeddings, trained on `stratified_subsample(train_labels, n, seed=0)`. Two students: n=770 (test acc 0.821) and n=3080 (test acc 0.910; the 770 rows are a subset of the 3080).
- "Reviewer changes this output" := student prediction != gold label (gold plays the human). Applied to all 3076 test rows.
- Split of the 3076 test rows (seed 0 `rng.permutation`): head_train 1000, head_cal 500, holdout 1576. Headline tables are on the holdout.
- Change counts: 770-student holdout has 286 changes in 1576 rows (18.1%); 3080-student holdout has 135 changes (8.6%).
- Heads (target = P(change)); features = [top-2 gap, max prob, entropy, sorted top-5 probs (8 numeric), predicted-class one-hot (77)]:
  - `GBC feat`: GradientBoostingClassifier, 150 trees, depth 2, lr 0.05, subsample 0.8, random_state 0 (fixed, not tuned).
  - `LR feat`: logistic regression on standardized numeric features + one-hot, C chosen by 5-fold CV (average precision) on the head's own training rows from {0.003, 0.01, 0.03, 0.1, 0.3, 1, 3}.
  - `LR feat+emb`: same plus the standardized 384-d embedding, C chosen the same way (one C for all blocks).
- Baselines: raw top-2 gap (ascending gap = review first), max prob (ascending), random (mean of 200 permutations), and an oracle that ranks true changes first (upper bound).
- Certificate: `certify(conf[cal], correct=~change[cal], budget, delta=0.05)`, then `apply_certificate` on the target rows. conf = 1 - P(change) for heads; raw gap and max prob for comparison. Coverage = share auto-passed with no review.
- Calibration-size sweep (new): heads are the seed-0 fits on head_train (1000 rows). Cal rows are drawn at random from a 1500-row pool (head_cal plus the first 1000 holdout rows) that is disjoint from head_train and from a fixed 576-row eval set (the rest of the holdout). Sizes 250 / 500 / 1000 / 1500, 20 draws each (seeds 200 to 219); the 1500 size is the whole pool, so its draws are identical. Disjointness from head_train matters: using head_train rows as cal rows would score the learned heads on rows they were fit on. The eval set has 108 changes (770 student, 18.8%) and 53 (3080 student, 9.2%), so single-eval numbers are noisy.
- Seeds: student subsample seed 0; headline split seed 0; robustness splits seeds 0 to 4; head-size subsets seeds 100 to 104; sweep draws seeds 200 to 219; GBC random_state 0.
- Code: `run.py` (single file, depends on `smokes/common.py`, which is unmodified). Log in `log.txt`, raw tables in `tables.md`.

## Results: student trained on 770 rows (acc 0.821, 17.9% of test outputs changed)

### Changes caught vs share reviewed (holdout; fraction of the holdout's changes caught)

| ranking | 5% | 10% | 20% | 30% | 50% |
|---|---|---|---|---|---|
| random | 0.050 | 0.101 | 0.201 | 0.301 | 0.501 |
| max prob | 0.196 | 0.360 | 0.615 | 0.808 | 0.923 |
| top-2 gap | 0.189 | 0.343 | 0.661 | 0.818 | 0.941 |
| GBC feat | 0.206 | 0.385 | 0.668 | 0.822 | 0.941 |
| LR feat | 0.238 | 0.413 | 0.720 | 0.867 | 0.951 |
| LR feat+emb | 0.227 | 0.381 | 0.664 | 0.811 | 0.941 |
| oracle | 0.276 | 0.552 | 1.000 | 1.000 | 1.000 |

### Share of outputs reviewed to catch 80% / 90% of changes; average precision (holdout)

| ranking | share for 80% | share for 90% | AP |
|---|---|---|---|
| random | 0.797 | 0.899 | 0.186 |
| max prob | 0.296 | 0.431 | 0.589 |
| top-2 gap | 0.285 | 0.398 | 0.578 |
| GBC feat | 0.274 | 0.398 | 0.642 |
| LR feat | 0.242 | 0.359 | 0.720 |
| LR feat+emb | 0.289 | 0.368 | 0.663 |
| oracle | 0.145 | 0.164 | 1.000 |

### Certificate against corrections (head_cal 500 rows, delta 5%, single split seed 0)

"Changes caught" = share of holdout changes that fall in the reviewed region (below the cutoff).

| budget | confidence | cutoff | cal coverage | holdout coverage | holdout change rate above cutoff | changes caught |
|---|---|---|---|---|---|---|
| 5% | max prob | 0.542 | 0.342 | 0.378 | 0.017 | 0.965 |
| 5% | top-2 gap | 0.419 | 0.380 | 0.409 | 0.017 | 0.962 |
| 5% | GBC feat | - (none certifiable) | 0.000 | 0.000 | - | 1.000 |
| 5% | LR feat | 0.925 | 0.536 | 0.545 | 0.024 | 0.927 |
| 5% | LR feat+emb | 0.926 | 0.454 | 0.457 | 0.019 | 0.951 |
| 10% | max prob | 0.334 | 0.654 | 0.659 | 0.044 | 0.839 |
| 10% | top-2 gap | 0.182 | 0.666 | 0.700 | 0.047 | 0.818 |
| 10% | GBC feat | 0.840 | 0.680 | 0.700 | 0.046 | 0.822 |
| 10% | LR feat | 0.720 | 0.736 | 0.754 | 0.046 | 0.808 |
| 10% | LR feat+emb | 0.790 | 0.694 | 0.710 | 0.050 | 0.804 |

Diagnosis of the infeasible GBC 5% certificate (printed by `run.py`): SGR search has m=10 steps so per-step delta=0.0050; zero errors would need 104 top rows; first error at rank 81 (stable order); 313 unique scores among 500 cal rows, largest tie block 76; 0 of 313 tie-aware candidate cutoffs satisfy the bound.

### Head quality vs number of reviewed rows used to train the head (holdout; mean ± sd over 5 random subsets of head_train; 1000 = all of head_train, one fit)

Last three columns: in how many subsets the head beats the raw gap on that metric (the gap is one fixed ranking on this holdout).

| head rows | head | share for 80% | share for 90% | AP | caught @10% | share80 < gap | AP > gap | caught@10% > gap |
|---|---|---|---|---|---|---|---|---|
| any | top-2 gap (no head) | 0.285 | 0.398 | 0.578 | 0.343 | - | - | - |
| 100 | GBC feat | 0.332 ± 0.036 | 0.535 ± 0.128 | 0.544 ± 0.013 | 0.343 ± 0.009 | 1/5 | 0/5 | 3/5 |
| 100 | LR feat | 0.284 ± 0.005 | 0.415 ± 0.022 | 0.592 ± 0.010 | 0.359 ± 0.012 | 2/5 | 4/5 | 4/5 |
| 100 | LR feat+emb | 0.546 ± 0.057 | 0.694 ± 0.063 | 0.395 ± 0.052 | 0.256 ± 0.032 | 0/5 | 0/5 | 0/5 |
| 300 | GBC feat | 0.297 ± 0.017 | 0.438 ± 0.030 | 0.590 ± 0.019 | 0.351 ± 0.011 | 1/5 | 3/5 | 4/5 |
| 300 | LR feat | 0.281 ± 0.015 | 0.419 ± 0.017 | 0.628 ± 0.038 | 0.376 ± 0.031 | 4/5 | 5/5 | 5/5 |
| 300 | LR feat+emb | 0.374 ± 0.010 | 0.499 ± 0.022 | 0.547 ± 0.027 | 0.338 ± 0.018 | 0/5 | 1/5 | 2/5 |
| 1000 | GBC feat | 0.274 | 0.398 | 0.642 | 0.385 | 1/1 | 1/1 | 1/1 |
| 1000 | LR feat | 0.242 | 0.359 | 0.720 | 0.413 | 1/1 | 1/1 | 1/1 |
| 1000 | LR feat+emb | 0.289 | 0.368 | 0.663 | 0.381 | 0/1 | 1/1 | 1/1 |

### Robustness: 5 random head_train / head_cal / holdout splits of the same 3076 test rows (1000 head rows; mean ± sd)

| ranking | share for 80% | share for 90% | AP | caught @10% |
|---|---|---|---|---|
| max prob | 0.308 ± 0.006 | 0.435 ± 0.006 | 0.591 ± 0.006 | 0.366 ± 0.007 |
| top-2 gap | 0.288 ± 0.002 | 0.410 ± 0.015 | 0.583 ± 0.012 | 0.354 ± 0.009 |
| GBC feat | 0.275 ± 0.008 | 0.384 ± 0.009 | 0.645 ± 0.016 | 0.395 ± 0.016 |
| LR feat | 0.255 ± 0.010 | 0.364 ± 0.013 | 0.690 ± 0.018 | 0.424 ± 0.014 |
| LR feat+emb | 0.311 ± 0.014 | 0.423 ± 0.029 | 0.632 ± 0.021 | 0.381 ± 0.010 |

Certificate on those splits (SGR on each split's 500 cal rows, delta 5%). Coverage averages all splits, infeasible ones counted as 0. Change-rate columns are the realized rate above the cutoff: mean over feasible splits, then the maximum over splits.

| ranking | cert@5% coverage | infeasible @5% | change rate @5% mean / max | cert@10% coverage | infeasible @10% | change rate @10% mean / max |
|---|---|---|---|---|---|---|
| max prob | 0.320 ± 0.169 | 1/5 | 0.020 / 0.027 | 0.721 ± 0.033 | 0/5 | 0.064 / 0.074 |
| top-2 gap | 0.447 ± 0.103 | 0/5 | 0.020 / 0.030 | 0.751 ± 0.033 | 0/5 | 0.063 / 0.082 |
| GBC feat | 0.147 ± 0.187 | 3/5 | 0.012 / 0.017 | 0.765 ± 0.040 | 0/5 | 0.065 / 0.081 |
| LR feat | 0.458 ± 0.160 | 0/5 | 0.018 / 0.027 | 0.788 ± 0.023 | 0/5 | 0.062 / 0.079 |
| LR feat+emb | 0.172 ± 0.212 | 3/5 | 0.013 / 0.019 | 0.707 ± 0.017 | 0/5 | 0.056 / 0.070 |

Per-split certified holdout coverage (split seeds 0 to 4, in order):

| ranking | cert@5% | cert@10% |
|---|---|---|
| max prob | 0.378 / 0.316 / 0.482 / 0.424 / 0.000 | 0.659 / 0.739 / 0.720 / 0.753 / 0.736 |
| top-2 gap | 0.409 / 0.372 / 0.577 / 0.319 / 0.557 | 0.700 / 0.761 / 0.751 / 0.801 / 0.742 |
| GBC feat | 0.000 / 0.000 / 0.286 / 0.449 / 0.000 | 0.700 / 0.794 / 0.774 / 0.814 / 0.746 |
| LR feat | 0.545 / 0.312 / 0.635 / 0.574 / 0.225 | 0.754 / 0.789 / 0.780 / 0.826 / 0.788 |
| LR feat+emb | 0.457 / 0.000 / 0.404 / 0.000 / 0.000 | 0.710 / 0.692 / 0.702 / 0.739 / 0.691 |

Paired against the raw top-2 gap on the same split: mean delta [min, max] over the 5 splits, then the number of splits in which the ranking is better (lower share; higher AP, caught, coverage).

| ranking | share for 80% | share for 90% | AP | caught @10% | cert@5% coverage | cert@10% coverage |
|---|---|---|---|---|---|---|
| max prob | +0.020 [+0.011, +0.025] 0/5 | +0.024 [+0.005, +0.034] 0/5 | +0.007 [-0.016, +0.020] 4/5 | +0.012 [+0.011, +0.017] 5/5 | -0.127 [-0.557, +0.106] 1/5 | -0.030 [-0.048, -0.006] 0/5 |
| GBC feat | -0.013 [-0.026, -0.001] 5/5 | -0.027 [-0.056, +0.000] 4/5 | +0.062 [+0.045, +0.077] 5/5 | +0.041 [+0.021, +0.060] 5/5 | -0.300 [-0.557, +0.130] 1/5 | +0.014 [+0.000, +0.032] 4/5 |
| LR feat | -0.033 [-0.043, -0.018] 5/5 | -0.046 [-0.056, -0.037] 5/5 | +0.106 [+0.069, +0.142] 5/5 | +0.069 [+0.046, +0.093] 5/5 | +0.011 [-0.332, +0.256] 3/5 | +0.037 [+0.025, +0.054] 5/5 |
| LR feat+emb | +0.023 [+0.004, +0.038] 0/5 | +0.013 [-0.030, +0.042] 2/5 | +0.048 [+0.000, +0.085] 5/5 | +0.027 [+0.018, +0.038] 5/5 | -0.274 [-0.557, +0.048] 1/5 | -0.044 [-0.069, +0.010] 1/5 |

Chosen C (LR heads, 5-fold CV on average precision, grid 0.003 to 3): seed-0 split `LR feat` 1.0, `LR feat+emb` 0.01; across the 5 splits `LR feat` [1.0, 3.0, 1.0, 0.3, 1.0], `LR feat+emb` [0.01, 0.01, 0.01, 0.003, 0.003].

### Calibration-size sweep (new; eval coverage mean ± sd over 20 cal draws, infeasible draws of 20 in brackets)

Ceiling = largest eval coverage whose empirical change rate is within the budget, cutoff chosen on the eval rows themselves (an upper reference, not a certificate). Max change rate = largest realized change rate above the cutoff over feasible draws (draws above the budget / feasible draws).

Budget 5%

| ranking | cal 250 | cal 500 | cal 1000 | cal 1500 | ceiling | max change rate |
|---|---|---|---|---|---|---|
| max prob | 0.000 ± 0.000 (20) | 0.247 ± 0.192 (7) | 0.473 ± 0.049 (0) | 0.517 ± 0.000 (0) | 0.720 | 0.025 (0/53) |
| top-2 gap | 0.000 ± 0.000 (20) | 0.353 ± 0.161 (3) | 0.539 ± 0.043 (0) | 0.568 ± 0.000 (0) | 0.738 | 0.022 (0/57) |
| GBC feat | 0.032 ± 0.140 (19) | 0.200 ± 0.247 (12) | 0.531 ± 0.140 (1) | 0.587 ± 0.000 (0) | 0.752 | 0.030 (0/48) |
| LR feat | 0.080 ± 0.191 (17) | 0.497 ± 0.132 (1) | 0.623 ± 0.037 (0) | 0.667 ± 0.000 (0) | 0.780 | 0.034 (0/62) |
| LR feat+emb | 0.073 ± 0.174 (17) | 0.194 ± 0.242 (12) | 0.512 ± 0.028 (0) | 0.523 ± 0.000 (0) | 0.733 | 0.022 (0/51) |

Budget 10%

| ranking | cal 250 | cal 500 | cal 1000 | cal 1500 | ceiling | max change rate |
|---|---|---|---|---|---|---|
| max prob | 0.491 ± 0.229 (3) | 0.694 ± 0.036 (0) | 0.745 ± 0.017 (0) | 0.762 ± 0.000 (0) | 0.840 | 0.073 (0/77) |
| top-2 gap | 0.610 ± 0.165 (1) | 0.718 ± 0.029 (0) | 0.754 ± 0.021 (0) | 0.774 ± 0.000 (0) | 0.833 | 0.072 (0/79) |
| GBC feat | 0.606 ± 0.211 (2) | 0.722 ± 0.032 (0) | 0.752 ± 0.025 (0) | 0.774 ± 0.000 (0) | 0.856 | 0.079 (0/78) |
| LR feat | 0.688 ± 0.171 (1) | 0.783 ± 0.028 (0) | 0.809 ± 0.017 (0) | 0.821 ± 0.000 (0) | 0.868 | 0.081 (0/79) |
| LR feat+emb | 0.612 ± 0.175 (1) | 0.727 ± 0.030 (0) | 0.768 ± 0.014 (0) | 0.776 ± 0.000 (0) | 0.859 | 0.082 (0/79) |

## Results: stronger student trained on 3080 rows (acc 0.910, 9.0% of test outputs changed)

### Changes caught vs share reviewed (holdout)

| ranking | 5% | 10% | 20% | 30% | 50% |
|---|---|---|---|---|---|
| random | 0.048 | 0.100 | 0.200 | 0.302 | 0.502 |
| max prob | 0.230 | 0.504 | 0.800 | 0.896 | 0.926 |
| top-2 gap | 0.259 | 0.504 | 0.822 | 0.904 | 0.941 |
| GBC feat | 0.252 | 0.459 | 0.778 | 0.867 | 0.926 |
| LR feat | 0.311 | 0.519 | 0.830 | 0.911 | 0.933 |
| LR feat+emb | 0.259 | 0.511 | 0.763 | 0.889 | 0.941 |
| oracle | 0.585 | 1.000 | 1.000 | 1.000 | 1.000 |

### Share reviewed to catch 80% / 90% of changes; average precision (holdout)

| ranking | share for 80% | share for 90% | AP |
|---|---|---|---|
| random | 0.792 | 0.897 | 0.090 |
| max prob | 0.200 | 0.301 | 0.392 |
| top-2 gap | 0.190 | 0.300 | 0.459 |
| GBC feat | 0.213 | 0.401 | 0.392 |
| LR feat | 0.184 | 0.266 | 0.449 |
| LR feat+emb | 0.217 | 0.369 | 0.381 |
| oracle | 0.069 | 0.077 | 1.000 |

### Certificate against corrections (head_cal 500 rows, delta 5%, single split seed 0)

| budget | confidence | cutoff | cal coverage | holdout coverage | holdout change rate above cutoff | changes caught |
|---|---|---|---|---|---|---|
| 5% | max prob | 0.604 | 0.708 | 0.713 | 0.013 | 0.889 |
| 5% | top-2 gap | 0.371 | 0.778 | 0.779 | 0.016 | 0.852 |
| 5% | GBC feat | 0.933 | 0.734 | 0.756 | 0.021 | 0.815 |
| 5% | LR feat | 0.904 | 0.756 | 0.768 | 0.016 | 0.859 |
| 5% | LR feat+emb | 0.939 | 0.594 | 0.597 | 0.014 | 0.904 |
| 10% | max prob | 0.370 | 0.894 | 0.901 | 0.048 | 0.496 |
| 10% | top-2 gap | 0.142 | 0.900 | 0.917 | 0.051 | 0.452 |
| 10% | GBC feat | 0.739 | 0.872 | 0.892 | 0.048 | 0.496 |
| 10% | LR feat | 0.689 | 0.896 | 0.907 | 0.050 | 0.474 |
| 10% | LR feat+emb | 0.788 | 0.896 | 0.905 | 0.049 | 0.481 |

The 10% budget is above this student's base change rate (9.0%), so that certificate is close to vacuous: it certifies auto-passing almost everything.

### Head quality vs number of reviewed rows used to train the head (holdout; mean ± sd over 5 random subsets; 1000 = all of head_train)

At 100 head rows the 3080-student has only about 9 changes in the training subset, so the head sees very few positives; 14 sklearn warnings in `log.txt` come from CV folds of these 100-row fits that contained no positive.

| head rows | head | share for 80% | share for 90% | AP | caught @10% | share80 < gap | AP > gap | caught@10% > gap |
|---|---|---|---|---|---|---|---|---|
| any | top-2 gap (no head) | 0.190 | 0.300 | 0.459 | 0.504 | - | - | - |
| 100 | GBC feat | 0.236 ± 0.031 | 0.437 ± 0.073 | 0.348 ± 0.038 | 0.424 ± 0.066 | 1/5 | 0/5 | 1/5 |
| 100 | LR feat | 0.221 ± 0.047 | 0.375 ± 0.068 | 0.395 ± 0.044 | 0.495 ± 0.053 | 1/5 | 0/5 | 3/5 |
| 100 | LR feat+emb | 0.457 ± 0.069 | 0.615 ± 0.098 | 0.215 ± 0.020 | 0.290 ± 0.032 | 0/5 | 0/5 | 0/5 |
| 300 | GBC feat | 0.245 ± 0.027 | 0.416 ± 0.084 | 0.367 ± 0.021 | 0.421 ± 0.024 | 0/5 | 0/5 | 0/5 |
| 300 | LR feat | 0.215 ± 0.015 | 0.324 ± 0.032 | 0.403 ± 0.024 | 0.498 ± 0.044 | 0/5 | 0/5 | 2/5 |
| 300 | LR feat+emb | 0.324 ± 0.038 | 0.458 ± 0.026 | 0.288 ± 0.032 | 0.375 ± 0.038 | 0/5 | 0/5 | 0/5 |
| 1000 | GBC feat | 0.213 | 0.401 | 0.392 | 0.459 | 0/1 | 0/1 | 0/1 |
| 1000 | LR feat | 0.184 | 0.266 | 0.449 | 0.519 | 1/1 | 0/1 | 1/1 |
| 1000 | LR feat+emb | 0.217 | 0.369 | 0.381 | 0.511 | 0/1 | 0/1 | 1/1 |

### Robustness: 5 random splits (1000 head rows; mean ± sd)

| ranking | share for 80% | share for 90% | AP | caught @10% |
|---|---|---|---|---|
| max prob | 0.213 ± 0.011 | 0.321 ± 0.033 | 0.417 ± 0.021 | 0.495 ± 0.013 |
| top-2 gap | 0.200 ± 0.009 | 0.316 ± 0.044 | 0.472 ± 0.018 | 0.503 ± 0.008 |
| GBC feat | 0.214 ± 0.013 | 0.349 ± 0.037 | 0.428 ± 0.039 | 0.502 ± 0.030 |
| LR feat | 0.197 ± 0.017 | 0.297 ± 0.053 | 0.455 ± 0.024 | 0.529 ± 0.018 |
| LR feat+emb | 0.231 ± 0.034 | 0.377 ± 0.074 | 0.408 ± 0.025 | 0.522 ± 0.018 |

| ranking | cert@5% coverage | infeasible @5% | change rate @5% mean / max | cert@10% coverage | infeasible @10% | change rate @10% mean / max |
|---|---|---|---|---|---|---|
| max prob | 0.764 ± 0.031 | 0/5 | 0.020 / 0.029 | 0.930 ± 0.017 | 0/5 | 0.061 / 0.074 |
| top-2 gap | 0.799 ± 0.026 | 0/5 | 0.021 / 0.028 | 0.929 ± 0.020 | 0/5 | 0.059 / 0.077 |
| GBC feat | 0.707 ± 0.124 | 0/5 | 0.018 / 0.026 | 0.927 ± 0.023 | 0/5 | 0.059 / 0.074 |
| LR feat | 0.794 ± 0.034 | 0/5 | 0.021 / 0.029 | 0.935 ± 0.019 | 0/5 | 0.059 / 0.078 |
| LR feat+emb | 0.718 ± 0.106 | 0/5 | 0.020 / 0.036 | 0.940 ± 0.019 | 0/5 | 0.062 / 0.075 |

Per-split certified holdout coverage (split seeds 0 to 4, in order):

| ranking | cert@5% | cert@10% |
|---|---|---|
| max prob | 0.713 / 0.750 / 0.780 / 0.802 / 0.778 | 0.901 / 0.926 / 0.930 / 0.951 / 0.942 |
| top-2 gap | 0.779 / 0.763 / 0.824 / 0.833 / 0.797 | 0.917 / 0.908 / 0.935 / 0.965 / 0.919 |
| GBC feat | 0.756 / 0.461 / 0.756 / 0.799 / 0.761 | 0.892 / 0.926 / 0.912 / 0.956 / 0.947 |
| LR feat | 0.768 / 0.742 / 0.817 / 0.834 / 0.810 | 0.907 / 0.937 / 0.931 / 0.966 / 0.932 |
| LR feat+emb | 0.597 / 0.587 / 0.792 / 0.843 / 0.773 | 0.905 / 0.949 / 0.933 / 0.959 / 0.952 |

Paired against the raw top-2 gap on the same split:

| ranking | share for 80% | share for 90% | AP | caught @10% | cert@5% coverage | cert@10% coverage |
|---|---|---|---|---|---|---|
| max prob | +0.013 [+0.008, +0.023] 0/5 | +0.006 [-0.017, +0.022] 1/5 | -0.055 [-0.082, -0.022] 0/5 | -0.008 [-0.024, +0.007] 1/5 | -0.035 [-0.066, -0.013] 0/5 | +0.001 [-0.016, +0.023] 2/5 |
| GBC feat | +0.015 [+0.001, +0.023] 0/5 | +0.034 [-0.048, +0.101] 1/5 | -0.044 [-0.067, +0.013] 1/5 | -0.001 [-0.044, +0.028] 2/5 | -0.092 [-0.303, -0.022] 0/5 | -0.002 [-0.025, +0.028] 2/5 |
| LR feat | -0.002 [-0.017, +0.008] 3/5 | -0.019 [-0.034, +0.004] 4/5 | -0.017 [-0.037, -0.007] 0/5 | +0.026 [+0.014, +0.047] 5/5 | -0.005 [-0.021, +0.013] 2/5 | +0.006 [-0.010, +0.029] 3/5 |
| LR feat+emb | +0.031 [+0.007, +0.077] 0/5 | +0.062 [+0.019, +0.114] 0/5 | -0.064 [-0.078, -0.037] 0/5 | +0.019 [+0.007, +0.043] 5/5 | -0.081 [-0.181, +0.010] 1/5 | +0.011 [-0.011, +0.041] 2/5 |

Chosen C: seed-0 split `LR feat` 1.0, `LR feat+emb` 0.003; across the 5 splits `LR feat` [1.0, 0.1, 0.3, 0.3, 0.3], `LR feat+emb` [0.003, 0.003, 0.003, 0.03, 0.003].

### Calibration-size sweep (new; same layout as above)

Budget 5%

| ranking | cal 250 | cal 500 | cal 1000 | cal 1500 | ceiling | max change rate |
|---|---|---|---|---|---|---|
| max prob | 0.353 ± 0.359 (10) | 0.764 ± 0.075 (0) | 0.814 ± 0.012 (0) | 0.826 ± 0.000 (0) | 0.889 | 0.031 (0/70) |
| top-2 gap | 0.410 ± 0.374 (9) | 0.797 ± 0.024 (0) | 0.826 ± 0.011 (0) | 0.840 ± 0.000 (0) | 0.903 | 0.027 (0/71) |
| GBC feat | 0.326 ± 0.366 (11) | 0.767 ± 0.039 (0) | 0.814 ± 0.014 (0) | 0.823 ± 0.000 (0) | 0.877 | 0.035 (0/69) |
| LR feat | 0.376 ± 0.379 (10) | 0.806 ± 0.028 (0) | 0.833 ± 0.013 (0) | 0.845 ± 0.000 (0) | 0.894 | 0.039 (0/70) |
| LR feat+emb | 0.143 ± 0.250 (15) | 0.716 ± 0.075 (0) | 0.778 ± 0.018 (0) | 0.797 ± 0.000 (0) | 0.889 | 0.022 (0/65) |

Budget 10% (ceiling is 1.000 because the eval set's change rate is below 10%)

| ranking | cal 250 | cal 500 | cal 1000 | cal 1500 | ceiling | max change rate |
|---|---|---|---|---|---|---|
| max prob | 0.876 ± 0.033 (0) | 0.915 ± 0.033 (0) | 0.945 ± 0.017 (0) | 0.964 ± 0.000 (0) | 1.000 | 0.085 (0/80) |
| top-2 gap | 0.886 ± 0.031 (0) | 0.927 ± 0.023 (0) | 0.949 ± 0.014 (0) | 0.957 ± 0.000 (0) | 1.000 | 0.084 (0/80) |
| GBC feat | 0.886 ± 0.038 (0) | 0.919 ± 0.028 (0) | 0.942 ± 0.014 (0) | 0.960 ± 0.000 (0) | 1.000 | 0.085 (0/80) |
| LR feat | 0.882 ± 0.036 (0) | 0.914 ± 0.027 (0) | 0.941 ± 0.016 (0) | 0.964 ± 0.000 (0) | 1.000 | 0.081 (0/80) |
| LR feat+emb | 0.862 ± 0.077 (0) | 0.918 ± 0.030 (0) | 0.950 ± 0.017 (0) | 0.965 ± 0.000 (0) | 1.000 | 0.089 (0/80) |

## Interpretation for the mission

- Routing a review budget works, and the raw top-2 gap is already a strong baseline. With the 82% student, reviewing 20% of outputs in gap order catches 66% of the changes, 30% catches 82%, and catching 80% needs 28.5% of outputs (random needs 79.7%, the oracle 14.5%). For the 91% student the same targets cost 19.0% (80%) and 30.0% (90%). An operator can read a review budget straight off these curves.
- A learned head helps most when the student is weak, and the help is metric-dependent. `LR feat` (scores only, 1000 labelled rows) is better than the gap in 5/5 splits on share-for-80% (28.8% to 25.5%), share-for-90%, AP (0.583 to 0.690) and caught@10% (35.4% to 42.4%) for the 770 student. For the 3080 student it is mixed: better at the top of the queue (caught@10% +1.4 to +4.7 points, 5/5 splits) and on share-for-90% (-1.9 points, 4/5), a wash on share-for-80% (19.7% vs 20.0%, 3/5), and lower on AP in 5/5 splits (0.455 vs 0.472). `GBC feat` improves on the gap for the 770 student (5/5 on share-for-80%, AP, caught@10%) but not for the 3080 student. The embedding head is worse than `LR feat` on every ranking metric for both students (for the 3080 student on caught@10% only by 0.007, inside the sd); against the gap it is mixed for the weak student (AP and caught@10% up in 5/5 splits, share-for-80% worse in 5/5) and worse for the strong one on AP and share. This is evidence about this one head design (shared C over scores, one-hots and 384 dims), not about whether embeddings can help. A head with block-wise regularization or a low-dimensional projection is untested.
- Label need depends on the metric. For the 770 student `LR feat` already beats the gap on AP and caught@10% with 300 rows (5/5 subsets each; 4/5 on both at 100 rows) but share-for-80% moves reliably only at 1000 rows (4/5 at 300, 2/5 at 100) and share-for-90% at 300 rows is worse than the gap (0.419 vs 0.398). For the 3080 student `LR feat` is below the gap on AP at every size (at 1000 rows too, worse in 5/5 splits) and on share-for-80% until 1000 rows, while caught@10% is above the gap in 5/5 splits at 1000 rows (3/5 and 2/5 subsets at 100 and 300 rows). So for a strong student a head pays only at the very top of the queue and only from about 1000 corrections; the break-even is metric-dependent, not a single threshold. Practical rule from this smoke: start on the raw gap, switch to `LR feat` for a weak student once a few hundred corrections exist, and keep it only if it beats the gap on held-out corrections for the metric that matters to the operator.
- The certificate against corrections depends on both calibration size and the ranker. Weak student, 5% budget: the gap's certified coverage is 0.000 / 0.353 / 0.539 / 0.568 at 250 / 500 / 1000 / 1500 cal rows (20 draws each; the realized no-penalty ceiling on the same eval rows is 0.738), so doubling the calibration set from 500 to 1000 rows adds more coverage (+0.19) than any ranker change at 1000 rows (`LR feat` +0.08). At 250 rows nothing is certifiable for the gap at 5%. The ranker matters mainly when its top rows contain an early error: `GBC feat` and `LR feat+emb` are infeasible in 12 of 20 draws at 500 rows (gap: 3, `LR feat`: 1) and in 3 of 5 splits in the robustness runs. `LR feat` vs gap at 5% is +0.011 across the 5 splits (range -0.33 to +0.26), i.e. not distinguishable; in the sweep (one fitted head, one 576-row eval set) `LR feat` is ahead by 0.08 to 0.14, which is a single-fit result and was a favorable split (seed 0). At the 10% budget `LR feat` is consistently ahead for the weak student (+0.025 to +0.054 coverage, 5/5 splits, equal realized change rate 6.2% vs 6.3%). For the strong student the ranker barely matters at 500 or more cal rows (within 0.035 of the gap except the embedding head, 0.04 to 0.08 below), and calibration size dominates: 0.410 / 0.797 / 0.826 / 0.840.
- The bound is conservative: the realized change rate above the cutoff averaged about 2% against a 5% budget and about 6% against a 10% budget; the maximum over the 5 splits was 3.0% / 2.7% (gap / `LR feat`, 770 student) and 3.6% at most (3080 student) at 5%, and 8.2% at most at 10%. No split, and none of the feasible sweep draws (271 at 5% and 392 at 10% for the 770 student; 345 and 400 for the 3080 student, summed over the five rankers), exceeded its budget on the eval set. For the operator: at a 5% residual-change budget with 500 labelled calibration rows, an 82% student still sends about 55% of outputs to review (gap, 5-split mean coverage 0.447; one split in five was infeasible for `max prob`).
- What breaks: rankings with an early error make the certificate infeasible (GBC on seed 0: first error at rank 81, 104 error-free rows needed; ties are not the cause); the naive embedding head (one shared C over 8 scores, 77 one-hots and 384 standardized dims) is far worse than `LR feat` at 100 and 300 head rows (770 student share-for-80% 0.546 and 0.374 vs 0.284 and 0.281) and still worse at 1000, and its cause (overfitting, the shared C, the standardization) was not isolated; heads trained on about 9 positives (100 rows, strong student) are noise and their CV folds can contain zero positives.

## Caveats

- Synthetic reviewer: "reviewer changes" is defined as student != gold. That is noise-free, deterministic, and exactly the student's error. A real reviewer's corrections include preference, inconsistency, and rubber-stamping, so the real target is noisier and a head's ceiling is lower than shown here. The certificate also then bounds disagreement with the gold label, not with a particular person.
- Head training data is a uniform random sample of the test pool (the best case). In deployment the head sees labels only on rows it queued, which biases it toward the region it already ranks high; an audit sample of random reviews is needed to keep it honest. Not tested here.
- Single pool: head_train, head_cal, holdout and the sweep pool/eval set are disjoint within one set of 3076 test rows, and there is one fixed student subsample per size (seed 0). The "± sd over 5 splits" understates true variation because the 5 holdouts overlap heavily, and differences of a point or two in the share tables should not be read as real. The paired counts (k/5) are consistency checks across overlapping splits, not independent trials.
- The sweep uses one fitted head per ranker and one 576-row eval set (108 changes for the weak student, 53 for the strong one), so ranker differences inside it carry the noise of that single fit and eval set; only the calibration-draw variance is averaged over 20 draws. At the 1500 size all draws are the same rows (sd 0.000 is not a measurement of stability). The sweep covers 250 to 1500 rows only; whether coverage saturates near the ceiling beyond that is untested.
- The head sees only student probabilities and the same bge-small embedding the student already used. Mission 4 also lists tool flags and raw input fields; none were used, so a head with genuinely new information (input length, tool outcome, vendor history) is untested. GBC hyperparameters were fixed, not tuned, so GBC may be understated.
- At 100 and 300 head rows, 5-fold CV inside the LR heads sometimes had folds with fewer than 5 positives (that specific warning is filtered). The 14 "No positive class found in y_true" warnings in the log are from 100-row fits for the strong student, where a fold had zero positives; results at those sizes are noisy by construction.
- `common.py` (unmodified): `certify` ranks tied confidences in arbitrary order while `apply_certificate` includes every row at `>= cutoff`, so with ties the cumulative error at k in `certify` need not equal the error of the applied set. `run.py` now counts how often this breaks the bound on the calibration rows: 6 of 715 feasible certificates for the 770 student run and 4 of 805 for the 3080 student run have a tie-inclusive cutoff set whose Clopper-Pearson bound at delta/m exceeds the budget. Those certificates are not strictly valid; none produced a realized change rate above budget. The SGR binary search also assumes roughly monotone error, so early errors can collapse the certified coverage (GBC, seed 0). No other bug found.
- sklearn emits two FutureWarnings about `LogisticRegressionCV` (`l1_ratios`, fitted attributes) that are filtered by message; behavior with the installed version is unchanged, but a newer sklearn may need `l1_ratios=(0,)`.
- Real deployment changes: a nonstationary stream (the head must retrain as the student changes, which also invalidates its certificate), reviewer fatigue and queue latency, and a certificate that must be recomputed on corrections from the actual reviewer.

## Time and cost

- Wall time: `run.py` 39 s on CPU for the final verification run (770-student section 16 s, 3080-student section 21 s, including the sweep); `log.txt` and `tables.md` are from that run. An earlier run of the same code took 101 s on a busier machine and its outputs are identical apart from the wall-time lines; the first version took 252 s. A first run of the revised code was stopped by me after its 770-student section (two sklearn FutureWarnings were flooding the log) and the code re-run from scratch with those two warnings filtered; `log.txt` and `tables.md` are from the complete second run, and its 770-student numbers matched the stopped run's. No embedding was computed (S1's cache reused).
- OpenRouter spend: $0.00 (no API calls). The existing line in `smokes/COSTS.md` stands (`s3_review_deferral,0.00`); no new spend.
