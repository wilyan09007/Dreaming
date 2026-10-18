Banking77: train 9993, test 3076; head_train 1000 / head_cal 500 / holdout 1576; embedding BAAI/bge-small-en-v1.5.

## Banking77: student trained on 770 rows

Student test accuracy 0.821; change rate: all test 0.179, head_train 0.177, head_cal 0.176, holdout 0.181 (286 changes in 1576 rows).

### Changes caught vs share reviewed (holdout; fraction of all changes in holdout caught)

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

### Certificate against corrections (conf = 1-P(change) for heads; certify on head_cal 500 rows, apply to holdout; delta 5%)

Coverage = share auto-passed (not reviewed). 'changes caught' = share of holdout changes that fall below the cutoff (i.e. get reviewed).

| budget | confidence | cutoff | cal coverage | holdout coverage | holdout change rate above cutoff | changes caught |
|---|---|---|---|---|---|---|
| 5% | max prob | - | 0.000 | 0.000 | - | 1.000 |
| 5% | top-2 gap | 0.475 | 0.324 | 0.346 | 0.011 | 0.979 |
| 5% | GBC feat | - | 0.000 | 0.000 | - | 1.000 |
| 5% | LR feat | 0.962 | 0.424 | 0.458 | 0.014 | 0.965 |
| 5% | LR feat+emb | 0.945 | 0.376 | 0.411 | 0.019 | 0.958 |
| 10% | max prob | 0.361 | 0.600 | 0.623 | 0.040 | 0.864 |
| 10% | top-2 gap | 0.200 | 0.650 | 0.670 | 0.043 | 0.843 |
| 10% | GBC feat | 0.944 | 0.600 | 0.633 | 0.033 | 0.885 |
| 10% | LR feat | 0.737 | 0.724 | 0.746 | 0.045 | 0.815 |
| 10% | LR feat+emb | 0.804 | 0.676 | 0.694 | 0.045 | 0.829 |

Infeasible certificates (seed-0 cal rows), diagnosed by run.py:

- max prob @ 5%: SGR search has m=10 steps so per-step delta=0.0050; zero errors would need 104 top rows; first error at rank 135 (stable order); 500 unique scores among 500 cal rows, largest tie block 1; 57 of 500 tie-aware candidate cutoffs satisfy the bound.
- GBC feat @ 5%: SGR search has m=10 steps so per-step delta=0.0050; zero errors would need 104 top rows; first error at rank 81 (stable order); 313 unique scores among 500 cal rows, largest tie block 76; 0 of 313 tie-aware candidate cutoffs satisfy the bound.

### Head quality vs number of reviewed rows used to train the head (holdout; mean ± sd over 5 random subsets of head_train, 1000 = all)

Last three columns: in how many of the subsets the head beats the raw gap on that metric (the gap is one fixed ranking on this holdout).

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

### Robustness: 5 different random head_train/head_cal/holdout splits (1000 head rows; mean ± sd)

| ranking | share for 80% | share for 90% | AP | caught @10% |
|---|---|---|---|---|
| max prob | 0.308 ± 0.006 | 0.435 ± 0.006 | 0.591 ± 0.006 | 0.366 ± 0.007 |
| top-2 gap | 0.288 ± 0.002 | 0.410 ± 0.015 | 0.583 ± 0.012 | 0.354 ± 0.009 |
| GBC feat | 0.275 ± 0.008 | 0.384 ± 0.009 | 0.645 ± 0.016 | 0.395 ± 0.016 |
| LR feat | 0.255 ± 0.010 | 0.364 ± 0.013 | 0.690 ± 0.018 | 0.424 ± 0.014 |
| LR feat+emb | 0.311 ± 0.014 | 0.423 ± 0.029 | 0.632 ± 0.021 | 0.381 ± 0.010 |

Certificate on those splits (cert = SGR on that split's 500 cal rows, delta 5%). Coverage averages every split, infeasible ones count as 0. Change-rate columns are the realized rate above the cutoff: mean over feasible splits only, then the maximum over splits; 'infeasible' = splits with nothing certified.

| ranking | cert@5% coverage | infeasible @5% | change rate @5% mean / max | cert@10% coverage | infeasible @10% | change rate @10% mean / max |
|---|---|---|---|---|---|---|
| max prob | 0.215 ± 0.178 | 2/5 | 0.017 / 0.023 | 0.686 ± 0.033 | 0/5 | 0.052 / 0.061 |
| top-2 gap | 0.210 ± 0.174 | 2/5 | 0.013 / 0.014 | 0.727 ± 0.041 | 0/5 | 0.056 / 0.079 |
| GBC feat | 0.054 ± 0.108 | 4/5 | 0.007 / 0.007 | 0.729 ± 0.053 | 0/5 | 0.054 / 0.067 |
| LR feat | 0.374 ± 0.216 | 1/5 | 0.018 / 0.025 | 0.760 ± 0.025 | 0/5 | 0.055 / 0.071 |
| LR feat+emb | 0.136 ± 0.172 | 3/5 | 0.010 / 0.019 | 0.662 ± 0.031 | 0/5 | 0.046 / 0.055 |

Per-split certified holdout coverage (splits seeds 0 to 4, in order):

| ranking | cert@5% | cert@10% |
|---|---|---|
| max prob | 0.000 / 0.310 / 0.416 / 0.347 / 0.000 | 0.623 / 0.712 / 0.687 / 0.695 / 0.712 |
| top-2 gap | 0.346 / 0.303 / 0.402 / 0.000 / 0.000 | 0.670 / 0.742 / 0.723 / 0.793 / 0.707 |
| GBC feat | 0.000 / 0.000 / 0.271 / 0.000 / 0.000 | 0.633 / 0.765 / 0.750 / 0.780 / 0.718 |
| LR feat | 0.458 / 0.279 / 0.598 / 0.536 / 0.000 | 0.746 / 0.755 / 0.728 / 0.803 / 0.768 |
| LR feat+emb | 0.411 / 0.000 / 0.269 / 0.000 / 0.000 | 0.694 / 0.650 / 0.667 / 0.691 / 0.610 |

Paired against the raw top-2 gap, same split: mean delta [min, max] over the 5 splits, then splits in which the ranking is better (lower share, higher AP / caught / coverage). The 5 holdouts overlap heavily, so read these as consistency checks, not independent trials.

| ranking | share for 80% | share for 90% | AP | caught @10% | cert@5% coverage | cert@10% coverage |
|---|---|---|---|---|---|---|
| max prob | +0.020 [+0.011, +0.025] 0/5 | +0.024 [+0.005, +0.034] 0/5 | +0.007 [-0.016, +0.020] 4/5 | +0.012 [+0.011, +0.017] 5/5 | +0.004 [-0.346, +0.347] 3/5 | -0.041 [-0.098, +0.004] 1/5 |
| GBC feat | -0.013 [-0.026, -0.001] 5/5 | -0.027 [-0.056, +0.000] 4/5 | +0.062 [+0.045, +0.077] 5/5 | +0.041 [+0.021, +0.060] 5/5 | -0.156 [-0.346, +0.000] 0/5 | +0.002 [-0.037, +0.027] 3/5 |
| LR feat | -0.033 [-0.043, -0.018] 5/5 | -0.046 [-0.056, -0.037] 5/5 | +0.106 [+0.069, +0.142] 5/5 | +0.069 [+0.046, +0.093] 5/5 | +0.164 [-0.024, +0.536] 3/5 | +0.033 [+0.005, +0.076] 5/5 |
| LR feat+emb | +0.023 [+0.004, +0.038] 0/5 | +0.013 [-0.030, +0.042] 2/5 | +0.048 [+0.000, +0.085] 5/5 | +0.027 [+0.018, +0.038] 5/5 | -0.074 [-0.303, +0.065] 1/5 | -0.065 [-0.102, +0.024] 1/5 |

Chosen C (LR heads, 5-fold CV on average precision; grid 0.003 to 3): headline split LR feat 1.0, LR feat+emb 0.01; across the 5 splits LR feat [1.0, 3.0, 1.0, 0.3, 1.0], LR feat+emb [0.01, 0.01, 0.01, 0.003, 0.003].

### Calibration-size sweep (heads fit on the same 1000 head_train rows; cal rows drawn at random from a 1500-row pool (head_cal + part of the holdout) that is disjoint from head_train and from the fixed 576-row eval set; 20 draws per size; the 1500 size is the whole pool, so its draws are identical)

Cell = eval coverage, mean ± sd over 20 draws, with the number of infeasible draws (nothing certified, coverage 0) in brackets. Ceiling = largest eval coverage with empirical change rate <= budget, cutoff chosen on the eval rows (an upper reference, not a certificate). Max change rate = largest realized change rate above the cutoff over all feasible draws (draws above the budget / feasible draws).

Budget 5%

| ranking | cal 250 | cal 500 | cal 1000 | cal 1500 | ceiling | max change rate |
|---|---|---|---|---|---|---|
| max prob | 0.000 ± 0.000 (20) | 0.147 ± 0.165 (11) | 0.424 ± 0.052 (0) | 0.488 ± 0.000 (0) | 0.720 | 0.025 (0/49) |
| top-2 gap | 0.000 ± 0.000 (20) | 0.208 ± 0.174 (8) | 0.488 ± 0.056 (0) | 0.557 ± 0.000 (0) | 0.738 | 0.022 (0/52) |
| GBC feat | 0.022 ± 0.098 (19) | 0.123 ± 0.192 (14) | 0.463 ± 0.135 (1) | 0.545 ± 0.000 (0) | 0.752 | 0.028 (0/46) |
| LR feat | 0.053 ± 0.159 (18) | 0.369 ± 0.172 (3) | 0.558 ± 0.044 (0) | 0.632 ± 0.000 (0) | 0.780 | 0.028 (0/59) |
| LR feat+emb | 0.024 ± 0.104 (19) | 0.155 ± 0.193 (12) | 0.478 ± 0.039 (0) | 0.502 ± 0.000 (0) | 0.733 | 0.020 (0/49) |

Budget 10%

| ranking | cal 250 | cal 500 | cal 1000 | cal 1500 | ceiling | max change rate |
|---|---|---|---|---|---|---|
| max prob | 0.422 ± 0.234 (4) | 0.648 ± 0.051 (0) | 0.714 ± 0.020 (0) | 0.731 ± 0.000 (0) | 0.840 | 0.068 (0/76) |
| top-2 gap | 0.492 ± 0.228 (3) | 0.679 ± 0.044 (0) | 0.731 ± 0.021 (0) | 0.745 ± 0.000 (0) | 0.833 | 0.062 (0/77) |
| GBC feat | 0.547 ± 0.237 (3) | 0.692 ± 0.032 (0) | 0.729 ± 0.014 (0) | 0.740 ± 0.000 (0) | 0.856 | 0.059 (0/77) |
| LR feat | 0.670 ± 0.095 (0) | 0.751 ± 0.028 (0) | 0.786 ± 0.019 (0) | 0.800 ± 0.000 (0) | 0.868 | 0.074 (0/80) |
| LR feat+emb | 0.569 ± 0.167 (1) | 0.695 ± 0.029 (0) | 0.738 ± 0.018 (0) | 0.755 ± 0.000 (0) | 0.859 | 0.075 (0/79) |

Tie check (Banking77, all certificates issued above): 0 of 690 feasible certificates have a tie-inclusive cutoff set (conf >= cutoff) whose Clopper-Pearson bound at delta/m exceeds the budget.

(section wall time 15s)

## Banking77: student trained on 3080 rows

Student test accuracy 0.910; change rate: all test 0.090, head_train 0.087, head_cal 0.108, holdout 0.086 (135 changes in 1576 rows).

### Changes caught vs share reviewed (holdout; fraction of all changes in holdout caught)

| ranking | 5% | 10% | 20% | 30% | 50% |
|---|---|---|---|---|---|
| random | 0.048 | 0.100 | 0.200 | 0.302 | 0.502 |
| max prob | 0.230 | 0.504 | 0.800 | 0.896 | 0.926 |
| top-2 gap | 0.259 | 0.504 | 0.822 | 0.904 | 0.941 |
| GBC feat | 0.252 | 0.459 | 0.778 | 0.867 | 0.926 |
| LR feat | 0.311 | 0.519 | 0.830 | 0.911 | 0.933 |
| LR feat+emb | 0.259 | 0.511 | 0.763 | 0.889 | 0.941 |
| oracle | 0.585 | 1.000 | 1.000 | 1.000 | 1.000 |

### Share of outputs reviewed to catch 80% / 90% of changes; average precision (holdout)

| ranking | share for 80% | share for 90% | AP |
|---|---|---|---|
| random | 0.792 | 0.897 | 0.090 |
| max prob | 0.200 | 0.301 | 0.392 |
| top-2 gap | 0.190 | 0.300 | 0.459 |
| GBC feat | 0.213 | 0.401 | 0.392 |
| LR feat | 0.184 | 0.266 | 0.449 |
| LR feat+emb | 0.217 | 0.369 | 0.381 |
| oracle | 0.069 | 0.077 | 1.000 |

### Certificate against corrections (conf = 1-P(change) for heads; certify on head_cal 500 rows, apply to holdout; delta 5%)

Coverage = share auto-passed (not reviewed). 'changes caught' = share of holdout changes that fall below the cutoff (i.e. get reviewed).

| budget | confidence | cutoff | cal coverage | holdout coverage | holdout change rate above cutoff | changes caught |
|---|---|---|---|---|---|---|
| 5% | max prob | 0.639 | 0.676 | 0.672 | 0.012 | 0.904 |
| 5% | top-2 gap | 0.445 | 0.724 | 0.734 | 0.014 | 0.881 |
| 5% | GBC feat | 0.934 | 0.726 | 0.743 | 0.019 | 0.837 |
| 5% | LR feat | 0.925 | 0.724 | 0.735 | 0.012 | 0.896 |
| 5% | LR feat+emb | 0.946 | 0.550 | 0.539 | 0.012 | 0.926 |
| 10% | max prob | 0.392 | 0.876 | 0.889 | 0.043 | 0.556 |
| 10% | top-2 gap | 0.170 | 0.876 | 0.900 | 0.047 | 0.504 |
| 10% | GBC feat | 0.797 | 0.850 | 0.869 | 0.037 | 0.630 |
| 10% | LR feat | 0.736 | 0.876 | 0.888 | 0.042 | 0.563 |
| 10% | LR feat+emb | 0.802 | 0.876 | 0.891 | 0.045 | 0.533 |

### Head quality vs number of reviewed rows used to train the head (holdout; mean ± sd over 5 random subsets of head_train, 1000 = all)

Last three columns: in how many of the subsets the head beats the raw gap on that metric (the gap is one fixed ranking on this holdout).

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

### Robustness: 5 different random head_train/head_cal/holdout splits (1000 head rows; mean ± sd)

| ranking | share for 80% | share for 90% | AP | caught @10% |
|---|---|---|---|---|
| max prob | 0.213 ± 0.011 | 0.321 ± 0.033 | 0.417 ± 0.021 | 0.495 ± 0.013 |
| top-2 gap | 0.200 ± 0.009 | 0.316 ± 0.044 | 0.472 ± 0.018 | 0.503 ± 0.008 |
| GBC feat | 0.214 ± 0.013 | 0.349 ± 0.037 | 0.428 ± 0.039 | 0.502 ± 0.030 |
| LR feat | 0.197 ± 0.017 | 0.297 ± 0.053 | 0.455 ± 0.024 | 0.529 ± 0.018 |
| LR feat+emb | 0.231 ± 0.034 | 0.377 ± 0.074 | 0.408 ± 0.025 | 0.522 ± 0.018 |

Certificate on those splits (cert = SGR on that split's 500 cal rows, delta 5%). Coverage averages every split, infeasible ones count as 0. Change-rate columns are the realized rate above the cutoff: mean over feasible splits only, then the maximum over splits; 'infeasible' = splits with nothing certified.

| ranking | cert@5% coverage | infeasible @5% | change rate @5% mean / max | cert@10% coverage | infeasible @10% | change rate @10% mean / max |
|---|---|---|---|---|---|---|
| max prob | 0.708 ± 0.090 | 0/5 | 0.017 / 0.024 | 0.902 ± 0.009 | 0/5 | 0.050 / 0.058 |
| top-2 gap | 0.754 ± 0.025 | 0/5 | 0.017 / 0.024 | 0.913 ± 0.022 | 0/5 | 0.053 / 0.074 |
| GBC feat | 0.673 ± 0.114 | 0/5 | 0.015 / 0.024 | 0.905 ± 0.026 | 0/5 | 0.050 / 0.068 |
| LR feat | 0.764 ± 0.048 | 0/5 | 0.017 / 0.029 | 0.908 ± 0.016 | 0/5 | 0.049 / 0.063 |
| LR feat+emb | 0.665 ± 0.123 | 0/5 | 0.017 / 0.034 | 0.923 ± 0.019 | 0/5 | 0.055 / 0.066 |

Per-split certified holdout coverage (splits seeds 0 to 4, in order):

| ranking | cert@5% | cert@10% |
|---|---|---|
| max prob | 0.672 / 0.547 / 0.780 / 0.781 / 0.760 | 0.889 / 0.911 / 0.895 / 0.912 / 0.904 |
| top-2 gap | 0.734 / 0.718 / 0.760 / 0.786 / 0.772 | 0.900 / 0.897 / 0.904 / 0.956 / 0.909 |
| GBC feat | 0.743 / 0.455 / 0.709 / 0.778 / 0.681 | 0.869 / 0.893 / 0.898 / 0.945 / 0.918 |
| LR feat | 0.735 / 0.691 / 0.780 / 0.833 / 0.779 | 0.888 / 0.896 / 0.914 / 0.935 / 0.905 |
| LR feat+emb | 0.539 / 0.502 / 0.768 / 0.809 / 0.706 | 0.891 / 0.934 / 0.913 / 0.932 / 0.947 |

Paired against the raw top-2 gap, same split: mean delta [min, max] over the 5 splits, then splits in which the ranking is better (lower share, higher AP / caught / coverage). The 5 holdouts overlap heavily, so read these as consistency checks, not independent trials.

| ranking | share for 80% | share for 90% | AP | caught @10% | cert@5% coverage | cert@10% coverage |
|---|---|---|---|---|---|---|
| max prob | +0.013 [+0.008, +0.023] 0/5 | +0.006 [-0.017, +0.022] 1/5 | -0.055 [-0.082, -0.022] 0/5 | -0.008 [-0.024, +0.007] 1/5 | -0.046 [-0.171, +0.020] 1/5 | -0.011 [-0.044, +0.015] 1/5 |
| GBC feat | +0.015 [+0.001, +0.023] 0/5 | +0.034 [-0.048, +0.101] 1/5 | -0.044 [-0.067, +0.013] 1/5 | -0.001 [-0.044, +0.028] 2/5 | -0.081 [-0.263, +0.009] 1/5 | -0.008 [-0.031, +0.010] 1/5 |
| LR feat | -0.002 [-0.017, +0.008] 3/5 | -0.019 [-0.034, +0.004] 4/5 | -0.017 [-0.037, -0.007] 0/5 | +0.026 [+0.014, +0.047] 5/5 | +0.010 [-0.027, +0.047] 4/5 | -0.005 [-0.021, +0.011] 1/5 |
| LR feat+emb | +0.031 [+0.007, +0.077] 0/5 | +0.062 [+0.019, +0.114] 0/5 | -0.064 [-0.078, -0.037] 0/5 | +0.019 [+0.007, +0.043] 5/5 | -0.089 [-0.216, +0.023] 2/5 | +0.010 [-0.024, +0.038] 3/5 |

Chosen C (LR heads, 5-fold CV on average precision; grid 0.003 to 3): headline split LR feat 1.0, LR feat+emb 0.003; across the 5 splits LR feat [1.0, 0.1, 0.3, 0.3, 0.3], LR feat+emb [0.003, 0.003, 0.003, 0.03, 0.003].

### Calibration-size sweep (heads fit on the same 1000 head_train rows; cal rows drawn at random from a 1500-row pool (head_cal + part of the holdout) that is disjoint from head_train and from the fixed 576-row eval set; 20 draws per size; the 1500 size is the whole pool, so its draws are identical)

Cell = eval coverage, mean ± sd over 20 draws, with the number of infeasible draws (nothing certified, coverage 0) in brackets. Ceiling = largest eval coverage with empirical change rate <= budget, cutoff chosen on the eval rows (an upper reference, not a certificate). Max change rate = largest realized change rate above the cutoff over all feasible draws (draws above the budget / feasible draws).

Budget 5%

| ranking | cal 250 | cal 500 | cal 1000 | cal 1500 | ceiling | max change rate |
|---|---|---|---|---|---|---|
| max prob | 0.320 ± 0.325 (10) | 0.735 ± 0.080 (0) | 0.794 ± 0.012 (0) | 0.806 ± 0.000 (0) | 0.889 | 0.019 (0/70) |
| top-2 gap | 0.374 ± 0.347 (9) | 0.761 ± 0.067 (0) | 0.806 ± 0.013 (0) | 0.828 ± 0.000 (0) | 0.903 | 0.025 (0/71) |
| GBC feat | 0.356 ± 0.358 (10) | 0.727 ± 0.064 (0) | 0.790 ± 0.025 (0) | 0.816 ± 0.000 (0) | 0.877 | 0.027 (0/70) |
| LR feat | 0.379 ± 0.350 (9) | 0.762 ± 0.065 (0) | 0.813 ± 0.016 (0) | 0.819 ± 0.000 (0) | 0.894 | 0.033 (0/71) |
| LR feat+emb | 0.085 ± 0.204 (17) | 0.562 ± 0.248 (3) | 0.748 ± 0.016 (0) | 0.780 ± 0.000 (0) | 0.889 | 0.018 (0/60) |

Budget 10%

| ranking | cal 250 | cal 500 | cal 1000 | cal 1500 | ceiling | max change rate |
|---|---|---|---|---|---|---|
| max prob | 0.847 ± 0.034 (0) | 0.886 ± 0.027 (0) | 0.912 ± 0.016 (0) | 0.929 ± 0.000 (0) | 1.000 | 0.075 (0/80) |
| top-2 gap | 0.848 ± 0.036 (0) | 0.896 ± 0.025 (0) | 0.926 ± 0.014 (0) | 0.939 ± 0.000 (0) | 1.000 | 0.071 (0/80) |
| GBC feat | 0.841 ± 0.042 (0) | 0.892 ± 0.030 (0) | 0.922 ± 0.014 (0) | 0.936 ± 0.000 (0) | 1.000 | 0.078 (0/80) |
| LR feat | 0.849 ± 0.032 (0) | 0.892 ± 0.027 (0) | 0.916 ± 0.011 (0) | 0.920 ± 0.000 (0) | 1.000 | 0.079 (0/80) |
| LR feat+emb | 0.823 ± 0.080 (0) | 0.881 ± 0.031 (0) | 0.915 ± 0.015 (0) | 0.927 ± 0.000 (0) | 1.000 | 0.075 (0/80) |

Tie check (Banking77, all certificates issued above): 0 of 802 feasible certificates have a tie-inclusive cutoff set (conf >= cutoff) whose Clopper-Pearson bound at delta/m exceeds the budget.

(section wall time 23s)

Total wall time 40s on CPU.