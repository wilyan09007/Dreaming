# S1 — student + certificate (frozen bge-small + logistic regression)


## Banking77: 77 labels, train=9993, test=3076

Day-zero label-name cosine accuracy (no rows): **0.680**

| rows | acc | cert@5% cutoff | coverage | holdout err above cutoff | cert@2% coverage | holdout err | train s |
|---|---|---|---|---|---|---|---|
| 77 | 0.601 | - | 0.00 | - | 0.00 | - | 0.1 |
| 154 | 0.465 | - | 0.00 | - | 0.00 | - | 0.2 |
| 385 | 0.688 | 0.427 | 0.17 | 0.011 | 0.00 | - | 2.0 |
| 770 | 0.821 | 0.266 | 0.57 | 0.032 | 0.00 | - | 4.7 |
| 1540 | 0.885 | 0.323 | 0.72 | 0.027 | 0.00 | - | 7.7 |
| 3080 | 0.910 | 0.280 | 0.85 | 0.025 | 0.00 | - | 12.6 |
| 6000 | 0.923 | 0.314 | 0.88 | 0.024 | 0.38 | 0.003 | 25.2 |
| 9993 | 0.933 | 0.403 | 0.88 | 0.019 | 0.48 | 0.004 | 54.0 |

Calibration-set size vs certified coverage (full-train student, budget 5%, delta 5%), holdout = other half of test:

| cal rows | cutoff | coverage | holdout err |
|---|---|---|---|
| 100 | - | 0.00 | - |
| 200 | 0.741 | 0.70 | 0.004 |
| 500 | 0.504 | 0.85 | 0.012 |
| 1000 | 0.399 | 0.88 | 0.020 |
| 1538 | 0.397 | 0.88 | 0.019 |

Confidence = top-2 gap: coverage 0.88; confidence = max prob: coverage 0.88 (budget 5%).

## CLINC150 (in-scope): 150 labels, train=15000, test=4500

Day-zero label-name cosine accuracy (no rows): **0.695**

| rows | acc | cert@5% cutoff | coverage | holdout err above cutoff | cert@2% coverage | holdout err | train s |
|---|---|---|---|---|---|---|---|
| 150 | 0.724 | 0.026 | 0.32 | 0.030 | 0.00 | - | 0.1 |
| 300 | 0.542 | 0.145 | 0.22 | 0.023 | 0.00 | - | 0.1 |
| 750 | 0.783 | 0.343 | 0.30 | 0.024 | 0.15 | 0.000 | 0.2 |
| 1500 | 0.906 | 0.135 | 0.80 | 0.031 | 0.50 | 0.007 | 0.3 |
| 3000 | 0.942 | 0.125 | 0.92 | 0.023 | 0.78 | 0.006 | 0.7 |
| 7500 | 0.956 | 0.057 | 0.98 | 0.027 | 0.88 | 0.008 | 1.5 |
| 15000 | 0.962 | 0.089 | 0.98 | 0.024 | 0.88 | 0.006 | 2.8 |

Calibration-set size vs certified coverage (full-train student, budget 5%, delta 5%), holdout = other half of test:

| cal rows | cutoff | coverage | holdout err |
|---|---|---|---|
| 100 | - | 0.00 | - |
| 200 | 0.408 | 0.93 | 0.009 |
| 500 | 0.217 | 0.95 | 0.015 |
| 1000 | 0.211 | 0.95 | 0.015 |
| 2250 | 0.091 | 0.98 | 0.024 |

Confidence = top-2 gap: coverage 0.98; confidence = max prob: coverage 0.98 (budget 5%).

Total wall time 199s on CPU, embedding model BAAI/bge-small-en-v1.5.
