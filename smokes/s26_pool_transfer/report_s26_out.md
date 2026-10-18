records 243; arms 9; tasks 3; k [np.int64(1), np.int64(3), np.int64(10)]; draws [np.int64(0), np.int64(1), np.int64(2)]

### Accuracy, mean over the 3 tasks (+- s.d. over 3 draws of the task-averaged value)

| arm | k=1 | k=3 | k=10 |
|---|---|---|---|
| 1 baseline | 0.513 +- 0.014 | 0.629 +- 0.005 | 0.724 +- 0.013 |
| 2 synthetic prior | 0.683 +- 0.005 | 0.709 +- 0.004 | 0.757 +- 0.009 |
| 3 hypernetwork | 0.574 +- 0.031 | 0.616 +- 0.009 | 0.625 +- 0.009 |
| 4 encoder pretrained | 0.557 +- 0.016 | 0.654 +- 0.014 | 0.736 +- 0.009 |
| 4b encoder pretrained + synth | 0.664 +- 0.004 | 0.697 +- 0.001 | 0.746 +- 0.005 |
| 5 shared LoRA | 0.346 +- 0.030 | 0.487 +- 0.031 | 0.645 +- 0.011 |
| 5c LoRA from base | 0.244 +- 0.050 | 0.392 +- 0.048 | 0.560 +- 0.052 |
| 5b shared LoRA + synth | 0.627 +- 0.008 | 0.668 +- 0.017 | 0.714 +- 0.015 |
| 5d LoRA from base + synth | 0.540 +- 0.031 | 0.561 +- 0.041 | 0.633 +- 0.045 |

### Accuracy, clinc_travel only (+- s.d. over draws)

| arm | k=1 | k=3 | k=10 |
|---|---|---|---|
| 1 baseline | 0.870 +- 0.034 | 0.975 +- 0.015 | 0.993 +- 0.005 |
| 2 synthetic prior | 0.967 +- 0.003 | 0.984 +- 0.013 | 0.998 +- 0.000 |
| 3 hypernetwork | 0.839 +- 0.036 | 0.866 +- 0.020 | 0.878 +- 0.009 |
| 4 encoder pretrained | 0.925 +- 0.019 | 0.981 +- 0.007 | 0.987 +- 0.003 |
| 4b encoder pretrained + synth | 0.957 +- 0.007 | 0.976 +- 0.014 | 0.991 +- 0.000 |
| 5 shared LoRA | 0.634 +- 0.080 | 0.822 +- 0.055 | 0.961 +- 0.016 |
| 5c LoRA from base | 0.442 +- 0.159 | 0.651 +- 0.059 | 0.920 +- 0.056 |
| 5b shared LoRA + synth | 0.925 +- 0.027 | 0.948 +- 0.032 | 0.979 +- 0.011 |
| 5d LoRA from base + synth | 0.829 +- 0.090 | 0.835 +- 0.077 | 0.960 +- 0.027 |

### Accuracy, mean of bank_us and amazon (+- s.d. over draws)

| arm | k=1 | k=3 | k=10 |
|---|---|---|---|
| 1 baseline | 0.335 +- 0.030 | 0.456 +- 0.006 | 0.590 +- 0.016 |
| 2 synthetic prior | 0.541 +- 0.007 | 0.571 +- 0.004 | 0.636 +- 0.014 |
| 3 hypernetwork | 0.442 +- 0.044 | 0.490 +- 0.013 | 0.499 +- 0.009 |
| 4 encoder pretrained | 0.372 +- 0.024 | 0.490 +- 0.021 | 0.610 +- 0.012 |
| 4b encoder pretrained + synth | 0.517 +- 0.003 | 0.558 +- 0.008 | 0.624 +- 0.007 |
| 5 shared LoRA | 0.202 +- 0.020 | 0.319 +- 0.020 | 0.487 +- 0.021 |
| 5c LoRA from base | 0.145 +- 0.015 | 0.262 +- 0.044 | 0.380 +- 0.056 |
| 5b shared LoRA + synth | 0.478 +- 0.020 | 0.529 +- 0.012 | 0.582 +- 0.028 |
| 5d LoRA from base + synth | 0.396 +- 0.003 | 0.424 +- 0.034 | 0.470 +- 0.062 |

### Certified coverage at 5%, mean over the 3 tasks (+- s.d. over 3 draws of the task-averaged value)

| arm | k=1 | k=3 | k=10 |
|---|---|---|---|
| 1 baseline | 0.134 +- 0.116 | 0.328 +- 0.003 | 0.332 +- 0.003 |
| 2 synthetic prior | 0.318 +- 0.004 | 0.330 +- 0.003 | 0.332 +- 0.000 |
| 3 hypernetwork | 0.070 +- 0.121 | 0.216 +- 0.050 | 0.235 +- 0.013 |
| 4 encoder pretrained | 0.262 +- 0.045 | 0.328 +- 0.004 | 0.333 +- 0.001 |
| 4b encoder pretrained + synth | 0.305 +- 0.006 | 0.323 +- 0.010 | 0.331 +- 0.002 |
| 5 shared LoRA | 0.000 +- 0.000 | 0.110 +- 0.105 | 0.309 +- 0.013 |
| 5c LoRA from base | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.244 +- 0.064 |
| 5b shared LoRA + synth | 0.279 +- 0.029 | 0.298 +- 0.006 | 0.324 +- 0.005 |
| 5d LoRA from base + synth | 0.129 +- 0.119 | 0.083 +- 0.144 | 0.284 +- 0.052 |

### Certified coverage at 5%, clinc_travel only (+- s.d. over draws)

| arm | k=1 | k=3 | k=10 |
|---|---|---|---|
| 1 baseline | 0.401 +- 0.348 | 0.985 +- 0.010 | 0.996 +- 0.008 |
| 2 synthetic prior | 0.954 +- 0.013 | 0.990 +- 0.008 | 0.996 +- 0.001 |
| 3 hypernetwork | 0.209 +- 0.362 | 0.648 +- 0.151 | 0.705 +- 0.038 |
| 4 encoder pretrained | 0.787 +- 0.134 | 0.983 +- 0.013 | 0.998 +- 0.002 |
| 4b encoder pretrained + synth | 0.915 +- 0.018 | 0.970 +- 0.031 | 0.993 +- 0.007 |
| 5 shared LoRA | 0.000 +- 0.000 | 0.330 +- 0.315 | 0.927 +- 0.039 |
| 5c LoRA from base | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.733 +- 0.191 |
| 5b shared LoRA + synth | 0.837 +- 0.087 | 0.895 +- 0.017 | 0.973 +- 0.016 |
| 5d LoRA from base + synth | 0.386 +- 0.358 | 0.250 +- 0.432 | 0.851 +- 0.156 |

### Per task bank_us: accuracy / coverage / realized error on answered rows (n certified of 3 draws), mean over draws

| arm | k=1 | k=3 | k=10 |
|---|---|---|---|
| 1 baseline | 0.335 / 0.000 / - (0) | 0.443 / 0.000 / - (0) | 0.573 / 0.000 / - (0) |
| 2 synthetic prior | 0.510 / 0.000 / - (0) | 0.560 / 0.000 / - (0) | 0.627 / 0.000 / - (0) |
| 3 hypernetwork | 0.400 / 0.000 / - (0) | 0.462 / 0.000 / - (0) | 0.468 / 0.000 / - (0) |
| 4 encoder pretrained | 0.340 / 0.000 / - (0) | 0.443 / 0.000 / - (0) | 0.571 / 0.000 / - (0) |
| 4b encoder pretrained + synth | 0.481 / 0.000 / - (0) | 0.542 / 0.000 / - (0) | 0.608 / 0.000 / - (0) |
| 5 shared LoRA | 0.194 / 0.000 / - (0) | 0.299 / 0.000 / - (0) | 0.468 / 0.000 / - (0) |
| 5c LoRA from base | 0.133 / 0.000 / - (0) | 0.244 / 0.000 / - (0) | 0.374 / 0.000 / - (0) |
| 5b shared LoRA + synth | 0.441 / 0.000 / - (0) | 0.506 / 0.000 / - (0) | 0.557 / 0.000 / - (0) |
| 5d LoRA from base + synth | 0.361 / 0.000 / - (0) | 0.371 / 0.000 / - (0) | 0.445 / 0.000 / - (0) |

### Per task amazon: accuracy / coverage / realized error on answered rows (n certified of 3 draws), mean over draws

| arm | k=1 | k=3 | k=10 |
|---|---|---|---|
| 1 baseline | 0.335 / 0.000 / - (0) | 0.469 / 0.000 / - (0) | 0.607 / 0.000 / - (0) |
| 2 synthetic prior | 0.573 / 0.000 / - (0) | 0.583 / 0.000 / - (0) | 0.646 / 0.000 / - (0) |
| 3 hypernetwork | 0.483 / 0.000 / - (0) | 0.518 / 0.000 / - (0) | 0.530 / 0.000 / - (0) |
| 4 encoder pretrained | 0.405 / 0.000 / - (0) | 0.537 / 0.000 / - (0) | 0.648 / 0.000 / - (0) |
| 4b encoder pretrained + synth | 0.553 / 0.000 / - (0) | 0.575 / 0.000 / - (0) | 0.640 / 0.000 / - (0) |
| 5 shared LoRA | 0.211 / 0.000 / - (0) | 0.339 / 0.000 / - (0) | 0.507 / 0.000 / - (0) |
| 5c LoRA from base | 0.158 / 0.000 / - (0) | 0.280 / 0.000 / - (0) | 0.386 / 0.000 / - (0) |
| 5b shared LoRA + synth | 0.515 / 0.000 / - (0) | 0.551 / 0.000 / - (0) | 0.606 / 0.000 / - (0) |
| 5d LoRA from base + synth | 0.431 / 0.000 / - (0) | 0.478 / 0.000 / - (0) | 0.495 / 0.000 / - (0) |

### Per task clinc_travel: accuracy / coverage / realized error on answered rows (n certified of 3 draws), mean over draws

| arm | k=1 | k=3 | k=10 |
|---|---|---|---|
| 1 baseline | 0.870 / 0.401 / 0.007 (2) | 0.975 / 0.985 / 0.019 (3) | 0.993 / 0.996 / 0.006 (3) |
| 2 synthetic prior | 0.967 / 0.954 / 0.007 (3) | 0.984 / 0.990 / 0.011 (3) | 0.998 / 0.996 / 0.000 (3) |
| 3 hypernetwork | 0.839 / 0.209 / 0.011 (1) | 0.866 / 0.648 / 0.014 (3) | 0.878 / 0.705 / 0.017 (3) |
| 4 encoder pretrained | 0.925 / 0.787 / 0.004 (3) | 0.981 / 0.983 / 0.010 (3) | 0.987 / 0.998 / 0.012 (3) |
| 4b encoder pretrained + synth | 0.957 / 0.915 / 0.005 (3) | 0.976 / 0.970 / 0.007 (3) | 0.991 / 0.993 / 0.006 (3) |
| 5 shared LoRA | 0.634 / 0.000 / - (0) | 0.822 / 0.330 / 0.004 (2) | 0.961 / 0.927 / 0.009 (3) |
| 5c LoRA from base | 0.442 / 0.000 / - (0) | 0.651 / 0.000 / - (0) | 0.920 / 0.733 / 0.010 (3) |
| 5b shared LoRA + synth | 0.925 / 0.837 / 0.012 (3) | 0.948 / 0.895 / 0.021 (3) | 0.979 / 0.973 / 0.008 (3) |
| 5d LoRA from base + synth | 0.829 / 0.386 / 0.003 (2) | 0.835 / 0.250 / 0.027 (1) | 0.960 / 0.851 / 0.004 (3) |

### Share of cells (task x k x draw, 27 per arm) where the 5% certificate succeeded at all, and worst realized error among certified cells

| arm | certified cells | of which on clinc_travel | max realized error |
|---|---|---|---|
| 1 baseline | 8 / 27 | 8 | 0.034 |
| 2 synthetic prior | 9 / 27 | 9 | 0.020 |
| 3 hypernetwork | 7 / 27 | 7 | 0.027 |
| 4 encoder pretrained | 9 / 27 | 9 | 0.016 |
| 4b encoder pretrained + synth | 9 / 27 | 9 | 0.013 |
| 5 shared LoRA | 5 / 27 | 5 | 0.014 |
| 5c LoRA from base | 3 / 27 | 3 | 0.017 |
| 5b shared LoRA + synth | 9 / 27 | 9 | 0.038 |
| 5d LoRA from base + synth | 6 / 27 | 6 | 0.027 |

### Paired differences in accuracy (A - B, same task / k / draw), all 3 tasks (9 cells per k): mean +- s.d. over cells [A better / A worse]

| A - B | k=1 | k=3 | k=10 |
|---|---|---|---|
| 4 encoder pretrained - 1 baseline | +0.043 +- 0.038 [8/1] | +0.025 +- 0.036 [6/3] | +0.012 +- 0.024 [4/5] |
| 4b encoder pretrained + synth - 2 synthetic prior | -0.020 +- 0.011 [0/9] | -0.012 +- 0.010 [1/7] | -0.011 +- 0.010 [0/9] |
| 5 shared LoRA - 5c LoRA from base | +0.102 +- 0.094 [7/1] | +0.095 +- 0.070 [8/1] | +0.085 +- 0.067 [8/1] |
| 5b shared LoRA + synth - 5d LoRA from base + synth | +0.086 +- 0.042 [9/0] | +0.107 +- 0.065 [9/0] | +0.081 +- 0.064 [9/0] |
| 2 synthetic prior - 1 baseline | +0.170 +- 0.066 [9/0] | +0.080 +- 0.054 [8/1] | +0.033 +- 0.023 [9/0] |
| 3 hypernetwork - 1 baseline | +0.061 +- 0.089 [7/2] | -0.013 +- 0.077 [5/4] | -0.099 +- 0.024 [0/9] |
| 3 hypernetwork - 2 synthetic prior | -0.109 +- 0.044 [0/9] | -0.093 +- 0.034 [0/9] | -0.132 +- 0.023 [0/9] |
| 4 encoder pretrained - 5 shared LoRA | +0.210 +- 0.076 [9/0] | +0.167 +- 0.053 [9/0] | +0.090 +- 0.056 [9/0] |
| 4b encoder pretrained + synth - 5b shared LoRA + synth | +0.037 +- 0.024 [8/1] | +0.029 +- 0.018 [8/0] | +0.032 +- 0.028 [8/0] |
| 1 baseline - 5c LoRA from base | +0.269 +- 0.138 [9/0] | +0.237 +- 0.084 [9/0] | +0.164 +- 0.083 [9/0] |
| 2 synthetic prior - 5d LoRA from base + synth | +0.143 +- 0.055 [9/0] | +0.148 +- 0.067 [9/0] | +0.124 +- 0.079 [9/0] |

### Paired differences in accuracy (A - B, same task / k / draw), bank_us + amazon (6 cells per k): mean +- s.d. over cells [A better / A worse]

| A - B | k=1 | k=3 | k=10 |
|---|---|---|---|
| 4 encoder pretrained - 1 baseline | +0.037 +- 0.046 [5/1] | +0.034 +- 0.042 [4/2] | +0.020 +- 0.026 [4/2] |
| 4b encoder pretrained + synth - 2 synthetic prior | -0.024 +- 0.010 [0/6] | -0.013 +- 0.011 [1/5] | -0.013 +- 0.012 [0/6] |
| 5 shared LoRA - 5c LoRA from base | +0.057 +- 0.050 [4/1] | +0.057 +- 0.037 [5/1] | +0.107 +- 0.067 [6/0] |
| 5b shared LoRA + synth - 5d LoRA from base + synth | +0.081 +- 0.032 [6/0] | +0.104 +- 0.061 [6/0] | +0.112 +- 0.054 [6/0] |
| 2 synthetic prior - 1 baseline | +0.206 +- 0.043 [6/0] | +0.116 +- 0.012 [6/0] | +0.047 +- 0.013 [6/0] |
| 3 hypernetwork - 1 baseline | +0.106 +- 0.057 [6/0] | +0.035 +- 0.033 [5/1] | -0.091 +- 0.026 [0/6] |
| 3 hypernetwork - 2 synthetic prior | -0.100 +- 0.047 [0/6] | -0.081 +- 0.034 [0/6] | -0.138 +- 0.027 [0/6] |
| 4 encoder pretrained - 5 shared LoRA | +0.170 +- 0.042 [6/0] | +0.171 +- 0.058 [6/0] | +0.122 +- 0.034 [6/0] |
| 4b encoder pretrained + synth - 5b shared LoRA + synth | +0.040 +- 0.021 [6/0] | +0.030 +- 0.020 [5/0] | +0.042 +- 0.029 [6/0] |
| 1 baseline - 5c LoRA from base | +0.190 +- 0.040 [6/0] | +0.194 +- 0.056 [6/0] | +0.209 +- 0.046 [6/0] |
| 2 synthetic prior - 5d LoRA from base + synth | +0.145 +- 0.041 [6/0] | +0.147 +- 0.071 [6/0] | +0.166 +- 0.055 [6/0] |

### Paired differences in accuracy (A - B, same task / k / draw), clinc_travel (3 cells per k): mean +- s.d. over cells [A better / A worse]

| A - B | k=1 | k=3 | k=10 |
|---|---|---|---|
| 4 encoder pretrained - 1 baseline | +0.056 +- 0.016 [3/0] | +0.007 +- 0.011 [2/1] | -0.005 +- 0.003 [0/3] |
| 4b encoder pretrained + synth - 2 synthetic prior | -0.010 +- 0.008 [0/3] | -0.008 +- 0.007 [0/2] | -0.007 +- 0.000 [0/3] |
| 5 shared LoRA - 5c LoRA from base | +0.192 +- 0.104 [3/0] | +0.171 +- 0.057 [3/0] | +0.041 +- 0.050 [2/1] |
| 5b shared LoRA + synth - 5d LoRA from base + synth | +0.096 +- 0.064 [3/0] | +0.113 +- 0.086 [3/0] | +0.019 +- 0.018 [3/0] |
| 2 synthetic prior - 1 baseline | +0.098 +- 0.034 [3/0] | +0.009 +- 0.010 [2/1] | +0.005 +- 0.005 [3/0] |
| 3 hypernetwork - 1 baseline | -0.031 +- 0.069 [1/2] | -0.109 +- 0.010 [0/3] | -0.115 +- 0.010 [0/3] |
| 3 hypernetwork - 2 synthetic prior | -0.129 +- 0.036 [0/3] | -0.118 +- 0.019 [0/3] | -0.120 +- 0.009 [0/3] |
| 4 encoder pretrained - 5 shared LoRA | +0.291 +- 0.064 [3/0] | +0.159 +- 0.052 [3/0] | +0.026 +- 0.019 [3/0] |
| 4b encoder pretrained + synth - 5b shared LoRA + synth | +0.032 +- 0.034 [2/1] | +0.027 +- 0.018 [3/0] | +0.012 +- 0.011 [2/0] |
| 1 baseline - 5c LoRA from base | +0.427 +- 0.125 [3/0] | +0.324 +- 0.058 [3/0] | +0.073 +- 0.057 [3/0] |
| 2 synthetic prior - 5d LoRA from base + synth | +0.139 +- 0.090 [3/0] | +0.149 +- 0.073 [3/0] | +0.038 +- 0.027 [3/0] |

### Paired differences in certified coverage (A - B, same task / k / draw), all 3 tasks (9 cells per k): mean +- s.d. over cells [A better / A worse]

| A - B | k=1 | k=3 | k=10 |
|---|---|---|---|
| 4 encoder pretrained - 1 baseline | +0.129 +- 0.223 [3/0] | -0.001 +- 0.011 [1/2] | +0.001 +- 0.005 [1/2] |
| 4b encoder pretrained + synth - 2 synthetic prior | -0.013 +- 0.020 [0/3] | -0.007 +- 0.017 [1/2] | -0.001 +- 0.004 [1/2] |
| 5 shared LoRA - 5c LoRA from base | +0.000 +- 0.000 [0/0] | +0.110 +- 0.228 [2/0] | +0.065 +- 0.125 [3/0] |
| 5b shared LoRA + synth - 5d LoRA from base + synth | +0.150 +- 0.263 [3/0] | +0.215 +- 0.393 [3/0] | +0.040 +- 0.096 [3/0] |
| 2 synthetic prior - 1 baseline | +0.184 +- 0.324 [3/0] | +0.002 +- 0.008 [1/2] | +0.000 +- 0.004 [1/2] |
| 3 hypernetwork - 1 baseline | -0.064 +- 0.368 [1/2] | -0.112 +- 0.185 [0/3] | -0.097 +- 0.146 [0/3] |
| 3 hypernetwork - 2 synthetic prior | -0.248 +- 0.416 [0/3] | -0.114 +- 0.186 [0/3] | -0.097 +- 0.147 [0/3] |
| 4 encoder pretrained - 5 shared LoRA | +0.262 +- 0.399 [3/0] | +0.218 +- 0.363 [3/0] | +0.024 +- 0.041 [3/0] |
| 4b encoder pretrained + synth - 5b shared LoRA + synth | +0.026 +- 0.054 [3/0] | +0.025 +- 0.038 [3/0] | +0.007 +- 0.015 [2/1] |
| 1 baseline - 5c LoRA from base | +0.134 +- 0.266 [2/0] | +0.328 +- 0.493 [3/0] | +0.088 +- 0.164 [3/0] |
| 2 synthetic prior - 5d LoRA from base + synth | +0.189 +- 0.333 [3/0] | +0.247 +- 0.431 [3/0] | +0.048 +- 0.107 [3/0] |

### Paired differences in certified coverage (A - B, same task / k / draw), clinc_travel (3 cells per k): mean +- s.d. over cells [A better / A worse]

| A - B | k=1 | k=3 | k=10 |
|---|---|---|---|
| 4 encoder pretrained - 1 baseline | +0.386 +- 0.224 [3/0] | -0.002 +- 0.023 [1/2] | +0.002 +- 0.010 [1/2] |
| 4b encoder pretrained + synth - 2 synthetic prior | -0.039 +- 0.006 [0/3] | -0.021 +- 0.027 [1/2] | -0.003 +- 0.007 [1/2] |
| 5 shared LoRA - 5c LoRA from base | +0.000 +- 0.000 [0/0] | +0.330 +- 0.315 [2/0] | +0.194 +- 0.157 [3/0] |
| 5b shared LoRA + synth - 5d LoRA from base + synth | +0.451 +- 0.271 [3/0] | +0.645 +- 0.449 [3/0] | +0.121 +- 0.149 [3/0] |
| 2 synthetic prior - 1 baseline | +0.553 +- 0.340 [3/0] | +0.005 +- 0.015 [1/2] | +0.001 +- 0.007 [1/2] |
| 3 hypernetwork - 1 baseline | -0.193 +- 0.710 [1/2] | -0.337 +- 0.151 [0/3] | -0.290 +- 0.033 [0/3] |
| 3 hypernetwork - 2 synthetic prior | -0.745 +- 0.370 [0/3] | -0.342 +- 0.144 [0/3] | -0.291 +- 0.037 [0/3] |
| 4 encoder pretrained - 5 shared LoRA | +0.787 +- 0.134 [3/0] | +0.653 +- 0.316 [3/0] | +0.071 +- 0.040 [3/0] |
| 4b encoder pretrained + synth - 5b shared LoRA + synth | +0.078 +- 0.074 [3/0] | +0.075 +- 0.016 [3/0] | +0.021 +- 0.022 [2/1] |
| 1 baseline - 5c LoRA from base | +0.401 +- 0.348 [2/0] | +0.985 +- 0.010 [3/0] | +0.263 +- 0.196 [3/0] |
| 2 synthetic prior - 5d LoRA from base + synth | +0.568 +- 0.347 [3/0] | +0.741 +- 0.439 [3/0] | +0.145 +- 0.157 [3/0] |

### Per-draw transfer gain in accuracy, decoder (5 - 5c), by task and k, mean over draws

| task | k=1 | k=3 | k=10 |
|---|---|---|---|
| bank_us | +0.060 | +0.055 | +0.093 |
| amazon | +0.053 | +0.059 | +0.120 |
| clinc_travel | +0.192 | +0.171 | +0.041 |

### Encoder transfer (4 - 1) accuracy by task and k, mean over draws

| task | k=1 | k=3 | k=10 |
|---|---|---|---|
| bank_us | +0.005 | +0.000 | -0.002 |
| amazon | +0.069 | +0.067 | +0.042 |
| clinc_travel | +0.056 | +0.007 | -0.005 |

### Seconds per run (mean)

| arm | sec per cell |
|---|---|
| 1 baseline | 0.05 |
| 2 synthetic prior | 0.02 |
| 3 hypernetwork | 0.03 |
| 4 encoder pretrained | 0.04 |
| 4b encoder pretrained + synth | 0.02 |
| 5 shared LoRA | 11.86 |
| 5c LoRA from base | 11.59 |
| 5b shared LoRA + synth | 11.86 |
| 5d LoRA from base + synth | 11.86 |

### Certificate arithmetic (Clopper-Pearson one-sided, delta 0.05 / 40 grid levels, budget 5%): fewest answered calibration rows needed

| errors among answered cal rows | min rows answered | share of the 300 | min accuracy of the answered rows |
|---|---|---|---|
| 0 | 131 | 0.44 | 1.000 |
| 1 | 176 | 0.59 | 0.994 |
| 2 | 215 | 0.72 | 0.991 |
| 3 | 251 | 0.84 | 0.988 |
| 4 | 285 | 0.95 | 0.986 |
| 5 | 318 | 1.06 | 0.984 |
| 6 | 349 | 1.16 | 0.983 |

(The table above ignores the grid: it is the fewest rows at the single level delta/40. `common.certify` only tests levels round(n*j/40), so the real minimum is the first grid level that passes, below.)

### Certificate arithmetic with the 40-level grid actually used by common.certify (first passing grid level, zero-padded to the top e errors)

| errors | n_cal=300: level (rows) | share | n_cal=1000: level (rows) | share |
|---|---|---|---|---|
| 0 | 135 | 0.450 | 150 | 0.150 |
| 1 | 180 | 0.600 | 200 | 0.200 |
| 2 | 218 | 0.727 | 225 | 0.225 |
| 3 | 255 | 0.850 | 275 | 0.275 |
| 4 | 285 | 0.950 | 300 | 0.300 |
| 5 | None | - | 325 | - |
| 6 | None | - | 350 | - |

### Overall accuracy versus certification (what decides it is the top-2-gap ranking, not overall accuracy)

- lowest overall accuracy among the 65 certified cells: 0.829; highest overall accuracy among the 178 uncertified cells: 0.887
- highest accuracy on bank_us / amazon (any cell): 0.659; certified cells with accuracy below 0.90: 13

| travel cell (arm, k) | mean accuracy | draws certified (of 3) | mean coverage |
|---|---|---|---|
| 5c LoRA from base, k=1 | 0.442 | 0 | 0.000 |
| 5 shared LoRA, k=1 | 0.634 | 0 | 0.000 |
| 5c LoRA from base, k=3 | 0.651 | 0 | 0.000 |
| 5 shared LoRA, k=3 | 0.822 | 2 | 0.330 |
| 5d LoRA from base + synth, k=1 | 0.829 | 2 | 0.386 |
| 5d LoRA from base + synth, k=3 | 0.835 | 1 | 0.250 |
| 3 hypernetwork, k=1 | 0.839 | 1 | 0.209 |
| 3 hypernetwork, k=3 | 0.866 | 3 | 0.648 |
| 1 baseline, k=1 | 0.870 | 2 | 0.401 |
| 3 hypernetwork, k=10 | 0.878 | 3 | 0.705 |

Travel cells with accuracy >= 0.84 that did not certify:

- 3 hypernetwork, k=1, seed 0: accuracy 0.840
- 1 baseline, k=1, seed 1: accuracy 0.840
- 5d LoRA from base + synth, k=3, seed 0: accuracy 0.887

### Compact per-task tables (mean over 3 draws)

Accuracy

| arm | bank_us k=1 | bank_us k=3 | bank_us k=10 | amazon k=1 | amazon k=3 | amazon k=10 | clinc_travel k=1 | clinc_travel k=3 | clinc_travel k=10 |
|---|---|---|---|---|---|---|---|---|---|
| 1 baseline | 0.335 | 0.443 | 0.573 | 0.335 | 0.469 | 0.607 | 0.870 | 0.975 | 0.993 |
| 2 synthetic prior | 0.510 | 0.560 | 0.627 | 0.573 | 0.583 | 0.646 | 0.967 | 0.984 | 0.998 |
| 3 hypernetwork | 0.400 | 0.462 | 0.468 | 0.483 | 0.518 | 0.530 | 0.839 | 0.866 | 0.878 |
| 4 encoder pretrained | 0.340 | 0.443 | 0.571 | 0.405 | 0.537 | 0.648 | 0.925 | 0.981 | 0.987 |
| 4b encoder pretrained + synth | 0.481 | 0.542 | 0.608 | 0.553 | 0.575 | 0.640 | 0.957 | 0.976 | 0.991 |
| 5 shared LoRA | 0.194 | 0.299 | 0.468 | 0.211 | 0.339 | 0.507 | 0.634 | 0.822 | 0.961 |
| 5c LoRA from base | 0.133 | 0.244 | 0.374 | 0.158 | 0.280 | 0.386 | 0.442 | 0.651 | 0.920 |
| 5b shared LoRA + synth | 0.441 | 0.506 | 0.557 | 0.515 | 0.551 | 0.606 | 0.925 | 0.948 | 0.979 |
| 5d LoRA from base + synth | 0.361 | 0.371 | 0.445 | 0.431 | 0.478 | 0.495 | 0.829 | 0.835 | 0.960 |

Certified coverage at 5%

| arm | bank_us k=1 | bank_us k=3 | bank_us k=10 | amazon k=1 | amazon k=3 | amazon k=10 | clinc_travel k=1 | clinc_travel k=3 | clinc_travel k=10 |
|---|---|---|---|---|---|---|---|---|---|
| 1 baseline | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.401 | 0.985 | 0.996 |
| 2 synthetic prior | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.954 | 0.990 | 0.996 |
| 3 hypernetwork | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.209 | 0.648 | 0.705 |
| 4 encoder pretrained | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.787 | 0.983 | 0.998 |
| 4b encoder pretrained + synth | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.915 | 0.970 | 0.993 |
| 5 shared LoRA | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.330 | 0.927 |
| 5c LoRA from base | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.733 |
| 5b shared LoRA + synth | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.837 | 0.895 | 0.973 |
| 5d LoRA from base + synth | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.386 | 0.250 | 0.851 |

