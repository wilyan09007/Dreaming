
### Accuracy (test rows), mean over 5 tasks x 3 draws (+- s.d. over draws)

| arm | k=1 | k=3 | k=10 |
|---|---|---|---|
| 1 baseline (frozen bge-small + LR) | 0.802 +- 0.015 | 0.925 +- 0.003 | 0.968 +- 0.002 |
| 2 synthetic prior | 0.914 +- 0.005 | 0.943 +- 0.002 | 0.969 +- 0.003 |
| 3 hypernetwork (100) | 0.856 +- 0.006 | 0.882 +- 0.003 | 0.893 +- 0.006 |
| 3 hypernetwork (100, early-stop on MASSIVE) | 0.856 +- 0.016 | 0.888 +- 0.000 | 0.906 +- 0.004 |
| 3 hypernetwork (25) | 0.829 +- 0.007 | 0.841 +- 0.011 | 0.853 +- 0.004 |
| 3 hypernetwork (25, early-stop on MASSIVE) | 0.839 +- 0.008 | 0.876 +- 0.005 | 0.890 +- 0.001 |
| 3 hypernetwork (3000) | 0.854 +- 0.006 | 0.882 +- 0.002 | 0.899 +- 0.002 |
| 3 hypernetwork (3000, early-stop on MASSIVE) | 0.848 +- 0.011 | 0.896 +- 0.004 | 0.909 +- 0.004 |
| 3 hypernetwork (800) | 0.861 +- 0.006 | 0.890 +- 0.003 | 0.898 +- 0.002 |
| 3 hypernetwork (800, early-stop on MASSIVE) | 0.872 +- 0.010 | 0.898 +- 0.001 | 0.914 +- 0.004 |
| 3 hypernetwork (clinc7) | 0.783 +- 0.031 | 0.806 +- 0.013 | 0.821 +- 0.004 |
| 4 encoder pretrained on pool (all) | 0.795 +- 0.014 | 0.905 +- 0.005 | 0.957 +- 0.005 |
| 4 encoder pretrained on pool (nosib) | 0.736 +- 0.006 | 0.877 +- 0.006 | 0.941 +- 0.003 |
| 4b encoder + synthetic prior | 0.906 +- 0.003 | 0.930 +- 0.001 | 0.959 +- 0.002 |
| 5 Reptile start | 0.873 +- 0.009 | 0.919 +- 0.006 | 0.955 +- 0.007 |
| 5 Reptile start (gentle, eps 0.05) | 0.885 +- 0.017 | 0.937 +- 0.010 | 0.966 +- 0.001 |
| 5c control: identity start | 0.889 +- 0.015 | 0.937 +- 0.011 | 0.967 +- 0.002 |
| 5c control: identity start (gentle, eps 0.05) | 0.885 +- 0.016 | 0.937 +- 0.012 | 0.967 +- 0.002 |
| 6 active selection (baseline student) | 0.824 +- 0.008 | 0.814 +- 0.020 | 0.958 +- 0.011 |
| 6b active selection (synthetic-prior student) | 0.921 +- 0.002 | 0.962 +- 0.003 | 0.978 +- 0.002 |

### Certified coverage at 5% budget (share of test rows answered by the student), mean over 5 tasks x 3 draws (+- s.d. over draws)

| arm | k=1 | k=3 | k=10 |
|---|---|---|---|
| 1 baseline (frozen bge-small + LR) | 0.143 +- 0.063 | 0.597 +- 0.101 | 0.906 +- 0.040 |
| 2 synthetic prior | 0.798 +- 0.012 | 0.870 +- 0.008 | 0.951 +- 0.019 |
| 3 hypernetwork (100) | 0.145 +- 0.027 | 0.428 +- 0.027 | 0.535 +- 0.061 |
| 3 hypernetwork (100, early-stop on MASSIVE) | 0.182 +- 0.080 | 0.403 +- 0.157 | 0.523 +- 0.055 |
| 3 hypernetwork (25) | 0.040 +- 0.069 | 0.171 +- 0.111 | 0.231 +- 0.015 |
| 3 hypernetwork (25, early-stop on MASSIVE) | 0.171 +- 0.148 | 0.438 +- 0.092 | 0.396 +- 0.067 |
| 3 hypernetwork (3000) | 0.351 +- 0.071 | 0.417 +- 0.034 | 0.519 +- 0.067 |
| 3 hypernetwork (3000, early-stop on MASSIVE) | 0.302 +- 0.144 | 0.556 +- 0.077 | 0.609 +- 0.027 |
| 3 hypernetwork (800) | 0.256 +- 0.023 | 0.451 +- 0.023 | 0.561 +- 0.098 |
| 3 hypernetwork (800, early-stop on MASSIVE) | 0.275 +- 0.110 | 0.473 +- 0.172 | 0.591 +- 0.015 |
| 3 hypernetwork (clinc7) | 0.031 +- 0.054 | 0.107 +- 0.021 | 0.110 +- 0.016 |
| 4 encoder pretrained on pool (all) | 0.175 +- 0.009 | 0.604 +- 0.156 | 0.886 +- 0.032 |
| 4 encoder pretrained on pool (nosib) | 0.140 +- 0.011 | 0.382 +- 0.052 | 0.756 +- 0.059 |
| 4b encoder + synthetic prior | 0.714 +- 0.058 | 0.815 +- 0.013 | 0.937 +- 0.007 |
| 5 Reptile start | 0.376 +- 0.092 | 0.644 +- 0.142 | 0.871 +- 0.009 |
| 5 Reptile start (gentle, eps 0.05) | 0.454 +- 0.020 | 0.758 +- 0.064 | 0.916 +- 0.046 |
| 5c control: identity start | 0.466 +- 0.019 | 0.767 +- 0.079 | 0.933 +- 0.028 |
| 5c control: identity start (gentle, eps 0.05) | 0.453 +- 0.025 | 0.731 +- 0.056 | 0.916 +- 0.045 |
| 6 active selection (baseline student) | 0.196 +- 0.080 | 0.166 +- 0.054 | 0.731 +- 0.199 |
| 6b active selection (synthetic-prior student) | 0.776 +- 0.062 | 0.909 +- 0.006 | 0.965 +- 0.005 |

### Realized error on the answered rows (budget 5%), mean over 5 tasks x 3 draws (+- s.d. over draws)

| arm | k=1 | k=3 | k=10 |
|---|---|---|---|
| 1 baseline (frozen bge-small + LR) | 0.007 +- 0.006 | 0.022 +- 0.010 | 0.013 +- 0.004 |
| 2 synthetic prior | 0.013 +- 0.002 | 0.017 +- 0.002 | 0.015 +- 0.004 |
| 3 hypernetwork (100) | 0.009 +- 0.009 | 0.012 +- 0.006 | 0.011 +- 0.002 |
| 3 hypernetwork (100, early-stop on MASSIVE) | 0.007 +- 0.006 | 0.009 +- 0.006 | 0.011 +- 0.003 |
| 3 hypernetwork (25) | 0.004 +- nan | 0.009 +- 0.007 | 0.006 +- 0.001 |
| 3 hypernetwork (25, early-stop on MASSIVE) | 0.014 +- 0.008 | 0.010 +- 0.006 | 0.011 +- 0.003 |
| 3 hypernetwork (3000) | 0.011 +- 0.002 | 0.013 +- 0.003 | 0.011 +- 0.002 |
| 3 hypernetwork (3000, early-stop on MASSIVE) | 0.013 +- 0.003 | 0.016 +- 0.004 | 0.018 +- 0.001 |
| 3 hypernetwork (800) | 0.010 +- 0.006 | 0.015 +- 0.004 | 0.016 +- 0.006 |
| 3 hypernetwork (800, early-stop on MASSIVE) | 0.010 +- 0.003 | 0.015 +- 0.003 | 0.011 +- 0.001 |
| 3 hypernetwork (clinc7) | 0.000 +- nan | 0.000 +- 0.000 | 0.000 +- 0.000 |
| 4 encoder pretrained on pool (all) | 0.004 +- 0.004 | 0.014 +- 0.003 | 0.013 +- 0.001 |
| 4 encoder pretrained on pool (nosib) | 0.002 +- 0.004 | 0.017 +- 0.006 | 0.014 +- 0.004 |
| 4b encoder + synthetic prior | 0.013 +- 0.005 | 0.013 +- 0.002 | 0.017 +- 0.001 |
| 5 Reptile start | 0.017 +- 0.003 | 0.023 +- 0.007 | 0.012 +- 0.002 |
| 5 Reptile start (gentle, eps 0.05) | 0.015 +- 0.003 | 0.012 +- 0.004 | 0.014 +- 0.003 |
| 5c control: identity start | 0.018 +- 0.003 | 0.015 +- 0.006 | 0.014 +- 0.003 |
| 5c control: identity start (gentle, eps 0.05) | 0.015 +- 0.003 | 0.013 +- 0.006 | 0.013 +- 0.003 |
| 6 active selection (baseline student) | 0.010 +- 0.007 | 0.013 +- 0.007 | 0.008 +- 0.002 |
| 6b active selection (synthetic-prior student) | 0.012 +- 0.005 | 0.013 +- 0.001 | 0.011 +- 0.003 |

### Certified coverage by family (CLINC 15-label tasks | Banking77 12-label tasks)

| arm | k=1 | k=3 | k=10 |
|---|---|---|---|
| 1 baseline (frozen bge-small + LR) | 0.20 \| 0.11 | 0.49 \| 0.67 | 0.90 \| 0.91 |
| 2 synthetic prior | 0.84 \| 0.77 | 0.90 \| 0.85 | 0.96 \| 0.95 |
| 3 hypernetwork (100) | 0.26 \| 0.07 | 0.42 \| 0.43 | 0.43 \| 0.61 |
| 3 hypernetwork (100, early-stop on MASSIVE) | 0.37 \| 0.06 | 0.39 \| 0.41 | 0.38 \| 0.62 |
| 3 hypernetwork (25) | 0.10 \| 0.00 | 0.31 \| 0.08 | 0.26 \| 0.21 |
| 3 hypernetwork (25, early-stop on MASSIVE) | 0.25 \| 0.12 | 0.40 \| 0.46 | 0.42 \| 0.38 |
| 3 hypernetwork (3000) | 0.40 \| 0.32 | 0.42 \| 0.42 | 0.41 \| 0.59 |
| 3 hypernetwork (3000, early-stop on MASSIVE) | 0.40 \| 0.24 | 0.43 \| 0.64 | 0.44 \| 0.72 |
| 3 hypernetwork (800) | 0.36 \| 0.18 | 0.43 \| 0.47 | 0.48 \| 0.61 |
| 3 hypernetwork (800, early-stop on MASSIVE) | 0.41 \| 0.18 | 0.51 \| 0.45 | 0.45 \| 0.69 |
| 3 hypernetwork (clinc7) | 0.08 \| 0.00 | 0.27 \| 0.00 | 0.27 \| 0.00 |
| 4 encoder pretrained on pool (all) | 0.44 \| 0.00 | 0.59 \| 0.61 | 0.89 \| 0.88 |
| 4 encoder pretrained on pool (nosib) | 0.35 \| 0.00 | 0.46 \| 0.33 | 0.80 \| 0.73 |
| 4b encoder + synthetic prior | 0.81 \| 0.65 | 0.85 \| 0.79 | 0.97 \| 0.91 |
| 5 Reptile start | 0.44 \| 0.33 | 0.57 \| 0.69 | 0.92 \| 0.84 |
| 5 Reptile start (gentle, eps 0.05) | 0.48 \| 0.44 | 0.73 \| 0.78 | 0.91 \| 0.92 |
| 5c control: identity start | 0.48 \| 0.46 | 0.71 \| 0.80 | 0.93 \| 0.93 |
| 5c control: identity start (gentle, eps 0.05) | 0.48 \| 0.44 | 0.72 \| 0.74 | 0.91 \| 0.92 |
| 6 active selection (baseline student) | 0.30 \| 0.12 | 0.33 \| 0.06 | 0.80 \| 0.68 |
| 6b active selection (synthetic-prior student) | 0.85 \| 0.73 | 0.93 \| 0.89 | 0.98 \| 0.95 |

### Real rows per label needed to reach 50% certified coverage at 5% (linear interpolation in log k between k = 1, 3, 10)

| arm | all 5 tasks | CLINC tasks | Banking77 tasks |
|---|---|---|---|
| 1 baseline (frozen bge-small + LR) | 2.4 | 3.1 | 2.2 |
| 2 synthetic prior | <=1 | <=1 | <=1 |
| 3 hypernetwork (100) | 6.7 | >10 | 4.8 |
| 3 hypernetwork (100, early-stop on MASSIVE) | 7.9 | >10 | 5.1 |
| 3 hypernetwork (25) | >10 | >10 | >10 |
| 3 hypernetwork (25, early-stop on MASSIVE) | >10 | >10 | >10 |
| 3 hypernetwork (3000) | 8.0 | >10 | 5.3 |
| 3 hypernetwork (3000, early-stop on MASSIVE) | 2.4 | >10 | 2.0 |
| 3 hypernetwork (800) | 5.1 | >10 | 3.9 |
| 3 hypernetwork (800, early-stop on MASSIVE) | 3.9 | 2.7 | 3.9 |
| 3 hypernetwork (clinc7) | >10 | >10 | >10 |
| 4 encoder pretrained on pool (all) | 2.3 | 1.6 | 2.4 |
| 4 encoder pretrained on pool (nosib) | 4.4 | 3.5 | 5.0 |
| 4b encoder + synthetic prior | <=1 | <=1 | <=1 |
| 5 Reptile start | 1.7 | 1.7 | 1.7 |
| 5 Reptile start (gentle, eps 0.05) | 1.2 | 1.1 | 1.2 |
| 5c control: identity start | 1.1 | 1.1 | 1.1 |
| 5c control: identity start (gentle, eps 0.05) | 1.2 | 1.1 | 1.3 |
| 6 active selection (baseline student) | 6.1 | 4.7 | 7.0 |
| 6b active selection (synthetic-prior student) | <=1 | <=1 | <=1 |

### Pod seconds and parameters

arm seconds (eval fit + predict, summed over all cells): {"1 baseline (frozen bge-small + LR)": 1.7, "2 synthetic prior": 0.9, "6 active selection (baseline student)": 1.5, "6b active selection (synthetic-prior student)": 3.4, "3 hypernetwork (100)": 0.1, "3 hypernetwork (25)": 0.1, "3 hypernetwork (3000)": 0.1, "3 hypernetwork (800)": 0.1, "3 hypernetwork (clinc7)": 0.1, "4 encoder pretrained on pool (all)": 1.0, "4b encoder + synthetic prior": 0.8, "4 encoder pretrained on pool (nosib)": 0.5, "5 Reptile start": 1.6, "5c control: identity start": 0.8, "3 hypernetwork (100, early-stop on MASSIVE)": 0.1, "3 hypernetwork (25, early-stop on MASSIVE)": 0.1, "3 hypernetwork (3000, early-stop on MASSIVE)": 0.1, "3 hypernetwork (800, early-stop on MASSIVE)": 0.1, "5 Reptile start (gentle, eps 0.05)": 1.5, "5c control: identity start (gentle, eps 0.05)": 0.7}
trainable parameters: {"3 hypernet": 1350530, "4 encoder": 33360000, "5 reptile": 147456}