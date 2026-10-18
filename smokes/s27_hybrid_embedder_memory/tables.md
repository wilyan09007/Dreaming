### Main table (mean over 5 tasks x 3 draws; +- = s.d. over the 3 draws of the task-averaged value)

| arm | acc k=1 | acc k=3 | acc k=10 | cert. cov k=1 | cov k=3 | cov k=10 |
|---|---|---|---|---|---|---|
| LR bge-small (baseline) | 0.811 +- 0.026 | 0.925 +- 0.001 | 0.962 +- 0.004 | 0.128 +- 0.035 | 0.635 +- 0.055 | 0.869 +- 0.015 |
| LR LFM2.5-350M | 0.793 +- 0.034 | 0.916 +- 0.006 | 0.962 +- 0.005 | 0.072 +- 0.125 | 0.589 +- 0.076 | 0.869 +- 0.025 |
| LR bge-base | 0.833 +- 0.015 | 0.934 +- 0.003 | 0.969 +- 0.005 | 0.179 +- 0.079 | 0.778 +- 0.004 | 0.911 +- 0.028 |
| delta-rule head (bge-small) | 0.801 +- 0.026 | 0.907 +- 0.006 | 0.955 +- 0.004 | 0.135 +- 0.129 | 0.519 +- 0.085 | 0.831 +- 0.030 |
| ridge least squares (bge-small) | 0.807 +- 0.027 | 0.917 +- 0.004 | 0.959 +- 0.004 | 0.156 +- 0.083 | 0.614 +- 0.111 | 0.809 +- 0.020 |
| Hopfield read (bge-small) | 0.799 +- 0.026 | 0.913 +- 0.003 | 0.953 +- 0.002 | 0.072 +- 0.069 | 0.656 +- 0.016 | 0.778 +- 0.103 |

### Paired difference to the baseline (same task, k, draw; mean +- standard error over 15 cells)

| arm | d-cov k=1 | k=3 | k=10 | d-acc k=1 | k=3 | k=10 |
|---|---|---|---|---|---|---|
| LR LFM2.5-350M | -0.056 +- 0.065 | -0.046 +- 0.049 | +0.000 +- 0.019 | -0.018 +- 0.013 | -0.009 +- 0.005 | -0.001 +- 0.003 |
| LR bge-base | +0.051 +- 0.033 | +0.143 +- 0.045 | +0.042 +- 0.015 | +0.022 +- 0.006 | +0.010 +- 0.002 | +0.006 +- 0.003 |
| delta-rule head (bge-small) | +0.007 +- 0.048 | -0.116 +- 0.074 | -0.038 +- 0.034 | -0.010 +- 0.005 | -0.018 +- 0.005 | -0.008 +- 0.002 |
| ridge least squares (bge-small) | +0.028 +- 0.035 | -0.021 +- 0.032 | -0.060 +- 0.029 | -0.004 +- 0.003 | -0.008 +- 0.004 | -0.004 +- 0.002 |
| Hopfield read (bge-small) | -0.056 +- 0.035 | +0.020 +- 0.057 | -0.091 +- 0.070 | -0.012 +- 0.003 | -0.012 +- 0.004 | -0.009 +- 0.002 |

### Realized error above the cutoff on the test rows (mean over cells that certified something; budget 0.05) and cells with nothing certified

| arm | err k=1 | err k=3 | err k=10 | zero-coverage cells k=1 / 3 / 10 (of 15) |
|---|---|---|---|---|
| LR bge-small (baseline) | 0.005 | 0.011 | 0.008 | 12 / 2 / 0 |
| LR LFM2.5-350M | 0.024 | 0.015 | 0.009 | 13 / 3 / 0 |
| LR bge-base | 0.006 | 0.011 | 0.010 | 11 / 0 / 0 |
| delta-rule head (bge-small) | 0.016 | 0.009 | 0.013 | 12 / 5 / 0 |
| ridge least squares (bge-small) | 0.008 | 0.013 | 0.009 | 11 / 3 / 0 |
| Hopfield read (bge-small) | 0.003 | 0.011 | 0.012 | 13 / 0 / 2 |

### Cost: fit time per cell (s, mean over the 15 cells at each k) and parameters

| arm | fit s k=1 | k=3 | k=10 | trained params (head) | frozen encoder params (M) | embed ms per text (CPU) |
|---|---|---|---|---|---|---|
| LR bge-small (baseline) | 0.061 | 0.075 | 0.105 | 5775 | 33 | 4.8 |
| LR LFM2.5-350M | 0.087 | 0.094 | 0.915 | 15375 | 354 | 151.4 |
| LR bge-base | 0.071 | 0.083 | 0.660 | 11535 | 109 | 17.3 |
| delta-rule head (bge-small) | 0.107 | 0.209 | 0.627 | 5775 | 33 | 4.8 |
| ridge least squares (bge-small) | 0.029 | 0.025 | 0.027 | 5775 | 33 | 4.8 |
| Hopfield read (bge-small) | 0.002 | 0.002 | 0.005 | 1 | 33 | 4.8 |

### Per-task accuracy / coverage at k=1, 3, 10

| arm | task | acc k=1 | k=3 | k=10 | cov k=1 | k=3 | k=10 |
|---|---|---|---|---|---|---|---|
| LR bge-small (baseline) | clinc/travel | 0.875 | 0.972 | 0.980 | 0.495 | 0.963 | 0.981 |
| LR bge-small (baseline) | clinc/work | 0.767 | 0.896 | 0.946 | 0.000 | 0.489 | 0.752 |
| LR bge-small (baseline) | banking/sub0 | 0.794 | 0.897 | 0.956 | 0.000 | 0.447 | 0.813 |
| LR bge-small (baseline) | banking/sub1 | 0.792 | 0.917 | 0.967 | 0.147 | 0.803 | 0.969 |
| LR bge-small (baseline) | banking/sub2 | 0.825 | 0.941 | 0.962 | 0.000 | 0.473 | 0.830 |
| LR LFM2.5-350M | clinc/travel | 0.859 | 0.973 | 0.984 | 0.227 | 0.919 | 0.990 |
| LR LFM2.5-350M | clinc/work | 0.793 | 0.907 | 0.950 | 0.136 | 0.573 | 0.871 |
| LR LFM2.5-350M | banking/sub0 | 0.747 | 0.874 | 0.949 | 0.000 | 0.228 | 0.766 |
| LR LFM2.5-350M | banking/sub1 | 0.783 | 0.898 | 0.963 | 0.000 | 0.803 | 0.951 |
| LR LFM2.5-350M | banking/sub2 | 0.781 | 0.927 | 0.962 | 0.000 | 0.424 | 0.768 |
| LR bge-base | clinc/travel | 0.886 | 0.979 | 0.982 | 0.550 | 0.981 | 0.996 |
| LR bge-base | clinc/work | 0.796 | 0.908 | 0.961 | 0.000 | 0.639 | 0.831 |
| LR bge-base | banking/sub0 | 0.799 | 0.912 | 0.959 | 0.000 | 0.706 | 0.899 |
| LR bge-base | banking/sub1 | 0.833 | 0.928 | 0.967 | 0.345 | 0.878 | 0.965 |
| LR bge-base | banking/sub2 | 0.850 | 0.943 | 0.974 | 0.000 | 0.686 | 0.864 |
| delta-rule head (bge-small) | clinc/travel | 0.876 | 0.976 | 0.977 | 0.498 | 0.957 | 0.995 |
| delta-rule head (bge-small) | clinc/work | 0.763 | 0.853 | 0.934 | 0.000 | 0.000 | 0.530 |
| delta-rule head (bge-small) | banking/sub0 | 0.773 | 0.883 | 0.947 | 0.000 | 0.587 | 0.807 |
| delta-rule head (bge-small) | banking/sub1 | 0.788 | 0.901 | 0.960 | 0.178 | 0.762 | 0.976 |
| delta-rule head (bge-small) | banking/sub2 | 0.804 | 0.921 | 0.955 | 0.000 | 0.289 | 0.846 |
| ridge least squares (bge-small) | clinc/travel | 0.870 | 0.970 | 0.979 | 0.465 | 0.955 | 0.992 |
| ridge least squares (bge-small) | clinc/work | 0.758 | 0.890 | 0.937 | 0.000 | 0.372 | 0.556 |
| ridge least squares (bge-small) | banking/sub0 | 0.787 | 0.885 | 0.948 | 0.000 | 0.442 | 0.746 |
| ridge least squares (bge-small) | banking/sub1 | 0.794 | 0.903 | 0.974 | 0.316 | 0.772 | 0.969 |
| ridge least squares (bge-small) | banking/sub2 | 0.827 | 0.935 | 0.955 | 0.000 | 0.532 | 0.784 |
| Hopfield read (bge-small) | clinc/travel | 0.853 | 0.950 | 0.964 | 0.362 | 0.856 | 0.933 |
| Hopfield read (bge-small) | clinc/work | 0.768 | 0.894 | 0.937 | 0.000 | 0.442 | 0.291 |
| Hopfield read (bge-small) | banking/sub0 | 0.778 | 0.899 | 0.946 | 0.000 | 0.665 | 0.864 |
| Hopfield read (bge-small) | banking/sub1 | 0.778 | 0.906 | 0.967 | 0.000 | 0.718 | 0.972 |
| Hopfield read (bge-small) | banking/sub2 | 0.818 | 0.918 | 0.951 | 0.000 | 0.596 | 0.830 |

### Chosen hyper-parameters

| arm | k | chosen values (count over 15 cells) |
|---|---|---|
| delta-rule head (bge-small) | 1 | 0.03/10: 1, 0.03/30: 7, 0.1/30: 2, 0.3/10: 1, 0.3/30: 4 |
| delta-rule head (bge-small) | 3 | 0.03/10: 1, 0.03/30: 2, 0.1/30: 8, 0.3/10: 2, 0.3/30: 2 |
| delta-rule head (bge-small) | 10 | 0.03/30: 6, 0.1/10: 1, 0.1/30: 5, 0.3/3: 1, 0.3/10: 1, 0.3/30: 1 |
| ridge least squares (bge-small) | 1 | 0.01: 1, 0.1: 5, 1: 4, 10: 5 |
| ridge least squares (bge-small) | 3 | 0.01: 1, 0.1: 7, 1: 5, 10: 2 |
| ridge least squares (bge-small) | 10 | 0.1: 3, 1: 9, 10: 3 |
| Hopfield read (bge-small) | 1 | 20: 1, 40: 14 |
| Hopfield read (bge-small) | 3 | 40: 12, 80: 3 |
| Hopfield read (bge-small) | 10 | 40: 5, 80: 10 |
