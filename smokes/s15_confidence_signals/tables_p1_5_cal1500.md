## S15 part 1: student-side confidence signals (frozen BAAI/bge-small-en-v1.5 + LR student, C=10)
Signals (higher = more confident): a top-2 gap; b max prob; c -entropy; d top-2 gap after temperature scaling (T fit on the head-training rows); e share of the 10 nearest student-training rows (cosine) with the predicted label; f LR head on the z-scored prompt embedding predicting 'student is wrong'; g LR head on z-scored (a-e) features (+log gap, log temp-gap); h gap + kNN share (parameter-free sum).

### Banking77, 770 rows: student trained on 770 rows (19s), test accuracy 0.821 (551 errors of 3076), head-training pool 9223 unused train rows, 5 random splits (cal 1500 / holdout 1576)

**Banking77, 770 rows, heads trained on 1000 rows** (temperature fit T = 0.46 ± 0.02). AUROC = ranking errors above correct answers on the holdout; coverage = share of holdout answered by the student at the certified cutoff (certificate fit on 1500 cal rows, delta 5%); error = realized error among those answers. Cells are mean (min over splits) for coverage, mean (max) for error.
| signal | AUROC mean ± sd | cov @2% | err @2% | cov @5% | err @5% | cov @10% | err @10% |
|---|---|---|---|---|---|---|---|
| a gap | 0.878 ± 0.006 | 0.05 (0.00) | 0.005 (0.005) | 0.53 (0.45) | 0.027 (0.035) | 0.77 (0.73) | 0.069 (0.080) |
| b maxp | 0.868 ± 0.010 | 0.00 (0.00) | - | 0.47 (0.36) | 0.027 (0.038) | 0.73 (0.70) | 0.070 (0.086) |
| c -entropy | 0.818 ± 0.014 | 0.00 (0.00) | - | 0.20 (0.00) | 0.022 (0.036) | 0.63 (0.56) | 0.070 (0.090) |
| d temp-gap | 0.876 ± 0.005 | 0.09 (0.00) | 0.003 (0.003) | 0.55 (0.47) | 0.028 (0.035) | 0.76 (0.74) | 0.070 (0.080) |
| e kNN agree | 0.753 ± 0.012 | 0.00 (0.00) | - | 0.02 (0.00) | 0.022 (0.022) | 0.34 (0.23) | 0.051 (0.063) |
| f head(emb) | 0.658 ± 0.042 | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.01 (0.00) | 0.085 (0.085) |
| g head(a-e) | 0.883 ± 0.006 | 0.06 (0.00) | 0.013 (0.013) | 0.55 (0.43) | 0.026 (0.033) | 0.76 (0.73) | 0.067 (0.088) |
| h gap+kNN | 0.835 ± 0.009 | 0.00 (0.00) | - | 0.39 (0.23) | 0.025 (0.040) | 0.66 (0.57) | 0.067 (0.082) |

Paired difference vs the student's own gap (mean ± sd over 5 splits; wins = splits where the signal is better):
| signal | dAUROC | AUROC wins | boot 95% CI dAUROC (split 0) | dcov @5% | wins | dcov @10% | wins |
|---|---|---|---|---|---|---|---|
| b maxp | -0.0097 ± 0.0047 | 0/5 | [-0.0131, +0.0050] | -0.064 ± 0.034 | 0/5 | -0.033 ± 0.008 | 0/5 |
| c -entropy | -0.0595 ± 0.0091 | 0/5 | [-0.0679, -0.0302] | -0.332 ± 0.082 | 0/5 | -0.134 ± 0.027 | 0/5 |
| d temp-gap | -0.0015 ± 0.0021 | 2/5 | [-0.0087, +0.0005] | +0.021 ± 0.023 | 3/5 | -0.003 ± 0.012 | 2/5 |
| e kNN agree | -0.1247 ± 0.0079 | 0/5 | [-0.1489, -0.1021] | -0.512 ± 0.061 | 0/5 | -0.423 ± 0.043 | 0/5 |
| f head(emb) | -0.2192 ± 0.0423 | 0/5 | [-0.2437, -0.1721] | -0.529 ± 0.057 | 0/5 | -0.752 ± 0.027 | 0/5 |
| g head(a-e) | +0.0051 ± 0.0024 | 5/5 | [-0.0009, +0.0124] | +0.018 ± 0.028 | 4/5 | -0.005 ± 0.013 | 2/5 |
| h gap+kNN | -0.0423 ± 0.0045 | 0/5 | [-0.0546, -0.0314] | -0.136 ± 0.048 | 0/5 | -0.110 ± 0.025 | 0/5 |

### Banking77, 3080 rows: student trained on 3080 rows (15s), test accuracy 0.910 (276 errors of 3076), head-training pool 6913 unused train rows, 5 random splits (cal 1500 / holdout 1576)

**Banking77, 3080 rows, heads trained on 1000 rows** (temperature fit T = 0.55 ± 0.02). AUROC = ranking errors above correct answers on the holdout; coverage = share of holdout answered by the student at the certified cutoff (certificate fit on 1500 cal rows, delta 5%); error = realized error among those answers. Cells are mean (min over splits) for coverage, mean (max) for error.
| signal | AUROC mean ± sd | cov @2% | err @2% | cov @5% | err @5% | cov @10% | err @10% |
|---|---|---|---|---|---|---|---|
| a gap | 0.903 ± 0.008 | 0.33 (0.00) | 0.006 (0.009) | 0.84 (0.82) | 0.028 (0.033) | 0.97 (0.95) | 0.074 (0.080) |
| b maxp | 0.893 ± 0.009 | 0.33 (0.00) | 0.007 (0.010) | 0.82 (0.78) | 0.028 (0.043) | 0.96 (0.92) | 0.072 (0.080) |
| c -entropy | 0.850 ± 0.014 | 0.09 (0.00) | 0.009 (0.009) | 0.71 (0.66) | 0.030 (0.044) | 0.96 (0.93) | 0.077 (0.083) |
| d temp-gap | 0.905 ± 0.007 | 0.24 (0.00) | 0.007 (0.009) | 0.84 (0.81) | 0.028 (0.034) | 0.96 (0.95) | 0.074 (0.082) |
| e kNN agree | 0.887 ± 0.008 | 0.18 (0.00) | 0.007 (0.007) | 0.73 (0.70) | 0.023 (0.030) | 0.98 (0.94) | 0.076 (0.084) |
| f head(emb) | 0.612 ± 0.017 | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.71 (0.00) | 0.083 (0.090) |
| g head(a-e) | 0.914 ± 0.007 | 0.32 (0.00) | 0.007 (0.009) | 0.83 (0.81) | 0.025 (0.033) | 0.97 (0.94) | 0.072 (0.081) |
| h gap+kNN | 0.913 ± 0.006 | 0.44 (0.00) | 0.006 (0.007) | 0.82 (0.81) | 0.026 (0.033) | 0.97 (0.95) | 0.072 (0.079) |

Paired difference vs the student's own gap (mean ± sd over 5 splits; wins = splits where the signal is better):
| signal | dAUROC | AUROC wins | boot 95% CI dAUROC (split 0) | dcov @5% | wins | dcov @10% | wins |
|---|---|---|---|---|---|---|---|
| b maxp | -0.0101 ± 0.0035 | 0/5 | [-0.0140, -0.0004] | -0.024 ± 0.023 | 1/5 | -0.007 ± 0.009 | 1/5 |
| c -entropy | -0.0533 ± 0.0100 | 0/5 | [-0.0655, -0.0305] | -0.128 ± 0.033 | 0/5 | -0.002 ± 0.008 | 2/5 |
| d temp-gap | +0.0020 ± 0.0017 | 5/5 | [-0.0020, +0.0059] | -0.003 ± 0.003 | 1/5 | -0.001 ± 0.002 | 2/5 |
| e kNN agree | -0.0165 ± 0.0061 | 0/5 | [-0.0421, -0.0026] | -0.107 ± 0.035 | 0/5 | +0.011 ± 0.011 | 4/5 |
| f head(emb) | -0.2917 ± 0.0231 | 0/5 | [-0.3175, -0.2178] | -0.841 ± 0.015 | 0/5 | -0.257 ± 0.351 | 1/5 |
| g head(a-e) | +0.0108 ± 0.0026 | 5/5 | [+0.0013, +0.0131] | -0.007 ± 0.004 | 0/5 | +0.001 ± 0.006 | 2/5 |
| h gap+kNN | +0.0099 ± 0.0036 | 5/5 | [-0.0030, +0.0143] | -0.018 ± 0.011 | 0/5 | +0.004 ± 0.004 | 4/5 |

### CLINC150 (in-scope), 1500 rows: student trained on 1500 rows (12s), test accuracy 0.906 (424 errors of 4500), head-training pool 13500 unused train rows, 5 random splits (cal 1500 / holdout 3000)

**CLINC150 (in-scope), 1500 rows, heads trained on 1000 rows** (temperature fit T = 0.38 ± 0.01). AUROC = ranking errors above correct answers on the holdout; coverage = share of holdout answered by the student at the certified cutoff (certificate fit on 1500 cal rows, delta 5%); error = realized error among those answers. Cells are mean (min over splits) for coverage, mean (max) for error.
| signal | AUROC mean ± sd | cov @2% | err @2% | cov @5% | err @5% | cov @10% | err @10% |
|---|---|---|---|---|---|---|---|
| a gap | 0.891 ± 0.002 | 0.31 (0.00) | 0.004 (0.005) | 0.80 (0.79) | 0.028 (0.032) | 0.95 (0.94) | 0.071 (0.084) |
| b maxp | 0.859 ± 0.002 | 0.20 (0.00) | 0.003 (0.005) | 0.72 (0.70) | 0.031 (0.040) | 0.95 (0.94) | 0.074 (0.083) |
| c -entropy | 0.798 ± 0.004 | 0.16 (0.00) | 0.005 (0.006) | 0.52 (0.42) | 0.025 (0.036) | 0.93 (0.90) | 0.075 (0.080) |
| d temp-gap | 0.899 ± 0.003 | 0.39 (0.00) | 0.006 (0.009) | 0.81 (0.79) | 0.027 (0.034) | 0.95 (0.94) | 0.072 (0.086) |
| e kNN agree | 0.785 ± 0.007 | 0.00 (0.00) | - | 0.52 (0.49) | 0.026 (0.040) | 0.89 (0.87) | 0.069 (0.080) |
| f head(emb) | 0.646 ± 0.018 | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.60 (0.00) | 0.073 (0.080) |
| g head(a-e) | 0.894 ± 0.001 | 0.35 (0.00) | 0.006 (0.009) | 0.82 (0.79) | 0.028 (0.032) | 0.95 (0.94) | 0.070 (0.082) |
| h gap+kNN | 0.854 ± 0.004 | 0.42 (0.30) | 0.006 (0.008) | 0.72 (0.67) | 0.029 (0.036) | 0.93 (0.89) | 0.071 (0.080) |

Paired difference vs the student's own gap (mean ± sd over 5 splits; wins = splits where the signal is better):
| signal | dAUROC | AUROC wins | boot 95% CI dAUROC (split 0) | dcov @5% | wins | dcov @10% | wins |
|---|---|---|---|---|---|---|---|
| b maxp | -0.0315 ± 0.0019 | 0/5 | [-0.0430, -0.0252] | -0.077 ± 0.024 | 0/5 | +0.001 ± 0.003 | 2/5 |
| c -entropy | -0.0931 ± 0.0037 | 0/5 | [-0.1154, -0.0798] | -0.281 ± 0.063 | 0/5 | -0.025 ± 0.017 | 1/5 |
| d temp-gap | +0.0085 ± 0.0013 | 5/5 | [+0.0035, +0.0141] | +0.011 ± 0.018 | 3/5 | +0.001 ± 0.002 | 1/5 |
| e kNN agree | -0.1062 ± 0.0054 | 0/5 | [-0.1292, -0.0832] | -0.277 ± 0.046 | 0/5 | -0.057 ± 0.027 | 0/5 |
| f head(emb) | -0.2446 ± 0.0195 | 0/5 | [-0.2972, -0.2243] | -0.802 ± 0.013 | 0/5 | -0.351 ± 0.315 | 0/5 |
| g head(a-e) | +0.0034 ± 0.0007 | 5/5 | [+0.0010, +0.0037] | +0.014 ± 0.013 | 3/5 | -0.001 ± 0.002 | 1/5 |
| h gap+kNN | -0.0368 ± 0.0025 | 0/5 | [-0.0489, -0.0250] | -0.080 ± 0.038 | 0/5 | -0.023 ± 0.022 | 1/5 |

wall time 183s