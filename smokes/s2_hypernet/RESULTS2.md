# S2b — weight generator, second pass (low-capacity generators, metric learning, synthetic examples)

LINGEN val acc 0.937
METRIC val acc 0.937
SETTR-REG val acc 0.799
generators trained in 175s; fit domains ['auto_and_commute', 'banking', 'credit_cards', 'home', 'kitchen_and_dining', 'travel'], validation domain utility, held-out ['meta', 'work', 'small_talk']
Accuracy on the test split, mean over 5 support draws; k = real examples per label. All tasks below were never used to fit a generator.

### Held-out CLINC domains (15-way)

| method | k=0 | k=1 | k=2 | k=5 | k=10 |
|---|---|---|---|---|---|
| desc-cos | 0.713 | 0.713 | 0.713 | 0.713 | 0.713 |
| name-cos | 0.765 | 0.765 | 0.765 | 0.765 | 0.765 |
| proto | - | 0.733 | 0.824 | 0.887 | 0.903 |
| desc+proto | - | 0.823 | 0.864 | 0.872 | 0.874 |
| LR k-shot | - | 0.745 | 0.853 | 0.920 | 0.941 |
| LINGEN | 0.659 | 0.805 | 0.841 | 0.861 | 0.865 |
| METRIC | 0.690 | 0.811 | 0.858 | 0.866 | 0.870 |
| SETTR-REG | 0.441 | 0.594 | 0.615 | 0.630 | 0.630 |
| SYNTH(+k real) | 0.870 | 0.889 | 0.905 | 0.925 | 0.939 |

### Banking77 (77-way, different dataset)

| method | k=0 | k=1 | k=2 | k=5 | k=10 |
|---|---|---|---|---|---|
| desc-cos | 0.664 | 0.664 | 0.664 | 0.664 | 0.664 |
| name-cos | 0.680 | 0.680 | 0.680 | 0.680 | 0.680 |
| proto | - | 0.604 | 0.680 | 0.773 | 0.812 |
| desc+proto | - | 0.706 | 0.743 | 0.763 | 0.779 |
| LR k-shot | - | 0.611 | 0.718 | 0.828 | 0.872 |
| LINGEN | 0.603 | 0.697 | 0.745 | 0.777 | 0.784 |
| METRIC | 0.637 | 0.708 | 0.746 | 0.778 | 0.786 |
| SETTR-REG | 0.300 | 0.464 | 0.481 | 0.487 | 0.493 |
| SYNTH(+k real) | 0.726 | 0.765 | 0.788 | 0.837 | 0.872 |

### Banking77 12-label sub-tasks (x5)

| method | k=0 | k=1 | k=2 | k=5 | k=10 |
|---|---|---|---|---|---|
| desc-cos | 0.885 | 0.885 | 0.885 | 0.885 | 0.885 |
| name-cos | 0.876 | 0.876 | 0.876 | 0.876 | 0.876 |
| proto | - | 0.811 | 0.884 | 0.922 | 0.937 |
| desc+proto | - | 0.896 | 0.918 | 0.931 | 0.931 |
| LR k-shot | - | 0.822 | 0.894 | 0.939 | 0.954 |
| LINGEN | 0.848 | 0.883 | 0.917 | 0.930 | 0.933 |
| METRIC | 0.871 | 0.890 | 0.918 | 0.929 | 0.931 |
| SETTR-REG | 0.643 | 0.764 | 0.779 | 0.781 | 0.781 |
| SYNTH(+k real) | 0.928 | 0.934 | 0.944 | 0.951 | 0.958 |

### Part 2 — Banking77 student on n rows with an L2 prior toward W0 (NaN bug fixed; lambda picked on a 20% validation slice)

| rows | prior | acc | cert coverage @5% | holdout err |
|---|---|---|---|---|
| 77 | zero (lam=1.0) | 0.594 | 0.00 | - |
| 77 | desc (lam=1.0) | 0.704 | 0.00 | - |
| 77 | LINGEN (lam=1.0) | 0.723 | 0.00 | - |
| 77 | SYNTH-LR (lam=1.0) | 0.735 | 0.29 | 0.009 |
| 154 | zero (lam=0.01) | 0.566 | 0.00 | - |
| 154 | desc (lam=0.01) | 0.657 | 0.19 | 0.025 |
| 154 | LINGEN (lam=0.01) | 0.651 | 0.22 | 0.035 |
| 154 | SYNTH-LR (lam=0.01) | 0.743 | 0.26 | 0.029 |
| 385 | zero (lam=0.001) | 0.756 | 0.31 | 0.017 |
| 385 | desc (lam=0.001) | 0.780 | 0.41 | 0.025 |
| 385 | LINGEN (lam=0.01) | 0.765 | 0.21 | 0.009 |
| 385 | SYNTH-LR (lam=0.001) | 0.791 | 0.51 | 0.027 |
| 770 | zero (lam=0.001) | 0.833 | 0.62 | 0.025 |
| 770 | desc (lam=0.001) | 0.844 | 0.65 | 0.024 |
| 770 | LINGEN (lam=0.01) | 0.834 | 0.55 | 0.031 |
| 770 | SYNTH-LR (lam=0.001) | 0.846 | 0.69 | 0.031 |
| 1540 | zero (lam=0.001) | 0.893 | 0.80 | 0.032 |
| 1540 | desc (lam=0.001) | 0.893 | 0.80 | 0.030 |
| 1540 | LINGEN (lam=0.001) | 0.893 | 0.80 | 0.029 |
| 1540 | SYNTH-LR (lam=0.001) | 0.897 | 0.80 | 0.033 |

### Synthetic-example student (LR on 770 synthetic rows + n real rows, Banking77)

| real rows | acc | cert coverage @5% | holdout err |
|---|---|---|---|
| 0 | 0.726 | 0.29 | 0.017 |
| 77 | 0.755 | 0.27 | 0.008 |
| 154 | 0.770 | 0.42 | 0.027 |
| 385 | 0.810 | 0.54 | 0.033 |
| 770 | 0.852 | 0.65 | 0.028 |
| 1540 | 0.891 | 0.79 | 0.036 |

Wall time 786s on CPU.
