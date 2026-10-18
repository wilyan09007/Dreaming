## S15 part 2: router-side confidence signals (cheap model vs gpt-5.4 low-effort 'E1', 200 Banking77 inputs from S5)

E1 = openai/gpt-5.4 effort low (S5 responses, reused, not re-run). E1 gold acc 0.870, E1b 0.895; **noise floor of the reference: E1 vs E1b agree on 0.940 of inputs (12 of 200 differ)**. E1 cost $1.719/1k calls.

### Cheap runs: accuracy, agreement with the reference, cost (every run is one call per input, provider pinned)
| run | model @ provider | temp sent | n | gold acc | match E1 | match E1b | match E1 on E1==E1b | rows with logprobs | parse fails | $/1k calls |
|---|---|---|---|---|---|---|---|---|---|---|
| nano_A | openai/gpt-5.4-nano @ OpenAI | 0 | 200 | 0.750 | 0.800 | 0.805 | 0.830 | 1.00 | 0 | 0.0979 |
| nano_Ab | openai/gpt-5.4-nano @ OpenAI | 0 | 200 | 0.755 | 0.785 | 0.810 | 0.824 | 1.00 | 0 | 0.0979 |
| nano_S | openai/gpt-5.4-nano @ OpenAI | 0.7 | 200 | 0.780 | 0.810 | 0.850 | 0.856 | 1.00 | 1 | 0.0978 |
| nano_V | openai/gpt-5.4-nano @ OpenAI | 0 | 200 | 0.735 | 0.790 | 0.805 | 0.824 | 1.00 | 1 | 0.1094 |
| nano_Vb | openai/gpt-5.4-nano @ OpenAI | 0 | 200 | 0.745 | 0.775 | 0.795 | 0.814 | 1.00 | 2 | 0.1095 |
| g41_A | openai/gpt-4.1-nano @ OpenAI | 0 | 200 | 0.625 | 0.690 | 0.685 | 0.707 | 1.00 | 9 | 0.0459 |
| g41_Ab | openai/gpt-4.1-nano @ OpenAI | 0 | 200 | 0.620 | 0.685 | 0.680 | 0.702 | 1.00 | 9 | 0.0459 |
| g41_S | openai/gpt-4.1-nano @ OpenAI | 0.7 | 200 | 0.595 | 0.660 | 0.660 | 0.681 | 1.00 | 8 | 0.0459 |
| qnx_A | qwen/qwen3-next-80b-a3b-instruct @ Parasail | 0 | 200 | 0.740 | 0.795 | 0.800 | 0.819 | 1.00 | 4 | 0.0483 |
| qnx_Ab | qwen/qwen3-next-80b-a3b-instruct @ Parasail | 0 | 200 | 0.740 | 0.795 | 0.800 | 0.819 | 1.00 | 4 | 0.0483 |
| qnx_S | qwen/qwen3-next-80b-a3b-instruct @ Parasail | 0.7 | 200 | 0.735 | 0.790 | 0.795 | 0.814 | 1.00 | 4 | 0.0483 |
| (S5) C1 nano reasoning off, no logprobs | openai/gpt-5.4-nano @ OpenAI | 0 | 200 | 0.760 | 0.815 | 0.825 | 0.846 | - | - | 0.0978 |

Noise floor, same config twice (share of inputs with the same label):
| pair | agree | differ |
|---|---|---|
| E1 vs E1b | 0.940 | 12 |
| nano_A vs nano_Ab | 0.890 | 22 |
| nano_A vs nano_S (T=0.7 sent) | 0.870 | 26 |
| nano_Ab vs nano_S | 0.895 | 21 |
| S5 C1 vs S5 C1b | 0.905 | 19 |
| nano_A vs S5 C1 | 0.880 | 24 |
| nano_V vs nano_Vb | 0.910 | 18 |
| nano_V vs nano_A | 0.865 | 27 |
| g41_A vs g41_Ab | 0.965 | 7 |
| g41_A vs g41_S (T=0.7) | 0.950 | 10 |
| g41_Ab vs g41_S (T=0.7) | 0.930 | 14 |
| qnx_A vs qnx_Ab | 0.980 | 4 |
| qnx_A vs qnx_S (T=0.7) | 0.975 | 5 |
| qnx_Ab vs qnx_S (T=0.7) | 0.965 | 7 |

Student (frozen bge-small + LR, full Banking77 train, 9993 rows) on the same 200 inputs: gold acc 0.930, agrees with E1 on 0.845, E1b 0.865.
Constants: competitor labels the API did not return are floored at logprob -12.0; returned alternatives only count when they still lead to a different listed label.

## Cheap model: gpt-5.4-nano (reasoning off, logprobs)
cheap label == E1 on 0.800 of inputs (E1b 0.805; on E1==E1b inputs 0.830; gold acc 0.750). Cost per 1000: cheap call $0.098, E1 $1.719. A perfect router could send 80% of calls to the cheap model.

**AUROC of each signal for 'cheap answer == reference'** (95% bootstrap CI over inputs, 1000 resamples, for the E1 column). A signal that uses its own call (verbalized) is scored on that call's answer. 0.5 = random; with 200 inputs a difference below about 0.08 is within the CI.
| signal | extra cheap calls before deciding | AUROC vs E1 [95% CI] | AUROC vs E1b | AUROC on E1==E1b inputs | AUROC vs gold | rows with a score |
|---|---|---|---|---|---|---|
| b1 returned-label logprob | none | 0.894 [0.849, 0.937] | 0.893 | 0.904 | 0.838 | 1.00 |
| b2 gap to best rival label | none | 0.892 [0.846, 0.935] | 0.890 | 0.901 | 0.834 | 1.00 |
| b3 first-token logprob | none | 0.869 [0.813, 0.919] | 0.867 | 0.872 | 0.798 | 1.00 |
| b4 mode-path logprob (top tokens) | none | 0.884 [0.837, 0.927] | 0.885 | 0.892 | 0.834 | 1.00 |
| b5 mode gap (top vs runner-up) | none | 0.881 [0.836, 0.923] | 0.882 | 0.888 | 0.829 | 1.00 |
| d verbalized conf (own call) | none | 0.782 [0.693, 0.862] | 0.776 | 0.782 | 0.777 | 1.00 |
| c1 self-consistency (A==S) | 1 extra | 0.684 [0.606, 0.767] | 0.738 | 0.715 | 0.667 | 1.00 |
| c2 self-consistency of 3 (A,Ab,S) | 2 extra | 0.712 [0.635, 0.792] | 0.753 | 0.736 | 0.688 | 1.00 |
| c3 2-agree, then logprob (lexi) | 1 extra | 0.895 [0.850, 0.936] | 0.898 | 0.906 | 0.839 | 1.00 |
| e student top-2 gap | none, decided before the cheap call | 0.665 [0.574, 0.748] | 0.666 | 0.680 | 0.720 | 1.00 |
| f1 student==cheap | none | 0.897 [0.834, 0.949] | 0.908 | 0.921 | 0.927 | 1.00 |
| f2 student==cheap, then student gap (lexi) | none | 0.902 [0.832, 0.954] | 0.921 | 0.931 | 0.968 | 1.00 |
| f3 student==cheap, then logprob gap (lexi) | none | 0.955 [0.923, 0.979] | 0.954 | 0.971 | 0.943 | 1.00 |
| a prompt-embedding predictor (LR) | none, decided before the cheap call | 0.583 [0.479, 0.680] | 0.536 | 0.585 | 0.486 | 1.00 |
| z all-signal LR (exploratory) | 1 extra | 0.924 [0.856, 0.968] | 0.925 | 0.936 | 0.920 | 1.00 |
| g random mixing | none, decided before the cheap call | 0.482 [0.382, 0.572] | 0.494 | 0.500 | 0.488 | 1.00 |

**Paired bootstrap of AUROC differences** (same 200 inputs, 1000 resamples; common target 'nano_A answer == E1', so the verbalized score from its own call is judged on this target too). A difference is real at this n only if the interval excludes 0.
| signal 1 minus signal 2 | AUROC diff | 95% CI |
|---|---|---|
| f3 student==cheap, then logprob gap (lexi) minus b1 returned-label logprob | +0.061 | [+0.028, +0.099] |
| f1 student==cheap minus b1 returned-label logprob | +0.003 | [-0.064, +0.068] |
| f3 student==cheap, then logprob gap (lexi) minus f1 student==cheap | +0.058 | [+0.023, +0.100] |
| b1 returned-label logprob minus d verbalized conf (own call) | +0.106 | [+0.012, +0.207] |
| b1 returned-label logprob minus c1 self-consistency (A==S) | +0.209 | [+0.139, +0.274] |
| b1 returned-label logprob minus e student top-2 gap | +0.228 | [+0.122, +0.326] |
| b1 returned-label logprob minus a prompt-embedding predictor (LR) | +0.310 | [+0.191, +0.431] |
| b1 returned-label logprob minus g random mixing | +0.411 | [+0.294, +0.523] |
| b1 returned-label logprob minus b4 mode-path logprob (top tokens) | +0.010 | [-0.010, +0.031] |
| b1 returned-label logprob minus b2 gap to best rival label | +0.002 | [-0.002, +0.006] |

**Descriptive risk-coverage on all 200 rows (no certificate; cutoffs chosen in-sample, so optimistic):** mismatch with E1 among the most-confident 25/50/75% of inputs (base rate 0.200; random mixing gets the base rate at every share), the largest routed share whose mismatch stays within the budget, and the blended cost per 1000 calls at the 5% and 10% operating points (includes every extra cheap call; unrouted calls pay E1 on top).
| signal | mismatch top 25% | top 50% | top 75% | max share at <=2% | <=5% | <=10% | <=15% | $/1k at 5% share | $/1k at 10% share |
|---|---|---|---|---|---|---|---|---|---|
| b1 returned-label logprob | 0.000 | 0.010 | 0.073 | 0.57 | 0.67 | 0.80 | 0.89 | 0.67 | 0.44 |
| b2 gap to best rival label | 0.000 | 0.010 | 0.073 | 0.57 | 0.67 | 0.78 | 0.90 | 0.67 | 0.48 |
| b3 first-token logprob | 0.000 | 0.020 | 0.087 | 0.52 | 0.62 | 0.80 | 0.89 | 0.75 | 0.44 |
| b4 mode-path logprob (top tokens) | 0.000 | 0.010 | 0.080 | 0.57 | 0.67 | 0.77 | 0.89 | 0.67 | 0.49 |
| b5 mode gap (top vs runner-up) | 0.000 | 0.000 | 0.087 | 0.57 | 0.67 | 0.78 | 0.88 | 0.67 | 0.48 |
| d verbalized conf (own call) | 0.065 | 0.085 | 0.110 | 0.01 | 0.01 | 0.69 | 0.90 | 1.82 | 0.63 |
| c1 self-consistency (A==S) | 0.129 | 0.130 | 0.131 | 0.00 | 0.00 | 0.00 | 0.87 | 1.91 | 1.91 |
| c2 self-consistency of 3 (A,Ab,S) | 0.116 | 0.118 | 0.119 | 0.00 | 0.00 | 0.00 | 0.83 | 2.01 | 2.01 |
| c3 2-agree, then logprob (lexi) | 0.000 | 0.010 | 0.073 | 0.57 | 0.67 | 0.81 | 0.90 | 0.77 | 0.51 |
| e student top-2 gap | 0.060 | 0.130 | 0.167 | 0.14 | 0.20 | 0.35 | 0.69 | 1.39 | 1.15 |
| f1 student==cheap | 0.034 | 0.035 | 0.033 | 0.00 | 0.76 | 0.76 | 0.76 | 0.51 | 0.51 |
| f2 student==cheap, then student gap (lexi) | 0.040 | 0.030 | 0.027 | 0.14 | 0.79 | 0.83 | 0.89 | 0.47 | 0.39 |
| f3 student==cheap, then logprob gap (lexi) | 0.000 | 0.000 | 0.033 | 0.68 | 0.79 | 0.85 | 0.94 | 0.46 | 0.35 |
| a prompt-embedding predictor (LR) | 0.160 | 0.150 | 0.167 | 0.00 | 0.00 | 0.00 | 0.57 | 1.72 | 1.72 |
| z all-signal LR (exploratory) | 0.020 | 0.010 | 0.033 | 0.66 | 0.77 | 0.84 | 0.92 | 0.59 | 0.47 |
| g random mixing | 0.240 | 0.200 | 0.193 | 0.01 | 0.01 | 0.01 | 0.01 | 1.69 | 1.69 |

**Certified routing (Clopper-Pearson, delta 5% over a 40-level grid), 5 random calibration/holdout half-splits (100/100 rows).** Zero-error rows needed to certify at all with delta/40: {2: 331, 5: 131, 10: 64, 15: 42, 20: 30} (the calibration half has 100 rows). cov = share of holdout inputs routed to the cheap model (mean, min over rotations); mism = realized mismatch with E1 among routed holdout inputs (mean, max over rotations that certified).
| signal | cov @2% | mism | cov @5% | mism | cov @10% | mism | cov @15% | mism | cov @20% | mism |
|---|---|---|---|---|---|---|---|---|---|---|
| b1 returned-label logprob | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.63 (0.44) | 0.062 (0.114) | 0.75 (0.65) | 0.084 (0.148) |
| b2 gap to best rival label | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.64 (0.44) | 0.064 (0.113) | 0.74 (0.65) | 0.082 (0.159) |
| b3 first-token logprob | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.24 (0.00) | 0.078 (0.114) | 0.73 (0.62) | 0.097 (0.163) |
| b4 mode-path logprob (top tokens) | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.59 (0.44) | 0.041 (0.076) | 0.72 (0.66) | 0.083 (0.136) |
| b5 mode gap (top vs runner-up) | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.59 (0.44) | 0.041 (0.090) | 0.72 (0.66) | 0.083 (0.134) |
| d verbalized conf (own call) | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.42 (0.00) | 0.135 (0.136) |
| c1 self-consistency (A==S) | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - |
| c2 self-consistency of 3 (A,Ab,S) | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - |
| c3 2-agree, then logprob (lexi) | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.63 (0.44) | 0.063 (0.114) | 0.73 (0.64) | 0.080 (0.148) |
| e student top-2 gap | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - |
| f1 student==cheap | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.15 (0.00) | 0.066 (0.066) | 0.47 (0.00) | 0.047 (0.066) | 0.76 (0.72) | 0.034 (0.066) |
| f2 student==cheap, then student gap (lexi) | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.15 (0.00) | 0.041 (0.041) | 0.48 (0.00) | 0.059 (0.087) | 0.80 (0.75) | 0.061 (0.099) |
| f3 student==cheap, then logprob gap (lexi) | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.53 (0.00) | 0.035 (0.065) | 0.76 (0.65) | 0.049 (0.138) | 0.81 (0.76) | 0.064 (0.167) |
| a prompt-embedding predictor (LR) | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - |
| z all-signal LR (exploratory) | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.39 (0.00) | 0.033 (0.066) | 0.69 (0.58) | 0.036 (0.066) | 0.74 (0.60) | 0.056 (0.119) |
| g random mixing | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - |

Random mixing at any routed share has mismatch 0.200 among routed calls, so it cannot meet any of these budgets. Blended $/1k for the certified operating points above (mean over the rotations that certified; n = how many of 5 did); always-E1 is 1.72, always-cheap 0.098:
| signal | $/1k @10% (n) | $/1k @15% (n) | $/1k @20% (n) |
|---|---|---|---|
| b1 returned-label logprob | - | 0.73 (5) | 0.53 (5) |
| b2 gap to best rival label | - | 0.71 (5) | 0.55 (5) |
| b3 first-token logprob | - | 0.80 (2) | 0.57 (5) |
| b4 mode-path logprob (top tokens) | - | 0.80 (5) | 0.57 (5) |
| b5 mode gap (top vs runner-up) | - | 0.80 (5) | 0.57 (5) |
| d verbalized conf (own call) | - | - | 0.64 (3) |
| c1 self-consistency (A==S) | - | - | - |
| c2 self-consistency of 3 (A,Ab,S) | - | - | - |
| c3 2-agree, then logprob (lexi) | - | 0.84 (5) | 0.66 (5) |
| e student top-2 gap | - | - | - |
| f1 student==cheap | 0.51 (1) | 0.48 (3) | 0.51 (5) |
| f2 student==cheap, then student gap (lexi) | 0.54 (1) | 0.45 (3) | 0.45 (5) |
| f3 student==cheap, then logprob gap (lexi) | 0.68 (4) | 0.52 (5) | 0.43 (5) |
| a prompt-embedding predictor (LR) | - | - | - |
| z all-signal LR (exploratory) | 0.81 (3) | 0.73 (5) | 0.64 (5) |
| g random mixing | - | - | - |

**Projection: calibration rows needed.** If the 200-row risk-coverage curve were the true curve (optimistic: it was read off the same rows), the certified share at calibration size n (same 40-level grid certificate) would be:
| signal | b=5%, n=200 | b=5%, n=500 | b=5%, n=1000 | b=5%, n=2000 | b=10%, n=200 | b=10%, n=500 | b=10%, n=1000 | b=10%, n=2000 |
|---|---|---|---|---|---|---|---|---|
| b1 returned-label logprob | 0.00 | 0.50 | 0.62 | 0.65 | 0.62 | 0.65 | 0.70 | 0.75 |
| d verbalized conf (own call) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| e student top-2 gap | 0.00 | 0.00 | 0.00 | 0.12 | 0.00 | 0.00 | 0.20 | 0.25 |
| f1 student==cheap | 0.00 | 0.00 | 0.00 | 0.76 | 0.00 | 0.76 | 0.76 | 0.76 |
| f3 student==cheap, then logprob gap (lexi) | 0.00 | 0.68 | 0.70 | 0.78 | 0.72 | 0.80 | 0.80 | 0.80 |
| z all-signal LR (exploratory) | 0.00 | 0.65 | 0.65 | 0.75 | 0.65 | 0.78 | 0.78 | 0.80 |

Verbalized confidence on the common target (nano_A answer == E1): AUROC 0.787 [0.702, 0.866]. Parsed confidences: 200/200; most common values {92: 51, 96: 28, 78: 22, 93: 14, 86: 13, 72: 12}; nano_V label equals nano_A label on 0.865 of inputs.

## Cheap model: qwen3-next-80b-a3b-instruct @ Parasail (substitute second cheap model: qwen3-32b returns no label logprobs)
cheap label == E1 on 0.795 of inputs (E1b 0.800; on E1==E1b inputs 0.819; gold acc 0.740). Cost per 1000: cheap call $0.048, E1 $1.719. A perfect router could send 80% of calls to the cheap model.

**AUROC of each signal for 'cheap answer == reference'** (95% bootstrap CI over inputs, 1000 resamples, for the E1 column). A signal that uses its own call (verbalized) is scored on that call's answer. 0.5 = random; with 200 inputs a difference below about 0.08 is within the CI.
| signal | extra cheap calls before deciding | AUROC vs E1 [95% CI] | AUROC vs E1b | AUROC on E1==E1b inputs | AUROC vs gold | rows with a score |
|---|---|---|---|---|---|---|
| b1 returned-label logprob | none | 0.788 [0.700, 0.858] | 0.773 | 0.794 | 0.725 | 1.00 |
| b2 gap to best rival label | none | 0.788 [0.701, 0.860] | 0.772 | 0.791 | 0.723 | 1.00 |
| b3 first-token logprob | none | 0.818 [0.747, 0.885] | 0.790 | 0.817 | 0.765 | 1.00 |
| b4 mode-path logprob (top tokens) | none | 0.788 [0.700, 0.858] | 0.773 | 0.794 | 0.725 | 1.00 |
| b5 mode gap (top vs runner-up) | none | 0.751 [0.654, 0.837] | 0.743 | 0.753 | 0.689 | 1.00 |
| c1 self-consistency (A==S) | 1 extra | 0.546 [0.505, 0.597] | 0.547 | 0.556 | 0.535 | 1.00 |
| c2 self-consistency of 3 (A,Ab,S) | 2 extra | 0.566 [0.515, 0.622] | 0.568 | 0.581 | 0.551 | 1.00 |
| c3 2-agree, then logprob (lexi) | 1 extra | 0.790 [0.701, 0.859] | 0.774 | 0.795 | 0.726 | 1.00 |
| e student top-2 gap | none, decided before the cheap call | 0.651 [0.550, 0.743] | 0.637 | 0.646 | 0.649 | 1.00 |
| f1 student==cheap | none | 0.916 [0.865, 0.959] | 0.928 | 0.950 | 0.945 | 1.00 |
| f2 student==cheap, then student gap (lexi) | none | 0.929 [0.886, 0.964] | 0.935 | 0.949 | 0.959 | 1.00 |
| f3 student==cheap, then logprob gap (lexi) | none | 0.946 [0.910, 0.976] | 0.939 | 0.969 | 0.926 | 1.00 |
| a prompt-embedding predictor (LR) | none, decided before the cheap call | 0.574 [0.479, 0.668] | 0.510 | 0.549 | 0.509 | 1.00 |
| z all-signal LR (exploratory) | 1 extra | 0.898 [0.816, 0.961] | 0.921 | 0.948 | 0.916 | 1.00 |
| g random mixing | none, decided before the cheap call | 0.405 [0.305, 0.504] | 0.381 | 0.388 | 0.437 | 1.00 |

**Paired bootstrap of AUROC differences** (same 200 inputs, 1000 resamples; common target 'qnx_A answer == E1', so the verbalized score from its own call is judged on this target too). A difference is real at this n only if the interval excludes 0.
| signal 1 minus signal 2 | AUROC diff | 95% CI |
|---|---|---|
| f3 student==cheap, then logprob gap (lexi) minus b1 returned-label logprob | +0.158 | [+0.088, +0.231] |
| f1 student==cheap minus b1 returned-label logprob | +0.128 | [+0.041, +0.222] |
| f3 student==cheap, then logprob gap (lexi) minus f1 student==cheap | +0.030 | [+0.003, +0.061] |
| b1 returned-label logprob minus c1 self-consistency (A==S) | +0.243 | [+0.158, +0.323] |
| b1 returned-label logprob minus e student top-2 gap | +0.137 | [+0.023, +0.252] |
| b1 returned-label logprob minus a prompt-embedding predictor (LR) | +0.214 | [+0.076, +0.341] |
| b1 returned-label logprob minus g random mixing | +0.384 | [+0.262, +0.504] |
| b1 returned-label logprob minus b4 mode-path logprob (top tokens) | +0.000 | [+0.000, +0.000] |
| b1 returned-label logprob minus b2 gap to best rival label | +0.000 | [-0.017, +0.015] |

**Descriptive risk-coverage on all 200 rows (no certificate; cutoffs chosen in-sample, so optimistic):** mismatch with E1 among the most-confident 25/50/75% of inputs (base rate 0.205; random mixing gets the base rate at every share), the largest routed share whose mismatch stays within the budget, and the blended cost per 1000 calls at the 5% and 10% operating points (includes every extra cheap call; unrouted calls pay E1 on top).
| signal | mismatch top 25% | top 50% | top 75% | max share at <=2% | <=5% | <=10% | <=15% | $/1k at 5% share | $/1k at 10% share |
|---|---|---|---|---|---|---|---|---|---|
| b1 returned-label logprob | 0.060 | 0.070 | 0.100 | 0.01 | 0.23 | 0.76 | 0.86 | 1.38 | 0.47 |
| b2 gap to best rival label | 0.060 | 0.070 | 0.100 | 0.01 | 0.23 | 0.75 | 0.85 | 1.38 | 0.48 |
| b3 first-token logprob | 0.043 | 0.053 | 0.100 | 0.12 | 0.18 | 0.78 | 0.87 | 1.45 | 0.44 |
| b4 mode-path logprob (top tokens) | 0.060 | 0.070 | 0.100 | 0.01 | 0.23 | 0.76 | 0.86 | 1.38 | 0.47 |
| b5 mode gap (top vs runner-up) | 0.120 | 0.090 | 0.100 | 0.00 | 0.00 | 0.75 | 0.85 | 1.77 | 0.48 |
| c1 self-consistency (A==S) | 0.190 | 0.185 | 0.188 | 0.00 | 0.00 | 0.00 | 0.00 | 1.82 | 1.82 |
| c2 self-consistency of 3 (A,Ab,S) | 0.184 | 0.178 | 0.182 | 0.00 | 0.00 | 0.00 | 0.00 | 1.86 | 1.86 |
| c3 2-agree, then logprob (lexi) | 0.060 | 0.070 | 0.100 | 0.01 | 0.23 | 0.76 | 0.86 | 1.43 | 0.52 |
| e student top-2 gap | 0.140 | 0.150 | 0.153 | 0.04 | 0.04 | 0.18 | 0.74 | 1.66 | 1.41 |
| f1 student==cheap | 0.023 | 0.023 | 0.034 | 0.00 | 0.73 | 0.73 | 0.73 | 0.50 | 0.50 |
| f2 student==cheap, then student gap (lexi) | 0.000 | 0.010 | 0.033 | 0.72 | 0.76 | 0.82 | 0.88 | 0.46 | 0.36 |
| f3 student==cheap, then logprob gap (lexi) | 0.000 | 0.010 | 0.033 | 0.66 | 0.77 | 0.84 | 0.91 | 0.45 | 0.32 |
| a prompt-embedding predictor (LR) | 0.120 | 0.170 | 0.200 | 0.01 | 0.01 | 0.22 | 0.34 | 1.70 | 1.35 |
| z all-signal LR (exploratory) | 0.060 | 0.030 | 0.033 | 0.02 | 0.78 | 0.83 | 0.91 | 0.48 | 0.38 |
| g random mixing | 0.320 | 0.230 | 0.213 | 0.01 | 0.01 | 0.01 | 0.01 | 1.69 | 1.69 |

**Certified routing (Clopper-Pearson, delta 5% over a 40-level grid), 5 random calibration/holdout half-splits (100/100 rows).** Zero-error rows needed to certify at all with delta/40: {2: 331, 5: 131, 10: 64, 15: 42, 20: 30} (the calibration half has 100 rows). cov = share of holdout inputs routed to the cheap model (mean, min over rotations); mism = realized mismatch with E1 among routed holdout inputs (mean, max over rotations that certified).
| signal | cov @2% | mism | cov @5% | mism | cov @10% | mism | cov @15% | mism | cov @20% | mism |
|---|---|---|---|---|---|---|---|---|---|---|
| b1 returned-label logprob | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.09 (0.00) | 0.109 (0.109) |
| b2 gap to best rival label | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.27 (0.00) | 0.095 (0.097) |
| b3 first-token logprob | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.40 (0.00) | 0.095 (0.103) |
| b4 mode-path logprob (top tokens) | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.09 (0.00) | 0.109 (0.109) |
| b5 mode gap (top vs runner-up) | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.27 (0.00) | 0.095 (0.097) |
| c1 self-consistency (A==S) | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - |
| c2 self-consistency of 3 (A,Ab,S) | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - |
| c3 2-agree, then logprob (lexi) | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.09 (0.00) | 0.109 (0.109) |
| e student top-2 gap | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - |
| f1 student==cheap | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.59 (0.00) | 0.017 (0.026) | 0.73 (0.69) | 0.013 (0.026) |
| f2 student==cheap, then student gap (lexi) | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.15 (0.00) | 0.026 (0.026) | 0.71 (0.62) | 0.010 (0.026) | 0.80 (0.73) | 0.079 (0.171) |
| f3 student==cheap, then logprob gap (lexi) | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.69 (0.53) | 0.014 (0.026) | 0.80 (0.74) | 0.074 (0.136) |
| a prompt-embedding predictor (LR) | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - |
| z all-signal LR (exploratory) | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.57 (0.00) | 0.017 (0.026) | 0.75 (0.70) | 0.036 (0.083) |
| g random mixing | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - |

Random mixing at any routed share has mismatch 0.205 among routed calls, so it cannot meet any of these budgets. Blended $/1k for the certified operating points above (mean over the rotations that certified; n = how many of 5 did); always-E1 is 1.72, always-cheap 0.048:
| signal | $/1k @10% (n) | $/1k @15% (n) | $/1k @20% (n) |
|---|---|---|---|
| b1 returned-label logprob | - | - | 0.98 (1) |
| b2 gap to best rival label | - | - | 0.59 (2) |
| b3 first-token logprob | - | - | 0.63 (3) |
| b4 mode-path logprob (top tokens) | - | - | 0.98 (1) |
| b5 mode gap (top vs runner-up) | - | - | 0.59 (2) |
| c1 self-consistency (A==S) | - | - | - |
| c2 self-consistency of 3 (A,Ab,S) | - | - | - |
| c3 2-agree, then logprob (lexi) | - | - | 1.02 (1) |
| e student top-2 gap | - | - | - |
| f1 student==cheap | - | 0.50 (4) | 0.51 (5) |
| f2 student==cheap, then student gap (lexi) | 0.44 (1) | 0.54 (5) | 0.40 (5) |
| f3 student==cheap, then logprob gap (lexi) | - | 0.59 (5) | 0.39 (5) |
| a prompt-embedding predictor (LR) | - | - | - |
| z all-signal LR (exploratory) | - | 0.59 (4) | 0.52 (5) |
| g random mixing | - | - | - |

**Projection: calibration rows needed.** If the 200-row risk-coverage curve were the true curve (optimistic: it was read off the same rows), the certified share at calibration size n (same 40-level grid certificate) would be:
| signal | b=5%, n=200 | b=5%, n=500 | b=5%, n=1000 | b=5%, n=2000 | b=10%, n=200 | b=10%, n=500 | b=10%, n=1000 | b=10%, n=2000 |
|---|---|---|---|---|---|---|---|---|
| b1 returned-label logprob | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 |
| e student top-2 gap | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| f1 student==cheap | 0.00 | 0.00 | 0.73 | 0.73 | 0.73 | 0.73 | 0.73 | 0.73 |
| f3 student==cheap, then logprob gap (lexi) | 0.00 | 0.65 | 0.72 | 0.75 | 0.72 | 0.78 | 0.78 | 0.80 |
| z all-signal LR (exploratory) | 0.00 | 0.00 | 0.72 | 0.75 | 0.72 | 0.78 | 0.80 | 0.80 |

## Cheap model: gpt-4.1-nano (extra: returns the full top-5 alternatives)
cheap label == E1 on 0.690 of inputs (E1b 0.685; on E1==E1b inputs 0.707; gold acc 0.625). Cost per 1000: cheap call $0.046, E1 $1.719. A perfect router could send 69% of calls to the cheap model.

**AUROC of each signal for 'cheap answer == reference'** (95% bootstrap CI over inputs, 1000 resamples, for the E1 column). A signal that uses its own call (verbalized) is scored on that call's answer. 0.5 = random; with 200 inputs a difference below about 0.08 is within the CI.
| signal | extra cheap calls before deciding | AUROC vs E1 [95% CI] | AUROC vs E1b | AUROC on E1==E1b inputs | AUROC vs gold | rows with a score |
|---|---|---|---|---|---|---|
| b1 returned-label logprob | none | 0.724 [0.654, 0.790] | 0.729 | 0.730 | 0.684 | 1.00 |
| b2 gap to best rival label | none | 0.707 [0.636, 0.775] | 0.715 | 0.714 | 0.675 | 1.00 |
| b3 first-token logprob | none | 0.721 [0.650, 0.792] | 0.729 | 0.727 | 0.694 | 1.00 |
| b4 mode-path logprob (top tokens) | none | 0.724 [0.654, 0.790] | 0.729 | 0.730 | 0.684 | 1.00 |
| b5 mode gap (top vs runner-up) | none | 0.707 [0.635, 0.776] | 0.715 | 0.714 | 0.674 | 1.00 |
| c1 self-consistency (A==S) | 1 extra | 0.499 [0.469, 0.534] | 0.510 | 0.505 | 0.503 | 1.00 |
| c2 self-consistency of 3 (A,Ab,S) | 2 extra | 0.516 [0.478, 0.557] | 0.526 | 0.525 | 0.515 | 1.00 |
| c3 2-agree, then logprob (lexi) | 1 extra | 0.721 [0.648, 0.788] | 0.728 | 0.729 | 0.683 | 1.00 |
| e student top-2 gap | none, decided before the cheap call | 0.683 [0.608, 0.766] | 0.656 | 0.672 | 0.677 | 1.00 |
| f1 student==cheap | none | 0.921 [0.880, 0.956] | 0.925 | 0.940 | 0.961 | 1.00 |
| f2 student==cheap, then student gap (lexi) | none | 0.937 [0.899, 0.971] | 0.931 | 0.947 | 0.972 | 1.00 |
| f3 student==cheap, then logprob gap (lexi) | none | 0.944 [0.905, 0.974] | 0.950 | 0.963 | 0.960 | 1.00 |
| a prompt-embedding predictor (LR) | none, decided before the cheap call | 0.628 [0.541, 0.718] | 0.616 | 0.628 | 0.545 | 1.00 |
| z all-signal LR (exploratory) | 1 extra | 0.916 [0.868, 0.958] | 0.922 | 0.933 | 0.954 | 1.00 |
| g random mixing | none, decided before the cheap call | 0.485 [0.388, 0.574] | 0.514 | 0.499 | 0.463 | 1.00 |

**Paired bootstrap of AUROC differences** (same 200 inputs, 1000 resamples; common target 'g41_A answer == E1', so the verbalized score from its own call is judged on this target too). A difference is real at this n only if the interval excludes 0.
| signal 1 minus signal 2 | AUROC diff | 95% CI |
|---|---|---|
| f3 student==cheap, then logprob gap (lexi) minus b1 returned-label logprob | +0.220 | [+0.156, +0.290] |
| f1 student==cheap minus b1 returned-label logprob | +0.197 | [+0.122, +0.279] |
| f3 student==cheap, then logprob gap (lexi) minus f1 student==cheap | +0.023 | [+0.004, +0.042] |
| b1 returned-label logprob minus c1 self-consistency (A==S) | +0.226 | [+0.153, +0.292] |
| b1 returned-label logprob minus e student top-2 gap | +0.041 | [-0.075, +0.150] |
| b1 returned-label logprob minus a prompt-embedding predictor (LR) | +0.096 | [-0.009, +0.204] |
| b1 returned-label logprob minus g random mixing | +0.239 | [+0.144, +0.338] |
| b1 returned-label logprob minus b4 mode-path logprob (top tokens) | +0.000 | [+0.000, +0.000] |
| b1 returned-label logprob minus b2 gap to best rival label | +0.017 | [+0.003, +0.030] |

**Descriptive risk-coverage on all 200 rows (no certificate; cutoffs chosen in-sample, so optimistic):** mismatch with E1 among the most-confident 25/50/75% of inputs (base rate 0.310; random mixing gets the base rate at every share), the largest routed share whose mismatch stays within the budget, and the blended cost per 1000 calls at the 5% and 10% operating points (includes every extra cheap call; unrouted calls pay E1 on top).
| signal | mismatch top 25% | top 50% | top 75% | max share at <=2% | <=5% | <=10% | <=15% | $/1k at 5% share | $/1k at 10% share |
|---|---|---|---|---|---|---|---|---|---|
| b1 returned-label logprob | 0.060 | 0.170 | 0.253 | 0.07 | 0.23 | 0.30 | 0.47 | 1.36 | 1.25 |
| b2 gap to best rival label | 0.040 | 0.200 | 0.260 | 0.07 | 0.28 | 0.33 | 0.41 | 1.28 | 1.21 |
| b3 first-token logprob | 0.040 | 0.180 | 0.240 | 0.12 | 0.25 | 0.31 | 0.46 | 1.34 | 1.23 |
| b4 mode-path logprob (top tokens) | 0.060 | 0.170 | 0.253 | 0.07 | 0.23 | 0.30 | 0.47 | 1.36 | 1.25 |
| b5 mode gap (top vs runner-up) | 0.040 | 0.200 | 0.253 | 0.00 | 0.28 | 0.33 | 0.41 | 1.28 | 1.20 |
| c1 self-consistency (A==S) | 0.313 | 0.309 | 0.314 | 0.00 | 0.00 | 0.00 | 0.00 | 1.81 | 1.81 |
| c2 self-consistency of 3 (A,Ab,S) | 0.304 | 0.299 | 0.306 | 0.00 | 0.00 | 0.00 | 0.00 | 1.86 | 1.86 |
| c3 2-agree, then logprob (lexi) | 0.060 | 0.170 | 0.247 | 0.07 | 0.23 | 0.30 | 0.47 | 1.41 | 1.30 |
| e student top-2 gap | 0.160 | 0.190 | 0.253 | 0.04 | 0.13 | 0.18 | 0.21 | 1.50 | 1.41 |
| f1 student==cheap | 0.020 | 0.024 | 0.150 | 0.00 | 0.63 | 0.63 | 0.63 | 0.68 | 0.68 |
| f2 student==cheap, then student gap (lexi) | 0.000 | 0.010 | 0.140 | 0.62 | 0.66 | 0.71 | 0.76 | 0.64 | 0.54 |
| f3 student==cheap, then logprob gap (lexi) | 0.000 | 0.030 | 0.133 | 0.37 | 0.67 | 0.71 | 0.78 | 0.62 | 0.54 |
| a prompt-embedding predictor (LR) | 0.280 | 0.230 | 0.247 | 0.01 | 0.01 | 0.01 | 0.01 | 1.71 | 1.71 |
| z all-signal LR (exploratory) | 0.020 | 0.030 | 0.153 | 0.27 | 0.65 | 0.68 | 0.74 | 0.70 | 0.64 |
| g random mixing | 0.360 | 0.320 | 0.300 | 0.00 | 0.00 | 0.00 | 0.00 | 1.72 | 1.72 |

**Certified routing (Clopper-Pearson, delta 5% over a 40-level grid), 5 random calibration/holdout half-splits (100/100 rows).** Zero-error rows needed to certify at all with delta/40: {2: 331, 5: 131, 10: 64, 15: 42, 20: 30} (the calibration half has 100 rows). cov = share of holdout inputs routed to the cheap model (mean, min over rotations); mism = realized mismatch with E1 among routed holdout inputs (mean, max over rotations that certified).
| signal | cov @2% | mism | cov @5% | mism | cov @10% | mism | cov @15% | mism | cov @20% | mism |
|---|---|---|---|---|---|---|---|---|---|---|
| b1 returned-label logprob | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - |
| b2 gap to best rival label | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - |
| b3 first-token logprob | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - |
| b4 mode-path logprob (top tokens) | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - |
| b5 mode gap (top vs runner-up) | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - |
| c1 self-consistency (A==S) | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - |
| c2 self-consistency of 3 (A,Ab,S) | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - |
| c3 2-agree, then logprob (lexi) | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - |
| e student top-2 gap | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.07 (0.00) | 0.297 (0.297) |
| f1 student==cheap | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.39 (0.00) | 0.031 (0.032) | 0.65 (0.62) | 0.018 (0.032) |
| f2 student==cheap, then student gap (lexi) | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.52 (0.00) | 0.051 (0.187) | 0.68 (0.62) | 0.055 (0.187) |
| f3 student==cheap, then logprob gap (lexi) | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.25 (0.00) | 0.032 (0.033) | 0.65 (0.58) | 0.029 (0.072) |
| a prompt-embedding predictor (LR) | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - |
| z all-signal LR (exploratory) | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.36 (0.00) | 0.022 (0.034) | 0.66 (0.62) | 0.036 (0.062) |
| g random mixing | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - | 0.00 (0.00) | - |

Random mixing at any routed share has mismatch 0.310 among routed calls, so it cannot meet any of these budgets. Blended $/1k for the certified operating points above (mean over the rotations that certified; n = how many of 5 did); always-E1 is 1.72, always-cheap 0.046:
| signal | $/1k @10% (n) | $/1k @15% (n) | $/1k @20% (n) |
|---|---|---|---|
| b1 returned-label logprob | - | - | - |
| b2 gap to best rival label | - | - | - |
| b3 first-token logprob | - | - | - |
| b4 mode-path logprob (top tokens) | - | - | - |
| b5 mode gap (top vs runner-up) | - | - | - |
| c1 self-consistency (A==S) | - | - | - |
| c2 self-consistency of 3 (A,Ab,S) | - | - | - |
| c3 2-agree, then logprob (lexi) | - | - | - |
| e student top-2 gap | - | - | 1.10 (1) |
| f1 student==cheap | - | 0.64 (3) | 0.65 (5) |
| f2 student==cheap, then student gap (lexi) | - | 0.65 (4) | 0.60 (5) |
| f3 student==cheap, then logprob gap (lexi) | - | 0.69 (2) | 0.64 (5) |
| a prompt-embedding predictor (LR) | - | - | - |
| z all-signal LR (exploratory) | - | 0.77 (3) | 0.68 (5) |
| g random mixing | - | - | - |

**Projection: calibration rows needed.** If the 200-row risk-coverage curve were the true curve (optimistic: it was read off the same rows), the certified share at calibration size n (same 40-level grid certificate) would be:
| signal | b=5%, n=200 | b=5%, n=500 | b=5%, n=1000 | b=5%, n=2000 | b=10%, n=200 | b=10%, n=500 | b=10%, n=1000 | b=10%, n=2000 |
|---|---|---|---|---|---|---|---|---|
| b1 returned-label logprob | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.23 | 0.23 | 0.25 |
| e student top-2 gap | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 |
| f1 student==cheap | 0.00 | 0.00 | 0.63 | 0.63 | 0.63 | 0.63 | 0.63 | 0.63 |
| f3 student==cheap, then logprob gap (lexi) | 0.00 | 0.35 | 0.62 | 0.62 | 0.62 | 0.65 | 0.68 | 0.68 |
| z all-signal LR (exploratory) | 0.00 | 0.00 | 0.62 | 0.62 | 0.62 | 0.62 | 0.65 | 0.65 |

## Sensitivity: how good must the shadow student be? (the student here is trained on GOLD labels; E1 itself has gold acc 0.870)
| student rows | student gold acc | agrees with E1 | AUROC student gap (vs nano_A==E1) | AUROC student==nano_A | lexi(student==nano_A, nano logprob gap) | AUROC student==qnx_A | lexi(student==qnx_A, qnx logprob gap) |
|---|---|---|---|---|---|---|---|
| 77 | 0.640 | 0.605 | 0.647 | 0.731 | 0.882 | 0.742 | 0.844 |
| 385 | 0.725 | 0.670 | 0.656 | 0.800 | 0.923 | 0.816 | 0.894 |
| 770 | 0.840 | 0.755 | 0.626 | 0.847 | 0.938 | 0.854 | 0.905 |
| 3080 | 0.905 | 0.820 | 0.665 | 0.894 | 0.951 | 0.907 | 0.929 |
| 9993 | 0.935 | 0.850 | 0.670 | 0.900 | 0.958 | 0.919 | 0.951 |