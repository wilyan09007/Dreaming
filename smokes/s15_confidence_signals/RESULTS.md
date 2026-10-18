## In plain words

The claim under test: "a second model that predicts whether the first will be right is useless, because everything useful is already in the first model's own confidence." We tested it where Dreaming decides whether to trust a cheap answer: the small student model, and the cheap-versus-expensive router. For the student the claim holds: its own top-2 gap ranks errors correctly 88-90% of the time, and a second model reading only the prompt manages 56-76%. A second model reading the student's own scores adds at most one point, which does not change how many calls can be certified. For the router the claim is half right: a prompt-only predictor is useless (57-63%, and nothing certifies), and the cheap model's token probabilities are good for gpt-5.4-nano (89%) but weak for the other two cheap models (72-79%). What the probabilities miss is a second, independent answer: asking whether the free student gives the same label as the cheap model reaches 90-92% on every cheap model, and adding the probability gap reaches 94-96%. With only 100 calibration rows nothing certifies at a 2% or 5% error budget on the router side. At 15-20% budgets the best signal routes about 76-81% of calls to the cheap model, 70-75% cheaper than always using the expensive one, and 5-10% budgets need 500-1000 calibration rows.

Best single gate: **student side, the student's own top-2 gap** (a temperature-scaled gap is a statistical tie). **Router side, "student label equals cheap label", with the cheap model's logprob gap as tie-break** (use agreement alone where the API gives no logprobs).

## Setup

- Everything here ran on a RunPod RTX A5000 pod (`pod.py`; 15.5 min for Part 1 plus Part 2, 3.1 min for the repeat-noise table). Embeddings used the pod GPU (a 4-line wrapper in `run.py` around `SentenceTransformer`; `common.py` untouched), so third decimals differ slightly from the earlier laptop attempt, which this run replaces. API calls were made earlier and are read from the cache (`responses.jsonl`, 2200 rows); this session made no new cheap-model calls and did not re-run E1.
- **Part 1 (student side, $0).** Frozen `BAAI/bge-small-en-v1.5` embeddings, student = `LogisticRegression(C=10)`. Banking77 trained on 770 rows (test acc 0.821) and 3080 rows (0.910); CLINC150 in-scope trained on 1500 rows (0.906). Signals (higher = more confident): a top-2 gap; b max prob; c -entropy; d top-2 gap after temperature scaling (T fit on the head-training rows, about 0.46 / 0.55 / 0.38); e share of the 10 nearest student-training rows (cosine) carrying the predicted label; f LR head on the z-scored prompt embedding predicting "student is wrong"; g LR head on z-scored (a)-(e) features; h gap + kNN share. Heads train on 1000 unused training rows (300 and 3000 as sensitivity for f and g). Certificate: `certify()` fixed grid, delta 5%, fit on a 500-row calibration slice of the test set, applied to the rest; 5 random splits, mean (min over splits). Budgets 2%, 5% and also 10% (added because 500 rows certify nothing at 2% for these students). A second pass uses 1500 calibration rows. Extra: bootstrap 95% CI of the AUROC difference vs the gap over holdout rows (split 0).
- **Part 2 (router side).** The 200 Banking77 test inputs, system prompt and parser of S5 (`b_data()` identical to `s5_tier_router/run.py`). Reference E1 = openai/gpt-5.4 effort low from S5's `responses.jsonl` (not re-run). Target = "cheap label == E1" (also reported vs E1b and on inputs where E1 == E1b). Cheap calls, all provider-pinned and logged: **openai/gpt-5.4-nano** @ OpenAI (reasoning none; logprobs work), **qwen/qwen3-next-80b-a3b-instruct** @ Parasail, and **openai/gpt-4.1-nano** @ OpenAI as an extra. Each model: A and Ab (same request twice, temperature 0), S (temperature 0.7 sent), and for nano also V and Vb (label + " | " + confidence 0-100). `logprobs=True, top_logprobs=5`. Labels are scored by the logprob of the returned label (b1), the gap to the best rival listed label built from the returned top-5 alternatives with a floor of -12 for unlisted rivals (b2), the first token only (b3). The shadow student is the S1 student (bge-small + LR) trained on all 9993 Banking77 training rows. Certified routing: calibration half = 100 inputs, holdout = 100, 5 random rotations, same `certify()`; a 100-row half needs 331 / 131 / 64 / 42 / 30 zero-error rows to certify 2 / 5 / 10 / 15 / 20% at all, so 15% and 20% budgets are added.
- **Substitution, evidence.** The spec asked for qwen/qwen3-32b with logprobs. `probe_q32.py` shows: DeepInfra (fp8) returns no logprobs; Tenstorrent claims support but returns logprobs for the stop token only (`<|im_end|>`), none for the label tokens; SiliconFlow lists no logprob support; Groq, Cerebras, Nebius, Together, Fireworks and SambaNova have no endpoint. Hence qwen3-next-80b-a3b-instruct (Parasail) stands in as the second cheap model and gpt-4.1-nano is added.

## Results

### Part 1: student side

Banking77 and CLINC, 500 calibration rows, heads on 1000 rows. AUROC = ranking errors above correct answers on the holdout (mean ± sd over 5 splits). Cov = share of holdout answered at the certified cutoff, mean (min over splits). The 2% column is 0.00 for every signal at 500 calibration rows; realized error at the 5% budget stayed under 5% in every cell that certified (max 0.038).

| student | signal | AUROC | cov @5% | cov @10% | dAUROC vs gap (bootstrap 95% CI, split 0) | dcov @5% (sd, wins) |
|---|---|---|---|---|---|---|
| B77, 770 rows | a gap | 0.878 ± 0.003 | 0.12 (0.00) | 0.69 (0.60) | - | - |
| | b max prob | 0.869 | 0.11 | 0.61 | -0.010 [-0.015, -0.001] | -0.014 ± 0.029 (1/5) |
| | c -entropy | 0.820 | 0.06 | 0.29 | -0.059 [-0.072, -0.042] | -0.057 ± 0.145 (1/5) |
| | d temp gap | 0.876 | 0.20 | 0.67 | -0.002 [-0.007, +0.000] | +0.077 ± 0.131 (2/5) |
| | e kNN agree | 0.749 | 0.00 | 0.12 | -0.129 [-0.141, -0.103] | -0.121 ± 0.151 (0/5) |
| | f head(emb) | 0.659 | 0.00 | 0.00 | -0.219 [-0.255, -0.195] | -0.121 ± 0.151 (0/5) |
| | g head(a-e) | 0.883 | 0.17 | 0.70 | +0.005 [-0.003, +0.008] | +0.047 ± 0.249 (2/5) |
| | h gap+kNN | 0.835 | 0.00 | 0.53 | -0.044 [-0.049, -0.031] | -0.121 ± 0.151 (0/5) |
| B77, 3080 rows | a gap | 0.898 ± 0.004 | 0.75 (0.63) | 0.92 (0.87) | - | - |
| | b max prob | 0.889 | 0.71 | 0.91 | -0.009 [-0.015, -0.004] | -0.043 ± 0.045 (1/5) |
| | c -entropy | 0.848 | 0.51 | 0.87 | -0.050 [-0.064, -0.035] | -0.247 ± 0.224 (0/5) |
| | d temp gap | 0.899 | 0.74 | 0.92 | +0.001 [-0.002, +0.005] | -0.018 ± 0.052 (3/5) |
| | e kNN agree | 0.879 | 0.65 | 0.92 | -0.018 [-0.033, -0.000] | -0.099 ± 0.072 (0/5) |
| | f head(emb) | 0.613 | 0.00 | 0.46 | -0.285 [-0.322, -0.241] | -0.753 ± 0.083 (0/5) |
| | g head(a-e) | 0.908 | 0.78 | 0.92 | +0.010 [+0.006, +0.016] | +0.031 ± 0.054 (3/5) |
| | h gap+kNN | 0.907 | 0.77 | 0.92 | +0.009 [+0.003, +0.018] | +0.012 ± 0.058 (3/5) |
| CLINC, 1500 rows | a gap | 0.891 ± 0.001 | 0.65 (0.56) | 0.92 (0.90) | - | - |
| | b max prob | 0.860 | 0.51 | 0.90 | -0.031 [-0.040, -0.025] | -0.137 ± 0.078 (0/5) |
| | c -entropy | 0.798 | 0.23 | 0.85 | -0.093 [-0.111, -0.080] | -0.413 ± 0.137 (0/5) |
| | d temp gap | 0.899 | 0.69 | 0.92 | +0.008 [+0.003, +0.013] | +0.049 ± 0.079 (3/5) |
| | e kNN agree | 0.788 | 0.00 | 0.73 | -0.103 [-0.124, -0.085] | -0.646 ± 0.088 (0/5) |
| | f head(emb) | 0.644 | 0.00 | 0.08 | -0.247 [-0.291, -0.225] | -0.646 ± 0.088 (0/5) |
| | g head(a-e) | 0.894 | 0.62 (0.45) | 0.92 | +0.003 [+0.001, +0.004] | -0.022 ± 0.046 (2/5) |
| | h gap+kNN | 0.856 | 0.54 | 0.87 | -0.035 [-0.046, -0.026] | -0.103 ± 0.136 (1/5) |

How the external prompt-embedding head f scales with its own training size (AUROC): B77-770 0.607 / 0.659 / 0.755 for 300 / 1000 / 3000 head rows; B77-3080 0.557 / 0.613 / 0.680. It improves with data but stays 0.12-0.34 below the gap. The (a)-(e) head g is within 0.006 AUROC of its 1000-row value at 300 rows.

Larger calibration (1500 rows, so the 2% budget becomes testable; heads on 1000 rows; mean (min) coverage):

| student | signal | AUROC | cov @2% | cov @5% | cov @10% | dAUROC vs gap [CI] | dcov @5% (sd, wins) |
|---|---|---|---|---|---|---|---|
| B77, 770 rows | a gap | 0.878 ± 0.006 | 0.05 (0.00) | 0.53 (0.45) | 0.77 (0.73) | - | - |
| | d temp gap | 0.876 | 0.09 | 0.55 | 0.76 | -0.002 [-0.009, +0.001] | +0.021 ± 0.023 (3/5) |
| | g head(a-e) | 0.883 | 0.06 | 0.55 | 0.76 | +0.005 [-0.001, +0.012] | +0.018 ± 0.028 (4/5) |
| | f head(emb) | 0.658 | 0.00 | 0.00 | 0.01 | -0.219 | -0.529 |
| B77, 3080 rows | a gap | 0.903 ± 0.008 | 0.33 (0.00) | 0.84 (0.82) | 0.97 (0.95) | - | - |
| | d temp gap | 0.905 | 0.24 | 0.84 | 0.96 | +0.002 [-0.002, +0.006] | -0.003 ± 0.003 (1/5) |
| | g head(a-e) | 0.914 | 0.32 | 0.83 | 0.97 | +0.011 [+0.001, +0.013] | -0.007 ± 0.004 (0/5) |
| | h gap+kNN | 0.913 | 0.44 | 0.82 | 0.97 | +0.010 [-0.003, +0.014] | -0.018 ± 0.011 (0/5) |
| | f head(emb) | 0.612 | 0.00 | 0.00 | 0.71 | -0.292 | -0.841 |
| CLINC, 1500 rows | a gap | 0.891 ± 0.002 | 0.31 (0.00) | 0.80 (0.79) | 0.95 (0.94) | - | - |
| | d temp gap | 0.899 | 0.39 | 0.81 | 0.95 | +0.009 [+0.004, +0.014] | +0.011 ± 0.018 (3/5) |
| | g head(a-e) | 0.894 | 0.35 | 0.82 | 0.95 | +0.003 [+0.001, +0.004] | +0.014 ± 0.013 (3/5) |
| | h gap+kNN | 0.854 | 0.42 (0.30) | 0.72 | 0.93 | -0.037 [-0.049, -0.025] | -0.080 ± 0.038 (0/5) |

The 2% column is the least stable number in the study: min coverage over the 5 splits is 0.00 for every signal in every cell except gap+kNN on CLINC (0.30), because a 2% Clopper-Pearson bound with delta/40 sits right at the edge of 1500 calibration rows.

**Part 1 answer.** Nothing beats the student's own gap by more than split noise at the level that matters, certified coverage. The two small AUROC gains are real in sign but tiny: the (a)-(e) head g gains +0.003 to +0.011 AUROC (5/5 splits in every cell; bootstrap CI excludes 0 in 4 of 6 cells; the two Banking77-770 cells include 0), and the temperature-scaled gap gains +0.001 to +0.002 on Banking77 and +0.008 on CLINC. Neither turns into coverage: g's coverage changes at 5% are +0.047 ± 0.249, +0.031 ± 0.054, -0.022 ± 0.046 (500 cal rows) and +0.018, -0.007, +0.014 (1500 cal rows), all inside one sd of their own spread or of the wrong sign. A head on the prompt embedding alone (f) is clearly worse at every head size and every student. Max prob, -entropy and kNN agreement are significantly worse than the gap in most cells. Note the AUROC sd across splits understates test-sampling noise (splits re-partition the same test rows), which is why the bootstrap CI is shown alongside.

### Part 2: router side

**Cheap models and the noise floor** (200 inputs; E1 = gpt-5.4 low, $1.719 per 1000 calls; gold accuracy of E1 0.870, E1b 0.895).

| model (provider pinned) | match E1 (A / Ab / S) | match E1 where E1 == E1b | gold acc (A) | $/1k calls | same-config repeat agreement (A vs Ab) |
|---|---|---|---|---|---|
| gpt-5.4-nano, OpenAI | 0.800 / 0.785 / 0.810 | 0.830 | 0.750 | 0.098 | 0.890 (22 of 200 differ; temperature is ignored, every draw is a T=1 sample) |
| qwen3-next-80b-a3b, Parasail | 0.795 / 0.795 / 0.790 | 0.819 | 0.740 | 0.048 | 0.980 (4 differ) |
| gpt-4.1-nano, OpenAI | 0.690 / 0.685 / 0.660 | 0.707 | 0.625 | 0.046 | 0.965 (7 differ) |
| reference E1 vs E1b | | | | 1.719 | 0.940 (12 differ) |

A perfect router could therefore send at most 80% (nano, qwen) or 69% (gpt-4.1-nano) of calls to the cheap model. The reference itself differs from its own repeat on 6% of inputs, so no signal can meaningfully certify below about that budget.

**AUROC for "cheap label == E1"** (95% bootstrap CI over inputs; n = 200, so differences under about 0.08 are inside the CI unless paired). "Extra" = extra cheap calls needed before deciding.

| signal | extra | gpt-5.4-nano | qwen3-next-80b | gpt-4.1-nano |
|---|---|---|---|---|
| g random mixing | none | 0.482 [0.382, 0.572] | 0.405 [0.305, 0.504] | 0.485 [0.388, 0.574] |
| a prompt-embedding predictor (5-fold LR, decided before the call) | none | 0.583 [0.479, 0.680] | 0.574 [0.478, 0.668] | 0.628 [0.541, 0.718] |
| e student top-2 gap | none | 0.665 [0.574, 0.748] | 0.651 [0.550, 0.743] | 0.683 [0.608, 0.766] |
| c1 self-consistency (A == S) | 1 call | 0.684 [0.606, 0.767] | 0.546 [0.505, 0.597] | 0.499 [0.469, 0.534] |
| c2 self-consistency of 3 (A, Ab, S) | 2 calls | 0.712 [0.635, 0.792] | 0.566 [0.515, 0.622] | 0.516 [0.478, 0.557] |
| d verbalized confidence 0-100 (own call, $0.109/1k) | own call | 0.782 [0.693, 0.862] (parsed 200/200, 51 answers say 92) | n/a | n/a |
| b3 first-token logprob | none | 0.869 [0.813, 0.919] | 0.818 [0.747, 0.885] | 0.721 [0.650, 0.792] |
| b1 returned-label logprob | none | 0.894 [0.849, 0.937] | 0.788 [0.700, 0.858] | 0.724 [0.654, 0.790] |
| b2 gap to best rival label | none | 0.892 [0.846, 0.935] | 0.788 [0.701, 0.860] | 0.707 [0.636, 0.775] |
| f1 student label == cheap label | none | 0.897 [0.834, 0.949] | 0.916 [0.865, 0.959] | 0.921 [0.880, 0.956] |
| f2 f1, then student gap as tie-break | none | 0.902 [0.832, 0.954] | 0.929 [0.886, 0.964] | 0.937 [0.899, 0.971] |
| **f3 f1, then logprob gap (b2) as tie-break** | none | **0.955 [0.923, 0.979]** | **0.946 [0.910, 0.976]** | **0.944 [0.905, 0.974]** |
| z all-signal LR (exploratory, 5-fold) | 1 call | 0.924 [0.856, 0.968] | 0.898 [0.816, 0.961] | 0.916 [0.868, 0.958] |

Same signals against E1b, and on inputs where E1 == E1b, move by under 0.04 (for example nano b1 0.893 / 0.904, f3 0.954 / 0.971). Against gold labels b1 is 0.838 (nano), so the logprob also tracks "right", not just "agrees with E1".

**Paired differences** (same 200 inputs, 1000 bootstrap resamples) and **the same-config repeat spread** (the signal difference recomputed on each independent draw of the same cheap model: A, Ab, S):

| comparison | gpt-5.4-nano | qwen3-next-80b | gpt-4.1-nano |
|---|---|---|---|
| f3 minus b1 (student agreement + logprob gap vs logprob) | +0.061 [+0.028, +0.099]; draws +0.061 / +0.059 / +0.078 | +0.158 [+0.088, +0.231]; draws +0.152 to +0.183 | +0.220 [+0.156, +0.290]; draws +0.174 to +0.220 |
| f1 minus b1 (student agreement alone vs logprob) | +0.003 [-0.064, +0.068]; draws -0.012 to +0.003 | +0.128 [+0.041, +0.222]; draws +0.124 to +0.145 | +0.197 [+0.122, +0.279]; draws +0.148 to +0.197 |
| b1 minus a (logprob vs prompt-only predictor) | +0.310 [+0.191, +0.431] | +0.214 [+0.076, +0.341] | +0.096 [-0.009, +0.204] |
| b1 minus e (logprob vs student gap) | +0.228 [+0.122, +0.326] | +0.137 [+0.023, +0.252] | +0.041 [-0.075, +0.150] |
| b1 minus c1 (logprob vs two-draw self-consistency) | +0.209 [+0.139, +0.274] | +0.243 [+0.158, +0.323] | +0.226 [+0.153, +0.292] |
| b1 minus d (nano only) | +0.106 [+0.012, +0.207] | | |

The draw-to-draw spread of every difference is smaller than the difference itself, so the ordering is not a lucky single draw. b1 vs b2 and b4 (mode-path logprob) are within 0.02: the choice of logprob variant does not matter.

**Certified routing** (calibration 100 rows, holdout 100, 5 rotations; delta 5%, 40-level grid). Coverage = share of holdout sent to the cheap model, mean (min over rotations); mismatch = realized mismatch with E1 among routed calls, mean (max). **At the 2% and 5% budgets every signal certifies nothing on every model.** Blended $/1k includes any extra cheap call and sends the rest to E1 ($1.72 always-E1; nano $0.098, qwen $0.048 always-cheap); n = rotations that certified.

| model | signal | cov @10% | cov @15% | mismatch @15% | $/1k @15% (n) | cov @20% | mismatch @20% | $/1k @20% (n) |
|---|---|---|---|---|---|---|---|---|
| nano | b1 logprob | 0.00 | 0.63 (0.44) | 0.062 (0.114) | 0.73 (5) | 0.75 (0.65) | 0.084 (0.148) | 0.53 (5) |
| | d verbalized | 0.00 | 0.00 | - | - | 0.42 (0.00) | 0.135 (0.136) | 0.64 (3) |
| | c1 / c2 self-consistency | 0.00 | 0.00 | - | - | 0.00 | - | - |
| | e student gap | 0.00 | 0.00 | - | - | 0.00 | - | - |
| | a prompt-embedding predictor | 0.00 | 0.00 | - | - | 0.00 | - | - |
| | f1 student == cheap | 0.15 (0.00) | 0.47 (0.00) | 0.047 (0.066) | 0.48 (3) | 0.76 (0.72) | 0.034 (0.066) | 0.51 (5) |
| | f3 f1 + logprob gap | **0.53 (0.00)** | **0.76 (0.65)** | 0.049 (0.138) | **0.52 (5)** | **0.81 (0.76)** | 0.064 (0.167) | **0.43 (5)** |
| | z all-signal LR | 0.39 (0.00) | 0.69 (0.58) | 0.036 (0.066) | 0.73 (5) | 0.74 (0.60) | 0.056 (0.119) | 0.64 (5) |
| qwen3-next | b1 / b2 logprob | 0.00 | 0.00 | - | - | 0.09 / 0.27 (0.00) | 0.109 / 0.095 | 0.98 (1) / 0.59 (2) |
| | f1 student == cheap | 0.00 | 0.59 (0.00) | 0.017 (0.026) | 0.50 (4) | 0.73 (0.69) | 0.013 (0.026) | 0.51 (5) |
| | f3 | 0.00 | 0.69 (0.53) | 0.014 (0.026) | 0.59 (5) | 0.80 (0.74) | 0.074 (0.136) | 0.39 (5) |
| gpt-4.1-nano | b1 / b2 logprob | 0.00 | 0.00 | - | - | 0.00 | - | - |
| | f1 / f3 | 0.00 | 0.39 / 0.25 (0.00) | 0.031 / 0.032 | 0.64 / 0.69 (3 / 2) | 0.65 / 0.65 | 0.018 / 0.029 | 0.65 / 0.64 (5) |

Random mixing never certifies (mismatch 0.200 at every routed share on nano). Descriptive in-sample numbers (no certificate, optimistic): for nano the logprob alone could send 67% at a 5% mismatch and 80% at 10%, while the prompt-embedding predictor sends 0% at 5% and 10% and the student gap 20% / 35%. Projection if the 200-row risk-coverage curve were the true curve (optimistic), certified nano coverage at 5% budget: logprob 0.50 / 0.62 / 0.65 at n = 500 / 1000 / 2000 calibration rows; f3 0.68 / 0.70 / 0.78; f1 alone 0.00 / 0.00 / 0.76; at 10%: logprob 0.65 / 0.70 / 0.75, f3 0.80 / 0.80 / 0.80. For qwen3-next and gpt-4.1-nano the logprob alone reaches nothing useful at 5% even with 2000 rows (0.00) and 0.50 / 0.25 at 10%, while f3 reaches 0.62-0.80 from 1000 rows.

**How good must the shadow student be?** (student trained on gold labels with N Banking77 rows; AUROC for nano_A / qwen_A label == E1; nano logprob b1 = 0.894, qwen b1 = 0.788)

| student rows | student gold acc | agrees with E1 | student gap alone | f1 nano | f3 nano | f1 qwen | f3 qwen |
|---|---|---|---|---|---|---|---|
| 77 | 0.640 | 0.605 | 0.647 | 0.731 | 0.882 | 0.742 | 0.844 |
| 385 | 0.725 | 0.670 | 0.656 | 0.800 | 0.923 | 0.816 | 0.894 |
| 770 | 0.840 | 0.755 | 0.626 | 0.847 | 0.938 | 0.854 | 0.905 |
| 3080 | 0.905 | 0.820 | 0.665 | 0.894 | 0.951 | 0.907 | 0.929 |
| 9993 | 0.935 | 0.850 | 0.670 | 0.900 | 0.958 | 0.919 | 0.951 |

A 385-row student already makes f3 beat the nano logprob by 0.03 and the qwen logprob by 0.11; below that (77 rows) it still beats qwen's logprob but not nano's.

## Interpretation for the mission

- **Mission 2 (student gate).** Keep the confidence definition missions.md already uses, the top-2 gap. Entropy, max prob and kNN agreement are worse; the temperature-scaled gap is a statistical tie (+0.001 to +0.008 AUROC, no coverage gain). A learned second model is not worth building here: a prompt-embedding head loses 12-34 AUROC points to the gap, and a head on the student's own scores adds at most one point and no certified coverage. The reason is mechanical: the gap already holds the student's view of the prompt, and an embedding-only head would have to relearn where the student fails from 300-3000 labels.
- **Mission 4 (review deferral).** This is the one place a learned confidence head is justified, because it trains on a new label source (human corrections) the student's logits never saw. S15 does not contradict that; it only says that a head trained on the same teacher-agreement labels as the student adds little.
- **Mission 1 (tier router).** The user's hypothesis is right about prompt-only routers (AUROC 0.57-0.63, certify nothing, in line with S5) and right that the cheap model's token probabilities are the strongest single-model signal where the API exposes them and the model is sharp (gpt-5.4-nano 0.894). It is wrong in the sense that matters for the product: the cheap model's logits miss errors that an independent model catches. Dreaming already runs a student at every call site (it shadows the call for free), so "does the student's label equal the cheap model's label" costs nothing and is the best signal wherever logprobs are weak or absent (qwen 0.79 to 0.92, gpt-4.1-nano 0.72 to 0.92; for Anthropic-style APIs with no logprobs it is the only strong cheap signal). Adding the logprob gap as a tie-break gave the best AUROC on all three models (0.944-0.955).
- Two ways to get "more confidence" cost money and bought nothing: a second cheap draw (self-consistency 0.50-0.68, worst on the models that honour temperature 0, whose repeat draws nearly always agree) and verbalized confidence (0.78, saturated at 92-96, 12% extra cost, 11 points under the logprob).
- Money at the budgets that certify: with 100 calibration rows, f3 on nano routes 76-81% of calls to the cheap model at a 15-20% budget for $0.52 / $0.43 per 1000 calls against $1.72 (70-75% cheaper), versus $0.73 / $0.53 for the logprob alone. Budgets at or below 10% need roughly 500-1000 calibration rows (projection table); the S18 audit certificate is the lever for that, not a better signal.

## Caveats

- One task (Banking77, 77 typed labels), 200 inputs, one reference (E1, itself 6% self-inconsistent and 87-90% gold-accurate). Target = agreement with a noisy reference; results on E1b and E1==E1b inputs are within 0.04.
- The shadow student in Part 2 is trained on gold labels with 9993 rows; in Dreaming it is trained on the teacher's own outputs and may be weaker. The sensitivity table shows the advantage survives a 385-row student for nano but shrinks with a poor one.
- f2, f3, c3 and z are my combinations, not in the spec's list; treat them as exploratory (f3 is a lexicographic sort: agreement first, logprob gap inside each group). The spec's pre-listed comparison (b1 vs f1 vs a vs c vs d) is the safe reading: f1 ties b1 on nano and beats it on qwen and gpt-4.1-nano.
- Logprob gap uses only the returned top-5 alternatives with a -12 floor for unlisted rivals; gpt-5.x ignore temperature, so a nano answer is a T=1 sample and its logprob describes that draw (the repeat spread table covers this). qwen3-32b gives no label logprobs (probe above), so the second cheap model is a substitute.
- Part 1 splits re-partition the same test rows, so split sd is an optimistic noise estimate; bootstrap CIs over holdout rows are shown for split 0 only. Heads draw from the training pool unused by the student, at most 9223 rows for B77-770; the embedding head still improves with data, though it starts 0.12-0.34 below the gap.
- The 2% and 5% router columns are empty on purpose: a 100-row calibration half cannot certify them for any signal. 15-20% budgets were added to make the signals comparable; that is not a product budget.
- An earlier attempt (before this session) ran the same analysis on the laptop and was interrupted; its logs `log_p1*.txt`, `log_p2.txt` are superseded by the pod run (numbers agree to the third decimal except as noted above). The `probe_lp*.py` files are its logprob availability probes.

## Time and cost

- Pod: A5000 at $0.27/h. Main run (Part 1 five splits at 500 and 1500 calibration rows plus Part 2) 15.5 min = $0.070; repeat-noise table 3.1 min = $0.014. Total pod $0.084. Laptop: one 1-split CLINC plumbing check (about 1 s of compute after load) and a 2-call-per-provider qwen3-32b probe.
- API spend: $0.159 from the cached 2200 cheap-model calls (11 runs x 200; sum of `usage.cost`, all provider-pinned: OpenAI for nano and gpt-4.1-nano, Parasail for qwen3-next), plus less than $0.01 for logprob and provider probes; about $0.17 of the $5 cap. E1 was not re-run. `COSTS.md` has the `s15_confidence_signals` line and the two pod lines.
- Reproduce: `python smokes/pod.py --tag s15 --bundle smokes/common.py smokes/s15_confidence_signals smokes/s5_tier_router/responses.jsonl --cmd "python smokes/s15_confidence_signals/run.py all"` (add `rep` for the repeat table). Raw tables: `tables_p1_5.md`, `tables_p1_5_cal1500.md`, `tables_p2.md`, `tables_rep.md`.
