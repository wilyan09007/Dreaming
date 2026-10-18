## In plain words

S17 asked how to give a new workflow step's small student a head start, and found that LLM-written practice examples beat everything else while pre-training on other tasks did nothing for a frozen encoder. The user's side review objected that S17's pool was intent-classification data, unlike the real call sites (short records such as bank-statement lines or product titles), and S8 and S17 disagreed on whether training on sibling tasks helps at all. S26 reruns the warm-start arms on one pool of 8 public "categorize this short record" datasets, one pretraining budget (600 steps, batch 32) for both a 33M encoder and a 0.5B decoder with LoRA, and three test tasks that were never in the pool: US bank-statement descriptors, Amazon product titles, and the CLINC travel domain. On the two record tasks the answer is mostly the same as in S17. LLM-written examples help most (+0.206, +0.116, +0.047 accuracy at 1 / 3 / 10 rows per label on the record tasks, 6 of 6 cells at each k; +17 points at 1 row per label over all three tasks, 9 of 9 cells); pre-training the encoder on the pool gives +0.037 / +0.034 / +0.020 on the record tasks without the examples (5/1, 4/2, 4/2 cells), almost all of it from `amazon` (`bank_us` gets about 0), and slightly less than nothing with them; the hypernetwork stays 8 to 14 points below the examples on the record tasks (9 to 13 over all three tasks). Sibling-task pretraining helps the 0.5B decoder, which is what S8 saw: +0.057 / +0.057 / +0.107 on the record tasks (4/1, 5/1, 6/0), +0.085 to +0.102 over all three tasks, and most of the k=1 and k=3 gain comes from travel (+0.192 / +0.171). At k=1 on the record tasks the decoder's gain (+0.057) is only slightly larger than the encoder's (+0.037), and it is measured against a weak 60-step from-base control (0.244 accuracy at k=1 over all tasks). Even with transfer the decoder is 9 to 21 points behind the pre-trained encoder over all tasks, and 16 to 27 points behind the plain encoder with no pre-training at all. This is consistent with S17, not a change to it: a small encoder gain that depends on the task and goes away once the examples are added. The most important thing the run shows is about the yardstick, not the arms: on the two record tasks no arm certifies any coverage at a 5% error budget, because accuracy tops out at 66% and, with the 40-level grid, a 300-row calibration set cannot certify anything below 45% coverage (135 rows); only the travel task certifies. So at this accuracy level the certificate is an on/off switch that reads "off" for every arm, and arms can only be compared by accuracy. Run: 243 cells, 15.7 pod minutes on an A100, $0.47 of pod time plus $0.0065 of LLM calls.

## Setup

Protocol shared with the other lite smokes (frozen bge-small embeddings, 300 calibration rows, certificate at a 5% budget); everything specific to S26 is in `run.py` (docstring and code).

- **Pool (9 datasets, 15,910 rows, 228 labels; 8 datasets / 201 labels / 13,210 rows are trained on, the 9th is the early-stopping set).** Rows per label are capped at min(100, max(30, 3000 / labels in that dataset)). 0 pool rows equal an eval text.

| dataset | what it is | labels | rows in pool | role |
|---|---|---|---|---|
| dikshith | bank transaction descriptors, spending category | 8 | 800 | train |
| sumeet | Indian bank narrations (UPI, IMPS, POS), spending category | 21 | 859 | train |
| rajesh | card merchant names, spending category | 7 | 213 | train |
| tickets | English support tickets, queue | 10 | 1,000 | train |
| lezh | marketplace listings, product category | 33 | 2,915 | train |
| flip | Flipkart product names, top-level category | 15 | 1,500 | train |
| banking77 | banking support queries, intent | 77 | 2,923 | train |
| clinc_fin | CLINC banking + credit-card domains, intent | 30 | 3,000 | train |
| bitext | customer-service messages, intent | 27 | 2,700 | validation only (checkpoint / early stopping) |

- **Eval tasks (3, whole datasets never in the pool).**
  - `bank_us`: US bank-feed descriptors (`DoDataThings/us-bank-transaction-categories-v2`, a synthetic dataset), 17 labels. 1,007 merchant groups (first brand-like token) split into disjoint support / calibration / test groups (496 / 209 / 302); rows per merchant capped at 15; 300 calibration rows, 800 test rows. Support / calibration / test group overlap asserted 0 / 0 / 0.
  - `amazon`: product listings "Title | Store | Manufacturer" (`SurAyush/ProductTitle-To-Category`, 80,000-row sample, top-level department), 14 labels. 2,186 store groups split 1,073 / 432 / 681; same caps; 300 calibration, 800 test rows; overlap 0 / 0 / 0.
  - `clinc_travel`: CLINC150 travel domain, 15 labels, official splits (1,500 support pool, 300 calibration, 450 test). No group field. CLINC is also the source of one pool dataset (clinc_fin), so this is the task with the closest sibling.
  - The dataset and merchant / store hold-out applies only to `bank_us` and `amazon`. `clinc_travel` is not a held-out dataset: it comes from the same CLINC150 source as the pool's `clinc_fin`, uses the official splits, and is not held out by group. As in S17, intents can overlap across datasets (for example `exchange_rate`, present in Banking77 in the pool and in CLINC travel in the eval), which S17 flagged and which is not controlled here.
  - Split by merchant or store (bank_us, amazon), never by row. The split is made once (fixed seed) and shared by all arms and draws; the draws (seeds 0-2) change only which k rows per label are taken from the support groups (and the decoder's initialisation).
- **Arms, k = 1 / 3 / 10 rows per label, 3 draws, 3 tasks = 27 cells per arm, 243 in all.**
  1. Baseline: frozen bge-small + LogisticRegression (C=10) on the k rows.
  2. Synthetic prior: the same, plus 10 LLM-written examples per label (ASI:One asi1-mini).
  3. Hypernetwork: S17 set-transformer (1.35M parameters; label-description embedding + mean of the k rows in, cosine head out), 800 tasks sampled from the 8 training datasets, shots 1/2/3/5/10, early stopping on episodes from the held-out bitext dataset. One forward pass, no fitting on the rows.
  4. Encoder pre-trained on the pool: bge-small fully fine-tuned (lr 3e-5, batch 32, 600 steps, 201-way head thrown away), then frozen + LR. 4b adds the synthetic prior.
  5. Shared LoRA: Qwen2.5-0.5B with a sequence-classification head and LoRA (r=16, alpha 32, q/k/v/o projections), pre-trained on the same 13,210 rows with the same 600 steps and batch 32 (201-way head), then on each cell fine-tuned for 60 steps (batch 32, lr 2e-4, S8 recipe) from the shared adapter with a fresh head. 5c is the same fine-tune from base (the no-transfer control). 5b / 5d add the synthetic prior to 5 / 5c. The prompt is `Step: <step name>\nRecord: <text>\nLabel:` with a per-dataset step name.
  - Pairing: arm 4 is paired with 1 and 4b with 2 (encoder transfer); 5 with 5c and 5b with 5d (decoder transfer). Checkpoint choice for 4 and 5 uses the held-out bitext dataset only (3-shot LR for the encoder; 3-shot 60-step fine-tune for the decoder), never the eval tasks.
- **Description text.** The label-description cache was regenerated in this session (new LLM text), and the synthetic examples for the three eval tasks were generated fresh (16 asi1-mini calls). Description-based arms (2, 3, 4b, 5b, 5d) are therefore not byte-identical to S17's. Arms 1 and 4 do not use the text; the baseline on `clinc_travel` reproduces S17 to the digit (0.870 accuracy, 0.401 coverage at k=1), which checks the shared pipeline.
- **Metrics.** Accuracy on the test rows; certified coverage at a 5% budget (`common.certify`: 40-level grid, Clopper-Pearson, delta 0.05, confidence = top-2 gap) fixed on the 300 calibration rows and applied to the test rows; realized error on the answered rows.

## Results

All numbers recomputed from `results.json` (243 records) by `report_s26.py`; its full output is `report_s26_out.md`. "+-" is the s.d. over the 3 draws of the task-averaged value, so it does not contain task-to-task spread, calibration resampling or pretraining-seed variation (each base was pre-trained once). "[a/b]" in paired differences counts cells where arm A is better / worse.

### Headline: mean over the 3 tasks

| arm | acc k=1 | acc k=3 | acc k=10 | cov k=1 | cov k=3 | cov k=10 |
|---|---|---|---|---|---|---|
| 1 baseline | 0.513 +- 0.014 | 0.629 +- 0.005 | 0.724 +- 0.013 | 0.134 +- 0.116 | 0.328 +- 0.003 | 0.332 +- 0.003 |
| 2 synthetic prior | 0.683 +- 0.005 | 0.709 +- 0.004 | 0.757 +- 0.009 | 0.318 +- 0.004 | 0.330 +- 0.003 | 0.332 +- 0.000 |
| 3 hypernetwork | 0.574 +- 0.031 | 0.616 +- 0.009 | 0.625 +- 0.009 | 0.070 +- 0.121 | 0.216 +- 0.050 | 0.235 +- 0.013 |
| 4 encoder pre-trained | 0.557 +- 0.016 | 0.654 +- 0.014 | 0.736 +- 0.009 | 0.262 +- 0.045 | 0.328 +- 0.004 | 0.333 +- 0.001 |
| 4b encoder pre-trained + synth | 0.664 +- 0.004 | 0.697 +- 0.001 | 0.746 +- 0.005 | 0.305 +- 0.006 | 0.323 +- 0.010 | 0.331 +- 0.002 |
| 5 shared LoRA | 0.346 +- 0.030 | 0.487 +- 0.031 | 0.645 +- 0.011 | 0.000 +- 0.000 | 0.110 +- 0.105 | 0.309 +- 0.013 |
| 5c LoRA from base | 0.244 +- 0.050 | 0.392 +- 0.048 | 0.560 +- 0.052 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.244 +- 0.064 |
| 5b shared LoRA + synth | 0.627 +- 0.008 | 0.668 +- 0.017 | 0.714 +- 0.015 | 0.279 +- 0.029 | 0.298 +- 0.006 | 0.324 +- 0.005 |
| 5d LoRA from base + synth | 0.540 +- 0.031 | 0.561 +- 0.041 | 0.633 +- 0.045 | 0.129 +- 0.119 | 0.083 +- 0.144 | 0.284 +- 0.052 |

Read the coverage columns with care: only `clinc_travel` ever certifies, so the task mean is capped at 1/3 (0.333) and every coverage figure in this table is travel's coverage divided by three.

### Per task: accuracy

| arm | bank_us k=1 | k=3 | k=10 | amazon k=1 | k=3 | k=10 | clinc_travel k=1 | k=3 | k=10 |
|---|---|---|---|---|---|---|---|---|---|
| 1 baseline | 0.335 | 0.443 | 0.573 | 0.335 | 0.469 | 0.607 | 0.870 | 0.975 | 0.993 |
| 2 synthetic prior | 0.510 | 0.560 | 0.627 | 0.573 | 0.583 | 0.646 | 0.967 | 0.984 | 0.998 |
| 3 hypernetwork | 0.400 | 0.462 | 0.468 | 0.483 | 0.518 | 0.530 | 0.839 | 0.866 | 0.878 |
| 4 encoder pre-trained | 0.340 | 0.443 | 0.571 | 0.405 | 0.537 | 0.648 | 0.925 | 0.981 | 0.987 |
| 4b + synth | 0.481 | 0.542 | 0.608 | 0.553 | 0.575 | 0.640 | 0.957 | 0.976 | 0.991 |
| 5 shared LoRA | 0.194 | 0.299 | 0.468 | 0.211 | 0.339 | 0.507 | 0.634 | 0.822 | 0.961 |
| 5c LoRA from base | 0.133 | 0.244 | 0.374 | 0.158 | 0.280 | 0.386 | 0.442 | 0.651 | 0.920 |
| 5b shared LoRA + synth | 0.441 | 0.506 | 0.557 | 0.515 | 0.551 | 0.606 | 0.925 | 0.948 | 0.979 |
| 5d LoRA from base + synth | 0.361 | 0.371 | 0.445 | 0.431 | 0.478 | 0.495 | 0.829 | 0.835 | 0.960 |

The best accuracy any arm reached on `bank_us` or `amazon`, in any single cell, was 0.659.

### Per task: certified coverage at 5% (bank_us and amazon are 0.000 for every arm, k and draw)

| arm | clinc_travel k=1 | k=3 | k=10 | travel cells certified (of 9) | realized error on answered rows, certified cells |
|---|---|---|---|---|---|
| 1 baseline | 0.401 | 0.985 | 0.996 | 8 | 0.007 / 0.019 / 0.006 |
| 2 synthetic prior | 0.954 | 0.990 | 0.996 | 9 | 0.007 / 0.011 / 0.000 |
| 3 hypernetwork | 0.209 | 0.648 | 0.705 | 7 | 0.011 / 0.014 / 0.017 |
| 4 encoder pre-trained | 0.787 | 0.983 | 0.998 | 9 | 0.004 / 0.010 / 0.012 |
| 4b + synth | 0.915 | 0.970 | 0.993 | 9 | 0.005 / 0.007 / 0.006 |
| 5 shared LoRA | 0.000 | 0.330 | 0.927 | 5 | - / 0.004 / 0.009 |
| 5c LoRA from base | 0.000 | 0.000 | 0.733 | 3 | - / - / 0.010 |
| 5b shared LoRA + synth | 0.837 | 0.895 | 0.973 | 9 | 0.012 / 0.021 / 0.008 |
| 5d LoRA from base + synth | 0.386 | 0.250 | 0.851 | 6 | 0.003 / 0.027 / 0.004 |

Realized error (k = 1 / 3 / 10, mean over the certified cells) never exceeded the 5% budget; the worst single certified cell was 3.8% (arm 5b). Certified cells in total: 65 of 243, all on `clinc_travel`; none on `bank_us` or `amazon`.

### Paired differences in accuracy (same task / k / draw; mean +- s.d. over cells [A better / A worse])

All 3 tasks, 9 cells per k:

| A - B | k=1 | k=3 | k=10 |
|---|---|---|---|
| 4 - 1 (encoder transfer) | +0.043 +- 0.038 [8/1] | +0.025 +- 0.036 [6/3] | +0.012 +- 0.024 [4/5] |
| 4b - 2 (encoder transfer, with synth) | -0.020 +- 0.011 [0/9] | -0.012 +- 0.010 [1/7] | -0.011 +- 0.010 [0/9] |
| 5 - 5c (decoder transfer) | +0.102 +- 0.094 [7/1] | +0.095 +- 0.070 [8/1] | +0.085 +- 0.067 [8/1] |
| 5b - 5d (decoder transfer, with synth) | +0.086 +- 0.042 [9/0] | +0.107 +- 0.065 [9/0] | +0.081 +- 0.064 [9/0] |
| 2 - 1 (synthetic prior) | +0.170 +- 0.066 [9/0] | +0.080 +- 0.054 [8/1] | +0.033 +- 0.023 [9/0] |
| 3 - 1 (hypernetwork vs baseline) | +0.061 +- 0.089 [7/2] | -0.013 +- 0.077 [5/4] | -0.099 +- 0.024 [0/9] |
| 3 - 2 (hypernetwork vs synthetic prior) | -0.109 +- 0.044 [0/9] | -0.093 +- 0.034 [0/9] | -0.132 +- 0.023 [0/9] |
| 4 - 5 (encoder-transfer vs decoder-transfer) | +0.210 +- 0.076 [9/0] | +0.167 +- 0.053 [9/0] | +0.090 +- 0.056 [9/0] |
| 4b - 5b (same, with synth) | +0.037 +- 0.024 [8/1] | +0.029 +- 0.018 [8/0] | +0.032 +- 0.028 [8/0] |
| 1 - 5c (plain encoder vs plain decoder) | +0.269 +- 0.138 [9/0] | +0.237 +- 0.084 [9/0] | +0.164 +- 0.083 [9/0] |

Record tasks only (`bank_us` + `amazon`, 6 cells per k), the S17 question:

| A - B | k=1 | k=3 | k=10 |
|---|---|---|---|
| 4 - 1 | +0.037 +- 0.046 [5/1] | +0.034 +- 0.042 [4/2] | +0.020 +- 0.026 [4/2] |
| 4b - 2 | -0.024 +- 0.010 [0/6] | -0.013 +- 0.011 [1/5] | -0.013 +- 0.012 [0/6] |
| 5 - 5c | +0.057 +- 0.050 [4/1] | +0.057 +- 0.037 [5/1] | +0.107 +- 0.067 [6/0] |
| 5b - 5d | +0.081 +- 0.032 [6/0] | +0.104 +- 0.061 [6/0] | +0.112 +- 0.054 [6/0] |
| 2 - 1 | +0.206 +- 0.043 [6/0] | +0.116 +- 0.012 [6/0] | +0.047 +- 0.013 [6/0] |
| 3 - 1 | +0.106 +- 0.057 [6/0] | +0.035 +- 0.033 [5/1] | -0.091 +- 0.026 [0/6] |
| 3 - 2 | -0.100 +- 0.047 [0/6] | -0.081 +- 0.034 [0/6] | -0.138 +- 0.027 [0/6] |
| 4 - 5 | +0.170 +- 0.042 [6/0] | +0.171 +- 0.058 [6/0] | +0.122 +- 0.034 [6/0] |

Where the transfer gain sits (mean accuracy difference over 3 draws, k = 1 / 3 / 10): encoder transfer (4 - 1) `bank_us` +0.005 / +0.000 / -0.002, `amazon` +0.069 / +0.067 / +0.042, `clinc_travel` +0.056 / +0.007 / -0.005; decoder transfer (5 - 5c) `bank_us` +0.060 / +0.055 / +0.093, `amazon` +0.053 / +0.059 / +0.120, `clinc_travel` +0.192 / +0.171 / +0.041. The encoder gain is concentrated on `amazon` and, at k=1, on `clinc_travel`; the task with the three most similar pool datasets (`bank_us`, next to dikshith, sumeet and rajesh) gets nothing from the encoder.

Paired differences in certified coverage are in `report_s26_out.md`. They carry little beyond the accuracy table because only `clinc_travel` certifies: on that task encoder transfer is +0.386 +- 0.224 at k=1 (3/0), and about 0 at k=3 and k=10, where the baseline is already at 0.98 to 1.00; the synthetic prior is +0.553 +- 0.340 at k=1 (3/0) and +0.005 / +0.001 after; decoder transfer is +0.330 / +0.194 at k=3 / k=10 (arm 5 against 5c) and +0.451 / +0.645 / +0.121 with the synthetic prior (5b against 5d, 3/0 in every k).

### Compared with S17 and S22 on `clinc_travel` (the only task all three share)

Accuracy / certified coverage, mean over 3 draws, k = 1 / 3 / 10. S17 columns are its `clinc/travel` cells.

| arm | S17 | S26 |
|---|---|---|
| baseline | 0.870 / 0.401, 0.975 / 0.985, 0.993 / 0.996 | 0.870 / 0.401, 0.975 / 0.985, 0.993 / 0.996 |
| synthetic prior | 0.993 / 0.982, 0.998 / 0.999, 0.996 / 0.995 | 0.967 / 0.954, 0.984 / 0.990, 0.998 / 0.996 |
| encoder pre-trained on pool | 0.949 / 0.873, 0.983 / 0.979, 0.993 / 0.991 | 0.925 / 0.787, 0.981 / 0.983, 0.987 / 0.998 |
| encoder + synth | 0.986 / 0.975, 0.991 / 0.975, 0.992 / 0.999 | 0.957 / 0.915, 0.976 / 0.970, 0.991 / 0.993 |
| hypernetwork (S17: 800 tasks, stop on TREC) | 0.933 / 0.729, 0.943 / 0.852, 0.944 / 0.846 | 0.839 / 0.209, 0.866 / 0.648, 0.878 / 0.705 |

The baseline is identical (nothing in it changed), which validates the shared pipeline. The description-based arms are a few points lower than S17 on travel; the LLM text was regenerated, the S26 pool differs from S17's (it holds all 77 Banking77 labels, 2,923 rows, but only 30 CLINC labels, from the banking and credit-card domains, against S17's 120 CLINC labels and 41 Banking77 labels), and the S26 encoder trained on 13k rows for 600 steps of 32 instead of 24k rows for up to 1,000 steps of 64, so the gap cannot be attributed to any one of these. Encoder gain on travel at k=1 in both runs: S17 +0.079 accuracy (0.949 vs 0.870) and +0.472 coverage (0.873 vs 0.401); S26 +0.056 and +0.386. So the one task both runs share shows a sizeable encoder gain in both, larger in S17. S17's "no encoder gain" was its 5-task mean (0.795 vs 0.802 at k=1).

S22 on travel only (same definition as here): k=1 accuracy / coverage for the shared-pool LoRA 0.764 / 0.135 and from-base LoRA 0.572 / 0.000, a transfer gain of +0.192; S26 has 0.634 / 0.000 and 0.442 / 0.000, also +0.192. The transfer gain is identical; both S26 arms are about 13 points lower, which fits 60 against 300 fine-tune steps (S22 also used max length 64 against 80 here and the prompt `Query:` against `Record:`). S22 calibrated on 300 train-split rows rather than the CLINC validation split, so its coverage is not the same quantity as here.

### Pool and pre-training details

| item | encoder (arm 4) | decoder (arm 5) | hypernetwork (arm 3) |
|---|---|---|---|
| trained on | 201 labels, 13,210 rows | the same | 800 tasks (5-15 labels each) from the same 201 labels |
| trainable parameters | 33.36M (full fine-tune) | LoRA r=16 (about 2.2M) + head, 0.5B base frozen | 1.35M |
| steps x batch (truncation) | 600 x 32 (48 tokens) | 600 x 32 (80 tokens) | 100 steps used (best), stopped at 500 |
| selection signal | 3-shot LR accuracy on bitext tasks | 3-shot, 60-step fine-tune accuracy on bitext tasks | episode accuracy on bitext tasks |
| curve (step: value) | 0: 0.9379, 200: 0.9382, 400: 0.9356, 600: 0.9357 | 0: 0.7625 (from base), 200: 0.810, 400: 0.8275, 600: 0.820 | 100: 0.868, 200: 0.857, 300: 0.838, 400: 0.810, 500: 0.785 |
| chosen | step 200 | step 400 | step 100 |
| training loss at the end | 1.12 (mean of the last 50 steps) | 0.39 (a single batch at step 600; logged values 1.18 / 0.91 / 0.39, not comparable with the encoder's mean) | 0.53 |
| training time (A100 / 2 CPU threads) | 31 s | 147 s | 23 s |

- The encoder's validation signal did not move (0.938 at the start, 0.938 at the chosen step 200), because bitext intents are already near the ceiling for 3-shot LR on frozen embeddings; the "pre-trained" encoder used here is a lightly changed copy of bge-small, and the +4 points it gives at k=1 on `amazon` and `clinc_travel` (not on `bank_us`) were not predicted by the validation set.
- The decoder's validation signal moved +6.5 points (0.7625 to 0.8275) and was 0.8 points lower at step 600 (0.8275 to 0.820, 3 of 400 query rows, noise level); the hypernetwork peaked at its first checkpoint and then declined by 8 points while its loss stayed near 0.51 to 0.53, the same early over-fit as S17. Held-out bitext episode accuracy of the best hypernetwork checkpoint (k=1 / 3 / 10: 0.831 / 0.882 / 0.892) is 4.5 to 10.7 points above plain description-cosine on the same episodes (0.786 / 0.790 / 0.785).

### Timings

| item | measured |
|---|---|
| Pod | A100 80GB PCIe, 128 CPUs; script wall 763 s (12.7 min); 15.7 pod minutes including boot and bundle |
| prep (7 dataset downloads, LLM descriptions + synthetic examples, embedding 15.9k pool rows and 3 eval sets) | 142 s |
| encoder pre-training / decoder pre-training / hypernetwork | 31 s / 147 s / 23 s (run in parallel) |
| arms 1, 2 / 3 / 4, 4b evaluation | finished at 148 s / 172 s / 187 s on the script clock (prep ended at 142 s), all under 1 s of compute per cell |
| LoRA arms (5, 5b, 5c, 5d), 108 runs on 3 workers sharing the GPU | 11.9 s per run (11.6 to 11.9 across arms); the three workers finished at 691 s, 724 s and 763 s |
| arms 1-4b per cell | 0.02 to 0.05 s |

## Caveats

- **Why nothing certifies on `amazon` and `bank_us`.** Accuracy is the first reason: the best arm reached 0.66 on a 14- or 17-way task (baseline 0.34-0.61 over k, synthetic prior 0.51-0.65). The second is the certificate itself. `common.certify` tests only 40 coverage levels, round(300 j / 40) rows, each with delta 0.05 / 40 = 0.00125, and a 5% budget. With 300 calibration rows the first passing level is 135 rows (0.45) with zero errors, 180 rows (0.60) with one error, 218 (0.73) with two, 255 (0.85) with three, 285 (0.95) with four; five errors never pass (last grid table of `report_s26_out.md`; the table just above it is the same Clopper-Pearson arithmetic without the grid and understates the minimum, 131 rows / 44%). The smallest calibration coverage of any certified cell in the run was 0.45, and the smallest test coverage 0.36. So the certificate is all-or-nothing at this calibration size: it returns either nothing or a cutoff covering about half the traffic or more. A task where the top-2-gap ranking does not isolate 45% of rows with about zero errors gets zero, and the metric cannot see the difference between a 0.34 and a 0.65 student. A larger calibration set does not remove the floor in rows but lowers it as a share of traffic: with 1,000 rows the zero-error floor is 150 rows, 15% (200 / 20% with one error). That would let a mid-accuracy student certify a small slice, but it is not tested here.
- **What this implies for the certificate at these accuracy levels.** Overall accuracy does not decide certification; whether the ranking isolates at least 45% of calibration rows with about zero errors does. In this run nothing certified below about 0.83 accuracy (lowest certified cell 0.829) and at 0.66 or below, the record tasks, nothing does. Between 0.83 and 0.89 it is mixed: the hypernetwork at 0.866 (travel, k=3) certified in 3 of 3 draws with 0.648 coverage, arm 5 at 0.822 (k=3) in 2 of 3, while arm 5d at 0.887 (k=3, seed 0) and the baseline at 0.840 (k=1, seed 1) did not. Two consequences for the plan. First, the cost-saving claim "the student answers X% of traffic at 5% error" cannot be made for record-style steps at these accuracies, and how many rows per label would be needed was not measured (accuracy was still rising at k=10 on both tasks: +13 to +14 points from k=3 to k=10 for the baseline). Second, for comparing arms on such tasks use accuracy, or a risk-coverage curve at a larger budget; the 5% coverage column saturates at zero. Sending everything to the teacher is the certificate working as designed, not failing.
- **Why `amazon` and `bank_us` may be hard beyond sample size (not diagnosed).** `bank_us` is a synthetic dataset with labels such as Shopping, Groceries, Subscription and Transfer that overlap; the `amazon` labels are top-level departments (Home & Kitchen, Tools & Home Improvement, Patio Lawn & Garden, Health & Household, Beauty & Personal Care) that also overlap. A ceiling set by label ambiguity is plausible and was not checked (no per-label confusion was saved). The hold-out is by merchant or store, which removes the near-duplicates that make row-split numbers look better.
- **Single split, single pre-training seed.** The merchant or store split is fixed once; the 3 draws change only the support rows (and decoder initialisation). Each base was pre-trained once, so "transfer" differences (arm 4 vs 1, arm 5 vs 5c) include pre-training noise that the s.d. does not show. With 9 paired cells per k, treat differences below about 0.03 as unresolved; the 3 tasks x 3 draws are not independent (draws share the calibration and test rows of a task). Paired counts such as 9/0 are the more robust reading.
- **The encoder is barely changed by pre-training.** The checkpoint selected (step 200) was indistinguishable from the base on the validation set; later checkpoints were 0.2 to 0.3 points lower. The encoder arm therefore tests a light adaptation, not a strongly specialised encoder; a larger pool or a harder validation dataset might give a different picture. The same stands for the S17-style recipe (full fine-tune with lr 3e-5); no lr or step sweep.
- **Equal pretraining budget is equal in steps, rows and batch, not in compute.** Truncation also differs: 48 tokens for the encoder, 80 for the decoder, 128 at embedding time. The decoder is 15 times larger (0.5B against 33M) and took 4.7 times the wall time (147 s against 31 s); only a LoRA adapter (about 2.2M parameters) is trained in it. The decoder's pre-training validation was highest at step 400 and within noise at 600, so more steps might not help, but a longer or higher-rank run, or a larger fine-tune (60 steps here), was not tried. The decoder fits a new 14- to 17-way head from scratch at every cell, which at k=1 means 14 to 17 rows; this hurts it more than a logistic regression on embeddings.
- **Sibling definition.** Each record task has pool datasets of the same shape (bank descriptors: dikshith, sumeet, rajesh; product titles: lezh, flip) but with different label sets, languages (sumeet is Indian UPI/IMPS text) or generators; `clinc_travel` has one same-family dataset (clinc_fin). The size of "transfer" depends on how close the siblings are, and with three eval tasks the dependence cannot be separated from task identity: the decoder's gain is largest on the task with the same-family sibling (travel, +19 points at k=1) and the encoder's gain is largest on `amazon`, not on `bank_us`.
- **Hypernetwork.** One recipe from S17 and one stopping signal; it is trained on 201 labels, fewer than S17's 248, and with only two CLINC / banking intent sets among them, and it is clearly worse on travel than S17's (0.839 against 0.933 at k=1). The best checkpoint is the first one checked (step 100), so the true optimum may be earlier. Using its output as a prior for a fitted classifier was not tested.
- **LLM text.** A single regeneration of the descriptions and examples; the travel synthetic prior is 2.6 points of accuracy and 2.8 points of coverage lower at k=1 than S17's. Whether that is generation noise or the effect of a different cache was not measured (no repeat of the generation).
- **Decoder-transfer control is weak.** Arm 5c fine-tunes from base for 60 steps and reaches only 0.244 at k=1 on 14- to 17-way tasks (S22's 300-step from-base LoRA reached 0.572 on travel), so part of the 5 - 5c gain reflects the short fine-tune, not only the pretraining. One pretraining seed per base, one fine-tune length.
- **Selection.** Arm 6 (the better of 4b and 5b by coverage over k, chosen on the same tasks) is in `summary.md` and is 4b; with every coverage but travel's at zero, it carries no more than the accuracy table does.
- **Data mismatch with S17.** The task set differs (3 tasks, of which two are record datasets and one is S17's travel task; S17 had 5 intent tasks), so the S17 numbers in the headline table are not comparable; only the travel comparison above is.

## Verdict

- **Does sibling-task transfer change the S17 verdict on structurally similar record tasks? No.** The warm start of choice remains the LLM-written examples on the plain encoder: +0.170 / +0.080 / +0.033 accuracy over the baseline at k = 1 / 3 / 10 (9/0, 8/1, 9/0 cells), and +0.206 / +0.116 / +0.047 on the record tasks alone (6/0 at each k). Pre-training the encoder on 8 sibling datasets gives +0.037 / +0.034 / +0.020 on the record tasks without the examples (5/1, 4/2, 4/2), almost all from `amazon` (+0.069 / +0.067 / +0.042; `bank_us` +0.005 / 0.000 / -0.002), and -0.024 / -0.013 / -0.013 with them (0/6, 1/5, 0/6). Each base was pretrained once. This is consistent with S17: a small encoder gain that depends on the task and disappears once the examples are added. S17's "no gain" was its 5-task mean; on travel, the one shared task, S17's encoder gained more than S26's (k=1: +0.079 accuracy / +0.472 coverage against +0.056 / +0.386). It does not change the choice because the examples give several times the gain. Nothing certifies on the record tasks, so S26 cannot confirm or refute the certified-coverage side of S17 for them.
- **Which base transfers better at equal pre-training budget, 33M encoder or 0.5B decoder? Depends on what is meant.** Gain from the pool: the decoder gains more (+0.102 / +0.095 / +0.085 accuracy, 7/1, 8/1, 8/1 cells, and +0.086 / +0.107 / +0.081 with the examples, 9/0 at every k) than the encoder (+0.043 / +0.025 / +0.012). This is the S8 effect and S22's transfer gain on travel again (+0.192 at k=1 in both), and it points to a resolution of the S8-vs-S17 contradiction: a decoder starts from a place where sibling data is worth more than it is to a frozen embedding. Hedge: one pretraining seed per base, one fine-tune length, and the gain partly reflects the weak 60-step from-base control (0.244 at k=1); on the record tasks the decoder gain at k=1 is +0.057 against the encoder's +0.037. Result after transfer: the encoder with pool pre-training beats the decoder with pool pre-training by +0.210 / +0.167 / +0.090 accuracy (9/0 at every k), and by +0.037 / +0.029 / +0.032 with the examples (8/1, 8/0, 8/0). The plain, untrained encoder already beats the transferred decoder (baseline 0.513 / 0.629 / 0.724 against 0.346 / 0.487 / 0.645). Certified coverage on travel follows (encoder 0.787 / 0.983 / 0.998, decoder 0.000 / 0.330 / 0.927). Use the 33M encoder with a linear head; the decoder route stays closed unless it is given far more rows or a better recipe than tested here.
- **Does the synthetic prior still dominate? Yes on accuracy, and it ties the best on the one task that certifies.** The prior is best or within about 1 point of best in all 9 task x k cells of accuracy; on `clinc_travel` at k=1 it certifies 0.954 of traffic against 0.401 for the baseline (3/0 cells), and from k=3 it equals the baseline (0.990 / 0.996 against 0.985 / 0.996). On the record tasks it adds +0.206 / +0.116 / +0.047 accuracy but no certified coverage; the best single cell on these tasks is 0.659 (arm 4, `amazon`, k=10, one draw), but the means are 0.648 for arm 4 against 0.646 for the prior (the prior's best cell in the same setting is 0.6575): a tie.
- **Is the hypernetwork still behind? Yes.** It is below the synthetic prior in 27 of 27 cells at all three k (-0.109 / -0.093 / -0.132 accuracy, 0/9 at each k), below the baseline at k=10 (-0.099, 0/9), roughly level with it at k=3 (-0.013, 5/4), and ahead of it only at k=1 (+0.061, 7/2; +0.106 on the record tasks, 6/0), where the examples are another 0.1 higher. On travel it certifies 0.209 / 0.648 / 0.705 against 0.401 / 0.985 / 0.996 for the baseline. Its accuracy is flat in k (0.574 / 0.616 / 0.625), the same early plateau as S17.
- **Action for the plan.** Keep schema -> LLM-written examples -> ordinary encoder student as the day-zero warm start; do not pre-train the encoder or the decoder on sibling tasks for it, and do not build the hypernetwork. Treat the 5% certified-coverage readout as unusable for steps where the student is below about 0.83 accuracy (in this run nothing certified below that; the two record tasks, at 0.66 or below, never did): track accuracy and a risk-coverage curve there, and expect to need many more labelled rows or an easier label scheme before the student can answer alone. Unrun and worth running if this matters: k = 30 / 100 on the record tasks, a larger calibration set (1,000 rows, zero-error floor 150 rows), and per-label confusion to check how much of the 0.66 ceiling is label ambiguity.

## Time and cost

| item | measured |
|---|---|
| Pod | 1 launch, A100 80GB PCIe, 15.7 pod minutes including boot (script wall 12.7 min), $0.468 (logged by `pod.py` as `pod_s26`); the 52-minute budget was not used |
| LLM spend | $0.0065 (16 ASI:One asi1-mini calls: one ping, 9 description calls for the 7 other pool datasets plus `bank_us` and `amazon` (Banking77 and CLINC descriptions, travel included, come from the shared regenerated cache), 6 calls of synthetic examples for the 46 eval labels at 10 each); cap was $0.50; logged in `COSTS.md` as `s26_pool_transfer` |
| Total | about $0.47 and 15.7 pod minutes for 243 cells, including both pre-trainings and the hypernetwork |

Files: `run.py` (all code, not modified), `report_s26.py` (recomputes every table here from `results.json`), `report_s26_out.md` (its output), `results.json` (243 records), `results_*.json` (per-stage records and pre-training curves), `summary.md` (the run's own aggregate), `prep_info.json`, `responses.jsonl` (LLM cache), `logs/` (stage logs), and `../pod_log_s26.txt`.


## Revision

Changes made after review of the first version; tables were already correct (rerunning `report_s26.py` reproduces them and `summary.md`), the text was not.

- **S17's encoder result.** The first version said S17 found no encoder gain and S26 found a few-point one. S17's "no gain" was its 5-task mean; on travel S17's encoder gained more than S26's (k=1: +0.079 accuracy / +0.472 coverage vs +0.056 / +0.386). On the record tasks the S26 gain (+0.037 / +0.034 / +0.020) comes almost entirely from `amazon`; `bank_us` gets about 0, with one pretraining seed. The verdict now says S26 is consistent with S17.
- **Certification threshold.** "Roughly 90% accuracy" is withdrawn. Certified cells go down to 0.829 overall accuracy, the hypernetwork at 0.866 certifies 3 of 3, while 0.887 and 0.840 cells did not. What decides it is whether the top-2-gap ranking isolates at least 45% of calibration rows with about zero errors. Wording is now: nothing certified below about 0.83; at 0.66 or below nothing does.
- **Grid arithmetic.** Minimum rows with the 40-level grid: 135 (0.45) with zero errors, 180 (0.60) with one; 150 rows (15%) with zero errors at 1,000 calibration rows. The earlier table (131 / 176 / 13%) is the same bound without the grid; `report_s26.py` now prints both and labels them.
- **S22 comparison.** Now on travel only: S22 0.764 / 0.135 vs 0.572 / 0.000 (+0.192), S26 0.634 / 0.000 vs 0.442 / 0.000 (+0.192). The gain is identical; the levels differ by about 13 points (60 vs 300 fine-tune steps, max length, prompt). The garbled sentence is replaced.
- **Record-task vs all-task numbers.** The plain-words paragraph now gives record-task figures (+0.206 examples, hypernetwork 8 to 14 below, decoder +0.057 / +0.057 / +0.107) beside the all-task ones.
- **Decoder-transfer wording** is hedged: one pretraining seed per base, one fine-tune length, and a weak 60-step from-base control.
- **Pool and leakage.** The S26 pool has all 77 Banking77 labels (2,923 rows) and 30 CLINC labels. The cross-dataset `exchange_rate` overlap is noted. The dataset and merchant / store hold-out applies only to `bank_us` and `amazon`; `clinc_travel` shares its source with `clinc_fin`.
- **Minor.** Best-cell 0.659 vs prior mean replaced by means (0.648 vs 0.646, a tie); hypernetwork margin 4.5 to 10.7; timings are script-clock exit times; decoder loss 0.39 is one batch; the 0.8275 to 0.820 drop is noise (3 of 400 rows); encoder truncation 48 tokens added to the budget differences; description call counts corrected (9 description calls, 6 synthetic-example calls).
