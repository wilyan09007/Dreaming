## In plain words

The question: can we tell, before paying for the expensive model, whether a cheap model's answer will match what the expensive model (E1 = gpt-5.4, low effort) would say, by looking inside the cheap model (a linear probe on its hidden states)? And does that beat the signals S15 found: the cheap model's logprob gap, the student's top-2 gap, and "the free student says the same label as the cheap model" (f1)?

Answer, on this setup: **the probe is a real signal (AUROC 0.775 against the cheap model's own logprob gap at 0.61) but it is far weaker than f1 (0.950), and adding it to f1 does not change the ranking (0.959 vs 0.950, interval includes 0).** The probe beat a linear probe on the student's embedding (f2, 0.706) by +0.069, so the hidden state carries information the 384-d sentence embedding does not, but agreement between two independent predictors carries much more. Nothing certifies at 5-20% budgets on the 200 rows, for a reason that is not the signal: the chosen cheap model (Qwen2.5-0.5B-Instruct) matches E1 on only 27% of inputs, so a 100-row calibration half contains about 27 positives and cannot hold the 30 clean rows a 20% budget needs. On a pool of 1200 rows (gold target) f1 certifies 28% coverage at 20% budget and 13.5% at 15%; the probe alone certifies nothing. Used only as a tie-break inside the f1 = 1 block, the probe does help certified coverage at 15% (23.4% vs 17.3% for the logprob-gap tie-break and 13.5% for f1), but that is not worth a GPU forward pass at this match rate (see Revision).

## Setup

- **Cheap model:** Qwen/Qwen2.5-0.5B-Instruct, bf16, on a RunPod RTX A5000 (`pod.py`; extraction 3.7 min, 66 ms median batch-1 greedy latency). Prompt, system prompt and parser are S5's (`data.json`). Hidden states at layers 6, 10, 13, 17, 20, 24 of 24 (last token and mean over the query tokens), float16; greedy answer (16 tokens) with token logprobs and top-2 gaps, plus two T=0.7 samples. `extract.py` now saves results after every input set.
- **Rows:** the 200 Banking77 inputs of S5/S15 (target: cheap label == E1, from S5's cached responses, no LLM call made); 1000 train rows (proxy target: cheap label == gold) and 1000 new test rows (gold target only, E1 was not queried: no new API calls); RouterBench 2000-row subset of S5's sample (gpt-3.5 vs gpt-4 scores already in the data). Qwen's answers are parsed with S5/S15's nearest-label parse (`difflib`), as for every cheap model there; 17% of its answers are not an exact label (see Revision).
- **Probe (arm "probe"):** logistic regression on standardized states, (layer, C) and pool (last vs mean) picked by 3-fold inner AUROC on the 1000 train rows only (chosen: last-token, layer 20, C 0.01, inner AUROC 0.78), trained with the proxy target and applied to the 200 rows with the real target, so no label of the 200 is ever used. A second version trained 5-fold on the 200 is reported for AUROC only.
- **Shadow student:** S1 recipe as in S15: frozen BAAI/bge-small-en-v1.5 + LogisticRegression(C=10) on all 9993 Banking77 train rows (gold acc 0.935 on the 200; agrees with E1 on 0.850).
- **New arms:** f1 = student label == cheap label; f3 = f1 then the cheap logprob gap (S15's definition; for Qwen the gap is the minimum token top-2 gap of the answer); f2 = logistic regression on the bge student embedding predicting the same target, trained the same way as the probe (proxy-trained on the 1000 train rows; 5-fold version AUROC only); combinations "f1 then probe", "f1 then f2", and a stack LR on [probe, gap, student gap, f1, f2].
- **Certificates:** `common.certify` (fixed 40-level grid, delta 0.05), 100-row calibration half and 100-row holdout, the same 5 rotations as S15 (`default_rng(200+r)`), and a second view with 200 random splits. S15's signals for gpt-5.4-nano and qwen3-next-80b are recomputed from the S15 cache on the identical 200 rows (their own cheap answers and targets); the probe and f2 cannot be built for them (no cheap-model labels on train rows), only Qwen has them.
- **Budget:** pod cost about $0.13 this round, about $0.21 for S21 including the two earlier attempts (see Time and cost). Analysis (probes, bootstrap, certificates) ran on the local CPU.

## Results

### A. Qwen2.5-0.5B as the cheap model, 200 rows, target cheap == E1 (base rate 0.270)

AUROC with 95% bootstrap CI over inputs (1000 resamples). Paired differences vs the cheap logprob gap and vs f1.

| signal | AUROC [95% CI] | diff vs cheap logprob gap | diff vs f1 |
|---|---|---|---|
| cheap logprob, sequence | 0.684 [0.596, 0.769] | +0.074 [+0.024, +0.128] | -0.266 [-0.366, -0.176] |
| cheap logprob gap (S15 style) | 0.610 [0.522, 0.693] | - | -0.340 [-0.442, -0.246] |
| student top-2 gap | 0.521 [0.431, 0.609] | -0.089 [-0.211, +0.035] | -0.430 [-0.529, -0.322] |
| **f1 student == cheap** | **0.950 [0.910, 0.986]** | +0.340 [+0.246, +0.442] | - |
| f3 f1, then cheap gap | 0.955 [0.901, 0.995] | +0.346 [+0.256, +0.440] | +0.005 [-0.022, +0.034] |
| probe (hidden state, proxy-trained) | 0.775 [0.699, 0.839] | +0.166 [+0.065, +0.271] | -0.175 [-0.250, -0.109] |
| f2 bge-embedding LR (proxy-trained) | 0.706 [0.615, 0.785] | +0.096 [-0.014, +0.205] | -0.245 [-0.337, -0.162] |
| f1, then probe | 0.959 [0.907, 0.995] | +0.349 [+0.258, +0.446] | +0.009 [-0.019, +0.035] |
| f1, then f2 | 0.955 [0.898, 0.998] | +0.345 [+0.248, +0.447] | +0.005 [-0.029, +0.038] |
| stack LR (5-fold OOF) | 0.929 [0.859, 0.981] | +0.319 [+0.213, +0.427] | -0.022 [-0.054, +0.006] |
| probe, 5-fold on the 200 (OOF) | 0.745 [0.661, 0.815] | +0.135 [+0.020, +0.255] | -0.205 [-0.288, -0.124] |
| f2, 5-fold on the 200 (OOF) | 0.620 [0.532, 0.709] | +0.011 [-0.114, +0.134] | -0.330 [-0.420, -0.250] |
| random | 0.445 [0.358, 0.534] | -0.165 [-0.296, -0.028] | -0.505 [-0.605, -0.413] |

Probe minus f2: +0.069 (0.775 vs 0.706).

The same signals for the cheap models of S15, same 200 rows, same bootstrap (own cheap answers, base rates 0.800 and 0.795):

| cheap model | cheap logprob gap | student top-2 gap | f1 | f3 |
|---|---|---|---|---|
| gpt-5.4-nano | 0.892 [0.846, 0.935] | 0.667 [0.575, 0.748] | 0.900 [0.837, 0.952] | 0.958 [0.926, 0.982] |
| qwen3-next-80b | 0.788 [0.701, 0.860] | 0.650 [0.548, 0.741] | 0.919 [0.867, 0.960] | 0.951 [0.915, 0.978] |

(These reproduce S15's ordering: f1 0.90-0.92, f3 0.95-0.96.) The Qwen2.5-0.5B row above has the highest f1 (0.950) mostly because a tiny model that agrees with the student is almost always right; the positives are rare (27%), which also lowers the logprob-based numbers.

**Certified coverage on the 200 rows, Qwen2.5-0.5B, budgets 5/10/15/20%: zero for every signal in the 5 rotations** (the 100-row calibration half has about 27 positives; with the 40-level grid and delta 0.05 a 20% budget needs 30 clean rows). In 200 random splits only the f1-based signals ever certify, and only at 20% budget: f1 in 3% of splits (0.6% coverage), f3 3%, f1 then probe 6% (1.0%), f1 then student gap 6%, f1 then f2 5%. The probe, f2, logprob signals and the student gap alone never certify.

Certified coverage on the 200 rows for the other cheap models (S15 cache; mean holdout coverage %, min over 5 rotations in brackets, worst realised mismatch % among routed holdout rows, count of rotations that certified):

| cheap model | signal | 10% | 15% | 20% |
|---|---|---|---|---|
| gpt-5.4-nano | cheap logprob gap | 0 | 64 (44), mism 11, 5/5 | 74 (65), mism 16, 5/5 |
| | f1 | 15 (0), mism 6, 1/5 | 47 (0), mism 6, 3/5 | 77 (73), mism 6, 5/5 |
| | f3 | 53 (0), mism 6, 4/5 | 76 (65), mism 14, 5/5 | 83 (77), mism 16, 5/5 |
| | student gap | 0 | 0 | 0 |
| qwen3-next-80b | cheap logprob gap | 0 | 0 | 27 (0), mism 10, 2/5 |
| | f1 | 0 | 60 (0), mism 3, 4/5 | 74 (70), mism 3, 5/5 |
| | f3 | 0 | 69 (53), mism 3, 5/5 | 81 (75), mism 13, 5/5 |
| | student gap | 0 | 0 | 14 (0), mism 19, 1/5 |

At 10% and below nothing is reliable at 100 calibration rows (as S15 found). Mismatch 16% at the 20% budget is under budget in all rotations.

### B. Pool of 1200 rows (200 + 1000 new test rows), target cheap == gold, Qwen2.5-0.5B (base rate 0.271)

The probe and f2 never saw these rows and the student never saw test rows. AUROC with 95% bootstrap CI, then the clean certificate over 200 random 600/600 splits. Certificate cells: mean realised coverage % / share of splits that certified / pooled realised error % / share of certified splits whose holdout error exceeded the budget ("viol").

| signal | AUROC [95% CI] | diff vs cheap gap | cov @10% | cov @15% | cov @20% |
|---|---|---|---|---|---|
| cheap logprob, sequence | 0.707 [0.674, 0.740] | +0.045 [+0.023, +0.068] | 0 | 0 | 0 |
| cheap logprob gap | 0.661 [0.625, 0.697] | - | 0 | 0 | 0 |
| student top-2 gap | 0.596 [0.558, 0.631] | -0.065 [-0.115, -0.014] | 0 | 0 | 0 |
| f1 student == cheap | 0.972 [0.961, 0.982] | +0.311 [+0.275, +0.345] | 0.1 / 0.01 / err 11.3 / viol 1.00 | 13.5 / 0.47 / err 8.6 / viol 0 | 28.3 / 0.99 / err 7.4 / viol 0 |
| f3 f1, then cheap gap | 0.979 [0.969, 0.988] | +0.318 [+0.284, +0.352] | 0.4 / 0.02 / err 8.8 / viol 0.25 | 17.3 / 0.69 / err 8.0 / viol 0 | 28.6 / 1.00 / err 8.5 / viol 0 |
| probe (proxy-trained) | 0.792 [0.766, 0.818] | +0.131 [+0.092, +0.171] | 0 | 0 | 0 |
| f2 bge-embedding LR | 0.744 [0.713, 0.775] | +0.083 [+0.041, +0.124] | 0 | 0 | 0 |
| f1, then probe | 0.981 [0.971, 0.989] | +0.320 [+0.284, +0.355] | 2.2 / 0.12 / err 6.7 / viol 0 | 23.4 / 0.89 / err 6.2 / viol 0 | 28.4 / 1.00 / err 8.1 / viol 0.01 |
| random | 0.551 [0.516, 0.588] | -0.110 [-0.159, -0.064] | 0 | 0 | 0 |

"f1 then probe" vs f1 is +0.009 [+0.003, +0.015] in AUROC, a detectable but tiny gain. In certified coverage the probe is the better tie-break inside the f1 = 1 block: at 15% budget 23.4% coverage (certifying in 89% of splits) against 17.3% for the logprob-gap tie-break (69%) and 13.5% for f1 alone (47%); at 10% 2.2% vs 0.4% vs 0.1%; at 20% all three are about 28% (capped by the 27% base rate). At 5% nothing certifies. **Budget violations at 10%:** the one split in which f1 certified had 11.0% holdout error (above the 10% budget, viol 1.00), and f3 exceeded the budget in 25% of its (few) certified splits; coverage there is 0.1-0.4%, so no conclusion rests on it. This rests on the gold target, not E1.

### C. RouterBench, 2000 rows (gpt-3.5 vs gpt-4), 5-fold OOF with nested (layer, C) selection

| target | probe last-token | probe mean-token | bge embedding LR (S5 features) | family prior |
|---|---|---|---|---|
| gpt-3.5 correct | 0.711 [0.686, 0.735] | 0.706 | 0.682 [0.656, 0.709] | 0.624 |
| gpt-3.5 >= gpt-4 | 0.652 [0.626, 0.680] | 0.641 | 0.638 [0.607, 0.666] | 0.577 |

Probe minus embedding LR: +0.029 [+0.010, +0.047] on "gpt-3.5 correct", +0.015 [-0.008, +0.035] on "gpt-3.5 >= gpt-4". In cost-saving terms the in-sample figures (cutoff picked on the same rows, optimistic) are a saving of 0.3 points for the probe against 0.0 for random mixing at zero accuracy loss, and 38.1 against 30.3 (+7.8) at 5 points of loss. The bootstrap means in `log.txt` (+1.4 [+0.1, +4.0] at zero loss, +6.3 at 5 points) are means over resamples in which the cutoff is re-picked each time, so they are optimistically biased and are not point estimates; the probe's saving is not distinguishable from 0 at 0.5-5 points of loss. The probe therefore adds about 3 AUROC points over the S5 embedding features and does not turn into an usable saving at this size.

## Caveats

- **The cheap model is too weak for the question the spec asks.** Qwen2.5-0.5B matches E1 on 27% of the 200 inputs (gold accuracy 0.230 with the nearest-label parse; 17% of answers are not an exact label). Positives are rare, so 100-row calibration halves certify nothing at any budget up to 20%, whatever the signal. A 0.5B model is also not the cheap tier Dreaming would route to (S15 used models at 69-80% match); this smoke measures whether hidden states help, not whether Qwen2.5-0.5B is a good router tier.
- Probe and f2 are trained with a proxy target (cheap == gold on 1000 train rows) and applied to a different target (cheap == E1). The 5-fold-on-200 versions, using the real target, are worse (0.745, 0.620), a sign that 200 rows are too few to train a 896-d probe, not that the proxy is better. Probes were not trained for the S15 cheap models.
- The 1000 new test rows have no E1 answer (no new LLM calls were allowed), so the pool in B uses the gold target; f1 benefits there because the student is trained on gold labels, a regime in which "student == cheap" is close to "both correct".
- Stack LR uses OOF scores for AUROC; its certificate row uses a nested fit inside each calibration half. 5 rotations of a 100/100 split is a very coarse view (min over rotations is 0 for most cells); the 200-split table was added for the Qwen rows.
- The AUROC bootstrap resamples inputs, not model draws; E1 itself differs from its own repeat on 6% of inputs (S15), a ceiling on any agreement-based estimate.
- 17% of Qwen's answers are not an exact label; they are matched to the nearest label as in S5/S15 (the first version of this smoke scored them as wrong, see Revision). Gap and logprob are computed on the generated answer tokens only.
- The earlier extraction (an RTX A5000 run, `pod_launch_s21b.txt`) was lost with its pod; this one saves after every input set. The 4090 pod of this round (`pod_launch_s21c.txt`) came up without CUDA again.

## Verdict

- **Do not add a hidden-state probe to the router gate.** On the like-for-like comparison it ranks 0.775, between the cheap logprob gap (0.61) and f1 (0.950), and "f1 then probe" is a statistical tie with f1 on the 200 rows (+0.009, CI includes 0) and a +0.009 gain on 1200 rows. It does beat a probe on the student's embedding (f2, +0.069) and the logprob gap, so hidden states are informative, and as a tie-break inside f1 = 1 it certifies more coverage at 15% than the logprob gap does (23.4% vs 17.3%, gold target). That does not make it worth a GPU forward pass at this match rate: it does not change the ranking (AUROC tie), and all three variants reach about 28% at 20%.
- **Confirms S15:** f1 (student == cheap), with the cheap logprob gap or nothing as tie-break, stays the router gate; the student's own gap is a poor router signal (0.52-0.67).
- **Not resolved:** whether a probe helps for a cheap model that is good enough to route to (match rate near 80%); that needs the probe trained on cheap-vs-E1 labels for such a model (needs hidden states of an open model of that quality, e.g. a 7-8B, plus E1 answers on a few hundred rows). Certification at budgets of 15-20% needs a base rate above about 50% and 100+ calibration rows, or 500+ rows for 10%.

## Time and cost

- Pod: RTX A5000, 3.7 min of extraction ($0.02 metered by the launcher), plus about 18 min of boot/idle time before the launcher was re-attached (the first launcher process was killed by a shell timeout while the pod kept booting; $0.08), plus the failed RTX 4090 launch without CUDA ($0.03): about $0.13 this round (`smokes/COSTS.md`); adding the earlier attempts (pod_s21 $0.010, pod_s21b $0.070) S21 has cost about $0.21 in total, under the $0.5 budget. No API spend.
- Local CPU: arms 10 min, RouterBench probe analysis 5 min (bge embedding of 2000 RouterBench texts 176 s).
- Files: `extract.py` (GPU), `arms.py` (Banking77 arms, f1/f2/probe, certificates, S15 recomputation), `analyze.py rb` (RouterBench), `results_gpu.json` (raw generations and logprobs), `results.json` (summary numbers), `log.txt`; the 87 MB hidden-state file `feats_Qwen2.5-0.5B-Instruct_gpu.npz` is git-ignored.

## Revision

A review found five issues; all are addressed.

1. **Parse.** The first version scored Qwen's answers exact-only (non-exact answers got f1 = 0 and target = 0, easy negatives), unlike S5/S15, which use a nearest-label parse (`difflib`) for every cheap model. `analyze.py gen_table` now uses S15's parse, and all Banking77 tables above were recomputed (`arms.py`). Match rate with E1 on the 200 rows: 0.270 (was 0.230). Changes: probe AUROC 0.811 to 0.775, cheap logprob gap 0.669 to 0.610, sequence logprob 0.754 to 0.684, f1 0.967 to 0.950, f3 0.973 to 0.955, f2 0.685 to 0.706, f1 then probe 0.972 to 0.959, stack 0.952 to 0.929, probe 5-fold on the 200 0.717 to 0.745; probe minus f2 shrinks from +0.126 to +0.069. Ordering and the tie of f1 with f1+probe are unchanged. RouterBench numbers do not depend on the parse.
2. **Certified coverage verdict.** The earlier statement that the probe changes nothing over f1 was too strong for certified coverage. On the 1200-row gold pool the probe is a better tie-break than the logprob gap at 15% (23.4% vs 17.3% vs 13.5% for f1; before the parse fix: 14.7 / 12.6 / 8.2) and at 10% (2.2 / 0.4 / 0.1), and about equal at 20% (about 28%). It does not matter for ranking and is not worth a GPU pass at this match rate.
3. **Violations.** Table B now keeps the violation column: at 10% the single split where f1 certified had 11.0% holdout error (budget 10%), and f3 violated in 25% of its certified splits (11% before the parse fix); coverage there is 0.1-0.4%.
4. **RouterBench saving.** The +1.4 [+0.1, +4.0] and +6.3 figures are bootstrap means with the cutoff re-picked on each resample, hence optimistically biased; in-sample figures are 0.3 vs 0.0 (random mix) at zero loss and 38.1 vs 30.3 at 5 points. The "no usable saving" conclusion stands.
5. **Cost.** $0.13 is this round only (s21c 4090 $0.031, s21d A5000 $0.017, idle boot $0.080); with pod_s21 ($0.010) and pod_s21b ($0.070) S21 totals about $0.21.

Checked and unchanged by the review: no leakage between the 200, 1000 train and 1000 new rows (0 shared texts; probe and f2 selected and fitted on train only); S15 signals are recomputed (within 0.005 AUROC of S15's published figures); certificates use `common.certify` as stated.
