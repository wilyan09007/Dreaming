## In plain words

PARTIAL RESULTS. The pod was cut off before the run finished, so this is 36 of 45 planned cells (188 of 225 planned arm runs), no cell at k=10 for seed 2, and no critic pass was done. Every number below is recomputed directly from `salvaged_results.json`.

The question: when a new workflow step needs a classifier, is it better to use a small decoder language model (0.5B parameters, adapted with LoRA, which trains only a small add-on) than the 33M encoder student (frozen bge-small embeddings + logistic regression)? And does it help if the decoder starts from adapters trained on sibling steps (a shared pool adapter, or a mix of per-domain adapters)? Answer from the cells that finished: no on the first, yes a little on the second. The plain 33M encoder beat every decoder arm on accuracy at every k, in 15 of 15 paired cells at k=1 and k=3 for the from-base, composition and composition+fine-tune arms and 14 of 15 for the shared-pool arm. It also had the highest certified coverage at 5% error at k=1 and k=3 (12% and 65% vs at most 3% and 26% for any decoder arm). Starting the decoder from the shared pool adapter helps relative to LoRA from base (+6.9 accuracy points at k=1, +4.7 at k=3), so sibling-step transfer is real for a decoder, but it closes only part of the gap. At k=10 all arms are within about 4 accuracy points of each other (93.0 to 96.9; the four decoder arms within 1.5), but only 6-9 of 15 cells exist per arm, so the k=10 comparison is weak. Nothing here used LLM-written synthetic examples, which S17 showed to be the strongest warm start.

## Setup

Protocol shared with the other lite smokes, with one difference from S17: here every task is calibrated on 300 rows drawn from the train split, whereas S17 calibrated CLINC tasks on the validation split (20 per label). The encoder arm otherwise matches S17 (logistic regression, C=10) and its coverage here (12/65/92%) is close to S17's 14/60/91%, but the calibration source differs, so it is not an exact replication. code in `run.py` (orchestrator) and `worker.py` (3 workers sharing one RTX 4090).

- **Tasks (5, never trained on):** CLINC150 domains `travel` and `work` (15 labels each; 450 test rows each), and three 12-label Banking77 sub-tasks `bank0-2` (480, 480 and 479 test rows). Banking77 sub-task ids come from `default_rng(0).permutation(77)`: sub0 = perm[0:12], sub1 = perm[12:24], sub2 = perm[24:36].
- **Disjointness (asserted in code):** the CLINC pool is the 8 other CLINC domains (120 labels) and the Banking77 pool is the 41 labels at perm[36:]; both are disjoint from the evaluation labels. CLINC tasks use only CLINC-domain pools and Banking tasks only Banking pools, so the S17 cross-dataset `exchange_rate` overlap does not arise here.
- **Rows:** k real rows per label, k in {1, 3, 10}, seeds 0-2 (15 cells per k planned). Calibration 300 rows, test = the test rows above.
- **Metrics:** accuracy on the test rows; certified coverage at a 5% error budget via `common.certify` (fixed grid, Clopper-Pearson) on the calibration rows with top-2 gap confidence, cutoff read off on the test rows. A cell where nothing certifies counts as 0 coverage. Means are over available (task, seed) cells; "±" is the standard error of the cell mean (cells of the same task share data, so it is rough).
- **Decoder:** Qwen2.5-0.5B with a sequence-classification head, prompt `Step: ... Query: ... Label:`, bf16 autocast, batch 32, max length 64, 300 training steps per run, LoRA learning rate 2e-4, no tuning per arm.
- **Arms**
  - F: frozen bge-small embeddings + LogisticRegression (the S17 encoder recipe without synthetic examples; calibration source differs, see above).
  - A: LoRA from base, trained on the k rows per label (2.18M trainable parameters).
  - B: LoRA initialised from a shared pool adapter (trained once on the whole sibling pool: 750 steps on 120 CLINC labels, or 600 steps on the 41 Banking labels), then 300 steps on the k rows.
  - D (composition): the new head plus softmax mixing weights over per-domain pool adapters (8 CLINC domains, 4 Banking77 label groups); 300 steps; 10.8k-13.4k trainable parameters.
  - D+ (composition, then +100 LoRA steps): D followed by 100 steps of full LoRA training, 400 steps in total.

## Results

### Coverage of the run versus the plan

Plan (from `run.py`/`worker.py`): 5 tasks x 3 seeds x 3 values of k = 45 cells, 5 arms each = 225 arm runs, plus 14 pool adapters.

| k | F | A | B | D | D+ | cells with all five arms |
|---|---|---|---|---|---|---|
| 1 | 15 | 15 | 15 | 15 | 15 | 15 of 15 |
| 3 | 15 | 15 | 15 | 15 | 15 | 15 of 15 |
| 10 | 9 | 6 | 9 | 7 | 7 | 6 of 15 |

- All 14 pool adapters finished (9 CLINC jobs and 5 Banking jobs, including the two shared pools). The run recorded 0 errors.
- Missing at k=10: all seed-2 cells for every task and arm, plus further seed-1 cells (F lacks 6 cells, A lacks 9, B lacks 6, D and D+ lack 8 each). Per the pod log, bank2 seed 1 at k=10 and every seed-2 cell at k=10 were never started (not cut off midway). A at k=10 covers travel (2 seeds), work (1), and one seed each for bank0-2; so k=10 means mix different tasks across arms. The matched table below restricts to the 6 cells where all five arms exist.
- Every task is present at k=1 and k=3 with all three seeds, so the k=1 and k=3 columns are complete.

### Main table, all five tasks

Accuracy (%) / certified coverage at 5% (%), mean over available cells (n in the table above).

| arm | acc k=1 | acc k=3 | acc k=10 | cov k=1 | cov k=3 | cov k=10 |
|---|---|---|---|---|---|---|
| F frozen bge-small + LR | 79.9 ± 1.9 | 92.1 ± 0.9 | 96.9 ± 0.6 | 12.2 ± 6.7 | 65.3 ± 7.7 | 91.5 ± 4.0 |
| A LoRA from base | 50.8 ± 2.6 | 80.3 ± 1.7 | 94.3 ± 1.0 | 0.0 ± 0.0 | 14.8 ± 7.9 | 88.8 ± 2.9 |
| B LoRA from shared pool | 57.7 ± 3.9 | 84.9 ± 1.7 | 94.5 ± 1.0 | 2.7 ± 2.7 | 21.3 ± 9.6 | 77.9 ± 10.1 |
| D composition (head + mixing only) | 53.2 ± 2.5 | 82.8 ± 1.6 | 94.4 ± 0.9 | 0.0 ± 0.0 | 25.5 ± 8.9 | 76.2 ± 13.0 |
| D+ composition + 100 LoRA steps | 55.5 ± 3.0 | 79.9 ± 2.4 | 93.0 ± 1.7 | 0.0 ± 0.0 | 16.6 ± 7.8 | 66.5 ± 13.0 |

Matched k=10 cells only (the same 6 task/seed cells for all five arms: travel s0, work s0, bank0 s0, bank1 s0, bank2 s0, travel s1): accuracy F 97.0 ± 0.9, A 94.3 ± 1.0, B 94.5 ± 1.4, D 94.2 ± 1.0, D+ 92.6 ± 1.9; coverage F 93.7 ± 3.8, A 88.8 ± 2.9, B 73.4 ± 15.3, D 74.3 ± 15.2, D+ 63.1 ± 14.9. The ordering is the same as the unmatched means.

Share of cells where the 5% certificate succeeded at all (coverage above 0): F 20% / 87% / 100% at k=1 / 3 / 10; A 0% / 20% / 100%; B 7% / 27% / 89%; D 0% / 40% / 86%; D+ 0% / 27% / 86%. Realized error among answered rows in cells that certified was 0.7-2.7% on average and at most 4.4% (A at k=3), so no certificate was violated on the test rows.

### By task family

Accuracy (%) / certified coverage (%).

| arm | CLINC k=1 | CLINC k=3 | CLINC k=10 | Banking77 k=1 | Banking77 k=3 | Banking77 k=10 |
|---|---|---|---|---|---|---|
| F | 82.3 / 22.4 | 93.5 / 79.8 | 97.8 / 97.6 | 78.3 / 5.5 | 91.2 / 55.7 | 96.2 / 86.5 |
| A | 59.8 / 0.0 | 83.9 / 36.9 | 95.9 / 93.2 | 44.8 / 0.0 | 77.9 / 0.0 | 92.8 / 84.4 |
| B | 74.5 / 6.7 | 91.1 / 53.3 | 97.1 / 92.7 | 46.5 / 0.0 | 80.8 / 0.0 | 92.3 / 66.1 |
| D | 60.3 / 0.0 | 86.9 / 48.2 | 95.8 / 91.8 | 48.4 / 0.0 | 80.0 / 10.4 | 92.6 / 55.4 |
| D+ | 65.1 / 0.0 | 88.3 / 41.5 | 95.2 / 85.8 | 49.1 / 0.0 | 74.3 / 0.0 | 90.1 / 40.9 |

n per cell: CLINC 6 at k=1 and k=3, 3-4 at k=10; Banking77 9 at k=1 and k=3, 3-5 at k=10. The shared-pool adapter helps much more on CLINC (+14.7 points over A at k=1, +7.2 at k=3) than on Banking77 (+1.7 and +2.9), where the pool covers different intents of the same dataset. Per-task numbers are in the script output.

### Paired differences, same task, seed and rows (all five tasks)

Accuracy points, mean (s.e.), then wins/ties/losses of the first arm over the second.

| pair | k=1 | k=3 | k=10 |
|---|---|---|---|
| B - F | -22.2 (3.7); 1/0/14 | -7.2 (1.4); 1/0/14 | -2.5 (0.7); 0/0/9 |
| A - F | -29.1 (2.9); 0/0/15 | -11.8 (1.2); 0/0/15 | -2.6 (0.3); 0/0/6 |
| D - F | -26.7 (2.5); 0/0/15 | -9.4 (1.4); 0/0/15 | -2.6 (0.5); 0/0/7 |
| D+ - F | -24.4 (3.0); 0/0/15 | -12.2 (1.9); 0/0/15 | -4.1 (1.0); 0/0/7 |
| B - A | +6.9 (2.2); 12/0/3 | +4.7 (1.2); 13/0/2 | +0.1 (0.7); 4/0/2 |
| D - A | +2.4 (1.9); 10/0/5 | +2.5 (1.3); 10/0/5 | -0.2 (0.4); 3/1/2 |
| D - B | -4.5 (2.8); 4/0/11 | -2.2 (1.1); 4/1/10 | -0.4 (0.5); 3/0/4 |
| D+ - D | +2.3 (3.2); 9/0/6 | -2.9 (1.8); 4/0/11 | -1.4 (1.0); 3/0/4 |

Certified coverage, points: B - A +2.7 / +6.5 / -15.4 (k=1 / 3 / 10); D - A +0.0 / +10.7 / -14.6; D - B -2.7 / +4.2 / +0.9; D+ - D +0.0 / -8.9 / -9.7; any decoder arm minus F at k=3: -39.9 to -50.6 (12 of 15 cells lower for B, 13 of 15 for A, D and D+). The k=10 coverage differences have standard errors of 3-14 points on 6-9 cells and should be read as noise.

### Cost

| arm | trainable parameters (CLINC, 15 labels / Banking, 12 labels) | steps | measured s per run (3 workers shared the GPU) | solo-equivalent s |
|---|---|---|---|---|
| F | 5,775 / 4,620 | 0 | <0.1 | <0.1 |
| A | 2,176,128 / 2,173,440 | 300 | 33.7 | 30.7 |
| B | 2,176,128 / 2,173,440 | 300 | 33.5 | 30.7 |
| D | 13,448 / 10,756 | 300 | 40.8 | 35.6 |
| D+ | 2,176,128 / 2,173,440 | 400 | 52.1 | 45.9 |

Solo-equivalent seconds use the solo calibration on the RTX 4090 (LoRA 9.77 steps/s, composition 8.42 steps/s; peak 4.2 GB). Composition is slower than LoRA per step because it re-mixes all K pool adapters into one adapter at every step. One-time pool cost: CLINC 8 domain adapters + shared = 223 s measured (200 s solo-equivalent), Banking 4 groups + shared = 148 s (123 s solo-equivalent). Pool adapters fit their own data (accuracy on a 300-row sample 0.97-1.00). Every decoder run is roughly a half-minute of GPU time against about 10 ms for the encoder student's LR fit (excluding the one-time embedding pass).

Learned mixing weights in D were close to uniform: CLINC mean largest weight 0.19 over 8 adapters (effective number of adapters, measured as exp of the entropy of the mixing weights: 7.6, uniform 0.12), Banking 0.35 over 4 (3.8, uniform 0.25; the inverse-Simpson measure would give about 7.3 and 3.7). The composition did not pick out a few relevant sibling adapters.

## Caveats

- **Partial run.** 36 of 45 cells and 188 of 225 arm runs; k=10 is missing seed 2 entirely and parts of seed 1, and the arms have different numbers of k=10 cells (6 to 9), so the unmatched k=10 means compare different task mixes. The matched 6-cell table gives the same ordering. The pod log also holds 2 finished D runs at k=10 that the JSON lacks (bank1 seed 1: accuracy 0.9208, coverage 0.7521; bank0 seed 1: accuracy 0.925, coverage 0.8542), logged after the last merge at +1800 s; they are not included, so D at k=10 has 9 finished runs in reality against 7 used here. No critic pass was done on this file.
- **Disagreement with FINDINGS (denominator only).** My recomputation reproduces the FINDINGS round-3 numbers for F, A, B and D at every k (for example F 0.80/0.12, 0.92/0.65, 0.97/0.92; B 0.58/0.03, 0.85/0.21, 0.95/0.78; B at k=10 is 94.5, which FINDINGS rounds to 0.95). It does not reproduce "188 of ~270 cells": the script's plan is 225 arm runs (45 cells x 5 arms), of which 188 are done, and 36 of 45 cells are complete. FINDINGS also gives no numbers for D+; the table above supplies them (it agrees that +100 steps is no better: it is lower than D at k=3 and k=10).
- **FINDINGS conclusion "the 33M encoder beats all at every k".** Supported for accuracy at every k, and for coverage at k=1 and k=3. At k=10 the coverage gap to A is small (91.5 vs 88.8, matched 93.7 vs 88.8) and within noise on these few cells.
- **Not a like-for-like comparison to S17's best arm.** F is the plain frozen-embedding LR without LLM-written examples. The decoder arms also have no synthetic examples. S17 showed that the encoder with LLM-written examples reaches 80% / 87% / 95% coverage; that arm was not run against the decoder here, so the claim is only that the plain encoder beats the plain decoders.
- **One fixed decoder recipe.** 300 steps, learning rate 2e-4, LoRA rank 16, alpha 32, dropout 0.05 on q/k/v/o projections, head learning rate 2e-3, no per-arm tuning, one 0.5B model, bf16. The decoder arms may be under-trained at k=1 (the classification head is newly initialised). Tuning could narrow the gap; my judgement, not tested, is that it is unlikely to make a 0.5B decoder beat the encoder at 1 to 3 rows per label given gaps of 7 to 29 points.
- **D+ is a single configuration** (+100 LoRA steps from the composition result) and its lower k=3 accuracy than D may be noise (s.e. of the paired difference 1.8).
- **Small sample.** Five tasks and three seeds, with s.e. of 1-4 accuracy points and 5-15 coverage points. Coverage at k=1 and k=3 is jumpy because a cutoff either certifies or does not (many cells are exactly 0).
- **Pool relevance.** CLINC `travel` and `work` are well separated from the 8 pool domains. The Banking77 pool is semantically close to the Banking tasks (sibling banking intents), which is the intended transfer; B still gains only +1.7 to +2.9 points over A there.
- The certificate uses the same 300 calibration rows for threshold and certification, as in the other lite smokes; it is not a separate third holdout.

## Verdict

For a new workflow step, keep the 33M encoder student with a linear head as the learned model. On the finished cells a 0.5B decoder with LoRA is 7 to 29 accuracy points worse at 1 to 3 rows per label and certifies almost nothing (at most 26% coverage at k=3 against 65%), at roughly half a minute of GPU per run instead of milliseconds. Sibling-step pretraining does help a decoder at 3 rows per label (shared-pool LoRA 84.9% vs 80.3% from base, composition 82.8%), which is consistent with S8's shared-LoRA finding; but S17's frozen-encoder finding stands as well, because the encoder with no transfer is still better than the best decoder arm. Composition of per-domain adapters is not better than the single shared-pool adapter (D - B is -4.5 / -2.2 / -0.4 accuracy points, coverage within noise) and fine-tuning the composition for 100 more steps does not help. The decoder route should be revisited only if the comparison is re-run with LLM-written examples on both sides, on the complete k=10 grid, with a tuned decoder recipe.

Files: `run.py`, `worker.py`, `report.py` (original report script for a finished `results.json`), `salvaged_results.json`, `salvaged_pod_log.txt`, `launch_shard.sh`.

## Revision

Applied after a numeric check of this file against `salvaged_results.json` (no material numeric errors found; all tables reproduced). Changes:
- Paired coverage line: B is lower than F in 12 of 15 cells at k=3, not 13 (A, D, D+ are 13 of 15).
- k=10 spread corrected from "about 3" to "about 4" accuracy points (93.0 to 96.9 on all cells; 4.4 on the matched cells; 1.5 among decoder arms).
- Caveats now say that 2 finished D runs at k=10 sit in the pod log but not the JSON, and that bank2 seed 1 and all seed-2 cells at k=10 were never started.
- Setup notes that calibration rows come from the train split here, versus the validation split in S17, so the comparison with S17 is approximate.
- Cost wording: the 10 ms excludes the one-time embedding pass; composition is slower because it re-mixes K adapters each step.
- Mixing-weight measure named (exp of entropy).
- The "unlikely to beat the encoder after tuning" claim is marked as a judgement.
- Removed references to the helper script `report_salvaged.py`, which was outside the brief; numbers are recomputed from the JSON as stated.
