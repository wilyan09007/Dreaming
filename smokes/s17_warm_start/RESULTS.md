## In plain words

A "step" is a new classification task in the workflow (here: sort messages into 12 or 15 labels), and we wanted to know how few labelled examples it needs before the small student model can answer on its own and still be right 95% of the time. Out of six ways of giving the student a head start, one clearly won: asking a cheap LLM to write 10 practice messages per label and training on those together with the real rows. With just 1 real row per label it can answer about 80% of traffic at the 5% error budget, against 14% for a plain student. The hypernetwork (a model that writes the student's weights from the label descriptions) improved a lot when we gave it hundreds of tasks to learn from, so it was starved before, but it levels off at about 90% accuracy and never beats either the plain student from 3 rows per label on or the LLM-written examples at any row count. Pre-training the encoder on the other tasks, and the Reptile meta-learned start, did not beat the plain student or their own no-training controls. Letting the student pick which rows to label hurt when the student was cold and helped only slightly when it started from the LLM-written examples. Recommendation: use the LLM-written examples as the standard day-zero warm start, skip the hypernetwork, encoder pre-training and Reptile. The whole study cost about $0.17 and 14 pod minutes.

## Setup

Protocol shared with the other lite smokes.

- **New-step tasks (5, never trained on):** CLINC150 domain `travel` (15 labels), CLINC150 domain `work` (15 labels), and three 12-label Banking77 sub-tasks. Banking77 sub-task label ids come from `default_rng(0).permutation(77)`: sub0 = perm[0:12], sub1 = perm[12:24], sub2 = perm[24:36] (label names are in `prep_info.json`).
- **Disjointness enforcement:** CLINC pool = the 8 other CLINC domains (120 labels), travel and work excluded by the domain map. Banking pool = the 41 labels at perm[36:], outside all three sub-tasks. All other pool sources are different datasets. Code asserts no label-name overlap inside CLINC and inside Banking77, and removes pool rows whose text equals any eval row (0 found). One cross-dataset name overlap remains and is left in: Banking77 `exchange_rate` (pool) vs CLINC `exchange_rate` (travel eval); see Caveats.
- **Rows:** k real rows per label, k in {1, 3, 10}, 3 draws (seeds 0-2). CLINC: support = train split of the domain (100 per label), calibration = validation split (20 per label = 300 rows), test = test split (450 rows). Banking77: calibration = 300 random train rows of the sub-task (fixed per task), support = its other train rows, test = official test rows of the 12 labels (480).
- **Metrics:** accuracy on the test rows and certified coverage at a 5% budget: `common.certify` (fixed grid, Clopper-Pearson, delta 0.05) on the 300 calibration rows with confidence = top-2 gap, cutoff applied to the test rows ("realized coverage"). Realized error on the answered rows was 0.0-2.3% in every arm, so no certificate was violated.
- **Student:** frozen bge-small embeddings + LogisticRegression (C=10). Baseline row in every table.
- **Pool for arms 3-5 (283 labels, 26,622 rows, <=100 rows per label):** CLINC train domains 120 labels, Banking77 41 labels, MASSIVE en 57 labels (`mteb/amazon_massive_intent`; 3 of the 60 intents have fewer than 20 rows and were dropped), TREC-QC 35 fine labels (those with >=20 rows; of 50), emotion 6, ag_news 4, 20 Newsgroups 20. Label descriptions for the non-CLINC/Banking labels were written by the LLM (0 fell back to the label name). TREC is held out of all fitting and used as the early-stopping / checkpoint dataset.
- **Arms**
  1. Baseline: LR on the k rows.
  2. Synthetic prior: LR on 10 LLM-written examples per label (ASI:One asi1-mini; S2b cache, regenerated for the 14 travel/work labels missing from it) + the k rows.
  3. Many-task hypernetwork: the S2b set-transformer (1.35M parameters; input = label-description embedding + mean of the k rows, output = a cosine head per label), trained on N distinct tasks (5-15 label subsets from one pool source, shots 1/2/3/5/10), AdamW 1e-3, early stopping on held-out episodes at k = 1, 3, 10. N = 25, 100, 800, 3000, and "CLINC-only" (the S2b recipe: 8 CLINC domains, 2000 episodes). Two early-stopping signals: held-out TREC (checked every 250 steps; launch 1) and held-out MASSIVE intents (TREC then joins the training pool; checked every 100 steps; launch 2). No fine-tuning on the rows: a single forward pass.
  4. Workflow-pretrained encoder: bge-small fully fine-tuned with cross-entropy over the union of pool labels (248 labels, 24,109 rows, TREC excluded), lr 3e-5, batch 64, up to 1000 steps, head discarded, then frozen + LR. Checkpoint chosen among the fine-tuned checkpoints (250/500/750/1000 steps) by 3-shot LR accuracy on held-out TREC tasks. Variant `nosib` = same recipe without any CLINC or Banking77 labels (87 labels, 8,099 rows). 4b adds the synthetic prior on top.
  5. Reptile: a 384x384 projection M (init identity) with a cosine head whose class vectors are `M·description + r` (r per task, zero init); inner loop 8 Adam steps on fresh k-row batches, outer Reptile step toward the adapted M, 4 tasks per step, early stopping on held-out TREC episodes; at test time the same 8 steps on the k rows. Control 5c = identical adaptation starting from M = identity (no meta-training). Launch 1 used outer step 0.5; launch 2 a gentler outer step 0.05 and inner M learning rate 2e-4.
  6. Active selection: the student picks the most uncertain rows (smallest top-2 gap) from an unlabelled pool of 40 random support rows per label; gold labels stand in for the teacher; budget k*K rows. 6 = baseline student (starts from one random pool row per label, then picks (k-1)*K rows in up to 3 rounds, so at k=1 it equals the baseline up to the draw). 6b = synthetic-prior student picks all k*K rows in up to 4 rounds.

## Results

All numbers: mean over 5 tasks x 3 draws; "+-" is the s.d. over the 3 draws of the task-averaged value (task-to-task spread is not in it). Full tables incl. per-family coverage and error are in `summary.md`; raw cells in `results.json` (900 records, 20 arms).

### Main table (headline version of each arm)

| arm | acc k=1 | acc k=3 | acc k=10 | cert. coverage k=1 | k=3 | k=10 | rows per label to 50% coverage |
|---|---|---|---|---|---|---|---|
| 1 baseline (frozen bge-small + LR) | 0.802 +- 0.015 | 0.925 +- 0.003 | 0.968 +- 0.002 | 0.143 +- 0.063 | 0.597 +- 0.101 | 0.906 +- 0.040 | 2.4 |
| 2 synthetic prior | 0.914 +- 0.005 | 0.943 +- 0.002 | 0.969 +- 0.003 | 0.798 +- 0.012 | 0.870 +- 0.008 | 0.951 +- 0.019 | <=1 |
| 3 hypernetwork, 800 tasks, stop on TREC | 0.861 +- 0.006 | 0.890 +- 0.003 | 0.898 +- 0.002 | 0.256 +- 0.023 | 0.451 +- 0.023 | 0.561 +- 0.098 | 5.1 |
| 3 hypernetwork, 800 tasks, stop on MASSIVE | 0.872 +- 0.010 | 0.898 +- 0.001 | 0.914 +- 0.004 | 0.275 +- 0.110 | 0.473 +- 0.172 | 0.591 +- 0.015 | 3.9 |
| 4 encoder pre-trained on pool (all) | 0.795 +- 0.014 | 0.905 +- 0.005 | 0.957 +- 0.005 | 0.175 +- 0.009 | 0.604 +- 0.156 | 0.886 +- 0.032 | 2.3 |
| 4 encoder pre-trained, no CLINC/Banking siblings | 0.736 +- 0.006 | 0.877 +- 0.006 | 0.941 +- 0.003 | 0.140 +- 0.011 | 0.382 +- 0.052 | 0.756 +- 0.059 | 4.4 |
| 4b pre-trained encoder + synthetic prior | 0.906 +- 0.003 | 0.930 +- 0.001 | 0.959 +- 0.002 | 0.714 +- 0.058 | 0.815 +- 0.013 | 0.937 +- 0.007 | <=1 |
| 5 Reptile start (launch 1, outer step 0.5) | 0.873 +- 0.009 | 0.919 +- 0.006 | 0.955 +- 0.007 | 0.376 +- 0.092 | 0.644 +- 0.142 | 0.871 +- 0.009 | 1.7 |
| 5 Reptile start (launch 2, outer step 0.05) | 0.885 +- 0.017 | 0.937 +- 0.010 | 0.966 +- 0.001 | 0.454 +- 0.020 | 0.758 +- 0.064 | 0.916 +- 0.046 | 1.2 |
| 5c control: identity start (no meta-training) | 0.889 +- 0.015 | 0.937 +- 0.011 | 0.967 +- 0.002 | 0.466 +- 0.019 | 0.767 +- 0.079 | 0.933 +- 0.028 | 1.1 |
| 6 active selection, baseline student | 0.824 +- 0.008 | 0.814 +- 0.020 | 0.958 +- 0.011 | 0.196 +- 0.080 | 0.166 +- 0.054 | 0.731 +- 0.199 | 6.1 |
| 6b active selection, synthetic-prior student | 0.921 +- 0.002 | 0.962 +- 0.003 | 0.978 +- 0.002 | 0.776 +- 0.062 | 0.909 +- 0.006 | 0.965 +- 0.005 | <=1 |

"Rows per label to 50% coverage": linear interpolation in log k between the measured k = 1, 3, 10 using the mean over all five tasks; "<=1" means already above 50% at k=1 (smaller k was not measured; see Caveats), ">10" means below 50% at k=10. Per family (CLINC tasks | Banking77 tasks): baseline 3.1 | 2.2; synthetic prior <=1 | <=1; hypernetwork 800/TREC >10 | 3.9; hypernetwork 800/MASSIVE 2.7 | 3.9; pre-trained encoder 1.6 | 2.4; Reptile launch 1 1.7 | 1.7; control 1.1 | 1.1; active selection (baseline student) 4.7 | 7.0.

### Was the hypernetwork starved of tasks?

Accuracy / certified coverage at k = 1, 3, 10, by number of training tasks (all rows are the same architecture, 1,350,530 parameters).

| tasks (labels seen) | early stop on TREC: accuracy | TREC: coverage | rows to 50% | early stop on MASSIVE: accuracy | MASSIVE: coverage | rows to 50% |
|---|---|---|---|---|---|---|
| CLINC-only, 2000 episodes (120 labels; the S2b recipe) | 0.783 / 0.806 / 0.821 | 0.03 / 0.11 / 0.11 | >10 | - | - | - |
| 25 (156 labels TREC-stop; 141 MASSIVE-stop) | 0.829 / 0.841 / 0.853 | 0.04 / 0.17 / 0.23 | >10 | 0.839 / 0.876 / 0.890 | 0.17 / 0.44 / 0.40 | >10 |
| 100 (246; 218) | 0.856 / 0.882 / 0.893 | 0.15 / 0.43 / 0.54 | 6.7 | 0.856 / 0.888 / 0.906 | 0.18 / 0.40 / 0.52 | 7.9 |
| 800 (248; 226) | 0.861 / 0.890 / 0.898 | 0.26 / 0.45 / 0.56 | 5.1 | 0.872 / 0.898 / 0.914 | 0.28 / 0.47 / 0.59 | 3.9 |
| 3000 (248; 226) | 0.854 / 0.882 / 0.899 | 0.35 / 0.42 / 0.52 | 8.0 | 0.848 / 0.896 / 0.909 | 0.30 / 0.56 / 0.61 | 2.4 |
| reference: baseline LR | 0.802 / 0.925 / 0.968 | 0.14 / 0.60 / 0.91 | 2.4 | same | same | same |
| reference: synthetic prior | 0.914 / 0.943 / 0.969 | 0.80 / 0.87 / 0.95 | <=1 | same | same | same |

Training time of one hypernetwork run was 58-61 s on pod CPU (2 threads) with TREC stopping and 25-30 s with MASSIVE stopping; the best checkpoint was the first one checked in every TREC-stopped run (step 250) and the first or second one in the MASSIVE-stopped runs (step 100; step 200 for 800 tasks), after which held-out accuracy fell.

### Reptile, per family

Certified coverage (CLINC | Banking77), k = 1 / 3 / 10: Reptile launch 1 0.44 | 0.33, 0.57 | 0.69, 0.92 | 0.84; Reptile launch 2 0.48 | 0.44, 0.73 | 0.78, 0.91 | 0.92; identity-start control 0.48 | 0.46, 0.71 | 0.80, 0.93 | 0.93. Held-out TREC accuracy during meta-training: identity control 0.718, launch 1 started at 0.677 after 250 steps and fell to 0.647 by step 1000; launch 2 stayed at 0.711-0.716 (essentially the control).

## Interpretation

- **Synthetic prior (2) is the warm start.** It lifts certified coverage at 1 real row per label from 14% to 80%, and from 60% to 87% at 3 rows; by 10 rows the baseline has caught up (91% vs 95% coverage, same accuracy), so the prior matters only in the first handful of rows. Cost: one cheap LLM call per 15 labels (about $0.0001 per label), no training beyond the LR fit.
- **Hypernetwork (3): it was starved, but the fix runs out.** Going from the S2b recipe (CLINC only, 120 labels, 0.82 accuracy at k=10, 11% coverage) to 100 tasks over six sources lifts k=10 accuracy to 0.89-0.91 and coverage to about 0.52-0.54, and 800 adds a little more; 3000 gives no further gain (the extra tasks are new label subsets of the same ~250 labels). It beats the baseline only at k=1 (accuracy 0.86-0.87 vs 0.80; coverage 0.26-0.28 vs 0.14) and loses to it from k=3 on, because it only sees the row mean and cannot fit the rows. It never reaches the synthetic prior at any k (0.91 accuracy and 80% coverage at k=1). Rows to 50% coverage: 2.4-8.0 depending on task count and stopping signal, vs 2.4 baseline and <=1 synthetic. Verdict: with hundreds of tasks it no longer fails, but it does not earn a place as a warm start.
- **Workflow-pretrained encoder (4): no help for a frozen+LR student.** Cross-entropy fine-tuning on the pool never improved held-out TREC 3-shot accuracy by more than 0.003 (0.702 to 0.705 at 250 steps, then down), and on the new tasks it is at or below the baseline at every k (0.795 / 0.905 / 0.957 vs 0.802 / 0.925 / 0.968). Without the CLINC/Banking77 siblings it is clearly worse (0.736 at k=1), so what little transfer there is comes from same-benchmark siblings. This does not contradict S8 phase B (shared LoRA helped at 3 rows/label), which fine-tuned the shared model end to end on the new step; this arm tested the frozen version only.
- **Reptile (5): the gain over baseline is from the description-initialised head, not from meta-learning.** The identity-start control (cosine head initialised from the label descriptions, 8 Adam steps on the k rows) already gets 0.889 accuracy at k=1 and 50% coverage at 1.1 rows, better than the plain LR baseline and with no LLM data. Meta-training M did not improve on it in either configuration (launch 1 worse, launch 2 equal within noise), and the held-out TREC curve says the same.
- **Active selection (6): do not use it with a cold student.** The baseline student picks confusable, atypical rows and leaves some labels with a single example: at k=3 accuracy is 0.814 (below its own k=1) and coverage 0.17. With the synthetic-prior student it is a small plus (accuracy +0.019 at k=3, +0.009 at k=10; coverage +0.04 at k=3, +0.014 at k=10; at k=1 coverage is slightly lower, 0.78 vs 0.80, within noise), so it is worth revisiting only once labelling rows costs real money.
- **Combining (4b) adds nothing over arm 2 alone.**

Recommendation. Day-zero warm start = LLM-written practice examples (arm 2); on top of it, optionally prior-aware active selection of rows (6b) once teacher labels are costly. Do not build the many-task hypernetwork for this purpose; if one wants a no-LLM fallback, the description-initialised cosine head (5c) is cheaper and better than the hypernetwork and reaches 50% coverage at about 1 row per label. Do not pre-train the encoder on sibling tasks for a frozen student, and drop Reptile.

## Caveats

- Five tasks and three draws; the "+-" shows only draw-to-draw spread. Coverage differences below about 0.1 (especially at k=1 and k=3, where a single draw moves a cutoff) are noise. The task set is two CLINC domains and three Banking77 sub-tasks: well-separated intent labels; the earlier larger S17 plan also had a bank-transaction task, not run here.
- "<=1 row per label" is a bound at the bottom of the tested grid (k = 1 is the smallest). Day zero (k = 0) was not part of this protocol; S2b measured the same recipe certifying 29% of traffic with no real rows on 77-way Banking77.
- Pool contamination risk: the pool has CLINC `banking` and `credit_cards` domains and 41 Banking77 labels that are semantically close to the Banking77 sub-tasks (sibling steps, as intended), and one identical concept across datasets (`exchange_rate`: Banking77 in the pool, CLINC `travel` in the eval; two Banking77 sub2 labels are also exchange-rate related). The `nosib` encoder variant shows how much siblings matter for arm 4; arms 3 and 5 were not re-run without siblings.
- Hypernetwork: "800 tasks" are label subsets of at most 248 labels, not 800 independent datasets (the published Text-to-LoRA setting is 64-479 tasks each with its own data), so label diversity, not task count, may be the real limit. Held-out accuracy peaked at the first or second checkpoint we looked at (step 100-250), so the true optimum may be earlier still. Two early-stopping signals were tried; the MASSIVE one is closer to the evaluation tasks (intent classification) and was slightly better. Using the hypernetwork's output as a prior for a fitted classifier (rather than as the classifier) was not tested; that is the main untested way it could still add value.
- Encoder arm: one recipe (lr 3e-5, 1000 steps, checkpoints every 250 steps; the earliest checkpoint was chosen in both variants), no tuning of lr or steps, no fine-tuning on the k rows afterwards.
- Reptile: launch 2 was a second try chosen after launch 1 was negative; selection used held-out TREC episodes only, not the evaluation tasks, but it is still one extra configuration, and only one value each of inner steps (8) and inner learning rate for M was tried (the class-residual rate was picked from three values on TREC).
- Active selection: gold labels stand in for the teacher (no label noise); the pool is 40 random support rows per label; the baseline student's seed is one random pool row per label (a stratified assumption); only plain top-2-gap uncertainty was tried.
- The certificate is computed on the fixed 300 calibration rows and read off on the test rows (as specified); it is not a separate third holdout, and the 12-15 label tasks make cutoffs jumpy at low k.
- The laptop `--tiny` check took 77 s wall (imports and dataset loading dominate), above the 60 s target. Launch 1 ran on a pod found through the "any cloud" fallback ($0.89/h) and launch 2 on a community pod ($0.34/h), so the two per-hour rates differ.

## Time and cost

| item | measured |
|---|---|
| Pod launches | 2 (RTX 4090 both; CUDA present both times). Launch 1: 9.4 min wall including boot, $0.139. Launch 2: 4.8 min wall, $0.027. |
| Data prep on the pod | 100 s (dataset downloads, LLM label descriptions, embedding 26.6k pool rows and the eval sets) |
| LLM spend | $0.0014 new (283 pool-label descriptions + 14 synthetic-example labels), cap was $0.5; logged in COSTS.md |
| Total | about $0.17 and 14 pod minutes for everything, including a hypernetwork scale sweep, two early-stopping signals and two Reptile configurations |

Per arm (trainable parameters; time to train; time to fit on a new step):

| arm | trainable parameters | training time | fitting a new step |
|---|---|---|---|
| 1 baseline | 5.8k (15 labels) / 4.6k (12 labels): LR weights | none | about 40 ms per fit on one CPU thread |
| 2 synthetic prior | same as 1 | none (LLM call about seconds, $0.0001 per label) | about 20 ms per fit |
| 3 hypernetwork | 1,350,530 | 25-61 s per run on 2 CPU threads (best checkpoint reached within 12 s) | one forward pass, about 2 ms; nothing to train per step |
| 4 pre-trained encoder | 33.36M fine-tuned once (248-way head of 95k discarded); 5.8k per step in the LR | 38.5 s per variant on the 4090 (1000 steps, 24k rows, including 5 validation passes); 78 s for both variants | about 20 ms per fit after re-embedding the step's texts |
| 5 Reptile | 147,456 shared (M) + 5.8k per step (r) | 80 s (launch 1) / 26 s (launch 2) on 2 CPU threads | 8 Adam steps, about 35 ms |
| 6 active selection | as 1 | none | 33 ms (baseline student) / 75 ms (synthetic-prior student) per cell, i.e. a few LR refits |

Files: `lite.py` (all code; reuses `lib.py` and `run.py` from the earlier S17 attempt), `results.json` (every cell), `summary.md` (all tables), `prep_info.json`, `results_enc.json`, `results_reptile.json`, `results_reptile2.json`, `models_json/` and `models_json2/` (hypernetwork training curves), `logs_pod1/` and `logs/` (pod stage logs), `synth_lite.json` (new synthetic examples), `responses.jsonl` (LLM cache).
