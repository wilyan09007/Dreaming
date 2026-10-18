# S6b - Context pruner, second pass: typed yes/no output, pinned low-noise teacher, majority-of-3 labels

## Setup
- **Why a second pass:** S6 used a free-text teacher (gpt-5.4-nano) that disagreed with itself on 36% of identical prompts, so single-call ablation labels were noise and no agreement could be certified. S6b changes two things: (1) a typed output (one word, yes/no, anything else = parse failure), (2) a low-noise teacher with the provider pinned, and every label is a majority of 3 calls.
- **Data:** HotpotQA distractor validation, `type == "comparison"` and gold answer exactly yes/no: 458 candidates; 200 drawn with seed 0 (also requiring exactly 10 paragraphs and exactly 2 gold supporting titles; yes share 0.525). Split by question: first 120 = scorer training, last 80 = held-out. Prompt: system "Answer the question using only the context. Reply with exactly one word: yes or no."; user = `[n] Title: text` paragraphs, then `Question: ...`. Mean full prompt 1,989 tokens, mean completion 3.4 tokens.
- **Teacher:** `anthropic/claude-haiku-5.5`, `reasoning.enabled=false`, `max_tokens=5`, `usage.include=true`. 3 unpinned probe calls all reported provider "Claude Platform on AWS"; every later call used `provider={"order": ["Claude Platform on AWS"], "allow_fallbacks": false}` (the display name is accepted as the slug). Verified over all 9,706 non-probe responses: provider = Claude Platform on AWS on 9,706/9,706, reasoning tokens = 0 on all, model = claude-haiku-5.5. Parse failures 5/7,200 label-phase calls (all `finish=length`, 0.07%); 2 of 2,000 ablation majorities were "fail" (3-way split counts as fail), 0 of 200 full-context majorities.
- **Labels:** A_full = majority of 3 full-context calls (reps 0-2). I also made 3 more independent full-context calls (reps 3-5) to measure majority-vs-majority. For each of the 10 paragraphs: remove it, ask 3 times, majority; "matters" iff that majority differs from A_full. 200 x 11 x 3 = 6,600 label calls (+600 extra full-context calls). The pilot on 5 questions projected $2.47 for 200 questions (< $3 threshold), so no shrink to 150.
- **Scorers (rank the 10 paragraphs per question):** (1) learned: StandardScaler + LogisticRegression on 7 features [cosine(bge-small question, paragraph), position, word length, question content-word overlap, title-in-question, cosine rank within question, cosine minus question max], trained on the 120 training questions' majority ablation labels (1,200 rows, 96 positive); (1b, my extra) the same model trained on gold-supporting labels instead; (2) zero-shot `cross-encoder/ms-marco-MiniLM-L-6-v2`; (3) oracle = the 2 gold paragraphs first, remaining slots random; (4) random. Top-k kept in original order, k in {2, 3, 4}, then re-asked 3 times (majority) on the 80 held-out questions.
- **Metrics:** agreement of the pruned majority with A_full (n = 80), Clopper-Pearson one-sided 95% upper bound (UB) on disagreement (`cp_upper`), tokens removed from API `prompt_tokens`. "Certifiable at budget b" = UB <= b on the 80 rows. The selective version (`certify`, SGR) uses the score gap between the k-th and (k+1)-th ranked paragraph as confidence and falls back to the full prompt below the cutoff.
- **Files:** `run.py` (resumable; `responses.jsonl` is cache and ledger, 9,709 raw responses), `log.txt` (final run, all from cache), `log_live_run.txt` / `stdout_live_run.txt` / `results_live_run.json` (the original live run with timings), `results.json`, `xenc_scores.json`.
- **Provenance:** all API calls were made by the earlier, interrupted attempt (it finished all calls and its printout but wrote no RESULTS.md). In this session I made **0 new API calls**; I re-ran `run.py` from the cache after fixing a reporting bug (the "both gold kept" column used `.all()` over the kept mask, which is only correct for k=2, so it printed 0.00 for k=3,4 in the live run) and adding diagnostics (label reliability, bootstrap AUC CIs, error breakdown, SGR effective tokens, provider check over all calls). No result in the live run changed apart from that column.

## Results

### 1. Teacher noise and label quality (200 questions)
| quantity | value |
|---|---|
| Full prompt x 3 calls: all three identical | 97.5% of questions |
| Full prompt x 3 calls: pairwise disagreement (noise floor) | **1.67%** (S6 nano free-text: 36%; S13 haiku Banking77: 3.5%) |
| Full prompt x 6 calls: all identical / pairwise | 96.5% / 1.57% |
| Majority-of-3 vs independent majority-of-3 on the same full prompt | disagree 1/200 = 0.5% |
| A_full accuracy vs gold | 0.880 (yes share of A_full 0.505) |
| Fraction of paragraphs that matter (majority label) | **0.076**: gold 0.228, non-gold 0.038 (S6 strict: 0.389 / 0.554 / 0.348) |
| Same, single call (rep 0) | 0.075: gold 0.233, non-gold 0.036 |
| Mean mattering paragraphs per question | 0.76; 59% of questions have none (82/200 have at least one) |
| A_full right on questions with a mattering paragraph vs the rest | 0.780 vs 0.949; 2.04 mattering paragraphs/question when A_full is wrong vs 0.59 when right |
| Label reliability: Jaccard of positive sets between two single-call replicates | 0.79 |
| P(single-call flip given majority label = matters) / given not | 0.943 / 0.005 |
| Noise-ceiling AUC (one replicate's flips used as a scorer for another replicate's flips) | 0.937 |

### 2. Scorer AUC at predicting the majority ablation label, held-out 80 questions (800 paragraphs, 56 positives)
Pooled mixes questions; within-question is the mean per-question AUC over the 26 held-out questions that have at least one mattering paragraph (the only ones where it is defined). CI = bootstrap over questions (1,000 resamples).

| scorer | ablation AUC pooled [95% CI] | ablation within-q (n_q = 26) | gold-label AUC pooled / within-q |
|---|---|---|---|
| learned (trained on ablation labels) | **0.714** [0.637, 0.797] | 0.749 | 0.991 / 0.992 |
| learned_gold (extra, trained on gold labels) | 0.668 [0.582, 0.775] | 0.737 | 0.998 / 0.996 |
| cross-encoder | 0.659 [0.564, 0.768] | 0.712 | 0.939 / 0.977 |
| oracle (gold paragraphs) | 0.648 [0.564, 0.767] | 0.737 | 1.000 / 1.000 |
| random | 0.578 [0.502, 0.659] | 0.628 | 0.515 / 0.505 |

Learned minus cross-encoder pooled ablation AUC: +0.055, bootstrap 95% CI [-0.004, +0.115] (not clearly different). Held-out matters rate 0.070 (gold 0.169, non-gold 0.045). Learned standardized coefficients: cos +1.01, title-in-question +0.46, question overlap -0.22, cosine rank +0.22, position -0.17, cos-minus-max -0.15, length -0.07. S6 for comparison: learned 0.52-0.53, oracle 0.57-0.66 (free-text strict labels).

### 3. Pruned re-asks on the 80 held-out questions (majority of 3 vs A_full)
UB = Clopper-Pearson 95% upper bound on the disagreement rate. "Per-call agree" = mean agreement of the individual calls. "Both gold kept" = both supporting paragraphs survived pruning. Reference: the full context asked again (independent majority).

| scorer | k | agree | disagree | UB95 | per-call agree | accuracy vs gold | tokens removed | both gold kept |
|---|---|---|---|---|---|---|---|---|
| full context, 2nd majority | 10 | 1.000 | 0/80 | 0.037 | 0.988 | 0.863 | 0% | 1.00 |
| learned | 2 | 0.925 | 6/80 | 0.143 | 0.921 | 0.912 | 82.0% | 0.96 |
| learned | 3 | 0.912 | 7/80 | 0.158 | 0.917 | 0.875 | 73.9% | 0.99 |
| learned | 4 | 0.887 | 9/80 | 0.188 | 0.892 | 0.900 | 64.7% | 0.99 |
| learned_gold (extra) | 2 | 0.925 | 6/80 | 0.143 | 0.921 | 0.912 | 82.0% | 0.99 |
| learned_gold (extra) | 3 | 0.912 | 7/80 | 0.158 | 0.925 | 0.900 | 73.1% | 0.99 |
| learned_gold (extra) | 4 | 0.925 | 6/80 | 0.143 | 0.929 | 0.887 | 64.4% | 0.99 |
| cross-encoder | 2 | 0.838 | 13/80 | 0.246 | 0.833 | 0.825 | 82.0% | 0.75 |
| cross-encoder | 3 | 0.887 | 9/80 | 0.188 | 0.900 | 0.850 | 72.9% | 0.90 |
| cross-encoder | 4 | 0.912 | 7/80 | 0.158 | 0.912 | 0.875 | 63.9% | 0.99 |
| oracle | 2 | 0.925 | 6/80 | 0.143 | 0.921 | 0.912 | 81.8% | 1.00 |
| oracle | 3 | 0.925 | 6/80 | 0.143 | 0.929 | 0.912 | 71.7% | 1.00 |
| oracle | 4 | 0.950 | 4/80 | 0.111 | 0.938 | 0.875 | 60.6% | 1.00 |
| random | 2 | 0.625 | 30/80 | 0.473 | 0.621 | 0.613 | 78.7% | 0.03 |
| random | 3 | 0.688 | 25/80 | 0.408 | 0.692 | 0.675 | 69.2% | 0.05 |
| random | 4 | 0.738 | 21/80 | 0.356 | 0.729 | 0.700 | 58.9% | 0.16 |

Where the disagreements come from. On the held-out set A_full is wrong against gold on 11/80 questions (accuracy 0.863). Learned k=2: 6 disagreements, 5 of which are questions where A_full is wrong and the pruned answer is the gold answer (1 true loss). Oracle k=2: 6, same 5. Oracle k=4: 4, 2 of them pruned-right. Cross-encoder k=2: 13, 9 with a gold paragraph dropped. Random: all 21-30 disagreements involve a dropped gold paragraph.

### 4. Certifiability (UB computed on the 80 held-out rows; delta = 0.05)
Whole-set certifiable = UB <= budget for the pruned prompt on all questions. "n needed" = smallest n at which the Clopper-Pearson bound at the observed disagreement rate would be <= budget ("never" = observed rate >= budget). SGR = selective version with fall-back to the full prompt; effective tokens removed = coverage x tokens removed.

| scorer | k | observed disagree | whole-set certifiable 5% / 10% / 15% | n needed 5% / 10% / 15% | SGR coverage 5% / 10% / 15% |
|---|---|---|---|---|---|
| learned | 2 | 7.5% | N / N / **Y** | never / 379 / 59 | 0 / 0 / 0.613 (emp. error 2.0%, effective tokens removed 50.2%) |
| learned | 3 | 8.8% | N / N / N | never / 1,537 / 85 | 0 / 0 / 0 |
| learned | 4 | 11.3% | N / N / N | never / never / 243 | 0 / 0 / 0 |
| learned_gold (extra) | 2 | 7.5% | N / N / Y | never / 379 / 59 | 0 / 0 / 0 |
| learned_gold (extra) | 3 | 8.8% | N / N / N | never / 1,537 / 85 | 0 / 0 / 0 |
| learned_gold (extra) | 4 | 7.5% | N / N / Y | never / 379 / 59 | 0 / 0 / 0 |
| cross-encoder | 2 | 16.3% | N / N / N | never / never / never | 0 / 0 / 0 |
| cross-encoder | 3 | 11.3% | N / N / N | never / never / 243 | 0 / 0 / 0 |
| cross-encoder | 4 | 8.8% | N / N / N | never / 1,537 / 85 | 0 / 0 / 0 |
| oracle | 2 | 7.5% | N / N / Y | never / 379 / 59 | n/a (no confidence) |
| oracle | 3 | 7.5% | N / N / Y | never / 379 / 59 | n/a |
| oracle | 4 | 5.0% | N / N / Y | never / 89 / 40 | n/a |
| random | 2-4 | 26-38% | N / N / N | never / never / never | n/a |
| full context, 2nd majority | 10 | 0.0% | Y / Y / Y | 59 / 29 / 19 (at zero observed errors) | n/a |

## Interpretation
- **Yes: typed output plus a pinned low-noise teacher makes the ablation labels repeatable, which S6 could not claim.** Identical full prompts now disagree 1.67% of the time (36% in S6), 97.5% of questions are unanimous over 3 calls, and two independent majorities disagree 0.5%. A single-call ablation flip equals the majority flip 94% of the time (0.5% false flips), the noise-ceiling AUC of the labels is 0.94, and the single-call matters rate (0.075) equals the majority rate (0.076). With this teacher, majority-of-3 is nearly redundant: one call per ablation would give essentially the same labels at a third of the cost (about $0.4 instead of $1.2 for the label calls).
- **But repeatable is not rich, and the scorer only partly learns them.** Only 7.6% of paragraphs matter (22.8% of gold, 3.8% of non-gold; 59% of questions have none), against 38.9% in S6. Removing a gold paragraph from a yes/no comparison question flips the answer only 23% of the time, and the mattering paragraphs concentrate on questions where the full-context answer is already fragile or wrong (2.04 per question vs 0.59). So the label is mostly a "this question is fragile" signal, not a "this paragraph is needed" signal. Learned AUC at the ablation label is 0.714 [0.64, 0.80] pooled (0.75 within question on 26 questions), up from 0.52 in S6, above random (0.58) and the cross-encoder (0.66), but the gap to the cross-encoder is not significant and the gold oracle itself only reaches 0.65. Against the gold label the same learned model reaches AUC 0.99: finding the relevant paragraphs is solved on this task (the question names both entities and the title-in-question feature carries it) while predicting ablation sensitivity is not.
- **The ablation labels did not add deployment value over gold relevance here.** A scorer trained on gold labels prunes as well or better (agreement 0.925 / 0.912 / 0.925 at k = 2 / 3 / 4 vs 0.925 / 0.912 / 0.887 for the ablation-trained one), so the ablation calls bought nothing a gold-labeled set would not. The mission's premise (labels from ablation, no annotation) is validated mechanically, not economically, on this call site.
- **What is deployable, and at which budget.** Best setting: learned (or gold-supervised) scorer at k=2: removes 82.0% of input tokens, keeps both gold paragraphs in 96-99% of held-out questions, agrees with the full-context majority on 92.5% (6/80, UB 14.3%) and is more accurate than the full prompt against gold (0.912 vs 0.863; within noise at n = 80). It is certifiable at the 15% disagreement budget on the 80 held-out rows, not at 10% or 5%. At the observed 7.5% rate a 10% budget needs about 379 held-out rows (assuming the rate holds); a 5% budget cannot be reached, because even the gold oracle disagrees with the full-context teacher 5-7.5% of the time. With selective gating (use the pruned prompt only when the k-th vs (k+1)-th score gap is large, otherwise send the full prompt) the learned k=2 scorer covers 61% of questions at 2.0% empirical error under a 15% certified budget, an effective 50% input-token saving; nothing is selectively certifiable at 10% or 5%. The zero-shot cross-encoder is not deployable at k=2 (drops a gold paragraph on 25% of questions, 16% disagreement) and needs k=4 (64% tokens removed, 8.8% disagreement, UB 15.8%) to approach the same quality; it does not certify at n = 80 at any budget. Random pruning is far off (26-38% disagreement), so the agreement numbers are not a binary-chance artifact (chance agreement on a yes/no output is about 50%; random k=2 gets 62.5%).
- **The remaining disagreement is mostly the teacher being wrong on the full prompt, not pruning harm.** 5 of the 6 disagreements of the gold-keeping k=2 scorers are questions where A_full contradicts gold and the pruned prompt gets it right; about 1 in 80 is a true loss. Agreement-with-full is therefore a conservative metric on this task: it penalizes the pruner for fixing distractor-induced errors, and a certificate on it cannot go below roughly 5-7% here no matter how good the scorer. Certifying tighter budgets would need a different reference (for example non-inferiority against gold, or agreement with a majority over both prompts), which is a different claim.

## Caveats
- **Single narrow task.** HotpotQA comparison yes/no, 10 paragraphs, 2 gold, one teacher on one endpoint. The questions name both entities, which makes gold retrieval easy (gold AUC 0.99) and is why the pruner looks good. Bridge questions, free-text outputs and other call sites are not tested; the typed-output fix addresses noise, but a free-text step cannot use it.
- **Small held-out set.** n = 80 questions, 56 mattering paragraphs; the ablation AUC CIs are about +/-0.08 and within-question AUC rests on 26 questions. Agreement counts of 6/80 vs 7/80 vs 9/80 are not distinguishable between learned k=2..4 and cross-encoder k=4. The "certifiable at 15%" entries sit near the edge (UB 0.143 vs 0.150) and depend on one disagreement.
- **Selection.** I evaluated 15 scorer+k configurations on the same 80 rows and report the best; no multiplicity correction. The SGR certificate for learned k=2 has its own delta split over the cutoff search, but the choice of this configuration among the others is not corrected, and the learned model was fit on only 96 positive rows. A fresh held-out split is needed before promotion.
- **Agreement vs reference.** A_full is itself wrong on 12% of questions (0.88 accuracy; 11/80 held-out). Agreement was defined against the teacher's full-context majority as in the mission, not against gold.
- **Correlated re-asks.** Scorers that choose the same paragraph set at the same k share the same 3 cached calls (keys dedup by question + kept set), so rows for oracle, learned and learned_gold are not independent samples.
- **Provider pinning.** The pin used the display name "Claude Platform on AWS" as the slug; OpenRouter accepted it and served 9,706/9,706 non-probe calls from it. S13 showed other endpoints of the same model differ; these results hold for this endpoint only. The 3 probe calls were unpinned but reported the same provider.
- **Noise estimate** rests on 200 prompts x 3 or 6 calls (no bootstrap CI computed; with 200 prompts the 1.67% figure is roughly +/-1 pp).
- The live run's "both gold kept" column was wrong for k=3,4 (printed 0.00); the tables above are from the corrected re-run on the same cached responses.

## Time and cost
- 9,709 API calls, all claude-haiku-5.5: 3 unpinned probe + 1 pin test + 7,200 label-phase calls (6,600 ablation/full + 600 extra full; the 165 pilot calls are part of these) + about 2,500 distinct pruned re-ask calls after dedup. Average $0.000153 per call, 14.7M input tokens. **Total $1.4842** (from `usage.cost`; about $1.33 through the label phase, $0.15 for the pruned re-asks), under the $4 cap and under the $3 pilot threshold (pilot projected $2.47; actual was lower because pruned prompts are cheaper than full ones). This session spent $0.00: every response was served from the cache left by the interrupted attempt.
- Wall clock of the live run 1,136 s (about 19 min with 8 threads: 723 s through the label phase, about 165 s of scorer features including embedding 2,200 texts on CPU, about 250 s for the pruned re-asks). The cached re-run takes about 18 s.
