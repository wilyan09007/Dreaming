# S5: Tier / thinking router (Mission 1)

Files: `run.py` (stages: `check`, `a`, `probe`, `live`, `analyze`, `evidence`), `rb_load.py` (allow-list unpickler for the RouterBench pickle), `responses.jsonl` (2,200 raw OpenRouter responses with tokens, latency, cost), `evidence.jsonl` (238 raw calls from the `evidence` stage), `rb_data/routerbench_0shot.pkl` (downloaded, 99.6 MB). Logs: `log.txt` is the concatenation of every stage log with a header per section: `probe_log.txt` and `live_log.txt` (original runs, unchanged), `check_log.txt`, `rb_load_check_log.txt`, `alpha_log.txt` (`alpha_check`: how the ridge strengths were picked), `partA_log.txt` (revised `a`), `analyze_log.txt` (revised `analyze`), `evidence_log.txt`. `v1_orig/` keeps the first version of `run.py`, `RESULTS.md`, and logs for comparison.

## Revision (v2): what changed and how the numbers moved

A review found that the v1 Part A conclusion ("a prompt router captures 8-11% at <=2pt and 18-34% at <=5pt, about half from knowing the benchmark") was not supported. v2 adds a random-mix baseline, a cost-aware ranking, equal-cost comparisons, bootstrap intervals, and a cutoff chosen on a calibration half. Part B data was not re-collected; the analysis was re-run on the saved responses.

**Part A, main change.** A random mix (route to the cheap model with probability p, no features) is linear between always-expensive and always-cheap. v1's embedding router (ranks by P(cheap >= expensive)) is at or below that line at every tolerance, and the v1 tables compared policies at equal share routed, not at equal cost (the router sent cheap-to-serve prompts, so a 30% share saved less money than a random 30%). Max cost saving vs always-GPT-4 within an accuracy-loss tolerance, same data, same cutoff rule (cutoff chosen on the same out-of-fold data, so these are optimistic for every learned policy):

| setup | policy | <=0.5pt | <=1pt | <=2pt | <=5pt |
|---|---|---|---|---|---|
| gpt-3.5, bge-small (5k) | v1 router, as reported | 3.1 | 5.4 | 9.4 | 22.0 |
| | v2 router (same policy, every cutoff instead of 101 shares) | 3.4 | 5.6 | 9.4 | 22.3 |
| | **random mix** | **3.0** | **6.0** | **12.0** | **30.0** |
| gpt-3.5, TF-IDF (36.5k) | v1 router | 3.7 | 5.8 | 11.2 | 33.7 |
| | v2 router | 3.9 | 6.1 | 11.4 | 33.9 |
| | **random mix** | **2.9** | **5.7** | **11.4** | **28.6** |
| mixtral, bge-small (5k) | v1 router | 2.0 | 3.6 | 8.3 | 17.6 |
| | v2 router | 2.0 | 3.8 | 8.7 | 17.8 |
| | **random mix** | **2.0** | **4.1** | **8.1** | **20.3** |
| mixtral, TF-IDF (36.5k) | v1 router | 2.6 | 4.3 | 8.4 | 19.8 |
| | v2 router | 2.6 | 4.7 | 8.8 | 20.2 |
| | **random mix** | **2.0** | **4.1** | **8.2** | **20.5** |

The random-mix rows reproduce the reviewer's figures. The v1 "family prior" (per-benchmark base rate of cheap >= expensive) is worse than random at 5pt (19% vs 30% for gpt-3.5), so "about half of that comes from knowing the benchmark" is withdrawn: the router has no demonstrated lift, and the benchmark prior as built in v1 did not help either. What does carry the savings is cost concentration (see Part A results), which v1 never measured.

**Other changes**

| item | v1 | v2 |
|---|---|---|
| Interpretation of Part A | "router weak, captures 8-11% / 18-34%" | no better than random mixing for savings at <=5pt loss; AUC 0.63 does not convert to savings; large savings exist only through cost-aware ranking, which comes from benchmark identity and dollar concentration |
| llama-2-70b / mistral-7b pairs | not flagged | flagged: llama-2-70b scores 0.027 on MMLU (below chance 0.25, unparsed output); mistral-7b is at chance on MMLU/hellaswag/winogrande. llama-70b "49.3% oracle saving" is understated by this artifact (55.1% without MMLU) |
| within-family AUC | "0.45-0.65" | full range with n: bge gpt-3.5 0.179 (mtbench, n=11) to 0.637; mbpp 0.356 (n=58); mixtral winogrande 0.442 (n=174). n<100 flagged. TF-IDF on all rows: 0.484-0.702 |
| `rb_load.py` | substring filter on callable names (an exec-capable numpy callable passed it) | exact allow-list of the 11 globals the file references; `python rb_load.py` self-check shows `numpy.testing._private.utils.runstring` is blocked |
| min zero-error rows for a certificate | 258 / 102 / 50 (N=200, m=9) | 252 / 99 / 49 on the 100 calibration rows actually used (m=8); 258 / 102 / 50 if all 200 rows were calibration. Conclusion unchanged |
| Part B router split | OOF scores on all 200 rows, then split (holdout labels influenced calibration scores) | split first; calibration scores are OOF inside the calibration half; holdout scores come from a model fit on the calibration half only |
| "empirical 15%" result | 45% routed, 6.7% disagreement (3/45, 95% CI 1.4-18.3%) | disappears on the clean split (no cutoff meets 15% on the seeded split). Over 20 random splits at budget 15%: no cutoff in 2/20; otherwise 56% routed (sd 22) with 14.4% holdout disagreement (n=1013). Not a result |
| "Ton changes 16.5% of answers" | no noise floor | same-config noise floor (C2 vs C2b) is 9.0%, so about 7.5 points of flips are beyond noise (rough). Accuracy effect holds: Ton right only 18 vs C2 right only 4 (exact sign test p=0.0043); vs C2b 18 vs 5 (p=0.0106) |
| "+6.7 pts (0.84 vs 0.77)" | | +7.0pt vs C2 alone, +6.5pt vs C2b, +6.75pt vs their mean (0.7725) |
| temperature / qwen claims | partly unlogged (about $0.004) | re-run and saved to `evidence.jsonl` ($0.174, see Part B); same conclusions |
| latency means | included 3 cold probe calls per config; configs ran at different times | cold calls (i<3) excluded: E1 2311 -> 2298 ms, C1 843 -> 826, C2 866 -> 851, Ton 2665 -> 2651. Plus an interleaved check (30 inputs, 7 configs together): E1/C1 2.1x (was 2.7x), Ton/C2 3.6x mean, 2.4x median (was 3.1x) |
| setup note on C2b | "added after finding the noise problem ($0.75)" | C2b ran in the original batch; E1b, C1b and E2b were added later: $0.733 for those three, $0.806 with C2b |
| `run.py` comment "latest" | code used the first record | comment fixed; `responses.jsonl` has no duplicate (cfg, i) so first = only |
| log files | `log.txt` mixed stages, extra logs unlisted | listed above, section headers in `log.txt` |

Not changed: the Part B accuracy, match, and cost tables (same raw data), the Part A agreement/oracle table (except the two flagged pairs), the certificate results (nothing certifiable).

## Setup

**Part A (offline, free).** `withmartian/routerbench` is public, no login. `routerbench_0shot.pkl` (99.6 MB; the 5-shot is 171 MB, the raw file 1.2 GB, not used) loaded with an allow-list unpickler (11 exact globals, all pandas/numpy reconstruct functions; enumerated with an inert-stub unpickler that imports and runs nothing). 36,497 prompts x 11 models (2023 vintage), 8 benchmark families (mmlu 14,042; hellaswag 10,042; gsm8k 7,450; other 1,719; arc 1,470; winogrande 1,267; mbpp 427; mtbench 80). Columns used: `<model>` (score in [0,1]; binary for arc/hellaswag/mbpp/mmlu/winogrande, 0.25 steps for gsm8k and "other", 0.1 steps for mtbench), `<model>|total_cost` (USD per sample at 2023 prices). Definitions: "ok" = score >= 0.5; "agree" = both ok or both not ok; `y1 = (score_cheap >= score_expensive)` ("cheap is at least as good"; ties, including both wrong, count as cheap fine); `y2 = (score_cheap >= 0.5)` (the spec's target P(cheap correct)); per-row accuracy loss from routing to cheap `dacc = score_exp - score_cheap`; per-row dollars saved `dcost = cost_exp - cost_cheap`. Pairs: gpt-4-1106-preview vs gpt-3.5-turbo-1106, mixtral-8x7b-chat, llama-2-70b-chat, mistral-7b-chat; plus claude-v2 vs claude-instant-v1.

Routers (gpt-3.5 and mixtral vs gpt-4; features either bge-small embeddings of `prompt[:400]` on a 5,000-row subsample stratified by family, seed 0, because CPU embedding ran at about 3.6 texts/s, or TF-IDF 1-2 grams, 30k features, on all 36,497 rows). All scorers are fit out-of-fold (5-fold, seed 0) unless stated. Policies, each a ranking of prompts to send to the cheap model first:

| policy | score |
|---|---|
| random mix | none; analytic line between always-expensive and always-cheap |
| family P(cheap>=exp) | v1 "family prior": per-benchmark base rate of y1 |
| family loss/$ | per-benchmark mean dacc / mean dcost (no ML, only the benchmark label and its average gap and dollars) |
| router P(cheap>=exp) | LogisticRegression(C=1) on y1 (the v1 router) |
| P(cheap ok) [spec] | LogisticRegression(C=1) on y2 (the mission's stated target) |
| cost-aware | ridge-predicted dacc / ridge-predicted dcost (fractional-knapsack greedy: lowest predicted loss per dollar first); ridge alpha 10 for dacc, 1 for dcost, picked from {0.1, 1, 10, 100, 1000} by OOF R2 on the 5k embedding features, reused for TF-IDF |
| saving-only | ridge-predicted dcost (quality-blind: send the prompts that cost the most to the cheap model) |
| oracle (true loss/$) | hindsight ranking by true dacc / dcost; an upper bound, not a policy |

Evaluation: (1) equal-cost view: accuracy loss at a fixed saving; (2) max saving within a tolerated mean accuracy loss, cutoff picked on the same OOF data (optimistic), with a 200-resample bootstrap over rows for the gap to the random mix; (3) clean split: a random half is the calibration half (scorer fit on it only, inner 5-fold OOF for its scores), the cutoff is the largest saving within tolerance on it, realised on the other half (20 random halves for the 5k embedding runs, 10 for TF-IDF); the random mix picks its share on the calibration half the same way. Certificate: `certify()` on a random half of OOF scores for y1.

**Part B (live, OpenRouter).** 200 Banking77 test inputs from `stratified_subsample(test_labels, 200, seed=0)` (all 77 classes present). System prompt exactly as specified (77 labels one per line), user = query, `temperature=0`, `max_tokens` 60 (reasoning off) / 2000 (E1) / 4000 (T-on), 8 threads. Parse: exact match after strip/lower (also stripping quotes/backticks/trailing period); else nearest label by `difflib` similarity and counted as a parse failure. Spend tracked from `usage` tokens x the given prices (equals OpenRouter's reported `usage.cost` to the digit).

| cfg | model | reasoning setting | note |
|---|---|---|---|
| E1 | openai/gpt-5.4 | effort "low" | expensive reference (default effort is not "none"; "low" as the task allowed) |
| C1 | openai/gpt-5.4-nano | effort "none" | verified: 0 reasoning tokens (gpt-5.4 family has no "minimal", "none" is the off switch) |
| C2 | openai/gpt-5.4-mini | effort "none" | 0 reasoning tokens; also serves as T-off |
| C3 | qwen/qwen3-32b | `enabled:false` + " /no_think" appended to the user message | the OpenRouter reasoning param is ignored by the provider (DeepInfra): `effort:none`, `enabled:false`, `exclude`, `chat_template_kwargs` all still produced 65-77 reasoning tokens and hit the length cap (re-verified in `evidence.jsonl`, below). Only the Qwen3 soft switch worked (1 reasoning token). |
| Ton | openai/gpt-5.4-mini | effort "high" | T-on |
| E2 | anthropic/claude-sonnet-4.6 | `enabled:false` | 0 reasoning tokens |
| C4 | anthropic/claude-haiku-4.5 | `enabled:false` | 0 reasoning tokens |
| C2b | repeat of C2 | same | ran in the original batch (it appears in `probe_log.txt`) |
| E1b, C1b, E2b | repeats of E1, C1, E2 | same | added later, after finding that OpenAI ignores temperature=0 through OpenRouter ($0.733 for the three; $0.806 with C2b) |

## Results

### Part A: RouterBench

Per-model (all 36,497 prompts): gpt-4 acc 0.781, $3.29/1k calls; gpt-3.5 0.619, $0.24; mixtral 0.547, $0.14; claude-v2 0.636, $2.42; claude-instant 0.598, $0.23; llama-2-70b 0.329, $0.20; mistral-7b 0.306, $0.05.

Agreement and oracle, cheap vs expensive (all benchmarks). "Oracle" = route to cheap iff y1:

| expensive | cheap | acc exp | acc cheap | agree (both ok / both not ok) | P(cheap ok given exp ok) | oracle acc | oracle cost saving | cheap/exp cost |
|---|---|---|---|---|---|---|---|---|
| gpt-4 | gpt-3.5 | 0.781 | 0.619 | 0.730 | 0.734 | 0.833 | 67.3% | 0.074 |
| gpt-4 | mixtral-8x7b | 0.781 | 0.547 | 0.654 | 0.632 | 0.821 | 58.1% | 0.041 |
| gpt-4 | llama-2-70b (**grader artifact on MMLU**) | 0.781 | 0.329 | 0.471 | 0.394 | 0.804 | 49.3% | 0.062 |
| gpt-4 | llama-2-70b, MMLU excluded (n=22,455) | 0.762 | 0.517 | 0.631 | 0.607 | 0.799 | 55.1% | 0.056 |
| gpt-4 | mistral-7b (**at chance on MMLU/hellaswag/winogrande**) | 0.781 | 0.306 | 0.396 | 0.319 | 0.814 | 39.1% | 0.014 |
| gpt-4 | mistral-7b, gsm8k/mbpp/mtbench/other only (n=9,676) | 0.644 | 0.394 | 0.462 | 0.402 | 0.668 | 38.7% | 0.011 |
| claude-v2 | claude-instant | 0.636 | 0.598 | 0.779 | 0.802 | 0.723 | 74.2% | 0.096 |

Grader/parse artifacts (per-model by-family accuracy matrix in `log.txt`, `!` = at or below multiple-choice chance + 0.03): llama-2-70b scores 0.027 on MMLU (4-way, chance 0.25) while 0.734 on ARC and 0.526 on hellaswag, so the MMLU output was almost certainly not parsed (its winogrande 0.482 is also below the 2-way chance of 0.5); code-llama-34b scores 0.005 on MMLU (not used in the pairs). mistral-7b is 0.254 on MMLU, 0.255 on hellaswag, 0.524 on winogrande (chance 0.25 / 0.25 / 0.5) and 0.388 on ARC; whether that is parsing or capability cannot be decided from the scores. Effect on the table: for llama the artifact removes MMLU from the oracle's savings (20.3% on MMLU vs 55.1% overall without it), so 49.3% is an understatement, not an inflation; for any model near chance the tie rule (both wrong counts as "cheap fine") makes the oracle saving partly an artifact of GPT-4's own failures. Read the llama-70b and mistral-7b rows as unreliable. Rows for gpt-3.5, mixtral and the claude pair show no below-chance cells.

By benchmark, gpt-4 vs gpt-3.5 (the oracle can exceed gpt-4 accuracy because it picks whichever scored higher):

| family | n | acc gpt-4 | acc gpt-3.5 | agree | P(cheap ok given exp ok) | oracle acc | oracle cost saving |
|---|---|---|---|---|---|---|---|
| arc | 1470 | 0.962 | 0.831 | 0.850 | 0.854 | 0.971 | 77.0% |
| gsm8k | 7450 | 0.659 | 0.605 | 0.841 | 0.847 | 0.699 | 67.1% |
| hellaswag | 10042 | 0.840 | 0.587 | 0.620 | 0.623 | 0.903 | 58.7% |
| mbpp | 427 | 0.686 | 0.653 | 0.813 | 0.840 | 0.763 | 85.0% |
| mmlu | 14042 | 0.812 | 0.647 | 0.738 | 0.737 | 0.860 | 69.7% |
| mtbench | 80 | 0.806 | 0.689 | 0.800 | 0.800 | 0.860 | 50.1% |
| other | 1719 | 0.564 | 0.484 | 0.768 | 0.741 | 0.637 | 81.0% |
| winogrande | 1267 | 0.819 | 0.579 | 0.621 | 0.622 | 0.889 | 62.5% |

**Where the dollars are.** GPT-4's cost is concentrated: gsm8k is 20% of rows but 53% of GPT-4 dollars (long chain-of-thought outputs; $8.55 per 1k calls vs $1.43 for MMLU) with only a 5.4pt accuracy gap to gpt-3.5 (mbpp: 1% of rows, 3% of dollars, 3.3pt gap). hellaswag is 28% of rows, 18% of dollars, 25.3pt gap. Ranked by GPT-4's own cost, the top 10% of rows hold 34% of dollars and routing them to gpt-3.5 loses 0.80pt overall; the top 20% hold 56% of dollars and lose 1.2pt (5k subsample, gpt-3.5; mixtral: 1.7pt and 2.9pt).

**Equal-cost view, gpt-4 vs gpt-3.5, bge-small + LR, 5,000-row subsample, out-of-fold.** Accuracy loss in points at a given cost saving vs always-gpt-4 (lower is better; base rate y1 = 0.752; always-expensive acc 0.7865 at $3.28/1k, always-cheap acc 0.6320 at $0.24/1k, max saving 92.6%). AUC of the router for y1: 0.631 (family prior 0.559); P(cheap ok) model AUC 0.679 for y2. OOF R2 of the ridge regressors: dacc 0.043, dcost 0.725.

| saving | random mix | family P(cheap>=exp) | family loss/$ | router P(cheap>=exp) | P(cheap ok) [spec] | cost-aware | saving-only |
|---|---|---|---|---|---|---|---|
| 10% | 1.67 | 1.03 | 0.31 | 2.24 | 0.75 | 0.43 | 0.18 |
| 20% | 3.34 | 4.95 | 0.57 | 4.37 | 1.03 | 0.71 | 0.45 |
| 30% | 5.00 | 7.12 | 0.84 | 6.68 | 1.34 | 0.98 | 0.67 |
| 40% | 6.67 | 7.70 | 1.05 | 7.22 | 1.49 | 1.18 | 0.82 |
| 50% | 8.34 | 8.03 | 1.20 | 7.53 | 1.80 | 1.32 | 1.12 |

The v1 router (column 5) loses more accuracy than the random mix at every saving from 10% to 40% and is slightly better only at 50% (7.53 vs 8.34); the three cost-aware columns lose 0.2-1.3pt where the random mix loses 1.7-8.3pt. The TF-IDF and mixtral versions are in `log.txt` and agree in direction (e.g. TF-IDF gpt-3.5 at 50% saving: random 8.75, router 6.57, cost-aware 1.06, saving-only 1.08).

**Max cost saving (%) within a tolerated accuracy loss, cutoff chosen on the same OOF data.** Optimistic for the learned policies (best cutoff on the data it is scored on).

| setup | loss <= | random mix | family P(cheap>=exp) | family loss/$ | router P(cheap>=exp) | P(cheap ok) [spec] | cost-aware | saving-only | oracle (hindsight) |
|---|---|---|---|---|---|---|---|---|---|
| gpt-3.5, bge (5k) | 0.5pt | 3.0 | 4.1 | 18.3 | 3.4 | 1.9 | 13.4 | 21.4 | 86.3 |
| | 1pt | 6.0 | 10.0 | 37.4 | 5.6 | 19.6 | 32.2 | 46.6 | 86.7 |
| | 2pt | 12.0 | 12.3 | 61.1 | 9.4 | 52.9 | 60.6 | 59.5 | 87.4 |
| | 5pt | 30.0 | 20.2 | 68.4 | 22.3 | 63.4 | 71.7 | 68.4 | 89.3 |
| gpt-3.5, TF-IDF (36.5k) | 0.5pt | 2.9 | 4.1 | 23.5 | 3.9 | 16.9 | 26.2 | 25.8 | 85.9 |
| | 1pt | 5.7 | 9.7 | 46.4 | 6.1 | 29.3 | 47.7 | 46.4 | 86.3 |
| | 2pt | 11.4 | 11.9 | 60.7 | 11.4 | 45.3 | 62.2 | 61.6 | 87.0 |
| | 5pt | 28.6 | 19.0 | 67.6 | 33.9 | 63.7 | 72.0 | 70.1 | 89.0 |
| mixtral, bge (5k) | 0.5pt | 2.0 | 0.8 | 7.0 | 2.0 | 3.1 | 8.4 | 9.5 | 83.8 |
| | 1pt | 4.1 | 3.3 | 15.9 | 3.8 | 5.4 | 17.5 | 18.6 | 84.3 |
| | 2pt | 8.1 | 5.5 | 33.7 | 8.7 | 7.6 | 34.3 | 36.2 | 85.2 |
| | 5pt | 20.3 | 13.0 | 64.7 | 17.8 | 17.6 | 67.2 | 62.6 | 87.6 |
| mixtral, TF-IDF (36.5k) | 0.5pt | 2.0 | 1.0 | 9.4 | 2.6 | 4.1 | 13.7 | 10.1 | 84.0 |
| | 1pt | 4.1 | 4.1 | 17.7 | 4.7 | 5.8 | 23.9 | 19.7 | 84.5 |
| | 2pt | 8.2 | 6.2 | 36.4 | 8.8 | 11.1 | 43.0 | 39.2 | 85.3 |
| | 5pt | 20.5 | 14.4 | 64.7 | 20.2 | 36.8 | 68.8 | 65.5 | 87.7 |

At 0pt tolerance every non-oracle policy saves 0-3.6% (full table in `log.txt`). The "oracle (hindsight)" column ranks by true loss per dollar and is not attainable.

**Router P(cheap>=exp) minus random mix, saving in points (bootstrap 95% interval over rows, 200 resamples; scores fixed).**

| setup | <=0.5pt | <=1pt | <=2pt | <=5pt |
|---|---|---|---|---|
| gpt-3.5, bge (5k) | +0.6 [-1.3, +2.3] | -0.3 [-1.8, +1.4] | -2.6 [-4.2, -0.5] | -7.5 [-10.0, -4.9] |
| gpt-3.5, TF-IDF | +1.0 [+0.5, +1.5] | +0.4 [-0.2, +1.1] | -0.1 [-1.0, +0.8] | +5.4 [+2.6, +8.4] |
| mixtral, bge (5k) | +0.1 [-0.6, +1.0] | -0.1 [-1.2, +1.2] | +0.6 [-1.3, +2.7] | -2.3 [-4.1, -0.7] |
| mixtral, TF-IDF | +0.7 [+0.4, +1.1] | +0.6 [+0.1, +1.1] | +0.7 [+0.0, +1.6] | -0.2 [-0.9, +0.4] |

Only gpt-3.5 TF-IDF at 5pt (+5.4) is clearly better than a coin-flip mix; the embedding router is significantly worse at 2pt and 5pt for gpt-3.5 and at 5pt for mixtral. Same bootstrap for the cost-aware columns at 2pt (family loss/$, cost-aware, saving-only): +47 to +51 points on gpt-3.5, +26 to +35 on mixtral, all intervals well above zero.

**Clean split** (scorer fit on the calibration half only, cutoff = largest saving within tolerance on it, realised on the other half). Cells: mean saving % / mean realised loss pt / share of splits with realised loss <= tolerance. Shown for tolerance 2pt and 5pt; 0.5pt and 1pt are in `log.txt`.

| setup | loss <= | random mix | family P(cheap>=exp) | family loss/$ | router P(cheap>=exp) | P(cheap ok) [spec] | cost-aware | saving-only |
|---|---|---|---|---|---|---|---|---|
| gpt-3.5, bge, 20 splits | 2pt | 12.2 / 2.07 / 0.30 | 8.3 / 0.84 / 1.00 | 59.7 / 1.56 / 1.00 | 10.7 / 2.41 / 0.10 | 53.5 / 2.29 / 0.25 | 60.4 / 2.00 / 0.50 | 59.3 / 1.97 / 0.55 |
| | 5pt | 30.5 / 5.16 / 0.30 | 22.8 / 6.52 / 0.10 | 62.8 / 2.87 / 0.80 | 22.8 / 5.48 / 0.30 | 64.1 / 5.36 / 0.25 | 70.9 / 5.20 / 0.55 | 68.2 / 4.97 / 0.50 |
| gpt-3.5, TF-IDF, 10 splits | 2pt | 11.5 / 2.04 / 0.40 | 9.7 / 0.95 / 1.00 | 59.8 / 1.58 / 1.00 | 11.9 / 2.20 / 0.30 | 48.1 / 2.11 / 0.30 | 63.1 / 2.17 / 0.20 | 61.9 / 2.15 / 0.10 |
| | 5pt | 28.8 / 5.09 / 0.40 | 23.3 / 6.73 / 0.10 | 64.3 / 3.55 / 0.70 | 31.4 / 5.08 / 0.30 | 63.7 / 5.04 / 0.50 | 72.1 / 5.13 / 0.30 | 70.4 / 5.14 / 0.30 |
| mixtral, bge, 20 splits | 2pt | 8.2 / 2.03 / 0.40 | 3.6 / 0.77 / 1.00 | 38.7 / 2.23 / 0.30 | 9.1 / 2.16 / 0.30 | 8.5 / 2.19 / 0.35 | 47.3 / 2.67 / 0.10 | 51.2 / 2.83 / 0.00 |
| | 5pt | 20.4 / 5.06 / 0.40 | 14.8 / 5.55 / 0.35 | 61.7 / 3.65 / 1.00 | 17.9 / 5.19 / 0.40 | 26.3 / 5.31 / 0.50 | 66.5 / 5.10 / 0.45 | 62.0 / 5.04 / 0.45 |
| mixtral, TF-IDF, 10 splits | 2pt | 8.2 / 1.98 / 0.80 | 5.7 / 0.88 / 1.00 | 45.1 / 2.46 / 0.20 | 9.7 / 2.07 / 0.40 | 11.4 / 2.08 / 0.40 | 42.5 / 2.02 / 0.30 | 38.7 / 2.00 / 0.40 |
| | 5pt | 20.4 / 4.96 / 0.80 | 14.2 / 4.32 / 0.50 | 61.6 / 3.57 / 1.00 | 19.7 / 4.84 / 0.80 | 36.4 / 4.93 / 0.60 | 69.0 / 5.15 / 0.40 | 65.7 / 5.06 / 0.40 |

Three readings. (1) The router P(cheap>=exp) does not beat the random mix on a clean split either (gpt-3.5 bge: 10.7% at 2.41pt realised loss vs random 12.2% at 2.07pt). (2) The cost-aware policies and the no-ML "family loss/$" policy keep most of their advantage out of sample (about 60% saving at a realised loss of 1.6-2.2pt for gpt-3.5; mixtral 39-51% at 2.0-2.8pt). (3) A cutoff placed at the tolerance boundary on the calibration half overshoots the tolerance on the other half in many splits (share within tolerance 0.0-0.55 for the cost-aware and saving-only columns at 2pt): selecting the largest saving within tolerance is a winner's-curse estimator, which is the reason for a certificate or a margin in a real system. Family loss/$ is a step function over 8 benchmarks, so it lands below the tolerance for gpt-3.5 and overshoots for mixtral.

**Learned quality signal vs benchmark identity and cost.** At 2pt on the same-data table, cost-aware vs saving-only vs family loss/$: gpt-3.5 bge 60.6 / 59.5 / 61.1, gpt-3.5 TF-IDF 62.2 / 61.6 / 60.7, mixtral bge 34.3 / 36.2 / 33.7, mixtral TF-IDF 43.0 / 39.2 / 36.4. At 5pt the cost-aware column leads saving-only by 1.9-4.6 points. The ridge regressor for accuracy loss is nearly uninformative (OOF R2 0.04-0.10) while the one for dollars is good (R2 0.73-0.78); on the gpt-3.5 5k embedding features a per-benchmark mean alone gives R2 0.017 for the loss (ridge 0.043) and 0.77 for the dollars (ridge 0.73; `alpha_log.txt`). No interval was computed for these contrasts, so differences of a few points are not separated from noise here.

**Within-family AUC of the router P(cheap>=exp)** (a measure of signal beyond "which benchmark is this"):

| features | pair | within-family AUC (n) |
|---|---|---|
| bge (5k) | gpt-3.5 | arc 0.542 (201); gsm8k 0.461 (1021); hellaswag 0.635 (1376); **mbpp 0.356 (58, n<100)**; mmlu 0.637 (1924); **mtbench 0.179 (11, n<100)**; other 0.609 (235); winogrande 0.525 (174) |
| bge (5k) | mixtral | arc 0.638 (201); gsm8k 0.489 (1021); hellaswag 0.604 (1376); mbpp 0.496 (58, n<100); mmlu 0.648 (1924); mtbench 0.536 (11, n<100); other 0.613 (235); **winogrande 0.442 (174)** |
| TF-IDF (all) | gpt-3.5 | arc 0.592; gsm8k 0.515; hellaswag 0.625; mbpp 0.499 (427); mmlu 0.658; mtbench 0.547 (80, n<100); other 0.676; winogrande 0.484 |
| TF-IDF (all) | mixtral | arc 0.538; gsm8k 0.554; hellaswag 0.615; mbpp 0.569 (427); mmlu 0.655; mtbench 0.619 (80, n<100); other 0.702; winogrande 0.498 |

Overall AUC for y1: gpt-3.5 0.631 (bge), 0.644 (TF-IDF); mixtral 0.668 (bge), 0.688 (TF-IDF); family-prior AUC 0.559 / 0.575 / 0.625 / 0.637. The tiny-n cells (mbpp, mtbench in the 5k subsample) are noise; the large families sit at 0.46-0.70, and cells at or below 0.5 are gsm8k (bge, gpt-3.5: 0.461), winogrande (mixtral bge 0.442, gpt-3.5 TF-IDF 0.484, mixtral TF-IDF 0.498) and mbpp (0.356, 0.496, 0.499).

**Certificates** (`certify` on y1 = "cheap at least as good", calibration half of the OOF scores, holdout half): nothing certifiable at budgets 1%, 2% and 5% for either pair and either feature set (unchanged from v1).

### Part B: live Banking77 (200 inputs, 77 labels)

Mean/p50 latency exclude the three cold probe calls per config (i<3). Everything else is the same raw data as v1.

| cfg | model | n | gold acc | parse fails | match vs E1 | mean ms | p50 ms | prompt tok | compl tok | reasoning tok | $/1000 calls |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | gpt-5.4, low | 200 | 0.870 | 0 | (1.000) | 2298 | 2228 | 439 | 41.4 | 31.4 | 1.719 |
| E1b | gpt-5.4, low (repeat) | 200 | 0.895 | 0 | 0.940 | 2180 | 2135 | 439 | 42.6 | 32.6 | 1.737 |
| C1 | gpt-5.4-nano, none | 200 | 0.760 | 0 | 0.815 | 826 | 831 | 439 | 8.0 | 0 | 0.098 |
| C1b | gpt-5.4-nano (repeat) | 200 | 0.755 | 0 | 0.805 | 796 | 784 | 439 | 8.0 | 0 | 0.098 |
| C2 (= T-off) | gpt-5.4-mini, none | 200 | 0.770 | 1 | 0.805 | 851 | 821 | 439 | 8.0 | 0 | 0.366 |
| C2b | gpt-5.4-mini (repeat) | 200 | 0.775 | 1 | 0.805 | 831 | 812 | 439 | 8.0 | 0 | 0.366 |
| C3 | qwen3-32b, /no_think | 200 | 0.690 | 1 | 0.770 | 573 | 453 | 434 | 7.8 | 1.0 | 0.037 |
| Ton | gpt-5.4-mini, high | 200 | 0.840 | 0 | 0.890 | 2651 | 2131 | 439 | 166.5 | 156.5 | 1.079 |
| E2 | claude-sonnet-4.6, off | 200 | 0.815 | 0 | 0.860 | 947 | 874 | 568 | 8.5 | 0 | 1.831 |
| E2b | claude-sonnet-4.6 (repeat) | 200 | 0.815 | 0 | 0.855 | 901 | 875 | 568 | 8.5 | 0 | 1.831 |
| C4 | claude-haiku-4.5, off | 200 | 0.770 | 2 | 0.800 | 803 | 769 | 567 | 8.7 | 0 | 0.611 |

Ton reasoning tokens: mean 156, max 1577; all 200 finished with `stop`. Parse failures (5 total, none empty): four near-miss label strings (`getting_physical_card`, `get_virtual_card`) resolved by nearest-label fallback, and one haiku reply (input 197) that was a prose refusal ("I'm specifically designed to classify bank support intents...") which the fallback mapped to an arbitrary label.

Pairwise same-label rate (n=200 each; a 95% interval at 0.8 is about +/-5.5pt):

| pair | match | disagreements | gold acc (first, second) |
|---|---|---|---|
| C2 vs C2b (same config twice) | 0.910 | 18 | 0.770, 0.775 |
| E1 vs E1b (same config twice) | 0.940 | 12 | 0.870, 0.895 |
| C1 vs C1b (same config twice) | 0.905 | 19 | 0.760, 0.755 |
| E2 vs E2b (same config twice) | 0.990 | 2 | 0.815, 0.815 |
| T-on vs T-off (Ton vs C2) | 0.835 | 33 | 0.840, 0.770 |
| Ton vs C2b | 0.835 | 33 | 0.840, 0.775 |
| Ton vs E1 | 0.890 | 22 | 0.840, 0.870 |
| C4 vs E2 (haiku vs sonnet) | 0.825 | 35 | 0.770, 0.815 |
| E2 vs E1 | 0.860 | 28 | 0.815, 0.870 |
| C1 vs E1 | 0.815 | 37 | 0.760, 0.870 |
| C2 vs E1 | 0.805 | 39 | 0.770, 0.870 |
| C4 vs E1 | 0.800 | 40 | 0.770, 0.870 |
| C3 vs E1 | 0.770 | 46 | 0.690, 0.870 |

**Reasoning effect on the same model (Ton = gpt-5.4-mini high; C2 and C2b = two independent runs with reasoning off).** Ton changes 33/200 labels (16.5%) against either off run; two off runs differ on 18/200 (9.0%) by themselves, so about 7.5 points of the flips are beyond sampling noise (rough; noise flips are not strictly additive). Accuracy: Ton right only 18 vs C2 right only 4 (exact sign test p=0.0043; +7.0pt); Ton right only 18 vs C2b right only 5 (p=0.0106; +6.5pt); +6.75pt against the mean of the two off runs. Cost 3.0x ($1.079 vs $0.366 per 1k), latency 3.1x in the sequential run.

When a cheap config disagrees with E1: E1 is right and cheap wrong in 24-37 of the 37-46 cases (C1: 25 of 37; cheap right only: 3; both wrong: 9). Cheap configs are wrong on 21-25 of E1's 26 errors. Most frequent swaps are adjacent labels (`exchange_via_app` vs `exchange_rate`, `getting_virtual_card` vs `get_disposable_virtual_card`, `change_pin` vs `passcode_forgotten`).

**Evidence stage (`run.py evidence`, 238 calls, $0.174, raw in `evidence.jsonl`).**
- *Temperature is not honored for the GPT-5.4 family through OpenRouter.* The OpenRouter model list (fetched during the run) has `temperature` in `supported_parameters` for sonnet-4.6, haiku-4.5 and qwen3-32b and not for openai/gpt-5.4, -mini or -nano. Six identical `temperature=0` gpt-5.4-mini (effort none) calls per input: input 17 returned `wrong_exchange_rate_for_cash_withdrawal` x5 and `exchange_rate` x1; input 18 returned `card_swallowed` x3, `cash_withdrawal_charge` x1, `cash_withdrawal_not_recognised` x2. The same call with `provider.require_parameters=true` fails with 404 "No endpoints found that can handle the requested parameters". Claude (temperature supported) repeats at 99% (E2 vs E2b). This shows the parameter is dropped and outputs vary; it does not separate provider nondeterminism from sampling.
- *Qwen3-32b reasoning switches.* Three calls each: `effort:none`, `enabled:false`, `exclude:true` and `chat_template_kwargs.enable_thinking=false` all gave 65-77 reasoning tokens and finish `length` at the 60-token cap; `enabled:false` plus ` /no_think` gave 1 reasoning token, finish `stop`.
- *Interleaved latency* (inputs 3-32, every config called on the same input at about the same time, 8 threads): mean / median / p90 ms: C1 846 / 822 / 976; C2 849 / 824 / 981; C4 895 / 835 / 1041; E2 956 / 904 / 1173; E1 1797 / 1756 / 2407; Ton 3080 / 2011 / 4787; C3 1128 / 732 / 1767. Mean ratios vs C1: E1 2.1x, Ton 3.6x (Ton vs C2: 3.6x mean, 2.4x median), C3 1.3x. The sequential run's ratios (E1 2.7x, Ton 3.1x, and qwen as the fastest at 573 ms) were confounded by provider load at different times; qwen's interleaved mean is 2x its sequential mean with a long tail.

### Part B: router on live data

Best cheap config by match vs E1 (ties to the cheaper): C1 (nano), disagreement 0.185, $0.098 vs $1.719 per 1000 calls (17.5x cheaper, about 2.1x faster interleaved). Label = (cheap != E1), 5-fold LR on bge-small embeddings (seed 0):

| cheap cfg | disagreements vs E1 | AUC predicting disagreement |
|---|---|---|
| C1 | 37 / 200 | 0.598 |
| C2 | 39 / 200 | 0.564 |
| C3 | 46 / 200 | 0.577 |
| C4 | 40 / 200 | 0.598 |

Certificate for C1 at a 2% disagreement budget: **0% of calls routable, nothing certifiable.** Calibration/holdout are split first (100/100, seeded); calibration scores are out-of-fold inside the calibration half and holdout scores come from a model fit on the calibration half only. With delta = 0.05 split over the search steps (`certify`), a zero-error prefix needs 252 / 99 / 49 calibration rows at budgets 2% / 5% / 10% for the 100 rows used (m=8); even all 200 rows would need 258 / 102 / 50 (m=9). Budgets 2%, 5%, 10%, 20% all returned nothing certifiable. Without a guarantee, on the seeded split no cutoff meets 2%, 5%, 10% or 15% on the calibration half (v1's "15% budget: 45% routed, 3/45 = 6.7%" came from scoring before splitting and does not reproduce; 3/45 has a 95% interval of 1.4-18.3%). Over 20 random splits with the same clean protocol, empirical cutoff on the calibration half, realised on the holdout half:

| calibration budget | splits with no cutoff | mean share routed (sd) | pooled holdout disagreement (n routed) |
|---|---|---|---|
| 5% | 9 / 20 | 2.9% (1.8) | 25.0% (32) |
| 10% | 5 / 20 | 24.2% (19.5) | 14.9% (363) |
| 15% | 2 / 20 | 56.3% (21.8) | 14.4% (1013) |
| 20% | 0 / 20 | 98.7% (2.1) | 17.7% (1973) |

Always-cheap disagrees 18.5%, which is also what a random routing would give at any share. Realised disagreement at the 10% and 15% budgets (14.9%, 14.4%) is above the 10% budget and about 4 points below random, i.e. a weak lift at best. Descriptive curve on all 200 (OOF confidence, no guarantee), share routed -> disagreement vs E1: 10% -> 0.25; 20% -> 0.15; 30% -> 0.15; 40% -> 0.113; 50% -> 0.12; 60% -> 0.133; 80% -> 0.169; 100% -> 0.185. No subset got near 2%.

## Interpretation for the mission

- **Offline, a prompt-level quality router did not beat random mixing for savings at <=5pt accuracy loss; AUC 0.63-0.69 does not convert to savings.** Against a coin-flip mix between GPT-4 and the cheap model, the embedding router (P(cheap >= expensive)) is not distinguishable from random mixing or worse at every tolerance (bootstrap intervals include zero or lie below it), significantly worse at 2pt and 5pt for gpt-3.5 (-2.6 and -7.5 points) and at 5pt for mixtral (-2.3); only TF-IDF/gpt-3.5 at 5pt is clearly better (+5.4). A clean split gives the same picture. The cause is the ranking rule: it ignores per-prompt dollars, so it routes the prompts that are cheap to serve. No router passed a 1%, 2% or 5% certificate. v1's "8-11% at <=2pt and 18-34% at <=5pt" is the router's number, but a random mix gets 8-12% and 20-30%, so it is not a router effect.
- **Large savings do exist offline, and they come from cost concentration, not from per-prompt quality prediction.** gsm8k is 20% of rows, 53% of GPT-4's dollars and a 5.4pt gap to gpt-3.5. Ranking by predicted dollars saved, or by benchmark loss-per-dollar with no ML at all, saves 60-62% (gpt-3.5) and 34-43% (mixtral) within 2pt in-sample, and 59-63% at a realised loss of 1.6-2.2pt (gpt-3.5) / 39-51% at 2.0-2.8pt (mixtral) on clean splits. The learned accuracy-loss regressor adds little to that (-2 to +5 points over quality-blind saving-only, not separated from noise). For the mission this supports measuring per call site (which step is expensive, what is the cheap model's gap on it) and routing whole call sites or long-output steps, and it does not support a per-input classifier on the prompt text.
- **Those cost-aware numbers need a margin.** Choosing the largest saving within a tolerance on a calibration half leaves the realised loss above the tolerance in a large share of splits (share within tolerance 0.0-0.55 for cost-aware and saving-only at 2pt). A deployment needs a conservative bound (as `certify` does) or an explicit safety margin, and a bound at 2% needs about 250 zero-error rows. RouterBench's 0-shot grader noise, 2023 prices, and tie rule (both wrong = cheap fine) also set the ceiling on how literally these percentages transfer.
- **On a typed 77-label step, cheap configurations do not clear a 2% budget.** Nano, mini, haiku and qwen3-32b match gpt-5.4(low) on 77-82% of inputs and lose 10-18 points of gold accuracy (0.69-0.77 vs 0.87-0.895). Reasoning off is not free here: turning on reasoning for the same mini model changes 16.5% of answers (about 7.5 points beyond sampling noise) and recovers about +7 points gold accuracy (18 vs 4 and 18 vs 5 right-only, p<0.02) at 3.0x the cost per call and 2.4-3.6x the latency. The worked example in the mission (thinking off changes 3 of 200) did not hold on this step.
- **Cost and latency spreads are real, so the lever exists when match is high.** E1 costs $1.72/1k calls at about 1.8-2.3 s; nano costs $0.098 (17.5x less) at about 0.8 s; sonnet-4.6 with no reasoning costs $1.83 and haiku-4.5 $0.61. Cost is dominated by the 439-token label list in the prompt for the reasoning-off configs (8 output tokens) and by reasoning tokens when on.
- **The teacher is noisy, which sets the ceiling on any certificate.** Identical repeated calls disagree 6% (E1), 9% (C1, C2) of the time on the OpenAI models because temperature is dropped. A 2% budget against a single E1 sample is below the teacher's own self-disagreement. A usable certificate has to be against the teacher's distribution (or a majority of repeated teacher calls), and these OpenAI calls cannot be replayed through OpenRouter. Claude Sonnet 4.6 at temperature 0 repeats at 99%.
- **Live router: no signal at n=200.** AUC 0.56-0.60 with 37-46 positives. With 100 calibration rows a 2% (or 5%, or 10%) certificate is impossible regardless of the router (needs 252 / 99 / 49 zero-error rows). On clean splits an empirical 10-15% disagreement budget realises 14-15% disagreement while routing 24-56%, against 18.5% for random routing. A real call site would need on the order of 1,000+ shadow rows and a much better-than-chance router before a 2% budget is reachable.
- **Net:** feasible as measurement (match-rate table per call site, cost per config, cost concentration) and as a coarse per-call-site decision; not demonstrated as a per-input router at a 2% budget on either dataset, and on RouterBench not demonstrated as better than random mixing when the ranking ignores cost.

## Caveats

- Banking77 is public and models may have seen it; gold labels exist here, but real call sites have none. 77 overlapping labels is a stress case: even E1 is at 87-89.5% gold, and swaps are between near-synonym labels. A step with few, well-separated outputs would likely show a much higher match rate. Do not read 80% as typical.
- n=200 per config: intervals of about +/-5.5pt on match rates; the C2 vs C2b and E1 vs E1b repeats show the sampling noise is of the same size as many of the differences between cheap configs. Single run, no repeated seeds for the sampled models.
- The GPT-5.4 results are at the default sampling temperature (OpenRouter drops `temperature`). A direct OpenAI key may allow temperature 0 with reasoning "none"; not tested.
- Qwen3-32b "reasoning off" needs the `/no_think` soft switch in the user message; the other reasoning-off settings were verified from usage (0 reasoning tokens). Anthropic and OpenAI reasoning settings were not varied beyond on/off (E1 "low", T-on "high", one run each).
- Latency is wall time per call through OpenRouter with 8 concurrent calls (network and provider queueing included). The sequential runs happened at different times per config; the interleaved check (30 inputs x 7 configs) is the better cross-config comparison and shows ratios that differ from the sequential ones by up to 0.6x. Cost is tokens x list prices; router inference and the audit-sample overhead are not included in the blended number.
- RouterBench: 2023 models and prices; 0-shot subset only (36,497 of ~405k rows); scores are per-benchmark graders (some GPT-4-judged), "ok" is a >= 0.5 threshold; fractional scores appear in gsm8k and "other" (0.25 steps) where the dataset card says 0/1, source unverified; `y1` counts a tie as "cheap fine"; "agree" is correctness agreement, not identical answer strings (the raw file with strings was not downloaded). The oracle can beat the expensive model, so its saving is not strictly "same accuracy". The llama-2-70b and mistral-7b rows contain grader/parse artifacts (above). The embedding router used a 5,000-row subsample at 400 characters; TF-IDF on all rows gives similar or slightly better AUC and the same conclusions, so the subsample is not what limits it. RouterBench has no thinking on/off pairs.
- The cost-aware policies use the dataset's realised per-row dollars (as targets for the ridge regressors, and for the hindsight oracle). A deployed router predicts them from the prompt (OOF R2 0.73-0.78 here); output length of the expensive model is the part that is not known before the call. The gsm8k dollar share depends on chain-of-thought length in this dataset.
- Leakage: the Part A router CV is by row, not by benchmark prompt template, so near-duplicate prompts across folds can inflate AUC and the learned policies (optimistic, not pessimistic). The same holds for the live router (random folds, queries from the same intent classes). The Part A bootstrap resamples rows with fixed OOF scores, so it measures evaluation noise, not training noise. The clean-split tables fit the scorer on the calibration half only; the "same-data" tables do not (their cutoff is picked on the data being scored).
- Real deployment changes: the teacher would be the production model at its production sampling settings, the call-site would have a narrower output set, shadow volume would be thousands of rows, and the expensive configuration should be sampled more than once per input to estimate the teacher's own disagreement before setting a budget.
- `common.py`: no bug found. `certify` is correct but cannot return a certificate below about 250 rows at budget 2% and delta 0.05 (by design); worth surfacing to users as a minimum-rows warning.
- Code state: `run.py` is the final version. The first live batch ran before the E1b/C1b/E2b repeats were added and before the "who is right" block was added to `analyze` (re-run on the saved responses). The 3 initial qwen3-32b probe rows (reasoning leaked, all length-truncated) were removed from `responses.jsonl` and replaced by the `/no_think` runs. `rb_load.py` now allow-lists exact globals; the pickle itself comes from a public, ungated HF repo (sizes match).

## Time and cost

- Wall: Part A revised run about 6 min (embeddings were cached from v1; router blocks 27-28 s per bge block and 138-155 s per TF-IDF block, including 20 or 10 clean-split repeats; `partA_log.txt` is from a second full run of `a` that reproduced every number in this file, the first run's blocks took 29 s and 152-158 s); the v1 Part A run took about 25 min (embedding 5,000 prompts 1383 s under heavy CPU contention). Part B analysis seconds on the saved responses; `evidence` stage 72 s.
- OpenRouter spend: $1.954 in `responses.jsonl` (2,200 calls, all configs; by config: E2/E2b $0.366 each, E1 $0.344, E1b $0.347, Ton $0.216, C4 $0.122, C2/C2b $0.073 each, C1/C1b $0.020 each, C3 $0.007) plus about $0.004 of unlogged qwen and temperature tests in v1 (superseded by `evidence.jsonl`) plus $0.1736 for the `evidence` stage (238 calls) in this revision. Total about $2.13 of the $7 cap. `smokes/COSTS.md` has the v1 line ($1.96) and a new `s5_revision` line ($0.1736). Part A cost $0.
