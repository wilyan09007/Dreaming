# S5: Tier / thinking router (Mission 1)

Files: `run.py` (all stages: `a`, `probe`, `live`, `analyze`), `rb_load.py` (restricted unpickler for the RouterBench pickle), `responses.jsonl` (2200 raw OpenRouter responses with tokens, latency, cost), `log.txt` (stdout of all stages), `rb_data/routerbench_0shot.pkl` (downloaded, 99.6 MB).

## Setup

**Part A (offline, free).** `withmartian/routerbench` is public, no login. `routerbench_0shot.pkl` (99.6 MB; the 5-shot is 171 MB, the raw file 1.2 GB, not used) loaded with a restricted unpickler (only pandas/numpy reconstruct callables allowed). 36,497 prompts x 11 models (2023 vintage), 8 benchmark families (mmlu 14,042; hellaswag 10,042; gsm8k 7,450; other 1,719; arc 1,470; winogrande 1,267; mbpp 427; mtbench 80). Columns used: `<model>` (score in [0,1]; binary for arc/hellaswag/mbpp/mmlu/winogrande, 0.25 steps for gsm8k and "other", 0.1 steps for mtbench), `<model>|total_cost` (USD per sample at 2023 prices). Definitions: "ok" = score >= 0.5; "agree" = both ok or both not ok; label for the router `y = (score_cheap >= score_expensive)` ("cheap is at least as good"); oracle routes to cheap iff `y=1`. Pairs: gpt-4-1106-preview (expensive) vs gpt-3.5-turbo-1106, mixtral-8x7b-chat, llama-2-70b-chat, mistral-7b-chat; plus claude-v2 vs claude-instant-v1. Router: LogisticRegression(C=1) on `embed(prompt[:400 chars])` (bge-small, via `common.embed`), 5-fold stratified out-of-fold (seed 0) on a 5,000-row subsample stratified by family (seed 0), because CPU embedding on this shared machine ran at about 3.6 texts/s (1383 s). A first attempt at all 36.5k prompts x 1200 chars was killed after ~17 min. Check on full data: TF-IDF (1-2 grams, 30k features) + LR on all 36,497 rows. Baseline: per-family base rate. Certificate: `certify()` on a random half of the OOF scores, applied to the other half (seed 1).

**Part B (live, OpenRouter).** 200 Banking77 test inputs from `stratified_subsample(test_labels, 200, seed=0)` (all 77 classes present). System prompt exactly as specified (77 labels one per line), user = query, `temperature=0`, `max_tokens` 60 (reasoning off) / 2000 (E1) / 4000 (T-on), 8 threads. Parse: exact match after strip/lower (also stripping quotes/backticks/trailing period); else nearest label by `difflib` similarity and counted as a parse failure. Spend tracked from `usage` tokens x the given prices (equals OpenRouter's reported `usage.cost` to the digit).

| cfg | model | reasoning setting | note |
|---|---|---|---|
| E1 | openai/gpt-5.4 | effort "low" | expensive reference (default effort is not "none"; "low" as the task allowed) |
| C1 | openai/gpt-5.4-nano | effort "none" | verified: 0 reasoning tokens (gpt-5.4 family has no "minimal", "none" is the off switch) |
| C2 | openai/gpt-5.4-mini | effort "none" | 0 reasoning tokens; also serves as T-off |
| C3 | qwen/qwen3-32b | `enabled:false` + " /no_think" appended to the user message | the OpenRouter reasoning param is ignored by the provider (DeepInfra): `effort:none`, `enabled:false`, `exclude`, `chat_template_kwargs` all still produced 65-77 reasoning tokens and hit the length cap. Only the Qwen3 soft switch worked (1 reasoning token). |
| Ton | openai/gpt-5.4-mini | effort "high" | T-on |
| E2 | anthropic/claude-sonnet-4.6 | `enabled:false` | 0 reasoning tokens |
| C4 | anthropic/claude-haiku-4.5 | `enabled:false` | 0 reasoning tokens |
| C2b, E1b, C1b, E2b | repeats of C2, E1, C1, E2 | same | added after finding the noise problem below (not in the original spec; $0.75) |

## Results

### Part A: RouterBench

Per-model (all 36,497 prompts): gpt-4 acc 0.781, $3.29/1k calls; gpt-3.5 0.619, $0.24; mixtral 0.547, $0.14; claude-v2 0.636, $2.42; claude-instant 0.598, $0.23; llama-2-70b 0.329, $0.20; mistral-7b 0.306, $0.05.

Agreement and oracle, cheap vs expensive (all benchmarks):

| expensive | cheap | acc exp | acc cheap | agree (both ok / both not ok) | P(cheap ok given exp ok) | oracle acc | oracle cost saving | cheap/exp cost |
|---|---|---|---|---|---|---|---|---|
| gpt-4 | gpt-3.5 | 0.781 | 0.619 | 0.730 | 0.734 | 0.833 | 67.3% | 0.074 |
| gpt-4 | mixtral-8x7b | 0.781 | 0.547 | 0.654 | 0.632 | 0.821 | 58.1% | 0.041 |
| gpt-4 | llama-2-70b | 0.781 | 0.329 | 0.471 | 0.394 | 0.804 | 49.3% | 0.062 |
| gpt-4 | mistral-7b | 0.781 | 0.306 | 0.396 | 0.319 | 0.814 | 39.1% | 0.014 |
| claude-v2 | claude-instant | 0.636 | 0.598 | 0.779 | 0.802 | 0.723 | 74.2% | 0.096 |

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

Router, gpt-4 vs gpt-3.5, bge-small + LR, 5,000-row subsample, out-of-fold (base rate `y=1` is 0.752; AUC 0.631 for the embedding router, 0.559 for the family-prior baseline; TF-IDF on all 36.5k rows: AUC 0.644 vs family prior 0.575):

| policy | share to cheap | accuracy | $ per 1k calls |
|---|---|---|---|
| always-expensive | 0% | 0.7865 | 3.28 |
| always-cheap | 100% | 0.6320 | 0.24 |
| oracle | 75% | 0.8392 | 1.09 |
| router top 20% | 20% | 0.7667 | 2.98 |
| router top 40% | 40% | 0.7438 | 2.63 |
| router top 60% (ties) | 64% | 0.7119 | 1.67 |
| router top 80% | 80% | 0.6962 | 0.63 |
| random 30% / 60% | 30% / 60% | 0.7424 / 0.7000 | 2.39 / 1.43 |
| family-prior top 30% / 60% | 33% / 62% | 0.7396 / 0.7048 | 2.66 / 1.43 |

Gpt-4 vs mixtral: base rate 0.671, AUC 0.668 (embedding), 0.625 (family prior), 0.688 (TF-IDF, all rows). Full curve in `log.txt`.

Maximum cost saving vs always-expensive at a given accuracy loss (out-of-fold router, same pair, sweep over the share routed):

| pair / features | loss <= 0 | <= 0.5pt | <= 1pt | <= 2pt | <= 5pt |
|---|---|---|---|---|---|
| gpt-3.5, bge-small LR (n=5000) | 0% | 3.1% | 5.4% | 9.4% | 22.0% |
| gpt-3.5, TF-IDF LR (n=36.5k) | 0% | 3.7% | 5.8% | 11.2% | 33.7% |
| gpt-3.5, family prior | 0% | 4.0% | 9.6-9.9% | 9.6-9.9% | 18.7-19.0% |
| mixtral, bge-small LR | 0% | 2.0% | 3.6% | 8.3% | 17.6% |
| mixtral, TF-IDF LR | 0% | 2.6% | 4.3% | 8.4% | 19.8% |

Certificates (`certify`, "cheap at least as good", calibration half, holdout half): nothing certifiable at budgets 1%, 2% and 5% for either pair and either feature set. Within-family AUC of the embedding router is 0.45-0.65 (e.g. gpt-3.5: mmlu 0.637, hellaswag 0.635, other 0.609, gsm8k 0.461), so most of the router's signal is "which benchmark is this" plus weak difficulty.

### Part B: live Banking77 (200 inputs, 77 labels)

| cfg | model | n | gold acc | parse fails | match vs E1 | mean ms | p50 ms | prompt tok | compl tok | reasoning tok | $/1000 calls |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | gpt-5.4, low | 200 | 0.870 | 0 | (1.000) | 2311 | 2229 | 439 | 41.4 | 31.4 | 1.719 |
| E1b | gpt-5.4, low (repeat) | 200 | 0.895 | 0 | 0.940 | 2186 | 2118 | 439 | 42.6 | 32.6 | 1.737 |
| C1 | gpt-5.4-nano, none | 200 | 0.760 | 0 | 0.815 | 843 | 834 | 439 | 8.0 | 0 | 0.098 |
| C1b | gpt-5.4-nano (repeat) | 200 | 0.755 | 0 | 0.805 | 798 | 787 | 439 | 8.0 | 0 | 0.098 |
| C2 (= T-off) | gpt-5.4-mini, none | 200 | 0.770 | 1 | 0.805 | 866 | 822 | 439 | 8.0 | 0 | 0.366 |
| C2b | gpt-5.4-mini (repeat) | 200 | 0.775 | 1 | 0.805 | 830 | 810 | 439 | 8.0 | 0 | 0.366 |
| C3 | qwen3-32b, /no_think | 200 | 0.690 | 1 | 0.770 | 572 | 454 | 434 | 7.8 | 1.0 | 0.037 |
| Ton | gpt-5.4-mini, high | 200 | 0.840 | 0 | 0.890 | 2665 | 2125 | 439 | 166.5 | 156.5 | 1.079 |
| E2 | claude-sonnet-4.6, off | 200 | 0.815 | 0 | 0.860 | 945 | 872 | 568 | 8.5 | 0 | 1.831 |
| E2b | claude-sonnet-4.6 (repeat) | 200 | 0.815 | 0 | 0.855 | 902 | 876 | 568 | 8.5 | 0 | 1.831 |
| C4 | claude-haiku-4.5, off | 200 | 0.770 | 2 | 0.800 | 806 | 769 | 567 | 8.7 | 0 | 0.611 |

Ton reasoning tokens: mean 156, max 1577; all 200 finished with `stop`. Parse failures (5 total, none empty): four near-miss label strings (`getting_physical_card`, `get_virtual_card`) resolved by nearest-label fallback, and one haiku reply (input 197) that was a prose refusal ("I'm specifically designed to classify bank support intents...") which the fallback mapped to an arbitrary label.

Pairwise same-label rate (n=200 each; a 95% interval at 0.8 is about +/-5.5pt):

| pair | match | disagreements | gold acc (first, second) |
|---|---|---|---|
| C2 vs C2b (same config twice) | 0.910 | 18 | 0.770, 0.775 |
| E1 vs E1b (same config twice) | 0.940 | 12 | 0.870, 0.895 |
| C1 vs C1b (same config twice) | 0.905 | 19 | 0.760, 0.755 |
| E2 vs E2b (same config twice) | 0.990 | 2 | 0.815, 0.815 |
| T-on vs T-off (Ton vs C2) | 0.835 | 33 | 0.840, 0.770 |
| Ton vs E1 | 0.890 | 22 | 0.840, 0.870 |
| C4 vs E2 (haiku vs sonnet) | 0.825 | 35 | 0.770, 0.815 |
| E2 vs E1 | 0.860 | 28 | 0.815, 0.870 |
| C1 vs E1 | 0.815 | 37 | 0.760, 0.870 |
| C2 vs E1 | 0.805 | 39 | 0.770, 0.870 |
| C4 vs E1 | 0.800 | 40 | 0.770, 0.870 |
| C3 vs E1 | 0.770 | 46 | 0.690, 0.870 |

When a cheap config disagrees with E1: E1 is right and cheap wrong in 24-37 of the 37-46 cases (C1: 25 of 37; cheap right only: 3; both wrong: 9). Cheap configs are wrong on 21-25 of E1's 26 errors. Most frequent swaps are adjacent labels (`exchange_via_app` vs `exchange_rate`, `getting_virtual_card` vs `get_disposable_virtual_card`, `change_pin` vs `passcode_forgotten`).

**Temperature is not honored for the GPT-5.4 family through OpenRouter.** `temperature=0` was sent on every call. A direct test: repeated identical gpt-5.4-mini calls on inputs 17 and 18 returned different labels (4 calls: `wrong_exchange_rate_for_cash_withdrawal`, `exchange_rate`, `exchange_rate`, `wrong_exchange_rate_for_cash_withdrawal`), and the same call with `provider.require_parameters=true` fails with 404 "No endpoints found that can handle the requested parameters". The model listing also lacks `temperature` in `supported_parameters` for openai/gpt-5.4*. Claude (temperature supported) repeats at 99%.

### Part B: router on live data

Best cheap config by match vs E1 (ties to the cheaper): C1 (nano), disagreement 0.185, $0.098 vs $1.719 per 1000 calls (17.5x cheaper, 2.7x faster). Label = (cheap != E1), 5-fold LR on bge-small embeddings (seed 0):

| cheap cfg | disagreements vs E1 | AUC predicting disagreement |
|---|---|---|
| C1 | 37 / 200 | 0.598 |
| C2 | 39 / 200 | 0.564 |
| C3 | 46 / 200 | 0.577 |
| C4 | 40 / 200 | 0.598 |

Certificate at a 2% disagreement budget for C1: **0% of calls routable, nothing certifiable.** With delta = 0.05 split over the search steps (`certify`), the minimum number of calibration rows with zero errors is 258 for a 2% budget, 102 for 5%, 50 for 10%; we have 100 calibration rows. Budgets 2%, 5%, 10%, 20% all returned nothing certifiable. Without a guarantee (empirical cutoff on the calibration half, applied to the holdout half): budget 2%, 5%, 10% have no cutoff at all; budget 15% gives cutoff 0.831, 45% of holdout calls routed, 6.7% disagreement (n=45), blended cost $0.99 per 1000 calls (E1 only: $1.72; all-cheap: $0.098 at 18.5% disagreement). Descriptive curve on all 200 (OOF confidence, no guarantee), share routed -> disagreement vs E1: 10% -> 0.25; 20% -> 0.15; 30% -> 0.15; 40% -> 0.113; 50% -> 0.12; 60% -> 0.133; 80% -> 0.169; 100% -> 0.185. No subset got near 2%.

## Interpretation for the mission

- **Offline there is large headroom but a prompt-only router captures little.** In RouterBench the cheap model is at least as good as GPT-4 on 75% of prompts (gpt-3.5) and an oracle saves 67% of cost at slightly higher accuracy. A router trained on prompt embeddings (AUC 0.63-0.69) saves only 8-11% at <=2pt accuracy loss and 18-34% at <=5pt, and about half of that comes from knowing the benchmark. No router passed a 1%, 2% or 5% certificate. For mission 1 as written ("route when the match rate clears the error budget"), the usable result here is per-call-site routing at low budgets only if the whole call site matches; per-input routing from the prompt alone is weak.
- **On a typed 77-label step, cheap configurations do not clear a 2% budget.** Nano, mini, haiku and qwen3-32b match gpt-5.4(low) on 77-82% of inputs and lose 10-18 points of gold accuracy (0.69-0.77 vs 0.87-0.895). Reasoning off is not free here: turning on reasoning for the same mini model (Ton) changes 16.5% of answers vs the off setting and recovers +6.7 pts gold accuracy (0.84 vs 0.77) at 3.0x the cost per call ($1.08 vs $0.37 per 1k) and 3.1x the latency. The worked example in the mission (thinking off changes 3 of 200) did not hold on this step; the expensive configuration contributes about 10 points of accuracy.
- **Cost and latency spreads are real, so the lever exists when match is high.** E1 costs $1.72/1k calls at 2.3 s; nano costs $0.098 (17.5x less) at 0.84 s; sonnet-4.6 with no reasoning costs $1.83 and haiku-4.5 $0.61. Cost is dominated by the 439-token label list in the prompt for the reasoning-off configs (8 output tokens), and by reasoning tokens when on.
- **The teacher is noisy, which sets the ceiling on any certificate.** Identical repeated calls disagree 6% (E1), 9% (C1, C2) of the time on the OpenAI models because temperature is ignored. So the best achievable match of a cheap config to a single E1 call is about 0.91-0.94, and a 2% budget against a single E1 sample is below the teacher's own self-disagreement. A usable certificate has to be against the teacher's distribution (or a majority of repeated teacher calls), and these OpenAI calls cannot be replayed through OpenRouter. Claude Sonnet 4.6 at temperature 0 repeats at 99%.
- **Live router: no signal at n=200.** AUC 0.56-0.60 with 37-46 positives, and with 100 calibration rows a 2% (or 5%, or 10%) certificate is impossible regardless of the router (needs 258 / 102 / 50 zero-error rows). Even the best score (top 40% by confidence) still shows 11% disagreement. A real call site would need on the order of 1,000+ shadow rows and a much better-than-chance router before a 2% budget is reachable.
- **Net:** feasible as measurement (match-rate table per call site, cost per config) and as a coarse per-call-site decision; not demonstrated as a per-input router at a 2% budget on either dataset.

## Caveats

- Banking77 is public and models may have seen it; gold labels exist here, but real call sites have none. 77 overlapping labels is a stress case: even E1 is at 87-89.5% gold, and swaps are between near-synonym labels. A step with few, well-separated outputs would likely show a much higher match rate. Do not read 80% as typical.
- n=200 per config: intervals of about +/-5.5pt on match rates; the C2 vs C2b and E1 vs E1b repeats show the sampling noise is of the same size as many of the differences between cheap configs. Single run, no repeated seeds for the sampled models.
- The GPT-5.4 results are at the default sampling temperature (OpenRouter drops `temperature`). A direct OpenAI key may allow temperature 0 with reasoning "none"; not tested.
- Qwen3-32b "reasoning off" needs the `/no_think` soft switch in the user message; the other reasoning-off settings were verified from usage (0 reasoning tokens). Anthropic and OpenAI reasoning settings were not varied beyond on/off (E1 "low", T-on "high", one run each).
- Latency is wall time per call with 8 concurrent calls through OpenRouter (includes network and provider queueing, and the first 3 probe calls per config which were cold). Cost is tokens x list prices; router inference and the audit-sample overhead are not included in the blended number.
- RouterBench: 2023 models and prices; 0-shot subset only (36,497 of ~405k rows); scores are per-benchmark graders (some GPT-4-judged), "ok" is a >= 0.5 threshold; fractional scores appear in gsm8k and "other" (0.25 steps) where the dataset card says 0/1, source unverified; `y` counts a tie as "cheap fine"; "agree" is correctness agreement, not identical answer strings (the raw file with strings was not downloaded). The oracle can beat the expensive model, so its saving is not strictly "same accuracy". The embedding router used a 5,000-row subsample at 400 characters because CPU embedding was the bottleneck; TF-IDF on all rows gives similar or slightly better AUC, so the subsample is not what limits it. RouterBench has no thinking on/off pairs.
- Leakage: the router CV is by row, not by benchmark prompt template, so near-duplicate prompts across folds can inflate AUC (the numbers are optimistic, not pessimistic). The same holds for the live router (random folds, queries from the same intent classes).
- Real deployment changes: the teacher would be the production model at its production sampling settings, the call-site would have a narrower output set, shadow volume would be thousands of rows, and the expensive configuration should be sampled more than once per input to estimate the teacher's own disagreement before setting a budget.
- `common.py`: no bug found. `certify` is correct but cannot return a certificate below about 258 rows at budget 2% and delta 0.05 (by design); worth surfacing to users as a minimum-rows warning.
- Code state: `run.py` is the final version; the first live batch ran before the E1b/C1b/E2b repeats were added and before the "who is right" block was added to `analyze` (re-run on the saved responses). The 3 initial qwen3-32b probe rows (reasoning leaked, all length-truncated) were removed from `responses.jsonl` and replaced by the `/no_think` runs.

## Time and cost

- Wall: Part A final run about 25 min (embedding 5,000 prompts 1383 s under heavy CPU contention from other smokes; pickle load, tables, TF-IDF and CV about 3 min); a first full-set embedding attempt was abandoned after about 17 min. Part B: probe about 1 min, 1,579 live calls about 5 min, 600 repeat calls about 2 min, analysis seconds.
- OpenRouter spend: $1.954 in `responses.jsonl` (2,200 calls, all configs; by config: E2/E2b $0.366 each, E1 $0.344, E1b $0.348, Ton $0.216, C4 $0.122, C2/C2b $0.073 each, C1/C1b $0.020 each, C3 $0.007) plus about $0.004 of unlogged qwen and temperature tests; total about $1.96 of the $7 cap. Appended to `smokes/COSTS.md`. Part A cost $0.
