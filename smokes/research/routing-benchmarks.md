# Routing benchmarks (tier / thinking router): research notes

Researched 2026-10-17. All prices are USD per 1M tokens (input/output) from the live, unauthenticated
`https://openrouter.ai/api/v1/models` listing fetched 2026-10-17 (470 models, 788,576 bytes) unless noted.
Everything marked "unverified" was not confirmed from a primary source.

## Summary

- RouterBench is downloadable with no login, contains per-sample responses + correctness + cost for 11 models, so cheap-vs-expensive agreement can be computed offline for free. But the models are 2023 vintage (GPT-4, GPT-3.5, Claude v1/v2/instant, Llama-70B...) and there are no thinking-on/off variants, so it only proves the pipeline, not modern tier pairs.
- LLMRouterBench (Jan 2026, ACL Findings 2026) is the better offline substrate: 33 models, 21 datasets, 400K+ per-query records (prediction, score, tokens, cost), includes GPT-5-chat vs GPT-5-medium, Qwen3-235B instruct vs thinking-2507, Gemini-2.5-flash vs pro, Claude-sonnet-4. 1.28 GB tarball on HF, no gate. Tasks are reasoning/QA/code, not bookkeeping-style typed steps. License ambiguous.
- RouteLLM (Apache-2.0) predicts "strong model wins a preference comparison" from chat prompts (Arena + GPT-4-judge labels). That is not the same target as "cheap config gives the same answer". Reuse the router classes (mf/bert) and threshold-calibration idea, not its labels. Repo last pushed 2024-08-10 (stale).
- DART (arXiv 2606.23181, EMNLP 2026 Findings, MIT, v4 2026-10-03) is almost exactly our certification idea in label-free form: sample 2 cheap no-think drafts, accept if they agree, else think with a budget from draft entropy. 32-73% fewer thinking tokens, matches/exceeds always-think accuracy on 13 of 14 model-benchmark pairs. Evaluated on math/code only; core is stdlib-only Python.
- No paper found that reports a per-question "thinking on vs off changes the answer" rate. Closest: OverthinkingBench (1,440 simple questions): Qwen3-235B 96.6% non-think vs 96.9% think; Qwen3-8B 95.4 vs 96.3; aggregate only, no paired flips. We must measure the flip rate ourselves; it is cheap (see Numbers).
- OpenRouter `reasoning` param works as documented: `effort` (max/xhigh/high/medium/low/minimal/none) OR `max_tokens`, plus `enabled` / `exclude`. `enabled:false` == `effort:"none"`, but many models are `mandatory` reasoning and reject "none" (gpt-5/-mini/-nano, gemini-2.5-pro, gemini 3.x-flash, claude-sonnet-5.5, claude-opus-5.5). Check `reasoning.mandatory` per model from `/api/v1/models` before designing the "thinking off" arm.
- Price spread inside a family is 5-20x (gpt-4.1 -> nano 20x; gpt-5 -> nano 25x on output), so routing has room; reasoning tokens bill as output and dominate cost when on.
- A full cheap-vs-expensive paired smoke of 1,000 prompts (800 in / 150 out) costs about $3.4 for gpt-4.1 + gpt-4.1-mini and about $3.3 for sonnet-5 + haiku-5.5 (at list price). Well inside the ~$20 OpenRouter credit.

## Findings

### (a) RouterBench (Martian)
- Paper: https://arxiv.org/abs/2403.12031 (v2 2024-03-28). Code: https://github.com/withmartian/routerbench (MIT, last push 2024-06-13). Data: https://huggingface.co/datasets/withmartian/routerbench (DOI 10.57967/hf/1996, lastModified 2024-03-27, 8,481 downloads/month at fetch).
- Contents (paper HTML https://arxiv.org/html/2403.12031v2): 405,467 samples; 11 LLMs; MMLU, MT-Bench, MBPP, HellaSwag, Winogrande, GSM8K, ARC-Challenge (+ a 800-query RAG set with 3 extra models). Models: GPT-4, GPT-3.5-turbo, Claude-instant-v1, Claude-v1, Claude-v2, Llama-70B-chat, Mixtral-8x7B-chat, Yi-34B-chat, Code Llama-34B, Mistral-7B-chat, WizardLM-13B.
- Per-sample record per paper: sample_id, model_name, eval_name, prompt, model_response, performance, cost, true_label. Scoring: exact-match 0/1 for MMLU/HellaSwag/GSM8K/ARC/Winogrande; GPT-4-judged for MBPP/MT-Bench/RAG (normalised to [0,1]). So correctness for many models per prompt IS included, and raw responses are in the raw file.
- Files (HF API https://huggingface.co/api/datasets/withmartian/routerbench?blobs=true): `routerbench_0shot.pkl` 99,567,659 B; `routerbench_5shot.pkl` 171,091,324 B; `routerbench_raw.pkl` 1,198,134,681 B (card says 1.47 GB total). Pickle only (parquet request open: https://huggingface.co/datasets/withmartian/routerbench/discussions). HF license field empty; GitHub code is MIT.
- Oracle router (paper Table 1) picks the best/cheapest model per prompt and "GPT-4 is seldom chosen": oracle perf MMLU 0.957, MT-Bench 0.996, MBPP 0.899, HellaSwag 0.994, Winogrande 1.0, GSM8K 0.748, ARC 0.977. Direct evidence that cheap models are right whenever the expensive one is, on most prompts.
- Caveats: 2023 models and 2023 price points; correctness agreement is derivable, but "same answer" (string equality) needs the raw file; pickle must be loaded in a sandbox (unpickling untrusted data executes code).

### (a2) LLMRouterBench (modern successor)
- Paper https://arxiv.org/abs/2601.07206 (2026-01-12; ACL Findings 2026 https://aclanthology.org/2026.findings-acl.1881/). Repo https://github.com/ynulihao/LLMRouterBench (last push 2026-10-15; README badge says MIT but GitHub API reports no license: unverified).
- Data: https://huggingface.co/datasets/NPULH/LLMRouterBench: `bench-release.tar.gz` 1,283,503,080 B, not gated, lastModified 2025-12-08, no HF license. Layout `results/bench/<dataset>/<split>/<model>/<timestamp>.json`; fields origin_query, prompt, prediction, raw_output, ground_truth, score, prompt_tokens, completion_tokens, cost, index.
- 33 models, 21 datasets, 400K+ instances, ~1.8B tokens. Flagship "performance-cost" set (13 models): Claude-sonnet-4, Gemini-2.5-flash, Gemini-2.5-pro, GPT-5-chat, GPT-5-medium, Qwen3-235B-a22b-2507, Qwen3-235B-a22b-thinking-2507, DeepSeek-v3-0324, v3.1-terminus, r1-0528, GLM-4.6, Kimi-k2-0905, Intern-S1. Datasets there: AIME, LiveMathBench, LiveCodeBench, SWE-Bench, GPQA, HLE, MMLU-Pro, SimpleQA, ArenaHard, tau2-Bench. 20 small (7-9B) models in the "performance" set.
- Findings: many routers perform about the same, a simple baseline is hard to beat, large gap to oracle remains, embedding backbone matters little. 10 baselines integrated (RouterDC, EmbedLLM, Model-SAT, Avengers, HybridLLM, FrugalGPT, RouteLLM, GraphRouter, Avengers-Pro, OpenRouter).
- The natural thinking-tier pairs inside it: GPT-5-chat vs GPT-5-medium; Qwen3-235B instruct-2507 vs thinking-2507; Gemini-2.5-flash vs pro.

### (b) RouteLLM (LMSYS)
- Repo https://github.com/lm-sys/RouteLLM (Apache-2.0, 5.6k stars, last push 2024-08-10). `pip install "routellm[serve,eval]"`. Paper https://arxiv.org/abs/2406.18665.
- Routers: `mf` (matrix factorization, recommended), `sw_ranking` (similarity-weighted Elo), `bert` (BERT-base classifier), `causal_llm` (Llama-3-8B classifier), `random`. Interface: implement `calculate_strong_win_rate(prompt)`; a threshold picks strong vs weak. Models go through LiteLLM; any OpenAI-compatible endpoint works. `mf` and `sw_ranking` call OpenAI embeddings so need `OPENAI_API_KEY` unless patched.
- Training data: Chatbot Arena 80K battles (5K val; ~65K after pruning prompts <16 chars; 64 models mapped to 10 tiers); GPT-4-judge labels on ~120K Nectar prompts (about $700 of GPT-4 calls); ~1.5K MMLU gold-labelled. Eval: MMLU, MT-Bench, GSM8K. Strong/weak = gpt-4-1106-preview / Mixtral-8x7B; transfer to Claude-3 Opus/Sonnet and Llama-3.1 70B/8B without retraining.
- Results: README "up to 85% cost reduction at 95% GPT-4 quality" (MT-Bench). Paper Table 6 cost-saving at CPT(50%): MT-Bench 3.66x, MMLU 1.41x, GSM8K 1.49x. Judge-data augmentation raised MF APGR on MT-Bench 0.580 -> 0.802.
- Hugging Face assets under https://huggingface.co/routellm : datasets `gpt4_dataset` (train.jsonl 289,790,211 B + valid.jsonl 26,716,111 B, Apache-2.0), `gpt4_judge_battles` (~109k rows), `arena_battles_embeddings` (55.4k), `mmlu_battles` (1.53k), `lmsys-arena-human-preference-55k-thresholds` (57.5k); models mf (205k params), bert (0.3B), causal_llm (8B), plus *_gpt4_augmented / *_mmlu_augmented variants.
- Training cost (paper): MF ~10 epochs on an 8 GB GPU; BERT ~2,000 steps on 2x L4; causal LLM ~2,000 steps on 8x A100 80GB. Router overhead (paper): MF 155 req/s and $3.32 per 1M requests; BERT 69.6 req/s; causal LLM 42.5 req/s; SW ranking 2.9 req/s and $39.26 per 1M.
- Fit: label is preference win-rate on open chat, not answer-equivalence on a typed step. Reuse the MF/BERT router skeleton and threshold sweep; relabel with our own agreement(cheap, expensive).

### (c) FrugalGPT cascades
- Paper https://arxiv.org/abs/2305.05176 (Chen, Zaharia, Zou, 2023): three levers (prompt adaptation, LLM approximation, LLM cascade); claims match GPT-4 at up to 98% lower cost or +4% accuracy at equal cost; API prices differ by ~2 orders of magnitude.
- Repo https://github.com/stanford-futuredata/FrugalGPT (Apache-2.0, last push 2025-02-10). Releases HEADLINES data (zip + sqlite from GitHub release v0.0.1 of lchen001/DataHolder) and, per README, generations from 12 commercial LLM APIs per paper dataset as CSV (names of APIs/datasets not confirmed). Cascade = ordered model list + learned scorer that accepts or escalates; the scorer training method was not confirmed.
- Fit: the cascade accept/escalate-with-scorer is the same shape as "answer from predictor above cutoff, else call real model". Treat as design reference; the code is notebook-grade.

### (d) DART and thinking-budget routing
- DART https://arxiv.org/abs/2606.23181 (v1 2026-06-22, v4 2026-10-03; EMNLP 2026 Findings). Code https://github.com/js-lee-AI/DART (MIT, pushed 2026-10-10, 18 stars). Stage 1 SC-Route: K=2 no-think drafts, accept if equivalent (`answer_extraction.py` has math, MCQ, code equivalence); Stage 2: query-specific thinking budget from draft entropy. Evaluated on Qwen3-8B/14B/32B (0.6B-32B range in abstract) and DeepSeek-V3.2 via hosted API; MATH-500, OlympiadBench, HumanEval, MBPP, AIME24/25 (datasets not redistributed). Results: up to +9.0 pts Olympiad math, +22.5 pts code, 32-73% fewer thinking tokens (v1 HTML said 15-69%). Needs only text API access, so it works on OpenRouter models. Demo ran on one RTX A6000.
- HRBench https://arxiv.org/abs/2605.28398 (2026-05-27, under review): 12+ thinking-mode-switch methods reimplemented (prompt-based, external router, speculative), 6 LLMs (Qwen3.5-2B to Kimi-K2.5), 5 benchmarks. Findings: routing gives stable cost cuts, speculative execution raises accuracy at higher token cost. Repo https://github.com/usail-hkust/HRBench (license NOASSERTION; data availability unverified).
- R2-Router https://arxiv.org/abs/2602.02823 (ICML 2026): router picks model AND output-length budget; ships R2-Bench (accuracy across length budgets), code https://github.com/UCF-ML-Research/R2-Router (no license in API). Claims 4-5x lower cost than prior routers.
- Thinking on vs off evidence (aggregate only, no paired flip rates found):
  - OverthinkingBench in OptimalThinkingBench https://arxiv.org/html/2508.13141v1: 1,440 simple questions, 72 domains. Non-think / think accuracy: Qwen3-8B 95.4 / 96.3, Qwen3-235B 96.6 / 96.9, Claude-Sonnet-4 97.8 / 97.7; Qwen3 think-gain 3.7 pts (1.7B) down to 0.2 pts (235B). Thinking costs 100-854 extra tokens on simple queries. Code https://github.com/facebookresearch/RAM/tree/main/projects/otb (dataset license not stated).
  - Same paper, hard side: Qwen3 non-thinking <=20% on UnderthinkingBench; Qwen3-14B 11% -> 45% with thinking.
  - Secondary (blog, https://huggingface.co/blog/AmirMohseni/reasoning-router, via search snippet): Qwen3-4B 73.4% / 592 tokens non-think vs 82.4% / 3,113 tokens think.
  - Implication: for easy typed steps (classification, yes/no) thinking adds ~0-1 pt and 100-850 tokens; for hard reasoning it matters a lot. Our bookkeeping steps are the easy kind, but this is unmeasured on our data.

### (e) OpenRouter `reasoning` parameter
Source: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens (markdown copy fetched 2026-10-17).
- Request: `reasoning: {effort | max_tokens, exclude?, enabled?}`; send `effort` OR `max_tokens`, not both. Effort values max, xhigh (both ~95% of `max_tokens`), high 80%, medium 50%, low 20%, minimal 10%, none = disabled. `enabled:true` = medium; if unset it is inferred. Anthropic Messages mapping: `thinking:{type:"disabled"}` == `reasoning:{enabled:false}`; `adaptive` == `enabled:true`; `budget_tokens` == `reasoning.max_tokens`.
- `max_tokens` mode honoured by: Anthropic (min 1024, `budget_tokens = max(min(max_tokens*ratio,128000),1024)`, `max_tokens` must exceed budget), Gemini (passes `thinkingBudget`; Gemini 3 maps to `thinkingLevel`, imprecise), some Qwen thinking models (`thinking_budget`). Effort mode honoured by OpenAI reasoning models, Grok, and anything else is translated to nearest supported level. In the live listing `supports_max_tokens:true` appears on qwen3.7-flash, qwen3.8-flash, qwen3.8-omni-flash, claude-opus-4.6 (among models I filtered); others use effort only.
- `exclude:true` hides reasoning but it is still billed. Reasoning tokens bill as output tokens and count against `max_tokens`; if they use the whole budget you get `finish_reason:"length"` with empty content (still billed). OpenAI o-series/GPT-5 do not return reasoning text, only token counts. Detect via `completion_tokens - reasoning_tokens`.
- Discovery: `GET /api/v1/models` returns per model `reasoning:{mandatory, default_enabled, supported_efforts, default_effort, supports_max_tokens}`. `mandatory:true` means "none"/enabled:false is rejected. The docs text does not itself describe `enabled:false` semantics beyond the Messages mapping above; per-model behaviour of `enabled:false` on Qwen3-32B / Gemini-2.5-flash was not tested.
- Where "thinking off" is possible (mandatory=false and `none` in efforts or default off): gpt-5.1/5.2/5.4 family (efforts incl. none), claude-sonnet-5, claude-haiku-5.5, claude-sonnet-4.6, claude-opus-4.6+, gemini-3.1-flash-lite, gemini-2.5-flash/-flash-lite (mandatory false), deepseek-v3.2 (default off), deepseek-v4.x, qwen3-32b/14b/30b-a3b, qwen3.7/3.8 flash. Not possible: gpt-5, gpt-5-mini, gpt-5-nano (lowest = minimal), gemini-2.5-pro, gemini-3.1-pro, gemini-3.5/3.6/3.7/3.8-flash, claude-sonnet-5.5, claude-opus-5.5, grok-4.5+.
- Platform: no per-request markup; 5.5% fee on credit purchases (Standard), 8% Business; free tier 20 req/min and 50 req/day, buying >= $10 credits lifts to 1,000 free-model requests/day (https://openrouter.ai/pricing, fetched 2026-10-17).

## Reusable components

| name | url | license | why |
|---|---|---|---|
| RouterBench data + loader | https://huggingface.co/datasets/withmartian/routerbench ; https://github.com/withmartian/routerbench | HF: none stated; code MIT | Free offline cheap-vs-expensive agreement/oracle/threshold-certification demo; 0-shot file is only 95 MiB |
| LLMRouterBench data + harness | https://huggingface.co/datasets/NPULH/LLMRouterBench ; https://github.com/ynulihao/LLMRouterBench | unclear (badge MIT, API none, HF none) | Modern flagship pairs incl. instruct vs thinking variants, per-query score/tokens/cost, 10 router baselines |
| RouteLLM | https://github.com/lm-sys/RouteLLM | Apache-2.0 | Router class API (`calculate_strong_win_rate`), threshold calibration, MF/BERT routers, OpenAI-compatible server; relabel for agreement |
| RouteLLM HF assets | https://huggingface.co/routellm | datasets Apache-2.0 (gpt4_dataset), others not shown | 119k GPT-4-judged preference rows, precomputed embeddings, pretrained mf/bert routers |
| FrugalGPT | https://github.com/stanford-futuredata/FrugalGPT | Apache-2.0 | Cascade/scorer reference and released generations CSVs for 12 APIs (2023 models) |
| DART | https://github.com/js-lee-AI/DART | MIT | Stdlib-only draft-agreement + thinking-budget router with answer-equivalence checkers; works against any text API |
| OverthinkingBench / OptimalThinkingBench | https://github.com/facebookresearch/RAM/tree/main/projects/otb | repo MIT (RAM); project data license unstated | 1,440 simple-query eval to sanity check thinking-off parity |
| HRBench | https://github.com/usail-hkust/HRBench | NOASSERTION | Reference impls of 12+ thinking-switch strategies |
| R2-Router / R2-Bench | https://github.com/UCF-ML-Research/R2-Router | none stated | Length-budget routing dataset and router |
| Router-R1 router dataset | https://huggingface.co/datasets/ulab-ai/Router-R1-Router-Dataset | Apache-2.0 | 14k NQ-style questions x 6 small LLMs with perf_list/cost_list; tiny, quick smoke |
| vLLM Semantic Router | https://github.com/vllm-project/semantic-router | Apache-2.0 | Maintained (pushed 2026-10-17, 6.2k stars) policy-driven model-selection layer; reasoning-toggle support not confirmed |

## Numbers

OpenRouter list prices, USD per 1M tokens in/out, fetched 2026-10-17. "R-off" = reasoning can be disabled per `/models` metadata.

| model (slug) | in | out | R-off / notes |
|---|---|---|---|
| openai/gpt-4.1 | 2.00 | 8.00 | no reasoning param; ctx 1,047,576; batch 1.00/4.00 |
| openai/gpt-4.1-mini | 0.40 | 1.60 | no reasoning |
| openai/gpt-4.1-nano | 0.10 | 0.40 | no reasoning |
| openai/gpt-5 | 1.25 | 10.00 | mandatory; efforts high..minimal |
| openai/gpt-5-mini | 0.25 | 2.00 | mandatory; high..minimal |
| openai/gpt-5-nano | 0.05 | 0.40 | mandatory; high..minimal |
| openai/gpt-5.4 | 2.50 | 15.00 | R-off yes (none); default_enabled false |
| openai/gpt-5.4-mini | 0.75 | 4.50 | R-off yes |
| openai/gpt-5.4-nano | 0.20 | 1.25 | R-off yes |
| openai/gpt-5.5 | 5.00 | 30.00 | R-off yes |
| openai/gpt-5.6-sol / terra / luna | 4/20 ; 2/12 ; 0.2/1.2 | | R-off yes; efforts max..none |
| anthropic/claude-sonnet-4.6 | 3.00 | 15.00 | R-off yes |
| anthropic/claude-sonnet-5 | 2.00 | 10.00 | R-off yes (default on, high) |
| anthropic/claude-sonnet-5.5 | 2.00 | 10.00 | mandatory |
| anthropic/claude-haiku-4.5 | 1.00 | 5.00 | reasoning optional |
| anthropic/claude-haiku-5.5 | 0.10 | 0.50 | R-off yes; price looks like an outlier, sanity-check before relying on it |
| anthropic/claude-opus-5 / 5.5 | 5/25 ; 4/20 | | 5 off-able; 5.5 mandatory |
| google/gemini-2.5-pro | 1.25 | 10.00 | mandatory |
| google/gemini-2.5-flash | 0.30 | 2.50 | optional |
| google/gemini-2.5-flash-lite | 0.10 | 0.40 | optional |
| google/gemini-3.1-pro-preview | 2.00 | 12.00 | mandatory; high/medium/low |
| google/gemini-3.1-flash-lite | 0.25 | 1.50 | R-off yes |
| google/gemini-3.6-flash / 3.8-flash | 0.75/3.75 ; 0.825/4.125 | | mandatory |
| deepseek/deepseek-v3.2 | 0.28 | 0.42 | default non-thinking; toggle |
| deepseek/deepseek-v4-flash / v4-pro | 0.091/0.182 ; 0.2088/0.4176 | | efforts xhigh/high/none |
| deepseek/deepseek-v4.1-flash | 0.15 | 0.60 | none supported; weekday UTC 01-04 and 06-10 priced 2x (`pricing.overrides`) |
| qwen/qwen3-32b | 0.08 | 0.28 | thinking toggle |
| qwen/qwen3-30b-a3b | 0.12 | 0.50 | toggle, default on |
| qwen/qwen3-235b-a22b-2507 (instruct) vs -thinking-2507 | 0.09/0.55 vs 0.45/3.50 | | separate slugs; thinking mandatory |
| qwen/qwen3.7-flash | 0.03 | 0.13 | supports reasoning.max_tokens |
| qwen/qwen3.8-flash | 0.15 | 0.47 | supports reasoning.max_tokens |

Cost per 1,000 calls at an assumed 800 input / 150 output tokens (my arithmetic, list price; reasoning adds hidden output tokens, the +1,000-token column is an assumption, not a measurement):

| model | no reasoning | +1,000 reasoning tok |
|---|---|---|
| gpt-4.1 / mini / nano | $2.80 / $0.56 / $0.14 | n/a |
| gpt-5 / mini / nano | $2.50 / $0.50 / $0.10 | $12.50 / $2.50 / $0.50 |
| gpt-5.4 / mini / nano | $4.25 / $1.28 / $0.35 | $19.25 / $5.78 / $1.60 |
| sonnet-5 / haiku-5.5 | $3.10 / $0.155 | $13.10 / $0.66 |
| gemini-3.1-pro / 2.5-flash / 2.5-flash-lite | $3.40 / $0.62 / $0.14 | $15.40 / $3.12 / $0.54 |
| deepseek-v3.2 / qwen3-32b | $0.29 / $0.11 | $0.71 / $0.39 |

- Smoke budget: a paired 1,000-prompt run of gpt-4.1 + gpt-4.1-mini = $3.36; plus gpt-4.1-nano = $3.50. A thinking-on/off flip-rate run on qwen3-32b or deepseek-v3.2 for 1,000 prompts = under $1.
- OpenRouter credit: $20 purchased nets about $18.96 after the 5.5% fee (if the ~$20 is already balance, no fee applies).
- Dataset sizes: RouterBench 0-shot 99.6 MB / 5-shot 171 MB / raw 1.20 GB; LLMRouterBench 1.28 GB; RouteLLM gpt4_dataset 290 MB + 27 MB; Router-R1 dataset 14k rows.
- Router inference throughput (RouteLLM paper): MF 155 req/s, BERT 70 req/s, causal-LLM 42 req/s, SW-ranking 2.9 req/s. DART demo: one RTX A6000 for Qwen3-8B.
- Token overhead of thinking on simple queries (OverthinkingBench): o3 ~135, GPT-OSS-120B ~110, Claude-Sonnet-4 ~332, Qwen3-8B ~854 tokens.
- Training cost of baselines (RouteLLM paper): judge labels ~$700 (GPT-4); MF fits on an 8 GB GPU; causal-LLM router needs 8x A100 (out of budget, skip).

## Blockers / plan changes

1. RouterBench cannot validate the modern tier/thinking claim: all models are 2023, no reasoning variants. Use it only to build/test the offline "agreement + certification curve" code at $0, then switch to LLMRouterBench or live OpenRouter calls.
2. No "same answer" rate for thinking on/off exists in the literature I found. Plan a measured smoke: for N=500-1,000 labelled items per step type, run model X with reasoning off vs a low/high effort and record paired flips (right->wrong, wrong->right, answer string changes). Expected cheap: under $1 on qwen3-32b/deepseek.
3. "Thinking off" is not available on many frontier-cheap models (gpt-5/-mini/-nano, gemini-2.5-pro, gemini 3.x flash, claude-sonnet-5.5). Pairs for the thinking mission must come from models whose metadata says mandatory=false; otherwise the cheapest arm is `minimal`/`low` effort, which still bills reasoning tokens.
4. RouteLLM labels are preference win-rates, not agreement; relabel with agreement(cheap, expensive) computed from our own logs. `mf`/`sw_ranking` need an embeddings key; swap in a local sentence-transformer. The repo is unmaintained since Aug 2024, so vendor and pin rather than depend on upstream.
5. DART suggests a cheap label-free signal (two cheap samples agree) usable as the predictor's confidence even before training; costs 2x cheap calls but avoids needing paired expensive labels. Evaluated only on math/code, so check on typed-step tasks.
6. Prices and model catalog churn fast (e.g. the live list has gpt-5.6-*, claude-haiku-5.5, gemini-4-argon). Snapshot `/api/v1/models` JSON with a date next to each experiment; deepseek-v4.1-flash has time-of-day pricing.
7. OpenRouter hides OpenAI reasoning text and bills it; cost accounting must read `reasoning_tokens` from usage, and `max_tokens` too small yields empty content with full billing. Set `max_tokens` generously for any reasoning arm.
8. Licence hygiene: RouterBench and LLMRouterBench HF cards state no licence and contain third-party model outputs; fine for internal smokes, flag before redistributing in the product.

## Unverified

- RouterBench pickle schema (column names, long vs wide format) and exact per-dataset prompt counts: not confirmed; I did not unpickle the files. The per-record fields listed come from the paper text. Whether HF requires login for it: the unauthenticated API returned the file list and sizes, no gating flag was seen.
- RouterBench paper is inconsistent about dataset count (7 vs 8); 405,467 samples / 64 tasks as stated.
- LLMRouterBench licence (badge MIT, GitHub API and HF none); `results/download.md` contents not read.
- FrugalGPT: names of the 12 APIs, dataset list in the CSV release, scorer training details.
- HRBench data availability; R2-Router licence; vllm-project/semantic-router reasoning-toggle features and OpenAI-compat mode.
- Per-model behaviour of `reasoning.enabled:false` on Qwen3-32B and Gemini-2.5-flash via OpenRouter (docs only describe the Anthropic Messages mapping); effort-to-token mapping accuracy per provider.
- claude-haiku-5.5 price ($0.10/$0.50) and model names like gpt-5.6-luna / gemini-4-argon come only from the live OpenRouter listing; I did not cross-check provider pricing pages.
- Qwen3-4B 73.4% vs 82.4% numbers are from a blog snippet via search, not the primary paper.
- Any measured latency or throughput for OpenRouter-hosted models (not collected), and OpenRouter paid-tier rate limits.
- Whether thinking-on/off changes answers on typed bookkeeping steps: no data anywhere; must be measured.
