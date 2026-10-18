# Context pruning: research for Dreaming Mission 5 (date: 2026-10-17)

Scope: learned prompt/context pruning, span-importance labels by ablation, what to reuse, cheapest honest way to build ablation labels for ~500 prompts x ~8 spans.
Method: web fetches of arXiv/HF/GitHub pages and the public OpenRouter model-list API (curl, no key), all on 2026-10-17. "unverified" = could not confirm.

## Summary

- No public pruner targets typed agent prompts (bookkeeping steps with ~8 labeled spans). Public scorers are trained on meeting transcripts (LLMLingua-2), RAG passages (Provence, RECOMP) or code (SWE-Pruner, LaMR). Plan to fine-tune our own small scorer.
- Closest prior art to our exact task is AGORA (arXiv 2605.26596, May 2026): a 125M RoBERTa-base scorer trained on counterfactual "next action changed when this step was removed" soft labels; MIT code incl. label pipeline + training script (no weights, no label data). Reuse its recipe.
- ProCut (LinkedIn, EMNLP-Industry 2025) shows leave-one-out (LOO) over prompt segments matches Shapley/LASSO/greedy and the brute-force 2^M oracle (0.748 vs 0.752 vs 0.749 avg score, M=5 segments). But it is template-level (one score per segment for the whole dataset), not per-instance. No code found.
- ContextCite (MIT, `pip install context_cite`): 32 random ablations + LASSO is enough for faithful attribution; AttriBoT (ICLR 2025) approximates LOO >300x cheaper but only for self-hosted models (KV cache, proxy model).
- For 8 spans, LOO is cheap: 9 calls/prompt = 4,500 calls = ~6M input tokens = $0.3 (Llama-3.1-8B) to $13 (Sonnet-class) on current OpenRouter prices. Labels must come from the call-site model (or a calibrated proxy).
- LOO misses redundancy (two spans carry the same fact: both look unimportant). ContextCite itself warns ablation can change meaning via dependencies. Mitigation: validate on ~50 prompts with the full 2^8=256 subset sweep on a cheap model, and always verify the pruned prompt end-to-end (matches Dreaming's shadow/certify loop).
- Token-level pruners (LLMLingua-2, Selective Context) are weak on structured/code text and not query-aware: on code, LLMLingua-2 at 3.3x gets EM 15.5 vs 40.5 full-context (SWE-Pruner paper, Table 4). Use them only as a baseline/feature, not as the scorer.
- Claude models on OpenRouter do not expose logprobs (per `supported_parameters`); OpenAI-mini, Llama, Qwen, DeepSeek, Gemma entries do. Soft labels from logprob drop are only available on those.

## Findings

### Pruners with open weights/code
- **LLMLingua-2** (arXiv 2403.12968, https://arxiv.org/abs/2403.12968; repo https://github.com/microsoft/LLMLingua, MIT). Token classification (keep/drop) with a bidirectional encoder, trained on GPT-4-distilled extractive compression of MeetingBank (5,169 meetings -> 41,746 chunks of <=512 tokens). Paper numbers: MeetingBank QA EM 86.92 at 3.1x vs 87.75 uncompressed; LongBench avg 39.1 at ~5x (2k-token budget) vs 44.0 full, 42.4 at ~3x; ZeroSCROLLS 33.4 at ~5x vs 34.7 full. Compressor latency 0.4-0.5 s vs 1.5-2.9 s (LLMLingua) and ~15.5 s (Selective-Context); peak GPU 2.1 GB vs 16.6 / 26.5 GB. Training: xlm-roberta-large 23 h, mBERT 16 h for 10 epochs (hardware not stated). Annotation $ cost not reported. Task-agnostic, not conditioned on the question.
  - HF: https://huggingface.co/microsoft/llmlingua-2-xlm-roberta-large-meetingbank (MIT, 559M params, 2.27 GB, 139k downloads) and https://huggingface.co/microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank (Apache-2.0 per HF API, 177M params, 709 MB, 547k downloads).
  - API detail: `compress_prompt_llmlingua2(..., return_word_label=True)` returns per-word keep labels; `token_to_word='mean'` averages token probs; `device_map='cpu'` supported (https://raw.githubusercontent.com/microsoft/LLMLingua/main/DOCUMENT.md). PyPI llmlingua latest 0.2.2 released 2024-04-09 (https://pypi.org/project/llmlingua/). Raw per-token keep-probabilities can be had by loading the HF checkpoint as a standard token-classification model (unverified, standard for that head).
  - Training data: https://huggingface.co/datasets/microsoft/MeetingBank-LLMCompressed (cc-by-nc-sa-4.0, 5,169 rows, 137 MB, not gated); eval QA: https://huggingface.co/datasets/microsoft/MeetingBank-QA-Summary (cc-by-nc-sa-4.0, 862 test rows, 7 MB).
- **SWE-Pruner** (arXiv 2601.16746, v4 2026-05-07; https://github.com/Ayanami1314/swe-pruner, MIT, 319 stars). Agent writes a "goal hint"; 0.6B skimmer (finetuned from Qwen3-Reranker-0.6B, dual head: line-level compression + doc relevance) keeps relevant lines. Weights: https://huggingface.co/ayanami-kitasan/code-pruner (MIT, 634M params, 1.36 GB, needs the repo's custom head code). Training data: 61,184 (query, code, line-mask, relevance) samples; teacher Qwen3-Coder-30B-A3B writes queries + masks, Qwen3-Next-80B-A3B-Thinking judges and keeps ~1/6; 3 epochs, 8 GPUs, batch 128, lr 3e-5. Labels are LLM-annotated, NOT ablation-based. Reported: 23-54% token reduction on SWE-Bench Verified; LongCodeQA 13.95x at acc 59.46 vs 54.05 full; LCC 5.56x EM 31.5 vs 40.5 full. First-token latency <100 ms up to 8K tokens (GPU not stated). Baselines in its tables: LLMLingua-2, Selective-Context, RAG, LongCodeZip. Python-only training set.
- **LaMR** (arXiv 2605.15315, May 2026, https://arxiv.org/abs/2605.15315). Two CRF heads (semantic, dependency; AST-derived labels) + MoE gate on the SWE-Pruner backbone (0.6B, same 61K data). LCC 9.17x EM 34.5 vs 40.5 full; LongCodeQA 16.48x acc 61.26; SWE-Bench Verified (Sonnet 4.5) 30.5% fewer tokens, 71.8% vs 70.6% solved. Code/weights link: none found. Code-specific; not reusable for bookkeeping text.
- **Provence** (arXiv 2501.16214, ICLR 2025). Sentence-level, query-conditioned pruning unified with reranking; 430M DeBERTa-v3; labels synthetic from Llama-3-8B; trained on MS MARCO + NQ. https://huggingface.co/naver/provence-reranker-debertav3-v1. License is non-commercial (card says both CC BY-NC-ND 4.0 and CC BY-NC 4.0; inconsistent). 512-token limit. Threshold 0.1 conservative / 0.5 aggressive.
- **RECOMP** (arXiv 2310.04408; https://github.com/carriex/recomp, MIT; models + `fangyuan/recomp_training` 6.4 GB, no license tag). Extractive compressor labels = effect on the LM of PREPENDING each candidate sentence (positive = best log-likelihood/correct answer; up to 5 hard negatives) - an "add-one" cousin of ablation labels. Abstractive: GPT-3.5 summaries (4 per example, ~48k Wikitext examples), keep the best by end-task score, empty string if no help. Results (Flan-UL2): NQ ~36 vs 660 tokens (~5%) EM 37.0 vs 39.4; TriviaQA 58.7 vs 62.4; HotpotQA extractive ~11% EM 30.4 vs 32.8. Dollar cost not reported.
- **Selective Context** (https://github.com/liyucheng09/Selective_Context, MIT, `pip install selective-context`). Self-information from GPT-2 at sentence/phrase/token level; no benchmark table in README; 15.5 s compressor latency in LLMLingua-2's table.

### Ablation / counterfactual labelling work
- **AGORA** (arXiv 2605.26596, https://arxiv.org/abs/2605.26596; code https://github.com/ranranrannervous/agoracompression, MIT, 1 commit, 2 stars - very new). Label for (current observation, past step) = fraction of paired rollouts whose canonicalized next action differs: K=8 re-queries per pair at T=1.0, 4 with full trajectory, 4 with the step removed. 21,523 pairs from 1,244 trajectories (ALFWorld, ScienceWorld, WebShop) -> ~172k backbone calls (my arithmetic: 21,523 x 8; paper gives no $). Scorer: RoBERTa-base 125M, linear-probe 5 epochs then full FT 4 epochs; ~2 ms/step inference. Result: token-level extractive compressors reach mean reward <=0.05 in every cell at 1.3-13.3x, AGORA keeps >=75% of uncompressed performance in 8 of 9 cells. Released: label pipeline (`cf_label_pipeline.py`), `train_scorer.py`; no weights or label file evident. Caveat: fresh repo, results unreplicated.
- **ProCut** (LinkedIn; EMNLP 2025 Industry pp.285-309, arXiv 2508.02053; https://aclanthology.org/2025.emnlp-industry.20.pdf). Segments prompt templates (pre-defined / sentence-paragraph / LLM-driven), scores segments by SHAP, LOO, LASSO, greedy; LOO needs M calls, SHAP up to 2^M. GPT-4.1 mini, T=0, 5 iterations, 10 threads. Avg score at 25/50/75% retention: LOO 0.748, SHAP 0.752, LASSO 0.748, greedy 0.752, brute-force oracle 0.749 (uncompressed 0.817); beats Selective Context (0.090) and LLMLingua-2 on SQuAD. LLM-driven estimator needs <M calls and cuts latency ~52%. Production prompts: 73% and 84% token cuts. Finding relevant to us: few-shot examples carry most attribution; role-play/zero-shot-CoT segments ~0. Code: none found in paper text (unverified).
- **ContextCite** (arXiv 2409.00729, https://github.com/MadryLab/context-cite, MIT). Random ablation masks -> logit of p(original response) -> sparse linear (LASSO) surrogate. "32 context ablations suffice" (=32x the cost of one generation, fully parallelizable); evaluated at 32/64/128/256. Keeping top-k ContextCite sources raised F1 for Llama-3-8B on HotpotQA and NQ. Warns ablation can change meaning (pronoun example, App. C). API: `ContextCiter.from_pretrained(hf_model, context, query)`; HF models only (GPU in README).
- **AttriBoT** (arXiv 2411.15102, ICLR 2025, https://github.com/r-three/AttriBoT; license unverified). Approximates exact LOO via KV cache reuse, hierarchical (paragraph then sentence; R=0.97 between paragraph LOO and sum of sentence LOO on HotpotQA), and small proxy models (1B-8B proxies correlate highly with Llama-70B LOO; 90% recall keeping half the sources). >300x faster than exact LOO. Needs self-hosted weights for the KV tricks; the proxy idea works through an API.
- **PoC** (arXiv 2603.19733, Mar 2026): predicts task performance to pick the most aggressive compression meeting a user-set floor. Conceptually the same as Dreaming's certification. No code found.
- **ACON** (arXiv 2510.00615, ICLR/ICML 2026; https://github.com/microsoft/acon): learns compression guidelines from failure analysis, no span labels; 26-54% peak-token reduction on AppWorld/OfficeBench/multi-objective QA; distilled compressor keeps >95% accuracy (as summarized in search results; license unverified). Alternative if span labelling disappoints.
- **Surveys**: Li, Liu, Su, Collier, "Prompt Compression for LLMs: A Survey", NAACL 2025 (https://arxiv.org/abs/2410.12388) - hard vs soft prompts taxonomy. A 2026 agent-context-compression survey exists only as an ACL ARR May 2026 OpenReview submission and a July 2026 preprints.org review (https://www.preprints.org/manuscript/202607.0924); contents not read.
- **Repo-level compression study** (arXiv 2604.13725, Apr 2026, WIP): 8 methods, 4x compression, latent-vector methods beat full context by up to 28.3% BLEU on repo code tasks; text/visual compression cuts latency up to 50%. Not read in depth.

### Benchmarks / downloadable datasets (no login)
- LongBench: https://huggingface.co/datasets/THUDM/LongBench (not gated, 618 MB, files LongBench.py + data.zip, no license tag, last modified 2024-12-18). Script-based loader; whether it works on current `datasets` is unverified - download `data.zip` directly.
- ZeroSCROLLS: https://huggingface.co/datasets/tau/zero_scrolls (not gated, 378 MB of per-task zips + `zero_scrolls.py`, no license tag, modified 2025-07-20).
- MeetingBank-LLMCompressed / MeetingBank-QA-Summary: see above (cc-by-nc-sa-4.0, not gated).
- HotpotQA: https://huggingface.co/datasets/hotpotqa/hotpot_qa (CC-BY-SA-4.0, parquet, not gated, `supporting_facts` field with sentence ids = free gold span labels for a sanity check of any span scorer).
- LongCodeQA, LCC: used by SWE-Pruner/LaMR; HF ids not verified.
- Kaggle: nothing here needs it.

## Reusable components

| name | url | license | why |
|---|---|---|---|
| AGORA label + scorer pipeline | https://github.com/ranranrannervous/agoracompression | MIT | Counterfactual next-action-change labels, RoBERTa-base scorer, LP-FT training script; closest recipe to our task. 1 commit, unreplicated. |
| ContextCite | https://github.com/MadryLab/context-cite | MIT | Random-ablation + LASSO attribution, 32 ablations; HF models only, but the surrogate-fit logic is ~50 lines to copy for API models. |
| AttriBoT | https://github.com/r-three/AttriBoT | unverified | Fast LOO for self-hosted models; proxy-model trick applies to any API. |
| LLMLingua-2 large (HF) | https://huggingface.co/microsoft/llmlingua-2-xlm-roberta-large-meetingbank | MIT | Baseline + init for a span scorer (559M; CPU fine-tune is slow). |
| LLMLingua-2 small (HF) | https://huggingface.co/microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank | Apache-2.0 (HF API) | 177M encoder, 709 MB: practical to fine-tune on CPU/small GPU with span-pooled head. |
| llmlingua package | https://github.com/microsoft/LLMLingua | MIT | `return_word_label`, CPU `device_map`; last PyPI release 2024-04-09. |
| SWE-Pruner weights/code | https://huggingface.co/ayanami-kitasan/code-pruner , https://github.com/Ayanami1314/swe-pruner | MIT | Query-conditioned line scorer on Qwen3-Reranker-0.6B; code-only, needs custom head code. Use as architecture template or for code spans. |
| Qwen3-Reranker-0.6B | https://huggingface.co/Qwen/Qwen3-Reranker-0.6B | unverified | Cross-encoder backbone behind SWE-Pruner/LaMR (per their cards). |
| RECOMP extractive | https://github.com/carriex/recomp | MIT | Insertion-style label recipe; NQ/TQA/HotpotQA checkpoints. |
| Provence | https://huggingface.co/naver/provence-reranker-debertav3-v1 | CC BY-NC(-ND) 4.0 (inconsistent card) | Sentence-level query-aware baseline; non-commercial, probably OK for a smoke only. |
| ProCut method | https://aclanthology.org/2025.emnlp-industry.20.pdf | paper only | Defines LOO/SHAP/LASSO segment protocol; reimplement (~30 lines). |
| HotpotQA | https://huggingface.co/datasets/hotpotqa/hotpot_qa | CC-BY-SA-4.0 | Gold supporting sentences to sanity-check any scorer. |

## Numbers (OpenRouter list prices fetched 2026-10-17 from https://openrouter.ai/api/v1/models, USD per 1M tokens in/out)

| model id | in | out | logprobs flag |
|---|---|---|---|
| meta-llama/llama-3.1-8b-instruct | 0.05 | 0.08 | yes |
| openai/gpt-oss-20b | 0.018 | 0.09 | yes |
| qwen/qwen3-14b | 0.12 | 0.24 | yes |
| deepseek/deepseek-v4-flash | 0.091 | 0.182 | yes |
| openai/gpt-4o-mini | 0.15 | 0.60 | yes |
| openai/gpt-4o-mini:batch | 0.075 | 0.30 | yes |
| anthropic/claude-haiku-5.5 | 0.10 | 0.50 | no |
| anthropic/claude-haiku-5.5:batch | 0.05 | 0.25 | no |
| google/gemini-3.8-flash | 0.83 | 4.12 | no |
| anthropic/claude-sonnet-5.5 | 2.00 | 10.00 | no |
| anthropic/claude-opus-5.5 | 4.00 | 20.00 | no |

(`:batch` variants are listed at half price; how to call them / turnaround is unverified.)

Cost to label 500 prompts x 8 spans (ASSUMPTIONS: avg prompt 1,500 tokens, 8 equal spans, 20 output tokens; scale linearly with real prompt size; my arithmetic):

| scheme | calls | input tok | Llama-3.1-8B | gpt-4o-mini | Haiku 5.5 | Gemini 3.8 Flash | Sonnet 5.5 |
|---|---|---|---|---|---|---|---|
| A. T=0 LOO: 1 base + 8 deletions | 4,500 | 6.0M | $0.31 | $0.95 | $0.65 | $5.35 | $12.9 |
| B. AGORA-style soft: 4 base + 4 per span (T=1) | 18,000 | 24M | $1.23 | $3.82 | $2.58 | $21.4 | $51.6 |
| C. ContextCite: 32 random masks | 16,000 | 12M | $0.63 | $1.99 | ~$1.5 | ~$12 | $27 |
| D. exhaustive 2^8 on 50 prompts (validation only) | 12,800 | 9.6M | $0.50 | $1.6 | ~$1.3 | ~$12 | ~$22 |

- Opus 5.5 on scheme A is ~$26 (over the ~$20 credit). Cached-input prices exist (e.g. Sonnet 5.5 cache read about $0.10/M) but LOO deletions only share the prefix before the deleted span.
- Wall-clock: ProCut ran 10 parallel threads against GPT-4.1 mini. At 10 concurrent and ~1 s/call, scheme A is ~8 min (my estimate; OpenRouter rate limits unverified).
- Published label-set sizes for scale: AGORA 21,523 pairs; SWE-Pruner 61,184 samples; LLMLingua-2 41,746 chunks; RECOMP ~48k examples. Ours is 4,000 span labels - small, so use a pretrained encoder with a light head, or a GBDT on span features first.
- Scorer latency (papers, hardware mostly unstated): LLMLingua-2 0.4-0.5 s per MeetingBank prompt, 2.1 GB GPU; SWE-Pruner <100 ms to 8K tokens; AGORA ~2 ms/step. CPU latency for any of these: unverified (torch/transformers not installed locally, no benchmark run).
- Local machine has 14 logical CPUs; no GPU assumed. Fine-tuning 177M-559M encoders on ~6M tokens/epoch on CPU: my estimate is hours per epoch for the 177M model (unverified); a single rented GPU for <1 h is the safer plan.

## Cheapest honest way to build ablation labels (recommended)

1. Segment by construction. Prompts are typed, so spans are the 8 template slots (ProCut's "pre-defined segmentation"). No segmenter needed. Mark spans that must never be removed (output schema, current record) as always-keep, AGORA-style; only ablate the rest.
2. Label with the call-site model. Per prompt: 1 base call + 1 deletion call per span, T=0 (scheme A). Delete the span, keep the template glue, so it matches what the pruner does at inference. Label = output change in the step's own equality (categorize_transaction label flip; flag_anomaly yes/no flip; match_invoice id flip; draft_note via embedding/ROUGE threshold or judge - skip draft_note in the first smoke).
3. Add a noise floor. Re-run the base call 3x on ~100 prompts; "output changed" is only counted beyond the base-vs-base flip rate (API nondeterminism at T=0 is common but its rate for these models is unmeasured here: unverified).
4. Soft labels where logprobs exist (Llama/Qwen/DeepSeek/gpt-4o-mini): label = drop in logprob of the base answer, 1 forward pass with max_tokens=1-5. AGORA's soft-label protocol (scheme B) is the upgrade if hard flips are too sparse (most spans will not flip an output; expect label imbalance - unmeasured).
5. Validate cheaply: on ~50 prompts run all 256 subsets on a cheap model (scheme D, ~$0.5-1.6) and check that LOO ranks agree with Banzhaf/Shapley ranks; if redundancy is large, add the 8 "keep-only-this-span" calls or the ContextCite 32-mask fit. ProCut found LOO ~ brute force at M=5 but at template level.
6. Proxy option: if the call-site model is Sonnet/Opus-class, label on a 4o-mini/Haiku/Llama proxy and verify on ~50 prompts with the real model (AttriBoT: small-proxy LOO correlates highly with 70B LOO on HotpotQA; transfer to our domain is untested).
7. Scorer: start with GBDT/logistic on span-type id, position, length, embedding similarity to the step input (likely strong because spans are typed). Then a span-pooled head on the 177M LLMLingua-2-small encoder (or RoBERTa-base as in AGORA), input = [step input] [SEP] [span]. Use LLMLingua-2 mean keep-prob per span as a zero-shot baseline feature.
8. Certify: pruned-prompt output vs base output agreement at a threshold, tied to the operator error budget - same machinery as Dreaming's shadow/certify, so the pruner needs no separate evaluation harness.

## Blockers / plan changes

- No reusable pretrained scorer for structured agent prompts; any "pruning works" claim for our domain needs our own labels. Budget for ~$1-13 of labels (scheme A, model-dependent) is fine; budget risk only if the call-site model is Opus/Fable-class (Fable 5.1 is $10/$50 per M: scheme A ~$65).
- LOO/hard-flip labels will be sparse and miss redundancy; plan for noise-floor + 256-subset validation, or use soft labels.
- Licenses: Provence weights and MeetingBank-LLMCompressed are non-commercial (CC BY-NC*); LongBench, ZeroSCROLLS, RECOMP data have no license tag; AttriBoT, Qwen3-Reranker, ACON licenses unverified. LaMR has no released code/weights.
- AGORA's repo is one commit with no weights/labels; treat as a recipe, not a dependency.
- Claude models via OpenRouter expose no logprobs, so soft logprob labels only work on non-Claude models.
- llmlingua pip package has not been released since 2024-04; may need version pinning with current transformers (unverified). Loading the HF checkpoints directly avoids this.
- LongBench/ZeroSCROLLS use loading scripts; download zips directly rather than rely on `datasets` script support (unverified whether it still works).

## Unverified

- Dollar cost of annotation for LLMLingua-2 (GPT-4), RECOMP (GPT-3.5), SWE-Pruner (Qwen teacher), AGORA labels; none reported.
- Hardware used for LLMLingua-2 training/latency tables and SWE-Pruner latency table.
- CPU latency/throughput of LLMLingua-2, SWE-Pruner, AGORA scorer (nothing run locally).
- Whether ProCut has public code; ProCut per-instance (vs template-level) performance.
- AttriBoT, Qwen3-Reranker-0.6B, ACON license.
- Provence "up to 80% compression / 2x speedup" (third-party summary only); its exact license.
- LaMR code/weights existence (none found in the paper text I read; page tail 745 chars unread).
- ACON numbers beyond search-result summaries; the 2026 agent-compression surveys (not read).
- OpenRouter `:batch` usage, rate limits, and that list price equals billed price (cache/provider variance).
- API nondeterminism rate at T=0 for the specific models used.
- Whether AGORA's label backbone matched its evaluation backbones (paper text unclear).
