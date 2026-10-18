# S13 - teacher noise floor and frontier-step latency

## Why this matters (missions.md: "Certify", "Reality is the teacher")
A certificate promises: above the cutoff, the student's error against the teacher is at most the operator's budget. If the teacher disagrees with itself on x% of identical calls, then no student can show an error below roughly x/2 against a single teacher label, and budgets below x cannot be measured reliably. This smoke measures x for five candidate teachers on a typed step (Banking77 intent, 77 labels in the system prompt), and what it costs and how long it takes.

## Setup
- 100 Banking77 test queries (`stratified_subsample(test labels, 100, seed=0)`), system prompt = "You are a bank support intent classifier. Reply with exactly one label from the list, nothing else." + 77 labels one per line, user = the query. `max_tokens=20`, 8 threads, OpenRouter, retries with backoff (3 of 5,600 calls needed one). Answers parsed exact after strip/lower, else nearest label by difflib (counted as a parse failure).
- Condition A: default routing (`temperature=0` is sent but was dropped for haiku, sonnet, gpt and, as routed, gemini; see finding 1), 5 repeats x 100 prompts per model (repeats are separate passes, so a prompt's repeats are minutes apart). Condition B (haiku-5.5, gpt-5.4-mini only): `provider.order=[the provider A reported]`, `allow_fallbacks=false`, `seed=0` (for haiku this pinned the endpoint default routing already used and `seed` is unsupported on Anthropic, so B did not really test pinning for haiku; see finding 5). Added: A2 (sonnet-5.5 with `max_tokens=300`, see finding 3) and C (same prompts pinned to every other endpoint of haiku/gpt/gemini/qwen, 2 repeats) to test whether the provider matters. 5,600 calls, 0 failed requests.
- Model settings actually used (probed first, see log.txt): haiku-5.5 `reasoning.enabled=false`; sonnet-5.5 default (reasoning cannot be disabled); gpt-5.4-mini `reasoning.effort="none"` (not "minimal": minimal still billed 10 hidden reasoning tokens per call and used 19 of the 20 output tokens); gemini-3.5-flash-lite default (reasoning is mandatory, 400 if disabled; measured 0 reasoning tokens on this task); qwen3-32b on DeepInfra: the `reasoning` param is ignored (content None, finish=length), so `"/no_think"` was appended to the system prompt, which leaves 1 reasoning token (empty think block).
- Files: `run.py` (probe | full | extra | report | tables), `responses.jsonl` (every raw response, 5,600 lines), `probe_responses.jsonl`, `stats.json`, `cross.json`, `tables.md`, `log.txt`.

## Headline findings
1. **Four of five teachers dropped `temperature=0` in default routing.** With `provider.require_parameters=true`, requests carrying `temperature` (and `seed` for Anthropic) return 404 "No endpoints found that can handle the requested parameters" for claude-haiku-5.5, claude-sonnet-5.5 and gpt-5.4-mini (gpt-5.4-mini accepts `seed`, not `temperature`). Without that flag OpenRouter just drops it. Gemini behaved the same way as it was actually run: all 500 condition-A gemini calls were served by provider "Google" (Vertex endpoint family, provider name checked in `responses.jsonl`), where OpenRouter's endpoint metadata lists `temperature` as unsupported; only the Google AI Studio endpoints support it. So gemini-A (4.2%) is a default-temperature number like haiku, sonnet and gpt, and the temperature-0 gemini figure is the pinned AI Studio run (condition C, 5.0% on 2 repeats / 100 pairs). Only qwen3-32b on DeepInfra (condition A) and gemini pinned to AI Studio (condition C) ran with temperature honoured, and even honoured temperature 0 is not deterministic: gemini AI Studio 5.0%, qwen DeepInfra 5.2%. This is the strongest single support for the "can only be recorded, not replayed" claim in missions.md, and it means a Dreaming trace row must record the teacher's answer, never assume it can be regenerated.
2. **Teacher noise on identical input** (pairwise disagreement of two calls, 100 prompts x 5 repeats, 95% bootstrap CI over prompts): sonnet-5.5 about 1-3%, weakly determined (1.2% [0.0, 2.8] at `max_tokens=300`: only 3 of 100 prompts flip, giving 12 disagreeing pairs, 8 of which involve an empty answer and 4 are real label flips on one prompt; the earlier "0.4% when it answers" figure drops every pair with an empty answer, including the prompt that was empty 4 times in 5, so it is a lower bound, not an estimate; at `max_tokens=20` it is 3.6% [1.1, 6.7], finding 3), haiku-5.5 3.5% [1.2, 6.4], gemini-3.5-flash-lite 4.2% [1.8, 7.0] (default temperature), qwen3-32b/DeepInfra 5.2% [2.2, 8.6] (temperature 0 honoured), gpt-5.4-mini 10.9% [6.9, 15.1] (23% of prompts get at least two different answers in 5 calls).
3. **Hidden reasoning can leave the answer empty.** Sonnet-5.5 cannot have reasoning disabled; at `max_tokens=20`, 47 of 500 calls (9.4%) came back empty (finish=length, thinking used the budget), single-run accuracy 78.4%. At `max_tokens=300` it is 5 of 500 empty (1%), accuracy 84.8%. Haiku-5.5's default also thinks adaptively (4 of 5 calls in one probe came back empty); `reasoning.enabled=false` gives 0 reasoning tokens and no empties. A typed step must set an output budget well above the label length and treat empty as "no label".
4. **Flips live on the hard prompts.** Prompts with any flip are wrong against gold 55-73% of the time (per call), stable prompts 13-33%. Majority-of-5 is no more accurate than a single run (within +1.2 pp), and 15-34 prompts per model are wrong in all 5 repeats: errors are systematic (Banking77 label ambiguity), noise is concentrated where the teacher is unsure.
5. **The endpoint is part of the teacher's identity; pinning is not shown to reduce within-endpoint noise.** Default routing was sticky: all 5 repeats of a prompt were served by the same provider (one qwen prompt went to Tenstorrent in all 5 repeats while the other 99 went to DeepInfra), so flips cannot be attributed to provider switches here; every flip happened on a single fixed endpoint. Condition B (pinning + seed) is weak evidence either way: for haiku it did not actually test pinning, because default routing already used the pinned endpoint (Claude Platform on AWS) and `seed` is unsupported on Anthropic, so B is effectively a replicate of A (3.6% vs 3.5%); for gpt-5.4-mini B is likewise already on OpenAI, so only `seed` differs: 8.0% vs 10.9% (difference 2.9 pp, CI -0.6 to 6.6, not significant). The pinned-slug conditions C do show that pinning works (every pinned slug was served 200 of 200 times by the requested endpoint, and answers shift with the endpoint). Different endpoints of the same model can disagree more than one endpoint disagrees with itself (table 5), but with 100 prompts the excess over the default-routing within-endpoint rate is statistically significant (paired bootstrap over prompts, 95% CI above 0) only for gemini AI Studio (10.3% vs 4.2%, +6.1 pp [2.0, 11.0]), qwen SiliconFlow (10.8% vs 5.2%, +5.6 pp [1.5, 10.4]) and qwen Tenstorrent (11.4% vs 5.2%, +6.2 pp [1.8, 11.6]; Tenstorrent is perfectly self-consistent, 0.0% on 100 pairs, yet disagrees with DeepInfra on 11.4% of pairs). It is borderline for haiku on Azure (6.8% vs 3.5%, +3.3 pp [0.1, 7.1]) and not significant for haiku on Bedrock, Anthropic and Vertex or gpt-5.4-mini on Azure. Accuracy also differs by qwen endpoint (64% / 67% / 69%).

## Table 1 - noise and parse failures (A = default routing, B = pinned+seed, A2 = sonnet max_tokens 300, C:tag = other endpoints, 2 repeats)
Flip rate = share of prompts whose repeats are not all identical. Pairwise = mean over prompts and repeat pairs of "answers differ". Floor = 1 - mean modal share: the best any fixed student could do against one teacher call. "Truncated" = finish=length with partial text (haiku: 2 long labels cut at 20 tokens, nearest-match recovers them); "off-list" = real label mistakes (for example "get_virtual_card"); the capitalised "Refund_not_showing_up" label is matched case-insensitively.
| model | cond | reps | calls | parse fails (empty / truncated / off-list) | flip rate | pairwise disagree | pairwise excl. empty-content calls | floor vs best fixed student |
|---|---|---|---|---|---|---|---|---|
| claude-haiku-5.5 | A | 5 | 500 | 16 (0 / 15 / 1) | 7.0% | 3.50% | 3.50% | 2.00% |
| claude-sonnet-5.5 | A | 5 | 500 | 65 (47 / 18 / 0) | 6.0% | 3.60% | 0.45% | 2.20% |
| gemini-3.5-flash-lite | A | 5 | 500 | 2 (0 / 0 / 2) | 9.0% | 4.20% | 4.20% | 2.40% |
| gpt-5.4-mini | A | 5 | 500 | 1 (0 / 0 / 1) | 23.0% | 10.90% | 10.90% | 6.00% |
| qwen3-32b | A | 5 | 500 | 7 (0 / 0 / 7) | 9.0% | 5.20% | 5.20% | 3.40% |
| claude-sonnet-5.5 | A2 | 5 | 500 | 5 (5 / 0 / 0) | 3.0% | 1.20% | 0.41% | 0.60% |
| claude-haiku-5.5 | B | 5 | 500 | 18 (0 / 16 / 2) | 8.0% | 3.60% | 3.60% | 2.00% |
| gpt-5.4-mini | B | 5 | 500 | 2 (0 / 0 / 2) | 15.0% | 8.00% | 8.00% | 4.80% |
| claude-haiku-5.5 | C:amazon-bedrock | 2 | 200 | 7 (0 / 6 / 1) | 2.0% | 2.00% | 2.00% | 1.00% |
| claude-haiku-5.5 | C:anthropic | 2 | 200 | 6 (0 / 6 / 0) | 3.0% | 3.00% | 3.00% | 1.50% |
| gpt-5.4-mini | C:azure | 2 | 200 | 1 (0 / 0 / 1) | 10.0% | 10.00% | 10.00% | 5.00% |
| claude-haiku-5.5 | C:azure/global | 2 | 200 | 6 (0 / 6 / 0) | 2.0% | 2.00% | 2.00% | 1.00% |
| gemini-3.5-flash-lite | C:google-ai-studio | 2 | 200 | 0 (0 / 0 / 0) | 5.0% | 5.00% | 5.00% | 2.50% |
| claude-haiku-5.5 | C:google-vertex/global | 2 | 200 | 6 (0 / 6 / 0) | 2.0% | 2.00% | 2.00% | 1.00% |
| qwen3-32b | C:siliconflow/fp8 | 2 | 200 | 5 (0 / 0 / 5) | 5.0% | 5.00% | 5.00% | 2.50% |
| qwen3-32b | C:tenstorrent/fp8 | 2 | 200 | 4 (0 / 0 / 4) | 0.0% | 0.00% | 0.00% | 0.00% |

## Table 2 - accuracy against gold (Banking77 test labels)
| model | cond | single-run acc (mean, min-max) | majority-vote acc | prompts wrong in all reps |
|---|---|---|---|---|
| claude-haiku-5.5 | A | 75.2% (73.0%-77.0%) | 76.0% | 23 |
| claude-sonnet-5.5 | A | 78.4% (76.0%-80.0%) | 79.0% | 20 |
| gemini-3.5-flash-lite | A | 73.6% (71.0%-75.0%) | 73.0% | 22 |
| gpt-5.4-mini | A | 70.8% (69.0%-72.0%) | 72.0% | 22 |
| qwen3-32b | A | 64.0% (63.0%-65.0%) | 65.0% | 34 |
| claude-sonnet-5.5 | A2 | 84.8% (84.0%-85.0%) | 85.0% | 15 |
| claude-haiku-5.5 | B | 75.6% (75.0%-77.0%) | 76.0% | 23 |
| gpt-5.4-mini | B | 70.6% (68.0%-72.0%) | 71.0% | 24 |
| claude-haiku-5.5 | C:amazon-bedrock | 74.0% (74.0%-74.0%) | 74.0% | 26 |
| claude-haiku-5.5 | C:anthropic | 73.0% (72.0%-74.0%) | 72.0% | 26 |
| gpt-5.4-mini | C:azure | 71.5% (70.0%-73.0%) | 73.0% | 25 |
| claude-haiku-5.5 | C:azure/global | 75.0% (75.0%-75.0%) | 75.0% | 25 |
| gemini-3.5-flash-lite | C:google-ai-studio | 74.0% (73.0%-75.0%) | 75.0% | 24 |
| claude-haiku-5.5 | C:google-vertex/global | 75.0% (74.0%-76.0%) | 74.0% | 24 |
| qwen3-32b | C:siliconflow/fp8 | 69.0% (67.0%-71.0%) | 71.0% | 29 |
| qwen3-32b | C:tenstorrent/fp8 | 67.0% (67.0%-67.0%) | 67.0% | 33 |

## Table 3 - latency, tokens, cost (latency client-side, 8 concurrent calls, ms)
| model | cond | p50 ms | p95 ms | prompt tok | completion tok | reasoning tok (mean/max) | $/call | $/1000 labels |
|---|---|---|---|---|---|---|---|---|
| claude-haiku-5.5 | A | 828 | 1047 | 787 | 10.1 | 0.00/0 | 0.000084 | 0.084 |
| claude-sonnet-5.5 | A | 1149 | 2264 | 788 | 10.9 | 1.87/20 | 0.001685 | 1.685 |
| gemini-3.5-flash-lite | A | 574 | 775 | 562 | 5.4 | 0.00/0 | 0.000182 | 0.182 |
| gpt-5.4-mini | A | 762 | 977 | 439 | 8.1 | 0.00/0 | 0.000366 | 0.366 |
| qwen3-32b | A | 390 | 1031 | 434 | 7.7 | 1.00/1 | 0.000037 | 0.037 |
| claude-sonnet-5.5 | A2 | 1148 | 3216 | 788 | 24.9 | 15.00/300 | 0.001825 | 1.825 |
| claude-haiku-5.5 | B | 818 | 976 | 787 | 10.1 | 0.00/0 | 0.000084 | 0.084 |
| gpt-5.4-mini | B | 762 | 973 | 439 | 8.1 | 0.00/0 | 0.000366 | 0.366 |
| claude-haiku-5.5 | C:amazon-bedrock | 1137 | 2488 | 787 | 10.2 | 0.00/0 | 0.000084 | 0.084 |
| claude-haiku-5.5 | C:anthropic | 727 | 1039 | 787 | 10.0 | 0.00/0 | 0.000084 | 0.084 |
| gpt-5.4-mini | C:azure | 951 | 1902 | 439 | 8.1 | 0.00/0 | 0.000366 | 0.366 |
| claude-haiku-5.5 | C:azure/global | 762 | 1023 | 787 | 10.0 | 0.00/0 | 0.000084 | 0.084 |
| gemini-3.5-flash-lite | C:google-ai-studio | 435 | 609 | 564 | 5.4 | 0.00/0 | 0.000183 | 0.183 |
| claude-haiku-5.5 | C:google-vertex/global | 533 | 772 | 787 | 10.0 | 0.00/0 | 0.000084 | 0.084 |
| qwen3-32b | C:siliconflow/fp8 | 1211 | 1446 | 433 | 6.7 | 2.00/2 | 0.000064 | 0.064 |
| qwen3-32b | C:tenstorrent/fp8 | 597 | 903 | 433 | 8.7 | 1.00/1 | 0.000045 | 0.045 |

## Table 4 - providers and flips
| model | cond | providers (calls) | prompts with provider change | flip rate if provider changed | flip rate if same provider | pair disagree diff-provider | pair disagree same-provider |
|---|---|---|---|---|---|---|---|
| claude-haiku-5.5 | A | {'Claude Platform on AWS': 500} | 0 | - | 7.0% | - | 3.50% |
| claude-sonnet-5.5 | A | {'Claude Platform on AWS': 500} | 0 | - | 6.0% | - | 3.60% |
| gemini-3.5-flash-lite | A | {'Google': 500} | 0 | - | 9.0% | - | 4.20% |
| gpt-5.4-mini | A | {'OpenAI': 500} | 0 | - | 23.0% | - | 10.90% |
| qwen3-32b | A | {'DeepInfra': 495, 'Tenstorrent': 5} | 0 | - | 9.0% | - | 5.20% |
| claude-sonnet-5.5 | A2 | {'Claude Platform on AWS': 500} | 0 | - | 3.0% | - | 1.20% |
| claude-haiku-5.5 | B | {'Claude Platform on AWS': 500} | 0 | - | 8.0% | - | 3.60% |
| gpt-5.4-mini | B | {'OpenAI': 500} | 0 | - | 15.0% | - | 8.00% |
| claude-haiku-5.5 | C:amazon-bedrock | {'Amazon Bedrock': 200} | 0 | - | 2.0% | - | 2.00% |
| claude-haiku-5.5 | C:anthropic | {'Anthropic': 200} | 0 | - | 3.0% | - | 3.00% |
| gpt-5.4-mini | C:azure | {'Azure': 200} | 0 | - | 10.0% | - | 10.00% |
| claude-haiku-5.5 | C:azure/global | {'Azure': 200} | 0 | - | 2.0% | - | 2.00% |
| gemini-3.5-flash-lite | C:google-ai-studio | {'Google AI Studio': 200} | 0 | - | 5.0% | - | 5.00% |
| claude-haiku-5.5 | C:google-vertex/global | {'Google': 200} | 0 | - | 2.0% | - | 2.00% |
| qwen3-32b | C:siliconflow/fp8 | {'SiliconFlow': 200} | 0 | - | 5.0% | - | 5.00% |
| qwen3-32b | C:tenstorrent/fp8 | {'Tenstorrent': 200} | 0 | - | 0.0% | - | 0.00% |

Default routing never switched provider across repeats of the same prompt, so "flips that coincide with provider changes" is 0 of 0 here; the question is answered by Table 5 instead.

## Table 5 - same prompts pinned to other endpoints (cross = default-routing A answers vs the pinned endpoint's answers, all pairs)
| model | pinned endpoint | served by | cross-provider disagree vs default-routing A (all pairs) | within pinned endpoint (2 reps) | single-run acc | p50 ms | p95 ms | $/call |
|---|---|---|---|---|---|---|---|---|
| claude-haiku-5.5 | amazon-bedrock | {'Amazon Bedrock': 200} | 4.50% | 2.0% | 74.0% | 1137 | 2488 | 0.000084 |
| claude-haiku-5.5 | anthropic | {'Anthropic': 200} | 5.50% | 3.0% | 73.0% | 727 | 1039 | 0.000084 |
| gpt-5.4-mini | azure | {'Azure': 200} | 12.00% | 10.0% | 71.5% | 951 | 1902 | 0.000366 |
| claude-haiku-5.5 | azure/global | {'Azure': 200} | 6.80% | 2.0% | 75.0% | 762 | 1023 | 0.000084 |
| gemini-3.5-flash-lite | google-ai-studio | {'Google AI Studio': 200} | 10.30% | 5.0% | 74.0% | 435 | 609 | 0.000183 |
| claude-haiku-5.5 | google-vertex/global | {'Google': 200} | 4.60% | 2.0% | 75.0% | 533 | 772 | 0.000084 |
| qwen3-32b | siliconflow/fp8 | {'SiliconFlow': 200} | 10.80% | 5.0% | 69.0% | 1211 | 1446 | 0.000064 |
| qwen3-32b | tenstorrent/fp8 | {'Tenstorrent': 200} | 11.40% | 0.0% | 67.0% | 597 | 903 | 0.000045 |

Modal answers (A vs pinned) differ on 4-6 of 100 prompts for haiku, 10 for gemini, 10-11 for qwen, 13 for gpt-5.4-mini/Azure.

## Consequence: smallest measurable error budget and price per 1000 teacher labels
| teacher | pairwise noise [95% CI] | floor for best fixed student | smallest measurable budget (> pairwise) | recommended floor (CI upper bound, rounded to nearest 1%) | $/1000 labels | p50 / p95 ms | single-run acc |
|---|---|---|---|---|---|---|---|
| claude-sonnet-5.5 (max_tokens 300) | about 1-3%, weakly determined: 1.2% [0.0, 2.8] (0.4% excl. empties is a lower bound) | 0.6% | > 1.2%, but not reliably measurable below ~3% | 3% | 1.825 | 1148 / 3216 | 84.8% |
| claude-haiku-5.5 (reasoning off) | 3.5% [1.2, 6.4] | 2.0% | > 3.5% | 6% | 0.084 | 828 / 1047 | 75.2% |
| gemini-3.5-flash-lite | 4.2% [1.8, 7.0] | 2.4% | > 4.2% | 7% | 0.182 | 574 / 775 | 73.6% |
| qwen3-32b (DeepInfra) | 5.2% [2.2, 8.6] | 3.4% | > 5.2% | 9% | 0.037 | 390 / 1031 | 64.0% |
| gpt-5.4-mini (reasoning none) | 10.9% [6.9, 15.1] | 6.0% | > 10.9% (8.0% pinned+seed) | 15% | 0.366 | 762 / 977 | 70.8% |

Cost is OpenRouter `usage.cost` per call (about 790 prompt tokens for the Claude tokenizer, 440-560 for the others; the 77-label system prompt dominates). Latency is wall time per call at 8-way concurrency with ~10-25 output tokens, so it is close to time-to-first-token plus network; a step that needs more output tokens will be slower.

## Interpretation
- **Certification reference.** Use claude-sonnet-5.5 as the reference teacher from roughly 2-3% budgets upward, where haiku is too noisy (floor ~6%): it has the lowest noise estimate (about 1-3%, weakly determined: 1.2% [0.0, 2.8] rests on 3 of 100 prompts, and the "0.4% when it answers" figure discards the prompt that was empty 4 times in 5) and the best accuracy (84.8%), at 22x haiku's price ($1.83 vs $0.084 per 1000 labels) and a 3.2 s p95. Give it `max_tokens` of at least 300 and treat an empty answer as "no label, retry". Budgets of 1-2% sit inside its own noise CI and are not reliably measurable against a single sonnet call. For budgets of about 6% and above, claude-haiku-5.5 with reasoning off is the better reference: 3.5% noise, $0.084 per 1000, 0.8 s p50. gpt-5.4-mini is a poor reference: 10.9% noise that cannot be reduced (no temperature control, pinning+seed gives at most a few points), so only budgets above ~15% would be measurable. qwen3-32b and gemini sit at 4-5% noise but are cheaper and faster; qwen's accuracy (64%) makes it a weak reference.
- **Pin the provider.** Yes, for any teacher that will certify a student: record the serving provider in every trace row and pin it (`order=[...]`, `allow_fallbacks=false`). Pinning was not shown to lower within-endpoint noise (finding 5), but a certificate computed against endpoint X is not a certificate against endpoint Y: cross-endpoint disagreement significantly exceeds the within-endpoint rate only for gemini AI Studio (10.3% vs 4.2%) and qwen SiliconFlow/Tenstorrent (10.8-11.4% vs 5.2%), is borderline for haiku/Azure (6.8% vs 3.5%), and the modal answer differs on 4-13 of 100 prompts. The cost of pinning is availability (a down endpoint fails instead of falling back); no failures were seen in 5,600 calls.
- **Budget floor to recommend.** Treat a budget as measurable only if it is above the teacher's pairwise noise, and set the product default above its upper CI bound: about 3% for sonnet-5.5 (weakly determined), 6% for haiku-5.5, 7% for gemini, 9% for qwen, 15% for gpt-5.4-mini. A student cannot show less than the floor column (0.6-6%) against a single teacher call even if it equals the modal answer. To certify tighter budgets, label with the majority of 3 calls (3x cost, not measured here) or certify against a lower-noise teacher.
- **Likely smaller in the certified region.** Flips sit on prompts where the teacher is unsure (finding 4), and a certificate only covers the confident region where the student is most often right. The noise inside the certified region is probably below these global rates. That needs the S1 student's confidence on these same 100 prompts and is not measured here.

## Caveats
- 100 prompts and 5 repeats: noise estimates have wide CIs (e.g. haiku 1.2-6.4%); within-endpoint estimates in Table 5 use 100 pairs, so 2% means 2 prompts. Rankings between haiku, gemini and qwen are not statistically separable, nor is sonnet from haiku (CIs 0.0-2.8% vs 1.2-6.4%); only gpt-5.4-mini has a CI mostly above the rest.
- One task (Banking77, heavily ambiguous labels, teachers at 64-85%). A cleaner task will flip less; a free-text step will flip more.
- One time window (about 15 minutes of calls). No day-to-day or load-related drift was measured; per-pass median latency was flat within 10%.
- Settings differ per model by necessity (see Setup), and the qwen prompt has "/no_think" appended to the system message. gpt used effort "none" instead of the requested "minimal".
- Truncation at `max_tokens=20` cut two long labels for the Claude tokenizer; the nearest-match parse recovers them, and truncation is deterministic so it does not add noise. Empty answers (thinking) count as wrong and as disagreements except where noted "excl. empties".
- ASI:One (asi1) models were not tested as teachers here; S11 covers that connector.

## Time and cost
- OpenRouter spend: $2.56 of the $4 cap (probe $0.02, main run $1.40, extra conditions $1.11, manual probing about $0.02), logged in `smokes/COSTS.md` as `s13_teacher_noise`. Wall time roughly 30 minutes including probing (estimate).

## Revision
Reviewer-requested corrections; the numbers in the tables are unchanged (the sonnet CIs and the cross-vs-within differences below were recomputed from `responses.jsonl`, CPU only, bootstrap over prompts, 10,000 resamples).
1. **Temperature claim.** Was: "only gemini and qwen honour temperature=0". Now: four of five teachers dropped temperature in default routing. All 500 condition-A gemini calls were served by provider "Google" (checked in `responses.jsonl`); the reviewer found that OpenRouter's endpoint metadata lists temperature as unsupported there and supported only on Google AI Studio (not re-queried here, no API calls). gemini-A (4.2%) is therefore a default-temperature number; the temperature-0 gemini figure is the pinned AI Studio run (5.0%, 2 repeats / 100 pairs). Added that honoured temperature 0 is still not deterministic (gemini AI Studio 5.0%, qwen DeepInfra 5.2%). The original probe line in `log.txt` ("gemini and qwen honour it") is superseded; it likely used `require_parameters`, which can route to a different endpoint than default routing.
2. **Sonnet recommendation.** Was: sonnet for "budgets of 1-2%", noise "0.4% when it answers", floor 2%. That contradicted the rule that a floor sits above the CI upper bound (2.8%). Now: noise "about 1-3%, weakly determined" (1.2% [0.0, 2.8] rests on 3 of 100 prompts: 12 disagreeing pairs, 8 involving an empty answer, 4 real label flips on one prompt; "0.4% when it answers" drops the prompt that was empty 4 of 5 times and is a lower bound). Sonnet is recommended from roughly 2-3% upward; the recommended floor is 3% (was 2%), and 1-2% budgets are called not reliably measurable. Related wording changes: haiku is recommended "for about 6% and above" (was 5%, now matching its 6% floor), and the caveat no longer says sonnet is statistically separable from the others.
3. **Condition B.** Stated plainly that B did not test pinning for haiku (default routing already used that endpoint and `seed` is unsupported on Anthropic), so "pinning does not reduce noise" is weakly supported; the pinned-slug conditions C do show pinning works. Finding 5 retitled accordingly.
4. **Cross- vs within-endpoint.** Now stated as significant only for gemini AI Studio (+6.1 pp [2.0, 11.0]), qwen SiliconFlow (+5.6 pp [1.5, 10.4]) and qwen Tenstorrent (+6.2 pp [1.8, 11.6]); borderline for haiku on Azure (+3.3 pp [0.1, 7.1]); not significant for the other endpoints. Baseline = the default-routing within-endpoint rate of the same prompts; against the pinned endpoint's own 2-repeat rate the same four are the only ones with a positive lower bound.
5. **Not changed.** The haiku floor stays 6% (CI upper bound 6.4%, rounded to nearest 1% like the other rows), so under a strict "above the upper bound" reading it would be about 6.5%.
