## In plain words

We priced a demo bookkeeping workflow (10,000 runs a day, three model calls per run) under every cost lever we could find, using real token counts from earlier smokes and today's published prices. Prompt caching saves nothing on the demo prompts as written, because the repeated part of each prompt (46 to 245 tokens) is shorter than the minimum the providers will cache (512 tokens on Claude Haiku 5.5 and Sonnet 5.5, 1,024 on OpenAI and Gemini). Caching would not replace the pruner anyway, since caching discounts the repeated instructions while pruning shrinks the per-record history, and the two stack. Once a prompt carries about 550 (Claude) or 1,050 (OpenAI, Gemini) tokens of fixed instructions, caching becomes the bigger saving and should simply be switched on in the runtime. On the demo, the biggest savings come from the small certified student (mission 2) and turning thinking off (mission 1); together they cut the Sonnet 5.5 bill from $76 to $40 a day. Adding the pruner takes it to $26, but only $3 of that last step comes from the step where pruning has been proven, and the other $11 depends on pruning the free-text note, which is still unproven (S16 is running). So build missions 1 and 2 before mission 5. Speculating the next step is cheap to break even on, but it can only hide as much time as the next step takes, so it saves 1 to 3 percent in long-tool workloads and 6 to 30 percent when tools take 0.3 to 3 seconds.

## Setup

Everything here is arithmetic on logged token counts plus pricing-page lookups; no pod was used. Files are in `smokes/s19_token_economics/`.

- **Workflow modeled** (missions.md): per run, one `categorize` (12 labels), one `flag_anomaly` (record + invoice + bank feed + vendor history), one `draft_note` (same context plus the flag, 1-3 sentence free text). `vendor_lookup` and `match_invoice` are tools, not LLM calls. 10,000 runs a day.
- **Token classes per call**: P fixed prefix (instructions + labels, cacheable), V per-call record (paid in full), H vendor history (variable, prunable), O output. From the S16 prompt set (300 prompts, mean 22.5 history lines, 573 o200k tokens of history, 103 of record), the S13 measured 77-label prompt for the long-prefix case, and a 10-pair paid token probe for Claude (Claude counts 1.51x the o200k tokens on records and 1.48x on history; `tokens.json`). Non-Claude teachers use o200k counts + 10 template tokens.

| class | Claude (Haiku/Sonnet 5.5) | OpenAI / Gemini (o200k) |
|---|---|---|
| categorize, 12 labels | P 245, V 31, O 3 | P 153, V 18, O 2 |
| categorize, 77 labels (S13) | P 771, V 16, O 10 | P 428 (mini) / 551 (Gemini), V 11-12, O 5-8 |
| flag_anomaly | P 114, V 133, H 846, O 2 | P 71, V 88, H 573, O 2 |
| draft_note | P 67, V 156, H 846, O 111 | P 46, V 103, H 573, O 75 |

- **Prices**: the OpenRouter `/models` listing fetched 2026-10-17 (`models.json`, includes the `:batch` variants and cache read/write fields), cross-checked against the vendor pages below. Teachers: claude-sonnet-5.5, claude-haiku-5.5, openai/gpt-5.4, gpt-5.4-mini, gpt-5.4-nano, gemini-3.5-flash-lite, qwen3-32b (no caching or Batch on OpenRouter).
- **Lever definitions** (all parameters at the top of `run.py`): prefix cache hit rate 0.95 on synchronous traffic (0.65 on Batch, the low end of Anthropic's 30-98% best-effort range); pruning removes 60-80% of history tokens (S6b removed 82% of tokens on a typed step) and 5% of pruned calls are re-asked on the full prompt through Batch (missions.md: one call in twenty); thinking delta 157 tokens on flag_anomaly (S5: gpt-5.4-mini effort high 166.5 completion tokens vs 8.0 with thinking off); certified student takes 60/75/90% of calls on the two typed steps (S1/S8: 59/89/94% coverage at 770/3,080/9,993 rows), is free at inference, and 5% of its answers are re-asked on the teacher through Batch; draft_note stays on the teacher.
- **Speculation**: TraceLab latency distribution (`tracelab_latency_s.npy`, 743,050 tool calls, Claude Code and Codex pooled), predictor latencies measured once on this laptop (batch 1, CPU), S14 launched-speculation precision as the hit rate.

### Pricing sources (fetched 2026-10-17)

| provider | page | what it says that matters here |
|---|---|---|
| Anthropic | platform.claude.com/docs/en/about-claude/pricing | Haiku 5.5 $0.10 in / $0.50 out per MTok (prompts up to 100k), 5m cache write $0.125, 1h write $0.20, read $0.01. Sonnet 5.5 $2 / $10, write $2.50, 1h $4, read $0.10 (0.05x). Batch API: 50% off input and output (Haiku 5.5 $0.05 / $0.25, Sonnet 5.5 $1 / $5). Cache multipliers "stack with ... the Batch API discount". |
| Anthropic | .../build-with-claude/prompt-caching | Write 1.25x (5m) / 2x (1h); read 0.1x, but 0.05x on Opus 5.5 and Sonnet 5.5. Minimum cacheable prompt 512 tokens for Haiku 5.5, Sonnet 5.5, Opus 5.5; 1,024 for Sonnet 5; 4,096 for Haiku 4.5. Hits need a 100% identical prefix in the order tools, system, messages; a shorter prompt is processed uncached with no error. |
| Anthropic | .../build-with-claude/batch-processing | Batches finish within 1 hour in most cases, 24 hours at most. With caching, hits are "best-effort ... typically 30% to 98%"; recommends the 1-hour cache for batches with shared context. |
| OpenAI | developers.openai.com/api/docs/pricing (platform.openai.com redirects here; no date on page) | gpt-5.4 $2.50 in / $0.25 cached / $15 out; Batch $1.25 / $0.13 / $7.50. gpt-5.4-mini $0.75 / $0.075 / $4.50; Batch $0.375 / $0.0375 / $2.25. gpt-5.4-nano $0.20 / $0.02 / $1.25; Batch $0.10 / $0.01 / $0.625. Batch is 50% of Standard. |
| OpenAI | developers.openai.com/api/docs/guides/prompt-caching (no date on page) | Automatic. For GPT-5.6 and later: 1,024-token minimum, read 0.1x, write 1.25x. For earlier models (the gpt-5.4 family): no cache-write charge, a model-dependent cached rate, and the minimum "varies with request settings". Retention 24h by default for orgs without ZDR. Example hit rates "about 70%" and "above 90%", illustrative only. |
| Google | ai.google.dev/gemini-api/docs/pricing ("Last updated 2026-10-16 UTC") | Gemini 3.5 Flash-Lite $0.30 in / $2.50 out (thinking tokens billed as output), context caching $0.03, storage $1.00 per MTok-hour; Batch $0.15 / $1.25 (50%). |
| Google | ai.google.dev/gemini-api/docs/caching | Implicit caching on by default for Gemini 2.5 and newer; "we automatically pass on cost savings"; listed minimums 2,048 (2.5) and 4,096 (3.x); Flash-Lite 3.5 is not listed. |
| OpenRouter | openrouter.ai/docs/guides/best-practices/prompt-caching (no date on page) | Anthropic needs `cache_control` (write 1.25x, read 0.1x); OpenAI and Gemini 2.5+ are automatic (OpenAI before GPT-5.6: no write cost; Gemini implicit: read 0.25x on this page, but Google's own table above says $0.03 = 0.1x, which is what the listing uses). Sticky routing to the provider that holds the cache, keyed on `session_id` (10-minute expiry), not used when `provider.order` is set. Usage fields `cached_tokens`, `cache_write_tokens`. Requests inside one Anthropic `:batch` job may run concurrently, so a cache write may not be visible to the others. |
| OpenRouter | openrouter.ai/docs/batch-quickstart (no date, no beta label) | `POST /api/v1/batches` with the plain model slug; "runs on `:batch` endpoint variants"; "typically billed at 50% of the model's standard per-token pricing"; only window is 24h; "prompt-caching rates vary by model"; the page does not say how caching combines with the Batch discount. The `/models` listing does price it: `claude-haiku-5.5:batch` cache read $0.005 = 0.5 x $0.01. |

All pages fetched; none failed. Where a page carries no date the fetch date (2026-10-17) is the stamp. No scout numbers were needed.

### Measured cache behavior (38 paid probes of at most 1 output token, $0.0036; made in the earlier attempt of this smoke, no calls this session)

| probe | result |
|---|---|
| Haiku 5.5, 77-label system prompt (777 tokens) with `cache_control`, provider pinned to Claude Platform on AWS | call 1: 764 tokens written, $9.73e-5 (124% of the uncached $7.82e-5); calls 2 and 3: 764 tokens read, $9.74e-6 and $9.64e-6 (12.5%). Matches 1.25x write and 0.1x read exactly. |
| Haiku 5.5, 12-label prompt (254-257 tokens) with `cache_control` | nothing written, nothing read, three times: below the minimum. |
| gpt-5.4-nano pinned to OpenAI, 433 tokens x2, 1,270 tokens x2, then 1,270 identical tokens x5 with a `session_id` | `cached_tokens` = 0 in all 9 calls, full price each time. OpenAI automatic caching was not observed at 1,270 tokens in this probe. Cause unknown (the page says the minimum on pre-5.6 models depends on request settings). Treat OpenAI caching on the gpt-5.4 family as unverified. |

The 512-token Claude minimum is from the docs; the probes only bracket it (254 not cached, 764 cached).

## Results

### 1. Cost per 1,000 calls, per step and teacher (USD)

Columns: no cache; prefix caching; caching + pruning 60% / 80% of history (the cat rows have no history, so pruning does nothing there); Batch (all of the step's calls batched); Batch + cache; thinking off; certified student at 60/75/90% coverage (with caching, 5% audit through Batch). "cacheable" means the fixed prefix reaches the provider minimum. `think off` only differs on flag_anomaly, the one step that thinks.

| teacher | step | P | V | H | O | cacheable | no cache | prefix cache | cache + prune 60% | cache + prune 80% | Batch | Batch + cache | think off | student 60% | student 75% | student 90% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sonnet-5.5 | cat12 | 245 | 31 | 0 | 3 | no | 0.5792 | 0.5792 | 0.5792 | 0.5792 | 0.2896 | 0.2896 | 0.5792 | 0.2404 | 0.1557 | 0.0710 |
| sonnet-5.5 | cat77 | 772 | 16 | 0 | 11 | yes | 1.6880 | 0.3140 | 0.3140 | 0.3140 | 0.8440 | 0.4349 | 1.6880 | 0.1387 | 0.0948 | 0.0510 |
| sonnet-5.5 | flag | 114 | 133 | 846 | 2 | no | 3.7749 | 3.7749 | 2.8545 | 2.5163 | 1.8875 | 1.8875 | 2.2049 | 1.5666 | 1.0145 | 0.4624 |
| sonnet-5.5 | note | 67 | 156 | 846 | 111 | no | 3.2508 | 3.2508 | 2.3173 | 1.9791 | 1.6254 | 1.6254 | 3.2508 | 1.3491 | 0.8737 | 0.3982 |
| haiku-5.5 | cat12 | 245 | 31 | 0 | 3 | no | 0.0290 | 0.0290 | 0.0290 | 0.0290 | 0.0145 | 0.0145 | 0.0290 | 0.0120 | 0.0078 | 0.0035 |
| haiku-5.5 | cat77 | 771 | 16 | 0 | 10 | yes | 0.0838 | 0.0188 | 0.0188 | 0.0188 | 0.0419 | 0.0227 | 0.0838 | 0.0082 | 0.0056 | 0.0029 |
| haiku-5.5 | flag | 114 | 133 | 846 | 2 | no | 0.1887 | 0.1887 | 0.1427 | 0.1258 | 0.0944 | 0.0944 | 0.1102 | 0.0783 | 0.0507 | 0.0231 |
| haiku-5.5 | note | 67 | 156 | 846 | 111 | no | 0.1625 | 0.1625 | 0.1159 | 0.0990 | 0.0813 | 0.0813 | 0.1625 | 0.0675 | 0.0437 | 0.0199 |
| gpt-5.4 | cat12 | 153 | 18 | 0 | 2 | no | 0.4600 | 0.4600 | 0.4600 | 0.4600 | 0.2300 | 0.2300 | 0.4600 | 0.1909 | 0.1236 | 0.0563 |
| gpt-5.4 | cat77 | 428 | 11 | 0 | 8 | no | 1.2169 | 1.2169 | 1.2169 | 1.2169 | 0.6084 | 0.6084 | 1.2169 | 0.5050 | 0.3270 | 0.1491 |
| gpt-5.4 | flag | 71 | 88 | 573 | 2 | no | 4.2139 | 4.2139 | 3.4603 | 3.1740 | 2.1069 | 2.1069 | 1.8589 | 1.7488 | 1.1325 | 0.5162 |
| gpt-5.4 | note | 46 | 103 | 573 | 75 | no | 2.9364 | 2.9364 | 2.1509 | 1.8646 | 1.4682 | 1.4682 | 2.9364 | 1.2186 | 0.7891 | 0.3597 |
| gpt-5.4-mini | cat12 | 153 | 18 | 0 | 2 | no | 0.1380 | 0.1380 | 0.1380 | 0.1380 | 0.0690 | 0.0690 | 0.1380 | 0.0573 | 0.0371 | 0.0169 |
| gpt-5.4-mini | cat77 | 428 | 11 | 0 | 8 | no | 0.3651 | 0.3651 | 0.3651 | 0.3651 | 0.1825 | 0.1825 | 0.3651 | 0.1515 | 0.0981 | 0.0447 |
| gpt-5.4-mini | flag | 71 | 88 | 573 | 2 | no | 1.2642 | 1.2642 | 1.0381 | 0.9522 | 0.6321 | 0.6321 | 0.5577 | 0.5246 | 0.3397 | 0.1549 |
| gpt-5.4-mini | note | 46 | 103 | 573 | 75 | no | 0.8809 | 0.8809 | 0.6453 | 0.5594 | 0.4405 | 0.4405 | 0.8809 | 0.3656 | 0.2367 | 0.1079 |
| gpt-5.4-nano | cat12 | 153 | 18 | 0 | 2 | no | 0.0369 | 0.0369 | 0.0369 | 0.0369 | 0.0185 | 0.0185 | 0.0369 | 0.0153 | 0.0099 | 0.0045 |
| gpt-5.4-nano | cat77 | 428 | 11 | 0 | 8 | no | 0.0977 | 0.0977 | 0.0977 | 0.0977 | 0.0489 | 0.0489 | 0.0977 | 0.0406 | 0.0263 | 0.0120 |
| gpt-5.4-nano | flag | 71 | 88 | 573 | 2 | no | 0.3451 | 0.3451 | 0.2850 | 0.2621 | 0.1725 | 0.1725 | 0.1488 | 0.1432 | 0.0927 | 0.0423 |
| gpt-5.4-nano | note | 46 | 103 | 573 | 75 | no | 0.2387 | 0.2387 | 0.1759 | 0.1530 | 0.1193 | 0.1193 | 0.2387 | 0.0991 | 0.0641 | 0.0292 |
| gemini-3.5-flash-lite | cat12 | 153 | 18 | 0 | 2 | no | 0.0567 | 0.0567 | 0.0567 | 0.0567 | 0.0284 | 0.0284 | 0.0567 | 0.0235 | 0.0152 | 0.0069 |
| gemini-3.5-flash-lite | cat77 | 551 | 12 | 0 | 5 | no | 0.1820 | 0.1820 | 0.1820 | 0.1820 | 0.0910 | 0.0910 | 0.1820 | 0.0755 | 0.0489 | 0.0223 |
| gemini-3.5-flash-lite | flag | 71 | 88 | 573 | 2 | no | 0.6170 | 0.6170 | 0.5293 | 0.4950 | 0.3085 | 0.3085 | 0.2245 | 0.2560 | 0.1658 | 0.0756 |
| gemini-3.5-flash-lite | note | 46 | 103 | 573 | 75 | no | 0.4052 | 0.4052 | 0.3123 | 0.2779 | 0.2026 | 0.2026 | 0.4052 | 0.1682 | 0.1089 | 0.0496 |
| qwen3-32b | cat12 | 153 | 18 | 0 | 2 | no | 0.0143 | 0.0143 | 0.0143 | 0.0143 | 0.0143 | 0.0143 | 0.0143 | 0.0061 | 0.0041 | 0.0021 |
| qwen3-32b | cat77 | 423 | 12 | 0 | 8 | no | 0.0369 | 0.0369 | 0.0369 | 0.0369 | 0.0369 | 0.0369 | 0.0369 | 0.0159 | 0.0106 | 0.0053 |
| qwen3-32b | flag | 71 | 88 | 573 | 2 | no | 0.1030 | 0.1030 | 0.0807 | 0.0715 | 0.1030 | 0.1030 | 0.0591 | 0.0443 | 0.0296 | 0.0149 |
| qwen3-32b | note | 46 | 103 | 573 | 75 | no | 0.0789 | 0.0789 | 0.0553 | 0.0462 | 0.0789 | 0.0789 | 0.0789 | 0.0339 | 0.0227 | 0.0114 |

Where the money sits without any lever (share of no-cache cost, Sonnet 5.5 and Haiku 5.5 alike): categorize 12 labels is 85% prefix, 11% record, 5% output; flag_anomaly is 45% history, 42% output (the 157 thinking tokens), 7% record, 6% prefix; draft_note is 52% history, 34% output, 10% record, 4% prefix. For gpt-5.4-mini the output share of flag is 57% because its output is priced 6x its input.

Caching sensitivity: the 77-label categorize on Sonnet 5.5 costs $2.07 per 1,000 calls at 0% hit rate (every call pays the 1.25x write), $1.15 at 50%, $0.59 at 80%, $0.31 at 95%, $0.24 at 99%; uncached it is $1.69. So the cache needs a hit rate above roughly 20% to break even on the 5-minute write.

### 2. Which lever saves what (Sonnet 5.5, baseline $76.05/day; single lever each, not additive)

| lever | saves $/day | share of bill |
|---|---|---|
| certified student, 90% coverage on the two typed steps (mission 2) | 38.21 | 50.2% |
| Batch for every call (only if all steps tolerate up to 24 h) | 38.03 | 50.0% |
| certified student, 75% | 31.84 | 41.9% |
| certified student, 60% | 25.47 | 33.5% |
| pruning 80% of history (mission 5) | 25.30 | 33.3% |
| pruning 60% of history | 18.54 | 24.4% |
| thinking off on flag_anomaly (mission 1) | 15.70 | 20.6% |
| prefix caching, if categorize carried the 77-label list | 13.74 | 15.8% |
| cache-friendly layout (flag writes the shared record+history, note reads it) | 12.52 | 16.5% |
| Batch for the audit calls only (inside student 75%) | 0.82 | 1.1% |
| prefix caching as specified (demo prompts) | 0.00 | 0.0% |

**One-line ranking (Sonnet 5.5):** certified student (34-50%) is about Batch-for-everything (50%, conditional on latency) > pruning (24-33%) > thinking off (21%) > cache layout or a 77-label prefix (16%) >> Batch on audits only (1%) > caching the demo prompts as written (0%).

Same ranking on the other teachers: Haiku 5.5 has near-identical percentages (same token mix; within 1.3 points on every lever). gpt-5.4 and gpt-5.4-mini: student 36-54%, Batch-all 50%, thinking off 31%, pruning 20-28%, audits 1.2%, caching 0 (no prefix reaches 1,024 tokens). Gemini 3.5 Flash-Lite: student 37-55%, Batch-all 50%, thinking off 36%, pruning 17-23%, audits 1.2%, caching 0. Not a lever but the largest number in the table: dropping a tier cuts the same workflow from $76.05 to $3.80/day (Sonnet 5.5 to Haiku 5.5, -95%), $76.10 to $22.83 (gpt-5.4 to mini, -70%), $22.83 to $6.21 (mini to nano, -73%); S5 found the cheaper tiers match the expensive one on only 77-82% of a typed step, so that saving needs a certificate before it counts.

### 3. Does the fixed prefix change the answer? (prefix-size sweep, same workflow, Sonnet 5.5, $/day saved)

The demo prefixes are tiny. If each step instead carried a fixed block (policy, labels, few-shot examples) of P tokens:

| fixed prefix P (tokens) | baseline $/day | caching saves | pruning 70% saves | student 75% saves | caching + pruning saves |
|---|---|---|---|---|---|
| 100 | 73.53 | 0.00 | 21.91 | 29.51 | 21.91 |
| 250 | 82.53 | 0.00 | 21.76 | 33.90 | 21.76 |
| 500 | 97.53 | 0.00 | 21.51 | 41.21 | 21.51 |
| 1,000 | 127.53 | 53.40 | 21.01 | 55.84 | 74.94 |
| 2,000 | 187.53 | 106.80 | 20.01 | 85.09 | 127.87 |
| 4,000 | 307.53 | 213.60 | 18.01 | 143.59 | 233.73 |
| 8,000 | 547.53 | 427.20 | 14.01 | 260.59 | 445.45 |

Caching switches on at 512 tokens on Claude, at 1,024 on OpenAI and Gemini. Crossovers (smallest fixed prefix where caching saves as much as the other lever, 50-token grid): caching beats pruning 70% at 550 tokens on Sonnet 5.5 and Haiku 5.5, 1,050 on gpt-5.4-mini and Gemini 3.5 Flash-Lite; caching beats student 75% at 1,150 (Sonnet), 1,250 (Haiku), 1,100 (mini), 1,350 (Gemini). Past those points caching dominates and the other levers add on top; below them it does nothing. Pruning savings shrink slowly as P grows because the 5% full-prompt audit calls get more expensive.

### 4. Headline: daily bill at 10,000 runs/day, per combination (USD/day)

| combination | sonnet-5.5 | haiku-5.5 | gpt-5.4 | gpt-5.4-mini | gemini-3.5-flash-lite |
|---|---|---|---|---|---|
| 0 baseline: no cache, flag thinking on, all real time | 76.05 | 3.80 | 76.10 | 22.83 | 10.79 |
| 1 + prefix caching | 76.05 | 3.80 | 76.10 | 22.83 | 10.79 |
| 2 + pruning 70% (no cache) | 54.13 | 2.71 | 57.85 | 17.35 | 8.64 |
| 2f pruning 70% on flag_anomaly only (typed step, S6b) | 65.15 | 3.26 | 67.14 | 20.14 | 9.74 |
| 2n pruning 70% on draft_note only (free text, S16 pending) | 65.02 | 3.25 | 66.82 | 20.04 | 9.69 |
| 3 caching + pruning 70% (= build the pruner first) | 54.13 | 2.71 | 57.85 | 17.35 | 8.64 |
| 1c caching with the cache-friendly layout | 63.53 | 3.22 | 76.10 | 22.83 | 10.79 |
| 4 thinking off on flag_anomaly (mission 1) | 60.35 | 3.02 | 52.55 | 15.77 | 6.86 |
| 5 certified student 75% on typed steps (mission 2) | 44.21 | 2.21 | 41.92 | 12.58 | 5.86 |
| 6 missions 1+2 | 39.99 | 2.00 | 35.60 | 10.68 | 4.81 |
| 7 missions 1+2 + caching | 39.99 | 2.00 | 35.60 | 10.68 | 4.81 |
| 7c missions 1+2 + cache-friendly layout | 36.86 | 1.85 | 35.60 | 10.68 | 4.81 |
| 8 missions 1+2+5 (pruning 70% on what still goes to the teacher) + caching | 25.92 | 1.30 | 23.73 | 7.12 | 3.40 |
| 8b missions 1+2 + pruning only flag_anomaly (the S6b-proven step) | 36.95 | 1.85 | 33.02 | 9.91 | 4.50 |
| 0b baseline if categorize had the 77-label list (S13 prompt) | 87.14 | 4.35 | 83.67 | 25.10 | 12.04 |
| 1b same, with prefix caching | 73.40 | 3.70 | 83.67 | 25.10 | 12.04 |

**Should mission 5 (pruner) be built before missions 1 and 2? No.** On Sonnet 5.5, pruner first saves $21.9/day (-29%); mission 1 alone saves $15.7 with no training, mission 2 alone $31.8, both together $36.1 (-47%). The pruner still has room afterwards: on top of 1+2 it saves another $14.1/day (to $25.92, -66% in total), but that increment is split $3.0 from flag_anomaly, the typed step where S6b proved pruning (82% of tokens removed at 92.5% agreement), and $11.0 from draft_note, a free-text step where S6 found a noisy teacher makes ablation labels unreliable and S16 has not finished. So the pruner's case rests on S16. One-time cost to start: about $2.2 in Sonnet teacher labels per 1,000 rows across the two typed sites for the student (Batch, thinking on for flag; S1/S8 want 770 to 9,993 rows for 59-94% coverage, so roughly $1.7 to $22), and about $3.7 of calibration calls for the pruner. At 10k runs/day on Sonnet that pays back in well under a day. On Haiku 5.5 the labels cost about $0.11 per 1,000 rows against a student saving of $1.59/day, so payback is also days, but the whole workflow is only $3.80/day and the absolute savings are small until volume or prompt size grows.

### 5. Speculation break-even (mission 3)

Rule: speculating the next step pays when the hit rate h times the time it hides exceeds predictor latency plus verify cost. The time hidden on a hit is at most min(tool latency L, next-step latency L2), because the next step cannot overlap more than its own length; a miss costs no latency, only wasted work.

TraceLab tool latency (743,050 tool calls, Claude Code + Codex pooled): median 0.218 s, p75 1.6 s, p90 10.1 s, p95 30 s, p99 152 s, mean 15.6 s. 33.8% of calls take 1 s or more and hold 99.3% of all tool time; 2.8% take over 60 s and hold 83%; 18.8% finish in under 50 ms. (The research file cites the Claude-only subset: median 0.3 s, p90 13.6 s, p99 3.6 min, mean 16.8 s; the pooled numbers are close.)

Predictor latency, batch 1, this laptop CPU (median of repeated calls): n-gram/trajectory-cache lookup 0.0003 ms, TF-IDF + logistic regression (77-class) 0.93 ms, 12-feature gradient boosting 11.5 ms, bge-small embedding on CPU 26.4 ms, JSON compare for verification 0.0076 ms, an LLM-written prediction (Haiku, S13 p50) 828 ms. Hit rate: S14 precision of launched speculations (read-only top-1 call correct), 0.19 to 0.79 across warm (W) and cold (X) regimes, retail/airline, two agents (S14 tables_E.md; S14 RESULTS.md was not yet written).

Break-even tool latency L* = overhead / h, and the share of TraceLab calls above it:

| predictor | overhead | h = 0.19: L* / calls above | h = 0.63: L* / calls above |
|---|---|---|---|
| n-gram + trajectory cache | 0.008 ms | 0.04 ms / 98.4% | 0.01 ms / 98.4% |
| gradient boosting or TF-IDF + LR | 11.5 ms | 60 ms / 77.0% | 18 ms / 91.4% |
| bge-small embed + LR | 26.4 ms | 139 ms / 58.3% | 42 ms / 83.8% |
| LLM-written prediction (Haiku) | 828 ms | 4.4 s / 18.2% | 1.3 s / 26.4% |

Hit rate needed to break even when the next step is a 0.83 s Haiku call (h* = overhead / min(L, L2)):

| predictor | tool 30 ms | 100 ms | 300 ms | 1 s | 3 s |
|---|---|---|---|---|---|
| n-gram / cache lookup | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| gradient boosting | 0.38 | 0.12 | 0.04 | 0.01 | 0.01 |
| bge-small + LR | 0.88 | 0.26 | 0.09 | 0.03 | 0.03 |
| LLM-written prediction | impossible | impossible | impossible | 1.00 | 1.00 |

An LLM-written prediction never helps latency: it takes as long as the step it predicts. A gradient-boosting predictor clears the bar for any tool of about 60 ms or more even at the cold-regime hit rate of 0.19 (bge-small needs 140 ms).

How much is actually saved (share of the sequential time L + L2, predictor = gradient boosting, verify included):

| setting | hit 0.47 | hit 0.63 | hit 0.73 | ceiling (hit 1.0, no overhead) |
|---|---|---|---|---|
| TraceLab mix, next step = Haiku 0.83 s, mean over all calls | 0.9% | 1.2% | 1.4% | 1.9% |
| TraceLab mix, next step = Sonnet 1.15 s | 1.1% | 1.5% | 1.8% | 2.4% |
| TraceLab mix, next step = thinking step 2.1 s | 1.6% | 2.2% | 2.5% | 3.5% |
| TraceLab mix, same, for the median call (0.22 s tool) | 4-10% | 6-13% | 7-15% | 9-21% |
| fixed tool 0.1 s, next = Haiku 0.83 s | 5.1% | 6.8% | | |
| fixed tool 0.3 s, next = Haiku 0.83 s | 12.5% | 16.8% | | |
| fixed tool 1 s, next = Sonnet 1.15 s | 21.9% | 29.3% | | |
| fixed tool 3 s, next = thinking 2.1 s | 19.4% | 26.0% | | |

In a coding-agent workload the mean time lives in a tail of tools that last tens of seconds, so hiding a 1 to 2 second LLM step saves 1 to 3.5% of total time even with a perfect predictor. Speculation pays in a regime where tool and next step are comparable (API tools of 0.3 to 3 s, LLM steps of 0.8 to 2.1 s): 6 to 30% at hit rates 0.47 to 0.63, in line with S4's synthetic 19-20%. Money: speculating a read-only tool call costs nothing but compute; speculating an LLM step wastes one call per miss, about (1 - h) x $0.0038 for a Sonnet 5.5 flag_anomaly call, so at h = 0.63 each speculation costs about $0.0014 in expectation and buys 0.3 to 0.5 s.

## Interpretation for the mission

1. **Caching and pruning address different tokens.** Caching discounts the fixed prefix (up to 90% on Haiku/OpenAI, 95% on Sonnet 5.5); pruning removes variable history that no cache can hold because it differs per record. They stack (Sonnet, P = 1,000: caching saves 53.4, caching plus pruning 74.9). So caching does not make the pruner moot.
2. **But on the demo workflow caching is worth exactly $0, and the pruner is not the first thing to build.** The prefixes (46 to 245 tokens) are under every provider minimum. The money in the demo sits in history (45 to 52% of the flag and note cost) and in output (thinking tokens, the note text). Mission 1 (thinking off) and mission 2 (student) go after those with no per-call machinery and rank above the pruner on every teacher. Mission 5's extra saving after 1+2 is real ($14.1/day on Sonnet) but 78% of it is the free-text note step the pruner has not been shown to handle.
3. **Make caching a runtime default, not a mission.** Dreaming's proxy and wrappers should put fixed content first and set `cache_control` for Anthropic (OpenAI and Gemini cache automatically), because the moment a customer prompt carries 550+ tokens of instructions or few-shot examples (Claude) or 1,050+ (OpenAI, Gemini) caching beats a 70% pruner. Our own 77-label categorize (771 tokens) already qualifies: -81% on Claude at a 95% hit rate (modeled), and a measured cache hit costs 12.5% of the uncached call. Order inside the prompt matters: a pruned history placed before the instructions would break the cache.
4. **Two cheap structural wins to note.** (a) A cache-friendly layout, [fixed prefix][record + history][step instruction last], lets flag_anomaly write the cache and draft_note read it within seconds: -16.5% on Claude when both calls reach the teacher (the flag call pays 1.25x on its shared block, the note call 0.05 to 0.1x). It shrinks to 7.8% once the student answers 75% of flags (no writer), and changing prompt order needs re-certification. (b) Batch halves every price but only fits async work: Dreaming's own audit/shadow stream is that work, yet it is only 1% of the bill at one call in twenty. If a customer's whole job tolerates a 24 hour window, Batch alone is the largest single lever (-50%).
5. **Speculation is a latency lever, not a token lever.** The break-even bar is low for anything cheaper than a model call, but the payoff is bounded by the next step's length and only shows up when tool and step times are comparable. That supports the earlier retargeting: speculate read-only tools whose latency is above roughly 100 ms, never speculate with an LLM-written prediction.
6. **Scale check.** At 10,000 runs/day the whole demo is $76/day on Sonnet 5.5 and $3.80 on Haiku 5.5, so even the best combination saves $50/day on Sonnet and $2.50 on Haiku. The levers are about relative savings (-47% for 1+2, -66% with 5) that grow with volume and prompt size; the tier drop itself (-70 to -95%) dwarfs them, which makes the mission 1 tier-router evidence (S5) the most valuable one to firm up.

## Caveats

- Costs are linear in tokens, so means over the logged prompts give exact totals; there is no per-call simulation. The only stochastic element is the speculation resampling (seed 0, 200,000 draws).
- Token profiles come from 300 synthetic S16 prompts and a 10-pair Claude token probe; the S16 response cache was still growing while this ran (n_notes went from 1,742 to 2,056), so figures can move in the third significant digit on a re-run. Tables here are from the final run (`results_econ.json`).
- Thinking delta (157 tokens) is from gpt-5.4-mini and applied to every teacher; Claude and Gemini thinking lengths may differ. Thinking-off is assumed to cost no accuracy (that is mission 1's job to certify).
- The student is free at inference and its build cost is amortized away; coverage 60/75/90% is borrowed from Banking77 (S1/S8), not from bookkeeping data. The pruner fractions (60-80%) are the mission range, not measured on this workflow; S6b measured 82% on a different typed step and S16 (draft_note) had not finished.
- Cache hit rates are assumptions (0.95 sync, 0.65 batch). 10k runs/day is about 7 runs a minute per call site, which keeps a 5-minute cache warm, so 0.95 is conservative for the demo and optimistic for a site that runs a few times an hour. The 1-hour cache (2x write) and Gemini explicit-cache storage are not modeled.
- OpenAI caching was not reproduced in the probe (0 of 9 calls cached at 1,270 tokens on gpt-5.4-nano through OpenRouter), so the OpenAI rows in the sweep and in the crossover (1,050-token threshold, 0.1x read) are the documented behavior, not a measured one. Gemini 3.5 Flash-Lite's minimum cacheable length is not published; the 1,024 used here is a guess from the OpenRouter page's 2.5 Flash figure (Google's own page says 2,048 for 2.5). Cache read is 0.1x from Google's price table; the OpenRouter page's 0.25x looks stale.
- The Claude minimum of 512 tokens is the documented value; the probes bracket it (254 uncached, 764 cached) but do not locate it.
- Prices are OpenRouter's list prices; OpenRouter's credit purchase fee and any provider-specific routing premium are not modeled. The vendor pages agree with the listing on every number used (the pages round OpenAI's batch cached input to $0.13 vs the listing's $0.125, and Gemini's batch cache price to $0.02 vs $0.015).
- TraceLab is pooled over Claude Code and Codex rather than Claude only (the cached `trace_stats.json` came from the earlier attempt; recomputing the Claude-only subset would re-parse a 101 MB file). The research file's Claude-only quantiles are close to the pooled ones. Its tools are coding tools, much slower and heavier-tailed than bookkeeping API calls; the fixed-latency rows are the closer analogue and are labeled as a regime, not a measurement. Hit rates are S14's tau-bench replay precision per launched speculation; the share of positions where a speculation is launched (59 to 81%) is not multiplied in, so the savings rows are per launched speculation. Wasted speculative work is treated as free in latency.
- The earlier attempt of this smoke spent $0.0036 on 38 tiny paid probes (cap for this smoke was $0). Nothing new was spent in this session.
- Layout and pruning change prompt text and therefore the teacher's behavior; both need their own certificates before the savings count.

## Time and cost

- API spend: $0.0036 total (38 probe calls with max_tokens = 1, all in the earlier attempt, cached in `responses.jsonl`; 0 new calls this session). Pod time: none, $0. Logged in `smokes/COSTS.md`.
- Local compute: econ stage 6.5 s (CPU, arithmetic), speculation stage 33 s once (predictor timing, bge-small load), then reruns of under 1 s from the saved timings. Pricing pages: 12 WebFetch calls and 1 web search, 2026-10-17.
- Files: `run.py` (stages `tokens`, `econ`, `spec`), `probe.py` (paid probes, cached), `models.json` (OpenRouter listing), `probe_A.json`, `tokens.json`, `trace_stats.json`, `tracelab_latency_s.npy`, `results_econ.json`, `results_spec.json`, `log_econ.txt`, `log_spec.txt`.
