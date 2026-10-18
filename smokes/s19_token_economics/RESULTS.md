## In plain words

We priced a demo bookkeeping workflow (10,000 runs a day, three model calls per run) under every cost lever, using token counts from earlier smokes and the prices published on 2026-10-17. Prompt caching saves nothing on the demo prompts, because their repeated part (46 to 245 tokens) is below the minimum size that providers cache, and it works on different tokens than pruning, so it does not make the pruner pointless. The first version of this report said to build the small certified student and "thinking off" before the pruner; that does not hold up. On a Sonnet 5.5 bill of $63.55 a day, the student at 75% coverage saves about $22.70, the pruner saves about $22.20 at the 70% removal the mission assumes (only $15.50 at what its own test supports), and turning thinking off saves $2.60, not $15.70, unless the model thinks for roughly 160 tokens or more; Sonnet 5.5 cannot switch thinking off at all. None of the three savings is proven on this workflow: the student's coverage comes from a 77-way task, the pruner passes its test only at a 15% disagreement budget, and the one thinking-off test changed 16.5% of answers. So the student and the pruner are close on dollars and stack (about -58% together), the student has the only certificate at a 5% budget, and thinking-off is the weakest cost lever. Outside Claude the cache minimums are not established (Google lists 2,048 to 4,096 tokens; OpenAI documents 1,024 only for GPT-5.6 and later, and our one test got no hit at 1,270 tokens). Speculating the next step can hide at most the shorter of the tool and the step: about 0.5 to 3% of time in a coding-agent trace (0.2 to 1.7% on its Claude-only part), and 6 to 22% when tools take 0.3 to 3 seconds.

## Revision

This is revision 2. A reviewer found six material problems; all are fixed in place (the same `run.py`, cached token profile and pricing listing; no new API spend). The v1 files are kept as `RESULTS_v1.md`, `results_econ_v1.json`, `results_spec_v1.json`.

| # | what was wrong | what changed | how the numbers moved |
|---|---|---|---|
| 1 | One thinking length (157 tokens, gpt-5.4-mini at high effort) was applied to every teacher, the baseline assumed thinking was on at that effort, thinking-off and the student were treated as proven while the pruner was doubted, and there was no sensitivity table. | Two thinking scenarios throughout: central 32 tokens (S5's expensive reference, gpt-5.4 at low effort) and upper 157. Thinking-off now also carries the 1-in-20 audit that the student and pruner already carried. A new `sens` stage gives tables for thinking length, student coverage, pruner removal, audit rate, cache hit rate, history size and extra LLM steps, plus break-even coverage and break-even thinking length. A new "equal-evidence" table puts all three levers at the budget their own evidence supports. S13 added: Sonnet 5.5 reasoning cannot be disabled and Gemini 3.5 Flash-Lite reasoning is mandatory, so thinking-off does not exist for those two. | Sonnet 5.5 baseline $76.05 -> $63.55 (central). Thinking off saves $15.70 (20.6%) -> $2.57 (4.0%) central, $14.76 (19.4%) upper. Student 75%: $31.84 -> $22.70 central. Pruner 70%: $21.92 -> $22.23 central. Mission 2 alone vs pruner alone: $31.8 vs $21.9 -> $22.7 vs $22.2 (a tie). Missions 1+2: -47% -> -37% central (-47% upper). Break-even student coverage against the pruner: 51% / 73% / 85% (central) against a pruner that removes 50 / 70 / 80%. The "mission 1+2 before mission 5" ordering is withdrawn. |
| 2 | S6b was reported as having "proved" pruning on a typed step (82% of tokens removed at 92.5% agreement). | Reworded everywhere: S6b ran HotpotQA yes/no, n = 80, Haiku 5.5; the k=2 pruner certifies only at a 15% disagreement budget (UB 14.3%), not at 10% or 5%, and with selective gating covers 61% of questions for an effective 50% token saving. New lever "pruning 50% (S6b at a 15% budget)" and "pruning at a 5% or 10% budget: $0". The "$3.0 from the proven step" split is withdrawn. | At a 5% or 10% budget the supported pruner saving is $0; at 15% it is $15.47 (gated, 50%) to $25.62 (whole set, 80%) a day on Sonnet central, against $22.23 at the 70% the v1 report modeled. |
| 3 | Interpretation point 2 said missions 1 and 2 "rank above the pruner on every teacher". | Rewritten from the tables. Mission 2 ties the pruner (Sonnet central: $22.70 vs $22.23) and beats it only above 51-85% coverage; mission 1 (thinking off) is below the pruner on every teacher at the central thinking length (Sonnet 4.0% vs 24-40%; gpt-5.4 7.3% vs 23-38%) and above it only at the upper length on gpt-5.4 and Gemini. | See the rankings in section 2. |
| 4 | "1,024 on OpenAI and Gemini" and "about 1,050" were stated as fact. | Re-fetched Google's caching page (updated 2026-10-13) and OpenAI's prompt-caching guide. Google: 2,048 (Gemini 2.5) and 4,096 (3.x), Gemini 3.5 Flash-Lite not listed; the model now uses 2,048 and shows 4,096 as an alternative. OpenAI: 1,024 is documented only for GPT-5.6 and later; for gpt-5.4 the minimum "varies with request settings" and our probe saw 0 of 9 cached calls at 1,270 tokens, so it is shown as unverified. Summary text corrected. | Gemini crossover 1,050 -> 2,050 (2,048 minimum) or 4,100 (4,096 minimum). OpenAI 1,050 stays only as a documented-for-5.6+ figure. Claude 550 is unchanged and is essentially the 512 minimum. |
| 5 | The speculation percentages came from 200,000 resamples (seed 0) of a heavy-tailed latency distribution (sd 789 s, max 440,698 s), inflating the mean to 19.36 s against the true 15.61 s and shrinking every percentage by 20-25%. | Every statistic is computed exactly on the 743,050 values. The Claude-only subset was extracted (14 s, one pass) and is reported next to the pooled data, plus a robustness row with latencies capped at 1 hour. One speculation model is used for every table (parallel and serial predictor). Hit rates are the S14 values (0.19, 0.47, 0.73, 0.79). | Perfect-predictor ceilings 1.9 / 2.4 / 3.5% -> 2.4 / 3.0 / 4.2% (Haiku step / Sonnet step / 2.1 s thinking step). At hit 0.47: 0.9 / 1.1 / 1.6% -> 1.1 / 1.4 / 2.0%. Claude-only ceilings 1.1 / 1.4 / 2.1%. |
| 6 | The $0 cap was breached by $0.0036 (38 paid probe calls in the earlier attempt). | Stated plainly (section "Measured cache behavior" and Time and cost). Nothing in the conclusions depends on the probes: the 1.25x write and 0.1x read multipliers and the 512-token minimum come from Anthropic's pricing and caching pages; the probes agree and are shown as a consistency check only. No call was made in the two sessions since. | None. |

Minor items, all handled: the five spec step types are now addressed (vendor_lookup is a tool in missions.md line 48 and match_invoice is never defined there; pricing both as extra flag-sized LLM calls is an upper-bound row in section 3d); the S6b prompt size (1,989 tokens) enters as a history-size sensitivity; S5 is used for per-configuration reasoning tokens (E1 31.4 / 32.6, mini-high 156.5), S13 for the Sonnet and Gemini reasoning facts, S6 is not used for tokens (its free-text teacher was noise); break-even tables use one model with the predictors separated (TF-IDF + LR 0.93 ms now apart from the 11.5 ms gradient boosting); a faster LLM predictor (qwen3-32b, 390 ms) is evaluated instead of only the 828 ms Haiku step; the 77-label caching row now says which baseline its percentage refers to; the Gemini cache-write price, the 5% audit and the 0.95 hit rate each get a band in section 3d. Still open: the research file's Claude-only latency statistics (median 0.3 s, mean 16.8 s, p99 3.6 min) do not match what the raw file gives for rows with provider "claude" (median 0.115 s, mean 28.7 s, p99 240 s); the difference is unexplained and both are reported.

## Setup

Everything here is arithmetic on logged token counts plus pricing-page lookups; no pod was used. Files are in `smokes/s19_token_economics/`.

- **Workflow modeled** (missions.md): per run, one `categorize` (12 labels), one `flag_anomaly` (record + invoice + bank feed + vendor history), one `draft_note` (same context plus the flag, 1-3 sentence free text). `vendor_lookup` is a tool in missions.md (line 48: "a tool is a call site exactly like a model step"); `match_invoice` is not defined there. Both are treated as non-LLM and priced as an upper-bound sensitivity in 3d. 10,000 runs a day.
- **Token classes per call**: P fixed prefix (instructions + labels, cacheable), V per-call record (paid in full), H vendor history (variable, prunable), O output. Profile is frozen in `tokens.json` (rebuild with `--retokens`): 300 S16 synthetic prompts (mean 22.5 history lines, 573 o200k tokens of history, 103 of record), a 10-pair paid token probe for Claude (Claude counts 1.51x the o200k tokens on records and 1.48x on history), and the S13 measured 77-label prompt for the long-prefix case. S5 contributes only reasoning-token counts per configuration (below), S13 the 77-label profile and the Sonnet and Gemini reasoning facts, S6b its 1,989-token prompt size as a history-size sensitivity; S6 contributes nothing numeric.

| class | Claude (Haiku/Sonnet 5.5) | OpenAI / Gemini (o200k) |
|---|---|---|
| categorize, 12 labels | P 245, V 31, O 3 | P 153, V 18, O 2 |
| categorize, 77 labels (S13) | P 771, V 16, O 10 | P 428 (mini) / 551 (Gemini), V 11-12, O 5-8 |
| flag_anomaly | P 114, V 133, H 846, O 2 | P 71, V 88, H 573, O 2 |
| draft_note | P 67, V 156, H 846, O 111 | P 46, V 103, H 573, O 75 |

- **Thinking length (the parameter v1 fixed at 157).** Measured: gpt-5.4 at low effort 31.4 and 32.6 reasoning tokens (S5, E1 and E1b, 200 inputs each, 77-label typed step); gpt-5.4-mini at high effort 156.5 (S5 Ton); Sonnet 5.5 default (cannot be disabled) 1.9 tokens at a 20-token output cap and 15 at a 300-token cap (S13); Gemini 3.5 Flash-Lite 0 (mandatory, measured on the same step); qwen3-32b 1. Not measured on a bookkeeping flag step for any model. The report runs two scenarios: **central = 32** and **upper = 157**, applied to every teacher, plus a sweep from 0 to 300 in 3a.
- **Evidence behind each lever** (used in section 2b):

| lever | where the evidence comes from | what it does and does not show |
|---|---|---|
| certified student (mission 2) | S1/S8, Banking77 (77-way): 59 / 89 / 94% certified coverage at a 5% budget with 770 / 3,080 / 9,993 rows | not tested on a yes/no or bookkeeping step; coverage 50-90% below is borrowed |
| pruner (mission 5) | S6b, HotpotQA yes/no, n = 80, Haiku 5.5: k=2 removes 82% of tokens, observed disagreement 7.5% (UB 14.3%), certifiable only at a 15% budget; gated version covers 61% for an effective 50% saving; 10% would need about 379 held-out rows if 7.5% holds; 5% is unreachable (even the gold-only prompt disagrees 5-7.5%) | not the flag_anomaly step; the draft_note half (free text) is untested (S16 has no RESULTS yet) |
| thinking off (mission 1) | S5, 77-label step: gpt-5.4-mini high vs none changes 16.5% of answers (9.0% between two none runs, so about 7.5 points beyond noise) and costs about 7 gold points; nothing passed a 2% budget. S13: Sonnet 5.5 cannot be switched off, Gemini 3.5 Flash-Lite cannot, Haiku 5.5 and gpt-5.4 can | one step, one model pair; no measurement on a yes/no step |

- **Prices**: the OpenRouter `/models` listing fetched 2026-10-17 (`models.json`, includes the `:batch` variants and cache read/write fields), cross-checked against the vendor pages below. Teachers: claude-sonnet-5.5, claude-haiku-5.5, openai/gpt-5.4, gpt-5.4-mini, gpt-5.4-nano, gemini-3.5-flash-lite, qwen3-32b (no caching or Batch on OpenRouter).
- **Lever definitions** (parameters at the top of `run.py`): prefix cache hit rate 0.95 on synchronous traffic (0.65 on Batch, the low end of Anthropic's 30-98% best-effort range; banded in 3d); pruning removes 60-80% of history tokens (mission range) or 50% (S6b at a 15% budget), and 5% of pruned calls are re-asked on the full prompt through Batch; thinking-off keeps 1 call in 20 re-run with thinking on through Batch (v2; v1 omitted this); certified student takes 50-90% of calls on the two typed steps, is free at inference, and 5% of its answers are re-asked on the teacher through Batch; draft_note stays on the teacher.
- **Speculation**: TraceLab latency distribution (`tracelab_latency_s.npy`, 743,050 tool calls, Claude Code and Codex pooled, and `tracelab_latency_claude_s.npy`, 304,290 Claude-provider calls), predictor latencies measured once on this laptop (batch 1, CPU), S14 launched-speculation precision as the hit rate (precisions 0.19, 0.20, 0.47, 0.48, 0.68, 0.73, 0.74, 0.79; the grid uses 0.19, 0.47, 0.73, 0.79).

### Pricing sources (fetched 2026-10-17)

| provider | page | what it says that matters here |
|---|---|---|
| Anthropic | platform.claude.com/docs/en/about-claude/pricing | Haiku 5.5 $0.10 in / $0.50 out per MTok (prompts up to 100k), 5m cache write $0.125, 1h write $0.20, read $0.01. Sonnet 5.5 $2 / $10, write $2.50, 1h $4, read $0.10 (0.05x). Batch API: 50% off input and output (Haiku 5.5 $0.05 / $0.25, Sonnet 5.5 $1 / $5). Cache multipliers "stack with ... the Batch API discount". |
| Anthropic | .../build-with-claude/prompt-caching | Write 1.25x (5m) / 2x (1h); read 0.1x, but 0.05x on Opus 5.5 and Sonnet 5.5. Minimum cacheable prompt 512 tokens for Haiku 5.5, Sonnet 5.5, Opus 5.5; 1,024 for Sonnet 5; 4,096 for Haiku 4.5. Hits need a 100% identical prefix in the order tools, system, messages; a shorter prompt is processed uncached with no error. |
| Anthropic | .../build-with-claude/batch-processing | Batches finish within 1 hour in most cases, 24 hours at most. With caching, hits are "best-effort ... typically 30% to 98%"; recommends the 1-hour cache for batches with shared context. |
| OpenAI | developers.openai.com/api/docs/pricing (no date on page) | gpt-5.4 $2.50 in / $0.25 cached / $15 out; Batch $1.25 / $0.13 / $7.50. gpt-5.4-mini $0.75 / $0.075 / $4.50; Batch $0.375 / $0.0375 / $2.25. gpt-5.4-nano $0.20 / $0.02 / $1.25; Batch $0.10 / $0.01 / $0.625. Batch is 50% of Standard. |
| OpenAI | developers.openai.com/api/docs/guides/prompt-caching (no date on page; re-fetched 2026-10-17 in this revision) | Automatic. "The minimum cacheable prompt length is 1,024 tokens for GPT-5.6 and later" (read 0.1x, write 1.25x). For earlier models including gpt-5.4: the minimum "varies with request settings, including tools, images, output schemas"; no fixed number; "no additional cache-write charge"; "model-dependent cached-input rate". The page names no minimum for gpt-5.4, mini or nano. |
| Google | ai.google.dev/gemini-api/docs/pricing ("Last updated 2026-10-16 UTC") | Gemini 3.5 Flash-Lite $0.30 in / $2.50 out (thinking tokens billed as output), context caching $0.03, storage $1.00 per MTok-hour; Batch $0.15 / $1.25 (50%). |
| Google | ai.google.dev/gemini-api/docs/caching (updated 2026-10-13 UTC; re-fetched in this revision) | Implicit caching on by default for Gemini 2.5 and newer. Implicit-cache minimums in the page's table: Gemini 2.5 Flash and 2.5 Pro 2,048; Gemini 3.1 Pro Preview, 3.6 Flash, 3.8 Flash and Gemini 4 Argon 4,096. **Gemini 3.5 Flash-Lite is not listed.** Explicit caching: no minimum given. |
| OpenRouter | openrouter.ai/docs/guides/best-practices/prompt-caching (no date on page) | Anthropic needs `cache_control` (write 1.25x, read 0.1x); OpenAI and Gemini 2.5+ are automatic (OpenAI before GPT-5.6: no write cost; Gemini implicit: read 0.25x on this page, but Google's own table says $0.03 = 0.1x, which is what the listing uses). Sticky routing to the provider that holds the cache, keyed on `session_id` (10-minute expiry), not used when `provider.order` is set. Usage fields `cached_tokens`, `cache_write_tokens`. Requests inside one Anthropic `:batch` job may run concurrently, so a cache write may not be visible to the others. |
| OpenRouter | openrouter.ai/docs/batch-quickstart (no date, no beta label) | `POST /api/v1/batches` with the plain model slug; "runs on `:batch` endpoint variants"; "typically billed at 50% of the model's standard per-token pricing"; only window is 24h; "prompt-caching rates vary by model"; the page does not say how caching combines with the Batch discount. The `/models` listing does price it: `claude-haiku-5.5:batch` cache read $0.005 = 0.5 x $0.01. |

All pages fetched; none failed. Where a page carries no date the fetch date (2026-10-17) is the stamp. No scout numbers were needed.

### Measured cache behavior (38 paid probes of at most 1 output token, $0.0036; made in the earlier attempt of this smoke, no calls in either later session)

**Spend-cap breach.** This smoke's cap was $0 (arithmetic and page fetches only). The earlier attempt spent $0.0036 on these probes (the probe script hard-coded a $0.05 cap). The amount is trivial but it breaches an explicit cap, so the probes are treated as a consistency check only: every conclusion rests on the documented multipliers and minimums above, and none relies on a probe result.

| probe | result |
|---|---|
| Haiku 5.5, 77-label system prompt (777 tokens) with `cache_control`, provider pinned to Claude Platform on AWS | call 1: 764 tokens written, $9.73e-5 (124% of the uncached $7.82e-5); calls 2 and 3: 764 tokens read, $9.74e-6 and $9.64e-6 (12.5%). Agrees with the documented 1.25x write and 0.1x read. |
| Haiku 5.5, 12-label prompt (254-257 tokens) with `cache_control` | nothing written, nothing read, three times: below the minimum. |
| gpt-5.4-nano pinned to OpenAI, 433 tokens x2, 1,270 tokens x2, then 1,270 identical tokens x5 with a `session_id` | `cached_tokens` = 0 in all 9 calls, full price each time. OpenAI automatic caching was not observed at 1,270 tokens. Cause unknown (the page says the minimum on pre-5.6 models depends on request settings). Treat OpenAI caching on the gpt-5.4 family as unverified. |

The 512-token Claude minimum is from the docs; the probes only bracket it (254 not cached, 764 cached).

## Results

### 1. Cost per 1,000 calls, per step and teacher (USD), central thinking length (32 tokens)

Columns: no cache; prefix caching; caching + pruning 60% / 80% of history (the cat rows have no history, so pruning does nothing there); Batch (all of the step's calls batched); Batch + cache; thinking off (the bare call cost, without the 1-in-20 audit that section 2 adds); certified student at 60/75/90% coverage (with caching, 5% audit through Batch). "cacheable" means the fixed prefix reaches the provider minimum (Gemini now 2,048). `think off` only differs on flag_anomaly, the one step that thinks.

| teacher | step | P | V | H | O | cacheable | no cache | prefix cache | cache + prune 60% | cache + prune 80% | Batch | Batch + cache | think off | student 60% | student 75% | student 90% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sonnet-5.5 | cat12 | 245 | 31 | 0 | 3 | no | 0.5792 | 0.5792 | 0.5792 | 0.5792 | 0.2896 | 0.2896 | 0.5792 | 0.2404 | 0.1557 | 0.0710 |
| sonnet-5.5 | cat77 | 772 | 16 | 0 | 11 | yes | 1.6880 | 0.3140 | 0.3140 | 0.3140 | 0.8440 | 0.4349 | 1.6880 | 0.1387 | 0.0948 | 0.0510 |
| sonnet-5.5 | flag | 114 | 133 | 846 | 2 | no | 2.5249 | 2.5249 | 1.5733 | 1.2350 | 1.2625 | 1.2625 | 2.2049 | 1.0478 | 0.6786 | 0.3093 |
| sonnet-5.5 | note | 67 | 156 | 846 | 111 | no | 3.2508 | 3.2508 | 2.3173 | 1.9791 | 1.6254 | 1.6254 | 3.2508 | 1.3491 | 0.8737 | 0.3982 |
| haiku-5.5 | cat12 | 245 | 31 | 0 | 3 | no | 0.0290 | 0.0290 | 0.0290 | 0.0290 | 0.0145 | 0.0145 | 0.0290 | 0.0120 | 0.0078 | 0.0035 |
| haiku-5.5 | cat77 | 771 | 16 | 0 | 10 | yes | 0.0838 | 0.0188 | 0.0188 | 0.0188 | 0.0419 | 0.0227 | 0.0838 | 0.0082 | 0.0056 | 0.0029 |
| haiku-5.5 | flag | 114 | 133 | 846 | 2 | no | 0.1262 | 0.1262 | 0.0787 | 0.0618 | 0.0631 | 0.0631 | 0.1102 | 0.0524 | 0.0339 | 0.0155 |
| haiku-5.5 | note | 67 | 156 | 846 | 111 | no | 0.1625 | 0.1625 | 0.1159 | 0.0990 | 0.0813 | 0.0813 | 0.1625 | 0.0675 | 0.0437 | 0.0199 |
| gpt-5.4 | cat12 | 153 | 18 | 0 | 2 | no | 0.4600 | 0.4600 | 0.4600 | 0.4600 | 0.2300 | 0.2300 | 0.4600 | 0.1909 | 0.1236 | 0.0563 |
| gpt-5.4 | cat77 | 428 | 11 | 0 | 8 | no | 1.2169 | 1.2169 | 1.2169 | 1.2169 | 0.6084 | 0.6084 | 1.2169 | 0.5050 | 0.3270 | 0.1491 |
| gpt-5.4 | flag | 71 | 88 | 573 | 2 | no | 2.3389 | 2.3389 | 1.5384 | 1.2522 | 1.1694 | 1.1694 | 1.8589 | 0.9706 | 0.6286 | 0.2865 |
| gpt-5.4 | note | 46 | 103 | 573 | 75 | no | 2.9364 | 2.9364 | 2.1509 | 1.8646 | 1.4682 | 1.4682 | 2.9364 | 1.2186 | 0.7891 | 0.3597 |
| gpt-5.4-mini | cat12 | 153 | 18 | 0 | 2 | no | 0.1380 | 0.1380 | 0.1380 | 0.1380 | 0.0690 | 0.0690 | 0.1380 | 0.0573 | 0.0371 | 0.0169 |
| gpt-5.4-mini | cat77 | 428 | 11 | 0 | 8 | no | 0.3651 | 0.3651 | 0.3651 | 0.3651 | 0.1825 | 0.1825 | 0.3651 | 0.1515 | 0.0981 | 0.0447 |
| gpt-5.4-mini | flag | 71 | 88 | 573 | 2 | no | 0.7017 | 0.7017 | 0.4615 | 0.3756 | 0.3508 | 0.3508 | 0.5577 | 0.2912 | 0.1886 | 0.0860 |
| gpt-5.4-mini | note | 46 | 103 | 573 | 75 | no | 0.8809 | 0.8809 | 0.6453 | 0.5594 | 0.4405 | 0.4405 | 0.8809 | 0.3656 | 0.2367 | 0.1079 |
| gpt-5.4-nano | cat12 | 153 | 18 | 0 | 2 | no | 0.0369 | 0.0369 | 0.0369 | 0.0369 | 0.0185 | 0.0185 | 0.0369 | 0.0153 | 0.0099 | 0.0045 |
| gpt-5.4-nano | cat77 | 428 | 11 | 0 | 8 | no | 0.0977 | 0.0977 | 0.0977 | 0.0977 | 0.0489 | 0.0489 | 0.0977 | 0.0406 | 0.0263 | 0.0120 |
| gpt-5.4-nano | flag | 71 | 88 | 573 | 2 | no | 0.1888 | 0.1888 | 0.1248 | 0.1019 | 0.0944 | 0.0944 | 0.1488 | 0.0784 | 0.0507 | 0.0231 |
| gpt-5.4-nano | note | 46 | 103 | 573 | 75 | no | 0.2387 | 0.2387 | 0.1759 | 0.1530 | 0.1193 | 0.1193 | 0.2387 | 0.0991 | 0.0641 | 0.0292 |
| gemini-3.5-flash-lite | cat12 | 153 | 18 | 0 | 2 | no | 0.0567 | 0.0567 | 0.0567 | 0.0567 | 0.0284 | 0.0284 | 0.0567 | 0.0235 | 0.0152 | 0.0069 |
| gemini-3.5-flash-lite | cat77 | 551 | 12 | 0 | 5 | no | 0.1820 | 0.1820 | 0.1820 | 0.1820 | 0.0910 | 0.0910 | 0.1820 | 0.0755 | 0.0489 | 0.0223 |
| gemini-3.5-flash-lite | flag | 71 | 88 | 573 | 2 | no | 0.3045 | 0.3045 | 0.2090 | 0.1747 | 0.1522 | 0.1522 | 0.2245 | 0.1264 | 0.0818 | 0.0373 |
| gemini-3.5-flash-lite | note | 46 | 103 | 573 | 75 | no | 0.4052 | 0.4052 | 0.3123 | 0.2779 | 0.2026 | 0.2026 | 0.4052 | 0.1682 | 0.1089 | 0.0496 |
| qwen3-32b | cat12 | 153 | 18 | 0 | 2 | no | 0.0143 | 0.0143 | 0.0143 | 0.0143 | 0.0143 | 0.0143 | 0.0143 | 0.0061 | 0.0041 | 0.0021 |
| qwen3-32b | cat77 | 423 | 12 | 0 | 8 | no | 0.0369 | 0.0369 | 0.0369 | 0.0369 | 0.0369 | 0.0369 | 0.0369 | 0.0159 | 0.0106 | 0.0053 |
| qwen3-32b | flag | 71 | 88 | 573 | 2 | no | 0.0680 | 0.0680 | 0.0440 | 0.0348 | 0.0680 | 0.0680 | 0.0591 | 0.0293 | 0.0196 | 0.0099 |
| qwen3-32b | note | 46 | 103 | 573 | 75 | no | 0.0789 | 0.0789 | 0.0553 | 0.0462 | 0.0789 | 0.0789 | 0.0789 | 0.0339 | 0.0227 | 0.0114 |

Only the flag_anomaly rows depend on the thinking length. At the upper length (157 tokens) the same rows read (no cache / cache+prune 60% / cache+prune 80% / Batch / student 60% / 75% / 90%; the `think off` column is unchanged because it removes the thinking): sonnet-5.5 3.7749 / 2.8545 / 2.5163 / 1.8875 / 1.5666 / 1.0145 / 0.4624; haiku-5.5 0.1887 / 0.1427 / 0.1258 / 0.0944 / 0.0783 / 0.0507 / 0.0231; gpt-5.4 4.2139 / 3.4603 / 3.1740 / 2.1069 / 1.7488 / 1.1325 / 0.5162; gpt-5.4-mini 1.2642 / 1.0381 / 0.9522 / 0.6321 / 0.5246 / 0.3397 / 0.1549; gemini-3.5-flash-lite 0.6170 / 0.5293 / 0.4950 / 0.3085 / 0.2560 / 0.1658 / 0.0756.

Where the money sits without any lever (share of no-cache cost, Sonnet 5.5 and Haiku 5.5 alike, central): categorize 12 labels is 85% prefix, 11% record, 5% output; flag_anomaly is 67% history, 13.5% output (32 thinking tokens plus the answer), 10.5% record, 9% prefix; draft_note is 52% history, 34% output, 10% record, 4% prefix. At the upper length the flag output share is 42% and history 45%. For gpt-5.4-mini the flag output share is 22% (central) because its output is priced 6x its input.

Caching sensitivity: the 77-label categorize on Sonnet 5.5 costs $2.07 per 1,000 calls at 0% hit rate (every call pays the 1.25x write), $1.15 at 50%, $0.59 at 80%, $0.31 at 95%, $0.24 at 99%; uncached it is $1.69. So the cache needs a hit rate above roughly 20% to break even on the 5-minute write.

### 2. Which lever saves what

#### 2a. Sonnet 5.5, single lever each (not additive), each against its own baseline

Baseline $63.55/day at the central thinking length, $76.05/day at the upper. "Evidence" is the status described in Setup.

| lever | evidence | saves $/day (central 32) | % of baseline | saves $/day (upper 157) | % of baseline |
|---|---|---|---|---|---|
| Batch for every call (only if every step tolerates up to 24 h) | pricing page | 31.77 | 50.0% | 38.02 | 50.0% |
| certified student, 90% coverage on the two typed steps (mission 2) | Banking77 94% at 9,993 rows | 27.24 | 42.9% | 38.21 | 50.2% |
| pruning 80% of history (mission 5) | S6b whole-set k=2, 15% budget | 25.62 | 40.3% | 25.30 | 33.3% |
| certified student, 75% | Banking77 89% at 3,080 rows | 22.70 | 35.7% | 31.84 | 41.9% |
| pruning 60% of history | mission range | 18.85 | 29.7% | 18.54 | 24.4% |
| certified student, 60% | Banking77 59% at 770 rows (5% budget) | 18.16 | 28.6% | 25.47 | 33.5% |
| pruning 50% (S6b gated, 15% budget) | S6b | 15.47 | 24.3% | 15.16 | 19.9% |
| certified student, 50% | borrowed | 15.13 | 23.8% | 21.23 | 27.9% |
| prefix caching, if categorize carried the 77-label list | docs + probe | 13.74 | 18.4% of $74.64 | 13.74 | 15.8% of $87.14 |
| cache-friendly layout (flag writes, note reads) | docs, modeled | 12.52 | 19.7% | 12.52 | 16.5% |
| thinking off on flag_anomaly (mission 1), with 1-in-20 audit | S5: failed; not possible on Sonnet 5.5 | 2.57 | 4.0% | 14.76 | 19.4% |
| Batch for the audit calls only (inside student 75%) | pricing page | 0.58 | 0.9% | 0.82 | 1.1% |
| pruning at a 5% or 10% budget | S6b: nothing certifies | 0.00 | 0.0% | 0.00 | 0.0% |
| prefix caching as specified (demo prompts) | below every minimum | 0.00 | 0.0% | 0.00 | 0.0% |

**One-line ranking (Sonnet 5.5, central):** Batch-for-everything (50%, conditional on latency) > certified student (29-43% for 60-90% coverage) ~ pruner at the mission range (30-40% for 60-80% removal) > pruner at what S6b supports (24%) ~ student at 50% coverage (24%) > 77-label prefix cache or cache layout (18-20%) >> thinking off (4%) > Batch on audits only (1%) > caching the demo prompts as written (0%); the pruner at a 5% or 10% budget is 0%. At the upper thinking length thinking off (19%) moves up between the pruner at 60% (24%) and the cache layout (16.5%). Thinking off is not available on Sonnet 5.5 at all (S13).

Other teachers, central (upper in brackets): Haiku 5.5 has the same percentages as Sonnet (same token mix). gpt-5.4: student 29-43% (36-54%), Batch-all 50%, pruning 23-38% (17-28%), thinking off 7.3% (29.6%), audits 0.9%, caching 0 (no prefix reaches 1,024). Gemini 3.5 Flash-Lite: student 28-41% (37-55%), Batch-all 50%, pruning 20-34% (14-23%), thinking off 9.4% (35%; not possible, thinking is mandatory), audits 0.9%, caching 0 (no prefix reaches 2,048). Not a lever but the largest number in the table: dropping a tier cuts the same workflow from $63.55 to $3.18 per day (Sonnet 5.5 to Haiku 5.5, -95%), $57.35 to $17.21 (gpt-5.4 to mini, -70%), $17.21 to $4.64 (mini to nano, -73%); S5 found the cheaper tiers match the expensive one on only 77-82% of a typed step, so that saving needs a certificate before it counts.

#### 2b. The levers at the budget their own evidence supports (Sonnet 5.5, central, $/day saved of $63.55)

| lever | at a 5% budget | at a ~15% budget | note |
|---|---|---|---|
| student | 59% coverage -> $17.86 (28%); 89% -> about $26.9 (42%) | at least that | Banking77 evidence, 77-way; needs 770 to 3,080 teacher labels |
| pruner | $0 (S6b: unreachable, teacher floor 5-7.5%) | gated 50% -> $15.47 (24%); whole set 82% -> about $26 (41%) | HotpotQA yes/no; 10% budget needs about 379 rows and 7.5% holding |
| thinking off | $0 (S5: nothing passed 2%) | $0 on the one measured step (16.5% flips, 9.0% of that is noise) | not available on Sonnet 5.5 |
| student 75% + pruner 50% | | $32.46 (51%) | stack; bill $31.09 |
| student 75% + pruner 70% | | $36.75 (58%) | stack; bill $26.80 |

At equal evidence only the student has a certificate at the tight budget. At the loose budget the pruner is competitive, and the two stack.

### 3. Sensitivity (new)

#### 3a. Thinking length against the thinking-off saving (includes the 1-in-20 audit)

| thinking tokens removed | Sonnet 5.5 / Haiku 5.5 (% of its baseline) | gpt-5.4 (% of its baseline) |
|---|---|---|
| 0 | -0.55 / -0.03 (-0.9%) | -0.46 (-0.9%) |
| 15 (Sonnet 5.5 measured, S13) | 0.91 / 0.05 (1.5%) | 1.73 (3.2%) |
| 32 (central, S5 E1) | 2.57 / 0.13 (4.0%) | 4.22 (7.3%) |
| 64 | 5.69 / 0.28 (8.5%) | 8.90 (14.3%) |
| 100 | 9.20 / 0.46 (13.1%) | 14.16 (21.0%) |
| 157 (upper, S5 mini-high) | 14.76 / 0.74 (19.4%) | 22.50 (29.6%) |
| 300 | 28.70 / 1.43 (31.8%) | 43.41 (44.5%) |

Break-even thinking length (thinking-off saves as much as the pruner): against a pruner that removes 50% / 70% / 80%, about 164 / 234 / 268 tokens on Sonnet 5.5 and Haiku 5.5, about 92 / 131 / 151 tokens on gpt-5.4 (output priced $15 per MTok). So thinking-off beats the pruner only where the model thinks for roughly 90 to 270 tokens or more per call, depending on the pruner's removal.

#### 3b. Student coverage and pruner removal (Sonnet 5.5, $/day saved)

| student coverage | central 32 | upper 157 | | pruner removes | central 32 | upper 157 |
|---|---|---|---|---|---|---|
| 25% | 7.57 | 10.61 | | 30% | 8.70 | 8.39 |
| 40% | 12.11 | 16.98 | | 50% | 15.47 | 15.16 |
| 50% | 15.13 | 21.23 | | 70% | 22.23 | 21.92 |
| 60% | 18.16 | 25.47 | | 80% | 25.62 | 25.30 |
| 75% | 22.70 | 31.84 | | | | |
| 90% | 27.24 | 38.21 | | | | |

**Break-even coverage (student saves as much as the pruner), Sonnet 5.5 and Haiku 5.5:** central 51% / 73% / 85% against a pruner removing 50 / 70 / 80%; upper 36% / 52% / 60%. gpt-5.4: central 48% / 69% / 79%; upper 27% / 40% / 46%. Reading: the S1/S8 coverage of 59% (770 rows) beats the S6b-anchored pruner and loses to the mission-range pruner; 89% (3,080 rows) beats both. At the central thinking length the student and pruner are a tie within the plausible range; v1's "$31.8 against $21.9" came entirely from the 157-token thinking baseline that the student avoids.

#### 3c. Does the ordering of the missions hold? (Sonnet 5.5, $/day saved)

| thinking | coverage | mission 1 alone | mission 2 alone | 1+2 | pruner 50% alone | pruner 70% alone | pruner 50% on top of 1+2 | pruner 70% on top of 1+2 | pruner alone beats 1+2? |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 50% | 2.57 | 15.13 | 16.46 | 15.47 | 22.23 | 11.81 | 16.97 | 70%: yes |
| 32 | 75% | 2.57 | 22.70 | 23.40 | 15.47 | 22.23 | 9.83 | 14.14 | no |
| 32 | 90% | 2.57 | 27.24 | 27.57 | 15.47 | 22.23 | 8.65 | 12.45 | no |
| 157 | 50% | 14.76 | 21.23 | 28.80 | 15.16 | 21.92 | 11.81 | 16.97 | no |
| 157 | 75% | 14.76 | 31.84 | 35.82 | 15.16 | 21.92 | 9.83 | 14.14 | no |
| 157 | 90% | 14.76 | 38.21 | 40.04 | 15.16 | 21.92 | 8.65 | 12.45 | no |

Mission 1 adds little over mission 2 alone at the central length ($0.3 to $1.3 a day more than mission 2 on Sonnet; on gpt-5.4 $0.5 to $2.2) and a lot at the upper one. Haiku 5.5 and gpt-5.4 give the same pattern with different scales (gpt-5.4: mission 1 alone $4.22 central, $22.50 upper). The pruner on top of 1+2 adds another $8.7 to $17.0 a day on Sonnet central, depending on coverage and removal, so the missions are not alternatives on dollars: the question is the order, and the dollars alone do not settle it.

#### 3d. Other assumptions that v1 fixed (Sonnet 5.5, central unless noted)

| assumption | values | effect |
|---|---|---|
| audit rate (1 in 20 = 5%) | 2% / 5% / 10% | student 75% saves 23.05 / 22.70 / 22.12; pruner 50% saves 16.34 / 15.47 / 14.02; pruner 70% saves 23.10 / 22.23 / 20.79; thinking off saves 2.95 / 2.57 / 1.94. The ordering does not change. |
| cache hit rate (sync) | 50% / 80% / 95% / 99% | cache-friendly layout saves 1.96 / 9.00 / 12.52 / 13.46; 77-label prefix cache saves 5.40 / 10.96 / 13.74 / 14.48. The 95% default is in the middle of the band for the demo and optimistic for a site that runs a few times an hour. |
| history size (S6b prompts average 1,989 tokens; the S16 flag prompt about 1,000) | x1 / x2.2 history | baseline 63.55 / 104.14; student 75% saves 22.70 / 37.54; pruner 50% saves 15.47 / 34.75; pruner 70% saves 22.23 / 49.63; thinking off 2.57 / 2.06. **With S6b-sized prompts the pruner overtakes the student at the mission range.** |
| vendor_lookup and match_invoice priced as two extra flag-sized LLM calls (upper bound) | adds $10.27/day (63.55 -> 73.82) | the student would also cover them (+$7.51), so student 75% saves $30.21 (41%) of $73.82; pruner and thinking are unchanged. |
| Gemini cache write = plain input price | | irrelevant on the demo (no prefix reaches 2,048); on a 4,000-token prefix Gemini's implicit cache saves $30.78 per day (sweep in `results_econ.json`), and a write surcharge would shave that. |

### 4. Does the fixed prefix change the answer? (prefix-size sweep, same workflow, Sonnet 5.5, central thinking, $/day saved)

The demo prefixes are tiny. If each step instead carried a fixed block (policy, labels, few-shot examples) of P tokens:

| fixed prefix P (tokens) | baseline $/day | caching saves | pruning 70% saves | pruning 50% saves | student 75% saves | caching + pruning 70% saves |
|---|---|---|---|---|---|---|
| 100 | 61.03 | 0.00 | 22.22 | 15.46 | 20.37 | 22.22 |
| 250 | 70.03 | 0.00 | 22.07 | 15.31 | 24.76 | 22.07 |
| 500 | 85.03 | 0.00 | 21.82 | 15.06 | 32.07 | 21.82 |
| 1,000 | 115.03 | 53.40 | 21.32 | 14.56 | 46.70 | 75.25 |
| 2,000 | 175.03 | 106.80 | 20.32 | 13.56 | 75.95 | 128.18 |
| 4,000 | 295.03 | 213.60 | 18.32 | 11.56 | 134.45 | 234.04 |
| 8,000 | 535.03 | 427.20 | 14.32 | 7.56 | 251.45 | 445.76 |

Caching switches on at the provider's minimum: **512 tokens on Claude (documented and bracketed by the probes), 2,048 or 4,096 on Gemini (Google's table; 3.5 Flash-Lite is not listed), 1,024 on OpenAI only for GPT-5.6 and later (unverified for the gpt-5.4 family: 0 of 9 cached calls at 1,270 tokens)**. Crossovers (smallest fixed prefix, 50-token grid, where caching saves at least as much as the other lever; central / upper thinking where they differ):

| provider (minimum used) | caching beats pruning 70% or 50% at | caching beats student 75% at |
|---|---|---|
| Sonnet 5.5 (512, documented) | 550 | 750 / 1,150 |
| Haiku 5.5 (512, documented) | 550 | 850 / 1,250 |
| gpt-5.4-mini (1,024, documented for 5.6+ only, unverified) | 1,050 | 1,050 / 1,100 |
| Gemini 3.5 Flash-Lite, 2,048 minimum | 2,050 | 2,050 |
| Gemini 3.5 Flash-Lite, 4,096 minimum | 4,100 | 4,100 |

The Claude figure of 550 is just the 512-token minimum plus the grid step: the moment caching switches on at all it already beats a pruner on this workflow, because it discounts 90-95% of a block that every call repeats. The OpenAI and Gemini figures are lower bounds set by the minimum, and the Gemini minimum is a choice between two published numbers, so they are not facts about those providers. Pruning savings shrink slowly as P grows because the 5% full-prompt audit calls get more expensive.

### 5. Headline: daily bill at 10,000 runs/day, per combination (USD/day, central 32 / upper 157)

Mission 1 (thinking off) does not exist for Sonnet 5.5 (S13), so Sonnet rows that include it are "if it could be switched off"; gpt-5.4 is the model here where it can.

| combination | sonnet-5.5 central | sonnet-5.5 upper | haiku-5.5 central | haiku-5.5 upper | gpt-5.4 central | gpt-5.4 upper |
|---|---|---|---|---|---|---|
| 0 baseline: no cache, flag thinking on, all real time | 63.55 | 76.05 | 3.18 | 3.80 | 57.35 | 76.10 |
| 1 + prefix caching | 63.55 | 76.05 | 3.18 | 3.80 | 57.35 | 76.10 |
| 2 + pruning 70% (no cache) | 41.32 | 54.13 | 2.07 | 2.71 | 38.63 | 57.85 |
| 2s pruning 50% (S6b gated, 15% budget) | 48.08 | 60.89 | 2.40 | 3.04 | 44.36 | 63.57 |
| 2f pruning 70% on flag_anomaly only | 52.34 | 65.15 | 2.62 | 3.26 | 47.92 | 67.14 |
| 2n pruning 70% on draft_note only (free text, S16 pending) | 52.52 | 65.02 | 2.63 | 3.25 | 48.07 | 66.82 |
| 1c + cache-friendly layout | 51.03 | 63.53 | 2.60 | 3.22 | 57.35 | 76.10 |
| 4 thinking off on flag_anomaly (mission 1) | 60.98 | 61.29 | 3.05 | 3.06 | 53.14 | 53.61 |
| 5 certified student 75% (mission 2) | 40.85 | 44.21 | 2.04 | 2.21 | 36.89 | 41.92 |
| 5a student 59% only (the 5%-budget certificate S1 supports at 770 rows) | 45.69 | 51.00 | 2.28 | 2.55 | 41.25 | 49.22 |
| 5b student 75% + pruning 50% | 31.09 | 34.53 | 1.55 | 1.73 | 28.68 | 33.84 |
| 5c student 75% + pruning 70% | 26.80 | 30.24 | 1.34 | 1.51 | 25.05 | 30.21 |
| 6 missions 1+2 (thinking off + student 75%) | 40.15 | 40.23 | 2.01 | 2.01 | 35.74 | 35.86 |
| 7c missions 1+2 + cache layout | 37.02 | 37.10 | 1.86 | 1.87 | 35.74 | 35.86 |
| 8 missions 1+2+5, pruning 70% | 26.00 | 26.08 | 1.30 | 1.30 | 23.82 | 23.93 |
| 8s missions 1+2+5, pruning 50% (S6b) | 30.32 | 30.39 | 1.52 | 1.52 | 27.47 | 27.58 |
| 8b missions 1+2, pruning 70% on flag_anomaly only | 37.03 | 37.11 | 1.85 | 1.86 | 33.10 | 33.22 |
| 0b baseline if categorize carried the 77-label list | 74.64 | 87.14 | 3.73 | 4.35 | 64.92 | 83.67 |
| 1b same, with prefix caching | 60.90 | 73.40 | 3.08 | 3.70 | 64.92 | 83.67 |

**Should mission 5 (pruner) be built before missions 1 and 2? It depends on which half of mission 1, and the dollar numbers do not decide the student-versus-pruner order.** On Sonnet 5.5 (central), pruner first saves $22.2/day (-35%) at the 70% removal the mission assumes and $15.5 (-24%) at what S6b supports; mission 2 alone saves $22.7 (-36%) at 75% coverage or $17.9 (-28%) at the 59% S1 shows for 770 rows; mission 1 (thinking) alone saves $2.6 (-4%), and Sonnet cannot do it at all. Missions 1+2 together save $23.4 (-37%, v1: -47%), and the pruner on top of that still saves another $14.1 (to $26.00, -59% in total). Of that last increment, $3.1 is flag_anomaly at 70% (a yes/no step; S6b tested a HotpotQA yes/no step, at a 15% budget only) and $11.0 is draft_note, a free-text step that S6 found unusable with a noisy teacher and that S16 has not finished; at the S6b-anchored 50% the whole increment is $9.8. So: (i) the thinking-off half of mission 1 should not come first on cost grounds unless the step really thinks for about 90 to 270 tokens or more; (ii) mission 2 and mission 5 are close on dollars and stack (together -51% to -58%), the student has the cleaner certificate (5% budget, mechanism tested on the right kind of task: classification into labels), and the pruner is better if prompts carry much more history than this demo; (iii) the pruner's draft_note half rests on S16. One-time cost to start: about $1.6 in Sonnet teacher labels per 1,000 rows across the two typed sites for the student (Batch, thinking on for flag; $2.2 at the upper thinking length; S1/S8 want 770 to 9,993 rows for 59-94% coverage, so roughly $1.2 to $16, upper $1.7 to $22), and about $2.2 ($3.7 upper) of calibration calls for the pruner. At 10k runs/day on Sonnet either pays back in well under a day. On Haiku 5.5 the labels cost about $0.08 per 1,000 rows against a student saving of $1.13/day (central), so payback is also days, but the whole workflow is only $3.18/day and the absolute savings are small until volume or prompt size grows.

### 6. Speculation break-even (mission 3), recomputed exactly

Rule: speculating the next step pays when the hit rate h times the time it hides exceeds the cost of the predictor and the verify. One model is used for every table below (`spec_gain` in `run.py`): the next step of length L2 can overlap at most min(L, L2) of a tool of length L. **Parallel** (default): the predictor runs alongside the previous step, only delaying the launch by its latency p, so a hit hides min(L, L2 - p) and every call pays the verify v; a miss costs no latency. **Serial** (worst case): the predictor blocks the critical path, a hit hides min(L, L2) and every call pays p + v. In v1 the first table used the serial rule and the savings table used the parallel rule; the two are now the same function.

TraceLab tool latency, every value (no resampling):

| subset | n | mean | sd | max | p50 | p90 | p99 | share >= 1 s | share of time in calls >= 1 s |
|---|---|---|---|---|---|---|---|---|---|
| pooled Claude Code + Codex | 743,050 | 15.61 s | 789 s | 440,698 s (5.1 days) | 0.218 s | 10.05 s | 151.6 s | 33.8% | 99.3% |
| Claude-provider rows only | 304,290 | 28.74 s | 1,220 s | 440,698 s | 0.115 s | 10.17 s | 240.1 s | 28.7% | 99.7% |
| pooled, capped at 1 hour | 743,050 | 9.55 s | 90.6 s | 3,600 s | 0.218 s | 10.05 s | 151.6 s | 33.8% | 98.8% |

The mean is set by a few outliers: a single 5-day call adds 0.6 s to the pooled mean. v1's 200,000 resamples gave a mean of 19.36 s (other seeds 15.4-21.2), which is why its percentages were 20-25% too small. The research file's Claude-only figures (median 0.3 s, p90 13.6 s, p99 3.6 min, mean 16.8 s) are not reproduced by filtering on `provider == "claude"` (0.115 s, 10.2 s, 240 s, 28.7 s); the cause is unknown, so the pooled and Claude-only results are both reported and the cap row shows how much of everything is the tail.

Predictor latency, batch 1, this laptop CPU (median of repeated calls): n-gram/trajectory-cache lookup 0.0003 ms, TF-IDF + logistic regression (77-class) 0.93 ms, 12-feature gradient boosting 11.5 ms, bge-small embedding on CPU 26.4 ms, JSON compare for verification 0.0076 ms, an LLM-written prediction 390 ms (qwen3-32b, S13 p50) or 828 ms (Haiku 5.5, S13 p50). Hit rate: S14 precision of launched speculations (read-only top-1 call correct), 0.19, 0.20, 0.47, 0.48, 0.68, 0.73, 0.74, 0.79 across cold (X) and warm (W) regimes, retail and airline, two agents (S14 `tables_E.md`); the grid is 0.19 (cold airline), 0.47 (cold retail), 0.73 (warm retail), 0.79 (best warm).

**Break-even tool latency L\*, against a Sonnet-length next step (1.15 s), and the share of pooled TraceLab calls above it.** Parallel L\* = v/h; serial L\* = (p + v)/h. "x" means the predictor cannot pay back (L\* exceeds the next step).

| predictor | overhead p | parallel L\* (any hit 0.19-0.79) | serial h = 0.19: L\* / calls above | serial h = 0.47 | serial h = 0.73 |
|---|---|---|---|---|---|
| n-gram + trajectory cache | 0.0003 ms | 0.01-0.04 ms, 98.4% of calls | 0.04 ms / 98.4% | 0.02 ms / 98.4% | 0.01 ms / 98.4% |
| TF-IDF + LR | 0.93 ms | 0.01-0.04 ms, 98.4% | 4.9 ms / 96.9% | 2.0 ms / 97.9% | 1.3 ms / 97.9% |
| gradient boosting (12 features) | 11.5 ms | 0.01-0.04 ms, 98.4% | 60 ms / 77.0% | 24 ms / 89.5% | 16 ms / 92.3% |
| bge-small embed (+LR) | 26.4 ms | 0.01-0.04 ms, 98.4% | 139 ms / 58.4% | 56 ms / 78.6% | 36 ms / 85.6% |
| qwen3-32b LLM prediction | 390 ms | 0.01-0.04 ms, 98.4% (window 0.76 s) | x | 830 ms / 35.5% | 534 ms / 39.2% |
| Haiku 5.5 LLM prediction | 828 ms | 0.01-0.04 ms, 98.4% (window 0.32 s) | x | x | 1.13 s / 32.1% |

The old "a gradient-boosting predictor clears the bar above about 60 ms" is the serial worst case at the cold hit rate; in the parallel model it pays for any tool longer than about 0.04 ms, because verification is the only cost on the critical path. The inconsistency (60 ms against 0.04 ms) was between the two v1 tables, not a property of the predictor.

Hit rate needed to break even when the next step is a 0.83 s Haiku call (h\* = need / min(L, window)):

| predictor | mode | tool 30 ms | 100 ms | 300 ms | 1 s | 3 s |
|---|---|---|---|---|---|---|
| n-gram / cache lookup, TF-IDF + LR, gradient boosting, bge-small | parallel | below 0.001 | below 0.001 | below 0.001 | below 0.001 | below 0.001 |
| TF-IDF + LR | serial | 0.031 | 0.009 | 0.003 | 0.001 | 0.001 |
| gradient boosting | serial | 0.38 | 0.11 | 0.04 | 0.014 | 0.014 |
| bge-small + LR | serial | 0.88 | 0.26 | 0.09 | 0.03 | 0.03 |
| qwen3-32b LLM prediction (390 ms) | parallel | below 0.001 | below 0.001 | below 0.001 | below 0.001 | below 0.001 |
| qwen3-32b LLM prediction | serial | impossible | impossible | impossible | 0.47 | 0.47 |
| Haiku 5.5 LLM prediction (828 ms) | parallel | impossible (window 0) | impossible | impossible | impossible | impossible |
| Haiku 5.5 LLM prediction | serial | impossible | impossible | impossible | 1.00 | 1.00 |

**"An LLM-written prediction never helps latency" (v1) is withdrawn as stated.** It holds for an 828 ms Haiku predictor against a 0.83 s step. A faster LLM predictor run in parallel with a slower step does hide time: qwen3-32b at 390 ms against a Sonnet-length step saves 1.0% (hit 0.47) to 1.6% (hit 0.73) of pooled TraceLab time and 5 to 17% in the fixed-latency regime (hit 0.47, tools 0.3 to 3 s), against 1.4% and 2.2% for gradient boosting, at the price of one model call per speculation (about $0.00004 for qwen3-32b). Run serially it loses time at every setting where the tool is under about 0.8 s. The 390 ms figure is the latency of a classification call; a call that writes tool arguments will be slower.

**How much is actually saved (share of the sequential time L + L2; gradient boosting, parallel; the serial worst case is 0.05 to 0.07 points lower in the pooled rows and up to 1.3 points lower in the fixed rows with a 0.1 s tool).**

| setting | hit 0.19 | hit 0.47 | hit 0.73 | hit 0.79 | ceiling (hit 1.0, no overhead) |
|---|---|---|---|---|---|
| pooled, next step = Haiku 0.83 s | 0.45% | 1.12% | 1.73% | 1.88% | 2.40% |
| pooled, next step = Sonnet 1.15 s | 0.57% | 1.40% | 2.17% | 2.35% | 3.00% |
| pooled, next step = thinking step 2.1 s | 0.80% | 1.98% | 3.08% | 3.33% | 4.23% |
| Claude-only rows, Haiku / Sonnet / thinking | 0.21 / 0.26 / 0.40% | 0.51 / 0.65 / 1.00% | 0.79 / 1.00 / 1.55% | 0.85 / 1.09 / 1.68% | 1.09 / 1.39 / 2.13% |
| pooled capped at 1 h, Haiku / Sonnet / thinking | 0.71 / 0.89 / 1.22% | 1.77 / 2.19 / 3.01% | 2.75 / 3.41 / 4.67% | 2.97 / 3.69 / 5.06% | 3.80 / 4.70 / 6.42% |
| median call (0.22 s tool), pooled, Haiku / Sonnet / thinking | 4.0 / 3.0 / 1.8% | 9.8 / 7.5 / 4.4% | 15.2 / 11.6 / 6.8% | 16.5 / 12.6 / 7.3% | |
| fixed tool 0.1 s, next = Haiku 0.83 s | 2.0% | 5.1% | 7.9% | 8.5% | |
| fixed tool 0.3 s, next = Haiku 0.83 s | 5.1% | 12.5% | 19.4% | 21.0% | |
| fixed tool 1 s, next = Sonnet 1.15 s | 8.8% | 21.9% | 34.0% | 36.8% | |
| fixed tool 3 s, next = thinking 2.1 s | 7.8% | 19.4% | 30.2% | 32.6% | |

In a coding-agent workload the mean time lives in a tail of tools that last tens of seconds to days, so hiding a 1 to 2 second LLM step saves 0.5 to 3.3% of total time at S14's hit rates (pooled, exact), 0.2 to 1.7% on the Claude-provider rows, and up to 5.1% if latencies are capped at an hour; even a perfect predictor is capped at 1.1 to 6.4%. v1 said 1.9-3.5% (ceiling) and 0.9-1.6% (hit 0.47); the exact values are 2.4-4.2% and 1.1-2.0%. The qualitative conclusion, a few percent, stands. Speculation pays in a regime where tool and next step are comparable: with tools of 0.3 to 3 s and LLM steps of 0.8 to 2.1 s the saving is 5.8 to 21.9% at hit 0.47 and 9 to 34% at 0.73 (2.3 to 8.8% at the cold 0.19), in line with S4's synthetic 19-20%. v1's "6 to 30%" was a sub-range of this, with 5.1% (tool 0.1 s) at its low end. Money: speculating a read-only tool call costs nothing but compute; speculating an LLM step wastes one call per miss, about (1 - h) x $0.0025 for a Sonnet 5.5 flag_anomaly call at the central thinking length ($0.0038 at 157 tokens), so at h = 0.47 each speculation costs about $0.0013 ($0.0020) in expectation and buys 0.3 to 0.5 s.

## Interpretation for the mission

1. **Caching and pruning address different tokens.** Caching discounts the fixed prefix (up to 90% on Haiku and OpenAI, 95% on Sonnet 5.5); pruning removes variable history that no cache can hold because it differs per record. They stack (Sonnet, P = 1,000: caching saves 53.4, caching plus pruning 75.3). So caching does not make the pruner moot.
2. **On the demo workflow caching is worth exactly $0, and no single lever dominates.** The prefixes (46 to 245 tokens) are under the Claude minimum and under every other provider minimum that is published. The money sits in history (52 to 67% of the flag and note cost) and in output (the note text; thinking tokens only if they are long). At the thinking length S5 measured on the expensive reference (32 tokens), thinking-off is the weakest lever on every teacher (Sonnet 4.0%, gpt-5.4 7.3%, Gemini 9.4% but not switchable) and sits below the pruner (24-40% on Sonnet) and the student (29-43%); it overtakes the pruner only if the step thinks for roughly 90 to 270 tokens. Mission 2 (student) and mission 5 (pruner) are within noise of each other at the central thinking length (break-even coverage 51-85%), and the pruner is ahead when prompts carry more history than this demo. v1's statement that missions 1 and 2 "rank above the pruner on every teacher" was false for the Claude teachers (and for mission 1 on every teacher at the central length) and is withdrawn.
3. **The evidence for the levers is uneven, and v1 was sceptical only of the pruner.** The student has a 5%-budget certificate on a 77-way task; the pruner has a 15%-budget certificate on one yes/no task and nothing at 5% or 10%; thinking-off has one measurement and it failed. None has been shown on flag_anomaly or draft_note. The dollar ranking above is "if each lever delivers its borrowed number"; the evidence ranking is student first, pruner second (and only if a 15% disagreement budget is acceptable), thinking-off last. The next experiments that would change the order are a student on a yes/no bookkeeping step (does coverage reach the 51-85% break-even?) and thinking-off on that same step.
4. **Make caching a runtime default, not a mission.** Dreaming's proxy and wrappers should put fixed content first and set `cache_control` for Anthropic (OpenAI and Gemini cache automatically). The moment a customer prompt carries about 550 tokens of fixed instructions or examples on Claude, caching beats a 70% pruner (documented minimum 512). For OpenAI and Gemini the threshold is at least the provider's minimum, and neither minimum is established for the models used here: Google lists 2,048 to 4,096 and omits 3.5 Flash-Lite, OpenAI documents 1,024 only for GPT-5.6 and later and our one test on gpt-5.4-nano saw no hit at 1,270 tokens. Our own 77-label categorize (771 tokens) qualifies on Claude: -81% at a 95% hit rate (modeled), and the 12.5% measured on a cache read matches the documented 0.1x. Order inside the prompt matters: a pruned history placed before the instructions would break the cache.
5. **Two cheap structural wins to note.** (a) A cache-friendly layout, [fixed prefix][record + history][step instruction last], lets flag_anomaly write the cache and draft_note read it within seconds: -19.7% on Sonnet central (-16.5% upper) when both calls reach the teacher, but it depends on the hit rate (1.96 at 50%, 12.52 at 95% $/day) and shrinks once the student answers 75% of flags (no writer); changing prompt order needs re-certification. (b) Batch halves every price but only fits async work: Dreaming's own audit/shadow stream is that work, yet it is only about 1% of the bill at one call in twenty. If a customer's whole job tolerates a 24 hour window, Batch alone is the largest single lever (-50%).
6. **Speculation is a latency lever, not a token lever.** It is cheap to break even on for anything cheaper than a model call, but the payoff is bounded by the next step's length and shows up as a few percent of time in a heavy-tailed coding-agent trace and 6 to 22% when tool and step times are comparable. That supports the earlier retargeting: speculate read-only tools whose latency is above roughly 100 ms; an LLM-written predictor is only worth considering if it is several times faster than the step it predicts and runs in parallel.
7. **Scale check.** At 10,000 runs/day the whole demo is $63.55/day on Sonnet 5.5 (central; $76.05 at the upper thinking length) and $3.18 on Haiku 5.5, so even the best combination saves about $37/day on Sonnet and $1.90 on Haiku. The levers are about relative savings (-51% to -58% for the student plus the pruner) that grow with volume and prompt size; the tier drop itself (-70 to -95%) dwarfs them, which makes the mission 1 tier-router evidence (S5) the most valuable one to firm up.

## Caveats

- Costs are linear in tokens, so means over the logged prompts give exact totals; there is no per-call simulation. The speculation numbers are now exact over the raw latency file; they carry no sampling error, but the heavy tail makes the mean-based percentages depend on a handful of outliers (see the capped row).
- Token profiles come from 300 synthetic S16 prompts and a 10-pair Claude token probe, frozen in `tokens.json` (the S16 response cache was still growing when the profile was made; n_notes was 2,056). Nothing here is bookkeeping data.
- **Thinking length** is the largest source of change in this revision: 32 and 157 are two measured points on two models on a 77-label step, applied to every teacher; Claude and Gemini measured 15 and 0 on that step, and nothing is measured on a flag_anomaly step. Thinking-off is shown as "if certified" and its only measurement failed.
- **Student coverage** 50-90% is borrowed from Banking77 (S1/S8), not from bookkeeping data; the student is free at inference and its build cost is amortized away. **Pruner** removal is the mission range (60-80%) or S6b's 50% effective (HotpotQA yes/no, n = 80, the selected best of 15 configurations on the same rows, no multiplicity correction); S16 (draft_note) had not finished.
- Cache hit rates are assumptions (0.95 sync, 0.65 batch; banded 0.5-0.99 in 3d). 10k runs/day is about 7 runs a minute per call site, which keeps a 5-minute cache warm, so 0.95 is conservative for the demo and optimistic for a site that runs a few times an hour. The 1-hour cache (2x write) and Gemini explicit-cache storage are not modeled. The Gemini cache-write price is set to the plain input price (implicit caching has no write fee); the 5% audit rate is banded 2-10%.
- OpenAI caching was not reproduced in the probe (0 of 9 calls cached at 1,270 tokens on gpt-5.4-nano through OpenRouter), so the OpenAI rows in the sweep are the documented behavior for GPT-5.6 and later, not a measured one for gpt-5.4. Gemini 3.5 Flash-Lite's minimum cacheable length is not published; the model uses the lowest listed figure (2,048) and shows 4,096. Cache read is 0.1x from Google's price table; the OpenRouter page's 0.25x looks stale.
- The Claude minimum of 512 tokens is the documented value; the probes bracket it (254 uncached, 764 cached) but do not locate it. The probes breached the $0 cap by $0.0036 and carry no weight in any conclusion.
- Prices are OpenRouter's list prices; OpenRouter's credit purchase fee and any provider-specific routing premium are not modeled. The vendor pages agree with the listing on every number used (the pages round OpenAI's batch cached input to $0.13 vs the listing's $0.125, and Gemini's batch cache price to $0.02 vs $0.015).
- TraceLab is pooled over Claude Code and Codex and also reported for the rows with provider "claude"; the research file's own Claude-only statistics do not match either (see section 6). Its tools are coding tools, much slower and heavier-tailed than bookkeeping API calls; the fixed-latency rows are the closer analogue and are labeled as a regime, not a measurement. Hit rates are S14's tau-bench replay precision per launched speculation; the share of positions where a speculation is launched (59 to 81%) is not multiplied in, so the savings rows are per launched speculation. Wasted speculative work is treated as free in latency. The predictor latencies were measured on one laptop CPU at batch 1.
- vendor_lookup and match_invoice are treated as tools (missions.md line 48; match_invoice is never defined) and appear only in the upper-bound sensitivity.
- Layout and pruning change prompt text and therefore the teacher's behavior; both need their own certificates before the savings count.

## Time and cost

- API spend: $0.0036 total (38 probe calls with max_tokens = 1, all in the earlier attempt, cached in `responses.jsonl`; 0 new calls in the first finishing session and 0 in this revision). This exceeds the smoke's $0 cap and is disclosed above. Pod time: none, $0. Logged in `smokes/COSTS.md` (unchanged; nothing new was spent).
- Local compute in this revision: `econ` 2 s (both thinking scenarios), `sens` 2 s, `spec` 4 s on the exact 743,050-value arrays (predictor timings reused), `trace` 14.4 s once at below-normal priority to extract the Claude-only array. Web fetches: 2 in this revision (Google caching page, OpenAI prompt-caching guide); 12 WebFetch calls and 1 web search in the first session, all on 2026-10-17.
- Files: `run.py` (stages `tokens`, `trace`, `econ`, `sens`, `spec`), `probe.py` (paid probes, cached), `models.json` (OpenRouter listing), `probe_A.json`, `tokens.json`, `trace_stats.json`, `tracelab_latency_s.npy`, `tracelab_latency_claude_s.npy`, `results_econ.json`, `results_sens.json`, `results_spec.json`, `log_econ.txt`, `log_sens.txt`, `log_spec.txt`, `log_trace.txt`, and the v1 copies `RESULTS_v1.md`, `results_econ_v1.json`, `results_spec_v1.json`.
