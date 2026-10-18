# Tool prediction and speculative execution (research, 2026-10-17)

Scope: predicting tool/API returns, speculative launch, world-model/tool simulators, latency prediction, public tool-call logs, and the cheapest honest mock bookkeeping tool env. All claims carry a URL; "unverified" = not confirmed from a primary page in this session. Paper numbers are author-reported unless stated.

## Summary

- Precedent map for our three tool cases:
  - (1) Return = function of inputs: strongest precedent, and trivially cheap (exact-match cache or lookup table). Learned simulators (MirrorAPI, GTM) exist but are for APIs with no implementation; for our own deterministic mock tools a predictor is only interesting for latency, not content.
  - (2) Return depends on hidden state: real precedent is AOSpec (terminal output, 33.3% exact hit at 5 candidates) and TVCache (state-aware cache keyed on call history). Naive input-keyed caching is unsafe here.
  - (3) Call changes the world: nobody speculates the write itself. PASTE blocks side-effecting speculation; AOSpec runs it only inside copy-on-write sandbox forks and commits on exact state+action match. Our version: SQLite savepoint/rollback or a dry-run variant.
- Measured speculation hit rates are modest: 22-38% (tau-bench retail API calls), up to 55% (chess opponent move, 3 guesses), up to 46% (HotpotQA, top-3), 27.8% top-1 / 43.9% top-3 (PASTE), 33.3% exact on tool output text (AOSpec). End-to-end latency saved: ~20% (Speculative Actions), 11.8-32.5% (AOSpec), 43.5-48.5% (PASTE, tool-latency-heavy workloads).
- Speculation only pays when tools are slow relative to the predictor: in real coding-agent traces 17% of calls (>=1 s) hold 97% of tool time (AOSpec), and calls >1 min are 4.9% of calls but 92% of tool time (TraceLab, Claude Code). A bookkeeping agent with millisecond DB lookups has nothing to hide; the demo needs injected slow external tools (bank feed, ERP) and must say they are injected.
- Exact byte-identity verification is the standard acceptance rule (AOSpec, Speculative Actions "lossless"). That matches our "certify then fallback to the real call" design and our byte-replay constraint; no new theory needed.
- No public dataset has real bookkeeping tool returns plus latencies. Best proxies: TraceLab (real latencies, args/outputs redacted, CC BY 4.0), Exgentic/agent-llm-traces (span timestamps + tau2 retail/airline/telecom, CDLA-Permissive-2.0), Toucan-1.5M (real MCP returns, no latency, Apache-2.0). All download without login.
- Cheapest honest mock env: in-process Python tools over a seeded SQLite DB (tau2-bench retail/airline pattern, MIT) + stdlib `random.lognormvariate` latency on a virtual clock + Bernoulli error injection. Zero dollars, fully deterministic, hashable state. Lognormal parameters can be fit to TraceLab/Exgentic.
- Nemotron "tool simulators": only found NVIDIA's Nemotron-Agentic-v1 dataset whose tool responses are LLM-simulated (not real logs). The "SimuAgent" name from the brief did not match a world-model paper (see Unverified).

## Findings

### A. Speculative execution papers (with measured hit rates)

- **Speculative Actions** (Columbia; arXiv Oct 2025, ICLR 2026 oral). https://arxiv.org/abs/2510.04371 , HTML https://arxiv.org/html/2510.04371v1
  - Small/fast model guesses next action; kept only if it matches (lossless).
  - Chess (TextArena): 54.7% avg next-action accuracy with 3 guesses, 19.5% time saved. E-commerce (tau-bench retail): 22-38% of API calls correctly predicted, no e2e latency number. HotpotQA: up to 46% top-3 strict match, no latency number. OS tuning (lossy): p95 latency 37.93 ms vs 54.00 ms.
  - Scores predicted actions/calls, NOT tool-return content. Code (Apache-2.0, 39 stars): https://github.com/naimengye/speculative-action
- **PASTE** "Act While Thinking" (SJTU/MSR/Stevens; arXiv 2603.18897, Mar 2026). https://arxiv.org/html/2603.18897 ; MSR page https://www.microsoft.com/en-us/research/publication/act-while-thinking-accelerating-llm-agents-via-pattern-aware-speculative-tool-execution/
  - Mines tool-call sequence patterns from history; learned "argument mappers" (e.g. first URL of a search result) fill args; only fully instantiated predictions execute.
  - Up to 27.8% top-1, 43.9% top-3 recall, 93.8% "hit rate" (any of many launched speculations matched; not comparable to top-1). Cuts task time 48.5% in abstract (43.5% in another version), 1.8x tool throughput.
  - Overhead: 1-3 idle CPU cores, 250 MB RAM, predictor <100 ms. 602 of >20,000 speculative actions flagged side-effecting and blocked; final outputs unchanged vs baseline.
  - Eligible: side-effect-free, fully parameterized tools; side-effect tools only via dry-run/staging. No code link or software license found (paper is CC BY-NC-ND 4.0). Workloads: DeepResearchBench, SWE-bench, ScholarQA.
- **AOSpec** (arXiv 2608.00881, Aug 2026; CC BY 4.0). https://arxiv.org/abs/2608.00881 , HTML https://arxiv.org/html/2608.00881v1
  - The only paper here that predicts tool-return text. A fine-tuned Qwen3-0.6B observation model drafts 5 candidate outputs while the real tool runs; candidates ranked by P(output) x expected tool time ("Expected Value Decoding"). Continuation kept only if draft is byte-identical to real output.
  - Observation hit: top-1 29.1% fine-tuned (2.4% frozen); 32.9% at 5 candidates; 33.3% full-context exact; hides 27.1% of tool time. Trained on 1,921 Terminal-Bench trajectories, no manual labels.
  - E2E mean latency saved 11.8% (20 ms/token decode) to 32.5% (1 ms/token); p99 -42.8%. Transfers to SWE-bench Verified (18.9% at 1 ms) without retraining.
  - State handling: speculative actions run in copy-on-write filesystem forks; reuse only if action matches AND fork's origin state hash matches ("Joint Action-State Verification"). Non-filesystem effects ineligible.
  - Caveats: trace replay, not live; strict byte-identity (UUIDs/timestamps kill hits; canonicalization not implemented); long-range action prediction decays (23.7% at distance 1 to 7.7% at distance 10, width 32). No code link found.
- **Speculate While You Reason** (UCSB/LinkedIn; arXiv 2607.25816, Jul 2026). https://arxiv.org/abs/2607.25816 : one 4B agent also predicts its own next tool call; Hit@1 Qwen3-4B 44.1 to 61.2, Qwen3.5-4B 48.9 to 66.3 after joint RL. Datasets unnamed in abstract; no code link found.
- **Optimizing Agentic LM Inference via Speculative Tool Calls** (arXiv 2512.15834, Dec 2025). https://arxiv.org/abs/2512.15834 : "several hundred tokens/s" throughput gain, proposes a provider "tool cache" API. No other numbers readable; no code.
- Related, not read in depth: B-PASTE (beam-style branch speculation; found only via search summary), SPORK https://www.alphaxiv.org/abs/2607.03333 , Ghost Tool Calls (privacy) https://arxiv.org/pdf/2606.02483 .

### B. Stateful correctness of cached/predicted tool values

- **TVCache** (arXiv 2602.10986). https://arxiv.org/abs/2602.10986 : tree of observed tool-call sequences; reuse only when the FULL call history matches (guarantees same env state). Up to 70% hit rate, median tool-call time down up to 6.9x, no reward degradation (terminal, SQL, video tasks). A slide deck (search snippet only, unverified) says a stateless naive cache gives ~27% incorrect hits: https://icml.cc/media/icml-2026/Slides/61148.pdf . No code link found.
- Takeaway for case (2): key any cache/predictor on (tool, args, state_hash or call-history), not (tool, args).

### C. World models / tool simulators

- **MirrorAPI** (arXiv 2503.20527, Mar 2025). https://arxiv.org/abs/2503.20527 : Qwen2.5-7B-Instruct SFT on 95,272 SFT + 44,440 CoT request-response pairs from ~7,437 real RapidAPI APIs (up to 200 real calls per API). vs real responses, OOD successful calls: BLEU 35.6 / LLM-cosine 69.9 (GPT-4o: 13.4 / 63.1). Cache-finetuned variant BLEU-4 84.9 on StableToolBench cache. Stateful/side-effect APIs NOT addressed. Training: 8x A100-40GB, 5 epochs. No release link found.
- **GTM** Generalist Tool Model (arXiv 2512.04535, Dec 2025). https://arxiv.org/abs/2512.04535 : 1.5B and 0.5B (Qwen2.5-based) simulator over >20,000 tools. ~0.46-0.48 s/query on A800 vs 0.73-14.84 s for real search APIs; RL training 6.3x (search) and 11x (CUDA) faster than real tools. Pure-simulator RL degrades late; hybrid (30 sim steps then real) recovers to 0.41 vs 0.424 real. Quality scored by a Qwen2.5-72B judge, not byte match. No release links found.
- **Simia** (arXiv 2511.01824). https://arxiv.org/abs/2511.01824 : LLM-simulated environments for SFT/RL; claims beating GPT-4o and near o4-mini on tau2-bench. No numbers/links readable.
- **Qwen-AgentWorld** (arXiv 2606.24597, Jun 2026). https://arxiv.org/abs/2606.24597 : language world models 35B-A3B and 397B-A17B, >10M trajectories, 7 domains, AgentWorldBench. Code https://github.com/QwenLM/Qwen-AgentWorld (license/weights not confirmed). Too large for our budget except as reference.
- **DreamGym** (Meta et al.; arXiv 2511.03773, ICLR 2026). https://arxiv.org/abs/2511.03773 , HTML https://arxiv.org/html/2511.03773v2 : Llama-3.1-8B-Instruct "experience model" predicts next state+reward by reasoning; trained on only 3.6k (WebShop), 5.2k (ALFWorld), 4.8k (WebArena) real trajectories; 3B backbone reaches ~55% WebShop with 20k samples; competitive at 2k-10k transition samples. Training effort ~1/3 to 1/5 of real-env RL (no absolute figures). Purpose is training agents, not speculation; no code link found.
- **WebDreamer** (arXiv 2411.06559, TMLR 2025). https://arxiv.org/abs/2411.06559 , code https://github.com/OSU-NLP-Group/WebDreamer : LLM imagines outcome of each candidate action (natural-language, not exact). VisualWebArena 17.6% to 23.6%; Dreamer-7B https://huggingface.co/osunlp/Dreamer-7B (image obs only). Semantic prediction, not byte-exact; 4-5x cheaper than tree search. Repo license not seen.
- **Trace2Env** (arXiv 2610.06100) rebuilds stateful worlds from traces with persistent episodic state: only a secondary summary seen (https://hyper.ai/en/papers/2610.06100), unverified.
- **Gecko** (arXiv 2602.19218) rules+LLM tool-response simulator for BFCLv3/tau2: https://arxiv.org/pdf/2602.19218 (not read).
- **Mock Worlds, Real Skills** (arXiv 2601.22511): flags that non-reproducible simulated responses give different observations for the same action: https://arxiv.org/pdf/2601.22511 (search summary only). Relevant to our byte-replay requirement.

### D. Environments with real stateful tools

- **tau2-bench / tau-bench** (Sierra; MIT). https://github.com/sierra-research/tau2-bench : domains mock, airline, retail, telecom, banking_knowledge. Each domain = `db.json`/`db.toml` + `tools.py` (ToolKitBase subclass) + policy + tasks; adding a domain needs data_model.py, tools.py, environment.py (https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/domains/README.md). Original v1: retail 500 users/50 products/1,000 orders, 7 write + 8 read tools (secondary: https://alphaxiv.org/benchmarks/sierra/tau-bench). User is an LLM (costs tokens); tool calls are Python. Latency/error injection: not present (unverified in code). `banking_knowledge` domain is a candidate bookkeeping-adjacent domain (contents unchecked).
- **AppWorld** (Stony Brook; Apache-2.0). https://github.com/StonyBrookNLP/appworld : 9 apps, 457 APIs, 100+ tables, SQLite base DBs, state-based eval, in-process FastAPI TestClient (no server), `save_state/load_state`, task load 4-5 s first / <0.5 s later. `appworld download data` with no login step described; app code ships as encrypted bundles. Only real "call changes the world" env with checkpointing found.
- **StableToolBench** https://github.com/zhichengg/StableToolBench : API response cache + GPT-4-Turbo simulator for dead RapidAPI endpoints; HF license not seen.
- **ToolSandbox** (Apple, arXiv 2408.04682) stateful tool-use benchmark with implicit state deps: https://arxiv.org/pdf/2408.04682v1 (not read).

### E. Latency prediction and measured latency distributions

- No learned tool-latency predictor paper found. Closest: JAUNT uses EWMA of recent per-tool end-to-end latency (https://arxiv.org/pdf/2510.18550); TraceLab lists "semantic-aware tool-latency prediction" as future work and notes tool type alone leaves large within-type variance (https://arxiv.org/abs/2606.30560). So latency prediction as a learned task has no direct benchmark; our baseline should be per-(tool, arg-features) quantile regression vs EWMA.
- **TraceLab** real numbers (~4,300 sessions; 432,510 tool calls; Claude Code + Codex; https://arxiv.org/html/2606.30560v2): mean tool call 16.8 s; Claude median 0.3 s, p90 13.6 s, p99 3.6 min; per-step tool time median 0.1 s, p90 10.0 s, p99 3.2 min. Tools = 59.3% of per-step time, LLM generation 40.7%. Measured e2e latency exceeds runner-reported internal latency by ~18.6% (Codex): train on wall-clock, not internal.
- AOSpec: ~17% of calls >=1 s hold 97% of tool time (https://arxiv.org/html/2608.00881v1).
- GTM Table: measured web-search API latency 0.73-14.84 s, Jina avg 0.923 s (https://arxiv.org/html/2512.04535v1).
- WS-DREAM (339 users x 5,825 web services RTT matrices, mean ~0.91 s, response time 0-20 s) per https://arxiv.org/pdf/2006.01287 ; it is the standard QoS-prediction benchmark (MAE/RMSE at 5-20% density) but is 2009-vintage SOAP/WS data. Download source not confirmed (my guessed GitHub URL returned 404; IEEE DataPort page https://ieee-dataport.org/documents/ws-dream likely needs a free account, unverified).

### F. Public datasets of tool calls WITH returns (no login unless stated)

| Dataset | Real returns? | Latency? | Notes |
|---|---|---|---|
| TraceLab coding trace | No (outputs/args redacted, only `result_chars`) | Yes: `emitted_at`, `result_at`, `tool_wall_latency_ms`, `tool_internal_latency_ms`, `is_error` | 432,510 tool records, 43 users; GitHub release v0.0.1 (jsonl.gz + duckdb); data CC BY 4.0. https://github.com/uw-syfi/TraceLab |
| Exgentic/agent-llm-traces | Tool calls with results claimed; result field not confirmed in schema | Derive from span start/end; OTel status codes | 1,781 traces, ~984 MB; SWE-bench 391, AppWorld 406, BrowseComp+ 133, tau2 airline 196 / retail 469 / telecom 186; CDLA-Permissive-2.0. https://huggingface.co/datasets/Exgentic/agent-llm-traces |
| Toucan-1.5M | Yes, real MCP server executions (495 servers, >2,000 tools; many flaky servers; responses reflect Jun-Sep 2025) | None | 1,646,546 rows, 21.8 GB parquet, Apache-2.0. https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M |
| Nemotron-Agentic-v1 | No: tool env simulated by LLMs (Qwen3-235B, Qwen3-32B, GPT-OSS-120B) | None | 335,122 samples (316,094 tool_calling + 19,028 interactive_agent), ~5.5 GB, CC BY 4.0 + Apache-2.0 for Glaive part; viewer errors on schema. https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1 |
| StableToolBench cache / MirrorAPI data | Real RapidAPI responses (cached, 2024 vintage) | None | License unverified |
| Glaive function-calling v2 | Synthetic | None | Apache-2.0 (per Nemotron card) |

Kaggle: none required for anything above. No public dataset of real accounting/bank/ERP API traces found.

## Reusable components

| name | url | license | why |
|---|---|---|---|
| tau2-bench domain pattern (db.json + tools.py) | https://github.com/sierra-research/tau2-bench | MIT | Template for a bookkeeping domain: JSON DB + Python tools + tasks; copy the layout, skip the LLM user sim |
| AppWorld | https://github.com/StonyBrookNLP/appworld | Apache-2.0 (app code encrypted bundles) | Only real stateful multi-app API env with SQLite + save/load state; use for case (3) validation |
| speculative-action | https://github.com/naimengye/speculative-action | Apache-2.0 | Reference lossless speculation loop on tau-bench retail / HotpotQA |
| TraceLab trace + pipeline | https://github.com/uw-syfi/TraceLab | Apache-2.0 code, CC BY 4.0 data | Real tool-latency distributions (432k calls) to fit lognormal/mixture parameters |
| Exgentic/agent-llm-traces | https://huggingface.co/datasets/Exgentic/agent-llm-traces | CDLA-Permissive-2.0 | tau2 retail/airline spans with timestamps and error status |
| Toucan-1.5M | https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M | Apache-2.0 | Real MCP tool returns for return-prediction smokes |
| Qwen3-0.6B (AOSpec observation model recipe) | https://arxiv.org/html/2608.00881v1 | CC BY 4.0 (paper); model license unchecked | Recipe: NLL on (history, action) to observation from 1.9k trajectories; fits CPU/RunPod budget |
| Python stdlib `random.lognormvariate`, `sqlite3` | https://docs.python.org/3/library/random.html | PSF | Latency + state for the mock env with no dependency |
| WireMock lognormal delay + faults (only if a real HTTP boundary is needed) | https://wiremock.org/docs/simulating-faults/ | Apache-2.0 (from memory, page did not state; unverified) | `delayDistribution: lognormal {median, sigma, maxValue}`; CONNECTION_RESET fault reportedly hangs on Windows |
| Toxiproxy | https://github.com/Shopify/toxiproxy | MIT (from memory, unverified) | TCP-level latency/timeout/reset; overkill for in-process tools |

## Numbers

| Item | Value | Source (date) |
|---|---|---|
| Speculative Actions chess hit / time saved | 54.7% / 19.5% (3 guesses) | arXiv 2510.04371 (Oct 2025) |
| Speculative Actions tau-bench retail | 22-38% calls predicted | same |
| Speculative Actions HotpotQA | up to 46% top-3 | same |
| PASTE top-1 / top-3 / "hit" / time cut | 27.8% / 43.9% / 93.8% / 43.5-48.5% | arXiv 2603.18897 (Mar 2026) |
| PASTE overhead | 1-3 idle cores, 250 MB, <100 ms | same |
| AOSpec exact obs hit | 29.1% top-1, 32.9% at 5 cand., 33.3% full-context | arXiv 2608.00881 (Aug 2026) |
| AOSpec tool time hidden / e2e saving | 27.1% / 11.8-32.5% mean, p99 -42.8% | same |
| AOSpec hardware | drafters on 2x H200; tools on Azure D4s_v5 | same |
| Self-speculating 4B agent Hit@1 | 44.1 to 61.2 (Qwen3-4B) | arXiv 2607.25816 (Jul 2026) |
| TVCache | hit rate up to 70%, 3.4-6.9x median speedup | arXiv 2602.10986; slides |
| Tool latency, Claude Code | median 0.3 s, p90 13.6 s, p99 3.6 min | arXiv 2606.30560 (Jun 2026) |
| Latency concentration | 17% of calls >=1 s = 97% tool time; >1 min = 4.9% calls, 92% time | AOSpec; TraceLab |
| Real web search API latency | 0.73-14.84 s; Jina 0.923 s mean | arXiv 2512.04535 (Dec 2025) |
| GTM 1.5B simulator latency | 0.46-0.48 s/query on A800 | same |
| MirrorAPI training | 8x A100-40GB, 5 epochs, ~140k samples | arXiv 2503.20527 (Mar 2025) |
| DreamGym experience model | Llama-3.1-8B; 3.6k-5.2k real trajectories | arXiv 2511.03773 |
| Mock env cost | $0 (stdlib + SQLite, CPU) | this plan |
| Fine-tuning a 0.6B observation model | no cost figure in paper; my estimate is a few GPU-hours on a RunPod 24 GB card (unverified; RunPod prices not checked in this topic) | estimate |

## Blockers / plan changes

1. **Speculation has nothing to hide in a ms-latency bookkeeping env.** Median real tool call is 0.1-0.3 s and speculation value sits in the >=1 s tail. Plan change: define the demo's slow tools explicitly (bank-feed fetch, OCR/receipt lookup, ERP posting) with injected lognormal latency, and report results as "under injected latency fit to TraceLab", not as measured production latency. Always report the break-even: predictor latency + verify cost < tool latency x hit rate.
2. **Case (2) and (3) need a state key.** Add `state_hash` (SQLite content hash or write-log hash) to every logged tool call and make predictors/caches conditional on it (TVCache, AOSpec JASV). Without it, replayed or speculated returns can silently be wrong.
3. **Case (3) writes: do not speculate; shadow only.** Precedent (PASTE, AOSpec) either blocks or sandbox-forks. For us: execute the write inside a SQLite SAVEPOINT on a copy, commit only on verified match (or just predict "will this write succeed / return value" and never launch it). AppWorld `save_state/load_state` is the reference.
4. **Byte-identity vs real tools.** AOSpec hits fall with nondeterministic fields (UUIDs, timestamps). Our mock tools must emit deterministic ids (seeded counter) or the predictor is scored on a canonicalized form; decide this before logging. This also serves the byte-for-byte replay requirement.
5. **Realism of mock latency/errors is only as good as the fit.** TraceLab covers coding tools (bash/read/edit), not finance APIs; the only external-API numbers found are search APIs (0.73-14.84 s) and old WS-DREAM data. State this limitation in any claim.
6. **Do not use a learned simulator as ground truth.** MirrorAPI/GTM/Nemotron-style simulated returns are LLM output, not reality; Nemotron-Agentic-v1's tool responses are LLM-simulated. They are fine for training/ablation, not for certifying a predictor against "real".
7. **Hit-rate expectations.** Plan for 20-35% exact next-call/return prediction on generic workloads and much higher only on narrow typed steps (our vendor_lookup/categorize have tiny label/arg spaces, so higher is plausible but unmeasured). Don't promise PASTE's 93.8% (different metric).

## Cheapest honest bookkeeping tool environment (recipe)

1. State: one SQLite file seeded from `random.Random(seed)` + Faker (vendors, accounts, transactions, invoices); tools are plain Python functions (`vendor_lookup`, `fetch_bank_feed`, `match_invoice`, `post_entry`, `flag_anomaly_inputs`) in the tau2 `db + tools.py` layout (MIT).
2. Determinism: ids from a seeded counter; time from a virtual clock; hash the DB after every call (`state_hash`) and log (tool, args, state_hash_before, result, state_hash_after, latency_ms, error).
3. Latency: virtual clock, not `sleep`. Draw `lognormvariate(mu, sigma)` per tool with a mixture for the tail (e.g. 90% fast mode, 10% slow mode), parameters fit to TraceLab `tool_wall_latency_ms` quantiles; add an arg-dependent term (rows scanned, date range) so latency is actually predictable from inputs, otherwise a latency predictor has nothing to learn. Flag this dependence as an assumption.
4. Errors: seeded Bernoulli per tool for timeout, 429 with retry-after, 5xx, and malformed payload; correlate with a "bad hour" regime so errors are also non-trivially predictable. WireMock has equivalent fault types if a real HTTP boundary is needed, but the CONNECTION_RESET fault is documented to hang on Windows.
5. Validation on real stateful data at near-zero cost: run the same speculation harness on tau2-bench retail (MIT) and AppWorld (Apache-2.0) tool calls, plus Exgentic tau2 traces for real timing; this shows the method is not overfit to our synthetic mock.
6. Optional real-return smoke: fit a 0.6B observation model per AOSpec recipe on Toucan MCP returns (Apache-2.0) to check the "return = f(inputs)" case against real, messy outputs.

## Unverified

- "SimuAgent" as a world-model/tool-simulator paper: not found (search returned a Simulink assistant, arXiv 2601.05187, and SimuRA, arXiv 2507.23773).
- "Nemotron tool simulators" as a model: none found; only the Nemotron-Agentic-v1 dataset with LLM-simulated tool responses.
- B-PASTE details, SPORK, Trace2Env, Gecko, Simia numbers: only abstracts or secondary summaries seen.
- AOSpec, PASTE, TVCache, Simia, Qwen-AgentWorld, Speculate-While-You-Reason: no code release located; software licenses unknown.
- AOSpec overhead of wasted forks and sandbox fork latency: not reported in the paper.
- TVCache "27% incorrect hits" for naive caching: from a search snippet of conference slides only.
- Whether Exgentic traces include full tool result text (card overview says yes; schema sample shows no result field).
- WS-DREAM download path and login requirement; WireMock and Toxiproxy licenses (stated from memory); DreamGym and WebDreamer repo licenses; StableToolBench HF dataset license; tau2 `banking_knowledge` domain contents and whether tau2 hashes DB state (I recall DB-hash comparison but did not confirm).
- Tool latencies for finance/accounting SaaS APIs (QuickBooks, Xero, Plaid): not researched; no public datasets located.
- RunPod GPU pricing and fine-tune time for a 0.6B observation model: out of scope for this topic, not checked.
- Hit rates are author-reported on their own harnesses (several via trace replay, not live) and not directly comparable across papers (metrics differ: top-k action, any-of-many hit, exact output text).

## Addendum: Mission 7 episode-level metric for chained predictors (gap 5, 2026-10-17)

Verdict on the gap claim: PARTLY CORRECTED. None of the three named candidates defines an episode-level compounding metric or reports fidelity at 5-10 rollout steps. The AOSpec numbers are real but are not a plain action-accuracy curve. The suggestion that open-loop imagination needs per-step certification holds up, and the evidence for it is stronger than the 23.7% to 7.7% figure.

Facts (all author-reported; "unverified" marks what I could not confirm from a primary page):

- AOSpec (https://arxiv.org/html/2608.00881v1): 23.7% at distance 1 and 7.7% at distance 10 are "trace-matched target-action accuracy" at candidate width 32 (any of 32 forked candidates matches tool + args), a nine-cell mean over Terminal-Bench 2.0 traces with an untrained Qwen3.6-35B-A3B-FP8 drafter. Default deployment uses 8 forks; width 1 is 11.0% (not tied to a distance). The paper does not define "distance" and does not say whether the prefix is real or predicted (unverified; my reading is real-prefix trace replay, i.e. not free-running). The actual chain-compounding numbers are in Sec 3.3: exact-chain accuracy "collapses multiplicatively from 20.9% at one step to 2.4% at three and 1.2% at five", cited for prior multi-step approaches (setting not specified). AOSpec itself avoids chains: it predicts only the target action and verifies the launch state.
- Qwen-AgentWorld / AgentWorldBench (https://arxiv.org/abs/2606.24597, https://arxiv.org/html/2606.24597; submitted 2026-06-23): turn-level, not episode-level. An LLM judge (GPT-5.2; cross-judge Spearman 0.92-0.99) scores each predicted observation 1-5 on Format, Factuality, Consistency, Realism, Quality, given the REAL history; primary score is the mean scaled to 0-100. 2,170 samples; text domains use 5 turns per trajectory (first, last, 3 uniform). No horizon curve, no error-accumulation measurement, no 5/10-step figure in the first 100k of 242k characters of the HTML (remaining 142k chars unread). tau-bench/tau2 not named in the portion read; abstract says "9 established benchmarks", names Tool Decathlon, Terminal-Bench 1.0/2.0, OSWorld-Verified.
- DreamGym (https://arxiv.org/abs/2511.03773, https://arxiv.org/html/2511.03773v2): fidelity of the experience model is judged by GPT-4o on 4 criteria (consistency, diversity, informativeness, hallucination), scores 0/1/2, 100 sampled trajectories, no per-score values in the text read and no horizon analysis; trajectories "often ~10 steps". The theory (App. B) bounds the real-vs-synthetic gap by one-step reward error and transition error, with no empirical values. The only episode-level number is downstream RL success (e.g. Llama-3.1-8B WebShop 63.9, ALFWorld 66.3, WebArena-Lite 9.1 with DreamGym GRPO), which measures usefulness for training, not rollout fidelity.
- The closest pre-registerable episode metric found is the Consistency Ratio in "From Word to World" (https://arxiv.org/html/2512.18832v2, arXiv 2512.18832): CR = W2R / Real, where W2R is the success rate when the world model's action sequence is replayed in the real environment. Qwen2.5-7B SFT world model: ALFWorld 0.96, SciWorld 0.91, TextWorld 0.92, WebShop 0.67 (Llama3.1-8B: 0.98, 0.89, 0.88, 0.76). Per-step next-state exact match for the same models is 99.9/98.6/70.6/79.1%, so high per-step accuracy did not imply high episode transfer (WebShop). No horizon-binned curve reported, and the paper only says collection used up to 50 turns.
- tau-bench scoring (https://arxiv.org/abs/2406.12045): reward compares the database state at the end of the conversation with an annotated goal state; pass^k measures reliability over k trials (agents "pass^8 < 25% in retail"). This gives a ready end-state criterion for an imagined episode (final DB state matches), and the repo ships free historical trajectories (see datasets-typed-steps.md, row tau-bench v1). Whether tau2 hashes DB state internally remains unconfirmed.
- Chained per-step predictors are reported to fail while exact executable models do not (https://arxiv.org/abs/2609.09163, submitted 2026-07-07): abstract says synthesized-and-checked code world models are "exact over 20-step rollouts" while per-step LLM and MLP predictors "compound error and collapse". The "24 of 24 bit-exact vs 0 of 24" figure came from a search snippet and is unverified (not in the abstract). Single-source preprint.
- Survey-level statement that evaluation of long-horizon consistency is a field gap: https://www.emergentmind.com/open-problems/llm-world-models-coherent-multi-step-trajectories-executable (secondary page; not a primary source). Foresight paper https://arxiv.org/pdf/2601.03905v1 (agents rarely call simulators, misuse rollouts) seen via snippet only, unverified.

### Proposed pre-registered Mission 7 metric (my design, not from a paper)

Data: tau-bench historical trajectories (free, in repo) plus our seeded bookkeeping mock; split by task id, never by step. Report all of the following, fixed before any model is trained:

1. Survival curve S(k) for k = 1..10: fraction of held-out episodes whose free-running chain (predictor consumes its own prior outputs) has no mismatch through step k, under a pre-registered canonicalization (strip ids/timestamps, or use seeded deterministic ids). Exact-chain accuracy at k = 1, 3, 5, 10 is directly comparable with AOSpec's 20.9 / 2.4 / 1.2%.
2. Control: teacher-forced per-step accuracy a_k on the same steps. The gap between S(k) and the product of the a_i quantifies compounding beyond independence.
3. Episode outcome: replay the imagined action sequence in the real deterministic tool layer, then compare final DB state hash to the real episode's (CR = W2R / Real, as in Word to World; tau-bench end-state criterion).
4. Certified rollout (the only mode we ship): chain a step only if its confidence passes that step's certified cutoff; otherwise call the real model. Report coverage (share of steps and episodes imagined without a real call) at each k, and the audited episode error among fully-imagined episodes.

Budget arithmetic (derived, not from sources): an independent per-step error budget e gives episode error 1-(1-e)^k, which is 40.1% at e = 5%, k = 10 (union bound 50%). To hold an episode budget of 5% at k = 10 requires roughly 0.5% per step, and certifying 0.5% at 95% confidence with zero observed errors needs about ln(20)/0.005 = 600 error-free accepted calibration rows per step. This adds to the calibration-size plan in certification-deferral.md.

Caveats: tau-bench trajectories contain an LLM-played user, so "imagine the episode" there must either condition on the real user turns (teacher-forced user) or include a user-simulator step; the bookkeeping agent has no user turn. Trajectory counts and step-length distribution of the tau-bench historical files were not checked here (unverified); check before fixing the k = 10 horizon, since many episodes may be shorter or far longer than 10 steps.
