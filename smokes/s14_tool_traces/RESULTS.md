# S14 - Mission 3 on real agent tool traces

## In plain words

We took logs of real AI agents calling tools (two public benchmark sets of logged agent runs, a log of real coding-agent tool latencies, and a sample of real tool calls to hundreds of public tool servers) and asked how often a tool's answer could be reused, predicted, or started early.
When the same workflow is run again, the answer is "almost always": a certified cache (error budget 2%, both agents pooled) skipped 96-98% of the read-only calls, and those skips were wrong at most 0.4% of the time (0.0% when the cache also keys on what the agent had already changed).
When the task is new, the share drops to about 55% in the retail domain and to about 0-12% in the airline domain, and in a mixed sample of real tool servers it is 5-12%; so the benefit is capped by how repetitive the traffic is, which has to be measured per customer.
Whether a world-changing call will fail can be predicted from the earlier calls in the same run (AUC 0.74-0.85 against 0.59-0.65 when only the tool name is used), and the next call can be guessed correctly and safely ahead of time in 38-61% of retail steps.
Real tools also drift: a third of the tools with enough repeats return different text for identical arguments (weather, status, trending lists), so those must be speculated on, not skipped.
Tool latency is dominated by a few slow calls (34% of calls carry 99% of tool time) and is predicted much better from the session's recent latencies than from the tool's name.
For Dreaming: the tool-prediction idea holds up on real traffic and is worth benching, but the headline number to report is the repeat rate of the customer's own log, not a universal skip rate.

## Setup

Everything ran on RunPod pods through `smokes/pod.py` (no LLM calls, $0 API spend). Data were downloaded on the pod into `/workspace/job/data` by `fetch.py` (sizes and sha256 are in `log.txt`; they match the earlier local download byte for byte); only derived tables are kept here.

| Data | Size | License | Used for |
|---|---|---|---|
| tau-bench historical trajectories (gpt-4o and Claude 3.5 Sonnet "new" on retail and airline) | 4 files, 52.4 MB | MIT | A-E (primary). 1,980 episodes = 115 retail tasks x (4 gpt-4o + 8 sonnet trials) + 50 airline tasks x (4 + 8); 14,285 tool calls, each with its real return |
| tau2-bench `banking_knowledge`, gold actions replayed against the deterministic tools, no LLM | 955 rows (1 MB) | MIT | T. 97 tasks, 196 read-only `get_*` calls, 457 calls that change the database |
| TraceLab v0.0.2 trace (real coding-agent sessions, args redacted) | 100.9 MB | code Apache-2.0, data CC BY 4.0 | F. 743,819 tool records, 7,244 sessions, 52 users (305,005 Claude Code, 438,814 Codex) |
| Toucan-1.5M, OSS shard 10 of 47 | 146.9 MB | Apache-2.0 | G. 9,726 prompts, 29,398 tool calls, 1,083 tool names, 399 MCP servers |

Definitions. An episode is one (task, trial, agent) run. Tools are labelled read (lookups and searches), write (the 14 world-changing tools including `transfer_to_human_agents`) or pure (`calculate`, `think`). All splits are by task or by session, never by row. The reuse pass (A) is chronological, trial-major.

Method per analysis (code: `run.py` A-E, `extra.py` F, G, T; one `--tiny` plumbing check each, under 25 s on the laptop):
- **A** reuse. For each call, was the canonical (tool, args) seen in an earlier episode, and did that earlier return equal the new one exactly? Keys: (i) input-keyed; (ii) state-aware (tool, args, hashes of the successful writes earlier in the same episode, as in TVCache); (iii) entity-scoped (only writes that share an id with the call's arguments). The only canonicalization is key order, plus whitespace in `calculate`.
- **B** certified skipping. Memory returns the majority earlier return; confidence = k/(n+1) (k votes for the majority, n prior observations). `certify()` (fixed-grid Clopper-Pearson, delta 0.05, grid 40) on calibration episodes at 1%, 2%, 5%; applied to test episodes; 10 random task splits. Unseen keys always go to the real tool, so the certificate is over answerable rows and "skipped" is reported as a share of all reads. Two regimes: **W** (the workflow is re-run: memory = first half of the trials of every task, calibration/test = later trials of two disjoint task halves) and **X** (new tasks: memory/calibration/test = disjoint 40/30/30% of tasks, all trials).
- **C** outage survival for the three most-called read tools, W splits, 10 seeds: circuit breaker vs plain last-value cache vs certified cache.
- **D** write outcome (success vs `Error` return) from tool, argument features and earlier calls (including a few fields of the latest earlier read of the same entity); grouped 5-fold by task, task-bootstrap CI.
- **E** next-call speculation: n-gram (order <= 3) over tool names, a copy-argument heuristic (same-named argument from earlier calls or returns) and a trajectory cache (longest matching suffix of exact earlier calls); 5 seeds; W and X regimes.
- **F** latency on TraceLab: chronological split by session start (5,070 train sessions up to 2026-07-01, 2,174 test sessions after). History features (previous latencies, session EWMA) count only calls that had already finished when the new call was emitted. GBMs fitted on a fixed 300k-row sample of the training calls.
- **G** Toucan: identical (server, tool, arguments) in different prompts, identical return text?
- **Noise floor.** There is no LLM in this smoke, so the teacher-noise issue does not arise: the tau-bench tools are deterministic (state-aware repeats returned the identical string 100% of the time). The analogues here are the 10 random task splits (B, C) and the task bootstrap (D). Three full executions (laptop CPU in the earlier attempt, a pod, a second pod) produced identical tables for A, B, C, E, G, T; the only differences are tie-ordering in D's precision at top-20% for models with tied scores (1-2 points) and row order of tied rows.

## Results

### A. Reuse (pooled over both agents; the first trial of each task is a cold start and is included)

| domain | kind | calls | (i) seen in an earlier episode | of those, earlier return equal | (ii) state-aware: seen | of those, equal |
|---|---|---|---|---|---|---|
| retail | read | 7,939 | 94.1% | 99.5% | 91.5% | 100.0% |
| retail | write | 2,122 | 79.3% | 91.7% | = (i) | |
| airline | read | 2,669 | 86.1% | 99.8% | 82.6% | 100.0% |
| airline | write | 923 | 50.3% | 90.3% | = (i) | |

Per tool (retail then airline; (i) seen % / equal %, (ii) seen % / equal %):

| tool | calls | (i) | (ii) |
|---|---|---|---|
| get_order_details | 3,455 | 95.1 / 99.1 | 92.3 / 100 |
| get_user_details | 1,233 | 95.7 / 99.2 | 95.1 / 100 |
| get_product_details | 1,207 | 96.4 / 100 | 89.5 / 100 |
| find_user_id_by_name_zip | 1,159 | 94.7 / 100 | 94.7 / 100 |
| find_user_id_by_email | 695 | 80.3 / 100 | 80.3 / 100 |
| list_all_product_types | 190 | 99.5 / 100 | 85.8 / 100 |
| calculate (pure) | 103 | 44.7 / 100 | 44.7 / 100 |
| get_reservation_details (airline) | 1,419 | 91.1 / 99.5 | 88.9 / 100 |
| search_direct_flight (airline) | 538 | 74.5 / 100 | 67.5 / 100 |
| get_user_details (airline) | 486 | 91.8 / 100 | 89.9 / 100 |
| search_onestop_flight (airline) | 208 | 68.8 / 100 | 60.6 / 100 |
| writes, retail: return_delivered 489, modify_pending_order_items 458, exchange_delivered 416, cancel_pending 291, modify_pending_address 227 | | seen 88.5 / 79.9 / 76.2 / 87.3 / 87.7 | equal 95.6 / 82.5 / 95.0 / 98.8 / 80.4 |
| writes, airline: update_flights 287, cancel_reservation 188, book_reservation 149, update_baggages 66 | | seen 62.4 / 80.9 / 30.9 / 68.2 | equal 92.2 / 98.7 / 87.0 / 71.1 |

- Where the input-keyed cache is wrong, it is because a write changed the state: error among repeated reads is 0.2% before any write in the episode and 4.1% after one in retail (n = 6,930 / 541), and 0.2% vs 1.1% in airline (n = 2,119 / 180). The state-aware key removes all of it (100% equal) at a cost of 2-9 points of coverage (for example `get_product_details` 96.4% -> 89.5%).
- Cold start vs warm (state-aware reusable share of reads by trial index): retail gpt-4o 44% (trial 0) -> 89% -> 95% (trial 3); retail sonnet 48% -> 93% -> 97% (trial 7); airline gpt-4o 16% -> 75% -> 81%; airline sonnet 18% -> 80% -> 99%.
- Writes repeat too: the same write call in a fresh state returned the same outcome 55-100% of the time depending on the tool (cancel 98.8%, address changes 80-100%, airline passenger updates 54.5% on 24 calls). `transfer_to_human_agents` never repeats (0% seen).
- Only 2.6-7.0% of reads were seen *only* in other tasks (84.5% / 80.0% of retail / airline reads were also seen in the same task's earlier trials), so most reuse is "same task, again". The cross-task share is measured properly in B-X.

### B. Certified skipping (read calls, pooled agents, 10 splits per cell; each cell: % of all test reads skipped / realized error % of those skipped, with the number of splits that certified)

Regime W (workflow re-run):

| domain | key | 1% budget | 2% budget | 5% budget |
|---|---|---|---|---|
| retail | input-keyed | 72.0 / 0.35 (9/10) | 98.5 / 0.40 (10/10) | 98.5 / 0.40 |
| retail | state-aware | 97.1 / 0.00 (10/10) | 97.1 / 0.00 (10/10) | 97.1 / 0.00 |
| retail | entity-scoped | 98.5 / 0.1 (10/10) | 98.5 / 0.1 | 98.5 / 0.1 |
| airline | input-keyed | none (0/10) | 97.4 / 0.00 (10/10) | 97.4 / 0.00 |
| airline | state-aware | none (0/10) | 95.7 / 0.00 (10/10) | 95.7 / 0.00 |

Oracle (share of test reads whose majority earlier return equals the real one): retail 97.1-98.4%, airline 95.7-97.4%. The certificate gives up almost nothing against it.

Regime X (new tasks, memory from other tasks only):

| domain | key | 1% budget | 2% budget | 5% budget | oracle |
|---|---|---|---|---|---|
| retail | input-keyed | 32.3 / 0.33 (6/10) | 58.4 / 0.31 (10/10) | 58.4 / 0.31 | 58.3% |
| retail | state-aware | 54.6 / 0.00 (10/10) | 54.6 / 0.00 (10/10) | 54.6 / 0.00 | 54.7% |
| airline | input-keyed | none | none | 10.2 / 0.71 (6/10) | 18.2% |
| airline | state-aware | none | none | 11.7 / 0.00 (7/10) | 17.7% |

- Of the 108 (regime, domain, agent, key, budget) cells, 69 certified in at least one split; in none of them did any split realize an error above its budget (0 violations; the largest pooled realized error is 0.71%).
- Per tool, W retail state-aware at 2% (share of that tool's test calls skipped; errors in brackets): get_order_details 99% of 6,159 (0), get_user_details 99% of 2,120 (0), get_product_details 95% of 2,111 (0), find_user_id_by_name_zip 99% of 1,920 (0), find_user_id_by_email 87% of 1,312 (0), list_all_product_types 92% of 216 (0). Airline: get_reservation_details 99% of 2,663 (0), get_user_details 98% of 946 (0), search_direct_flight 89% of 967 (0), search_onestop_flight 82% of 378 (0). The input-keyed errors in retail W come from two tools only: get_order_details 40 wrong of 6,153 skipped, get_user_details 14 of 2,120.
- The 1% budget is often infeasible for sample-size reasons, not error reasons: with a zero-error calibration stretch, a Clopper-Pearson bound with delta 0.05/40 needs about 670 rows to reach 1% (about 330 for 2%). Per-agent cells with 200 episodes (gpt-4o airline: 2% infeasible in W, 5% certifies in 6-9 of 10 splits) show the same effect.

### C. Outage survival (regime W, pooled, dead tool = one of the three most-called read tools; certificate at 2%)

| dead tool | circuit breaker answers | plain last-value cache: answered / error of answered | certified state-aware cache: answered / error | episodes served with every answer correct (breaker / plain / certified) |
|---|---|---|---|---|
| retail get_order_details (6,159 calls) | 0% | 99.9% / 0.65% | 98.7% / 0.00% | 0 / 98.0 / 97.0 % |
| retail get_user_details (2,120) | 0% | 100% / 0.66% | 99.5% / 0.00% | 0 / 99.6 / 99.5 % |
| retail get_product_details (2,111) | 0% | 99.7% / 0.00% | 95.0% / 0.00% | 0 / 99.6 / 93.5 % |
| airline get_reservation_details (2,663) | 0% | 99.2% / 0.00% | 99.2% / 0.00% | 0 / 99.2 / 99.2 % |
| airline search_direct_flight (967) | 0% | 94.1% / 0.00% | 89.0% / 0.00% | 0 / 93.1 / 91.3 % |
| airline get_user_details (946) | 0% | 98.5% / 0.00% | 97.7% / 0.00% | 0 / 98.4 / 97.5 % |

The circuit breaker completes no episode that calls the dead tool. In this benign setting (static database, workflow already seen) a plain cache also does well; what certification adds is the ability to say which answers are safe: the stale-after-write errors (0.65%) go to 0.00% and the cache only answers keys it has evidence for, at a coverage cost of 0.5-5 points. With only 200 gpt-4o airline episodes the certified cache at 2% answers nothing (too few calibration rows) while the plain cache answers 91%: certification needs data to show its safety.

### D. Write outcomes (predict success vs an `Error` return; `transfer_to_human_agents` excluded)

| domain | writes (errors, base rate) | model | AUC (task-grouped 5-fold) | AUC 95% CI | precision / recall at the top 20% flagged |
|---|---|---|---|---|---|
| retail | 2,018 (209, 10.4%) | per-tool base rate | 0.598 | [0.517, 0.688] | 18% / 34% |
| retail | | tool-only LR | 0.585 | [0.508, 0.675] | 17% / 33% |
| retail | | all-context LR | 0.740 | [0.644, 0.818] | 26% / 50% |
| retail | | all-context GBM | 0.795 | [0.724, 0.868] | 29% / 56% |
| airline | 744 (148, 19.9%) | per-tool base rate | 0.654 | [0.516, 0.764] | 14% / 14% |
| airline | | tool-only LR | 0.643 | [0.510, 0.752] | 15% / 16% |
| airline | | all-context LR | 0.850 | [0.780, 0.914] | 56% / 56% |
| airline | | all-context GBM | 0.789 | [0.712, 0.862] | 46% / 47% |

Error rates differ by tool (retail exchange 18%, modify_items 12%, return 10%, cancel 1%; airline book_reservation 39%, update_flights 29%, cancel 0%), but the tool name alone explains little (AUC about 0.6). Context helps: strongest features are the number of items or flights in the call, the payment type (certificate vs card), the status of the order or reservation read earlier in the run, and how many times the same write was already tried. Flagging the top 20% of write calls catches 50-56% of the failures (chance would be 20%).

### E. Next-call speculation (every position with a following call; `think` calls dropped; prediction made when the call is issued)

| regime | domain | agent | positions/seed | next is read | name top-1 | name top-3 | name+args, copy heuristic top-1 | name+args, with trajectory cache top-1 | useful speculation (next is read-only AND exactly right) | gate tau 0.7: launched / precision |
|---|---|---|---|---|---|---|---|---|---|---|
| W | retail | gpt-4o | 1,380 | 72.8% | 66.8% | 86.5% | 29.3% | 67.1% | 55.1% | 25.9% / 93% |
| W | retail | sonnet | 3,039 | 75.2% | 67.1% | 87.8% | 24.4% | 72.7% | 60.8% | 44.3% / 95% |
| W | airline | gpt-4o | 455 | 56.0% | 55.2% | 79.8% | 9.2% | 49.0% | 40.7% | 14.3% / 89% |
| W | airline | sonnet | 1,108 | 70.7% | 61.3% | 84.0% | 12.0% | 64.5% | 54.8% | 36.7% / 93% |
| X | retail | gpt-4o | 825 | 73.1% | 60.3% | 78.2% | 28.1% | 40.5% | 38.1% | 20.1% / 79% |
| X | retail | sonnet | 1,822 | 75.6% | 63.0% | 82.0% | 24.1% | 41.1% | 39.0% | 26.6% / 75% |
| X | airline | gpt-4o | 261 | 58.8% | 51.6% | 71.6% | 10.0% | 15.6% | 14.2% | 5.9% / 75% |
| X | airline | sonnet | 730 | 70.9% | 55.5% | 71.2% | 11.0% | 16.7% | 15.9% | 7.8% / 75% |

- Literature check: Speculative Actions reports 22-38% next-call hits on tau-bench retail. Our copy-argument heuristic alone gets 24-29% for name+args top-1 in retail (both regimes), which sits inside that range; the trajectory cache lifts it to 67-73% when the workflow is re-run (W) and 41% for new tasks (X).
- The hard steps are the ones right after a user message: W retail sonnet 53.7% (39% of positions) vs 84.9% inside an uninterrupted chain of calls (61%). Predicting that the next call is a write is poor (name+args top-1 48.2% vs 80.8% when the next call is a read, W retail sonnet), which is the right way round for speculation, since only reads are launched.
- Launching every read-only guess has a precision of 19-79% depending on regime and domain (73-79% W retail, 47-48% X retail, 19-20% X airline); gating on the agreement score (tau 0.7) trades volume for precision (93-95% in W retail). The score is a heuristic, not a calibrated probability.

### F. Latency (TraceLab; chronological split by session start; 196,501 test calls)

- Calls above 1 s are 33.8% of calls and 99.3% of the summed tool time (above 10 s: 11.2% of calls, 94.7% of time). Median call 218 ms, mean 15.6 s, p90 10.1 s. Summed over parallel calls, so not elapsed time.
- Per tool (median / p90 / share of calls > 1 s): exec_command (251,607 calls) 265 ms / 1.6 s / 26.8%; Bash (158,471) 329 ms / 20.4 s / 41.4%; Read (63,703) 27 ms / 245 ms / 2.8%; Edit (36,533) 75 ms / 2.0 s / 17.5%; write_stdin (75,952) 5.0 s / 44.7 s / 69.5%; wait (11,404) 11.9 s / 18.4 s / 79.9%; WebFetch (3,636) 6.3 s / 22.6 s / 92.8%.

| predictor (test sessions) | MAE ms | p90 abs error ms | mean abs log-ratio |
|---|---|---|---|
| global median | 14,382 | 11,154 | 2.04 |
| per-tool median | 14,275 | 12,091 | 1.48 |
| per-(tool, first executable) median | 14,256 | 12,678 | 1.45 |
| per-tool EWMA over the whole stream (alpha 0.1) | 13,448 | 8,827 | 1.18 |
| per-session same-tool EWMA (0.5) | 13,520 | 8,692 | 1.05 |
| GBM on call-time features (tool, executable, input size, provider, position, hour) | 14,006 | 10,359 | 1.75 |
| **GBM + session history (previous finished latencies)** | **12,673** | **8,515** | **0.99** |

- Excluding the long-wait tools (sub-agents, wait, write_stdin, human prompts; 88% of calls kept): per-tool median MAE 7,931 / p90 8,405; per-tool EWMA 7,559 / 6,868; session EWMA 7,617 / 6,773; GBM + history 7,215 / 6,057.
- "Will this call take over 1 s?" (test base rate 38.1%): AUC 0.717 per-tool slow rate, 0.769 GBM on call-time features, 0.884 GBM + session history; precision when flagging as many calls as there are slow ones 60% / 61% / 75%.
- Versus the per-tool median, GBM + session history cuts MAE 11%, p90 error 30%, log error 33%; versus the simplest history baseline (session EWMA) the extra gain is small (MAE 6%, p90 2%, log error 6%). So most of the gain is "use what this session's recent calls took", not model capacity.
- The call-time-only GBM does not beat the per-tool median on later sessions (log error 1.73-1.78 vs 1.48 across three feature sets), but it does on a random hold-out of train-period sessions (1.49 vs 1.72). The gap is drift between the two periods, not a bug: latency for the same tool shifts over weeks, so static latency models go stale and recency features do not.

### G. Toucan shard (real MCP tool calls)

- 29,398 calls, 20,796 distinct (server, tool, args); 36.4% of calls have an exact repeat somewhere in the shard. Returns that look like errors or unavailable-tool messages (keyword heuristic): 18.5%.
- Keys seen in 2 or more different prompts: 1,846. Identical return text for 65.7% (77.5% after masking numbers and timestamps); 61.9% when restricted to the 1,513 keys with no error-looking return.
- Of 1,297 tools, 923 (71%) have too few repeats to observe. Of the 374 observable: 188 (50%) deterministic (at least 95% of repeated keys identical), 61 mixed, 125 (33%) time- or state-varying. Of the 188 deterministic tools, 24 are deterministically broken (constant failure messages: missing credentials, wrong platform). Deterministic tools carry 51% of the calls of the observable tools.
- Deterministic examples: `drawing_generateCanvas`, `convert_volume` (250 ml -> cup 1.0567), `get_fruit_nutrition`, `get_component_examples`. Varying examples: `get_weather_forecast` (40% of repeated keys identical; "last updated" timestamp changes), `get_weather` London (3 distinct returns in 8 calls), `status --all` (5 distinct in 12), `fetch_reddit_hot_threads` (score changes), `resolve-library-id` (25% identical).
- Certified skipping, episodes = prompts (50/25/25% split, 10 splits, every tool treated as read-only): 1% budget certifies in 1/10 splits (0.9% of calls skipped, 0.2% error), 2% in 5/10 (4.6%, 0.7%), 5% in 10/10 (12.3%, 2.3%); oracle (key seen in memory and majority correct) 20.0%.

### T. tau2-bench banking_knowledge gold replay

97 tasks, 955 calls, 457 change the database, 196 are read-only. A read is seen in an earlier task for 14.3% of reads by input key (23 of 28 equal) and for only 3.1% when keyed on the hash of the whole database before the call (6 of 6 equal). One scripted trajectory per task, so this measures the task set and the tools, not an agent.

## Transfer to Dreaming

| Tool case in missions.md | Real-trace number that supports | Real-trace number that undercuts |
|---|---|---|
| Return is a function of inputs we can see (e.g. vendor lookup) | tau-bench repeated reads return the identical string 99.5-100%; certified 2% cache skips 96-98% of reads (82-100% per tool) when the workflow is re-run, 0.0-0.4% realized error, 0 budget violations in the 69 certified cells; 50% of the 374 observable Toucan tools are deterministic | Skip share is bounded by the repeat rate: new retail tasks 55%, new airline tasks 0-12%, tau2 banking 14% of reads seen before, Toucan shard 4.6% of calls at 2%; 71% of Toucan tools never repeat; the 1% budget needs about 670 clean calibration rows |
| Return depends on state we cannot see (e.g. refund status) | State-keyed caching removes stale answers (0.4% -> 0.0%); next-call speculation with a heuristic gate (tau 0.7) is right 93-95% of the time on 26-44% of retail steps when the workflow repeats; without a gate 38-61% of retail steps are launched and correct; name+args 24-29% from the copy heuristic is inside the published 22-38% | Input-keyed errors rise from 0.2% to 4.1% after a write; 33% of observable Toucan tools return changing text for identical arguments (weather, status, trending); keying on the whole database leaves 3.1% of tau2 reads reusable, so keys must be scoped to the entities involved; new airline tasks: only 14-16% of next calls speculated correctly |
| Call changes the world (e.g. issue a refund) | Failure of a write is predictable from the run so far: AUC 0.74-0.85 vs 0.59-0.65 from the tool name; the top 20% flagged holds 50-56% of failures; latency: session history cuts p90 error 30% and lifts slow-call AUC from 0.72 to 0.88 | Only 209 (retail) and 148 (airline) failures, wide CIs (about +-0.08); static latency features go stale across weeks (drift check); writes repeat with the same outcome only 55-100% of the time depending on the tool |

**Does tool prediction hold up, or does it need more benching?** On real traces it holds up where tool traffic repeats (a certified 2% cache skipped 96-98% of re-run reads with no budget violation, and rode out a simulated tool outage where a circuit breaker answered 0%), but the achievable skip rate is the customer's repeat rate (55% on new retail tasks, about 10% on new airline tasks, 5-12% on a mixed sample of public tool servers), so it still needs benching on each target agent's own log, and on a live API with drift and TTLs, before a number is promised.

## Interpretation for the mission

- The synthetic S4 result (70% skipped at 0.12% error, outage 89% vs 0%, speculation hit 0.63, write-failure AUC 0.71) is in the right range for real traces, and in tau-bench it is conservative for repeated workflows (96-98% skipped, outage episodes 91-99.5% served with every answer correct) and optimistic for novel inputs.
- Dreaming's headline claim for mission 3 should be stated as a function of the repeat rate: certified skip share is close to (repeat rate) x (share of repeats that are safe), and the second factor was 99.5-100% here. The first factor is cheap to measure from the log before any model is trained, which makes it a good sales-time number.
- Hidden state is mostly the agent's own writes in tau-bench, and a state key plus an entity scope handles it; external drift (Toucan) needs a TTL or a freshness check, and speculation rather than skipping.
- For slow tools, the learned latency matters more than the learned return: a third of calls carry 99% of tool time, and recent session latencies beat the tool name. A latency feature belongs in the shadow log from day one.
- The 1% budget needs far more calibration rows than 2%; a Dreaming product should show the operator "rows needed" per budget.

## Caveats

- tau-bench tools are deterministic in-memory functions over a fixed database, and the same 115 retail and 50 airline tasks are run 4-8 times, so W numbers are the upper end for a recurring workflow. The agents (GPT-4o, Claude 3.5 Sonnet) are 2024 models; the tools themselves are cheap, so the real saving there is not the tool call but the overlap or outage case.
- The certificate is over answerable rows (unseen keys go to the real tool by construction). "Skipped" is a share of all reads. The B/C/E tau-bench rows pool 4-8 trials per task, so test episodes of the same task are correlated even though calibration and test tasks are disjoint.
- E's gate score is a heuristic, not a calibrated probability; D's precision at top-20% moves by 1-2 points with tie ordering (AUCs do not).
- F: wall latency includes time waiting for permission or human answers and for sub-agents (the `wait`, `write_stdin`, `AskUserQuestion` tools); the Codex `shell` tool reports about 0 ms; TraceLab is one lab's 52 users; the 11% MAE gain is dominated by the heavy tail. Provider mix shifts over time (Claude Code and Codex).
- G: one shard; the keyword error heuristic is crude; "unobservable" tools (71%) may be deterministic or not; Toucan questions are LLM-written, only the returns are real; no timestamps, so drift is inferred from changing text.
- T: gold actions only (one trajectory per task).
- Provenance: the earlier attempt ran A-E, G and the tau2 gold replay on the laptop (the replay is `tau2_replay.py`, run in a separate venv that is now in `smokes/.data/s14_tau2_venv`, with an 874 MB repo clone) and its F run never finished (killed after 14 min 45 s without output). This final session re-ran A-E, F, G, T on pods; F was rewritten with latency features that only use calls already finished at emission time and a 300k-row training sample. Both pod runs reproduced the earlier A-E, G, T tables exactly (up to tie ordering).

## Time and cost

- API spend: $0.00 (no LLM calls).
- Pods (RunPod, `smokes/pod.py`): run 1 on an RTX 3090 (A5000 and RTX 4000 Ada had no stock; secure cloud, $0.50/h) 4.8 min = $0.040; final run on an RTX A5000 ($0.27/h) 5.2 min = $0.023. Total about $0.06. Each pod job (boot, pip, 300 MB of downloads, A-E, F, G, T) took about 5 min wall; F itself 35 s after loading.
- Laptop: two `--tiny` plumbing checks (16 s and 21 s). The earlier attempt (not counted here) ran A-E and G on the laptop CPU in a few minutes.
- Files: `run.py` (A-E), `extra.py` (F, G, T), `fetch.py` (pod downloads), `tau2_replay.py` + `tau2_calls.jsonl` (T input), `tables_*.md` and `results_tau.json`, `results_extra.json` (all numbers), `log.txt` (final pod log).
