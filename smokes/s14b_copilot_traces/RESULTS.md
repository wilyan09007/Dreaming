# S14b - S14 on the GitHub Copilot coding-agent traces (2026)

## In plain words

S14 asked three questions about real agent tool traffic: can a write failure be predicted, can the next call be guessed ahead of time, and how predictable is tool latency. It used tens of thousands of calls from benchmark runs and 0.7 M calls from one lab's coding sessions. S14b asks the same three questions on a corpus about 4.5 times larger and from many ordinary users: two days (2026-06-01 and 2026-06-02) of GitHub Copilot coding-agent traces, 3.31 M tool calls (the corpus has 116,015 sessions; about 78,000 of them contain a tool call; T uses the 55,870 + 60,144 sessions with timed turns on day 1 and day 2) (metadata only: tool name, status, duration, plus LLM call timestamps, durations and token counts).
Failure prediction transfers: a failed tool call (status 2) is predictable from what the session did just before (AUC 0.878 with a gradient-boosted model on session context against 0.798 from the tool's base failure rate), and the chronological test (train day 1, test day 2) gives the same numbers as the session-grouped one.
Next-tool guessing is much weaker than in S14's benchmark runs: the best top-1 name accuracy is about 48-49% (trigram, 4-gram, GBM), and a confidence gate at 0.7 launches on only 10.5% of positions with 71% precision (trigram). Right after a user message it is 36.3-37.8%, no better than always guessing the most common tool (37.6%); the GBM (36.3%) is worse.
Latency is dominated by a small slow tail (24% of calls above 1 s carry 99% of summed tool time), and the best predictor combines the tool name with the session's recent latencies (log error 0.75 against 1.10 for the per-tool median; "slower than 1 s" AUC 0.942). Recent latencies alone, without the tool name, do not beat the per-tool median here.
New in S14b: where the time goes. Inside active turns, LLM calls take 40-41% of the time, tool batches 35-42% (19-26% without the 0.2% of turns that contain a batch over 1 h, which carry about a fifth of the time), and gaps 18-25%. With a perfect predictor, S19's rule (min of tool batch and next LLM step) hides 2.9% (day 1) and 3.9% (day 2) of active turn time and 0.8% of session wall clock. That sits inside S19's perfect-predictor range of 1.1-6.4%, not against its 0.5-3% (which is the saving at S14's hit rates). The rule leaves out the wait between batch end and the next LLM start: adding it, the true perfect-predictor ceiling is 13.1% / 17.9%, or 6.2% / 8.2% if only waits up to 60 s can be hidden (79% of wait time is in waits over 60 s, probably humans or permissions). So the honest range is about 3-4% (S19 rule), 6-8% (waits up to 60 s) and at most 13-18%; these ceiling figures come from a separate check script, not from results.json.
For Dreaming: tool-failure prediction and a latency feature are worth putting in the shadow log; next-call speculation on coding traces is a narrow win by name accuracy, and the time-share ceiling is 3-4% under the S19 rule and up to 6-8% (13-18% at most) if the waits between batch and next LLM call can also be hidden, so it is not clearly a headline for coding agents.

## Setup

No LLM calls and no pod: everything ran on the local machine (4 CPUs, 15 GB), $0.00 spend.

| Data | Size | License | Used for |
|---|---|---|---|
| GitHub Copilot coding-agent traces 2026 (Azure/AzurePublicDataset release `ghcp-coding-agent-2026`), day 1 = 2026-06-01 (train), day 2 = 2026-06-02 (test) | day 2 archive 621 MB; flattened to 1,635,686 + 1,674,818 tool calls, 1,748,037 + 1,827,928 LLM calls, 1,488,918 + 1,524,765 batches | CC BY 4.0 | D', E', F', S', T, X |
| TraceLab v0.0.2 trace (the S14 F data) | 101 MB, 743,819 tool records, 743,050 with a wall latency | code Apache-2.0, data CC BY 4.0 | X (cross-dataset latency check) |

Metadata only: no prompts, arguments or return values are in the corpus. Model names are anonymised in the data (Model A to Model I below).
Pipeline (code in this folder): `fetch.py` downloads day 2 and TraceLab; `flatten.py` turns the shards into flat call, LLM-call and batch tables (about 2 min per day with 4 processes); `common14b.py` holds shared helpers; `run.py` runs the sections D, E, F, S, T, X and writes `results.json` and `tables_*.md`.

Definitions that differ from S14 (read before comparing numbers):
- **Status.** 1 = success, 2 = failure, 0 = undocumented (3.6% of calls, short, concentrated in `run_command_in_terminal` and `run_build`). D' labels only status 1/2 calls (failure = status 2); status-0 calls stay in the history features as non-failures.
- **D'** covers all tools (reads and writes alike), not only the 14 world-changing tools of S14 D. Base rate is 4.7%, not 10-20%; sample sizes are 1.6 M calls per day, not 2,018 and 744, so confidence intervals are far tighter (a session bootstrap with 200 resamples).
- **E'** predicts the next tool name only (no arguments exist), over a vocabulary of the 24 most common day-1 tools plus OTHER (OTHER counts as a miss). Positions are every tool call that has a following call in the same session; the `think` filter of S14 does not apply. The gate probability is the n-gram count share or the GBM probability, not S14's agreement score. Two position types are separated: the next call opens a new turn (after a user message) versus in-chain.
- **F'.** Timestamps are completion times (checked: a batch starts no earlier than the issuing LLM call in 100.0% of pairs, and the next LLM call starts no earlier than the batch end in 98.7%). A call is issued at its batch start and ends after its own duration. Split by day, not by session start; 40,156 test sessions never seen in training. Errors are in milliseconds on a heavier tail than S14 (max 5.3e8 ms), so `MAE clip 1 h` is also given. GBMs use a 300,000-row training sample.
- **Same-tool EWMA** is per session, as in S14. S14 also had a "per-(tool, first executable)" row and a provider feature; neither exists here.
- **Units of time.** An active turn runs from its first LLM call's start to its last event's end; user idle time between turns is excluded from the active shares and reported separately.
- **Noise floor.** There is no LLM in the loop, so teacher noise does not apply. The analogue is the session bootstrap (D, F) and the day-1 session hold-out comparison (F), reported below. The pipeline ran once (`log.txt`): stages F and E first, then D, S, T, X after a column-overlap bug in D was fixed (the history features were read back from the cache written by F). No other differences between runs.

## Results

### S'. Structure of the corpus

Structure of the corpus (day 1 = 2026-06-01, day 2 = 2026-06-02; numbers for the pooled two days are in results.json):
| quantity | day 1 | day 2 |
|---|---|---|
| tool calls / sessions with a tool call / distinct tool names | 1635686 / 37521 / 855 | 1674818 / 40156 / 872 |
| LLM calls / turns | 1748037 / 222774 | 1827928 / 235008 |
| calls per session: mean / median / p90 | 43.6 / 22 / 103 | 41.7 / 21 / 99 |
| turns per session (mean) | 3.99 | 3.91 |
| tool calls per turn (turns with >= 1 call): mean / median / p90 | 10.5 / 6 / 26 | 10.2 / 6 / 26 |
| LLM calls per turn: mean / median | 7.8 / 4 | 7.8 / 4 |
| turns with no tool call | 30.0% | 30.2% |
| batch size: 1 / 2 / 3 / 4 / 5+ (share of batches) | 93.0 / 4.7 / 1.4 / 0.6 / 0.3 | 92.9 / 4.9 / 1.4 / 0.6 / 0.3 |
| mean batch size | 1.11 | 1.11 |
| call has the same tool as the previous call in the session | 38.6% | 37.9% |
|   ... when the previous call is in the same batch / in an earlier batch | 75.1% / 34.4% | 74.2% / 33.8% |
|   share of consecutive pairs that are inside one batch | 10.2% | 10.2% |
| runs of the same tool: mean / median / p90 / max length | 1.60 / 1 / 3 / 1757 | 1.59 / 1 / 3 / 158 |
| share of calls inside runs of length >= 3 / >= 10 | 36.1% / 7.9% | 35.3% / 7.4% |
| distinct models; top-5 share of tool calls | 26; 70.9% | 28; 70.2% |
| agent-initiated share of LLM calls | 87.7% | 87.6% |
| LLM calls with token counts | 97.3% | 97.3% |
| cache hit share of prompt tokens: all / agent calls / user-turn calls | 85.0% / 89.9% / 47.4% | 85.5% / 90.4% / 47.7% |
| agent LLM calls with any cached token | 97.5% | 97.7% |
| median prompt / completion tokens per LLM call | 56910 / 170 | 55596 / 172 |
| top-3 tools by calls; share of calls (top 10 share) | get_file, run_command_in_terminal, replace_string_in_file; 61.8% (90.1%) | get_file, run_command_in_terminal, replace_string_in_file; 61.2% (89.9%) |
| top-3 tools by summed tool time; share of tool time | run_command_in_terminal, run_build, get_file; 88.8% | run_command_in_terminal, run_build, tool_960e73fa6c1c; 86.1% |
| LLM call results | Success 98.76%, UserCancel 0.73%, Failure 0.44%, UserFault 0.08% | Success 98.69%, UserCancel 0.76%, Failure 0.45%, UserFault 0.10% |

Model mix (share of tool calls), day 1: Model A 29.5%, Model B 12.7%, Model C 11.7%, Model D 9.1%, Model E 8.0%, Model F 6.5%, Model I 6.4%, Model G 4.1%. Day 2: Model A 28.8%, Model B 13.1%, Model C 10.8%, Model D 9.2%, Model E 8.2%, Model F 7.1%, Model I 5.6%, Model G 4.5%.

### D'. Tool failure prediction (S14 D equivalent; all tools, status 2 vs status 1)

Status shares over all tool calls: success (1) 91.8%, failure (2) 4.5%, status 0 3.6% (not described in the schema; its durations are short, median 32 ms; concentrated in run_command_in_terminal and run_build). Status-0 calls are dropped from every D' label set but stay in the history features (as non-failures). Failure = status 2 among status 1/2 calls: day 1 4.73% of 1574247, day 2 4.70% of 1616565.
Failures are clustered: the top 5% of day-1 sessions by failure count hold 43.8% of all failures. Failure rate of a call by status of the latest finished call in its session (day 1): none yet 4.41% (n=45661), success 4.16% (n=1458597), failure 16.92% (n=69989).

Session-grouped 5-fold on day 1 (1574247 calls, 37202 sessions, failure base rate 4.73%); AUC with 95% session-bootstrap CI (200 resamples), AUC difference to the per-tool base rate (paired), precision / recall when the top 20% of calls by score are flagged (random ties):
| predictor | AUC | AUC gain vs per-tool rate | precision / recall at top 20% |
|---|---|---|---|
| per-tool base rate | 0.798 (0.795-0.801) | - | 15.7% (15.4%-16.1%) / 66.4% (65.7%-67.1%) |
| LR call-time features | 0.814 (0.812-0.818) | +0.017 (+0.015 to +0.018) | 15.8% (15.5%-16.1%) / 66.7% (65.9%-67.4%) |
| LR + session context | 0.859 (0.857-0.862) | +0.061 (+0.059 to +0.064) | 17.4% (17.1%-17.7%) / 73.4% (72.6%-74.0%) |
| GBM call-time features | 0.831 (0.828-0.834) | +0.033 (+0.031 to +0.035) | 16.1% (15.9%-16.5%) / 68.2% (67.5%-69.0%) |
| GBM + session context | 0.878 (0.876-0.880) | +0.080 (+0.078 to +0.083) | 18.2% (17.9%-18.5%) / 76.8% (76.2%-77.4%) |

Chronological: train day 1, test day 2 (1616565 calls, 39857 sessions, failure base rate 4.70%); AUC with 95% session-bootstrap CI (200 resamples), AUC difference to the per-tool base rate (paired), precision / recall when the top 20% of calls by score are flagged (random ties):
| predictor | AUC | AUC gain vs per-tool rate | precision / recall at top 20% |
|---|---|---|---|
| per-tool base rate | 0.799 (0.796-0.802) | - | 15.7% (15.5%-16.0%) / 67.0% (66.2%-67.8%) |
| LR call-time features | 0.815 (0.811-0.818) | +0.016 (+0.014 to +0.017) | 15.7% (15.5%-16.0%) / 66.9% (66.1%-67.5%) |
| LR + session context | 0.859 (0.857-0.861) | +0.060 (+0.058 to +0.063) | 17.2% (17.0%-17.5%) / 73.3% (72.8%-73.8%) |
| GBM call-time features | 0.832 (0.828-0.835) | +0.033 (+0.031 to +0.035) | 16.1% (15.8%-16.4%) / 68.5% (67.9%-69.2%) |
| GBM + session context | 0.878 (0.877-0.880) | +0.080 (+0.077 to +0.082) | 18.0% (17.8%-18.3%) / 76.8% (76.3%-77.3%) |

Per-tool failure rate (status 2 among status 1/2 calls), top 15 tools by calls (both days):
| tool | calls | failure rate | failures | median ms (success) | median ms (failure) | % status 0 |
|---|---|---|---|---|---|---|
| get_file | 1155368 | 2.46% | 28411 | 68 | 80 | 0.3% |
| run_command_in_terminal | 472549 | 13.09% | 61839 | 529 | 4967 | 14.0% |
| replace_string_in_file | 324689 | 0.86% | 2802 | 261 | 101 | 1.1% |
| code_search | 266614 | 0.67% | 1776 | 1764 | 943 | 0.0% |
| file_search | 152659 | 1.94% | 2961 | 59 | 36 | 0.0% |
| run_build | 134888 | 25.54% | 34447 | 5303 | 11360 | 17.3% |
| apply_patch | 111685 | 2.40% | 2680 | 299 | 64 | 0.5% |
| create_file | 96787 | 5.17% | 5004 | 619 | 64 | 1.1% |
| update_plan_progress | 81443 | 0.07% | 61 | 18 | 43 | 0.0% |
| multi_replace_string_in_file | 68219 | 2.03% | 1383 | 373 | 34 | 1.1% |
| get_errors | 63992 | 1.52% | 975 | 63 | 52 | 0.0% |
| edit_file | 38478 | 11.73% | 4515 | 5299 | 2083 | 1.3% |
| get_files_in_project | 28087 | 0.22% | 61 | 66 | 18 | 0.0% |
| get_projects_in_solution | 25469 | 0.16% | 41 | 95 | 108 | 0.0% |
| find_symbol | 25123 | 0.42% | 105 | 219 | 14703 | 0.0% |

Reading it. Failure is concentrated in two tools (`run_command_in_terminal` 13.1% and `run_build` 25.5%) and in failing sessions: the next call fails 16.9% of the time after a failed call against 4.2% after a success. The tool name alone already gives AUC 0.798-0.799 because of that, but session context still adds a clear gain with a tight CI: +0.060 (LR) to +0.080 (GBM) AUC, and the gains are the same on the chronological split as in the grouped CV (0.878 against 0.878), so the signal is not leakage between sessions of the same day. In S14 the equivalent contrast was 0.59-0.65 (tool name) against 0.74-0.85 (context) on writes only.

### E'. Next-tool-name speculation (S14 E equivalent)

Positions: every tool call that has a following call in the same session. Train 1598165 positions (day 1), test 1634662 (day 2). Target = next call's tool name, vocabulary = 24 most common day-1 tools + OTHER (1.2% of test targets are OTHER and count as misses; OTHER is never predicted). There are 855 distinct names on day 1 but the top 24 carry 98.8% of next calls. The GBM is fitted on 200000 random day-1 positions.

Top-1 / top-3 accuracy (%):
| positions | n | (a) most common tool | (b) bigram | (b) trigram | (b) 4-gram | (c) GBM |
|---|---|---|---|---|---|---|
| all positions | 1634662 | 35.0 / 61.7 | 45.3 / 73.1 | 47.8 / 77.2 | 49.0 / 79.0 | 48.1 / 79.5 |
| next call opens a new turn (after a user message) | 123904 | 37.6 / 60.5 | 37.8 / 60.5 | 37.8 / 66.9 | 37.1 / 68.4 | 36.3 / 68.6 |
| inside a turn, current call in the first batch | 172330 | 38.8 / 59.4 | 46.6 / 76.1 | 47.9 / 78.0 | 48.1 / 79.4 | 46.5 / 79.8 |
| inside a turn, later batch | 1338428 | 34.2 / 62.1 | 45.9 / 73.9 | 48.7 / 78.1 | 50.2 / 79.9 | 49.4 / 80.4 |
| next call in the same batch (parallel) | 166488 | 64.9 / 71.6 | 73.6 / 89.4 | 72.5 / 90.6 | 72.1 / 90.8 | 72.7 / 91.2 |
| model Model A | 467659 | 35.4 / 69.6 | 48.6 / 75.2 | 50.9 / 79.3 | 52.0 / 81.0 | 51.1 / 81.1 |
| model Model B | 212280 | 34.1 / 64.1 | 42.4 / 75.1 | 45.2 / 77.6 | 46.5 / 78.8 | 45.3 / 78.7 |
| model Model C | 174823 | 35.5 / 68.1 | 46.9 / 75.4 | 49.3 / 78.6 | 50.1 / 80.4 | 49.1 / 80.1 |
| model Model D | 148442 | 37.6 / 50.5 | 47.0 / 70.0 | 48.9 / 74.6 | 49.9 / 76.4 | 48.8 / 76.4 |
| model Model E | 132518 | 33.6 / 62.8 | 42.5 / 75.3 | 45.9 / 78.1 | 47.2 / 79.6 | 46.2 / 79.3 |

Gate on the predicted probability of the top-1 tool (launch the speculation only if p >= tau). n-gram p = count share of the top tool in its context; GBM p = predicted probability. `correct launches / all positions` is the share of all positions where a correct speculation was launched:
| predictor | tau | launched share | precision of launched | correct launches / all positions |
|---|---|---|---|---|
| bigram (last tool) | 0.5 | 57.3% | 53.1% | 30.4% |
| bigram (last tool) | 0.7 | 0.0% | n/a | 0.0% |
| bigram (last tool) | 0.9 | 0.0% | n/a | 0.0% |
| trigram (last 2 tools, backoff to bigram if < 5 seen) | 0.5 | 50.3% | 58.5% | 29.4% |
| trigram (last 2 tools, backoff to bigram if < 5 seen) | 0.7 | 10.5% | 71.1% | 7.5% |
| trigram (last 2 tools, backoff to bigram if < 5 seen) | 0.9 | 0.0% | 28.6% | 0.0% |
| 4-gram (last 3 tools, backoff trigram then bigram) | 0.5 | 48.7% | 60.2% | 29.3% |
| 4-gram (last 3 tools, backoff trigram then bigram) | 0.7 | 9.5% | 75.6% | 7.2% |
| 4-gram (last 3 tools, backoff trigram then bigram) | 0.9 | 0.1% | 89.2% | 0.1% |
| GBM (last 3 tools, status, position, turn, model, batch size) | 0.5 | 40.5% | 62.4% | 25.3% |
| GBM (last 3 tools, status, position, turn, model, batch size) | 0.7 | 16.1% | 69.7% | 11.2% |
| GBM (last 3 tools, status, position, turn, model, batch size) | 0.9 | 4.3% | 50.4% | 2.2% |

Trigram gate at tau 0.7 by position type: all positions: launched 10.5%, precision 71.1%; next call opens a new turn (after a user message): launched 8.6%, precision 49.7%; inside a turn, current call in the first batch: launched 4.9%, precision 74.4%; inside a turn, later batch: launched 11.4%, precision 72.4%

Batch level (day 2; 7.1% of 1508330 batches have 2+ tools; among those, 62.2% contain a single distinct tool). Given the first tool of a batch (and the fact that the batch has a slot there, i.e. its size is at least the slot index; the size is also used to split the columns), predict each of the other slots. Top-1 / top-3 accuracy (%) over all non-first slots (166488):
| predictor | all non-first slots | batch of 2 | batch of 3-4 | batch of 5+ |
|---|---|---|---|---|
| unconditional (mode of non-first slots) | 64.9 / 80.2 | 51.6 / 77.7 | 76.9 / 85.5 | 71.9 / 73.0 |
| P(tool given previous tool in the batch) | 76.4 / 92.9 | 59.9 / 87.8 | 87.0 / 96.7 | 96.2 / 97.6 |
| P(tool given first tool and slot index) | 75.6 / 92.5 | 60.1 / 87.8 | 85.4 / 95.9 | 94.8 / 97.2 |
| rest identical to the first tool | 71.8 / 71.8 | 56.0 / 56.0 | 80.8 / 80.8 | 93.5 / 93.5 |

All non-first slots of a batch right (argmax of P(tool given first tool and slot index)): 66.1% of multi-tool batches; baseline 'rest identical to the first tool': 62.2%.

Reading it. S14's benchmark (retail and airline) gave top-1 name accuracy of 55.2-67.1% when the workflow was re-run and 51.6-63.0% for new tasks; here the best is 49.0% (4-gram) over all positions. Per model, the trigram, 4-gram and GBM top-1 accuracies range from 45.2% to 52.0%. A trigram with 71% precision on 10.5% of positions is the safe regime; S14's 93-95% precision at 26-44% launched came from re-run workflows with argument-level agreement and is not reproduced. Within a parallel batch, the next tool is much more predictable (72.7-73.6% top-1, 76.4% when conditioning on the previous tool in the batch), but only 7.1% of batches have 2+ tools, so this covers a small share of calls. Calls in one batch are issued together at the same start time, so there is no wait to hide: this is a structural statistic, not a speculation win.

Calibration. The gate scores are not calibrated probabilities. Precision falls as the threshold rises: GBM 69.7% at tau 0.7 against 50.4% at 0.9, trigram 71.1% against 28.6%. A GBM refitted on held-out day-1 rows shows the same pattern (71.0% at 0.7, 52.9% at 0.9), so this is miscalibration, not day-to-day drift. Positions with p >= 0.9 spread over 17.5k sessions, and 46% of them are same-batch positions. The tau 0.7 operating point is set empirically.

### F'. Latency (S14 F equivalent)

Timestamp convention check (day 1): a batch's start (timestamp - duration) is no earlier than the issuing LLM call's timestamp in 100.0% of pairs, and the next LLM call's start (timestamp - duration) is no earlier than the batch end in 98.7% (median gap 229 ms). So timestamps are completion times. Each tool call is issued at its batch start and finishes at batch start + its own duration.
Tool calls: day 1 1635686 (sessions 37521), day 2 1674818 (sessions 40156). Calls with duration > 1 s: 2026-06-01 23.7% of calls / 99.1% of summed tool time, 2026-06-02 23.8% of calls / 98.9% of summed tool time, pooled 23.8% of calls / 99.0% of summed tool time; > 10 s: 2026-06-01 5.96% of calls / 95.2% of time, 2026-06-02 6.06% of calls / 94.3% of time, pooled 6.01% of calls / 94.8% of time. Summed over parallel calls (not elapsed time). Per batch (wall time of the whole batch, 3013683 batches): > 1 s 24.8% of batches / 99.1% of batch time, > 10 s 6.47% / 95.1%. Duration quantiles pooled: median 169 ms, p90 4946 ms, max 5.32e+08 ms.

Per-tool latency, top 15 tools by calls (both days):
| tool | calls | median ms | p90 ms | % calls > 1 s | % calls > 10 s | % of its time in calls > 1 s | % of all tool time |
|---|---|---|---|---|---|---|---|
| get_file | 1158566 | 68 | 439 | 4.8 | 0.43 | 85.6 | 2.0 |
| run_command_in_terminal | 549188 | 379 | 20219 | 40.4 | 15.82 | 99.7 | 56.3 |
| replace_string_in_file | 328188 | 258 | 1077 | 10.9 | 0.89 | 89.6 | 1.7 |
| code_search | 266614 | 1759 | 5665 | 73.2 | 4.61 | 94.0 | 1.8 |
| run_build | 163196 | 5040 | 45633 | 77.0 | 33.85 | 100.0 | 28.5 |
| file_search | 152659 | 58 | 173 | 1.0 | 0.07 | 54.6 | 0.1 |
| apply_patch | 112276 | 295 | 3171 | 18.7 | 5.64 | 92.3 | 0.7 |
| create_file | 97891 | 582 | 2847 | 31.0 | 2.19 | 92.1 | 0.8 |
| update_plan_progress | 81443 | 18 | 187 | 1.0 | 0.06 | 36.7 | 0.0 |
| multi_replace_string_in_file | 68949 | 360 | 1669 | 18.2 | 1.07 | 87.6 | 0.3 |
| get_errors | 63992 | 63 | 233 | 1.6 | 0.08 | 53.0 | 0.0 |
| edit_file | 39002 | 4931 | 27277 | 91.9 | 28.17 | 99.8 | 1.1 |
| get_files_in_project | 28087 | 66 | 234 | 1.2 | 0.12 | 48.5 | 0.0 |
| get_projects_in_solution | 25469 | 95 | 274 | 1.6 | 0.07 | 29.2 | 0.0 |
| find_symbol | 25123 | 219 | 1534 | 18.7 | 0.98 | 91.7 | 0.1 |

Test = day 2 (1674818 calls, 40156 sessions never seen in training); train = day 1; GBMs fitted on 300000 random day-1 calls. Errors in ms; `MAE clip` clips true and predicted values at 1 h (a few multi-hour calls dominate raw MAE).

| predictor | MAE ms | MAE clip 1 h | median abs err | p90 abs err ms | mean abs log1p-ratio |
|---|---|---|---|---|---|
| global median | 13030 | 6776 | 134 | 4802 | 1.60 |
| per-tool median (day 1) | 12790 | 6536 | 125 | 4234 | 1.10 |
| per-tool EWMA over the stream (alpha 0.1) | 12832 | 6578 | 155 | 3808 | 1.15 |
| per-session same-tool EWMA (0.5) | 16341 | 7088 | 92 | 4143 | 0.85 |
| GBM, call-time features only | 12753 | 6499 | 111 | 4042 | 1.03 |
| GBM + session history | 12422 | 6169 | 71 | 3491 | 0.75 |

Log-error gain of GBM + history over per-tool median: 0.352 (95% session-bootstrap CI 0.347 to 0.356).

Will this call take > 1 s? Test base rate 23.8% of calls. AUC, and precision when flagging as many calls as there are slow ones (k = 399371):
| predictor | AUC | precision at top-k |
|---|---|---|
| per-tool slow rate | 0.873 | 64% |
| GBM call-time features | 0.884 | 67% |
| GBM + session history | 0.942 | 79% |

Same-day random hold-out of 20% of day-1 sessions (326032 calls) for comparison, mean abs log1p-ratio: per-tool median 1.09, GBM call-time 1.02, GBM + history 0.75 (day-1 to day-2 values are in the table above).

By model (top 5 by test calls), mean abs log1p-ratio:
| model | test calls | median ms | per-tool median | GBM call-time | GBM + history |
|---|---|---|---|---|---|
| Model A | 478101 | 158 | 1.08 | 1.03 | 0.76 |
| Model B | 217909 | 159 | 1.09 | 1.04 | 0.71 |
| Model C | 179344 | 172 | 1.08 | 1.03 | 0.74 |
| Model D | 152276 | 193 | 1.12 | 0.97 | 0.74 |
| Model E | 135439 | 157 | 1.03 | 0.98 | 0.71 |

Reading it. The ladder differs from S14 in two ways. First, here per-tool statistics are strong (log error 1.10 against 1.48 in S14) because the tool vocabulary is rich (855 names) and day 1 is close to day 2; the drift gap seen in S14 (static GBM worse than per-tool median on later sessions) does not appear (GBM call-time 1.03 against 1.10, and 1.02 against 1.09 on a same-day hold-out). Second, the per-session same-tool EWMA is the best baseline in median error (92 ms) and log error (0.85) but the worst in raw MAE (16,341 ms), because one multi-hour call poisons the average; the clipped MAE (7,088 ms) is also the worst. GBM with session history improves the per-tool median by 2.9% in MAE, 17.5% in p90 error and 32% in log error (gain 0.352, 95% CI 0.347-0.356); in S14 the corresponding cuts were 11%, 30% and 33%. The ">1 s" AUC rises from 0.873 (per-tool slow rate) to 0.942, against 0.717 to 0.884 in S14.

### T. Where the time goes, and the ceiling for speculating the next LLM call (new in S14b)

Time split inside turns (a turn runs from its first LLM call's start to its last event's end; between-turn user idle time is excluded here). Shares are of summed active turn time; per-session rows use each session's own turn time:
| share of active turn time | day 1 | day 2 |
|---|---|---|
| LLM calls (union of intervals) | 39.6% | 40.8% |
| tool batches (union) | 35.3% | 41.9% |
| both at once (overlap) | 0.3% | 0.4% |
| neither (gaps inside a turn: client work, waits) | 25.5% | 17.7% |
| only LLM | 39.2% | 40.4% |
| only tool | 35.0% | 41.5% |
| per-session median: LLM / tool / gap | 86.8% / 3.7% / 1.9% | 86.6% / 3.7% / 1.9% |

Estimate of what next-step speculation can hide under S19's rule, assuming a perfect predictor (next LLM step starts when the tool batch is issued and is committed when the tool finishes):
| quantity | day 1 | day 2 |
|---|---|---|
| active turn time (hours) / sessions / turns | 16889 / 55870 / 222774 | 13038 / 60144 / 235007 |
| between-turn user idle, share of session wall clock | 73.8% | 79.9% |
| per-session tool share of turn time, p10 - p90 | 0.0% - 47.3% | 0.0% - 47.6% |
| S19-rule estimate of time saved with a perfect predictor, min(tool batch, next LLM step): share of active turn time | 2.91% | 3.91% |
|   same, share of session wall clock incl. user idle | 0.77% | 0.79% |
|   same, per session median / p90 | 2.56% / 12.61% | 2.54% / 12.51% |
|   same, batches under 1 h only | 2.91% | 3.90% |
| robust variant: drop turns that contain a batch over 1 h (share of active time kept) -> LLM / tool / gap shares | 79.3% kept -> 49.8% / 18.5% / 32.1% | 77.9% kept -> 52.2% / 25.6% / 22.7% |
|   same robust subset: S19-rule estimate of time saved | 3.64% | 4.98% |
| looser bound: next LLM call starts at batch issue (saves its whole wait), share of active turn time | 26.99% | 35.25% |

Reading it. Per-session medians are dominated by sessions in which the LLM does everything (median 86.7% LLM, 3.7% tool): speculation cannot help those, and the tool share is highly skewed (p10 0%, p90 47-48%). The perfect-predictor estimate by S19's rule, min(tool batch, next LLM step) summed over batches, is 2.91% of active turn time on day 1 and 3.91% on day 2; dropping turns that contain a batch over 1 hour (21-22% of the time) raises it to 3.64% and 4.98% because the retained time is more evenly split. Against the full session wall clock including user idle time the estimate is 0.77-0.79%. The looser bound that lets the next LLM call start at batch issue and save its whole wait is 27.0% and 35.3%; it needs the next call's content to be right, and this corpus cannot measure that hit rate (E' predicts the next tool name, not the tool output the next call depends on).

The S19 rule is not a true ceiling: it omits the wait between batch end and the next LLM start, which a perfect predictor also hides. From a separate check with the same join (not in results.json), as a share of active turn time: rule 2.91% / 3.91%, the wait alone 10.2% / 13.9%, real ceiling 13.1% / 17.9%, real ceiling with waits capped at 60 s 6.2% / 8.2% (day 1 / day 2). About 79% of wait time is in waits over 60 s and the median wait is 229 ms. The comparison with S19 is to its perfect-predictor range of 1.1-6.4%, which the rule values (2.9-5.0%) fall inside; S19's 0.5-3% is the saving at S14's hit rates and is not comparable. The tool share is driven by few turns: dropping the 0.2% of turns with a batch over 1 h keeps 79.3% / 77.9% of the time and cuts the tool share to 18.5% / 25.6% (from 35.3% / 41.9%).

### X. Cross-dataset latency check against TraceLab

TraceLab v0.0.2: 743819 tool records, 743050 with wall latency; chronological split by session start as in S14 F: 546549 train / 196501 test calls. Copilot: train day 1 (1635686 calls), test day 2 (1674818 calls). History-only model = HistGradientBoosting (absolute-error loss on log1p latency, 300000 training rows) on tool-agnostic causal features only: latencies of the last 3 finished calls of the session, session EWMA (0.5), log number of finished calls. No tool name, no model, no time. Slow base rate (> 1 s): Copilot day 2 23.8%, TraceLab test 38.1%. AUC scores the > 1 s question with the predicted latency itself.

| test set | predictor | MAE ms | p90 abs err ms | mean abs log1p-ratio | AUC > 1 s | log-error gain vs own per-tool median (95% CI) |
|---|---|---|---|---|---|---|
| Copilot day 2 | own global median | 13030 | 4802 | 1.60 | 0.500 | - |
| Copilot day 2 | own per-tool median | 12790 | 4234 | 1.10 | 0.864 | - |
| Copilot day 2 | history-only GBM, trained on the same dataset | 12993 | 4578 | 1.47 | 0.634 | -0.37 (-0.38 to -0.36) |
| Copilot day 2 | history-only GBM, trained on TraceLab | 13031 | 4492 | 1.54 | 0.633 | -0.44 (-0.44 to -0.43) |
| TraceLab test sessions | own global median | 14382 | 11154 | 2.04 | 0.500 | - |
| TraceLab test sessions | own per-tool median | 14275 | 12091 | 1.48 | 0.734 | - |
| TraceLab test sessions | history-only GBM, trained on the same dataset | 13695 | 9422 | 1.44 | 0.798 | +0.03 (-0.08 to +0.14) |
| TraceLab test sessions | history-only GBM, trained on Copilot day 1 | 14148 | 10666 | 1.61 | 0.758 | -0.13 (-0.18 to -0.09) |

Reading it. A model that sees only the session's recent latencies (no tool name, model or time) does not beat the per-tool median on Copilot (log error 1.47 against 1.10, gain -0.37, CI -0.38 to -0.36), and one trained on TraceLab is worse still (1.54). On TraceLab, the same history-only model matches the per-tool median within noise (gain +0.03, CI -0.08 to +0.14) and has a higher slow-call AUC (0.798 against 0.734); trained on Copilot day 1 it is worse (1.61 and gain -0.13). So the S14 conclusion "recent latencies beat the tool name" holds only in combination with the tool name (F': 0.75 against 1.10 on Copilot); it does not hold alone, and models do not transfer across the two corpora. Note that the per-tool-median AUC in this table (0.864) scores the predicted latency itself, while F' uses the per-tool slow rate (0.873).

## Transfer to Dreaming

| Tool case in missions.md | Number from S14b that supports | Number from S14b that undercuts |
|---|---|---|
| Call changes the world or can fail (e.g. a build or command) | Failure is predictable from the session so far: AUC 0.878 (GBM + context) against 0.798 (per-tool rate) chronologically; flagging the top 20% of calls captures 76.8% of failures (precision 18.0%, base rate 4.7%); the same on both validation schemes; after a failed call the next fails 16.9% | The tool name carries most of the signal (0.798), though context still adds +0.080 AUC with a tight CI; precision at top 20% is only 18%, so it is a triage score, not a veto; failures are clustered (top 5% of sessions hold 43.8% of failures) |
| Next-call speculation | Trigram gate at 0.7 launches 10.5% of positions with 71.1% precision (7.5% of all positions correct) | Within a parallel batch the 72.7-73.6% top-1 is structural (calls start together, nothing to hide); overall top-1 is 47.8-49.0% and 36-38% after a user message; metadata gives no arguments, so only the name is guessed; the best gated GBM is right on 11.2% of all positions at 69.7% precision |
| Slow tools / latency | 24% of calls carry 99% of tool time; the per-tool median plus session history lowers the log error from 1.10 to 0.75 and p90 error 17.5%; "> 1 s" AUC 0.942 | MAE gain is only 2.9%; history alone is worse than the per-tool median on this corpus (X) and does not transfer between corpora |
| Hiding LLM latency by speculating the next step | S19-rule estimate 2.9-3.9% of active turn time with a perfect predictor (0.8% of wall clock), inside S19's 1.1-6.4%; 6-8% if waits up to 60 s are hidden, 13-18% at most | The larger figures need the next LLM call's content to be right and the waits to be hideable (79% of wait time is in waits over 60 s); this corpus cannot measure the hit rate |

**Does tool prediction hold up on a large, heterogeneous corpus?** Failure prediction and latency prediction do, in the form "tool name plus the session's recent history" and with a triage-level precision; next-call name prediction holds up only behind a 0.7 gate on a tenth of the positions (within-batch accuracy is a structural statistic, not a gain). The time-share result gives a ceiling that does not depend on the model: about 3-4% of active turn time under the S19 rule, 6-8% if waits up to 60 s can be hidden and up to 13-18% at most, so the value of speculation on coding traces depends on whether those waits are machine waits.

## Caveats

- **Repeat rate and skip rate cannot be measured on this corpus.** There are no arguments and no return values, so the certified-cache analyses of S14 A-C (repeat rate, skip rate, realized error) have no equivalent here. Nothing in this smoke says how many calls could be skipped.
- E' guesses the tool name only; argument-level hits (S14's 9-73% across retail and airline) are unknowable. The OTHER class (1.2% of test targets) is counted as a miss. S14's trajectory cache (exact earlier calls) cannot be built without arguments, so there is no workflow re-run regime.
- The E' gate scores (n-gram shares, GBM probabilities) are not calibrated; tau 0.7 is an empirical operating point (see the calibration note in E').
- The T ceiling figures with the wait (13.1% / 17.9%, 6.2% / 8.2%) come from a reviewer's check script, not from results.json.
- Status 0 (3.6%) is undocumented; its calls are excluded from D' labels. If it hides failures, the D' base rate is understated.
- Only two days of one product, anonymised models; day 2 follows day 1 directly, so long-term drift (S14's weeks-long drift) is not tested. Chronological transfer here is one day.
- Wall latency includes waiting for permissions, humans and sub-agents; a few multi-hour calls dominate raw MAE (max 5.3e8 ms), hence the clipped MAE. The per-tool and per-session medians are less affected.
- T uses completion timestamps and the ordering check (98.7% of next-LLM starts at or after the batch end); the remaining 1.3% overlap slightly, which can shift the shares by tenths of a point at most (the overlap row is 0.3-0.4%). The ceiling assumes the speculated LLM call finishes with the same latency as the real one and ignores the cost of wrong speculations.
- The cross-dataset check (X) compares a tool-agnostic history model only. TraceLab is not the same product and has fewer users (52).
- Precision/recall at top 20% in D' use random tie-breaking; the per-tool table counts both days.
- The work order mentioned `log_STX.txt` and `log_EDT.txt`; they are not in the folder, and `log.txt` covers both runs.
- Figures in "In plain words" are the same as in the tables; no number appears that is not in `results.json` or `tables_*.md`, except the derived ratios stated explicitly (2.9% MAE cut = 12,422 against 12,790 ms; 17.5% p90 cut = 3,491 against 4,234 ms; the 4.5 times corpus size comparison = 3.31 M against 0.74 M tool records).

## Time and cost

- API spend: $0.00. No pods; local machine only (4 CPUs, 15 GB).
- Download: day 2 archive (621 MB) and TraceLab (101 MB) with `fetch.py` (about 5 s each at 25 MB/s).
- Flatten: about 2 min per day (4 processes).
- `run.py`: first run loaded 3,310,504 calls in 17 s, F in 194 s, E in 243 s (stopped at D by a column-overlap bug, since fixed); second run D 477 s, S 494 s, T 652 s, X 720 s (12 min), history features read from the F cache. About 16 min of compute in total for the sections, none above 8 min.
- Files: `run.py`, `common14b.py`, `flatten.py`, `fetch.py`, `results.json` (all numbers), `tables_{D,E,F,S,T,X}.md`, `log.txt`, `fetch_log.txt`.

## Revision

Review pass after the first draft. No pipeline re-run; tables unchanged. Changes: (1) T reworded: the S19 rule is an estimate, not an upper bound, and the wait between batch end and next LLM start adds up to 13.1% / 17.9% (6.2% / 8.2% with waits capped at 60 s); (2) the S19 comparison is now against its perfect-predictor range of 1.1-6.4%; (3) the E' cross-reference to T removed and within-batch accuracy described as structural; (4) tool share noted as driven by the 0.2% of turns with a batch over 1 h; (5) calibration note for the E' gate; (6) wording fixes: no better than the most common tool, S14 benchmark ranges (55.2-67.1%, 51.6-63.0%, 9-73%), session counts, D' context gain +0.080 AUC with a tight CI.
