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
