TraceLab v0.0.2 (CC BY 4.0): 743819 tool records, 7244 sessions, 52 users, providers {'codex': 438814, 'claude': 305005}; records with wall latency 99.9%, with tool-reported internal latency 46.5%.
Chronological split by session start: 5070 train sessions (546549 calls, up to 2026-07-01) / 2174 test sessions (196501 calls, from 2026-07-01). History features (previous latencies, session EWMA) only use calls that had already finished when the new call was emitted.
All calls: median latency 218 ms, mean 15606 ms, p90 10052 ms; calls > 1 s = 33.8% of calls but 99.3% of summed tool time (summed over parallel calls, so not elapsed time); calls > 10 s = 11.21% of calls, 94.7% of time.
| tool | calls | median ms | p90 ms | % calls > 1 s | % of tool time in calls > 1 s |
|---|---|---|---|---|---|
| exec_command | 251607 | 265 | 1600 | 26.8 | 93.1 |
| Bash | 158471 | 329 | 20404 | 41.4 | 99.6 |
| write_stdin | 75952 | 5006 | 44715 | 69.5 | 99.8 |
| Read | 63703 | 27 | 245 | 2.8 | 92.1 |
| exec | 46756 | 1457 | 10050 | 61.3 | 97.3 |
| Edit | 36533 | 75 | 1997 | 17.5 | 98.9 |
| apply_patch | 32330 | 45 | 1218 | 12.5 | 97.1 |
| wait | 11404 | 11903 | 18392 | 79.9 | 99.8 |
| TaskUpdate | 8824 | 50 | 327 | 6.8 | 92.1 |
| Write | 8602 | 68 | 2105 | 19.8 | 99.4 |
| shell_command | 7623 | 51 | 867 | 8.3 | 93.1 |
| shell | 5416 | 0 | 1 | 0.0 | 0.0 |
| TaskCreate | 4877 | 45 | 2660 | 13.8 | 94.1 |
| WebFetch | 3636 | 6260 | 22628 | 92.8 | 99.9 |
| update_plan | 3473 | 21 | 65 | 0.1 | 6.0 |
| Grep | 3355 | 130 | 367 | 5.2 | 87.8 |

Test-set error of point predictions (ms). MAE is dominated by the heavy tail (subagent, wait and long shell calls); the log error is the scale-free view.
| predictor | MAE ms | median abs err ms | p90 abs err ms | mean |log1p ratio| | median |log1p ratio| |
|---|---|---|---|---|---|
| global median | 14382 | 190 | 11154 | 2.04 | 1.72 |
| per-tool median | 14275 | 357 | 12091 | 1.48 | 1.20 |
| per-(tool, executable) median | 14256 | 790 | 12678 | 1.45 | 1.13 |
| per-tool EWMA over the stream (alpha 0.1) | 13448 | 445 | 8827 | 1.18 | 0.84 |
| per-session same-tool EWMA (0.5) | 13520 | 252 | 8692 | 1.05 | 0.53 |
| GBM, call-time features only (tool, executable, input size, hour) | 14006 | 423 | 10359 | 1.75 | 1.25 |
| GBM + session history (previous latencies) | 12673 | 223 | 8515 | 0.99 | 0.57 |

Same, excluding the long-wait tools ['Agent', 'AskUserQuestion', 'ExitPlanMode', 'ScheduleWakeup', 'Skill', 'Task', 'TaskOutput', 'wait', 'write_stdin'] (88% of test calls kept):
| predictor | MAE ms | p90 abs err ms | mean |log1p ratio| |
|---|---|---|---|
| per-tool median | 7931 | 8405 | 1.30 |
| per-tool EWMA | 7559 | 6868 | 1.13 |
| per-session same-tool EWMA | 7617 | 6773 | 1.00 |
| GBM, call-time features only | 8021 | 9921 | 1.61 |
| GBM + session history | 7215 | 6057 | 0.94 |

Will this call take > 1 s? Test base rate 38.1% of calls (99% of test tool time). AUC and precision when flagging as many calls as there are slow ones:
| predictor | AUC | precision at top-k (k = #slow) |
|---|---|---|
| per-tool slow rate | 0.717 | 60% |
| GBM call-time features | 0.769 | 61% |
| GBM + session history | 0.884 | 75% |

Shell tools only (Bash/exec_command, 84266 test calls): MAE per-tool median 11676 ms, per-(tool, executable) median 11633 ms, GBM call-time 11525 ms, GBM+history 10954 ms; p90 abs err 6581 / 7240 / 6969 / 6470 ms.

Ablation of the call-time GBM on the later (test) sessions:
| GBM features | MAE ms | mean |log1p ratio| |
|---|---|---|
| tool, executable, provider | 14306 | 1.73 |
| + input size | 13926 | 1.78 |
| all call-time features (+ n_exe, position in session, hour) | 14006 | 1.75 |
| (reference) per-tool median | 14275 | 1.48 |

Drift check, same period (hold out a random 20% of TRAIN-period sessions, 120902 calls): mean |log1p ratio| per-tool median 1.72, call-time GBM 1.49 (MAE 11349 vs 10604 ms). If the GBM wins here but not on later sessions, the gap is drift, not a bug.