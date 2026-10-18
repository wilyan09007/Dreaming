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
