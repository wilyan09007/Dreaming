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
