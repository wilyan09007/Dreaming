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

Upper bound on what next-step speculation can hide, assuming a perfect predictor (next LLM step starts when the tool batch is issued and is committed when the tool finishes):
| quantity | day 1 | day 2 |
|---|---|---|
| active turn time (hours) / sessions / turns | 16889 / 55870 / 222774 | 13038 / 60144 / 235007 |
| between-turn user idle, share of session wall clock | 73.8% | 79.9% |
| per-session tool share of turn time, p10 - p90 | 0.0% - 47.3% | 0.0% - 47.6% |
| upper bound on time saved, perfect prediction, S19 rule min(tool batch, next LLM step): share of active turn time | 2.91% | 3.91% |
|   same, share of session wall clock incl. user idle | 0.77% | 0.79% |
|   same, per session median / p90 | 2.56% / 12.61% | 2.54% / 12.51% |
|   same, batches under 1 h only | 2.91% | 3.90% |
| robust variant: drop turns that contain a batch over 1 h (share of active time kept) -> LLM / tool / gap shares | 79.3% kept -> 49.8% / 18.5% / 32.1% | 77.9% kept -> 52.2% / 25.6% / 22.7% |
|   same robust subset: upper bound on time saved (S19 rule) | 3.64% | 4.98% |
| looser bound: next LLM call starts at batch issue (saves its whole wait), share of active turn time | 26.99% | 35.25% |
