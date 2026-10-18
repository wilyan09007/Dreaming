# S28 Router certificate versus calibration size (Banking77, cheap vs expensive model)

## In plain words

The router sends a call to a cheap model when a certificate says the cheap answer will equal the expensive model's answer (E1, here gpt-5.4 at low reasoning effort) at a stated error budget. The certificate is fitted on calibration rows, calls where both models answered. S15 showed that 100 rows certify nothing at 2-5% and guessed that 5-10% budgets need 500-1,000 rows. This smoke measured it: 1,129 Banking77 test inputs answered by both E1 and the cheap model (gpt-5.4-nano with token probabilities), then 200 random calibration sets of 100, 200, 300, 500 and 1,000 rows per size, each certified and checked on the rows left out.

With the best gate (the free student model agrees with the cheap model, the cheap model's own probability gap breaks ties inside the agree group; "f3"), the rule fixed in `run.py` before looking (the smallest calibration size that certifies in at least 95% of draws and reaches 90% of the coverage seen at 1,000 rows) recommends **1,000 / 300 / 200 / 200 rows for budgets of 5 / 10 / 15 / 20%**. At those sizes the cheap model takes 71 / 78 / 83 / 89% of calls and the blended cost is 65 / 72 / 77 / 83% below always-E1. A looser test, certifying in at least 95% of draws with no condition on coverage, would stop earlier, at 500 / 200 / 200 / 100 rows (cheap model on 62 / 72 / 83 / 83% of calls, 57 / 66 / 77 / 77% cheaper), but at the 5, 10 and 20% budgets coverage there is under 90% of the 1,000-row coverage; only the 15% budget passes both tests at 200 rows. More rows add coverage slowly: 1,000 rows reach 71 / 83 / 89 / 96% (65 / 77 / 83 / 90% cheaper). S15's guess holds for the 5% budget and was too pessimistic for 10%, which needs 200-300 rows. The 5% budget is also below E1's own 6% repeat disagreement and has not been checked against a second E1 draw (see Transfer), so treat the 5% row as unproven. Without logprobs (agreement alone, "f1") the certificate exists at 15-20% from 200 rows, at 10% from 500, and essentially never at 5% (1% of draws at 300 rows, none at 100, 200, 500, 1,000). The student's own confidence alone certifies nothing at 5-10% (bar one draw) and needs 1,000 rows for 15%.

The two models agree on 81.5% of inputs; against the true labels the cheap model is 76.2% right, E1 85.7%, and the free student (trained on all 9,993 gold training rows) 92.9%. The router here is a measurement vehicle for calibration size; at this call site the student alone already beats both models on gold (see Caveats).

## Setup

- **Rows.** 1,129 = S5's 200 stratified test inputs (E1 from `s5_tier_router/responses.jsonl`, cheap model from `s15_confidence_signals/responses.jsonl`, both reused) + 929 new test inputs (seed 28, drawn from the other 2,880 test rows, round-robin over the 77 labels). 1,000 new inputs were drawn and the cheap model answered all 1,000, but **E1 answered only 929: the last 71 E1 calls could not be made** (OpenRouter returned HTTP 402 "exceeds your available credits"; the account balance was about $0.03 and the 71 calls cost about $0.12). They are missing for a platform reason, not a content reason (60 different labels). The new-row column below is therefore 929 rows, not 1,000 (`log.txt` says "new-1000"; read it as 929). The 929 used rows cover all 77 labels, 11-13 each. Cheap-model gold accuracy on the 71 missing inputs is 0.859 against 0.764 on the used ones, so the headline match rate may be slightly low; that is the only selection effect seen.
- **Models and prompts.** Identical system prompt and nearest-label parser as S5/S15. E1 = openai/gpt-5.4, reasoning effort low, provider pinned to OpenAI with fallbacks off on the 929 new calls (S5's E1 request carried no provider pin; all 200 reused S5 calls and all 929 new ones were nonetheless served by OpenAI); max_tokens 800 on these calls (S5 used 2,000; the largest completion seen was 309 tokens). Cheap = openai/gpt-5.4-nano, reasoning none, OpenAI, temperature sent as 0 (ignored by the model, so every call is one sample), `logprobs=True, top_logprobs=5`. Cost is the `usage.cost` field of each response.
- **Target.** "cheap label == E1 label" (one E1 draw per row; E1 disagrees with its own repeat on 6% of inputs, S15).
- **Signals (higher = more confident), as in S15 Part 2.** f1: the shadow student's label equals the cheap label (student = bge-small-en-v1.5 embeddings + logistic regression C=10 on all 9,993 Banking77 training rows, free). b2: the cheap label's log-probability minus the log-mass of the best rival label among the returned top-5 alternatives (floor 12 nats when no rival is listed). f3: f1 first, b2 breaks ties inside each group (score = 1000 x f1 + b2). e: the student's own top-2 gap (needs no cheap call). The student is built as in S15 but is not byte-identical to S15's: on the same 200 rows its AUROCs here vs S15's are f1 0.900 vs 0.897, e 0.671 vs 0.665 and f3 0.958 vs 0.955 (b2 identical), probably from re-embedding the texts for this run (not checked). 68% of cheap answers list no rival label; those rows sit at the b2 floor but are not tied, because b2 = 12 + the label's logprob (560 distinct values among the 765 rows).
- **Certificate.** `common.certify`: 40 coverage levels, Clopper-Pearson upper bound at delta 5% / 40 per level, largest level whose bound is <= budget, ties closed. Budgets 5, 10, 15, 20%.
- **Sweep.** For each n in 100, 200, 300, 500, 1,000: 200 random splits (`default_rng(2800 + s)`), calibration = n rows, holdout = the other 1,129 - n rows. n = 1,000 was planned with a 200-row holdout (1,200 rows); with the 71 missing E1 labels it leaves 129, below that floor, so a supplementary n = 929 row (holdout exactly 200, same seeds) is reported beside it and the n = 1,000 rows are marked *. Reported per cell: cert% = share of splits where any coverage certified; cov = share of the holdout routed to the cheap model (0 when nothing certified), mean over splits and [5th percentile]; realized mismatch among routed holdout rows, mean over certifying splits (max); viol% = share of certifying splits whose realized holdout mismatch exceeded the budget; saving = 1 - blended $/1k / E1 $/1k at the mean coverage, where blended = cheap + (1 - cov) x E1 for f1/b2/f3 (the cheap call is made first), and cov x cheap + (1 - cov) x E1 for e. A split that certifies nothing still pays the cheap call, hence -6%.
- **AUROC.** Over all 1,129 rows, 1,000-resample bootstrap 95% CI; paired differences on the same resamples.
- **Recommended n (rule fixed before looking, in `run.py`).** Smallest n where >= 95% of splits certify and mean coverage >= 90% of the n=1,000 coverage. The cert%-only test (>= 95% of splits certify, nothing about coverage) is reported separately as the looser test.

## Results

### Models on these rows

| quantity | value |
|---|---|
| match rate, cheap == E1 | 0.815 [Wilson 95% 0.791, 0.836]; the S5 200 rows 0.800, the 929 new rows 0.818 (S15: 0.800 on the 200) |
| gold accuracy | E1 0.857 (S5 200: 0.870, new: 0.854); cheap 0.762; student 0.929 |
| student == E1 / student == cheap | 0.849 / 0.764 |
| observed $/1k calls | E1 $1.732; cheap $0.098; ratio 17.7x (new rows: E1 $1.735, cheap $0.098) |
| E1 self-disagreement floor (S15, 200 inputs, repeat run) | 6% |
| cheap answers with no rival label in the top-5 alternatives (b2 at its 12-nat floor) | 765 of 1,129 (68%), mismatch with E1 4.4% (34 rows); the other 364 rows (32%) mismatch 48.1% (175 rows, 175 of the 209 mismatches). The 538 rows with b2 >= 11.999 (no rival and label logprob above -0.001) mismatch 1.1% (6 rows), the other 591 rows 34.3% |
| parse fallbacks (nearest label used) | E1 1, cheap 3 of 1,129 |
| exact-text duplicates of a Banking77 training input (the student's training set) | 3 of 1,129 (1 reused row, 2 new); the split is disjoint by index |

### AUROC for "cheap == E1" (all 1,129 rows)

| signal | AUROC [95% CI] | S5 200 rows | new 929 rows | S15 (200 rows, cheap A draw) |
|---|---|---|---|---|
| f1 student == cheap | 0.875 [0.849, 0.901] | 0.900 | 0.869 | 0.897 |
| b2 cheap logprob gap | 0.895 [0.874, 0.913] | 0.892 | 0.896 | 0.892 |
| f3 f1 then b2 | 0.946 [0.931, 0.960] | 0.958 | 0.944 | 0.955 |
| e student top-2 gap | 0.692 [0.652, 0.732] | 0.671 | 0.698 | 0.665 |

Paired: f3 minus b2 +0.051 [+0.035, +0.068]; f3 minus f1 +0.071 [+0.054, +0.090]; f1 minus b2 -0.020 [-0.051, +0.008] (includes 0). The S15 ordering reproduces on 5x the data: f3 clearly best, agreement and logprob gap roughly tied, the student's own gap far behind.

### Calibration-size sweep, signal f3 (agreement, then logprob gap)

Certified coverage, mean [5th percentile] (share of the 200 splits that certified anything).

| calibration rows (holdout) | budget 5% | budget 10% | budget 15% | budget 20% |
|---|---|---|---|---|
| 100 (1029) | 0.00 [0.00] (0%) | 0.20 [0.00] (28%) | 0.68 [0.00] (92%) | 0.83 [0.73] (100%) |
| 200 (929) | 0.04 [0.00] (6%) | 0.72 [0.57] (100%) | 0.83 [0.77] (100%) | 0.89 [0.84] (100%) |
| 300 (829) | 0.32 [0.00] (56%) | 0.78 [0.72] (100%) | 0.85 [0.81] (100%) | 0.91 [0.86] (100%) |
| 500 (629) | 0.62 [0.44] (100%) | 0.80 [0.77] (100%) | 0.87 [0.84] (100%) | 0.93 [0.90] (100%) |
| 929 (200), supplement | 0.70 [0.65] (100%) | 0.82 [0.78] (100%) | 0.88 [0.84] (100%) | 0.95 [0.93] (100%) |
| 1000 (129) * | 0.71 [0.64] (100%) | 0.83 [0.76] (100%) | 0.89 [0.84] (100%) | 0.96 [0.92] (100%) |

Realized mismatch among routed holdout rows, mean (max); viol% (share of certifying splits whose holdout mismatch exceeded the budget); cost saving vs always-E1 at mean coverage.

| calibration rows | budget 5% | budget 10% | budget 15% | budget 20% |
|---|---|---|---|---|
| 100 | - | 0.026 (0.076); 0.0%; 14% | 0.037 (0.109); 0.0%; 63% | 0.067 (0.159); 0.0%; 77% |
| 200 | 0.020 (0.026); 0.0%; -2% | 0.032 (0.088); 0.0%; 66% | 0.064 (0.132); 0.0%; 77% | 0.103 (0.167); 0.0%; 83% |
| 300 | 0.015 (0.049); 0.0%; 26% | 0.042 (0.083); 0.0%; 72% | 0.078 (0.141); 0.0%; 79% | 0.120 (0.191); 0.0%; 85% |
| 500 | 0.016 (0.048); 0.0%; 57% | 0.051 (0.098); 0.0%; 75% | 0.090 (0.154); 0.5%; 81% | 0.137 (0.204); 0.5%; 88% |
| 929 (supplement) | 0.023 (0.077); 3.5%; 64% | 0.061 (0.136); 6.5%; 77% | 0.101 (0.218); 5.0%; 82% | 0.153 (0.246); 8.0%; 90% |
| 1000 * | 0.023 (0.075); 6.0%; 65% | 0.060 (0.128); 5.0%; 77% | 0.102 (0.186); 7.5%; 83% | 0.153 (0.234); 14.5%; 90% |

\* n = 1,000 leaves 129 holdout rows, below the 200-row floor the design wanted (the 71 missing E1 labels). The n = 929 row (calibration 929, holdout 200; do not confuse it with the 200 (929) row above) has exactly 200 holdout rows; its numbers are close (coverage 0.70 / 0.82 / 0.88 / 0.95 against 0.71 / 0.83 / 0.89 / 0.96 at n = 1,000) and the rule's outcome is the same with either as the reference (1,000 / 300 / 200 / 200, `log.txt` Revision check 1).

Reading it. (1) The column that moves with n is cert%: 100 rows certify 28% of the time at 10% and never at 5%; 200 rows certify 100% at 10-20% and 6% at 5%; 300 rows 56% at 5%; 500 rows 100% everywhere. (2) Past the point where it certifies reliably, more rows buy coverage slowly: at 15% going from 200 to 1,000 rows lifts coverage 0.83 to 0.89, saving 77% to 83%. (3) The mean realized mismatch is well inside the budget at every n (15% budget: 3.7% at 100 rows, 6.4% at 200, 10.2% at 1,000), because the bound is a worst-case guarantee and the mismatch concentrates in the 32% of rows where the cheap model lists a rival label (48.1% mismatch there, against 4.4% on the other 68%; 175 of the 209 mismatches). (4) The viol% of 5-14.5% at n = 1,000 is holdout sampling noise, not certificate failure. The holdout is 129 rows and only 79-105 of them are routed at the 5% budget (115-129 at 20%), so one holdout's mismatch has a standard deviation of about 1.6 points at 5% and about 3.2 at 20% (binomial, at the observed routed count and mean mismatch), and the mean sits only 1.5-1.7 standard deviations below the budget. To check, every certified f3 cutoff at n = 500, 929 and 1,000 was also scored on all 1,129 rows, taken as the truth: 0 of 200 splits exceeded the budget in all 12 cells (`log.txt` Revision check 2). That check is weaker at large n, because the pool is then mostly calibration rows (89% at n = 1,000, 82% at 929, 44% at 500); at n <= 500 (holdout >= 629) the holdout viol% is at most 0.5% anyway.

### Other signals (mean coverage, share of splits that certify in brackets)

| signal | calibration rows | budget 5% | budget 10% | budget 15% | budget 20% |
|---|---|---|---|---|---|
| f1 student == cheap (no logprobs needed) | 100 | 0.00 (0%) | 0.02 (3%) | 0.32 (42%) | 0.71 (93%) |
|  | 200 | 0.00 (0%) | 0.29 (38%) | 0.75 (98%) | 0.76 (100%) |
|  | 300 | 0.01 (1%) | 0.58 (76%) | 0.76 (100%) | 0.76 (100%) |
|  | 500 | 0.00 (0%) | 0.76 (99%) | 0.76 (100%) | 0.76 (100%) |
|  | 1000 | 0.00 (0%) | 0.76 (100%) | 0.76 (100%) | 0.76 (100%) |
| b2 cheap logprob gap alone | 100 | 0.00 (0%) | 0.08 (12%) | 0.48 (76%) | 0.75 (100%) |
|  | 200 | 0.01 (1%) | 0.55 (94%) | 0.74 (100%) | 0.84 (100%) |
|  | 300 | 0.21 (44%) | 0.65 (100%) | 0.78 (100%) | 0.88 (100%) |
|  | 500 | 0.48 (97%) | 0.71 (100%) | 0.82 (100%) | 0.91 (100%) |
|  | 1000 | 0.58 (100%) | 0.75 (100%) | 0.84 (100%) | 0.95 (100%) |
| e student top-2 gap (decides before the cheap call) | 100 | 0.00 (0%) | 0.00 (0%) | 0.02 (4%) | 0.12 (21%) |
|  | 200 | 0.00 (0%) | 0.00 (0%) | 0.06 (13%) | 0.34 (58%) |
|  | 300 | 0.00 (0%) | 0.00 (0%) | 0.10 (20%) | 0.56 (86%) |
|  | 500 | 0.00 (0%) | 0.00 (1%) | 0.23 (48%) | 0.74 (100%) |
|  | 1000 | 0.00 (0%) | 0.00 (0%) | 0.59 (100%) | 0.85 (100%) |

Reading it. f1 is two-valued (agree or not), so it cannot carve a safer subset out of the agree group: its mismatch is about 3.7% at 0.76 coverage, but a bound at delta 5%/40 over about 760 rows sits near 6%, so it essentially never certifies at 5% (1% of draws at 300 rows) and certifies at 10% only from about 300-500 rows. b2 orders rows inside the group and reaches 5% from 300 rows (44% of splits) and 500 rows (97%), at lower coverage than f3 at every n (0.48 vs 0.62 at 500 rows). The student's gap alone behaves as in S15: nothing at 5-10%, 15% only at 1,000 rows (13% viol; not checked against the pool as the f3 cells were), 20% from 500 rows; its single certifying split at 500 rows and 10% exceeded the budget.

### Recommended calibration size per budget (signal f3) and what it saves

Rule fixed in `run.py` before looking at results (Setup): the smallest n where >= 95% of splits certify and mean coverage >= 90% of the n = 1,000 coverage. At the 5% budget, 1,000 is the largest size tested, so it means that no smaller tested size qualified (500 rows give coverage 0.623, below 0.9 x 0.708 = 0.637).

| budget | recommended n (rule) | cov [p5] | realized mismatch among routed, mean (max) | about this share of ALL calls differs from E1 | blended $/1k | saving vs always-E1 | cov at n=1000 | in-sample ceiling (no guarantee) |
|---|---|---|---|---|---|---|---|---|
| 5% | 1,000 | 0.71 [0.64] | 0.023 (0.075) | 1.7% | $0.604 | 65% | 0.71 | 0.80 |
| 10% | 300 | 0.78 [0.72] | 0.042 (0.083) | 3.2% | $0.484 | 72% | 0.83 | 0.88 |
| 15% | 200 | 0.83 [0.77] | 0.064 (0.132) | 5.2% | $0.400 | 77% | 0.89 | 0.95 |
| 20% | 200 | 0.89 [0.84] | 0.103 (0.167) | 9.2% | $0.295 | 83% | 0.96 | 1.00 |

Looser test, for comparison only: the smallest n where >= 95% of splits certify, with no condition on coverage.

| budget | smallest n certifying in >= 95% of splits | cov, saving, $/1k there | share of the n = 1,000 coverage | meets the 90% coverage condition |
|---|---|---|---|---|
| 5% | 500 | 0.62, 57%, $0.752 | 88% | no (0.623 < 0.637) |
| 10% | 200 | 0.72, 66%, $0.582 | 87% | no (0.72 < 0.744) |
| 15% | 200 | 0.83, 77%, $0.400 | 93% | yes |
| 20% | 100 | 0.83, 77%, $0.397 | 86% | no (0.827 < 0.861) |

Blended $/1k = $0.098 + (1 - cov) x $1.732, from observed `usage.cost`; always-E1 is $1.732, always-cheap $0.098. The "share of ALL calls differing from E1" column is mean coverage x mean mismatch (approximate): at the 15% operating point about 5.2% of all calls would get a different label than E1 gave, about the size of E1's own 6% repeat disagreement. In-sample ceiling = the largest routed share on all 1,129 rows with mismatch <= budget (no guarantee, it uses the labels it is scored on); certified coverage at n=1,000 reaches 88% (5%), 94% (10%), 93% (15%) and 96% (20%) of it. With the n = 929 supplement (200 holdout rows) as the coverage reference the rule picks the same sizes.

Calibration price. The n calibration rows are calls E1 would have answered anyway; the extra spend is the shadow cheap call, $0.020 for 200 rows and $0.098 for 1,000. The real cost of waiting is the saving not yet taken: at the 15% point the first 200 calls forgo about $0.27.

## Transfer to Dreaming

- **Shadow phase length.** Run E1 plus the free student plus a cheap logprob call side by side and switch routing on at the recommended size for the budget: 200 calls for a 15-20% budget, 300 for 10%, 1,000 before offering 5% (and then only after the second-E1-draw check in the last bullet). The looser cert%-only test would have said 200 for 10% and 500 for 5%, but at those sizes coverage is below 90% of the 1,000-row coverage. Keep E1-labelling a sampled fraction of routed calls afterwards (missions.md, router step 4): that grows the calibration set to the 500-1,000 range over time and buys 5-7 points of coverage at 10-20% over the recommended sizes.
- **Budget the operator should expect to see.** Realized mismatch with E1 averaged 2-10% at the recommended points, well below the stated budget; the certificate is a bound, not a forecast.
- **Without logprobs (reasoning models, other providers).** Agreement alone is the fallback: 15-20% budgets from 200 rows at 0.75-0.76 coverage (69-71% cheaper), 10% from 500 rows, no 5% certificate. The logprob tie-break is worth +0.07 AUROC and +4 to +43 points of coverage at 10-20% budgets with 200-500 rows (largest at 10% with 200 rows, 0.72 vs 0.29), and it is what makes 5% reachable (0.62 vs 0 at 500 rows).
- **The student's own gap cannot start a router on its own** at these sizes, as S15 found; it is useful only after a cheap call has been made.
- **A 5% budget is below E1's 6% average repeat disagreement.** It certified here because the routed rows are the ones where the cheap model lists no rival or the student agrees, and those are plausibly rows where E1 is also stable; that was not tested (S28 has one E1 draw per row), so a second E1 draw on routed rows is the missing check before promising 5% to an operator.

## Caveats

- One call site (Banking77, 77 labels, short inputs, about 440 prompt tokens), one cheap model (gpt-5.4-nano on OpenAI), one reference (gpt-5.4 low), one student. Other sites with a different match rate move every number; the shape (cert% rises steeply over a few hundred rows, coverage slowly after) should travel, the exact n will not.
- The student here is trained on 9,993 gold labels and reaches 0.929 against gold, above E1 (0.857) and the cheap model (0.762); it agrees with E1 on 0.849. In Dreaming the student trains on logged E1 answers, so f1 and e will behave differently. At this site the student alone would be the better route; the router was kept as the measurement vehicle because S15 used it.
- Splits are random subsets of one fixed pool of 1,129 rows. They are not independent draws from the world: for n=1,000 two calibration sets share at least 87% of rows and the holdout is only 129 rows (below the 200-row floor, see the n = 929 row), so p5 coverage and viol% at large n are noisy and the pool's own error profile is treated as the truth; the pool check in Reading (4) is itself weaker at large n. No drift is simulated; the certificate assumes calibration and deployment rows are exchangeable.
- Parse choices. Labels are the nearest valid label (4 fallbacks in 2,258 answers); b2 uses only the returned top-5 alternatives and floors unlisted rivals at 12 nats; 68% of cheap answers list no rival, and they do not tie (b2 = 12 + the label's logprob, 560 distinct values among 765 rows). The target is one E1 draw (6% self-disagreement).
- 71 of 1,000 planned E1 labels are missing (credits ran out, see Setup). The numbers use 929 new + 200 reused rows. To complete: top up the OpenRouter balance (about $0.12 needed), run `python calls.py run` (cached rows are skipped, the spend cap is $3.00) and `python run.py`.
- 3 of the 1,129 inputs (1 reused, 2 new) have an exact-text duplicate in the Banking77 training set the student is trained on. The split is disjoint by index and the rows were kept; 3 rows of 1,129 are unlikely to move any table here, but the student's agreement on them is not out-of-sample.
- The ledger `responses.jsonl` holds 531 error rows at $0: 528 are 402 "in-flight budget exhausted" responses from concurrent threads against a nearly empty balance (a request reserves its maximum cost while in flight), 3 are 402 "maximum cost exceeds available credits"; all but the 71 missing inputs were later retried successfully.
- `log_partial.txt` is an earlier analysis on the 904 new rows that had E1 labels at the time; `log.txt` is this final run on 929. Numbers differ by under 0.01 in AUROC and by up to 0.07 in coverage (the f1 cells move in steps).

## Verdict

Router certificates need fewer calibration rows than S15 guessed, except at 5%. With the agreement-plus-logprob-gap gate, the rule fixed before looking recommends 1,000 / 300 / 200 / 200 calibration rows for budgets of 5 / 10 / 15 / 20%, at 65 / 72 / 77 / 83% lower cost (cheap model on 71 / 78 / 83 / 89% of calls). The looser test of certifying in >= 95% of draws alone would say 500 / 200 / 200 / 100 rows, but at three of those four budgets coverage is under 90% of the 1,000-row coverage. Default shadow length: 200 rows for a 15-20% budget, 300 for 10%. Offer 5% only after 1,000 rows, and only after a second E1 draw on routed rows shows E1 is stable there: 5% is below E1's own 6% repeat disagreement and that check has not been run. Agreement alone works as a no-logprobs fallback from 15%. The student's own confidence is not a router.

## Revision (2026-10-17, after review) - what changed

| review point | what was done | result |
|---|---|---|
| Verdict, plain words and Transfer used the looser ">= 95% of splits certify" test instead of the rule fixed in `run.py` | All three now lead with the rule (1,000 / 300 / 200 / 200 rows for budgets 5 / 10 / 15 / 20%); the cert%-only sizes (500 / 200 / 200 / 100) are a separate, labelled table. The Verdict and plain words carry the 5% caveat (below E1's 6% repeat disagreement; a second E1 draw on routed rows is the missing check) | "offer 5% after 500" and "200 rows at 10-15%" are gone; 500 rows give coverage 0.623 < 0.637 at 5%, 200 rows 0.72 < 0.744 at 10% |
| n = 1,000 holdout is 129 rows, below the 200-row floor | Rows marked * with a footnote; supplementary n = 929 row (holdout exactly 200, same seeds, 200 splits) in both sweep tables; `run.py` Revision check 1 also re-applies the rule with the 929 coverage as the reference | n = 929: cert 100% everywhere, coverage 0.700 / 0.824 / 0.880 / 0.955, mismatch 0.023 / 0.061 / 0.101 / 0.153, viol% 3.5 / 6.5 / 5.0 / 8.0; the rule picks the same 1,000 / 300 / 200 / 200 |
| "No rival / floor" numbers wrong in three places | Recomputed from `lp_parts` (new, the rival half of `lp_gap`): Models table row, Reading (3) and the Caveats parse bullet corrected; Signals bullet now says no-rival rows are not tied. The earlier 538 / 6 rows / 34.3% was the b2 >= 11.999 subset and is labelled as that | no rival listed: 765 of 1,129 (68%), mismatch 4.4% (34); rival listed: 364 rows (32%), mismatch 48.1% (175 of the 209 mismatches); 560 distinct b2 values among the 765 (917 over all rows). "48% tie at the floor" deleted |
| viol% hedge could be firmer; noise figures slightly off | Every certified f3 cutoff at n = 500, 929 and 1,000 scored on all 1,129 rows (Revision check 2); Reading (4) rewritten with the actual routed holdout counts and standard deviations | 0 of 200 splits exceed the budget in all 12 cells, so the n = 1,000 viol% (5-14.5%) is holdout noise, not certificate failure (weaker check at large n: the pool is mostly calibration rows). Routed holdout rows 79-105 at 5% and 115-129 at 20% (not "90-125"); standard deviation about 1.6 points at 5%, 3.2 at 20% |
| Provider-pin wording | Setup now says S5's E1 request had no provider pin | all 200 reused S5 E1 calls and all 929 new ones were served by OpenAI, so the data stand |
| Student not byte-identical to S15's | One sentence in the Signals bullet | on the same 200 rows f1 0.900 vs 0.897, e 0.671 vs 0.665, f3 0.958 vs 0.955, b2 identical; probably re-embedding for this run (not checked) |
| Exact-text duplicates of training inputs | Models table row and a Caveats bullet; `run.py` Revision check 3 | 3 of 1,129 (1 reused, 2 new); split disjoint by index; rows kept |

Unchanged: every sweep, AUROC and recommended-n number in the first `log.txt` reproduces exactly (diff of the two logs shows only the added checks). `results.json` differs from the first run only in the fifth decimal of the e-signal AUROC bounds (one-thread BLAS) and in the new `revision` block. The rule in `run.py` was not touched; `lp_gap` was split into `lp_parts` plus `lp_gap` (same numbers) and `split_run` and the three checks were added. No API calls.

## Time and cost

- API spend this smoke, sum of `usage.cost` over `responses.jsonl`: **$1.7096** = E1 929 calls $1.6117 ($1.735 per 1k) + cheap 1,000 calls $0.0979 ($0.098 per 1k). Error rows cost $0. The 200 reused rows were paid in S5 and S15. A few in-flight calls when earlier runs were stopped may have completed unrecorded (under $0.01 each). The $3.00 cap was not reached. Mean latency: E1 2.1 s, cheap 0.8 s per call.
- Analysis: one process on the laptop CPU, 198 s (95 s to embed the 9,993 training texts because the embedding cache was not present, 17 s for the 1,129 test texts, the rest the 200-split sweep and bootstraps). No pods, no GPU. Revision rerun: 47 s, one thread, embedding cache present, so the two `[embed]` lines of the first run's log (95 s and 17 s, quoted above) are not in the rerun's `log.txt`. No API calls in the revision; spend unchanged.
- Files: `calls.py` (calls and ledger), `run.py` (analysis), `responses.jsonl`, `results.json`, `log.txt`.
