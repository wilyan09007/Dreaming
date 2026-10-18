# S4: Tool return predictor and next-step speculation (Mission 3)

**The environment is synthetic.** Every tool, merchant, descriptor, latency, error and order status below comes from `env.py`, a seeded generator I wrote. The results therefore only test the mechanics (log -> predict -> certify -> skip / speculate / fall back -> outage handling), not how predictable real bank descriptors, real order statuses or a real refund API are. Real-world accuracy and coverage numbers will differ, in either direction; do not quote any number here as a real-world estimate.

## Revision

A reviewer found material problems in the first version. This section says what changed; every table below is from the revised `run.py` (`log.txt`). The first version's log is kept as `log_v1.txt`.

**1. The shipped certifier was path dependent (a bug in `common.certify`, which I did not modify).** `certify` binary-searches the coverage k, starting at k = n/2, and moves up when the Clopper-Pearson (CP) bound at k is <= budget and down otherwise. The bound is U-shaped in k (loose for small k, tighter as k grows, rising again as errors accumulate), so failing at the probe says nothing about smaller or larger k. Minimal case (`selfcheck()` in `run.py`, runs on every start): 1000 rows, the top 480 correct, 5 errors in rows 480-484, budget 2%: `certify` returns "nothing certifiable"; the bound is 1.5% at full coverage. The first version's "common.py: no bug found" was wrong.
Fix in `run.py`: `certify_grid` evaluates 20 fixed coverage levels (top j/20 of the rows by confidence, tie-closed), delta/20 each (union bound, same overall delta 0.05) and keeps the largest level whose CP bound is <= budget. Every certificate below uses it; `common.certify` is kept only for the before/after columns. The grid is coarser (5-point coverage steps, delta/20 instead of about delta/12 per probe), so at equal data it sometimes certifies slightly less coverage than the original (65.1% vs 68.4% for LR in the main world).

**2. How the numbers moved** (main world = env seed 4, budget 1% unless stated):

| quantity | first version | revised |
|---|---|---|
| NN at 1%, main world | "not certifiable at 1% at all" | certifiable: cal coverage 35.2%, test skipped 37.0%, 0/436 errors, 0/213 false skips on novel merchants |
| NN at 1%, other worlds | nothing certifiable in seeds 4 and 12 | certifiable in all 6 worlds |
| LR (kept), main world | cutoff 0.839, cal coverage 68.4%, test skipped 69.9%, 1/823 errors | cutoff 0.870, cal coverage 65.1%, test skipped 67.3%, 1/792 errors, false-skip novel 1/213 (0.47%, unchanged) |
| LR at 2%, main world | cutoff 0.514, 4 errors in 942 skipped (from the 2% sweep row of the first version's certifier) | cutoff 0.691, 1/904 errors, 1/213 false skips |
| pooled over 6 worlds, kept model, 1% | 28/5107 errors above cutoff (0.55%), 28/1425 false skips on novel merchants (1.96%), 72.2% skipped | 21/4943 (0.42%), 21/1425 (1.47%), 69.9% skipped |
| worlds that break the 1% budget | 1 of 6 (seed 11, 2.2%, 20/909) | 1 of 6 (seed 11, 1.46%, 13/889) |
| worlds that break a 2% budget | 2 of 6 (seeds 11 and 13) | 0 of 6 for the kept LR |
| outage, predictor + idealised snapshot | 89.3% completed, 1.68% error | 87.8% completed, 1.94% error (cutoff 0.870 instead of 0.839) |
| vendor_lookup time saved | 70.0% (141 ms/call), 27 of 37 tail calls | 66.0% (133 ms/call), 24 of 37 tail calls |

(The reviewer quoted "LR gets 4/942 at 2%". That is the original certifier's LR; under the grid certifier LR at 2% is 1/904. The reviewer's NN numbers (0/436 at 1%, 0/723 at 2%) reproduce exactly.)

**3. The "keep the model with higher certified coverage" rule is not safe by itself.** In seed 11 it keeps LR, which breaks the 1% budget (13/889 = 1.46%, 5.3% false skips on novel merchants) while NN holds (3/638 = 0.47%). In seed 13 it keeps LR, which is fine (5/759 = 0.66%), while NN breaks (27/792 = 3.41%). Pooled over six worlds LR has 0.42% error above cutoff, NN 0.82%. Neither is safe by default and coverage cannot tell them apart. I kept the rule (it is the spec's "keep the better") and report both models everywhere.

**4. The "naive calibration" bullet was wrong in what it measured.** With the original certifier the naive (known-merchants-only) calibration in the main world gave 0.95% test error, under the 1% budget; the harm shown was 9 versus 1 false skips on novel merchants, not a breach. "98% versus 68% coverage" compared two different calibration slices; the comparable number is test skipped (80.5% versus 69.9% then; 78.4% versus 67.3% now). With the grid certifier the main world shows no harm from naive calibration at all (1 false skip each). The new pooled six-world evidence (below) is what supports the point: in world 11 naive calibration at 1% gives a cutoff of 0.029, 19.7% test error and 91% of novel-merchant calls confidently skipped; at a 2% budget 5 of 6 worlds exceed the budget (17.3% pooled error, 80% of novel calls skipped).

**5. Smaller items fixed.** The headline false-skip is now pooled (1.47% over six worlds) next to the seed-4 value (0.47%). Per-merchant evidence for "errors cluster by merchant" is in the robustness table. Speculation now states that the HGB at tau 0.5/0.7/0.9 saves less than the constant "always speculate delivered". Fallback-first now reports calls flagged (38 at 0.5) and a precision interval. The outage table now has a realistic snapshot-resolution variant and says next to the table that the idealised rows are upper bounds and that the outage stream shares its unseen merchants with the test stream.

**6. Second review round (no change to any earlier number: the re-run's `log.txt` differs from the previous one only by the new tables and the wall seconds, `traces.jsonl` is byte-identical).**
- *Novelty mix.* The calibration and test streams both have about 20% novel merchants by construction, so the "deployment-like" certificate holds partly because of that match. I did not simulate a different novelty rate; instead `run.py` now re-weights the existing test stream to a different novel share p with the certificate fixed (new tables after the pooled table in section 1). The kept LR's pooled error above the cutoff crosses the 1% budget between p = 30% (0.69%) and p = 40% (1.07%), and the 2% budget is still held at p = 50% (1.54%) pooled, though seed 11 (30%+) and seed 13 (50%) exceed 2% individually. This is a re-weighting of one test stream per world, not a new simulation.
- *Certifier validity.* `certify_grid`'s 20 coverage levels are fixed in coverage but their cutoffs are data-dependent quantiles of the calibration scores, so the delta/20 union bound is not strictly valid; "delta 0.05" is nominal (see Caveats).
- *Latency scope.* The vendor_lookup time-saved numbers are computed on successful calls only (error rows are dropped before the split), so the 5 s timeout rows are not in the denominator; the effect is small.
- *Label noise.* The generator has one cross-merchant descriptor collision in seed 11 (`PAYPAL *EVERGREENW`, from two merchants, caused by the 10-character PAYPAL truncation) and none in seeds 4 and 9; negligible.

## Setup

- **Env** (`env.py`, run `python env.py` for its self-check: same seed gives identical returns and latencies, 300 unique merchants, disjoint seen/unseen groups, outage gives 100% errors; `env.py` was not changed in this revision and the revised run regenerates a byte-identical `traces.jsonl`).
  - 300 canonical merchants, 12 categories x 25, names built from word parts (40 prefixes x 12 category suffixes), plus one hand-made "Amazon Marketplace" (most frequent merchant). Merchant frequency is Zipf (exponent 0.7). Each merchant has 1-4 raw descriptor templates; render adds processor prefixes (`SQ *`, `TST* `, `PAYPAL *`, `POS `, `CHECKCARD `, `AMZN Mktp US*2K3AB`), store numbers, city/state, trailing digits, abbreviation/space removal, 22-char truncation, random case.
  - `vendor_lookup(descriptor) -> vendor_id`: deterministic function of the canonical merchant; latency lognormal median 120 ms plus a 3% tail > 1.5 s; 2% timeout errors (5 s); outage flag gives 100% errors.
  - `lookup_order(order_id) -> status` in {pending, shipped, delivered, refunded, cancelled}: hidden daily Markov chain run for `age_days` steps; transition rates depend on channel, amount, category and a hidden per-order speed factor (lognormal). Visible features: age_days, channel, amount, category. Latency lognormal median 200 ms (my choice; the task did not give one).
  - `issue_refund(order_id, amount) -> ok | insufficient_funds | vendor_error | timeout`: multinomial logit on amount, category risk, hour-of-day and a hidden two-state "degraded bank" regime (persistent, ~14% of the time). Latency lognormal by outcome (timeouts 8 s), slower for some categories, busier in the day, slower when degraded. The trace carries `recent_error_rate` (last 30 calls of that tool).
  - Every call appends one row to `traces.jsonl` (tool, args, visible features, return, latency_ms, ts, recent_error_rate, plus a `hid` field with ground-truth diagnostics that predictors never read; it is used only for the oracle rows, the generation flags and the per-merchant error counts).
- **Stream**: 6000 transactions, each makes one call to each tool (18,000 rows; for `issue_refund` every transaction is refunded, a simplification). Chronological split: train = first 3000, calibration = next 1800, test = last 1200. Train transactions come only from 240 "seen" merchants. Calibration and test are 80% seen / 20% from "unseen" merchants; the 30 unseen merchants in calibration are disjoint from the 30 in test. "Novel" below means the true vendor never appears in the training traces (this matched the generation flag exactly in the main run: 18.1% of test).
- **Seeds**: main env seed 4 (all tables below unless stated), robustness seeds 11-15, HGB `random_state=0`.
- **Predictors**
  - vendor_lookup: char 2-4gram TF-IDF (lowercased, digits -> 0) + either multinomial LogisticRegression (C=30) or cosine nearest neighbour (per-class max similarity). Confidence = top-2 gap (`common.confidence`). The one with higher certified coverage on the calibration slice is kept (ties: smaller CP bound). Trained on successful calls only. Certified with `certify_grid` at budget 1% (also 0.5%, 2%, 5% in the sweep), delta 0.05 on the calibration slice, then applied to the test stream.
  - lookup_order, issue_refund: `HistGradientBoosting` (regularised: 8 leaves, min 40 samples/leaf, L2 1, early stopping) trained on train + calibration (4800 rows), evaluated on the last 1200. The refund model also gets "previous call failed" and "error rate of the previous 5 calls", derived from the trace.
- Run: `python run.py` (log in `log.txt`, stderr timing in `err.txt`).

## Results

### 1. vendor_lookup (return is a function of the input)

Rows (successful calls): train 2936, calibration 1771, test 1177; 240 vendors in training; novel-merchant share 19.5% (cal), 18.1% (test).

| model | cal acc | test acc (always answer) | certified cutoff @1% | cal coverage | cal CP bound | test calls skipped | test errors above cutoff | false-skip, novel merchants |
|---|---|---|---|---|---|---|---|---|
| LR (kept) | 0.801 | 0.811 | 0.870 | 0.651 | 0.0052 | 0.673 | 1/792 = 0.13% | 1/213 = 0.47% |
| NN | 0.779 | 0.784 | 0.423 | 0.352 | 0.0096 | 0.370 | 0/436 = 0.00% | 0/213 = 0.00% |

Test accuracy if always answered is 0.990 on known merchants and 0 on novel ones (a closed-set classifier cannot output an unseen vendor).

Calibration that ignores novelty versus calibration that includes it (kept LR, same test stream; main world only):

| calibration slice | cutoff | cal coverage | test skipped | test errors above cutoff | 95% upper (CP) | false-skip on novel merchants |
|---|---|---|---|---|---|---|
| includes novel merchants (deployment-like) | 0.870 | 0.651 | 0.673 | 1/792 = 0.13% | 0.0060 | 1/213 = 0.47% |
| known merchants only (naive) | 0.619 | 0.950 | 0.784 | 1/923 = 0.11% | 0.0051 | 1/213 = 0.47% |

In this world naive calibration costs nothing. It does in other worlds, see the robustness tables below (seed 11: cutoff 0.029, 227/1155 = 19.65% test error, 225 of 247 novel-merchant calls confidently skipped).

By test slice at the deployment-like cutoff: known merchants (n=964) 82.1% skipped, 0/791 errors; novel merchants (n=213) 0.47% skipped (the single skip is a confident wrong answer).

Budget sweep, both models, and what the original `common.certify` reported for the same calibration data:

| error budget | model | cutoff | cal coverage (grid certifier) | cal coverage (original certify) | test skipped | test errors above cutoff | false-skip novel |
|---|---|---|---|---|---|---|---|
| 0.5% | LR (kept) | inf | 0 | 0 | 0 | 0/0 | 0/213 |
| 0.5% | NN | inf | 0 | 0 | 0 | 0/0 | 0/213 |
| 1% | LR (kept) | 0.870 | 0.651 | 0.684 | 0.673 | 1/792 = 0.13% | 1/213 = 0.47% |
| 1% | NN | 0.423 | 0.352 | 0 | 0.370 | 0/436 = 0.00% | 0/213 = 0.00% |
| 2% | LR (kept) | 0.691 | 0.752 | 0.793 | 0.768 | 1/904 = 0.11% | 1/213 = 0.47% |
| 2% | NN | 0.296 | 0.600 | 0.619 | 0.614 | 0/723 = 0.00% | 0/213 = 0.00% |
| 5% | LR (kept) | 0.461 | 0.801 | 0.822 | 0.811 | 12/954 = 1.26% | 10/213 = 4.69% |
| 5% | NN | 0.214 | 0.700 | 0.718 | 0.692 | 1/814 = 0.12% | 1/213 = 0.47% |

Nothing is certifiable at 0.5% with 1771 calibration rows for either model. Latency (kept model, 1% budget): mean 201 ms/call; skipping saves 133 ms/call (66.0% of total vendor_lookup time) and removes 24 of the 37 tail (> 1.5 s) calls in the test stream.

**Does the certificate hold on other worlds?** Whole pipeline (new merchants, templates, streams) re-run with 5 other env seeds. The last column counts the merchants behind the errors above the cutoff and the most errors from a single merchant.

| env seed | model | test acc (always answer) | cutoff | cal coverage (grid) | cal coverage (original certify) | test skipped | test errors above cutoff | false-skip novel | distinct merchants behind the errors (max from one) |
|---|---|---|---|---|---|---|---|---|---|
| 4 | LR (kept) | 0.811 | 0.870 | 0.651 | 0.684 | 0.673 | 1/792 = 0.13% | 1/213 = 0.47% | 1 (max 1) |
| 4 | NN | 0.784 | 0.423 | 0.352 | 0 | 0.370 | 0/436 = 0.00% | 0/213 = 0.00% | 0 |
| 4 | LR naive (known-merchant cal rows only) | 0.811 | 0.619 | 0.950 | 0.982 | 0.784 | 1/923 = 0.11% | 1/213 = 0.47% | 1 (max 1) |
| 11 | LR (kept) | 0.788 | 0.710 | 0.750 | 0.765 | 0.755 | **13/889 = 1.46%** | **13/247 = 5.26%** | 6 (max 4) |
| 11 | NN | 0.771 | 0.345 | 0.551 | 0.584 | 0.542 | 3/638 = 0.47% | 3/247 = 1.21% | 2 (max 2) |
| 11 | LR naive | 0.788 | 0.029 | 1.000 | 1.000 | 0.981 | **227/1155 = 19.65%** | **225/247 = 91.09%** | 31 (max 30) |
| 12 | LR (kept) | 0.795 | 0.887 | 0.650 | 0.682 | 0.649 | 0/767 = 0.00% | 0/236 = 0.00% | 0 |
| 12 | NN | 0.773 | 0.463 | 0.403 | 0 | 0.368 | 0/435 = 0.00% | 0/236 = 0.00% | 0 |
| 12 | LR naive | 0.795 | 0.733 | 0.950 | 0.997 | 0.760 | 4/898 = 0.45% | 4/236 = 1.69% | 1 (max 4) |
| 13 | LR (kept) | 0.777 | 0.876 | 0.650 | 0.687 | 0.645 | 5/759 = 0.66% | 5/258 = 1.94% | 1 (max 5) |
| 13 | NN | 0.762 | 0.263 | 0.650 | 0.667 | 0.673 | **27/792 = 3.41%** | **27/258 = 10.47%** | 4 (max 16) |
| 13 | LR naive | 0.777 | 0.744 | 0.951 | 0.997 | 0.738 | 8/868 = 0.92% | 8/258 = 3.10% | 2 (max 5) |
| 14 | LR (kept) | 0.806 | 0.825 | 0.700 | 0.702 | 0.724 | 0/854 = 0.00% | 0/222 = 0.00% | 0 |
| 14 | NN | 0.776 | 0.341 | 0.550 | 0.589 | 0.564 | 0/665 = 0.00% | 0/222 = 0.00% | 0 |
| 14 | LR naive | 0.806 | 0.675 | 0.951 | 0.982 | 0.771 | 2/909 = 0.22% | 2/222 = 0.90% | 1 (max 2) |
| 15 | LR (kept) | 0.784 | 0.736 | 0.750 | 0.775 | 0.746 | 2/882 = 0.23% | 2/249 = 0.80% | 1 (max 2) |
| 15 | NN | 0.765 | 0.344 | 0.601 | 0.637 | 0.601 | 0/710 = 0.00% | 0/249 = 0.00% | 0 |
| 15 | LR naive | 0.784 | 0.643 | 0.951 | 0.979 | 0.761 | 2/900 = 0.22% | 2/249 = 0.80% | 1 (max 2) |

(Naive rows: the "cal coverage" columns are measured on the known-merchant calibration slice, so they are not comparable with the other rows. Every error above a cutoff in the kept-LR rows is a confident wrong answer on a novel merchant. The per-world table for the 2% budget is in `log.txt`.)

Pooled over the six worlds:

| budget | certifier / selection | errors above cutoff (pooled) | false-skip novel (pooled) | share of test skipped (pooled) | env seeds whose test error exceeds the budget |
|---|---|---|---|---|---|
| 1% | first version: original certify, kept by coverage | 28/5107 = 0.55% | 28/1425 = 1.96% | 72.2% | 11 |
| 1% | grid, LR (also the kept model in all 6 worlds) | 21/4943 = 0.42% | 21/1425 = 1.47% | 69.9% | 11 |
| 1% | grid, NN | 30/3676 = 0.82% | 30/1425 = 2.11% | 52.0% | 13 |
| 1% | grid, kept model, naive calibration (known merchants only) | 244/5653 = 4.32% | 242/1425 = 16.98% | 79.9% | 11 |
| 2% | first version: original certify, kept by coverage | 51/5335 = 0.96% | 51/1425 = 3.58% | 75.4% | 11, 13 |
| 2% | grid, LR (kept) | 21/5117 = 0.41% | 21/1425 = 1.47% | 72.4% | none |
| 2% | grid, NN | 31/4085 = 0.76% | 31/1425 = 2.18% | 57.8% | 13 |
| 2% | grid, kept model, naive calibration | 1168/6737 = 17.34% | 1139/1425 = 79.93% | 95.3% | 4, 11, 12, 13, 15 |

**Novelty shift** (certificate from the ~20%-novel calibration slice, kept LR; the test stream is re-weighted to a different novel-merchant share p using each world's per-group skip and error rates; no new simulation). A skipped novel-merchant call is always a wrong answer (closed-set predictor). The "20%" row is the nominal share; the measured shares are 18-22%, hence 0.40% here against 0.42% (21/4943) in the table above.

| novel share of test | pooled error above cutoff, 1% budget | pooled share skipped | env seeds exceeding 1% | pooled error, 2% budget | pooled share skipped | env seeds exceeding 2% |
|---|---|---|---|---|---|---|
| 10% | 0.18% | 78.6% | none | 0.17% | 81.4% | none |
| 20% | 0.40% | 70.0% | 11 | 0.39% | 72.5% | none |
| 30% | 0.69% | 61.4% | 11, 13 | 0.67% | 63.6% | 11 |
| 40% | 1.07% | 52.9% | 11, 13 | 1.03% | 54.7% | 11 |
| 50% | 1.59% | 44.3% | 11, 13 | 1.54% | 45.8% | 11, 13 |

The 1% certificate is therefore safe pooled up to roughly a 38% novel share (linear interpolation between the 30% and 40% rows) and the per-world breaches grow from 1 to 2 of 6 at 30%. Below the certificate's design mix it is conservative (fewer skips but also fewer errors); above it, it silently loses its guarantee unless the mix is monitored and recertified.

### 2. lookup_order (return depends on hidden state: speculate, do not skip)

Train 4800, test 1200. Test status mix: delivered 0.50, refunded 0.20, pending 0.12, cancelled 0.11, shipped 0.08. Mean lookup latency 226 ms; the speculated next step costs 150 ms and runs concurrently, so a hit saves min(latency, 150) ms and a miss saves nothing (the work is discarded, no extra latency). Sequential baseline = mean(L + 150) = 376 ms.

| status predictor | top-1 accuracy | mean log-loss |
|---|---|---|
| majority class ("delivered") | 0.500 | - |
| HGB | 0.568 | 1.050 |
| oracle: env's exact P(status) given the visible features | 0.580 | 1.020 |

Speculation rule: launch the next read-only step when top probability > tau.

| predictor | tau | speculated (share of calls) | hit rate when speculated | wasted launches (of 1200) | mean ms saved / call | saved / sequential mean | ms saved per launch |
|---|---|---|---|---|---|---|---|
| HGB | 0 (always) | 1.000 | 0.568 | 518 | 79.6 | 21.2% | 79.6 |
| HGB | **0.5** | 0.712 | 0.631 | 315 | 63.0 | 16.8% | 88.6 |
| HGB | **0.7** | 0.164 | 0.761 | 47 | 17.3 | 4.6% | 105.4 |
| HGB | **0.9** | 0.028 | 1.000 | 0 | 3.9 | 1.0% | 137.2 |
| oracle | 0.5 | 0.729 | 0.619 | 333 | 63.8 | 17.0% | 87.5 |
| oracle | 0.7 | 0.186 | 0.767 | 52 | 19.9 | 5.3% | 106.8 |
| oracle | 0.9 | 0.028 | 1.000 | 0 | 3.9 | 1.0% | 137.2 |
| no model: always speculate "delivered" | - | 1.000 | 0.500 | 600 | 70.2 | 18.7% | 70.2 |

**Comparison with the no-model constant.** The constant "always speculate delivered" saves 70.2 ms/call. The HGB at the requested tau values saves less: 63.0 (tau 0.5), 17.3 (0.7), 3.9 (0.9). Only tau = 0 (speculate on everything) beats the constant, by 9.5 ms, which is 2.5% of the 376 ms sequential time. A wasted read-only launch is free in this accounting, so tau has no latency value; it only removes wasted launches (518 -> 315 -> 47 -> 0). The 100% hit rate at tau 0.9 rests on 34 calls.

HGB calibration of the top probability: bin 0-0.5 hit 0.413 (mean p 0.448, n=346); 0.5-0.7 hit 0.592 (0.583, n=657); 0.7-0.9 hit 0.712 (0.752, n=163); 0.9-1.0 hit 1.000 (0.989, n=34).

### 3. issue_refund (the call changes the world: predict the outcome, set up the fallback)

Train 4800, test 1200. Error rate 0.209 (train), 0.204 (test). Test outcomes: ok 0.796, vendor_error 0.081, insufficient_funds 0.077, timeout 0.046.

| P(error) predictor | AUC-ROC | Brier |
|---|---|---|
| HGB (amount, category, hour, recent error rate, last outcomes) | 0.706 | 0.1487 |
| base rate only | 0.500 | 0.1625 |
| `recent_error_rate` alone (raw feature) | 0.594 | - |
| oracle: env's true P(error), including the hidden regime | 0.762 | 0.1381 |

"Fallback-first" rule (set up the fallback before calling when predicted P(error) exceeds a threshold):

| predicted P(error) > | calls flagged (of 1200) | precision | precision 95% CI | recall (any error) | recall (timeouts) | base error rate |
|---|---|---|---|---|---|---|
| **0.5** | 38 | 0.421 | [0.26, 0.59] | 0.065 | 0.018 | 0.204 |
| 0.35 | 204 | 0.412 | [0.34, 0.48] | 0.343 | 0.255 | 0.204 |
| 0.2 | 488 | 0.336 | [0.29, 0.38] | 0.669 | 0.618 | 0.204 |

At the specified threshold 0.5 the rule flags 38 calls and catches 6.5% of errors; that is not the "useful precision/recall" the spec asks for (precision is about 2x the base rate, but on 38 calls with a wide interval, and almost no recall).

Given an error occurred, the most likely error type is identified 63.3% of the time (always "vendor_error": 39.6%).

Latency regression (HGB, absolute-error loss) versus predicting the training median (583 ms):

| latency predictor | MAE ms, all test calls | MAE ms, excluding timeouts |
|---|---|---|
| HGB regressor | 750 | 239.7 |
| constant train median | 791.4 | 275.7 |

### 4. Outage test (1000-transaction import, vendor_lookup at 100% errors)

Circuit breaker (opens after 5 consecutive failures; half-open probes would also fail in a full outage) made 5 real calls and then fast-failed everything. Dreaming uses the certified predictor (cutoff 0.870, kept LR) and otherwise a registered fallback: a stale snapshot covering a random 60% of merchants, 3% of whose entries point at the wrong vendor. 21.1% of the transactions are from merchants novel to the predictor; 2 of them were confidently predicted (and wrong).

How the snapshot resolves a raw descriptor matters, so there are two variants. **Idealised** looks the descriptor up in the env's hidden descriptor-to-merchant map (perfect resolution: an upper bound). **Normalised-descriptor match** is an exact match on lowercase, digit-masked descriptors seen in the training window for covered merchants (what a real string-keyed snapshot could do). The outage stream is drawn from the same 30 unseen merchants as the test stream, so it is not independent of the test stream (the cutoff was not tuned on it).

| strategy | snapshot resolves descriptors by | completed | error rate among completed | n completed |
|---|---|---|---|---|
| circuit breaker (baseline) | - | 0% | - | 0 |
| certified predictor only (rest incomplete) | - | 65.6% | 0.30% | 656 |
| registered fallback only (stale snapshot) | idealised (upper bound) | 68.3% | 2.78% | 683 |
| **Dreaming: certified predictor, else fallback** | idealised (upper bound) | **87.8%** | **1.94%** | 878 |
| - answered by fallback (rows tagged for downstream gates) | idealised (upper bound) | 22.2% | 6.76% | 222 |
| registered fallback only (stale snapshot) | normalised-descriptor match | 35.0% | 1.43% | 350 |
| **Dreaming: certified predictor, else fallback** | normalised-descriptor match | **67.3%** | **0.45%** | 673 |
| - answered by fallback | normalised-descriptor match | 1.7% | 5.88% | 17 |

## Interpretation for the mission

- **Skip for input-determined tools is mechanically feasible and pays.** With a 1% budget the kept TF-IDF + LR predictor certified 65% calibration coverage and skipped 67% of the main-world test calls with 1 error in 792 (0.13%); pooled over six worlds it skipped 69.9% at 0.42% error (21/4943). It cut vendor_lookup time by 66% including most of the slow tail. NN certifies too (35% calibration coverage in the main world, certifiable in all six worlds); the first version's "NN cannot be certified at 1%" was a search-path artifact.
- **No predictor is safe by default and certified coverage cannot rank them.** LR was kept in all six worlds. It broke the 1% budget in seed 11 (1.46%) where NN held (0.47%); NN broke it in seed 13 (3.41%) where LR held (0.66%). So "the choice of predictor matters" stands, but the rule "keep higher certified coverage" picked the riskier model in one world and the safer one in another; it is blind to the failure mode (novel merchants). The choice also reuses the calibration slice (two candidates, uncorrected).
- **The certificate is only as good as the novelty mix in the calibration slice, and the test mix matches it here by construction.** Re-weighting the test stream to 30-50% novel merchants (novelty-shift table in section 1) pushes the pooled error above the 1% budget at about 35-40% novel and adds seed 13 to the budget breaches at 30%. Calibrating on known merchants only breaks the 1% budget in 1 of 6 worlds (seed 11: it certified full coverage, 19.7% test error, 91% of novel-merchant calls confidently skipped; the other five stay under 1%, worst 0.92% in seed 13, but with 0.8-3.1% false skips on novel merchants). At a 2% budget it breaks the budget in 5 of 6 worlds (17.3% pooled error, 80% of novel-merchant calls skipped). The main-world comparison alone (1 false skip either way) shows nothing; the pooled one does.
- **What breaks: the iid assumption behind the Clopper-Pearson bound.** Errors cluster by merchant: the 21 pooled errors of the kept LR (all confident wrong answers on novel merchants) come from 9 merchants (6 in seed 11 with at most 4 from one; all 5 of seed 13's from one merchant; seed 15's 2 from one; seed 4's single error); NN's 27 errors in seed 13 come from 4 merchants (16 from one). 1 of 6 worlds breaks the 1% budget (seed 11, 1.46% against 1%; 5.3% false skips on novel merchants) and 0 of 6 break 2%. Mission step 5 (audit sample, re-certify) is not optional, and calibration splits should be grouped by input identity. One seed per world: 1-in-6 has very wide error bars.
- **Speculation on a hidden-state tool buys almost nothing here.** The status is only partly predictable from visible features (HGB 0.568 top-1, exact Bayes predictor 0.580, majority 0.500). The HGB at tau 0.5/0.7/0.9 saves less latency (63.0, 17.3, 3.9 ms/call) than the no-model constant "always speculate delivered" (70.2 ms/call); only tau 0 beats it, by 9.5 ms (2.5% of the 376 ms sequential time). Wasted launches are free in this accounting, so a threshold has no latency value; the model only reduces wasted launches (518 -> 315 -> 47 -> 0). It would matter only if a wasted launch had a cost.
- **Fallback-first for the world-changing tool is weak.** AUC 0.706 against an oracle ceiling of 0.762. At the specified threshold of 0.5 the rule flags 38 of 1200 calls with precision 0.42 (95% CI 0.26-0.59, base rate 0.20) and recall 6.5%: it does not show useful precision and recall. At 0.35 it flags 17% for 0.41 precision and 0.34 recall. Latency regression beats the median by 5% overall and 13% excluding timeouts.
- **Outage: completion is real, the size of it depends on how the snapshot resolves descriptors.** The breaker completes 0%. Predictor plus idealised-resolution snapshot completes 87.8% at 1.94% error (upper bound). With an exact-match normalised-descriptor snapshot only 35% of the transactions resolve from the snapshot, and Dreaming completes 67.3% at 0.45% error, almost all of it from the certified predictor (65.6% at 0.30%). Fallback-sourced rows carry 5.9-6.8% error and should be tagged for downstream gates.

## Caveats

- Synthetic: merchant names, descriptor noise, order dynamics, refund error structure and the "degraded bank" regime are my inventions. Real descriptors are messier and more adversarial; real merchants share names and acquirers; real order state and refund failures depend on things the env does not model. Predictability numbers (67-70% skipped, 0.57 status accuracy, AUC 0.71) mean nothing about real tools.
- The label noise floor is set by the env (vendor errors are 0 for known merchants; refund outcomes are sampled from known probabilities). A real tool has returns that change over time (vendor re-mapped, order states edited), so a certificate would drift and need audit sampling, which is not simulated.
- `certify_grid` is my replacement, not a published algorithm: a union bound over 20 pre-specified coverage levels with tie-closed cutoff sets. It is path independent, but coarse (5-point coverage steps), and it uses a smaller per-level delta than the original, so it can certify a little less. Its cutoffs are data-dependent quantiles of the calibration scores, so the union bound over the 20 levels is not strictly valid either; "delta 0.05" is nominal. The calibration and test streams have the same novelty rate (about 20% novel merchants) by construction; a different test novelty rate was only approximated by re-weighting (novelty-shift table). The LR-vs-NN selection and the certification share the same calibration slice (two candidates; no multiplicity correction). The oracle rows use the env's hidden ground truth and are diagnostics only.
- `common.py` bug (reported, not fixed there): `certify` is path dependent (see Revision). It also raises on empty input (`np.log2(0)`); S9 guards the empty case.
- Fallback is idealised in the first outage variant: the snapshot resolves any descriptor of a covered merchant via the env's hidden map. The second variant (normalised exact match) is closer to a real string-keyed snapshot but still generous (it only sees descriptors in the training window and has no fuzzy matching). Staleness is a fixed 3% of entries (chosen by merchant, so lumpy). The outage stream shares its unseen merchants with the test stream.
- `issue_refund` is called for every transaction and the refund has no effect on `lookup_order` or later calls; successive refund outcomes depend on each other only through the hidden regime.
- Speculation accounting ignores the cost of wasted launches and assumes the speculated step starts at t=0 and takes 150 ms.
- Process notes (not results): an early version of `env.py` had a slice bug (the "unseen test" group overlapped the seen group), found because the test novel share was 8% instead of 20%; fixed before the numbers above. HistGradientBoosting was 100x slower than expected on this machine under CPU contention until `OMP_NUM_THREADS=1` was set (now set at the top of `run.py`). In this revision an edit to the fallback-first table silently did not apply once (a string replacement with an escaped newline matched nothing) and was caught only because the table in `log.txt` still had the old columns; it is now applied and visible in the table above.

## Time and cost

- Wall time for the final `python run.py` (simulation, all fits, outage test, 5 robustness worlds, novelty-shift tables): 83 s on a shared CPU with one thread (earlier runs of the same code: 55-156 s depending on CPU contention).
- Cost: $0 (no API calls).
- Files: `env.py` (unchanged), `run.py` (revised: `certify_grid`, `selfcheck`, pooled/per-merchant tables, speculation comparison, refund CI, outage variants), `log.txt` (revised run), `log_v1.txt` (first version), `err.txt` (progress timing), `traces.jsonl` (18,000 rows, seed 4).
