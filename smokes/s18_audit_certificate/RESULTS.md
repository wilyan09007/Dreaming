# S18: an anytime-valid audit certificate (orthogonal to the one-shot certificate)

## In plain words

Dreaming promises that a student's mistakes stay under a budget such as 1%. Today that promise is checked once, on old examples, and S4 showed it can be wrong when mistakes clump by merchant. I tested a standing guarantee instead: keep sending a random 5-20% of the student's confident answers to the real tool, and keep a running upper limit on the error rate that stays valid however long you watch and however the errors clump or drift. The valid version I built (a Chernoff-form e-process, my own construction) never fell below the true error in 192 test configurations (worst 4% of audit draws, nominal 5%), while the "just average the audited answers" estimate was wrong 88% of the time and an exact confidence interval re-read every 20 calls was wrong 6.5% of the time (up to 28%). The price is time and money: with 10% audits it takes about 5,300 served calls to confirm a 1% budget even if the student made no mistakes at all (the rule is roughly 5.3 divided by audit rate times budget), and the audits add 24% more real calls (36.8% of traffic is paid instead of 29.7%). In the one world where the one-shot certificate was wrong (1.57% real error against a 1% promise) the audit limit never confirmed 1%, so it would not have made the false claim, but it could not prove the breach quickly either (only 2-19% of runs raised a proven alarm within 10,000 calls). For Dreaming: keep the one-shot certificate as the day-zero promotion gate and make the audit bound the standing certificate, but it is a slow, paid confirmation and not a faster certificate, and on streams whose errors do not clump (Banking77) the one-shot certificate never broke and the audit bound only adds cost.

## Setup

- **Worlds.** The S4 synthetic environment (`smokes/s4_tool_predictor/env.py`, not edited) for seeds 4, 11, 12, 13, 14, 15, `vendor_lookup` only. Chronological split as in S4: transactions 0-2999 train, 3000-4799 calibration, 4800 onward is the serving stream, extended from S4's 1,200 test transactions to 13,200 (about 12,950 successful calls per world, about 20% from merchants never seen in training). The first 1,200 transactions are bit-identical to S4: the S4 reproduction check printed world 11 LR 13 errors in 889 skipped calls and world 4 LR 1 in 792 at a 1% budget with S4's grid-20 certifier, matching S4's RESULTS (`log_build_earlier_local.txt`).
- **Predictors.** S4's two text predictors (char n-gram TF-IDF with logistic regression, "LR", and per-class nearest neighbour, "NN"), confidence = top-2 gap. Streams: S4 LR (6 worlds), S4 NN (6), and "S4 LR shift" (6): the LR stream whose second half is resampled with novel-merchant calls 4 times as likely (a bootstrap of the same pool, not a new world).
- **Control.** Banking77 with the S1 student (frozen bge-small embeddings + logistic regression) trained on 770 and 3,080 rows, served on every row it never trained on (12,299 and 9,989 rows, accuracy 0.832 and 0.910), three random calibration/serving splits per size (calibration 1,540 rows, serving order random, so errors do not cluster by construction).
- **One-shot certificates (the current method and (c)).** (a) `common.certify`, fixed-grid Clopper-Pearson (grid 40, delta 0.05) on the calibration slice, budgets 1%, 2%, 5%. (c) Learn-then-Test fixed-sequence testing with data splitting (`cs.ltt_fst`: half the calibration rows pick the starting cutoff, the other half test upward in coverage with the exact binomial test, stop at the first non-rejection). The served set is every call with confidence at or above the cutoff (70-80% of traffic).
- **Audit stream.** Every served call is audited with known probability p in {5%, 10%, 20%} by an independent coin; an audit reveals the error indicator e_t. Y_t = A_t e_t; the inverse-propensity estimate is Y_t / p. 100 independent coin sequences per stream x budget x p (40 for WSR, which is slow). The target is the **realized running error rate of all served traffic**, the quantity S4's certificate failed on.
- **Bounds compared.** (b) `naive`: running audited error rate. `cp_peek`: fixed-sample Clopper-Pearson on the audit stream recomputed every 20 calls (continuous peeking, not valid). `wsr`: predictable-plug-in betting confidence sequence of Waudby-Smith and Ramdas (JRSSB 2024, arXiv 2010.09686), one-sided, my own ~25-line implementation in `cs.py` because pip `confseq` needs a C++ toolchain and did not build here. `chern`: design-based e-process (my construction): for any lambda > 0, exp(-lambda sum Y + (1-e^-lambda) p sum e) is a nonnegative supermartingale for **any** error sequence because only the audit coins are random; a union over a 10-point lambda grid plus Ville's inequality gives a time-uniform bound on the running error count (and a mirror-image lower bound). (d) Group-conditional variants: the same bounds per group at delta/G, summed. Groups: `tpl` (descriptor template never seen in training; observable at serving time), `orc` (merchant never seen in training; oracle, not observable at serving time), `confbin` (confidence above or below the served median; Banking77).
- **Reference cross-check.** The WSR code is checked against the reference `confseq.betting.betting_cs` (confseq 0.0.11 from the PyPI sdist; its `betting.py` is pure Python, imported with `matplotlib`/`multiprocess` stubbed and a `np.float_` shim). See R8 below.
- **Files.** `run.py` (build / analyze / confseq / report), `cs.py` (bounds + self-check), `rows.json` (1,271 result rows), `report.md` (all tables R1-R8), `log.txt` (pod log), `confseq_check.json`, `cache/*.npz` (world and Banking77 streams).

## Results

Self-check (`cs.py`, pod): on an iid 1% error stream the share of audit-coin sequences where the bound ever fell below the realized error is 0.000 for the e-process and 0.020 for WSR; on an adversarial stream with 40 errors in the first 60 calls it is 0.000 and 0.007; the lower bound never exceeded the realized error. With zero errors at t = 1000 and p = 10% the e-process bound is 5.3% (rule of three would say 3%: the price of validity at every stopping time).

### 1. Does a bound ever fall below the realized error? (violations)

Share of audit-coin sequences in which the bound is below the running realized error at some call, averaged over streams x budgets x audit rates (`report.md` R3):

| bound | valid by theory? | configs | mean violation | worst config | configs above 5% |
|---|---|---|---|---|---|
| naive audited error rate | no (a point estimate) | 192 | 87.8% | 100% | 177 |
| Clopper-Pearson re-read every 20 calls | no (peeking) | 192 | 6.5% | 28% | 93 |
| WSR betting CS | only for a constant mean | 100 | 1.6% | 32.5% | 7 |
| **Chernoff e-process, pooled** | **yes, any error sequence** | **192** | **0.32%** | **4%** | **0** |
| Chernoff, per observable template group | yes | 159 | 0.01% | 1% | 0 |
| Chernoff, per oracle novelty group | yes | 159 | 0.01% | 1% | 0 |
| WSR, per template group | constant mean per group | 89 | 0.11% | 10% | 1 |

By setting (mean violation): Chernoff S4 LR 0.31%, S4 NN 0.31%, S4 LR shift 0.26%, Banking77 0.42%; WSR 1.25%, 0.88%, 3.6%, 1.14%. WSR's seven configs above 5% are four S4 LR stationary ones (7.5-10%, i.e. 3-4 of 40 sequences; three of them at the 5% budget) and three shift ones (7.5%, 12.5% and 32.5%, the last in world 11 at the 5% budget): clustered and drifting errors break its constant-mean assumption, as expected. The Chernoff form is the only pooled bound with no config above 5%. (Statistical noise: 100 coin sequences per config gives about +/-2 points on a rate near 5%; WSR has 40.)

Per world, S4 LR, audit rate 10%, mean over the three budgets (R3):

| world | naive | CP re-read | WSR | Chernoff | Chernoff per template group | one-shot cp over budget (of 3 budgets) |
|---|---|---|---|---|---|---|
| 4 | 0.947 | 0.03 | 0 | 0 | 0 | 0/3 |
| 11 | 0.983 | 0.11 | 0 | 0.0067 | 0 | 2/3 |
| 12 | 0.917 | 0.013 | 0 | 0 | 0 | 0/3 |
| 13 | 0.983 | 0.083 | 0.042 | 0.010 | 0 | 0/3 |
| 14 | 0.303 | 0.033 | 0.025 | 0 | 0 | 0/3 |
| 15 | 0.96 | 0.053 | 0.025 | 0.0033 | 0 | 0/3 |

**False certification** (R3b): on streams whose final realized error is above the budget, the share of coin sequences in which the bound still dropped under the budget at some call: Chernoff 0% (S4 LR, S4 NN), 0.2% (shift); WSR 0% (LR, NN), 1.7% (shift); CP re-read 0.8% / 1.0% / 9.1% (worst config 38% under drift); naive 99-100%. The one-shot certificates did it in R2 below.

### 2. The one-shot certificates on the long serving stream (R1, R2)

Streams (stream x budget) whose realized error over all served calls exceeded the budget:

| setting | one-shot cp (fixed-grid CP) | LTT fixed-sequence | which |
|---|---|---|---|
| S4 LR | 2 of 18 (1% budget: 1 of 6 worlds; 2%: 0 of 6; 5%: 1 of 6) | 3 of 16 | world 11: 1.57% against 1%, 6.66% against 5%; LTT also breaks 2% in world 11 (2.42%) |
| S4 NN | 3 of 17 | 2 of 14 | world 13: 2.53% against 1%, 3.43% against 2%; world 11: 6.93% against 5% |
| S4 LR shift | 3 of 18 | 3 of 16 | world 11 at all three budgets (3.19% against 1% and 2%, 13.0% against 5%) |
| Banking77 control | **0 of 11** | **0 of 7** | iid errors: the one-shot certificate held every time (realized 0.3-0.7% at 2%, 1.9-3.2% at 5%) |

Same failure as S4 (1 of 6 worlds at 1%, realized 1.46% in S4's first 1,200 transactions, 1.57% over the long stream). LTT with data splitting does not fix it: it is also an iid-calibration argument.

### 3. Calls until the audit bound confirms the budget, and tightness (R4, R6)

S4 LR stationary, audit rate 10%, worlds whose realized error is within the budget (5, 6, 5 worlds), median over worlds of the per-world median over coin sequences. "Calls" = served calls; realized error at the end is 0.17%, 0.18%, 0.94%:

| budget | bound | calls to confirm (share of sequences that confirm by the end) | bound at 1k | at 5k | at all (~9k) |
|---|---|---|---|---|---|
| 1% | Chernoff pooled | 8,500 (68%) | 5.3% | 1.06% | 0.96% |
| 1% | WSR | 4,270 (71%) | 4.0% | 0.81% | 0.82% |
| 1% | Chernoff per template group (observable) | never (0%) | 12.0% | 2.4% | 1.72% |
| 1% | Chernoff per oracle novelty group | 7,490 (80%) | 6.2% | 1.3% | 0.84% |
| 2% | Chernoff pooled | 2,650 (77%) | 5.3% | 1.38% | 0.97% |
| 2% | WSR | 2,130 (77%) | 4.0% | 1.02% | 0.76% |
| 2% | Chernoff per template group | 7,620 (65%) | 12.0% | 2.7% | 1.66% |
| 5% | Chernoff pooled | 1,700 (99.8%) | 8.5% | 2.9% | 2.35% |
| 5% | WSR | 803 (99%) | 6.2% | 2.5% | 2.16% |
| 5% | Chernoff per template group | 3,900 (97%) | 15.2% | 4.3% | 3.06% |

The floor with zero errors is calls >= ln(10 / 0.05) / (p x budget) = 5.3 / (p x budget): 5,300 calls at p = 10% and a 1% budget, 2,650 at 2%, 1,060 at 5%; 10,600 at p = 5% and 2,650 at p = 20% for the 1% budget. The zero-error worlds match it (world 14: 5,300 and 2,650; Chernoff). Even after about 9,000 calls the bound is 5.6x (1% budget), 5.4x (2%) and 2.5x (5%) the realized error.

Audit rate sweep, 1% budget, S4 LR (R4/R6): median calls to confirm (share confirming) Chernoff: p = 5% never (0%), 10% 8,500 (68%), 20% 4,250 (85%); WSR: 8,540 (57%), 4,270 (71%), 2,140 (85%).

### 4. What the audit stream triggers where the one-shot certificate broke (R5)

Lower-bound alarm = the lower bound on the running error rises above the budget. Streams where the cp certificate exceeded its budget; call of the first alarm, median over coin sequences (share of sequences that ever alarm in brackets; "never" = fewer than half alarm in about 10,000 calls):

| stream | budget (realized) | audit rate | pooled lower bound (valid) | per template group, any group above budget (observable) | per oracle novelty group, any (not observable) | CP re-read (not valid) |
|---|---|---|---|---|---|---|
| S4 LR world 11 | 1% (1.57%) | 5% | never (2%) | 8,660 (61%) | 2,230 (100%) | 9,470 (55%) |
| | | 10% | never (13%) | 3,750 (91%) | 1,050 (100%) | 3,520 (74%) |
| | | 20% | never (19%) | 2,400 (100%) | 588 (100%) | 2,790 (85%) |
| S4 LR world 11 | 5% (6.66%) | 10% | never (29%) | 1,210 (100%) | 328 (100%) | 1,810 (96%) |
| | | 20% | 9,740 (55%) | 594 (100%) | 185 (100%) | 710 (98%) |
| S4 NN world 13 | 1% (2.53%) | 10% | 5,050 (81%) | 1,040 (100%) | 438 (100%) | 700 (100%) |
| | | 20% | 2,500 (99%) | 533 (100%) | 246 (100%) | 380 (100%) |
| S4 LR shift world 11 | 1% (3.19%) | 10% | 6,790 (94%) | 4,420 (100%) | 1,370 (100%) | 5,360 (99%) |

False alarms at audit rate 10% on streams where the cp certificate held (share of coin sequences that alarmed anyway): pooled lower bound 0% (S4 LR, shift, Banking77) and 0.07% (NN); per-template-group alarm 2.4% (LR; worst config 30%), 8.9% (shift; worst 88%), 7.1% (NN); per-oracle-group alarm 57% (LR), 66% (shift), 44% (NN) because the novel-merchant group's error rate legitimately sits above the pooled budget; CP re-read 0.7-1.8%. Summing the template-group lower bounds into a pooled bound is no more powerful than the plain pooled bound (6,790 vs 7,120 calls in shift world 11 at 10%).

Upper-bound side in world 11 at p = 10%: the Chernoff bound is 8.5% at 1k, 4.25% at 5k and 3.34% at the end (10,055 calls, realized 1.57%). It ends at 3.34% against the 1% budget, 3.27% against the 2% budget (the realized 1.57% is inside 2%, so the audit bound is simply too loose to confirm it) and 9.6% against the 5% budget (realized 6.66%). The share of sequences that ever confirm the budget is 0% (1%), 2% (2%) and 0% (5%). So the audit certificate never makes the false claim the one-shot certificate made at 1% and 5%, but it shows the failure as "not confirmed" and not as a proven breach.

### 5. Cost of auditing vs tightness (R6)

S4 LR, Chernoff pooled bound, 1% budget, the 5 worlds within budget. The student serves 70.3% of calls; without audits 29.7% of calls still go to the real tool:

| audit rate | paid share of calls | extra real calls vs no audit | bound at 1k | at 5k | at ~9k | calls to confirm |
|---|---|---|---|---|---|---|
| 5% | 33.2% | +3.5 points (+12%) | 10.6% | 2.12% | 1.21% | never |
| 10% | 36.8% | +7.0 points (+24%) | 5.3% | 1.06% | 0.96% | 8,500 |
| 20% | 43.8% | +14.1 points (+47%) | 2.65% | 0.85% | 0.71% | 4,250 |

The bound at call t scales like 1/(p t): doubling the audit rate roughly halves the calls needed, at the cost of the extra real calls above.

### 6. Banking77 control (iid errors), audit rate 10% (R7)

| budget | method | violation | calls to confirm | share confirming | bound at all calls | realized |
|---|---|---|---|---|---|---|
| 2% (5 streams) | Chernoff | 0% | never | 18% | 2.68% | 0.46% |
| 2% | WSR | 0% | 3,130 | 53% | 2.01% | 0.46% |
| 2% | CP re-read | 2% | 2,250 | 77% | - | 0.46% |
| 5% (6 streams) | Chernoff | 0.7% | never | 46% | 5.53% | 2.91% |
| 5% | WSR | 2.1% | never | 48% | 5.35% | 2.91% |
| 5% | CP re-read | 13.8% | 2,030 | 94% | - | 2.91% |
| 5% | naive | 97.7% | 8 | 100% | 2.87% | 2.91% |

The one-shot certificate held in all 11 certifiable stream-budget pairs. The audit bound ends at about 2x (5% budget) to 6x (2% budget) the realized error. Within the 2,300-7,200 served calls of these streams the Chernoff bound confirms the budget in 18% (2%) and 46% (5%) of audit runs, WSR in 53% and 48%. Per-confidence-bin group bounds are looser still (4.3% and 6.8% at all calls, confirming in 0% and 6% of runs).

### 7. Cross-check of my WSR against the reference `confseq` (R8; 1% budget, p = 10%, 3 coin sequences each)

| world | method | bound at 1k | at 5k | at the end | ever below realized | calls to under 1% |
|---|---|---|---|---|---|---|
| 4 (T = 8,872, 15 errors, realized 0.169%) | mine (`cs.wsr`) | 4.04% | 1.24% | 0.71% | 0 | 6,380 |
| 4 | `confseq.betting_cs` | 6.00% | 1.50% | 0.85% | 0 | 7,370 |
| 11 (T = 10,055, 158 errors, realized 1.57%) | mine | 6.15% | 2.85% | 2.48% | 0 | never |
| 11 | `confseq.betting_cs` | 7.35% | 2.90% | 2.45% | 0 | never |

The two agree to within 2 points of error rate at 1k calls and 0.15 points at the end; the reference is the more conservative one throughout (its bet sizes differ and its grid adds 1/20000 on the audited-error scale). Neither ever fell below the realized error and both confirm or fail to confirm the 1% budget in the same worlds. The S4 results above use mine.

## Interpretation for the mission

**Should the certificate be the audit confidence sequence, with the one-shot certificate only as the promotion gate? Yes for the standing guarantee, with three conditions.**

1. **Use the e-process form, not a betting CS or a re-read confidence interval, if the claim is about the realized error of what was served.** It is the only pooled bound with no violating config (0 of 192, worst 4%) under clustered errors, drift and adversarial ordering, because it only relies on the audit coin. WSR is about 24% tighter (4.0% vs 5.3% at 1k; confirms up to 2x sooner at a 1% budget, 1.25x at 2%) but broke its 5% allowance in 7 of 100 configs (worst 32.5% under drift). The naive running error is not a bound at all (wrong 88% of the time), and peeking at Clopper-Pearson fails 6.5% of the time on average and 28% at worst.
2. **It cannot replace the day-zero gate.** Confirming a 1% budget needs about 5.3 / (p x 1%) served calls even for a perfect student: 5,300 at p = 10%. A call site with 100 calls a day waits 53 days at that rate. The one-shot certificate answers on day zero with 1,800 calibration rows, and on Banking77 (iid errors) it never broke. So: one-shot certificate = provisional promotion; audit bound = what stays on the certificate record ("confirmed" when the upper bound is under the budget).
3. **The audit adds real cost and is slow to react.** At p = 10% it adds 24% more real calls on top of the 29.7% still paid (7.0 points of traffic), and halving the confirmation time costs another doubling of audits. In the world where the one-shot certificate was wrong the pooled valid alarm fired in only 2-19% of runs (1% budget, p up to 20%, 10,000 calls); it needs a large breach: NN world 13 at 2.5x the budget alarmed in 81% of runs at p = 10% and 99% at p = 20%, while LR world 11 at 1.3x the 5% budget alarmed in 55% of runs at p = 20% and fewer at lower rates. The reliable detector in these runs was a group alarm: "some template group's error rate is certainly above the budget" fired in 91% of runs by 3,750 calls (p = 10%) in world 11, at a false-alarm rate of 2.4% on healthy streams (worst config 30%; 7-9% on NN and shifted streams). That alarm is a diagnostic that triggers re-certification, not a budget statement, and the oracle novelty variant is not available in production (it alarms 44-66% of the time on healthy streams because the novel-merchant group is legitimately worse than the pool).

What the design should be, from these numbers: promote on the one-shot certificate; start auditing at a high rate (20%: confirms a 2% budget in about 1,300 calls and a 1% budget in about 4,250) and decay it as the bound tightens; treat "bound not under budget after N calls" as a re-certify trigger, and add an observable group alarm (input template or novelty flag) for faster detection. Group-conditional bounds do not help certification (they are 2-3x looser: 1% never vs 8,500 calls) and only help diagnosis.

Not tested here: a decaying audit rate (the e-process works with any known p_t, but I only ran constant p), audit coins correlated with the input (the guarantee requires independent coins), and label noise in the audit (the audit is assumed to reveal the true error).

## Caveats

- **Synthetic worlds, few breaches.** The S4 environment is my own generator; the clustering comes from how it samples merchants. Only one LR world (11) and one NN world (13 at 1-2%, 11 at 5%) breach their budget, so the breach-detection numbers in section 4 rest on essentially one stream per row, and coin-sequence variation (100 sequences) is the only replication. The 192 validity configs are 192 views of 24 fixed error streams, not 192 independent worlds.
- **Violation rates are over audit coins only.** The guarantee is design-based (the error sequence is held fixed), which is why it covers clustering and drift. For the "shift" stream the bound at time t describes traffic served so far; a rising error later leaves it stale until new audits arrive (Chernoff compared with the **final** realized error is "below" in 12.9% of coin sequences on average in the shift streams, by construction; the check that matters, against the running error, is the 0.26% above).
- **WSR and the group variants have fewer sequences (40 for WSR) and only run at all audit rates on the main S4 LR stationary streams; elsewhere WSR runs at p = 10%.** Its 32.5% worst-config violation is 13 of 40 sequences.
- **LTT is a simple fixed-sequence implementation with data splitting.** Half the calibration rows test, so it may certify less than the best published variants; it was not tuned.
- **The shift scenario is a bootstrap of the same pool** (second half resampled with novel-merchant calls 4x as likely), not a new simulation.
- **Data provenance.** The `cache/*.npz` streams were built by `run.py build` on the laptop in an earlier interrupted attempt (125 s, 6 worlds in parallel; `log_build_earlier_local.txt`) before the pod-only rule applied to this smoke; this session ran only two `--tiny` checks locally (5-7 s each) and all analysis on pods. Rebuilding needs `smokes/s4_tool_predictor/env.py` and embeddings.
- **Reference cross-check is on 3 coin sequences x 2 worlds** and used confseq 0.0.11's pure-Python module with stubs and a NumPy shim, not a pip install.
- **The Banking77 control is one student family.** The streams use the S1 student only; the pool excludes the student's training rows, and the serving order is a random permutation, so any real-world temporal clustering is absent.

## Time and cost

- API spend: $0 (no model calls).
- Pod run 1: RTX 3090 at $0.50/h (the A5000 had no supply: SUPPLY_CONSTRAINT), 5.9 min wall, $0.049. Pod run 2 (adds the pooled group lower bound, the CP re-read lower alarm, the false-certification table R3b, and fixes the OVER column in the report; all 1,271 result rows shared with run 1 are identical, so the analysis is deterministic): RTX 4090 at $0.89/h (A5000, 3090, A4000 and A4500 had no supply), 5.9 min wall, $0.088. Total pod cost $0.137. Compute on the pod: analysis 33 s, reference cross-check about 155 s; the rest is boot and upload.
- Laptop: two `--tiny` checks (5-7 s each) plus a few seconds of report and aggregation; the earlier 125 s cache build is described under Caveats.
