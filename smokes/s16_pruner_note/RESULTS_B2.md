# S16b v2 - Does pruning draft_note stay at the noise floor under a fact-based equivalence?

Cost $1.63 of a $3.00 cap (4,482 new calls, 557 s). Code: `run_b2.py` (driver), `equiv.py` (equivalence), `../s16c_judge_floor_offline/canon.py` (canonicalizer); plan and definitions in `RUN_B2_PLAN.md`; raw log `log_b2_live.txt`, results `results_b2.json`. Every number below is recomputed from `results_b2.json` (and the by-scenario floor lines of the log) by `report_b2.py`, which also writes `tables_b2.md` with the full tables. Nothing in `run_b2.py`, `equiv.py` or the caches was changed.

## In plain words

S16 said two independent full-prompt notes disagree 13-16% of the time under a strict judge, so a 5% or 10% error budget looked unreachable on prose. S16c showed that almost none of that disagreement is about facts. S16b v2 redefines "same note" as "same facts" (vendor, amounts, transaction date, invoice outcome, flag decision, flag reason), decided by a deterministic extractor where it can and by a two-order judge with an adjudicator only on the leftovers, and then re-measures everything on the same 140 prompts with that definition.

Four answers.

1. **The floor fell from 11.7% (strict) to 2.2% (new) on the same 180 pairs (4/180, one-sided 95% upper bound 5.0%).** That says two independent full-prompt notes agree on the facts in 98% of pairs; it does not say the notes are the same prose. The drop is by construction partly a change of question: optional history detail (does the note mention the vendor's history, a range, a count, the bank amount?) no longer counts, and 150 of the 180 pairs are decided "same" by the extractor without any judge. The strict column still shows 11.7% raw / 7.8% confirmed on those pairs, so if a user does care about that detail, nothing has changed for them.
2. **With about 5 or 2 history records kept per prompt, free-text pruning is not distinguishable from the new floor; with no history, or random history, it is distinguishable.** Learned, cross-encoder and oracle selection at m=5 and m=2 give 2.9-3.6% DIFFERENT against a 2.1% floor on the same rows (discordant pairs 3 vs 1-2, p = 0.63-1.00) while removing 59-70% of input tokens. Dropping all history (m=0) gives 11.4% (15 vs 2 discordant, p = 0.002) and the rule that keeps only transaction, invoice and flag gives 6.4% (8 vs 2, p = 0.11). Random selection of history at m=2 gives 9.3% (12 vs 2, p = 0.013). These tests have little power (3 vs 2 discordant pairs cannot separate anything), the upper bounds are 6.4-7.4%, and "indistinguishable" is not "equal". Under the strict rubric on the same 140 rows the floor is 12.1% raw / 7.1% confirmed, and learned m=5 is 24.3% / 22.1% and m=2 is 30.0% / 28.6%, roughly two to four times the strict floor: the core equivalence leaves out exactly the history evidence that pruning removes, so the pruned notes keep the facts but visibly lose history detail.
3. **Typed is not better than free text here.** The typed floor is 3.6% (5/140) against 2.1% free text on the same rows, and every typed cell that keeps history records and the rule cell is higher than its free-text twin (8.6-12.9% against 2.9-9.3% for the history-keeping cells, 20.7% against 6.4% for the rule); only m=0 is lower (8.6% against 11.4%). Seven of the ten typed cells are above the typed floor at p < 0.05, against two of ten free-text cells. The typed equivalence is stricter (the reason category must match exactly), so this is not a like-for-like comparison, but it is the opposite of the hope that typing is the safe route to 5%.
4. **Nothing certifies, cross-fitted, at 5, 10 or 15%, in any of the 23 cells, plain or excess-over-floor.** The reason is entirely structural, so the empty table is not a finding about pruning quality: every calibration fold is exactly 28 prompts (vendor GroupKFold gives 28/28/28/28/28), and the grid certifier (40 coverage levels, union bound) needs 42 error-free calibration rows for 15%, 64 for 10% and 131 for 5%. A perfect sample of 28 cannot certify, whatever the model did. If the 2.9% text rate held exactly, about 30 rows (plain bound) or 69 rows (grid certifier) would be the minimum for 15%, 61 or 140 for 10%, and 5% needs more than 330 even with the plain bound; but those are minimums, not achievable sizes. With binomial sampling at 2.9% the grid certifier reaches 80% power only at about 92 calibration rows for 15% and about 202 for 10% (pools of about 460 and 1,000 prompts under one calibration fold), and even that takes the 2.9% point estimate, whose upper bound is 6.4%. 15% is plausible with a larger set, 10% needs a much larger one, and 5% is not supported by these data.

## Setup

- **Pool and split.** The 140 labelled synthetic prompts (ids 0-79 and 200-259), 84 vendors, five vendor-grouped folds (asserted: no vendor in two folds). One teacher, synthetic prompts from the generator in `gen.py`. The floor stage adds floor ids 260-299, 180 prompts in all.
- **Equivalence definition** (`equiv.py`, evidence policy `core`, decided before the run, `RUN_B2_PLAN.md`). A pruned or second note is compared with the reference full-prompt note A0. (1) DIFFERENT, without a judge, if the canonicalizer finds a conflict in a core fact or the reference states a required field that the candidate does not. (2) SAME, without a judge, if the core canonical facts are identical. (3) Otherwise the residual judge reads the pair in both presentation orders with the core rubric; equal answers decide, any disagreement or unparsed answer goes to a second-family judge as adjudicator; an unparsed adjudicator answer is FAIL. FAIL is excluded from every rate and counted next to it (it was 0 everywhere in this run).
- **Evidence policy `core`.** History evidence (counts, ranges, earlier-charge dates, whether history is mentioned) is not a core fact. This is a design choice, not a measurement, and it is exactly the thing pruning removes; the strict rubric column is kept so the other reading stays visible. Under `loose` the dry estimate was above the cap, so it was not run.
- **Judge fix.** Judge outputs now allow 200 tokens (S16 used 40/30 and 3 of its 16 floor "DIFFERENT" verdicts were cut-off answers); the verdict is the first SAME/DIFFERENT word; old cached verdicts are reused only if they parsed. Canonicalizer fixes after review: bank posting dates are not transaction-date candidates, a negative or bank-adjacent amount counts as stated when it equals the reference value, an invoice amount is ignored when the outcome is "no invoice" or the ids differ.
- **Three-way rotation cross-fitting.** Rotation f: test fold f, calibration fold f+1, span scorer trained on the other three folds. The cutoff applied to a fold never depends on that fold's labels, and the scorer never saw labels of the rows it calibrates on. Reported DIFFERENT rates use the test-role rows only. Cost of this design: one calibration fold = 140/5 = about 28 prompts per rotation, low-confidence prompts fall back to the full prompt, and the learned cells need up to two keep-sets per prompt.
- **Certifier.** `common.certify`: 40 coverage levels, each with delta/40 (union bound), one-sided Clopper-Pearson, largest coverage whose bound is at or under the budget. Budgets 5, 10, 15%, and an "excess over floor" variant that counts a pruned DIFFERENT as an error only when the floor pair of that prompt is SAME.
- **Typed arm.** The teacher returns a JSON record (flag, reason categories, invoice outcome/id/amount, transaction amount/date, bank mismatch yes/no); the note is rendered from a fixed template; equivalence is field equality after normalisation (reasons mapped to five categories, amounts absolute). Typed token savings use the typed full-prompt token count as the denominator. Typed floor = two independent typed full-prompt records.
- **Cells.** Learned scorer at m = 0, 5, 2; rule (transaction + invoice + flag only); cross-encoder, oracle, random at m = 5 and 2; each as free text and as typed. Fixed fields (transaction, invoice, flag, bank) are always kept except in the rule cell; a scorer ranks history records and the top m are kept (about 22.2 history records per prompt on average).
- **Cost.** $1.628 of the $3.00 cap; preflight estimated $2.33. Observed prices per call: teacher $0.00014, primary judge $0.00130, second-family judge $0.00028. Detail in Time and cost.

## Results

### 1. The floor: new equivalence against the strict rubric, same pairs


| pairs | n | new equivalence DIFFERENT (CP UB) | strict raw (CP UB) | strict confirmed (CP UB) |
|---|---|---|---|---|
| pool 140 (labelled) | 140 | 3/140 = 2.1% (5.4%) | 17/140 = 12.1% (17.7%) | 10/140 = 7.1% (11.8%) |
| original floor ids 200-299 | 100 | 1/100 = 1.0% (4.7%) | 13/100 = 13.0% (19.9%) | 11/100 = 11.0% (17.6%) |
| all floor prompts | 180 | 4/180 = 2.2% (5.0%) | 21/180 = 11.7% (16.4%) | 14/180 = 7.8% (11.9%) |

By scenario (the 140 labelled pairs; the log's by-scenario lines are over the labelled pool, not all 180; new equivalence; parsed from log_b2_live.txt):

| scenario | n | DIFFERENT | CP UB |
|---|---|---|---|
| dup | 20 | 0/20 = 0.0% | 13.9% |
| first | 28 | 0/28 = 0.0% | 10.1% |
| noinv | 30 | 0/30 = 0.0% | 9.5% |
| normal | 45 | 2/45 = 4.4% | 13.3% |
| spike | 17 | 1/17 = 5.9% | 25.0% |

Typed floor (two independent typed records, field equality): 5/140 = 3.6% (CP UB 7.4%); differing fields {'bank_mismatch': 1, 'reasons': 4}.
Typed record vs prompt ground truth (copy fidelity, n=140): flag 140, reasons 131, invoice_outcome 140, invoice_id 140, invoice_amount 140, tx_amount 140, tx_date 140, bank_mismatch 139

Where the 180 floor pairs went: 150 identical canonical facts (SAME with no judge), 4 byte-identical, 26 read by the judge; no pair had a canonical conflict or a lost required field. The 4 DIFFERENT pairs are therefore judge verdicts on the 26 residual pairs. By scenario no cell is above 6%, but every upper bound is wide (n = 17-45) and the 3 DIFFERENT floor pairs on the labelled rows sit in normal (2) and spike (1). The S16 per-scenario floor (noinv 19%, spike 35%, dup 17%) came from the earlier 100-prompt floor set under the strict rubric, so comparing it with these rates changes the rubric and the prompt set at once. Strict rubric on the same 140 labelled pairs, raw / confirmed: dup 1/20 / 1/20, first 1/28 / 1/28, noinv 7/30 (23.3%) / 2/30 (6.7%), normal 3/45 / 2/45, spike 5/17 (29.4%) / 4/17 (23.5%); the same pairs on the 180 are in `tables_b2.md`. So on identical pairs the strict rubric still has noinv and spike high and the new equivalence does not; the fall is mostly the rubric, not the prompt set.

What the strict column still shows: 11.7% raw (21/180, upper bound 16.4%) and 7.8% confirmed (14/180, 11.9%) on exactly the same pairs. The strict rubric counts optional history detail: its raw floor (11.7%) is above 10%, so neither 5% nor 10% can be certified at any n under it; the confirmed floor (7.8%) is under 10% but its bound at n=180 is 11.9%.

### 2. Span labels, relabelled with the new equivalence


| ablated span | new DIFFERENT | old strict | pairs |
|---|---|---|---|
| tx | 100.0% (140/140) | 62.9% | 140 |
| inv | 69.3% (97/140) | 77.9% | 140 |
| flag | 70.0% (98/140) | 95.0% | 140 |
| bank | 6.4% (9/140) | 17.9% | 140 |
| whole history (H0) | 11.4% (16/140) | 40.7% (57/140, from the log) | 140 |

Relabel paths over 748 pairs: {'canon-conflict': 272, 'canon-identical': 309, 'judge-onesided': 96, 'canon-lost:tx_date': 37, 'canon-lost:inv': 25, 'canon-lost:reason': 2, 'canon-lost:tx_amt,tx_date': 5, 'canon-lost:flag,inv,reason': 2}; FAIL 0.

(old strict values are the S16 labels in `labels.json`; the H0 old value is the strict-confirmed count the relabel stage printed on the same pairs.) The labels moved a lot: history is needed on 16 of 140 prompts (was 57), a single history record is positive for 10.5% of 3,102 labelled records (was 27.6%), the flag span matters on 70% of prompts (was 95%), the bank-feed span on 6% (was 18%). The transaction label is 100% because 139 of its 140 pairs turn on the transaction date: 35 are canon-lost (the ablated note does not state it), 104 are canon-conflicts on `tx_date` (97 on tx_date alone, 7 with tx_amt), and of those 104, 78 are the extractor taking the invoice issue date as the transaction date (`canon2` sets `tx_date` without excluding `inv_date`) and 26 are bank or history dates the posted/feed regex missed. The remaining conflict is reason plus tx_amt. So this label measures the extractor and the date requirement, not real dependence on the transaction span; ignoring the date would leave at most about 13 of 140 (about 9%, a rough estimate that was not computed). The pruned cells always keep the tx span, so the headline rates are not affected. Because only 3 prompts have both positive and negative history records, the scorer's within-prompt AUC on the new labels (learned 0.45, cross-encoder 0.50, recency 0.31, random 0.53) is meaningless; the learned scorer's advantage over random in the cells below is therefore not backed by a label-level AUC.

### 3. Every pruned cell

DIFFERENT is the share of the 140 held-out rows whose pruned note is not equivalent to A0; "floor, same rows" is the A0-vs-A3 rate on those same rows (3/140 text, 5/140 typed); discordant = rows where only the pruned note differs / only the floor pair differs, exact two-sided binomial p. Tokens removed is the share of the arm's own full-prompt input tokens (different denominators for text and typed). Vendor-bootstrap is the 95% range resampling vendors.

**Free text**


| cell | tokens removed | DIFFERENT (k/n) | CP UB | vendor-bootstrap | floor, same rows | excess | discordant pruned-only/floor-only | exact p |
|---|---|---|---|---|---|---|---|---|
| learned m=0 | 77.8% | 16/140 = 11.4% | 16.8% | 6.5%-16.7% | 2.1% | +9.3 pt | 15/2 | 0.002 |
| learned m=5 | 59.4% | 4/140 = 2.9% | 6.4% | 0.6%-5.8% | 2.1% | +0.7 pt | 3/2 | 1.000 |
| learned m=2 | 69.7% | 4/140 = 2.9% | 6.4% | 0.6%-6.2% | 2.1% | +0.7 pt | 3/2 | 1.000 |
| rule (tx+inv+flag) | 82.7% | 9/140 = 6.4% | 10.9% | 2.5%-11.2% | 2.1% | +4.3 pt | 8/2 | 0.109 |
| cross-encoder m=5 | 59.3% | 5/140 = 3.6% | 7.4% | 0.7%-7.2% | 2.1% | +1.4 pt | 3/1 | 0.625 |
| cross-encoder m=2 | 69.6% | 4/140 = 2.9% | 6.4% | 0.6%-6.1% | 2.1% | +0.7 pt | 3/2 | 1.000 |
| oracle m=5 | 59.2% | 5/140 = 3.6% | 7.4% | 0.7%-7.2% | 2.1% | +1.4 pt | 3/1 | 0.625 |
| random m=5 | 59.3% | 10/140 = 7.1% | 11.8% | 2.7%-12.0% | 2.1% | +5.0 pt | 10/3 | 0.092 |
| oracle m=2 | 69.7% | 4/140 = 2.9% | 6.4% | 0.6%-6.0% | 2.1% | +0.7 pt | 3/2 | 1.000 |
| random m=2 | 69.6% | 13/140 = 9.3% | 14.4% | 4.5%-15.3% | 2.1% | +7.1 pt | 12/2 | 0.013 |

**Typed**


| cell | tokens removed | DIFFERENT (k/n) | CP UB | vendor-bootstrap | floor, same rows | excess | discordant pruned-only/floor-only | exact p |
|---|---|---|---|---|---|---|---|---|
| learned m=0 | 67.2% | 12/140 = 8.6% | 13.5% | 4.0%-13.4% | 3.6% | +5.0 pt | 9/2 | 0.065 |
| learned m=5 | 51.4% | 15/140 = 10.7% | 16.0% | 5.9%-15.9% | 3.6% | +7.1 pt | 12/2 | 0.013 |
| learned m=2 | 60.3% | 16/140 = 11.4% | 16.8% | 5.7%-17.5% | 3.6% | +7.9 pt | 13/2 | 0.007 |
| rule (tx+inv+flag) | 71.5% | 29/140 = 20.7% | 27.2% | 13.8%-27.7% | 3.6% | +17.1 pt | 26/2 | 0.000 |
| cross-encoder m=5 | 51.2% | 12/140 = 8.6% | 13.5% | 3.8%-13.8% | 3.6% | +5.0 pt | 9/2 | 0.065 |
| cross-encoder m=2 | 60.2% | 15/140 = 10.7% | 16.0% | 5.7%-15.9% | 3.6% | +7.1 pt | 12/2 | 0.013 |
| oracle m=5 | 51.2% | 12/140 = 8.6% | 13.5% | 3.6%-13.9% | 3.6% | +5.0 pt | 10/3 | 0.092 |
| random m=5 | 51.2% | 15/140 = 10.7% | 16.0% | 5.1%-17.6% | 3.6% | +7.1 pt | 13/3 | 0.021 |
| oracle m=2 | 60.2% | 15/140 = 10.7% | 16.0% | 5.6%-16.4% | 3.6% | +7.1 pt | 13/3 | 0.021 |
| random m=2 | 60.2% | 18/140 = 12.9% | 18.5% | 7.2%-18.7% | 3.6% | +9.3 pt | 15/2 | 0.002 |

Strict rubric on the same rows (secondary, first three cells):

| cell | strict raw | strict confirmed |
|---|---|---|
| learned m=0 | 56/140 = 40.0% (UB 47.3%) | 52/140 = 37.1% (UB 44.4%) |
| learned m=5 | 34/140 = 24.3% (UB 31.0%) | 31/140 = 22.1% (UB 28.7%) |
| learned m=2 | 42/140 = 30.0% (UB 37.0%) | 40/140 = 28.6% (UB 35.5%) |

By scenario (DIFFERENT / rows; selected cells, all in `tables_b2.md`):

Text DIFFERENT by scenario (n: dup 20, first 28, noinv 30, normal 45, spike 17):

| cell | dup | first | noinv | normal | spike |
|---|---|---|---|---|---|
| text: learned m=0 | 3/20 | 3/28 | 2/30 | 5/45 | 3/17 |
| text: learned m=5 | 1/20 | 0/28 | 0/30 | 1/45 | 2/17 |
| text: learned m=2 | 1/20 | 0/28 | 0/30 | 2/45 | 1/17 |
| text: rule (tx+inv+flag) | 0/20 | 1/28 | 1/30 | 5/45 | 2/17 |
| text: oracle m=2 | 0/20 | 0/28 | 0/30 | 2/45 | 2/17 |
| text: random m=2 | 8/20 | 0/28 | 0/30 | 2/45 | 3/17 |
| typed: learned m=0 | 0/20 | 4/28 | 6/30 | 0/45 | 2/17 |
| typed: learned m=5 | 0/20 | 4/28 | 9/30 | 0/45 | 2/17 |
| typed: learned m=2 | 0/20 | 4/28 | 10/30 | 0/45 | 2/17 |
| typed: rule (tx+inv+flag) | 3/20 | 6/28 | 9/30 | 6/45 | 5/17 |
| typed: oracle m=2 | 0/20 | 3/28 | 10/30 | 0/45 | 2/17 |
| typed: random m=2 | 0/20 | 4/28 | 12/30 | 0/45 | 2/17 |

Reading the cells. Free text at m=5 and m=2 with a sensible selector (learned, cross-encoder, oracle) is 2.9-3.6%, a 0.7-1.4 point excess over the 2.1% floor; random selection at m=2 is 9.3% and the damage is concentrated in the `dup` scenario (8 of 20), where the earlier duplicate charge is in the history. Learned m=0 gives 16/140, the same count as the whole-history label H0 (16/140), as it should. In the typed arm the excess concentrates in `noinv` (6-12 of 30) and `first` (3-6 of 28); the typed floor by scenario was not stored, so I cannot say how much of that is floor, and that the reason category depends on history in those scenarios is a hypothesis, not something this run tested.

### 4. Certificates

Rotation cross-fitting, 140 usable rows per cell (0 excluded FAIL/unknown), 5 folds, budgets 5/10/15%, plain and excess over floor. Covered = held-out rows above their fold's certified cutoff. The last two columns run the same grid certifier once on all 140 rows, with no calibration/test split: that is in-sample, not a certificate, and is shown only to show what the observed rates would allow.


Held-out covered rows per budget; every one is 0. 'In-sample pooled' = the same grid certifier run once on all 140 rows (no calibration/test split, NOT a valid certificate, shown only to show what the observed rates would allow).

| arm | cell | plain 5/10/15% covered | excess 5/10/15% covered | in-sample pooled plain 10% (coverage, err, bound) | in-sample pooled plain 15% |
|---|---|---|---|---|---|
| text | learned m=0 | 0/0/0 | 0/0/0 | none | none |
| text | cross-encoder m=0 | 0/0/0 | 0/0/0 | none | none |
| text | oracle m=0 | 0/0/0 | 0/0/0 | n/m (label-leaked) | n/m (label-leaked) |
| text | random m=0 | 0/0/0 | 0/0/0 | none | none |
| text | learned m=5 | 0/0/0 | 0/0/0 | 100%, 2.9%, 10.0% | 100%, 2.9%, 10.0% |
| text | learned m=2 | 0/0/0 | 0/0/0 | 100%, 2.9%, 10.0% | 100%, 2.9%, 10.0% |
| text | rule (tx+inv+flag) | 0/0/0 | 0/0/0 | none | none |
| text | cross-encoder m=5 | 0/0/0 | 0/0/0 | 93%, 2.3%, 9.5% | 100%, 3.6%, 11.1% |
| text | cross-encoder m=2 | 0/0/0 | 0/0/0 | 100%, 2.9%, 10.0% | 100%, 2.9%, 10.0% |
| text | oracle m=5 | 0/0/0 | 0/0/0 | 93%, 2.3%, 9.5% | 100%, 3.6%, 11.1% |
| text | random m=5 | 0/0/0 | 0/0/0 | none | none |
| text | oracle m=2 | 0/0/0 | 0/0/0 | 100%, 2.9%, 10.0% | 100%, 2.9%, 10.0% |
| text | random m=2 | 0/0/0 | 0/0/0 | none | none |
| typed | learned m=0 | 0/0/0 | 0/0/0 | none | none |
| typed | learned m=5 | 0/0/0 | 0/0/0 | none | none |
| typed | learned m=2 | 0/0/0 | 0/0/0 | none | none |
| typed | rule (tx+inv+flag) | 0/0/0 | 0/0/0 | none | none |
| typed | cross-encoder m=5 | 0/0/0 | 0/0/0 | none | none |
| typed | cross-encoder m=2 | 0/0/0 | 0/0/0 | none | none |
| typed | oracle m=5 | 0/0/0 | 0/0/0 | none | none |
| typed | random m=5 | 0/0/0 | 0/0/0 | none | none |
| typed | oracle m=2 | 0/0/0 | 0/0/0 | none | none |
| typed | random m=2 | 0/0/0 | 0/0/0 | none | none |

Folds with a cutoff: 0 of 5 in every cell. Calibration rows per rotation: 28.
- Zero errors, grid certifier, delta=0.05/40: smallest n with CP UB <= budget: 5%: 131, 10%: 64, 15%: 42
- Zero errors, plain one-sided 95%: smallest n with CP UB <= budget: 5%: 59, 10%: 29, 15%: 19

Result: **0 of 5 folds had a cutoff in every cell, so tokens saved by certified pruning is 0.0% everywhere** (uncertified prompts fall back to the full prompt). Why:

- Calibration fold size. Each rotation calibrates on exactly 28 prompts (checked: 28/28/28/28/28). With delta/40 the smallest error-free calibration set that gives a bound at or under the budget is 131 rows for 5%, 64 for 10% and 42 for 15% (plain one-sided 95%: 59, 29, 19). 28 rows cannot pass even with zero errors, so the table is zero before the model's behaviour matters: it is a foregone conclusion of the design, carries no information about pruning quality and should not be counted as a finding. The `min 10 calibration rows` guard in `run_b2.py` is not the binding constraint.
- Rotation. The cutoff for fold f is fitted on a different fold and scorer than the one it is applied to; this costs rows but is the part that makes the number honest.
- Grid certifier. The union bound over 40 levels is conservative at small n: at 2.9% observed error, 140 rows give an in-sample bound of 10.0% (so even the in-sample version only just clears 10%, and only for text with history kept), while the plain 95% bound would be 6.4%.
- The oracle m=0 in-sample row (89% coverage, 0.0% error) is not meaningful: the m=0 prompt is the same for every scorer, only the confidence differs, and the oracle's confidence comes from the labels. It is marked n/m above.
- The in-sample column also shows what does not certify even in-sample: all typed cells, the rule cell, random cells and free text m=0.

### 5. What budget and calibration size would certify

Smallest n such that the one-sided Clopper-Pearson upper bound at the observed rate (errors = round(rate x n)) is at or under the budget, and stays so for the next 40 sizes. "Plain" is delta = 0.05; "grid certifier" is delta = 0.05/40, the cost of searching coverage levels.


| observed DIFFERENT rate | source | 5%, plain | 10%, plain | 15%, plain | 10%, grid certifier | 15%, grid certifier |
|---|---|---|---|---|---|---|
| 2.2% | new-equivalence floor, 180 pairs | 208 | 46 | 30 | 123 | 69 |
| 2.9% | text learned m=5 / m=2 (4/140) | 336 | 61 | 30 | 140 | 69 |
| 3.6% | text cross-encoder/oracle m=5 (5/140) | 717 | 76 | 30 | 187 | 81 |
| 6.4% | text rule (9/140) | never (rate >= budget) | 227 | 59 | 632 | 152 |
| 9.3% | text random m=2 (13/140) | never (rate >= budget) | 4971 | 126 | 16085 | 352 |
| 11.4% | text learned m=0 (16/140) | never (rate >= budget) | never (rate >= budget) | 304 | never (rate >= budget) | 907 |
| 3.6% | typed floor (5/140) | 717 | 76 | 30 | 187 | 81 |
| 8.6% | typed cross-encoder/oracle m=5 (12/140) | never (rate >= budget) | 1280 | 102 | 4009 | 276 |

With one calibration fold per rotation (a fifth of the pool), the pool size is five times the 'grid certifier' columns. These assume errors = round(rate x n); they are the minimum, not a power calculation, and treat rows as independent (rows share vendors and prompts).

With one calibration fold per rotation, the pool needed is five times the calibration size, so the grid-certifier 15% column (69 rows at 2.9%) needs a pool of about 350 prompts and the 10% column (140 rows) about 700. Alternatively `--cal-folds 3` (the maximum with five folds) gives 84 calibration rows from the same 140 prompts, which would clear the 15% threshold at about 3% observed error but leaves one fold to train the scorer and 28 rows to evaluate per rotation (all 140 rows are still evaluated across the rotations). These are minimums at a fixed rate: they treat rows as independent (rows share vendors and prompts), they assume the rate does not rise at lower coverage, and they say nothing about power.

## Transfer to Dreaming

- A certified error budget on free prose depends on the equivalence definition first and the model second. Under a facts-only definition the full-prompt floor is about 2%, so 15% is a plausible target (10% needs a much larger calibration set) for a free-text step with a pruner that keeps a few history records; under a definition that counts optional detail it is 8-12% (confirmed / raw), 5% is unreachable and 10% is marginal at best. The definition must be a product decision made and frozen before certification.
- For the pruner: keeping the transaction, invoice, flag and about 2-5 history records removed 59-70% of input tokens at 2.9-3.6% facts-level disagreement on this synthetic set; dropping history entirely costs 11%, and a rule-only pruner costs 6% (free text) or 21% (typed). A learned selector looks no better than the cross-encoder or the oracle selector here, and has fewer DIFFERENT than random at m=5 (4 vs 10) and m=2 (4 vs 13), mostly on duplicates; no paired test was run for that.
- Do not assume typing buys a lower floor. On this task the typed floor is 3.6%, above the free-text 2.1%, and typed pruned cells are further above it.
- Calibration set size is the practical gate: a few dozen error-free rows per fold at 15%, a few hundred at 5%, and more if the rate is not close to the floor. With cross-fitting the pool is about five times that. Reuse of the same 140 prompts for tuning and reporting is fine for these descriptive tables but not for a certificate.
- The extractor-first path (decide SAME/DIFFERENT from canonical facts, judge only residuals) removed the judge from 86% of floor pairs (154/180) and kept FAIL at 0; it is the part most likely to transfer.

## Caveats

- Synthetic prompts, one teacher, 140 prompts (180 for the floor). Scenario mix, history length and noise are generator choices.
- The equivalence decides the headline. Under `core` evidence, history mentions are not facts, which is the thing pruning removes; the 2.2% floor and the 2-4% pruned rates are conditional on that. The strict column is the other view and is 3.5-10.5 times higher (7.8/2.2 on the floor, 40.0/11.4 for m=0, 30.0/2.9 for m=2).
- The extractor is regex-based. Its accuracy was checked on the 140 reference notes (all core fields stated and correct, amount 139/140) and fixed after review, but "canon-identical" pairs are SAME by an extractor, not by a human. No human adjudication was done and no Rogan-Gladen correction was applied.
- The `tx` label of 100% reflects the transaction-date requirement plus an extractor weakness (139 of 140 pairs turn on `tx_date`; 78 of the 104 date conflicts are the invoice date read as the transaction date), not real dependence on the transaction span; the span labels, and hence the scorers, depend on required-field choices and extractor accuracy.
- Small counts: the pruned-vs-floor tests have 3-15 discordant pairs; a p of 1.0 with 3 vs 2 is no evidence of equality. Upper bounds are one-sided 95% Clopper-Pearson and ignore vendor clustering; the vendor-bootstrap ranges are the clustering-aware view and are wider on the upper side for some cells.
- No cell was selected after seeing results, but there are 20 cells and no multiplicity correction; nine of 20 cells have p < 0.05 against the floor, and most are the cells whose pruning removes information, so this is not a chance finding, but the individual p-values are not adjusted.
- Typed pruned cells' differing fields were not stored (only the floor's: 4 `reasons`, 1 `bank_mismatch`), so the typed breakdown is by scenario only.
- The judge provider pin was not audited in this report.

## Time and cost

557 s wall clock, 4,482 new calls, $1.628 of the $3.00 cap (preflight estimate $2.33; the gap was not decomposed). Cumulative spend at the end of each stage: relabel $0.170 (22 s); floor, typed floor and the learned m=0 cell $0.383 (673 calls, 225 s); learned m=5 $0.730; learned m=2 $1.082; rule $1.160; cross-encoder m=5 and 2 $1.237 and $1.297; oracle m=5 $1.380; random m=5 $1.476; oracle m=2 $1.540; random m=2 $1.614. Embedding 3,242 texts took 84 s on CPU. Observed prices per call: teacher $0.00014, primary judge $0.00130, second-family judge $0.00028.

## Revision

A review of this report changed the following; the body above has been edited to match, and `report_b2.py` now also prints the revision tables (power, fold sizes, `tx` anatomy, strict rubric by scenario; cache-only, no model call). No headline rate changed.

1. **The 100% transaction label was misexplained.** It is not "35 of 140 canon-lost". 139 of 140 `tx` pairs turn on `tx_date`: 35 canon-lost, 104 conflicting (97 on tx_date alone, 7 with tx_amt); of the 104, 78 are the extractor taking the invoice issue date as the transaction date and 26 are bank or history dates the posted/feed regex missed. The label measures the extractor and the date requirement. Pruned cells always keep the tx span, so the cell rates are unaffected.
2. **"No certificate" is entirely structural.** Every calibration fold has exactly 28 rows; the grid certifier needs at least 42 error-free rows for 15%, so no fold could certify any budget at or under 15% whatever the model did. The 0/138 table is a foregone conclusion and not a finding.
3. **Minimum sizes are not achievable sizes.** With binomial sampling at 2.9% (grid certifier, delta/40), the chance a calibration sample certifies:

   | calibration rows | budget | chance |
   |---|---|---|
   | 28 | 15% | 0.00 |
   | 69 | 15% | 0.68 |
   | 84 (`--cal-folds 3`) | 15% | 0.78 |
   | 84 (`--cal-folds 3`) | 10% | 0.09 |
   | 140 | 10% | 0.63 |

   For 80% power: about 92 rows at 15% and about 202 at 10% (113 and 259 at 3.6%), i.e. pools of about 460 and 1,000 prompts under one calibration fold (565 and 1,295 at 3.6%). These still assume the 2.9% point estimate, whose upper bound is 6.4%. "10% is plausible" was too optimistic and is now "needs a much larger set".
4. **Answer (2) is softened and the strict comparison added.** Free-text pruning at m=5 or m=2 is "not distinguishable from the floor" (upper bounds 6.4-7.4%, 3 vs 1-2 discordant pairs), not "at the floor". On the same 140 rows the strict floor is 12.1% raw / 7.1% confirmed, while learned m=5 is 24.3% / 22.1% and m=2 is 30.0% / 28.6%, two to four times the strict floor, because the core equivalence omits the history evidence that pruning removes.
5. **Oracle m=0 in-sample row is label-leaked** (the m=0 prompt is identical for all scorers; only the confidence differs, and the oracle's comes from labels). Marked not meaningful.
6. **Wording and one table label.** "12-16%" is 13-16%; strict versus core is 3.5-10.5 times; "rows of an error-free-looking sample" now says "errors = round(rate x n)"; "28 rows to evaluate" is per rotation. The by-scenario floor table was labelled "all 180 floor pairs" but the log prints it over the 140 labelled pairs (n sums to 140); relabelled. The S16 per-scenario floor came from a different prompt set and rubric, so strict-by-scenario is now shown on the same pairs: the strict rubric is still high for noinv (23.3% raw) and spike (29.4%) on the 140, so the fall is mostly the rubric, not the prompt set.

Checked and unchanged: all floor, cell, typed, strict, certificate, cost and time numbers against the log and `results_b2.json`; discordant counts and p-values; path counts (748 = 140x5 + 16x3; floor 150 + 4 + 26 = 180, 154/180 = 86% judge-free); `run_b2.py`, `equiv.py` and the caches untouched.
