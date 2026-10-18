# S16c - Does draft_note just need a judge? Offline re-analysis of the S16 floor

Cost $0. No network, no model call: every note and verdict below was re-derived from `s16_pruner_note/responses.jsonl` with S16's own `core.py` functions.

## In plain words

S16 found that two independent full-prompt notes from the same pinned teacher are called DIFFERENT by the strict judge 16% of the time (13% after the majority-of-3 confirmation). That makes a 5% budget look impossible. This smoke asks where those 16 disagreements come from, using only what is already cached.

Answer in four sentences. (1) A judge alone does not fix it: with the strict rubric the floor stays at 11-16% (upper bound 18-23%; 3 of the 16 raw verdicts are cut-off judge answers, see Revision), so neither a 5% nor a 10% budget is reachable however many rows are calibrated. (2) The disagreement is not about facts: a deterministic extractor finds **no** conflict in vendor, amount, invoice outcome or flag decision in any of 520 pairs, and no conflict in the flag reason except one note in 640; the 16 judged pairs differ in optional detail (does the note mention the vendor's history, a spending range, a count, the bank amount?) and in judge noise. (3) So the floor is set by the equivalence definition: canonical-fact conflicts alone are 0% (0/100, UB 3.0%); the residual that a core-rubric judge would have to read is 42 of 100 pairs and the cache measures it on only 6, so the floor with residual judging is unmeasured (0% to 36%). It is about 2-5% if history evidence must also agree, and 11-16% under the strict rubric. (4) The research note's recipe "call DIFFERENT without a judge when canonical facts conflict" never fires on this task (0 of 420 pairs after fixing a follow-up false positive); the canonicalizer's real use is to certify SAME and to shrink the set a judge must read, and the real decision is which details count. Corrections are in the Revision section at the end.

## Setup

- **Code:** `run.py` (analysis), `canon.py` (the canonicalizer, pure regex, no model), `cacheonly.py` (stand-in for `llm.py`: same sha1 cache key, raises on a miss, counts misses), `manual_classification.json` (my hand labels for 10 pairs, read by `run.py`). Outputs: `log.txt`, `results.json`, `tables.md` (all tables, incl. the 8+8 verbatim examples).
- **Recovered from the cache:** notes via `core.note(p, keep_all(p), rep)` for rep 0-2 (A0, A1, A2) on the 140 labelled prompts (ids 0-79, 200-259; identical to `labels.json` refs, asserted) and on 260-299 where S16's floor run had cached them (180 prompts), and rep 3 (the 4th note, A3) on the 100 floor prompts 200-299. Judge verdicts via `core.J` / `core.verdict`; the recomputed A0-A3 strict verdicts reproduce `results.json["floor"]` exactly (16/100 raw, 13/100 confirmed; an unparseable verdict, FAIL, counts as DIFFERENT as in S16).
- **Missing from the cache:** 0 note keys and 0 A0-A3 strict-judge keys. Strict-judge verdicts for A0-A1, A0-A2 and A1-A2 do **not** exist (S16 never judged two reference notes against each other): 404 of 420 are missing; of the rest, 9 are byte-identical note pairs (SAME without a call) and 7 non-identical pairs happen to be in S16's judge-validation sample. A1-A3 and A2-A3 strict verdicts exist for 17 and 6 pairs (S16's confirmation calls, only where A0-A3 was DIFFERENT). Core-rubric verdicts exist for 12 non-identical A0-A3 pairs (+3 byte-identical), order-swapped for 12, the second judge for 52. Consequently definitions (a), (b), (e) are measured on A0-A3 only; (c), (d), (f) on all four pair types.
- **Canonicalizer** (`canon.py`): per note it extracts vendor (fuzzy token match to the vendor string in `tx`), transaction amount, ISO dates, invoice id and invoice amount, bank-feed amount (negative-signed or after "bank/feed/posted"), flag decision (flagged / not flagged / unclear after removing negations), flag reason set (duplicate, first purchase, unusually high, no invoice, none, other), follow-up requested, history mentioned. After reading the judged pairs I added an evidence family: history count, history range, highest prior amount, other dates (earlier-charge date). These are used only in the (c+) rows, not in the originally specified (c)/(d).
- **Pair comparison:** a field conflicts only if both notes state it with different values. A field stated in one note only is "one-sided" (optional details) or "unstated" (required fields); these are not conflicts. A pair is "identical" if every extracted fact and the history/follow-up mention flags are equal, "residual" if there is no conflict but the facts are not identical.
- **Definitions:** (a) strict judge raw; (b) strict judge confirmed; (c) any conflict among vendor, tx amount, tx date, invoice id, invoice amount, bank amount, flag, reason, follow-up; (c') = (c) without follow-up; (c+) = (c) plus history/earlier-charge evidence; (c+tol) = (c+) with numbers within max(1, 5%) equal; (d) conflict on core fields only (vendor, amount, invoice id and amount, flag); (e) canonical conflict or the cached core-rubric judge; (f) exact match after lowercase / whitespace / number normalisation.
- **Bounds:** one-sided 95% Clopper-Pearson (`common.cp_upper`). "UB@n" = the bound if n rows showed the same rate. "n for b" = fewest rows whose bound at the observed rate is below budget b. The grid certifier (`common.certify`) is a selective-prediction tool and is not used for a single global rate; for a selective certificate with 40 grid levels and no errors the row count needed is 131 (5%) / 64 (10%) instead of 59 / 29.

## Results

### 1. Canonicalizer vs ground truth (640 full-prompt notes, 180 prompts)

| field | stated | unclear / not stated | accuracy among stated |
|---|---|---|---|
| vendor name (fuzzy vs tx string) | 97.8% | 2.2% | 100% |
| transaction amount | 99.7% | 0.3% | 100% |
| transaction date | 100% | 0% | 100% |
| invoice outcome (id or none) | 100% | 0% | 100% |
| invoice amount (439 notes with invoice) | 100% | 0% | 100% |
| invoice issue date (same 439) | 69.5% | 30.5% | 100% |
| bank amount, bank differs from tx (90 notes) | 73.3% | 26.7% | 100% |
| bank amount, bank equals tx (550 notes) | 10.9% | 89.1% | 100% |
| flag decision | 100% | **0%** | 100% |
| flag reason category (set) | 100% | **0%** | 99.8% (1 note) |
| follow-up requested | 0.0% of notes (0) after the Revision fix | n/a (no ground truth) | n/a |

Whole-note check: flag, reason, invoice outcome and tx amount all equal the record in 637/640 notes (99.5%). The three misses: two notes with a tx amount the extractor could not isolate from a signed bank amount, and one note that states "no matching invoice" as a fact but names only "unusually high" as the reason.

### 2. The floor under different equivalence definitions (DIFFERENT rate, 95% upper bound)

| definition | pair set | k/n | rate | UB95 | UB@300 | UB@1000 | n for 5% | n for 10% |
|---|---|---|---|---|---|---|---|---|
| (a) strict judge, raw | A0-A3 | 16/100 | 16.0% | 23.3% | 19.9% | 18.0% | never | never |
| (b) strict judge, confirmed | A0-A3 | 13/100 | 13.0% | 19.9% | 16.6% | 14.9% | never | never |
| (a)/(b) on A0-A1, A0-A2, A1-A2 | | not in cache | | | | | | |
| (c) any canonical conflict | A0-A1 / A0-A2 / A1-A2 / A0-A3 | 0/140, 0/140, 0/140, 0/100 | 0% | 2.1%, 2.1%, 2.1%, 3.0% | 1.0% | 0.3% | 59 | 29 |
| (c) pooled 3 pair types (not independent) | 420 pairs | 0/420 | 0% | 0.7% (cluster bootstrap 0.0%) | 1.0% | 0.3% | 59 | 29 |
| (c') without follow-up (identical to (c) after the fix) | all four types | 0/140, 0/140, 0/140, 0/100 | 0% | 2.1-3.0% | 1.0% | 0.3% | 59 | 29 |
| (d) core fields only | all four types | 0 conflicts in 520 pairs | 0% | 2.1-3.0% (pooled 0.7%) | 1.0% | 0.3% | 59 | 29 |
| (c+) incl. history evidence, exact | A0-A1 / A0-A2 / A1-A2 / A0-A3 | 6/140, 3/140, 6/140, 5/100 | 4.3%, 2.1%, 4.3%, 5.0% | 8.3%, 5.4%, 8.3%, 10.2% | 7.6% (A0-A3) | 6.3% | 2461 / 153 / 2461 / never | 76 / 46 / 76 / 89 |
| (c+tol) numbers within 5% | A0-A1 / A0-A2 / A1-A2 / A0-A3 | 3/140, 1/140, 2/140, 2/100 | 2.1%, 0.7%, 1.4%, 2.0% | 5.4%, 3.3%, 4.4%, 6.2% | 3.9% (A0-A3) | 2.9% | 153 / 59 / 93 / 124 | 46 / 29 / 29 / 46 |
| (e) canonical or core-rubric judge, pairs with a cached core verdict only (15 pairs = 6 residual + 9 SAME by construction; residual only: 0/6, UB 39.3%) | A0-A3 subset | 0/15 | 0% | 18.1% | | | | |
| (e) range over all 100 pairs | A0-A3 | | 0% (residual all SAME) to 36% (residual all DIFFERENT) | | | | | |
| (f) exact match after normalisation | A0-A1 / A0-A2 / A1-A2 / A0-A3 | 139, 138, 133, 95 of 140, 140, 140, 100 | 99.3%, 98.6%, 95.0%, 95.0% | 97.6-100% | | | never | never |

Notes on (e): 15 A0-A3 pairs have a cached core-rubric verdict, but 9 of them are SAME by construction (3 byte-identical, 6 identical canonical facts), so only 6 are real residual pairs (0 DIFFERENT, UB 39.3%). Under (c) 42 of 100 pairs are residual (no conflict, facts not identical); 36 have no core verdict, so (e) is a range 0% to 36%, not a measurement. The strict judge calls 13 of the 42 residual pairs DIFFERENT (11 real, 2 cut-off) and 3 of the 55 non-byte-identical pairs with field-for-field identical facts DIFFERENT.

The first version reported 4 (c)-conflicts (prompts 32 and 57, note A0). They were extractor false positives: "a difference ... that the reviewer should note" is an observation, not a follow-up request. After the fix (c) is 0 of 520. Those pairs differ in whether the real bank-feed mismatch is mentioned (a one-sided omission, see Revision).

### 3. Judge verdict vs canonical facts (A0-A3, n = 100)

| strict raw verdict \ canonical | conflict (c) | residual | identical | total |
|---|---|---|---|---|
| DIFFERENT | 0 | 13 | 3 | 16 |
| SAME | 0 | 29 | 55 | 84 |

With (c+) (history evidence counted as a fact): judge DIFFERENT = 3 conflict / 10 residual / 3 identical; judge SAME = 2 conflict / 27 residual / 55 identical. Including A1-A3 and A2-A3 pairs that have a cached verdict (a biased subset): 29 judge-DIFFERENT pairs, none with a (c) conflict, 4 with field-for-field identical facts.

My reading of the 16 judge-DIFFERENT A0-A3 pairs: 10 differ in history detail (the history is mentioned or summarised in one note and not the other: 5 pairs; a count or range differs, e.g. 27 vs 26 prior transactions, a lower bound of $50 vs $49: 5 pairs), 2 differ in whether the bank amount is stated (one of them a real $7.26 bank-feed mismatch that one note omits), 1 is an unparseable judge answer counted as DIFFERENT (its confirmation says SAME), and 3 show no visible difference in facts (judge strictness or noise). Zero are fact conflicts.

**Hand inspection of 10 judge-SAME pairs with a canonical difference.** There are no judge-SAME pairs with a (c) conflict (0 of 100 on A0-A3, 0 of 120 incl. extras), so I inspected the 10 best candidates (history-evidence conflicts first, then bank amount stated once). Result: 6 are an optional detail (bank amount, equal to the transaction) omitted in one note, judge right; 2 are rounding of the same history range ($99-$249 vs $100-$250, $400-$900 vs $400-$925), judge right and extractor over-strict; 1 is a loose characterisation ("typically around $100-$200" vs "$82 to $210"), borderline; **1 is a judge miss on a real error** (prompt 288: one note says typical charges "roughly $150-$480" while the record's lowest prior charge is $167.71). No extractor error produced a false conflict on a fact; all false positives came from my new strict evidence fields, which is why (c+tol) exists. Details in `tables.md`.

### 4. Where the differences come from

By scenario (A0-A3 judge, n = 100; canonical rates over the 4 pair types):

| scenario | prompts floor | judge raw DIFFERENT | judge confirmed | canonical (c) | facts not identical |
|---|---|---|---|---|---|
| normal | 34 | 2 (6%) | 2 | 0% | 17% |
| noinv | 21 | 4 (19%) | 2 | 0% | 63% |
| dup | 12 | 2 (17%) | 2 | 0% | 53% |
| first | 16 | 2 (13%) | 2 | 0% | 26% |
| spike | 17 | 6 (35%) | 5 | 0% | 63% |

By history mention (A0-A3): when exactly one of the two notes mentions the history, the strict judge says DIFFERENT in **7 of 11** pairs; when both mention it 6 of 29; when neither 3 of 60. Seven of the 16 judged differences are therefore a history-mention asymmetry by the extractor's mention flag (my own reading of the 16, above, counts 5 mention asymmetries and 5 history number or wording differences). By canonical field over 520 pairs: conflicts are only follow-up (4); one-sided items are the bank amount (12.3% of pairs), vendor name not recognisable in one note (1.3%), tx amount not isolable (0.6%), invoice amount (0.2%); the history is mentioned in exactly one of the two notes in 7.9% of pairs.

Clustering: the optional details are prompt-driven, not random per note. All three notes of a prompt agree on whether to mention history in 89% of prompts (73% expected if each note chose independently at its scenario's rate), on whether to state the bank amount in 81% (53% expected), on the earlier-charge date 81% (77%). History-evidence conflicts (c+) fall on 9 of 140 prompts (5 noinv, 3 spike, 1 dup), 5 of them in two or three pairs. For the judge, the number of non-identical-facts pairs among a prompt's reference pairs predicts its A0-A3 judge verdict with AUC 0.88 (n = 60 prompts that have reference pairs, 10 positives; 0.94 with the 3 cut-off prompts dropped); the (c+) count gives 0.70. This replaces 0.71, which wrongly scored prompts 260-299 (no reference pairs) as 0.

Prompt-level predictors (scenario, spike, has_inv, bank_mismatch, number of history records, flagged; logistic regression, stratified 5-fold by prompt, 20 repeats; CI = bootstrap over prompts of the out-of-fold score):

| target | prompts | positives | AUC |
|---|---|---|---|
| strict judge raw DIFFERENT (A0-A3) | 100 | 16 | 0.69 (0.57-0.82) |
| strict judge confirmed DIFFERENT | 100 | 13 | 0.63 (0.51-0.79) |
| any non-identical facts in 3 pairs | 140 | 73 | 0.71 (0.61-0.80) |
| any (c+) conflict in 3 pairs | 140 | 9 | 0.74 (0.62-0.83) |
| any (c) / core conflict | 140 | 0 / 0 | too few positives |

Prompt features predict "this prompt's notes will disagree" weakly for the judge (AUC 0.6-0.7, the interval touches 0.5 for the confirmed verdict) and moderately for non-identical facts. The strongest single signals are scenario = spike and the flag being on; the pairs that are stable are normal prompts with no history spans.

### 5. What each route reaches and what it costs to certify

| route | floor (A0-A3 unless noted) | UB95 | rows to certify at 5% / 10% (rate-matched) |
|---|---|---|---|
| (i) judge only, strict rubric | 16% raw, 13% confirmed | 23.3%, 19.9% | never / never |
| (ii) canonical + residual judge with the core rubric | 0/100 canonical-only (the lower end; no residual judging); 2% (c+tol), 5% (c+); with residual judging unmeasured, range 0% to 36% (6 of 42 residual pairs have a core verdict, 0 DIFFERENT, UB 39%) | 3.0%, 6.2%, 10.2% (upper end of the range: not measured) | 59 / 29 (c); 124 / 46 (c+tol); never / 89 (c+) |
| (iii) typed fields (flag, reason, invoice outcome, tx amount) | 0/100 (0/140 on each reference pair; 1/100 if an unstated required field counts) | 3.0% (4.7%) | 59 (93) / 29 |
| (f) exact match (reference) | 95-99% DIFFERENT | | never |

Estimate for (iii): the canonical fields are the typed output, and their teacher self-agreement is 100% (0 conflicts in 520 pairs on four typed fields, 0 in 420 on the labelled prompts) and 99.5% against the ground truth. A typed step would also fix the bank-amount and history mentions by template, which removes the largest source of residual differences; but that floor is measured on free text from which the fields were extracted, not on a typed call.

Rows needed. The floor only has to be certified once and is part of the budget the pruned step must stay under: with a floor of 2% the pruning headroom at a 5% budget is 3 points, at 10% it is 8. A 5% budget is therefore plausible only under (ii)-core or (iii); under (i) neither budget is reachable. For a single global rate and no errors, 59 rows suffice for 5% and 29 for 10%; for a selective (grid) certificate 131 / 64. The 140 labelled prompts are enough for 10% under (ii)/(iii); 5% needs near-full coverage and about 100 zero-error pruned rows.

### 6. What the new definition does to S16's ablation labels (free, from the cached variant notes)

| ablation | strict confirmed DIFFERENT | canonical core change | evidence change |
|---|---|---|---|
| drop tx line (n=140) | 88 (63%) | 140 (100%) | 36 |
| drop invoice line | 109 (78%) | 97 (69%) | 45 |
| drop flag line | 133 (95%) | 95 (68%) | 47 |
| drop bank line | 25 (18%) | 1 (0.7%) | 41 |
| drop whole history (H0) | 57 (41%) | 9 (6.4%) | 46 |
| history block 0 / 1 / 2 (57 prompts each) | 31 / 15 / 19 | 1 / 0 / 0 | 38 / 30 / 37 |

Dropping the whole history changes the strict verdict on 57 of 140 prompts and the canonical core facts on 9. By scenario (strict / canonical / prompts): normal 3/2/45, noinv 23/2/30, dup 20/0/20, first 3/4/28, spike 8/1/17. History matters for the strict rubric mainly because it supplies evidence (which earlier charge a duplicate matches, how the amount compares with the usual range, how many prior transactions lack an invoice); under a core-facts definition it is almost never needed. This is the most consequential effect of the definition: the pruning headroom under the new definition is far larger than S16's labels suggest. Caveat: "drop tx = 100%" is partly mechanical (without the transaction line the note's date is the bank date); the tx and flag rows should be read as sensitivity checks, not labels.

## Caveats

- **Synthetic prompts, one teacher, one prompt template.** The note restates a record that already contains the flag decision, its reason, the invoice and the amount; so near-zero fact conflicts are expected here and say little about a draft_note whose content requires a decision. The floor under (c)/(d)/(iii) is a property of this task being nearly a copy task.
- **The canonicalizer was developed on the same 640 notes it is validated on.** The 100% field accuracies are in-sample and optimistic; the regexes were tuned to this teacher's phrasing ("flagged", "no matching invoice", "-$x"). It must be validated on fresh notes (S16b generates them) before it is trusted. Vendor conflicts cannot be detected at all (an unrecognised name is "unstated"). Follow-up detection has 2 positives and no ground truth.
- **Evidence fields (c+) were added after reading the judged pairs**, so (c+) is partly fitted to explain the judge's DIFFERENT calls. (c+tol) uses a 5% tolerance chosen by me.
- **Judge-based rows rest on one pair type** (A0-A3, n = 100, 16 positives). The strict judge was never run on reference-vs-reference pairs; A1-A3 / A2-A3 are a biased subset. Core-rubric coverage is 12 pairs. The cross-tab, the history-asymmetry result (7/11 vs 9/89) and the prompt AUCs rest on 100 prompts and wide intervals.
- **Pooled pair rates are not independent** (pairs share notes and prompts); only per-pair-type rows carry a valid Clopper-Pearson bound, pooled rows show a prompt-cluster bootstrap.
- **My hand classifications are single-rater**, written by the builder.
- The "hard 16%" figure itself comes from an LLM judge that is not a ground truth; its self-agreement was 99% but it flips 13% on order swap (S16).
- No new model call was made, so nothing here tests whether a core-rubric judge would agree with a human on the residual.

## Verdict

**Does draft_note just need a judge? No.** A judge changes what is counted, not how often two samples disagree: with the strict rubric the floor is 11-16% (UB 18-23%) and a 5% or 10% budget is unreachable at any calibration size. **But the floor is mostly the definition, not the model.** On this task the two notes never disagree about the transaction, invoice outcome or flag, and disagree on the flag reason in 1 of 640 notes; (vendor conflicts are not measured, see Revision) the strict judge's 16 raw DIFFERENT calls are 3 cut-off judge answers (207, 295, 296), 9 history or earlier-charge detail, 2 bank-amount mention and 2 with nothing visible (see Revision). The research note's mechanism (canonical conflict => DIFFERENT without a judge) never fires here: 0 of 420 pairs. The canonical layer is useful the other way round: it certifies identical facts (58 of 100 A0-A3 pairs, strict judge says DIFFERENT on 3 of them) and leaves a small residual.

Floors reachable (A0-A3, n = 100, upper bounds in brackets): (i) judge only 11-16% (18-23%); (ii) canonical only 0% (3.0%), which is the floor before any residual judging; canonical plus a core-rubric residual judge is NOT measured by this cache (0% to 36%), 2% (6.2%) / 5% (10.2%) if history evidence must match; (iii) typed fields 0% (3%), 1% (4.7%) if unstated required fields count, but this is the floor of a weaker output definition on a copy task, not a reachable floor for draft_note. Calibration rows: (i) none suffice; (ii) 46-124 for the floor alone at 10% / 5% under (c+tol), 59 / 29 under (c); (iii) 59 / 29 with no errors, 93 for 5% with one error; a selective grid certificate needs 131 / 64 at zero errors. The budget must also cover the pruned step's own errors, so 10% is credible and 5% is borderline on 140 rows.

**Should S16b run with the new equivalence? Yes, and it must change in five places** (`s16_pruner_note/run_b.py`; I did not edit it):
0. Fix the judge call first: `core.J` uses max_tokens 40 and a strict exact-word parse, and 3 of 100 floor verdicts were cut off (finish=length) and counted as DIFFERENT. Raise max_tokens or parse the first word, and stop counting FAIL as DIFFERENT.
1. Replace `core.verdict` (strict judge, majority-of-3) by an `equiv(refs, x)` that returns DIFFERENT only if `canon` finds a conflict or loses a required field (flag, amount, invoice, reason, date) relative to A0, otherwise runs the core-rubric judge (`JSYS_CORE`) in both orders, DIFFERENT only if both say DIFFERENT, with the second-family judge as adjudicator. Keep the strict verdict as a secondary column; report the definition and the history/evidence policy with the certificate.
2. Relabel the spans before training the scorers: the cached ablation notes can be relabelled with `core_change` at no cost (table 6: history needed on 9 of 140 prompts instead of 57). Three prompts that are canonical-only positives have no block-level (B0-B2) notes cached; treat them as "evidence spread over the history" as S16 does, or generate those 9 notes. Decide explicitly whether history evidence (which earlier charge, usual range) is a core fact: the label rate for dropping the history moves from 6% (core facts only) to as much as 33% (46 of 140 variants show any evidence change under the loose evidence test), so this choice, not the model, sets how much can be pruned.
3. Recompute the floor with the same `equiv` (floor_cell on 0-79 needs the 4th note for 80 prompts, about $0.1) and keep the "excess over floor" certificate; with a floor of 0-2% the excess correction matters little.
4. Add a typed arm (flag decision, reason, invoice outcome, amount, bank mismatch as fields rendered by template), because it is the only route that removes the optional-detail variance by construction. Validate `canon` on the fresh S16b notes before trusting it (ground truth check is free), and hand-check about 50 residual pairs judged by the core rubric as the calibration slice (the research note's 200 stratified pairs is more than this synthetic task needs).

## Verbatim examples

Generated by `run.py` (same text is in `tables.md`). Category 1 = strict judge DIFFERENT while no extracted fact conflicts; category 2 = strict judge SAME while the canonical facts conflict or are unstated (there is no (c) conflict with judge SAME in the cache, so the list is filled as stated).


### Category 1: judge DIFFERENT, canonical facts do not conflict under (c) (presentation / emphasis / omission differences)

29 pairs in this category in the cache; showing 8.

**1. prompt 227 (noinv), pair A0-A3, judge strict=DIFFERENT, confirmed=SAME; canonical conflict=none; one-sided/unstated=none; history mention A/B=True/True**

> A (A0): Transaction 2025-06-03 at Orchid Gas Station #9670 for $89.33 (posted 2025-06-04, card ****9161) has no matching invoice and was flagged for that reason. The ledger shows no invoice on any of the 30 recent Orchid Gas Station or PayPal *ORCHIDGASS transactions listed.
>
> B (A3): Transaction 2025-06-03, Orchid Gas Station #9670, $89.33 (posted 2025-06-04 to card ****9161) has no matched invoice and was flagged for that reason. Ledger history shows over 30 prior Orchid Gas Station transactions, all also without invoices.

**2. prompt 240 (normal), pair A0-A3, judge strict=DIFFERENT, confirmed=DIFFERENT; canonical conflict=none; one-sided/unstated=none; history mention A/B=False/False**

> A (A0): Transaction of $28.48 at Pacific Fulfillment, Phoenix AZ, on 2026-03-20 matched invoice INV-001533 (vendor V0028, $28.45, within the 2% tolerance). Not flagged: the invoice matched and the amount is within the usual range. Bank feed posted 2026-03-21 at -$29.33.
>
> B (A3): Transaction of $28.48 at Pacific Fulfillment (Phoenix, AZ) on 2026-03-20 matched INV-001533 (vendor V0028, $28.45, within the 2% tolerance), so it was not flagged as an anomaly. The bank feed posted it 2026-03-21 at -$29.33 on card ****4468.

**3. prompt 201 (noinv), pair A0-A3, judge strict=DIFFERENT, confirmed=DIFFERENT; canonical conflict=none; one-sided/unstated=none; history mention A/B=True/True**

> A (A0): Transaction 2025-06-08, SQ *SUNRISE CARGO, $12.36 (posted 2025-06-10, card ****8463) has no matching invoice and was flagged for that reason. The vendor has 27 prior transactions in the ledger from 2024-07-05 through 2025-05-31, none with a matching invoice.
>
> B (A3): Transaction 2025-06-08, SQ *SUNRISE CARGO, $12.36, posted 2025-06-10 to card ****8463. No matching invoice exists. Flagged because no invoice matched. The ledger shows about 26 prior Sunrise Cargo charges from July 2024 to May 2025, none with an invoice.

**4. prompt 205 (noinv), pair A0-A3, judge strict=DIFFERENT, confirmed=DIFFERENT; canonical conflict=none; one-sided/unstated=none; history mention A/B=True/True**

> A (A0): On 2025-04-13, a $18.46 GOLDEN NOODLE HOUSE transaction (posted 2025-04-14 to card ****4030) has no matched invoice and was flagged for no matching invoice. The vendor has 24 prior ledger entries, each with an invoice number, most recently INV-304847 on 2025-03-29.
>
> B (A3): Transaction 2025-04-13 at GOLDEN NOODLE HOUSE for $18.46 (posted 2025-04-14, card ****4030) has no matched invoice and was flagged for that reason. The vendor has 24 prior invoiced transactions in the ledger history, so this is the first unmatched payment in the records provided.

**5. prompt 206 (dup), pair A0-A3, judge strict=DIFFERENT, confirmed=DIFFERENT; canonical conflict=none; one-sided/unstated=['bank_amt']; history mention A/B=True/False**

> A (A0): Transaction 2026-06-21, SQ *BLUE FRESH MARKET #455, $23.33, has no matched invoice and was flagged as a possible duplicate of a recent charge. The bank feed shows the same merchant and amount posted 2026-06-22, and the ledger shows an identical $23.33 charge at #455 on 2026-06-17.
>
> B (A3): Transaction 2026-06-21, SQ *BLUE FRESH MARKET #455, $23.33, has no matched invoice and was flagged as a possible duplicate of a recent charge. The bank feed shows the same merchant and amount posted 2026-06-22 (-$23.33, card ****5388). The matching $23.33 charge at #455 on 2026-06-17 also has no invoice.

**6. prompt 222 (first), pair A0-A3, judge strict=DIFFERENT, confirmed=DIFFERENT; canonical conflict=none; one-sided/unstated=none; history mention A/B=False/True**

> A (A0): Transaction 2026-02-27 for lakesidemarket, $60.03, has no matched invoice and was flagged as an anomaly because it is the first purchase from this vendor. The bank feed shows the posted amount as -$61.83 on 2026-03-01 (card ****8283), which differs from the ledger amount.
>
> B (A3): Transaction 2026-02-27 for lakesidemarket, $60.03, has no matched invoice and was flagged as a first purchase from this vendor. The bank feed shows the posted amount as -$61.83 on 2026-03-01 (card ****8283), a $1.80 difference from the transaction amount. Recent history shows many prior vendor purchases with and without invoices.

**7. prompt 242 (spike), pair A0-A3, judge strict=DIFFERENT, confirmed=DIFFERENT; canonical conflict=none; one-sided/unstated=none; history mention A/B=True/False**

> A (A0): Transaction 2026-10-05, Riverside Car Rental #2136 San Jose, $1,818.52, posted 2026-10-07, matched to INV-001554 (vendor V0265, $1,825.09, issued 2026-09-21) within the 2% tolerance. Flagged as an anomaly for unusually high amount for this vendor; the ledger shows prior charges mostly between $150 and $720, with one at $1,482.71.
>
> B (A3): Transaction 2026-10-05 for $1,818.52 (Riverside Car Rental, San Jose CA; posted 2026-10-07) matches INV-001554 ($1,825.09, issued 2026-09-21) within the 2% tolerance. It was flagged as an anomaly because the amount is unusually high for this vendor.

**8. prompt 246 (spike), pair A0-A3, judge strict=DIFFERENT, confirmed=DIFFERENT; canonical conflict=none; one-sided/unstated=none; history mention A/B=True/True**

> A (A0): Transaction 2026-07-16 ($298.02, Riverside Analytics Dallas) matches INV-001605 ($302.53, vendor V0165) within the 2% tolerance and posted 2026-07-17. It was flagged as anomalous because the amount is unusually high for this vendor, whose prior charges mostly fall between about $50 and $105.
>
> B (A3): Transaction 2026-07-16 ($298.02, Riverside Analytics Dallas TX) matched INV-001605 (V0165, $302.53), within the 2% tolerance. It was flagged as an anomaly because the amount is unusually high for this vendor, whose prior charges mostly range about $49 to $105, with one at $177.07.

### Category 2: judge SAME, canonical facts conflict or are unstated

Strictly under (c) there are 0 such pairs in the cache. To give examples anyway the list is filled in this order: (c) conflicts (0), history-evidence conflicts (4), a core field unstated or the bank amount stated in only one note (11); each example says which.

**1. prompt 265 (spike), pair A0-A3, judge strict=SAME, confirmed=SAME; canonical conflict=none; one-sided/unstated=none; history mention A/B=True/True; why listed: history-evidence conflict ['hist_hi', 'hist_lo']**

> A (A0): Transaction 2025-05-01 for $1,082.57 (PRIMETELEPHONE #3592, posted 2025-05-03) matches INV-001772 from vendor V0012 ($1,085.12, issued 2025-04-20) within the 2% tolerance. Flagged as an anomaly because the amount is unusually high for this vendor; prior transactions in the ledger range from about $99 to $249.
>
> B (A3): Transaction 2025-05-01 for $1,082.57 (PRIMETELEPHONE #3592, posted 2025-05-03) matches INV-001772 (vendor V0012, $1,085.12), within the 2% tolerance. Flagged as an anomaly because the amount is unusually high for this vendor; prior charges in the ledger range from about $100 to $250.

**2. prompt 288 (spike), pair A0-A3, judge strict=SAME, confirmed=SAME; canonical conflict=none; one-sided/unstated=none; history mention A/B=True/True; why listed: history-evidence conflict ['hist_hi', 'hist_lo']**

> A (A0): Transaction 2026-01-19 ($2,675.50, Granite Surety Nashville) matched INV-001937 (vendor V0156, $2,719.22, within the 2% tolerance). Flagged as an anomaly because the amount is unusually high for this vendor; recent history shows typical charges of roughly $150–$480.
>
> B (A3): Transaction 2026-01-19 ($2,675.50, Granite Surety Nashville) matched INV-001937 (vendor V0156, $2,719.22), within the 2% tolerance. It was flagged because the amount is unusually high for this vendor; the prior 20 transactions range from about $167 to $481.

**3. prompt 252 (spike), pair A2-A3, judge strict=SAME; canonical conflict=none; one-sided/unstated=none; history mention A/B=True/True; why listed: history-evidence conflict ['hist_hi', 'hist_lo']**

> A (A2): Transaction 2026-10-10 for $846.26 to Prime Power & Light (posted 2026-10-11) matches INV-001719 ($861.51, issued 2026-09-21) within the 2% tolerance. It was flagged because the amount is unusually high for this vendor, where recent charges have typically been around $100–$200.
>
> B (A3): Transaction 2026-10-10 for $846.26 to Prime Power & Light matched INV-001719 (vendor V0007, $861.51, issued 2026-09-21) within the 2% tolerance. It was flagged as an anomaly because the amount is unusually high for this vendor, whose prior payments ranged roughly $82 to $210.

**4. prompt 294 (spike), pair A1-A3, judge strict=SAME; canonical conflict=none; one-sided/unstated=none; history mention A/B=True/True; why listed: history-evidence conflict ['hist_hi']**

> A (A1): Transaction 2025-05-03 for $3,075.40 (Prime Prote, Boston MA, pending) matches INV-002007 from vendor V0158 at $3,069.59, within the 2% tolerance. It was flagged because the amount is unusually high for this vendor, whose prior ledger transactions range roughly $400 to $900.
>
> B (A3): Transaction 2025-05-03 for $3,075.40 (Prime Prote, Boston MA, pending) matches INV-002007 (vendor V0158, $3,069.59, within 2%). It was flagged because the amount is unusually high for this vendor, whose prior transactions range roughly $400 to $925.

**5. prompt 215 (noinv), pair A0-A3, judge strict=SAME, confirmed=SAME; canonical conflict=none; one-sided/unstated=['bank_amt']; history mention A/B=True/True; why listed: bank amount stated in one note only**

> A (A0): Transaction 2025-02-10, Silver Suppl, $87.75 (bank posted 2025-02-11, card ****6538) has no matching invoice and was flagged for that reason. The ledger shows many prior Silver Suppl transactions, all without invoices, so the lack of a match appears consistent with this vendor's history.
>
> B (A3): Transaction 2025-02-10 with Silver Suppl for $87.75 has no matching invoice and was flagged for that reason. The bank feed shows it posted 2025-02-11 as -$87.75 on card ****6538. Recent ledger history shows numerous prior Silver Suppl transactions, all without invoices.

**6. prompt 219 (normal), pair A0-A3, judge strict=SAME, confirmed=SAME; canonical conflict=none; one-sided/unstated=['bank_amt']; history mention A/B=False/False; why listed: bank amount stated in one note only**

> A (A0): Transaction 2025-05-02 (WILLOW ANALYTICS #6273, $48.47) matched to INV-001406 (vendor V0150, $47.56, issued 2025-04-14), within the 2% tolerance. Not flagged: invoice matched and amount is in the usual range. Bank feed posted 2025-05-03.
>
> B (A3): Transaction 2025-05-02 (WILLOW ANALYTICS #6273, $48.47) matched to INV-001406 (vendor V0150, $47.56, issued 2025-04-14), within the 2% tolerance. Not flagged: invoice matched and amount in usual range. Bank feed posted 2025-05-03 for -$48.47 on card ****6449.

**7. prompt 226 (spike), pair A0-A3, judge strict=SAME, confirmed=SAME; canonical conflict=none; one-sided/unstated=['bank_amt']; history mention A/B=True/True; why listed: bank amount stated in one note only**

> A (A0): Transaction 2025-08-30 POS NOVAMOVERS DENVER CO, $230.89, posted 2025-09-01 to card ****5046, matched to INV-001489 (vendor V0164, $232.49, issued 2025-08-27) within the 2% tolerance. Flagged as an anomaly because the amount is unusually high for this vendor; prior ledger charges range from about $9 to $38.
>
> B (A3): Transaction 2025-08-30 POS NOVAMOVERS DENVER CO for $230.89 matches INV-001489 (vendor V0164, $232.49, issued 2025-08-27) within the 2% tolerance, and posted 2025-09-01 at -$230.89 on card ****5046. It was flagged as an anomaly because the amount is unusually high for this vendor; the ledger history shows prior charges ranging from about $9 to $38.

**8. prompt 238 (normal), pair A0-A3, judge strict=SAME, confirmed=SAME; canonical conflict=none; one-sided/unstated=['bank_amt']; history mention A/B=False/False; why listed: bank amount stated in one note only**

> A (A0): Transaction 2026-11-28 ORCHID LEGAL PHOENIX AZ $297.30 (posted 2026-11-30, card ****9247) matches INV-001524 (vendor V0230, $292.62, issued 2026-11-10) within the 2% tolerance. Not flagged: invoice matched and the amount is in the usual range.
>
> B (A3): 2026-11-28 ORCHID LEGAL PHOENIX AZ $297.30 matched to INV-001524 (vendor V0230, $292.62, issued 2026-11-10), within the 2% tolerance. Not flagged: invoice matched and amount in the usual range. Bank posted 2026-11-30 at -$297.30.


## Revision

Review corrections, applied by editing `canon.py` (one regex) and `run.py` and re-running; the tables above were updated where numbers moved.

1. **Route (ii) is overstated by the first version.** What the cache measures for "canonical + residual core-rubric judge" is the canonical-only lower end, 0/100 (UB 3.0%). Under (c) 42 of 100 A0-A3 pairs are residual (no conflict, facts not identical). Only 6 of them have a cached core-rubric verdict (prompts 201, 239, 243, 258, 265, 273), all SAME: 0/6, UB 39.3%. The earlier "15 pairs, 0 DIFFERENT" counted 9 pairs that are SAME by construction (3 byte-identical, 6 identical canonical facts). The strict judge calls 13 of the 42 residual pairs DIFFERENT (11 real, 2 cut-off); in S16's 30-pair sample the core rubric still said DIFFERENT about 10% of the time. The honest range with residual judging is 0% to 36%; the reachable floor under (ii) is **unmeasured**, and the line "a 5% budget is plausible under (ii)" rested on that unmeasured number. "0-2%" is the floor before any residual judging.
2. **Three floor verdicts are cut-off judge answers, not one.** Prompts 207, 295, 296 have raw verdict FAIL because the judge output was cut at max_tokens 40 on all 3 retries (295 confirmed DIFFERENT on FAIL, FAIL; 296 on FAIL, SAME, FAIL; 207 confirmed SAME). Real DIFFERENT verdicts are 13/100 raw (UB 19.9%); confirmed DIFFERENT backed by at least one real DIFFERENT vote is 11/100 (UB 17.6%). Rows (a') and (b') in `log.txt`/`tables.md` give them. The attribution of the 16 raw DIFFERENT calls is now: 3 cut-off answers, 9 history or earlier-charge detail (201, 205, 242, 246, 252, 276, 279, 294 and 222, see item 7), 2 bank amount stated in one note (206, 284; both also differ in a history mention, so the two categories overlap), 2 with nothing visible (227, 240). Prompts 295 and 296 carry a one-sided earlier-charge date but were wrongly credited to content before; they are format failures. The judge-alone conclusion stands at 11-13% (UB 18-20%): neither budget is certifiable. Missing change for `run_b.py` added as item 0 of the Verdict list.
3. **Reproducibility AUC bug fixed.** The reference-pair score exists only for prompts 200-259 (60 prompts); prompts 260-299 silently scored 0. On the 60: non-identical-facts count AUC 0.878 (0.940 with the 3 cut-off prompts dropped, 9 positives), (c+) count 0.700 (0.722). The earlier 0.714 and 0.625 are replaced. This strengthens "disagreement is prompt-driven". The model-based AUC with the 3 cut-off prompts dropped is 0.707 (0.687 before).
4. **The typed floor (iii) is close to a tautology.** It assumes (a) that values extracted from free text equal what a typed call would emit (no typed call was made), (b) that the extractor is accurate, checked only on its own tuning set, and (c) that the prompt already contains every typed field verbatim ("Anomaly flag: yes (reason: ...)", invoice id and amount, tx amount), so 0/100 measures copying fidelity, not agreement on a judgment. The typed fields also exclude the evidence the strict rubric asks for and the bank mismatch (stated in only 73% of mismatch notes), so a typed arm in S16b would show history and bank as prunable by template construction. Read 0% (UB 3%) as the floor of a weaker output definition on a copy task, not as a reachable floor for draft_note.
5. **"0 of 520 core conflicts" is partly by construction.** The vendor field returns only "recognised" or "not stated", so a vendor conflict cannot occur, and in validation "accuracy among stated" for vendor is 100% by definition. The statement "two notes never disagree about the transaction" holds for amount, invoice outcome and flag decision; vendor agreement is **not measured**.
6. **The 4 (c) conflicts were extractor false positives.** The FOLLOW regex matched "reviewer should" in "a difference ... that the reviewer should note" (A0 of prompts 32 and 57), which is an observation, not a follow-up request. The regex now requires an action verb after "reviewer should/to/may". Follow-up requested is 0 of 640 notes and (c) is 0 of 520. Those pairs actually differ in whether a real bank-feed mismatch is mentioned: across the 72 reference pairs on bank-mismatch prompts, 12 state the mismatch in exactly one note. (c) treats that as one-sided, not as a conflict; it is a substantive omission and a stricter core definition may want it counted.
7. **Canonical facts miss at least one defensible judge-DIFFERENT pair.** Example 6 (prompt 222, first purchase): A3 says "Recent history shows many prior vendor purchases", but the record's history is all other vendors, so it reads as contradicting the stated reason. The judge was arguably right; "presentation difference" undersells it, so the residual is not purely presentation, which further weakens item 1 of the verdict.
8. **Two caveats on the relabel claim (H0 changes core facts on 9 of 140 prompts, against 57 strict).** (a) The canonicalizer was never validated on ablated notes, which are out of distribution (missing lines, the bank date standing in for the tx date). (b) The 57 to 9 drop comes from deciding that the evidence is not a fact. The strict rubric lists the earlier charge a duplicate matches and the usual-range comparison as required, and run_b.py item 2 above admits the label rate swings from 6% to 33% on this choice. "Pruning headroom far larger" is therefore **conditional on that decision**, not a finding.

Corrected verdict in one line: a judge alone cannot certify 10% (floor 11-13%, UB 18-20%); canonical conflicts alone are 0/100 (UB 3.0%) but the residual judging step that route (ii) needs is unmeasured (0% to 36%); the typed floor is a copy-task floor of a weaker output; whether S16b may use the new equivalence depends on the explicit decision whether history evidence is a core fact.
