# S16b v2 run plan (output of `python run_b2.py --dry`, no network)

Code: `run_b2.py` (driver) and `equiv.py` (equivalence, typed arm). Nothing was run live. `run.py`, `run_b.py`, `core.py`, `llm.py` and `responses.jsonl` are untouched; the canonicalizer comes from `../s16c_judge_floor_offline/canon.py`.

## What will run
1. **Relabel** the per-span ablation labels (tx, inv, flag, bank, whole history H0, history blocks B0-B2) from the cached ablation notes, using the new equivalence against the reference note A0. Cached B0-B2 block notes are judged only when the new H0 label is DIFFERENT or unknown (otherwise `span_new` ignores them). Missing block notes are generated only for prompts whose new H0 is DIFFERENT. **Relabel makes new calls only with `--live`, in every mode** (without it, the stage is cache-only and the unanswerable pairs stay masked in scorer training); the relabel row below is therefore NOT spent by a run without `--live`.
2. **Floor**: A0 vs an independent full-prompt note (rep 3), same equivalence, on the 140 pool prompts plus floor ids 260-299 (180 prompts). The rep-3 notes for ids 200-299 are all cached; the 74 missing ones are ids 6-79 (ids 0-5 came from the earlier `run_b.py --tiny` run).
3. **Typed floor and typed cells**: teacher returns a JSON record (flag, reasons, invoice outcome/id/amount, transaction amount/date, bank mismatch yes/no); the note is rendered from a fixed template ("No bank-feed mismatch noted." when there is no mismatch) and a sample of rendered typed notes per cell is saved in `results_b2*.json` under `typed_samples` and printed once per cell. Equivalence = field equality after normalisation (reasons mapped to the five allowed categories, flag and bank mismatch to yes/no, invoice outcome to matched/none, amounts absolute). Typed token savings use the typed full-prompt token count as denominator (own system prompt), not the free-text one.
4. **Pruned cells** in priority order: learned m=0, m=5, m=2, rule (tx+inv+flag), cross-encoder m=5/2, oracle m=5/2, random m=5/2, each as free text and as typed, with the certificate and the "excess over floor" variant at 5/10/15%. The strict-rubric secondary column runs on the floor and the first 3 cells (`--strict-cells`).
5. **Rotation cross-fitting** (replaces run_b.py's single-scorer folds): vendor-grouped GroupKFold (5 folds, asserted that no vendor is in two folds). Rotation f = test fold f, calibration fold f+1 (`--cal-folds`, default 1), scorer trained on the remaining three folds, so the cutoff applied to fold f never depends on fold f's labels and the scorer has not seen the labels of the rows it calibrates on. The learned cells therefore need pruned notes for up to two keep-sets per prompt (own test rotation and the rotation that calibrates on its fold); non-learned scorers and m=0 share one keep-set. Reported rates use the test-role rows only. Trade-off: one calibration fold is about 28 prompts, so the 5% budget cutoff is conservative; `--cal-folds 2` (3 keep-sets per prompt) is about $2.81 total in the dry estimate instead of $2.31 (with relabel).

Equivalence (`equiv.py`): DIFFERENT if the canonicalizer finds a conflict or a lost required field relative to A0; SAME without a judge if canonical facts are identical (history evidence ignored under `--evidence core`, required to match under `--evidence loose`); otherwise the residual judge (core rubric under both policies; the rubric is reported next to every rate) in both presentation orders, second-family judge as adjudicator on any non-agreement. Judge output is parsed for the first SAME/DIFFERENT word with max_tokens 200; an unparsed verdict is FAIL, excluded from every rate and counted next to it. Old cached verdicts (max_tokens 40/30) are reused only when they parsed. Canonicalizer fixes after review: a transaction amount counts as stated if its value appears among any amount in the candidate (negative or bank-adjacent wording), an invoice amount is ignored when the outcome is "no invoice" or the invoice ids differ, and bank posting dates are not transaction-date candidates.

## What the cache already covers (core evidence policy)
- Relabel (799 pairs after skipping block notes of H0-SAME prompts): 648 answerable now; the rest need judge calls or block notes. Dry-run `canon-lost` paths: tx_date 37 (+5 with tx_amt), inv 25, reason 2, flag/inv/reason 2.
- Floor: 93 of 180 pairs answerable. 74 need the rep-3 note (ids 6-79). The floor DIFFERENT count on the answerable ids 200-299 is now 0/88; the two DIFFERENTs the first version reported (prompts 262 and 285) were both extractor mistakes (a lost invoice amount under "no invoice", a negative or bank-adjacent transaction amount), fixed in `canon_decide`/`canon2`.
- m=0 cell: 118 of 140 answerable; rule cell 56 of 140; every other pruned cell and all typed calls are fresh.
- The `tx` label is 1.00 because the `tx_date` requirement alone makes H0 positive on those prompts: the candidates give only the bank posting date (prompts 24, 42, 70, 209, 212, 219 and most of the tx ablations), which is now removed from the date candidates, so they show up as `canon-lost:tx_date` rather than as a date conflict. Whether a missing transaction date should be a required field is a design choice, not a model measurement; check this before trusting the span labels.
- Partial-cache numbers printed by the dry run are biased toward canon-decided pairs and no certificate is computed from them. Do not quote them.
- Free facts from the dry run: the extractor states and gets right flag, date, invoice outcome and reason on all 140 reference notes (amount on 139); new span labels under core evidence: tx 1.00, inv 0.76, flag 0.75, bank 0.00 (answerable part), history needed on 11 of 118 answerable prompts (old strict labels: 57 of 140), so the history-evidence decision, not the model, sets the pruning headroom; the strict rubric reproduces on the answerable floor pairs: raw 12 of 98 (old 13/100 minus the cut-off verdicts that now need a re-ask).

## Estimated calls and dollars (observed prices: teacher $0.00014, primary judge $0.0013, second-family judge $0.00028 per call; ledger overall $0.00074/call over 6274 calls, $4.56 per log.txt)
```
stage                             answerable/pairs teacher   judge1  judge2    est $
relabel                               648/799           63      194    12.2    0.264
floor                                  93/180           74      163     2.9    0.223
typed_floor                             0/140          280        0     0.0    0.040
cell:learned m=0                      118/140            0       49     2.9    0.065
typed:learned m=0                       0/140          140        0     0.0    0.020
cell:learned m=5                        0/215          215      398     3.5    0.550
typed:learned m=5                       0/215          215        0     0.0    0.031
cell:learned m=2                        2/192          189      352     3.2    0.486
typed:learned m=2                       0/192          192        0     0.0    0.027
cell:rule (tx+inv+flag)                56/140           74       39     2.5    0.062
typed:rule (tx+inv+flag)                0/140          140        0     0.0    0.020
cell:cross-encoder m=5                  1/140          138       37     2.4    0.068
typed:cross-encoder m=5                 0/140          140        0     0.0    0.020
cell:cross-encoder m=2                  7/140          132       35     2.3    0.066
typed:cross-encoder m=2                 0/140          140        0     0.0    0.020
cell:oracle m=5                         1/140          138       37     2.4    0.068
typed:oracle m=5                        0/140          140        0     0.0    0.020
cell:random m=5                         0/140          140       35     2.3    0.067
typed:random m=5                        0/140          140        0     0.0    0.020
cell:oracle m=2                         7/140          132       35     2.3    0.066
typed:oracle m=2                        0/140          140        0     0.0    0.020
cell:random m=2                         0/140          140       35     2.3    0.067
typed:random m=2                        0/140          140        0     0.0    0.020
TOTAL                                                 3242     1410    41.3    2.307   (cap $3.0; policy evidence=core; relabel row is spent only with --live)
```
[dry] network calls made: 0. Cache hits 727. Calls still needed in total: teacher/typed 3242, judge1 1410, judge2 41.3 (needs totals; the shim's miss log is empty because note_c / J2 check the cache before calling it). Scorer embedding STUB(dry: zero embeddings, cos feature = 0), cross-encoder STUB(dry).
[time] 4s

Without `--live` the relabel row ($0.264) is not spent, so a run without it costs about $2.04; with `--live` about $2.31. Counts for pairs whose notes do not exist yet are extrapolations (canonical residual share of ablation-like pairs, 1.6 strict-column calls per pair on the first 3 cells, 13% adjudicator rate); counts for cached notes are exact. The only price risk is a slightly higher primary-judge price now that judge output can run to 200 tokens.
Under `--evidence loose` the dry run gives $4.02 with relabel ($3.18 without), because history-mention differences go to the judge. That exceeds the $3.0 cap: the run does not crash, each stage and cell is pre-checked against `left()` and wrapped in a handler for the spend-cap error, so lower-priority cells are skipped with a `[SKIP]` line, partial results are saved, and the headline and summary are still written. Run loose as a separate, smaller run (for example `--stages relabel,floor --live`).

## Launch
`python run_b2.py --tiny --live` (live wiring check, a dozen prompts, 3 folds, well under $0.2), then `python run_b2.py --live --cap-usd 3.0` (preflight = this script's own `--dry` estimate in a subprocess, printed first; a failed estimate aborts the run; spend is capped through `llm.CAP`). Stages can be split with `--stages relabel,floor,cells,typed`. `--dry --evidence loose --tag _loose` reproduces the loose-policy estimate.

## Open design choices
- Evidence policy default `core`; `loose` changes only which pairs reach the judge (the rubric stays the core one).
- Lost required field is one-directional (reference states it, candidate does not) and decided without a judge; sending `canon-lost` pairs to the two-order judge is the alternative if the extractor is still suspected.
- Strict column is `core.verdict` logic re-implemented with the fixed parse (the original would reuse the 40-token bug on new calls).
- Embeddings and cross-encoder need Hugging Face; both fall back to marked stubs (embedding) or are skipped (cross-encoder) if unavailable, and are always stubs in `--dry`.
