### Floor on the same pairs: new equivalence vs strict rubric

| pairs | n | new equivalence DIFFERENT (CP UB) | strict raw (CP UB) | strict confirmed (CP UB) |
|---|---|---|---|---|
| pool 140 (labelled) | 140 | 3/140 = 2.1% (5.4%) | 17/140 = 12.1% (17.7%) | 10/140 = 7.1% (11.8%) |
| original floor ids 200-299 | 100 | 1/100 = 1.0% (4.7%) | 13/100 = 13.0% (19.9%) | 11/100 = 11.0% (17.6%) |
| all floor prompts | 180 | 4/180 = 2.2% (5.0%) | 21/180 = 11.7% (16.4%) | 14/180 = 7.8% (11.9%) |

By scenario (the 140 labelled pairs only: the log prints by-scenario over LAB, not the 180; new equivalence; parsed from log_b2_live.txt):

| scenario | n | DIFFERENT | CP UB |
|---|---|---|---|
| dup | 20 | 0/20 = 0.0% | 13.9% |
| first | 28 | 0/28 = 0.0% | 10.1% |
| noinv | 30 | 0/30 = 0.0% | 9.5% |
| normal | 45 | 2/45 = 4.4% | 13.3% |
| spike | 17 | 1/17 = 5.9% | 25.0% |

Typed floor (two independent typed records, field equality): 5/140 = 3.6% (CP UB 7.4%); differing fields {'bank_mismatch': 1, 'reasons': 4}.
Typed record vs prompt ground truth (copy fidelity, n=140): flag 140, reasons 131, invoice_outcome 140, invoice_id 140, invoice_amount 140, tx_amount 140, tx_date 140, bank_mismatch 139

### Span labels, new equivalence vs old strict labels (140 prompts)

| ablated span | new DIFFERENT | old strict | pairs |
|---|---|---|---|
| tx | 100.0% (140/140) | 62.9% | 140 |
| inv | 69.3% (97/140) | 77.9% | 140 |
| flag | 70.0% (98/140) | 95.0% | 140 |
| bank | 6.4% (9/140) | 17.9% | 140 |
| whole history (H0) | 11.4% (16/140) | 40.7% (57/140, from the log) | 140 |

Relabel paths over 748 pairs: {'canon-conflict': 272, 'canon-identical': 309, 'judge-onesided': 96, 'canon-lost:tx_date': 37, 'canon-lost:inv': 25, 'canon-lost:reason': 2, 'canon-lost:tx_amt,tx_date': 5, 'canon-lost:flag,inv,reason': 2}; FAIL 0.

### Pruned cells, free text

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

### Pruned cells, typed

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

Text DIFFERENT by scenario (n: dup 20, first 28, noinv 30, normal 45, spike 17):

| cell | dup | first | noinv | normal | spike |
|---|---|---|---|---|---|
| text: learned m=0 | 3/20 | 3/28 | 2/30 | 5/45 | 3/17 |
| text: learned m=5 | 1/20 | 0/28 | 0/30 | 1/45 | 2/17 |
| text: learned m=2 | 1/20 | 0/28 | 0/30 | 2/45 | 1/17 |
| text: rule (tx+inv+flag) | 0/20 | 1/28 | 1/30 | 5/45 | 2/17 |
| text: cross-encoder m=5 | 0/20 | 1/28 | 0/30 | 2/45 | 2/17 |
| text: cross-encoder m=2 | 0/20 | 0/28 | 0/30 | 3/45 | 1/17 |
| text: oracle m=5 | 0/20 | 0/28 | 0/30 | 2/45 | 3/17 |
| text: random m=5 | 7/20 | 0/28 | 0/30 | 0/45 | 3/17 |
| text: oracle m=2 | 0/20 | 0/28 | 0/30 | 2/45 | 2/17 |
| text: random m=2 | 8/20 | 0/28 | 0/30 | 2/45 | 3/17 |
| typed: learned m=0 | 0/20 | 4/28 | 6/30 | 0/45 | 2/17 |
| typed: learned m=5 | 0/20 | 4/28 | 9/30 | 0/45 | 2/17 |
| typed: learned m=2 | 0/20 | 4/28 | 10/30 | 0/45 | 2/17 |
| typed: rule (tx+inv+flag) | 3/20 | 6/28 | 9/30 | 6/45 | 5/17 |
| typed: cross-encoder m=5 | 0/20 | 3/28 | 7/30 | 0/45 | 2/17 |
| typed: cross-encoder m=2 | 0/20 | 4/28 | 9/30 | 0/45 | 2/17 |
| typed: oracle m=5 | 0/20 | 3/28 | 8/30 | 0/45 | 1/17 |
| typed: random m=5 | 0/20 | 4/28 | 8/30 | 1/45 | 2/17 |
| typed: oracle m=2 | 0/20 | 3/28 | 10/30 | 0/45 | 2/17 |
| typed: random m=2 | 0/20 | 4/28 | 12/30 | 0/45 | 2/17 |

### Certificate table (rotation cross-fitting, 140 usable rows each)

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

### Rows needed if the observed rate held (CP upper bound <= budget, errors = round(rate x n))

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

With one calibration fold per rotation (a fifth of the pool), the pool size is five times the 'grid certifier' columns. These are rows of an error-free-looking sample at the observed rate; they are the minimum, not a power calculation, and treat rows as independent (rows share vendors and prompts).

### REVISION: chance a calibration sample certifies at true error rate p (binomial sampling), grid certifier delta=0.05/40

| calibration rows | budget | p=2.9% | p=3.6% |
|---|---|---|---|
| 28 | 15% | 0.00 | 0.00 |
| 28 | 10% | 0.00 | 0.00 |
| 69 | 15% | 0.68 | 0.55 |
| 84 | 15% | 0.78 | 0.65 |
| 84 | 10% | 0.09 | 0.05 |
| 140 | 15% | 0.98 | 0.94 |
| 140 | 10% | 0.63 | 0.44 |

Rows for 80% power (grid certifier; power stays >= 80% for the next 150 sizes) and the one-fold pool (x5):

| true rate | budget | calibration rows | pool (x5) |
|---|---|---|---|
| 2.9% | 15% | 92 | 460 |
| 2.9% | 10% | 202 | 1010 |
| 3.6% | 15% | 113 | 565 |
| 3.6% | 10% | 259 | 1295 |

CP upper bound (plain 95%) of 4/140: 6.4%; 5/140: 7.4%

Vendor GroupKFold(5) fold sizes on the 140 prompts: [28, 28, 28, 28, 28]. Smallest error-free calibration n for the grid certifier at 15%: 42; so no fold of 28 rows can certify any budget <= 15%.

`tx` ablation pairs (140): canon-lost 35; canon-conflict by conflicting fields {('tx_date',): 97, ('tx_amt', 'tx_date'): 7, ('reason', 'tx_amt'): 1}; of the tx_date conflicts: {'extractor took the invoice issue date': 78, 'other date mismatch': 26}. Pairs that turn on tx_date: 139 of 140.

Strict rubric by scenario on the 180 floor pairs (cache-only):

| scenario | n | strict raw | strict confirmed |
|---|---|---|---|
| dup | 26 | 1/26 = 3.8% | 1/26 = 3.8% |
| first | 32 | 2/32 = 6.2% | 2/32 = 6.2% |
| noinv | 36 | 7/36 = 19.4% | 2/36 = 5.6% |
| normal | 59 | 3/59 = 5.1% | 2/59 = 3.4% |
| spike | 27 | 8/27 = 29.6% | 7/27 = 25.9% |
| all | 180 | 21/180 = 11.7% | 14/180 = 7.8% |

Same, 140 labelled pairs only (the rows the log's by-scenario new-equivalence lines use):

| scenario | n | strict raw | strict confirmed |
|---|---|---|---|
| dup | 20 | 1/20 = 5.0% | 1/20 = 5.0% |
| first | 28 | 1/28 = 3.6% | 1/28 = 3.6% |
| noinv | 30 | 7/30 = 23.3% | 2/30 = 6.7% |
| normal | 45 | 3/45 = 6.7% | 2/45 = 4.4% |
| spike | 17 | 5/17 = 29.4% | 4/17 = 23.5% |
