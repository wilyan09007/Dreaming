Positions: every tool call that has a following call in the same session. Train 1598165 positions (day 1), test 1634662 (day 2). Target = next call's tool name, vocabulary = 24 most common day-1 tools + OTHER (1.2% of test targets are OTHER and count as misses; OTHER is never predicted). There are 855 distinct names on day 1 but the top 24 carry 98.8% of next calls. The GBM is fitted on 200000 random day-1 positions.

Top-1 / top-3 accuracy (%):
| positions | n | (a) most common tool | (b) bigram | (b) trigram | (b) 4-gram | (c) GBM |
|---|---|---|---|---|---|---|
| all positions | 1634662 | 35.0 / 61.7 | 45.3 / 73.1 | 47.8 / 77.2 | 49.0 / 79.0 | 48.1 / 79.5 |
| next call opens a new turn (after a user message) | 123904 | 37.6 / 60.5 | 37.8 / 60.5 | 37.8 / 66.9 | 37.1 / 68.4 | 36.3 / 68.6 |
| inside a turn, current call in the first batch | 172330 | 38.8 / 59.4 | 46.6 / 76.1 | 47.9 / 78.0 | 48.1 / 79.4 | 46.5 / 79.8 |
| inside a turn, later batch | 1338428 | 34.2 / 62.1 | 45.9 / 73.9 | 48.7 / 78.1 | 50.2 / 79.9 | 49.4 / 80.4 |
| next call in the same batch (parallel) | 166488 | 64.9 / 71.6 | 73.6 / 89.4 | 72.5 / 90.6 | 72.1 / 90.8 | 72.7 / 91.2 |
| model Model A | 467659 | 35.4 / 69.6 | 48.6 / 75.2 | 50.9 / 79.3 | 52.0 / 81.0 | 51.1 / 81.1 |
| model Model B | 212280 | 34.1 / 64.1 | 42.4 / 75.1 | 45.2 / 77.6 | 46.5 / 78.8 | 45.3 / 78.7 |
| model Model C | 174823 | 35.5 / 68.1 | 46.9 / 75.4 | 49.3 / 78.6 | 50.1 / 80.4 | 49.1 / 80.1 |
| model Model D | 148442 | 37.6 / 50.5 | 47.0 / 70.0 | 48.9 / 74.6 | 49.9 / 76.4 | 48.8 / 76.4 |
| model Model E | 132518 | 33.6 / 62.8 | 42.5 / 75.3 | 45.9 / 78.1 | 47.2 / 79.6 | 46.2 / 79.3 |

Gate on the predicted probability of the top-1 tool (launch the speculation only if p >= tau). n-gram p = count share of the top tool in its context; GBM p = predicted probability. `correct launches / all positions` is the share of all positions where a correct speculation was launched:
| predictor | tau | launched share | precision of launched | correct launches / all positions |
|---|---|---|---|---|
| bigram (last tool) | 0.5 | 57.3% | 53.1% | 30.4% |
| bigram (last tool) | 0.7 | 0.0% | n/a | 0.0% |
| bigram (last tool) | 0.9 | 0.0% | n/a | 0.0% |
| trigram (last 2 tools, backoff to bigram if < 5 seen) | 0.5 | 50.3% | 58.5% | 29.4% |
| trigram (last 2 tools, backoff to bigram if < 5 seen) | 0.7 | 10.5% | 71.1% | 7.5% |
| trigram (last 2 tools, backoff to bigram if < 5 seen) | 0.9 | 0.0% | 28.6% | 0.0% |
| 4-gram (last 3 tools, backoff trigram then bigram) | 0.5 | 48.7% | 60.2% | 29.3% |
| 4-gram (last 3 tools, backoff trigram then bigram) | 0.7 | 9.5% | 75.6% | 7.2% |
| 4-gram (last 3 tools, backoff trigram then bigram) | 0.9 | 0.1% | 89.2% | 0.1% |
| GBM (last 3 tools, status, position, turn, model, batch size) | 0.5 | 40.5% | 62.4% | 25.3% |
| GBM (last 3 tools, status, position, turn, model, batch size) | 0.7 | 16.1% | 69.7% | 11.2% |
| GBM (last 3 tools, status, position, turn, model, batch size) | 0.9 | 4.3% | 50.4% | 2.2% |

Trigram gate at tau 0.7 by position type: all positions: launched 10.5%, precision 71.1%; next call opens a new turn (after a user message): launched 8.6%, precision 49.7%; inside a turn, current call in the first batch: launched 4.9%, precision 74.4%; inside a turn, later batch: launched 11.4%, precision 72.4%

Batch level (day 2; 7.1% of 1508330 batches have 2+ tools; among those, 62.2% contain a single distinct tool). Given the first tool of a batch (and the fact that the batch has a slot there, i.e. its size is at least the slot index; the size is also used to split the columns), predict each of the other slots. Top-1 / top-3 accuracy (%) over all non-first slots (166488):
| predictor | all non-first slots | batch of 2 | batch of 3-4 | batch of 5+ |
|---|---|---|---|---|
| unconditional (mode of non-first slots) | 64.9 / 80.2 | 51.6 / 77.7 | 76.9 / 85.5 | 71.9 / 73.0 |
| P(tool given previous tool in the batch) | 76.4 / 92.9 | 59.9 / 87.8 | 87.0 / 96.7 | 96.2 / 97.6 |
| P(tool given first tool and slot index) | 75.6 / 92.5 | 60.1 / 87.8 | 85.4 / 95.9 | 94.8 / 97.2 |
| rest identical to the first tool | 71.8 / 71.8 | 56.0 / 56.0 | 80.8 / 80.8 | 93.5 / 93.5 |

All non-first slots of a batch right (argmax of P(tool given first tool and slot index)): 66.1% of multi-tool batches; baseline 'rest identical to the first tool': 62.2%.
