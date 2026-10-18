Positions = every tool call that has a following tool call in the same episode (think calls dropped). Prediction is made when the call is issued, before its return.
| regime | domain | agent | positions/seed | next is read % | unigram top-1 % | name n-gram top-1 % | name top-3 % | name+args (copy heuristic) top-1 % | name+args top-3 % | + trajectory cache top-1 % | + trajectory cache top-3 % | traj. cache matches % | traj. cache precision % | useful speculation (read-only AND exactly right) % of next calls | speculations launched % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W | retail | gpt-4o | 1380 | 72.8 | 36.7 | 66.8 | 86.5 | 29.3 | 34.2 | 67.1 | 71.2 | 93.6 | 71.5 | 55.1 | 75.6 |
| W | retail | sonnet-3.5 | 3039 | 75.2 | 40.0 | 67.1 | 87.8 | 24.4 | 27.7 | 72.7 | 75.9 | 96.9 | 75.0 | 60.8 | 76.7 |
| W | airline | gpt-4o | 455 | 56.0 | 34.1 | 55.2 | 79.8 | 9.2 | 14.7 | 49.0 | 51.4 | 74.9 | 64.8 | 40.7 | 59.6 |
| W | airline | sonnet-3.5 | 1108 | 70.7 | 39.1 | 61.3 | 84.0 | 12.0 | 19.0 | 64.5 | 66.2 | 92.3 | 69.6 | 54.8 | 74.1 |
| X | retail | gpt-4o | 825 | 73.1 | 37.4 | 60.3 | 78.2 | 28.1 | 32.6 | 40.5 | 46.0 | 51.0 | 52.4 | 38.1 | 81.2 |
| X | retail | sonnet-3.5 | 1822 | 75.6 | 39.6 | 63.0 | 82.0 | 24.1 | 27.2 | 41.1 | 45.3 | 51.7 | 55.5 | 39.0 | 80.5 |
| X | airline | gpt-4o | 261 | 58.8 | 38.6 | 51.6 | 71.6 | 10.0 | 12.8 | 15.6 | 18.4 | 15.2 | 53.5 | 14.2 | 74.9 |
| X | airline | sonnet-3.5 | 730 | 70.9 | 42.7 | 55.5 | 71.2 | 11.0 | 15.9 | 16.7 | 20.9 | 16.7 | 47.3 | 15.9 | 80.1 |

Breakdown (combined predictor, name+args top-1 / name top-1; seeds pooled):
- W/retail/gpt-4o: after user turn: 58.8% (share 57%), immediate chain: 77.8% (share 43%), name top1, next is read: 80.1% (share 73%), name top1, next is write: 31.2% (share 27%), name+args top1, next is read: 75.6% (share 73%), name+args top1, next is write: 44.3% (share 27%), name:after user turn: 55.1% (share 57%), name:immediate chain: 82.0% (share 43%)
- W/retail/sonnet-3.5: after user turn: 53.7% (share 39%), immediate chain: 84.9% (share 61%), name top1, next is read: 77.2% (share 75%), name top1, next is write: 36.5% (share 25%), name+args top1, next is read: 80.8% (share 75%), name+args top1, next is write: 48.2% (share 25%), name:after user turn: 45.4% (share 39%), name:immediate chain: 81.0% (share 61%)
- W/airline/gpt-4o: after user turn: 29.8% (share 40%), immediate chain: 61.7% (share 60%), name top1, next is read: 70.2% (share 56%), name top1, next is write: 36.0% (share 44%), name+args top1, next is read: 72.5% (share 56%), name+args top1, next is write: 19.0% (share 44%), name:after user turn: 25.4% (share 40%), name:immediate chain: 74.8% (share 60%)
- W/airline/sonnet-3.5: after user turn: 37.1% (share 33%), immediate chain: 78.0% (share 67%), name top1, next is read: 70.5% (share 71%), name top1, next is write: 39.1% (share 29%), name+args top1, next is read: 77.5% (share 71%), name+args top1, next is write: 33.2% (share 29%), name:after user turn: 29.1% (share 33%), name:immediate chain: 77.0% (share 67%)
- X/retail/gpt-4o: after user turn: 35.1% (share 56%), immediate chain: 47.3% (share 44%), name top1, next is read: 76.0% (share 73%), name top1, next is write: 17.6% (share 27%), name+args top1, next is read: 52.1% (share 73%), name+args top1, next is write: 9.0% (share 27%), name:after user turn: 48.2% (share 56%), name:immediate chain: 75.3% (share 44%)
- X/retail/sonnet-3.5: after user turn: 20.5% (share 38%), immediate chain: 54.0% (share 62%), name top1, next is read: 75.9% (share 76%), name top1, next is write: 23.3% (share 24%), name+args top1, next is read: 51.7% (share 76%), name+args top1, next is write: 8.6% (share 24%), name:after user turn: 37.5% (share 38%), name:immediate chain: 79.0% (share 62%)
- X/airline/gpt-4o: after user turn: 12.7% (share 38%), immediate chain: 17.4% (share 62%), name top1, next is read: 74.7% (share 59%), name top1, next is write: 18.8% (share 41%), name+args top1, next is read: 24.1% (share 59%), name+args top1, next is write: 3.5% (share 41%), name:after user turn: 21.7% (share 38%), name:immediate chain: 70.4% (share 62%)
- X/airline/sonnet-3.5: after user turn: 5.3% (share 31%), immediate chain: 21.9% (share 69%), name top1, next is read: 69.5% (share 71%), name top1, next is write: 21.3% (share 29%), name+args top1, next is read: 22.3% (share 71%), name+args top1, next is write: 3.0% (share 29%), name:after user turn: 20.7% (share 31%), name:immediate chain: 71.2% (share 69%)

Speculation gate (launch only when the top-1 call is read-only and score >= tau; score = trajectory-cache agreement, else 0.5 x name probability): launched % of positions / precision %
- W/retail/gpt-4o: tau 0.0: 75.6% / 73%; tau 0.3: 68.8% / 78%; tau 0.5: 62.0% / 83%; tau 0.7: 25.9% / 93%; tau 0.9: 1.4% / 100%
- W/retail/sonnet-3.5: tau 0.0: 76.7% / 79%; tau 0.3: 74.7% / 81%; tau 0.5: 66.5% / 86%; tau 0.7: 44.3% / 95%; tau 0.9: 9.3% / 99%
- W/airline/gpt-4o: tau 0.0: 59.6% / 68%; tau 0.3: 54.1% / 74%; tau 0.5: 47.0% / 80%; tau 0.7: 14.3% / 89%; tau 0.9: 0.0% / nan%
- W/airline/sonnet-3.5: tau 0.0: 74.1% / 74%; tau 0.3: 71.2% / 76%; tau 0.5: 57.3% / 85%; tau 0.7: 36.7% / 93%; tau 0.9: 2.7% / 97%
- X/retail/gpt-4o: tau 0.0: 81.2% / 47%; tau 0.3: 63.2% / 60%; tau 0.5: 36.3% / 73%; tau 0.7: 20.1% / 79%; tau 0.9: 0.7% / 100%
- X/retail/sonnet-3.5: tau 0.0: 80.5% / 48%; tau 0.3: 69.9% / 55%; tau 0.5: 40.4% / 69%; tau 0.7: 26.6% / 75%; tau 0.9: 3.7% / 88%
- X/airline/gpt-4o: tau 0.0: 74.9% / 19%; tau 0.3: 48.8% / 29%; tau 0.5: 14.3% / 48%; tau 0.7: 5.9% / 75%; tau 0.9: 0.0% / nan%
- X/airline/sonnet-3.5: tau 0.0: 80.1% / 20%; tau 0.3: 48.8% / 24%; tau 0.5: 12.5% / 59%; tau 0.7: 7.8% / 75%; tau 0.9: 0.4% / 100%