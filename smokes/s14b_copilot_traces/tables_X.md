TraceLab v0.0.2: 743819 tool records, 743050 with wall latency; chronological split by session start as in S14 F: 546549 train / 196501 test calls. Copilot: train day 1 (1635686 calls), test day 2 (1674818 calls). History-only model = HistGradientBoosting (absolute-error loss on log1p latency, 300000 training rows) on tool-agnostic causal features only: latencies of the last 3 finished calls of the session, session EWMA (0.5), log number of finished calls. No tool name, no model, no time. Slow base rate (> 1 s): Copilot day 2 23.8%, TraceLab test 38.1%. AUC scores the > 1 s question with the predicted latency itself.

| test set | predictor | MAE ms | p90 abs err ms | mean abs log1p-ratio | AUC > 1 s | log-error gain vs own per-tool median (95% CI) |
|---|---|---|---|---|---|---|
| Copilot day 2 | own global median | 13030 | 4802 | 1.60 | 0.500 | - |
| Copilot day 2 | own per-tool median | 12790 | 4234 | 1.10 | 0.864 | - |
| Copilot day 2 | history-only GBM, trained on the same dataset | 12993 | 4578 | 1.47 | 0.634 | -0.37 (-0.38 to -0.36) |
| Copilot day 2 | history-only GBM, trained on TraceLab | 13031 | 4492 | 1.54 | 0.633 | -0.44 (-0.44 to -0.43) |
| TraceLab test sessions | own global median | 14382 | 11154 | 2.04 | 0.500 | - |
| TraceLab test sessions | own per-tool median | 14275 | 12091 | 1.48 | 0.734 | - |
| TraceLab test sessions | history-only GBM, trained on the same dataset | 13695 | 9422 | 1.44 | 0.798 | +0.03 (-0.08 to +0.14) |
| TraceLab test sessions | history-only GBM, trained on Copilot day 1 | 14148 | 10666 | 1.61 | 0.758 | -0.13 (-0.18 to -0.09) |
