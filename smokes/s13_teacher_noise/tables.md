| model | cond | reps | calls | parse fails (empty / truncated / off-list) | flip rate | pairwise disagree | pairwise excl. empty-content calls | floor vs best fixed student |
|---|---|---|---|---|---|---|---|---|
| claude-haiku-5.5 | A | 5 | 500 | 16 (0 / 15 / 1) | 7.0% | 3.50% | 3.50% | 2.00% |
| claude-sonnet-5.5 | A | 5 | 500 | 65 (47 / 18 / 0) | 6.0% | 3.60% | 0.45% | 2.20% |
| gemini-3.5-flash-lite | A | 5 | 500 | 2 (0 / 0 / 2) | 9.0% | 4.20% | 4.20% | 2.40% |
| gpt-5.4-mini | A | 5 | 500 | 1 (0 / 0 / 1) | 23.0% | 10.90% | 10.90% | 6.00% |
| qwen3-32b | A | 5 | 500 | 7 (0 / 0 / 7) | 9.0% | 5.20% | 5.20% | 3.40% |
| claude-sonnet-5.5 | A2 | 5 | 500 | 5 (5 / 0 / 0) | 3.0% | 1.20% | 0.41% | 0.60% |
| claude-haiku-5.5 | B | 5 | 500 | 18 (0 / 16 / 2) | 8.0% | 3.60% | 3.60% | 2.00% |
| gpt-5.4-mini | B | 5 | 500 | 2 (0 / 0 / 2) | 15.0% | 8.00% | 8.00% | 4.80% |
| claude-haiku-5.5 | C:amazon-bedrock | 2 | 200 | 7 (0 / 6 / 1) | 2.0% | 2.00% | 2.00% | 1.00% |
| claude-haiku-5.5 | C:anthropic | 2 | 200 | 6 (0 / 6 / 0) | 3.0% | 3.00% | 3.00% | 1.50% |
| gpt-5.4-mini | C:azure | 2 | 200 | 1 (0 / 0 / 1) | 10.0% | 10.00% | 10.00% | 5.00% |
| claude-haiku-5.5 | C:azure/global | 2 | 200 | 6 (0 / 6 / 0) | 2.0% | 2.00% | 2.00% | 1.00% |
| gemini-3.5-flash-lite | C:google-ai-studio | 2 | 200 | 0 (0 / 0 / 0) | 5.0% | 5.00% | 5.00% | 2.50% |
| claude-haiku-5.5 | C:google-vertex/global | 2 | 200 | 6 (0 / 6 / 0) | 2.0% | 2.00% | 2.00% | 1.00% |
| qwen3-32b | C:siliconflow/fp8 | 2 | 200 | 5 (0 / 0 / 5) | 5.0% | 5.00% | 5.00% | 2.50% |
| qwen3-32b | C:tenstorrent/fp8 | 2 | 200 | 4 (0 / 0 / 4) | 0.0% | 0.00% | 0.00% | 0.00% |

| model | cond | single-run acc (mean, min-max) | majority-vote acc | prompts wrong in all reps |
|---|---|---|---|---|
| claude-haiku-5.5 | A | 75.2% (73.0%-77.0%) | 76.0% | 23 |
| claude-sonnet-5.5 | A | 78.4% (76.0%-80.0%) | 79.0% | 20 |
| gemini-3.5-flash-lite | A | 73.6% (71.0%-75.0%) | 73.0% | 22 |
| gpt-5.4-mini | A | 70.8% (69.0%-72.0%) | 72.0% | 22 |
| qwen3-32b | A | 64.0% (63.0%-65.0%) | 65.0% | 34 |
| claude-sonnet-5.5 | A2 | 84.8% (84.0%-85.0%) | 85.0% | 15 |
| claude-haiku-5.5 | B | 75.6% (75.0%-77.0%) | 76.0% | 23 |
| gpt-5.4-mini | B | 70.6% (68.0%-72.0%) | 71.0% | 24 |
| claude-haiku-5.5 | C:amazon-bedrock | 74.0% (74.0%-74.0%) | 74.0% | 26 |
| claude-haiku-5.5 | C:anthropic | 73.0% (72.0%-74.0%) | 72.0% | 26 |
| gpt-5.4-mini | C:azure | 71.5% (70.0%-73.0%) | 73.0% | 25 |
| claude-haiku-5.5 | C:azure/global | 75.0% (75.0%-75.0%) | 75.0% | 25 |
| gemini-3.5-flash-lite | C:google-ai-studio | 74.0% (73.0%-75.0%) | 75.0% | 24 |
| claude-haiku-5.5 | C:google-vertex/global | 75.0% (74.0%-76.0%) | 74.0% | 24 |
| qwen3-32b | C:siliconflow/fp8 | 69.0% (67.0%-71.0%) | 71.0% | 29 |
| qwen3-32b | C:tenstorrent/fp8 | 67.0% (67.0%-67.0%) | 67.0% | 33 |

| model | cond | p50 ms | p95 ms | prompt tok | completion tok | reasoning tok (mean/max) | $/call | $/1000 labels |
|---|---|---|---|---|---|---|---|---|
| claude-haiku-5.5 | A | 828 | 1047 | 787 | 10.1 | 0.00/0 | 0.000084 | 0.084 |
| claude-sonnet-5.5 | A | 1149 | 2264 | 788 | 10.9 | 1.87/20 | 0.001685 | 1.685 |
| gemini-3.5-flash-lite | A | 574 | 775 | 562 | 5.4 | 0.00/0 | 0.000182 | 0.182 |
| gpt-5.4-mini | A | 762 | 977 | 439 | 8.1 | 0.00/0 | 0.000366 | 0.366 |
| qwen3-32b | A | 390 | 1031 | 434 | 7.7 | 1.00/1 | 0.000037 | 0.037 |
| claude-sonnet-5.5 | A2 | 1148 | 3216 | 788 | 24.9 | 15.00/300 | 0.001825 | 1.825 |
| claude-haiku-5.5 | B | 818 | 976 | 787 | 10.1 | 0.00/0 | 0.000084 | 0.084 |
| gpt-5.4-mini | B | 762 | 973 | 439 | 8.1 | 0.00/0 | 0.000366 | 0.366 |
| claude-haiku-5.5 | C:amazon-bedrock | 1137 | 2488 | 787 | 10.2 | 0.00/0 | 0.000084 | 0.084 |
| claude-haiku-5.5 | C:anthropic | 727 | 1039 | 787 | 10.0 | 0.00/0 | 0.000084 | 0.084 |
| gpt-5.4-mini | C:azure | 951 | 1902 | 439 | 8.1 | 0.00/0 | 0.000366 | 0.366 |
| claude-haiku-5.5 | C:azure/global | 762 | 1023 | 787 | 10.0 | 0.00/0 | 0.000084 | 0.084 |
| gemini-3.5-flash-lite | C:google-ai-studio | 435 | 609 | 564 | 5.4 | 0.00/0 | 0.000183 | 0.183 |
| claude-haiku-5.5 | C:google-vertex/global | 533 | 772 | 787 | 10.0 | 0.00/0 | 0.000084 | 0.084 |
| qwen3-32b | C:siliconflow/fp8 | 1211 | 1446 | 433 | 6.7 | 2.00/2 | 0.000064 | 0.064 |
| qwen3-32b | C:tenstorrent/fp8 | 597 | 903 | 433 | 8.7 | 1.00/1 | 0.000045 | 0.045 |

| model | cond | providers (calls) | prompts with provider change | flip rate if provider changed | flip rate if same provider | pair disagree diff-provider | pair disagree same-provider |
|---|---|---|---|---|---|---|---|
| claude-haiku-5.5 | A | {'Claude Platform on AWS': 500} | 0 | - | 7.0% | - | 3.50% |
| claude-sonnet-5.5 | A | {'Claude Platform on AWS': 500} | 0 | - | 6.0% | - | 3.60% |
| gemini-3.5-flash-lite | A | {'Google': 500} | 0 | - | 9.0% | - | 4.20% |
| gpt-5.4-mini | A | {'OpenAI': 500} | 0 | - | 23.0% | - | 10.90% |
| qwen3-32b | A | {'DeepInfra': 495, 'Tenstorrent': 5} | 0 | - | 9.0% | - | 5.20% |
| claude-sonnet-5.5 | A2 | {'Claude Platform on AWS': 500} | 0 | - | 3.0% | - | 1.20% |
| claude-haiku-5.5 | B | {'Claude Platform on AWS': 500} | 0 | - | 8.0% | - | 3.60% |
| gpt-5.4-mini | B | {'OpenAI': 500} | 0 | - | 15.0% | - | 8.00% |
| claude-haiku-5.5 | C:amazon-bedrock | {'Amazon Bedrock': 200} | 0 | - | 2.0% | - | 2.00% |
| claude-haiku-5.5 | C:anthropic | {'Anthropic': 200} | 0 | - | 3.0% | - | 3.00% |
| gpt-5.4-mini | C:azure | {'Azure': 200} | 0 | - | 10.0% | - | 10.00% |
| claude-haiku-5.5 | C:azure/global | {'Azure': 200} | 0 | - | 2.0% | - | 2.00% |
| gemini-3.5-flash-lite | C:google-ai-studio | {'Google AI Studio': 200} | 0 | - | 5.0% | - | 5.00% |
| claude-haiku-5.5 | C:google-vertex/global | {'Google': 200} | 0 | - | 2.0% | - | 2.00% |
| qwen3-32b | C:siliconflow/fp8 | {'SiliconFlow': 200} | 0 | - | 5.0% | - | 5.00% |
| qwen3-32b | C:tenstorrent/fp8 | {'Tenstorrent': 200} | 0 | - | 0.0% | - | 0.00% |

| model | pinned endpoint | served by | cross-provider disagree vs default-routing A (all pairs) | within pinned endpoint (2 reps) | single-run acc | p50 ms | p95 ms | $/call |
|---|---|---|---|---|---|---|---|---|
| claude-haiku-5.5 | amazon-bedrock | {'Amazon Bedrock': 200} | 4.50% | 2.0% | 74.0% | 1137 | 2488 | 0.000084 |
| claude-haiku-5.5 | anthropic | {'Anthropic': 200} | 5.50% | 3.0% | 73.0% | 727 | 1039 | 0.000084 |
| gpt-5.4-mini | azure | {'Azure': 200} | 12.00% | 10.0% | 71.5% | 951 | 1902 | 0.000366 |
| claude-haiku-5.5 | azure/global | {'Azure': 200} | 6.80% | 2.0% | 75.0% | 762 | 1023 | 0.000084 |
| gemini-3.5-flash-lite | google-ai-studio | {'Google AI Studio': 200} | 10.30% | 5.0% | 74.0% | 435 | 609 | 0.000183 |
| claude-haiku-5.5 | google-vertex/global | {'Google': 200} | 4.60% | 2.0% | 75.0% | 533 | 772 | 0.000084 |
| qwen3-32b | siliconflow/fp8 | {'SiliconFlow': 200} | 10.80% | 5.0% | 69.0% | 1211 | 1446 | 0.000064 |
| qwen3-32b | tenstorrent/fp8 | {'Tenstorrent': 200} | 11.40% | 0.0% | 67.0% | 597 | 903 | 0.000045 |
