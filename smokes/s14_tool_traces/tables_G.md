Toucan-1.5M (Apache-2.0), one shard OSS/train-00010-of-00047 (147 MB; all rows are subset single-turn-original; the first shard I tried, train-00027, was all 'irrelevant' prompts with no tool calls): 9726 prompts, 29398 tool calls, 1083 distinct tool names, 399 distinct MCP servers (tool -> server via each prompt's server tool list; '?' when unresolved: 1.5% of calls), 20796 distinct (server, tool, args). Returns that look like errors / unavailable-tool messages (keyword heuristic on the first 300 chars): 18.5% of calls.
Calls whose exact (server, tool, args) occurs in at least one other call: 10693 = 36.4% of calls (2091 keys). (Within the same prompt too, so see the cross-prompt number below.)
Keys seen in >= 2 different prompts: 1846; all returns byte-identical (after JSON key-order normalization) for 65.7% of them; identical after masking numbers and timestamps for 77.5%.
Among those cross-prompt keys, 16.5% are mostly error-looking returns; restricted to keys with no error-looking return: identical for 61.9% (n=1513).

Tools (MCP server, tool name): 1297. 923 (71%) have fewer than 2 repeated argument keys in this shard, so their behaviour is unobservable here. Of the 374 with >= 2 repeated keys: 188 (50%) deterministic (>= 95% of repeated keys return identical text), 61 mixed, 125 (33%) time- or state-varying (< 50% of repeated keys identical). Weighted by calls: deterministic tools carry 51% of the calls of these 374 tools.
Of the 188 deterministic tools, 24 return mostly error-looking text (constant failure messages, e.g. missing credentials or a Windows-only tool in a Linux sandbox), so 'deterministic' there means 'deterministically broken'.

Most-called observable tools:
| tool | calls | repeated arg keys | % repeated keys with identical return | % of those calls error-looking |
|---|---|---|---|---|
| get_area_code | 274 | 8 | 88% | 100% |
| find-place | 258 | 33 | 100% | 100% |
| drawing_fillRectangle | 216 | 2 | 100% | 0% |
| pinterest_search | 216 | 5 | 100% | 0% |
| search-news | 204 | 22 | 68% | 0% |
| search_tour_info | 186 | 19 | 95% | 98% |
| search | 172 | 3 | 100% | 0% |
| resolve-library-id | 161 | 8 | 25% | 44% |
| create_desktop_txt_file | 157 | 4 | 100% | 0% |
| get_movies | 129 | 8 | 100% | 0% |
| get_member_phone | 126 | 6 | 100% | 0% |
| get-weibo-trending | 117 | 6 | 83% | 98% |
| get_weather_forecast | 117 | 20 | 40% | 0% |
| pokemon_query | 117 | 2 | 100% | 0% |

Examples of tools whose identical calls returned different text (excluding error-looking returns):
- `resolve-library-id` args `{"libraryName":"jsonwebtoken"}`: 4 calls, 4 distinct returns, e.g. `'Available Libraries (top matches):\n\nEach result includes:\n- Library ID: Context7-compatible identifier (format'` vs `'Available Libraries (top matches):\n\nEach result includes:\n- Library ID: Context7-compatible identifier (format'`
- `get_weather_forecast` args `{"city":"Braga","days":5}`: 2 calls, 2 distinct returns, e.g. `'**Previsão para Braga**\n\nCoordenadas: 41.5475, -8.4227\nÚltima atualização: 2025-08-14T14:31:02\n\n**2025-08-14**'` vs `'**Previsão para Braga**\n\nCoordenadas: 41.5475, -8.4227\nÚltima atualização: 2025-08-14T21:31:02\n\n**2025-08-14**'`
- `status` args `{"command":"--all"}`: 12 calls, 5 distinct returns, e.g. `'Status for All Platforms:\n\nAmplitude: Operational ✅\n\nAnthropic: Operational ✅\n\nAsana: Operational ✅\n\nAtlassian'` vs `'Status for All Platforms:\n\nAmplitude: Operational ✅\n\nAnthropic: Operational ✅\n\nAsana: Operational ✅\n\nAtlassian'`
- `fetch_reddit_hot_threads` args `{"limit":10,"subreddit":"Christianity"}`: 2 calls, 2 distinct returns, e.g. `"Title: August's Banner -- World Humanitarian Day\nScore: 7\nComments: 10\nAuthor: McClanky\nType: text\nContent: Th"` vs `"Title: August's Banner -- World Humanitarian Day\nScore: 8\nComments: 10\nAuthor: McClanky\nType: text\nContent: Th"`
- `get_weather` args `{"city":"London"}`: 8 calls, 3 distinct returns, e.g. `'🌤️ Weather in London, United Kingdom:\n📍 Location: London, United Kingdom\n🌡️ Temperature: 26°C (79°F)\n🌈 Conditi'` vs `'🌤️ Weather in London, United Kingdom:\n📍 Location: London, United Kingdom\n🌡️ Temperature: 22°C (72°F)\n🌈 Conditi'`
- `get_current_weather` args `{"location":"25.7617,-80.1918"}`: 4 calls, 4 distinct returns, e.g. `'# Current Weather\n\n**Location:** 25.7617, -80.1918\n**Station:** KMIA\n**Observed:** 8/14/2025, 8:25 AM (local t'` vs `'# Current Weather\n\n**Location:** 25.7617, -80.1918\n**Station:** KMIA\n**Observed:** 8/17/2025, 1:45 PM (local t'`

Examples of deterministic tools with a large repeated-key set:
- `drawing_generateCanvas`: 109 calls, 18 repeated keys, e.g. args `{"height":600,"width":800}` -> `'Canvas generated with width: 800, height: 600'`
- `get_component_examples`: 52 calls, 15 repeated keys, e.g. args `{"componentName":"data-table"}` -> `'{\'title\': \'Code Example 1\', \'code\': \'"use client"\\n\\nimport * as React from "react"\\nimport {\\n  ColumnDef,\\n  ColumnFil'`
- `convert_volume`: 101 calls, 15 repeated keys, e.g. args `{"from_unit":"ml","to_unit":"cup","value":250}` -> `'{"from_unit":"ml","to_unit":"cup","value":1.0567}'`
- `get_fruit_nutrition`: 72 calls, 14 repeated keys, e.g. args `{"fruit_name":"apple"}` -> `'{"family":"Rosaceae","genus":"Malus","id":6,"name":"Apple","nutritions":{"calories":52,"carbohydrates":11.4,"fat":0.4,"p'`

Certified skipping on Toucan (episodes = prompts; memory 50% / calibration 25% / test 25% of prompts, 10 random splits; confidence = k/(n+1) as for tau-bench; every tool treated as read-only):
| budget | certified (splits) | test calls skipped % | realized error % of skipped | oracle: seen in memory and majority correct % |
|---|---|---|---|---|
| 1% | 1/10 | 0.9 | 0.2 | 20.0 |
| 2% | 5/10 | 4.6 | 0.7 | 20.0 |
| 5% | 10/10 | 12.3 | 2.3 | 20.0 |