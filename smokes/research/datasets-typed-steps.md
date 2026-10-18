# datasets-typed-steps: public datasets as typed-step traces

Checked 2026-10-17 via HF API / GitHub API / curl HEAD (unauthenticated) and page fetches. "no-login" = an anonymous curl got HTTP 200 on the data file.

## Summary

- There is NO public real-world bank-transaction dataset with raw merchant descriptors that we can download without login. Every HF set found is synthetic or LLM-generated. Kaggle has some (flag: login) but I did not confirm any.
- Best categorize_transaction input source: `DoDataThings/us-bank-transaction-categories-v2` (MIT, 68,000 rows, 17 labels, no-login CSV 3.5 MB). It has realistic descriptors like `[debit] PP*SAFEWAY`, `[credit] ACME CORP PAYROLL PPD ID: 123456789`, `TST*GOLDEN DRAGON - OAKLAND`. Its generator is `scripts/generate-training-data.js` in `wnstnb/foliome` (Apache-2.0).
- `mitulshah/transaction-categorization` (4.5M rows, MIT) is gated "auto": anonymous download returns HTTP 401. It needs an HF login/token. Skip it, since the 4.5M rows are template-synthetic anyway.
- Dreaming certifies against the teacher (frontier model) output, not human gold. So these datasets only need realistic INPUTS. Re-label with the teacher and keep the dataset label only as a sanity check. That makes the synthetic-only situation tolerable. The real risk is over-optimistic certification (see Blockers).
- Banking77, CLINC150, MASSIVE and TREC are all no-login and cleanly licensed. They give real human-written text with real label confusion, so they are good for stress-testing the certification curve (a student's error rate vs confidence). They map to classification call sites with 77, 151, 60 and 50 labels.
- Loading-script datasets break on current `datasets` (v4+ dropped scripts): `PolyAI/banking77`, `CogComp/trec` and `AmazonScience/massive` are script-only. Use `mteb/banking77`, `SetFit/TREC-QC`, the MASSIVE S3 tarball or its parquet branch, and `clinc/clinc_oos` (native parquet).
- For tool calls with real tool returns, the best fit is deterministic local environments. `tau2-bench` (MIT) has a `banking_knowledge` domain (97 tasks, 17-table bank DB, 266 bank transactions with raw descriptions like `ATM WITHDRAWAL - CHASE BANK #4521 CHICAGO IL`). Tool code is plain Python over JSON, so replay is byte-deterministic.
- Large trace corpora with tool role messages: Nemotron-Agentic-v1 (335k, CC-BY-4.0, no-login, tool returns are LLM-simulated), Toucan-1.5M (1.6M rows, Apache-2.0, real MCP server executions), ToolACE (11.3k, Apache-2.0), ToolBench (126k instances, Apache-2.0, recorded RapidAPI responses). Simulated returns are easier to predict than real ones, so use them for plumbing, not for the headline tool-return-predictor result.
- Nothing found for match_invoice, draft_note, or reviewer-edit (human-review deferral) traces. I did not search those specifically. Plan to synthesize them.

## Findings

### A. Single-step classification benchmarks (mission 2 students, certification curves)

| Dataset | Exact id / URL | Size | Labels | License | No-login? | Maps to |
|---|---|---|---|---|---|---|
| Banking77 | `mteb/banking77` https://huggingface.co/datasets/mteb/banking77 (original `PolyAI/banking77` is script-only, https://huggingface.co/datasets/PolyAI/banking77) | mteb: 9,993 train + 3,076 test = 13,069 (original card: 13,083) | 77 intents | original CC-BY-4.0; mteb tag says MIT | Yes. `train.jsonl` gives 200, rows like `{"text":..., "label":11, "label_text":"card_arrival"}` | categorize_transaction analog (bank-domain intent routing). Fine-grained, 77-way, so a hard classifier test. |
| CLINC150 | `clinc/clinc_oos` https://huggingface.co/datasets/clinc/clinc_oos | small 7,600/3,100/5,500; imbalanced 10,625/3,100/5,500; plus 15,250/3,100/5,500 (train/val/test) | 150 intents x 10 domains + `oos` = 151 | CC-BY-3.0 | Yes. Native parquet, e.g. `plus/train-00000-of-00001.parquet` | Classification with an abstain class. The `oos` vs in-scope binary maps directly to flag_anomaly yes/no and to "defer to teacher". |
| MASSIVE | `AmazonScience/massive` https://huggingface.co/datasets/AmazonScience/massive. Direct tarball (HTTP 200, 39.5 MB): https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz. Parquet branch: `.../resolve/refs%2Fconvert%2Fparquet/en-US/train/0000.parquet` (HTTP 200) | 52 languages; en-US 11,514 train / 2,033 dev / 2,974 test (=16,521; card text says 19,521 per language, a discrepancy) | 60 intents, 18 scenarios, 55 slot types | CC-BY-4.0 | Yes | Two typed outputs per utterance: intent (classification) and slot filling (extraction). Candidate for the "shared student across steps" mission, e.g. one student with a scenario tag vs per-step students. |
| TREC-QC | `SetFit/TREC-QC` https://huggingface.co/datasets/SetFit/TREC-QC (original `CogComp/trec` is script-only, license "unknown") | train.jsonl 5,452 rows | 6 coarse, 50 fine | not stated on SetFit; source Li and Roth COLING 2002 (https://cogcomp.seas.upenn.edu/Data/QA/QC/) | Yes | Small, easy classifier. Good for the fastest smoke (tiny train time). Coarse vs fine labels give a two-level label hierarchy test. |

### B. Bank-transaction categorization (raw descriptor strings)

All results: HF search on "transaction categor", "merchant", "bank statement" (2026-10-17). I found no real-world, human-labeled public set.

| Dataset | URL | Size | Labels | License | No-login? | Notes |
|---|---|---|---|---|---|---|
| DoDataThings v2 (recommended) | https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2 | 68,000 rows (4,000/class), CSV `description,category`, 3.5 MB | 17: Restaurants, Groceries, Shopping, Transportation, Entertainment, Utilities, Subscription, Healthcare, Insurance, Mortgage, Rent, Travel, Education, Personal Care, Transfer, Income, Fees | MIT | Yes (200). Modified 2026-04-06 | Synthetic from 500+ real merchant names with randomized store numbers, addresses and reference codes. Sign prefix `[debit]`/`[credit]`. Sample rows: `[debit] NORTHWESTERN MUTUALEFT PYMT PPD ID: 9901629141` (Insurance), `[debit] PYPL*365 MARKET` (Groceries). Closest to what a bookkeeping agent sees. Clones exist (`kaifkazi13/...-v2`, `DEVILHADYOURMOM/...`); use the original. |
| DoDataThings v1 | https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories | about 16,000 per its model card | 17 | MIT | Yes | Superseded by v2. |
| foliome generator | https://github.com/wnstnb/foliome (file `scripts/generate-training-data.js`, 60 KB) | n/a | n/a | Apache-2.0 | Yes | Probable source of the v2 data (the dataset card links this repo; the repo README says its DistilBERT classifier is trained on synthetic US bank data). Lets us generate more rows or unseen-merchant splits. I did not read the script. |
| karthiksagarn/bank-statement-categorization | https://huggingface.co/datasets/karthiksagarn/bank-statement-categorization | 7,512 rows | 12 (Education, Travel & Transport, Groceries, Miscellaneous, Bills & Utilities, Health & Fitness, Shopping, Entertainment, Investments, Income, Food & Drinks, Withdrawals) | NOT stated, empty card | Yes | Indian UPI/NEFT/POS style: `UPI/P2M/695829655510/COURSERA`, `POS DEBIT 150368212498 RELIANCEDIGITAL`. Exactly 12 labels, matching our demo. License gap means research use only. Provenance unknown. |
| Dikshith4500/bank-transaction-categories-50k | https://huggingface.co/datasets/Dikshith4500/bank-transaction-categories-50k | 6,973 rows (title says 50k) | 8 | not stated | Yes | `source` column says "generated". Examples: `BEST BUY #2156 *CD9012`, `CHIPOTLE MEXICAN GR #0123 TX`. Mislabeled size, empty card. |
| mitulshah/transaction-categorization | https://huggingface.co/datasets/mitulshah/transaction-categorization | 4,501,043 rows, parquet about 71 MB, fields `transaction_description, category, country, currency` | 10 | MIT | NO. gated "auto"; anonymous GET returned HTTP 401 | Tagged `annotations_creators: machine-generated`. Descriptions are mostly clean (`Uber Ride`, `Netflix Subscription`), not bank-statement raw. Not worth a login. |
| Sumeetgpt/indian-transaction-categorization-synthetic | https://huggingface.co/datasets/Sumeetgpt/indian-transaction-categorization-synthetic | under 1k rows | not checked | MIT | Yes | Too small. |
| Spanish synthetic | https://huggingface.co/marinamen/gastos-bancarios-classifier (model card; companion `marinamen/gastos-bancarios-es`, not opened) | 1,400 rows | 14 | not checked | not checked | Spanish; low value. |
| AgamiAI/Indian-Bank-Statements | https://huggingface.co/datasets/AgamiAI/Indian-Bank-Statements | 10k-100k docs, synthetic | n/a | Apache-2.0 | Yes (not opened) | Document-AI (statement parsing) rather than per-line labels. Many identical copies under other usernames. |
| MCC tables | https://huggingface.co/datasets/jayjayjet/merchant-category-codes (MIT); https://github.com/greggles/mcc-codes | reference tables | n/a | MIT / license unverified | Yes | Code-to-description maps for a label taxonomy or a tool backing `vendor_lookup`. No transactions. |
| Kaggle (fallback) | e.g. "Personal Budget Transactions Dataset" (ismetsemedov), about 3,600 rows, no description column per aggregator listing: https://baselight.app/u/kaggle/dataset/ismetsemedov_personal_budget_transactions_dataset | | | | LOGIN REQUIRED | Not downloaded, not recommended. Mint-format exports with `Original Description` exist only as personal data. |

Reality check: a DistilBERT on a similar synthetic set reports 99.88% test accuracy (https://huggingface.co/fahadkamraan/transaction-categorizer). Treat anything near that as a template-memorization signal, not evidence the certifier works.

### C. Multi-step agent traces with tool calls and tool returns

| Dataset | URL | Size | License | No-login? | Tool returns? | Maps to |
|---|---|---|---|---|---|---|
| tau2-bench (tau-cubed) | https://github.com/sierra-research/tau2-bench (v1.0.1, July 2026; last push 2026-10-16) | tasks: airline 50, retail 114, telecom 20 (small; `tasks_full.json` 14 MB), banking_knowledge 97, mock 10. DBs: airline 7.0 MB, retail 2.8 MB, banking_knowledge 270 KB | MIT | Yes (git clone, `uv sync`, Python >=3.12 and <3.14) | Yes. Tools are Python methods over JSON DBs in `src/tau2/domains/<d>/tools.py`; DBs in `data/tau2/domains/<d>/db.json`. Deterministic. | Mission 3 (tool return and latency predictor), mission 7 (rollout). Needs an LLM (via LiteLLM) only for the agent and user simulator. |
| tau2 `banking_knowledge` | https://github.com/sierra-research/tau2-bench/tree/main/data/tau2/domains/banking_knowledge | 17 tables incl. 338 credit-card txns (`merchant_name`, `category`: "Costco"/Groceries) and 266 bank txns (`description`: `ATM WITHDRAWAL - CHASE BANK #4521 CHICAGO IL`, `EVERYONEPAY - RECEIVED FROM DAD`); 698 knowledge docs; 97 tasks | MIT | Yes | Yes | The only bank-domain agent environment found. Read tools include `get_credit_card_transactions_by_user`, `get_user_information_by_id`, and `query_db`. Closest public stand-in for a `vendor_lookup` tool. Needs the `knowledge` extra. |
| tau-bench (v1) | https://github.com/sierra-research/tau-bench | airline and retail DBs (retail orders.json 1.8 MB); `historical_trajectories/`: gpt-4o-airline 4.1 MB, gpt-4o-retail 10.8 MB, sonnet-35-new-airline 10.9 MB, sonnet-35-new-retail 26.5 MB | MIT | Yes | Yes (in trajectories) | Repo says tasks are not updated; use tau2. The historical trajectories are free, ready-made logs of tool calls and returns, so they need no LLM spend. |
| AppWorld | https://github.com/StonyBrookNLP/appworld ; paper https://arxiv.org/abs/2407.18901 (ACL 2024) | 9 apps, 457 APIs, 100+ tables, 750 tasks; splits train/dev/test_normal/test_challenge | Apache-2.0 plus encrypted-bundle terms | Yes (`pip install appworld; appworld install; appworld download data`, Python 3.11+; no login mentioned) | Yes. Local sandbox executes API code, no LLM needed. train and dev ship solution code. | Mission 3/7. Heavier (code-as-action agent, not JSON tool calls). Terms: do not post extracted bundle data in plain text, do not tune on test splits. |
| BFCL | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard ; code https://github.com/ShishirPatil/gorilla | 52 files, 11.7 MB. Multi-turn (v3): 200 base + 800 augmented. Live: simple 258, multiple 1,037, irrelevance 875 | Apache-2.0 | Yes | No returns in the data files. Multi-turn execution uses Python backends in the gorilla repo (e.g. `GorillaFileSystem`, `TravelAPI` per changelog). | Function-call-routing student: which function and which args (extraction). `irrelevance` detection is a clean yes/no. The HF card still describes V3 (09/2024); the repo is at V4 (2025-07-17; last changelog entry 2025-10-01). |
| API-Bank | `liminghao1630/API-Bank` https://huggingface.co/datasets/liminghao1630/API-Bank ; https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank | Eval: 73 runnable APIs, 314 dialogues, 753 calls. Train: 2,138 APIs, 1,888 dialogues | MIT (HF) | Yes | Eval tools are runnable Python with a DB (`init_database/`); `response` fields exist | Small, old (2023). Level-1/2/3 gives an easy-to-hard tool-use ladder. Low priority. |
| ToolBench | Official: https://github.com/OpenBMB/ToolBench (data via Google Drive / Tsinghua Cloud). HF mirror: `tuandunghcmut/toolbench-v1` https://huggingface.co/datasets/tuandunghcmut/toolbench-v1 (188,304 train rows + 1,100 benchmark; 541 MB; apache-2.0) and `Maurus/ToolBench` (2.1 GB, no license tag) | 126,486 instances, 16,464 RapidAPI APIs, 3,451 tools, 469,585 real API calls | Apache-2.0, "research and educational purposes" | Yes via HF mirror | Yes, recorded RapidAPI responses. Many of those APIs are dead now; StableToolBench (https://github.com/zhichengg/StableToolBench) uses simulated responses. | Mission 3 volume source. Returns are noisy and heterogeneous, so a hard tool-return-prediction test. |
| Toucan-1.5M | `Agent-Ark/Toucan-1.5M` https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M ; paper https://arxiv.org/abs/2510.01179 | 1,646,546 rows (Kimi-K2 519k, OSS 457k, Qwen3 552k, SFT 119k); 495 real MCP servers, 2,000+ tools; 21.8 GB | Apache-2.0 | Yes | Yes, real MCP executions (collected June to Sept 2025, so tool outputs are time-stamped) | MCP-proxy connector (mission 6) fixtures and tool-return prediction. Too big to pull whole; stream one shard or use HTTP Range. |
| Nemotron-Agentic-v1 | `nvidia/Nemotron-Agentic-v1` https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1 | 335,122: `interactive_agent` 19,028 (449 MB), `tool_calling` 316,094 (5.3 GB). JSONL | CC-BY-4.0 | Yes (HEAD 200; I read one record by Range request) | Yes, schema is messages with roles system/user/assistant(tool_calls, reasoning_content)/tool(tool_call_id, content). Tool returns are LLM-SIMULATED. | Plumbing and format reference. HF dataset viewer is broken (schema cast error), so load with raw JSONL. Fully synthetic, LLM-judged. |
| ToolACE | `Team-ACE/ToolACE` https://huggingface.co/datasets/Team-ACE/ToolACE ; https://arxiv.org/abs/2409.00920 | 11,300 rows, 37 MB, 26,507-API pool, 2 to 12 turns | Apache-2.0 | Yes | Yes (`tool` turns with JSON). Whether returns are simulated: card does not say (unverified). | Small single-file function-call corpus. Quick to parse. |
| APIGen-MT-5k | `Salesforce/APIGen-MT-5k` https://huggingface.co/datasets/Salesforce/APIGen-MT-5k | 5,000 multi-turn trajectories, 138 MB | CC-BY-NC-4.0 | Yes (API gated=False, file HTTP 200; the card text still shows gate-form fields) | tau-bench-style retail and airline | Non-commercial, so research only. |
| AgentInstruct (AgentTuning) | `zai-org/AgentInstruct` (was THUDM) https://huggingface.co/datasets/zai-org/AgentInstruct | 1,866 trajectories: alfworld 336, webshop 351, mind2web 122, kg 324, os 195, db 538 | not stated | Yes | Environment observations appear as human-role turns | Too small and off-domain. `db` (SQL) is the only near-fit. Skip. |
| xlam-function-calling-60k | `Salesforce/xlam-function-calling-60k` | 60k | CC-BY-4.0 | NO. gated auto, HTTP 401 | single-turn, no returns | Skip (login). |
| glaive-function-calling-v2 | https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2 | about 1.7 GB used storage | Apache-2.0 | Yes | simulated | Optional. Folded into Nemotron-Agentic. |

## Reusable components

| name | url | license | why |
|---|---|---|---|
| DoDataThings v2 CSV | https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2 | MIT | 68k raw bank descriptors; the teacher re-labels them to build the categorize_transaction demo set |
| foliome data generator | https://github.com/wnstnb/foliome | Apache-2.0 | generate unseen-merchant or higher-noise descriptor splits |
| mteb/banking77 | https://huggingface.co/datasets/mteb/banking77 | MIT tag (orig CC-BY-4.0) | 77-way parquet/jsonl, loads without scripts |
| clinc/clinc_oos | https://huggingface.co/datasets/clinc/clinc_oos | CC-BY-3.0 | 151 labels incl. `oos` abstain class, native parquet |
| MASSIVE tarball | https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz | CC-BY-4.0 | 39.5 MB, intent plus slots, avoids script loader |
| SetFit/TREC-QC | https://huggingface.co/datasets/SetFit/TREC-QC | unstated | tiny 6/50-label smoke set |
| tau2-bench | https://github.com/sierra-research/tau2-bench | MIT | deterministic bank-domain tools and DBs, no LLM needed for the tool layer |
| tau-bench historical trajectories | https://github.com/sierra-research/tau-bench/tree/main/historical_trajectories | MIT | 4 free JSON logs (4 to 27 MB) of GPT-4o and Sonnet 3.5 tool calls and returns |
| AppWorld | https://github.com/StonyBrookNLP/appworld | Apache-2.0 (+bundle terms) | local executable environment with 457 APIs |
| BFCL | https://github.com/ShishirPatil/gorilla | Apache-2.0 | function-selection and irrelevance labels plus Python backends |
| Toucan-1.5M | https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M | Apache-2.0 | real MCP tool executions; fixtures for the MCP proxy |
| Nemotron-Agentic-v1 | https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1 | CC-BY-4.0 | OpenAI-style tool_calls/tool messages at scale (format reference) |
| ToolBench / StableToolBench | https://github.com/OpenBMB/ToolBench ; https://github.com/zhichengg/StableToolBench | Apache-2.0 | recorded real API responses; StableToolBench for a stable local server |
| ToolACE | https://huggingface.co/datasets/Team-ACE/ToolACE | Apache-2.0 | 11.3k, single 37 MB file |

## Numbers

| Item | Value | Date / source |
|---|---|---|
| HF dataset downloads | free, no login for all recommended sets | 2026-10-17, curl HEAD |
| DoDataThings v2 | 68,000 rows, 3.47 MB | https://huggingface.co/api/datasets/DoDataThings/us-bank-transaction-categories-v2/tree/main |
| mteb/banking77 | 13,069 rows (9,993 + 3,076) | https://huggingface.co/api/datasets/mteb/banking77 |
| CLINC150 plus | 15,250 / 3,100 / 5,500 | https://huggingface.co/api/datasets/clinc/clinc_oos |
| MASSIVE en-US | 11,514 / 2,033 / 2,974; tarball 39.5 MB | HF card; S3 Content-Length |
| TREC-QC | 5,452 train, 50 fine / 6 coarse | read from `train.jsonl` |
| tau2-bench repo | 461 MB repo size; MIT; v1.0.1 | https://api.github.com/repos/sierra-research/tau2-bench |
| tau2 banking_knowledge | db.json 270 KB; tasks.json 1.1 MB; 97 tasks | GitHub trees API |
| tau-bench trajectories | 4.1 / 10.8 / 10.9 / 26.5 MB | GitHub contents API |
| Nemotron-Agentic | interactive 449 MB, tool_calling 5.34 GB | `X-Linked-Size` header |
| Toucan-1.5M | 21.8 GB data (43 GB with HF conversions) | HF card; HF API `usedStorage` |
| ToolBench HF mirror | 541 MB; 188,304 rows | datasets-server `/size` |
| ToolACE | 37 MB, 11,300 rows | datasets-server `/size` |
| Dollar cost of acquiring data | $0 (disk only; Nemotron/Toucan can be sampled with HTTP Range) | this session |
| Dollar cost of teacher re-labeling | not computed here (token prices are outside this topic) | n/a |

## Blockers / plan changes

1. No public real raw-descriptor dataset. Certification on DoDataThings (template-generated, 500+ merchants) will look too good. Mitigations: (a) hold out merchants, not rows, for the cert split; (b) mix in noisier karthiksagarn-style strings (license caveat); (c) add a CLINC/Banking77 run where human-written text gives realistic error rates; (d) label with the teacher, not the dataset column.
2. Label-space mismatch: DoDataThings has 17 labels, the demo plans about 12. Merge classes (e.g. Mortgage+Rent -> Housing, Subscription+Entertainment) or define the demo taxonomy as 17. Decide before generating teacher labels.
3. mitulshah (4.5M) and xlam-60k need an HF login. Do not depend on them.
4. `datasets` v4+ cannot run loading scripts. Pin to the parquet/jsonl mirrors listed above. Downloaded files are untrusted; parse with plain `json`/`pandas`/`pyarrow`.
5. For mission 3 (tool-return predictor), simulated-return corpora (Nemotron, ToolACE, APIGen-MT) overstate predictability because an LLM wrote the returns. Use tau2-bench, tau-bench trajectories, AppWorld, BFCL backends, Toucan (real) for the real test.
6. tau2-bench can emit (tool call, tool return) pairs with no LLM spend if we replay each task's gold `evaluation_criteria.actions` against the environment. This is my inference from the docs; I did not run it. The full agent/user-simulator loop needs an LLM and Python >=3.12, <3.14. Check the local Python version (miniconda is present; version not checked).
7. AppWorld licensing: derived data must stay encrypted if redistributed, and test splits must not be used for tuning. Keep it out of any committed demo fixtures.
8. Gaps with no dataset found: match_invoice (document-to-record matching), draft_note (free text), reviewer-edit traces for mission 4, latency data for mission 3. Synthesize them, or run the tau2 environment to log timings (measures our own Python tools, not real API latency).
9. The tau2 `banking_knowledge` tool set is the closest public analog to `vendor_lookup`; consider adopting its data model (users, accounts, transactions, disputes) as the demo's mock DB to save build time.

## Unverified

- Whether any public dataset of REAL labeled bank descriptors exists on Kaggle or elsewhere. I only searched HF (by keyword: transaction categor / merchant / bank statement) and general web. I did not log in to Kaggle.
- `scripts/generate-training-data.js` in foliome being the exact generator of the v2 CSV (the v2 card links the repo; I did not read the script).
- AppWorld per-split task counts and download size (README gives neither; the paper abstract says 750 tasks, 9 apps, 457 APIs).
- Whether ToolACE tool returns are LLM-simulated (the card does not say).
- BFCL: whether the data files include any tool returns; I only inferred that execution happens in repo Python backends from the changelog.
- tau2-bench telecom `tasks_full.json` task count, and the cost to run a full tau2 evaluation.
- Licenses: `SetFit/TREC-QC`, `karthiksagarn/...`, `Dikshith4500/...`, `zai-org/AgentInstruct`, `greggles/mcc-codes` not stated or not checked. API-Bank MIT is from the HF tag only.
- MASSIVE per-language count: the card text says 19,521; the en-US splits sum to 16,521.
- Toucan finance/banking MCP coverage (card names no domains).
- Whether the HF "convert/parquet" branch for `PolyAI/banking77` exists; the API said invalid rev. I used `mteb/banking77` instead.


## Addendum: verify-0, "no real bank-transaction dataset" claim (checked 2026-10-17)

Verdict: CORRECTED. The claim in line 7 is false as written. The search behind it covered HF only, so it should read "none on HF". MoneyData is a real, anonymized, labeled bank-transaction set with raw (truncated) bank descriptors, downloadable with no login. Do not drop it. Also correct the `Unverified` bullet about "any public dataset of REAL labeled bank descriptors": it exists (below).

- Dataset page: https://data.mendeley.com/datasets/dnxtg6n4rv/1 ("Bank Transactions Dataset", contributor Robert Laramee, Univ. of Nottingham; DOI 10.17632/dnxtg6n4rv.1; Mendeley v1 dated 14 Jan 2026; CC BY 4.0; described as anonymized real retail bank transactions, July 2015 onward, seven years). The page's Files tab is empty (only "Download All"), but the public file URL works.
- No-login download re-verified: `curl -L https://data.mendeley.com/public-files/datasets/dnxtg6n4rv/files/36b3d359-2491-4282-850f-3f03e00fbfe3/file_downloaded` returned HTTP 200, 507,389 bytes, xlsx. No cookies or token.
- CORRECTION to bookkeeping-data.md row counts: Sheet1 ("Anonymized Original with Catego") has 8,315 `<row>` elements, but rows 6,569-8,315 are empty styled rows. Real data = 6,567 transactions (matches the page's "over 6,500", not 8,314). 6,543 have a Category. 908 distinct descriptions (bookkeeping-data.md said 909). 30 distinct category strings, some are typos or one-offs ("Groceries " with trailing space x5, "Food Shopping" x1, "Purchase of uk.eg.org" x1).
- Columns: Transaction Number, Date (dd/mm/yyyy), Type (DEB/BP/FPO/FPI/DD/CPT...), Description, Debit, Credit, Balance, Category, Location City, Location Country (UK in sampled rows, Nottingham). Sheet2 "Description of Categories" has 42 lines of category definitions plus the required citation: Firat, Vytla, Vasudeva Singh, Jiang, Laramee, "MoneyVis: Open Bank Transaction Data for Visualization and Beyond", EuroVis 2023 short papers, https://doi.org/10.2312/evs20231052.
- Skew (matters for certification): top labels Savings 1,249, Amazon 1,195, Groceries 487, Bills 474, Cash 453, Dine Out 448, Others 351. One descriptor, "SAVE THE CHANGE", is 1,165 rows (17.7% of all rows). Top 3 descriptors (SAVE THE CHANGE 1,165; Amazon UK Marketpl 421; AMZNMktplace 275) are 28% of rows. Single person, personal-finance categories (not accounting GL), descriptors truncated to ~18 chars. Use for a held-out merchant-split realism eval, dedupe before computing error rates, and do not train a student on it alone.
- Paper: https://diglib.eg.org/items/e5a24478-3ec5-4e33-bab1-e5b27b06c6bd/full and PDF https://diglib7.eg.org/bitstream/handle/10.2312/evs20231052/109-113.pdf both returned HTTP 403 to WebFetch, so the anonymization method, who assigned categories (manual vs rule), and the source bank are UNVERIFIED. Search snippet only: authors from Cukurova University and Univ. of Nottingham; claims to be the first real retail bank transaction dataset anonymized and published for visualization research. Residual risk: anonymization may have altered some payee strings (person-name payees like "MISS D WU" appear to be pseudonyms); category labels are likely one author's judgment, not gold.
- Still unresolved: whether Kaggle or other sites have more real labeled descriptors (not searched here; Kaggle needs login). Other real-format sets found elsewhere in this folder lack descriptor strings (TabFormer merchant is an int hash).
