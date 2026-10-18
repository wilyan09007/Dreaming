# Sprint 2026-10-18: finish Dreaming and its demo by 10:00 PDT

Status: IN_PROGRESS (started 03:00 PDT). Goal text: the `/goal` of the 03:10 session. Design docs are frozen; build to them.

## Critical path
- [x] `mock-dir/` → `dreaming/`; venv resynced; docs repointed
- [ ] StepFun record pass, 5-wide, seeds 0 and 1 → `dreaming/data/recorded.jsonl` + `embeddings.npz` (running since 03:03, restarted 03:22 with retry passes; ~46 answers/min ≈ 2.5 h per seed)
- [x] Review-and-fix workflow (15 agents, 0 errors): campaign, memory, quickbooks, systemone/campaign routes, Sentry, web rewire, speculation, veri, benchmark, S29 — all merged by 05:05
- [x] Merged; ruff clean, pyright 0, 195 passed / 17 skipped (05:05)
- [x] Campaign app deployed (v7); tier1-1018-rule-free: 31 cells incl. 4 T4 finetune, $0.06; memory seed+real best on categorize (0.90 unseen / 0.83 repeat); coverage@5% = 0 everywhere (calib ~95 rows) until seed 0 lands
- [x] 06:56 seed 0 complete (8,219 answers, $6.88; 42/2000 transactions with one unparsed step); uploaded; tier1-1018-final (45 cells) done 07:20: memory(seed) 89% unseen coverage on categorize, memory 78% repeat coverage on match_invoice; qb-1018b: bill_account head 93-97% at 5%; tier1-1018-final2 (deduped rows) running 07:30
- [x] Deploy #2 05:00: designed page live at https://dreaming-demo.fly.dev (TEACHER=rule for QA); demo tables reset
- [x] Deploy #3 07:31: TEACHER=stepfun (RecordedTeacher), RECORDED_SEEDS=0, recorded.jsonl + embeddings.npz, scores tab, evidence rows, quickbooks in the image; demo-reset 07:33; verification batch of 2,000 done 07:44: 11 min, 12 live StepFun calls, 0% failed, replayable 28%, cost/run 0.42c; replay 3/3 same bytes on the deployed encoder
- [x] Fetch round trip on the deployed stack passes 07:40 (planner quoted the agent's receipt verbatim)
- [ ] Visual QA loop round 1 running (05:04): 4 lenses offline + 1 live; fixers in worktrees
- [x] Benchmark: GET /benchmark(.md) with repeat/unseen columns + accuracy; scores tab merged 05:29 (needs deploy #3); README/Devpost evidence later

## Side quests (parallel, scrap if they run long)
- [x] Devin on `hdc` (PR #4 merged 04:19; 9 min, byte-stable, count-normalized score): session https://app.devin.ai/sessions/a713adfe0fc64befb33e9b2c0f64439f started 04:07 from aidenhongg/dreaming `sprint-1018`; PR expected against that branch
- [x] CREAO agent 'Dreaming benchmark runner' ca9c3ce9-78d9-4bc6-ad8f-5ec1d3275991 built in the UI, ran via UI and API (05:20); human: add DREAMING_API_KEY to CREAO Secrets (scratchpad/creao/dreaming_api_key.env; also set as a Fly secret), re-run, share the run thread
- [x] Paper: dreams tab + 3 variants built in Paper by a sibling session (Desktop\paper-variants, file 'Unique yonder'); user closed it 03:40 — claim 'designed in Paper', no more MCP calls
- [ ] Speculation / DAG parallelism behind `speculate=False`
- [x] veri.studio fine-tune cell: L4 job ran, artifact downloaded + sha matched (merged 04:22)

## Evidence landed
- Fetch.ai: ASI:One -> Agentverse -> agent round trip passes (branch fetch-roundtrip merged 03:55; needs the next Fly deploy)
- S29 (RunPod 4090 15 min $0.23 + Modal CPU): seeded Hopfield ties LR on certified coverage at <=77 labels (k=3/10), loses at 150; use 1/count vote weighting; 0.3 ms/query at 1k rows, 17 ms at 100k; byte-stable across processes
- Sentry: Tracing + Logs + gen_ai spans (merged 04:12)
- Add-on in progress (05:31): memory evidence rows (top-3 voting rows) in the trace card
