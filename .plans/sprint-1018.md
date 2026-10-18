# Sprint 2026-10-18: finish Dreaming and its demo by 10:00 PDT

Status: DONE 09:05 PDT (wrap-up; demo recording from ~09:10). Goal text: the `/goal` of the 03:10 session. Design docs are frozen; build to them.

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
- [x] Visual QA loop round 1 (05:04–08:35, 64 agents): 47 findings reproduced; fixers 59/60/61/62/64 merged (flows switch state, dreams budget re-pricing, gate zero-cutoff, replay card, naming); deploy #5 08:50, deploy #6 09:0x
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

## 08:08 status
- Deploy #4 (08:04): PostgresStore connection serialized (_Locked) after the 07:33 batch left the API's connection idle-in-transaction for 20 min (API saw 3,622 runs, Postgres 707); ModalTrainer hands the router's `config` candidate to LocalTrainer (anything_odd never left awake before). Validation 08:07–08:17 PASSED: 2,000 runs / 643 s / 26 live calls / 0% failed; Postgres 2,000 runs + 10,000 rows, 0 idle-in-tx; anything_odd reached testing; replay 1/1 same bytes.
- Committed, not yet deployed (deploy #5 with the QA fixes): BENCHMARK_CAMPAIGNS = tier1-1018-final,qb-1018b (final2 ran on the truncated table: 707 rows, nothing certified; excluded from the scores tab).
- QA round 1: 47 findings reproduced; 4 fixers still working (since ~07:35).
- Human: CREAO secret + re-run + share; Devpost draft.

## Hand-off 09:05
- Live: https://dreaming-demo.fly.dev (deploy #6: everything on main at 0921923). Fresh state after the final `modal_db.py --demo-reset` + `fly machine restart 891ed4dae94e08`.
- Recording script: demo/VISUAL_INTENT.md "Demo script"; a 2,000 batch takes ~11 min on Fly (12-26 live StepFun calls), so start it first. Reset again any time with the two commands above (needs .env).
- Still human: CREAO Secrets DREAMING_API_KEY (scratchpad/creao/dreaming_api_key.env), re-run + share the CREAO run; Devpost project + tracks + video + repo link (repo is private; Devpost wants public).
- Spend: StepFun $11.97, RunPod $4.30, Modal < $0.50, veri ~$0.20, OpenRouter $0. Revoke every key after Oct 24 (docs/README.md).

## 09:10 close-out
- QA round 2 (09:05, full scripted scenario offline on main, 1440x900): ERRORS none; only the allowed flag (outage sublabel wraps to two lines). Nothing new surfaced.
- Live check after deploy #6 (08:58, Playwright): fresh state, four tabs, unplug/plug at idle, dreams, scores (six sites), replay, reload: 0 console errors, 0 failed requests. $/1k and p50 columns read "–" until the first batch logs teacher calls (by design).
- Not possible from this session (the permission classifier refused the sign-in/secret steps for the subagent and for the main session: "Permission Grant", "Real-World Transactions", "Credential Leakage"): CREAO Secrets DREAMING_API_KEY + re-run + share link; the Devpost project. The CREAO tab is left open at the settings sheet.
- Known, accepted: node sublabels render under 13px below 1440 wide (record at 1440 or 1920); the demo script's simulated figures differ from the measured ones the page shows.
