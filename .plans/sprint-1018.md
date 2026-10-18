# Sprint 2026-10-18: finish Dreaming and its demo by 10:00 PDT

Status: IN_PROGRESS (started 03:00 PDT). Goal text: the `/goal` of the 03:10 session. Design docs are frozen; build to them.

## Critical path
- [x] `mock-dir/` → `dreaming/`; venv resynced; docs repointed
- [ ] StepFun record pass, 5-wide, seeds 0 and 1 → `dreaming/data/recorded.jsonl` + `embeddings.npz` (running since 03:03, restarted 03:22 with retry passes; ~46 answers/min ≈ 2.5 h per seed)
- [ ] Review-and-fix workflow over the unreviewed drafts (campaign, memory, quickbooks, systemone/campaign routes, Sentry) + new work (web rewire, speculation, veri cell, benchmark report, S29)
- [ ] Merge branches, `uv run ruff/pyright/pytest` green, commit
- [ ] Deploy campaign app; smoke 3 cells; launch tier 1 (QuickBooks first, bookkeeper on rule labels; re-prepare on StepFun labels when seed 0 lands)
- [ ] `TEACHER=stepfun` in fly.toml; deploy API with the designed page; replay check on the deployed stack
- [ ] Visual QA loop on the deployed page against `demo/VISUAL_INTENT.md` until nothing new surfaces
- [ ] Benchmark table from the campaign; README/Devpost evidence (user-facing text is the user's call)

## Side quests (parallel, scrap if they run long)
- [ ] Devin on `hdc`: session https://app.devin.ai/sessions/a713adfe0fc64befb33e9b2c0f64439f started 04:07 from aidenhongg/dreaming `sprint-1018`; PR expected against that branch
- [ ] CREAO benchmark-runner agent (after `/campaign/*` deploys)
- [x] Paper: dreams tab + 3 variants built in Paper by a sibling session (Desktop\paper-variants, file 'Unique yonder'); user closed it 03:40 — claim 'designed in Paper', no more MCP calls
- [ ] Speculation / DAG parallelism behind `speculate=False`
- [ ] veri.studio fine-tune cell
