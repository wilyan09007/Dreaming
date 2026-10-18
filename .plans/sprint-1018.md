# Sprint 2026-10-18: finish Dreaming and its demo by 10:00 PDT

Status: IN_PROGRESS (started 03:00 PDT). Goal text: the `/goal` of the 03:10 session. Design docs are frozen; build to them.

## Critical path
- [x] `mock-dir/` → `dreaming/`; venv resynced; docs repointed
- [ ] StepFun record pass, 5-wide, seeds 0 and 1 → `dreaming/data/recorded.jsonl` + `embeddings.npz` (started 03:03; ~2.5 h per seed)
- [ ] Review-and-fix workflow over the unreviewed drafts (campaign, memory, quickbooks, systemone/campaign routes, Sentry) + new work (web rewire, speculation, veri cell, benchmark report, S29)
- [ ] Merge branches, `uv run ruff/pyright/pytest` green, commit
- [ ] Deploy campaign app; smoke 3 cells; launch tier 1 (QuickBooks first, bookkeeper on rule labels; re-prepare on StepFun labels when seed 0 lands)
- [ ] `TEACHER=stepfun` in fly.toml; deploy API with the designed page; replay check on the deployed stack
- [ ] Visual QA loop on the deployed page against `demo/VISUAL_INTENT.md` until nothing new surfaces
- [ ] Benchmark table from the campaign; README/Devpost evidence (user-facing text is the user's call)

## Side quests (parallel, scrap if they run long)
- [ ] Devin on `hdc` (needs the GitHub app connected by the repo owner)
- [ ] CREAO benchmark-runner agent (after `/campaign/*` deploys)
- [ ] Paper frames (needs Paper Desktop installed by a human)
- [ ] Speculation / DAG parallelism behind `speculate=False`
- [ ] veri.studio fine-tune cell
