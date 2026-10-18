# HANDOFF — Dreaming feasibility work (paused on the local machine 2026-10-17 12:40 PDT)

This repo is being handed from a local Claude Code session (Windows laptop) to a remote session. Everything below is what a fresh session needs to continue without re-deriving anything.

## 1. What this is
`missions.md` is the product spec (a runtime that logs an agent's calls, trains per-call-site predictors, certifies them against an error budget, and answers from predictions). `smokes/` holds the feasibility study: research reports, 20+ smoke experiments, a cost ledger, and the roll-up.

**Start by reading `smokes/FINDINGS.md`** (headline answers, per-mission verdicts, plan changes, round-2 and round-3 results, open items). Then `smokes/research/*.md` as needed. Each smoke folder has `RESULTS.md` (if finished), `run.py`, `log.txt`, caches.

## 2. State of every smoke
| id | topic | state |
|---|---|---|
| S1–S13, S2b, S6b | original missions (students, certificate, deferral, tools, router, pruner, determinism, GPU ladder, rollout, connectors, ASI:One, wrappers, teacher noise) | **done + critiqued** |
| S14 real tool traces, S15 confidence signals, S18 audit certificate, S19 token economics | round 2 | **done** |
| S17 warm start (incl. many-task hypernetwork) | round 3 | **done** (RESULTS.md) |
| S22 decoder adapters, S23 PAW/stream probes/TSDAE, S24 memory/Bayesian/symbolic students | round 3 | **partial run, RESULTS.md written + critiqued** from `salvaged_results.json` (remote session) |
| S16 / S16b v2 / S16c draft_note | rounds 2–4 | **done + critiqued**: strict floor 12%; core-facts equivalence floor 2.2%; pruning 60–70% of tokens at the floor; certificate needs 100–200 cal rows (RESULTS_B2.md, S16c RESULTS.md) |
| S25 exotic families, S26 similar-task pool transfer | round 4 | **done + critiqued** (RESULTS.md in each folder; no exotic family or sibling-task transfer beats encoder + LR; LLM-written examples remain the warm start) |
| S14b Copilot traces, S27 hybrid embedder + memory heads, S28 router calibration sweep | round 4 | **done + critiqued** (RESULTS.md; see FINDINGS round-4 notes 1, 3, 4) |
| S21 hidden-state probe | round 4 | **done + critiqued** (RESULTS.md; probe not worth a GPU pass vs student==cheap) |
| S20 TabICL head | round 2 | scripts only; low priority (scout: TabPFN loses to LR at few shots) |

## 3. Cloud resources (as of 2026-10-17 23:40 UTC)
- **No Dreaming pods are running.** RunPod balance ≈ $12.90. The user's own pods (`promo-*`) are not ours; do not touch them.
- **OpenRouter prepaid credits are exhausted** ($0.03 of $495 left; the key itself has ~$31 of its $50 limit unused). Any API-bound smoke needs a top-up first. ASI:One and Agentverse keys work.
- The cloud session (claude.ai/code) that ran round 4 is stopped; the laptop session owns the branch; copy `.env` into any new worktree.
- `smokes/pod.py` now has `--pull` (fetch partial results on every poll) and `--attach`; a re-attach from a second machine terminates the pod when it fetches, so never run two launchers against one pod.

## 4. How compute works here
- **Laptop must stay idle** (user request). All compute goes to RunPod pods through `smokes/pod.py` (read its docstring): ships a tar of the listed files over the pod's HTTP proxy, runs the command from the project root on the pod, streams `log.txt`, brings the bundled folders back, self-terminates the pod, appends cost to `smokes/COSTS.md`. Community 4090 $0.34/h (secure $0.89), A5000 $0.27, A100 80GB $1.19, H100 PCIe ~$2. Boot ≈ 10 min. `--attach <podId>` reuses an already-booted pod.
- GPU training script for the student ladder: `smokes/s8_gpu_student/train_student.py` (+ `launch_runpod.py`, the older single-file launcher).
- Keys: `.env` (git-ignored; copy from the user) — OPENROUTERKEY (OpenRouter; ~$42 of the $50 key limit left), RUNPODKEY (balance ≈ $17), ASI1KEY (ASI:One, OpenAI-compatible at https://api.asi1.ai/v1, model asi1-mini), AGENTVERSEKEY. AWS CLI exists but has no GPU quota and the IAM user cannot write S3; EC2 CPU quota is 256 vCPU.
- Spend so far ≈ $12 OpenRouter + ≈ $3 RunPod + $0.05 ASI:One. Hard cap $100.

## 5. Rules the user set (keep them)
- Balance information against cost; reuse before building; no Fable fan-out, few Opus agents (Sonnet builders + one critic worked well).
- A first failure is not a verdict: research the latest methods and try orthogonal ideas before concluding infeasible. Keep checking 2025–26 literature.
- If data is synthesized, use track-sponsor services where possible (ASI:One for LLM calls).
- Hold out by group (whole datasets/workflows, merchants, episodes), never by row.
- Flag anything needing a human sign-in instead of blocking.

## 6. Workflow specs
`smokes/_workflows/*.js` are the exact multi-agent workflow scripts used (build → critique → fix pipelines). The spec text for every smoke is inside them; the `dreaming-finish-s25-s26-s16b-*.js` script is the one that was running at handoff. They reference local paths (`C:\Users\<user>\...`); adjust `ROOT`/`SCRIPTS` if re-used.

## 7. Suggested next steps (in order)
0. **Decision: train every candidate student per site in parallel (memory, linear head, HDC bundle, fine-tuned encoder where a GPU exists) and promote the one with the best certificate; memory is the day-zero tier.** Certify at delta/N or on a separate slice from the one used to pick. S29 (memory de-risk at scale) is written in `smokes/s29_memory_student/` but was cancelled unrun; optional.
1. Top up OpenRouter credits if any API-bound smoke is wanted.
2. MCPMark trajectory logs (HF): the repeat-rate measurement on real servers with repeated runs that S14/S14b cannot give ($0 compute).
3. Second E1 draw on the S28 routed rows (~$0.3): is a 5% router budget measurable at all given E1's 6% self-disagreement?
4. Hybrid smokes A (Qwen3.5-0.8B in-context k-shot) and C (batch-invariance of Gated DeltaNet / Mamba2 hybrids), < $1 of pod each; S20 TabICL head, low priority.
5. Mission-4 deferral head with features the logits cannot contain, evaluated across a mid-stream reviewer change (CPU).
6. Keep `smokes/FINDINGS.md` and `smokes/COSTS.md` current; commit after each smoke.

## 8. Memory notes carried over
- The hypernetwork question is settled: all designs (3 home-made, 800-task version, public PAW) lose to a 33M encoder + linear head; LLM-written examples from the schema are the day-zero warm start (80% certified coverage at 1 row/label); a Hopfield memory seeded with those examples is a credible no-training, byte-replayable alternative.
- Teacher noise bounds every certificate (pin providers; budgets below ~3–11% depending on teacher are unmeasurable).
- The certificate must use a fixed coverage grid (`common.certify`), not a binary search.

---

# HANDOFF 2 — mock-dir build order (local session paused 2026-10-17 23:15 PDT)

The design docs are frozen and the sign-ins are all cleared; what remains is the build order in `mock-dir/DESIGN.md` §12 and the
clock in `mock-dir/docs/training-plan.md` §7. Hackathon deadline: **2026-10-18 10:00 PDT** (a little over is tolerated).

## 1. Read first, in this order
`mock-dir/docs/README.md` (index, decided/open, sign-in status: everything done; Devin's repo connection is deliberately NOT done),
`mock-dir/DESIGN.md` §4-§5 and §12, `mock-dir/docs/training-plan.md` §3-§4 (the campaign and the monitor), §7 (the clock), §8-§9,
`mock-dir/TREE.md`. `mock-dir/smokes/RESULTS.md` has tonight's measured rows.

## 2. State of the code (all committed on this branch)
- **Done and reviewed:** the candidate slot (`predictors/candidates.py`), the provider table (`teachers/openai_compat.py`, StepFun pinned
  with `floor=0.035`, 1024-token budget, tolerant label parser), the campaign contracts + migration 0002 (applied to the live Supabase
  database 22:45 PDT), the trainer app redeployed on Modal, and **the candidate loop in `learning.py`** (one job per
  candidate, spares shadow into `TraceRow.shadow`, each certified at δ/N, best coverage promoted, handover on revocation; test
  `packages/dreaming/tests/test_candidates.py`).
- **Drafts, UNREVIEWED:** five sonnet builders were stopped mid-way; their files are in the tree because ruff
  and pyright are clean and 140 of 141 tests pass. They are: `predictors/memory.py` and `predictors/finetune.py` (+ tests),
  `apps/trainer/.../campaign.py` and `monitor.py` bodies (+ `apps/trainer/tests/test_campaign.py`), the `agents/quickbooks` package
  (`data.py`, `workflow.py`, `baselines.py`, tests), `apps/api/.../systemone.py` and `campaign_routes.py`, the Sentry sink in
  `telemetry.py` and `wiring.py`, deps (`quickbooks` workspace member, `sentry-sdk`). **The critic pass never ran.** Review each
  against the specs it was written to (quoted in the workflow script
  `dreaming-build-order-wf_7b399f44-9ed.js`, which is not in this repo; the gist is in `docs/training-plan.md` §3 "Interfaces pinned for the build").
- **The one failing test:** `agents/bookkeeper/tests/test_agent.py::test_a_chat_line_is_acked_relayed_to_the_api_and_answered_in_words`
  (it expects only `predicted` decisions; with `memory` as a second default candidate the loop now logs `called` rows too). Decide
  whether the test or `DEFAULT[Kind.model]` is right; do not paper over it.
- **Data:** `mock-dir/data/recorded.jsonl` holds 232 StepFun answers for ledger seed 0 (a resumable partial file).

## 3. Measured tonight (do not re-measure)
- StepFun `step-5-preview`: self-disagreement floor 3.5%; 10/10 labels parse at `max_tokens=1024`; ≈ 47 calls/min at 5-way
  concurrency; **sequential record is ≈ 14 answers/min with a long tail (p50 2.9 s, p90 10 s, max 17.5 s)**, so `just record` must
  run the typed sites 5-wide or a seed takes ≈ 7 h. Balance $25 voucher.
- Modal: card on file, a T4 runs (`Tesla T4, 15360 MiB`), trainer app deployed, migration 0002 applied (tables: train_runs,
  checkpoints, eval_sets, memory_rows). OpenRouter ≈ $20 credit. veri $20 credit, CLI logged in. Sentry project `dreaming-api`
  (issue DREAMING-API-1 proves ingest). Browser Use key expires Oct 24. Devin personal key works (0 sessions).

## 4. Next steps, in order (each is one commit)
1. Review the WIP drafts (one critic pass; fix or delete), make the failing test honest, `just check` green.
2. `just deploy-campaign`; smoke `just campaign night-1017 bookkeeper.categorize 3`; read `just monitor night-1017`; fix; then
   `just campaign night-1017` (tier 1: 24 CPU + 6 exact + 4 T4 cells; the `seed` variant is not launched, nothing writes seed.jsonl).
3. Make `bookkeeper.record` 5-wide, finish seed 0 (`--seeds 0`), re-prepare the bookkeeper sites with StepFun labels, relaunch
   their cells; flip `TEACHER=stepfun` in `infra/fly.toml`; `just deploy-api` (Sentry sink goes live with `SENTRY_DSN`).
4. QuickBooks: run `quickbooks.data.rows` on the real download inside `prepare_site`; report label counts; the three sites' cells.
5. Sponsor evidence: the veri cell (`experiments/veri_finetune/` was never written; spec in `docs/devtools.md` §veri and
   `training-plan.md` §3 last paragraph), the CREAO agent against `/campaign/*`, Sentry's caught-log, Paper frames. Devin: do not
   connect the repo until the user says so.
6. 08:00–09:30: the tables in `smokes/RESULTS.md`; then the video, Devpost, README blocks.

## 5. Environment the remote session needs
Secrets are not in git. Set these in the cloud environment (values are in the laptop's `.env`): OPENROUTERKEY, RUNPODKEY,
AGENTVERSEKEY, ASI1KEY, SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY, DATABASE_URL, MODAL_TOKEN_ID, MODAL_TOKEN_SECRET, FLY_API_TOKEN,
AGENT_SEED, LOGFIRE_TOKEN, GEMINIKEY, CREAOKEY, STEPFUNKEY, DEVINKEY, SENTRY_DSN, VERIKEY, BROWSERUSEKEY. The cloud "Default"
network policy denied RunPod/OpenRouter/HF earlier today; widen it for modal.com, api.stepfun.ai, openrouter.ai, huggingface.co,
*.supabase.co, sentry.io, api.veri.studio. Never `source .env` (FLY_API_TOKEN contains a comma). Windows-only gotchas do not apply there.

## 6. Rules the user set tonight
Design docs are frozen; build in the order above; commit after each phase; keep code minimal, no god modules; sonnet for fan-out,
one opus critic at most, never fan out Fable; no Devin repo connection yet; revoke every key after Oct 24 (`docs/README.md`).
