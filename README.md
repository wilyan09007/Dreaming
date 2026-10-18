# Dreaming

![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
![tag:hackathon](https://img.shields.io/badge/hackathon-5F43F1)

LA AI Hackathon 2026. Dreaming is a runtime that learns a model of every call an agent makes, then stops making the calls it can predict.
It logs each call (a frontier-model step, a tool, a routing choice), trains predictors per call site on those rows, certifies each one
against an error budget the operator sets, and answers from the best certified one. The real call stays as the fallback and as a
1-in-20 spot check. Every saving comes with a certified error bound, and every decision a trained predictor answered can be replayed
byte for byte. Product spec: `missions.md`. System design: `dreaming/DESIGN.md`. Repo: https://github.com/aidenhongg/dreaming

## Live demo

https://dreaming-demo.fly.dev (Fly.io; the API serves the page). The demo agent is a bookkeeper: categorize a bank-feed line, look up the
vendor, match an invoice, flag anything odd, post to the books or wake a human. Four tabs (spec: `demo/VISUAL_INTENT.md`):

- **flows**: run 2,000 transactions and watch each step move from the real call to a dream; cost per run, seconds per run, failed runs and replayable decisions are charted. "Unplug the API" takes the vendor tool down.
- **dreams**: the dream journal and the error-budget slider. Moving the slider re-certifies every step; each step shows its promise (wrong at most N%), the evidence and the serving candidate.
- **replay**: pick a past run and re-run it. Steps a hosted dream answered come back with the same bytes; the rest are read from the log.
- **scores**: the benchmark table, read from `GET /benchmark` (`/benchmark.md` for the Markdown form).

## Fetch.ai agent

![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
![tag:hackathon](https://img.shields.io/badge/hackathon-5F43F1)

| | |
|---|---|
| Agent name (Agentverse) | `dreaming-bookkeeper` |
| Agent address | `agent1q2eaums3hgpp2d68npndfnfdnkyskjt2kgz5ajpvdpdht5mk4lgvx9n6uuw` |
| Protocol | Agent Chat Protocol (`AgentChatProtocol` 0.3.0), Agentverse mailbox, Almanac testnet |
| Code | `dreaming/agents/bookkeeper/src/bookkeeper/agent.py`, `dreaming/packages/dreaming/src/dreaming/connectors/uagents.py` |

Send it one bank-feed line from ASI:One (for example `NOTION *WEB 96.00`). The agent relays it to the Dreaming API
(`POST /run`), which runs the bookkeeping workflow (categorize, vendor lookup, invoice match, anything odd, post to books)
and answers with a receipt: what was posted, and for each step whether the real call or a trained model ("a dream")
answered, with that model's weights hash and certificate.

## What is in the box

All under `dreaming/`, a uv workspace (layout in `dreaming/TREE.md`).

| Part | What it is | Where |
|---|---|---|
| Runtime | `@step` / `@tool` / `@effect` call sites; one trace row per call; a router that decides called, predicted, audited or fallback; a state machine awake, watching, testing, dreaming | `packages/dreaming/src/dreaming/` (`registry.py`, `router.py`, `runtime.py`, `learning.py`) |
| Candidates | trained per site in parallel and kept in shadow; listed below | `packages/dreaming/src/dreaming/predictors/` |
| Certificate | fixed-grid Clopper-Pearson bound (delta 0.05) over shadow rows the weights never trained on; names the candidate, weights, environment and teacher hashes; revoked when the 1-in-20 audits exceed the budget | `gate.py`, `learning.py` |
| Replay | loads the exact weights a row names and re-runs the forward pass with caches bypassed | `replay.py` |
| Campaign | a Modal app that runs a matrix of cells (site, candidate, seed, rows seen) on CPU or a T4, with a monitor cron | `apps/trainer/src/dreaming_trainer/campaign.py` |
| API and page | FastAPI on Fly: control routes, `/v1/systemone`, an OpenAI-compatible proxy, `/benchmark`; the static page | `apps/api`, `apps/web` |
| Agents | `bookkeeper` (the demo and the uAgent), `quickbooks` (the benchmark workflow) | `agents/` |
| Store | Postgres on Supabase; Realtime feeds the page | `db/migrations/` |

The candidates, in plain words (`training-plan.md` section 1). The certificate picks the serving one per site; the others stay as spares.

- `memory`: a softmax-weighted vote over stored (embedding, label) rows, seeded with LLM-written examples. Learning is appending a row, forgetting is deleting one, and every answer can show the rows that voted. `predictors/memory.py`
- `head`: a frozen bge-small encoder (33M parameters, 384-d) with a linear head fit per site in seconds. `predictors/student.py`
- `finetune`: the same encoder unfrozen on the site's rows; needs a GPU. `predictors/finetune.py`
- `hdc`: one integer bundle per class over 10,000-bit codes; after the encoder it is integer arithmetic, so the same embedding gives the same bytes anywhere. `predictors/hdc.py`
- `exact`: tool memory (the same input gave the same return), no model. `config`: the router's cheaper-configuration choice. `predictors/candidates.py`

## Numbers we measured tonight

Provisional: the final run with the full seed-0 StepFun labels was still pending when this was written (`.plans/sprint-1018.md`).
`GET /benchmark` is the current table and supersedes anything here.

- **Campaign `tier1-1018-rule-free`**: 31 bookkeeper cells, 4 of them T4 fine-tunes, cost $0.06 (`.plans/sprint-1018.md`).
- **`categorize`, seeded memory against the linear head** (same campaign, seed rows plus real rows): the memory reached 0.90 accuracy on unseen merchants and 0.83 on repeat traffic; the head reached 0.35-0.58 and 0.67-0.70. Certified coverage at 5% was 0 for every candidate in that campaign (about 95 calibration rows), so these are accuracies, not certificates. Source: the campaign monitor table, summarised in `.plans/sprint-1018.md`.
- **QuickBooks `bill_account`, held out by company** (campaign `qb-1018b`, 06:30): the linear head certifies 93-97% coverage at a 5% budget with 1-3% realized error. The live `GET /benchmark.md` at 06:48 shows 94.3% coverage and 1.8% error for it (3 seeds), marked as read from the checkpoints' own summaries.
- **S29, the memory's scale check** (RunPod RTX 4090, $0.23, plus Modal CPU; `smokes/s29_memory_student/RESULTS.md`, "In plain words"): the seeded memory ties logistic regression on certified coverage at 77 labels or fewer with 3 or 10 rows per label, and loses at 150 labels (0.28-0.33 less at 10 rows per label). A query costs 0.31 ms at 1k stored rows and 17.1 ms at 100k. The read-out was byte-identical across three fresh processes. Verdict: a day-zero tier and a hot spare, not the winner at many labels.
- **StepFun self-disagreement**: 3.5% on 200 identical calls (Other 193, Office supplies 2, 5 unparsed; `smokes/RESULTS.md`). It floors every certificate, and the API refuses a budget below it.
- **veri.studio before and after** (below): held-out accuracy 26.0% to 37.9%, neither certified.
- **Tests**: 195 passed, 17 skipped at 05:05 (`.plans/sprint-1018.md`).

## Run it

```sh
cd dreaming
uv sync --all-packages --extra student
# the API offline: memory store, rule teacher (labelled "rule" in every row), hashed-trigram encoder, local training, no accounts
DATABASE_URL= TEACHER=rule ENCODER=char TRAINER=local uv run uvicorn dreaming_api.wiring:app --factory --port 8080
uv run python -m pytest -q
```

The page is then at http://127.0.0.1:8080/. `just` is not required: the recipes in `dreaming/justfile` are plain `uv run` lines (`dev` is the API line above;
`fast-forward` runs the whole loop offline from recorded answers).

## Sponsor tracks

Each block says what ran. Anything not listed here is not claimed.

### Main track: Model Learning
Log, shadow every candidate, certify on shadow rows, promote the candidate with the most certified coverage, and replay run end to end in fast-forward;
audits and revocation are covered by `packages/dreaming/tests/test_runtime.py`. The page charts cost per run, seconds per run, failed % and replayable % over a
2,000-run batch. Evidence: `dreaming/DESIGN.md` sections 1-4, `GET /benchmark`, the replay tab.

### Fetch.ai
A uAgent registered on Agentverse (block above), on the Agent Chat Protocol, relays an ASI:One chat line to `POST /run`; the receipt names which steps a dream
answered and under which certificate. The ASI:One to Agentverse to agent round trip passed and merged at 03:55 (`.plans/sprint-1018.md`, which notes it
ships with the next Fly deploy). Multi-agent calls and Interactive Cards were not built.

### StepFun
The pinned teacher of the typed steps is `step-5-preview`: `TeacherSpec(provider="stepfun", model="step-5-preview", ...)` at
`dreaming/agents/bookkeeper/src/bookkeeper/workflow.py` line 35 (selected by `TEACHER=stepfun`, set in `dreaming/infra/fly.toml`). The provider row
(base URL `https://api.stepfun.ai/v1`, key `STEPFUNKEY`) is at `dreaming/packages/dreaming/src/dreaming/teachers/openai_compat.py` line 41. The cheaper
configuration the router tries is the same model at `reasoning_effort="minimal"` (workflow.py line 36). We dropped `step-3.5-flash` for that role: on the
real anything-odd prompts it thought 500-4,096 tokens even at low effort (p50 13 s, p90 57 s, 12% unparsed), so it was slower and no cheaper than the
teacher. Its self-disagreement floor is 3.5% (above). The other models and services: `BAAI/bge-small-en-v1.5` (the encoder, pinned revision `5c38ec7c`);
`asi1-mini` for the LLM-written seed examples (per `predictors/seeds.py`); OpenRouter and ASI:One exist as alternate teacher rows and are not the submitted
configuration; Modal, Fly.io, Supabase, Sentry, veri.studio, CREAO and Devin are described in this list.

### Intuit / QuickBooks
`dreaming/agents/quickbooks`: three call sites (`invoice_paid`, `bill_account`, `bank_match`) over the `CL251215/QuickBooks-Hackathon-2026` dataset, with
the books as the teacher and the split held out by company. `bank_match` carries injected mess (10% amount noise, 30% memo truncation, 2% duplicates),
disclosed in the docstring of `agents/quickbooks/src/quickbooks/data.py`. `POST /v1/systemone` (`apps/api/src/dreaming_api/systemone.py`) serves any trained
site in TypeSafe's System One wire format (state and typed questions in, probabilities out); a test drives it through `typesafe_sdk.TypeSafeClient`. The
`bill_account` number is above. The live table had no rule baselines and no LLM-alone or Jev rows at 06:48, so we do not claim to beat a baseline.

### CREAO
Agent "Dreaming benchmark runner", ID `ca9c3ce9-78d9-4bc6-ad8f-5ec1d3275991` (https://agent.creao.ai/apps/ca9c3ce9-78d9-4bc6-ad8f-5ec1d3275991), built on
agent.creao.ai. It calls our API (`POST /campaign/run {provider, site}`), reads `GET /campaign/{name}` and writes the provider's benchmark row. It ran once from
the UI and once through the API at about 05:20 (fallback: no `DREAMING_API_KEY` yet). With the secret set (09:10), the UI run at 09:13 did the
core work end to end: from CREAO's sandbox our log shows `GET /healthz` 200, `POST /campaign/run` 200 (campaign `stepfun-bookkeeper.categorize`,
cells launched on Modal) and `GET /campaign/stepfun-bookkeeper.categorize` 200. Run thread: https://agent.creao.ai/chat?threadId=e70ad788-f390-457a-9441-2d8aec4a80b7
(public share link added by the team at submission).

### veri.studio
The `finetune` candidate (bge-small unfrozen plus a head) ran as a `veri run-script` job on an L4: `dreaming/experiments/veri_finetune/` (`train.py` is the job,
`compare.py` the offline before and after, `results.json` its output). Two jobs, `f3fa8651ff45` and `ad7dba4ee80c`, 300 steps at batch 32, about 6 minutes end to
end, $0.34 each; both produced the same weights (`dreaming/docs/devtools.md`). The artifact was downloaded, loaded through `finetune.load` and hashed here. On
held-out merchants of the `categorize` site (259 train rows, 428 StepFun labels in all), accuracy went from 26.0% (frozen head) to 37.9% (fine-tuned), against
19.5% for the majority label. Neither certified at 5% (73 calibration rows), so this shows the platform ran our training, not a better certificate.

### Sentry
`dreaming/packages/dreaming/src/dreaming/telemetry.py` sends Tracing (FastAPI and httpx integrations, plus hand-made `gen_ai.invoke_agent`, `gen_ai.chat` and
`gen_ai.execute_tool` spans per run and per routed call) and Logs (the JSON event lines) to Sentry, and the AI Agents overview fills from those spans. What it
caught, dated, is in `dreaming/docs/devtools.md` under Sentry. Honestly: the build's own runs raised no error issue. The first traffic arrived as 56
`gen_ai.invoke_agent`, 168 `gen_ai.chat` and 112 `gen_ai.execute_tool` spans plus the Logs, and in the same row we changed `telemetry.py` so batch runs no longer
hang off an already-sent request transaction (their spans were dropped) and issues group by component, kind and site instead of by message.

### Cognition (Devin)
Built by Devin: the `hdc` candidate, `dreaming/packages/dreaming/src/dreaming/predictors/hdc.py`, with `tests/test_hdc.py`. PR https://github.com/aidenhongg/dreaming/pull/4
("hdc candidate (built by Devin)"), session https://app.devin.ai/sessions/a713adfe0fc64befb33e9b2c0f64439f. The spec is `training-plan.md` section 1. The session took
9 minutes and the PR merged at 04:19; the merged commit (`c4965e1`) carries a `Co-Authored-By: Devin AI` trailer. The candidate is byte-stable and scores by
count-normalized bundle.

### Paper
The dashboard's dreams tab was rebuilt in Paper from `demo/index.html` by the team (file "Unique yonder").

## Honesty notes

- The ledger is synthetic: 120 merchants and a seeded stream (`agents/bookkeeper/src/bookkeeper/tools.py`), and the page says so. The vendor API is a mock; its 1.2 s latency is charged to a virtual clock and never slept, so "seconds per run" is that model, not wall time.
- Fast-forward answers come from pre-recorded StepFun answers for seed 0 (`dreaming/data/recorded.jsonl`, `RECORDED_SEEDS=0`) and are labelled recorded; a miss goes to StepFun live, one call at a time.
- A certificate holds on rows like the ones it was calibrated on. On held-out companies the memory candidate's realized error at the 5% budget was 8.4% (`bill_account`) and 8.7% (`invoice_paid`) in the 06:48 table; the head's was 1.8% and 4.9%. Every row of that table came from checkpoint summaries and was not recomputed.
- "Byte for byte" means the same label and quantized confidence under the same weights and environment hashes: CPU, batch 1, one thread. A different thread count or a GPU changes the bytes (S29). A promoted "anything odd?" with thinking off is still a frontier call, so replay reads it from the log.
- Budgets below the teacher's self-disagreement (3.5% for StepFun) cannot be measured, and the API refuses them.
- Not claimed here: the LLM-alone and Jev baselines, and Gemini, Snowflake, Tiger Data, Solana, GoDaddy and Browser Use. Hypernetworks and several exotic heads were rejected with numbers (`training-plan.md` section 1). Speculation (running predicted values ahead of the real call) sits behind `speculate=False` and is not part of the demo.
