# Sponsor devtools: how to use each one during development

For every sponsor tool: what it does for Dreaming, how to install and sign in, the smallest real use that lets the Devpost say
"we used it", what to capture as evidence, the gotchas that cost time, and when to quit it. Facts were checked against the
vendors' own docs on 2026-10-17 (Saturday; hacking ends Sunday 2026-10-18 12:00 PDT); anything a primary page did not confirm
is marked *unverified*. Which track each serves, and the sentence we can claim: `tracks.md`.

Rule of thumb for the night: a tool earns at most 30 minutes before its quit rule applies, except Modal and StepFun, which the
campaign and the demo depend on.

## Modal (the campaign's compute)

**For:** every training cell of the matrix, the fine-tunes on a T4, the monitor; `training-plan.md` §3.
**Install / auth:** `modal==1.6.2` is pinned; tokens (7-day, made 2026-10-17) are in `.env` as `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`.
**Human step:** add a payment method in the dashboard. The GPU guide says any GPU needs one, and our workspace shows $1 of the
$30 Starter credit unlocked without it. Set a workspace budget in Settings at the same time.
**Facts that shape the campaign** (modal.com/pricing and the 1.6.2 client):
- Starter: 100 containers, 10 concurrent GPUs, 5 deployed crons, 1-day logs. T4 $0.59/h, L4 $0.80/h, A10 $1.10/h; CPU $0.047
  per physical core-hour, memory $0.008/GiB-h. Tonight's tier-1 matrix (54 CPU cells × ≤ 30 s, 4 GPU cells × ≤ 5 min on T4) is
  under $1; tier 2 adds cents (`training-plan.md` §3).
- The campaign is its own **deployed** app (`apps/trainer/.../campaign.py`, `just deploy-campaign`): the deployed
  `dreaming-trainer.train` has `max_containers=2`, and an ephemeral `modal run` app stops with the client, so cells are spawned
  by name from `just campaign` and the monitor is a cron inside the app; nothing depends on a laptop staying awake.
- `fn.spawn(cell)` per cell (measured: 8 concurrent cold starts back in 21 s); `FunctionCall.get(timeout=0)` polls once and raises
  the **builtin** `TimeoutError` while the call is still running; a timed-out or expired call raises
  `modal.exception.FunctionTimeoutError` / `OutputExpiredError`, which are modal errors (not builtin `TimeoutError` subclasses in
  1.6.2), so a poll loop catches those two first and the builtin one as "still running". Results live 7 days;
  `FunctionCall.from_id("fc-…")` works from any process; `.cancel()` prunes a GPU cell.
- Public `spawn_map` returns nothing in 1.6.2: spawn per cell. `fn.get_current_stats()` gives backlog and running counts.
- Deploy (`modal deploy`) and spawn by name; an ephemeral `modal run` app stops with the client.
- Threads: Modal injects `OMP_NUM_THREADS` and friends from the cpu request (values undocumented). Keep `Image.env({...: "1"})`
  and `torch.set_num_threads(1)` inside the function; S7 says a thread change changes bytes.
- Database fan-out: 100 containers each holding a Supabase connection will exhaust the free project. `train_cell` caps at 12
  containers, connects only at a checkpoint through the session pooler the `dreaming-db` secret already holds (the transaction
  pooler would reject `PostgresStore`'s prepared statements and `SET`), and closes; artifacts go to the Volume, not Postgres.
- Matrix cells never call a teacher (StepFun's voucher tier allows 5 in flight): labels are read from rows.
**Evidence:** `modal app list`, `modal function calls dreaming-campaign/train_cell --json`, `modal billing report`, and every
`TrainRun.where = "modal:cpu" | "modal:T4"` with `MODAL_IMAGE_ID` in `env_sha`.
**Quit:** if after the card is added 50 spawns do not reach ~50 containers in 2 minutes, or GPU inputs sit pending 5 minutes with
under 10 GPU containers: GPU cells go to a RunPod 4090 pod (`../../smokes/pod.py`, ~10-minute boot, $0.34/h); CPU cells run in
one Modal container sequentially (they are seconds each).

## StepFun (the pinned teacher)

**For:** `step-5-preview` as the teacher of the demo's typed steps and the recorded ledger; a benchmark row.
**Install:** the `openai` client; one `PROVIDERS` row in `teachers/openai_compat.py`: base `https://api.stepfun.ai/v1`, key
`STEPFUNKEY`, prices below.
**Human steps:** claim the $25 voucher through the Starter Pack link (50 total, first come, claim by 2026-10-31), redeem under
User Center → Payment → Gift Records, create a key on the Interface Keys page. Identity verification may be asked (*unverified*
whether a voucher-funded key needs it).
**Minimal use** (verbatim shape from the quickstart):
```python
client = OpenAI(api_key=STEPFUNKEY, base_url="https://api.stepfun.ai/v1")
r = client.chat.completions.create(model="step-5-preview", messages=msgs, reasoning_effort="low", temperature=0, max_tokens=256)
r.choices[0].message.content, r.choices[0].message.reasoning, r.usage  # usage has cached_tokens; r.id is the trace id
```
`GET /v1/models` lists models; `GET /v1/accounts` returns `balance`, `total_voucher_balance`.
**Facts:** $1.00 input (cache hit $0.10) / $2.70 output per million; reasoning tokens bill as output and there is **no off
switch**, only `reasoning_effort` low | medium | high. Siblings: `step-3.7-flash` ($0.20/$1.15), `step-3.5-flash` ($0.10/$0.30).
Default temperature 0.5. Voucher-only accounts stay at tier V0: 5 concurrent, 100 RPM, 500k TPM. `response_format`
json_object and function tools exist; built-in web search does not on step-5-preview.
**What this changes in the design:** the router site's cheap configuration cannot be "thinking off" on StepFun; it is the
cheaper tier `step-3.5-flash` (mission 1's original tier router). The record pass is rate-bound: 2,000 runs × 3 typed steps =
6,000 calls per seed ≈ 60 minutes at 100 RPM, est. $3–5 per seed at low effort; record seed 0 first and decide on 1–2 after.
With `max_tokens` small, the reply can be all reasoning and empty content: check `finish_reason` and that the label parsed;
`max_tokens=256` is the safe setting for a one-word label at low effort (*unverified* whether reasoning tokens count toward it).
**Measured on the key, 2026-10-17 22:05 PDT** (`GET /v1/accounts`: balance $25, all voucher):
- 10 typed-label calls at `reasoning_effort=low`, `temperature=0`: with `max_tokens=256` only 5 of 10 parsed (the model echoed
  "Meals — restaurants, coffee, food delivery" in 4, and one spent all 256 tokens on reasoning and returned empty content with
  `finish_reason=length`). With `max_tokens=1024`, a one-word prompt and the tolerant parser (`teachers.parse_label` accepts a
  label followed by a separator): 10 of 10. Output tokens 20–220 per call (reasoning included), so ≈ $0.0003–0.0008 per call.
- **Self-disagreement floor 3.5%**: 200 identical calls on one ambiguous input (AMZN MKTP $63.20) gave Other 193, Office
  supplies 2, unparsed 5. Budgets of 5% and up are measurable; 2% is not. `TeacherSpec.floor=0.035` in `workflow._teachers`.
- Latency: 0.8–3.5 s per call sequentially; at 5-way concurrency p50 6.4 s, p95 11.4 s (the queue, not the 100 RPM cap, binds).
  Throughput ≈ 47 calls/min either way, so the 6,000-call record of one seed is ≈ 2 h, not 1 h: start it early and record seed 0 only.
**Evidence:** the file and line of the `TeacherSpec(provider="stepfun", model="step-5-preview")`, trace rows with
`teacher.provider = "stepfun"`, one saved response with `id`, `usage`, `reasoning`, the Gift Records screenshot, the README
paragraph naming what every other model does (bge-small students, ASI:One, Gemini, Jev as rows).
**Quit (as live teacher only; the README-linked call stays):** vouchers gone before we claim; 401/402 for 30 minutes after
redeeming with no answer from `builders@stepfun.com`; p50 latency at low effort too slow for the loop. Fallback teacher for the
demo: ASI:One (quota) or OpenRouter (top-up).

## Sentry (observability; the track needs two products beyond errors)

**For:** tracing + logs + AI-agent spans across the API, the agent and the trainer; uptime on `/healthz`; replay on the page.
**Install:** `uv add "sentry-sdk[fastapi]==2.72.0"` (pin; 3.x is alpha). `SENTRY_DSN` in Fly secrets, a Modal secret `sentry`,
and `.env`; `SENTRY_ENVIRONMENT=demo`.
**Human steps:** sign up at sentry.io (14-day Business trial, then the free Developer plan: 1 user, 5k errors, 5M spans, 50
replays, 1 uptime monitor, 1 cron, no profiling); create a Python project, copy the DSN; Client Keys → Configure to turn on
Replay for the loader script; create the uptime monitor at sentry.io/monitors/new (UI only) on `https://dreaming-demo.fly.dev/healthz`.
**Minimal use** (every line ran on 2.72.0 with a fake transport):
```python
import logging, os, sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration
logging.basicConfig(level=logging.INFO)  # or INFO records never reach Sentry Logs
sentry_sdk.init(dsn=os.environ["SENTRY_DSN"], environment=os.environ.get("SENTRY_ENVIRONMENT", "demo"), send_default_pii=True,
                traces_sample_rate=float(os.environ.get("SENTRY_TRACES", "1.0")), integrations=[LoggingIntegration(capture_sentry_logs=True)])
# FastAPI, Starlette and httpx integrations switch on by themselves. There is NO psycopg 3 integration: wrap queries by hand
# with start_span(op="db", name=sql) and set_data("db.system", "postgresql"), or skip db spans.
with sentry_sdk.start_transaction(op="task", name="dreaming.run"):          # needed outside an HTTP request (Modal, uAgent)
    with sentry_sdk.start_span(op="gen_ai.invoke_agent", name="invoke_agent bookkeeper") as a:
        a.set_data("gen_ai.operation.name", "invoke_agent"); a.set_data("gen_ai.agent.name", "bookkeeper")
        with sentry_sdk.start_span(op="gen_ai.execute_tool", name="execute_tool vendor_lookup") as t:   # one per routed call
            t.set_data("gen_ai.operation.name", "execute_tool"); t.set_data("gen_ai.tool.name", "vendor_lookup")
            t.set_data("dreaming.decision", "predicted"); t.set_data("dreaming.candidate", "head"); t.set_data("dreaming.cert_id", cert_id)
        # a real teacher call: op="gen_ai.chat", name="chat step-5-preview", gen_ai.request.model / gen_ai.usage.* after the call
sentry_sdk.logger.info("site {site} promoted {candidate}", site=site_id, candidate=name)   # Sentry Logs, trace-linked
sentry_sdk.flush()  # before a Modal function or a short script returns
```
Page: `window.sentryOnLoad = () => Sentry.init({replaysSessionSampleRate: 1.0, integrations: [Sentry.replayIntegration({maskAllText: false})]})`
before `<script src="https://js.sentry-cdn.com/<public-key>.min.js" crossorigin="anonymous">`.
**Where it goes in our code:** `telemetry.py` gains a sink that opens the agent span in `run_boundary` and the tool/chat span
around `router.route`; `boundary()` sets the span status on failure. Sampling: 1.0 for the demo, `SENTRY_TRACES=0.05` for the
2,000-run batches (a batch is ~8,000 spans at 1.0; the free quota is 5M).
**Gotchas:** `start_span(..., attributes={...})` raises `TypeError` in the default mode on 2.72.0: use `set_data`. Spans outside a
transaction are dropped. Prompts and tool arguments are captured only with `send_default_pii=True`. The OpenAI integration labels
the provider "openai" and only patches the `openai` package; our teachers use it, so chat spans come for free but the provider
name needs `set_data("gen_ai.provider.name", …)`. Crons does not count toward the two products; profiling is not on the free plan.
**The judged item is the log of what Sentry caught.** Keep it here, dated, with the issue link and the fixing commit:

| when | what Sentry showed | what we changed |
|---|---|---|
| (starts when the DSN is live) | | |

Caught before Sentry existed (stdout and the `events` table, 2026-10-17, for honesty's sake): the 402 from OpenRouter when
`max_tokens` was unset; the model step marked as an outage after three 429s; `/state` timing out while the batch thread embedded
on the shared CPU. These are the kinds of things the Sentry rows will show.
**Evidence:** Explore → Agents overview for agent `bookkeeper`; one trace API → Modal under one trace id; a Logs query on
`dreaming.candidate`; the uptime monitor green; one replay URL; the table above.
**Quit:** if the Agents view stays empty 30 minutes after spans arrive, keep Tracing + Logs (both plain SDK calls, both count).

## veri.studio (the sponsor training platform)

**For:** one `finetune` cell (bge-small unfrozen + head, S8 v2) per site runs there; its weights come back by download, are hashed
and certified like any other candidate (`TrainRun.where = "veri"`).
**Install:** `uv tool install --upgrade 'veri-sdk[mcp]'` (0.2.44, Python ≥ 3.10; verified to install on this Windows box; set
`PYTHONUTF8=1`), then `veri version`, `veri doctor`.
**Human steps:** sign up at veri.studio; `veri login` (browser authorize; or `veri login --token vk_…`); get the credit code from
@daniung in the LA Hacks Discord and redeem it at Settings → Billing **before the first job** (a $0 balance kills the job after
it boots and bills). Minimum top-up otherwise $5 via Stripe.
**Minimal use, Dreaming-shaped** (`veri run-script --help`, docs/training/custom-script):
```
veri run-script ./experiments/veri_finetune \
  --entrypoint 'python train.py' --base-image veri/base --gpu-type L4-24GB --gpu-count 1 \
  --requirements ./experiments/veri_finetune/requirements.txt --output-name dreaming-finetune-categorize
veri jobs get <id> --format json      # status queued -> provisioning -> configuring -> running -> completed | failed
veri jobs logs <id> -f                # no --follow on run-script itself
veri jobs download <id> --output-dir ./checkpoints
```
The code dir is tarred and uploaded (excludes `.git`, venvs, caches); bundle the rows as one JSONL and the 133 MB bge-small
checkpoint inside it (outbound access to Hugging Face from a job is *unverified*). **The artifact contract**: `train.py` writes
under `$VERI_OUTPUT_DIR` the fine-tuned encoder as `encoder/` in Hugging Face layout (`save_pretrained`) plus `head.npz` in
`Head.to_bytes` layout (`W`, `b`, `labels`, `encoder_sha`); the `finetune` candidate's `load` reads exactly that, whether the job
ran on Modal or on veri. It calls `veri_sdk.log_metrics(step=…, loss=…)` per epoch. The base image is Python 3.10, torch
2.8.0+cu128, transformers 4.56.2: do **not** import the `dreaming` package there; `train.py` is the S8 v2 `ft_encoder` recipe
standalone, ~80 lines. Single quotes around the entrypoint. Test the download path on a tiny job before the real cell.
**Fallback honest use** (if run-script artifacts cannot be downloaded, *unverified* for custom scripts): one managed `sft_text`
job on `Qwen/Qwen3-0.6B` over one site's logged (prompt, label) rows exported as `{"prompt","completion"}` JSONL (`veri datasets
upload`, `veri jobs estimate`, `veri jobs create configs/train.toml --follow`), 50 steps on an L4, with a before/after on our
held-out rows. It is a decoder the smokes say loses to the encoder on typed steps; report that honestly if it does.
**Facts:** L4-24GB $1.046/h on AWS (also vast/digitalocean H100s); 4 in-flight jobs per account; 120 requests/min; jobs that never
reach a GPU are not billed; boots 5 min on AWS, 5–15 on vast; `CAPACITY_UNAVAILABLE` lists alternatives. MCP: `veri mcp serve`
(28 read tools, 7 write) if an agent should watch the job.
**Evidence:** job id and `dashboard_url`, `veri jobs get <id> --format json` pasted, the full log, the downloaded weights' sha in our
`weights` table, the two certificates (frozen head vs fine-tuned) side by side, the `train.py` and the command line committed.
**Quit:** no credit code by late evening and no appetite for $5; a tiny job cannot reach `running` after two resubmits across
providers; the artifact cannot be downloaded after one debugging pass. Confirm with @daniung that a run-script job counts.

## CREAO (the benchmark runner agent)

**For:** one saved CREAO agent that does core work: synthesizes seed rows with CREAO's model, calls our API to train and evaluate
one provider row, polls, and reports the benchmark row. Their team said on 2026-10-17 that synthesize + train + evaluate counts.
**Install:** none (web app; there is no CLI despite the docs' wording). Our side calls it with `httpx`.
**Human steps:** sign up through the hackathon link (`creao.ai/signup?hackathon=ucla-ai&…`, before signing in, for the Pro Plus
upgrade and credits; send the account email to the booth if not upgraded); at agent.creao.ai → Secrets add `DREAMING_API_URL` and
`DREAMING_API_KEY` (a hackathon-only key); build the agent in chat; click **Create Agent** with an input `prompt`; on the agent's
page copy the agent ID and create its `capi_` key; set autonomy to Auto (API runs cannot pause for approval). The CREAO workshop
is tonight 21:00–22:00 PDT on the second-floor patio.
**Minimal use** (docs.creao.ai/pro/api-trigger, verbatim shape):
```
POST https://agent.creao.ai/api/v1/apps/<APP_ID>/runs   Authorization: Bearer capi_…   {"inputs": {"prompt": "benchmark stepfun on categorize"}}
-> 202 {"id": "<RUN_ID>", "status": "pending", ...}
GET  https://agent.creao.ai/api/v1/apps/<APP_ID>/runs/<RUN_ID>   until status in completed | failed | cancelled; result.text, threadId
```
In the sidebar the run shows as an `[API] …` conversation; its share icon gives the read-only link the README needs.
**What our API must offer the agent** (two routes, `apps/api`): `POST /campaign/run {provider, site}` → `{campaign_id}` (seeds
rows if given, spawns the cells) and `GET /campaign/{id}` → the table row when done. Auth by the hackathon key in a header. The
agent waits by polling; runs take minutes and there is no streaming.
**Facts:** runs are async; webhooks are unsigned and cannot reach localhost; packages installed in a chat are kept only when you
click Create Agent; 30 min sandbox idle pause; judging 30% "AI as foundation" (remove the agent and the product stops), 25%
depth on CREAO, 10% demo; select the CREAO track on Devpost or it is not judged; an agent must never start itself.
**Evidence:** agent ID, the shared run link, a Sessions-tab screenshot, our API's access log lines from the sandbox, the clip in
the video, the Devpost track selection.
**Quit:** no upgraded account or credits by late evening; the sandbox cannot reach the Fly API after 60 minutes of tries; runs
take over 10 minutes. Minimum that still satisfies the rule: one saved agent, one shared run, the ID in the README, shown running.

## Cognition Devin (builds one module)

**For:** the `hdc` candidate (`predictors/hdc.py`, build order step 4) from a committed failing test; a second, smaller spec is the
`/v1/systemone` route.
**Install:** none (app.devin.ai). The CLI and API are optional and not needed.
**Human steps:** sign in at app.devin.ai; Settings → Connections → GitHub → add the repo owner's account (personal accounts are
allowed per the FAQ) and select this repo; submit the #cognition credits form (how credits land is *unverified*; Free
has "limited usage"); set Settings → Devin → Pull requests → Open PRs as: Devin.
**Minimal use (UI, ~30 minutes):** commit `packages/dreaming/tests/test_hdc.py` (pure Python, no `.env`, no network) and an
`AGENTS.md` whose first lines give the working directory `dreaming/`, `uv sync`, `uv run pytest packages/dreaming/tests/test_hdc.py -q`,
`uv run ruff check`, and the spec pointer (`docs/training-plan.md` §1 `hdc`, `smokes/s25_exotic/arms.py`). Prompt:
"In this repo, on the build branch, implement `dreaming/packages/dreaming/src/dreaming/predictors/hdc.py`
so the committed test passes; touch only that file and `candidates.py` to register it; do not edit tests; run ruff and pyright;
open a PR titled `hdc candidate`." Review, merge.
**Evidence:** the session URL, the PR (its footer carries "Link to Devin session"), the commit author line from `git log`, a README
line "built by Devin: hdc candidate (PR #n, session …)". One honest line about the name: Devin has a feature called Dreaming.
**Quit:** the GitHub app cannot be installed in 20 minutes, or the first session has no PR after 45 minutes. It is a cheap add.

## TypeSafe System One and Jev (Intuit's preferred API, and a baseline)

**For:** Jev as a baseline row on the QuickBooks sites; our own `POST /v1/systemone` so any certified student is served in the
System One shape (the "compatible API" Intuit prefers; the meaning is inferred from the one-line brief, *confirm with the mentor*).
**Install:** `uv add typesafe-sdk==0.7.5` (Python ≥ 3.10).
**Auth, easiest first:** through OpenRouter with the existing key (`TYPESAFE_BASE_URL=https://openrouter.ai/api`,
`TYPESAFE_API_KEY=$OPENROUTERKEY`, model `typesafe/jev-1.13`; needs OpenRouter credit, $0.042 per million input tokens, output free);
or a TypeSafe console key (signups were reported paused in September, *unverified* now). No key needed to serve our endpoint.
**The wire format** (docs.typesafe.ai/api): request `{"model", "state": str|object|array, "questions": {id: {"type": "choice"|"score"|"noul",
"instructions", "criteria"}}}`; choice criteria is a map option → description (≤ 255 options), score criteria an ordered list of
2–10 levels, noul optional. Response `{"model", "answers": {id: {"type": "choice", "choice", "probabilities", "confidence"} | {"type":
"noul", "noul"} | {"type": "score", "score", "legend", "probabilities", "confidence"}}, "usage": {"input_tokens", "output_tokens"}}`.
typesafe-sdk 0.7.5 was run against a stub server: it posts exactly this and parses the OpenRouter-shaped reply.
**Our route:** one handler in `apps/api/main.py`: question id → call site by name (unknown id → 422); `criteria` must equal the
site's labels (else 422, the student has a fixed label set); state → the site's rendered input; `Prediction.scores` →
`probabilities`, top-2 gap → `confidence`, rounded to 4 places; `usage` from the tokenizer count. Then
`client.system_one(...)` from the SDK with `TYPESAFE_BASE_URL` pointed at Fly is the compatibility proof.
**Baseline pass:** `invoice_paid` as a `choice` with three options, `bill_account` as a `choice` over the company's chart (≤ 255),
state = the rendered row; through OpenRouter the model id is `typesafe/jev-1.13` and the response `model` string
(`typesafe/jev-1.13-20260917`) is what gets logged, with `usage`; `jev-1.13.0` is the id only on api.typesafe.ai. Jev is weak at
arithmetic and dates (its own jaggedness page): keep amounts as text and expect that in the row.
**Evidence:** the JSON log of the pass (model string, usage, request ids), the benchmark row, the curl transcript against our
`/v1/systemone`, the SDK call succeeding against Fly.
**Quit:** OpenRouter route returns 4xx for 30 minutes with no credit; then only our endpoint remains, which costs nothing. Ollaya
(local, same wire format) is optional and not planned.

## Gemini and Snowflake (benchmark rows)

**Gemini:** key at aistudio.google.com/apikey (no card); base `https://generativelanguage.googleapis.com/v1beta/openai/`, models
`gemini-3.8-flash` or `gemini-3.5-flash-lite` (the 2.5 ids are closed to new projects); `reasoning_effort` minimal..high, thinking
cannot be off; free tier limits are only visible in AI Studio and prompts are used for training there (no private data). One
`PROVIDERS` row, 200 calls for the floor, one benchmark row. Evidence: trace rows with `teacher.provider = "gemini"`.
**Snowflake:** 120-day student trial via mlh.link/snowflake-signup; Cortex REST `POST https://<org-account>.snowflakecomputing.com/api/v2/cortex/v1/chat/completions`
with a Programmatic Access Token, OpenAI chat shape (`max_completion_tokens`, not `max_tokens`); models per region
(`openai-gpt-5-mini`, `claude-haiku-4-5`, `llama3.1-8b`…); a PAT only works once the user is under a network policy or the
bypass is set (`ALTER USER … ADD PAT … MINS_TO_BYPASS_NETWORK_POLICY_REQUIREMENT = 240`). Lowest priority of the rows; quit at 30
minutes of auth trouble and say so in the README.

## Browser Use, Paper, Tiger Data, Solana, GoDaddy (cheap adds; each is optional except the domain)

| tool | install / auth | minimal real use | evidence | gotcha | quit |
|---|---|---|---|---|---|
| **Browser Use** | `pip install browser-use-sdk` (3.12.1); key `bu_…` at cloud.browser-use.com/settings, header `X-Browser-Use-API-Key` (no Bearer); the rep's $250 credit after signup (how it is applied is *unverified*) | one `@tool browser_task` wrapping `client.runs.create(task)` + `wait_for_completion(run.id)`, logging `run.id` and `totalCostUsd`; Dreaming learns and speculates over its return like any slow tool | run ids, the runs list screenshot, the trace rows | the cloud has no custom LLM base URL (only the open-source library does; fix in `../../tracks.md`); runs are not deterministic, do not claim replay of the tool itself; `maxCostUsd` caps a run | no result 15 minutes after the key |
| **Paper** | Paper Desktop from paper.design/downloads; redeem `LAHACKS26` on a monthly plan (cancel before renewal or email proof of enrollment); Claude Code: `claude plugin marketplace add paper-design/agent-plugins && claude plugin install paper-desktop@paper`; free plan is 100 MCP calls/week | lay out the dashboard frames in Paper (the Lucid Doodle look from `../../demo/VISUAL_INTENT.md`), hand-edited; copy as Tailwind (`Alt+T`) for the candidate rows | file link (`Cmd/Ctrl+L`; whether it opens logged-out is *unverified*), two screenshots, the redemption screen | judged on craft that is not "clearly AI-generated": do not let an agent generate the look; the submission format is not stated, ask at the booth | the app or plugin does not connect in 20 minutes |
| **Tiger Data** | `irm https://cli.tigerdata.com/install.ps1 \| iex`; signup via mlh.link/tigerdata-quickstart ($1,000 for 30 days); `tiger auth login`; `tiger service create --name dreaming-audit`; `tiger db uri dreaming-audit --with-password` | the same `PostgresStore` driver against the Tiger URI for an audit copy: one hypertable of certificate calibration snapshots, one continuous aggregate (calls and cost per site per hour), one `tiger service fork dreaming-audit --now` per certificate (30–90 s on free) | service id, the fork id per certificate, a query on the fork, the console screenshot | the live database stays on Supabase (Realtime feeds the page); free services are us-east-1 only, `sslmode=require`, no pooler; `--cpu shared --memory shared` flags are *unverified* | two failed `service create` or psycopg cannot connect |
| **Solana** | `uv pip install solders`; no account; fund a devnet keypair at faucet.solana.com by hand (2 requests per 8 h; the faucet tells agents not to use it) | on `cert.issued`, a Memo (program `MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr`) of the certificate's sha256 on devnet via plain JSON-RPC to `https://api.devnet.solana.com`; ~5,000 lamports each | the signature and `https://explorer.solana.com/tx/<sig>?cluster=devnet` showing `Memo (len 64)`, the sha recomputed from the repo | thin for the prize (Unruggable Hardware Wallets); frame it as an attestation anyone can recompute, nothing more; keep the keypair as a Fly secret | no funded wallet in 15 minutes |
| **GoDaddy Registry** | log in at mlh.com to see the code (the public page hides it; `MLHLHA2685` from the Discord), redeem at tech.study (claim by 2026-10-22, one per account) | `fly certs add <domain> -a dreaming-demo`, A + AAAA on the apex as it prints (`fly ips allocate-v4 --shared` if no v4), `fly certs check` | the domain resolving to the app, `fly certs show` Ready, the tech.study confirmation (code hidden) | the DNS host tech.study hands out is *unverified*; a cert stays pending until DNS is right | cert not Ready 30 minutes after correct DNS; `dreaming-demo.fly.dev` stays the demo URL |

## The sign-in checklist

One ordered list, in `README.md` here.
