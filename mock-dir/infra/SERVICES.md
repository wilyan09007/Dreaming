# External services reference

Consolidated 2026-10-17 from four research briefs (modal, supabase-neon, hosting, telemetry-web-openrouter). Choices and reasons: `DESIGN.md` section 9. Prices are as of 2026-10-17 and move often; re-check vendor pages before budgeting.
Trailing tags: [D] first-party docs or source read directly, [T] run in the research container (briefs: LOCAL, measured), [W] web search snippet or third-party page, [M] model knowledge or inference.
modal.com, supabase.com, openrouter.ai, asi1.ai, pydantic.dev and vercel.com were unreachable to the researchers, so facts from those hosts are [W] unless a repo clone is named.

## Supabase

Inspected: supabase/supabase@7425d49 docs (2026-10-17); supabase-js 2.x (npm latest 2.118.0 on 2026-10-15, pin 2.117.1, `postgres_changes_options.wait` needs >= 2.113.0, v3 exists only on the `next` tag so do not use it); supabase-py 2.33.0; psycopg 3.3.6; asyncpg 0.32.0; Postgres 16 for the SQL test. [D][T]
- Browser: `supabase.createClient(url, 'sb_publishable_...', {realtime: {worker: true, heartbeatCallback}})`, then `sb.channel(name, {config: {postgres_changes_options: {wait: true}}}).on('postgres_changes', {event: 'INSERT'|'UPDATE'|'DELETE'|'*', schema: 'public', table, filter: 'run_id=eq.1'}, cb).subscribe(status => ...)`. Statuses: SUBSCRIBED, CHANNEL_ERROR, TIMED_OUT, CLOSED. Payload: `{schema, table, commit_timestamp, errors, eventType, new, old}`. [D]
- Keys: `sb_publishable_...` is role `anon`; `sb_secret_...` is `service_role` (BYPASSRLS, rejected from browser User-Agents). Legacy anon/service_role JWTs still work and are deprecated by end of 2026. [D]
- SQL per live table: `alter publication supabase_realtime add table public.<t>`, `enable row level security`, `grant select ... to anon`, `create policy ... for select to anon using (true)`. Python writers use the `postgres.<ref>` pooler role (table owner, bypasses RLS). The migration ran twice (idempotent) on stub roles. [D][T]
- Python: `psycopg.connect(DSN, autocommit=True)`; `psycopg_pool.AsyncConnectionPool(DSN, min_size=1, max_size=5, open=False, kwargs={"autocommit": True, "prepare_threshold": None})`; asyncpg `statement_cache_size=0` on :6543. Storage only: `storage3.create_client(url, headers, is_async=False)`, `.from_('weights').upload(key, data, {'content-type': 'application/octet-stream', 'upsert': 'true'})`, `.download`, `.create_signed_url(path, expires_in)`. [D][T]
- Migrations: `psql "$SESSION_POOLER_URL" -v ON_ERROR_STOP=1 -f db/migrations/0001_init.sql`. The CLI is optional: `supabase migration new`, `supabase db push [--dry-run]`, `supabase link --project-ref`, `--db-url`. [D]

| Connection | Host:port, user | Free plan | Prepared stmts / LISTEN |
|---|---|---|---|
| Direct | `db.<ref>.supabase.co:5432`, `postgres` | IPv6 only (IPv4 add-on about $0.0055/h, Pro+) | yes / yes |
| Session pooler | `aws-<n>-<region>.pooler.supabase.com:5432`, `postgres.<ref>` | IPv4 | yes / yes |
| Transaction pooler | same host `:6543`, `postgres.<ref>` | IPv4 | no / no (also no SET, advisory locks, temp tables) |

Use the session pooler from Fly and the transaction pooler from Modal fan-out. The pooler host cannot be derived from the region; copy it from Dashboard > Connect. Free: 60 direct connections, 200 pooler clients. [D]

| Free plan limit (2026-10-17) | Value |
|---|---|
| Price | $0; Pro $25/mo never pauses [D] |
| Database | 500 MB then read-only (SQLSTATE 25006); 2 active projects [D] |
| Storage | 1 GB total, 50 MB per file, not raisable on Free [D] |
| Egress, auth | 5 GB plus 5 GB cached; 50,000 MAU; unlimited API requests [D] |
| Realtime rates | 200 concurrent connections, 100 messages/s, 100 joins/s, 100 channels per connection, 2M messages/month [D] |
| Realtime payloads | Broadcast 256 KB; Postgres Changes 1,024 KB (larger rows keep only fields <= 64 bytes) [D] |
| Pausing | paused after about 7 days of low DB activity, restorable for 1 year; public Realtime connections last at most 24 h [D] |

Gotchas:
- No events usually means grant, RLS policy or publication, in that order; test `select` as anon first. New projects since 2026-05-30 (existing ones 2026-10-30, date from secondary pages) no longer auto-grant `anon`, so `GRANT` explicitly. `alter publication ... add table` errors if already a member. [D][W]
- `SUBSCRIBED` can fire before the replication subscription exists: use `wait: true`, backfill by REST after every SUBSCRIBED, dedupe by `id`. Postgres Changes has no replay. [D]
- One DB change is one message per listening client, so inserts/s x viewers must stay under 100/s (10/s x 10 viewers saturates it). UPDATE events need the old and new row to pass the policy and `old` carries only the primary key under RLS; use `event: '*'` for tables whose rows change. Keep raw trace rows out of the publication. [D]
- 10 rows/s x 1 KB is about 860 MB/day, so the 500 MB cap falls in about 14 h. Weights over 50 MB cannot go in Free Storage; `bytea` is fine up to about 5 MB in its own table outside the publication with `set storage external`. [D]
- Free projects pause: ping every 1 to 3 days from outside (`pg_cron` dies with the project); GitHub Actions schedules auto-disable after 60 days of repo inactivity. [D][M]
- Transaction pooler breaks prepared statements, LISTEN and SET; percent-encode special characters in the DB password; hidden tabs drop heartbeats (`worker: true`). [D]

Human sign-in: Supabase account and project (region near Fly and Modal, save the DB password); Settings > API Keys for `sb_publishable_...` and `sb_secret_...`; Dashboard > Connect for both pooler strings; optional `weights` bucket; GitHub secret for the keep-alive. Optional CLI route: `supabase login`. [D]
Unverified: Postgres Changes latency on a real project (only synthetic Broadcast figures exist: median 6 to 58 ms); default Supavisor pool size on Free; whether a REST ping counts as pause activity; `sb_publishable_` with private Broadcast; whether the 24 h public-connection cap reconnects cleanly.

## Modal

Inspected: `modal==1.6.2` (PyPI 2026-10-16, Python >=3.10,<3.16), installed source plus modal-labs/modal-examples@65241421; docs and prices from search snippets of modal.com. Pin it on caller and image. [D][W]
- Renamed, old names are hard errors: `keep_warm` to `min_containers`, `concurrency_limit` to `max_containers`, `container_idle_timeout` to `scaledown_window`, `allow_concurrent_inputs` to `@modal.concurrent(max_inputs=N)`, `@modal.web_endpoint` to `@modal.fastapi_endpoint`, `modal.Stub` to `modal.App`. 1.6.0: `@app.cls` may not define `__init__` (use `modal.parameter`). [D]
- Function: `app = modal.App("dreaming-trainer", image=image)`; `@app.function(gpu="T4", cpu=2.0, memory=8192, timeout=900, max_containers=3, retries=modal.Retries(max_retries=2, initial_delay=5.0, backoff_coefficient=2.0), volumes={"/weights": vol}, secrets=[...], min_containers=, scaledown_window=, single_use_containers=)`. Web Functions reject `retries`. [D]
- Call from outside (deployed apps only): `fn = modal.Function.from_name("dreaming-trainer", "train")`; `fn.remote(...)` blocks; `call = fn.spawn(...)` then `call.object_id` (`fc-...`); `modal.FunctionCall.from_id(id).get(timeout=0)` raises builtin `TimeoutError` while running and `modal.exception.OutputExpiredError` when expired; `call.cancel()`. Caller env: `MODAL_TOKEN_ID` (`ak-`) and `MODAL_TOKEN_SECRET` (`as-`). Args and returns are pickled, so return plain dicts. [D]
- Image: `modal.Image.debian_slim(python_version="3.12.10")` (give the full version), `.uv_pip_install(...)`, `.run_function(fn)` to bake HF weights, `.env({...})`, `.add_local_python_source(..., copy=True)`. Env ids: `MODAL_IMAGE_ID` inside, `Function.info().image_info.image_id` outside. [D]
- Storage and secrets: `modal.Volume.from_name(n, create_if_missing=True)`, `vol.commit()`, `vol.reload()`, `vol.read_file(path)`; `modal.Secret.from_name("dreaming-db", required_keys=["DATABASE_URL"])`. CLI: `modal token new`, `modal secret create dreaming-db DATABASE_URL=... [--force]`, `modal deploy f.py [--strategy rolling|recreate] [--tag]`, `modal app logs <app> -f --since 2h`. [D]
- GPU strings: `"T4"`, `"L4"`, `"A10"`, `"L40S"`, `"A100-40GB"`, `"A100-80GB"`, `"H100"`, `"H100!"`, `"H200"`, `"B200"`. Plain `A100` may upgrade to 80 GB and plain `H100` to H200 (`"H100!"` blocks it); a list means ordered fallbacks, so never pass one. [D][W]

| Starter plan (2026-10-17) | Value |
|---|---|
| Plan | $0 plus usage; $30/mo recurring credit; 100 containers; 10 GPU concurrency; 5 deployed crons; 3 seats; no static-IP proxy [W] |
| GPU per hour | T4 $0.59 ($0.000164/s), L4 $0.80, A10 $1.10, L40S $1.95, A100-40GB $2.10, A100-80GB $2.50, H100 $3.95 [W] |
| CPU, memory | $0.0000131 per physical core-s (1 core = 2 vCPU), $0.00000222 per GiB-s; minimum container 0.125 core + 128 MiB [W] |
| Job cost | 2-min T4 job at 2 cores / 8 GiB about $0.025, so about 1,200 per $30; GPU time alone: 50.8 T4 hours per $30 [W] |
| Always-on, 730 h | 0.125c/128 MiB $5.0; 0.25c/512 MiB $11.5; 0.25c/1 GiB $14.4; 0.5c/1 GiB $23.1; 1c/2 GiB $46.1 per month [W] |
| Multipliers | `nonpreemptible=True` 3x CPU/mem; region pin 1.15x broad to 1.75x narrow; Sandboxes about 3x [W] |
| Egress | billed from 2026-10-01 beyond 1 TiB per cycle at $0.04/GiB; Volume reads and writes excluded [W] |
| Limits | Web Function request 150 s then 303; WebSocket message 2 MiB; 200 req/s default; `timeout` default 300 s, max 24 h; `scaledown_window` 2 to 3600 s [D][W] |
| Retention | `.remote()` results 24 h; `.spawn()` results 7 days; `modal.Dict` entries expire after 7 days idle [D][W] |

Billing is per second from container start, including image pull and `@modal.enter`. [W] The briefs conflict on two points: `discord_bot.py` says a live container costs about 2 cents/hour (about $14/mo) against a $4.96/mo floor from the rates; the docs snippet caps `scaledown_window` at 20 min while 1.6.2 source allows 3600 s. [D][W]

Gotchas:
- The caller needs the deployed app and both token vars, so the Fly API holds `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` as secrets; `modal serve` apps fail lookup. Starter has no static IP, so the DB must accept public TLS (the Supabase pooler does). [D]
- Determinism: no CPU SKU control and the CPU mix is unknown. Modal injects `OMP_NUM_THREADS` and friends from the CPU request, so set `.env(...)` and `torch.set_num_threads(1)` plus `use_deterministic_algorithms(True)`. CUDA training is not bitwise deterministic, so certify the CPU path. Record CPU model, flags and `MODAL_IMAGE_ID`. [W][M]
- `min_containers=1` bills 24/7; an API without `@modal.concurrent` serves one request per container and fans out to `max_containers`. DESIGN keeps the API on Fly; Modal only trains. [D]
- Volumes need `commit()` and `reload()` (reload fails with open files); use sha256 filenames; never `modal.Dict` for weights. `debian_slim()` without `python_version` follows the caller's Python. [D]
- Do not host the uAgent here: 24 h timeout cap, preemption, 5 crons. Web Functions have the 150 s cap, and the 303 mechanism breaks under CORS. [D][W]

Human sign-in: Modal account in the browser, `modal token new` (or Settings in the dashboard) for the token pair, then `modal secret create dreaming-db DATABASE_URL=...`. [D]
Unverified: CPU-API cold start; whether `Image.env` thread vars win over Modal's injected ones; whether an identical recipe gives the same `im-` id; CPU vendor mix; `RTX PRO 6000` string; log retention; all prices (snippets, not fetched pages).

## Fly.io

Inspected: superfly/docs@2f68a40 (2026-10-16) and superfly/flyctl@19b6f4c source; nothing deployed. [D]
- CLI: `fly apps create <app> -o <org>`, `fly secrets set --stage K=V`, `fly deploy -c infra/fly.toml --ha=false -y`, `fly status`, `fly checks list`, `fly logs -a <app> --no-tail [-m <machine>]`, `fly ssh console -a <app> -C "cmd" [-g agent]`, `fly machine stop|start <id>`, `fly scale count api=1 agent=1`, `fly tokens create org -o <org> -x 720h`. [D]
- Auth: `FLY_API_TOKEN` (also `FLY_ACCESS_TOKEN`), value `FlyV1 fm2_...` contains a space, so quote it. An app-scoped deploy token cannot create the app; use the org token. `fly auth login` refuses non-interactive use. [D]
- `fly.toml` keys used: `[processes]` (api, agent), `[http_service] processes=["api"] internal_port force_https auto_stop_machines="off" auto_start_machines min_machines_running`, `[[http_service.checks]]`, `[[vm]] processes size memory`, `[[restart]] policy="always" processes`, `kill_timeout="30s"`; default `kill_signal` is SIGINT. The agent group has no service. [D]

| Price, 30 days, iad/ewr (2026-10-17) | Value |
|---|---|
| shared-cpu-1x 256 MB / 512 MB / 1 GB | $2.19 / $3.69 / $6.70 [D] |
| 2 x 512 MB (DESIGN figure) | $7.38 ($7.63 for 31 days) [D] |
| API 1 GB + agent 512 MB (as `infra/fly.toml` is written) | $10.39, the sum of two rows above [D] |
| Default HA left on | about $11.1 for the 512 MB pair [D] |
| Other regions vs iad | ams/arn 1.04x, lhr/cdg 1.13x, fra 1.15x, sjc 1.19x, ord/dfw 1.25x, gru 1.62x [D] |
| Network, disk | egress $0.02/GB NA/EU, inbound free; shared IPv4 and IPv6 free, dedicated IPv4 $2/mo; volumes and stopped rootfs $0.15/GB/mo [D] |
| Free allowance | none for new orgs; trial is 2 VM-hours or 7 days and trial machines auto-stop after 5 min; no-card route is prepaid credit, minimum $25 [D] |
| CPU | shared baseline 6.25% of a core, burst balance max 500 s [D] |

The modal brief suggests a 256 MB agent ($2.19); the hosting brief measured agent RSS 83 to 96 MB and calls 512 MB the sensible floor for agent plus API. DESIGN section 9 says 2 x 512 MB, but `infra/fly.toml` gives the API 1 GB for torch (the modal brief says torch plus bge needs >= 1 GiB). [D][T]

Gotchas:
- `fly deploy` makes 2 machines for a group with a service and a stopped standby for one without; use `--ha=false` or pay about $11.1 instead of $7.38. [D]
- `fly secrets set` without `--stage` restarts every machine in every group, and `fly deploy` restarts the agent too; freeze deploys before the demo. [D]
- Default restart policy `on-failure` ignores exit 0, so set `[[restart]] policy = "always"`. Health checks gate routing or report; they never restart a machine. [D]
- Keep SIGINT: uAgents exits cleanly on it and dies at once on SIGTERM (exit 143). [T]
- Fly Proxy closes idle HTTP streams at 60 s (`http_options.idle_timeout` reportedly has no effect); send an SSE comment heartbeat every 15 s. [W]
- `[processes]` commands replace CMD but not ENTRYPOINT (keep the image ENTRYPOINT-free); bind 0.0.0.0; a leftover `[http_service]` without `processes=` applies to all groups. Outbound IPv6 to Supabase direct hosts was reported lossy from gru and fra in 2026, so use the IPv4 pooler. [D][W]

The briefs conflict on IPv6: hosting says outbound IPv6 works with one gru caveat; the supabase brief says Fly docs and forum are contradictory with gru and fra timeouts. Third-party trackers list many Fly incidents in 2026 [W]; keep the laptop as cold standby with the same seed (stop the Fly agent machine first). [T]
Human sign-in: Fly account in the browser with a card (or >= $25 prepaid credit; prepaid cards cannot be the saved method), then mint the org token once. [D]
Unverified: deploy duration; `[processes]` quoting and exec parsing; that an org token can `fly apps create` without extra flags; `fly machine run --dockerfile` from cwd.

## OpenRouter

Inspected: OpenRouterTeam/docs@09ad514 (2026-10-17) including `openapi.yaml`; openai SDK 3.33.0 against a mock transport (body keys and headers only; no live call). Base URL `https://openrouter.ai/api/v1`. [D][T]
- Client: `OpenAI(base_url=..., api_key=..., default_headers={"HTTP-Referer": url, "X-OpenRouter-Title": "dreaming"})`; `X-Title` is legacy. [D]
- Pin: `extra_body={"provider": {"order": ["anthropic"], "allow_fallbacks": False, "require_parameters": True, "data_collection": "deny"}}`. `ProviderPreferences` fields: `order`, `allow_fallbacks` (true), `require_parameters` (false), `data_collection` (allow|deny), `zdr`, `enforce_distillable_text`, `only`, `ignore`, `quantizations`, `sort`, `preferred_min_throughput`, `preferred_max_latency`, `max_price`, `options`; `additionalProperties: false`. `order` takes lowercase slugs. [D]
- Fallback models: `extra_body={"models": [...]}`, top-level array; billed at the model actually used. [D]
- Served provider: header `X-OpenRouter-Metadata: enabled`, read `r.model_extra["openrouter_metadata"]` with `requested`, `strategy`, `attempt`, `endpoints.available[{provider,model,selected}]`, `attempts[{provider,model,status}]`, `is_byok`, `pipeline`; absent on cache hits. Or `GET /api/v1/generation?id=gen-...` for `provider_name`, `total_cost`, tokens, `latency`, `finish_reason`. [D]
- Cost: `usage.cost` and `usage.cost_details.upstream_inference_cost` are always returned; `usage:{include:true}` and `stream_options.include_usage` are deprecated no-ops. [D]
- Other endpoints: `GET /api/v1/key` (`limit`, `limit_remaining`, `limit_reset`), `GET /api/v1/models/{author}/{slug}/endpoints` (`supported_parameters`), `POST /api/v1/embeddings`, `GET /api/v1/embeddings/models`. [D]

| Money and limits (2026-10-17) | Value |
|---|---|
| Fees | no inference markup; card purchase 5.5% ($0.80 minimum), crypto 5% [D] |
| Rate limits | `:free` models 20 rpm, 50/day if under 10 credits ever bought, else 1000/day; paid models have no platform request cap [D] |
| Errors | 402 = balance, per-key cap or in-flight budget (`error.metadata.limit_source`, honour `Retry-After`); 403 = guardrail; 429 = free models or upstream [D] |
| Embeddings | `openai/text-embedding-3-small` $0.02/M [D]; `baai/bge-base-en-v1.5` $0.005/M, `baai/bge-large-en-v1.5` $0.01/M, `baai/bge-m3` $0.01/M [W]; no bge-small listed |
| Teacher cost | no model prices in the briefs; read `usage.cost` per call |

Gotchas:
- A typo in `provider` returns 400. Setting `order` or `sort` disables load balancing; with `allow_fallbacks: false` upstream errors reach you (an outage from over-pinning was reported [W]). [D]
- `require_parameters: true` fails the call when no endpoint supports every sent param. Anthropic's native API has no `seed` [M], so `seed` plus this flag plus an Anthropic pin may not route: check `supported_parameters` first. `quantizations` on closed models filters out everything. [D][M]
- The 2026 response schema has no top-level `provider`; use `openrouter_metadata` or `/generation`. The briefs list no `provider` key directly under `openrouter_metadata` (DESIGN section 11 expects `openrouter_metadata.provider`). [D]
- How `models` fallbacks interact with a `provider.order` pin is undocumented; check `attempt` greater than 1. [D]
- Determinism is "not guaranteed for some models" [D]; log seed, temperature, provider and `system_fingerprint` per call.
- Parallel large `max_tokens` can give 402 with a positive balance on small prepaid accounts (in-flight budget). [D]
- Embeddings stay local: bge-small-en-v1.5 (384-d, about 33M params [M]) on CPU gives identical vectors in library and Modal; a hosted fallback must be pinned and store model and dimension with each vector. [D][M]

Human sign-in: OpenRouter account, credits, API key (optional per-key credit limit); DESIGN assumes the key exists. [D]
Unverified: every price (snippets); top-level `provider` in non-streaming responses; `/generation` freshness lag; `seed` in Anthropic `supported_parameters`; metadata present on every response.

## ASI:One + Agentverse

Inspected: `uagents==0.26.1` (uagents-core 0.4.12, Python >=3.10,<4.0; fetchai/uAgents@90e2f6a, 2026-10-13), run against a local fake Agentverse, never the real one. ASI:One facts are [W] only (docs.asi1.ai unreachable). [D][T]
- ASI:One: OpenAI SDK with `base_url="https://api.asi1.ai/v1"`; Chat Completions and Responses formats. Body fields: `model, messages, temperature (0-2), top_p, max_tokens, stop, stream, tools, parallel_tool_calls, response_format`; custom via `extra_body`: `planner_mode, agents, enable_thinking, thinking_budget, web_search, agent_address`. Send header `x-session-id: <run_id>`. Unknown top-level fields return 400 naming the field; `seed` is not listed. [W]
- uAgent: `Agent(name=, seed=os.environ["AGENT_SEED"], mailbox=True, port=8000)`, `@agent.on_event("startup")`, `@agent.on_interval(period=30.0)`, `@agent.on_message(Model)`, `ctx.send(addr, msg)`, `agent.run()` or `run_async()`, `handle_messages_concurrently=True`. [D]
- Mailbox: `MailboxClient` polls `GET {agentverse}/v2/agents/<addr>/mailbox` every `MAILBOX_POLL_INTERVAL_SECONDS = 1.0`, deletes envelopes after dispatch, and the server stores messages while the agent is offline. Needs outbound HTTPS to agentverse.ai only; no inbound port. Importing `uagents` also builds Fetch ledger clients and only logs "Ledger unavailable ... will retry later" on failure. [D][T]
- Create the mailbox once: run locally with the production seed, open the printed `https://agentverse.ai/inspect/?uri=http%3A//127.0.0.1%3A8000&address=agent1q...`, Connect > Mailbox. Same seed gives the same address on any host. Programmatic route from source, untested: `POST http://127.0.0.1:8000/connect {"user_token": "<Agentverse API key>", "agent_type": "mailbox"}`. [D][T]

| Item (2026-10-17) | Value |
|---|---|
| `asi1` | about 196k context, default, image input [W] |
| `asi1-ultra` | about 203k context, most capable [W] |
| `asi1-mini` | fastest and cheapest; context 262k per docs, 200k per a blog (conflict) [W] |
| Prices, rate limits | not published in any reachable page; limits are plan based with 429 on excess, no numbers [W] |
| Agent footprint | idle RSS 83 to 96 MB, about 0.7% of a core, 1 poll/s per agent forever [T] |

Gotchas:
- Always pass `seed=`; without it keys go to `./private_keys.json`, lost on every Fly restart, and the address and mailbox change. `AGENT_SEED` is a secret. [T]
- `enable_agent_inspector=False` with `mailbox=True` and no REST handlers makes `agent.run()` return at once. Keep the inspector on; it binds 0.0.0.0:8000, which stays unreachable when the Fly group has no service. [T]
- Python 3.14 breaks `Agent()` ("no current event loop"); stay on 3.12 or 3.13 (`serve.Dockerfile` uses 3.12) or pass `loop=`. [T]
- Two processes with one seed split or duplicate mailbox messages; never run Fly and laptop agents together. [T]
- ASI:One rejects unknown body fields: build its request separately from OpenRouter's and never reuse `extra_body`. Its docs pages disagree on model names, context sizes and penalty params. [W]
- Messages are handled sequentially by default; any Agentverse outage stops the chat path whatever the host. [D]

Human sign-in: ASI:One account and API key; Agentverse login once to connect the mailbox (or an Agentverse API key for the `/connect` route). [D][W]
Unverified: real mailbox creation, `/connect` payload, Almanac registration, rate limits, offline envelope retention; `asi1-mini` price; whether `temperature` is honoured in thinking mode; `x-session-id` required or optional (pages disagree).

## Logfire

Inspected: `logfire==5.2.0` (PyPI 2026-10-13, Python >=3.10, ships `py.typed`) run here. Repo HEAD is the v6.0.0b8 pre-release line, so pin `logfire>=5.2,<6`. [T][D]
- Setup: `logfire.configure(service_name="dreaming-api", send_to_logfire="if-token-present", environment="prod", metrics=False, scrubbing=logfire.ScrubbingOptions(callback=keep))`. Env: `LOGFIRE_TOKEN`, `LOGFIRE_SEND_TO_LOGFIRE=false`, `LOGFIRE_SERVICE_NAME`. `send_to_logfire=False` prints to console only and sends nothing. [T][D]
- One-liners, after `configure()`: `instrument_fastapi(app)` (`logfire[fastapi]`), `instrument_httpx()` (`logfire[httpx]`), `instrument_psycopg()` (`logfire[psycopg]`), `instrument_openai(client)`, `instrument_pydantic(record="failure")`. httpx injects and FastAPI extracts W3C `traceparent`, giving the same trace_id across a real HTTP hop. [T]
- Manual: `logfire.span("step {n}", n=3, component="trainer")`, `logfire.info/exception(..., run_id=...)`, `@logfire.instrument`, `logfire.LogfireLoggingHandler()`, `logfire.force_flush()`, `logfire.shutdown()`, `logfire.get_context()` with `logfire.attach_context(ctx)`. [T]
- OTel: `OTEL_EXPORTER_OTLP_ENDPOINT` makes the SDK also export there (protobuf over HTTP only); `additional_span_processors=[...]` accepts a custom exporter. [D][T]

| Plan (2026-10-17) | Value |
|---|---|
| Personal | $0, never billed; 10M spans + logs + metrics per month combined; 30-day retention; 1 seat, 3 projects (the last two [W] only) [D] |
| Over the cap | Live view and single-trace views stop returning data, then new data is dropped; both lift at monthly reset [D] |
| Team / Growth | $49/mo (5 seats, +$25/seat to 12) / $249/mo; overage $2 per million [D][W] |
| Ingest limits | 100 MB per request; timestamps within 24 h past and 1 h future, otherwise silently dropped [D] |

Gotchas:
- Bare `logfire.configure()` with no token raises `LogfireConfigError`; use `"if-token-present"` (console only when absent). [T]
- The default scrubber redacts keys and text containing auth, session, api key, secret, token, password, cookie, jwt: `error="401 Unauthorized: authentication failed"` became `[Scrubbed due to 'auth']`. Add a callback returning `m.value` for `error`, `session_id`, `call_site`. [T]
- `instrument_pydantic()` defaults to a span per validation (use `record="failure"`) and only covers models defined after the call; metrics count toward the 10M (`metrics=False`). [D]
- `instrument_httpx` does not hook `httpx.ASGITransport`, so test `traceparent` over real sockets; `instrument_openai` suppresses inner httpx spans by default. [T]
- Modal jobs may be hard-killed: call `logfire.force_flush()`; pass `logfire.get_context()` in job payloads and uAgent messages to keep one trace. [D]

Human sign-in: optional; Logfire account (GitHub, Google or email code), a project, and a write token (or API key with `project:write_otlp`) set as `LOGFIRE_TOKEN`. `logfire auth` is a browser device flow, so this cannot be headless. [D]
Unverified: Personal "1 seat, 3 projects"; behaviour of `instrument_openai` suppression against a live host; free tiers of Honeycomb, Axiom and Grafana (third-party snippets).

## Static page / types

Inspected: fastapi 0.143.3 `StaticFiles`; json-schema-to-typescript 16.0.0 (`json2ts`); datamodel-code-generator 0.85.0; tsc 5.9.3; Pydantic v2. Cost $0, no accounts. [T]
- Serve: `app.mount("/", StaticFiles(directory="apps/web", html=True), name="web")` registered after every API route; `/` returned index.html and `/api/healthz` returned JSON in one app. [T]
- Schema to types: `Model.model_json_schema(mode="serialization")`, then `npx json2ts -i schemas/event.schema.json -o web/types/event.d.ts --no-additionalProperties`. Reverse: `datamodel-codegen --input f.schema.json --input-file-type jsonschema --output-model-type pydantic_v2.BaseModel --output gen_models.py`. [T]
- Page: `// @ts-check` plus JSDoc `/** @type {import('./types/event').Event[]} */`; check with `tsc --noEmit --allowJs --checkJs page.js`; CI guard `git diff --exit-code schemas web/types`. [T]
- Supabase client: `<script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2.117.1/dist/umd/supabase.js">` (global `supabase`, 218 KB, about 55 KB gzip). [T]
- If ever split from the API: `CORSMiddleware(allow_origins=[...], allow_methods=["GET"])` echoed the header. [T]

Gotchas:
- A mount at `/` swallows routes registered after it. [T]
- Without `--no-additionalProperties` (or `extra="forbid"`) json2ts adds `[k: string]: unknown` and typos like `e.errorr` type-check silently. [T]
- The page dies with the API: keep the Supabase-direct failure banner and the `GET /events?after=` poll; send an SSE heartbeat every 15 s if SSE is used (Fly 60 s idle close). [T][W]
- Datetimes are plain `string` in JS; field titles create noisy aliases such as `Error` and `Ts`. [T]
- Pin supabase-js to an exact 2.x; never `@2` or `@next`. [D]

Human sign-in: none.
Unverified: the `tsc` check ran on a test page, not `apps/web`; supabase-js UMD against a live project.

## Rejected

- Neon: browser realtime needs a TypeScript backend to seal 60-second queries (`@neon/realtime` 0.1.1, beta, no RLS, 4 AWS regions); scale-to-zero after 5 min cannot be disabled on Free and kills LISTEN. Upsides: IPv4 and IPv6, 5 GB free Object Storage, 100 CU-hours. [D]
- Railway: Hobby $5 includes $5 usage, but `railway.json` config-as-code stops being read 2026-12-01 and unverified GitHub accounts on the trial get restricted outbound ports. [D]
- Render: free web service spins down after 15 min; workers have no free tier; web Starter $7/mo. [W]
- Modal as API or agent host: API feasible at $5 to $23/mo but 150 s request cap; agent needs a 24 h function plus respawn hacks. [D][W]
- Vercel (CLI 64.5.0): `vercel deploy ./dash --prod --yes --token "$VERCEL_TOKEN"`; Hobby is non-commercial and preview URLs may sit behind Vercel Authentication. [T][W]
- Cloudflare Pages (wrangler 4.151.1): `wrangler pages deploy`; needs token, account id and project create; 25 MiB per file; docs steer new projects to Workers static assets. [D]
- GitHub Pages via Actions: no secrets (OIDC) but Settings > Pages must be set to Actions; free only on public repos; 1 GB site, 100 GB/mo soft. [D]
- Honeycomb, Axiom, Grafana Cloud with the plain OTel SDK: more bootstrap code, no LLM-aware UI; free tiers are [W] only. structlog to stdout stays as the baseline (Fly keeps about 7 days, Modal retention is plan based). [W]
- Hosted embeddings (OpenRouter bge-base, -large, -m3; Cloudflare Workers AI bge-small at $0.0202/M input): dimension or hardware drift versus the in-process encoder. [W]
- supabase-py for DB traffic: psycopg 3 is one hop with real transactions; supabase-py 2.33.0 (or `storage3`) only for Storage. Supabase Storage for weights over 50 MB, and `modal.Dict` for weights, also rejected. [D]
- RunPod, LiteLLM, Helicone, Langfuse, Temporal, Next.js, a GPU predictor pool: rejected in DESIGN section 9 and not researched in these briefs.
