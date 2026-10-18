# Tracks: what each sponsor requires, which component serves it, what we show

The engineering view of `../../tracks.md` (the facts: prizes, rules, codes, sources, checked 2026-10-17). This file says, per
track, the component that does the work, the evidence to capture, the sentence we can say, and what is still open. How to use
each sponsor's tool during development: `devtools.md`. Deadline: Devpost Sunday 2026-10-18 12:00 PM, no late submissions;
company judges may judge from the Devpost text alone, so every track below gets its own Devpost section with the evidence linked.

## 1. The rule for claims

A claim is made only when the thing it names ran. "Minimal" is fine; "nominal" is fine; "did not happen" is not written. For
each track the table says the smallest real use that satisfies the sponsor's own requirement, so a tight night still ends with
honest sections. Where a sponsor's requirement cannot be met in time, the section says what was built and what was not.

## 2. Main track: Model Learning

**Requirement:** a system that learns, adapts and improves from data.
**Component:** the whole loop (`../DESIGN.md` §1): per call site, candidates train on the agent's own trace rows, are certified
on shadow rows, promoted, audited, revoked and replaced. **Evidence:** the `run 2,000` chart (cost per run, sec per run, failed
%, replayable %), the certificates table per site and candidate, one replay byte-identical. **Claim:** "the agent's cost, latency
and failure rate fall as it runs, every saving carries a certified error bound, every promoted decision replays byte for byte."

## 3. Company tracks

| Track | Sponsor requires | Component that does it | Evidence to capture | Claim we can make | Open |
|---|---|---|---|---|---|
| **Fetch.ai** ($1,500 / $1,000 / $500 + interview) | an Agentverse agent on the Chat Protocol, usable from ASI:One, real tool execution, the primary workflow finished inside an ASI:One chat, README with agent name + address + the two Innovation Lab badges, registration via the LA Hacks Submission Agent | `agents/bookkeeper/agent.py` (uAgent, mailbox, relays to `POST /run`), `connectors/uagents.py`; the workflow's tools are real calls on the API side; the reply names which calls were predicted and under which certificate | agent address (registered: `agent1q2eaums3hgpp2d68npndfnfdnkyskjt2kgz5ajpvdpdht5mk4lgvx9n6uuw`), a screen recording of the ASI:One chat, the README section | "a uAgent on Agentverse, reachable from ASI:One, runs the bookkeeping workflow through Dreaming; tool recovery under an injected outage is shown live" | the planner round trip (`smokes/smoke_agent.py`) once the ASI:One key has quota; multi-agent (`send_and_receive` wrapper) and Interactive Cards are bonus, not planned |
| **StepFun** (prize TBA) | the running workflow calls `step-5-preview`; an explanation of every other model and service; reviewers pointed to the call in the code | `teachers/openai_compat.py` row `stepfun` (base URL `https://api.stepfun.ai/v1`, OpenAI-compatible, `reasoning_effort="low"`; thinking cannot be switched off); `agents/bookkeeper/workflow.py` pins it as the teacher of the three typed steps; the router site's cheap config is the cheaper tier `step-3.5-flash` ($0.10/$0.30 per M vs $1.00/$2.70) | the file and line of the `TeacherSpec(provider="stepfun", model="step-5-preview")`, trace rows with `teacher.provider = stepfun`, the self-disagreement number for the certificate floor | "every typed step's teacher is step-5-preview; the students are distilled from it and certified against it; here is the call" | the key (Starter Pack voucher, 50 total); the model's self-agreement on identical inputs (S13 protocol, ~200 calls); the record pass is rate-bound at 100 RPM (≈ 1 h per 2,000-run seed) |
| **Intuit** ($200 / $100 / $50 per person) | something QuickBooks users could not get from classical ML or conventional GenAI, "System One" spirit; a realistic evaluation beating classical-ML and LLM baselines; a System-One-compatible API preferred | `agents/quickbooks`: three call sites over the hackathon dataset (`training-plan.md` §2.2); one pooled student per site trained on 70 companies, certified and tested separately on each of 15 held-out companies (forward-in-time within a company only where a company has ≥ 500 rows); the `/v1/systemone` route in `apps/api` serves any certified student in TypeSafe's wire format (state + typed questions → probabilities); Jev is a baseline row through OpenRouter | the benchmark table (coverage at budget, realized error per held-out company, cost and latency per 1,000 calls) with the customer / vendor rule, the exact cache, an LLM (if a key with quota lands) and Jev (if OpenRouter is topped up) as baselines; the API's request and response checked with `typesafe-sdk`; the disclosure of what was changed in the data | "one System One predictor per bookkeeping decision, learned from the books, with a certified error bound per customer and byte-exact replay, served on a System One-compatible API; here is where it beats the rule and the baselines, and where it does not" | the TypeSafe/OpenRouter credit for Jev, how much messiness to inject and disclose, what Intuit means by "System-One-compatible" (confirm with the mentor) |
| **Creao** ($2,000 / $1,000 / $500 + Launchpad) | a CREAO agent built on agent.creao.ai doing core work; shared run + agent ID in the README; the agent shown running in the demo; CREAO selected on Devpost; agents never start other agents; `capi_` keys server-side; runs are async | one CREAO agent, "benchmark runner": it calls our API (`POST /campaign/run {provider, site}`), waits for the cells, reads `GET /campaign/{id}`, writes the provider's benchmark row; their team said synthesize + train + evaluate counts as core use | the agent ID, one shared run link, a clip of it running in the video | "the provider benchmark is run by a CREAO agent that talks to Dreaming's API" | which category (AI-First Disruptor is the natural one); whether the agent also synthesizes the seed rows with CREAO's model (one more real use) |
| **veri.studio** ($800 split) | train (post-train or fine-tune) an open-source model on their platform; show before and after | the `finetune` candidate (bge-small unfrozen + head, the S8 v2 recipe as a standalone `train.py`) run as a `veri run-script` job, writing the same artifact format the candidate loads (`encoder/` + `head.npz`); weights downloaded, hashed, certified here like any other candidate; before = frozen head's certificate, after = the fine-tuned one's | the veri job id and log, the weights sha in our `weights` table, the two certificates side by side | "one candidate per site was trained on veri.studio; the certificate picked between it and the CPU candidates" | credits redeemed before the first job; whether the artifact downloads from a custom-script job; the GPU type it gives; confirm with their rep that run-script counts (managed SFT on Qwen3-0.6B is the fallback) |
| **Sentry** (interviews) | at least two products beyond error monitoring; judged on how observability shaped the build | `telemetry.py` Sentry sink (`SENTRY_DSN`, `sentry-sdk==2.72.0`): Tracing (FastAPI and httpx integrations, hand-made `gen_ai.*` spans per run and per routed call; no psycopg integration exists) and Logs from the JSON lines are the two qualifying products; AI-agent monitoring is the third; an uptime check on `/healthz` and Session Replay on the page are extras; no profiling on the free plan | the Sentry project's Agents view, one trace of a run, one log query, the uptime monitor, plus the dated log of what Sentry caught during the build (kept in `devtools.md` §Sentry) | "Tracing + Logs + AI-agent monitoring across the API, the agent and the trainer; here is what they caught" | installing it before the campaign so the night's failures land in it |
| **Cognition** ($5,000 Devin credits) | creativity, novelty, polish; free credits via the form | Devin builds one well-specified module with tests: the `hdc` candidate (`predictors/hdc.py`: 384 → 10,000-bit codes, integer bundles, byte-exact) from `training-plan.md` §1 and the smoke it ports | the Devin session link, the PR it opened, the commit author line, the README line "built by Devin: …" | "the HDC candidate was built by Devin from the spec in training-plan.md" | credits; whether the session finishes tonight (a second, smaller spec is ready: the `/v1/systemone` route) |
| **Paper** ($1,500 / $1,000 / $500) | design fundamentals, ease of use, personal craft | the dashboard's design in Paper (the Lucid Doodle look from `../../demo/VISUAL_INTENT.md`), exported or linked | the Paper file link, two screenshots in the Devpost | "the dashboard was designed in Paper" | the submission format (ask at the booth); whether the existing page is re-drawn in Paper or Paper is used for the new candidate rows only |
| **Gemini API** (MLH swag) | the Gemini API used inside the project | `teachers/openai_compat.py` row `gemini` (OpenAI-compatible endpoint); one benchmark row | trace rows with `teacher.provider = gemini`, the benchmark row | "Gemini is one teacher row in the provider benchmark" | the key |
| **Snowflake API** (MLH swag) | the REST API or Cortex used | a Cortex completion as one benchmark row through the same `Teacher` protocol (not OpenAI-compatible: one small class) | the benchmark row | same as Gemini | the trial; lowest priority of the model rows |
| **GoDaddy Registry** (MLH) | a domain registered with the code | the deployed app on that domain (`fly certs add`) | the URL | "live at <domain>" | registration |
| **Tiger Data** (MLH, optional) | Tiger Cloud used | the same `PostgresStore` against a Tiger service for the audit copy of each certificate's calibration rows (a zero-copy fork per certificate) | the fork id per certificate | only if done | moving the live database is not planned; an audit copy is |
| **Solana** (MLH, optional; prize: Unruggable hardware wallets) | Solana used | a devnet memo of each certificate's sha | the transaction links | only if done | not planned unless time remains |
| **Browser Use** (optional) | an agent built with Browser Use turning a goal into web action | a browser task as one slow tool in the workflow (cloud API v4; its model list is fixed, no custom base URL), whose return Dreaming learns and speculates over; the tool itself is not replayable | the run ids | only if done | not planned unless time remains |

## 4. The provider benchmark (what Creao runs, what Intuit judges, where every model row lands)

One protocol, one table. Rows tonight: StepFun `step-5-preview` on the bookkeeper sites (its record pass is the labelling
pass) and the QuickBooks sites, where the books are the teacher and no provider is needed. Every other row needs its own
labelling pass (≈ 6,000 calls at the provider's rate) and a key with quota, so it is listed as pending until it runs: ASI:One
`asi1-mini` (Pro code), OpenRouter Anthropic Haiku-5.5 (top-up), Gemini flash (key), Snowflake Cortex (trial), Jev as a
predictor baseline through OpenRouter (top-up), CREAO's model (only through a run). Columns, per call site and per provider:
- the provider's **self-disagreement** on identical inputs (the floor of every certificate it backs; S13 protocol);
- for each **candidate** trained from that provider's labels: certified coverage at 2 / 5 / 10 / 20% budgets, realized error
  above the cutoff on the held-out slice, seconds to the first certificate with seed rows only;
- **cost and latency per 1,000 calls** with and without Dreaming (teacher dollars avoided minus embedding compute; p50 and p95);
- for the Intuit sites: the **vendor rule**, the **LLM alone** and **Jev** as baselines on the same held-out companies.
Every number comes from `train_runs` and `checkpoints` (`training-plan.md` §4) so it can be recomputed at any budget later.

## 5. The video and the Devpost, in one pass

One 3:00 video serves Devpost (2–3 min) and Fetch (3–5 min): run 2,000 → budget slider → unplug → replay → the ASI:One chat →
the benchmark table → the CREAO agent running. The Devpost description has one section per track above, each with its evidence
links, and the README carries the Fetch agent block, the CREAO agent ID and run, the Devin module line, the StepFun call pointer.

## 6. Registrations (who clicks what)

The one ordered list is in `README.md` here (Modal card, StepFun, OpenRouter top-up, ASI:One Pro code, veri, CREAO, Devin,
Sentry, the cheap adds, Devpost and the Fetch Submission Agent).
