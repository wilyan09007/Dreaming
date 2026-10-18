# Dreaming — Tracks and Sponsors

Companion to `missions.md`. What we enter, what each sponsor requires, and where each fits Dreaming.

Requirements, prizes and rules below are facts from the sponsor briefs, the LA Hacks Discord, and the opening and pitching slides (checked 2026-10-17). How we use each product is kept general on purpose: the specific design is settled by experiment. Anything not yet decided is listed as open.

## Event

- LA Hacks AI Hackathon 2026, Oct 17–18, James West Alumni Center, UCLA. Teams of up to 4; the whole team must be there in person.
- **Entry:** one main track, plus any number of company challenges. The Agentverse track also requires a main track.
- **Submission:** Devpost, one per team, due Sunday 12:00 PM, no late submissions. It needs a project description, a public GitHub repo, a 2–3 minute demo video as a YouTube link, and every teammate added with the email they applied with.
- **Company judges may decide from the Devpost alone**, so each challenge gets its own section in the description.
- **Judging, Sunday:**
  - Science fair 12:30–1:45 PM: Elo-ranked; 2 min pitch + 3 min Q&A per judge group, many times.
  - Top 5 on stage 2:15–3:00 PM: 3 min demo + 4 min Q&A, majority vote.
  - Closing ceremony 3:15 PM.
- **Best Overall:** $2,000 / $1,500 / $1,000. Each track prize goes to the best project in that track outside the overall top 3.

## Main track: Model Learning

"Build systems that learn, adapt, and improve from data." Dreaming is that system. It trains predictors from an agent's own call log, certifies them against reality, and promotes them. The before/after is how cost, latency, failures and the replayable share move as runs accumulate. Track prize: AirPods 4.

## Where each sponsor fits

| Role | Sponsors |
|---|---|
| Where the agent runs and how users reach it | Fetch.ai |
| Models the workflow calls, compared in one benchmark | StepFun (required at runtime), ASI:One, Gemini, Snowflake, Jev, CREAO's model |
| Runs the benchmark | Creao |
| Evaluation data and API format | Intuit |
| Trains an open model | veri.studio |
| Observes the system | Sentry |
| Builds code | Cognition |
| Designs the UI | Paper |
| Audit trail (optional) | Tiger Data, Solana |
| A slow tool to learn (optional) | Browser Use |
| Domain | GoDaddy Registry |

Hosting is Fly.io, which is not a sponsor.

## Constraints that hold across sponsors

- **One live teacher per call site.** A certificate names the model it was checked against, so changing a site's teacher sends it back through the gate. StepFun's rule makes `step-5-preview` the natural live teacher, once its self-agreement and latency are measured. Other providers are compared in the benchmark.
- **Fetch's main workflow must finish inside ASI:One chat**, so nothing in it can depend on the dashboard.
- **One database is the source of truth.** Nothing is split across two.
- **One demo video of about 3:00** covers both Devpost (2–3 min) and Fetch (3–5 min).

## The provider benchmark

One protocol, run with each model or agent provider in turn: the same call sites, the same held-out data, the same error budgets. Each row reports:
- how much of that provider's calls Dreaming can answer, and at what certified error
- cost and latency with and without Dreaming

A teacher's disagreement with itself bounds every certificate (smokes S13), so it is measured first for each provider. Call sites, sample sizes and budgets are set when the harness is built.

## Sponsors

Each entry gives what the sponsor requires (fixed), how it fits (general), and what is still open.

### Fetch.ai: $1,500 / $1,000 / $500, each with an internship interview

- **Requires:**
  - at least one agent registered on Agentverse, implementing the Agent Chat Protocol
  - discoverable and usable through ASI:One
  - real tool execution or multi-agent orchestration
  - the primary workflow completed inside an ASI:One conversation
  - a public repo whose README lists each agent's name and address plus the two Innovation Lab badges
  - registration through the LA Hacks ASI:One Submission Agent, in addition to Devpost
- **Judging:** functionality 25%, use of Fetch technology 20%, innovation 20%, real-world impact 20%, UX and presentation 15%.
- **Bonus:** multi-agent collaboration; the Payment Protocol with a credible monetization model; ASI1 Interactive Cards; recovery from failed tool calls; an agent that keeps running after the event.
- **Fit:** the demo agent is a uAgent on Agentverse, reached from ASI:One. Dreaming wraps its calls (mission 6). Its learned tool and agent-to-agent predictions are what let it keep working when a tool fails. ASI:One's models are one row of the benchmark.
- **Open:** how far to take multi-agent work, Interactive Cards and the Payment Protocol.
- **Known limits:**
  - ASI:One rate-limited us at three live calls per run.
  - Plain ASI:One chat may not render the payment checkout card.
- **Codes:** ASI:One Pro `LAHACKSAI26`, Agentverse Premium `LAHACKSAI26AV`.

### Intuit: $200 / $100 / $50 gift card per person

- **Challenge:** build something QuickBooks users could not get from classical ML or conventional GenAI, in the spirit of "System One" models (fast, specialized, efficient).
- **Judging:**
  - a meaningful technical unlock
  - something previously impossible, or too slow or costly
  - clear customer value
  - a realistic evaluation that beats classical-ML and LLM baselines, expected to be a large part of the work

  A System-One-compatible API is preferred.
- **Data:** [QuickBooks-Hackathon-2026](https://huggingface.co/datasets/CL251215/QuickBooks-Hackathon-2026).
  - 100 synthetic companies over one year, with linked bank lines, bills, invoices, payments and accounts.
  - Vendor names are clean, and there are no anomaly or correction labels. Modifications are allowed if disclosed.
  - Vendors repeat heavily: a per-company vendor rule already gets about 97% on bill → account. So hold out by company.
- **System One:** TypeSafe's term. Their model, Jev, is served at `POST /v1/systemone`: a state plus typed questions in, probabilities out, in one forward pass. Ollaya serves the same API locally. Pin the model version.
- **Fit:** Dreaming's students are System One models trained from a customer's own traffic, with a certificate.
  - Intuit's data is the evaluation set.
  - Their API format is a way to serve the students.
  - Jev is a baseline.
- **Open:**
  - which call sites to evaluate
  - which baselines beyond the vendor rule, an LLM and Jev
  - whether the live demo also runs on this data

### StepFun: prize to be announced

- **Requires:**
  - the running app or workflow calls `step-5-preview`
  - an explanation of what every other model and service does alongside it
  - reviewers pointed to the call in the code
- **Judging:** task completion, reliability, and whether users can understand and control the workflow.
- **API:** OpenAI-compatible at `https://api.stepfun.ai/v1`. $25 vouchers come from the StepFun Hacker Starter Pack (50 in total, first come).
- **Fit:** called inside the running workflow, most naturally as the teacher (see Constraints). Also a benchmark row, and possibly a data synthesizer or judge.

### Creao: AI-First Disruptor $2,000, One-Person Company $1,000, Team Empowerment $500, plus CREAO Launchpad

- **Requires:**
  - a CREAO agent, built on agent.creao.ai, doing core work
  - a shared run and the agent ID in the README
  - the agent shown running in the demo
  - the CREAO track selected on Devpost
- **Their rules:** agents must not start other agents on their own; keep `capi_` keys server-side; runs are async and take minutes.
- **Judging:** AI as foundation 30%, depth on CREAO 25% (extending the agent with your own tools and APIs counts), originality 20%, track fit 15%, demo 10%.
- **Told to the team on Oct 17:** an agent that synthesizes data with their model, trains and evaluates counts as core use.
- **Fit:** a CREAO agent runs the provider benchmark: it synthesizes data, trains, evaluates and reports. It talks to Dreaming's API, never to other agents.
- **Open:**
  - which category to enter
  - whether training runs in its sandbox or through our API
  - whether scheduled runs are included on the hackathon plan

### veri.studio: $800, split across the team

- **Challenge:** train an open-source model on their platform (post-train or fine-tune it to be better at a task) and show before and after.
- **Credits:** from their rep in #veri-studio. Redeem under settings → billing before the first job: jobs started at a $0 balance are stopped. Install with `uv tool install --upgrade 'veri-sdk[mcp]'`, then `veri login`.
- **Platform:**
  - managed SFT, DPO and GRPO, aimed at text-generation models
  - any PyTorch script via `veri run-script`
  - weights can be downloaded
- **Fit:** fine-tune an open model for one of Dreaming's predictors. The training must happen on veri to count. Publishing the weights is optional.
- **Open:** which model and which call site.
  - Typed steps favour a small encoder (smokes S8, S22, S26: a 33M encoder beat every 0.5B decoder).
  - A free-text step would need a small decoder.

### Sentry: a guaranteed interview for each member of the winning team

- **Requires:** at least two Sentry products beyond error monitoring: Session Replay, Logs, Tracing, Profiling, Uptime, or MCP / AI agent monitoring.
- **Judged on** how observability shaped the build. Sentry is virtual this weekend, so the Devpost section carries it.
- **Fit:** instrument the API, the agent and the trainer early. Cover every wrapped call and who answered it, and keep a log of what Sentry caught during the build. Postgres stays the system of record, since Sentry spans are sampled.
- **Open:** which products beyond the first two.

### Cognition: $5,000 of Devin credits (one winner)

- **Judging:** creativity, novelty, polish. Free credits through the form in #cognition.
- **Fit:** Devin builds well-specified parts of the code, and the README says what it built.

### Paper: $1,500 / $1,000 / $500, Best Designed Hack

- **Judging:**
  - design fundamentals: hierarchy, legibility, consistency, contrast
  - ease of use
  - personal craft over "clearly AI-generated" design
- **Code:** Paper Pro is free for a month with `LAHACKS26`. It renews unless cancelled.
- **Submission:** the format is not stated; ask at their booth.
- **Fit:** the dashboard and the agent's UI are designed in Paper.

### Tiger Data (MLH): Stream Deck Mini

- **Credits:** $1,000 for new signups (mlh.link/tigerdata).
- **Product:** Postgres with hypertables, continuous aggregates, compression, and zero-copy forks.
- **Fit (optional):** the audit side of the trace store, such as a frozen snapshot of each certificate's calibration rows.
- **Caution:** adopting it means moving the database. Supabase Realtime currently feeds the dashboard.

### Solana (MLH): Unruggable hardware wallets (per the MLH prize page, 2026-10-17)

**Fit (optional):** publish each certificate's fingerprint so it can be shown unchanged later. No tokens.

### Browser Use: prize not announced

- **Credits:** $250 of test credits from their rep; sign up at cloud.browser-use.com first.
- **Challenge:** an agent built with Browser Use that turns a goal into action on the web.
- **Fit (optional):** a browser task as a slow tool in the workflow, whose results Dreaming learns to predict. The cloud API (`api.browser-use.com/api/v4`, header `X-Browser-Use-API-Key`) takes a fixed model list; only the open-source library accepts an OpenAI-compatible base URL (checked 2026-10-17). Runs are not deterministic.

### Gemini API and Snowflake API (MLH): swag kits; Arduino kits

- **Gemini:** the prize is for the Gemini API used inside the project, not the chat assistant. Free tier; OpenAI-compatible endpoint.
- **Snowflake:** the REST API and Cortex, with a 120-day student trial.
- **Fit:** providers in the benchmark; possibly synthesis or judging.

### GoDaddy Registry (MLH): digital gift card

Register a domain through GoDaddy Registry with code `MLHLHA2685`. It becomes the domain for the deployed app.

## Registrations

- [ ] StepFun voucher (first come)
- [ ] veri.studio credits, redeemed before the first job
- [ ] CREAO account
- [ ] Devin credits form
- [ ] Sentry account
- [ ] TypeSafe key for Jev
- [ ] Gemini key and Snowflake trial
- [ ] ASI:One and Agentverse promo codes
- [ ] Paper Pro code
- [ ] GoDaddy domain
- [ ] Devpost project with every teammate
- [ ] Optional: Browser Use credits, Tiger Data credits

## Not entering

Visa, Workday, Base44, Photon, Even Realities, QNX, Espressif, and the MLH ElevenLabs and Vultr prizes. AWS and Annapurna Labs have no challenge.

## Sources

- **Event:** LA Hacks Discord (#announcements, #ask-an-organizer, sponsor channels); [opening slides](https://docs.google.com/presentation/d/1nMyGE5odyLhMVIDoKPlKW2nW3kD7QllUoN-Wst7tDGs/edit); [pitching and submission slides](https://docs.google.com/presentation/d/1d5z3RdWRorg9HMZR6TZ6xAdqgDJ2HxhlTnrLlpqTpvE/edit).
- **Fetch.ai:** [hackpack](https://www.fetch.ai/events/hackathons/la-hacks-ai-hackathon-2026/hackpack); [submission agent guide](https://docs.google.com/document/d/1UDW-X1C24hxZviFOQzjTeh0pXRNAoflMb8lhJqZP9Z0/edit).
- **StepFun:** [starter pack](https://docs.google.com/document/d/1XAfIQC6vhA3WRy_SyCd2774esPRovuQVZHkL-OC7kRY/edit).
- **Creao:** [UCLA hackathon page](https://community.creao.ai/AI_hackathon_UCLA2026); [docs](https://docs.creao.ai/).
- **veri.studio:** [LA Hacks page](https://veri.studio/lahacks); [docs](https://docs.veri.studio/).
- **Intuit / System One:** [Jev access guide](https://systemonemodels.org/guides/how-to-get-jev-access/); [Ollaya](https://github.com/ollaya-dev/ollaya).
- **Sentry:** [AI agent monitoring](https://docs.sentry.io/product/insights/ai/agents/).
- **Paper:** paper.design/la-hacks.
- **Browser Use:** browser-use.com.
- **MLH:** mlh.link/ucla-ai.
