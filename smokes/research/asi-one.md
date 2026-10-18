# ASI:One / Agentverse / uAgents integration (researched 2026-10-17)

All facts fetched 2026-10-17 from the URLs given. Nothing below was executed locally; the example is assembled from docs patterns.

## Summary

- **Stack is current and permissive.** `uagents` 0.26.1 (PyPI, 2026-10-08), `uagents-core` 0.4.12 (2026-10-13), both Apache-2.0, Python >=3.10 (README says 3.10-3.13). Releases land every 1-4 weeks, so pin versions.
- **ASI:One discoverability = Agent Chat Protocol (ACP)** from `uagents_core.contrib.protocols.chat` (ChatMessage / ChatAcknowledgement / TextContent / StartSessionContent / EndSessionContent), plus `agent.include(chat_proto, publish_manifest=True)`, plus the agent being registered and *running* on Agentverse.
- **Three ways to connect:** (a) local agent + Agentverse **mailbox** (no public endpoint; the connect step is a click in the local Inspector web UI), (b) own server with public HTTPS endpoint registered by script (`register_chat_agent`, fully scriptable), (c) **hosted** agent whose code lives on Agentverse (restricted imports, wrong fit for our runtime).
- **Human sign-in is required once, for keys.** Agentverse API key = logged-in UI only (Profile > New API Key; docs say login with a Gmail address). ASI:One API key = asi1.ai > Developer > Create New. No documented programmatic key creation. After the keys exist, registration and runtime are scriptable.
- **ASI:One LLM API is OpenAI-compatible:** base `https://api.asi1.ai/v1`, models `asi1`, `asi1-mini`, `asi1-ultra`. Prices per 1M tokens (in/out): $0.50/$1.90, $0.18/$0.45, $1.26/$3.96 (docs.asi1.ai). No free tier or credit amount documented; rate limits "scoped to your plan", unpublished.
- **No on-chain fee needed for the basic path.** Default registration posts a signed attestation to the Almanac API; ledger (on-chain) registration only runs if a contract is configured. Testnet funding is automatic; mainnet needs real FET only for the contract step.
- **Hackathon rules found are per-event, not a generic "ASI Alliance track".** Common eligibility: register >=1 agent on Agentverse, implement ACP, be usable via ASI:One, whole workflow inside an ASI:One chat (no custom frontend), public GitHub repo. Judging is 25% functionality, 20-25% Fetch tech use, 20% innovation, 20% impact, 10-15% UX. Latest found (MHacks, Oct 3-4 2026) already passed; no later event found.
- **Integration estimate for a Python runtime:** about 1 hour for an echo agent answering in ASI:One, 3-5 hours to wrap the Dreaming runtime behind the chat handler, plus 0.5 day for demo/README/video polish. Not on the critical path for missions 1-7.

## Findings

### uAgents framework
- Repo `fetchai/uAgents`, Apache-2.0, ~1.6k stars, `python/` holds `uagents` and `uagents-core`. https://github.com/fetchai/uAgents
- `uagents` 0.26.1 deps: aiohttp, cosmpy, cronsim, pydantic>=2.8, uagents-core>=0.4.11, uvicorn; `litellm` only via `uagents[llm]`/`[all]`. https://pypi.org/pypi/uagents/json
- `Agent.__init__` (main branch): `port` default 8000, `mailbox: bool=False`, `proxy: bool=False`, `network` default **"mainnet"**, `publish_agent_details=True`, `readme_path`, `description`, `handle`, `enable_agent_inspector=True`, `mark_inactive_on_shutdown=True`. https://raw.githubusercontent.com/fetchai/uAgents/main/python/src/uagents/agent.py
- Identity is deterministic from `seed`; same seed => same `agent1q...` address. Without a seed a key lands in `private_keys.json`. Treat seeds as secrets.

### Chat Protocol (what ASI:One speaks)
- Canonical tutorial pattern: `chat_proto = Protocol(spec=chat_protocol_spec)`, handlers for `ChatMessage` and `ChatAcknowledgement`, `agent.include(chat_proto, publish_manifest=True)`. https://uagents.fetch.ai/docs/examples/asi-1 and https://docs.asi1.ai/documentation/tutorials/agent-chat-protocol.md
- Hard rules from the Fetch skill file: ack first, iterate `msg.content` (never `[0]`), tz-aware `datetime.now(timezone.utc)`, fresh `uuid4()` per message, handle unknown content without crashing, reply via `TextContent`, append `EndSessionContent` to close. https://cdn.jsdelivr.net/npm/fetch-skills@2.0.3/skills/chat-protocol/SKILL.md
- MHacks hackpack warns its own sample code is broken (missing imports, indentation, deprecated `utcnow`). Do not copy it. https://innovationlab.fetch.ai/events/hackathons/mhacks-2026/hackpack
- Users can force routing by @mentioning an address in the ASI:One prompt (`@agent1q... do X`). https://docs.asi1.ai/documentation/tutorials/agent-chat-protocol.md
- Discovery otherwise is ASI:One keyword + semantic matching over agent keywords/README ("Search Visibility" tab). https://docs.agentverse.ai/documentation/launch-agents/external-agents/u-agents

### Registration modes
| Mode | Needs | Human step | Notes |
|---|---|---|---|
| Mailbox (`mailbox=True`) | seed, running process, Agentverse account | Run agent, open Inspector URL, Connect > Mailbox > Finish, restart | Agent may be offline; messages queue. Chrome v142+/Brave need Local Network Access permission for Inspector. Each local agent needs its own mailbox. https://uagents.fetch.ai/docs/agentverse/mailbox (updated 2025-07-02) |
| Public endpoint + script | public HTTPS URL, API key, seed | Agentverse "Evaluate Registration" click in UI flow | Tunnel e.g. `cloudflared tunnel --url http://localhost:8000`. https://docs.agentverse.ai/documentation/launch-agents/agentverse-sdk/fast-api |
| Programmatic | API key + seed + endpoint | None after key | `uagents_core.utils.registration.register_chat_agent(name, endpoint, active, credentials, track_interactions=True, description, readme, starter_prompts, ...) -> bool`; credentials = `RegistrationRequestCredentials(agentverse_api_key, agent_seed_phrase)`. `track_interactions=True` routes via Agentverse proxy endpoint. https://raw.githubusercontent.com/fetchai/uAgents/main/python/uagents-core/uagents_core/utils/registration.py |
| Hosted on Agentverse | API key (REST) or UI | UI or REST | `POST /v1/hosting/agents`, `PUT .../code`, `POST .../start` (tutorial; may be out of date). Globals reset per invocation, imports whitelisted, one agent per project. https://docs.agentverse.ai/documentation/launch-agents/hosted-agents , https://fetch.ai/docs/guides/apis/agent-function-creation-apis |

- Mailbox client authenticates to Agentverse with a `user_token` at connect time (identity challenge/proof), then polls with an agent-signed attestation (poll interval 1.0 s). https://raw.githubusercontent.com/fetchai/uAgents/main/python/src/uagents/mailbox.py
- Registration endpoint flow (all Bearer API key): GET agent, GET identity challenge, POST identity proof, POST agent details. `update_agent_status` needs only the agent signature. (registration.py above)

### Keys and accounts
- Agentverse API key: login (Gmail documented) > Profile > "+ New API Key" > access "write" to all resources > Generate. Shown once. One mirror page says permission lasts 30 days (unverified expiry). https://agentverse.ai/docs/apitokens , https://innovationlab.fetch.ai/resources/docs/1.0.4/agentverse/agentverse-api-key
- ASI:One API key: sign up at asi1.ai > Developer section > Create New; manage at asi1.ai/dashboard/api-keys; shown once. https://docs.asi1.ai/documentation/getting-started/quickstart.md , https://innovationlab.fetch.ai/resources/docs/asione/asi1-getting-started
- Env var names used in Fetch examples: `ASI_ONE_API_KEY`, `AGENTVERSE_KEY` (older SDK), `AGENT_SEED_PHRASE`.

### ASI:One LLM API
- OpenAI Chat Completions + Responses API compatible; extras: `web_search` (via `extra_body` in Python), `x-session-id` header, Planner mode, response fields `executable_data`, `intermediate_steps`, `thought`. https://innovationlab.fetch.ai/resources/docs/asione/build/openai-compatibility , https://docs.asi1.ai/documentation/models
- Models page (newest, docs.asi1.ai): asi1 409,600 ctx; asi1-ultra 1,048,576 ctx; asi1-mini 262,144 ctx; all support streaming, tool calling, image input, structured data. Older innovationlab pages disagree (say only `asi1` exists; other context sizes): treat docs.asi1.ai as authoritative.
- Errors: 429 = plan limit used (backoff+jitter); 402 = account cannot be billed, different body shape. https://docs.asi1.ai/documentation/build-with-asi-one/errors.md
- Usage and balance shown in an "ASI:One account wallet"; no public free-credit or top-up terms found. https://docs.asi1.ai/documentation/models

### Almanac / funding / network
- Registration policies: `AlmanacApiRegistrationPolicy` (signed POST, no funds), `LedgerBasedRegistrationPolicy` (on-chain, needs funds), `DefaultRegistrationPolicy` (API first, ledger only if contract provided). https://raw.githubusercontent.com/fetchai/uAgents/main/python/src/uagents/registration.py
- Testnet: funds auto-added at startup; `TESTNET_REGISTRATION_FEE = 5e17` base units (0.5 test FET). Mainnet: log asks you to send FET; contract step fails with "insufficient funds" but Almanac API registration still succeeds. https://raw.githubusercontent.com/fetchai/uAgents/main/python/src/uagents/config.py , https://uagents.fetch.ai/docs/getting-started/tokens (updated 2025-07-02)
- Mainnet contract fee amount not found. `fund_agent_if_low(str(agent.wallet.address()))` from `uagents.setup` is the helper used in the Fetch skill file.

### Quotas (Agentverse)
- Free: 8 hosted + 8 local agents, 60K compute-seconds, 5 MB storage, 1 GB message bytes, 10K processed messages. Premium $25/mo: 25 agents, 500K s, 100 MB, 1M msgs. Free-tier mailbox deactivates if you do not log in to Agentverse for 30 days. https://docs.agentverse.ai/documentation/advanced-usages/agentverse-subscriptions-and-quotas

### Hackathon rules (event-specific; no generic "ASI Alliance track" page found)
- MHacks 2026 "ASI:One Agent Challenge", Oct 3-4 2026 (passed). Eligibility: Agentverse agent + ACP + ASI:One usable + meaningful tool use/orchestration + whole workflow in ASI:One chat + public GitHub. Weights: functionality 25, Fetch tech 20, innovation 20, impact 20, UX 15. Prizes $1,250/$750/$500. Submission: Devpost, submission agent registration, README with agent names+addresses and badges `tag:innovationlab` `tag:hackathon`, 3-5 min video. Bonus: Payment Protocol, multi-agent, error recovery, ASI interactive cards. https://innovationlab.fetch.ai/events/hackathons/mhacks-2026/hackpack
- UC Berkeley AI Hackathon, Jun 20-21 2026 (passed). Weights 25/25/20/20/10. Also needs public shared ASI:One chat URL and Agentverse profile URL(s). "Integration central, not added for eligibility." Prizes $1,500/$1,000/$500. https://innovationlab.fetch.ai/events/hackathons/uc-berkeley-ai-hackathon-2026/hackpack
- Other 2026 hackpacks exist (TreeHacks, Hack@Brown, ProduHacks, UK AI Agent Hack Ep5); rubrics not read. Events index: https://innovationlab.fetch.ai/events

### Minimal working example (not executed; assembled from the docs above)
```python
# pip install "uagents==0.26.1" "uagents-core==0.4.12" openai
import os
from datetime import datetime, timezone
from uuid import uuid4
from openai import OpenAI
from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement, ChatMessage, EndSessionContent, StartSessionContent,
    TextContent, chat_protocol_spec)

llm = OpenAI(api_key=os.environ["ASI_ONE_API_KEY"], base_url="https://api.asi1.ai/v1")
agent = Agent(name="dreaming-demo", seed=os.environ["AGENT_SEED_PHRASE"], port=8000,
              mailbox=True, publish_agent_details=True, readme_path="README.md")
proto = Protocol(spec=chat_protocol_spec)

@proto.on_message(ChatMessage)
async def on_chat(ctx: Context, sender: str, msg: ChatMessage):
    await ctx.send(sender, ChatAcknowledgement(
        timestamp=datetime.now(timezone.utc), acknowledged_msg_id=msg.msg_id))
    text = " ".join(c.text for c in msg.content if isinstance(c, TextContent))
    if not text:                       # StartSessionContent etc.
        return
    r = llm.chat.completions.create(model="asi1-mini",
            messages=[{"role": "user", "content": text}])
    await ctx.send(sender, ChatMessage(timestamp=datetime.now(timezone.utc), msg_id=uuid4(),
        content=[TextContent(type="text", text=r.choices[0].message.content),
                 EndSessionContent(type="end-session")]))

@proto.on_message(ChatAcknowledgement)
async def on_ack(ctx: Context, sender: str, msg: ChatAcknowledgement): pass

agent.include(proto, publish_manifest=True)
if __name__ == "__main__":
    agent.run()   # then open Inspector URL in terminal > Connect > Mailbox
```
Note: the `openai` call is synchronous and blocks the event loop; use `AsyncOpenAI` for real use. Replace the LLM call with a call into the Dreaming runtime (the point of the demo).

## Reusable components

| name | url | license | why |
|---|---|---|---|
| uAgents (`uagents`) | https://github.com/fetchai/uAgents | Apache-2.0 | Agent runtime, mailbox client, Almanac registration, Inspector |
| uagents-core | https://pypi.org/project/uagents-core/ | Apache-2.0 | Chat Protocol models, `register_chat_agent`, `Identity`, envelope/message utils; usable without the full uagents loop (FastAPI path) |
| innovation-lab-examples | https://github.com/fetchai/innovation-lab-examples | Apache-2.0 per README (GitHub sidebar says MIT; unresolved) | Working examples: `asi1-llm-example/`, `mcp-agents/`, `openai-agent-sdk/`, `Crewai-agents/`, `dynamic-ui-on-asi1/` (interactive cards) |
| ASI:One OpenAI-compatible API | https://docs.asi1.ai/documentation/build-with-asi-one/openai-compatibility.md | commercial API | Stock `openai` SDK with `base_url`; also usable as a cheap "frontier" step provider |
| Agentverse MCP servers | https://mcp.agentverse.ai/sse and https://mcp-lite.agentverse.ai/mcp | n/a (hosted) | Create/manage agents from Claude Code/Desktop; listed in hackpack |
| fetch-skills (npm) | https://cdn.jsdelivr.net/npm/fetch-skills@2.0.3/skills/chat-protocol/SKILL.md | unverified | Concise, correct ACP rules; feed to a coding agent |
| Agentverse hosting REST API | https://agentverse.ai/docs/api/hosting | n/a | Scripted create/upload/start of hosted agents (docs partly stale) |

## Numbers

| item | value | date | source |
|---|---|---|---|
| asi1 price | $0.50 in / $0.05 cached / $1.90 out per 1M tokens | 2026-10-17 | https://docs.asi1.ai/documentation/models |
| asi1-mini price | $0.18 / $0.018 / $0.45 per 1M | 2026-10-17 | same |
| asi1-ultra price | $1.26 / $0.126 / $3.96 per 1M | 2026-10-17 | same |
| 1M-token demo budget | 500k in + 500k out on asi1-mini = about $0.31; on asi1 about $1.20 | derived | derived from above |
| Agentverse free tier | 8 hosted + 8 local agents, 60K compute s, 5 MB storage, 10K msgs | page undated | https://docs.agentverse.ai/documentation/advanced-usages/agentverse-subscriptions-and-quotas |
| Agentverse Premium | $25/month | page undated | same |
| Testnet almanac fee | 0.5 test FET, auto-funded | 2026-10-17 (main branch) | uagents `config.py` |
| Mailbox poll interval | 1.0 s | 2026-10-17 | uagents `config.py` |
| Message history retention | 86,400 s, 1,000 msgs | 2026-10-17 | uagents `config.py` |
| Almanac registration refresh | every 3600 s | 2026-10-17 | uagents `config.py` |
| ASI:One rate limit / free credits | not published | 2026-10-17 | https://docs.asi1.ai/documentation/build-with-asi-one/errors.md |

## Blockers / plan changes

- **One human-only step before any smoke:** create the Agentverse API key (Google/Gmail login documented) and the ASI:One API key. Budget 15 minutes of the user's time; an agent cannot do this.
- **Mailbox connect needs a human with a browser** (Inspector click). If the runtime is headless (RunPod, CI), use the programmatic path: public endpoint (tunnel or pod port) + `register_chat_agent`, or run the agent locally on the laptop and connect the mailbox once; the seed keeps the address stable afterwards.
- **The agent must be running for ASI:One to answer.** For the demo/judging window, keep a stable host (laptop is fine if mailbox; RunPod CPU pod otherwise). Free-tier mailbox dies after 30 days without an Agentverse login.
- **Hackathon fit:** "Fetch.ai integration central, not added for eligibility" is scored (20-25%). Plan the Dreaming chat agent so the ASI:One chat IS the control surface (e.g. "run bookkeeping episode", "show certified predictors / savings", "replay episode hash"), not a bolt-on. Whole workflow must complete in the chat; no custom frontend.
- **Pin versions.** `uagents`/`uagents-core` release often and docs lag (several doc pages contradict each other and the source). Trust source + docs.asi1.ai over innovationlab mirrors.
- **`network` defaults to "mainnet"** in current source. Harmless for API-based registration, but logs an insufficient-funds warning on the contract step; set `network="testnet"` for dev to get auto-funding.
- **Hosted agents are not suitable** for Dreaming (restricted imports, per-call global reset, small quotas). Keep the runtime external.
- No change to missions 1-7: the Fetch connector is a thin ACP wrapper around the same runtime entry point. Do it last.

## Unverified

- Whether Agentverse/asi1.ai sign-up supports anything besides Google (docs only mention Gmail).
- ASI:One free credits, wallet top-up method (card vs FET), and actual rate limits.
- Agentverse API key expiry (one mirror page says "30 days").
- Mainnet Almanac contract fee amount; which registration policy a plain `Agent(mailbox=True)` uses by default.
- Clean `pip install uagents` on Windows + Python 3.13/3.14 (README states 3.10-3.13; cosmpy wheel availability not checked).
- ASI:One's timeout for waiting on an agent reply; ASI:One planner behavior when several agents match.
- Hosted-agent allowed-imports list; `agentverse_init(AGENT_URI, mailbox=True)` helper (mentioned on a docs page, package not identified).
- License of `fetch-skills`; GitHub sidebar vs README license conflict on innovation-lab-examples.
- Which Fetch hackathon (if any) the team is targeting; no event after Oct 4 2026 was found. Payment Protocol and ASI interactive cards not researched.
- Example code above was not run; it was cross-checked against the tutorial page and the Fetch skill file rules, not executed.
