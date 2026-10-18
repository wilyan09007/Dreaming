# S11 - ASI:One / Agentverse connector feasibility (run 2026-10-17, revised the same day)

Verdict (corrected after review): **the plumbing works through the API with only the two keys, no UI click, no money.** ASI:One's API planner (`planner_mode:true`, optionally `agents:[...]`) routed a prompt to our agent, the agent answered, and its reply with a certificate line came back in the completion. This holds in both mailbox mode (no tunnel) and tunnel mode (`register_chat_agent`).

What is **not** proven, and matters for the demo:
1. The product path is a human typing in the asi1.ai chat UI with the Agents toggle (or an `@agent1q...` mention). That needs a browser login and was never run. Everything here goes through the API.
2. Discovery by a stranger's vague prompt is a coin flip. Nameless prompts found our agent in 2 of 3 differently worded tries (Q3 rows 5a-5c). One wording ("auditable receipt") was not routed at all.
3. Slow steps are lost. A reply after 25 s arrived; a reply after 90 s did not (`[NO RESPONSE RECEIVED]`), and an immediate ack or a progress message does not help (Q3 rows 8-10).

No blocker for building the connector, but two design constraints follow from 2 and 3 (Q4). Stack: Python 3.12.12 (Windows), `uagents==0.26.1`, `uagents-core==0.4.12`, cloudflared 2026.10.1 (quick tunnel). Time and spend are in the tables below.

## Agents (all left registered; nothing running)

| name | address | where | state at end of revision |
|---|---|---|---|
| dreaming-smoke-hosted | agent1qgulmx22hrwhmnzrt28cwfdx48kn8qv50fe35tp9ywn8k44d2tqkvt2ef7e | Agentverse hosted | stopped (`running:false`, search status `inactive`). Its `ASI1KEY` secret was overwritten with a placeholder (see Q1) |
| dreaming-smoke-local | agent1qtz8j3j97yxgeaktlvdhaff2sy89ux3nyhpxeacxtg0fezzhmpypxcaywnq | this laptop | process stopped. Almanac record: type `mailbox`, `status: active`, expiry 2026-11-16T09:45Z. A signed `status=False` update returns 2xx but the status stays `active` (Almanac GET and Agentverse search both read back `active`), so the only real "off" is the process being down; mailbox messages queue meanwhile |
| dreaming-smoke-client | agent1q2d5t6nc83wzz96nnu0u0yssw0u52uyhlnxdajaasc8t9lp852u7652pmk9 | local test client, NOT an Agentverse agent | process exits after one message. Its Almanac testnet entry is still `active` (endpoint `http://127.0.0.1:8001/submit`, expiry 2026-10-24); harmless, and the same `status=False` call had no effect on it either |
| ASI:One's own sender address (seen in logs) | agent1q0upkvglmcayp2sdt9eymxy26vuc2rggaua32rfgkenjgk7ns87hcalccns | ASI:One | n/a |

Only 2 agents exist in the Agentverse account (hosted + local). The local agent was first registered in tunnel mode (`register_chat_agent`, Almanac type `proxy`) and then flipped to mailbox with the same seed (same address). It was flipped to tunnel and back once more during the revision; the final record type is `mailbox`. Seeds are in `seeds.txt` (SECRET, do not commit or share). A leak scan over every file in this folder found none of the four .env keys and no seed outside `seeds.txt`.

## Q1 - Hosted path (REST only, AGENTVERSEKEY)

Fully programmatic: create, secret, upload, start, logs, stop all work with just the API key. Hosted code called ASI:One (`requests` import allowed; the key was stored as a hosted secret and appeared as an env var).

| step | request | status | notes |
|---|---|---|---|
| list | `GET /v1/hosting/agents` | 200 | `{"items":[],"next_cursor":null}` |
| create | `POST /v1/hosting/agents` `{"name":"dreaming-smoke-hosted"}` | 200 | returns `address`, `wallet_address`, `running:false` |
| set secret | `POST /v1/hosting/secrets` `{"address","name":"ASI1KEY","secret":"..."}` | 200 | response ECHOES the secret value (the harness redacts it from log.txt); `GET /v1/hosting/secrets?address=` returns it masked `***` |
| delete secret | `DELETE /v1/hosting/secrets?address=&name=`; same path with a JSON body; `DELETE /v1/hosting/secrets/ASI1KEY?address=` | **405**; **405**; **404** | no delete route found. Workaround used: `POST` the same name with a placeholder value, 200, timestamp changed, so the real key no longer sits on Agentverse. Re-POST the real key before using the hosted agent's `llm` command again |
| upload code, docs shape | `PUT /v1/hosting/agents/{a}/code` `{"code":[{id,name,value,language}]}` | **422** | `string_type`: the docs example (array) is stale |
| upload code, working shape | same, `{"code": json.dumps([{id:0,name:"agent.py",value:<src>,language:"python"}])}` | 200 | `code` must be a JSON-encoded STRING; returns `{"digest"}` |
| start | `POST /v1/hosting/agents/{a}/start` | 200 | `running:true`; compiled and up about 14 s later (started again in the revision and answered within 20 s) |
| logs | `GET /v1/hosting/agents/{a}/logs/latest` | 200 | newest first; entries have `log_type` system/execution; `ctx.logger.info` lines show as `execution`; shows "Registered to Almanac fast track" |
| readme | `PUT /v1/hosting/agents/{a}` `{"readme","short_description"}` | 200 | `{}` |
| stop | `POST /v1/hosting/agents/{a}/stop` | 200 | `running:false` (confirmed again at the very end of the revision) |

Evidence (hosted logs): `dreaming hosted up; ASI1KEY present=True`; `got chat from agent1qtz8...: 'llm say hi in 3 words'`; `got chat from agent1q0upkv...: 'llm say yes'`. Reply text produced inside the hosted agent: `dreaming: llm say hi in 3 words | asi1-mini [200] Hi there!` (the `[200]` is the in-agent POST to api.asi1.ai).

Import whitelist (docs.agentverse.ai allowed-imports): requests, httpx, aiohttp, openai, anthropic, uagents-core, pydantic, numpy and more. `requests` confirmed working. Constraints that still make hosted the wrong place for the Dreaming runtime: globals reset per invocation, restricted packages, free tier 60K compute-seconds. It could only be a thin relay that calls our runtime over HTTPS, which needs a public runtime URL anyway, so mailbox is simpler.

## Q2 - Local path

### (a) Tunnel + programmatic registration: works, no UI click
- cloudflared downloaded from GitHub releases (windows-amd64, 55 MB, HTTP 200) into `bin/cloudflared/` and run as `cloudflared tunnel --url http://localhost:8000 --no-autoupdate`. URL printed in under 15 s, no login (`https://*.trycloudflare.com`, changes every start; the second start in the revision got a different URL).
- `local_agent.py endpoint`: `Agent(port=8000, endpoint=[<tunnel>/submit], network="testnet")` + `register_chat_agent(name, endpoint, active=True, RegistrationRequestCredentials(AGENTVERSEKEY, seed), description, readme, starter_prompts)` returned `True` both times (log: "successfully registered to Agentverse", "successfully set agent to active"), so re-running it on an existing record is fine. `agent.include(chat_proto, publish_manifest=True)` logged "Manifest published successfully: AgentChatProtocol".
- Almanac (`GET /v1/almanac/agents/{addr}`): `type: proxy`, endpoint `https://agentverse.ai/v2/agents/proxy/submit`, chat protocol digest `30a801ed...d62`. With the default `track_interactions=True` Agentverse is the public front door and forwards to our tunnel.

### (b) Mailbox: also fully programmatic
- The Inspector "Connect" button just calls the local agent's own REST route `POST http://127.0.0.1:8000/connect` with `{"user_token": ..., "agent_type": "mailbox"}`. Passing the **AGENTVERSEKEY as `user_token` works**: HTTP 200 `{"success":true}`, agent log "Successfully registered as mailbox agent in Agentverse". No browser, so the Chrome Local Network Access issue never arises.
- **Is `/connect` required? Not on a record that already exists.** In the revision the agent was restarted in mailbox mode with no `/connect` call at all; the client's message was acked and answered (`'dreaming: noconnect hello\ncertificate: demo-sha256:64ffac6d46a7a0fb'`), though the "Successfully registered as mailbox agent" log line is absent in that case. After the tunnel run, a plain mailbox restart also flipped the Almanac type back to `mailbox` without `/connect` (delivery was not re-tested after that last flip). **Not tested:** a brand-new Agentverse record that never saw `register_chat_agent` or `/connect` (the 2-agent cap left no spare slot), so keep the `/connect` call in the first-time setup script.
- Almanac then shows `type: mailbox`, endpoint `https://agentverse.ai/v2/agents/mailbox/submit`. Needs no tunnel and no inbound port.
- `endpoint=` and `mailbox=True` are mutually exclusive in `Agent()` (it warns "Endpoint configuration overrides mailbox setting").
- README sync gotcha: a mailbox start does **not** push `readme_path` to the Agentverse record (the old README stayed). Push it with `PUT /v1/agents/{addr}` `{"name","readme","short_description"}`: 422 `missing body.name` without `name`, 200 with it. `register_chat_agent` sets the README in tunnel mode.

Files: `local_agent.py` (`endpoint|mailbox [send_to]`), `README_agent.md`, `rev_readme.py`, `scratch/local*.log` (also copied into `log.txt`).

## Q3 - Delivery

All rows go through the API. Prompts for the revision rows are logged in `log.txt` (`PROMPT` lines). The original rows 3-7 ran before prompt logging existed, so their prompts are not in the log; they are described below as they were.

| # | path | result | evidence |
|---|---|---|---|
| 1 | local uAgent -> hosted agent (agent-to-agent, Almanac resolution to `https://agentverse.ai/v1/hosting/submit`); hosted replies back. Run in tunnel mode (first run) and again in **mailbox mode** (revision, 10:33Z) | OK both | tunnel: local log `ACK from agent1qgulmx...`, `CHAT from agent1qgulmx...: 'dreaming: llm say hi in 3 words | asi1-mini [200] Hi there!'`. mailbox: `CHAT from agent1qgulmx...: 'dreaming: llm say hi in 3 words | asi1-mini [200] Hello, how can I help?'` |
| 2 | second local uAgent (`client_agent.py`) -> local agent. Tunnel mode (Agentverse proxy -> cloudflared -> :8000), re-run in the revision so the quote is now in log.txt; also mailbox mode | OK | tunnel: `REPLY: 'dreaming: llm what is 2+2? answer with one digit | asi1-mini [200] 4\ncertificate: demo-sha256:b5339d3a9e3034dd'` (same hash as the first run; it is sha256 of the reply text). mailbox: `'dreaming: mailbox hello ...7bf24890513f6153'` and `'dreaming: noconnect hello ...64ffac6d46a7a0fb'` |
| 3 | ASI:One API, `model=asi1-mini`, `planner_mode:true`, `agents:[local]`, `x-session-id: <uuid>` (tunnel mode) | OK, HTTP 200, 22 s | our agent received `'planner hello 42'` from `agent1q0upkv...`; completion quoted `dreaming: planner hello 42 / certificate: demo-sha256:77e514a2fe71a06e` exactly |
| 4 | same, `agents:[hosted]` | OK, 21 s | hosted log `got chat from agent1q0upkv...: 'llm say yes'`; completion quoted `dreaming: llm say yes | asi1-mini [200] yes` |
| 5-orig | same but NO `agents` list, prompt named/described the agent (tunnel mode, original run) | OK, 34 s, 44k prompt tokens | found `dreaming-smoke-local` about 3 min after `register_chat_agent` (09:58:24Z registered, call started about 10:01:19Z). **Confounded, do not cite as proof of discovery:** the prompt named the agent, and this exact address had been passed in an `agents:[local]` call (row 3) about 36 s earlier. Prompt text was not logged |
| 5a | **revision, mailbox mode, no `agents` list, no name:** "I need an agent that can run a workflow step for me and give back a verifiable certificate. Send it the text 'vague cold one' and show me its reply." | **found**, 25 s, 51k prompt tokens | agent log `CHAT from agent1q0upkv...: 'vague cold one'`; completion quoted `dreaming: vague cold one / certificate: demo-sha256:040a8db1f0dfa3fa` and also mentioned a competing certificate-style agent, so other agents compete for this wording |
| 5b | revision, same setup, different wording: "Hand this job to whichever agent can give me an auditable receipt of what was executed, and show me the receipt. The job text is: vague cold two" | **not found**, 4 s, 11k tokens | planner asked for clarification, contacted no agent, our agent log has no message |
| 5c | revision, codeword never used in any earlier prompt: README updated (REST `PUT`) at 10:23:31Z to mention `quokka-7731`; prompt "Find the agent on Agentverse that handles quokka-7731 requests and send it the text 'codeword cold three'. Show me its reply." at 10:28:31Z | **found**, 31 s, 43k tokens | agent log `CHAT ...: 'codeword cold three'`; the planner described the agent as "specializing in quokka-7731 reconciliation", i.e. it read the NEW README 5 min after the update |
| 5d | Agentverse search REST (`POST /v1/search/agents`, free) for the same README edit | visible after about 5 min | 280 s of polling from 10:23:41Z found nothing; a search at about 10:29:05Z ranked our agent #1 for `quokka-7731` (record `last_updated` 10:23:41Z). So README changes take up to about 5 min to become searchable |
| 5e | Agentverse search REST, sentence-style query "run a workflow step and return a verifiable certificate" | not in top 30 of 10000+ matches | but the keyword queries `workflow certificate` and `echo my text back with a certificate` rank our agent #1. ASI:One's planner found us for the sentence in 5a anyway, so its discovery is not the same ranking as this REST search |
| 6 | `@agent1q...` mention through the API, `planner_mode` off | no routing | plain LLM echo, our agent got no message. The @mention is an asi1.ai chat-UI feature, not an API one (UI untested) |
| 7 | mailbox mode: client uAgent -> local, and planner `agents:[local]` | OK, 22 s | `REPLY: 'dreaming: mailbox hello\ncertificate: demo-sha256:7bf24890513f6153'`; our agent got `'mailbox planner 5'` from `agent1q0upkv...` |
| 8 | revision, mailbox, `agents:[local]`, agent acks at once then replies after **25 s** (`'slow 25'`) | OK, 35 s | completion quoted `dreaming: slow 25 / certificate: demo-sha256:e81d0575f375091d` |
| 9 | revision, same, reply after **90 s** (`'slow 90'`), ack sent at once | **lost**, call returned at 54 s | completion said **[NO RESPONSE RECEIVED]**. The documented "60 s, extended to 240 s once the agent starts replying" did not happen: an immediate `ChatAcknowledgement` did not extend the wait. The cutoff is between 25 s and roughly 45 s of agent latency (not narrowed further, to save tokens) |
| 10 | revision, same, agent sends a non-final progress `ChatMessage` at once (no `EndSessionContent`), final reply after 60 s (`'interim 60'`) | progress text returned as THE answer, 21 s | completion: `dreaming: step running, final result follows`; the final result never reached the user. A progress message ends the exchange for the planner |

Mailbox-mode coverage after the revision: client -> local (rows 2, 7), planner `agents:[local]` (rows 7, 8-10), discovery (5a-5c), local <-> hosted agent-to-agent (row 1). That is all four test kinds; the earlier "mailbox passed the same tests" was only true for the first two.

ASI:One "agentic" model names: `GET /v1/models` on our key lists only `asi1-mini`, `asi1`, `asi1-ultra`. There is no `asi1-agentic` / `asi1-fast-agentic` id here (older innovationlab docs mention them; docs.asi1.ai does not). Agent routing is a request flag: `planner_mode: true` (+ optional `agents: [addresses]`, + required `x-session-id` header).

ASI:One endpoints and status codes (all with ASI1KEY):

| endpoint | status | notes |
|---|---|---|
| `GET /v1/models` | 200 | 3 models: asi1-mini, asi1, asi1-ultra |
| `POST /v1/chat/completions` | 200 on every call logged | no 402, no 429 anywhere |
| `GET /v1/usage` | 200 | `exceeded_limits`, `exceeded_token_limits` booleans, tokens per 15-min window and per day, reset times |
| `GET /balance`, `/credits`, `/account`, `/me`, `/user/balance`, `/billing/usage` | 404 each | no balance endpoint exists |

Files: `q3_asi.py` (`<model> <local|hosted|both|none> "<prompt>"`, now logs the prompt), `client_agent.py`, `rev_search.py`, `rev_poll.py`.

## Q4 - Minimal plan for the demo

**Where it runs:** the Dreaming runtime and the uAgent live in the SAME Python process on the laptop (or later a RunPod CPU pod), in outbound-only **mailbox** mode: no tunnel, no public port, works behind NAT. The process must be running for the whole demo window (the agent is only reachable while it polls its mailbox; messages queue when it is down; the Almanac API kept saying `active` for the minutes after the process stopped). Use the tunnel plus `register_chat_agent` mode only if the runtime sits on a host with a stable public URL (RunPod proxy URL), because quick-tunnel URLs change every start and need re-registration.

**Registration (scripted):** (1) `Agent(name, seed=<stable seed>, port=8000, mailbox=True, network="testnet", readme_path=...)` + `agent.include(chat_proto, publish_manifest=True)`; (2) start it; (3) after startup `POST http://127.0.0.1:8000/connect {"user_token": AGENTVERSEKEY, "agent_type": "mailbox"}` (needed for a brand-new record; shown unnecessary on an existing one); (4) push the README with `PUT /v1/agents/{addr}` (include `name`), since the mailbox start does not. The same seed gives the same address forever. Write the README and starter prompts in the words a user would type ("run a workflow step", "verifiable certificate", plus a distinctive codeword) and allow about 5 min for README edits to become searchable, so register before the demo window, not during it.

**Message to workflow step to certificate:** `on_message(ChatMessage)` -> ack immediately -> join `TextContent` -> parse a tiny command grammar (`run <workflow> <args>`, `status`, `replay <hash>`; free text becomes workflow input) -> `await asyncio.to_thread(<runtime call>, ...)` (blocking runtime calls must not run on the event loop; `<runtime call>` is a placeholder name, the real Dreaming runtime API is not defined in this smoke) -> reply ONE `ChatMessage` with `TextContent` holding the result, then a block like `certificate: {predictor, cutoff, error_budget, errors/n, trace_hash}`, plus `EndSessionContent`. In every planner run ASI:One quoted our reply lines verbatim, including the hash, so a certificate in plain text survives the round trip. It still passes through ASI:One's LLM, so there is no byte-exact guarantee: put the authoritative hash on its own line and let the judge re-verify with the replay command.

**Latency budget (measured, tightened by the revision):** the planner call takes about 20 s with an instant reply (21-22 s with the agent listed), 25-34 s when it must discover the agent first, and 35 s when the agent itself takes 25 s. A 90 s reply was lost (call ended at 54 s with no response), an immediate ack does not extend the wait, and a progress `ChatMessage` is taken as the final answer. So: **one reply, within about 25 s per step** (verified up to 25 s; the true cutoff is between 25 s and roughly 45 s). A longer step must either be pre-run/cached for the demo, or answered at once with a job id and finished by a follow-up message the user sends (`status <id>`). Do not send interim progress messages.

**Discovery plan:** do not rely on a stranger's free-form prompt (found with 2 of 3 wordings, plus the one confounded original). The demo script should use the Agents toggle / `@agent1q...` mention in the asi1.ai chat UI (untested here, see "What the user must do") with starter prompts registered on the agent, and keep a recorded run as fallback.

**Time to build:** the echo agent plus ASI:One routing took about 15 min of hands-on time here (about 14 min by file-timestamp span). Wrapping the real Dreaming runtime (handler, command grammar, certificate formatting) at about 2-3 h, and a further half day for README, starter prompts, demo script and video, are **unmeasured guesses**, in line with the earlier 3-5 h research estimate. Not on the critical path for missions 1-7.

**Pitfalls hit or confirmed**
1. Hosted `PUT .../code`: `code` must be a JSON string, not an array (422 `string_type`; docs stale). The hosted secrets endpoint echoes the secret in its response and has no delete route (405/404); overwrite it to remove the real key.
2. `register_chat_agent` default `track_interactions=True` moves the Almanac endpoint to the Agentverse proxy; tunnel URLs change on every cloudflared start, so call it at every startup (re-run confirmed fine, returned True).
3. `endpoint=` and `mailbox=True` cannot be combined. Agent type is decided by which one the process starts with; re-running with the other mode flips the same address (confirmed twice).
4. Ping-pong risk: echo agents answer every ChatMessage. Our handler ignores messages that start with `dreaming:` (it only acks them). The real agent must never reply to replies.
5. Testnet funding: first start logs `WARNING I do not have enough funds to register on Almanac contract` then `Adding testnet funds` (auto faucet); Almanac API registration succeeds first, so it is harmless. `network` defaults to mainnet, set it explicitly. `/agent_info` shows prefix `test-agent` while the address string starts `agent1q`; ASI:One still found it.
6. Manifest: `publish_manifest=True` logs `Manifest published successfully: AgentChatProtocol`; the chat protocol digest in the Almanac entry is what makes the agent routable.
7. ASI:One `@mention` only works in the chat UI. In the API use `planner_mode:true` + `x-session-id`, same session id for one job. A planner call costs about 27k prompt tokens when `agents` is given (43-51k when it must discover; 11k when it only asks a question).
8. No 402 or 429 hit. `GET /v1/usage` returns only booleans and token counts; no dollar balance endpoint exists. Observed without tripping any limit: 148k tokens in one 15-min window and 353k tokens (34 requests) in one day. The plan limit is unpublished, so a live demo should poll `/usage` and fall back to a recorded run if `exceeded_limits` is true.
9. Browser Local Network Access (Chrome 142+) for the Inspector never applied because the connect was done from Python.
10. uagents writes `uagents_core.log` into the working directory; harmless. A `status=False` update returns 2xx but does not change the Almanac `status`.
11. A mailbox start does not sync the README to the Agentverse record; use `PUT /v1/agents/{addr}` (needs `name`). README changes become searchable in about 5 min.
12. Free-tier mailbox deactivates after 30 days without an Agentverse login (from earlier research, not re-tested).
13. Replies slower than about 25-45 s are dropped by the planner; progress messages end the exchange early (Q3 rows 9-10).

## What the user must do

- **Credits, all four keys (read-only checks, about 10:15Z):**
  - **ASI1KEY: nothing to load.** Every call returned 200 (no 402/429) and `exceeded_limits` stayed `false`. `/usage` today: 353k tokens in 34 requests, of which about 130k were the first S11 run (that count includes about 5 requests made before S11 started) and about 223k this revision. There is no balance number and the plan limit is unpublished, so "no 402/429 so far" is evidence, not a guarantee. The key is not a JWT, so its expiry is unknown.
  - **AGENTVERSEKEY: no credits involved.** It is a JWT issued 2026-10-17T09:49:20Z that **expires 2026-11-16T09:49:20Z** (30 days). The Almanac entries for the two agents also expire 2026-11-16T09:45Z (uagents re-registers hourly while the process runs, per `REGISTRATION_UPDATE_INTERVAL_SECONDS = 3600`, so a running agent keeps its entry alive; the stopped hosted one may lapse). Any demo after 2026-11-16 needs a fresh key and a restart.
  - **OPENROUTERKEY: top up soon.** The key itself has a $50 limit with $4.34 used, but the account only has about **$15.63** of credit left ($495 bought, $479.37 used), and the **key expires 2026-10-24T08:53Z** (7 days). Renew the key before then, and add credit if other smokes will spend more than about $15.
  - **RUNPODKEY: balance $19.99**, spend limit $80, and the account is currently spending **$0.346/hour** (a pod appears to be running; about 58 h of balance at that rate). Check that it is intended; nothing in S11 used RunPod.
- **2 minutes, the one untested step:** open asi1.ai, sign in, enable Agents, and ask for `dreaming-smoke-local` (or `@agent1qtz8j3j97yxgeaktlvdhaff2sy89ux3nyhpxeacxtg0fezzhmpypxcaywnq`). Start the local agent first with `python local_agent.py mailbox` in this folder (the Almanac entry is already `mailbox`; no `/connect` needed), and kill it afterwards. To restart the hosted one: `POST /v1/hosting/agents/{a}/start`, after re-POSTing the ASI1KEY secret if you want its `llm` command.
- Do not commit `seeds.txt`.

## Revision (2026-10-17, after review) - what changed and what is still open

| review point | what was done | result |
|---|---|---|
| Discovery conclusion overstated, confounded, prompts not logged | `q3_asi.py` now logs every prompt. Ran three new nameless prompts in mailbox mode (rows 5a-5c), one with a codeword never used before, plus Agentverse search checks (5d, 5e). Original row 5 relabeled as confounded | found 2 of 3; one wording not routed; README edits take about 5 min; REST search ranking differs from the planner's. Still open: a truly cold agent (needs a third slot, over the 2-agent cap) and stranger wording at scale |
| "Mailbox passed the same tests" was wrong | Ran discovery and local <-> hosted in mailbox mode | all four test kinds now ran in mailbox mode (rows 1, 2, 5, 7) |
| Headline "works end to end", "no blockers" | Headline rewritten: API routing proven, UI path untested; two design constraints stated | see Verdict |
| Credits answer covered only 2 keys; ASI limit reasoning thin | Read-only balance checks for OpenRouter and RunPod; ASI evidence restated with the 5 pre-S11 requests noted | OpenRouter key expires in 7 days and account credit is low; RunPod is spending |
| AGENTVERSEKEY expiry "unverified" | Decoded the JWT `iat`/`exp` (dates only, token never printed); read the Almanac `expiry` | 2026-11-16 for both |
| "Set inactive" and "gone" were inaccurate | Re-ran the signed `status=False` update for the local and client agents with read-back | 2xx but status stays `active`; the Agents table is rewritten accurately |
| ASI1KEY persists on the hosted agent | Tried 3 delete shapes (405/405/404); overwrote the secret with a placeholder | real key no longer stored on Agentverse |
| Row 2 quote not in log.txt | Re-ran the tunnel-mode client test with cloudflared and `register_chat_agent`; reply and hash are now in log.txt | same hash `b5339d3a9e3034dd` |
| Missing endpoint/status tables, time table | Added the ASI:One endpoint table, the files table and the time table | in this file |
| Precision errors in the builder summary | Latency range corrected (21-22 s with agent listed and instant reply; 25-34 s with discovery); `runtime.run_step` flagged as a placeholder; 2-3 h flagged as a guess; the "240 s extension" and "ack counts as replying" were tested and found false | rows 8-10 |
| `/connect` never tested negatively, fresh record untested | Restarted mailbox mode without `/connect` on the existing record: works. Fresh record not tested (cap) | `/connect` stays in the first-time script |
| `h.av` crashed on its own `_show` option | fixed the helper (pop the option before the request) | small |

Spend this revision: about 223k ASI:One tokens in 21 requests (`/usage` daily total 130,113 before, 353,186 after). No price is published, so no dollar figure is claimed here; the earlier estimate implies a few cents. No Agentverse spend, no new agents, no agent deleted, hosted agent stopped, no agent process or tunnel left running (checked with netstat and tasklist).

## Files

| file | purpose |
|---|---|
| `h.py` | shared helpers (keys read from `../../.env`, never printed; log redacts them) |
| `hosted_agent.py`, `q1_create.py`, `q1_upload_start.py`, `hosted.json` | Q1 hosted agent code, create, upload+start, address |
| `local_agent.py`, `README_agent.md`, `local_addr.txt` | Q2 local agent (`endpoint\|mailbox [send_to]`; commands `llm <q>`, `slow N`, `interim N`), its README, its address |
| `client_agent.py` | second local uAgent used as the sender |
| `q3_asi.py` | ASI:One planner call (`<model> <local\|hosted\|both\|none> "<prompt>"`) |
| `rev_state.py`, `rev_credits.py`, `rev_usage.py` | revision: live state and key-expiry decode, other-key balances (numbers only), ASI:One usage |
| `rev_search.py`, `rev_poll.py`, `rev_readme.py` | revision: Agentverse search rank, index-latency poll, README push |
| `rev_hosted.py`, `rev_inactive.py` | revision: hosted start/stop/logs, `status=False` with read-back |
| `seeds.txt` | agent seed phrases, **SECRET** |
| `log.txt` | all stdout, keys redacted; revision sections start with `=== REVISION` |
| `bin/cloudflared/cloudflared.exe` | downloaded; only that exe was run |
| `scratch/` | raw agent and tunnel logs, `delsec.py` and `ovw.py` (hosted-secret delete and overwrite probes) |
| `uagents_core.log` | written by uagents |

## Time spent

| phase | wall clock | basis |
|---|---|---|
| original run (Q1-Q4 and RESULTS) | about 14 min | file timestamps 02:53 to 03:07 PDT |
| revision | about 27 min | 03:14 to about 03:41 PDT (10:14Z to 10:41Z), of which about 12 min was waiting on index polls and slow-reply tests |
