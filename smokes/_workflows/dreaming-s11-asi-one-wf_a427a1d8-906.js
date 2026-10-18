export const meta = {
  name: 'dreaming-s11-asi-one',
  description: 'Feasibility smoke for the ASI:One / Agentverse connector: register an agent, answer chat-protocol messages, reach it from ASI:One',
  phases: [{ title: 'Build' }, { title: 'Critique' }, { title: 'Fix' }],
}
const ROOT = 'C:\\Users\\<user>\\Desktop\\personalprojects\\Dreaming'
const COMMON = `
ENVIRONMENT AND RULES:
- Project root: ${ROOT}. Python: "${ROOT}\\.venv\\Scripts\\python.exe" (Windows, CPU). Install packages only with: uv pip install --python "${ROOT}\\.venv\\Scripts\\python.exe" <pkg>. If you need a binary (e.g. cloudflared), download it into its own new empty folder under ${ROOT}\\smokes\\s11_asi_one\\bin\\ and never run anything else from that folder.
- Keys are in ${ROOT}\\.env as AGENTVERSEKEY and ASI1KEY (also OPENROUTERKEY). Read them with python/re or bash source; NEVER print them, never write them into any file other than .env, never put them in command lines that get logged.
- Read ${ROOT}\\smokes\\research\\asi-one.md first: it already contains the API surfaces (uagents, uagents_core.utils.registration.register_chat_agent, Agentverse hosting REST POST /v1/hosting/agents + PUT code + start, mailbox flow, ASI:One OpenAI-compatible base https://api.asi1.ai/v1 with models asi1-mini/asi1/asi1-ultra, chat protocol). Verified today: ASI1KEY works on POST https://api.asi1.ai/v1/chat/completions; AGENTVERSEKEY works on GET https://agentverse.ai/v1/hosting/agents (returns {"items":[]}).
- Read ${ROOT}\\missions.md section "Connector to ASI:One (Fetch)" for what the product needs: a user in ASI:One submits work to the Dreaming agent, the agent runs a workflow step on our runtime and returns the result and its certificate through chat.
- Use WebFetch/WebSearch for docs (fetch.ai, agentverse.ai, docs.asi1.ai, github fetchai/uAgents, innovationlab.fetch.ai). Never use mcp__claude-in-chrome tools.
- Keep code minimal (ponytail). Everything in ${ROOT}\\smokes\\s11_asi_one\\. Keep stdout in log.txt. Spend on ASI:One: a few cents at most (asi1-mini, short prompts). Do not create more than 2 agents on Agentverse; name them "dreaming-smoke-*"; leave them registered (do not delete) so the user can see them in the UI, but STOP any hosted agent you started at the end.
- Final return: 10-20 lines of key findings and blockers; this is data for the orchestrator.
`
const SPEC = `
SMOKE S11 — ASI:One / Agentverse connector feasibility. Answer, with evidence (HTTP codes, log lines, agent addresses):
Q1 Hosted path (most scriptable): using only the REST API with AGENTVERSEKEY, create a hosted agent (POST https://agentverse.ai/v1/hosting/agents), upload code (PUT .../code) implementing the chat protocol (uagents_core.contrib.protocols.chat: ChatMessage/ChatAcknowledgement/TextContent, publish_manifest=True) that replies to any text with "dreaming: <echo>" and, if feasible, calls ASI:One asi1-mini from inside the hosted agent (check the hosted-agent import whitelist: requests? openai?), then start it (POST .../start) and read its logs. Record every endpoint, payload shape, and status code. Report whether this path is fully programmatic.
Q2 Local path: a local uAgents agent (pip uagents) implementing the same chat protocol. Try to make it reachable WITHOUT a UI click: (a) programmatic registration via uagents_core.utils.registration.register_chat_agent with a public endpoint — a tunnel is needed; try downloading cloudflared (GitHub releases, windows amd64) into smokes/s11_asi_one/bin/ and running "cloudflared tunnel --url http://localhost:8000" (quick tunnel, no login); if a tunnel is not possible in this environment, say so and explain what the user would need (e.g. a RunPod CPU pod or ngrok); (b) mailbox=True: see whether the mailbox connection can be completed from code (uagents mailbox API / Agentverse API) or requires the Inspector UI "Connect" click. Use network="testnet".
Q3 Delivery check: send a chat message to the registered agent and get a reply. First via agent-to-agent (a second local uAgent or the uagents Client / send_sync_message / Agentverse "submit message" REST if any) and then via ASI:One itself: check docs.asi1.ai for an "agentic" chat mode (e.g. extra headers like x-session-id, model asi1-agentic / asi1-fast-agentic, or a tools field) that lets ASI:One route the user's request to a registered agent by address or name; try it with asi1-mini / the agentic model and record whether the reply came from our agent. If ASI:One discovery takes time to index (research says it may), say what you observed.
Q4 Integration estimate: given the answers, write the exact minimal plan for the demo: where the Dreaming runtime must run (laptop + tunnel vs cloud), which registration path, how a message maps to a workflow step and how the certificate rides back in the chat text, and the time to build it. Also list the pitfalls hit (ports, Local Network Access, almanac/testnet funding logs, manifest publishing, rate limits, 402/429 from ASI:One).
Write ${ROOT}\\smokes\\s11_asi_one\\RESULTS.md with sections Q1-Q4, tables for endpoints/status codes, code file names, time spent, and "What the user must do" (if anything). Keep the agent seed phrases in a local file seeds.txt that you mention is secret.
`
const CRITIC_SCHEMA = { type: 'object', properties: { material_issues: { type: 'array', items: { type: 'string' } }, minor_issues: { type: 'array', items: { type: 'string' } }, verdict: { type: 'string' } }, required: ['material_issues', 'minor_issues', 'verdict'] }

phase('Build')
const built = await agent(`${COMMON}\n${SPEC}`, { label: 'build:s11', phase: 'Build', model: 'sonnet' })
phase('Critique')
const crit = await agent(`${COMMON}
You are a skeptical reviewer. Read ${ROOT}\\smokes\\s11_asi_one\\RESULTS.md, log.txt and the code files. Builder summary:
---
${built}
---
Spec:
${SPEC}
Refute: claims without a status code or log line, "works" without an actual reply received, questions skipped silently, secrets written to files other than .env/seeds.txt, hosted agents left running, misreadings of the docs. Material = would change a conclusion or violates the spec. Be concrete.`, { label: 'critique:s11', phase: 'Critique', model: 'sonnet', schema: CRITIC_SCHEMA })
phase('Fix')
const fix = (crit && crit.material_issues.length) ? await agent(`${COMMON}
Reviewer found material issues with ${ROOT}\\smokes\\s11_asi_one:
${crit.material_issues.map((m, i) => `${i + 1}. ${m}`).join('\n')}
Minor: ${crit.minor_issues.join(' | ')}
Spec:
${SPEC}
Fix what can be fixed, re-run the affected checks, add a "## Revision" section to RESULTS.md, make sure no hosted agent is left running. Return revised key findings.`, { label: 'fix:s11', phase: 'Fix', model: 'sonnet' }) : null
return { built, crit, fix }