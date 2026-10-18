export const meta = {
  name: 'dreaming-s12-wrappers',
  description: 'Smoke the revised mission 6: four in-process wrappers (model client, tool, agent-to-agent send_and_receive, run-boundary) plus record-and-replay, on LangGraph and uAgents locally',
  phases: [{ title: 'Build' }, { title: 'Critique' }, { title: 'Fix' }],
}
const ROOT = 'C:\\Users\\<user>\\Desktop\\personalprojects\\Dreaming'
const COMMON = `
ENVIRONMENT AND RULES:
- Project root: ${ROOT}. Python: "${ROOT}\\.venv\\Scripts\\python.exe" (Windows, CPU; openai, httpx, langgraph may or may not be installed). Install packages only with: uv pip install --python "${ROOT}\\.venv\\Scripts\\python.exe" <pkg>. Set HF_HUB_DISABLE_SYMLINKS_WARNING=1.
- OPENROUTERKEY is in ${ROOT}\\.env (read with python/re; never print it). Model for any live call: openai/gpt-5.4-nano via base_url https://openrouter.ai/api/v1, temperature 0, max_tokens <= 60, extra_body={"reasoning": {"effort": "minimal"}}. Spend cap $0.50; append "s12_wrappers,<usd>,<note>" to ${ROOT}\\smokes\\COSTS.md (price 0.2/1.25 per M in/out).
- Read ${ROOT}\\missions.md FIRST, especially "### 6. Connectors" (the four wrappers; "what wrapping buys"), "### Our own orchestration runtime" (run-boundary decorator, call wrappers, replay mode are the core), and "### Connector to ASI:One / Agentverse" (uAgents is a framework like any other: model client wrapper, tool decorator, send_and_receive wrapper, run-boundary on the chat handler). Everything you build must match that text; where the text is ambiguous, pick the simplest reading and state it in RESULTS.md.
- Another agent is concurrently working in ${ROOT}\\smokes\\s10_connectors (HTTP proxy, LangGraph, Agents SDK, MCP) and ${ROOT}\\smokes\\s11_asi_one (Agentverse registration). Do NOT touch those folders; work only in ${ROOT}\\smokes\\s12_wrappers\\. You may READ s10's proxy.py for reuse ideas.
- Code style: ponytail (minimal, stdlib first). No Agentverse/ASI:One network calls in this smoke: uAgents agents run purely locally (Bureau, or two agents with local endpoints on localhost ports), network="testnet" if needed, and it is fine if almanac registration warns/fails as long as local messaging works.
- Never use mcp__claude-in-chrome tools. WebFetch/WebSearch for docs is fine (uAgents docs: https://uagents.fetch.ai/docs, github fetchai/uAgents; langgraph docs).
- Final return: 10-20 lines of key findings, gotchas and blockers; it is data for the orchestrator.
`
const SPEC = `
SMOKE S12 — the four wrappers + record-and-replay (revised mission 6). Folder: ${ROOT}\\smokes\\s12_wrappers.
Build dreaming_wrap.py, a tiny library (target under ~200 lines) with:
1. run_boundary(fn): decorator for sync and async entry points (a LangGraph graph.invoke, a uAgents message handler). Opens a run_id (deterministic counter or uuid), stores it in a contextvars.ContextVar so every wrapped call inside lands under it, closes the run on return/exception, writes a run row.
2. wrap_openai(client): wraps client.chat.completions.create (sync OpenAI and AsyncOpenAI) so each call logs {run_id, call_site, model, messages (structured, not flattened), tools if any, sha256 of the rendered messages JSON, response text, latency_ms, usage} to traces.jsonl. The call site comes from a call_site("name") context manager / ContextVar or from extra_headers["X-Dreaming-Call-Site"]; untagged calls are logged under a hash of the system prompt. If a predictors dict maps call_site -> {hash: text} and the hash is present, return an openai.types.chat.ChatCompletion object built from the predicted text with no network call and mark the row predicted=true.
3. tool(call_site, fallback=None, speculate_next=None): decorator for tool functions (sync and async) that logs args/return/latency/errors; consults an optional predictor callable (args -> (return, confidence)) and a certified cutoff; above the cutoff it skips the real call; on exception it uses the fallback if given; speculate_next is only logged (a hint).
4. wrap_send_and_receive(ctx) for uAgents: wraps ctx.send_and_receive(destination, message, response_type=...) so the call site is (destination, type(message).__name__); same predictor/fallback logic as tools; one-way ctx.send is logged as an effect only.
5. Replay mode: Recorder(mode="record"|"replay", path). In replay, every wrapped call is served from the logged rows of the given run_id in order with no network/tool execution; if the next call's call_site or request hash differs from the log, replay stops and reports the first divergence (row index, expected vs actual). Also "recertify"-style check: re-executing a predicted call on its logged prompt must give the same text.
Tests (each a separate small script, all must actually run; keep stdout in log.txt):
T1 LangGraph (pip langgraph langchain-openai if missing): a 2-node StateGraph (node A categorize, node B draft_note) where each node uses the wrapped OpenAI client under its own call_site; graph.invoke wrapped with run_boundary. Show: both calls logged under ONE run_id with the right call sites; then copy node A's answer into predictors and rerun -> node A served predicted=true, node B live; then replay mode -> zero network calls and identical outputs; then change node B's prompt and replay -> divergence reported at row 2. IMPORTANT: verify whether LangGraph runs nodes in worker threads/tasks that lose the ContextVar (run_id missing in rows); if so, fix it (e.g. contextvars.copy_context, or LangGraph config/callbacks) and document the gotcha.
T2 uAgents (pip uagents): two local agents (responder exposes a "VendorLookup" request/response via @on_message or @on_query; requester has a chat-style handler decorated with run_boundary that calls wrapped ctx.send_and_receive). Run them in a Bureau or as two processes on localhost ports. Show: the agent-to-agent call logged as a call site under the requester's run_id; a predictor that answers for known inputs makes the second request skip the send; stopping the responder (outage) with a registered fallback keeps the handler answering. Report the uagents version and whether ctx.send_and_receive exists with that signature; if it does not, use the closest primitive and say so.
T3 tool decorator: a vendor_lookup function with a dict-based predictor and cutoff: log, skip above cutoff, exception -> fallback; async variant.
RESULTS.md: what works (table per wrapper: sync/async, LangGraph, uAgents), lines of code, gotchas (ContextVar propagation across threads/tasks, building ChatCompletion objects, streaming not supported, uAgents async handlers and message type registration, ports), the exact one-line change per framework, time and cost, and what is NOT covered (Responses API, MCP, streaming).
`
const CRITIC_SCHEMA = { type: 'object', properties: { material_issues: { type: 'array', items: { type: 'string' } }, minor_issues: { type: 'array', items: { type: 'string' } }, verdict: { type: 'string' } }, required: ['material_issues', 'minor_issues', 'verdict'] }
phase('Build')
const built = await agent(`${COMMON}\n${SPEC}`, { label: 'build:s12', phase: 'Build', model: 'sonnet' })
phase('Critique')
const crit = await agent(`${COMMON}
Skeptical reviewer. Read ${ROOT}\\smokes\\s12_wrappers\\* (library, tests, log.txt, RESULTS.md). Builder summary:
---
${built}
---
Spec:
${SPEC}
Refute: tests that did not actually run (no log evidence), replay that still makes network calls, run_id missing from rows in LangGraph nodes, divergence detection that cannot fire, uAgents test that never exchanged a real message, send_and_receive not actually wrapped, claims unsupported by log lines, mismatch with the missions.md text. You may re-run the CPU-only scripts (T2, T3, replay runs) but not spend on OpenRouter. Material = changes a conclusion or violates the spec.`, { label: 'critique:s12', phase: 'Critique', model: 'sonnet', schema: CRITIC_SCHEMA })
phase('Fix')
const fix = (crit && crit.material_issues.length) ? await agent(`${COMMON}
Reviewer found material issues with ${ROOT}\\smokes\\s12_wrappers:
${crit.material_issues.map((m, i) => `${i + 1}. ${m}`).join('\n')}
Minor: ${crit.minor_issues.join(' | ')}
Spec:
${SPEC}
Fix in place, re-run the affected tests (reuse traces for replay; stay under the spend cap), add "## Revision" to RESULTS.md. Return revised findings.`, { label: 'fix:s12', phase: 'Fix', model: 'sonnet' }) : null
return { built, crit, fix }