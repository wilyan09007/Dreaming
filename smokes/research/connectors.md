# Connectors research: OpenAI-compatible proxy + tool-call interception

Date: 2026-10-17. Tested items were run locally (Windows, Python 3.13.2) against a fake upstream; versions listed under Numbers.
Test files (not in repo): a local `proxy\` folder (`dproxy.py`, `fake_upstream.py`, `clients_test.py`, `mcp_dreaming.py`).

## Summary

- Build our own ~60-line FastAPI proxy as the logging/answering layer. It is tested (see sketch) and adds about +4.7 ms p50 locally. LiteLLM would work but is heavy, had a PyPI supply-chain compromise in March 2026, and its pre-call hook cannot be shown to return a full response object (docs only show returning a string).
- The call-site tag is one custom header (`x-dreaming-site`). Every framework tested can attach it per model/agent/call. Tested OK: openai SDK, LangChain `ChatOpenAI`, OpenAI Agents SDK, Pydantic AI. CrewAI and Vercel AI SDK confirmed from source/docs only.
- Biggest trap: Responses API is the default in OpenAI Agents SDK, Pydantic AI (`openai:` prefix) and Vercel AI SDK (`openai(...)`), and LangChain `ChatOpenAI` auto-switches to it for some models. A `/v1/chat/completions`-only proxy is silently bypassed (verified: Agents SDK default and `ChatOpenAI(model="gpt-6-astra")+tools` both hit `/v1/responses`).
- OpenAI says GPT-6 Astra / 6.1 Sol function calling requires Responses; Sol and Luna do Chat Completions tools only with `reasoning_effort: "none"`. So an OpenAI-direct frontier step with tools forces a Responses implementation in the proxy. Routing via OpenRouter Chat Completions is the simpler path (unverified for these models).
- OpenRouter does not take custom tag headers as a logging key; it uses body fields (`user`, `session_id`, `trace{...}`) that its Broadcast feature forwards to Langfuse etc. Fine as an upstream; our own proxy sits in front and owns the tag.
- Helicone is in maintenance mode (acquired by Mintlify, announced 2026-03-03) and its new Rust gateway repo is GPL-3.0 and stale (last push 2025-11-21). Portkey OSS gateway is MIT but last pushed 2026-05-25 and its docs show no self-hosted log store. Langfuse self-host needs Postgres + ClickHouse + Redis + S3. None is a good fit as the logging layer for a local smoke.
- MCP: FastMCP 4.1.0 `create_proxy(...)` + `Middleware.on_call_tool` intercepts and can short-circuit tool calls (tested, returns a `ToolResult` without calling the backend). sparfenyuk/mcp-proxy is only a transport bridge (stdio/SSE/streamable HTTP), no interception.
- Responses API remote MCP tools (`type: "mcp"`) are called by OpenAI's side, not locally, so a local MCP proxy only works there via a public URL or tunnel. Avoid for the demo; use client-side MCP or function tools.

## Findings

### Proxy / logging layers

| Tool | Logs every call? | Custom headers | Verdict |
|---|---|---|---|
| LiteLLM proxy | Yes, via callbacks; spend logs | Headers starting `x-` (not `x-stainless-*`) forwarded to provider if `forward_client_headers_to_llm_api: true`; `x-pass-*` always forwarded with prefix stripped; `Authorization` never forwarded. https://docs.litellm.ai/docs/proxy/forward_client_headers | Works; optional. Request headers readable in `async_pre_call_hook` at `data["proxy_server_request"]["headers"]` and `data["metadata"]["headers"]` (credentials redacted) - verified in source `litellm/proxy/litellm_pre_call_utils.py`, main branch 2026-10-17. https://github.com/BerriAI/litellm/blob/main/litellm/proxy/litellm_pre_call_utils.py |
| Helicone | Yes (hosted/self-host) | `Helicone-Property-[Name]`, `Helicone-Session-Id`, `Helicone-User-Id`, `Helicone-Auth`. https://docs.helicone.ai/helicone-headers/header-directory | Skip: maintenance mode since 2026-03 (secondary sources: https://dev.to/spanlens/helicone-was-acquired-by-mintlify-here-is-a-migration-checklist-if-you-are-moving-off-41m3). Main repo Apache-2.0. |
| Portkey gateway | Hosted dashboards log; OSS log store not found | `x-portkey-metadata` (JSON), `x-portkey-trace-id`. https://github.com/Portkey-AI/docs/blob/v2/api-reference/headers.md | Skip. `npx @portkey-ai/gateway` runs on :8787; MIT; has guardrails and an MCP gateway. https://github.com/Portkey-AI/gateway |
| OpenRouter | Activity logs/export on all plans; Broadcast forwards traces | Tag via body: `user`, `session_id`, `trace{trace_id,trace_name,span_name,generation_name,...}` (docs mention no custom header). https://openrouter.ai/docs/guides/features/broadcast/langfuse ; attribution headers `HTTP-Referer`, `X-OpenRouter-Title` (`X-Title` legacy) https://openrouter.ai/docs/app-attribution | Use as upstream only. |
| Langfuse | Yes (SDK wrapper `from langfuse.openai import openai`; metadata keys `langfuse_session_id` etc.). https://langfuse.com/integrations/model-providers/openai-py | n/a (SDK, not header based) | Too heavy: Postgres, ClickHouse, Redis, S3 + web + worker. https://langfuse.com/self-hosting . ClickHouse acquired Langfuse Jan 2026, MIT kept (secondary source). Page does not document Responses API support. |
| OpenLLMetry (Traceloop) | Yes, OTel spans for openai/langchain etc. | Not header based | Apache-2.0; optional export format later. Local-exporter recipe unconfirmed. https://github.com/traceloop/openllmetry |

LiteLLM hook facts (https://docs.litellm.ai/docs/proxy/call_hooks): `async_pre_call_hook(self, user_api_key_dict, cache, data, call_type)`; returning a string produces a normal chat/text completion response and skips the LLM call (400 for other endpoints); raising `HTTPException` rejects. Returning a `ModelResponse` is not documented. `async_post_call_success_hook` return semantics not documented. `async_moderation_hook` runs in parallel with the call, so it cannot answer instead.

### Per-framework one-liner (base_url + call-site header)

Constants: `BASE="http://127.0.0.1:8788/v1"`, `H={"x-dreaming-site":"categorize_transaction","x-dreaming-run":"r1"}`. "Tested" = header observed at upstream via the sketch proxy.

| Framework (version tested) | Change | Status |
|---|---|---|
| openai SDK 3.33.0 | `OpenAI(base_url=BASE, api_key=k, default_headers=H)`; per call `create(..., extra_headers=H)` | tested |
| LangChain / LangGraph (`langchain-openai` 1.7.2) | `ChatOpenAI(model=m, base_url=BASE, default_headers=H, use_responses_api=False)`; per call `llm.bind(extra_headers=H)` or `llm.invoke(x, extra_headers=H)`. Per LangGraph node: build one `ChatOpenAI` per node, or `.bind(extra_headers=...)` inside the node. `base_url` is alias of `openai_api_base`; env `OPENAI_API_BASE` / `OPENAI_BASE_URL` also work | tested (all four forms) |
| OpenAI Agents SDK 0.24.0 | `Agent(model=OpenAIChatCompletionsModel(model=m, openai_client=AsyncOpenAI(base_url=BASE, api_key=k)), model_settings=ModelSettings(extra_headers=H))`. Global alternative: `set_default_openai_client(AsyncOpenAI(base_url=BASE, api_key=k), use_for_tracing=False)` + `set_default_openai_api("chat_completions")` (+ `set_tracing_disabled(True)`). `ModelSettings.extra_headers` works on Responses model too | per-agent tested; the two global setters: set_default_openai_client tested (hit `/v1/responses` by default), `set_default_openai_api` documented, not run. https://openai.github.io/openai-agents-python/models/ ; https://openai.github.io/openai-agents-python/ref/model_settings/ |
| Pydantic AI 2.57.0 | `Agent(OpenAIChatModel(m, provider=OpenAIProvider(base_url=BASE, api_key=k)), model_settings={"extra_headers": H})`. Provider signature: `OpenAIProvider(base_url, api_key, openai_client, http_client)`. String `"openai:m"` = Responses model; `"openai-chat:m"` = Chat | tested (chat and responses) . https://pydantic.dev/docs/ai/models/openai/ |
| CrewAI (main, not installed) | `LLM(model="openai/<m>", base_url=BASE, api_key=k, default_headers=H)`; native OpenAI provider has `default_headers` and `api: "completions"\|"responses"` (default completions); one `LLM` per Agent. Docs page https://docs.crewai.com/en/learn/llm-connections lists `base_url` only, not headers | source-read only: `lib/crewai/src/crewai/llms/providers/openai/completion.py` |
| Vercel AI SDK | `createOpenAI({ baseURL: BASE, apiKey: k, headers: H })` then `provider.chat(m)` (plain `provider(m)` = Responses); per call `generateText({ model, headers: H })` ("only applicable for HTTP-based providers"). https://ai-sdk.dev/providers/ai-sdk-providers/openai ; https://ai-sdk.dev/docs/reference/ai-sdk-core/generate-text | docs only |

### Chat Completions vs Responses when proxying

- OpenAI: Chat Completions "remains supported", Responses recommended; Responses stored by default (`store:false` to disable); output is a typed `output` array (`message`, `function_call`, `function_call_output`, reasoning items), streaming uses typed SSE events (`response.created`, `response.output_text.delta`, `response.completed`). https://developers.openai.com/api/docs/guides/migrate-to-responses
- `previous_response_id` is server-side state: a pass-through proxy keeps it working only if upstream stores state; a translating proxy must store responses itself. OpenRouter Responses is stateless and returns 400 for `store:true` or non-null `previous_response_id`; clients must resend history. https://openrouter.ai/docs/api/reference/responses/overview
- LiteLLM proxy serves `/v1/responses` and bridges non-OpenAI providers through chat completions; `previous_response_id` session management needs `store_prompts_in_spend_logs` and S3 (s3_v2); response-ID ownership checks give 403 on raw provider IDs. https://docs.litellm.ai/docs/response_api
- Encrypted reasoning items (`encrypted_content`) are opaque; a student model cannot consume or produce them, so Responses calls with reasoning must be logged/replayed as opaque blobs.
- GPT-6 note (OpenAI docs): Astra and 6.1 Sol "support Chat Completions, but tool calling requires Responses"; Sol/Luna function calling in Chat Completions only with `reasoning_effort: "none"`. https://developers.openai.com/api/docs/guides/latest-model
- Framework defaults that hit Responses: Agents SDK default model class; Pydantic AI `openai:` prefix; Vercel `openai()`. LangChain `_use_responses_api` returns True for `reasoning`, `include`, `truncation`, `previous_response_id`, built-in tools, models in `_RESPONSES_API_ONLY_PREFIXES` (`gpt-5-pro`, `gpt-5.5-pro`, `gpt-6.x-sol`, ...), and `gpt-6*` with tools; fix with `use_responses_api=False`. Source: https://github.com/langchain-ai/langchain/blob/master/libs/partners/openai/langchain_openai/chat_models/base.py (lines ~763, ~2012)

### MCP interception

- FastMCP (Apache-2.0, 4.1.0 on 2026-10-08): `create_proxy(url | path | config dict | FastMCP instance)`, then `proxy.add_middleware(...)`. Hooks: `on_message`, `on_request`, `on_call_tool` (`context.message.name`, `.arguments`), `on_list_tools`, ... Not calling `call_next` stops processing; `call_next` in `on_call_tool` returns a `ToolResult`. Parent middleware also runs for mounted servers. Built-in `ResponseCachingMiddleware` caches tool calls (TTL, per-tool include list, file-tree store); cache key excludes custom HTTP headers. https://gofastmcp.com/servers/middleware ; https://gofastmcp.com/servers/providers/proxy ; source docs https://github.com/PrefectHQ/fastmcp/blob/main/docs/servers/middleware.mdx
- FastMCP proxy forwards incoming HTTP headers (incl. `Authorization` and custom) to HTTP/SSE backends by default; `ProxyClient(..., forward_incoming_headers=False)` disables. Call-site header can therefore reach the real MCP server.
- sparfenyuk/mcp-proxy (MIT, v0.13.0 2026-10-10, 2.8k stars): stdio <-> SSE/streamable HTTP bridge, named servers. No interception. https://github.com/sparfenyuk/mcp-proxy
- LiteLLM MCP gateway: guardrail modes `pre_mcp_call`, `during_mcp_call`, `post_mcp_call`; per-server upstream auth via `x-mcp-{server_alias}-{header_name}`. Interception only, answering unverified. https://docs.litellm.ai/docs/mcp_guardrail
- OpenAI Responses `type:"mcp"` tool: "the API will make a request to the remote MCP server"; servers must be public (or Secure MCP Tunnel); approval required by default; auth via `authorization` field, no headers parameter. https://developers.openai.com/api/docs/guides/tools-connectors-mcp

### Code sketch 1: logging + answer-from-predictor proxy (tested; full file `dproxy.py`, not in repo)

```python
# pip install fastapi uvicorn httpx ; uvicorn dproxy:app --port 8788
import hashlib, json, os, random, time, uuid, httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
UPSTREAM = os.environ.get("UPSTREAM", "https://openrouter.ai/api/v1")
CUTOFF = {"categorize_transaction": 0.9}            # certified per-site cutoff
AUDIT_RATE = float(os.environ.get("AUDIT_RATE", 0.1))
DROP = {"host", "content-length", "connection", "accept-encoding"}
app, http = FastAPI(), httpx.AsyncClient(timeout=120)

def predict(site, body):                            # stub -> (text, confidence) | None
    return ("groceries", 0.97) if "grocer" in str(body["messages"][-1]["content"]).lower() else None
def log(rec):
    open("calls.jsonl", "a").write(json.dumps(rec) + "\n")

@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()                          # ponytail: log raw bytes too if byte-exact replay needed
    site = req.headers.get("x-dreaming-site", "unknown")
    t0 = time.time()
    rec = {"ts": t0, "site": site, "run": req.headers.get("x-dreaming-run"),
           "key": hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "body": body}
    pred = None if body.get("stream") or body.get("tools") else predict(site, body)
    if pred and pred[1] >= CUTOFF.get(site, 2) and random.random() >= AUDIT_RATE:
        log(rec | {"served_by": "predictor", "out": pred[0], "conf": pred[1]})
        return JSONResponse({"id": "chatcmpl-" + uuid.uuid4().hex[:24], "object": "chat.completion", "created": int(t0),
            "model": body.get("model", "dreaming"), "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": pred[0]}}]})
    hdrs = {k: v for k, v in req.headers.items() if k.lower() not in DROP}   # forwards x-dreaming-* and Authorization
    up = await http.post(f"{UPSTREAM}/chat/completions", json=body, headers=hdrs)   # ponytail: buffers; tee SSE for streaming
    log(rec | {"served_by": "upstream", "status": up.status_code, "out": up.text, "ms": (time.time() - t0) * 1e3,
               "pred": pred, "audit": bool(pred)})          # audit rows give predictor-vs-real pairs for certification
    return Response(up.content, up.status_code, media_type=up.headers.get("content-type"))

@app.api_route("/v1/{path:path}", methods=["GET", "POST"])    # /v1/responses etc: log + pass through (REQUIRED, see blockers)
async def passthrough(path: str, req: Request):
    raw = await req.body()
    up = await http.request(req.method, f"{UPSTREAM}/{path}", content=raw,
                            headers={k: v for k, v in req.headers.items() if k.lower() not in DROP})
    log({"ts": time.time(), "site": req.headers.get("x-dreaming-site", "unknown"), "path": path, "status": up.status_code, "req": raw.decode(errors="replace")})
    return Response(up.content, up.status_code, media_type=up.headers.get("content-type"))
```

Ceilings (ponytail): non-streaming only for predictor answers; Responses API is log-only passthrough (buffered, so streaming works but not incrementally); no tool-call answers; file append per call (swap for SQLite/parquet writer); no auth on the proxy.

### Code sketch 2: MCP tool interception (tested on fastmcp 4.1.0, mcp 2.3.0; `mcp_dreaming.py`)

```python
from fastmcp.server import create_proxy
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.tools import ToolResult
class Dreaming(Middleware):
    async def on_call_tool(self, ctx: MiddlewareContext, call_next):
        name, args = ctx.message.name, ctx.message.arguments
        pred = predict_tool(name, args)                      # (structured_dict, confidence) | None
        if pred and pred[1] >= CUTOFF.get(name, 2):
            log(...); return ToolResult(structured_content=pred[0])   # answered, backend never called
        res = await call_next(ctx)                           # real call; log args + res.structured_content + latency
        log(...); return res
proxy = create_proxy("http://localhost:9000/mcp", name="dreaming")   # or config dict / script path / stdio cmd
proxy.add_middleware(Dreaming()); proxy.run(transport="http", port=9001)
```

## Reusable components

| name | url | license | why |
|---|---|---|---|
| Own FastAPI proxy (sketch 1) | `proxy\dproxy.py` (local test file, not in repo) | ours | Tested; call-site header, JSONL log, predictor answer, audit sampling, Responses passthrough |
| FastMCP (proxy + middleware) | https://github.com/PrefectHQ/fastmcp | Apache-2.0 | `on_call_tool` intercept/short-circuit, built-in caching, header forwarding; 4.1.0 (2026-10-08) |
| MCP Python SDK | https://github.com/modelcontextprotocol/python-sdk | MIT | Client/server types FastMCP builds on (mcp 2.3.0) |
| sparfenyuk/mcp-proxy | https://github.com/sparfenyuk/mcp-proxy | MIT | Transport bridge only (stdio<->HTTP) so stdio MCP servers can sit behind a URL |
| LiteLLM proxy | https://github.com/BerriAI/litellm | MIT (enterprise/ dir separate) | Optional if we need Anthropic/Gemini -> OpenAI format translation; header passthrough and hooks exist; pin + hash |
| OpenLLMetry | https://github.com/traceloop/openllmetry | Apache-2.0 | Possible later OTel export of logged calls |
| Langfuse | https://github.com/langfuse/langfuse | MIT (ee/ dirs separate) | Later UI for traces only if we can afford the 4-service stack |
| Framework SDKs | https://github.com/openai/openai-agents-python , https://github.com/pydantic/pydantic-ai , https://github.com/crewAIInc/crewAI , https://github.com/langchain-ai/langgraph | MIT | Target frameworks; all expose header hooks above |
| Vercel AI SDK | https://github.com/vercel/ai | Apache-2.0 | Target framework (TypeScript) |

## Numbers

| Item | Value | Date / source |
|---|---|---|
| LiteLLM proxy gateway overhead, 4 instances x (4 vCPU, 8 GB), 1,170 RPS | p50 2 ms, p95 8 ms, p99 13 ms | page undated (2026 footer) https://docs.litellm.ai/docs/benchmarks |
| Same, 2 instances | 1,035.7 RPS; p50 12 ms, p95 29 ms | same |
| Our sketch proxy, local Windows laptop, 1 uvicorn worker, fake upstream, 200 calls | direct 4.98 ms p50 / 5.57 p95; via proxy 9.64 / 16.63 (about +4.7 ms p50); predictor hit 5.46 / 6.64 | measured 2026-10-17 (machine shared with other jobs) |
| OpenRouter platform fee | 5.5% on credit purchases (Standard), 8% Business; BYOK free up to $25k/month list-price then 5% | https://openrouter.ai/pricing (2026-10-17). With ~$20 credit expect about $18.9 usable (arithmetic, minimum fee unverified) |
| OpenRouter free models | 50 req/day (20/min) with no credits; 1,000/day after buying >= $10 credits | https://openrouter.ai/pricing (2026-10-17) |
| OpenRouter Broadcast | async, no added API latency; Privacy Mode drops prompt/completion content | https://openrouter.ai/docs/guides/features/broadcast/langfuse |
| Portkey gateway claims | "<1 ms latency", "122kb" footprint (vendor claims, no benchmark) | https://github.com/Portkey-AI/gateway |
| Versions (2026-10) | litellm v1.105.1 (2026-10-14); fastmcp 4.1.0 (2026-10-08); mcp-proxy v0.13.0 (2026-10-10); openai-python 3.33.0; openai-agents 0.24.0; pydantic-ai 2.57.0; langchain-openai 1.7.2 | PyPI installs / GitHub API, 2026-10-17 |
| Repo stars / last push | litellm 61.4k; fastmcp 28.1k; mcp-proxy 2.8k; Portkey gateway 13.2k (push 2026-05-25); Helicone 6.2k; Helicone ai-gateway 636 (GPL-3.0, push 2025-11-21) | GitHub API 2026-10-17 |

## Blockers / plan changes

1. Must implement or pass through `/v1/responses`. Verified: Agents SDK default and LangChain `gpt-6*`+tools bypass a chat-completions-only proxy. Plan: in the demo, force Chat Completions in every framework (`use_responses_api=False`, `OpenAIChatCompletionsModel`, `OpenAIChatModel`, `provider.chat()`), keep Responses as log-only passthrough, and add answer-from-predictor for Responses only later (needs synthesized typed output items and SSE event sequence).
2. OpenAI-direct GPT-6 Astra/Sol tool-calling steps need Responses (per OpenAI docs). Use OpenRouter (Chat Completions) as the frontier upstream for the smoke; whether OpenRouter's Chat Completions handles these tool calls is unverified.
3. Do not adopt LiteLLM without pinning: PyPI versions 1.82.7 and 1.82.8 (2026-03-24) were malicious (credential stealer, TeamPCP; 1.82.6 last clean). Sources: https://www.netspi.com/blog/executive-blog/ai-ml-pentesting/litellm-supply-chain-compromise/ , https://cycode.com/blog/lite-llm-supply-chain-attack/ . Our own proxy avoids it.
4. Helicone (maintenance mode), Helicone ai-gateway (GPL-3.0, stale), Portkey OSS (stale since May, log store unconfirmed), Langfuse (4-service stack) are not viable as the local logging layer. Log to JSONL/SQLite first.
5. Byte-for-byte replay: the sketch hashes canonical JSON, not raw bytes. If replay must be byte exact, store the raw request body and raw response bytes (and SSE stream) in the log.
6. OpenAI-hosted remote MCP tool executes on OpenAI's side; an MCP proxy there requires a public URL/tunnel. Keep MCP interception on the client side (Agents SDK/Pydantic AI/FastMCP client talking to our proxy URL).
7. Streaming: predictor answers need synthesized SSE chunks; upstream pass-through needs a streaming tee (the sketch buffers). Tool-call outputs (predicting `tool_calls` arguments) not implemented.
8. On OpenRouter-direct (no proxy) the call-site tag must ride in body fields (`user`, `session_id`, `trace.trace_name`), not a header.

## Unverified

- Whether LiteLLM `async_pre_call_hook` may return a `ModelResponse` (docs only show string / `HTTPException`); `async_post_call_success_hook` return semantics.
- CrewAI `default_headers` behavior at runtime (source-read only); CrewAI LiteLLM-fallback `extra_headers` passthrough; Vercel AI SDK per-call `headers` and `createOpenAI.headers` runtime behavior (docs only).
- Pydantic AI per-run `model_settings` override; Agents SDK `set_default_openai_api("chat_completions")` (documented, not run).
- OpenRouter: whether arbitrary request headers are logged/forwarded; Chat Completions tool calling for GPT-6 models; minimum fee on credit purchases.
- Helicone acquisition details (only secondary sources; official post helicone.ai/blog/joining-mintlify not read); Helicone current base URL (docs show both `oai.helicone.ai` and `gateway.helicone.ai`).
- Portkey OSS gateway request-log storage and whether it honors `x-portkey-metadata` without the hosted platform.
- LiteLLM benchmark page date; OpenLLMetry local exporter recipe; Langfuse minimum hardware.
- Responses API pass-through of streaming and SSE through the sketch proxy (not tested); LangChain `invoke(extra_headers=)` works in 1.7.2 but is not a documented API.
