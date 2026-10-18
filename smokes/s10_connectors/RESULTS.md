# S10 - Mission 6 connectors: OpenAI-compatible proxy + MCP proxy

## Setup

- Model: `openai/gpt-5.4-nano` through OpenRouter, temperature 0, max_tokens 60. No reasoning field was needed (the probe returned 5 completion tokens with no reasoning tokens).
- Versions (installed into `.venv` with uv; the two installs took 60 s and 16 s, so pydantic-ai was within the 2-minute limit): fastapi 0.143.3, uvicorn 0.54.0, httpx 0.28.1, openai 3.33.0, langgraph 1.2.16, langchain-openai 1.7.2, openai-agents 0.24.0, pydantic-ai-slim 2.57.0, fastmcp 4.1.0. Windows 11, Python 3.12.12.
- Files (all in `smokes/s10_connectors/`):
  - `proxy.py`: the proxy. 98 lines, 84 excluding blanks and comment-only lines.
  - `client_langgraph.py` (27), `client_agents.py` (26, three modes plus `stream`), `client_pydanticai.py` (14, plus `stream`).
  - `mcp_demo.py` (66, 59 excluding blanks and comments).
  - `run.py` (114): the harness. It starts the proxy and runs every client twice. Between the two runs it copies the first answer into `predictors.json`. It then runs the protocol probes and the MCP demo, checks `traces.jsonl`, and sums spend.
  - `log.txt`, `traces.jsonl`, `predictors.json`, `mcp_traces.jsonl` and `mcp_cache.json` are the outputs of the final run.
- "Predictor" here is an exact lookup: `predictors.json[call_site]["answers"][sha256(messages)]`. The trainer is a stand-in (`learn()` in `run.py` copies the first real answer). There is no student, no certificate and no generalisation in this smoke. It tests the plumbing a certified student would plug into.
- No random seeds are involved. Temperature is 0 and the harness is deterministic apart from upstream latency.

## Results

### 1. Framework clients (pass 1 = real call through the proxy, pass 2 = answered from `predictors.json`)

Every row was checked by assertion in `run.py`:

- The header arrived: the `call_site` in `traces.jsonl` is the tagged name, not the `sha1:` fallback.
- Pass 1 reached upstream, and pass 2 had `predicted=true` with the same message hash.
- The client's final output was identical, and pass 2 logged zero upstream usage.

| Client | Endpoint | Call sites seen | Pass 1 latency (ms) | Pass 2 latency (ms) | Result |
|---|---|---|---|---|---|
| LangGraph, 2 nodes | chat | `categorize` (own ChatOpenAI), `summarize` (`.bind(extra_headers=)`) | 727 / 647 | 0 / 0 | pass |
| Agents SDK, `OpenAIChatCompletionsModel` | chat | `triage-chat` | 752 | 1 | pass |
| Agents SDK, global default client + `chat_completions` | chat | `triage-global` | 688 | 0 | pass |
| Agents SDK, `OpenAIResponsesModel` (the SDK default API) | responses | `triage-responses` | 875 | 2 | pass |
| pydantic-ai, `OpenAIChatModel` | chat | `sentiment` | 731 | 1 | pass |
| Agents SDK, chat, `Runner.run_streamed` | chat, stream | `triage-chat-stream` | 599 | 2 | pass |
| pydantic-ai, `run_stream` | chat, stream | `sentiment-stream` | 1140 | 0 | pass |

### 2. Protocol probes (final run, 10 checks, all pass; 32 PASS lines in `log.txt` in total)

| Probe | Outcome |
|---|---|
| Raw OpenAI SDK, chat, 2nd call | Served from `predictors.json`. The raw client sees `resp.model_extra["dreaming"] == {"predicted": True}`; pass 1 has no such field. |
| Raw OpenAI SDK streaming with `include_usage` | Pass 1 was forwarded; the trace holds the full text and usage. The 2nd call was a synthesized stream (content chunk, stop chunk, `[DONE]`) with the same text. |
| LangChain `.stream()` | Pass 1 forwarded, 2nd call served as a synthesized stream. |
| `/v1/responses` via raw SDK | Logged with `api=responses`; 2nd call predicted; the client sees the `dreaming` flag. |
| LangChain `use_responses_api=True` | Goes to `/v1/responses`; predicted on the 2nd call. |
| pydantic-ai `OpenAIResponsesModel` | Goes to `/v1/responses`; predicted on the 2nd call. |
| Forced tool call (`tools` + `tool_choice`) | Passed through; the client got a `vendor_lookup` tool call. The trace stores `tool_calls` and an empty `response`, and the harness's `learn()` skips it. The identical 2nd call was not predicted. |
| Follow-up turn with `role: "tool"` message | Passed through; the trace holds the model's text. |
| Untagged calls via `OPENAI_BASE_URL` only (no header, no code change) | Two calls with one system prompt got the same site `sha1:8e955670d92b`. A different system prompt got `sha1:3bc623a0e3ec`. |
| Invalid model slug | Upstream 400 relayed to the SDK as a 400. The trace row has `status=400`. |

### 3. Overhead

| Measure | Value |
|---|---|
| Median latency, direct to OpenRouter (8 paired calls, unique prompts) | 706 ms |
| Median latency, through proxy (same prompts) | 639 ms |
| Predicted-hit round trip, 10 calls, from the OpenAI SDK on localhost | median 10 ms, min 9 ms |

Proxy overhead is below upstream jitter. n = 8, so this is a statement of "not measurable here", not "zero". The negative delta is noise.

### 4. MCP (fastmcp 4.1.0)

A `vendor_lookup(descriptor)` tool sits behind `create_proxy(backend)` with a `Middleware.on_call_tool` tap. The tap logs to `mcp_traces.jsonl` and consults `mcp_cache.json`.

| Step | Transport | Predicted | Latency (ms) | Check |
|---|---|---|---|---|
| `"adobe *creative cld "` (sloppy) | in-memory | no | 215.5 | The backend received `"ADOBE *CREATIVE CLD"`, so the request rewrite reached it. |
| same key after copying the answer into the cache | in-memory | yes | 1.0 | The backend call counter did not change. |
| backend marked down, cached key | in-memory | yes | 0.8 | The agent keeps working through the outage. |
| backend down, unseen key | in-memory | no | 419 | Error passes through. It arrives in the middleware as `ToolResult.is_error=True`, not as an exception. |
| child process over stdio, cached key | stdio | yes | 1.4 | The cache is a file, so a separate process sees it. |
| child process over stdio, unseen key | stdio | no | 300.6 | Real call. |

Variance across about six runs of the demo: the first cache hit after the cache file was written sometimes took 30-220 ms. I did not diagnose this and suspect file-read cost on Windows. Steady-state hits were 0.8-2.4 ms. Real calls to the 200 ms sleeping backend took 215-390 ms through the proxy.

Can the proxy intercept both the request and the result? Yes. `on_call_tool(context, call_next)` sees `context.message.name` and `.arguments` before the backend is called. It can rewrite them via `context.copy(message=...)`, as shown above. It sees the `ToolResult` after, and can pass it through, replace it, or short-circuit without calling `call_next`.

### 5. Spend

| Item | USD |
|---|---|
| Final run: 28 upstream calls, tokens x gpt-5.4-nano price; matches the OpenRouter-reported `usage.cost` sum | 0.00042 |
| Whole smoke including five earlier harness runs (one passed, four failed or were stopped) and manual probes (upper estimate; traces were reset each run) | about 0.003 |
| Cap | 1.00 |

## The one-line change per framework

All point at `http://127.0.0.1:8765/v1`. The key can be any string; the proxy substitutes its own.

| Framework | Change | Notes |
|---|---|---|
| Any code using the OpenAI SDK defaults | `OPENAI_BASE_URL=http://127.0.0.1:8765/v1` and `OPENAI_API_KEY=x` in the environment (zero code lines) | Verified. There is no call-site header, so calls group by `sha1(system prompt)`. |
| LangGraph / langchain_openai | `ChatOpenAI(..., base_url=BASE, api_key="x", default_headers={"X-Dreaming-Call-Site": "categorize"})` | One instance per node. Alternative: one shared instance and `llm.bind(extra_headers={"X-Dreaming-Call-Site": "summarize"})`. Both verified. `with_config` cannot set headers. |
| OpenAI Agents SDK | `model=OpenAIChatCompletionsModel(model="openai/gpt-5.4-nano", openai_client=AsyncOpenAI(base_url=BASE, api_key="x", default_headers={...}))` | Needs `set_tracing_disabled(True)`. One client (hence one header) per call site. Alternatives: `OpenAIResponsesModel(...)` with the same client; or the global `set_default_openai_client(client, use_for_tracing=False)` + `set_default_openai_api("chat_completions")`, which tags every agent with the same site. |
| pydantic-ai | `OpenAIChatModel("openai/gpt-5.4-nano", provider=OpenAIProvider(base_url=BASE, api_key="x"))` plus `model_settings={"extra_headers": {"X-Dreaming-Call-Site": "sentiment"}}` | I did not test the header via `OpenAIProvider(openai_client=AsyncOpenAI(default_headers=...))`. |
| MCP | Wrap the server: `proxy = create_proxy(target)` (target = FastMCP instance, URL, path or Client); `proxy.add_middleware(Tap())`; clients connect to `proxy` or run `proxy.run()` for stdio | `FastMCP.as_proxy` from the task text does not exist in fastmcp 4.1.0. |

So "one line" holds only when untagged grouping is acceptable. Tagging a call site is per-client configuration: a header on each ChatOpenAI instance, or one AsyncOpenAI client per Agents SDK call site.

## Gotchas hit

1. **Responses vs Chat Completions.**
   - The Agents SDK defaults to the Responses API. A proxy with only `/v1/chat/completions` would 404 it. The proxy serves both paths and each passed a pass-1/pass-2 round trip.
   - LangChain with `use_responses_api=True` and pydantic-ai's `OpenAIResponsesModel` also use `/v1/responses` (probes pass).
   - The two APIs render the same prompt differently (`messages` vs `instructions` + `input`), so the message hash differs. A predictor learned from chat traffic will not hit Responses traffic for the same prompt.
   - Not handled or tested: predicted answers for streamed Responses calls (they are forwarded and logged only), and stateful Responses (`previous_response_id`). In stateful mode the proxy sees only the new `input`, so an identical hash does not mean an identical context. This is a design risk, not a test result.
2. **Streaming.**
   - The raw SDK, LangChain, pydantic-ai and the Agents SDK all work for pass-through and for the synthesized predicted stream.
   - Bug I hit: I first logged after the last SSE line. The SDK stops reading at `[DONE]`, so the harness read `traces.jsonl` before the row was written (an `IndexError` in the harness). Fix: log before the terminal line goes out, with a `finally` for non-SSE bodies and hang-ups.
   - The upstream status has to be read before building the `StreamingResponse` (`http.send(..., stream=True)`). Otherwise upstream errors become a 200 with an error body.
   - Streamed tool calls are only flagged `tool_calls=true` in the trace, not reassembled.
3. **Tool-call passthrough.**
   - The body is forwarded unchanged, so `tools`, `tool_choice` and `role: "tool"` turns work.
   - Tool-call turns have empty `content`. If they were stored as the "answer", a predictor would replay an empty string. The trace keeps `tool_calls` separate and the trainer must skip those rows.
   - The hash covers only the messages, not `model`, `tools`, `response_format` or `temperature`. Two requests with the same messages but different tools or schemas would collide. A real deployment must hash those too. I did not.
4. **Headers.**
   - Custom headers survive in all three frameworks, so none strips the header. LangChain works with `default_headers` and with `.bind(extra_headers=)`, and pydantic-ai works with `model_settings.extra_headers`.
   - The proxy replaces `Authorization` with its own key and forwards no client headers upstream.
   - The `sha1(system prompt)` fallback fragments if a system prompt embeds dates, ids or retrieved text. I did not measure how often that happens in real agents.
5. **Agents SDK model strings.** In global mode the string `openai/gpt-5.4-nano` reached the proxy as `gpt-5.4-nano` (the SDK strips the `openai/` prefix; OpenRouter accepted the bare name). Other vendors' slugs would be routed by the SDK's own prefix logic, so use an explicit model class to keep slugs intact. I set `set_tracing_disabled(True)` from the start and did not test what happens without it.
6. **pydantic-ai 2.57:** `OpenAIModel` no longer exists (use `OpenAIChatModel`). It prints a banner to stdout, which breaks "last stdout line is JSON" parsing in the harness unless `PYDANTIC_AI_NO_BANNER=1` is set. `run.py` parses the last line only.
7. **fastmcp 4.1:** `FastMCP.as_proxy` is gone (use `create_proxy`). A cached reply must include `structured_content={"result": text}`. Otherwise the client raises `Tool vendor_lookup has an output schema but did not return structured content`.
8. **Fake response acceptance.** A minimal OpenAI-shaped body (id, object, created, model, choices, zero usage, extra `dreaming` field) was accepted by the raw SDK, LangChain, Agents SDK and pydantic-ai. LangChain, the Agents SDK and pydantic-ai drop the `dreaming` field, so only raw-SDK clients can see the flag. For the frameworks the trace is the only proof. A response header would fix this; I did not add one.
9. **Windows process flake.**
   - A client subprocess exited abnormally after printing its result in two of the harness iterations: pydantic-ai once with exit code 0xC0000005 (access violation), and the Agents SDK in global mode once with empty stderr and an unrecorded exit code.
   - Both happened while other smokes were loading the machine. The final run had none. Cause not found.
   - `run.py` now logs a `[WARN]` and continues if a client printed its result and then died. Separately, the Agents SDK sometimes prints `Cancelling an overlapped future failed ... WinError 6` to stderr at exit.
   - Orphaned proxies kept port 8765 bound twice (a `nohup` start, and a harness run that died before its `finally`). Check the port before restarting.

## Interpretation for the mission

- **Feasible: yes, for the plumbing.** Three frameworks and the raw SDK all route model calls through one 98-line proxy. Call-site headers arrive, calls are logged with latency and usage, and a hit in `predictors.json` returns a body every framework accepts. This holds for chat, streaming and Responses. In this smoke the round trip on a hit is about 10 ms from the SDK, and no proxy overhead was measurable against upstream jitter.
- **What was not shown.** No student, no certificate, and no generalisation. A hit means the exact same rendered messages were seen before. In real agents prompts contain timestamps, ids and retrieved text, so exact hashes will rarely recur. A certified student also needs the typed input fields. The proxy only sees a rendered prompt, so recovering fields from arbitrary messages (template extraction, or the call-site declaring its schema) is the open piece.
- **"Change one line" is only true for untagged grouping.** Per-call-site tagging needs a header per client: one ChatOpenAI instance or `.bind` per LangGraph node, one AsyncOpenAI client per Agents SDK call site. That is a small change per call site, not one line per agent.
- **MCP.** The proxy intercepts both the request (rewrite arguments) and the result (observe, replace, short-circuit). The cached answer worked across a stdio process boundary, and during a simulated outage a cached key kept answering while an unseen key failed through. That is the mission-3 "tool keeps working while down" behaviour at the protocol level, with the cache standing in for a certified predictor.
- **What breaks.**
  - The hash is too narrow (no `tools`, `response_format`, `model`, `temperature`, `previous_response_id`).
  - Predicted streaming is chat-only.
  - Streamed tool calls are not reassembled.
  - A non-JSON upstream error body would 500 the proxy.
  - There is no auth, no multi-tenancy, and one shared OpenRouter key.

## Caveats

- Prompts are single-sentence toy classifications, run once each. Latency numbers are single runs, and the overhead comparison is 8 pairs.
- The second-call answers match because I copied the first answer into `predictors.json`. That is replay, not prediction. Temperature 0 on a nano model gives the same text here but is not guaranteed in general.
- Only the OpenAI-compatible surface through OpenRouter was exercised. No Anthropic-native or Gemini-native endpoints. No concurrent load. Deviation from the shared rules: no `ThreadPoolExecutor` and no retries, because the harness is sequential and every call is a single shot.
- MCP was exercised with fastmcp's own client against a same-process backend. A real third-party MCP server over HTTP/SSE, and clients such as Claude Desktop, were not tested.
- A real deployment changes the following: per-tenant keys and auth; hashing the full request shape; a sampled shadow call on hits (the proxy never calls upstream on a hit, so there is no certificate upkeep yet); template normalisation for untagged sites; and a place for the student to run.

## Time and cost

- Wall time of the final harness run: 160 s. The whole smoke took about 55 minutes including installs, MCP API discovery and six harness runs (CPU contention from other smokes was present).
- OpenRouter spend: $0.00042 for the final run, about $0.003 for everything (cap $1). The line is appended to `smokes/COSTS.md`.
