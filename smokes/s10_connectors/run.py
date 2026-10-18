"""S10 harness. Starts proxy.py, runs each framework client twice (real call, then answered from predictors.json), runs protocol
probes (streaming, tool calls, Responses API, untagged calls, errors, overhead), runs the MCP demo, checks traces.jsonl, sums spend.
Run: python run.py | tee log.txt"""
import json, os, socket, statistics as st, subprocess, sys, time
from pathlib import Path
from openai import OpenAI, APIStatusError
from langchain_openai import ChatOpenAI

H, PY, T0 = Path(__file__).parent, sys.executable, time.time()
TR, PR = H / "traces.jsonl", H / "predictors.json"
BASE, M, PRICE = "http://127.0.0.1:8765/v1", "openai/gpt-5.4-nano", (0.2, 1.25)  # USD per 1M tokens in/out
KEY = dict(l.split("=", 1) for l in (H.parents[1] / ".env").read_text().splitlines() if "=" in l)["OPENROUTERKEY"].strip().strip('"')
env = {**os.environ, "HF_HUB_DISABLE_SYMLINKS_WARNING": "1", "PYDANTIC_AI_NO_BANNER": "1", "PYTHONIOENCODING": "utf-8"}

def rows(): return [json.loads(l) for l in TR.read_text(encoding="utf-8").splitlines()] if TR.exists() else []
def learn(new):  # stand-in for the trainer: copy real text answers into predictors.json (skips errors, tool-call turns, empty streams)
    p = json.loads(PR.read_text()) if PR.exists() else {}
    for r in new:
        if r["status"] == 200 and not r["predicted"] and not r["tool_calls"] and r["response"]:
            p.setdefault(r["call_site"], {"answers": {}})["answers"][r["msg_sha256"]] = r["response"]
    PR.write_text(json.dumps(p, indent=1))
def sh(*cmd):  # a client that printed its result but died at interpreter exit (observed: 0xC0000005 on Windows under load) still counts; the warning is logged
    r = subprocess.run([PY, "-X", "faulthandler", *cmd], cwd=H, env=env, capture_output=True, text=True, timeout=300, encoding="utf-8")
    assert r.stdout.strip(), (cmd, r.returncode, r.stderr[-1500:])
    if r.returncode: print(f"  [WARN] {' '.join(cmd)} exited {r.returncode & 0xFFFFFFFF:#x} after printing its result; stderr tail: {r.stderr[-400:]!r}")
    return r.stdout
def usd(r): u = r["usage"] or {}; return (u.get("prompt_tokens", u.get("input_tokens", 0)) * PRICE[0] + u.get("completion_tokens", u.get("output_tokens", 0)) * PRICE[1]) / 1e6
def oa(site=None): return OpenAI(base_url=BASE, api_key="x", default_headers={"X-Dreaming-Call-Site": site} if site else None)
def twice(call):  # call, learn from the trace rows it produced, call again; returns (out1, out2, rows1, rows2)
    n = len(rows()); a = call(); r1 = rows()[n:]; learn(r1); n = len(rows()); b = call(); return a, b, r1, rows()[n:]
def check(name, ok, detail=""): print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}"); assert ok, name

p = subprocess.Popen([PY, "proxy.py"], cwd=H, env=env, stdout=open(H / "proxy.log", "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(80):
        try: socket.create_connection(("127.0.0.1", 8765), 1).close(); break
        except OSError: time.sleep(0.5)
    TR.unlink(missing_ok=True); PR.unlink(missing_ok=True)

    print("== A. framework clients: pass 1 real, pass 2 from predictors.json")
    CL = {"LangGraph": ["client_langgraph.py"], "Agents SDK (chat model)": ["client_agents.py", "chat"], "Agents SDK (global default)": ["client_agents.py", "global"],
          "Agents SDK (responses)": ["client_agents.py", "responses"], "pydantic-ai": ["client_pydanticai.py"],
          "Agents SDK (chat, streamed)": ["client_agents.py", "chat", "stream"], "pydantic-ai (streamed)": ["client_pydanticai.py", "stream"]}
    for name, cmd in CL.items():
        n = len(rows()); o1 = sh(*cmd).strip().splitlines()[-1]; r1 = rows()[n:]; learn(r1)
        n = len(rows()); o2 = sh(*cmd).strip().splitlines()[-1]; r2 = rows()[n:]
        sites = [r["call_site"] for r in r1]
        print(f"{name}: sites={sites} apis={sorted({r['api'] for r in r1})} real_ms={[r['latency_ms'] for r in r1]} predicted_ms={[r['latency_ms'] for r in r2]} out={o1[:90]}")
        check("header arrived (no sha1: fallback site)", r1 and not any(s.startswith("sha1:") for s in sites))
        check("pass 1 hit upstream, pass 2 predicted, same hashes", not any(r["predicted"] for r in r1) and all(r["predicted"] for r in r2) and [r["msg_sha256"] for r in r1] == [r["msg_sha256"] for r in r2])
        check("pass 2 returned the same final output with zero upstream tokens", o1 == o2 and all(r["usage"] is None for r in r2))

    print("== B. protocol probes (OpenAI SDK direct)")
    msgs = [{"role": "system", "content": "Reply with one word."}, {"role": "user", "content": "Capital of France?"}]
    a, b, r1, r2 = twice(lambda: oa("probe-chat").chat.completions.create(model=M, messages=msgs, max_tokens=60, temperature=0))
    check("chat: client sees dreaming.predicted on 2nd call", b.model_extra.get("dreaming") == {"predicted": True} and a.model_extra.get("dreaming") is None and a.choices[0].message.content == b.choices[0].message.content, a.choices[0].message.content)
    sc = lambda: "".join(c.choices[0].delta.content or "" for c in oa("probe-stream").chat.completions.create(model=M, messages=msgs, max_tokens=60, stream=True, stream_options={"include_usage": True}) if c.choices)
    a, b, r1, r2 = twice(sc)
    check("stream (openai sdk): logged text+usage, 2nd call predicted", a == b != "" and r1[0]["stream"] and r1[0]["response"] == a and r1[0]["usage"] and r2[0]["predicted"], a)
    lcs = lambda: "".join(c.content for c in ChatOpenAI(model=M, base_url=BASE, api_key="x", max_tokens=60, default_headers={"X-Dreaming-Call-Site": "probe-lc-stream"}).stream("Say hi in one word."))
    a, b, r1, r2 = twice(lcs)
    check("stream (langchain): 2nd call predicted", a == b != "" and r2[0]["predicted"] and r2[0]["stream"], a)
    a, b, r1, r2 = twice(lambda: oa("probe-responses").responses.create(model=M, instructions="Reply with one word.", input="Capital of Italy?", max_output_tokens=60))
    check("/v1/responses: logged, 2nd call predicted, client sees flag", a.output_text == b.output_text != "" and r1[0]["api"] == "responses" and r2[0]["predicted"] and b.model_extra.get("dreaming") == {"predicted": True}, a.output_text)

    lcr = lambda: str(ChatOpenAI(model=M, base_url=BASE, api_key="x", use_responses_api=True, max_tokens=60, default_headers={"X-Dreaming-Call-Site": "probe-lc-responses"}).invoke("Say hi in one word.").text)
    a, b, r1, r2 = twice(lcr)
    check("langchain use_responses_api=True via /v1/responses: 2nd call predicted", a == b != "" and r1[0]["api"] == "responses" and r2[0]["predicted"], a)
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider
    pr = lambda: Agent(OpenAIResponsesModel(M, provider=OpenAIProvider(base_url=BASE, api_key="x")), model_settings={"max_tokens": 60, "extra_headers": {"X-Dreaming-Call-Site": "probe-pai-responses"}}).run_sync("Say hi in one word.").output
    a, b, r1, r2 = twice(pr)
    check("pydantic-ai OpenAIResponsesModel via /v1/responses: 2nd call predicted", a == b != "" and r1[0]["api"] == "responses" and r2[0]["predicted"], a)

    tool = {"type": "function", "function": {"name": "vendor_lookup", "description": "Resolve a merchant descriptor", "parameters": {"type": "object", "properties": {"descriptor": {"type": "string"}}, "required": ["descriptor"]}}}
    tm = [{"role": "user", "content": "Look up the vendor for ADOBE *CREATIVE CLD"}]
    kw = dict(model=M, max_tokens=60, tools=[tool], tool_choice={"type": "function", "function": {"name": "vendor_lookup"}})
    a, b, r1, r2 = twice(lambda: oa("probe-tools").chat.completions.create(messages=tm, **kw))
    tc = a.choices[0].message.tool_calls[0]
    check("tool call passes through; logged as tool_calls, never learned", tc.function.name == "vendor_lookup" and r1[0]["tool_calls"] and r1[0]["response"] == "" and not r2[0]["predicted"], tc.function.arguments)
    fu = tm + [a.choices[0].message.model_dump(exclude_none=True), {"role": "tool", "tool_call_id": tc.id, "content": "Adobe Inc"}]
    n = len(rows()); f = oa("probe-tools").chat.completions.create(messages=fu, model=M, max_tokens=60, tools=[tool])
    check("tool result turn (role=tool) passes through", f.choices[0].message.content and rows()[n:][0]["response"] == f.choices[0].message.content, f.choices[0].message.content[:60])

    os.environ.update(OPENAI_BASE_URL=BASE, OPENAI_API_KEY="x")  # zero-code-change path: env vars only, no header
    n = len(rows()); OpenAI().chat.completions.create(model=M, messages=msgs, max_tokens=60); OpenAI().chat.completions.create(model=M, messages=[msgs[0], {"role": "user", "content": "Capital of Spain?"}], max_tokens=60)
    OpenAI().chat.completions.create(model=M, messages=[{"role": "system", "content": "Reply in French, one word."}, msgs[1]], max_tokens=60)
    s = [r["call_site"] for r in rows()[n:]]
    check("untagged calls (OPENAI_BASE_URL only) group by sha1(system prompt)", s[0] == s[1] != s[2] and s[0].startswith("sha1:"), str(s))
    n = len(rows())
    try: oa("probe-error").chat.completions.create(model="openai/does-not-exist-9", messages=msgs, max_tokens=60); code = None
    except APIStatusError as e: code = e.status_code
    check("upstream error relayed with its status and logged", code in (400, 404) and rows()[n:][0]["status"] == code, f"status={code}")

    print("== C. overhead: 8 paired calls, direct OpenRouter vs through proxy (unique prompts, never predicted)")
    def ms(c, q): t = time.perf_counter(); c.chat.completions.create(model=M, messages=[{"role": "user", "content": q}], max_tokens=20, temperature=0); return (time.perf_counter() - t) * 1000
    direct, via = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=KEY), oa("overhead")
    d, v = zip(*[(ms(direct, f"Capital of {c}? One word."), ms(via, f"Capital of {c}? One word.")) for c in ["Peru", "Chad", "Laos", "Fiji", "Cuba", "Iran", "Mali", "Togo"]])
    hit = []
    for _ in range(10): t = time.perf_counter(); oa("probe-chat").chat.completions.create(model=M, messages=msgs, max_tokens=60, temperature=0); hit.append((time.perf_counter() - t) * 1000)
    print(f"  median ms direct={st.median(d):.0f} proxied={st.median(v):.0f} (delta {st.median(v) - st.median(d):+.0f}); predicted-hit round trip median={st.median(hit):.0f} min={min(hit):.0f}")

    print("== D. MCP (fastmcp) proxy")
    out = sh("mcp_demo.py"); print("  " + "\n  ".join(l for l in out.splitlines() if l.startswith(("tools", "  {", "mcp"))))
    check("mcp demo asserts passed", "mcp ok" in out)

    R = rows(); real = [r for r in R if not r["predicted"] and r["usage"]]
    cost = sum(usd(r) for r in real); reported = sum((r["usage"].get("cost") or 0) for r in real)
    print(f"== trace rows={len(R)} real={len(real)} predicted={sum(r['predicted'] for r in R)}  calls_by_site=" + json.dumps({s: sum(r['call_site'] == s for r in R) for s in sorted({r['call_site'] for r in R})}))
    print(f"== spend (this run, tokens x gpt-5.4-nano price) = ${cost:.5f}; OpenRouter-reported cost field = ${reported:.5f}; wall = {time.time() - T0:.0f}s")
    with open(H.parent / "COSTS.md", "a", encoding="utf-8") as f: f.write(f"s10,{cost:.5f},gpt-5.4-nano via local proxy; {len(real)} upstream calls (excludes ~$0.0005 of dev runs)\n")
finally:
    p.terminate()
