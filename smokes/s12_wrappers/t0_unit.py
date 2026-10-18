"""T0: offline unit checks of the OpenAI wrapper and run boundary against an httpx.MockTransport (no network, no cost)."""
import asyncio, concurrent.futures, contextvars, dataclasses, datetime, hashlib, json, uuid
import httpx, openai, pydantic
import dreaming_wrap as dw
from _t import HERE

TR = HERE / "t0_traces.jsonl"; TR.unlink(missing_ok=True)
HITS = []
def handler(req):
    if req.url.path.endswith("/embeddings"):
        HITS.append({"embeddings": True}); return httpx.Response(200, json={"object": "list", "model": "e", "usage": {"prompt_tokens": 1, "total_tokens": 1}, "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}]})
    HITS.append(json.loads(req.content)); body = HITS[-1]
    if body["messages"][-1]["content"] == "500": return httpx.Response(500, json={"error": {"message": "boom"}})
    msg = {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "vendor_lookup", "arguments": "{\"vendor\":\"acme\"}"}}]} \
        if body.get("tools") else {"role": "assistant", "content": "hello " + body["messages"][-1]["content"]}
    return httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 1, "model": body["model"], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                                     "choices": [{"index": 0, "finish_reason": "tool_calls" if body.get("tools") else "stop", "message": msg}]})
mk = lambda retries=0: openai.OpenAI(api_key="k", base_url="http://mock/v1", max_retries=retries, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
amk = lambda: openai.AsyncOpenAI(api_key="k", base_url="http://mock/v1", max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
def rows(kind=None): return [r for r in map(json.loads, TR.read_text(encoding="utf-8").splitlines()) if kind in (None, r["kind"])]
SYS = "You are terse."
def call(cl, user, **kw): return cl.chat.completions.create(model="m", messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}], **kw)

dw.Recorder("record", TR); cl = dw.wrap_openai(mk())

# a. call-site resolution: untagged -> hash of system prompt; context manager; header beats context manager
@dw.run_boundary
def sites():
    call(cl, "a")
    with dw.call_site("ctx"): call(cl, "b")
    with dw.call_site("ctx"): call(cl, "c", extra_headers={"X-Dreaming-Call-Site": "hdr"})
sites(); got = [r["call_site"] for r in rows("openai")]
print("sites:", got); assert got == ["sys:" + hashlib.sha1(SYS.encode()).hexdigest()[:12], "ctx", "hdr"]

# b. tools are logged; a tool-call response (content None) survives a replay as a real ChatCompletion
TOOLS = [{"type": "function", "function": {"name": "vendor_lookup", "parameters": {"type": "object", "properties": {"vendor": {"type": "string"}}}}}]
@dw.run_boundary
def with_tools(): return call(cl, "look up acme", tools=TOOLS).choices[0].message.tool_calls[0].function.name
live = with_tools(); run = rows("run")[-1]["run_id"]; r = [x for x in rows("openai") if x["run_id"] == run][0]
assert r["request"]["tools"] == TOOLS and r["text"] is None and live == "vendor_lookup"
n = len(HITS); dw.Recorder("replay", TR, run_id=run); again = with_tools()
print("tool-call replay:", again, "| extra HTTP:", len(HITS) - n); assert again == live and len(HITS) == n

# c. errors: logged, re-raised, run row carries the error, run id is reset afterwards
dw.Recorder("record", TR)
@dw.run_boundary
def fails(): call(cl, "500")
try: fails(); raise SystemExit("should raise")
except openai.InternalServerError: pass
rr = ERR_RUN = rows("run")[-1]; print("error run:", rr["error"][:60], "| row error:", rows("openai")[-1]["error"][:60]); assert "InternalServerError" in rr["error"] and rows("openai")[-1]["response"] is None and dw.RUN.get() is None

# d. nested boundaries join the outer run; concurrent calls from copied contexts get unique, gapless indices
@dw.run_boundary
def inner(): return call(cl, "in").choices[0].message.content
@dw.run_boundary
def outer():
    inner()
    ctxs = [contextvars.copy_context() for _ in range(20)]  # copy in the SUBMITTING thread: a copy taken inside the worker would not have the run id
    with concurrent.futures.ThreadPoolExecutor(8) as ex: list(ex.map(lambda p: p[0].run(call, cl, f"p{p[1]}"), zip(ctxs, range(20))))
outer(); run = rows("run")[-1]; rs = [x for x in rows("openai") if x["run_id"] == run["run_id"]]
print("nested+parallel: run calls =", run["calls"], "| distinct indices =", len({x["i"] for x in rs})); assert run["calls"] == 21 and sorted(x["i"] for x in rs) == list(range(1, 22)) and len([x for x in rows("run") if x["run_id"] == run["run_id"]]) == 1

# e. streaming: unlogged pass-through in record mode, refused in replay mode
dw.Recorder("replay", TR, run_id=run["run_id"])
@dw.run_boundary
def streams(): return call(cl, "x", stream=True)
try: streams(); raise SystemExit("should refuse")
except NotImplementedError as e: print("stream in replay:", e)

# f. async client against the mock
async def amain():
    dw.Recorder("record", TR); acl = dw.wrap_openai(amk())
    @dw.run_boundary
    async def ah(): return (await call(acl, "async")).choices[0].message.content
    a = await ah(); run = rows("run")[-1]["run_id"]; n = len(HITS)
    dw.Recorder("replay", TR, run_id=run); b = await ah(); print("async:", a, "->", b, "| extra HTTP:", len(HITS) - n); assert a == b == "hello async" and len(HITS) == n
asyncio.run(amain())

# g. with_options() / copy() return a NEW client whose create was not patched: they used to bypass the log and, in replay, the network was hit (review finding 1)
dw.Recorder("record", TR); cl = dw.wrap_openai(mk()); n = len(HITS)
@dw.run_boundary
def clones(): return [call(c, t).choices[0].message.content for c, t in ((cl, "w"), (cl.with_options(timeout=5), "o"), (cl.copy(), "c"))]
live = clones(); run = rows("run")[-1]; print("with_options/copy: calls logged =", run["calls"], "| HTTP =", len(HITS) - n); assert run["calls"] == 3 and len(HITS) - n == 3 and live == ["hello w", "hello o", "hello c"]
n = len(HITS); dw.Recorder("replay", TR, run_id=run["run_id"]); again = clones(); print("  replay:", again == live, "| extra HTTP:", len(HITS) - n); assert again == live and len(HITS) == n

# h. replay-mode transport guard: any request that is not a served wrapped call (embeddings, responses, parse, streams) fails loudly, default SDK retries included
dw.Recorder("record", TR); cl2 = dw.wrap_openai(mk(retries=2))
@dw.run_boundary
def leaky(swallow=False):
    out = call(cl2, "q").choices[0].message.content
    try: cl2.embeddings.create(model="e", input="x")
    except Exception:
        if not swallow: raise
    return out
leaky(); run = rows("run")[-1]["run_id"]; n = len(HITS); rec = dw.Recorder("replay", TR, run_id=run)
try: leaky(); raise SystemExit("expected ReplayLeak")
except dw.ReplayLeak as e: print("leak (raised):", str(e)[:100])
assert len(HITS) == n and len(rec.leaks) == 1
try: leaky(swallow=True); raise SystemExit("expected ReplayLeak")  # the handler swallowed the leak: the boundary still refuses
except dw.ReplayLeak as e: print("leak (swallowed):", str(e)[:100])
assert len(HITS) == n

# i. replay reproduces failures and return types exactly (review finding 2)
Pair = dataclasses.make_dataclass("Pair", [("net", str), ("ids", tuple)]); Pair.__module__ = "__main__"; globals()["Pair"] = Pair
class Mod(pydantic.BaseModel): vendor: str; limit: float
RICH = lambda: {"tuple": (1, ("a", 2)), "set": {3, 1}, "fs": frozenset({"x"}), "bytes": b"\x00\xff", "dc": Pair("net30", (1, 2)), "model": Mod(vendor="acme", limit=2.5),
                "dt": datetime.datetime(2026, 1, 2, 3, 4), "date": datetime.date(2026, 1, 2), "uuid": uuid.UUID(int=7), "kd": {1: "a", (1, 2): "b"}, "plain": {"$not": [1, None, True]}}
@dw.tool("rich")
def rich(kind):
    if kind == "timeout": raise TimeoutError("vendor db down", 7)
    if kind == "lossy": return object()
    return RICH()
@dw.run_boundary
def kinds():
    got = rich("ok")
    try: rich("timeout"); caught = None
    except TimeoutError as e: caught = ("TimeoutError", e.args)  # `except TimeoutError` must still match on replay
    return got, caught
dw.Recorder("record", TR); live = kinds(); run = rows("run")[-1]["run_id"]
rec = dw.Recorder("replay", TR, run_id=run); again = kinds()
print("rich returns:", sorted(type(v).__name__ for v in again[0].values()), "| caught:", again[1]); assert again == live and again[1] == ("TimeoutError", ("vendor db down", 7))
assert all(type(again[0][k]) is type(v) for k, v in RICH().items()) and again[0]["tuple"][1] == ("a", 2)
dw.Recorder("record", TR)
@dw.run_boundary
def lossy(): return rich("lossy")
lossy(); run = rows("run")[-1]["run_id"]; rec = dw.Recorder("replay", TR, run_id=run)
try: lossy(); raise SystemExit("expected Divergence")
except dw.Divergence as e: print("not reproducible ->", str(e)[:110]); assert rec.divergence["row"] == 1
# an openai API error replays as the same class with its attributes
dw.Recorder("replay", TR, run_id=ERR_RUN["run_id"]); n = len(HITS)
try: fails(); raise SystemExit("expected InternalServerError")
except openai.InternalServerError as e: print("openai error replay:", type(e).__name__, e.status_code, "| extra HTTP:", len(HITS) - n); assert e.status_code == 500 and len(HITS) == n
print("T0 OK")
