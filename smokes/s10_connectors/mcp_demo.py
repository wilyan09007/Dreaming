"""fastmcp: a vendor_lookup tool behind a Dreaming MCP proxy. The proxy middleware sees the request (name, args) and the result,
logs both to mcp_traces.jsonl, can rewrite args, and can answer from a cache (mcp_cache.json) without calling the backend.
Run: python mcp_demo.py          (in-memory client + one stdio client)   python mcp_demo.py serve   (the proxy as a stdio server)"""
import asyncio, json, os, sys, time
from pathlib import Path
from fastmcp import Client, FastMCP
from fastmcp.client.transports import PythonStdioTransport
from fastmcp.server import create_proxy
from fastmcp.server.middleware import Middleware
from fastmcp.tools import ToolResult

H = Path(__file__).parent
LOG, CACHE = H / "mcp_traces.jsonl", H / "mcp_cache.json"
backend, state = FastMCP("vendors"), {"calls": [], "down": False}

@backend.tool
def vendor_lookup(descriptor: str) -> str:
    """Resolve a merchant descriptor to a vendor name."""
    state["calls"].append(descriptor)
    if state["down"]: raise RuntimeError("vendor API unavailable")
    time.sleep(0.2)  # pretend network
    return {"ADOBE *CREATIVE CLD": "Adobe Inc"}.get(descriptor, "unknown vendor")

class Tap(Middleware):
    async def on_call_tool(self, context, call_next):
        m, t0 = context.message, time.perf_counter()
        args = {**(m.arguments or {}), "descriptor": m.arguments["descriptor"].strip().upper()}  # request interception: normalize before forwarding
        context = context.copy(message=m.model_copy(update={"arguments": args}))
        k, cache = f"{m.name}|{json.dumps(args, sort_keys=True)}", json.loads(CACHE.read_text()) if CACHE.exists() else {}
        predicted = k in cache
        # result interception: answer from cache (structured_content must match the tool's output schema or clients reject it) or pass through
        res = ToolResult(content=cache[k], structured_content={"result": cache[k]}) if predicted else await call_next(context)
        text = res.content[0].text  # backend failures arrive here as res.is_error=True, not as exceptions
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(pid=os.getpid(), tool=m.name, args=args, ret=text, error=text if res.is_error else None, predicted=predicted,
                                    latency_ms=round((time.perf_counter() - t0) * 1000, 1))) + chr(10))
        return res

proxy = create_proxy(backend, name="dreaming-mcp"); proxy.add_middleware(Tap())
rows = lambda: [json.loads(l) for l in LOG.read_text(encoding="utf-8").splitlines()] if LOG.exists() else []

async def main():
    LOG.unlink(missing_ok=True); CACHE.unlink(missing_ok=True)
    async with Client(proxy) as c:  # in-memory transport
        print("tools via proxy:", [t.name for t in await c.list_tools()])
        r1 = await c.call_tool("vendor_lookup", {"descriptor": "adobe *creative cld "})  # sloppy casing: normalized by the tap
        r = rows()[-1]; assert r["args"]["descriptor"] == "ADOBE *CREATIVE CLD" and r["ret"] == "Adobe Inc" and not r["predicted"], r
        assert state["calls"] == ["ADOBE *CREATIVE CLD"], state  # backend received the REWRITTEN arg
        CACHE.write_text(json.dumps({f"{r['tool']}|{json.dumps(r['args'], sort_keys=True)}": r["ret"]}))  # copy first answer into the cache
        r2 = await c.call_tool("vendor_lookup", {"descriptor": "ADOBE *CREATIVE CLD"})
        assert rows()[-1]["predicted"] and r2.content[0].text == r1.content[0].text and len(state["calls"]) == 1, (rows()[-1], state)
        state["down"] = True  # outage: cached key still answers, unseen key fails through to the real (dead) tool
        r3 = await c.call_tool("vendor_lookup", {"descriptor": "ADOBE *CREATIVE CLD"}); assert r3.content[0].text == "Adobe Inc"
        try: await c.call_tool("vendor_lookup", {"descriptor": "NEW MERCHANT"}); raise SystemExit("expected error")
        except Exception as e: assert "unavailable" in str(e) or "Error" in str(e), e
        assert rows()[-1]["error"] and not rows()[-1]["predicted"], rows()[-1]
        state["down"] = False
    async with Client(PythonStdioTransport(__file__, args=["serve"], python_cmd=sys.executable)) as c:  # stdio: the proxy runs as a child process
        n = len(rows())
        assert (await c.call_tool("vendor_lookup", {"descriptor": "Adobe *Creative Cld"})).content[0].text == "Adobe Inc"
        assert (await c.call_tool("vendor_lookup", {"descriptor": "starbucks #12"})).content[0].text == "unknown vendor"
        new = rows()[n:]; assert [x["predicted"] for x in new] == [True, False] and new[0]["pid"] != os.getpid(), new

if __name__ == "__main__":
    if "serve" in sys.argv: proxy.run()
    else: asyncio.run(main()); print("rows:"); [print(" ", json.dumps(r)) for r in rows()]; print("mcp ok")
