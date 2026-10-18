"""T2b: replay the live T2 runs through the real decorated handler with a fake ctx whose network methods explode; plus the soft-failure and divergence cases."""
import asyncio, json, uuid
from uagents_core.types import DeliveryStatus, MsgStatus
import dreaming_wrap as dw
from t2_common import *
import t2_requester as rq   # imports the real handler; its module-level Recorder is replaced below
TR = rq.TR

class Fake:  # same method signatures as uagents' ExternalContext
    session = uuid.uuid4()
    async def send_and_receive(self, destination, message, response_type, sync=False, timeout=30): raise AssertionError("network: send_and_receive executed in replay")
    async def send(self, destination, message, timeout=30): raise AssertionError("network: send executed in replay")

R = [json.loads(l) for l in TR.read_text(encoding="utf-8").splitlines()]
runs = [r["run_id"] for r in R if r["kind"] == "run"]
async def main():
    for rid in runs:
        rows = [r for r in R if r["run_id"] == rid]
        vendor = next(r for r in rows if r["kind"] == "a2a")["request"]["body"]["vendor"]
        logged = next(r for r in rows if r["kind"] == "effect")["request"]["body"]["text"]  # what the live handler sent back
        rec = dw.Recorder("replay", TR, run_id=rid)
        got = await rq.chat(Fake(), ADDR["driver"], Ask(text=vendor))
        print(f"  replay run={rid} ask={vendor!r:9} -> {got!r} logged={logged!r} same={got == logged} divergence={rec.divergence}")
        assert got == logged and rec.divergence is None
    rec = dw.Recorder("replay", TR, run_id=runs[0])  # run 1 asked globex; ask acme instead -> same call site, different request
    try: await rq.chat(Fake(), ADDR["driver"], Ask(text="acme")); raise SystemExit("expected Divergence")
    except dw.Divergence as e: print("  divergence:", str(e)[:150]); assert rec.divergence["row"] == 1

    # soft failure without a registered fallback: uAgents returns (None, status) instead of raising; the wrapper passes it through and logs it
    class Down(Fake):
        async def send_and_receive(self, destination, message, response_type, sync=False, timeout=30):
            return None, MsgStatus(status=DeliveryStatus.FAILED, detail="connection refused", destination=destination, endpoint="", session=self.session)
    f = TR.parent / "t2b_traces.jsonl"; f.unlink(missing_ok=True); dw.Recorder("record", f)
    @dw.run_boundary
    async def h(ctx): return await ctx.send_and_receive("agent1qdest", VendorLookup(vendor="x"), response_type=VendorInfo)
    out = await h(Down()); row = json.loads(f.read_text(encoding="utf-8").splitlines()[0])
    print("  no fallback ->", out[0], out[1].status.value, "| logged error:", row["error"], "| site:", row["call_site"])
    assert out[0] is None and row["error"] == "connection refused" and row["call_site"] == "agent1qdest|VendorLookup"

    # review finding 2a: replaying that soft failure must hand the handler the same (None, MsgStatus), not a raised RuntimeError
    rec = dw.Recorder("replay", f, run_id=row["run_id"]); again = await h(Fake())  # Fake's network methods raise: everything comes from the log
    print("  soft failure replay ->", again[0], again[1].status.value, repr(again[1].detail), "| divergence:", rec.divergence)
    assert again[0] is None and again[1].status == DeliveryStatus.FAILED and again[1].detail == "connection refused" and rec.divergence is None
    @dw.run_boundary
    async def branchy(ctx):  # a handler that branches on `if info is None`
        info, st = await ctx.send_and_receive("agent1qdest", VendorLookup(vendor="x"), response_type=VendorInfo)
        return "lookup failed: " + st.detail if info is None else "ok"
    dw.Recorder("record", f); live = await branchy(Down()); rid = json.loads(f.read_text(encoding="utf-8").splitlines()[-1])["run_id"]
    dw.Recorder("replay", f, run_id=rid); same = await branchy(Fake()); print("  branch on `info is None`:", live, "->", same); assert live == same == "lookup failed: connection refused"

    # review finding 3: ctx.send called with keyword args must not crash the host handler (positional-only indexing did)
    class Ok(Fake):
        ok = lambda self, dest: MsgStatus(status=DeliveryStatus.DELIVERED, detail="", destination=dest, endpoint="", session=self.session)
        async def send(self, destination, message, timeout=30): return self.ok(destination)
        async def send_and_receive(self, destination, message, response_type, sync=False, timeout=30): return VendorInfo(vendor="k", terms="net30"), self.ok(destination)
    g = TR.parent / "t2b_send_traces.jsonl"; g.unlink(missing_ok=True); dw.Recorder("record", g)
    @dw.run_boundary
    async def sends(ctx):
        m = Ask(text="hi"); a = await ctx.send("agent1qd", m); b = await ctx.send(destination="agent1qd", message=m); c = await ctx.send("agent1qd", message=m, timeout=5)
        d, _ = await ctx.send_and_receive(destination="agent1qd", message=VendorLookup(vendor="k"), response_type=VendorInfo, timeout=1)
        return [x.status.value for x in (a, b, c)] + [d.terms]
    live = await sends(Ok()); rr = [json.loads(l) for l in g.read_text(encoding="utf-8").splitlines()]
    print("  keyword sends:", live, "| effect rows:", [r["call_site"][-8:] for r in rr if r["kind"] == "effect"]); assert live == ["delivered"] * 3 + ["net30"] and len([r for r in rr if r["kind"] == "effect"]) == 3
    dw.Recorder("replay", g, run_id=rr[0]["run_id"]); assert await sends(Fake()) == live
asyncio.run(main()); print("T2b OK")
