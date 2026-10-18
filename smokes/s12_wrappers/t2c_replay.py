"""T2c replay: every logged chat run through the real decorated handler with a fake ctx (network methods raise) and the real wrapped OpenRouter client (any request would raise ReplayLeak)."""
import asyncio, json, uuid
from uagents_core.contrib.protocols.chat import ChatMessage, TextContent
import dreaming_wrap as dw
from t2_common import *
import t2c_agent as ag
TR = ag.TR

class Fake:
    session = uuid.uuid4()
    async def send_and_receive(self, destination, message, response_type, sync=False, timeout=30): raise AssertionError("network: send_and_receive executed in replay")
    async def send(self, destination, message, timeout=30): raise AssertionError("network: send executed in replay")

R = [json.loads(l) for l in TR.read_text(encoding="utf-8").splitlines()]
async def main():
    for rid in [r["run_id"] for r in R if r["kind"] == "run"]:
        rows = [r for r in R if r["run_id"] == rid and r["kind"] != "run"]
        vendor = next(r for r in rows if r["kind"] == "a2a")["request"]["body"]["vendor"]
        logged = [r for r in rows if r["kind"] == "effect"][-1]["request"]["body"]["content"][0]["text"]  # the reply the live handler sent
        rec = dw.Recorder("replay", TR, run_id=rid)
        got = await ag.on_chat(Fake(), ADDR["driver"], ChatMessage(content=[TextContent(type="text", text=vendor)]))  # new msg_id and timestamps: effects are hashed by destination and type only
        print(f"  replay run={rid} chat={vendor!r:9} -> {got!r} same={got == logged} divergence={rec.divergence} leaks={rec.leaks}")
        assert got == logged and rec.divergence is None and not rec.leaks
    # negative control: an unwrapped call (embeddings) on the same real client is refused in replay mode before any request is sent
    rec = dw.Recorder("replay", TR, run_id=R[0]["run_id"])
    @dw.run_boundary
    async def leak(): await ag.llm.embeddings.create(model="x", input="y")
    try: await leak(); raise SystemExit("expected ReplayLeak")
    except dw.ReplayLeak as e: print("  negative control:", str(e)[:110]); assert rec.leaks
asyncio.run(main()); print("T2c replay OK")
