"""T2c: chat-protocol agent (ChatMessage handler = run boundary) doing a2a + a model call under ONE run id, between real processes: responder :8101, chat agent :8104, driver :8103 (this process).
Shows: 3 call kinds under one run id, predictors skip both the send and the model call, outage + fallback, recertify of the predicted a2a + model rows against the live responder/model, replay of every run."""
import asyncio, json, os, socket, subprocess, sys, time
import httpx, openai
from t2_common import *
from uagents import Protocol
from uagents_core.contrib.protocols.chat import ChatAcknowledgement, ChatMessage, TextContent, chat_protocol_spec
import dreaming_wrap as dw
from _t import HERE, KEY, spend
TR = HERE / "t2c_traces.jsonl"; TR.unlink(missing_ok=True)
env = dict(os.environ, PYTHONUTF8="1", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
def spawn(script, log): return subprocess.Popen([sys.executable, script], cwd=HERE, stdout=open(HERE / log, "w"), stderr=subprocess.STDOUT, env=env)
def up(port):
    for _ in range(100):
        try: socket.create_connection(("127.0.0.1", port), 0.2).close(); return
        except OSError: time.sleep(0.2)
    raise SystemExit(f"port {port} never came up")
resp, bot = spawn("t2_responder.py", "t2c_responder.log"), spawn("t2c_agent.py", "t2c_agent.log")
up(PORTS["responder"]); up(PORTS["chat"])
driver = make("driver"); proto = Protocol(spec=chat_protocol_spec); inbox, acks = asyncio.Queue(), []
@proto.on_message(ChatMessage)
async def got(ctx, sender, msg: ChatMessage): await inbox.put(msg.text())
@proto.on_message(ChatAcknowledgement)
async def ack(ctx, sender, msg: ChatAcknowledgement): acks.append(str(msg.acknowledged_msg_id))
driver.include(proto, publish_manifest=False)
def log_lines(name, prefix): return [l.strip() for l in (HERE / name).read_text(encoding="utf-8").splitlines() if l.startswith(prefix)]
def rows(): return [json.loads(l) for l in TR.read_text(encoding="utf-8").splitlines()]

@driver.on_event("startup")
async def scenario(ctx):
    try:
        async def say(text, label):
            t0 = time.time(); await ctx.send(ADDR["chat"], ChatMessage(content=[TextContent(type="text", text=text)]))
            r = await asyncio.wait_for(inbox.get(), 30)
            print(f"  say({text!r:9}) -> {r!r}  [{label}] {time.time() - t0:.2f}s | responder hits: {len(log_lines('t2c_responder.log', 'RESPONDER'))}, model HTTP: {len(log_lines('t2c_agent.log', 'MODEL HTTP'))}", flush=True); return r
        print("== live (a2a + model), then the same chat again (both predicted)")
        r1 = await say("acme", "live a2a + live model"); r2 = await say("acme", "predicted a2a + predicted model")
        assert r1 == r2 and "net30" in r1 and len(acks) == 2
        await asyncio.sleep(0.5)  # the run row is written after the handler's last send
        R = rows(); runs = [r for r in R if r["kind"] == "run"]; run1, run2 = runs[0]["run_id"], runs[1]["run_id"]
        for rid in (run1, run2): print("  run", rid, [(r["i"], r["kind"], r["call_site"][-14:], "predicted" if r["predicted"] else "live") for r in R if r["run_id"] == rid and r["kind"] != "run"])
        assert [(r["kind"], r["predicted"]) for r in R if r["run_id"] == run1 and r["kind"] != "run"] == [("effect", False), ("a2a", False), ("openai", False), ("effect", False)]
        assert [(r["kind"], r["predicted"]) for r in R if r["run_id"] == run2 and r["kind"] != "run"] == [("effect", False), ("a2a", True), ("openai", True), ("effect", False)]
        assert len({r["run_id"] for r in R[:4]}) == 1 and [r["i"] for r in R[:4]] == [1, 2, 3, 4]
        assert len(log_lines("t2c_responder.log", "RESPONDER")) == 1 and len(log_lines("t2c_agent.log", "MODEL HTTP")) == 1  # run 2 sent nothing and called no model
        print("== recertify run 2's predicted rows live (driver ctx -> responder; plain async client -> model)")
        plain = openai.AsyncOpenAI(api_key=KEY, base_url="https://openrouter.ai/api/v1", http_client=httpx.AsyncClient(timeout=60))
        rc = await dw.arecertify(TR, run2, client=plain, ctx=ctx)
        for x in rc: print("  ", {k: v for k, v in x.items() if k != "usage"})
        assert [x["kind"] for x in rc] == ["a2a", "openai"] and rc[0]["same"], rc
        print("== kill the responder (outage)")
        resp.terminate(); resp.wait(); time.sleep(0.5)
        r3 = await say("acme", "predicted: works in outage"); r4 = await say("initech", "live send fails -> fallback, model live")
        assert r3 == r1 and "unknown (fallback" in r4, (r3, r4)  # the model may clip its own reply at max_tokens; the a2a row below proves the fallback
        await asyncio.sleep(1.0); R = rows(); a2a = [r for r in R if r["kind"] == "a2a"]
        assert [(r["predicted"], r["fallback"]) for r in a2a] == [(False, False), (True, False), (True, False), (False, True)]
        print("== replay each run through the real chat handler with a fake ctx and the REAL wrapped model client (guard: any request raises ReplayLeak)")
        rp = subprocess.run([sys.executable, "t2c_replay.py"], cwd=HERE, env=env, capture_output=True, text=True); print(rp.stdout, rp.stderr[-800:])
        assert rp.returncode == 0
        pin, pout, _ = spend(TR); pin += sum(x["usage"]["prompt_tokens"] for x in rc if x.get("usage")); pout += sum(x["usage"]["completion_tokens"] for x in rc if x.get("usage"))
        usd = (pin * 0.2 + pout * 1.25) / 1e6; print(f"== tokens in={pin} out={pout} usd={usd:.6f}")
        f = HERE / "cost_s12.json"; c = json.loads(f.read_text()) if f.exists() else {}; c["t2c"] = usd; f.write_text(json.dumps(c))
        print("T2c OK", flush=True); code = 0
    except BaseException:
        import traceback; traceback.print_exc(); code = 1
    finally:
        for p in (resp, bot): p.kill()
        sys.stdout.flush(); os._exit(code)

if __name__ == "__main__": driver.run()
