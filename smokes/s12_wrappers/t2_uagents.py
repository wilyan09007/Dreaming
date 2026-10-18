"""T2: two real uAgents processes on localhost (responder :8101, requester :8102) driven by a third agent (driver :8103, this process).
Shows: a2a call logged under the requester's run id; learned predictor skips the second send; responder outage + registered fallback keeps the handler answering."""
import json, os, socket, subprocess, sys, time
from t2_common import *
import uagents
import dreaming_wrap as dw
HERE = __import__("pathlib").Path(__file__).parent
TR = HERE / "t2_traces.jsonl"; TR.unlink(missing_ok=True)
env = dict(os.environ, PYTHONUTF8="1", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
def spawn(script, log): return subprocess.Popen([sys.executable, script], cwd=HERE, stdout=open(HERE / log, "w"), stderr=subprocess.STDOUT, env=env)
def up(port):
    for _ in range(100):
        try: socket.create_connection(("127.0.0.1", port), 0.2).close(); return
        except OSError: time.sleep(0.2)
    raise SystemExit(f"port {port} never came up")

print("uagents", uagents.__version__ if hasattr(uagents, "__version__") else __import__("importlib.metadata").metadata.version("uagents"))
print("ctx.send_and_receive:", __import__("inspect").signature(uagents.Context.send_and_receive))
resp, req = spawn("t2_responder.py", "t2_responder.log"), spawn("t2_requester.py", "t2_requester.log")
up(PORTS["responder"]); up(PORTS["requester"])
driver = make("driver")

def rows(): return [json.loads(l) for l in TR.read_text(encoding="utf-8").splitlines()]
def responder_hits(): return [l.strip() for l in (HERE / "t2_responder.log").read_text(encoding="utf-8").splitlines() if l.startswith("RESPONDER")]

@driver.on_event("startup")
async def scenario(ctx):
    try:
        out = []
        async def ask(text, label):
            t0 = time.time(); r, st = await ctx.send_and_receive(ADDR["requester"], Ask(text=text), response_type=Answer, timeout=20)
            print(f"  ask({text!r:10}) -> {r and r.text!r}  [{label}] {time.time() - t0:.2f}s | responder hits so far: {len(responder_hits())}", flush=True)
            out.append(r and r.text); return r
        print("== live, then the same request again (learned predictor), then a different vendor")
        await ask("globex", "live send"); await ask("globex", "predicted: send skipped"); await ask("acme", "live send")
        print("== recertify the predicted row (run 2) against the live responder, sent from the driver's ctx")
        run2 = [r for r in rows() if r["kind"] == "run"][1]["run_id"]; rc = await dw.arecertify(TR, run2, ctx=ctx); print("  ", rc)
        assert len(rc) == 1 and rc[0]["kind"] == "a2a" and rc[0]["same"], rc
        print("== kill the responder (outage)")
        resp.terminate(); resp.wait(); time.sleep(0.5)
        await ask("acme", "predicted: works in outage"); await ask("initech", "live send fails -> registered fallback")
        time.sleep(1.0)
        R = rows(); RESP = f"{ADDR['responder']}|VendorLookup"
        a2a = [r for r in R if r["kind"] == "a2a"]; runs = [r for r in R if r["kind"] == "run"]
        print("== trace rows (requester process)")
        for r in R:
            print(f"  run={r['run_id']} i={r.get('i')} {r['kind']:6} site={str(r.get('call_site', r.get('fn')))[-34:]} predicted={r.get('predicted')} fallback={r.get('fallback')} error={str(r.get('error'))[:60]} {r.get('latency_ms')}ms resp={r.get('response') if r['kind'] != 'run' else ''}")
        assert out == ["globex: net60", "globex: net60", "acme: net30", "acme: net30", "initech: unknown (fallback)"], out
        assert [r["call_site"] for r in a2a] == [RESP] * 5 and len({r["run_id"] for r in a2a}) == 5 and {r["run_id"] for r in a2a} == {r["run_id"] for r in runs}
        assert [r["predicted"] for r in a2a] == [False, True, False, True, False] and [r["fallback"] for r in a2a] == [False] * 4 + [True]
        assert all(r["i"] == 1 for r in a2a) and [r["i"] for r in R if r["kind"] == "effect"] == [2] * 5  # a2a is call 1, the reply ctx.send is effect 2
        assert responder_hits() == ["RESPONDER got VendorLookup(globex)", "RESPONDER got VendorLookup(acme)", "RESPONDER got VendorLookup(globex)"], responder_hits()  # skipped + outage sends never arrived; the 3rd is the recertify re-execution
        print("== replay each run with a fake ctx in a fresh process (no sockets, no agents)")
        rp = subprocess.run([sys.executable, "t2_replay.py"], cwd=HERE, env=env, capture_output=True, text=True); print(rp.stdout, rp.stderr[-600:])
        assert rp.returncode == 0
        print("T2 OK", flush=True); code = 0
    except BaseException as e:
        import traceback; traceback.print_exc(); code = 1
    finally:
        for p in (resp, req): p.kill()
        sys.stdout.flush(); os._exit(code)

if __name__ == "__main__": driver.run()
