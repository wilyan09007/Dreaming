"""T1: LangGraph 2-node graph (categorize -> draft_note) under the wrappers: one run id, predictor, replay, divergence, recertify, async."""
import asyncio, json, threading
from typing import TypedDict
from langgraph.graph import END, START, StateGraph
import dreaming_wrap as dw
from _t import GEN, HERE, NET, client, spend

TR = HERE / "t1_traces.jsonl"; TR.unlink(missing_ok=True)
SYS_A = "Classify the transaction as one of: travel, software, meals, other. Reply with one word."
SYS_B = "Write a one-sentence bookkeeping note for the transaction. Mention the category."
TEXT = "Delta Air Lines $412.80 flight SFO-JFK"

class S(TypedDict, total=False): text: str; category: str; note: str

def rows(run=None): return [r for r in map(json.loads, TR.read_text(encoding="utf-8").splitlines()) if run is None or r["run_id"] == run]
def show(run):
    for r in rows(run):
        print(f"  run={r['run_id']} i={r.get('i')} {r['kind']:6} site={r.get('call_site', r.get('fn'))} predicted={r.get('predicted')} "
              f"sha={str(r.get('req_sha'))[:8]} {r.get('latency_ms')}ms text={r.get('text')!r}" if r["kind"] != "run" else f"  run={r['run_id']} RUN-ROW fn={r['fn']} calls={r['calls']} error={r['error']}")

def msgs(sys, user): return [{"role": "system", "content": sys}, {"role": "user", "content": user}]

def build(cl, sys_b=SYS_B, seen=None):
    def categorize(s):
        seen is not None and seen.append(("categorize", threading.current_thread().name, dw.RUN.get() and dw.RUN.get()["id"]))
        r = cl.chat.completions.create(**GEN, messages=msgs(SYS_A, s["text"]), extra_headers={"X-Dreaming-Call-Site": "categorize"})  # tag by header
        return {"category": r.choices[0].message.content.strip().lower()}
    def draft_note(s):
        seen is not None and seen.append(("draft_note", threading.current_thread().name, dw.RUN.get() and dw.RUN.get()["id"]))
        with dw.call_site("draft_note"):  # tag by context manager
            r = cl.chat.completions.create(**GEN, messages=msgs(sys_b, f"Transaction: {s['text']}\nCategory: {s['category']}"))
        return {"note": r.choices[0].message.content.strip()}
    g = StateGraph(S); g.add_node("categorize", categorize); g.add_node("draft_note", draft_note)
    g.add_edge(START, "categorize"); g.add_edge("categorize", "draft_note"); g.add_edge("draft_note", END)
    return g.compile()

def build_async(cl, sys_b=SYS_B):
    async def categorize(s):
        r = await cl.chat.completions.create(**GEN, messages=msgs(SYS_A, s["text"]), extra_headers={"X-Dreaming-Call-Site": "categorize"})
        return {"category": r.choices[0].message.content.strip().lower()}
    async def draft_note(s):
        with dw.call_site("draft_note"):
            r = await cl.chat.completions.create(**GEN, messages=msgs(sys_b, f"Transaction: {s['text']}\nCategory: {s['category']}"))
        return {"note": r.choices[0].message.content.strip()}
    g = StateGraph(S); g.add_node("categorize", categorize); g.add_node("draft_note", draft_note)
    g.add_edge(START, "categorize"); g.add_edge("categorize", "draft_note"); g.add_edge("draft_note", END)
    return g.compile()

RC = []  # usage of recertify's live re-executions (they write no rows): counted into the cost
def last_run(): return [r for r in rows() if r["kind"] == "run"][-1]["run_id"]

# ---- 1. record: both calls under ONE run id with the right call sites
print("== 1. record (live)")
dw.Recorder("record", TR)
cl, seen = dw.wrap_openai(client()), []
invoke = dw.run_boundary(build(cl, seen=seen).invoke)  # <- the one-line LangGraph change (plus wrap_openai on the client)
n0 = NET["n"]; out1 = invoke({"text": TEXT}); run1 = last_run()
show(run1); print("  out:", out1); print("  node threads/run ids seen:", seen, "| main thread:", threading.current_thread().name)
r1 = [r for r in rows(run1) if r["kind"] == "openai"]
assert [r["call_site"] for r in r1] == ["categorize", "draft_note"] and len({r["run_id"] for r in r1}) == 1 and run1 is not None
assert NET["n"] - n0 == 2 and all(s[2] == run1 for s in seen), "run id must reach both nodes (ContextVar survived LangGraph's executor)"
assert isinstance(r1[0]["request"]["messages"], list) and r1[0]["request"]["messages"][0]["role"] == "system"  # structured, not flattened
assert r1[0]["req_sha"] == dw.sha(r1[0]["request"]["messages"]) and r1[0]["usage"]["prompt_tokens"] > 0

# ---- 2. predictor: copy node A's answer in -> A predicted (no network), B live
print("== 2. predictor for categorize")
dw.Recorder("record", TR, predictors={"categorize": {r1[0]["req_sha"]: r1[0]["text"]}})
n0 = NET["n"]; out2 = invoke({"text": TEXT}); run2 = last_run()
show(run2); print("  out:", out2, "| network calls this run:", NET["n"] - n0)
r2 = [r for r in rows(run2) if r["kind"] == "openai"]
assert [r["predicted"] for r in r2] == [True, False] and NET["n"] - n0 == 1 and out2["category"] == out1["category"]
assert r2[0]["usage"]["total_tokens"] == 0 and r2[0]["text"] == r1[0]["text"]

# ---- 3. replay: zero network, identical outputs (run1 = all live, run2 = includes a predicted row)
print("== 3. replay")
for run, out in ((run1, out1), (run2, out2)):
    rec = dw.Recorder("replay", TR, run_id=run); n0 = NET["n"]; got = invoke({"text": TEXT})
    print(f"  replay {run}: network calls={NET['n'] - n0} identical={got == out} divergence={rec.divergence}")
    assert NET["n"] == n0 and got == out and rec.divergence is None

# ---- 4. divergence: change node B's prompt, replay run1
print("== 4. divergence (node B prompt changed)")
rec = dw.Recorder("replay", TR, run_id=run1); n0 = NET["n"]
try: dw.run_boundary(build(cl, sys_b=SYS_B + " Use passive voice.").invoke)({"text": TEXT}); raise SystemExit("expected Divergence")
except dw.Divergence as e: print("  raised:", str(e)[:160]); print("  report:", json.dumps(rec.divergence))
assert rec.divergence["row"] == 2 and rec.divergence["expected"][0] == rec.divergence["actual"][0] == "draft_note" and NET["n"] == n0
assert rec.divergence["expected"][1] != rec.divergence["actual"][1]
# a handler that stops early is also a divergence
rec = dw.Recorder("replay", TR, run_id=run1)
def only_a(): return cl.chat.completions.create(**GEN, messages=msgs(SYS_A, TEXT), extra_headers={"X-Dreaming-Call-Site": "categorize"})
try: dw.run_boundary(only_a)(); raise SystemExit("expected Divergence")
except dw.Divergence as e: print("  early stop:", str(e)[:120]); assert rec.divergence["row"] == 2 and rec.divergence["actual"] is None

# ---- 5. recertify: re-execute the predicted call live on its logged prompt
print("== 5. recertify run2's predicted row")
dw.Recorder("record", TR); n0 = NET["n"]
rc = dw.recertify(TR, run2, client=cl); print("  ", [{k: v for k, v in x.items() if k != "usage"} for x in rc], "| network calls:", NET["n"] - n0)
assert rc and all(x["same"] for x in rc) and NET["n"] - n0 == 1; RC.extend(x["usage"] for x in rc)

# ---- 6. async client + ainvoke
print("== 6. AsyncOpenAI + ainvoke")
async def amain():
    dw.Recorder("record", TR)
    acl = dw.wrap_openai(client(async_=True)); print("  create is coroutine fn:", asyncio.iscoroutinefunction(acl.chat.completions.create))
    ainvoke = dw.run_boundary(build_async(acl).ainvoke); n0 = NET["n"]
    a1 = await ainvoke({"text": TEXT}); ra1 = last_run(); show(ra1)
    ra = [r for r in rows(ra1) if r["kind"] == "openai"]
    assert [r["call_site"] for r in ra] == ["categorize", "draft_note"] and len({r["run_id"] for r in ra}) == 1 and NET["n"] - n0 == 2
    dw.Recorder("record", TR, predictors={"categorize": {ra[0]["req_sha"]: ra[0]["text"]}}); n0 = NET["n"]
    a2 = await ainvoke({"text": TEXT}); ra2 = last_run(); show(ra2)
    assert [r["predicted"] for r in rows(ra2) if r["kind"] == "openai"] == [True, False] and NET["n"] - n0 == 1
    dw.Recorder("replay", TR, run_id=ra1); n0 = NET["n"]; a3 = await ainvoke({"text": TEXT})
    print(f"  async replay: network calls={NET['n'] - n0} identical={a3 == a1}"); assert NET["n"] == n0 and a3 == a1
    # async divergence (review: the async variant had only record/predictor/replay checks): node B's prompt changes
    rec = dw.Recorder("replay", TR, run_id=ra1); n0 = NET["n"]
    try: await dw.run_boundary(build_async(acl, sys_b=SYS_B + " Use passive voice.").ainvoke)({"text": TEXT}); raise SystemExit("expected Divergence")
    except dw.Divergence as e: print("  async divergence:", str(e)[:70])
    assert rec.divergence["row"] == 2 and rec.divergence["expected"][0] == rec.divergence["actual"][0] == "draft_note" and NET["n"] == n0
    # async recertify: the predicted row of the async run, re-executed live through the AsyncOpenAI client
    dw.Recorder("record", TR); n0 = NET["n"]
    rca = await dw.arecertify(TR, ra2, client=acl); print("  async recertify:", [{k: v for k, v in x.items() if k != "usage"} for x in rca], "| network calls:", NET["n"] - n0)
    assert rca and all(x["same"] for x in rca) and NET["n"] - n0 == 1; RC.extend(x["usage"] for x in rca)
    # sync graph, async entry (LangGraph runs sync nodes in an executor under ainvoke)
    dw.Recorder("record", TR); sc, seen2 = dw.wrap_openai(client()), []
    s1 = await dw.run_boundary(build(sc, seen=seen2).ainvoke)({"text": TEXT}); rs = last_run(); print("  sync nodes under ainvoke:", seen2)
    assert all(s[2] == rs for s in seen2) and len([r for r in rows(rs) if r["kind"] == "openai"]) == 2
asyncio.run(amain())

pin, pout, _ = spend(TR); pin += sum(u["prompt_tokens"] for u in RC); pout += sum(u["completion_tokens"] for u in RC); usd = (pin * 0.2 + pout * 1.25) / 1e6
print(f"== tokens in={pin} out={pout} usd={usd:.6f} (live rows + recertify re-executions)")
f = HERE / "cost_s12.json"; c = json.loads(f.read_text()) if f.exists() else {}; c["t1"] = usd; f.write_text(json.dumps(c))
print("T1 OK")
