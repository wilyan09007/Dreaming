"""T1b: does the run id (ContextVar) reach wrapped calls made from LangGraph worker threads / tasks / user-spawned threads? No network."""
import asyncio, contextvars, threading
from operator import add
from typing import Annotated, TypedDict
from langgraph.graph import END, START, StateGraph
import dreaming_wrap as dw
from _t import HERE

TR = HERE / "t1b_traces.jsonl"; TR.unlink(missing_ok=True)
dw.Recorder("record", TR)
OUT = []  # (tag, thread, run id the wrapped call saw)

@dw.tool("probe")
def probe(tag): OUT.append((tag, threading.current_thread().name, (dw.RUN.get() or {}).get("id"))); return tag

class S(TypedDict, total=False): log: Annotated[list, add]

def node(tag): return lambda s: {"log": [probe(tag)]}
def graph(**nodes):
    g = StateGraph(S)
    for k, f in nodes.items(): g.add_node(k, f); g.add_edge(START, k); g.add_edge(k, END)
    return g.compile()

def spawn_raw(s):  # a user node starting its own thread: plain Thread does NOT copy the context
    t = threading.Thread(target=probe, args=("raw_thread",)); t.start(); t.join(); return {}
def spawn_copied(s):  # the fix
    t = threading.Thread(target=contextvars.copy_context().run, args=(probe, "copied_thread")); t.start(); t.join(); return {}
async def exec_raw(s): await asyncio.get_running_loop().run_in_executor(None, probe, "run_in_executor"); return {}
async def exec_copied(s): await asyncio.get_running_loop().run_in_executor(None, contextvars.copy_context().run, probe, "run_in_executor+copy_context"); return {}
async def to_thread(s): await asyncio.to_thread(probe, "asyncio.to_thread"); return {}
async def task(s): await asyncio.create_task(asyncio.to_thread(probe, "create_task")); return {}
async def anode(s): probe("async_node"); return {"log": ["async_node"]}

print("== sync invoke: 1 node (LangGraph runs a lone task inline) / 3 parallel nodes (thread pool)")
one = graph(a=node("one_node")); dw.run_boundary(one.invoke)({"log": []})
par = graph(a=node("par_a"), b=node("par_b"), c=node("par_c")); dw.run_boundary(par.invoke)({"log": []})
print("== sync invoke: user-spawned threads")
dw.run_boundary(graph(r=spawn_raw, c=spawn_copied).invoke)({"log": []})
print("== ainvoke: sync parallel nodes (executor), async nodes, executors, tasks")
asyncio.run(dw.run_boundary(graph(a=node("sync_in_ainvoke_a"), b=node("sync_in_ainvoke_b"), n=anode, e1=exec_raw, e2=exec_copied, t=to_thread, k=task).ainvoke)({"log": []}))
for tag, th, rid in OUT: print(f"  {tag:28} thread={th:14} run_id={rid}")
lost = sorted(t for t, _, rid in OUT if rid is None)
print("run id LOST for:", lost)
assert lost == ["raw_thread", "run_in_executor"], lost  # only user-spawned threads / bare run_in_executor lose it; LangGraph's own threads and tasks keep it
assert any(th != "MainThread" for t, th, _ in OUT if t.startswith("par_")), "parallel nodes should have used worker threads"

# ---- the OpenAI wrapper itself on LangGraph worker threads (T1's nodes ran on MainThread), against a mock transport (no network, no cost)
import httpx, json, openai
HITS = []
def _mock(req):
    HITS.append(1); body = json.loads(req.content)
    return httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 1, "model": "m", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                                     "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "re:" + body["messages"][-1]["content"]}}]})
cl = dw.wrap_openai(openai.OpenAI(api_key="k", base_url="http://mock/v1", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(_mock))))
THREADS = {}
def llm_node(tag):
    def f(s):
        with dw.call_site(tag): cl.chat.completions.create(model="m", messages=[{"role": "user", "content": tag}])
        THREADS[tag] = threading.current_thread().name; return {"log": [tag]}
    return f
def rows(kind=None): return [r for r in map(json.loads, TR.read_text(encoding="utf-8").splitlines()) if kind in (None, r["kind"])]
par = graph(l1=llm_node("llm_a"), l2=llm_node("llm_b"), l3=llm_node("llm_c"))
print("== openai wrapper under parallel LangGraph nodes (worker threads)")
dw.run_boundary(par.invoke)({"log": []}); run = rows("run")[-1]; rs = [r for r in rows("openai") if r["run_id"] == run["run_id"]]
print("  threads:", THREADS, "| run id on every row:", {r["run_id"] for r in rs} == {run["run_id"]}, "| indices:", sorted(r["i"] for r in rs))
assert len(rs) == 3 and run["run_id"] and sorted(r["i"] for r in rs) == [1, 2, 3] and any(t != "MainThread" for t in THREADS.values())

print("== run_boundary on generator entry points: the run stays open until the stream is consumed")
chain = StateGraph(S)
chain.add_node("a", llm_node("seq_a")); chain.add_node("b", llm_node("seq_b")); chain.add_edge(START, "a"); chain.add_edge("a", "b"); chain.add_edge("b", END); chain = chain.compile()
stream = dw.run_boundary(chain.stream); n_runs = len(rows("run")); seen_open = []
for chunk in stream({"log": []}): seen_open.append(len(rows("run")) == n_runs)  # no run row yet while chunks are still coming
srun = rows("run")[-1]; srs = [r for r in rows("openai") if r["run_id"] == srun["run_id"]]
print("  sync stream: run row only after the last chunk:", all(seen_open), "| calls:", srun["calls"], "| call rows under the run id:", len(srs), "| fn:", srun["fn"])
assert all(seen_open) and srun["calls"] == 2 and len(srs) == 2 and len(rows("run")) == n_runs + 1
async def astream():
    async for chunk in dw.run_boundary(par.astream)({"log": []}): pass
asyncio.run(astream()); arun = rows("run")[-1]; ars = [r for r in rows("openai") if r["run_id"] == arun["run_id"]]
print("  astream (sync nodes in executor):", "calls:", arun["calls"], "| call rows under the run id:", len(ars), "| context left clean:", dw.RUN.get() is None)
assert arun["calls"] == 3 and len(ars) == 3 and dw.RUN.get() is None
n = len(HITS); rec = dw.Recorder("replay", TR, run_id=srun["run_id"]); out = [c for c in stream({"log": []})]
print("  replay of the stream: chunks", len(out), "| extra HTTP:", len(HITS) - n, "| divergence:", rec.divergence); assert len(HITS) == n and rec.divergence is None and len(out) == 2

print("== parallel-branch replay ordering (informational: calls are numbered by START order, so a different start order on replay is reported as a divergence)")
dw.Recorder("record", TR); dw.run_boundary(par.invoke)({"log": []}); prun = rows("run")[-1]["run_id"]; bad = 0
for _ in range(40):
    rec = dw.Recorder("replay", TR, run_id=prun)
    try: dw.run_boundary(par.invoke)({"log": []})
    except dw.Divergence: bad += 1
print(f"  {bad} divergences in 40 replays of a 3-way fan-out (mock transport, instant calls)")
import random, time
def slow_node(tag):  # same fan-out, but each branch takes a random 0-15 ms before it calls: start order now varies run to run
    f = llm_node(tag); return lambda s: (time.sleep(random.random() * 0.015), f(s))[1]
jit = graph(l1=slow_node("llm_a"), l2=slow_node("llm_b"), l3=slow_node("llm_c")); dw.Recorder("record", TR); dw.run_boundary(jit.invoke)({"log": []}); jrun = rows("run")[-1]["run_id"]; bad = 0
for _ in range(40):
    dw.Recorder("replay", TR, run_id=jrun)
    try: dw.run_boundary(jit.invoke)({"log": []})
    except dw.Divergence: bad += 1
print(f"  {bad} spurious divergences in 40 replays when branch start order varies (strict start-order replay: parallel branches need a deterministic order)")
print("T1b OK")
