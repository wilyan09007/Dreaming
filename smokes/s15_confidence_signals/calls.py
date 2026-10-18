"""S15 part 2, cheap-model calls with logprobs / self-consistency / verbalized confidence on the S5 inputs.
Usage: calls.py pilot | run | summary.   Every raw response is cached in responses.jsonl keyed by a hash of the request, so reruns are free.
ponytail: one helper file; thread pool of 8; provider pinned and the serving provider is logged per call."""
import sys, os, json, re, time, hashlib, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).parent
RESP = HERE / "responses.jsonl"
CAP = 4.0          # stop issuing calls past this (smoke cap is $5)
N = 200
BASE_SYS_HEAD = "You are a bank support intent classifier. Reply with exactly one label from the list, nothing else.\n"
VERB_TAIL = "\nAfter the label, add ' | ' and then your confidence from 0 to 100 that the label is correct, for example: card_arrival | 85. Output nothing else."
OFF = {"reasoning": {"effort": "none"}}
MODELS = {  # key: (model, pinned provider, extra_body, suffix on the user turn)
    "nano": ("openai/gpt-5.4-nano", "OpenAI", OFF, ""),
    "g41": ("openai/gpt-4.1-nano", "OpenAI", {}, ""),
    "qnx": ("qwen/qwen3-next-80b-a3b-instruct", "Parasail", {}, ""),
}
RUNS = {  # run -> (model key, temperature, verbalized)
    "nano_A": ("nano", 0, False), "nano_Ab": ("nano", 0, False), "nano_S": ("nano", 0.7, False), "nano_V": ("nano", 0, True), "nano_Vb": ("nano", 0, True),
    "g41_A": ("g41", 0, False), "g41_Ab": ("g41", 0, False), "g41_S": ("g41", 0.7, False),
    "qnx_A": ("qnx", 0, False), "qnx_Ab": ("qnx", 0, False), "qnx_S": ("qnx", 0.7, False),
}
lock = threading.Lock()

def b_data():  # identical to s5_tier_router/run.py b_data(): same 200 inputs
    d = load_banking77(); names = d["names"]; texts, labels = d["test"]
    idx = stratified_subsample(labels, N, seed=0)
    return names, [texts[i] for i in idx], labels[idx]

def sys_msg(names, verbal):
    return BASE_SYS_HEAD + "\n".join(names) + (VERB_TAIL if verbal else "")

def client_():
    from openai import OpenAI
    key = re.search(r"^OPENROUTERKEY=(.*)$", (HERE.parents[1] / ".env").read_text(encoding="utf-8"), re.M).group(1).strip()
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, timeout=180, max_retries=0)

def req_key(run, i, model, sysm, user, temp, eb):
    return hashlib.sha1(json.dumps([run, i, model, sysm, user, temp, eb], sort_keys=True).encode()).hexdigest()

def load_cache():
    out = {}
    if RESP.exists():
        for l in RESP.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("err") is None: out[r["key"]] = r
    return out

def spent():
    return sum(json.loads(l).get("usd", 0.0) for l in RESP.read_text(encoding="utf-8").splitlines() if l.strip()) if RESP.exists() else 0.0

def one(cl, run, i, names, text):
    mk, temp, verbal = RUNS[run]; model, prov, eb0, suf = MODELS[mk]
    eb = dict(eb0) | {"provider": {"order": [prov], "allow_fallbacks": False}}
    sysm = sys_msg(names, verbal); user = text + suf
    key = req_key(run, i, model, sysm, user, temp, eb)
    err = None
    for attempt in range(4):
        t0 = time.time()
        try:
            r = cl.chat.completions.create(model=model, temperature=temp, max_tokens=60, logprobs=True, top_logprobs=5, extra_body=eb,
                                           messages=[{"role": "system", "content": sysm}, {"role": "user", "content": user}])
            lp = r.choices[0].logprobs
            toks = [[t.token, t.logprob, [[a.token, a.logprob] for a in (t.top_logprobs or [])]] for t in (lp.content if lp and lp.content else [])]
            u = r.usage
            return dict(key=key, run=run, i=i, model=model, content=r.choices[0].message.content, toks=toks, finish=r.choices[0].finish_reason,
                        ms=(time.time() - t0) * 1000, pt=u.prompt_tokens, ct=u.completion_tokens, usd=float(getattr(u, "cost", None) or 0.0),
                        served_by=getattr(r, "provider", None), err=None)
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:300]}"
            if any(c in err for c in ("400", "401", "402", "404")): break
            time.sleep(2 ** attempt)
    return dict(key=key, run=run, i=i, model=model, content=None, toks=[], usd=0.0, err=err, served_by=None)

def go(runs, items):
    names, texts, y = b_data(); cl = client_(); cache = load_cache(); total = [spent()]
    jobs = [(r, i) for r in runs for i in items]
    todo = []
    for r, i in jobs:
        mk, temp, verbal = RUNS[r]; model, prov, eb0, suf = MODELS[mk]
        eb = dict(eb0) | {"provider": {"order": [prov], "allow_fallbacks": False}}
        if req_key(r, i, model, sys_msg(names, verbal), texts[i] + suf, temp, eb) not in cache: todo.append((r, i))
    print(f"[calls] jobs={len(jobs)} cached={len(jobs)-len(todo)} to_run={len(todo)} spent_so_far=${total[0]:.4f}", flush=True)
    def work(j):
        if total[0] >= CAP: return None
        res = one(cl, j[0], j[1], names, texts[j[1]])
        with lock:
            total[0] += res["usd"]
            with open(RESP, "a", encoding="utf-8") as f: f.write(json.dumps(res, ensure_ascii=False) + "\n")
        return res
    with ThreadPoolExecutor(8) as ex: res = [r for r in ex.map(work, todo) if r]
    bad = [r for r in res if r["err"]]
    print(f"[calls] done {len(res)} calls, {len(bad)} errors, spent=${total[0]:.4f}", flush=True)
    for r in bad[:5]: print("   ERR", r["run"], r["i"], r["err"])
    return res

def summary():
    c = load_cache(); import collections
    by = collections.defaultdict(list)
    for r in c.values(): by[r["run"]].append(r)
    for run in RUNS:
        rr = by.get(run, [])
        if rr: print(f"{run:8s} n={len(rr):3d} $/1k={1000*sum(r['usd'] for r in rr)/len(rr):.4f} served_by={dict(collections.Counter(r['served_by'] for r in rr))} "
                     f"mean_ms={sum(r['ms'] for r in rr)/len(rr):.0f} ntok_mean={sum(len(r['toks']) for r in rr)/len(rr):.1f} no_lp={sum(1 for r in rr if not r['toks'])}")
    print(f"[summary] total spent=${spent():.4f}")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    if cmd == "pilot": go(list(RUNS), range(5)); summary()
    elif cmd == "run": go(list(RUNS), range(N)); summary()
    elif cmd == "summary": summary()
