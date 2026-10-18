"""S28 calls: E1 (openai/gpt-5.4, reasoning low) and cheap (openai/gpt-5.4-nano, reasoning none, logprobs top-5) on 1,000 NEW Banking77 test inputs.
Usage: calls.py pilot | run | summary.  responses.jsonl is cache + spend ledger (key = sha1 of the full request, as in S15).
Hard cap CAP USD, enforced by summing usage.cost over the ledger before every call. ponytail: one file, 8 threads."""
import sys, json, re, time, hashlib, threading, collections
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_banking77, stratified_subsample
HERE = Path(__file__).parent
RESP = HERE / "responses.jsonl"
CAP = 3.0
NNEW, SEED = 1000, 28
SYS_HEAD = "You are a bank support intent classifier. Reply with exactly one label from the list, nothing else.\n"  # identical to S5 / S15
PIN = {"provider": {"order": ["OpenAI"], "allow_fallbacks": False}, "usage": {"include": True}}
RUNS = {  # run -> (model, extra_body, max_tokens, logprobs)
    "E1": ("openai/gpt-5.4", {"reasoning": {"effort": "low"}} | PIN, 800, False),  # S5 used 2000; 800 on the retry pass (observed max 309 completion tokens) to shrink the in-flight reservation
    "nano_A": ("openai/gpt-5.4-nano", {"reasoning": {"effort": "none"}} | PIN, 60, True),
}
lock = threading.Lock()

def data():
    """names, texts (all 3080 test), gold (ints), old_idx (S5's 200 = stratified_subsample(seed 0)), new_idx (1000, seed 28, round-robin over labels so any prefix is balanced)."""
    d = load_banking77(); names = d["names"]; texts, labels = d["test"]
    old = stratified_subsample(labels, 200, seed=0)
    rng = np.random.default_rng(SEED); pool = np.setdiff1d(np.arange(len(labels)), old)
    per = {c: list(rng.permutation(pool[labels[pool] == c])) for c in range(77)}
    order = list(rng.permutation(77)); new = []
    while len(new) < NNEW:
        for c in order:
            if per[c] and len(new) < NNEW: new.append(int(per[c].pop()))
    return names, list(texts), np.asarray(labels), np.asarray(old), np.array(new)

def sysmsg(names): return SYS_HEAD + "\n".join(names)

def api_key():
    return re.search(r"^OPENROUTERKEY=(.*)$", (HERE.parents[1] / ".env").read_text(encoding="utf-8"), re.M).group(1).strip().strip("\"'")

def req_key(run, i, model, sysm, user, eb): return hashlib.sha1(json.dumps([run, i, model, sysm, user, 0, eb], sort_keys=True).encode()).hexdigest()

def ledger():
    out = {}
    if RESP.exists():
        for l in RESP.read_text(encoding="utf-8").splitlines():
            if l.strip():
                try: r = json.loads(l)
                except Exception: continue
                if r.get("err") is None: out[r["key"]] = r
    return out

def spent(): return sum(json.loads(l).get("usd", 0.0) for l in RESP.read_text(encoding="utf-8").splitlines() if l.strip()) if RESP.exists() else 0.0

def one(cl, run, i, names, text):
    model, eb, mt, lpf = RUNS[run]; sm = sysmsg(names); key = req_key(run, i, model, sm, text, eb); err = None
    kw = dict(logprobs=True, top_logprobs=5) if lpf else {}
    for a in range(8):
        t0 = time.time()
        try:
            r = cl.chat.completions.create(model=model, temperature=0, max_tokens=mt, extra_body=eb, **kw,
                                           messages=[{"role": "system", "content": sm}, {"role": "user", "content": text}])
            lp = r.choices[0].logprobs
            toks = [[t.token, t.logprob, [[x.token, x.logprob] for x in (t.top_logprobs or [])]] for t in (lp.content if lp and lp.content else [])]
            u = r.usage
            return dict(key=key, run=run, i=i, model=model, content=r.choices[0].message.content, toks=toks, finish=r.choices[0].finish_reason, ms=(time.time() - t0) * 1000,
                        pt=u.prompt_tokens, ct=u.completion_tokens, usd=float(getattr(u, "cost", None) or 0.0), has_cost=getattr(u, "cost", None) is not None,
                        served_by=getattr(r, "provider", None), err=None)
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:300]}"
            if "in_flight" in err: time.sleep(5 + 5 * a); continue  # transient: other jobs share the key
            if any(c in err for c in ("400", "401", "402", "404")): break
            time.sleep(2 ** a)
    return dict(key=key, run=run, i=i, model=model, content=None, toks=[], usd=0.0, err=err, served_by=None)

def go(n):
    from openai import OpenAI
    names, texts, labels, old, new = data(); items = [int(i) for i in new[:n]]
    cl = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key(), timeout=180, max_retries=0)
    cache = ledger(); total = [spent()]; sm = sysmsg(names)
    todo = [(r, i) for i in items for r in RUNS if req_key(r, i, RUNS[r][0], sm, texts[i], RUNS[r][1]) not in cache]
    print(f"[calls] items={len(items)} jobs={2*len(items)} cached={2*len(items)-len(todo)} to_run={len(todo)} spent_so_far=${total[0]:.4f} cap=${CAP}", flush=True)
    def work(j):
        with lock:
            if total[0] >= CAP: return None
            total[0] += 0.003  # reserve worst-case per call so 8 in-flight calls cannot overshoot; refunded below
        res = one(cl, j[0], j[1], names, texts[j[1]])
        with lock:
            total[0] += res["usd"] - 0.003
            with open(RESP, "a", encoding="utf-8") as f: f.write(json.dumps(res, ensure_ascii=False) + "\n")
        return res
    with ThreadPoolExecutor(4) as ex: res = [r for r in ex.map(work, todo) if r]
    bad = [r for r in res if r["err"]]
    print(f"[calls] done {len(res)} calls, {len(bad)} errors, skipped_by_cap={len(todo)-len(res)}, spent=${spent():.4f}", flush=True)
    for r in bad[:5]: print("   ERR", r["run"], r["i"], r["err"])

def summary():
    names, texts, labels, old, new = data(); by = collections.defaultdict(list)
    for r in ledger().values():
        if r["i"] in set(new.tolist()): by[r["run"]].append(r)
    for run, rr in by.items():
        print(f"{run:7s} n={len(rr)} $/1k={1000*sum(r['usd'] for r in rr)/len(rr):.4f} cost_field={sum(r.get('has_cost',False) for r in rr)}/{len(rr)} served_by={dict(collections.Counter(r['served_by'] for r in rr))} "
              f"mean_ms={np.mean([r['ms'] for r in rr if r.get('ms')]):.0f} no_lp={sum(1 for r in rr if RUNS[run][3] and not r['toks'])} empty={sum(1 for r in rr if not (r['content'] or '').strip())}")
    print(f"[summary] total spent=${spent():.4f}")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    if cmd == "pilot": go(20); summary()
    elif cmd == "run": go(NNEW); summary()
    elif cmd == "summary": summary()
