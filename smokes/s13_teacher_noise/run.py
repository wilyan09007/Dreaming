"""S13: teacher noise floor. Same 100 Banking77 prompts x R repeats at temperature 0, per model/condition.
Usage: run.py probe | run.py full [reps] | run.py report.   ponytail: one file, resumable via responses.jsonl."""
import sys, re, json, time, difflib, threading, collections
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from openai import OpenAI

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "smokes")); from common import load_banking77, stratified_subsample

KEY = re.search(r"OPENROUTERKEY\s*=\s*(\S+)", (ROOT / ".env").read_text()).group(1).strip("\"'")
client = OpenAI(api_key=KEY, base_url="https://openrouter.ai/api/v1", max_retries=0, timeout=90)
RESP, LOG = HERE / "responses.jsonl", HERE / "log.txt"
MODELS = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5", "openai/gpt-5.4-mini",
          "google/gemini-3.5-flash-lite", "qwen/qwen3-32b"]
PINNED = ["anthropic/claude-haiku-5.5", "openai/gpt-5.4-mini"]  # condition B only for these
# haiku-5.5: default adaptive thinking fires on some prompts and eats the 20-token budget (content None) -> enabled:false; sonnet-5.5: reasoning mandatory (400 if disabled), default kept
# settings found by manual probing (see log.txt): gpt "minimal" still bills 10 reasoning tokens -> "none"; gemini reasoning
# is mandatory (cannot be disabled; adaptive, measured 0 on this task); qwen on DeepInfra ignores the reasoning param -> "/no_think" soft switch
CFG = {"anthropic/claude-haiku-5.5": dict(reasoning={"enabled": False}), "anthropic/claude-sonnet-5.5": dict(reasoning=None),
       "openai/gpt-5.4-mini": dict(reasoning={"effort": "none"}), "google/gemini-3.5-flash-lite": dict(reasoning=None),
       "qwen/qwen3-32b": dict(reasoning=None, suffix="\n/no_think")}
lock = threading.Lock()

def say(s=""):
    print(s, flush=True)
    with lock, open(LOG, "a", encoding="utf-8") as f: f.write(s + "\n")

b = load_banking77(); NAMES = b["names"]; texts, labels = b["test"]
IDX = stratified_subsample(labels, 100, seed=0)
QS = [texts[i] for i in IDX]; GOLD = [NAMES[labels[i]] for i in IDX]
SYSTEM = ("You are a bank support intent classifier. Reply with exactly one label from the list, nothing else.\n"
          + "\n".join(NAMES))

LOW = {n.lower(): n for n in NAMES}

def parse(raw):
    s = (raw or "").strip().strip("`'\".").strip().lower()
    if s in LOW: return LOW[s], False   # exact after strip/lower (Banking77 has one capitalised label)
    m = difflib.get_close_matches(s, list(LOW), n=1, cutoff=0.0)
    return (LOW[m[0]] if m else "__none__"), True

def call(model, q, reasoning, provider=None, seed=None, temperature=True, max_tokens=20):
    kw = dict(model=model, messages=[{"role": "system", "content": SYSTEM + CFG[model].get("suffix", "")}, {"role": "user", "content": q}], max_tokens=max_tokens)
    if temperature: kw["temperature"] = 0
    if seed is not None: kw["seed"] = seed
    eb = {}
    if reasoning is not None: eb["reasoning"] = reasoning
    if provider: eb["provider"] = {"order": [provider], "allow_fallbacks": False}
    if eb: kw["extra_body"] = eb
    err = None
    for k in range(5):
        t = time.perf_counter()
        try:
            r = client.chat.completions.create(**kw)
            ms = (time.perf_counter() - t) * 1000
            d = r.model_dump(); d["_ms"] = ms; d["_attempts"] = k + 1
            return d
        except Exception as e:
            err = e
            code = getattr(e, "status_code", None)
            if code and 400 <= code < 500 and code not in (408, 429): break
            time.sleep(2 ** k)
    return {"_error": f"{type(err).__name__}: {str(err)[:300]}", "_attempts": k + 1}

def content(d):
    try: return d["choices"][0]["message"]["content"]
    except Exception: return None
def rtok(d): return ((d.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
def cost(d): return float((d.get("usage") or {}).get("cost") or 0)

def add_cost(usd, note):
    with open(ROOT / "smokes" / "COSTS.md", "a", encoding="utf-8") as f: f.write(f"s13_teacher_noise,{usd:.4f},{note}\n")

def probe():  # 5 calls/model to project the cost of the full run
    est = {}
    for m in MODELS:
        c = CFG[m]; ds = [call(m, QS[i], c["reasoning"]) for i in range(5)]
        with open(HERE / "probe_responses.jsonl", "a", encoding="utf-8") as f:
            for d in ds: f.write(json.dumps(dict(model=m, resp=d)) + "\n")
        ok = [d for d in ds if "_error" not in d]; est[m] = float(np.mean([cost(d) for d in ok]))
        say(f"estimate {m}: ${est[m]:.6f}/call; ok={len(ok)}/5; providers={[d.get('provider') for d in ok]}; rtok={[rtok(d) for d in ok]}; "
            f"content={[content(d) for d in ok]}")
    proj = sum(est[m] * 500 for m in MODELS) + sum(est[m] * 500 for m in PINNED)
    say(f"projected full run (5 reps, A for all, B for pinned) = ${proj:.2f}")
    add_cost(sum(est[m] * 5 for m in MODELS), "probe: 5 calls/model")

def done_keys():
    s = set()
    if RESP.exists():
        for l in RESP.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            if "_error" not in r["resp"]: s.add((r["cond"], r["model"], r["rep"], r["i"]))
    return s

def full(reps):
    cfg = CFG; have = done_keys(); spent = [0.0]
    jobs = []
    for cond in "AB":
        for m in (MODELS if cond == "A" else PINNED):
            prov = None
            if cond == "B":  # provider the first A call reported
                for l in RESP.read_text(encoding="utf-8").splitlines():
                    r = json.loads(l)
                    if r["cond"] == "A" and r["model"] == m and r["resp"].get("provider"): prov = r["resp"]["provider"]; break
                say(f"condition B pinned provider for {m}: {prov}")
            for rep in range(reps):
                for i in range(100):
                    if (cond, m, rep, i) not in have: jobs.append((cond, m, rep, i, prov))
        if cond == "A": run_jobs(jobs, cfg, spent); jobs = []   # A must finish so B can read its provider
    run_jobs(jobs, cfg, spent)
    add_cost(spent[0], f"full run, {reps} reps; sum of usage.cost this invocation")
    say(f"spent this invocation: ${spent[0]:.4f}")

# extra conditions: A2 = sonnet with max_tokens 300 (its thinking cannot be disabled and ate the 20-token budget);
# C:<endpoint tag> = same prompts pinned to every other endpoint of the model, 2 reps (cross-provider disagreement)
CTAGS = {"anthropic/claude-haiku-5.5": ["anthropic", "amazon-bedrock", "google-vertex/global", "azure/global"],
         "openai/gpt-5.4-mini": ["azure"], "google/gemini-3.5-flash-lite": ["google-ai-studio"],
         "qwen/qwen3-32b": ["tenstorrent/fp8", "siliconflow/fp8"]}
def extra():
    have = done_keys(); spent = [0.0]; jobs = [("A2", "anthropic/claude-sonnet-5.5", r, i, None) for r in range(5) for i in range(100)]
    jobs += [(f"C:{t}", m, r, i, None) for m, ts in CTAGS.items() for t in ts for r in range(2) for i in range(100)]
    run_jobs([j for j in jobs if j[:4] not in have], CFG, spent)
    add_cost(spent[0], "extra: sonnet max_tokens=300 rerun + cross-provider pins (C)"); say(f"spent this invocation: ${spent[0]:.4f}")

def run_jobs(jobs, cfg, spent):
    if not jobs: return
    say(f"running {len(jobs)} calls")
    def one(j):
        cond, m, rep, i, prov = j; c = cfg[m]
        pin = prov if cond == "B" else cond[2:] if cond.startswith("C:") else None   # B: first provider seen in A; C:<tag>: other endpoints
        d = call(m, QS[i], c["reasoning"], provider=pin, seed=0 if cond == "B" else None, max_tokens=300 if cond == "A2" else 20)
        with lock:
            spent[0] += cost(d)
            with open(RESP, "a", encoding="utf-8") as f: f.write(json.dumps(dict(cond=cond, model=m, rep=rep, i=i, resp=d)) + "\n")
    # passes are ordered rep-major so repeats of a prompt are separated in time
    jobs.sort(key=lambda j: (j[0], j[1], j[2], j[3]))
    with ThreadPoolExecutor(8) as ex: list(ex.map(one, jobs))

def load():
    R = collections.defaultdict(dict)  # (cond, model) -> {(rep,i): resp}; last successful wins
    for l in RESP.read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        if "_error" not in r["resp"]: R[(r["cond"], r["model"])][(r["rep"], r["i"])] = r["resp"]
    return R

def pct(x, p): return float(np.percentile(x, p))

def stats(cond, m, rs):
    reps = max(r for r, _ in rs) + 1
    ans, fails, ms, pt, ct, rt, usd, provs = {}, 0, [], [], [], [], [], collections.Counter()
    pv, empty, trunc, off, bad = {}, 0, 0, 0, set()
    for (rep, i), d in rs.items():
        a, f = parse(content(d)); ans[(rep, i)] = a; fails += f
        if f:
            if not (content(d) or "").strip(): empty += 1
            elif d["choices"][0].get("finish_reason") == "length": trunc += 1
            else: off += 1
            if not (content(d) or "").strip(): bad.add((rep, i))
        ms.append(d["_ms"]); u = d.get("usage") or {}; pt.append(u.get("prompt_tokens") or 0); ct.append(u.get("completion_tokens") or 0)
        rt.append(rtok(d)); usd.append(cost(d)); provs[d.get("provider")] += 1; pv[(rep, i)] = d.get("provider")
    full_i = [i for i in range(100) if all((r, i) in ans for r in range(reps))]
    A = [[ans[(r, i)] for r in range(reps)] for i in full_i]; G = [GOLD[i] for i in full_i]
    flip = np.mean([len(set(a)) > 1 for a in A])
    pairs = [(x, y) for x in range(reps) for y in range(x + 1, reps)]
    pair = np.mean([a[x] != a[y] for a in A for x, y in pairs])
    maj = [collections.Counter(a).most_common(1)[0][0] for a in A]
    maj_acc = np.mean([x == g for x, g in zip(maj, G)])
    run_acc = [np.mean([a[r] == g for a, g in zip(A, G)]) for r in range(reps)]
    floor = 1 - np.mean([collections.Counter(a).most_common(1)[0][1] / reps for a in A])  # best fixed student vs one teacher call
    # provider coincidence
    PV = [[pv[(r, i)] for r in range(reps)] for i in full_i]
    chg = [len(set(p)) > 1 for p in PV]; fl = [len(set(a)) > 1 for a in A]
    flip_chg = np.mean([f for f, c in zip(fl, chg) if c]) if any(chg) else None
    flip_same = np.mean([f for f, c in zip(fl, chg) if not c]) if not all(chg) else None
    xs = [(a[x] != a[y], PV[k][x] != PV[k][y]) for k, a in enumerate(A) for x, y in pairs]
    dis_diffprov = np.mean([d for d, p in xs if p]) if any(p for _, p in xs) else None
    dis_sameprov = np.mean([d for d, p in xs if not p]) if any(not p for _, p in xs) else None
    n = len(ms)
    pair_ok = np.mean([a[x] != a[y] for k, a in enumerate(A) for x, y in pairs if (x, full_i[k]) not in bad and (y, full_i[k]) not in bad])
    return dict(cond=cond, model=m, reps=reps, n_calls=n, n_prompts=len(full_i), parse_fail=fails, empty=empty, trunc=trunc, off=off, pair_ok=pair_ok, flip=flip, pair=pair, floor=floor,
                single_acc=float(np.mean(run_acc)), single_min=min(run_acc), single_max=max(run_acc), maj_acc=maj_acc,
                p50=pct(ms, 50), p95=pct(ms, 95), ptok=np.mean(pt), ctok=np.mean(ct), rtok=np.mean(rt), rtok_max=max(rt),
                usd_call=np.mean(usd), usd_1k=np.mean(usd) * 1000, provs=dict(provs), flip_chg=flip_chg, flip_same=flip_same,
                n_chg=int(sum(chg)), dis_diffprov=dis_diffprov, dis_sameprov=dis_sameprov, wrong_all=int(sum(all(x != g for x in a) for a, g in zip(A, G))))

def report(reps):
    R = load(); S = [stats(c, m, rs) for (c, m), rs in sorted(R.items())]
    (HERE / "stats.json").write_text(json.dumps(S, indent=1, default=str))
    X = []  # cross-provider: answers of the default-routing run (A, 5 reps) vs each pinned endpoint (C, 2 reps), same prompts
    for (c, m), rs in sorted(R.items()):
        if not c.startswith("C:"): continue
        A_ = R[("A", m)]; reps_c = max(r for r, _ in rs) + 1
        cross = np.mean([parse(content(A_[(ra, i)]))[0] != parse(content(rs[(rc, i)]))[0] for i in range(100) for ra in range(5) for rc in range(reps_c)])
        within = np.mean([parse(content(rs[(0, i)]))[0] != parse(content(rs[(1, i)]))[0] for i in range(100)])
        acc = np.mean([parse(content(rs[(r, i)]))[0] == GOLD[i] for i in range(100) for r in range(reps_c)])
        prov = collections.Counter(d.get("provider") for d in rs.values()); ms = [d["_ms"] for d in rs.values()]
        X.append(dict(model=m, tag=c[2:], provider=dict(prov), cross=cross, within_pin=within, acc=acc, p50=pct(ms, 50), p95=pct(ms, 95), usd_call=np.mean([cost(d) for d in rs.values()])))
    (HERE / "cross.json").write_text(json.dumps(X, indent=1, default=lambda o: float(o)))
    for s in S + X: say(json.dumps(s, default=lambda o: float(o)))

def tables(reps):  # markdown tables from stats.json -> tables.md
    S = json.loads((HERE / "stats.json").read_text()); f = lambda x, k=1: "-" if x in (None, "None") else f"{float(x)*100:.{k}f}%"
    short = lambda m: m.split("/")[1]
    o = ["| model | cond | reps | calls | parse fails (empty / truncated / off-list) | flip rate | pairwise disagree | pairwise excl. empty-content calls | floor vs best fixed student |", "|---|---|---|---|---|---|---|---|---|"]
    for s in S: o.append(f"| {short(s['model'])} | {s['cond']} | {s['reps']} | {s['n_calls']} | {s['parse_fail']} ({s['empty']} / {s['trunc']} / {s['off']}) | {f(s['flip'])} | {f(s['pair'],2)} | {f(s['pair_ok'],2)} | {f(s['floor'],2)} |")
    o += ["", "| model | cond | single-run acc (mean, min-max) | majority-vote acc | prompts wrong in all reps |", "|---|---|---|---|---|"]
    for s in S: o.append(f"| {short(s['model'])} | {s['cond']} | {f(s['single_acc'])} ({f(s['single_min'])}-{f(s['single_max'])}) | {f(s['maj_acc'])} | {s['wrong_all']} |")
    o += ["", "| model | cond | p50 ms | p95 ms | prompt tok | completion tok | reasoning tok (mean/max) | $/call | $/1000 labels |", "|---|---|---|---|---|---|---|---|---|"]
    for s in S: o.append(f"| {short(s['model'])} | {s['cond']} | {s['p50']:.0f} | {s['p95']:.0f} | {float(s['ptok']):.0f} | {float(s['ctok']):.1f} | {float(s['rtok']):.2f}/{s['rtok_max']} | {float(s['usd_call']):.6f} | {float(s['usd_1k']):.3f} |")
    o += ["", "| model | cond | providers (calls) | prompts with provider change | flip rate if provider changed | flip rate if same provider | pair disagree diff-provider | pair disagree same-provider |", "|---|---|---|---|---|---|---|---|"]
    for s in S: o.append(f"| {short(s['model'])} | {s['cond']} | {s['provs']} | {s['n_chg']} | {f(s['flip_chg'])} | {f(s['flip_same'])} | {f(s['dis_diffprov'],2)} | {f(s['dis_sameprov'],2)} |")
    X = json.loads((HERE / "cross.json").read_text())
    o += ["", "| model | pinned endpoint | served by | cross-provider disagree vs default-routing A (all pairs) | within pinned endpoint (2 reps) | single-run acc | p50 ms | p95 ms | $/call |", "|---|---|---|---|---|---|---|---|---|"]
    for x in X: o.append(f"| {short(x['model'])} | {x['tag']} | {x['provider']} | {f(x['cross'],2)} | {f(x['within_pin'],1)} | {f(x['acc'])} | {x['p50']:.0f} | {x['p95']:.0f} | {x['usd_call']:.6f} |")
    (HERE / "tables.md").write_text("\n".join(o) + "\n", encoding="utf-8"); print("\n".join(o))

if __name__ == "__main__":
    mode = sys.argv[1]; reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    {"probe": probe, "full": lambda: full(reps), "report": lambda: report(reps), "extra": extra, "tables": lambda: tables(reps)}[mode]()
