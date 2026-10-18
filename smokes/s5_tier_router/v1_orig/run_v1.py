"""S5 tier / thinking router. Usage: run.py a | probe | live | analyze
a       Part A, RouterBench offline (needs rb_data/routerbench_0shot.pkl)
probe   Part B, 3 calls per config, prints usage
live    Part B, 200 Banking77 inputs x all configs -> responses.jsonl (resumable, spend-capped)
analyze Part B tables + router + certificate from responses.jsonl
ponytail: one file, no classes; results printed (stdout -> log.txt)."""
import sys, os, json, time, difflib, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\Users\<user>\Desktop\personalprojects\Dreaming\smokes")
from common import *  # noqa
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).parent
RESP = HERE / "responses.jsonl"
CAP = 7.0           # hard spend cap for this smoke (USD)
STOP_AT = 6.5       # stop issuing calls past this
N = 200
SUB = 5000        # Part A router subsample (CPU embedding is the bottleneck on this shared machine)

# ---------------------------------------------------------------- Part A
def family(e):
    if e.startswith("mmlu-"): return "mmlu"
    if e.startswith("mtbench"): return "mtbench"
    return {"grade-school-math": "gsm8k", "arc-challenge": "arc", "hellaswag": "hellaswag", "winogrande": "winogrande", "mbpp": "mbpp"}.get(e, "other")

def curve_row(name, acc, cost, share):
    return f"| {name} | {share*100:.0f}% | {acc:.4f} | {cost*1000:.2f} |"

def part_a():
    sys.path.insert(0, str(HERE))
    from rb_load import load
    df = load(HERE / "rb_data" / "routerbench_0shot.pkl")
    M = [c for c in df.columns if "|" not in c and c not in ("sample_id", "prompt", "eval_name", "oracle_model_to_route_to")]
    fam = df.eval_name.map(family).values
    S = {m: df[m].astype(float).values for m in M}
    C = {m: df[m + "|total_cost"].astype(float).values for m in M}
    print(f"[A] rows={len(df)} models={len(M)}")
    print("[A] rows per family:", pd.Series(fam).value_counts().to_dict())
    print("[A] per-family score values:", {f: sorted(set(np.round(np.concatenate([S[m][fam == f] for m in M]), 2)))[:6] + (["..."] if len(set(np.round(np.concatenate([S[m][fam == f] for m in M]), 2))) > 6 else []) for f in sorted(set(fam))})
    print("[A] mean score / total cost per model:")
    for m in M: print(f"   {m:40s} acc={S[m].mean():.4f} cost_per_1k=${C[m].mean()*1000:.3f}")
    EXP = "gpt-4-1106-preview"
    pairs = [("gpt-3.5-turbo-1106", EXP), ("mistralai/mixtral-8x7b-chat", EXP), ("meta/llama-2-70b-chat", EXP),
             ("mistralai/mistral-7b-chat", EXP), ("claude-instant-v1", "claude-v2")]
    out = []
    out.append("| family | n | exp | cheap | acc_exp | acc_cheap | ok-agree | P(cheap ok given exp ok) | oracle acc | oracle cost saving | cheap/exp cost |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for ch, ex in pairs:
        for f in ["ALL"] + sorted(set(fam)):
            k = np.ones(len(df), bool) if f == "ALL" else fam == f
            se, sc, ce, cc = S[ex][k], S[ch][k], C[ex][k], C[ch][k]
            oe, oc = se >= .5, sc >= .5
            r = sc >= se  # cheap at least as good -> route cheap
            acc_or = np.where(r, sc, se).mean(); cost_or = np.where(r, cc, ce).sum()
            out.append(f"| {f} | {k.sum()} | {ex} | {ch} | {se.mean():.3f} | {sc.mean():.3f} | {(oe == oc).mean():.3f} | "
                       f"{(oc[oe].mean() if oe.any() else float('nan')):.3f} | {acc_or:.3f} | {100*(1-cost_or/ce.sum()):.1f}% | {cc.sum()/ce.sum():.3f} |")
    print("\n".join(out))

    # ---- router: P(cheap at least as good) from the prompt, 5-fold OOF, main pairs
    import torch; torch.set_num_threads(4)
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    prompts = df.prompt.astype(str).values
    sub = np.sort(train_test_split(np.arange(len(df)), train_size=SUB, stratify=fam, random_state=0)[0])  # CPU-bound embedding: subsample
    t0 = time.time(); Xe = embed([p[:400] for p in prompts[sub]]); print(f"[A] embed {len(sub)} subsampled prompts (400 chars) done {time.time()-t0:.0f}s")
    Xt = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True, min_df=3).fit_transform([p[:1200] for p in prompts])
    for ch, ex in pairs[:2]:  # gpt-3.5 and mixtral vs gpt-4
        for tag, X, rows, full in (("bge-small emb-LR", Xe, sub, True), ("tfidf-LR (all 36.5k)", Xt, np.arange(len(df)), False)):
            router(tag, X, rows, S[ch][rows], S[ex][rows], C[ch][rows], C[ex][rows], fam[rows], f"{ch} vs {ex}", full)

def router(tag, X, rows, sc, se, cc, ce, fam, title, full):
    y = (sc >= se).astype(int); n = len(y)
    oof = np.zeros(n); oof_fam = np.zeros(n)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        oof[te] = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear" if "tfidf" in tag else "lbfgs").fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        rate = pd.Series(y[tr]).groupby(fam[tr]).mean()
        oof_fam[te] = pd.Series(fam[te]).map(rate).fillna(y[tr].mean()).values
    print(f"\n[A] ROUTER [{tag}] {title}: n={n} base rate cheap>=exp = {y.mean():.3f}; AUC router={roc_auc_score(y, oof):.4f} family-prior={roc_auc_score(y, oof_fam):.4f}")
    rng = np.random.default_rng(0)
    if full:
        lines = ["| policy | share to cheap | accuracy | $ per 1k calls |", "|---|---|---|---|"]
        lines.append(curve_row("always-expensive", se.mean(), ce.mean(), 0))
        lines.append(curve_row("always-cheap", sc.mean(), cc.mean(), 1))
        r = y == 1
        lines.append(curve_row("oracle (cheap iff cheap>=exp)", np.where(r, sc, se).mean(), np.where(r, cc, ce).mean(), r.mean()))
        for q in (.1, .2, .3, .4, .5, .6, .7, .8, .9):
            rt = oof >= np.quantile(oof, 1 - q)
            lines.append(curve_row(f"router top {int(q*100)}%", np.where(rt, sc, se).mean(), np.where(rt, cc, ce).mean(), rt.mean()))
        for q in (.3, .6):
            rt = oof_fam >= np.quantile(oof_fam, 1 - q)
            lines.append(curve_row(f"family-prior top {int(q*100)}%", np.where(rt, sc, se).mean(), np.where(rt, cc, ce).mean(), rt.mean()))
            rt = rng.random(n) < q
            lines.append(curve_row(f"random {int(q*100)}%", np.where(rt, sc, se).mean(), np.where(rt, cc, ce).mean(), rt.mean()))
        print("\n".join(lines))
    base_acc, base_cost = se.mean(), ce.mean(); res = []
    for name, sco in (("router", oof), ("family-prior", oof_fam)):
        res = []
        for tol in (0.0, 0.005, 0.01, 0.02, 0.05):
            best = 0.0
            for q in np.linspace(0, 1, 101):
                rt = sco >= np.quantile(sco, 1 - q) if q > 0 else np.zeros(n, bool)
                if np.where(rt, sc, se).mean() >= base_acc - tol:
                    best = max(best, 1 - np.where(rt, cc, ce).mean() / base_cost)
            res.append(f"loss<={tol*100:.1f}pt: {best*100:.1f}%")
        print(f"[A] [{tag}] max cost saving vs always-exp ({name}):", "; ".join(res))
    idx = np.random.default_rng(1).permutation(n); cal, ho = idx[: n // 2], idx[n // 2:]
    for b in (0.01, 0.02, 0.05):
        c = certify(oof[cal], y[cal] == 1, budget=b)
        if c["coverage"] == 0: print(f"[A] [{tag}] certify budget={b}: nothing certifiable"); continue
        h = apply_certificate(oof[ho], y[ho] == 1, c["cutoff"]); rt = oof[ho] >= c["cutoff"]
        print(f"[A] [{tag}] certify budget={b}: cutoff={c['cutoff']:.3f} cal coverage={c['coverage']:.3f} cal err={c['emp_error']:.4f} bound={c['bound']:.4f} | holdout coverage={h['coverage']:.3f} err={h['error']:.4f} "
              f"acc={np.where(rt, sc[ho], se[ho]).mean():.4f} (exp {se[ho].mean():.4f}) cost saving={100*(1-np.where(rt, cc[ho], ce[ho]).sum()/ce[ho].sum()):.1f}%")
    if full:
        wa = {f: roc_auc_score(y[fam == f], oof[fam == f]) for f in sorted(set(fam)) if 0 < y[fam == f].mean() < 1}
        print(f"[A] [{tag}] within-family AUC:", {k: round(v, 3) for k, v in wa.items()})

# ---------------------------------------------------------------- Part B
PRICE = {"openai/gpt-5.4": (2.5, 15), "openai/gpt-5.4-mini": (.75, 4.5), "openai/gpt-5.4-nano": (.2, 1.25),
         "qwen/qwen3-32b": (.08, .28), "anthropic/claude-sonnet-4.6": (3, 15), "anthropic/claude-haiku-4.5": (1, 5)}
OFF = {"reasoning": {"effort": "none"}}
CFG = {  # name: (model, extra_body, max_tokens)
    "E1": ("openai/gpt-5.4", {"reasoning": {"effort": "low"}}, 2000),
    "C1": ("openai/gpt-5.4-nano", OFF, 60),
    "C2": ("openai/gpt-5.4-mini", OFF, 60),
    "C2b": ("openai/gpt-5.4-mini", OFF, 60),  # repeat of C2: noise floor of "same config twice"
    "E1b": ("openai/gpt-5.4", {"reasoning": {"effort": "low"}}, 2000),  # repeats added after finding OpenAI ignores temperature=0 via OpenRouter
    "C1b": ("openai/gpt-5.4-nano", OFF, 60),
    "E2b": ("anthropic/claude-sonnet-4.6", {"reasoning": {"enabled": False}}, 60),
    "C3": ("qwen/qwen3-32b", {"reasoning": {"enabled": False}}, 60),  # OpenRouter reasoning param is ignored by DeepInfra qwen3-32b; only the " /no_think" soft switch (SUFFIX) turns thinking off
    "Ton": ("openai/gpt-5.4-mini", {"reasoning": {"effort": "high"}}, 4000),
    "E2": ("anthropic/claude-sonnet-4.6", {"reasoning": {"enabled": False}}, 60),
    "C4": ("anthropic/claude-haiku-4.5", {"reasoning": {"enabled": False}}, 60),
}
SUFFIX = {"C3": " /no_think"}
lock = threading.Lock()

def usd(model, pt, ct):
    i, o = PRICE[model]; return (pt * i + ct * o) / 1e6

def b_data():
    d = load_banking77(); names = d["names"]; texts, labels = d["test"]
    idx = stratified_subsample(labels, N, seed=0)
    return names, [texts[i] for i in idx], labels[idx]

def sys_prompt(names):
    return "You are a bank support intent classifier. Reply with exactly one label from the list, nothing else.\n" + "\n".join(names)

def parse(txt, names):
    t = (txt or "").strip().strip("`'\".").strip().lower()
    low = {n.lower(): n for n in names}
    if t in low: return low[t], False
    if not t: return None, True
    return low[difflib.get_close_matches(t, list(low), n=1, cutoff=0.0)[0]], True

def spent():
    if not RESP.exists(): return 0.0
    return sum(json.loads(l).get("usd", 0.0) for l in RESP.read_text(encoding="utf-8").splitlines() if l.strip())

def call(client, cfg, i, query, sysmsg):
    model, extra, mt = CFG[cfg]
    last = None
    for attempt in range(4):
        t0 = time.time()
        try:
            r = client.chat.completions.create(model=model, temperature=0, max_tokens=mt, extra_body=extra,
                messages=[{"role": "system", "content": sysmsg}, {"role": "user", "content": query + SUFFIX.get(cfg, "")}])
            ms = (time.time() - t0) * 1000; u = r.usage
            rt = ((getattr(u, "completion_tokens_details", None) and u.completion_tokens_details.reasoning_tokens) or 0)
            return dict(cfg=cfg, i=i, model=model, content=r.choices[0].message.content, finish=r.choices[0].finish_reason, ms=ms,
                        pt=u.prompt_tokens, ct=u.completion_tokens, rt=rt, usd=usd(model, u.prompt_tokens, u.completion_tokens),
                        usd_rep=getattr(u, "cost", None), served_by=getattr(r, "provider", None), err=None)
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:300]}"
            if "400" in last or "404" in last or "401" in last or "402" in last: break  # config problem: do not burn retries
            time.sleep(2 ** attempt)
    return dict(cfg=cfg, i=i, model=model, content=None, err=last, usd=0.0, ms=None, pt=0, ct=0, rt=0)

def client_():
    from openai import OpenAI
    key = [l.split("=", 1)[1].strip() for l in (HERE.parent.parent / ".env").read_text().splitlines() if l.startswith("OPENROUTERKEY=")][0]
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, timeout=180, max_retries=0)

def run_calls(jobs, names, texts, tag):
    cl = client_(); sysmsg = sys_prompt(names); bad = set(); total = [spent()]
    def work(j):
        cfg, i = j
        if cfg in bad: return None
        if total[0] >= STOP_AT: return None
        r = call(cl, cfg, i, texts[i], sysmsg)
        with lock:
            total[0] += r["usd"]
            with open(RESP, "a", encoding="utf-8") as f: f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if r["err"] and ("400" in r["err"] or "404" in r["err"]): bad.add(cfg); print(f"[{tag}] CONFIG ERROR {cfg}: {r['err']}")
        return r
    with ThreadPoolExecutor(8) as ex: res = [r for r in ex.map(work, jobs) if r]
    print(f"[{tag}] done calls={len(res)} total_spent=${total[0]:.4f}")
    return res

def probe():
    names, texts, y = b_data()
    res = run_calls([(c, i) for c in CFG for i in range(3)], names, texts, "probe")
    for r in res:
        lab, pf = parse(r["content"], names)
        print(f"{r['cfg']:4s} i={r['i']} gold={names[y[r['i']]]:34s} -> {str(r['content'])[:40]!r:44s} parse_fail={pf} ms={r['ms'] and int(r['ms'])} pt={r['pt']} ct={r['ct']} rt={r['rt']} usd={r['usd']:.5f} rep={r.get('usd_rep')} fin={r.get('finish')} err={r['err']}")
    per = pd.DataFrame(res).groupby("cfg").agg(usd=("usd", "mean"), ct=("ct", "mean"), rt=("rt", "mean"), pt=("pt", "mean"))
    per["proj_200"] = per.usd * 200
    print(per); print("projected total for 200 each: $%.2f (cap %.1f)" % (per.proj_200.sum(), CAP))

def done_set():
    return {(r["cfg"], r["i"]) for r in map(json.loads, RESP.read_text(encoding="utf-8").splitlines()) if r.get("err") is None} if RESP.exists() else set()

def live(cfgs=None):
    names, texts, y = b_data()
    dn = done_set(); cfgs = cfgs or list(CFG)
    jobs = [(c, i) for c in cfgs for i in range(N) if (c, i) not in dn]
    print(f"[live] jobs={len(jobs)} already_spent=${spent():.4f}")
    run_calls(jobs, names, texts, "live")

# ---------------------------------------------------------------- analysis
def analyze():
    names, texts, y = b_data()
    rows = [json.loads(l) for l in RESP.read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = [r for r in rows if r["err"] is None]
    last = {}  # latest successful record per (cfg, i); probe calls are duplicates of the same call
    for r in ok: last.setdefault((r["cfg"], r["i"]), r)
    cfgs = [c for c in CFG if any(k[0] == c for k in last)]
    err = {c: sum(1 for r in rows if r["cfg"] == c and r["err"]) for c in CFG}
    P = {}; PF = {}
    for c in cfgs:
        P[c] = np.array([parse(last[(c, i)]["content"], names)[0] if (c, i) in last else None for i in range(N)], dtype=object)
        PF[c] = np.array([parse(last[(c, i)]["content"], names)[1] if (c, i) in last else False for i in range(N)])
    gold = np.array([names[k] for k in y], dtype=object)
    def have(c): return np.array([(c, i) in last for i in range(N)])
    def match(a, b):
        m = have(a) & have(b); return float((P[a][m] == P[b][m]).mean()), int(m.sum())
    print(f"[B] total spend ${spent():.4f}; error rows per cfg: {err}")
    L = ["| cfg | model | setting | n | gold acc | parse fails | match vs E1 | mean ms | p50 ms | prompt tok | compl tok | reasoning tok | $/1000 calls |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    cost1k = {}
    for c in cfgs:
        rr = [last[(c, i)] for i in range(N) if (c, i) in last]
        n = len(rr); ms = [r["ms"] for r in rr]; cost1k[c] = 1000 * np.mean([r["usd"] for r in rr])
        m = have(c)
        mv = "-" if c == "E1" else "%.3f" % match(c, "E1")[0] if "E1" in P else "-"
        L.append(f"| {c} | {CFG[c][0]} | {json.dumps(CFG[c][1])} | {n} | {(P[c][m] == gold[m]).mean():.3f} | {PF[c][m].sum()} | {mv} | {np.mean(ms):.0f} | {np.median(ms):.0f} | "
                 f"{np.mean([r['pt'] for r in rr]):.0f} | {np.mean([r['ct'] for r in rr]):.1f} | {np.mean([r['rt'] for r in rr]):.1f} | {cost1k[c]:.3f} |")
    print("\n".join(L))
    print("\n[B] pairwise agreement (same-label rate):")
    for a, b in [("C2", "C2b"), ("E1", "E1b"), ("C1", "C1b"), ("E2", "E2b"), ("C1", "E1b"), ("Ton", "C2"), ("Ton", "E1"), ("C4", "E2"), ("E2", "E1"), ("C1", "E1"), ("C2", "E1"), ("C3", "E1"), ("C4", "E1")]:
        if a in P and b in P:
            m, n = match(a, b); both = have(a) & have(b)
            print(f"   {a:4s} vs {b:4s}: match={m:.3f} (n={n}) disagreements={int(round((1-m)*n))}; gold acc {a}={(P[a][both]==gold[both]).mean():.3f} {b}={(P[b][both]==gold[both]).mean():.3f}")
    for c in [c for c in ("C1", "C2", "C3", "C4", "Ton") if c in P and "E1" in P]:  # who is right when cheap != E1
        dd = P[c] != P["E1"]; eo, co = P["E1"] == gold, P[c] == gold
        print(f"   {c} vs E1 disagreements={dd.sum()}: E1 right only={(dd&eo&~co).sum()}, cheap right only={(dd&~eo&co).sum()}, both wrong={(dd&~eo&~co).sum()}; "
              f"E1 wrong overall={(~eo).sum()}, cheap wrong where E1 wrong={(~eo&~co).sum()}")
    if "C1" in P and "E1" in P:
        pairs_ = pd.Series([f"{a} <> {b}" for a, b in zip(P["E1"][P["C1"] != P["E1"]], P["C1"][P["C1"] != P["E1"]])]).value_counts().head(5)
        print("   most frequent E1 -> C1 label swaps:", pairs_.to_dict())
    if "Ton" in P and "C2" in P:
        rr = [last[("Ton", i)] for i in range(N) if ("Ton", i) in last]
        print(f"   Ton finish reasons: {pd.Series([r['finish'] for r in rr]).value_counts().to_dict()}; reasoning tok mean={np.mean([r['rt'] for r in rr]):.0f} max={max(r['rt'] for r in rr)}")
    # ---- router for the best cheap config vs E1
    cheap = [c for c in ("C1", "C2", "C3", "C4") if c in P and have(c).all() and have("E1").all()]
    if "E1" not in P or not cheap: print("[B] no router (missing E1/cheap)"); return
    X = embed(texts)
    print("\n[B] router, label = (cheap != E1), 5-fold LR on bge-small embeddings (AUC of predicting disagreement):")
    best = None
    for c in cheap:
        d = (P[c] != P["E1"]).astype(int)
        if d.sum() < 5 or d.sum() > N - 5: print(f"   {c}: disagreements={d.sum()} (too few/many for CV)"); aucs = None
        else:
            oof = np.zeros(N)
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, d):
                oof[te] = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], d[tr]).predict_proba(X[te])[:, 1]
            print(f"   {c}: disagreements={d.sum()}/{N} ({d.mean():.3f}) AUC={roc_auc_score(d, oof):.3f}")
        key = (1 - d.mean(), -cost1k[c])
        if best is None or key > best[0]: best = (key, c, d)
    _, bc, d = best; print(f"[B] best cheap config by match vs E1 (ties -> cheaper): {bc} (match {1-d.mean():.3f}, ${cost1k[bc]:.3f} vs E1 ${cost1k['E1']:.3f} per 1000)")
    # min rows to certify a 0-error prefix
    from math import ceil, log
    for b in (0.02, 0.05, 0.10):
        m_ = ceil(np.log2(N)) + 1; dl = 0.05 / m_
        print(f"   min zero-error rows to certify budget {b}: {ceil(log(dl)/log(1-b))} (delta/m={dl:.4f}); we have n={N}")
    if d.sum() >= 5:
        oof = np.zeros(N)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, d):
            oof[te] = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], d[tr]).predict_proba(X[te])[:, 1]
        conf = 1 - oof; agree = d == 0
        cal, ho = next(StratifiedKFold(2, shuffle=True, random_state=1).split(X, d))
        for b in (0.02, 0.05, 0.10, 0.20):
            c = certify(conf[cal], agree[cal], budget=b)
            if c["coverage"] == 0:
                print(f"   certify budget={b}: nothing certifiable on {len(cal)} calibration rows; no-guarantee version below")
            else:
                h = apply_certificate(conf[ho], agree[ho], c["cutoff"]); sh = h["coverage"]
                print(f"   certify budget={b}: cutoff={c['cutoff']:.3f} cal cov={c['coverage']:.3f} err={c['emp_error']:.3f} bound={c['bound']:.3f} | holdout share-cheap={sh:.3f} disagree={h['error']} n_above={h['n_above']} blended $/1000={sh*cost1k[bc]+(1-sh)*cost1k['E1']:.3f}")
        # empirical (uncertified) cutoff: lowest cutoff with cal disagreement <= budget
        for b in (0.02, 0.05, 0.10, 0.15):
            o = np.argsort(-conf[cal]); cc, ag = conf[cal][o], agree[cal][o]; e = np.cumsum(~ag) / (np.arange(len(ag)) + 1)
            k = np.where(e <= b)[0]; k = k.max() if len(k) else -1
            if k < 0: print(f"   empirical budget={b}: none"); continue
            cut = cc[k]; h = apply_certificate(conf[ho], agree[ho], cut)
            print(f"   empirical (no guarantee) budget={b}: cutoff={cut:.3f} cal cov={(k+1)/len(cc):.3f} | holdout share-cheap={h['coverage']:.3f} disagree={h['error']} n_above={h['n_above']} blended $/1000={h['coverage']*cost1k[bc]+(1-h['coverage'])*cost1k['E1']:.3f}")
        o = np.argsort(-conf)
        print("   descriptive (all 200, OOF conf; no guarantee): share routed -> disagreement vs E1 in routed set:",
              "; ".join(f"{q:.0%}->{(d[o][:int(q*N)]).mean():.3f}" for q in (.1, .2, .3, .4, .5, .6, .8, 1.0)))
        print(f"   always-cheap {bc}: disagree vs E1 = {d.mean():.3f}, $/1000={cost1k[bc]:.3f}")

if __name__ == "__main__":
    {"a": part_a, "probe": probe, "live": live, "analyze": analyze}[sys.argv[1]]()
