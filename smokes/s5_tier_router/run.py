"""S5 tier / thinking router. Usage: run.py check | a | alpha_check | probe | live | analyze | evidence
check   self-check of the Part A cost/accuracy path math
a       Part A, RouterBench offline (needs rb_data/routerbench_0shot.pkl)
alpha_check  OOF R2 of the ridge regressors vs alpha (how ALPHA was picked)
probe   Part B, 3 calls per config, prints usage
live    Part B, 200 Banking77 inputs x all configs -> responses.jsonl (resumable, spend-capped)
analyze Part B tables + router + certificate from responses.jsonl
evidence Part B, small paid re-runs that put the temperature / qwen / interleaved-latency evidence on disk (evidence.jsonl)
ponytail: one file, no classes; results printed (stdout -> log.txt)."""
import sys, os, json, time, difflib, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\Users\<user>\Desktop\personalprojects\Dreaming\smokes")
from common import *  # noqa
from sklearn.linear_model import LogisticRegression, Ridge
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

CHANCE = {"mmlu": .25, "arc": .25, "hellaswag": .25, "winogrande": .5}  # multiple-choice chance level
ALPHA = {"loss": 10.0, "saving": 1.0}  # ridge strengths; picked from {.1,1,10,100,1000} by OOF R2 on the 5k embedding features (loss R2 .04, saving R2 .73), reused for tf-idf
TOLS = (0.0, .005, .01, .02, .05)  # tolerated mean accuracy loss vs always-expensive
SAVS = (.05, .10, .20, .30, .40, .50)  # target cost saving vs always-expensive

def load_a():
    sys.path.insert(0, str(HERE))
    from rb_load import load
    df = load(HERE / "rb_data" / "routerbench_0shot.pkl")
    M = [c for c in df.columns if "|" not in c and c not in ("sample_id", "prompt", "eval_name", "oracle_model_to_route_to")]
    return df, M, df.eval_name.map(family).values

def part_a():
    df, M, fam = load_a()
    S = {m: df[m].astype(float).values for m in M}
    C = {m: df[m + "|total_cost"].astype(float).values for m in M}
    print(f"[A] rows={len(df)} models={len(M)}")
    print("[A] rows per family:", pd.Series(fam).value_counts().to_dict())
    print("[A] per-family score values:", {f: sorted(set(np.round(np.concatenate([S[m][fam == f] for m in M]), 2).tolist()))[:6] for f in sorted(set(fam))})
    print("[A] mean score / total cost per model:")
    for m in M: print(f"   {m:40s} acc={S[m].mean():.4f} cost_per_1k=${C[m].mean()*1000:.3f}")
    fams = sorted(set(fam))
    acc = pd.DataFrame({f: {m: f"{S[m][fam == f].mean():.3f}" + ("!" if f in CHANCE and S[m][fam == f].mean() <= CHANCE[f] + .03 else "") for m in M} for f in fams})
    print("[A] accuracy by model x family; '!' = at or below multiple-choice chance + 0.03 (chance: mmlu/arc/hellaswag .25, winogrande .5) -> grader/parse artifact suspected:")
    print(acc.to_string())
    EXP = "gpt-4-1106-preview"
    pairs = [("gpt-3.5-turbo-1106", EXP), ("mistralai/mixtral-8x7b-chat", EXP), ("meta/llama-2-70b-chat", EXP),
             ("mistralai/mistral-7b-chat", EXP), ("claude-instant-v1", "claude-v2")]
    out = ["| family | n | exp | cheap | acc_exp | acc_cheap | ok-agree | P(cheap ok given exp ok) | oracle acc | oracle cost saving | cheap/exp cost |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    def row(f, k, ch, ex):
        se, sc, ce, cc = S[ex][k], S[ch][k], C[ex][k], C[ch][k]
        oe, oc = se >= .5, sc >= .5
        r = sc >= se  # cheap at least as good -> route cheap
        out.append(f"| {f} | {k.sum()} | {ex} | {ch} | {se.mean():.3f} | {sc.mean():.3f} | {(oe == oc).mean():.3f} | "
                   f"{(oc[oe].mean() if oe.any() else float('nan')):.3f} | {np.where(r, sc, se).mean():.3f} | {100*(1-np.where(r, cc, ce).sum()/ce.sum()):.1f}% | {cc.sum()/ce.sum():.3f} |")
    for ch, ex in pairs:
        for f in ["ALL"] + fams: row(f, np.ones(len(df), bool) if f == "ALL" else fam == f, ch, ex)
        if "llama-2-70b" in ch: row("ALL ex-mmlu (mmlu score 0.027 = parse artifact)", fam != "mmlu", ch, ex)
        if "mistral-7b" in ch: row("ALL ex-MC (gsm8k/mbpp/mtbench/other only)", ~np.isin(fam, list(CHANCE)), ch, ex)
    print("\n".join(out))

    # ---- routers: 5-fold OOF on the main pairs; every policy judged at equal COST, against a random mix
    import torch; torch.set_num_threads(4)
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    prompts = df.prompt.astype(str).values
    sub = np.sort(train_test_split(np.arange(len(df)), train_size=SUB, stratify=fam, random_state=0)[0])  # CPU-bound embedding: subsample
    t0 = time.time(); Xe = embed([p[:400] for p in prompts[sub]]); print(f"[A] embed {len(sub)} subsampled prompts (400 chars) done {time.time()-t0:.0f}s")
    Xt = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True, min_df=3).fit_transform([p[:1200] for p in prompts])
    for ch, ex in pairs[:2]:  # gpt-3.5 and mixtral vs gpt-4
        for tag, X, rows, tf, R in (("bge-small emb-LR (5k subsample)", Xe, sub, False, 20), ("tfidf-LR (all 36.5k)", Xt, np.arange(len(df)), True, 10)):
            t0 = time.time()
            router(tag, X, S[ch][rows], S[ex][rows], C[ch][rows], C[ex][rows], fam[rows], f"{ch} vs {ex}", tf, R)
            print(f"[A] ({time.time()-t0:.0f}s)")

def pair_data(sc, se, cc, ce, fam):
    return dict(y1=(sc >= se).astype(int), y2=(sc >= .5).astype(int), dacc=se - sc, dcost=ce - cc, fam=fam, ce=ce)

def take(D, ix): return {k: v[ix] for k, v in D.items()}

def ridge(Xa, y, Xb, alpha):  # standardised target: dollar targets are ~1e-3
    m, s = y.mean(), y.std() + 1e-12
    return Ridge(alpha=alpha).fit(Xa, (y - m) / s).predict(Xb) * s + m

def fit_scores(Xa, A, Xb, fam_b, tf):
    """Fit on rows A, score rows Xb. router = P(cheap>=exp); spec = P(cheap ok) (the mission's target); ploss/saving = ridge predictions of
    (acc_exp - acc_cheap) and ($exp - $cheap); costaware ranks by predicted loss per predicted dollar saved (fractional-knapsack greedy)."""
    lr = lambda y: LogisticRegression(C=1.0, max_iter=2000, solver="liblinear" if tf else "lbfgs").fit(Xa, y).predict_proba(Xb)[:, 1]
    pl, ps = ridge(Xa, A["dacc"], Xb, ALPHA["loss"]), ridge(Xa, A["dcost"], Xb, ALPHA["saving"]); eps = .1 * np.median(A["dcost"])
    rate = pd.Series(A["y1"]).groupby(A["fam"]).mean()
    fm = pd.DataFrame(dict(f=A["fam"], l=A["dacc"], s=A["dcost"])).groupby("f").mean()  # per-benchmark mean loss and mean $ saved
    return dict(router=lr(A["y1"]), spec=lr(A["y2"]), ploss=pl, saving=ps, costaware=-pl / np.maximum(ps, eps),
                family=pd.Series(fam_b).map(rate).fillna(A["y1"].mean()).values,
                famca=-(pd.Series(fam_b).map(fm.l) / np.maximum(pd.Series(fam_b).map(fm.s), eps)).fillna(-1e9).values)

def oof(X, D, tf, seed=0):
    n = len(D["y1"]); out = {}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(n), D["y1"]):
        for k, v in fit_scores(X[tr], take(D, tr), X[te], D["fam"][te], tf).items(): out.setdefault(k, np.zeros(n))[te] = v
    return out

def path(score, dacc, dcost, rng):
    """Route rows to cheap in descending score order (random tie-break). Returns cumulative mean accuracy lost L[k], dollars saved S[k]
    after k rows routed (k=0..n), and the sorted scores."""
    o = np.lexsort((rng.random(len(score)), -score))
    return np.r_[0, np.cumsum(dacc[o])] / len(score), np.r_[0, np.cumsum(dcost[o])], score[o]

def sav_at(L, S, ct, tol): return S[L <= tol + 1e-12].max() / ct            # best saving (fraction of always-exp cost) within the loss tolerance
def loss_at(L, S, ct, s):  ok = S >= s * ct; return L[ok.argmax()] if ok.any() else np.nan  # smallest loss reaching saving s
def rand_sav(dacc, dcost, ct, tol):  # random mix, no features: linear between always-exp (0,0) and always-cheap (Ltot, Stot)
    Lt = dacc.mean(); return (1.0 if Lt <= tol else tol / Lt) * dcost.sum() / ct
def rand_loss(dacc, dcost, ct, s):
    St = dcost.sum() / ct; return s / St * dacc.mean() if s <= St else np.nan

def router(tag, X, sc, se, cc, ce, fam, title, tf, R):
    D = pair_data(sc, se, cc, ce, fam); n = len(sc); ct = ce.sum(); dacc, dcost = D["dacc"], D["dcost"]
    O = oof(X, D, tf); eps = .001 * np.median(dcost)
    sco = {"family P(cheap>=exp)": O["family"], "family loss/$": O["famca"], "router P(cheap>=exp)": O["router"], "P(cheap ok) [spec]": O["spec"],
           "cost-aware": O["costaware"], "saving-only": O["saving"], "oracle (true loss/$)": -dacc / np.maximum(dcost, eps)}
    r2 = lambda y, p: 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"\n[A] ROUTER [{tag}] {title}: n={n} base rate cheap>=exp={D['y1'].mean():.3f}; AUC router={roc_auc_score(D['y1'], O['router']):.4f} family-prior={roc_auc_score(D['y1'], O['family']):.4f} "
          f"P(cheap ok)->cheap-ok={roc_auc_score(D['y2'], O['spec']):.4f}; OOF R2 loss={r2(dacc, O['ploss']):.3f} saving={r2(dcost, O['saving']):.3f}")
    print(f"[A] always-exp acc={se.mean():.4f} ${ce.mean()*1000:.2f}/1k; always-cheap acc={sc.mean():.4f} ${cc.mean()*1000:.2f}/1k (max saving {100*(1-cc.sum()/ct):.1f}%, loss {100*dacc.mean():.2f}pt); "
          f"binary oracle (cheap iff cheap>=exp) acc={np.where(D['y1'] == 1, sc, se).mean():.4f} saving={100*(1-np.where(D['y1'] == 1, cc, ce).sum()/ct):.1f}%")
    o_ = np.argsort(-ce); print("[A] cost concentration (rows ranked by the expensive model's own $): " + "; ".join(
        f"top {q:.0%} rows = {100*ce[o_[:int(q*n)]].sum()/ct:.0f}% of $, routing them loses {100*dacc[o_[:int(q*n)]].sum()/n:.2f}pt" for q in (.05, .1, .2, .3)))
    print("[A] by benchmark: " + "; ".join(f"{f} {100*(fam == f).mean():.0f}% rows, {100*ce[fam == f].sum()/ct:.0f}% of $, gap {100*dacc[fam == f].mean():.1f}pt" for f in sorted(set(fam))))
    rng = np.random.default_rng(0); P = {k: path(v, dacc, dcost, rng) for k, v in sco.items()}
    pols = list(sco)
    print("[A] EQUAL-COST view: accuracy loss in points (smaller is better; negative = accuracy gain) at a given cost saving vs always-exp")
    print("| saving | random mix | " + " | ".join(pols) + " |"); print("|---|---|" + "---|" * len(pols))
    for s in SAVS: print(f"| {s*100:.0f}% | {100*rand_loss(dacc, dcost, ct, s):.2f} | " + " | ".join(f"{100*loss_at(*P[k][:2], ct, s):.2f}" for k in pols) + " |")
    print("[A] max cost saving (%) within an accuracy-loss tolerance; cutoff chosen on the SAME out-of-fold data (optimistic, see clean-split table)")
    print("| loss <= | random mix | " + " | ".join(pols) + " |"); print("|---|---|" + "---|" * len(pols))
    for t in TOLS: print(f"| {t*100:.1f}pt | {100*rand_sav(dacc, dcost, ct, t):.1f} | " + " | ".join(f"{100*sav_at(*P[k][:2], ct, t):.1f}" for k in pols) + " |")
    brng = np.random.default_rng(1); bk = ("family loss/$", "router P(cheap>=exp)", "P(cheap ok) [spec]", "cost-aware", "saving-only"); dif = {k: [] for k in bk}
    for _ in range(200):  # bootstrap over rows (scores are fixed OOF values): saving minus random-mix saving
        ix = brng.integers(0, n, n); cb = ce[ix].sum()
        for k in bk:
            L, Sv, _ = path(sco[k][ix], dacc[ix], dcost[ix], brng)
            dif[k].append([sav_at(L, Sv, cb, t) - rand_sav(dacc[ix], dcost[ix], cb, t) for t in TOLS])
    print("[A] saving minus random-mix saving, points (bootstrap 95% interval, 200 resamples of rows)")
    print("| loss <= | " + " | ".join(bk) + " |"); print("|---|" + "---|" * len(bk))
    for j, t in enumerate(TOLS):
        print(f"| {t*100:.1f}pt | " + " | ".join("%+.1f [%+.1f, %+.1f]" % (100*np.mean(np.array(dif[k])[:, j]), *(100*np.percentile(np.array(dif[k])[:, j], [2.5, 97.5]))) for k in bk) + " |")
    # clean protocol: random half = calibration (scorer trained on it only, inner OOF), cutoff picked there, realised on the other half
    key = {"family P(cheap>=exp)": "family", "family loss/$": "famca", "router P(cheap>=exp)": "router", "P(cheap ok) [spec]": "spec", "cost-aware": "costaware", "saving-only": "saving"}
    rec = {(t, k): [] for t in TOLS[1:] for k in list(key) + ["random mix"]}
    for r in range(R):
        rs = np.random.default_rng(100 + r); perm = rs.permutation(n); cal, ho = perm[: n // 2], perm[n // 2:]
        Dc, Dh = take(D, cal), take(D, ho); Sc = oof(X[cal], Dc, tf, seed=r); Sh = fit_scores(X[cal], Dc, X[ho], Dh["fam"], tf); cth = Dh["ce"].sum()
        for k, kk in key.items():
            L, Sv, ss = path(Sc[kk], Dc["dacc"], Dc["dcost"], rs)
            for t in TOLS[1:]:
                idx = np.flatnonzero(L <= t + 1e-12); kb = idx[Sv[idx].argmax()]; cut = ss[kb - 1] if kb > 0 else np.inf
                m = Sh[kk] >= cut; rec[(t, k)].append((Dh["dcost"][m].sum() / cth, Dh["dacc"][m].sum() / len(ho)))
        for t in TOLS[1:]:
            Lc = Dc["dacc"].mean(); p = 1.0 if Lc <= t else t / Lc
            rec[(t, "random mix")].append((p * Dh["dcost"].sum() / cth, p * Dh["dacc"].mean()))
    print(f"[A] CLEAN split ({R} random halves: scorer fit on the calibration half only, cutoff = max saving within tolerance on it, realised on the other half): mean saving % / mean loss pt / share of splits with loss<=tol")
    print("| loss <= | " + " | ".join(["random mix"] + list(key)) + " |"); print("|---|" + "---|" * (len(key) + 1))
    for t in TOLS[1:]:
        cells = []
        for k in ["random mix"] + list(key):
            a = np.array(rec[(t, k)]); cells.append(f"{100*a[:, 0].mean():.1f} / {100*a[:, 1].mean():.2f} / {np.mean(a[:, 1] <= t + 1e-12):.2f}")
        print(f"| {t*100:.1f}pt | " + " | ".join(cells) + " |")
    wa = []
    for f in sorted(set(fam)):
        m = fam == f
        if 0 < D["y1"][m].mean() < 1: wa.append(f"{f} {roc_auc_score(D['y1'][m], O['router'][m]):.3f} (n={m.sum()}, pos={D['y1'][m].sum()}{', n<100: unreliable' if m.sum() < 100 else ''})")
    print(f"[A] [{tag}] within-family AUC of router P(cheap>=exp):", "; ".join(wa))
    idx = np.random.default_rng(1).permutation(n); cal, ho = idx[: n // 2], idx[n // 2:]  # certificate on OOF scores, as in v1
    for b in (0.01, 0.02, 0.05):
        c = certify(O["router"][cal], D["y1"][cal] == 1, budget=b)
        print(f"[A] [{tag}] certify (cheap>=exp) budget={b}: " + ("nothing certifiable" if c["coverage"] == 0 else f"cutoff={c['cutoff']:.3f} cal coverage={c['coverage']:.3f}"))

def alpha_check():
    """OOF R2 of ridge on the 5k bge features (gpt-3.5 vs gpt-4) for the loss and dollar targets: how ALPHA was picked; per-benchmark means alone for comparison."""
    from sklearn.model_selection import KFold, train_test_split
    df, M, fam = load_a(); G, E = "gpt-3.5-turbo-1106", "gpt-4-1106-preview"
    sub = np.sort(train_test_split(np.arange(len(df)), train_size=SUB, stratify=fam, random_state=0)[0])
    X = embed([p[:400] for p in df.prompt.astype(str).values[sub]]); f = fam[sub]
    sc, se, cc, ce = (df[c].astype(float).values[sub] for c in (G, E, G + "|total_cost", E + "|total_cost"))
    r2 = lambda y, o: 1 - ((y - o) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    for name, y in (("dacc", se - sc), ("dcost", ce - cc)):
        for a in (.1, 1, 10, 100, 1000):
            o = np.zeros(len(y))
            for tr, te in KFold(5, shuffle=True, random_state=0).split(X): o[te] = ridge(X[tr], y[tr], X[te], a)
            print(f"[alpha] {name} ridge alpha={a}: OOF R2={r2(y, o):.4f}")
        o = np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=0).split(X): o[te] = pd.Series(f[te]).map(pd.Series(y[tr]).groupby(f[tr]).mean()).values
        print(f"[alpha] {name} per-benchmark mean only: OOF R2={r2(y, o):.4f}")

def selfcheck_a():
    rng = np.random.default_rng(0); n = 2000
    dacc = rng.normal(.15, .4, n); dcost = rng.uniform(.5, 2, n); ct = 3 * dcost.sum()
    L, Sv, _ = path(rng.random(n), dacc, dcost, rng)
    assert abs(L[-1] - dacc.mean()) < 1e-9 and abs(Sv[-1] - dcost.sum()) < 1e-9 and L[0] == 0 and Sv[0] == 0
    assert abs(rand_sav(dacc, dcost, ct, dacc.mean()) - dcost.sum() / ct) < 1e-12 and abs(rand_sav(dacc, dcost, ct, dacc.mean() / 2) - dcost.sum() / ct / 2) < 1e-12
    Lo, So, _ = path(-dacc / dcost, dacc, dcost, rng)
    assert all(sav_at(Lo, So, ct, t) >= rand_sav(dacc, dcost, ct, t) for t in TOLS)  # oracle ranking never loses to the random mix
    print("selfcheck_a ok")

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
    last = {}  # the successful record per (cfg, i); responses.jsonl holds no duplicates (live skips pairs the probe already did), so i<3 are the cold probe calls
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
        n = len(rr); ms = [r["ms"] for r in rr if r["i"] >= 3]; cost1k[c] = 1000 * np.mean([r["usd"] for r in rr])
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
    if all(c in P for c in ("Ton", "C2", "C2b")):  # reasoning effect on the same model, judged against the same-config noise floor
        from scipy.stats import binomtest
        gt = {c: P[c] == gold for c in ("Ton", "C2", "C2b")}; fl = lambda a, b: (P[a] != P[b]).mean()
        for off in ("C2", "C2b"):
            n10, n01 = int((gt["Ton"] & ~gt[off]).sum()), int((~gt["Ton"] & gt[off]).sum())
            print(f"   Ton vs {off}: flips={int((P['Ton'] != P[off]).sum())}/{N}; Ton right only={n10}, {off} right only={n01}, exact sign test p={binomtest(n10, n10 + n01, .5).pvalue:.4f}; gold acc diff={100*(gt['Ton'].mean() - gt[off].mean()):+.1f}pt")
        print(f"   flip rate Ton vs off {fl('Ton','C2'):.3f} / {fl('Ton','C2b'):.3f}; same-config noise floor (C2 vs C2b) {fl('C2','C2b'):.3f} -> flips beyond noise ~ {fl('Ton','C2')-fl('C2','C2b'):.3f} / {fl('Ton','C2b')-fl('C2','C2b'):.3f} (rough: noise flips are not strictly additive)")
        print(f"   gold acc Ton {gt['Ton'].mean():.3f} vs off C2 {gt['C2'].mean():.3f} (diff {100*(gt['Ton'].mean()-gt['C2'].mean()):+.1f}pt), vs mean of C2/C2b {(gt['C2'].mean()+gt['C2b'].mean())/2:.4f} (diff {100*(gt['Ton'].mean()-(gt['C2'].mean()+gt['C2b'].mean())/2):+.2f}pt)")
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
    # split calibration / holdout FIRST; the scorer for calibration rows is out-of-fold inside the calibration half, holdout rows are scored by a model fit on the calibration half only
    from math import ceil, log
    from scipy.stats import beta
    cal, ho = next(StratifiedKFold(2, shuffle=True, random_state=1).split(X, d)); agree = d == 0
    for b in (0.02, 0.05, 0.10):
        need = lambda n_: ceil(log(0.05 / (ceil(np.log2(n_)) + 1)) / log(1 - b))
        print(f"   min zero-error rows to certify budget {b}: {need(len(cal))} on the {len(cal)} calibration rows (certify() uses delta/m with m={ceil(np.log2(len(cal)))+1}); {need(N)} even if all {N} rows were calibration")
    if d.sum() >= 5:
        LRm = lambda: LogisticRegression(C=1.0, max_iter=2000)
        def split_conf(cal_, ho_):  # confidence = P(agree)
            oc = np.zeros(len(cal_))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X[cal_], d[cal_]):
                oc[te] = LRm().fit(X[cal_][tr], d[cal_][tr]).predict_proba(X[cal_][te])[:, 1]
            return 1 - oc, 1 - LRm().fit(X[cal_], d[cal_]).predict_proba(X[ho_])[:, 1]
        def emp_cut(conf, ag, b):  # lowest cutoff whose cumulative disagreement on the calibration half is <= b
            o = np.argsort(-conf); e = np.cumsum(~ag[o]) / (np.arange(len(ag)) + 1); k = np.where(e <= b)[0]
            return (conf[o][k.max()], (k.max() + 1) / len(ag)) if len(k) else (None, 0.0)
        cc_, ch_ = split_conf(cal, ho)
        for b in (0.02, 0.05, 0.10, 0.20):
            c = certify(cc_, agree[cal], budget=b)
            if c["coverage"] == 0:
                print(f"   certify budget={b}: nothing certifiable on {len(cal)} calibration rows")
            else:
                h = apply_certificate(ch_, agree[ho], c["cutoff"]); sh = h["coverage"]
                print(f"   certify budget={b}: cutoff={c['cutoff']:.3f} cal cov={c['coverage']:.3f} err={c['emp_error']:.3f} bound={c['bound']:.3f} | holdout share-cheap={sh:.3f} disagree={h['error']} n_above={h['n_above']} blended $/1000={sh*cost1k[bc]+(1-sh)*cost1k['E1']:.3f}")
        for b in (0.02, 0.05, 0.10, 0.15):  # empirical (uncertified) cutoff, one seeded split
            cut, cov = emp_cut(cc_, agree[cal], b)
            if cut is None: print(f"   empirical budget={b}: none"); continue
            ab = ch_ >= cut; n_ = int(ab.sum()); k_ = int((~agree[ho][ab]).sum()); sh = ab.mean()
            ci = (beta.ppf(.025, k_, n_ - k_ + 1) if k_ else 0.0, beta.ppf(.975, k_ + 1, n_ - k_)) if n_ else (np.nan, np.nan)
            print(f"   empirical (no guarantee, seeded split) budget={b}: cutoff={cut:.3f} cal cov={cov:.3f} | holdout share-cheap={sh:.3f} disagree={k_}/{n_}={k_/max(n_,1):.3f} (95% CP {ci[0]:.3f}-{ci[1]:.3f}) blended $/1000={sh*cost1k[bc]+(1-sh)*cost1k['E1']:.3f}")
        for b in (0.05, 0.10, 0.15, 0.20):  # same thing over 20 random splits: how much of the single-split number is noise
            sh_, dis, nn, none_ = [], 0, 0, 0
            for r in range(20):
                cal_, ho_ = next(StratifiedKFold(2, shuffle=True, random_state=100 + r).split(X, d)); c1, h1 = split_conf(cal_, ho_); cut, _ = emp_cut(c1, agree[cal_], b)
                if cut is None: none_ += 1; continue
                ab = h1 >= cut; sh_.append(ab.mean()); dis += int((~agree[ho_][ab]).sum()); nn += int(ab.sum())
            print(f"   empirical cutoff, 20 random splits, budget={b}: no cutoff in {none_}/20; else mean share routed {np.mean(sh_) if sh_ else float('nan'):.3f} (sd {np.std(sh_) if sh_ else float('nan'):.3f}), pooled holdout disagreement {dis/max(nn,1):.3f} (n={nn})")
        oof = np.zeros(N)  # descriptive only: out-of-fold scores on all 200 rows
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, d): oof[te] = LRm().fit(X[tr], d[tr]).predict_proba(X[te])[:, 1]
        o = np.argsort(-(1 - oof))
        print("   descriptive (all 200, OOF conf; no guarantee): share routed -> disagreement vs E1 in routed set:",
              "; ".join(f"{q:.0%}->{(d[o][:int(q*N)]).mean():.3f}" for q in (.1, .2, .3, .4, .5, .6, .8, 1.0)))
        print(f"   always-cheap {bc}: disagree vs E1 = {d.mean():.3f}, $/1000={cost1k[bc]:.3f}")

# ---------------------------------------------------------------- evidence (small paid re-runs, logged to evidence.jsonl)
def once(cl, model, extra, mt, query, sysmsg, **kw):
    t0 = time.time()
    try:
        r = cl.chat.completions.create(model=model, max_tokens=mt, extra_body=extra, messages=[{"role": "system", "content": sysmsg}, {"role": "user", "content": query}], **kw)
        u = r.usage; rt = ((getattr(u, "completion_tokens_details", None) and u.completion_tokens_details.reasoning_tokens) or 0)
        return dict(model=model, extra=extra, kw=kw, content=r.choices[0].message.content, finish=r.choices[0].finish_reason, ms=(time.time() - t0) * 1000,
                    pt=u.prompt_tokens, ct=u.completion_tokens, rt=rt, usd=usd(model, u.prompt_tokens, u.completion_tokens), err=None)
    except Exception as e:
        return dict(model=model, extra=extra, kw=kw, content=None, err=f"{type(e).__name__}: {str(e)[:300]}", usd=0.0, ms=None, pt=0, ct=0, rt=0)

def evidence():
    import httpx
    names, texts, y = b_data(); cl = client_(); sysm = sys_prompt(names); mini = "openai/gpt-5.4-mini"; jobs = []
    for i in (17, 18):  # temperature=0 repeated on the same input
        jobs += [("temp_repeat", mini, OFF, 60, i, {"temperature": 0}, "")] * 6
    jobs.append(("require_parameters", mini, {**OFF, "provider": {"require_parameters": True}}, 60, 17, {"temperature": 0}, ""))
    for nm, ex, suf in (("effort_none", {"reasoning": {"effort": "none"}}, ""), ("enabled_false", {"reasoning": {"enabled": False}}, ""), ("exclude", {"reasoning": {"exclude": True}}, ""),
                        ("chat_template_kwargs", {"chat_template_kwargs": {"enable_thinking": False}}, ""), ("enabled_false+no_think", {"reasoning": {"enabled": False}}, " /no_think")):
        jobs += [(f"qwen_{nm}", "qwen/qwen3-32b", ex, 60, i, {"temperature": 0}, suf) for i in (0, 1, 2)]
    for i in range(3, 33):  # interleaved latency: same input to every config at about the same time
        jobs += [(f"latency_{c}", *CFG[c], i, {"temperature": 0}, SUFFIX.get(c, "")) for c in ("E1", "C1", "C2", "C3", "Ton", "E2", "C4")]
    tot = [0.0]
    def work(j):
        if tot[0] > 0.6: return None
        k, m, ex, mt, i, kw, suf = j; r = once(cl, m, ex, mt, texts[i] + suf, sysm, **kw); r.update(kind=k, i=i); tot[0] += r["usd"]; return r
    with ThreadPoolExecutor(8) as exr: res = [r for r in exr.map(work, jobs) if r]
    (HERE / "evidence.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in res) + "\n", encoding="utf-8")
    print(f"[evidence] {len(res)} calls, spend ${sum(r['usd'] for r in res):.4f}")
    for i in (17, 18): print(f"  temp_repeat input {i} (gpt-5.4-mini, effort none, temperature=0):", [r["content"] for r in res if r["kind"] == "temp_repeat" and r["i"] == i])
    print("  require_parameters:", [r["err"] or r["content"] for r in res if r["kind"] == "require_parameters"])
    for k in sorted({r["kind"] for r in res if r["kind"].startswith("qwen_")}):
        rr = [r for r in res if r["kind"] == k]; print(f"  {k}: reasoning tok {[r['rt'] for r in rr]} compl tok {[r['ct'] for r in rr]} finish {[r.get('finish') for r in rr]} err {[r['err'] for r in rr if r['err']]}")
    lat = {k: [r["ms"] for r in res if r["kind"] == k and r["ms"]] for k in sorted({r["kind"] for r in res if r["kind"].startswith("latency_")})}
    for k, v in lat.items(): print(f"  {k}: n={len(v)} mean {np.mean(v):.0f} ms, median {np.median(v):.0f} ms, p90 {np.percentile(v, 90):.0f} ms")
    if "latency_C1" in lat: print("  latency ratio vs C1 (mean): " + ", ".join(f"{k[8:]} {np.mean(v)/np.mean(lat['latency_C1']):.2f}x" for k, v in lat.items()))
    try:
        mods = {m["id"]: m for m in httpx.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]}
        for m in sorted({c[0] for c in CFG.values()}): print(f"  models list: {m}: 'temperature' in supported_parameters = {'temperature' in mods[m]['supported_parameters']}" if m in mods else f"  models list: {m} missing")
    except Exception as e: print("  models list fetch failed:", e)

if __name__ == "__main__":
    globals()[{"check": "selfcheck_a", "a": "part_a"}.get(sys.argv[1], sys.argv[1])]()
