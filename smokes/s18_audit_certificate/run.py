"""S18: anytime-valid audit certificate vs one-shot certificates. Usage: python run.py build | analyze | confseq | report   (default: all four in order); --tiny = seconds-long smoke check of analyze+confseq+report on a few rows (writes *_tiny files).
build   : S4 worlds (6 seeds, LR + NN vendor predictor, long serving stream) and Banking77 control streams -> cache/*.npz
analyze : for every stream x budget x audit rate: one-shot (grid CP, LTT) and audit bounds (naive, CP-peek, WSR betting CS, Chernoff e-process, group variants) -> rows.json
confseq : cross-check of my betting CS against the reference confseq.betting.betting_cs on two streams
report  : markdown tables from rows.json"""
import os; os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, json, time, re, importlib.util, itertools
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent; SM = HERE.parent
sys.path.insert(0, str(SM)); sys.path.insert(0, str(SM / "s4_tool_predictor")); sys.path.insert(0, str(HERE))
from common import embed, load_banking77, confidence, certify, stratified_subsample
import cs
CACHE = HERE / "cache"; CACHE.mkdir(exist_ok=True)
SEEDS = (4, 11, 12, 13, 14, 15)          # S4's six worlds (4 = main)
EXTRA = 12000                             # test transactions served after S4's 6000 (S4's test stream = the first 1200 of the serving stream)
BUD = (.01, .02, .05); PS = (.05, .10, .20); S = 100; SW = 40; TINY = "--tiny" in sys.argv; SUF = "_tiny" if TINY else ""
if TINY: S, SW = 8, 4
  # SW: coin sequences used for the (slow) WSR betting CS   # error budgets, audit rates, audit-coin sequences per stream
norm = lambda s: re.sub(r"\d", "0", s.lower())

def s4_module():  # load S4's run.py under another name (my own file is also run.py); not edited
    sp = importlib.util.spec_from_file_location("s4run", SM / "s4_tool_predictor" / "run.py"); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m

# ---------------------------------------------------------------------------------------------------------------- build
def build_world(sd):
    f = CACHE / f"w{sd}.npz"
    if f.exists(): return sd
    from env import Env
    s4 = s4_module(); t0 = time.time(); env = Env(seed=sd); R = []
    for i in range(6000 + EXTRA):   # identical RNG sequence to S4's simulate() for i < 6000 (all three tools), then vendor_lookup only
        t = env.new_txn("train" if i < 3000 else "cal" if i < 4800 else "test"); ret, _ = env.vendor_lookup(t["descriptor"])
        if i < 6000: env.lookup_order(t["order_id"]); env.issue_refund(t["order_id"], t["amount"])
        if not ret.startswith("ERROR"): R.append((i, t["descriptor"], ret))
    I = np.array([r[0] for r in R]); D = [r[1] for r in R]; Y = np.array([r[2] for r in R])
    tr, ca, te = I < 3000, (I >= 3000) & (I < 4800), I >= 4800
    sel = lambda m: [d for d, k in zip(D, m) if k]
    known = set(Y[tr]); seen_tpl = {norm(d) for d in sel(tr)}
    out = dict(te_i=I[te], te_nov=~np.isin(Y[te], list(known)), te_tpl=np.array([norm(d) not in seen_tpl for d in sel(te)]))
    chk = {}
    for kind in ("lr", "nn"):
        pred = s4.fit_text(kind, sel(tr), Y[tr])
        for nm, m in (("cal", ca), ("te", te)):
            lab, conf = pred(sel(m)); out[f"{nm}_conf_{kind}"] = conf; out[f"{nm}_ok_{kind}"] = lab == Y[m]
        # S4 reproduction: its grid-20 certifier at 1%, applied to the first 1200 test transactions
        c = s4.certify_grid(out[f"cal_conf_{kind}"], out[f"cal_ok_{kind}"], .01); fi = out["te_i"] < 6000
        sk = fi & (out[f"te_conf_{kind}"] >= c["cutoff"]); chk[kind] = (int((~out[f"te_ok_{kind}"][sk]).sum()), int(sk.sum()))
    np.savez(f, **out); print(f"[build] world {sd}: {len(I)} calls, test {te.sum()}, novel share {out['te_nov'].mean():.3f}; S4 check (errors/skipped in first 1200 txns, grid-20 @1%): LR {chk['lr']} NN {chk['nn']}; {time.time()-t0:.0f}s", flush=True)
    return sd

def build_b77():
    f = CACHE / "b77.npz"
    if f.exists(): return
    from sklearn.linear_model import LogisticRegression
    b = load_banking77(); Xtr, ytr = embed(b["train"][0]), b["train"][1]; Xte, yte = embed(b["test"][0]), b["test"][1]; out = {}
    for n in (770, 3080):
        idx = stratified_subsample(ytr, n, seed=0); clf = LogisticRegression(C=10, max_iter=2000).fit(Xtr[idx], ytr[idx])
        rest = np.setdiff1d(np.arange(len(ytr)), idx); X = np.r_[Xtr[rest], Xte]; y = np.r_[ytr[rest], yte]   # every row the student never trained on
        P = clf.predict_proba(X); conf = confidence(P); ok = clf.classes_[P.argmax(1)] == y
        for ps in range(3):   # three random cal/serve splits of the pool: cal = 1540 rows (S1's size), serve = the rest, iid order
            perm = np.random.default_rng(100 + ps).permutation(len(y)); c, s = perm[:1540], perm[1540:]
            out[f"n{n}_s{ps}"] = np.stack([np.r_[conf[c], conf[s]], np.r_[ok[c], ok[s]].astype(float)])
            out[f"n{n}_s{ps}_ncal"] = np.array(1540)
        print(f"[build] banking77 student {n} rows: acc on pool {ok.mean():.3f}, pool {len(y)}", flush=True)
    np.savez(f, **out)

# ---------------------------------------------------------------------------------------------------------------- analyze
def served_stats(e, b):
    t = np.arange(1, len(e) + 1); run = np.cumsum(e) / t; at = lambda k: float(run[k - 1]) if len(e) >= k else None
    return dict(T=len(e), real1k=at(1000), real5k=at(5000), realT=float(run[-1]), ever_over_1k=bool((run[999:] > b).any()) if len(e) >= 1000 else None), run

def analyze_stream(job):
    name, scen, pred, world, cal_conf, cal_ok, st_conf, st_ok, groups, wsr_group = job
    rows = []; t0 = time.time(); rng_seed = sum(map(ord, name + scen + pred))
    for b in BUD:
        c = certify(cal_conf, cal_ok, b); l = cs.ltt_fst(cal_conf, cal_ok, b)
        base = dict(stream=name, scen=scen, pred=pred, world=world, budget=b)
        for meth, cert in (("cp", c), ("ltt", l)):   # one-shot certificates: what they promised and what the served stream realized
            sv = st_conf >= cert["cutoff"]
            r = dict(base, kind="oneshot", method=meth, cutoff=cert["cutoff"], cal_cov=cert["coverage"], cal_bound=cert.get("bound"), serve_cov=float(sv.mean()))
            if sv.sum() >= 30:
                st, _ = served_stats(~st_ok[sv], b); r.update(st); r["viol_T"] = st["realT"] > b
            rows.append(r)
        sv = st_conf >= c["cutoff"]
        if sv.sum() < 200: continue
        e = (~st_ok[sv]).astype(float); T = len(e); t = np.arange(1, T + 1); st, run = served_stats(e, b)
        g = {k: v[sv].astype(int) for k, v in groups.items()}
        if wsr_group == "confbin": g["confbin"] = (st_conf[sv] < np.median(cal_conf[cal_conf >= c["cutoff"]])).astype(int)
        for pi, p in enumerate(PS):
            rng = np.random.default_rng(rng_seed * 100 + int(b * 1000) * 10 + pi); A = rng.random((S, T)) < p; Y = (A & (e > 0)).astype(float); A = A.astype(float)
            B = dict(base, kind="audit", p=p, cutoff=c["cutoff"], cal_cov=c["coverage"], cal_bound=c["bound"], serve_cov=float(sv.mean()), paid_share=float(1 - sv.mean() + p * sv.mean()), **st)
            def add(method, U, L=None, idx=None, Lany=None):
                ii = np.arange(T) if idx is None else idx; ref = run[ii]; r = dict(B, kind="audit", method=method)
                call_at = lambda cond: (lambda f: f if idx is None else np.where(np.isnan(f), np.nan, ii[np.nan_to_num(f - 1).astype(int)] + 1.))(cs.first(cond))   # first call count where cond holds (NaN: never)
                r["viol"] = float((U < ref - 1e-12).any(1).mean()); r["viol_T"] = float((U < run[-1] - 1e-12).any(1).mean())
                for k, nm in ((1000, "U1k"), (5000, "U5k"), (T, "UT")):
                    j = np.searchsorted(ii + 1, k); r[nm] = float(np.median(U[:, j])) if k <= T and j < len(ii) and ii[j] + 1 == k else None
                tc = call_at(U <= b); r["cert_frac"] = float(np.isfinite(tc).mean()); r["tcert"] = float(np.median(np.where(np.isnan(tc), np.inf, tc)))
                for nm, LL in (("alarm", L), ("alarm_any", Lany)):   # alarm = pooled lower bound above the budget; alarm_any = some group's lower bound above the budget
                    if LL is not None:
                        ta = call_at(LL > b); r[nm + "_frac"] = float(np.isfinite(ta).mean()); r[nm + "_t"] = float(np.median(np.where(np.isnan(ta), np.inf, ta)))
                rows.append(r)
            add("naive", cs.naive(Y, A))
            U, idx, Lc = cs.cp_peek(Y, A); add("cp_peek", U, Lc, idx=idx)
            U, L, _ = cs.chernoff(Y, p); add("chern", U, L)
            for gn, gv in g.items():
                U, L, Lg = cs.chernoff(Y, p, g=gv); add(f"chern_g[{gn}]", U, L, Lany=(Lg > b).any(2))   # L: pooled (sum of group lower bounds); Lany: some group's error rate certainly above budget
            if p == .10 or (scen == "stationary" and pred == "lr" and name.startswith("world")):   # WSR is slow: all audit rates only on the main streams
                add("wsr", cs.wsr(Y[:SW], p))
                if wsr_group: add(f"wsr_g[{wsr_group}]", cs.wsr(Y[:SW], p, g=g[wsr_group]))
    print(f"[analyze] {name}/{scen}/{pred}: {len(rows)} rows, {time.time()-t0:.0f}s", flush=True)
    return rows

def jobs():
    J = []
    for sd in ((4,) if TINY else SEEDS):
        w = dict(np.load(CACHE / f"w{sd}.npz"))
        if TINY: w = {k: (v[:2500] if k.startswith("te_") else v) for k, v in w.items()}
        n = len(w["te_i"]); grp = dict(tpl=w["te_tpl"], orc=w["te_nov"])
        for pred in ("lr", "nn"):
            cc, co, tc, to = w[f"cal_conf_{pred}"], w[f"cal_ok_{pred}"], w[f"te_conf_{pred}"], w[f"te_ok_{pred}"]
            J.append((f"world{sd}", "stationary", pred, sd, cc, co, tc, to, grp, "tpl"))
            if pred == "lr":   # drift: first half as served, second half resampled with novel-merchant calls 4x as likely (a bootstrap of the same pool)
                r = np.random.default_rng(777 + sd); wt = 1 + 3 * w["te_nov"]; ix = np.r_[np.arange(n // 2), r.choice(n, n - n // 2, p=wt / wt.sum())]
                J.append((f"world{sd}", "shift", pred, sd, cc, co, tc[ix], to[ix], {k: v[ix] for k, v in grp.items()}, "tpl"))
    b = np.load(CACHE / "b77.npz")
    for k in b.files:
        if k.endswith("_ncal") or (TINY and k != "n770_s0"): continue
        nc = int(b[k + "_ncal"]); conf, ok = b[k][0], b[k][1].astype(bool)
        J.append((k, "stationary", "lr", k, conf[:nc], ok[:nc], conf[nc:][:2500 if TINY else None], ok[nc:][:2500 if TINY else None], {}, "confbin"))
    return J

# ---------------------------------------------------------------------------------------------------------------- confseq cross-check
def confseq_check(sds=(4, 11), b=.01, p=.10):
    """Cross-check of cs.wsr against the reference implementation, confseq.betting.betting_cs (Waudby-Smith & Ramdas). pip's confseq needs a C++ toolchain, but betting.py is
    pure Python, so fetch the sdist and import just that module (matplotlib / multiprocess are stubbed when absent: unused with parallel=False). Both give a one-sided 95% upper
    confidence sequence for E[Y] = p * error rate on the same audit streams. confseq's grid adds 1/breaks to its bound (a conservative offset on the Y scale)."""
    import io, tarfile, tempfile, types, urllib.request
    nseq, breaks = (1, 2000) if TINY else (3, 20000)
    try:
        d = Path(tempfile.mkdtemp()); raw = urllib.request.urlopen("https://files.pythonhosted.org/packages/source/c/confseq/confseq-0.0.11.tar.gz", timeout=60).read()
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as t:
            try: t.extractall(d, filter="data")
            except TypeError: t.extractall(d)
        sys.path.insert(0, str(d / "confseq-0.0.11" / "src"))
        for nm in ("matplotlib", "matplotlib.pyplot", "multiprocess"):
            try: __import__(nm)
            except Exception: sys.modules[nm] = types.ModuleType(nm)
        if "matplotlib.pyplot" in sys.modules and not hasattr(sys.modules["matplotlib"], "pyplot"): sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]
        if not hasattr(np, 'float_'): np.float_ = np.float64   # confseq 0.0.11 annotations use the NumPy-1 alias
        from confseq.betting import betting_cs
        res = dict(available=True, version="confseq 0.0.11 (PyPI sdist, pure-Python betting.py; betting_cs(alpha=0.10, theta=1/2) = one-sided 5%), predictable-plug-in EB bets", rows=[])
        for sd in sds[:1] if TINY else sds:
            w = dict(np.load(CACHE / f"w{sd}.npz"))
            if TINY: w = {k: (v[:2500] if k.startswith("te_") else v) for k, v in w.items()}
            c = certify(w["cal_conf_lr"], w["cal_ok_lr"], b); sv = w["te_conf_lr"] >= c["cutoff"]; e = (~w["te_ok_lr"][sv]).astype(float); T = len(e); run = np.cumsum(e) / np.arange(1, T + 1)
            Y = ((np.random.default_rng(sd).random((nseq, T)) < p) & (e > 0)).astype(float); t0 = time.time()
            mine = cs.wsr(Y, p); ref = np.stack([np.minimum(betting_cs(y, alpha=.10, breaks=breaks)[1] / p, 1.) for y in Y])
            for nm, U in (("mine (cs.wsr)", mine), ("confseq betting_cs", ref)):
                at = lambda k: float(np.median(U[:, k - 1])) if T >= k else None
                tc = cs.first(U <= b)
                res["rows"].append(dict(world=sd, T=T, errors=int(e.sum()), realized=float(run[-1]), method=nm, seqs=nseq, U1k=at(1000), U5k=at(5000), UT=at(T), ever_below_realized=float((U < run - 1e-12).any(1).mean()), calls_to_under_budget=float(np.median(np.where(np.isnan(tc), np.inf, tc)))))
            print(f"[confseq] world {sd}: T={T}, {time.time()-t0:.0f}s", flush=True)
    except Exception as ex:
        res = dict(available=False, note=f"{type(ex).__name__}: {str(ex)[:300]}"); print("[confseq] skipped:", res["note"], flush=True)
    json.dump(res, open(HERE / f"confseq_check{SUF}.json", "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))

# ---------------------------------------------------------------------------------------------------------------- report
def report():
    import pandas as pd
    D = pd.DataFrame(json.load(open(HERE / f"rows{SUF}.json")))
    for c in ("U1k", "U5k", "UT", "real1k", "real5k", "realT", "tcert", "alarm_t", "viol", "viol_T", "cutoff", "cal_bound", "T"): D[c] = pd.to_numeric(D[c], errors="coerce")
    out = []; P = out.append
    def md(df, nd=3):
        f = lambda x: "-" if x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)) else ("never" if isinstance(x, (float, np.floating)) and np.isinf(x) else f"{x:.{nd}g}" if isinstance(x, (float, np.floating)) else str(x))
        P("| " + " | ".join(map(str, df.columns)) + " |"); P("|" + "---|" * len(df.columns))
        for r in df.itertuples(index=False): P("| " + " | ".join(f(x) for x in r) + " |")
        P("")
    D["setting"] = np.where(D.stream.str.startswith("world"), np.where(D.scen == "shift", "S4 LR shift", np.where(D.pred == "lr", "S4 LR", "S4 NN")), "B77 control")
    A = D[D.kind == "audit"].copy(); O = D[D.kind == "oneshot"].copy(); O["over"] = np.where(O.viol_T == True, "OVER", np.where(O.viol_T == False, "ok", "-"))
    med = lambda s: float(np.median(s)) if len(s) else np.nan   # inf counts as 'never'
    ren = dict(serve_cov="served", real1k="err@1k", real5k="err@5k", realT="err@all")
    P("### R1. One-shot certificates on the long serving stream (cp = common.certify grid 40; ltt = fixed-sequence, split). OVER = realized error of all served calls exceeds the budget\n")
    for st, key in (("S4 LR", "world"), ("S4 NN", "world"), ("S4 LR shift", "world"), ("B77 control", "stream")):
        P(f"**{st}**\n"); x = O[(O.setting == st) & O["T"].notna()]
        t = x[x.method == "cp"][[key, "budget", "cal_cov", "serve_cov", "T", "real1k", "real5k", "realT", "over"]].rename(columns=ren).rename(columns=dict(over="cp"))
        l = x[x.method == "ltt"][[key, "budget", "serve_cov", "realT", "over"]].rename(columns=dict(serve_cov="ltt served", realT="ltt err@all", over="ltt"))
        md(t.merge(l, on=[key, "budget"], how="outer").sort_values(["budget", key]))
    P("### R2. How often did each one-shot certificate exceed its budget on the long stream? (streams where something was certifiable)\n")
    rows = []
    for st in ("S4 LR", "S4 NN", "S4 LR shift", "B77 control"):
        for m in ("cp", "ltt"):
            x = O[(O.setting == st) & (O.method == m) & O["T"].notna()]
            for b in BUD:
                y = x[x.budget == b]; rows.append([st, m, b, len(y), int((y.viol_T == True).sum()), float((y.viol_T == True).mean()) if len(y) else np.nan, float(y.serve_cov.mean()) if len(y) else np.nan])
    md(pd.DataFrame(rows, columns=["setting", "certificate", "budget", "streams certifiable", "over budget", "share over", "mean served share"]))
    P("### R3. Does the bound ever fall below the realized error? Share of audit-coin sequences with at least one violation (any call t), mean over budgets and audit rates\n")
    methods = ["naive", "cp_peek", "wsr", "chern", "wsr_g[tpl]", "chern_g[tpl]", "chern_g[orc]", "wsr_g[confbin]", "chern_g[confbin]"]
    rows = []
    for st in ("S4 LR", "S4 NN", "S4 LR shift", "B77 control"):
        for m in methods:
            x = A[(A.setting == st) & (A.method == m)]
            if len(x): rows.append([st, m, len(x), x.viol.mean(), x.viol.max(), x.viol_T.mean(), x.viol_T.max(), int((x.viol > .05).sum())])
    md(pd.DataFrame(rows, columns=["setting", "bound", "configs", "ever below running realized (mean)", "(worst config)", "ever below final realized (mean)", "(worst config)", "configs with >5%"]))
    P("Per world (S4 LR stationary), audit rate 10%, mean over the three budgets (naive and cp_peek are not valid bounds; the last column is the one-shot cp certificate):\n")
    cols = [c for c in methods if c in A[A.setting == "S4 LR"].method.unique()]
    x = A[(A.setting == "S4 LR") & (A.p == .10)].pivot_table(index="world", columns="method", values="viol", aggfunc="mean")[cols]
    o = O[(O.setting == "S4 LR") & (O.method == "cp") & O["T"].notna()].groupby("world").viol_T.agg(lambda s: f"{int(s.sum())}/{len(s)}").rename("cp over budget (budgets)")
    md(x.join(o).reset_index())
    P("### R3b. False certification: streams whose final realized error is ABOVE the budget; share of audit-coin sequences in which the bound nevertheless dropped under the budget at some call (the one-shot certificates did this in R2). Lower is safer; a valid bound should stay near or under 5% (it is a proxy: the running error at that call may still have been under the budget)\n")
    x = A[(A.realT > A.budget) & A.method.isin(["naive", "cp_peek", "wsr", "chern", "wsr_g[tpl]", "chern_g[tpl]", "chern_g[orc]"])]
    md(x.groupby(["setting", "method"]).agg(configs=("cert_frac", "size"), mean_share=("cert_frac", "mean"), worst=("cert_frac", "max")).reset_index())
    P("### R4. Calls until the audit bound is under the budget, and its value after 1k / 5k / all served calls (S4 stationary; worlds whose realized error is within the budget; median over coin sequences, WSR 40 sequences, rest 100)\n")
    rows = []
    for st in ("S4 LR", "S4 NN"):
        x = A[(A.setting == st) & A.method.isin(["chern", "wsr", "chern_g[orc]", "chern_g[tpl]", "wsr_g[tpl]"])]
        for (b, p, m), y in x.groupby(["budget", "p", "method"]):
            ok = y[y.realT <= b]
            rows.append([st, b, p, m, len(ok), med(ok.tcert), float(ok.cert_frac.mean()) if len(ok) else np.nan, med(ok.U1k), med(ok.U5k), med(ok.UT), med(ok.realT)])
    md(pd.DataFrame(rows, columns=["setting", "budget", "audit rate", "bound", "worlds in budget", "median calls to certify", "share certified by end", "bound@1k", "bound@5k", "bound@all", "realized@all"]))
    P("Per world, audit rate 10%, pooled Chernoff and WSR bounds:\n")
    y = A[(A.setting == "S4 LR") & (A.p == .10) & A.method.isin(["chern", "wsr"])]
    y = y.pivot_table(index=["budget", "world"], columns="method", values=["tcert", "U1k", "U5k", "UT"], aggfunc="first"); y.columns = [f"{a} {b}" for a, b in y.columns]
    r = A[(A.setting == "S4 LR") & (A.p == .10) & (A.method == "chern")].set_index(["budget", "world"])[["T", "realT"]]
    md(r.join(y).reset_index())
    P("### R5. Re-certify alarms (a lower bound above the budget). Streams where the one-shot cp certificate exceeded its budget, by audit rate. 'frac' = share of coin sequences that ever alarm; 'call' = median call of the first alarm (never = fewer than half alarmed). pooled = Chernoff lower bound on all served calls; tpl-pooled = sum of the per-template-group lower bounds (observable group); tpl-any / orc-any = some group's error rate is certainly above the budget (orc = oracle novelty flag, not observable at serving time); cp_peek = continuously peeked Clopper-Pearson lower bound (not valid)\n")
    bad = O[(O.method == "cp") & (O.viol_T == True) & O.setting.isin(["S4 LR", "S4 NN", "S4 LR shift"])][["setting", "world", "budget", "realT"]]
    alarms = [("pooled", "chern", "alarm"), ("tpl-pooled", "chern_g[tpl]", "alarm"), ("tpl-any", "chern_g[tpl]", "alarm_any"), ("orc-any", "chern_g[orc]", "alarm_any"), ("cp_peek", "cp_peek", "alarm")]
    cell = lambda y, pre, col: float(y[f"{pre}_{col}"].iloc[0]) if len(y) else np.nan
    rows = []
    for r_ in bad.itertuples(index=False):
        for p in PS:
            row = [r_.setting, r_.world, r_.budget, r_.realT, p]
            for nm, m, pre in alarms:
                y = A[(A.setting == r_.setting) & (A.world == r_.world) & (A.budget == r_.budget) & (A.p == p) & (A.method == m)]
                row += [cell(y, pre, "t"), cell(y, pre, "frac")]
            rows.append(row)
    md(pd.DataFrame(rows, columns=["setting", "world", "budget", "realized@all", "audit rate"] + [f"{nm} {c}" for nm, _, _ in alarms for c in ("call", "frac")]))
    P("False alarms: streams where the one-shot certificate held (final realized error within the budget); share of coin sequences in which the alarm fired anyway. The pooled lower bound is a statement about all served calls and is valid; group alarms are statements about each group, which can legitimately sit above the pooled budget while the pool is fine, so they are diagnostics and not budget alarms.\n")
    ok_ = O[(O.method == "cp") & (O.viol_T == False)][["setting", "world", "budget"]]; rows = []
    for nm, m, pre in alarms:
        y = A[A.method == m].merge(ok_, on=["setting", "world", "budget"])
        if f"{pre}_frac" in y: rows += [[nm, st, p, len(z), z[f"{pre}_frac"].mean(), z[f"{pre}_frac"].max()] for (st, p), z in y.groupby(["setting", "p"])]
    md(pd.DataFrame(rows, columns=["alarm", "setting", "audit rate", "configs", "mean false-alarm share", "worst config"]))
    P("### R6. Cost of auditing vs tightness (S4 LR stationary, Chernoff pooled bound, worlds in budget; budget in first column)\n")
    y = A[(A.setting == "S4 LR") & (A.method == "chern")]; y = y[y.realT <= y.budget]
    md(y.groupby(["budget", "p"]).agg(worlds=("serve_cov", "size"), served_share=("serve_cov", "mean"), paid_share=("paid_share", "mean"), bound_1k=("U1k", "median"), bound_5k=("U5k", "median"), bound_all=("UT", "median"), realized=("realT", "median"), calls_to_certify=("tcert", "median")).reset_index())
    P("### R7. Banking77 control (iid errors), audit rate 10%\n")
    y = A[(A.setting == "B77 control") & (A.p == .10) & A.method.isin(["naive", "cp_peek", "chern", "wsr", "chern_g[confbin]", "wsr_g[confbin]"])]
    md(y.groupby(["budget", "method"]).agg(streams=("viol", "size"), viol=("viol", "mean"), calls_to_certify=("tcert", "median"), certified=("cert_frac", "mean"), bound_1k=("U1k", "median"), bound_5k=("U5k", "median"), bound_all=("UT", "median"), realized=("realT", "median")).reset_index())
    f = HERE / f"confseq_check{SUF}.json"
    if f.exists():
        cc = json.load(open(f)); P("### R8. Cross-check: my betting CS (cs.wsr) against the reference confseq.betting.betting_cs, same audit streams (budget 1%, audit rate 10%)\n")
        if cc["available"]: P(cc["version"] + "\n"); md(pd.DataFrame(cc["rows"]))
        else: P("Reference confseq could not be used here: " + cc["note"] + "\n")
    txt = "\n".join(out); (HERE / f"report{SUF}.md").write_text(txt, encoding="utf-8"); print(txt)

if __name__ == "__main__":
    from multiprocessing import Pool
    mode = ([a for a in sys.argv[1:] if not a.startswith("--")] or ["all"])[0]; t0 = time.time()
    if mode in ("build", "all") and not TINY:
        with Pool(6) as pool: list(pool.imap_unordered(build_world, SEEDS))
        build_b77(); print(f"[build] done {time.time()-t0:.0f}s", flush=True)
    if mode in ("analyze", "all"):
        J = jobs(); rows = []
        it = map(analyze_stream, J) if TINY else Pool(min(int(os.environ.get("NPROC", os.cpu_count() or 4)), 16, len(J))).imap_unordered(analyze_stream, J)
        for r in it:
            rows += r; json.dump(rows, open(HERE / f"rows{SUF}.json", "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))   # incremental: a timeout still leaves data
        print(f"[analyze] {len(rows)} rows, {time.time()-t0:.0f}s", flush=True)
    if mode in ("confseq", "all"): confseq_check()
    if mode in ("report", "all"): report()
