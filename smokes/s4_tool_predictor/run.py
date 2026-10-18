"""S4: tool return predictor + speculation + outage test on the SYNTHETIC bookkeeping env (env.py). Prints markdown tables."""
import os; os.environ.setdefault("OMP_NUM_THREADS", "1")  # HGB busy-waits and crawls when the CPU is shared; 1 thread is plenty here
import sys, json, time, re
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from common import confidence, certify as certify_orig, cp_upper
from env import Env, CATS, CHANNELS, OUT, STATUS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from scipy.stats import beta
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error

N, NTR, NCAL, BUDGET = 6000, 3000, 4800, 0.01   # chronological: train [0,3000) | calibration [3000,4800) | test [4800,6000)

def certify_grid(conf, correct, budget, delta=0.05, G=20):
    """Path-independent replacement for common.certify. The Clopper-Pearson bound is U-shaped in the coverage k (loose for small k, rising with accumulated errors for large k), so
    certify's binary search from k=n/2 can step over the feasible region. Here: G fixed coverage levels (top j/G of the rows by confidence), delta/G each (union bound, same overall delta),
    keep the largest level whose bound is <= budget. Candidate sets are tie-closed (conf >= cutoff), matching how the cutoff is applied."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, bool); n = len(conf)
    best = dict(cutoff=float("inf"), coverage=0.0, emp_error=None, bound=None)
    if n == 0: return best
    o = np.argsort(-conf, kind="stable"); cs, ce = conf[o], np.cumsum(~correct[o])
    for j in range(1, G + 1):
        c = cs[int(np.ceil(n * j / G)) - 1]; k = int((cs >= c).sum()); e = int(ce[k - 1]); ub = cp_upper(e, k, delta / G)
        if ub <= budget and k / n >= best["coverage"]: best = dict(cutoff=float(c), coverage=k / n, emp_error=e / k, bound=ub)
    return best
def selfcheck():
    """The CP bound is U-shaped in coverage k: common.certify's search (first probe k=n/2) steps over a feasible region that the grid certifier finds."""
    n = 1000; conf = -np.arange(n, dtype=float); ok = np.ones(n, bool); ok[480:485] = False   # clean top 480 rows, 5 errors just before k=500
    a, g = certify_orig(conf, ok, .02), certify_grid(conf, ok, .02)
    assert np.isinf(a["cutoff"]) and g["coverage"] >= .4 and g["bound"] <= .02, (a, g)
certify = certify_grid  # every certificate below uses the grid certifier; certify_orig (common.certify) is kept only for the before/after comparison

def md(rows, cols):
    f = lambda x: "-" if x is None else f"{x:.4g}" if isinstance(x, (float, np.floating)) else str(x)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(f(x) for x in r) + " |" for r in rows])

# ---- text predictors (shared with S9) ---------------------------------------------------------------------------
def fit_text(kind, X, y):
    """char 2-4gram TF-IDF (lowercased, digits->0) + LR or cosine nearest-neighbour. Returns predict(X) -> (labels, top-2-gap confidence)."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), preprocessor=lambda s: re.sub(r"\d", "0", s.lower()), sublinear_tf=True)
    A = vec.fit_transform(X); y = np.asarray(y)
    if kind == "lr":
        clf = LogisticRegression(C=30, max_iter=300).fit(A, y); classes = clf.classes_
        score = lambda Z: clf.predict_proba(Z)
    else:  # per-class max cosine similarity to training rows
        classes = np.unique(y); cols = [np.where(y == c)[0] for c in classes]
        score = lambda Z: np.stack([(Z @ A[ix].T).max(axis=1).toarray().ravel() for ix in cols], axis=1)
    def predict(Xn):
        s = score(vec.transform(Xn)); return classes[s.argmax(1)], confidence(s)
    return predict

def pick_text(Xtr, ytr, Xcal, ycal, budget):
    """Fit LR and NN, keep the one with higher certified coverage on the calibration slice. Returns (name, predict, table rows)."""
    res = {}
    for kind in ("lr", "nn"):
        p = fit_text(kind, Xtr, ytr); lab, conf = p(Xcal); c = certify(conf, lab == np.asarray(ycal), budget)
        res[kind] = (p, c, float((lab == np.asarray(ycal)).mean()))
    best = max(res, key=lambda k: res[k][1]["coverage"])
    return best, res[best][0], res

# ---- simulate ---------------------------------------------------------------------------------------------------
def simulate(env):
    for i in range(N):
        t = env.new_txn("train" if i < NTR else "cal" if i < NCAL else "test")
        env.vendor_lookup(t["descriptor"]); env.lookup_order(t["order_id"]); env.issue_refund(t["order_id"], t["amount"])
    env.close()

def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    df = pd.json_normalize(rows)
    return {t: g.reset_index(drop=True) for t, g in df.groupby("tool")}

def split(df): return df[:NTR], df[NTR:NCAL], df[NCAL:]

# ---- Part B1: vendor_lookup -------------------------------------------------------------------------------------
INF = float("inf")
frac = lambda k, n: f"{k}/{n} = {k/n:.2%}" if n else f"{k}/0"

def vstats(P, k, b, nov, mt, cert=certify, cal_mask=None):
    """Certify model k on its calibration predictions at budget b, apply to the test stream. nov = test rows from merchants never in training; mt = true merchant of each test row."""
    cc, oc, ct, ot = P[k]; cm = np.ones(len(cc), bool) if cal_mask is None else cal_mask; c = cert(cc[cm], oc[cm], b); n_nov = int(nov.sum()); co = certify_orig(cc[cm], oc[cm], b)["coverage"]
    if np.isinf(c["cutoff"]): return dict(cutoff=INF, cal_cov=0., cal_cov_orig=co, bound=None, skipped=0., n_above=0, n_err=0, err=None, n_fs=0, n_nov=n_nov, fs=0., n_merch=0, max_merch=0)
    skip = ct >= c["cutoff"]; bad = skip & ~ot; cnt = Counter(mt[bad].tolist())
    return dict(cutoff=c["cutoff"], cal_cov=c["coverage"], cal_cov_orig=co, bound=c["bound"], skipped=skip.mean(), n_above=int(skip.sum()), n_err=int(bad.sum()), err=bad.sum() / max(skip.sum(), 1),
                n_fs=int((skip & nov).sum()), n_nov=n_nov, fs=(skip & nov).sum() / max(n_nov, 1), n_merch=len(cnt), max_merch=max(cnt.values(), default=0))

def part_vendor(V, env, out):
    V = V[~V.ret.str.startswith("ERROR")].reset_index(drop=True)  # labels only from successful calls
    tx = V.i // 3  # 3 tool calls per transaction
    tr, ca, te = V[tx < NTR], V[(tx >= NTR) & (tx < NCAL)], V[tx >= NCAL]
    desc = lambda d: d["args.descriptor"].tolist()
    known = set(tr.ret); nov_c = ~ca.ret.isin(known).values; nov = ~te.ret.isin(known).values; mt = te["hid.m"].values
    out.append(f"vendor_lookup rows (successful calls): train {len(tr)}, cal {len(ca)}, test {len(te)}; vendors in train {len(known)}; "
               f"novel-merchant share: cal {nov_c.mean():.3f}, test {nov.mean():.3f} (generation flag unseen share test {(te['hid.group'] != 'S').mean():.3f})\n")
    _, _, res = pick_text(desc(tr), tr.ret.values, desc(ca), ca.ret.values, BUDGET)  # fits LR and NN once
    P = {}
    for k in ("lr", "nn"):
        lc, cc = res[k][0](desc(ca)); lt, ct = res[k][0](desc(te)); P[k] = (cc, lc == ca.ret.values, ct, lt == te.ret.values)
    H = {k: {b: vstats(P, k, b, nov, mt) for b in (.005, .01, .02, .05)} for k in P}
    pick = lambda HH, b: max(P, key=lambda k: (HH[k][b]["cal_cov"], -(HH[k][b]["bound"] or 1.)))  # rule: higher certified calibration coverage (ties: smaller bound)
    kept = {b: pick(H, b) for b in H["lr"]}
    Ho = {k: {b: vstats(P, k, b, nov, mt, cert=certify_orig) for b in (.01, .02)} for k in P}   # as shipped: common.certify
    Hn = {k: {b: vstats(P, k, b, nov, mt, cal_mask=~nov_c) for b in (.01, .02)} for k in P}      # naive: grid certifier, calibration rows from known merchants only
    keptn = {b: pick(Hn, b) for b in (.01, .02)}; kepto = {b: pick(Ho, b) for b in (.01, .02)}
    best = kept[BUDGET]; cc, oc, ct, ot = P[best]
    acc = {k: (P[k][1].mean(), P[k][3].mean()) for k in P}
    rows = []
    for k in ("lr", "nn"):
        s = H[k][BUDGET]
        rows.append([k.upper() + (" (kept)" if k == best else ""), acc[k][0], acc[k][1], s["cutoff"], s["cal_cov"], s["bound"], s["skipped"], frac(s["n_err"], s["n_above"]), frac(s["n_fs"], s["n_nov"])])
    out.append(md(rows, ["model", "cal acc", "test acc (always answer)", "certified cutoff @1%", "cal coverage", "cal CP bound", "test calls skipped", "test errors above cutoff", "false-skip, novel merchants"]) + "\n")
    # naive vs deployment-like calibration for the kept predictor
    rows = []
    for name, mk in (("calibration includes novel merchants (deployment-like)", np.ones(len(ca), bool)), ("calibration on known merchants only (naive)", ~nov_c)):
        c = certify(cc[mk], oc[mk], BUDGET); skip = ct >= c["cutoff"]; err = int((~ot[skip]).sum()); ns = int((skip & nov).sum())
        rows.append([name, c["cutoff"], c["coverage"], skip.mean(), frac(err, int(skip.sum())), cp_upper(err, int(skip.sum()), .05) if skip.any() else None, frac(ns, int(nov.sum())), ns / max(skip.sum(), 1)])
    out.append(md(rows, ["calibration slice", "cutoff", "cal coverage", "test skipped", "test errors above cutoff", "95% upper (CP)", "false-skip novel", "novel share of skips"]) + "\n")
    c = certify(cc, oc, BUDGET); skip = ct >= c["cutoff"]
    rows = [[g, int(m.sum()), skip[m].mean(), frac(int((~ot[skip & m]).sum()), int((skip & m).sum())), ot[m].mean()] for g, m in (("known merchants", ~nov), ("novel merchants", nov))]
    out.append(md(rows, ["test slice (deployment-like cutoff, kept model)", "n", "share skipped", "errors among skipped", "accuracy if always answered"]) + "\n")
    rows = []
    for b in (.005, .01, .02, .05):
        for k in ("lr", "nn"):
            s = H[k][b]
            rows.append([b, k.upper() + (" (kept)" if k == kept[b] else ""), s["cutoff"], s["cal_cov"], s["cal_cov_orig"], s["skipped"], frac(s["n_err"], s["n_above"]), frac(s["n_fs"], s["n_nov"])])
    out.append(md(rows, ["error budget", "model", "cutoff", "cal coverage (grid certifier)", "cal coverage (original common.certify)", "test skipped", "test errors above cutoff", "false-skip novel"]) + "\n")
    L = te.latency_ms.values
    out.append(f"Latency (kept model, 1% budget): mean {L.mean():.0f} ms/call, skipping saves {L[skip].sum()/len(L):.0f} ms/call ({L[skip].sum()/L.sum():.1%} of total vendor_lookup time); "
               f"skipped calls include {(L[skip] > 1500).sum()} of {(L > 1500).sum()} tail (>1.5 s) calls.\n")
    pred = res[best][0]
    return pred, H[best][BUDGET]["cutoff"], tr, dict(H=H, Ho=Ho, Hn=Hn, kept=kept, keptn=keptn, kepto=kepto, acc={k: v[1] for k, v in acc.items()}, n_te=len(te))

# ---- Part B2: lookup_order speculation ---------------------------------------------------------------------------
def order_X(d):
    return np.c_[d["feat.age_days"], d["feat.channel"].map(CHANNELS.index), d["feat.amount"], d["feat.category"].map(CATS.index)]

HGB = dict(max_iter=300, learning_rate=.05, max_leaf_nodes=8, min_samples_leaf=40, l2_regularization=1., early_stopping=True, validation_fraction=.2, n_iter_no_change=15, random_state=0)

def part_order(O, env, out):
    O = O.reset_index(drop=True); tr, te = O[:NCAL], O[NCAL:]
    m = HistGradientBoostingClassifier(categorical_features=[1, 3], **HGB).fit(order_X(tr), tr.ret)
    L = te.latency_ms.values; base = (L + 150).mean(); maj = tr.ret.mode()[0]; y = te.ret.values
    orc = np.array([env.order_dist(a, c, m_, k) for a, c, m_, k in zip(te["feat.age_days"], te["feat.channel"], te["feat.amount"], te["feat.category"])])
    models = {"HGB": (m.predict_proba(order_X(te)), m.classes_), "oracle (true P given visible features)": (orc, np.array(STATUS))}
    out.append(f"lookup_order: train {len(tr)}, test {len(te)}; status mix test: " + ", ".join(f"{k} {v:.2f}" for k, v in te.ret.value_counts(normalize=True).items()) +
               f"; majority '{maj}' accuracy {(te.ret == maj).mean():.3f}; mean lookup latency {L.mean():.0f} ms; next step 150 ms\n")
    rows, rel = [], []
    for name, (P, cl) in models.items():
        top = P.max(1); hit = cl[P.argmax(1)] == y
        rows.append([name, hit.mean(), -np.log(P[np.arange(len(y)), [list(cl).index(v) for v in y]] + 1e-12).mean()])
    out.append(md(rows, ["status predictor", "top-1 accuracy", "mean log-loss"]) + "\n")
    for name, (P, cl) in models.items():
        top = P.max(1); hit = cl[P.argmax(1)] == y
        for tau in (0, .5, .7, .9):
            go = top > tau; saved = np.where(go & hit, np.minimum(L, 150), 0)
            rel.append([name, tau, go.mean(), hit[go].mean() if go.any() else None, int((go & ~hit).sum()), saved.mean(), saved.mean() / base, saved[go].mean() if go.any() else None])
    mh = (y == maj); saved = np.where(mh, np.minimum(L, 150), 0)
    rel.append([f"no model: always speculate '{maj}'", "-", 1.0, mh.mean(), int((~mh).sum()), saved.mean(), saved.mean() / base, saved.mean()])
    out.append(md(rel, ["predictor", "tau", "speculated (share of calls)", "hit rate when speculated", "wasted launches", "mean ms saved / call", "saved / mean (L+150)", "ms saved per launch"]) + "\n")
    sv = {(r[0], r[1]): r[5] for r in rel}; b0 = sv[(f"no model: always speculate '{maj}'", "-")]; top9 = int((models["HGB"][0].max(1) > .9).sum())
    out.append(f"Comparison with the no-model constant (a wasted read-only launch is free here, so tau has no latency value; it only removes wasted launches): always speculating '{maj}' saves {b0:.1f} ms/call; "
               "HGB saves " + ", ".join(f"{sv[('HGB', t)]:.1f} at tau {t} ({sv[('HGB', t)] - b0:+.1f})" for t in (0, .5, .7, .9)) +
               f". Only tau 0 beats the constant, by {sv[('HGB', 0)] - b0:.1f} ms = {(sv[('HGB', 0)] - b0) / base:.1%} of the {base:.0f} ms sequential time. The tau 0.9 hit rate of 1.0 rests on n={top9} calls.\n")
    P, cl = models["HGB"]; top = P.max(1); hit = cl[P.argmax(1)] == y
    rows = [[f"{lo:.1f}-{hi:.1f}", int(((top > lo) & (top <= hi)).sum()), hit[(top > lo) & (top <= hi)].mean(), top[(top > lo) & (top <= hi)].mean()]
            for lo, hi in ((0, .5), (.5, .7), (.7, .9), (.9, 1.01)) if ((top > lo) & (top <= hi)).any()]
    out.append("HGB calibration of the top probability:\n\n" + md(rows, ["top prob bin", "n", "actual hit rate", "mean top prob"]) + "\n")

# ---- Part B3: issue_refund -------------------------------------------------------------------------------------
def refund_X(d):  # amount, category, hour, recent error rate (30 calls), previous call failed, error rate of the previous 5 calls
    return np.c_[d["feat.amount"], d["feat.category"].map(CATS.index), d["feat.hour"], d["recent_error_rate"], d["prev_err"], d["last5"]]

def part_refund(R, out):
    R = R.reset_index(drop=True); e = (R.ret != "ok").astype(float); R["prev_err"] = e.shift(1).fillna(0); R["last5"] = e.shift(1).rolling(5, min_periods=1).mean().fillna(0)
    tr, te = R[:NCAL], R[NCAL:]
    y = (te.ret != "ok").values; ytr = (tr.ret != "ok").values
    m = HistGradientBoostingClassifier(categorical_features=[1], **HGB).fit(refund_X(tr), tr.ret)
    P = m.predict_proba(refund_X(te)); pe = 1 - P[:, list(m.classes_).index("ok")]
    rows = [["HGB (amount, category, hour, recent error rate, last outcomes)", roc_auc_score(y, pe), brier_score_loss(y, pe)],
            ["base rate only", .5, brier_score_loss(y, np.full(len(y), ytr.mean()))],
            ["recent_error_rate alone (raw feature)", roc_auc_score(y, te.recent_error_rate), None],
            ["oracle: env's true P(error), incl. hidden regime", roc_auc_score(y, te["hid.p_err"]), brier_score_loss(y, te["hid.p_err"])]]
    out.append(f"issue_refund: train {len(tr)}, test {len(te)}; error rate train {ytr.mean():.3f}, test {y.mean():.3f}; outcomes test: " +
               ", ".join(f"{k} {v:.3f}" for k, v in te.ret.value_counts(normalize=True).items()) + "\n")
    out.append(md(rows, ["P(error) predictor", "AUC-ROC", "Brier"]) + "\n")
    # fallback-first rule
    rows = []
    for thr in (.5, .35, .2):
        f = pe > thr; tp = int((f & y).sum()); nf = int(f.sum())
        lo, hi = (beta.ppf(.025, tp, nf - tp + 1) if tp else 0., beta.ppf(.975, tp + 1, nf - tp) if nf else 1.)  # Clopper-Pearson 95% interval on precision
        rows.append([thr, nf, f.mean(), tp / max(nf, 1), f"[{lo:.2f}, {hi:.2f}]", tp / y.sum(), y.mean(), (f & (te.ret == "timeout").values).sum() / (te.ret == "timeout").sum()])
    out.append(md(rows, ["predicted P(error) >", "calls flagged (of 1200)", "share flagged", "precision", "precision 95% CI", "recall (any error)", "base error rate", "recall (timeouts)"]) + "\n")
    errs = te.ret.values != "ok"; ec = [c for c in m.classes_ if c != "ok"]
    top_err = np.array(ec)[P[:, [list(m.classes_).index(c) for c in ec]].argmax(1)]
    out.append(f"Most likely error type, given an error occurred: accuracy {(top_err[errs] == te.ret.values[errs]).mean():.3f} "
               f"(always-'{pd.Series(tr.ret[tr.ret != 'ok']).mode()[0]}': {(te.ret.values[errs] == pd.Series(tr.ret[tr.ret != 'ok']).mode()[0]).mean():.3f})\n")
    # latency regression
    r = HistGradientBoostingRegressor(loss="absolute_error", categorical_features=[1], **HGB).fit(refund_X(tr), tr.latency_ms)
    pl = r.predict(refund_X(te)); med = tr.latency_ms.median(); nt = (te.ret != "timeout").values
    rows = [["HGB regressor (abs. error loss)", mean_absolute_error(te.latency_ms, pl), mean_absolute_error(te.latency_ms[nt], pl[nt])],
            [f"constant train median ({med:.0f} ms)", mean_absolute_error(te.latency_ms, np.full(len(te), med)), mean_absolute_error(te.latency_ms[nt], np.full(nt.sum(), med))]]
    out.append(md(rows, ["latency predictor", "MAE ms (all test calls)", "MAE ms (excluding timeouts)"]) + "\n")

# ---- Part B4: outage test --------------------------------------------------------------------------------------
def part_outage(env, pred, cutoff, tr, out, n=1000, seed=1):
    r = np.random.default_rng(seed)
    env.outage = True; txs = [env.new_txn("test") for _ in range(n)]  # NB: drawn from the same 30 unseen test merchants as the test stream (not independent of it; the cutoff was not tuned on it)
    descs = [t["descriptor"] for t in txs]; truth = np.array([env.truth_vendor(d) for d in descs])
    # baseline: circuit breaker (opens after 5 consecutive failures; half-open probes would also fail during a full outage)
    fails, is_open, done, called = 0, False, 0, 0
    for d in descs:
        if is_open: continue
        called += 1; ret, _ = env.vendor_lookup(d)
        if ret.startswith("ERROR"): fails += 1; is_open = fails >= 5
        else: fails = 0; done += 1
    # registered fallback: stale snapshot of 60% of merchants, 3% of its entries point at the wrong vendor
    nm = len(env.m); covered = set(r.choice(nm, int(.6 * nm), replace=False).tolist())
    stale = {i: env.vid[(i + 7) % nm] if r.random() < .03 else env.vid[i] for i in covered}
    lab, conf = pred(descs); skip = conf >= cutoff
    # how the snapshot resolves a raw descriptor: (a) idealised = the env's hidden descriptor->merchant map (perfect resolution, an UPPER bound);
    # (b) realistic = exact match on the normalised (lowercase, digits->0) descriptors seen in the training window for covered merchants
    norm = lambda s: re.sub(r"\d", "0", s.lower())
    key = {norm(d): stale[env.desc2m[d]] for d in tr["args.descriptor"] if env.desc2m[d] in stale}
    snaps = {"idealised resolution": np.array([stale.get(env.desc2m[d]) for d in descs], dtype=object), "normalised-descriptor match": np.array([key.get(norm(d)) for d in descs], dtype=object)}
    rows = [["circuit breaker (baseline)", "-", done / n, None, done], ["certified predictor only (rest incomplete)", "-", None, None, None]]
    pr = skip; wrong_p = lab != truth
    rows[1][2:] = [pr.mean(), (wrong_p & pr).sum() / max(pr.sum(), 1), int(pr.sum())]
    for nm_, snap in snaps.items():
        has = np.array([s is not None for s in snap]); fb = ~skip & has; wrong_s = np.array([s_ != t for s_, t in zip(snap, truth)])
        rows += [["registered fallback only (stale snapshot)", nm_, has.mean(), (wrong_s & has).sum() / max(has.sum(), 1), int(has.sum())],
                 ["**Dreaming: certified predictor, else fallback**", nm_, (pr | fb).mean(), ((wrong_p & pr) | (wrong_s & fb)).sum() / max((pr | fb).sum(), 1), int((pr | fb).sum())],
                 ["   answered by fallback (rows tagged for downstream gates)", nm_, fb.mean(), (wrong_s & fb).sum() / max(fb.sum(), 1), int(fb.sum())]]
    novel = ~np.isin(truth, list(set(tr.ret)))
    out.append(f"Outage: {n} transactions, vendor_lookup 100% errors; breaker made {called} real calls before opening; cutoff {cutoff:.3f}; "
               f"{novel.mean():.3f} of the transactions are from merchants novel to the predictor ({(skip & novel).sum()} of them were confidently predicted, all wrong). "
               "The idealised-resolution rows are upper bounds; the outage stream shares its 30 unseen merchants with the test stream.\n")
    out.append(md(rows, ["strategy", "snapshot resolves descriptors by", "completed", "error rate among completed", "n completed"]) + "\n")

def main():
    selfcheck(); t0 = time.time(); out = []
    env = Env(seed=4, trace_path=HERE / "traces.jsonl"); simulate(env)
    T = load(HERE / "traces.jsonl")
    out.append(f"traces.jsonl: {sum(len(v) for v in T.values())} rows; per tool " + ", ".join(f"{k} {len(v)}" for k, v in T.items()) +
               f"; vendor_lookup error rate {T['vendor_lookup'].ret.str.startswith('ERROR').mean():.3f}\n")
    tick = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", file=sys.stderr, flush=True)
    out.append("## vendor_lookup\n"); pred, cutoff, tr, h4 = part_vendor(T["vendor_lookup"], env, out); tick("vendor done")
    out.append("## lookup_order\n"); part_order(T["lookup_order"], env, out); tick("order done")
    out.append("## issue_refund\n"); part_refund(T["issue_refund"], out); tick("refund done")
    out.append("## outage\n"); part_outage(env, pred, cutoff, tr, out); tick("outage done")
    out.append("## certificate robustness: main run (seed 4) plus 5 other env seeds (new merchants, templates, streams)\n")
    W = {4: h4}
    for sd in (11, 12, 13, 14, 15):
        e2 = Env(seed=sd, trace_path=HERE / f"traces_seed{sd}.jsonl"); simulate(e2)
        W[sd] = part_vendor(load(HERE / f"traces_seed{sd}.jsonl")["vendor_lookup"], e2, [])[3]; os.remove(HERE / f"traces_seed{sd}.jsonl"); tick(f"seed {sd} done")
    for b in (.01, .02):
        rows = []
        for sd, h in W.items():
            for k in ("lr", "nn"):
                s = h["H"][k][b]
                rows.append([sd, k.upper() + (" (kept)" if k == h["kept"][b] else ""), h["acc"][k], s["cutoff"], s["cal_cov"], s["cal_cov_orig"], s["skipped"], frac(s["n_err"], s["n_above"]),
                             frac(s["n_fs"], s["n_nov"]), f"{s['n_merch']} (max {s['max_merch']})"])
            kn = h["keptn"][b]; s = h["Hn"][kn][b]   # same model family, but calibrated on the known-merchant calibration rows only
            rows.append([sd, f"{kn.upper()} naive (cal rows from known merchants only)", h["acc"][kn], s["cutoff"], s["cal_cov"], s["cal_cov_orig"], s["skipped"], frac(s["n_err"], s["n_above"]),
                         frac(s["n_fs"], s["n_nov"]), f"{s['n_merch']} (max {s['max_merch']})"])
        out.append(f"### Per world, error budget {b:.0%}\n\n" + md(rows, ["env seed", "model", "test acc (always answer)", "cutoff", "cal coverage (grid)", "cal coverage (original certify)", "test skipped",
                   "test errors above cutoff", "false-skip novel", "distinct merchants behind the errors (max from one)"]) + "\n")
        rows = []
        for name, get in (("shipped: original certify, kept by coverage", lambda h: h["Ho"][h["kepto"][b]][b]), ("grid, LR", lambda h: h["H"]["lr"][b]), ("grid, NN", lambda h: h["H"]["nn"][b]),
                          ("grid, kept by coverage", lambda h: h["H"][h["kept"][b]][b]), ("grid, kept, calibrated on known merchants only (naive)", lambda h: h["Hn"][h["keptn"][b]][b])):
            S = [(sd, get(h)) for sd, h in W.items()]; ea, na, fk, nn_ = (sum(s[x] for _, s in S) for x in ("n_err", "n_above", "n_fs", "n_nov")); ns = sum(h["n_te"] for h in W.values())
            rows.append([name, frac(ea, na), frac(fk, nn_), na / ns, ", ".join(str(sd) for sd, s in S if s["err"] is not None and s["err"] > b) or "none", ", ".join(str(sd) for sd, s in S if np.isinf(s["cutoff"])) or "none"])
        out.append(f"Pooled over the 6 worlds, budget {b:.0%}:\n\n" + md(rows, ["certifier / selection", "errors above cutoff (pooled)", "false-skip novel (pooled)", "share of test skipped (pooled)",
                   f"env seeds whose test error exceeds {b:.0%}", "env seeds with nothing certifiable"]) + "\n")
    # novelty shift: certificate fixed on the ~20%-novel calibration slice, test stream re-weighted to a different novel-merchant share p (per-group rates from the existing test stream; no new simulation).
    # A novel merchant's skip is always a wrong answer (closed-set predictor), so errors above cutoff = skipped novel + wrong skipped known.
    for b in (.01, .02):
        rows = []
        for p in (.1, .2, .3, .4, .5):
            em = sm = 0.; over = []
            for sd, h in W.items():
                s = h["H"][h["kept"][b]][b]; nk = h["n_te"] - s["n_nov"]; sk = (s["n_above"] - s["n_fs"]) / nk; er = (s["n_err"] - s["n_fs"]) / max(s["n_above"] - s["n_fs"], 1); sn = s["n_fs"] / s["n_nov"]
                skip = (1 - p) * sk + p * sn; err = ((1 - p) * sk * er + p * sn) / skip; em += err * skip; sm += skip
                if np.isfinite(s["cutoff"]) and err > b: over.append(sd)
            rows.append([f"{p:.0%}", em / sm, sm / len(W), ", ".join(map(str, over)) or "none"])
        out.append(f"### Novelty shift, kept model, budget {b:.0%}: certificate from the ~20%-novel calibration slice, test stream re-weighted to novel share p (pooled over 6 worlds)\n\n" +
                   md(rows, ["novel share of test", "pooled error above cutoff", "pooled share skipped", f"env seeds exceeding {b:.0%}"]) + "\n")
    print("\n".join(out)); print(f"\nwall seconds: {time.time()-t0:.1f}")

if __name__ == "__main__":
    main()
