"""S21 analysis (runs on the pod, CPU). python analyze.py all|s1|rb|s2 [model_tag]   (stdout -> log.txt)
s1: Banking77, cheap local model vs E1 on the S5 200 (+ proxy target cheap==gold on 1000 train rows). rb: RouterBench 2000.
s2: stage 2 (E1 on 1000 new rows via OpenRouter, cached in stage2_e1.jsonl). `all` runs s1, rb, and s2 only if the pre-stated stage-2 bar is met.
Env: S21_TINY=N local pipeline check on the first N rows of every set; S21_SUF / S21_FEATDIR locate results<SUF>.json + feats_<tag><SUF>.npz; S21_MOCK_E1=1 fakes E1.
ponytail: one file; sklearn LR probes with nested (layer, C) selection inside every training fold; certificates from common.certify (fixed-grid Clopper-Pearson)."""
import sys, os, json, warnings, time, threading, importlib.util
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")  # pod reports 100+ cores: one BLAS thread per joblib worker
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent; SM = HERE.parent
sys.path.insert(0, str(SM))
from common import *  # noqa
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
_sp = importlib.util.spec_from_file_location("s5run", SM / "s5_tier_router" / "run.py"); s5 = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(s5)  # S5's parse / pair_data / path helpers
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
NT = int(os.environ.get("S21_TINY", 0)); SUF = os.environ.get("S21_SUF", "_gpu"); FEATDIR = Path(os.environ.get("S21_FEATDIR", HERE)); MOCK = bool(os.environ.get("S21_MOCK_E1"))
NJ = 1 if NT else 8; BOOT = 40 if NT else 600; RC1 = 8 if NT else 300; RC2 = 8 if NT else 200  # bootstrap resamples; random calibration/holdout splits
T = json.loads((HERE / "targets.json").read_text(encoding="utf-8")); names = T["names"]
DATA = json.loads((HERE / "data.json").read_text(encoding="utf-8"))
if NT:  # local pipeline check: first NT rows of everything
    for k in ("gold200", "e1", "e1b", "tr_gold", "new_gold", "new_idx", "tr_idx"): T[k] = T[k][:NT]
    T["rb"] = {k: v[:NT] for k, v in T["rb"].items()}
    for k in ("b200", "b1000tr", "b1000new", "rb2000"): DATA[k] = DATA[k][:NT]
CS = (1e-3, 1e-2, 1e-1, 1.0)
BUD = (.02, .05, .10, .20)

def load_model(tag):
    R = [r for r in json.loads((FEATDIR / f"results{SUF}.json").read_text(encoding="utf-8")) if r["model"].endswith(tag)][0]
    F = dict(np.load(FEATDIR / f"feats_{tag}{SUF}.npz"))
    if NT:
        for s in R["sets"]: R["sets"][s]["gen"] = R["sets"][s]["gen"][:NT]
        F = {k: v[:NT] for k, v in F.items()}
    return R, F

# ---------------------------------------------------------------- generic pieces
def auc(y, s): y = np.asarray(y, int); return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan

def lr_fit(X, y, C):
    sc = StandardScaler().fit(X); return sc, LogisticRegression(C=C, max_iter=2000).fit(sc.transform(X), y)

def _cv_auc(X, y, C, splits):
    oof = np.zeros(len(y))
    for tr, te in splits: sc, m = lr_fit(X[tr], y[tr], C); oof[te] = m.decision_function(sc.transform(X[te]))
    return auc(y, oof)

def select_fit(Xs, y, seed=0):
    """Xs: candidate feature matrices (one per layer). Choose (layer, C) by 3-fold inner AUROC on the training rows only, refit on all of them.
    Returns predict(list_of_test_matrices) -> decision values, and (inner auc, layer index, C)."""
    splits = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(Xs[0], y)); grid = [(li, C) for li in range(len(Xs)) for C in CS]
    a = np.nan_to_num(np.array(Parallel(n_jobs=NJ)(delayed(_cv_auc)(Xs[li], y, C, splits) for li, C in grid)), nan=-1.0)
    k = int(a.argmax()); li, C = grid[k]; sc, m = lr_fit(Xs[li], y, C)
    return (lambda Xt: m.decision_function(sc.transform(Xt[li]))), (float(a[k]), li, C)

def cv_oof(Xs, y, seed=0, k=5):
    """Out-of-fold decision values over k stratified folds, nested (layer, C) selection inside every training fold."""
    oof = np.zeros(len(y)); chosen = []
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(Xs[0], y):
        f, best = select_fit([X[tr] for X in Xs], y[tr], seed); oof[te] = f([X[te] for X in Xs]); chosen.append(best[1:])
    return oof, chosen

def boot_auc(y, scores, refs=None, B=BOOT, seed=0):
    """Bootstrap over rows. Returns {name: (auc, lo, hi)} and {name: (auc - auc(ref), lo, hi)} for names in refs (paired resamples)."""
    y = np.asarray(y, int); n = len(y); rng = np.random.default_rng(seed); ix = [rng.integers(0, n, n) for _ in range(B)]
    ix = [i for i in ix if 0 < y[i].sum() < n]; out = {}; diff = {}
    for k, s in scores.items():
        b = np.array([auc(y[i], s[i]) for i in ix]); out[k] = (auc(y, s), *np.percentile(b, [2.5, 97.5]))
        if refs and k in refs:
            d = np.array([auc(y[i], s[i]) - auc(y[i], scores[refs[k]][i]) for i in ix]); diff[k] = (auc(y, s) - auc(y, scores[refs[k]]), *np.percentile(d, [2.5, 97.5]))
    return out, diff

def cert_table(scores, correct, R=200, seed=0, budgets=BUD):
    """Clean-split certificate: random half = calibration (certify), other half = realised. Scores must not have been fit on these rows' labels.
    cell = (mean realised coverage, share of splits that certified, pooled realised error, share of certified splits whose holdout error exceeded the budget)."""
    n = len(correct); rng = np.random.default_rng(seed); perms = [rng.permutation(n) for _ in range(R)]; rows = {}
    for k, s in scores.items():
        for b in budgets:
            cov = []; errs = cnt = viol = cert = 0
            for p in perms:
                cal, ho = p[: n // 2], p[n // 2:]; c = certify(s[cal], correct[cal], b)
                if c["coverage"] == 0: cov.append(0.0); continue
                h = apply_certificate(s[ho], correct[ho], c["cutoff"]); cov.append(h["coverage"]); cert += 1
                if h["n_above"]: errs += h["error"] * h["n_above"]; cnt += h["n_above"]; viol += h["error"] > b
            rows[(k, b)] = (float(np.mean(cov)), cert / R, errs / cnt if cnt else float("nan"), viol / max(cert, 1))
    return rows

def print_cert(rows, scores, title, budgets=BUD):
    print(f"\n{title}\ncell = mean realised coverage % / share of splits that certified / pooled realised error % / share of certified splits whose holdout error exceeded the budget")
    print("| score | " + " | ".join(f"budget {int(b*100)}%" for b in budgets) + " |"); print("|---|" + "---|" * len(budgets))
    for k in scores:
        print(f"| {k} | " + " | ".join("%.1f / %.2f / %s / %.2f" % (100 * rows[(k, b)][0], rows[(k, b)][1], "-" if np.isnan(rows[(k, b)][2]) else "%.1f" % (100 * rows[(k, b)][2]), rows[(k, b)][3]) for b in budgets) + " |")

def print_auc(title, o, dd, ref="embedding LR"):
    print(title); print(f"| score | AUROC [95% CI] | diff vs {ref} [95% CI] |"); print("|---|---|---|")
    for k in o: print("| %s | %.3f [%.3f, %.3f] | %s |" % (k, *o[k], "%+.3f [%+.3f, %+.3f]" % dd[k] if k in dd else "-"))

# ---------------------------------------------------------------- Banking77 tables
def exact(t):  # strict label parse: anything that is not exactly a label (case-insensitive, stripped) is None = counted as a mismatch
    l, pf = s5.parse(t, names); return None if pf else l

def gen_table(R, sname):
    G = R["sets"][sname]["gen"]; out = {}
    lab = [s5.parse(g["text"], names) for g in G]
    out["ans_exact"] = np.array([None if pf else l for l, pf in lab], dtype=object); out["ans"] = out["ans_fz"] = np.array([l for l, _ in lab], dtype=object);  # ans = nearest-label parse as S5/S15 (revision; the first version scored non-exact answers as wrong)
    out["pf"] = np.array([pf for _, pf in lab])
    sm = [[exact(t) if t.strip() else None for t in g["samples"]] for g in G]
    lp = [np.array(g["lp"]) for g in G]; gap = [np.array(g["gap"]) for g in G]
    out["lp_seq"] = np.array([l.sum() + (g["eos_lp"] or 0.0) for l, g in zip(lp, G)])
    out["lp_mean"] = np.array([l.mean() if len(l) else -20.0 for l in lp])
    out["gap_first"] = np.array([x[0] if len(x) else 0.0 for x in gap]); out["gap_min"] = np.array([x.min() if len(x) else 0.0 for x in gap])
    ok = lambda u, v: float(u is not None and u == v)
    out["sc"] = np.array([ok(s[0], s[1]) + ok(s[0], a) + ok(s[1], a) for s, a in zip(sm, out["ans"])])
    out["sc_tb"] = out["sc"] + 1e-3 * (out["lp_seq"] - out["lp_seq"].min()) / (np.ptp(out["lp_seq"]) + 1e-9)  # self-consistency, ties broken by sequence logprob
    return out

SIG = ("lp_seq", "lp_mean", "gap_first", "gap_min", "sc")
def sigX(g): return np.c_[tuple(g[k] for k in SIG)]
def layers_of(F, sname, pool): A = F[f"{sname}_{pool}"].astype(np.float32); return [A[:, i, :] for i in range(A.shape[1])]

def b77(tag, pools=("last", "mean")):
    R, F = load_model(tag); L = R["layers"]
    S = {s: gen_table(R, s) for s in ("b200", "b1000tr", "b1000new")}
    gold = {"b200": np.array(T["gold200"], dtype=object), "b1000tr": np.array(T["tr_gold"], dtype=object), "b1000new": np.array(T["new_gold"], dtype=object)}
    e1 = np.array(T["e1"], dtype=object); e1b = np.array(T["e1b"], dtype=object)
    print(f"\n==================== {tag}: Banking77 ====================\nsaved layers {L} of {R['n_layers']}; batch-1 latency {R.get('batch1_latency_ms')} ms; throughput " +
          ", ".join(f"{s} {R['sets'][s]['n']/R['sets'][s]['seconds']:.1f}/s" for s in R["sets"]))
    for s in S:
        g = S[s]
        print(f"[{s}] cheap non-exact-label rate {g['pf'].mean():.3f} (counted as wrong/mismatch), exact-label accuracy vs gold {(g['ans'] == gold[s]).mean():.3f} (fuzzy-matched {(g['ans_fz'] == gold[s]).mean():.3f}), self-consistency 3-of-3 {np.mean(g['sc'] == 3):.3f}" +
              (f", agree with E1 {np.mean(g['ans'] == e1):.3f} (fuzzy {np.mean(g['ans_fz'] == e1):.3f}; E1 gold acc {np.mean(e1 == gold[s]):.3f}; noise floor: E1 vs E1b match {np.mean(e1 == e1b):.3f}, cheap vs E1b {np.mean(g['ans'] == e1b):.3f})" if s == "b200" else ""))
    y_e1 = (S["b200"]["ans"] == e1).astype(int); y_e1b = (S["b200"]["ans"] == e1b).astype(int); y_g200 = (S["b200"]["ans"] == gold["b200"]).astype(int)
    y_gtr = (S["b1000tr"]["ans"] == gold["b1000tr"]).astype(int); y_gnew = (S["b1000new"]["ans"] == gold["b1000new"]).astype(int)
    print(f"targets: cheap==E1 on 200: {y_e1.mean():.3f}; cheap==E1b {y_e1b.mean():.3f}; cheap==gold on 200 {y_g200.mean():.3f}, 1000train {y_gtr.mean():.3f}, 1000new {y_gnew.mean():.3f}")
    E = {s: embed(DATA[s]) for s in S}
    raw = {k: S["b200"][k] for k in ("lp_seq", "lp_mean", "gap_first", "gap_min", "sc", "sc_tb")}
    res = {}
    for pool in pools:
        X = {s: layers_of(F, s, pool) for s in S}
        # ---- setting P: train on the 1000 train rows with the proxy target (cheap == gold); test on the 200 with the real target (cheap == E1)
        t0 = time.time(); f, best = select_fit(X["b1000tr"], y_gtr); pP = f(X["b200"])
        print(f"\n[{pool}] proxy probe: inner AUC {best[0]:.3f} layer {L[best[1]]} C {best[2]} ({time.time()-t0:.0f}s)")
        fe, _ = select_fit([E["b1000tr"]], y_gtr); eP = fe([E["b200"]])
        fs, _ = select_fit([sigX(S["b1000tr"])], y_gtr); sP = fs([sigX(S["b200"])])
        oof_tr = np.zeros(len(y_gtr))  # cross-fitted probe score on the training rows (layer/C fixed from the selection above) for the probe+signals stack
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X["b1000tr"][0], y_gtr):
            sc_, m_ = lr_fit(X["b1000tr"][best[1]][tr], y_gtr[tr], best[2]); oof_tr[te] = m_.decision_function(sc_.transform(X["b1000tr"][best[1]][te]))
        fst, _ = select_fit([np.c_[oof_tr, sigX(S["b1000tr"])]], y_gtr); stP = fst([np.c_[pP, sigX(S["b200"])]])
        # ---- setting C: 5-fold CV on the 200 with the real target
        pC, chC = cv_oof(X["b200"], y_e1); eC, _ = cv_oof([E["b200"]], y_e1); sC, _ = cv_oof([sigX(S["b200"])], y_e1)
        print(f"[{pool}] CV-on-200 chosen (layer, C) per fold: {[(L[a], c) for a, c in chC]}")
        rnd = np.random.default_rng(0).random(len(y_e1))
        sc_all = {"probe (proxy-trained on 1000 train)": pP, "probe (5-fold on the 200)": pC, "embedding LR (proxy-trained)": eP, "embedding LR (5-fold on the 200)": eC,
                  "signals LR (proxy-trained)": sP, "signals LR (5-fold on the 200)": sC, "probe+signals stack (proxy)": stP,
                  "logprob: sequence": raw["lp_seq"], "logprob: mean token": raw["lp_mean"], "logprob: first-token gap": raw["gap_first"], "logprob: min token gap": raw["gap_min"],
                  "self-consistency (3 votes)": raw["sc"], "self-consistency + logprob tiebreak": raw["sc_tb"], "random (random mixing)": rnd}
        refs = {k: ("embedding LR (5-fold on the 200)" if "5-fold" in k else "embedding LR (proxy-trained)") for k in sc_all if not k.startswith("embedding")}
        res[pool] = {}
        for lab, y in (("cheap==E1 (REAL target)", y_e1), ("cheap==E1b (second E1 draw: target-noise check)", y_e1b), ("cheap==gold (200)", y_g200)):
            o, dd = boot_auc(y, sc_all, refs); print_auc(f"\n[{pool}] AUROC for {lab}, n={len(y)}, base rate {y.mean():.3f}; bootstrap 95% CI; diff vs the embedding LR trained the same way (paired)", o, dd)
            if lab.startswith("cheap==E1 "): res[pool].update(auc=o, diff=dd)
        clean = {"probe (proxy-trained)": pP, "embedding LR (proxy-trained)": eP, "signals LR (proxy-trained)": sP, "probe+signals stack (proxy)": stP, "logprob: sequence": raw["lp_seq"],
                 "logprob: min token gap": raw["gap_min"], "self-consistency (3 votes)": raw["sc"], "self-consistency + logprob tiebreak": raw["sc_tb"], "random": rnd}
        ct = cert_table(clean, y_e1.astype(bool), R=RC1); print_cert(ct, clean, f"[{pool}] CLEAN certificate vs E1 (cheap==E1), {RC1} random 100/100 splits of the 200 (100 calibration rows cannot certify a <=5% budget with the 40-level grid: need ~134 clean rows)")
        res[pool]["cert"] = ct
        desc = {"probe (5-fold on the 200)": pC, "embedding LR (5-fold on the 200)": eC, "signals LR (5-fold on the 200)": sC, "random": rnd, "self-consistency": raw["sc"], "logprob: sequence": raw["lp_seq"]}
        print(f"[{pool}] DESCRIPTIVE (no holdout; cutoff chosen on all 200 rows; OOF scores) certified coverage % at budgets {[int(100*b) for b in BUD]}: " +
              "; ".join(f"{k}: " + "/".join("%.0f" % (100 * certify(v, y_e1.astype(bool), b)["coverage"]) for b in BUD) for k, v in desc.items()))
        # ---- larger gold-target pool: 200 + 1000 new test rows (every score is out-of-sample for the probe, which saw only the 1000 train rows)
        pool_y = np.r_[y_g200, y_gnew]
        p_new = f(X["b1000new"]); pp = np.r_[pP, p_new]; pe = np.r_[eP, fe([E["b1000new"]])]; ps = np.r_[sP, fs([sigX(S["b1000new"])])]
        pst = np.r_[stP, fst([np.c_[p_new, sigX(S["b1000new"])]])]
        rr = {k: np.r_[S["b200"][k], S["b1000new"][k]] for k in ("lp_seq", "gap_min", "sc", "sc_tb")}
        big = {"probe (proxy-trained)": pp, "embedding LR (proxy-trained)": pe, "signals LR (proxy-trained)": ps, "probe+signals stack": pst, "logprob: sequence": rr["lp_seq"],
               "logprob: min token gap": rr["gap_min"], "self-consistency (3 votes)": rr["sc"], "self-consistency + logprob tiebreak": rr["sc_tb"], "random": np.random.default_rng(1).random(len(pool_y))}
        o2, d2 = boot_auc(pool_y, big, {k: "embedding LR (proxy-trained)" for k in big if not k.startswith("embedding")})
        print_auc(f"\n[{pool}] AUROC for cheap==gold on {len(pool_y)} rows (200 + 1000 new test rows; probe never saw them), base rate {pool_y.mean():.3f}", o2, d2)
        ct2 = cert_table(big, pool_y.astype(bool), R=RC2, seed=2); print_cert(ct2, big, f"[{pool}] CLEAN certificate vs GOLD (cheap==gold) on {len(pool_y)} rows, {RC2} random half/half splits")
        res[pool].update(big_auc=o2, big_diff=d2, big_cert=ct2)
    # ---- the pre-stated bar for stage 2 (last-token probe)
    r = res["last"]; k1, k2, k3 = "probe (proxy-trained on 1000 train)", "probe (5-fold on the 200)", "probe (proxy-trained)"
    dP, dC = r["diff"][k1], r["diff"][k2]; cp, ce_, cr = r["cert"][(k3, .10)][0], r["cert"][("embedding LR (proxy-trained)", .10)][0], r["cert"][("random", .10)][0]
    a_ok = bool(dP[0] >= .05 and dP[1] > 0); b_ok = bool(dC[0] > 0); r_ok = bool(r["auc"][k1][1] > .5)
    print("\n==== STAGE-2 BAR (last-token probe, target cheap==E1; fixed before the run) ====")
    print(f"A. proxy-trained probe minus embedding LR AUROC {dP[0]:+.3f} [{dP[1]:+.3f}, {dP[2]:+.3f}]  -> {'PASS' if a_ok else 'FAIL'} (need >= +0.05 and CI above 0)")
    print(f"B. 5-fold probe minus 5-fold embedding LR AUROC {dC[0]:+.3f} [{dC[1]:+.3f}, {dC[2]:+.3f}]  -> {'PASS' if b_ok else 'FAIL'} (need point estimate > 0)")
    print(f"R. proxy-trained probe AUROC {r['auc'][k1][0]:.3f} [{r['auc'][k1][1]:.3f}, {r['auc'][k1][2]:.3f}] vs random mixing 0.5 -> {'PASS' if r_ok else 'FAIL'} (need CI lower end > 0.5)")
    print(f"C (informational only): clean certified coverage at 10% budget: probe {100*cp:.1f}% vs embedding {100*ce_:.1f}% vs random {100*cr:.1f}%")
    res["bar"] = a_ok and b_ok and r_ok; print(f"STAGE-2 BAR (A and B and R): {'MET' if res['bar'] else 'NOT MET'}")
    return res

# ---------------------------------------------------------------- RouterBench
def rb(tag, pools=("last", "mean")):
    R, F = load_model(tag); L = R["layers"]; B = T["rb"]
    sc, se, cc, ce = (np.array(B[k]) for k in ("sc", "se", "cc", "ce")); fam = np.array(B["fam"]); n = len(sc)
    D = s5.pair_data(sc, se, cc, ce, fam); y1, y2 = D["y1"], D["y2"]
    print(f"\n==================== {tag}: RouterBench gpt-3.5 vs gpt-4, n={n} ====================\nbase rates: gpt-3.5 correct (y2) {y2.mean():.3f}; cheap>=gpt-4 (y1) {y1.mean():.3f}; mean acc gpt-4 {se.mean():.3f} gpt-3.5 {sc.mean():.3f}")
    Xm = embed(DATA["rb2000"]); X4 = embed(B["prompt400"])  # the matched head+tail 1200-char text the probe saw, and S5's features (bge-small, prompt[:400])
    scores = {}
    for tgt, y in (("y2 gpt-3.5 correct", y2), ("y1 gpt-3.5 >= gpt-4", y1)):
        sc_t = {}; sel = {}
        for pool in pools:
            oof, ch = cv_oof(layers_of(F, "rb2000", pool), y); sc_t[f"probe ({pool}-token states)"] = oof; sel[pool] = [(L[a], c) for a, c in ch]
        sc_t["embedding LR (400 chars, S5 features)"], _ = cv_oof([X4], y); sc_t["embedding LR (matched 1200-char text)"], _ = cv_oof([Xm], y)
        fp = np.zeros(n)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(n), y): fp[te] = pd.Series(fam[te]).map(pd.Series(y[tr]).groupby(fam[tr]).mean()).fillna(y[tr].mean()).values
        sc_t["benchmark-family prior"] = fp; sc_t["random"] = np.random.default_rng(3).random(n)
        scores[tgt] = sc_t
        o, dd = boot_auc(y, sc_t, {k: "embedding LR (400 chars, S5 features)" for k in sc_t if k.startswith(("probe", "embedding LR (matched"))})
        print_auc(f"\nAUROC for {tgt} (5-fold OOF, nested layer/C selection), bootstrap 95% CI; diff vs the S5 embedding LR (400 chars)", o, dd, ref="S5 embedding LR")
        fams = [f for f in sorted(set(fam)) if (fam == f).sum() >= 100 and 0 < y[fam == f].sum() < (fam == f).sum()]
        print("within-family AUROC (families with n>=100): " + "; ".join(f"{f} (n={int((fam == f).sum())}): " + ", ".join(f"{k.split(' (')[0]}{' ' + k.split('(')[1].split('-')[0] if k.startswith('probe') else ''} {auc(y[fam == f], v[fam == f]):.3f}" for k, v in sc_t.items() if k.startswith(("probe", "embedding LR (400", "random"))) for f in fams))
        print("chosen (layer, C) per fold:", sel)
    # routing economics on the y1 router, equal-cost view vs random mix (S5 protocol; cutoff chosen on the same OOF data => optimistic for every learned policy)
    dacc, dcost, ct = D["dacc"], D["dcost"], ce.sum(); rng = np.random.default_rng(0)
    print("\nMax cost saving % within accuracy-loss tolerance, ranking by the OOF score for y1 (cheap>=gpt-4). Optimistic: cutoff chosen on the same data. random mix = analytic line.")
    pol = {k: v for k, v in scores["y1 gpt-3.5 >= gpt-4"].items() if k != "random"}
    P = {k: s5.path(v, dacc, dcost, rng) for k, v in pol.items()}
    print("| loss <= | random mix | " + " | ".join(pol) + " |"); print("|---|---|" + "---|" * len(pol))
    for t in s5.TOLS: print(f"| {t*100:.1f}pt | {100*s5.rand_sav(dacc, dcost, ct, t):.1f} | " + " | ".join(f"{100*s5.sav_at(*P[k][:2], ct, t):.1f}" for k in pol) + " |")
    brng = np.random.default_rng(1); dif = {k: [] for k in pol}
    for _ in range(40 if NT else 200):
        ix = brng.integers(0, n, n); cb = ce[ix].sum()
        for k in pol: Lb, Sb, _ = s5.path(pol[k][ix], dacc[ix], dcost[ix], brng); dif[k].append([s5.sav_at(Lb, Sb, cb, t) - s5.rand_sav(dacc[ix], dcost[ix], cb, t) for t in s5.TOLS])
    print("saving minus random-mix saving, points [bootstrap 95%, resamples of rows]"); print("| loss <= | " + " | ".join(pol) + " |"); print("|---|" + "---|" * len(pol))
    for j, t in enumerate(s5.TOLS): print(f"| {t*100:.1f}pt | " + " | ".join("%+.1f [%+.1f, %+.1f]" % (100 * np.mean(np.array(dif[k])[:, j]), *(100 * np.percentile(np.array(dif[k])[:, j], [2.5, 97.5]))) for k in pol) + " |")
    return scores

# ---------------------------------------------------------------- stage 2
E1_EXTRA = {"reasoning": {"effort": "low"}, "provider": {"order": ["OpenAI"], "allow_fallbacks": False}}  # S5's E1 (openai/gpt-5.4 effort low) with the provider pinned to the one that served all 400 S5 E1 calls
S2_CAP = 3.0; E1_PRICE = (2.5, 15)

def e1_new(texts, gold):
    """E1 labels for the 1000 new Banking77 inputs: same system prompt/inputs/config as S5's E1, cached in stage2_e1.jsonl (resumable; 5-item pilot first). Returns (list of labels or None, usd per call)."""
    f = HERE / "stage2_e1.jsonl"; done = {}
    if f.exists():
        for l in f.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                if r["err"] is None: done[r["i"]] = r
    todo = [i for i in range(len(texts)) if i not in done]; lock = threading.Lock()
    def save(r):
        with lock, open(f, "a", encoding="utf-8") as fh: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if MOCK:
        rng = np.random.default_rng(5)
        for i in todo: done[i] = dict(i=i, content=gold[i] if rng.random() < .9 else names[rng.integers(77)], usd=0.0017, served_by="MOCK", err=None)
    elif todo:
        from openai import OpenAI
        key = [l.split("=", 1)[1].strip() for l in (SM.parent / ".env").read_text().splitlines() if l.startswith("OPENROUTERKEY=")][0]
        cl = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, timeout=180, max_retries=0)
        def one(i):
            last = None
            for att in range(4):
                try:
                    t0 = time.time(); r = cl.chat.completions.create(model="openai/gpt-5.4", temperature=0, max_tokens=2000, extra_body=E1_EXTRA,
                        messages=[{"role": "system", "content": DATA["sys_prompt"]}, {"role": "user", "content": texts[i]}])
                    u = r.usage; usd = getattr(u, "cost", None)
                    return dict(i=i, content=r.choices[0].message.content, ms=(time.time() - t0) * 1000, pt=u.prompt_tokens, ct=u.completion_tokens,
                                usd=usd if usd is not None else (u.prompt_tokens * E1_PRICE[0] + u.completion_tokens * E1_PRICE[1]) / 1e6, usd_is_reported=usd is not None, served_by=getattr(r, "provider", None), err=None)
                except Exception as e:
                    last = f"{type(e).__name__}: {str(e)[:300]}"
                    if any(c in last for c in ("400", "401", "402", "404")): break
                    time.sleep(2 ** att)
            return dict(i=i, content=None, usd=0.0, served_by=None, err=last)
        if not done:  # pilot: 5 items, stop if anything looks wrong
            for i in todo[:5]:
                r = one(i); save(r); print("  pilot", {k: r.get(k) for k in ("i", "content", "served_by", "usd", "ms", "err")}, "gold", gold[i], flush=True)
                if r["err"] or r["served_by"] != "OpenAI" or not (r["content"] or "").strip(): print("PILOT FAILED: stage 2 aborted"); return None, None
                done[i] = r
            todo = todo[5:]
        spent = [sum(r["usd"] for r in done.values())]
        def work(i):
            if spent[0] >= S2_CAP: return None
            r = one(i); save(r)
            with lock: spent[0] += r["usd"]
            return r
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(8) as ex:
            for r in ex.map(work, todo):
                if r and r["err"] is None: done[r["i"]] = r
    rows = [done.get(i) for i in range(len(texts))]
    print(f"[stage2] E1 labels {sum(r is not None for r in rows)}/{len(texts)}; spend ${sum(r['usd'] for r in done.values()):.4f}; served_by {dict(pd.Series([r['served_by'] for r in done.values()]).value_counts())}; "
          f"median latency {np.median([r['ms'] for r in done.values() if r.get('ms')] or [0]):.0f} ms")
    lab = [s5.parse(r["content"], names)[0] if r is not None else None for r in rows]
    print(f"[stage2] E1 non-exact-label rate {np.mean([s5.parse(r['content'], names)[1] for r in rows if r is not None]):.3f}; E1 accuracy vs gold {np.mean([a == b for a, b in zip(lab, gold) if a is not None]):.3f}")
    return lab, float(np.mean([r["usd"] for r in done.values()]))

def s2(tag):
    R, F = load_model(tag); L = R["layers"]; S0 = gen_table(R, "b200"); S1 = gen_table(R, "b1000new")
    gold_new = np.array(T["new_gold"], dtype=object); gold0 = np.array(T["gold200"], dtype=object); e10 = np.array(T["e1"], dtype=object)
    print(f"\n==================== STAGE 2: {tag}, E1 = openai/gpt-5.4 effort low (OpenAI-pinned) on 1000 new Banking77 test inputs ====================")
    lab, e1_usd = e1_new(DATA["b1000new"], gold_new)
    if lab is None: return
    ok = np.array([l is not None for l in lab]); e1n = np.array(lab, dtype=object)
    # 1200 E1-labelled rows: the S5 200 (rows 0..199) + the new rows that got an E1 label
    keep = np.flatnonzero(ok); n0 = len(e10); m = n0 + len(keep)
    cheap = np.r_[S0["ans"], S1["ans"][keep]]; e1 = np.r_[e10, e1n[keep]]; gold = np.r_[gold0, gold_new[keep]]
    y = (cheap == e1).astype(int); gok_c = (cheap == gold).astype(int); gok_e = (e1 == gold).astype(int)
    G = {k: np.r_[S0[k], S1[k][keep]] for k in ("lp_seq", "lp_mean", "gap_first", "gap_min", "sc", "sc_tb")}
    sig = np.c_[tuple(G[k] for k in SIG)]
    E = embed(DATA["b200"] + [DATA["b1000new"][i] for i in keep])
    Xp = {pool: [np.r_[layers_of(F, "b200", pool)[j], layers_of(F, "b1000new", pool)[j][keep]] for j in range(len(L))] for pool in ("last", "mean")}
    print(f"[stage2] {m} E1-labelled rows (200 from S5 + {len(keep)} new); cheap==E1 {y.mean():.3f}; cheap==gold {gok_c.mean():.3f}; E1==gold {gok_e.mean():.3f}; E1 cost ${1000*e1_usd:.3f} per 1000 calls")
    # ---- (i) the specified split: probe trained on 800 new rows, tested on the other 200 new rows; the S5 200 rows are an independent holdout
    rng = np.random.default_rng(21); newrows = np.arange(n0, m); perm = rng.permutation(newrows); tr, te = perm[: len(perm) * 4 // 5], perm[len(perm) * 4 // 5:]; s5rows = np.arange(n0)
    out = {}
    for pool in ("last", "mean"):
        f, best = select_fit([X[tr] for X in Xp[pool]], y[tr]); out[f"probe ({pool}-token)"] = lambda ix, f=f, pool=pool: f([X[ix] for X in Xp[pool]])
        print(f"[stage2] {pool}-token probe trained on {len(tr)} rows: inner AUC {best[0]:.3f} layer {L[best[1]]} C {best[2]}")
    fe, _ = select_fit([E[tr]], y[tr]); out["embedding LR"] = lambda ix: fe([E[ix]])
    fs, _ = select_fit([sig[tr]], y[tr]); out["signals LR (logprob+self-consistency)"] = lambda ix: fs([sig[ix]])
    out["logprob: sequence"] = lambda ix: G["lp_seq"][ix]; out["logprob: min token gap"] = lambda ix: G["gap_min"][ix]; out["self-consistency (3 votes)"] = lambda ix: G["sc"][ix]
    out["self-consistency + logprob tiebreak"] = lambda ix: G["sc_tb"][ix]; rr = np.random.default_rng(7).random(m); out["random (random mixing)"] = lambda ix: rr[ix]
    sc_te = {k: v(te) for k, v in out.items()}; sc_s5 = {k: v(s5rows) for k, v in out.items()}
    refs = {k: "embedding LR" for k in sc_te if k != "embedding LR"}
    o, dd = boot_auc(y[te], sc_te, refs); print_auc(f"\n[stage2] AUROC for cheap==E1 on the 200 held-out NEW rows (probe trained on the other 800), base rate {y[te].mean():.3f}", o, dd)
    o, dd = boot_auc(y[s5rows], sc_s5, refs); print_auc(f"\n[stage2] AUROC for cheap==E1 on the S5 200 rows (independent: no model saw them), base rate {y[s5rows].mean():.3f}", o, dd)
    print("\n[stage2] certificate calibrated on the 200 held-out new rows (cutoff from certify(), delta .05), realised on the independent S5 200 rows; cell = cal coverage % (cutoff) -> realised coverage % / realised error % (n above)")
    print("| score | " + " | ".join(f"budget {int(b*100)}%" for b in (.02, .05, .10)) + " |"); print("|---|---|---|---|")
    for k in sc_te:
        cells = []
        for b in (.02, .05, .10):
            c = certify(sc_te[k], y[te].astype(bool), b)
            if c["coverage"] == 0: cells.append("none certifiable"); continue
            h = apply_certificate(sc_s5[k], y[s5rows].astype(bool), c["cutoff"]); cells.append("%.1f (%.2f) -> %.1f / %s (%d)" % (100 * c["coverage"], c["cutoff"], 100 * h["coverage"], "-" if h["error"] is None else "%.1f" % (100 * h["error"]), h["n_above"]))
        print(f"| {k} | " + " | ".join(cells) + " |")
    # ---- (ii) all 1200 rows, 5-fold cross-fitted scores, then clean random half/half certificates
    cf = {}
    for pool in ("last", "mean"): cf[f"probe ({pool}-token, 5-fold)"], ch = cv_oof(Xp[pool], y); print(f"[stage2] {pool} chosen (layer, C) per fold: {[(L[a], c) for a, c in ch]}")
    cf["embedding LR (5-fold)"], _ = cv_oof([E], y); cf["signals LR (5-fold)"], _ = cv_oof([sig], y)
    cf.update({"logprob: sequence": G["lp_seq"], "logprob: min token gap": G["gap_min"], "self-consistency (3 votes)": G["sc"], "self-consistency + logprob tiebreak": G["sc_tb"], "random": rr})
    o, dd = boot_auc(y, cf, {k: "embedding LR (5-fold)" for k in cf if k != "embedding LR (5-fold)"}); print_auc(f"\n[stage2] AUROC for cheap==E1 on all {m} rows (5-fold cross-fitted), base rate {y.mean():.3f}", o, dd)
    ct = cert_table(cf, y.astype(bool), R=RC2, seed=4, budgets=(.02, .05, .10)); print_cert(ct, cf, f"[stage2] CLEAN certificate vs E1 on {m} rows: cross-fitted scores, {RC2} random half/half splits (calibrate on {m//2}, realise on {m - m//2})", budgets=(.02, .05, .10))
    # ---- blended cost: all calls run the local model (probe needs its forward pass); routed calls skip E1
    thr = R["sets"]["b1000new"]["n"] / R["sets"]["b1000new"]["seconds"]; gpu_h = float(os.environ.get("GPU_USD_H", 0.34)); loc1k = gpu_h / 3600 * 1000 / thr
    print(f"\n[stage2] local model: {thr:.1f} prompts/s (batch 32, greedy + 2 samples per prompt) on a ${gpu_h}/h GPU = ${loc1k:.4f} per 1000 calls (all calls); E1 = ${1000*e1_usd:.3f} per 1000 calls")
    print("| score | budget | routed to cheap % | blended $/1000 calls | saving vs E1 only % | holdout error vs E1 among routed % | gold acc of blended system % (E1 alone %) |"); print("|---|---|---|---|---|---|---|")
    for k in ("probe (last-token, 5-fold)", "embedding LR (5-fold)", "self-consistency (3 votes)", "logprob: sequence", "random"):
        s = cf[k]
        for b in (.02, .05, .10):
            prng = np.random.default_rng(11); cov = []; er = []; ga = []; gae = []
            for _ in range(RC2):
                p = prng.permutation(m); cal, ho = p[: m // 2], p[m // 2:]; c = certify(s[cal], y[cal].astype(bool), b)
                if c["coverage"] == 0: cov.append(0.0); ga.append(gok_e[ho].mean()); gae.append(gok_e[ho].mean()); continue
                r_ = s[ho] >= c["cutoff"]; cov.append(r_.mean()); ga.append(np.where(r_, gok_c[ho], gok_e[ho]).mean()); gae.append(gok_e[ho].mean())
                if r_.any(): er.append(1 - y[ho][r_].mean())
            cv_ = float(np.mean(cov)); bl = 1000 * e1_usd * (1 - cv_) + loc1k
            print(f"| {k} | {int(b*100)}% | {100*cv_:.1f} | {bl:.3f} | {100*(1 - bl/(1000*e1_usd)):.1f} | {100*np.mean(er) if er else float('nan'):.1f} | {100*np.mean(ga):.1f} ({100*np.mean(gae):.1f}) |")

if __name__ == "__main__":
    what = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else "Qwen2.5-0.5B-Instruct"; t0 = time.time()
    if what in ("s1", "all"): r1 = b77(tag); print(f"[time] s1 {time.time()-t0:.0f}s", flush=True)
    if what in ("rb", "all"): rb(tag); print(f"[time] through rb {time.time()-t0:.0f}s", flush=True)
    if what == "s2" or (what == "all" and (r1["bar"] or MOCK)): s2(tag)  # MOCK (tiny check only) forces the stage-2 path
    elif what == "all": print("\nSTAGE 2 SKIPPED: the pre-stated bar was not met (see STAGE-2 BAR above).")
    print(f"[time] total {time.time()-t0:.0f}s")
