"""S3: review deferral on Banking77. Student = LR on a small train subsample; "reviewer changes output" := student != gold.
Confidence head predicts P(change) from student scores (+ embedding); compared with raw top-2 gap / max prob / random /
oracle on changes-caught-vs-share-reviewed, average precision, and the SGR certificate against corrections.
Revision: paired deltas vs the gap, infeasible-split counts, chosen C, certificate diagnostics, calibration-size sweep."""
import sys, time, warnings
for m_ in ("The least populated class", "The default value for l1_ratios", "The fitted attributes of LogisticRegressionCV"):
    warnings.filterwarnings("ignore", message=m_)  # CV-fold warning at tiny head sizes + two sklearn 1.10 FutureWarnings; convergence warnings stay visible
sys.stdout.reconfigure(encoding="utf-8")  # cp1252 stdout turned the +/- sign into mojibake in log.txt
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score
from common import *

OUT = ROOT / "s3_review_deferral" / "tables.md"
BUDGETS = (0.05, 0.10, 0.20, 0.30, 0.50)
lines = []
TIE = [0, 0]  # [certificates whose applied cutoff set (conf >= cutoff, ties included) has CP bound > budget, certificates issued]
def say(s=""):
    print(s, flush=True); lines.append(s)
def f(x, nd=3):
    return "-" if x is None or x == float("inf") else f"{x:.{nd}f}"
def ms(a, nd=3):
    return f"{np.nanmean(a):.{nd}f} ± {np.nanstd(a):.{nd}f}"

def feats(P):
    s = np.sort(P, 1)[:, ::-1]; ent = -(P * np.log(P + 1e-12)).sum(1)
    return np.column_stack([s[:, 0] - s[:, 1], s[:, 0], ent, s[:, :5]]), np.eye(P.shape[1])[P.argmax(1)]  # numeric(8), one-hot(77)

def zs(X, idx):
    return (X - X[idx].mean(0)) / (X[idx].std(0) + 1e-6)

def fit_heads(N, O, E, y, idx):
    """Train 3 heads on rows idx (reviewed rows); return P(change) for every row and the CV-chosen C of the LR heads."""
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)
    gb.fit(np.hstack([N, O])[idx], y[idx])
    cv = lambda: LogisticRegressionCV(Cs=[0.003, 0.01, 0.03, 0.1, 0.3, 1, 3], cv=5, scoring="average_precision", max_iter=3000)
    Xa = np.hstack([zs(N, idx), O]); Xb = np.hstack([Xa, zs(E, idx)])
    la, lb = cv().fit(Xa[idx], y[idx]), cv().fit(Xb[idx], y[idx])
    return ({"GBC feat": gb.predict_proba(np.hstack([N, O]))[:, 1], "LR feat": la.predict_proba(Xa)[:, 1], "LR feat+emb": lb.predict_proba(Xb)[:, 1]},
            {"LR feat": float(la.C_[0]), "LR feat+emb": float(lb.C_[0])})

def metrics(score, y):  # score high = review first
    c = np.cumsum(y[np.argsort(-score, kind="stable")]) / y.sum(); n = len(y)
    return dict(caught=[c[int(np.ceil(b * n)) - 1] for b in BUDGETS], n80=(np.argmax(c >= .8) + 1) / n, n90=(np.argmax(c >= .9) + 1) / n,
                ap=average_precision_score(y, score))

def rand_metrics(y, reps=200):
    rng = np.random.default_rng(0); m = [metrics(rng.random(len(y)), y) for _ in range(reps)]
    return {k: np.mean([x[k] for x in m], 0) for k in m[0]}

def split(n, seed):
    p = np.random.default_rng(seed).permutation(n); return p[:1000], p[1000:1500], p[1500:]

def cert(cf, ok, cal, tgt, bud):
    """certify on rows cal, apply on rows tgt; also count certificates whose tie-inclusive cutoff set breaks the bound."""
    c = certify(cf[cal], ok[cal], bud); h = apply_certificate(cf[tgt], ok[tgt], c["cutoff"])
    if c["cutoff"] != np.inf:
        a = cf[cal] >= c["cutoff"]; d = 0.05 / (int(np.ceil(np.log2(len(cal)))) + 1)
        TIE[0] += cp_upper(int((~ok[cal][a]).sum()), int(a.sum()), d) > bud; TIE[1] += 1
    return c, h

def diag(k, cf, ok, bud, delta=0.05):
    """Why was a certificate infeasible? Numbers the SGR search needs vs what the cal rows offer (tie-aware scan over every cutoff)."""
    n = len(cf); m = int(np.ceil(np.log2(n))) + 1; d = delta / m
    k0 = next((j for j in range(1, n + 1) if cp_upper(0, j, d) <= bud), None)
    o = np.argsort(-cf, kind="stable"); s, e = cf[o], ~ok[o]
    first = int(np.argmax(e)) + 1 if e.any() else None
    cnt = np.unique(cf, return_counts=True)[1]
    ends = np.r_[np.where(s[:-1] > s[1:])[0] + 1, n]  # candidate set sizes = ends of tie blocks (set == conf >= cutoff)
    feas = sum(cp_upper(int(e[:j].sum()), int(j), d) <= bud for j in ends)
    return (f"- {k} @ {bud:.0%}: SGR search has m={m} steps so per-step delta={d:.4f}; zero errors would need {k0} top rows; "
            f"first error at rank {first} (stable order); {len(cnt)} unique scores among {n} cal rows, largest tie block {cnt.max()}; "
            f"{feas} of {len(ends)} tie-aware candidate cutoffs satisfy the bound.")

def ceiling(cf, ok, bud):
    """Largest coverage whose empirical change rate is <= budget, cutoff picked on these very rows (no statistical penalty; optimistic)."""
    o = np.argsort(-cf, kind="stable"); w = np.where(np.cumsum(~ok[o]) / np.arange(1, len(o) + 1) <= bud)[0]
    return (w[-1] + 1) / len(o) if len(w) else 0.0

def calsweep(conf, ok, pool, ev):
    sizes = (250, 500, 1000, 1500)
    say(f"\n### Calibration-size sweep (heads fit on the same 1000 head_train rows; cal rows drawn at random from a {len(pool)}-row pool "
        f"(head_cal + part of the holdout) that is disjoint from head_train and from the fixed {len(ev)}-row eval set; 20 draws per size; "
        f"the 1500 size is the whole pool, so its draws are identical)\n")
    say("Cell = eval coverage, mean ± sd over 20 draws, with the number of infeasible draws (nothing certified, coverage 0) in brackets. "
        "Ceiling = largest eval coverage with empirical change rate <= budget, cutoff chosen on the eval rows (an upper reference, not a certificate). "
        "Max change rate = largest realized change rate above the cutoff over all feasible draws (draws above the budget / feasible draws).\n")
    for bud in (0.05, 0.10):
        say(f"Budget {bud:.0%}\n")
        say("| ranking | " + " | ".join(f"cal {s}" for s in sizes) + " | ceiling | max change rate |"); say("|---" * (len(sizes) + 3) + "|")
        for k, cf in conf.items():
            cells, er = [], []
            for s in sizes:
                out = [cert(cf, ok, np.random.default_rng(200 + d).permutation(pool)[:s], ev, bud) for d in range(20)]
                cov = np.array([h["coverage"] for _, h in out]); er += [h["error"] for _, h in out if h["error"] is not None]
                cells.append(f"{cov.mean():.3f} ± {cov.std():.3f} ({sum(c['cutoff'] == np.inf for c, _ in out)})")
            say(f"| {k} | " + " | ".join(cells) + f" | {ceiling(cf[ev], ok[ev], bud):.3f} | {max(er, default=float('nan')):.3f} ({sum(e > bud for e in er)}/{len(er)}) |")
        say()

def paired(R):
    """Per-split deltas of each ranking against the raw gap, 5 splits: mean [min, max] and the number of splits where it is better."""
    g = R["top-2 gap"]; cols = [("share for 80%", 0, -1), ("share for 90%", 1, -1), ("AP", 2, 1), ("caught @10%", 3, 1), ("cert@5% coverage", 4, 1), ("cert@10% coverage", 6, 1)]
    say("| ranking | " + " | ".join(c[0] for c in cols) + " |"); say("|---" * (len(cols) + 1) + "|")
    for k, a in R.items():
        if k == "top-2 gap": continue
        cells = []
        for _, j, sg in cols:
            d = a[:, j] - g[:, j]; cells.append(f"{d.mean():+.3f} [{d.min():+.3f}, {d.max():+.3f}] {int((sg * d > 0).sum())}/{len(d)}")
        say(f"| {k} | " + " | ".join(cells) + " |")

def study(name, Xtr, ytr, Xte, yte, n_student):
    TIE[:] = [0, 0]
    idx = stratified_subsample(ytr, n_student, seed=0)
    clf = LogisticRegression(C=10, max_iter=2000).fit(Xtr[idx], ytr[idx]); P = clf.predict_proba(Xte)
    ok = clf.classes_[P.argmax(1)] == yte; ch = (~ok).astype(int); N, O = feats(P); gap, mx = confidence(P), P.max(1)
    htr, cal, hold = split(len(yte), 0)
    say(f"\n## {name}: student trained on {len(idx)} rows\n")
    say(f"Student test accuracy {ok.mean():.3f}; change rate: all test {ch.mean():.3f}, head_train {ch[htr].mean():.3f}, "
        f"head_cal {ch[cal].mean():.3f}, holdout {ch[hold].mean():.3f} ({ch[hold].sum()} changes in {len(hold)} rows).\n")
    H, Cs = fit_heads(N, O, Xte, ch, htr)
    conf = {"max prob": mx, "top-2 gap": gap} | {k: 1 - v for k, v in H.items()}   # higher = more trusted; review priority = -conf
    M = {"random": rand_metrics(ch[hold]), "oracle": metrics(ch[hold].astype(float), ch[hold])} | {k: metrics(-v[hold], ch[hold]) for k, v in conf.items()}
    assert M["oracle"]["ap"] > 0.999 and abs(M["random"]["caught"][-1] - 0.5) < 0.03   # sanity
    order = ["random", "max prob", "top-2 gap", "GBC feat", "LR feat", "LR feat+emb", "oracle"]
    say("### Changes caught vs share reviewed (holdout; fraction of all changes in holdout caught)\n")
    say("| ranking | " + " | ".join(f"{int(b*100)}%" for b in BUDGETS) + " |"); say("|---" * (len(BUDGETS) + 1) + "|")
    for k in order: say(f"| {k} | " + " | ".join(f"{v:.3f}" for v in M[k]["caught"]) + " |")
    say("\n### Share of outputs reviewed to catch 80% / 90% of changes; average precision (holdout)\n")
    say("| ranking | share for 80% | share for 90% | AP |"); say("|---|---|---|---|")
    for k in order: say(f"| {k} | {M[k]['n80']:.3f} | {M[k]['n90']:.3f} | {M[k]['ap']:.3f} |")
    say("\n### Certificate against corrections (conf = 1-P(change) for heads; certify on head_cal 500 rows, apply to holdout; delta 5%)\n")
    say("Coverage = share auto-passed (not reviewed). 'changes caught' = share of holdout changes that fall below the cutoff (i.e. get reviewed).\n")
    say("| budget | confidence | cutoff | cal coverage | holdout coverage | holdout change rate above cutoff | changes caught |"); say("|---|---|---|---|---|---|---|")
    dg = []
    for bud in (0.05, 0.10):
        for k, cf in conf.items():
            c, h = cert(cf, ok, cal, hold, bud)
            miss = (cf[hold] >= c["cutoff"]) & ~ok[hold]
            say(f"| {bud:.0%} | {k} | {f(c['cutoff'])} | {c['coverage']:.3f} | {h['coverage']:.3f} | {f(h['error'])} | {1 - miss.sum() / ch[hold].sum():.3f} |")
            if c["cutoff"] == np.inf: dg.append(diag(k, cf[cal], ok[cal], bud))
    if dg: say("\nInfeasible certificates (seed-0 cal rows), diagnosed by run.py:\n"); [say(x) for x in dg]
    say("\n### Head quality vs number of reviewed rows used to train the head (holdout; mean ± sd over 5 random subsets of head_train, 1000 = all)\n")
    say("Last three columns: in how many of the subsets the head beats the raw gap on that metric (the gap is one fixed ranking on this holdout).\n")
    say("| head rows | head | share for 80% | share for 90% | AP | caught @10% | share80 < gap | AP > gap | caught@10% > gap |"); say("|---|---|---|---|---|---|---|---|---|")
    g = M["top-2 gap"]; say(f"| any | top-2 gap (no head) | {g['n80']:.3f} | {g['n90']:.3f} | {g['ap']:.3f} | {g['caught'][1]:.3f} | - | - | - |")
    for n in (100, 300, 1000):
        R = {}
        for r in range(5 if n < 1000 else 1):
            sub = htr if n == 1000 else np.random.default_rng(100 + r).permutation(htr)[:n]
            for k, v in (H if n == 1000 else fit_heads(N, O, Xte, ch, sub)[0]).items():
                m = metrics(v[hold], ch[hold]); R.setdefault(k, []).append([m["n80"], m["n90"], m["ap"], m["caught"][1]])
        for k, a in R.items():
            a = np.array(a); w = lambda x: f"{int(x.sum())}/{len(x)}"
            say(f"| {n} | {k} | " + " | ".join(ms(a[:, j]) if len(a) > 1 else f"{a[0, j]:.3f}" for j in range(4))
                + f" | {w(a[:, 0] < g['n80'])} | {w(a[:, 2] > g['ap'])} | {w(a[:, 3] > g['caught'][1])} |")
    R, Cl = {}, {"LR feat": [], "LR feat+emb": []}
    for s in range(5):
        a, c_, h = split(len(yte), s); Hs, Cs_ = fit_heads(N, O, Xte, ch, a)
        for k in Cl: Cl[k].append(Cs_[k])
        for k, cf in ({"max prob": mx, "top-2 gap": gap} | {k: 1 - v for k, v in Hs.items()}).items():
            m = metrics(-cf[h], ch[h]); row = [m["n80"], m["n90"], m["ap"], m["caught"][1]]
            for bud in (0.05, 0.10):
                r = cert(cf, ok, c_, h, bud)[1]; row += [r["coverage"], np.nan if r["error"] is None else r["error"]]
            R.setdefault(k, []).append(row)
    R = {k: np.array(a) for k, a in R.items()}
    say("\n### Robustness: 5 different random head_train/head_cal/holdout splits (1000 head rows; mean ± sd)\n")
    say("| ranking | share for 80% | share for 90% | AP | caught @10% |"); say("|---|---|---|---|---|")
    for k, a in R.items(): say(f"| {k} | " + " | ".join(ms(a[:, j]) for j in range(4)) + " |")
    say("\nCertificate on those splits (cert = SGR on that split's 500 cal rows, delta 5%). Coverage averages every split, infeasible ones count as 0. "
        "Change-rate columns are the realized rate above the cutoff: mean over feasible splits only, then the maximum over splits; 'infeasible' = splits with nothing certified.\n")
    say("| ranking | cert@5% coverage | infeasible @5% | change rate @5% mean / max | cert@10% coverage | infeasible @10% | change rate @10% mean / max |"); say("|---|---|---|---|---|---|---|")
    for k, a in R.items():
        say(f"| {k} | {ms(a[:, 4])} | {int(np.isnan(a[:, 5]).sum())}/5 | " + (f"{np.nanmean(a[:, 5]):.3f} / {np.nanmax(a[:, 5]):.3f}" if (~np.isnan(a[:, 5])).any() else "-")
            + f" | {ms(a[:, 6])} | {int(np.isnan(a[:, 7]).sum())}/5 | " + (f"{np.nanmean(a[:, 7]):.3f} / {np.nanmax(a[:, 7]):.3f}" if (~np.isnan(a[:, 7])).any() else "-") + " |")
    say("\nPer-split certified holdout coverage (splits seeds 0 to 4, in order):\n")
    say("| ranking | cert@5% | cert@10% |"); say("|---|---|---|")
    for k, a in R.items(): say(f"| {k} | " + " / ".join(f"{x:.3f}" for x in a[:, 4]) + " | " + " / ".join(f"{x:.3f}" for x in a[:, 6]) + " |")
    say("\nPaired against the raw top-2 gap, same split: mean delta [min, max] over the 5 splits, then splits in which the ranking is better "
        "(lower share, higher AP / caught / coverage). The 5 holdouts overlap heavily, so read these as consistency checks, not independent trials.\n")
    paired(R)
    say(f"\nChosen C (LR heads, 5-fold CV on average precision; grid 0.003 to 3): headline split LR feat {Cs['LR feat']}, LR feat+emb {Cs['LR feat+emb']}; "
        f"across the 5 splits LR feat {Cl['LR feat']}, LR feat+emb {Cl['LR feat+emb']}.")
    calsweep(conf, ok, np.r_[cal, hold[:1000]], hold[1000:])
    say(f"Tie check ({name}, all certificates issued above): {TIE[0]} of {TIE[1]} feasible certificates have a tie-inclusive cutoff set (conf >= cutoff) whose Clopper-Pearson bound at delta/m exceeds the budget.")

if __name__ == "__main__":
    t0 = time.time()
    b = load_banking77(); Xtr, ytr = embed(b["train"][0]), b["train"][1]; Xte, yte = embed(b["test"][0]), b["test"][1]
    say(f"Banking77: train {len(ytr)}, test {len(yte)}; head_train 1000 / head_cal 500 / holdout {len(yte) - 1500}; embedding {EMB_MODEL}.")
    for n in (770, 3080):
        t = time.time(); study("Banking77", Xtr, ytr, Xte, yte, n); say(f"\n(section wall time {time.time() - t:.0f}s)")
    say(f"\nTotal wall time {time.time() - t0:.0f}s on CPU.")
    OUT.write_text("\n".join(lines), encoding="utf-8")
