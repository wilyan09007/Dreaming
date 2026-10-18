"""S21 arms (CPU, local): Banking77 router side with Qwen2.5-0.5B as the cheap model. Target = cheap label == E1 (S5 cached), same 200 rows as S15.
Adds f1 (student==cheap), f3 (f1 then cheap logprob gap), f2 (linear probe on the bge student embedding) and probe+f1 combos, next to S15's signals recomputed from the S15 cache
for gpt-5.4-nano and qwen3-next on the identical 200 rows. Usage: python arms.py [tag]   (stdout -> log.txt part 2; writes arms.json)
ponytail: reuses analyze.py (probe fitting, bootstrap, clean certificates) and S15 run.py (lp_feats, pdiff, rotations)."""
import os, sys, json, importlib.util
os.environ.setdefault("S21_SUF", "_gpu")
from pathlib import Path
HERE = Path(__file__).resolve().parent; SM = HERE.parent
sys.path.insert(0, str(SM)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(SM / "s15_confidence_signals"))
import analyze as A
import run as s15            # S15 helpers
from calls import load_cache  # S15 cached cheap-model calls
from common import *          # noqa
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
tag = sys.argv[1] if len(sys.argv) > 1 else "Qwen2.5-0.5B-Instruct"
T, DATA, names = A.T, A.DATA, A.T["names"]
BUD3 = (.05, .10, .15, .20); OUTJ = {}
R, F = A.load_model(tag); L = R["layers"]
G = {s: A.gen_table(R, s) for s in ("b200", "b1000tr", "b1000new")}
gold = {"b200": np.array(T["gold200"], object), "b1000tr": np.array(T["tr_gold"], object), "b1000new": np.array(T["new_gold"], object)}
e1 = np.array(T["e1"], object); N = len(e1)
lab = G["b200"]["ans"]; y = (lab == e1)
y_tr = (G["b1000tr"]["ans"] == gold["b1000tr"]).astype(int); y_new = (G["b1000new"]["ans"] == gold["b1000new"]).astype(int); y_g200 = (lab == gold["b200"]).astype(int)
print(f"# S21 arms, cheap = {tag}. cheap==E1 on 200: {y.mean():.3f}; cheap==gold: 200 {y_g200.mean():.3f}, 1000 train {y_tr.mean():.3f}, 1000 new {y_new.mean():.3f}; non-exact answers (counted wrong) {G['b200']['pf'].mean():.3f}")
# ---- shadow student = S1/S15 recipe: frozen bge-small + LR(C=10) on all Banking77 train
bk = load_banking77(); Xtr, ytr = embed(bk["train"][0]), bk["train"][1]
st = LogisticRegression(C=10, max_iter=2000).fit(Xtr, ytr)
E = {s: embed(DATA[s]) for s in G}
def stu(s): P = st.predict_proba(E[s]); return confidence(P), np.array([names[k] for k in P.argmax(1)], object)
sg200, sp200 = stu("b200"); sgn, spn = stu("b1000new")
print(f"student (9993 train rows): gold acc on 200 {np.mean(sp200 == gold['b200']):.3f}, agrees with E1 {np.mean(sp200 == e1):.3f}; on 1000 new {np.mean(spn == gold['b1000new']):.3f}")
# ---- probes trained on the 1000 train rows with the proxy target (cheap == gold)
def probe(sname_tr, y_tr_, sets):
    best = None
    for pool in ("last", "mean"):
        X = {s: A.layers_of(F, s, pool) for s in sets + [sname_tr]}
        f, b = A.select_fit(X[sname_tr], y_tr_)
        print(f"[probe] pool={pool}: inner AUC {b[0]:.3f} layer {L[b[1]]} C {b[2]}", flush=True)
        if best is None or b[0] > best[0][0]: best = (b, pool, f, X)
    b, pool, f, X = best; print(f"[probe] chosen by inner AUC on the train rows: pool={pool} layer {L[b[1]]} C {b[2]}")
    return {s: f(X[s]) for s in sets}, pool, L[b[1]], b[2]
pr, pool, lay, C_ = probe("b1000tr", y_tr, ["b200", "b1000new"])
fe, be = A.select_fit([E["b1000tr"]], y_tr); print(f"[f2] bge-embedding LR proxy-trained: inner AUC {be[0]:.3f} C {be[2]}")
f2 = {s: fe([E[s]]) for s in ("b200", "b1000new")}
OUTJ["probe_choice"] = dict(pool=pool, layer=int(lay), C=C_)
# ---- 200-row signals
g = G["b200"]; ok = lambda a: np.nan_to_num(a, nan=-50)
f1 = (sp200 == lab).astype(float)
sc = {}
sc["cheap logprob: sequence"] = g["lp_seq"]; sc["cheap logprob gap (min token top-2 gap)"] = g["gap_min"]; sc["cheap first-token gap"] = g["gap_first"]
sc["student top-2 gap"] = sg200
sc["f1 student==cheap"] = f1
sc["f3 f1, then cheap logprob gap"] = f1 * 1000 + g["gap_min"]
sc["f1, then student gap"] = f1 * 1000 + sg200
sc["probe (hidden state, proxy-trained)"] = pr["b200"]
sc["f2 bge-embedding LR (proxy-trained)"] = f2["b200"]
sc["f1, then probe"] = f1 * 1000 + pr["b200"]
sc["f1, then f2"] = f1 * 1000 + f2["b200"]
sc["random"] = np.random.default_rng(7).random(N)
# stack LR over [probe, cheap gap, student gap, f1], nested inside every rotation for certificates; 5-fold OOF for AUROC
Z = np.c_[pr["b200"], g["gap_min"], sg200, f1, f2["b200"]]
def oof_all(Z, tgt):
    o = np.zeros(N)
    for a_, b_ in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(N), tgt):
        mu, sd = Z[a_].mean(0), Z[a_].std(0) + 1e-6
        o[b_] = LogisticRegression(C=1.0, max_iter=3000).fit((Z[a_] - mu) / sd, tgt[a_]).predict_proba((Z[b_] - mu) / sd)[:, 1]
    return o
stack_oof = oof_all(Z, y)
# 5-fold-on-200 variants (AUROC only: the 200 labels fit them)
pC, chC = A.cv_oof(A.layers_of(F, "b200", pool), y.astype(int)); eC, _ = A.cv_oof([E["b200"]], y.astype(int))
cvarms = {"probe (5-fold on the 200, OOF)": pC, "f2 bge-embedding LR (5-fold on the 200, OOF)": eC, "stack LR [probe, cheap gap, student gap, f1, f2] (5-fold OOF)": stack_oof}
# ---- S15 signals for the other cheap models on the same 200 rows (own cheap answers, own target)
cache = load_cache(); RR = {}
for r in cache.values(): RR.setdefault(r["run"], {})[r["i"]] = r
other = {}
for mk, nm in (("nano", "gpt-5.4-nano (S15)"), ("qnx", "qwen3-next-80b (S15)")):
    recs = [RR[f"{mk}_A"][i] for i in range(N)]
    l_ = np.array([s15.parse(r["content"], names)[0] for r in recs], object)
    fe_ = np.array([s15.lp_feats(recs[i], False, l_[i], names) for i in range(N)])
    ag = (sp200 == l_).astype(float); gp = np.nan_to_num(fe_[:, 2], nan=-50)
    other[nm] = (l_ == e1, {"cheap logprob gap (S15 b2)": gp, "student top-2 gap": sg200, "f1 student==cheap": ag, "f3 f1, then cheap logprob gap": ag * 1000 + gp, "random": np.random.default_rng(7).random(N)})
# ---- tables
def tab_auc(title, scores, yy, ref_keys):
    o, _ = A.boot_auc(yy.astype(int), scores, B=1000)
    print(f"\n{title}\n| signal | AUROC [95% CI] |" + "".join(f" diff vs {r} [95% CI] |" for r in ref_keys) + "\n|---|---|" + "---|" * len(ref_keys))
    for k, s in scores.items():
        cells = []
        for r in ref_keys:
            if r in scores and k != r: d = s15.pdiff(np.asarray(s, float), np.asarray(scores[r], float), yy); cells.append("%+.3f [%+.3f, %+.3f]" % d)
            else: cells.append("-")
        print("| %s | %.3f [%.3f, %.3f] | %s |" % (k, *o[k], " | ".join(cells)))
    return {k: [float(x) for x in v] for k, v in o.items()}
def rot_cert(scores, yy, nested=None):
    rows = {}
    for k, s in scores.items():
        for b in BUD3:
            cov, mm = [], []
            for r in range(5):
                perm = np.random.default_rng(200 + r).permutation(len(yy)); nc = min(100, len(yy) // 2); cal, hold = perm[:nc], perm[nc:]
                if k == "stack": cc, ch = nested(cal, hold)
                else: cc, ch = s[cal], s[hold]
                c = certify(cc, yy[cal], b)
                if c["cutoff"] == np.inf: cov.append(0.0); continue
                h = apply_certificate(ch, yy[hold], c["cutoff"]); cov.append(h["coverage"])
                if h["error"] is not None: mm.append(h["error"])
            rows[(k, b)] = (float(np.mean(cov)), float(np.min(cov)), float(np.max(mm)) if mm else None, sum(c_ > 0 for c_ in cov))
    return rows
def print_rot(rows, keys, title):
    print(f"\n{title}\ncell = mean holdout coverage % (min over rotations) / worst realised mismatch % among routed / rotations that certified (of 5)")
    print("| signal | " + " | ".join(f"budget {int(b*100)}%" for b in BUD3) + " |\n|---|" + "---|" * len(BUD3))
    for k in keys: print(f"| {k} | " + " | ".join("%.0f (%.0f) / %s / %d" % (100 * rows[(k, b)][0], 100 * rows[(k, b)][1], "-" if rows[(k, b)][2] is None else "%.0f" % (100 * rows[(k, b)][2]), rows[(k, b)][3]) for b in BUD3) + " |")
def stack_nested(cal, hold):
    mu, sd = Z[cal].mean(0), Z[cal].std(0) + 1e-6; Zt = (Z[cal] - mu) / sd; t = y[cal]; oof = np.zeros(len(cal))
    for a_, b_ in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(cal)), t):
        oof[b_] = LogisticRegression(C=1.0, max_iter=3000).fit(Zt[a_], t[a_]).predict_proba(Zt[b_])[:, 1]
    return oof, LogisticRegression(C=1.0, max_iter=3000).fit(Zt, t).predict_proba((Z[hold] - mu) / sd)[:, 1]
print(f"\n## A. {tag} as the cheap model: target cheap label == E1 (n=200, base rate {y.mean():.3f})")
allsc = {**sc, **cvarms}
OUTJ["auc_qwen"] = tab_auc("AUROC with 95% bootstrap CI over inputs (1000 resamples); paired diffs vs the cheap logprob gap and vs f1", allsc, y, ["cheap logprob gap (min token top-2 gap)", "f1 student==cheap"])
cs = {**sc, "stack": stack_oof}
rc = rot_cert(cs, y, stack_nested); print_rot(rc, list(cs), "Certified routing, 5 rotations (rng 200+r) of 100 calibration / 100 holdout, certify() grid 40, delta 0.05. Stack = nested LR on [probe, cheap gap, student gap, f1, f2] fit inside each calibration half")
OUTJ["cert_qwen_rot"] = {f"{k}|{b}": v for (k, b), v in rc.items()}
clean = {k: v for k, v in sc.items()}
ct = A.cert_table(clean, y, R=200, budgets=BUD3); A.print_cert(ct, clean, "Clean certificate, 200 random 100/100 splits (more rotations than S15's 5)", budgets=BUD3)
OUTJ["cert_qwen_200splits"] = {f"{k}|{b}": v for (k, b), v in ct.items()}
for nm, (yy, scs) in other.items():
    print(f"\n## B. {nm} as the cheap model (S15 cache, same 200 inputs): cheap==E1 base rate {yy.mean():.3f}")
    OUTJ["auc_" + nm] = tab_auc("AUROC [95% CI]", scs, yy, ["cheap logprob gap (S15 b2)"])
    rc2 = rot_cert(scs, yy); print_rot(rc2, list(scs), "Certified routing, same 5 rotations"); OUTJ["cert_" + nm] = {f"{k}|{b}": v for (k, b), v in rc2.items()}
# ---- bigger pool, gold target
yb = np.r_[y_g200, y_new].astype(int); gb = G["b1000new"]
big = {"cheap logprob: sequence": np.r_[g["lp_seq"], gb["lp_seq"]], "cheap logprob gap (min token top-2 gap)": np.r_[g["gap_min"], gb["gap_min"]], "student top-2 gap": np.r_[sg200, sgn],
       "f1 student==cheap": np.r_[f1, (spn == gb["ans"]).astype(float)], "probe (hidden state, proxy-trained)": np.r_[pr["b200"], pr["b1000new"]], "f2 bge-embedding LR (proxy-trained)": np.r_[f2["b200"], f2["b1000new"]], "random": np.random.default_rng(1).random(len(yb))}
big["f3 f1, then cheap logprob gap"] = big["f1 student==cheap"] * 1000 + big["cheap logprob gap (min token top-2 gap)"]
big["f1, then probe"] = big["f1 student==cheap"] * 1000 + big["probe (hidden state, proxy-trained)"]
print(f"\n## C. Pool of 200 + 1000 new test rows, target cheap==gold (n={len(yb)}, base rate {yb.mean():.3f}); probe and f2 never saw these rows; student never saw test rows")
OUTJ["auc_big"] = tab_auc("AUROC [95% CI]; paired diffs", big, yb, ["cheap logprob gap (min token top-2 gap)", "f1 student==cheap"])
ctb = A.cert_table(big, yb.astype(bool), R=200, seed=2, budgets=BUD3); A.print_cert(ctb, big, f"Clean certificate vs GOLD on {len(yb)} rows, 200 random 600/600 splits", budgets=BUD3)
OUTJ["cert_big"] = {f"{k}|{b}": v for (k, b), v in ctb.items()}
json.dump(OUTJ, open(HERE / "arms.json", "w"), indent=1, default=float)
