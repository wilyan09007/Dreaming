"""S15: which confidence signal should gate a call? Usage: run.py p1 [nsplit] [ncal] | p2 | all (pod: p2, p1 at 500 cal rows, p1 at 1500 cal rows).
p1 = student side (CPU, Banking77 @770/3080 rows, CLINC150 @1500 rows), p2 = router side (cheap vs E1, uses responses.jsonl from calls.py).
ponytail: one file; stdout is the log; tables are also written to tables_p1.md / tables_p2.md."""
import sys, json, re, time, difflib, warnings, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # smokes/ (common.py): portable laptop/pod
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:  # pod: common.embed hard-codes device="cpu"; use the GPU when there is one (common.py untouched)
    import torch, sentence_transformers as _st
    if torch.cuda.is_available():
        _Orig = _st.SentenceTransformer
        class _GPU(_Orig):
            def __init__(self, name, device=None, **k): super().__init__(name, device="cuda", **k)
        _st.SentenceTransformer = _GPU
except Exception: pass
from common import *  # noqa
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize_scalar
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).parent
OUT = []
def say(s=""):
    print(s, flush=True); OUT.append(s)
def f3(x, nd=3): return "-" if x is None or (isinstance(x, float) and (np.isnan(x) or x == np.inf)) else f"{x:.{nd}f}"
def msd(a, nd=3): a = np.asarray(a, float); return f"{np.nanmean(a):.{nd}f} ± {np.nanstd(a):.{nd}f}"
def mmin(a, nd=2): a = np.asarray(a, float); return f"{np.nanmean(a):.{nd}f} ({np.nanmin(a):.{nd}f})"
def mmax(a, nd=3): a = [x for x in a if x is not None]; return f"{np.mean(a):.{nd}f} ({np.max(a):.{nd}f})" if a else "-"

# ============================================================ PART 1: student side
NCAL, NHEAD, BUD = 500, 1000, (0.02, 0.05, 0.10)  # 10% added: at 500 cal rows the 2% budget certifies nothing for these students
SIGS = ["a gap", "b maxp", "c -entropy", "d temp-gap", "e kNN agree", "f head(emb)", "g head(a-e)", "h gap+kNN"]

def softmax_T(logp, T):
    z = logp / T; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def fit_T(P, yi):  # temperature on held-out rows: minimise NLL of the true label under softmax(log P / T)
    lp = np.log(P + 1e-12)
    nll = lambda T: -np.log(softmax_T(lp, T)[np.arange(len(yi)), yi] + 1e-12).mean()
    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)

def knn_share(Xq, pred, Xs, ys, k=10):
    S = Xq @ Xs.T; nn = np.argpartition(-S, k, axis=1)[:, :k]
    return (ys[nn] == pred[:, None]).mean(1)

def base_signals(clf, Xs, ys, Xq, T):
    P = clf.predict_proba(Xq); pred = clf.classes_[P.argmax(1)]
    gap, mp = confidence(P), P.max(1); ent = -(P * np.log(P + 1e-12)).sum(1); tg = confidence(softmax_T(np.log(P + 1e-12), T)); kn = knn_share(Xq, pred, Xs, ys)
    F = np.column_stack([gap, np.log(gap + 1e-4), mp, ent, tg, np.log(tg + 1e-4), kn])
    return dict(P=P, pred=pred, conf={"a gap": gap, "b maxp": mp, "c -entropy": -ent, "d temp-gap": tg, "e kNN agree": kn, "h gap+kNN": gap + kn}, F=F)

def zs(A, B): mu, sd = A.mean(0), A.std(0) + 1e-6; return (B - mu) / sd

def head(Xtr, err, Xq, Cs):  # P(error) from an LR head, C chosen by 5-fold CV log-loss
    m = LogisticRegressionCV(Cs=Cs, cv=5, scoring="neg_log_loss", max_iter=3000).fit(zs(Xtr, Xtr), err)
    return m.predict_proba(zs(Xtr, Xq))[:, 1]

def eval_conf(conf, ok, cal, hold):
    r = dict(auc=roc_auc_score(~ok[hold], -conf[hold]))
    for b in BUD:
        c = certify(conf[cal], ok[cal], b); h = apply_certificate(conf[hold], ok[hold], c["cutoff"])
        r[f"cov{b}"] = h["coverage"] if c["cutoff"] != np.inf else 0.0; r[f"err{b}"] = h["error"] if c["cutoff"] != np.inf else None
    return r

def student_block(name, tr, te, n_train, nsplit, head_sizes=(NHEAD,)):
    Xtr, ytr = embed(tr[0]), tr[1]; Xte, yte = embed(te[0]), te[1]
    u = np.unique(ytr); ytr, yte = np.searchsorted(u, ytr), np.searchsorted(u, yte)  # CLINC labels skip the dropped oos index
    idx = stratified_subsample(ytr, n_train, seed=0); pool = np.setdiff1d(np.arange(len(ytr)), idx)
    t = time.time(); clf = LogisticRegression(C=10, max_iter=2000).fit(Xtr[idx], ytr[idx]); Xs, ys = Xtr[idx], ytr[idx]
    assert (clf.classes_ == np.arange(len(clf.classes_))).all()
    Pte = clf.predict_proba(Xte); ok = clf.classes_[Pte.argmax(1)] == yte
    say(f"\n### {name}: student trained on {len(idx)} rows ({time.time()-t:.0f}s), test accuracy {ok.mean():.3f} ({(~ok).sum()} errors of {len(ok)}), head-training pool {len(pool)} unused train rows, {nsplit} random splits (cal {NCAL} / holdout {len(yte)-NCAL})")
    R = {nh: collections.defaultdict(lambda: collections.defaultdict(list)) for nh in head_sizes}
    boot = None
    for s in range(nsplit):
        rng = np.random.default_rng(100 + s); perm = rng.permutation(len(yte)); cal, hold = perm[:NCAL], perm[NCAL:]
        for nh in head_sizes:
            hs = rng.choice(pool, nh, replace=False)
            Ph = clf.predict_proba(Xtr[hs]); T = fit_T(Ph, ytr[hs]); errh = (clf.classes_[Ph.argmax(1)] != ytr[hs]).astype(int)
            Sh, St = base_signals(clf, Xs, ys, Xtr[hs], T), base_signals(clf, Xs, ys, Xte, T)
            conf = dict(St["conf"])
            conf["f head(emb)"] = -head(Xtr[hs], errh, Xte, [1e-3, 1e-2, 1e-1, 1])
            conf["g head(a-e)"] = -head(Sh["F"], errh, St["F"], [1e-2, 1e-1, 1, 10, 100])
            for k in SIGS:
                r = eval_conf(conf[k], ok, cal, hold)
                for kk, v in r.items(): R[nh][k][kk].append(v)
            R[nh]["_T"]["T"].append(T)
            if s == 0 and nh == NHEAD:  # bootstrap CI of AUROC difference vs the gap on split 0's holdout
                bs = np.random.default_rng(0); y_err = ~ok[hold]; boot = {}
                B = [bs.integers(0, len(hold), len(hold)) for _ in range(1000)]
                for k in SIGS[1:]:
                    d = []
                    for b in B:
                        if y_err[b].all() or (~y_err[b]).all(): continue
                        d.append(roc_auc_score(y_err[b], -conf[k][hold][b]) - roc_auc_score(y_err[b], -conf["a gap"][hold][b]))
                    boot[k] = (np.percentile(d, 2.5), np.percentile(d, 97.5))
    for nh in head_sizes:
        r = R[nh]
        say(f"\n**{name}, heads trained on {nh} rows** (temperature fit T = {msd(r['_T']['T'], 2)}). AUROC = ranking errors above correct answers on the holdout; coverage = share of holdout answered by the student at the certified cutoff (certificate fit on {NCAL} cal rows, delta 5%); error = realized error among those answers. Cells are mean (min over splits) for coverage, mean (max) for error.")
        say("| signal | AUROC mean ± sd | cov @2% | err @2% | cov @5% | err @5% | cov @10% | err @10% |"); say("|---|---|---|---|---|---|---|---|")
        for k in SIGS:
            x = r[k]; say(f"| {k} | {msd(x['auc'])} | {mmin(x['cov0.02'])} | {mmax(x['err0.02'])} | {mmin(x['cov0.05'])} | {mmax(x['err0.05'])} | {mmin(x['cov0.1'])} | {mmax(x['err0.1'])} |")
        say(f"\nPaired difference vs the student's own gap (mean ± sd over {nsplit} splits; wins = splits where the signal is better):")
        say("| signal | dAUROC | AUROC wins | boot 95% CI dAUROC (split 0) | dcov @5% | wins | dcov @10% | wins |"); say("|---|---|---|---|---|---|---|---|")
        g = r["a gap"]
        for k in SIGS[1:]:
            x = r[k]; da = np.array(x["auc"]) - np.array(g["auc"]); d2 = np.array(x["cov0.05"]) - np.array(g["cov0.05"]); d5 = np.array(x["cov0.1"]) - np.array(g["cov0.1"])
            ci = boot[k] if nh == NHEAD else (np.nan, np.nan)  # bootstrap only for the main head size
            say(f"| {k} | {da.mean():+.4f} ± {da.std():.4f} | {(da > 0).sum()}/{nsplit} | {'-' if np.isnan(ci[0]) else f'[{ci[0]:+.4f}, {ci[1]:+.4f}]'} | {d2.mean():+.3f} ± {d2.std():.3f} | {(d2 > 0).sum()}/{nsplit} | {d5.mean():+.3f} ± {d5.std():.3f} | {(d5 > 0).sum()}/{nsplit} |")
    return R

def p1(nsplit, ncal=500):
    global NCAL
    NCAL = ncal; OUT.clear(); t0 = time.time(); b = load_banking77(); c = load_clinc150()
    hs = (1000, 300, 3000) if ncal == 500 else (1000,)  # the larger-calibration pass only repeats the main head size
    say(f"## S15 part 1: student-side confidence signals (frozen {EMB_MODEL} + LR student, C=10)")
    say("Signals (higher = more confident): a top-2 gap; b max prob; c -entropy; d top-2 gap after temperature scaling (T fit on the head-training rows); e share of the 10 nearest student-training rows (cosine) with the predicted label; f LR head on the z-scored prompt embedding predicting 'student is wrong'; g LR head on z-scored (a-e) features (+log gap, log temp-gap); h gap + kNN share (parameter-free sum).")
    student_block("Banking77, 770 rows", b["train"], b["test"], 770, nsplit, head_sizes=hs)
    student_block("Banking77, 3080 rows", b["train"], b["test"], 3080, nsplit, head_sizes=hs)
    student_block("CLINC150 (in-scope), 1500 rows", c["train"], c["test"], 1500, nsplit, head_sizes=(1000,))
    say(f"\nwall time {time.time()-t0:.0f}s")
    (HERE / (f"tables_p1_{nsplit}.md" if ncal == 500 else f"tables_p1_{nsplit}_cal{ncal}.md")).write_text("\n".join(OUT), encoding="utf-8")

# ============================================================ PART 2: router side
S5RESP = HERE.parent / "s5_tier_router" / "responses.jsonl"
BUD2 = (0.02, 0.05, 0.10, 0.15, 0.20)
FLOOR = -12.0  # logprob assumed for a competitor label the API did not return (all alternatives that are returned are above this)

def parse(txt, names):  # same as s5 run.py parse(): exact (case-insensitive) else nearest label
    t = (txt or "").strip().strip("`'\".").strip().lower(); low = {n.lower(): n for n in names}
    if t in low: return low[t], False
    if not t: return None, True
    return low[difflib.get_close_matches(t, list(low), n=1, cutoff=0.0)[0]], True

def label_tokens(rec, verbal):
    out = []
    for t in rec["toks"]:
        if re.fullmatch(r"<\|.*\|>", t[0].strip()): continue  # stop tokens
        if verbal and "|" in t[0]: break
        out.append(t)
    while out and not out[0][0].strip(): out = out[1:]
    return out

def lp_feats(rec, verbal, lab, names):
    """Per-answer logprob features from the returned token logprobs + top-5 alternatives.
    seq   = logprob of the label actually returned (sum over its tokens).   first = logprob of its first token.
    gap   = seq minus log(mass of the best competing branch): at each token position, an alternative token that still leads to some other listed
            label starts a competing branch whose mass (prefix logprob + alternative logprob) upper-bounds the best label in it. Can be negative when
            the returned label was not the model's favourite (gpt-5.x ignore temperature, so their answers are T=1 samples).
    mode  = sum of the top-token logprobs along the label up to and including the first position where the returned token is not the top token
            (the model's confidence in its own favourite path, independent of which token got sampled).
    mgap  = smallest (top token minus runner-up among tokens that still lead to a listed label) over those same positions; FLOOR if no runner-up was returned."""
    toks = label_tokens(rec, verbal)
    if not toks or lab is None: return (np.nan,) * 5
    lab_l = [n.lower() for n in names]; chosen = lab.lower(); cum = 0.0; prefix = ""; best = -np.inf
    mode = 0.0; mgap = np.inf; dev = False
    for tok, lp, alts in toks:
        cands = {tok: lp}
        for a, alp in alts:
            if a == tok: continue
            cand = (prefix + a).strip().lower()
            if not cand: continue
            m = [l for l in lab_l if l.startswith(cand)]
            if not m or m == [chosen]: continue
            best = max(best, cum + alp); cands[a] = alp
        if not dev:
            v = sorted(cands.values(), reverse=True); mode += v[0]; mgap = min(mgap, v[0] - (v[1] if len(v) > 1 else FLOOR))
            if v[0] > lp + 1e-9: dev = True
        cum += lp; prefix += tok
    return cum, toks[0][1], cum - max(best, FLOOR), mode, mgap

def parse_verbal(txt, names):
    m = re.match(r"\s*(.*?)\s*[|,;:]\s*(\d{1,3})", txt or "")
    if m: return parse(m.group(1), names)[0], float(min(100, int(m.group(2))))
    return parse(txt, names)[0], np.nan

def auc_ci(score, pos, nb=1000, seed=0):
    score, pos = np.asarray(score, float), np.asarray(pos, bool)
    if pos.all() or (~pos).all(): return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed); a = roc_auc_score(pos, score); d = []
    for _ in range(nb if nb > 1 else 0):
        b = rng.integers(0, len(pos), len(pos))
        if pos[b].all() or (~pos[b]).all(): continue
        d.append(roc_auc_score(pos[b], score[b]))
    return (a, np.percentile(d, 2.5), np.percentile(d, 97.5)) if d else (a, np.nan, np.nan)

def jitter_curve(score, mism, qs=(.25, .5, .75), reps=50):  # mismatch rate among the top q share, ties broken at random (averaged)
    rng = np.random.default_rng(0); out = np.zeros(len(qs))
    for _ in range(reps):
        o = np.lexsort((rng.random(len(score)), -score)); m = mism[o]
        out += [m[:int(q * len(m))].mean() for q in qs]
    return out / reps

def emp_cov(score, mism, b):  # descriptive, no guarantee: largest top-k (ties closed) with cumulative mismatch <= b on all rows
    o = np.argsort(-score, kind="stable"); s, m = score[o], mism[o]; cum = np.cumsum(m) / (np.arange(len(m)) + 1); best = 0
    for k in range(len(m), 0, -1):
        if k < len(m) and s[k] == s[k - 1]: continue
        if cum[k - 1] <= b: best = k; break
    return best / len(m)

def proj_cov(score, mism, b, ncal, grid=40, delta=0.05):  # certified share at calibration size ncal if the observed curve were the true curve
    o = np.argsort(-score, kind="stable"); s, m = score[o], mism[o]; n = len(m); cum = np.cumsum(m)
    for j in range(grid, 0, -1):
        k = max(1, int(round(n * j / grid)))
        while k < n and s[k] == s[k - 1]: k += 1
        nr = int(round(ncal * k / n)); er = int(round(cum[k - 1] / k * nr))
        if cp_upper(er, nr, delta / grid) <= b: return k / n
    return 0.0

def pdiff(s1, s2, pos, nb=1000, seed=1):
    rng = np.random.default_rng(seed); pos = np.asarray(pos, bool); d = []
    for _ in range(nb):
        b = rng.integers(0, len(pos), len(pos))
        if pos[b].all() or (~pos[b]).all(): continue
        d.append(roc_auc_score(pos[b], s1[b]) - roc_auc_score(pos[b], s2[b]))
    return roc_auc_score(pos, s1) - roc_auc_score(pos, s2), np.percentile(d, 2.5), np.percentile(d, 97.5)

def clean(v): return np.where(np.isnan(v), np.nanmin(v) - 1, v)

def p2():
    OUT.clear()
    from calls import b_data, RUNS, MODELS, load_cache
    names, texts, y = b_data(); N = len(texts); gold = np.array([names[k] for k in y], object)
    say("## S15 part 2: router-side confidence signals (cheap model vs gpt-5.4 low-effort 'E1', 200 Banking77 inputs from S5)")
    s5 = {}
    for l in S5RESP.read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        if r["err"] is None: s5.setdefault((r["cfg"], r["i"]), r)
    lab5 = lambda cfg: np.array([parse(s5[(cfg, i)]["content"], names)[0] for i in range(N)], object)
    E1, E1b, C1, C1b = lab5("E1"), lab5("E1b"), lab5("C1"), lab5("C1b")
    cE1 = np.mean([s5[("E1", i)]["usd"] for i in range(N)])
    agreedE = E1 == E1b
    say(f"\nE1 = openai/gpt-5.4 effort low (S5 responses, reused, not re-run). E1 gold acc {np.mean(E1==gold):.3f}, E1b {np.mean(E1b==gold):.3f}; **noise floor of the reference: E1 vs E1b agree on {agreedE.mean():.3f} of inputs ({(~agreedE).sum()} of {N} differ)**. E1 cost ${1000*cE1:.3f}/1k calls.")
    # ---- load new runs
    cache = load_cache(); R = collections.defaultdict(dict)
    for r in cache.values(): R[r["run"]][r["i"]] = r
    for run in RUNS: assert len(R[run]) == N, (run, len(R[run]))
    A = {}; cost = {}; served = {}
    for run, (mk, temp, verbal) in RUNS.items():
        recs = [R[run][i] for i in range(N)]; cost[run] = np.mean([r["usd"] for r in recs]); served[run] = collections.Counter(r["served_by"] for r in recs)
        if verbal: lc = [parse_verbal(r["content"], names) for r in recs]; lab = np.array([x[0] for x in lc], object); vc = np.array([x[1] for x in lc])
        else: lab = np.array([parse(r["content"], names)[0] for r in recs], object); vc = None
        f = np.array([lp_feats(recs[i], verbal, lab[i], names) for i in range(N)])
        A[run] = dict(lab=lab, vc=vc, lp_seq=f[:, 0], lp_first=f[:, 1], lp_gap=f[:, 2], lp_mode=f[:, 3], lp_mgap=f[:, 4], has_lp=np.isfinite(f[:, 0]),
                      pfail=np.array([parse(r["content"].split("|")[0] if verbal else r["content"], names)[1] for r in recs]))
    # ---- basic table
    say("\n### Cheap runs: accuracy, agreement with the reference, cost (every run is one call per input, provider pinned)")
    say("| run | model @ provider | temp sent | n | gold acc | match E1 | match E1b | match E1 on E1==E1b | rows with logprobs | parse fails | $/1k calls |"); say("|---|---|---|---|---|---|---|---|---|---|---|")
    for run, (mk, temp, verbal) in RUNS.items():
        a = A[run]; say(f"| {run} | {MODELS[mk][0]} @ {','.join(served[run])} | {temp} | {N} | {np.mean(a['lab']==gold):.3f} | {np.mean(a['lab']==E1):.3f} | {np.mean(a['lab']==E1b):.3f} | {np.mean((a['lab']==E1)[agreedE]):.3f} | {a['has_lp'].mean():.2f} | {a['pfail'].sum()} | {1000*cost[run]:.4f} |")
    say(f"| (S5) C1 nano reasoning off, no logprobs | openai/gpt-5.4-nano @ OpenAI | 0 | {N} | {np.mean(C1==gold):.3f} | {np.mean(C1==E1):.3f} | {np.mean(C1==E1b):.3f} | {np.mean((C1==E1)[agreedE]):.3f} | - | - | {1000*np.mean([s5[('C1',i)]['usd'] for i in range(N)]):.4f} |")
    say("\nNoise floor, same config twice (share of inputs with the same label):")
    pairs = [("E1", "E1b", E1, E1b), ("nano_A", "nano_Ab", A["nano_A"]["lab"], A["nano_Ab"]["lab"]), ("nano_A", "nano_S (T=0.7 sent)", A["nano_A"]["lab"], A["nano_S"]["lab"]), ("nano_Ab", "nano_S", A["nano_Ab"]["lab"], A["nano_S"]["lab"]),
             ("S5 C1", "S5 C1b", C1, C1b), ("nano_A", "S5 C1", A["nano_A"]["lab"], C1), ("nano_V", "nano_Vb", A["nano_V"]["lab"], A["nano_Vb"]["lab"]), ("nano_V", "nano_A", A["nano_V"]["lab"], A["nano_A"]["lab"]),
             ("g41_A", "g41_Ab", A["g41_A"]["lab"], A["g41_Ab"]["lab"]), ("g41_A", "g41_S (T=0.7)", A["g41_A"]["lab"], A["g41_S"]["lab"]), ("g41_Ab", "g41_S (T=0.7)", A["g41_Ab"]["lab"], A["g41_S"]["lab"]),
             ("qnx_A", "qnx_Ab", A["qnx_A"]["lab"], A["qnx_Ab"]["lab"]), ("qnx_A", "qnx_S (T=0.7)", A["qnx_A"]["lab"], A["qnx_S"]["lab"]), ("qnx_Ab", "qnx_S (T=0.7)", A["qnx_Ab"]["lab"], A["qnx_S"]["lab"])]
    say("| pair | agree | differ |"); say("|---|---|---|")
    for a, b, x, z in pairs: say(f"| {a} vs {b} | {np.mean(x==z):.3f} | {int((x!=z).sum())} |")
    # ---- student (shadows every call site: free) and prompt-embedding predictor
    bk = load_banking77(); Xtr, ytr = embed(bk["train"][0]), bk["train"][1]; X = embed(texts)
    st = LogisticRegression(C=10, max_iter=2000).fit(Xtr, ytr); Ps = st.predict_proba(X); sg = confidence(Ps); sp = np.array([names[k] for k in Ps.argmax(1)], object)
    say(f"\nStudent (frozen bge-small + LR, full Banking77 train, {len(ytr)} rows) on the same {N} inputs: gold acc {np.mean(sp==gold):.3f}, agrees with E1 on {np.mean(sp==E1):.3f}, E1b {np.mean(sp==E1b):.3f}.")
    say(f"Constants: competitor labels the API did not return are floored at logprob {FLOOR}; returned alternatives only count when they still lead to a different listed label.")

    def run_model(mk):
        a, cA = A[f"{mk}_A"], cost[f"{mk}_A"]; lab = a["lab"]
        S = {}  # name -> (score array or None, learn-features or None, pre-decision $ per input (None = decided before the cheap call), label array used)
        S["b1 returned-label logprob"] = (a["lp_seq"], None, cA, lab)
        S["b2 gap to best rival label"] = (a["lp_gap"], None, cA, lab)
        S["b3 first-token logprob"] = (a["lp_first"], None, cA, lab)
        S["b4 mode-path logprob (top tokens)"] = (a["lp_mode"], None, cA, lab)
        S["b5 mode gap (top vs runner-up)"] = (a["lp_mgap"], None, cA, lab)
        if mk == "nano":
            v = A["nano_V"]; S["d verbalized conf (own call)"] = (np.where(np.isnan(v["vc"]), np.nanmedian(v["vc"]), v["vc"]), None, cost["nano_V"], v["lab"])
        sc2 = (A[f"{mk}_A"]["lab"] == A[f"{mk}_S"]["lab"]).astype(float); sc3 = ((A[f"{mk}_A"]["lab"] == A[f"{mk}_Ab"]["lab"]).astype(float) + sc2) / 2
        S["c1 self-consistency (A==S)"] = (sc2, None, cA + cost[f"{mk}_S"], lab)
        S["c2 self-consistency of 3 (A,Ab,S)"] = (sc3, None, cA + cost[f"{mk}_Ab"] + cost[f"{mk}_S"], lab)
        S["c3 2-agree, then logprob (lexi)"] = (sc2 * 1000 + np.nan_to_num(a["lp_seq"], nan=-50), None, cA + cost[f"{mk}_S"], lab)
        S["e student top-2 gap"] = (sg, None, None, lab)
        agree_s = (sp == lab).astype(float)
        S["f1 student==cheap"] = (agree_s, None, cA, lab)
        S["f2 student==cheap, then student gap (lexi)"] = (agree_s * 10 + sg, None, cA, lab)
        S["f3 student==cheap, then logprob gap (lexi)"] = (agree_s * 1000 + np.nan_to_num(a["lp_gap"], nan=-50), None, cA, lab)
        S["a prompt-embedding predictor (LR)"] = (None, X, None, lab)
        Zall = np.column_stack([np.nan_to_num(a["lp_seq"], nan=-50), np.nan_to_num(a["lp_gap"], nan=-50), sc2, sg, agree_s])
        S["z all-signal LR (exploratory)"] = (None, Zall, cA + cost[f"{mk}_S"], lab)
        S["g random mixing"] = (np.random.default_rng(7).random(N), None, None, lab)
        return S, lab, cA

    def fit_learn(Z, tgt, tr, te):  # nested: OOF scores for the calibration rows, a model fit on all calibration rows for the holdout rows
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-6; Zt = (Z[tr] - mu) / sd; Ze = (Z[te] - mu) / sd; t = tgt[tr]
        if t.sum() < 3 or (~t).sum() < 3: return np.zeros(len(tr)), np.zeros(len(te))
        oof = np.zeros(len(tr))
        for a_, b_ in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(tr)), t):
            oof[b_] = LogisticRegression(C=1.0, max_iter=3000).fit(Zt[a_], t[a_]).predict_proba(Zt[b_])[:, 1]
        return oof, LogisticRegression(C=1.0, max_iter=3000).fit(Zt, t).predict_proba(Ze)[:, 1]

    def oof_all(Z, tgt):
        o = np.zeros(N)
        for a_, b_ in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(N), tgt):
            mu, sd = Z[a_].mean(0), Z[a_].std(0) + 1e-6
            o[b_] = LogisticRegression(C=1.0, max_iter=3000).fit((Z[a_] - mu) / sd, tgt[a_]).predict_proba((Z[b_] - mu) / sd)[:, 1]
        return o

    for mk, title in (("nano", "gpt-5.4-nano (reasoning off, logprobs)"), ("qnx", "qwen3-next-80b-a3b-instruct @ Parasail (substitute second cheap model: qwen3-32b returns no label logprobs)"), ("g41", "gpt-4.1-nano (extra: returns the full top-5 alternatives)")):
        S, lab, cA = run_model(mk); t1 = lab == E1; m2 = agreedE
        say(f"\n## Cheap model: {title}")
        say(f"cheap label == E1 on {t1.mean():.3f} of inputs (E1b {np.mean(lab==E1b):.3f}; on E1==E1b inputs {t1[m2].mean():.3f}; gold acc {np.mean(lab==gold):.3f}). Cost per 1000: cheap call ${1000*cA:.3f}, E1 ${1000*cE1:.3f}. A perfect router could send {t1.mean():.0%} of calls to the cheap model.")
        sc = {}
        for k, (s, Z, c_pre, labk) in S.items(): sc[k] = oof_all(Z, labk == E1) if s is None else s
        say("\n**AUROC of each signal for 'cheap answer == reference'** (95% bootstrap CI over inputs, 1000 resamples, for the E1 column). A signal that uses its own call (verbalized) is scored on that call's answer. 0.5 = random; with 200 inputs a difference below about 0.08 is within the CI.")
        say("| signal | extra cheap calls before deciding | AUROC vs E1 [95% CI] | AUROC vs E1b | AUROC on E1==E1b inputs | AUROC vs gold | rows with a score |"); say("|---|---|---|---|---|---|---|")
        for k, (s0, Z, c_pre, labk) in S.items():
            s = clean(sc[k]); a1, lo, hi = auc_ci(s, labk == E1); a2 = auc_ci(s, labk == E1b, nb=1)[0]; a3 = auc_ci(s[m2], (labk == E1)[m2], nb=1)[0]; a4 = auc_ci(s, labk == gold, nb=1)[0]
            extra = "none, decided before the cheap call" if c_pre is None else ("none" if c_pre < cA * 1.2 else f"{c_pre/cA - 1:.0f} extra")
            say(f"| {k} | {extra} | {a1:.3f} [{lo:.3f}, {hi:.3f}] | {a2:.3f} | {a3:.3f} | {a4:.3f} | {np.mean(~np.isnan(sc[k])):.2f} |")
        pos = lab == E1; pk = lambda p: next(k for k in S if k.startswith(p)); sv = {k: clean(sc[k]) for k in S}
        specs = [("f3", "b1"), ("f1", "b1"), ("f3", "f1"), ("b1", "d"), ("b1", "c1"), ("b1", "e "), ("b1", "a "), ("b1", "g "), ("b1", "b4"), ("b1", "b2")]
        say(f"\n**Paired bootstrap of AUROC differences** (same {N} inputs, 1000 resamples; common target '{mk}_A answer == E1', so the verbalized score from its own call is judged on this target too). A difference is real at this n only if the interval excludes 0.")
        say("| signal 1 minus signal 2 | AUROC diff | 95% CI |"); say("|---|---|---|")
        for p1_, p2_ in specs:
            try: k1, k2 = pk(p1_), pk(p2_)
            except StopIteration: continue
            d = pdiff(sv[k1], sv[k2], pos); say(f"| {k1} minus {k2} | {d[0]:+.3f} | [{d[1]:+.3f}, {d[2]:+.3f}] |")
        mism = (lab != E1).astype(float)
        say(f"\n**Descriptive risk-coverage on all {N} rows (no certificate; cutoffs chosen in-sample, so optimistic):** mismatch with E1 among the most-confident 25/50/75% of inputs (base rate {mism.mean():.3f}; random mixing gets the base rate at every share), the largest routed share whose mismatch stays within the budget, and the blended cost per 1000 calls at the 5% and 10% operating points (includes every extra cheap call; unrouted calls pay E1 on top).")
        say("| signal | mismatch top 25% | top 50% | top 75% | max share at <=2% | <=5% | <=10% | <=15% | $/1k at 5% share | $/1k at 10% share |"); say("|---|---|---|---|---|---|---|---|---|---|")
        for k in S:
            s = clean(sc[k]); mm = (S[k][3] != E1).astype(float); q = jitter_curve(s, mm); shs = [emp_cov(s, mm, b) for b in (.02, .05, .10, .15)]
            c_pre = S[k][2]; cst = lambda sh: 1000 * ((cA * sh) if c_pre is None else c_pre) + 1000 * (1 - sh) * cE1
            say(f"| {k} | {q[0]:.3f} | {q[1]:.3f} | {q[2]:.3f} | " + " | ".join(f"{x:.2f}" for x in shs) + f" | {cst(shs[1]):.2f} | {cst(shs[2]):.2f} |")
        nreq = {b: int(np.ceil(np.log(0.05 / 40) / np.log(1 - b))) for b in BUD2}
        say(f"\n**Certified routing (Clopper-Pearson, delta 5% over a 40-level grid), 5 random calibration/holdout half-splits (100/100 rows).** Zero-error rows needed to certify at all with delta/40: { {int(b*100): v for b, v in nreq.items()} } (the calibration half has 100 rows). cov = share of holdout inputs routed to the cheap model (mean, min over rotations); mism = realized mismatch with E1 among routed holdout inputs (mean, max over rotations that certified).")
        say("| signal | " + " | ".join(f"cov @{int(b*100)}% | mism" for b in BUD2) + " |"); say("|---|" + "---|" * (2 * len(BUD2)))
        costs = {}
        for k, (s0, Z, c_pre, labk) in S.items():
            ag = labk == E1; cells = []
            for b in BUD2:
                cov, mmx, usd = [], [], []
                for r in range(5):
                    perm = np.random.default_rng(200 + r).permutation(N); cal, hold = perm[:100], perm[100:]
                    if s0 is None: cc, ch = fit_learn(Z, ag, cal, hold)
                    else: v = clean(sc[k]); cc, ch = v[cal], v[hold]
                    c = certify(cc, ag[cal], b)
                    if c["cutoff"] == np.inf: cov.append(0.0); continue
                    h = apply_certificate(ch, ag[hold], c["cutoff"]); cov.append(h["coverage"])
                    if h["error"] is not None: mmx.append(h["error"])
                    sh = h["coverage"]; usd.append(1000 * ((cA * sh) if c_pre is None else c_pre) + 1000 * (1 - sh) * cE1)
                cells.append(f"{np.mean(cov):.2f} ({np.min(cov):.2f}) | {mmax(mmx) if mmx else '-'}"); costs[(k, b)] = (np.mean(usd) if usd else None, sum(1 for x in cov if x > 0))
            say(f"| {k} | " + " | ".join(cells) + " |")
        say(f"\nRandom mixing at any routed share has mismatch {mism.mean():.3f} among routed calls, so it cannot meet any of these budgets. Blended $/1k for the certified operating points above (mean over the rotations that certified; n = how many of 5 did); always-E1 is {1000*cE1:.2f}, always-cheap {1000*cA:.3f}:")
        say("| signal | " + " | ".join(f"$/1k @{int(b*100)}% (n)" for b in BUD2[2:]) + " |"); say("|---|" + "---|" * len(BUD2[2:]))
        for k in S:
            say(f"| {k} | " + " | ".join((f"{costs[(k, b)][0]:.2f} ({costs[(k, b)][1]})" if costs[(k, b)][0] is not None else "-") for b in BUD2[2:]) + " |")
        keys = [k for k in S if k.split()[0] in ("b1", "f1", "f3", "z", "e", "d")]
        say(f"\n**Projection: calibration rows needed.** If the {N}-row risk-coverage curve were the true curve (optimistic: it was read off the same rows), the certified share at calibration size n (same 40-level grid certificate) would be:")
        say("| signal | " + " | ".join(f"b={int(b*100)}%, n={n}" for b in (0.05, 0.10) for n in (200, 500, 1000, 2000)) + " |"); say("|---|" + "---|" * 8)
        for k in keys:
            s = clean(sc[k]); mm = (S[k][3] != E1).astype(float)
            say(f"| {k} | " + " | ".join(f"{proj_cov(s, mm, b, n):.2f}" for b in (0.05, 0.10) for n in (200, 500, 1000, 2000)) + " |")
        if mk == "nano":
            vv = np.where(np.isnan(A["nano_V"]["vc"]), np.nanmedian(A["nano_V"]["vc"]), A["nano_V"]["vc"]); a1, lo, hi = auc_ci(vv, A["nano_A"]["lab"] == E1)
            say(f"\nVerbalized confidence on the common target (nano_A answer == E1): AUROC {a1:.3f} [{lo:.3f}, {hi:.3f}]. Parsed confidences: {int((~np.isnan(A['nano_V']['vc'])).sum())}/{N}; most common values {dict(collections.Counter(A['nano_V']['vc'][~np.isnan(A['nano_V']['vc'])].astype(int).tolist()).most_common(6))}; nano_V label equals nano_A label on {np.mean(A['nano_V']['lab']==A['nano_A']['lab']):.3f} of inputs.")
    say(f"\n## Sensitivity: how good must the shadow student be? (the student here is trained on GOLD labels; E1 itself has gold acc {np.mean(E1==gold):.3f})")
    say("| student rows | student gold acc | agrees with E1 | AUROC student gap (vs nano_A==E1) | AUROC student==nano_A | lexi(student==nano_A, nano logprob gap) | AUROC student==qnx_A | lexi(student==qnx_A, qnx logprob gap) |"); say("|---|---|---|---|---|---|---|---|")
    for n in (77, 385, 770, 3080, len(ytr)):
        idx = stratified_subsample(ytr, n, seed=0); cl = LogisticRegression(C=10, max_iter=2000).fit(Xtr[idx], ytr[idx]); Pn = cl.predict_proba(X); gn = confidence(Pn); spn = np.array([names[k] for k in Pn.argmax(1)], object)
        cells = [f"{n}", f"{np.mean(spn==gold):.3f}", f"{np.mean(spn==E1):.3f}", f"{auc_ci(gn, A['nano_A']['lab']==E1, nb=1)[0]:.3f}"]
        for mk in ("nano", "qnx"):
            lb = A[f"{mk}_A"]["lab"]; ag = (spn == lb).astype(float)
            cells += [f"{auc_ci(ag, lb==E1, nb=1)[0]:.3f}", f"{auc_ci(ag*1000 + np.nan_to_num(A[f'{mk}_A']['lp_gap'], nan=-50), lb==E1, nb=1)[0]:.3f}"]
        say("| " + " | ".join(cells) + " |")
    (HERE / "tables_p2.md").write_text("\n".join(OUT), encoding="utf-8")

def rep():
    """Generation-noise floor for the signal comparison: the same signals scored on every independent draw of the same cheap model
    (A, Ab = same config twice; S = T=0.7 sent). Signal differences smaller than the spread across draws are not real."""
    OUT.clear()
    from calls import b_data, load_cache
    names, texts, y = b_data(); N = len(texts); s5 = {}
    for l in S5RESP.read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        if r["err"] is None: s5.setdefault((r["cfg"], r["i"]), r)
    E1 = np.array([parse(s5[("E1", i)]["content"], names)[0] for i in range(N)], object)
    cache = load_cache(); R = collections.defaultdict(dict)
    for r in cache.values(): R[r["run"]][r["i"]] = r
    bk = load_banking77(); Xtr, ytr = embed(bk["train"][0]), bk["train"][1]; X = embed(texts)
    Ps = LogisticRegression(C=10, max_iter=2000).fit(Xtr, ytr).predict_proba(X); sp = np.array([names[k] for k in Ps.argmax(1)], object); sg = confidence(Ps)
    say("## S15 same-config repeat: the same signals on each independent draw of the same cheap model (target: draw's label == E1)")
    say("A and Ab are the same request twice; S sends temperature 0.7. gpt-5.4-nano ignores temperature (every draw is a T=1 sample); gpt-4.1-nano and qwen3-next honour it. b1 = returned-label logprob, b2 = gap to best rival label, e = student gap (does not depend on the draw's label except through the target), f1 = student label == draw's label, f3 = f1 then b2 as tie-break.")
    say("| draw | match E1 | AUROC b1 | b2 | e student gap | f1 | f3 | f3 minus b1 | f1 minus b1 |"); say("|---|---|---|---|---|---|---|---|---|")
    for mk in ("nano", "qnx", "g41"):
        d3, d1 = [], []
        for run in (f"{mk}_A", f"{mk}_Ab", f"{mk}_S"):
            recs = [R[run][i] for i in range(N)]; lab = np.array([parse(r["content"], names)[0] for r in recs], object)
            f = np.array([lp_feats(recs[i], False, lab[i], names) for i in range(N)]); pos = lab == E1; au = lambda s: roc_auc_score(pos, clean(s))
            ag = (sp == lab).astype(float); b1, b2, f1, f3 = au(f[:, 0]), au(f[:, 2]), au(ag), au(ag * 1000 + np.nan_to_num(f[:, 2], nan=-50)); d3.append(f3 - b1); d1.append(f1 - b1)
            say(f"| {run} | {pos.mean():.3f} | {b1:.3f} | {b2:.3f} | {au(sg):.3f} | {f1:.3f} | {f3:.3f} | {f3 - b1:+.3f} | {f1 - b1:+.3f} |")
        say(f"| {mk}: mean over the 3 draws, range | | | | | | | {np.mean(d3):+.3f} [{min(d3):+.3f}, {max(d3):+.3f}] | {np.mean(d1):+.3f} [{min(d1):+.3f}, {max(d1):+.3f}] |")
    (HERE / "tables_rep.md").write_text("\n".join(OUT), encoding="utf-8")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "p1"
    if cmd == "p1": p1(int(sys.argv[2]) if len(sys.argv) > 2 else 5, int(sys.argv[3]) if len(sys.argv) > 3 else 500)
    elif cmd == "all": p2(); p1(5, 500); p1(5, 1500)
    elif cmd == "p2": p2()
    elif cmd == "rep": rep()
