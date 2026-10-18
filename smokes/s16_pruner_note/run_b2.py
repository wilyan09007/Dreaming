"""S16b v2: draft_note context pruner with the new equivalence definition (equiv.py), relabelled spans, recomputed floor, and a typed arm.
Usage:
  python run_b2.py --dry            cache-only plan/estimate: no network, no model load; scorer embeddings and cross-encoder are STUBS; writes results_b2_dry.json + log_b2_dry.txt
  python run_b2.py --tiny           a handful of prompts, live (relabel stage stays cache-only unless --live is also passed)
  python run_b2.py                  full live run (preflight = this script's own --dry estimate, hard spend cap --cap-usd, default 3.0)
Flags: --evidence core|loose (history-evidence policy, see equiv.py), --stages relabel,floor,cells,typed (subset), --strict-cells N (strict-rubric secondary column on the first N pruned cells; floor always),
       --no-ce, --adj-p, --cal-folds N (calibration folds per rotation, default 1),
       --live (REQUIRED for the relabel stage to make any new judge / note call, in every mode; without it relabel is cache-only and its missing pairs stay unlabelled).
Pool = the 140 labelled prompts (ids 0-79, 200-259); split = GroupKFold by vendor (5 folds, asserted vendor-disjoint). Rotation f: test fold f, calibration fold(s) f+1.., scorer trained on the remaining folds,
so the scorer never sees the labels of the folds it is calibrated or tested on and the cutoff for fold f never depends on fold f's labels (tiny: 3 folds).
Reference note for every comparison = A0 (the rep-0 full-prompt note). Floor = A0 vs A3 (rep 3, independent full-prompt note). Every API call is cached in responses.jsonl."""
import sys, json, math, time, argparse, subprocess
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
ap = argparse.ArgumentParser()
ap.add_argument("--dry", action="store_true"); ap.add_argument("--tiny", action="store_true"); ap.add_argument("--live", action="store_true")
ap.add_argument("--cap-usd", type=float, default=3.0); ap.add_argument("--evidence", choices=["core", "loose"], default="core")
ap.add_argument("--stages", default="relabel,floor,cells,typed"); ap.add_argument("--strict-cells", type=int, default=3)
ap.add_argument("--cal-folds", type=int, default=1); ap.add_argument("--no-ce", action="store_true"); ap.add_argument("--tag", default=""); ap.add_argument("--no-preflight", action="store_true"); ap.add_argument("--adj-p", type=float, default=0.13)
A = ap.parse_args(); STAGES = set(A.stages.split(","))
import equiv as E
E.setup(A.dry); E.ADJ_P = A.adj_p
import core, llm
from core import *
from common import embed, cp_upper, certify
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest

SUF = "_b2" + ("_dry" if A.dry else "_tiny" if A.tiny else "") + A.tag
T0 = time.time(); LOG = open(HERE / f"log{SUF}.txt", "w", encoding="utf-8")
def say(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
RES = {}
def save(): json.dump(RES, open(HERE / f"results{SUF}.json", "w"), indent=1, default=str)
VAL = ("SAME", "DIFFERENT"); dif = lambda v: v == "DIFFERENT"
ub = lambda k, n: cp_upper(int(k), int(n), 0.05) if n else float("nan")
POL = f"evidence={A.evidence}"
LIVE = not A.dry
if LIVE: llm.CAP = llm.SPENT[0] + A.cap_usd; LEDGER0 = llm.SPENT[0]
left = lambda: (llm.CAP - llm.SPENT[0]) if LIVE else float("inf")

# ------------------------------------------------------------------ observed prices (responses.jsonl is the ledger) and need accounting
def prices():
    by = defaultdict(lambda: [0, 0.0])
    for r in llm.R.values():
        u = r["resp"].get("usage") or {}; by[r["model"]][0] += 1; by[r["model"]][1] += u.get("cost") or 0
    pr = {m: v[1] / v[0] for m, v in by.items() if v[0]}
    tot = sum(v[1] for v in by.values()); n = sum(v[0] for v in by.values())
    try:
        last = [l for l in open(HERE / "log.txt", encoding="utf-8") if l.startswith("[spend] end")][-1]
        lg = f"log.txt: {last.strip()}"
    except Exception: lg = "log.txt: n/a"
    return dict(teacher=pr.get(HAIKU, 0.00016), judge1=pr.get(SONNET, 0.0013), judge2=pr.get(MINI, 0.00028), ledger_per_call=tot / n, ledger_total=tot, calls=n, log_line=lg)
PR = prices(); STRICT_CALLS = 1.6   # expected judge1 calls per pair for the strict secondary column (raw + confirmation of the ~40% DIFFERENT)
NEEDS = defaultdict(lambda: defaultdict(float)); COVER = defaultdict(lambda: defaultdict(int))   # stage -> kind -> n
def need(stage, kind, n=1.0): NEEDS[stage][kind] += n
def cover(stage, kind, n=1): COVER[stage][kind] += n
def usd(d): return d.get("teacher", 0) * PR["teacher"] + d.get("judge1", 0) * PR["judge1"] + (d.get("judge2", 0)) * PR["judge2"]

# ------------------------------------------------------------------ pool, folds
labs = {int(k): v for k, v in json.load(open(HERE / "labels.json")).items()}
LAB = [0, 1, 2, 3, 4, 5, 200, 201, 202, 203, 204, 205] if A.tiny else sorted(labs)
K = 3 if A.tiny else 5
FOLD = {}
for f, (_, te) in enumerate(GroupKFold(K).split(LAB, groups=[P[i]["vendor"] for i in LAB])):
    for a in te: FOLD[LAB[a]] = f
_vf = defaultdict(set)
for i in LAB: _vf[P[i]["vendor"]].add(FOLD[i])
assert all(len(v) == 1 for v in _vf.values()), "a vendor landed in two folds"
NCAL = max(1, min(A.cal_folds, K - 2))
CALF = {f: [(f + j) % K for j in range(1, NCAL + 1)] for f in range(K)}              # calibration folds of rotation f
TRF = {f: [g for g in range(K) if g != f and g not in CALF[f]] for f in range(K)}     # scorer-training folds of rotation f (disjoint from test and calibration)
assert all(TRF[f] for f in range(K))
ROTS = {i: [FOLD[i]] + [f for f in range(K) if FOLD[i] in CALF[f]] for i in LAB}      # rotations in which prompt i is scored: first = its own test rotation
REF = {i: labs[i]["refs"] for i in LAB}; TOK = {i: labs[i]["tok_full"] for i in LAB}
FLOORIDS = LAB if A.tiny else sorted(set(LAB) | set(range(200, 300)))
say(f"=== S16b v2 {'DRY (cache-only, scorer embedding + cross-encoder are STUBS)' if A.dry else 'TINY' if A.tiny else 'FULL'} | pool {len(LAB)} prompts, {len(set(P[i]['vendor'] for i in LAB))} vendors, {K} vendor-grouped folds | {POL} | stages {sorted(STAGES)} | cap ${A.cap_usd}")
say(f"[prices] observed $/call from the response ledger: teacher {PR['teacher']:.5f}, judge1 judge {PR['judge1']:.5f}, judge2 judge {PR['judge2']:.5f}; overall {PR['ledger_per_call']:.5f}/call over {PR['calls']} calls (${PR['ledger_total']:.2f}); {PR['log_line']}")
if LIVE and not A.no_preflight:
    cmd = [sys.executable, str(HERE / "run_b2.py"), "--dry", "--tag", "_pre", "--cal-folds", str(A.cal_folds), "--evidence", A.evidence, "--stages", A.stages, "--strict-cells", str(A.strict_cells), "--adj-p", str(A.adj_p)] + (["--tiny"] if A.tiny else [])
    prf = HERE / "results_b2_dry_pre.json"; prf.unlink(missing_ok=True)
    rc = subprocess.run(cmd, capture_output=True, text=True)
    if rc.returncode != 0 or not prf.exists():
        print(rc.stdout[-2000:]); print(rc.stderr[-4000:], file=sys.stderr); raise SystemExit(f"[preflight] the --dry estimate subprocess failed (rc={rc.returncode}); refusing to start a live run on a stale or missing estimate")
    plan_ = json.load(open(prf))["plan"]; pre = dict(usd=sum(v["usd"] for k, v in plan_.items() if k != "TOTAL" and (A.live or k != "relabel")), calls=plan_["TOTAL"]["calls"])
    say(f"[preflight] cache-only estimate of the work still to buy: {pre['calls']} (relabel {'included' if A.live else 'EXCLUDED: needs --live'}) => ~${pre['usd']:.2f} (hard cap ${A.cap_usd}; stages/cells that no longer fit are skipped, never crash the run)")
    if pre["usd"] > A.cap_usd: say("[preflight] WARNING estimate exceeds the cap: lower-priority stages/cells will be skipped")
RES["setup"] = dict(mode="dry" if A.dry else "tiny" if A.tiny else "full", n=len(LAB), evidence=A.evidence, folds=K, prices=PR, stages=sorted(STAGES))

# ------------------------------------------------------------------ helpers: reporting
def vboot(vals, ids, B=2000):
    g = {}
    for v, i in zip(vals, ids): g.setdefault(P[i]["vendor"], []).append(float(v))
    if len(g) < 2: return [float("nan")] * 2
    s = np.array([sum(x) for x in g.values()]); c = np.array([len(x) for x in g.values()]); d = np.random.default_rng(0).integers(0, len(s), (B, len(s))); e = s[d].sum(1) / c[d].sum(1)
    return [float(np.percentile(e, 2.5)), float(np.percentile(e, 97.5))]
def rate_of(rows, key="v"):
    """DIFFERENT rate over parseable rows; FAIL / unanswerable are excluded and reported"""
    v = [r[key] for r in rows]; n = sum(x in VAL for x in v); k = sum(x == "DIFFERENT" for x in v)
    return dict(n_rows=len(v), n=n, k=k, rate=(k / n if n else None), ub=(ub(k, n) if n else None), fail=sum(x == "FAIL" for x in v), unknown=sum(x is None for x in v), policy=A.evidence)
def fmt(d):
    r = "n/a" if d["rate"] is None else f"{d['k']}/{d['n']} = {d['rate']:.3f} (CP UB {d['ub']:.3f})"
    return f"{r} | FAIL {d['fail']} (excluded) | not answerable {d['unknown']} | policy {d['policy']}"
def pmapx(fn, xs, w=12): return pmap(fn, xs, w if (LIVE and not A.tiny) else 1 if A.dry else 4)

# ------------------------------------------------------------------ 0. extractor sanity on the reference notes (free): how often can canon state each required field?
def stage0():
    Cn = E.C; gts = {i: Cn.gt_of(P[i]) for i in LAB}; tot = Counter(); ok = Counter(); st = Counter()
    close = lambda a, b: a is not None and b is not None and abs(a - b) < 0.005
    for i in LAB:
        c = Cn.canon(REF[i][0], P[i]["tx"]); g = gts[i]; tot["n"] += 1
        for f, s_, o_ in (("flag", c["flag"] is not None, c["flag"] == g["flag"]), ("tx_amt", c["tx_amt"] is not None, close(c["tx_amt"], g["tx_amt"])), ("tx_date", c["tx_date"] is not None, c["tx_date"] == g["tx_date"]),
                          ("inv", c["inv_id"] is not None, c["inv_id"] == g["inv_id"]), ("reason", c["reason"] not in (None, ("other",)), c["reason"] == g["reason"])):
            st[f] += s_; ok[f] += s_ and o_
    say(f"[extractor on the {len(LAB)} A0 notes] stated / correct-among-stated: " + ", ".join(f"{f} {st[f]}/{tot['n']} / {ok[f]}/{st[f]}" for f in st))
    RES["extractor"] = dict(n=tot["n"], stated=dict(st), correct=dict(ok))
stage0()

# ------------------------------------------------------------------ 2. relabel spans with the new equivalence
def span_new(p, nd, nb=NB):
    """per-span 0/1/None labels from new-equivalence ablation outcomes nd[name] in {True, False, None}; same logic as core.span_labels"""
    n = len(p["hist"]); fx = {f: (None if nd.get(f) is None else int(nd[f])) for f in FIELDS}; h0 = nd.get("H0")
    if h0 is None: return dict(fixed=fx, hist=[None] * n, h0=None, dist=None)
    if not h0: return dict(fixed=fx, hist=[0] * n, h0=0, dist=0)
    bl = [nd.get(f"B{b}") for b in range(nb)]
    if any(x is True for x in bl): hist = [None if bl[block_of(p, j)] is None else int(bl[block_of(p, j)]) for j in range(n)]; dist = 0
    elif all(x is False for x in bl): hist = [1] * n; dist = 1                    # evidence spread over the history: every record may be needed
    else: hist = [None] * n; dist = None                                          # blocks not (yet) answerable
    return dict(fixed=fx, hist=hist, h0=1, dist=dist)
ND, SLN, RLB = {}, {}, []
def relabel_one(i):
    p = P[i]; lab = labs[i]; A0 = lab["refs"][0]; nd = {}; rows = []
    def ev(nm, x):
        r = E.equiv(p, A0, x, A.evidence); nd[nm] = None if r["v"] not in VAL else dif(r["v"])
        rows.append(dict(i=i, name=nm, v=r["v"], path=r["path"], old=lab["var"][nm]["cons"] if nm in lab["var"] else None, need=r["need"]))
    for nm, v in lab["var"].items():
        if not nm.startswith("B"): ev(nm, v["note"])
    if nd.get("H0") is not False:                                  # cached block notes are judged only when the new H0 label is DIFFERENT or unknown (span_new ignores them otherwise)
        for nm, v in lab["var"].items():
            if nm.startswith("B"): ev(nm, v["note"])
    if nd.get("H0") is not False and not all(f"B{b}" in lab["var"] for b in range(NB)):   # new H0 DIFFERENT (or unknown): history blocks are needed and were only cached when the OLD H0 changed
        for b in range(NB):
            if f"B{b}" in lab["var"]: continue
            n = E.note_c(p, minus(p, hist=set(blk(p, b))), 0)
            if n is None: nd[f"B{b}"] = None; rows.append(dict(i=i, name=f"B{b}", v=None, path="note-missing", old=None, need=dict(judge1=0, judge2=0, adj_exp=0.0), note_missing=True))
            else: ev(f"B{b}", n["text"])
    return i, nd, rows
def stage_relabel():
    E.STATE["net"] = bool(A.live and LIVE)                          # new relabel judge / block-note calls ONLY with --live, in every mode
    t = time.time()
    for i, nd, rows in pmapx(relabel_one, LAB): ND[i] = nd; RLB.extend(rows); SLN[i] = span_new(P[i], nd)
    E.STATE["net"] = LIVE
    paths = Counter(r["path"] for r in RLB); known = [r for r in RLB if r["v"] in VAL]
    for r in RLB:
        need("relabel", "judge1", r["need"]["judge1"]); need("relabel", "judge2", r["need"]["judge2"] + r["need"]["adj_exp"])
        if r.get("note_missing"): need("relabel", "teacher", 1)
        cover("relabel", "pairs"); cover("relabel", "answerable", r["v"] in VAL)
    # expected block notes for H0-unknown prompts and expected judge calls for those blocks are small; counted as expected below
    h0 = [nd.get("H0") for nd in ND.values()]; h0k = [x for x in h0 if x is not None]
    frac = (sum(h0k) / len(h0k)) if h0k else 0.2; n_unk = sum(x is None for x in h0)
    need("relabel", "teacher", n_unk * frac * 3 * 0.9); need("relabel", "judge1", n_unk * frac * 3 * 0.9 * 0.5 * 2)
    say(f"\n=== stage relabel: {len(LAB)} prompts, {len(RLB)} ablation pairs (variant note vs A0), {POL}")
    say(f"[relabel paths] {dict(paths)} ; answerable now {len(known)}/{len(RLB)} ; FAIL {sum(r['v'] == 'FAIL' for r in RLB)} (excluded)")
    for nm in ("tx", "inv", "flag", "bank", "H0", "B0", "B1", "B2"):
        rs = [r for r in RLB if r["name"] == nm]
        if not rs: continue
        k = [r for r in rs if r["v"] in VAL]; ol = [r for r in rs if r["old"] is not None]
        say(f"   {nm:5s} pairs {len(rs):3d} answerable {len(k):3d} new DIFFERENT {sum(dif(r['v']) for r in k):3d} ({(sum(dif(r['v']) for r in k)/len(k) if k else float('nan')):.2f}) | "
            f"old strict-confirmed DIFFERENT on the same answerable pairs {sum(r['old'] != 'SAME' for r in k if r['old'] is not None)}/{len([r for r in k if r['old'] is not None])} | by canon: " +
            ", ".join(f"{p_} {c}" for p_, c in sorted(Counter(r["path"] for r in rs).items())))
    fxs = {f: [SLN[i]["fixed"][f] for i in LAB if SLN[i]["fixed"][f] is not None] for f in FIELDS}
    h0l = [SLN[i]["h0"] for i in LAB if SLN[i]["h0"] is not None]
    hist_known = [x for i in LAB for x in SLN[i]["hist"] if x is not None]
    old = {f: float(np.mean([SL_OLD[i]["fixed"][f] for i in LAB])) for f in FIELDS}
    say("[new span labels] " + ", ".join(f"{f} {np.mean(v):.2f} (n={len(v)}; old {old[f]:.2f})" for f, v in fxs.items()) + f" | H0 changes note {np.mean(h0l) if h0l else float('nan'):.2f} (n={len(h0l)}; old {np.mean([SL_OLD[i]['h0'] for i in LAB]):.2f}) | "
        f"history-record positives {np.mean(hist_known) if hist_known else float('nan'):.3f} over {len(hist_known)} labelled records (old {np.mean([x for i in LAB for x in SL_OLD[i]['hist']]):.3f}) | prompts needing history (new / old): "
        f"{sum(1 for i in LAB if SLN[i]['h0'] == 1)} / {sum(SL_OLD[i]['h0'] for i in LAB)}; prompts with H0 not answerable {sum(SLN[i]['h0'] is None for i in LAB)}; H0-changed prompts whose blocks are not answerable {sum(1 for i in LAB if SLN[i]['h0'] == 1 and SLN[i]['dist'] is None)}")
    RES["relabel"] = dict(pairs=len(RLB), answerable=len(known), paths=dict(paths), policy=A.evidence, fail=sum(r["v"] == "FAIL" for r in RLB),
                          new_fixed={f: (float(np.mean(v)) if v else None) for f, v in fxs.items()}, old_fixed=old, h0_new=(float(np.mean(h0l)) if h0l else None), h0_n=len(h0l),
                          needs=dict(NEEDS["relabel"]), per_name={nm: dict(pairs=sum(r["name"] == nm for r in RLB), known=sum(r["name"] == nm and r["v"] in VAL for r in RLB)) for nm in ("tx", "inv", "flag", "bank", "H0", "B0", "B1", "B2")})
    save(); say(f"[spend] relabel: {time.time()-t:.0f}s" + (f", ${llm.SPENT[0]-LEDGER0:.3f} spent" if LIVE else ""))
SL_OLD = {i: span_labels(P[i], labs[i]) for i in LAB}
if "relabel" in STAGES:
    try: stage_relabel()
    except RuntimeError as e:
        E.STATE["capped"] = E.STATE["capped"] or "spend cap" in str(e); say(f"[SKIP] relabel stopped early: {e} -- labels so far are used, unanswerable ones stay masked"); E.STATE["net"] = LIVE
    for i in LAB:
        if i not in SLN: SLN[i] = span_new(P[i], ND.get(i, {}))
else:
    for i in LAB: SLN[i] = SL_OLD[i]
    say("[relabel] stage not selected: the OLD strict-rubric span labels are used for the scorers")

# ------------------------------------------------------------------ scorers (out-of-fold by vendor), trained on the new labels
STUB = A.dry
def get_emb():
    texts = [P[i]["tx"] for i in LAB] + [h["text"] for i in LAB for h in P[i]["hist"]]
    if A.dry: return np.zeros((len(texts), 384), np.float32), "STUB(dry: zero embeddings, cos feature = 0)"
    try: return embed(texts), "bge-small-en-v1.5"
    except Exception as e: say(f"[WARN] embedding model unavailable ({type(e).__name__}); cos feature set to 0 and MARKED STUB"); return np.zeros((len(texts), 384), np.float32), "STUB(embedding unavailable)"
ET_, EMBTAG = get_emb(); ET = {i: ET_[k] for k, i in enumerate(LAB)}; off = len(LAB); EH = {}
for i in LAB: n = len(P[i]["hist"]); EH[i] = ET_[off:off + n]; off += n
NF = 10
def feats(i):
    p = P[i]; n = len(p["hist"]); fx = np.zeros((4, NF)); fx[range(4), range(4)] = 1; hs = np.zeros((n, NF)); hs[:, 4] = 1
    for j, h in enumerate(p["hist"]):
        rel = abs(h["amt"] - p["tx_amt"]) / p["tx_amt"]
        hs[j, 5] = min(rel, 5); hs[j, 6] = float(rel < .01 and h["days"] <= 7); hs[j, 7] = float(h["days"] == 0); hs[j, 8] = math.log1p(h["days"]); hs[j, 9] = float(EH[i][j] @ ET[i])
    f = p["flag"]; c = [("duplicate" in f), ("first purchase" in f), ("unusually high" in f), ("no matching invoice" in f)]; ctx = np.array(c + [not any(c)], float)
    return np.c_[fx, np.zeros((4, 5))], np.c_[hs, np.tile(ctx, (n, 1))]
XS = {i: feats(i) for i in LAB}
def stack(ids):
    X, y = [], []
    for i in ids:
        fx, hs = XS[i]; yy = [SLN[i]["fixed"][f] for f in FIELDS] + list(SLN[i]["hist"]); XX = np.vstack([fx, hs])
        keep = [k for k, v in enumerate(yy) if v is not None]; X.append(XX[keep]); y += [yy[k] for k in keep]   # unanswerable labels are masked, never trained as 0
    return (np.vstack(X) if X else np.zeros((0, NF))), np.array(y, int)
PFr, PHr = {}, {}; LBL_N = {}      # rotation f -> {prompt: scores}; scorer f is trained on TRF[f] only and applied to the test fold f and the calibration fold(s) CALF[f]
for f in range(K):
    X, y = stack([i for i in LAB if FOLD[i] in TRF[f]]); LBL_N[f] = (len(y), int(y.sum()) if len(y) else 0)
    m_ = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X, y) if len(set(y.tolist())) > 1 else None
    PFr[f], PHr[f] = {}, {}
    for i in LAB:
        if FOLD[i] != f and FOLD[i] not in CALF[f]: continue
        if m_ is None: pr = float(y.mean()) if len(y) else 0.5; PFr[f][i], PHr[f][i] = np.full(4, pr), np.full(len(P[i]["hist"]), pr)
        else: PFr[f][i], PHr[f][i] = m_.predict_proba(XS[i][0])[:, 1], m_.predict_proba(XS[i][1])[:, 1]
dys = lambda i: np.array([h["days"] for h in P[i]["hist"]], float)
CE = {}
if A.dry or A.no_ce:
    for i in LAB: CE[i] = np.zeros(len(P[i]["hist"]))
    CE_TAG = "STUB(dry)" if A.dry else "off"
else:
    try:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
        for i in LAB: CE[i] = ce.predict([(P[i]["tx"], h["text"]) for h in P[i]["hist"]], batch_size=64, show_progress_bar=False)
        CE_TAG = "ms-marco-MiniLM-L-6-v2"
    except Exception as e:
        say(f"[WARN] cross-encoder unavailable ({type(e).__name__}); cross-encoder arms are skipped"); CE_TAG = "off"
        for i in LAB: CE[i] = np.zeros(len(P[i]["hist"]))
rng = np.random.default_rng(16); RS, RC = {}, {}
for i in LAB: RS[i] = rng.random(len(P[i]["hist"])); RC[i] = float(rng.random())
lab0 = lambda x: 0 if x is None else x       # oracle ranks unanswerable labels as negative (marked in the report)
SCORE = {"learned": lambda i, r: PHr[r][i], "cross-encoder": lambda i, r: CE[i], "oracle": lambda i, r: 1000.0 * np.array([lab0(x) for x in SLN[i]["hist"]]) - dys(i), "random": lambda i, r: RS[i], "recency": lambda i, r: -dys(i)}
topm = lambda sc, i, m, r: tuple(sorted(int(j) for j in np.argsort(-SCORE[sc](i, r), kind="stable")[:m]))
def conf(sc, i, kept, r):          # r = rotation whose scorer produced the scores (the prompt's own fold for test rows, a neighbouring fold's rotation for calibration rows)
    n = len(P[i]["hist"]); Dr = [j for j in range(n) if j not in kept]
    if not Dr and sc in ("learned", "cross-encoder"): return 1e9
    if sc == "learned": return float(1 - PHr[r][i][Dr].max())
    if sc == "cross-encoder": return float(-CE[i][Dr].max())
    if sc == "oracle": pos = [j for j in range(n) if lab0(SLN[i]["hist"][j])]; return sum(j in kept for j in pos) / len(pos) if pos else 1.0
    if sc == "random": return RC[i]
    if sc == "rule": return float(1 - max(PFr[r][i][3], PHr[r][i].max()))
def keepfn(sc, m):
    def f(i, r):
        k = keep_all(P[i])
        if sc == "rule": k["bank"] = 0; k["hist"] = ()
        else: k["hist"] = topm(sc, i, m, r)
        return k
    return f
D = {}
for sc in ("learned", "cross-encoder", "recency", "random"):
    mixed = [i for i in LAB if all(x is not None for x in SLN[i]["hist"]) and 0 < sum(SLN[i]["hist"]) < len(SLN[i]["hist"])]
    withpos = [i for i in LAB if all(x is not None for x in SLN[i]["hist"]) and sum(SLN[i]["hist"]) > 0]
    d = dict(within_auc=float(np.mean([roc_auc_score(SLN[i]["hist"], SCORE[sc](i, FOLD[i])) for i in mixed])) if mixed else None, n_mixed=len(mixed), n_withpos=len(withpos))
    D[sc] = d; say(f"[scorer vs NEW history labels, vendor-OOF] {sc:14s} " + " ".join(f"{k} {round(v,3) if isinstance(v,float) else v}" for k, v in d.items()) + ("  <- STUB scores" if sc == "cross-encoder" and CE_TAG != "ms-marco-MiniLM-L-6-v2" else ""))
RES["scorers"] = dict(embedding=EMBTAG, cross_encoder=CE_TAG, train_label_counts_per_rotation={f: dict(rows=a, positives=b) for f, (a, b) in LBL_N.items()}, diag=D); save()
say(f"[scorers] embedding: {EMBTAG}; cross-encoder: {CE_TAG}; masked-label training rows per rotation (rows, positives): {LBL_N}; rotation f = test fold f, calibration fold(s) {CALF}, scorer trained on folds {TRF}")

# ------------------------------------------------------------------ guards (a spend-cap RuntimeError from llm.chat must never kill the run)
RES["scorers"]["rotations"] = dict(calibration_folds=CALF, scorer_train_folds=TRF, note="rotation f: test fold f, calibration CALF[f], scorer trained on TRF[f]; all vendor-disjoint")
def guarded(label, fn, *a):
    try: return fn(*a)
    except RuntimeError as e:
        if "spend cap" in str(e): E.STATE["capped"] = True
        say(f"[SKIP] {label}: {e} -- partial results saved, continuing with whatever still fits"); save(); return None
def typed_key(p, k, rep=0): return E.ckey(core.HAIKU, E.SYS_TY, core.render(p, k), 200, core.EX_T, rep)
def typed_cost(keys):
    n = sum(1 for key in keys if key not in llm.R); return n, n * PR["teacher"]

# ------------------------------------------------------------------ 3. floor under the same equivalence
FL, FLT = {}, {}
RESID = dict(floor=None, abl=None)
def floor_one(i):
    p = P[i]; k = keep_all(p); a0 = E.note_c(p, k, 0); a3 = E.note_c(p, k, 3)
    if a0 is None or a3 is None: return dict(i=i, v=None, miss=int(a3 is None) + int(a0 is None), need=dict(judge1=0, judge2=0, adj_exp=0.0), path="note-missing", sraw=None, scons=None)
    r = E.equiv(p, a0["text"], a3["text"], A.evidence); rows = [E.note_c(p, k, q) for q in (1, 2)]
    s = E.strict([a0["text"]] + [x["text"] if x else None for x in rows], a3["text"]) if all(rows) else dict(raw=None, cons=None, need=0, exp=0.3)
    return dict(i=i, v=r["v"], path=r["path"], need=r["need"], miss=0, sraw=s["raw"], scons=s["cons"], sneed=s["need"], tok=a3["tok"])
def stage_floor():
    E.STATE["net"] = LIVE
    nmiss = sum(not E.note_cached(P[i], keep_all(P[i]), 3) for i in FLOORIDS); est = nmiss * (PR["teacher"] + 0.4 * (2 * PR["judge1"] + A.adj_p * PR["judge2"]) + STRICT_CALLS * PR["judge1"])
    if LIVE and est > left() - 0.03: say(f"[floor] SKIPPED: needs ~${est:.2f} for {nmiss} rep-3 notes + judging, ${left():.2f} left under the cap"); return
    for r in pmapx(floor_one, FLOORIDS): FL[r["i"]] = r
    rows = list(FL.values())
    for r in rows:
        need("floor", "teacher", r["miss"]); need("floor", "judge1", r["need"]["judge1"] + r.get("sneed", 0)); need("floor", "judge2", r["need"]["judge2"] + r["need"]["adj_exp"]); cover("floor", "pairs"); cover("floor", "answerable", r["v"] in VAL)
    # expected judge calls for pairs whose notes are missing: use the canon residual share measured on the pairs we do have
    have = [r for r in rows if r["path"] != "note-missing"]; nm = len(rows) - len(have)
    rr = np.mean([r["path"].startswith("judge") for r in have]) if have else 0.5; RESID["floor"] = float(rr)
    need("floor", "judge1", nm * (rr * 2 + STRICT_CALLS)); need("floor", "judge2", nm * rr * A.adj_p)
    paths = Counter(r["path"] for r in rows)
    say(f"\n=== stage floor: A0 vs A3 (independent full-prompt notes), {len(rows)} prompts ({len(have)} with both notes cached; {nm} need the rep-3 note), {POL}")
    say(f"[floor paths] {dict(paths)}")
    for nm_, ids in (("pool 140 (labelled)", [i for i in LAB]), ("original floor ids 200-299", [i for i in FLOORIDS if 200 <= i < 300]), ("all floor prompts", FLOORIDS)):
        rs = [FL[i] for i in ids if i in FL]; d = rate_of(rs); s_raw = rate_of(rs, "sraw"); s_con = rate_of(rs, "scons")
        say(f"[floor {nm_}] NEW equivalence DIFFERENT {fmt(d)}")
        say(f"      strict rubric (secondary, same pairs): raw {fmt(s_raw)} | confirmed {fmt(s_con)}")
        RES.setdefault("floor", {})[nm_] = dict(new=d, strict_raw=s_raw, strict_cons=s_con)
    for s in sorted({P[i]["scn"] for i in LAB}):
        rs = [FL[i] for i in LAB if i in FL and P[i]["scn"] == s]; d = rate_of(rs); say(f"      by scenario {s:7s} n_rows={len(rs):3d} " + fmt(d))
    save()
def typed_pair(i, k, ref_rep=0, x_rep=0):
    """typed record of the pruned prompt vs the typed record of the FULL prompt (rep ref_rep); den = the typed full-prompt token count (the typed arm's own denominator)"""
    p = P[i]; ref = E.typed_c(p, keep_all(p), ref_rep); x = E.typed_c(p, k, x_rep)
    rm, xm = int(ref is None), int(x is None); miss = rm + xm
    if miss: return dict(i=i, v=None, miss=miss, xmiss=xm, tok=None if x is None else x["tok"], den=None if ref is None else ref["tok"])
    if ref["rec"] is None or x["rec"] is None: return dict(i=i, v="FAIL", miss=0, xmiss=0, tok=x["tok"], den=ref["tok"], diff=[])
    d = E.rec_diff(ref["rec"], x["rec"])
    return dict(i=i, v="DIFFERENT" if d else "SAME", miss=0, xmiss=0, tok=x["tok"], den=ref["tok"], diff=d, rec=x["rec"], note=E.render_template(x["rec"]), ref_note=E.render_template(ref["rec"]))
def typed_samples(name, rows, n=3):
    got = [r for r in rows if r.get("note")][:n]
    RES.setdefault("typed_samples", {})[name] = [dict(i=r["i"], v=r["v"], diff=r.get("diff"), ref_note=r["ref_note"], note=r["note"]) for r in got]
    for r in got[:1]: say(f"      typed note sample [{name}] prompt {r['i']} ({r['v']}, differing {r.get('diff')}): ref: {r['ref_note']} || pruned: {r['note']}")
def stage_typed_floor():
    # reference = typed record rep 0 on the full prompt; floor = typed record rep 1 on the full prompt
    keys = [typed_key(P[i], keep_all(P[i]), q) for i in LAB for q in (0, 1)]; nn, est = typed_cost(keys)
    if LIVE and est > left() - 0.03: say(f"[typed floor] SKIPPED: needs ~${est:.2f} for {nn} typed records, ${left():.2f} left under the cap"); return
    for r in pmapx(lambda i: typed_pair(i, keep_all(P[i]), 0, 1), LAB): FLT[r["i"]] = r
    rows = list(FLT.values())
    nmiss = sum(r["miss"] for r in rows); need("typed_floor", "teacher", nmiss); cover("typed_floor", "pairs", len(rows)); cover("typed_floor", "answerable", sum(r["v"] in VAL for r in rows))
    d = rate_of(rows); say(f"\n=== typed arm floor (two independent typed full-prompt records, field equality on normalised fields; template-rendered note) {len(rows)} prompts\n[typed floor] DIFFERENT {fmt(d)}")
    fd = Counter(f for r in rows if r.get("diff") for f in r["diff"]); say(f"      differing fields: {dict(fd)}")
    RES["typed_floor"] = dict(rate=d, fields=dict(fd)); typed_samples("floor", rows)
    gts = {i: E.gt_record(P[i]) for i in LAB}; acc = Counter(); nn = 0
    for r in rows:
        if r.get("rec"):
            nn += 1
            for f in E.TFIELDS:
                g = gts[r["i"]]; a_ = r["rec"][f]; acc[f] += (abs(a_ - g[f]) < 0.005 if isinstance(a_, float) and isinstance(g[f], float) else a_ == g[f])
    if nn: say("      typed record vs prompt ground truth (copy fidelity): " + ", ".join(f"{f} {acc[f]}/{nn}" for f in E.TFIELDS)); RES["typed_gt_accuracy"] = dict(n=nn, **{f: acc[f] for f in E.TFIELDS})
    save()
if "floor" in STAGES: guarded("floor", stage_floor)
if "typed" in STAGES: guarded("typed floor", stage_typed_floor)

# ------------------------------------------------------------------ 5. pruned cells + certificate (code path of run_b.py, with rotation cross-fitting)
CELLS, CERT, CELLS_T, CERT_T = {}, {}, {}, {}
kkey = lambda k: (tuple(int(k[f]) for f in FIELDS), tuple(k["hist"]))
def views(kf, ids):
    """distinct (prompt, keep-set) items over every rotation in which a prompt is scored (own test rotation + the rotations that calibrate on its fold), and the (prompt, rotation) -> item map"""
    items, vmap = {}, {}
    for i in ids:
        for r in ROTS[i]:
            k = kf(i, r); key = (i, kkey(k)); items.setdefault(key, (i, k)); vmap[(i, r)] = key
    return items, vmap
def text_cell_rows(items, strict_keys):
    def one(key):
        i, k = items[key]; p = P[i]; n = E.note_c(p, k, 0)
        if n is None: return key, dict(i=i, v=None, path="note-missing", need=dict(judge1=0, judge2=0, adj_exp=0.0), miss=1, tok=None, den=TOK[i], trunc=False, kept_h=len(k["hist"]), kept_f=sum(k[f] for f in FIELDS), nh=len(p["hist"]), sraw=None, scons=None)
        r = E.equiv(p, REF[i][0], n["text"], A.evidence); s = E.strict(REF[i], n["text"]) if key in strict_keys else dict(raw=None, cons=None, need=0, exp=0)
        return key, dict(i=i, v=r["v"], path=r["path"], need=r["need"], miss=0, tok=n["tok"], den=TOK[i], trunc=n["trunc"], kept_h=len(k["hist"]), kept_f=sum(k[f] for f in FIELDS), nh=len(p["hist"]), sraw=s["raw"], scons=s["cons"], sneed=s["need"])
    return dict(pmapx(one, list(items)))
def cell_cost(items, n_strict):
    nnew = sum(not E.note_cached(P[i], k, 0) for i, k in items.values()); pr_ = RESID["abl"] if RESID["abl"] is not None else 0.4
    return nnew, nnew * PR["teacher"] + len(items) * (pr_ * (2 * PR["judge1"] + A.adj_p * PR["judge2"])) + n_strict * STRICT_CALLS * PR["judge1"]
def summ(name, rows, FLd, arm):
    d = rate_of(rows); ids = [r["i"] for r in rows if r["v"] in VAL]; c = np.array([dif(r["v"]) for r in rows if r["v"] in VAL])
    have = [r for r in rows if r["tok"] and r.get("den")]
    o = dict(name=name, arm=arm, **d, ci_vendor=vboot(c, ids) if len(ids) else None, tokens_removed=(1 - sum(r["tok"] for r in have) / sum(r["den"] for r in have)) if have else None,
             hist_kept=float(np.mean([r["kept_h"] for r in rows])), hist_total=float(np.mean([r["nh"] for r in rows])), fixed_dropped=float(np.mean([4 - r["kept_f"] for r in rows])), by_scn={})
    if arm == "text":
        sr = rate_of(rows, "sraw"); sc_ = rate_of(rows, "scons")
        if sr["n"]: o["strict_raw"] = sr; o["strict_cons"] = sc_
    for s in sorted({P[i]["scn"] for i in LAB}):
        xs = [r for r in rows if P[r["i"]]["scn"] == s and r["v"] in VAL]
        if xs: o["by_scn"][s] = dict(n=len(xs), dis=float(np.mean([dif(r["v"]) for r in xs])))
    both = [r for r in rows if r["v"] in VAL and FLd.get(r["i"], {}).get("v") in VAL]
    if both:
        b = sum(dif(r["v"]) and not dif(FLd[r["i"]]["v"]) for r in both); cc = sum(not dif(r["v"]) and dif(FLd[r["i"]]["v"]) for r in both)
        o["vs_floor"] = dict(n=len(both), floor_rate=float(np.mean([dif(FLd[r["i"]]["v"]) for r in both])), pruned_only=int(b), floor_only=int(cc), p=float(binomtest(int(b), int(b + cc), .5).pvalue) if b + cc else 1.0)
    say(f"[cell/{arm}] {name:24s} rows {len(rows):3d} | tokens removed {('n/a' if o['tokens_removed'] is None else f'{o['tokens_removed']*100:5.1f}%')} ({'typed' if arm == 'typed' else 'free-text'} full-prompt denominator) | hist kept {o['hist_kept']:.1f}/{o['hist_total']:.1f} | DIFFERENT {fmt(d)}"
        + (f" | vendor-bootstrap {o['ci_vendor'][0]:.3f}-{o['ci_vendor'][1]:.3f}" if o["ci_vendor"] else "") + (f" | floor same rows {o['vs_floor']['floor_rate']:.3f}, discordant {o['vs_floor']['pruned_only']}/{o['vs_floor']['floor_only']} p={o['vs_floor']['p']:.3f}" if "vs_floor" in o else ""))
    if "strict_raw" in o: say(f"      strict rubric (secondary, same rows): raw {fmt(o['strict_raw'])} | confirmed {fmt(o['strict_cons'])}")
    return o
def cert(RV, CF, budget, excess, FLd):
    """RV / CF: (prompt, rotation) -> row / confidence. Rotation f: cutoff chosen on the calibration fold(s) CALF[f] scored by scorer f, applied to the test fold f scored by the same scorer."""
    usable = lambda r: r["v"] in VAL and (not excess or FLd.get(r["i"], {}).get("v") in VAL)
    ok = lambda r: not (dif(r["v"]) and (not excess or not dif(FLd[r["i"]]["v"])))
    tst = [(i, FOLD[i]) for i in LAB if usable(RV[(i, FOLD[i])])]; n_ = len(tst); tok_all = sum(RV[k]["den"] for k in tst) or 1
    cov = err = saved = 0; cuts = []
    for f in range(K):
        te = [k for k in tst if FOLD[k[0]] == f]; cal = [(i, f) for i in LAB if FOLD[i] in CALF[f] and usable(RV[(i, f)])]
        if not te or len(cal) < 10: continue
        c = certify([CF[k] for k in cal], [ok(RV[k]) for k in cal], budget); cuts.append(c["cutoff"])
        if np.isfinite(c["cutoff"]):
            sel = [k for k in te if CF[k] >= c["cutoff"]]; cov += len(sel); err += sum(not ok(RV[k]) for k in sel); saved += sum(RV[k]["den"] - RV[k]["tok"] for k in sel)
    pc = certify([CF[k] for k in tst], [ok(RV[k]) for k in tst], budget) if n_ >= 10 else dict(cutoff=float("inf"), coverage=0.0, emp_error=None, bound=None)
    ps = sum(RV[k]["den"] - RV[k]["tok"] for k in tst if CF[k] >= pc["cutoff"]) / tok_all if np.isfinite(pc["cutoff"]) else 0.0
    return dict(budget=budget, n_usable=n_, excluded_fail_or_unknown=len(LAB) - n_, covered=cov, coverage=cov / n_ if n_ else None, errors=err, sel_error=err / cov if cov else None, ub=ub(err, cov) if cov else None,
                tokens_saved=saved / tok_all, folds_with_cutoff=int(sum(np.isfinite(x) for x in cuts)), n_calibration_per_rotation=NCAL * len(LAB) // K, pooled_in_sample=dict(coverage=pc["coverage"], emp_error=pc["emp_error"], bound=pc["bound"], tokens_saved=float(ps)))
def certify_cell(name, RV, sc, kept_of, FLd, store, arm):
    na = sum(RV[(i, FOLD[i])]["v"] in VAL for i in LAB); out = {}
    if na < 10 or (A.dry and na < 0.9 * len(LAB)): say(f"[certificate/{arm}] {name}: skipped ({na}/{len(LAB)} rows answerable; a partial cache is biased toward canon-decided pairs, so no certificate is computed)"); return
    CF = {k: conf(sc, k[0], kept_of(*k), k[1]) for k in RV}
    for ex in ([False, True] if FLd else [False]):
        for b in (0.05, 0.10, 0.15):
            c = cert(RV, CF, b, ex, FLd); out[("excess@" if ex else "") + f"{int(b*100)}%"] = c
            say(f"[certificate/{arm}] {name:22s} {'(excess over floor) ' if ex else '                    '}budget {int(b*100):2d}%: covered {c['covered']:3d}/{c['n_usable']} | error among covered {c['sel_error'] if c['sel_error'] is None else round(c['sel_error'],3)} CP UB {c['ub'] if c['ub'] is None else round(c['ub'],3)} | tokens saved {c['tokens_saved']*100:4.1f}% | folds with a cutoff {c['folds_with_cutoff']}/{K} | excluded FAIL/unknown {c['excluded_fail_or_unknown']} | {POL}")
    store[name] = out
def do_cell_text(name, kf, sc, idx):
    ids = LAB; items, vmap = views(kf, ids); testk = {vmap[(i, FOLD[i])] for i in ids}; strict_keys = testk if idx < A.strict_cells else set()
    nnew, est = cell_cost(items, len(strict_keys))
    if LIVE and est > left() - 0.03: say(f"[cell] SKIPPED {name}: needs ~${est:.2f} for {nnew} uncached notes + judging ({len(items)} distinct pruned prompts over {K} rotations), ${left():.2f} left under the cap"); return None
    stg = f"cell:{name}"; rd = text_cell_rows(items, strict_keys)
    for r in rd.values():
        need(stg, "teacher", r["miss"]); need(stg, "judge1", r["need"]["judge1"] + r.get("sneed", 0)); need(stg, "judge2", r["need"]["judge2"] + r["need"]["adj_exp"]); cover(stg, "pairs"); cover(stg, "answerable", r["v"] in VAL)
    pr_ = RESID["abl"] if RESID["abl"] is not None else 0.4; nm = sum(r["miss"] for r in rd.values())
    need(stg, "judge1", nm * (pr_ * 2 + (STRICT_CALLS if strict_keys else 0))); need(stg, "judge2", nm * pr_ * A.adj_p)
    RV = {k: rd[v] for k, v in vmap.items()}; rows = [RV[(i, FOLD[i])] for i in ids]
    CELLS[name] = summ(name, rows, FL, "text"); RES["cells"] = CELLS
    certify_cell(name, RV, sc, lambda i, r: set(kf(i, r)["hist"]), FL if all(i in FL for i in LAB) else {}, CERT, "text"); RES["cert"] = CERT; save()
    if LIVE: say(f"[spend] {name}: ${llm.SPENT[0]-LEDGER0:.3f} of ${A.cap_usd} cap ({llm.NEW[0]} new calls, {time.time()-T0:.0f}s)")
    return RV
def do_cell_typed(name, kf, sc):
    ids = LAB; items, vmap = views(kf, ids)
    nn, est = typed_cost([typed_key(P[i], k, 0) for i, k in items.values()] + [typed_key(P[i], keep_all(P[i]), 0) for i in ids])
    if LIVE and est > left() - 0.03: say(f"[typed cell] SKIPPED {name}: needs ~${est:.2f} for {nn} typed records, ${left():.2f} left under the cap"); return None
    stg = f"typed:{name}"; keys = list(items)
    rd = dict(zip(keys, pmapx(lambda key: typed_pair(items[key][0], items[key][1], 0, 0), keys)))
    for r in rd.values(): need(stg, "teacher", r["xmiss"]); cover(stg, "pairs"); cover(stg, "answerable", r["v"] in VAL)
    for key, r in rd.items():
        i, k = items[key]; r.update(kept_h=len(k["hist"]), nh=len(P[i]["hist"]), kept_f=sum(k[f] for f in FIELDS), trunc=False)
    RV = {k: rd[v] for k, v in vmap.items()}; rows = [RV[(i, FOLD[i])] for i in ids]
    CELLS_T[name] = summ(name, rows, FLT, "typed"); RES["cells_typed"] = CELLS_T; typed_samples(name, rows)
    certify_cell(name, RV, sc, lambda i, r: set(kf(i, r)["hist"]), FLT if all(i in FLT for i in LAB) else {}, CERT_T, "typed"); RES["cert_typed"] = CERT_T; save()
    return RV
ORDER = [("learned", 0), ("learned", 5), ("learned", 2), ("rule", None), ("cross-encoder", 5), ("cross-encoder", 2), ("oracle", 5), ("random", 5), ("oracle", 2), ("random", 2)]
if A.tiny: ORDER = [("learned", 0), ("learned", 2), ("rule", None)]
if STAGES & {"cells", "typed"}:
    # residual share of ablation-like pairs (canon only, free) for dry-run extrapolation of pruned cells
    ab = [r for r in RLB if r["v"] in VAL or r["path"] in ("judge-onesided", "judge-evidence")]; RESID["abl"] = float(np.mean([r["path"].startswith("judge") for r in ab])) if ab else None
    for idx, (sc, m) in enumerate(ORDER):
        nm_ = "rule (tx+inv+flag)" if sc == "rule" else f"{sc} m={m}"
        if sc == "cross-encoder" and CE_TAG == "off": say(f"[cell] {nm_} skipped (cross-encoder unavailable)"); continue
        if E.STATE["capped"]: say(f"[cell] SKIPPED {nm_}: the spend cap was already hit"); continue
        kf = keepfn(sc, m); tr = guarded(f"text cell {nm_}", do_cell_text, nm_, kf, sc, idx) if "cells" in STAGES else None
        if "typed" in STAGES: guarded(f"typed cell {nm_}", do_cell_typed, nm_, kf, sc)
        if sc == "learned" and m == 0 and tr is not None:      # the m=0 prompt is the same for every scorer; only the confidence differs
            for sc2 in ("cross-encoder", "oracle", "random"):
                if sc2 == "cross-encoder" and CE_TAG == "off": continue
                guarded(f"certificate {sc2} m=0", certify_cell, f"{sc2} m=0", tr, sc2, lambda i, r: (), FL if all(i in FL for i in LAB) else {}, CERT, "text")
RES["cert"] = CERT; RES["cert_typed"] = CERT_T

# ------------------------------------------------------------------ headline + cost plan
if CERT or CERT_T:
    say(f"\n=== headline: share of input tokens removable at each budget (rotation cross-fitting: scorer, calibration and held-out folds vendor-disjoint; low-confidence prompts fall back to the full prompt; typed arm uses its own typed full-prompt token count), " + POL)
    for arm, C_ in (("text", CERT), ("typed", CERT_T)):
        for key in ("5%", "10%", "15%", "excess@5%", "excess@10%", "excess@15%"):
            best = sorted(((v[key]["tokens_saved"], k, v[key]) for k, v in C_.items() if key in v), key=lambda t: -t[0])[:3]
            if best: say(f"  [{arm}] budget {key:10s} " + " | ".join(f"{k}: saved {t*100:.1f}% coverage {(c['coverage'] or 0)*100:.0f}% error {c['sel_error']} (UB {c['ub']})" for t, k, c in best))
PLAN = {}
tot = Counter()
for st, d in NEEDS.items():
    d = dict(d); PLAN[st] = dict(calls=d, usd=usd(d), covered=dict(COVER[st])); tot.update(d)
PLAN["TOTAL"] = dict(calls=dict(tot), usd=usd(tot))
RES["plan"] = PLAN if A.dry else {}; RES["resid_share"] = RESID; save()
if A.dry:
    say("\n=== COST / COVERAGE PLAN (calls still to be made = misses against the cache; teacher = teacher note or typed record, judge1/judge2 = judge calls incl. expected extras for pairs whose notes do not exist yet)")
    say(f"{'stage':32s} {'answerable/pairs':>17s} {'teacher':>7s} {'judge1':>8s} {'judge2':>7s} {'est $':>8s}")
    for st, v in PLAN.items():
        if st == "TOTAL": continue
        c = v["covered"]; say(f"{st:32s} {c.get('answerable',0):>8d}/{c.get('pairs',0):<8d} {v['calls'].get('teacher',0):7.0f} {v['calls'].get('judge1',0):8.0f} {v['calls'].get('judge2',0):7.1f} {v['usd']:8.3f}")
    say(f"{'TOTAL':32s} {'':>17s} {PLAN['TOTAL']['calls'].get('teacher',0):7.0f} {PLAN['TOTAL']['calls'].get('judge1',0):8.0f} {PLAN['TOTAL']['calls'].get('judge2',0):7.1f} {PLAN['TOTAL']['usd']:8.3f}   (cap ${A.cap_usd}; policy {POL}; relabel row is spent only with --live)")
if A.dry:
    import cacheonly; T = PLAN["TOTAL"]["calls"]
    say(f"[dry] network calls made: 0. Cache hits {cacheonly.HITS[0]}. Calls still needed in total: teacher/typed {T.get('teacher',0):.0f}, judge1 {T.get('judge1',0):.0f}, judge2 {T.get('judge2',0):.1f} (needs totals; the shim's miss log is empty because note_c / J2 check the cache before calling it). Scorer embedding {EMBTAG}, cross-encoder {CE_TAG}.")
if LIVE: say(f"[spend] total this run ${llm.SPENT[0]-LEDGER0:.3f} of cap ${A.cap_usd}, {llm.NEW[0]} new calls")
say(f"[time] {time.time()-T0:.0f}s")
