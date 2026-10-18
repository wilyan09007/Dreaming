"""S9: full rollout on the SYNTHETIC 4-step workflow  categorize -> vendor_lookup -> match_invoice -> flag_anomaly  (same env as S4).
Imagine the trajectory from the raw transaction with chained predictors, execute the real one with real tools/teacher, compare.
Revision: grid certifier (S4 run.py) instead of common.certify; 'calls avoided' counts only fully specified outputs (match_invoice's output is an invoice id);
the invoice-consumption side effect of skipping match_invoice is replayed; all vendor_lookup attempts count toward latency; direct trajectory certificate tested."""
import os; os.environ.setdefault("OMP_NUM_THREADS", "1")  # HGB busy-waits and crawls when the CPU is shared
import sys, time, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).parent; S4 = HERE.parent / "s4_tool_predictor"
sys.path.insert(0, str(S4)); sys.path.insert(0, str(HERE.parent))
from common import confidence
from env import Env, CATS
spec = importlib.util.spec_from_file_location("s4run", S4 / "run.py"); s4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s4)
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC
md, pick_text, HGB, frac, cert, cert_orig = s4.md, s4.pick_text, s4.HGB, s4.frac, s4.certify_grid, s4.certify_orig
N, NTR, NCAL, SEL_BUDGET = 6000, 3000, 4800, 0.02   # chronological train | calibration | test; SEL_BUDGET picks LR vs NN per text step
STEPS = ["categorize", "vendor_lookup", "match_invoice", "flag_anomaly"]; BS = (.02, .03, .05, .10); INF = float("inf")
cf = lambda x: "inf" if np.isinf(x) else round(float(x), 3)

def simulate(env):
    rows, ev, snap = [], [], None
    for i in range(N):
        if i == NCAL: snap = {v: list(l) for v, l in env.invoices.items()}  # invoice table as the real train+cal executions left it
        n0 = env.n_inv
        t = env.new_txn("train" if i < NTR else "cal" if i < NCAL else "test")
        if i >= NCAL: ev.append([(env.vid[t["_m"]], x[0], x[1]) for x in env.invoices.get(env.vid[t["_m"]], []) if int(x[0][4:]) > n0])  # invoices created by this transaction
        cat, l1 = env.teacher_category(t["descriptor"]); l2 = 0.0
        for _ in range(20):  # the real tool times out ~2% of the time: retry; every attempt costs its latency
            v, l = env.vendor_lookup(t["descriptor"]); l2 += l
            if not v.startswith("ERROR"): break
        assert not v.startswith("ERROR")
        inv, l3 = env.match_invoice(v, t["amount"]); an, l4 = env.flag_anomaly(v, t["amount"], inv)
        rows.append(dict(i=i, desc=t["descriptor"], amount=t["amount"], cat=cat, vendor=v, inv=inv, found=int(inv != "NONE"), anomaly=an, l1=l1, l2=l2, l3=l3, l4=l4, group=t["_group"]))
    return pd.DataFrame(rows), snap, ev

def replay(env, snap, ev, amount, vendor, run):
    """Re-run the test stream's match_invoice calls against the invoice table, executing only rows with run[j] (with the given vendor id). Skipped calls do not consume their invoice."""
    env.invoices = {v: list(l) for v, l in snap.items()}; out = []
    for j in range(len(ev)):
        for v, iid, a in ev[j]: env.invoices.setdefault(v, []).append((iid, a))
        out.append(env.match_invoice(vendor[j], amount[j])[0] if run[j] else None)
    return np.array(out, dtype=object)

# ---- trace-derived features for the two tabular steps ---------------------------------------------------------
vnum = lambda v: np.array([int(x[1:]) for x in v])
def te_fit(vendor, found, a=2.0):  # smoothed per-vendor invoice rate from traces
    prior = found.mean(); g = pd.DataFrame(dict(v=vendor, f=found)).groupby("v").f.agg(["sum", "count"])
    return lambda vs: ((g["sum"].reindex(vs).fillna(0) + a * prior) / (g["count"].reindex(vs).fillna(0) + a)).values
def te_oof(vendor, found, k=5):  # out-of-fold on training rows so the model does not see its own label through the encoding
    out = np.zeros(len(vendor)); idx = np.arange(len(vendor))
    for f in range(k): out[idx % k == f] = te_fit(vendor[idx % k != f], found[idx % k != f])(vendor[idx % k == f])
    return out
def X3(vendor, te, cat, amount): return np.c_[vnum(vendor), te, [CATS.index(c) for c in cat], amount]       # categorical idx 2
def X4(found, amount, ratio, cat): return np.c_[found, amount, ratio, [CATS.index(c) for c in cat]]       # categorical idx 3

# ---- certified gating -------------------------------------------------------------------------------------------
def gate(Rc, Rt, te, b, certf, rp=None):
    """Sequential per-step cutoffs on the calibration slice (step i certified on the rows where steps 1..i-1 passed), applied to the test rollout."""
    ok_c, ok_t = Rc["m"], Rt["m"]; mask = np.ones(len(ok_c), bool); cuts = []
    for i in range(4):
        c = certf(Rc["conf"][mask, i], ok_c[mask, i], b) if mask.sum() > 1 else dict(cutoff=INF)
        cuts.append(c["cutoff"]); mask &= Rc["conf"][:, i] >= c["cutoff"]
    conf_t = Rt["conf"] >= np.array(cuts); full = conf_t.all(1); n = int(full.sum()); bad = ~ok_t.all(1)
    L = te[["l1", "l2", "l3", "l4"]].values; none = full & (Rt["pf"] == 0)  # none: confident AND imagined NONE = the only fully specified match_invoice output
    se = [(~ok_t[full, i]).mean() if n else None for i in range(4)]
    g = dict(cuts=cuts, full=full, n=n, share=full.mean(), n_err=int(bad[full].sum()), err=bad[full].mean() if n else None, step_err=se, prefix=np.cumprod(conf_t, axis=1).sum(1).mean(),
             mean_k=Rt["k"][full].mean() if n else None, nov=None, none_share=none.mean(), found_share=(full & (Rt["pf"] == 1)).mean(), tools=(full.mean() + none.mean()) / 2, tools_old=full.mean(),
             lat=(L[full][:, [0, 1, 3]].sum() + L[none][:, 2].sum()) / L.sum(), lat_old=L[full].sum() / L.sum(),
             both23=(~ok_t[full, 1] & ~ok_t[full, 2]).mean() if n else None, new4=(~ok_t[full, 3] & ok_t[full, 2]).mean() if n else None,
             p3_v=(~ok_t[full, 2])[~ok_t[full, 1]].mean() if n and (~ok_t[full, 1]).any() else None, n_vwrong=int((~ok_t[full, 1]).sum()))
    if rp:  # state divergence: replay match_invoice with the policy's skips; compare executed results with the all-real run
        real, pv = te.vendor.values, Rt["pv"]
        for key, run_m, vend in (("A", ~none, np.where(full, pv, real)), ("B", ~full, real)):
            o = rp(run_m, vend); r = te.inv.values
            g["rp" + key] = dict(n_run=int(run_m.sum()), diff=int((o[run_m] != r[run_m]).sum()), flip=int(((o[run_m] != "NONE") != (r[run_m] != "NONE")).sum()))
    return g

def direct(Rc, Rt, b):
    """Certify the trajectory outcome directly: confidence = smallest of the 4 step gaps, correct = every step matched."""
    c = cert(Rc["conf"].min(1), Rc["m"].all(1), b); full = Rt["conf"].min(1) >= c["cutoff"]; bad = ~Rt["m"].all(1)
    return dict(cutoff=c["cutoff"], cal_cov=c["coverage"], share=full.mean(), n=int(full.sum()), n_err=int(bad[full].sum()), err=bad[full].mean() if full.any() else None)

def world(seed, tick=lambda m: None):
    env = Env(seed=seed); D, snap, ev = simulate(env); tick("simulated")
    tr, ca, te = D[:NTR].reset_index(drop=True), D[NTR:NCAL].reset_index(drop=True), D[NCAL:].reset_index(drop=True)
    known = set(tr.vendor); nov = ~te.vendor.isin(known).values; y4 = (te.anomaly == "yes").values
    assert (replay(env, snap, ev, te.amount.values, te.vendor.values, np.ones(len(te), bool)) == te.inv.values).all()  # executing everything reproduces the logged outputs
    rp = lambda run_m, vend: replay(env, snap, ev, te.amount.values, vend, run_m)
    # step 1 and 2: text students (LR vs NN, kept by certified coverage on the calibration slice at SEL_BUDGET); also what the ORIGINAL certify would have kept
    res, kept, kept_orig = {}, {}, {}
    for step, col in (("cat", "cat"), ("ven", "vendor")):
        kept[step], _, res[step] = pick_text(tr.desc.tolist(), tr[col].values, ca.desc.tolist(), ca[col].values, SEL_BUDGET); tick(step + " student")
        cov = {}
        for k, (p, _, _) in res[step].items():
            lab, conf = p(ca.desc.tolist()); cov[k] = cert_orig(conf, lab == ca[col].values, SEL_BUDGET)["coverage"]
        kept_orig[step] = max(cov, key=cov.get)
    # step 3: invoice found?  HGB on [vendor id, trace-derived vendor invoice rate, category, amount]
    te_all = te_fit(tr.vendor.values, tr.found.values)
    m3 = HGBC(categorical_features=[2], **HGB).fit(X3(tr.vendor.values, te_oof(tr.vendor.values, tr.found.values), tr.cat.values, tr.amount.values), tr.found.values)
    # step 4: anomaly?  HGB on [invoice found, amount, amount / vendor mean (leave-one-out on train), category]
    g = tr.groupby("vendor").amount.agg(["sum", "count"]); vsum, vcnt = g["sum"], g["count"]
    nn_ = vcnt.reindex(tr.vendor).values
    loo = (vsum.reindex(tr.vendor).values - tr.amount.values) / np.where(nn_ > 1, nn_ - 1, np.nan)  # vendor mean without the row itself
    vmean = lambda vs: (vsum.reindex(vs) / vcnt.reindex(vs)).values  # NaN for vendors never seen in training
    m4 = HGBC(categorical_features=[3], **HGB).fit(X4(tr.found.values, tr.amount.values, tr.amount.values / loo, tr.cat.values), (tr.anomaly == "yes").values)
    tick("tabular steps")

    def run(d, cat_p, ven_p):  # imagined rollout (chained, each step consumes PREDICTED outputs) and teacher-forced step predictions
        R = {}
        R["pc"], R["c1"] = cat_p(d.desc.tolist()); R["pv"], R["c2"] = ven_p(d.desc.tolist())
        P3 = m3.predict_proba(X3(R["pv"], te_all(R["pv"]), R["pc"], d.amount.values)); R["pf"], R["c3"], R["p3"] = P3.argmax(1), confidence(P3), P3
        P4 = m4.predict_proba(X4(R["pf"], d.amount.values, d.amount.values / vmean(R["pv"]), R["pc"])); R["pa"], R["c4"], R["p4"] = P4[:, 1] > .5, confidence(P4), P4[:, 1]
        R["tf3"] = m3.predict_proba(X3(d.vendor.values, te_all(d.vendor.values), d.cat.values, d.amount.values)).argmax(1)   # teacher-forced: each step sees the REAL upstream outputs
        P4t = m4.predict_proba(X4(d.found.values, d.amount.values, d.amount.values / vmean(d.vendor.values), d.cat.values)); R["tf4"] = P4t[:, 1] > .5
        R["m"] = np.c_[R["pc"] == d.cat.values, R["pv"] == d.vendor.values, R["pf"] == d.found.values, R["pa"] == (d.anomaly == "yes").values]  # step-i imagined == real
        R["conf"] = np.c_[R["c1"], R["c2"], R["c3"], R["c4"]]; R["k"] = np.cumprod(R["m"], axis=1).sum(1)  # steps ahead matched = matching prefix length
        return R
    def analyze(students):
        cat_p, ven_p = res["cat"][students[0]][0], res["ven"][students[1]][0]
        Rc, Rt = run(ca, cat_p, ven_p), run(te, cat_p, ven_p)
        G = {(n_, b): gate(Rc, Rt, te, b, f, rp if n_ == "grid" else None) for n_, f in (("grid", cert), ("orig", cert_orig)) for b in BS}
        for gg in G.values(): gg["nov"] = nov[gg["full"]].mean() if gg["n"] else None
        return dict(Rc=Rc, Rt=Rt, G=G, D={b: direct(Rc, Rt, b) for b in (.02, .03, .05)}, students=students)
    W = dict(seed=seed, tr=tr, ca=ca, te=te, nov=nov, y4=y4, kept=kept, kept_orig=kept_orig, acc={s: {k: v[2] for k, v in res[s].items()} for s in res},
             vmean=vmean, A=analyze((kept["cat"], kept["ven"])))
    W["alt"] = analyze((kept_orig["cat"], kept_orig["ven"])) if kept_orig != kept else None
    tick("analysis"); return W

def report_main(W, out):
    te, nov, y4, A = W["te"], W["nov"], W["y4"], W["A"]; Rt, G = A["Rt"], A["G"]; tr = W["tr"]
    out.append(f"Real trajectories executed with real tools: train {len(tr)}, cal {len(W['ca'])}, test {len(te)}. Realized invoice rate: train {tr.found.mean():.3f}, test {te.found.mean():.3f}; "
               f"anomaly=yes rate: train {(tr.anomaly == 'yes').mean():.3f}, test {(te.anomaly == 'yes').mean():.3f}; test transactions with a vendor absent from training: {nov.mean():.3f}\n")
    a = W["acc"]; out.append(f"Step students kept (grid certifier, {SEL_BUDGET:.0%} budget): categorize = {W['kept']['cat'].upper()} (cal acc LR {a['cat']['lr']:.3f} / NN {a['cat']['nn']:.3f}), "
                             f"vendor_lookup = {W['kept']['ven'].upper()} (cal acc LR {a['ven']['lr']:.3f} / NN {a['ven']['nn']:.3f}). With the original certify the kept students would have been: "
                             f"categorize = {W['kept_orig']['cat'].upper()}, vendor_lookup = {W['kept_orig']['ven'].upper()}.\n")
    seg = {"all test": np.ones(len(te), bool), "known vendors": ~nov, "novel vendors": nov}
    rows = [[n, int(m.sum())] + [(Rt["k"][m] == j).mean() for j in range(5)] + [Rt["k"][m].mean()] for n, m in seg.items()]
    out.append("### Steps ahead matched (imagined trajectory equals the real one for the first k steps)\n\n" + md(rows, ["test slice", "n", "k=0", "k=1", "k=2", "k=3", "k=4 (whole trajectory)", "mean k"]) + "\n")
    tf = [Rt["m"][:, 0].mean(), Rt["m"][:, 1].mean(), (Rt["tf3"] == te.found.values).mean(), (Rt["tf4"] == y4).mean()]  # steps 1-2 have no upstream
    rows = [[STEPS[i], Rt["m"][:, i].mean(), tf[i], Rt["m"][~nov, i].mean(), Rt["m"][nov, i].mean()] for i in range(4)]
    out.append("### Per-step accuracy: chained (consumes predicted upstream) vs teacher-forced (consumes real upstream)\n\n" +
               md(rows, ["step", "chained acc", "teacher-forced acc", "chained acc, known vendors", "chained acc, novel vendors"]) + "\n")
    out.append(f"Baselines: invoice found = majority class {max(te.found.mean(), 1 - te.found.mean()):.3f}; anomaly majority class {max(y4.mean(), 1 - y4.mean()):.3f}. "
               "Teacher-forced is an upper bound only for step 4. For step 3 it can be lower than chained because real vendors absent from training reach the HGB as arbitrary vendor numbers, "
               "whereas a predicted vendor is always a training vendor.\n")
    rows = []
    def route(name, flag):
        tp = (flag & y4).sum(); rows.append([name, flag.mean(), tp / max(flag.sum(), 1), tp / y4.sum(), int(tp), int((flag & ~y4).sum())])
    route("send everything to review", np.ones(len(te), bool)); route("imagined anomaly = yes (chained rollout)", Rt["pa"])
    for thr in (.8, .95): route(f"imagined P(anomaly) > {thr}", Rt["p4"] > thr)
    route("step-4 predictor on the REAL upstream outputs (reference)", Rt["tf4"])
    route("approximate rule (no invoice OR amount > 3x TRAIN-estimated vendor mean) on imagined upstream outputs", (Rt["pf"] == 0) | (te.amount.values > 3 * W["vmean"](Rt["pv"])))
    out.append(f"### Early routing to review. Real anomaly=yes base rate in test: {y4.mean():.3f} ({int(y4.sum())} of {len(y4)})\n\n" +
               md(rows, ["policy", "share routed", "precision", "recall", "true anomalies routed", "false routes"]) + "\n")
    rows = [[n, int(m.sum()), y4[m].mean(), (Rt["pa"][m] & y4[m]).sum() / max(Rt["pa"][m].sum(), 1), (Rt["pa"][m] & y4[m]).sum() / max(y4[m].sum(), 1)] for n, m in (("known vendors", ~nov), ("novel vendors", nov))]
    out.append(md(rows, ["slice", "n", "real anomaly rate", "precision (chained)", "recall (chained)"]) + "\n")
    # ---- certified gating
    rows = [[b] + [cf(x) for x in G["grid", b]["cuts"]] + [G["grid", b]["share"], G["grid", b]["n"], frac(G["grid", b]["n_err"], G["grid", b]["n"]), G["grid", b]["prefix"]] for b in BS]
    out.append("### Confident trajectories, grid certifier (every step above its certified cutoff; per-step budget b, delta 0.05 per step, steps gated sequentially on the calibration slice)\n\n" +
               md(rows, ["budget b", "cut 1", "cut 2", "cut 3", "cut 4", "share of test with whole trajectory confident", "n", "trajectory error among them (any step wrong)", "mean confident prefix (steps skippable)"]) + "\n")
    rows = [[b] + [cf(x) for x in G["orig", b]["cuts"]] + [G["orig", b]["share"], G["orig", b]["n"], frac(G["orig", b]["n_err"], G["orig", b]["n"])] for b in BS]
    out.append("Same, with the original `common.certify` (as shipped; `inf` = it found nothing certifiable):\n\n" + md(rows, ["budget b", "cut 1", "cut 2", "cut 3", "cut 4", "share confident", "n", "trajectory error"]) + "\n")
    # ---- calls avoided, counting only fully specified outputs
    rows = [[b, g["share"], g["found_share"] / g["share"] if g["n"] else None, g["none_share"], g["tools"], g["tools_old"], g["lat"], g["lat_old"]]
            for b in BS for g in [G["grid", b]]]
    out.append("### Calls avoided when the whole imagined trajectory is confident\n\n"
               "categorize and flag_anomaly (teacher calls) and vendor_lookup have fully specified outputs (a category, a vendor id, yes/no), so all three are avoided whenever the trajectory is confident. "
               "match_invoice returns an invoice id, which the rollout cannot supply: it can be skipped only when the imagined output is NONE. "
               "Each test transaction has 2 teacher calls and 2 tool calls; 'old accounting' counted match_invoice as skipped for every confident trajectory.\n\n" +
               md(rows, ["budget b", "confident share = teacher calls avoided = vendor_lookup avoided", "share of confident trajectories whose imagined invoice is `found` (id unavailable)",
                         "match_invoice avoided (confident and imagined NONE, share of test)", "tool calls avoided, fully specified outputs", "tool calls avoided, old accounting",
                         "step latency avoided, fully specified", "step latency avoided, old accounting"]) + "\n")
    rows = []
    for b in BS:
        g = G["grid", b]
        for key, name in (("A", "skip match_invoice only when imagined NONE; executed matches use the predicted vendor"), ("B", "old accounting: skip match_invoice for every confident trajectory")):
            r = g["rp" + key]; rows.append([b, name, r["n_run"], frac(r["diff"], r["n_run"]), frac(r["flip"], r["n_run"]), g["n_err"]])
    out.append("### State divergence from skipping match_invoice (replayed on the invoice table; an executed match consumes its invoice)\n\n" +
               md(rows, ["budget b", "policy", "match_invoice calls still executed", "executed results that differ from the all-real run (invoice id)", "... of which found/NONE flips", "confident trajectories with an error"]) + "\n")
    # ---- error structure among confident trajectories
    rows = [[b, g["n"]] + g["step_err"] + [g["new4"], g["err"], (g["step_err"][1] + g["step_err"][2]) if g["n"] else None, g["err"] / b if g["n"] else None, g["p3_v"], g["both23"], g["mean_k"], g["nov"]]
            for b in BS for g in [G["grid", b]]]
    out.append("### Error structure among confident trajectories (grid certifier)\n\n" +
               md(rows, ["b", "n confident", "err step 1", "err step 2 (vendor)", "err step 3 (invoice)", "err step 4", "step 4 wrong while step 3 right", "trajectory error (any step)", "vendor + invoice error",
                         "trajectory error / b", "P(step 3 wrong | vendor wrong)", "P(vendor and invoice both wrong)", "mean k among confident", "novel-vendor share among confident"]) + "\n")
    # ---- direct trajectory certificate
    rows = []
    for b in (.02, .03, .05):
        d = A["D"][b]; g = G["grid", b]
        rows.append([b, cf(d["cutoff"]), d["cal_cov"], d["share"], frac(d["n_err"], d["n"]), g["share"], frac(g["n_err"], g["n"])])
    out.append("### Direct trajectory certificate (confidence = min of the 4 step gaps, correct = whole trajectory matched, budget applies to the trajectory) vs per-step budgets of the same size\n\n" +
               md(rows, ["budget", "direct: cutoff", "direct: cal coverage", "direct: share of test confident", "direct: trajectory error", "per-step b: share confident", "per-step b: trajectory error"]) + "\n")

def report_robust(Ws, out):
    S = list(Ws)
    rows = [[sd, Ws[sd]["nov"].mean(), Ws[sd]["A"]["Rt"]["k"].mean(), (Ws[sd]["A"]["Rt"]["k"] == 4).mean(), Ws[sd]["y4"].mean(), *(lambda R, y: ((R["pa"] & y).sum() / R["pa"].sum(), (R["pa"] & y).sum() / y.sum()))(Ws[sd]["A"]["Rt"], Ws[sd]["y4"]),
             f"{Ws[sd]['kept']['cat'].upper()}/{Ws[sd]['kept']['ven'].upper()}" + ("" if Ws[sd]["kept"] == Ws[sd]["kept_orig"] else f" (orig certify: {Ws[sd]['kept_orig']['cat'].upper()}/{Ws[sd]['kept_orig']['ven'].upper()})")] for sd in S]
    num = np.array([r[1:7] for r in rows], float)
    rows += [["mean over 6 worlds"] + list(num.mean(0)) + [""], ["min - max"] + [f"{lo:.3f} - {hi:.3f}" for lo, hi in zip(num.min(0), num.max(0))] + [""]]
    out.append("### Six worlds: rollout quality (main world = seed 9)\n\n" + md(rows, ["env seed", "novel-vendor share", "mean k", "whole trajectory matched", "anomaly base rate", "routing precision", "routing recall", "students (categorize/vendor)"]) + "\n")
    for name in ("grid", "orig"):
        rows = [[sd] + [x for b in (.02, .03, .05) for g in [Ws[sd]["A"]["G"][name, b]] for x in (g["share"], frac(g["n_err"], g["n"]) if g["n"] else "nothing certifiable")] for sd in S]
        out.append(f"### Six worlds: per-step gating with the {'grid certifier (this revision)' if name == 'grid' else 'original common.certify (as shipped)'}: share of test with a confident trajectory, trajectory error\n\n" +
                   md(rows, ["env seed"] + [f"b={b}%: {x}" for b in (2, 3, 5) for x in ("share", "trajectory error")]) + "\n")
    rows = [[name, sd, b, g["n"]] + g["step_err"] + [g["err"], g["step_err"][1] + g["step_err"][2], g["err"] / (b / 100)] for name in ("grid", "orig") for b in (3, 5) for sd in S for g in [Ws[sd]["A"]["G"][name, b / 100]] if g["n"]]
    out.append("### Six worlds: which step produces the errors among confident trajectories (grid certifier, and the shipped original certify for comparison)\n\n" +
               md(rows, ["certifier", "env seed", "b (%)", "n confident", "err step 1", "err step 2 (vendor)", "err step 3 (invoice)", "err step 4", "trajectory error", "vendor + invoice", "trajectory error / b"]) + "\n")
    rows = [[sd, b, g["share"], g["found_share"] / g["share"] if g["n"] else None, g["none_share"], g["tools"], g["tools_old"], g["lat"], g["lat_old"], frac(g["rpA"]["diff"], g["rpA"]["n_run"]), frac(g["rpB"]["diff"], g["rpB"]["n_run"])]
            for b in (3, 5) for sd in S for g in [Ws[sd]["A"]["G"]["grid", b / 100]]]
    out.append("### Six worlds: calls avoided, fully specified outputs only, and state divergence (grid certifier)\n\n" +
               md(rows, ["env seed", "b (%)", "confident share (teacher calls and vendor_lookup avoided)", "confident trajectories with imagined `found`", "match_invoice avoided (share of test)", "tool calls avoided (fully specified)", "tool calls avoided (old accounting)",
                         "latency avoided (fully specified)", "latency avoided (old accounting)", "policy A: executed matches changed", "policy B: executed matches changed"]) + "\n")
    rows = [[sd, b, Ws[sd]["A"]["D"][b]["share"], frac(Ws[sd]["A"]["D"][b]["n_err"], Ws[sd]["A"]["D"][b]["n"]), Ws[sd]["A"]["G"]["grid", b]["share"], frac(Ws[sd]["A"]["G"]["grid", b]["n_err"], Ws[sd]["A"]["G"]["grid", b]["n"])]
            for b in (.02, .03, .05) for sd in S]
    out.append("### Six worlds: direct trajectory certificate vs per-step budgets of the same size\n\n" +
               md(rows, ["env seed", "budget", "direct: share confident", "direct: trajectory error", "per-step: share confident", "per-step: trajectory error"]) + "\n")
    if any(Ws[sd]["alt"] for sd in S):
        rows = [[sd, b, Ws[sd]["alt"]["G"]["grid", b]["share"], frac(Ws[sd]["alt"]["G"]["grid", b]["n_err"], Ws[sd]["alt"]["G"]["grid", b]["n"])] for sd in S if Ws[sd]["alt"] for b in (.02, .03, .05)]
        out.append("### Worlds where the original certify would have kept different step students: grid-certified gating with those students\n\n" + md(rows, ["env seed", "b", "share confident", "trajectory error"]) + "\n")

def main():
    t0 = time.time(); out = []; Ws = {}
    for sd in (9, 21, 22, 23, 24, 25):
        Ws[sd] = world(sd, lambda m: print(f"[{time.time()-t0:6.1f}s] seed {sd}: {m}", file=sys.stderr, flush=True))
    out.append("## Main world (seed 9)\n"); report_main(Ws[9], out)
    out.append("## Six worlds (seed 9 plus robustness seeds 21-25)\n"); report_robust(Ws, out)
    print("\n".join(out)); print(f"\nwall seconds: {time.time()-t0:.1f}")

if __name__ == "__main__":
    main()
