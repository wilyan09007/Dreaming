"""S28 analysis: calibration-size sweep for the cheap-vs-E1 router. Usage: run.py   (stdout -> log.txt, tables -> results.json)
Rows: S5's 200 (E1 from s5 responses, cheap nano_A from s15 responses, both reused) + the new rows that have both answers (this smoke's responses.jsonl: 1,000 drawn, 929 with an E1 label).
Target = cheap label == E1 label (nearest-label parse). Signals: f1 student==cheap, b2 cheap logprob gap to best rival, f3 f1-then-b2, e student top-2 gap.
ponytail: one file; certificate = common.certify (40-level grid, delta 0.05)."""
import sys, json, re, difflib, warnings, collections, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from common import certify, apply_certificate, confidence, embed, load_banking77
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
from calls import data, ledger, sysmsg, RUNS, req_key
HERE = Path(__file__).parent; SM = HERE.parent
FLOOR = -12.0; BUDS = (0.05, 0.10, 0.15, 0.20); SIZES = (100, 200, 300, 500, 1000); NSPLIT = 200
OUT = []
def say(s=""): print(s, flush=True); OUT.append(s)

def parse(txt, names):  # = S5 / S15 parse()
    t = (txt or "").strip().strip("`'\".").strip().lower(); low = {n.lower(): n for n in names}
    if t in low: return low[t], False
    if not t: return None, True
    return low[difflib.get_close_matches(t, list(low), n=1, cutoff=0.0)[0]], True

def lp_parts(rec, lab, names):  # (label logprob, log-mass of the best listed rival branch, -inf if none is listed)
    toks = []
    for t in rec["toks"]:
        if re.fullmatch(r"<\|.*\|>", t[0].strip()): continue
        toks.append(t)
    while toks and not toks[0][0].strip(): toks = toks[1:]
    if not toks or lab is None: return np.nan, np.nan
    lab_l = [n.lower() for n in names]; chosen = lab.lower(); cum = 0.0; prefix = ""; best = -np.inf
    for tok, lp, alts in toks:
        for a, alp in alts:
            if a == tok: continue
            cand = (prefix + a).strip().lower()
            if not cand: continue
            m = [l for l in lab_l if l.startswith(cand)]
            if not m or m == [chosen]: continue
            best = max(best, cum + alp)
        cum += lp; prefix += tok
    return cum, best

def lp_gap(rec, lab, names):  # = S15 lp_feats()[2] (b2): returned-label logprob minus log-mass of the best rival branch
    cum, best = lp_parts(rec, lab, names); return cum - max(best, FLOOR)

def split_run(sc, tgt, n, b, N):  # the sweep's 200 splits for one signal/size/budget with per-split detail (revision checks only)
    cov, errs, nab, pv = [], [], [], 0
    for s in range(NSPLIT):
        perm = np.random.default_rng(2800 + s).permutation(N); cal, hold = perm[:n], perm[n:]
        c = certify(sc[cal], tgt[cal], b)
        if c["cutoff"] == np.inf: cov.append(0.0); continue
        h = apply_certificate(sc[hold], tgt[hold], c["cutoff"]); cov.append(h["coverage"]); nab.append(h["n_above"])
        if h["error"] is not None: errs.append(h["error"])
        pv += apply_certificate(sc, tgt, c["cutoff"])["error"] > b  # the certified cutoff scored on the whole pool (taken as the truth)
    return np.array(cov), np.array(errs), np.array(nab), int(pv)

def auc_ci(score, pos, nb=1000, seed=0):
    rng = np.random.default_rng(seed); a = roc_auc_score(pos, score); d = []
    for _ in range(nb):
        b = rng.integers(0, len(pos), len(pos))
        if pos[b].all() or (~pos[b]).all(): continue
        d.append(roc_auc_score(pos[b], score[b]))
    return a, np.percentile(d, 2.5), np.percentile(d, 97.5)

def pdiff(s1, s2, pos, nb=1000, seed=1):
    rng = np.random.default_rng(seed); d = []
    for _ in range(nb):
        b = rng.integers(0, len(pos), len(pos))
        if pos[b].all() or (~pos[b]).all(): continue
        d.append(roc_auc_score(pos[b], s1[b]) - roc_auc_score(pos[b], s2[b]))
    return roc_auc_score(pos, s1) - roc_auc_score(pos, s2), np.percentile(d, 2.5), np.percentile(d, 97.5)

def wilson(k, n, z=1.96):
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n); return c - h, c + h

def load_rows():
    names, texts, labels, old, new = data(); sm = sysmsg(names); led = ledger()
    s5 = {}
    for l in (SM / "s5_tier_router" / "responses.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        if r["err"] is None: s5.setdefault((r["cfg"], r["i"]), r)
    s15 = {}
    for r in (json.loads(l) for l in (SM / "s15_confidence_signals" / "responses.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()):
        if r["err"] is None and r["run"] == "nano_A": s15.setdefault(r["i"], r)
    rows = []  # (test index, E1 record, cheap record, source)
    for k, ti in enumerate(old): rows.append((int(ti), s5[("E1", k)], s15[k], "old"))
    for ti in new:
        ti = int(ti); e = led.get(req_key("E1", ti, RUNS["E1"][0], sm, texts[ti], RUNS["E1"][1])); c = led.get(req_key("nano_A", ti, RUNS["nano_A"][0], sm, texts[ti], RUNS["nano_A"][1]))
        if e and c: rows.append((ti, e, c, "new"))
    return names, texts, labels, rows

def main():
    t0 = time.time(); names, texts, labels, rows = load_rows(); N = len(rows)
    ti = np.array([r[0] for r in rows]); src = np.array([r[3] for r in rows]); gold = np.array([names[labels[i]] for i in ti], object)
    E1 = np.array([parse(r[1]["content"], names)[0] for r in rows], object); CH = np.array([parse(r[2]["content"], names)[0] for r in rows], object)
    pf_e = sum(parse(r[1]["content"], names)[1] for r in rows); pf_c = sum(parse(r[2]["content"], names)[1] for r in rows)
    b2 = np.array([lp_gap(r[2], CH[k], names) for k, r in enumerate(rows)]); has_lp = np.isfinite(b2)
    cE = np.mean([r[1]["usd"] for r in rows]); cC = np.mean([r[2]["usd"] for r in rows])
    cE_new = np.mean([r[1]["usd"] for r in rows if r[3] == "new"]); cC_new = np.mean([r[2]["usd"] for r in rows if r[3] == "new"])
    bk = load_banking77(); Xtr, ytr = embed(bk["train"][0]), bk["train"][1]; X = embed([texts[i] for i in ti])
    st = LogisticRegression(C=10, max_iter=2000).fit(Xtr, ytr); Ps = st.predict_proba(X); sp = np.array([names[k] for k in Ps.argmax(1)], object); sg = confidence(Ps)
    tgt = CH == E1
    say(f"# S28 run: {N} rows ({(src=='old').sum()} reused from S5/S15, {(src=='new').sum()} new). Parse fallbacks: E1 {pf_e}, cheap {pf_c}. Cheap rows with logprob gap: {has_lp.sum()}/{N}.")
    mt = tgt.mean(); lo, hi = wilson(int(tgt.sum()), N)
    say(f"match rate (cheap == E1): {mt:.3f} [Wilson 95% {lo:.3f}, {hi:.3f}]; old-200 {tgt[src=='old'].mean():.3f}, new-1000 {tgt[src=='new'].mean():.3f}")
    ge, gc, gs = np.mean(E1 == gold), np.mean(CH == gold), np.mean(sp == gold)
    say(f"gold accuracy: E1 {ge:.3f} (old {np.mean((E1==gold)[src=="old"]):.3f}, new {np.mean((E1==gold)[src=='new']):.3f}); cheap {gc:.3f}; student {gs:.3f}; student==E1 {np.mean(sp==E1):.3f}; student==cheap {np.mean(sp==CH):.3f}")
    say(f"cost per 1k calls (observed usage.cost): E1 ${1000*cE:.3f} (new rows ${1000*cE_new:.3f}), cheap ${1000*cC:.3f} (new rows ${1000*cC_new:.3f}); ratio {cE/cC:.1f}x. Self-disagreement floor of E1 (S15): 6% -> achievable mismatch cannot go reliably below ~6%.")
    f1 = (sp == CH).astype(float); f3 = f1 * 1000 + np.nan_to_num(b2, nan=-50)
    SIG = {"f1 student==cheap": (f1, True), "b2 cheap logprob gap": (np.nan_to_num(b2, nan=np.nanmin(b2) - 1), True), "f3 f1 then b2": (f3, True), "e student top-2 gap": (sg, False)}
    res = dict(n=N, n_new=int((src == "new").sum()), match=float(mt), gold=dict(E1=float(ge), cheap=float(gc), student=float(gs)), usd_per_1k=dict(E1=1000 * cE, cheap=1000 * cC), auroc={}, sweep={})
    say("\n## AUROC for 'cheap == E1' on all rows (95% bootstrap CI, 1000 resamples)")
    say("| signal | AUROC [CI] | old 200 | new 1000 |"); say("|---|---|---|---|")
    for k, (s, _) in SIG.items():
        a, l, h = auc_ci(s, tgt); ao = roc_auc_score(tgt[src == "old"], s[src == "old"]); an = roc_auc_score(tgt[src == "new"], s[src == "new"])
        res["auroc"][k] = [a, l, h]; say(f"| {k} | {a:.3f} [{l:.3f}, {h:.3f}] | {ao:.3f} | {an:.3f} |")
    for a_, b_ in (("f3 f1 then b2", "b2 cheap logprob gap"), ("f3 f1 then b2", "f1 student==cheap"), ("f1 student==cheap", "b2 cheap logprob gap")):
        d = pdiff(SIG[a_][0], SIG[b_][0], tgt); say(f"paired {a_} minus {b_}: {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"); res["auroc"][f"{a_} - {b_}"] = list(d)
    say("\n## Calibration-size sweep (200 random splits per size; calibration = n rows, holdout = the other rows; certificate = common.certify, 40-level grid, delta 5%)")
    say("cert% = share of splits where anything certified; cov = certified share of the holdout routed to the cheap model (0 when nothing certified; mean [5th pct]); err = realized holdout mismatch among routed rows (mean over certifying splits, max); viol% = share of certifying splits whose realized holdout error exceeded the budget; saving = 1 - blended $/1k / E1 $/1k at mean cov (f1/b2/f3 need the cheap call first, so blended = cheap + (1-cov) x E1; student gap decides first, blended = cov x cheap + (1-cov) x E1).")
    cells = {}
    for n in SIZES:
        splits = []
        for s in range(NSPLIT):
            perm = np.random.default_rng(2800 + s).permutation(N); splits.append((perm[:n], perm[n:]))
        for k, (sc, pre) in SIG.items():
            for b in BUDS:
                cov, errs, viol = [], [], 0
                for cal, hold in splits:
                    c = certify(sc[cal], tgt[cal], b)
                    if c["cutoff"] == np.inf: cov.append(0.0); continue
                    h = apply_certificate(sc[hold], tgt[hold], c["cutoff"]); cov.append(h["coverage"])
                    if h["error"] is not None: errs.append(h["error"]); viol += h["error"] > b
                cv = np.array(cov); mc = cv.mean(); cost = (cC + (1 - mc) * cE) if pre else (mc * cC + (1 - mc) * cE)
                cells[(n, k, b)] = dict(cert=float((cv > 0).mean()), cov=float(mc), cov_p5=float(np.percentile(cv, 5)), err=float(np.mean(errs)) if errs else None,
                                        err_max=float(np.max(errs)) if errs else None, viol=float(viol / len(errs)) if errs else None, saving=float(1 - cost / cE), nhold=N - n)
        res["sweep"][str(n)] = {k: {str(b): cells[(n, k, b)] for b in BUDS} for k in SIG}
    for k in SIG:
        say(f"\n### {k}")
        say("| cal n (holdout) | " + " | ".join(f"{int(b*100)}%: cert% / cov [p5] / err (max) / viol% / saving" for b in BUDS) + " |"); say("|---|" + "---|" * len(BUDS))
        for n in SIZES:
            row = []
            for b in BUDS:
                c = cells[(n, k, b)]; row.append(f"{100*c['cert']:.0f} / {c['cov']:.2f} [{c['cov_p5']:.2f}] / " + (f"{c['err']:.3f} ({c['err_max']:.3f}) / {100*c['viol']:.0f}" if c["err"] is not None else "- / -") + f" / {100*c['saving']:.0f}%")
            say(f"| {n} ({N-n}) | " + " | ".join(row) + " |")
    say("\n## Recommended calibration size per budget (signal f3): smallest n where >=95% of splits certify and mean coverage >= 90% of the n=1000 coverage")
    rec = {}
    for b in BUDS:
        ref = cells[(1000, "f3 f1 then b2", b)]["cov"]; pick = None
        for n in SIZES:
            c = cells[(n, "f3 f1 then b2", b)]
            if c["cert"] >= 0.95 and ref > 0 and c["cov"] >= 0.9 * ref: pick = n; break
        rec[str(b)] = pick; say(f"budget {int(b*100)}%: n = {pick}  (coverage at n=1000: {ref:.2f}; at pick: {cells[(pick,'f3 f1 then b2',b)]['cov']:.2f})" if pick else f"budget {int(b*100)}%: none of {SIZES} reaches the criterion (cov at n=1000 {ref:.2f})")
    res["recommended_f3"] = rec
    say("\n## In-sample descriptive ceiling (all rows, no guarantee): largest routed share (ties closed) with mismatch <= budget, f3")
    o = np.argsort(-f3, kind="stable"); s_, m_ = f3[o], (~tgt)[o]; cum = np.cumsum(m_) / (np.arange(N) + 1); ceil = {}
    for b in BUDS:
        best = 0
        for k in range(N, 0, -1):
            if k < N and s_[k] == s_[k - 1]: continue
            if cum[k - 1] <= b: best = k; break
        ceil[str(b)] = best / N; say(f"budget {int(b*100)}%: {best/N:.2f}")
    res["insample_f3"] = ceil
    # ---- Revision checks (added after the critique). The sweep, the rule and every number above are untouched.
    say("\n## Revision check 1: f3 at n=%d (holdout exactly 200 rows; n=1000 leaves only %d)" % (N - 200, N - 1000))
    n9 = N - 200; rv = dict(n929={}, pool={}, rule_ref929={}, rival={}); f3c = {}
    for b in BUDS:
        cv, er, nab, pv = split_run(f3, tgt, n9, b, N); f3c[b] = cv.mean()
        rv["n929"][str(b)] = d = dict(cert=float((cv > 0).mean()), cov=float(cv.mean()), cov_p5=float(np.percentile(cv, 5)), err=float(er.mean()), err_max=float(er.max()), viol=float((er > b).mean()), saving=float(1 - (cC + (1 - cv.mean()) * cE) / cE))
        say(f"budget {int(b*100)}%: cert {100*d['cert']:.0f}% cov {d['cov']:.3f} [p5 {d['cov_p5']:.2f}] mismatch {d['err']:.3f} ({d['err_max']:.3f}) viol {100*d['viol']:.1f}% saving {100*d['saving']:.0f}%")
    say("rule re-applied with the n=929 coverage as the reference instead of n=1000:")
    for b in BUDS:
        pick = next((n for n in SIZES if cells[(n, "f3 f1 then b2", b)]["cert"] >= 0.95 and cells[(n, "f3 f1 then b2", b)]["cov"] >= 0.9 * f3c[b]), None)
        rv["rule_ref929"][str(b)] = pick; say(f"budget {int(b*100)}%: n = {pick} (0.9 x cov at 929 = {0.9*f3c[b]:.3f}; 0.9 x cov at 1000 = {0.9*cells[(1000,'f3 f1 then b2',b)]['cov']:.3f})")
    say("\n## Revision check 2: are the holdout violations at n=500 / 929 / 1000 certificate failures? (f3; each certified cutoff also scored on all %d rows)" % N)
    for n in (500, n9, 1000):
        for b in BUDS:
            cv, er, nab, pv = split_run(f3, tgt, n, b, N); nc = len(er); p = er.mean(); m = nab.mean()
            rv["pool"][f"{n}_{b}"] = d = dict(n_cert=nc, pool_viol=pv, holdout_viol=int((er > b).sum()), routed_min=int(nab.min()), routed_max=int(nab.max()), routed_mean=float(m), sd_binom=float(np.sqrt(p * (1 - p) / m)), sd_emp=float(er.std()))
            say(f"n={n} budget {int(b*100)}%: pool-level violations {pv}/{nc}; holdout violations {d['holdout_viol']}/{nc}; routed holdout rows {d['routed_min']}-{d['routed_max']} (mean {m:.0f}); sd of one holdout's mismatch: binomial {100*d['sd_binom']:.1f} pts, across splits {100*d['sd_emp']:.1f} pts")
    say("\n## Revision check 3: cheap answers with no rival label in the top-5 alternatives (b2 uses the 12-nat floor for them)")
    best = np.array([lp_parts(r[2], CH[k], names)[1] for k, r in enumerate(rows)]); nr = ~np.isfinite(best); mm = ~tgt; hi_ = b2 >= 11.999
    rv["rival"] = dict(no_rival=int(nr.sum()), no_rival_mismatch=int(mm[nr].sum()), rival=int((~nr).sum()), rival_mismatch=int(mm[~nr].sum()), total_mismatch=int(mm.sum()),
                       distinct_b2_no_rival=len(set(b2[nr].round(9))), distinct_b2_all=len(set(b2.round(9))), b2_ge_11999=int(hi_.sum()), b2_ge_11999_mismatch=int(mm[hi_].sum()))
    say(f"no rival listed: {nr.sum()} of {N} ({100*nr.mean():.0f}%), mismatch with E1 {mm[nr].sum()} rows ({100*mm[nr].mean():.1f}%)")
    say(f"rival listed: {(~nr).sum()} rows, mismatch {mm[~nr].sum()} rows ({100*mm[~nr].mean():.1f}%) = {mm[~nr].sum()} of the {mm.sum()} mismatches ({100*mm[~nr].sum()/mm.sum():.0f}%)")
    say(f"distinct b2 values: {rv['rival']['distinct_b2_no_rival']} among the no-rival rows, {rv['rival']['distinct_b2_all']} over all rows (no-rival rows are not tied: b2 = 12 + label logprob)")
    say(f"subset with b2 >= 11.999 (no rival and label logprob above -0.001): {hi_.sum()} rows, mismatch {mm[hi_].sum()} ({100*mm[hi_].mean():.1f}%); the other {(~hi_).sum()} rows mismatch {100*mm[~hi_].mean():.1f}%")
    trn = {t.strip().lower() for t in bk["train"][0]}; dup = np.array([texts[i].strip().lower() in trn for i in ti])
    say(f"\nexact-text duplicates of a Banking77 training input among the {N} rows: {dup.sum()} (old {dup[src=='old'].sum()}, new {dup[src=='new'].sum()}); the split is disjoint by index")
    res["revision"] = rv
    say(f"\nwall time {time.time()-t0:.0f}s")
    (HERE / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

if __name__ == "__main__": main()
