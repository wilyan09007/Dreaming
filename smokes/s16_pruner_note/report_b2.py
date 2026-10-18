"""Recompute every number in RESULTS_B2.md from results_b2.json (and the by-scenario floor lines of log_b2_live.txt).
Usage: python report_b2.py   -> prints markdown tables and writes tables_b2.md. No network, no model call."""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
from math import ceil, log
from scipy.stats import beta, binomtest
HERE = Path(__file__).resolve().parent
R = json.load(open(HERE / "results_b2.json")); LOG = (HERE / "log_b2_live.txt").read_text(encoding="utf-8")
cp = lambda k, n, d=0.05: 1.0 if n == 0 or k >= n else float(beta.ppf(1 - d, k + 1, n - k))
pc = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"
out = []
def P(s=""): out.append(s); print(s)

# ---------------------------------------------------------------- 1. floor
P("### Floor on the same pairs: new equivalence vs strict rubric\n")
P("| pairs | n | new equivalence DIFFERENT (CP UB) | strict raw (CP UB) | strict confirmed (CP UB) |\n|---|---|---|---|---|")
for k, v in R["floor"].items():
    f = lambda o: f"{o['k']}/{o['n']} = {pc(o['rate'])} ({pc(o['ub'])})"
    P(f"| {k} | {v['new']['n']} | {f(v['new'])} | {f(v['strict_raw'])} | {f(v['strict_cons'])} |")
P("\nBy scenario (the 140 labelled pairs only: the log prints by-scenario over LAB, not the 180; new equivalence; parsed from log_b2_live.txt):\n")
P("| scenario | n | DIFFERENT | CP UB |\n|---|---|---|---|")
for m in re.finditer(r"by scenario (\w+)\s+n_rows=\s*(\d+) (\d+)/(\d+) = ([\d.]+) \(CP UB ([\d.]+)\)", LOG):
    s, nr, k, n, rt, u = m.groups(); P(f"| {s} | {n} | {k}/{n} = {pc(float(rt))} | {pc(float(u))} |")
tf = R["typed_floor"]["rate"]; P(f"\nTyped floor (two independent typed records, field equality): {tf['k']}/{tf['n']} = {pc(tf['rate'])} (CP UB {pc(tf['ub'])}); differing fields {R['typed_floor']['fields']}.")
P("Typed record vs prompt ground truth (copy fidelity, n=140): " + ", ".join(f"{k} {v}" for k, v in R["typed_gt_accuracy"].items() if k != "n"))

# ---------------------------------------------------------------- 2. relabel
rl = R["relabel"]; P("\n### Span labels, new equivalence vs old strict labels (140 prompts)\n")
P("| ablated span | new DIFFERENT | old strict | pairs |\n|---|---|---|---|")
for s in ("tx", "inv", "flag", "bank"): P(f"| {s} | {pc(rl['new_fixed'][s])} ({round(rl['new_fixed'][s]*140)}/140) | {pc(rl['old_fixed'][s])} | {rl['per_name'][s]['pairs']} |")
P(f"| whole history (H0) | {pc(rl['h0_new'])} ({round(rl['h0_new']*140)}/140) | 40.7% (57/140, from the log) | 140 |")
P(f"\nRelabel paths over {rl['pairs']} pairs: {rl['paths']}; FAIL {rl['fail']}.")

# ---------------------------------------------------------------- 3. cells
def cells(arm, key):
    P(f"\n### Pruned cells, {arm}\n")
    P("| cell | tokens removed | DIFFERENT (k/n) | CP UB | vendor-bootstrap | floor, same rows | excess | discordant pruned-only/floor-only | exact p |\n|---|---|---|---|---|---|---|---|---|")
    for name, o in R[key].items():
        v = o["vs_floor"]; b, c = v["pruned_only"], v["floor_only"]; p = binomtest(b, b + c, .5).pvalue if b + c else 1.0
        assert abs(p - v["p"]) < 1e-9
        P(f"| {name} | {pc(o['tokens_removed'])} | {o['k']}/{o['n']} = {pc(o['rate'])} | {pc(o['ub'])} | {pc(o['ci_vendor'][0])}-{pc(o['ci_vendor'][1])} | {pc(v['floor_rate'])} | {100*(o['rate']-v['floor_rate']):+.1f} pt | {b}/{c} | {p:.3f} |")
cells("free text", "cells"); cells("typed", "cells_typed")
P("\nStrict rubric on the same rows (secondary, first three cells):\n\n| cell | strict raw | strict confirmed |\n|---|---|---|")
for name, o in R["cells"].items():
    if "strict_raw" in o: P(f"| {name} | {o['strict_raw']['k']}/140 = {pc(o['strict_raw']['rate'])} (UB {pc(o['strict_raw']['ub'])}) | {o['strict_cons']['k']}/140 = {pc(o['strict_cons']['rate'])} (UB {pc(o['strict_cons']['ub'])}) |")
P("\nText DIFFERENT by scenario (n: dup 20, first 28, noinv 30, normal 45, spike 17):\n\n| cell | dup | first | noinv | normal | spike |\n|---|---|---|---|---|---|")
for arm, key in (("text", "cells"), ("typed", "cells_typed")):
    for name, o in R[key].items(): P(f"| {arm}: {name} | " + " | ".join(f"{round(o['by_scn'][s]['dis']*o['by_scn'][s]['n'])}/{o['by_scn'][s]['n']}" for s in ("dup", "first", "noinv", "normal", "spike")) + " |")

# ---------------------------------------------------------------- 4. certificates
P("\n### Certificate table (rotation cross-fitting, 140 usable rows each)\n")
P("Held-out covered rows per budget; every one is 0. 'In-sample pooled' = the same grid certifier run once on all 140 rows (no calibration/test split, NOT a valid certificate, shown only to show what the observed rates would allow).\n")
P("| arm | cell | plain 5/10/15% covered | excess 5/10/15% covered | in-sample pooled plain 10% (coverage, err, bound) | in-sample pooled plain 15% |\n|---|---|---|---|---|---|")
for arm, key in (("text", "cert"), ("typed", "cert_typed")):
    for name, c in R[key].items():
        g = lambda b: c[b]["covered"]; q = lambda b: c[b]["pooled_in_sample"]
        f = lambda x: "n/m (label-leaked)" if (arm, name) == ("text", "oracle m=0") else "none" if not x["coverage"] else f"{100*x['coverage']:.0f}%, {pc(x['emp_error'])}, {pc(x['bound'])}"
        P(f"| {arm} | {name} | {g('5%')}/{g('10%')}/{g('15%')} | {g('excess@5%')}/{g('excess@10%')}/{g('excess@15%')} | {f(q('10%'))} | {f(q('15%'))} |")
c0 = R["cert"]["learned m=0"]["10%"]; P(f"\nFolds with a cutoff: {max(c['folds_with_cutoff'] for k in ('cert','cert_typed') for v in R[k].values() for c in v.values())} of 5 in every cell. Calibration rows per rotation: {c0['n_calibration_per_rotation']}.")
# minimum error-free calibration size for the grid certifier (delta/40) and a plain one-sided 95% bound
for d, nm in ((0.05 / 40, "grid certifier, delta=0.05/40"), (0.05, "plain one-sided 95%")):
    P(f"- Zero errors, {nm}: smallest n with CP UB <= budget: " + ", ".join(f"{int(b*100)}%: {ceil(log(d)/log(1-b))}" for b in (0.05, 0.10, 0.15)))

# ---------------------------------------------------------------- 5. what would certify
def need(p, b, d):
    for n in range(5, 20000):
        k = round(p * n)
        if cp(k, n, d) <= b and all(cp(round(p * m), m, d) <= b for m in range(n, n + 40)): return n
P("\n### Rows needed if the observed rate held (CP upper bound <= budget, errors = round(rate x n))\n")
P("| observed DIFFERENT rate | source | 5%, plain | 10%, plain | 15%, plain | 10%, grid certifier | 15%, grid certifier |\n|---|---|---|---|---|---|---|")
src = [(R["floor"]["all floor prompts"]["new"]["rate"], "new-equivalence floor, 180 pairs"), (R["cells"]["learned m=5"]["rate"], "text learned m=5 / m=2 (4/140)"),
       (R["cells"]["cross-encoder m=5"]["rate"], "text cross-encoder/oracle m=5 (5/140)"), (R["cells"]["rule (tx+inv+flag)"]["rate"], "text rule (9/140)"),
       (R["cells"]["random m=2"]["rate"], "text random m=2 (13/140)"), (R["cells"]["learned m=0"]["rate"], "text learned m=0 (16/140)"),
       (R["typed_floor"]["rate"]["rate"], "typed floor (5/140)"), (R["cells_typed"]["cross-encoder m=5"]["rate"], "typed cross-encoder/oracle m=5 (12/140)")]
for p, s in src:
    cs = [need(p, b, 0.05) for b in (0.05, 0.10, 0.15)] + [need(p, b, 0.05 / 40) for b in (0.10, 0.15)]
    P(f"| {pc(p)} | {s} | " + " | ".join("never (rate >= budget)" if x is None else str(x) for x in cs) + " |")
P("\nWith one calibration fold per rotation (a fifth of the pool), the pool size is five times the 'grid certifier' columns. These are rows of an error-free-looking sample at the observed rate; they are the minimum, not a power calculation, and treat rows as independent (rows share vendors and prompts).")

# ---------------------------------------------------------------- REVISION tables (critique): power, tx-label anatomy, strict floor by scenario, vendor folds
from scipy.stats import binom
def cert_prob(n, p, b, d):
    """chance that a calibration sample of n rows with true error rate p has CP upper bound <= b (delta d) = P(K <= kmax)"""
    kmax = -1
    for k in range(n + 1):
        if cp(k, n, d) <= b: kmax = k
        else: break
    return 0.0 if kmax < 0 else float(binom.cdf(kmax, n, p))
def n_power(p, b, d, power=0.8):
    for n in range(10, 6000):
        if cert_prob(n, p, b, d) >= power and all(cert_prob(m, p, b, d) >= power for m in range(n, n + 150)): return n
P("\n### REVISION: chance a calibration sample certifies at true error rate p (binomial sampling), grid certifier delta=0.05/40\n")
P("| calibration rows | budget | p=2.9% | p=3.6% |\n|---|---|---|---|")
for n, b in ((28, .15), (28, .10), (69, .15), (84, .15), (84, .10), (140, .15), (140, .10)):
    P(f"| {n} | {int(b*100)}% | {cert_prob(n, 4/140, b, .05/40):.2f} | {cert_prob(n, 5/140, b, .05/40):.2f} |")
P("\nRows for 80% power (grid certifier; power stays >= 80% for the next 150 sizes) and the one-fold pool (x5):\n\n| true rate | budget | calibration rows | pool (x5) |\n|---|---|---|---|")
for p_, nm_ in ((4/140, "2.9%"), (5/140, "3.6%")):
    for b in (.15, .10):
        n = n_power(p_, b, .05 / 40); P(f"| {nm_} | {int(b*100)}% | {n} | {5*n} |")
P(f"\nCP upper bound (plain 95%) of 4/140: {pc(cp(4,140))}; 5/140: {pc(cp(5,140))}")

# folds: every rotation calibrates on exactly one vendor-grouped fold
try:
    from sklearn.model_selection import GroupKFold
    import sys as _s; _s.path.insert(0, str(HERE)); _s.path.insert(0, str(HERE.parent)); _s.path.insert(0, str(HERE.parent / "s16c_judge_floor_offline"))
    import equiv as E; E.setup(True)
    raw_p = {int(p["id"]): p for p in json.load(open(HERE / "prompts.json"))}
    LABS = sorted(int(k) for k in json.load(open(HERE / "labels.json")))
    sizes = [len(te) for _, te in GroupKFold(5).split(LABS, groups=[raw_p[i]["vendor"] for i in LABS])]
    P(f"\nVendor GroupKFold(5) fold sizes on the 140 prompts: {sizes}. Smallest error-free calibration n for the grid certifier at 15%: {ceil(log(0.05/40)/log(0.85))}; so no fold of {max(sizes)} rows can certify any budget <= 15%.")
    # tx-label anatomy, cache-only (same canonicalizer and compare as the run)
    L0 = json.load(open(HERE / "labels.json")); cnt = Counter(); src = Counter(); lost = 0
    for k, l in L0.items():
        p = raw_p[int(k)]; ca = E.canon2(l["refs"][0], p["tx"]); cb = E.canon2(l["var"]["tx"]["note"], p["tx"])
        v, path = E.canon_decide(ca, cb, "core")
        if path.startswith("canon-lost"): lost += 1
        if path == "canon-conflict":
            conf = E.compare(ca, cb)[0]; cnt[tuple(sorted(conf))] += 1
            if "tx_date" in conf: src["extractor took the invoice issue date" if cb["tx_date"] == cb.get("inv_date") else "other date mismatch"] += 1
    P(f"\n`tx` ablation pairs (140): canon-lost {lost}; canon-conflict by conflicting fields {dict(cnt)}; of the tx_date conflicts: {dict(src)}. Pairs that turn on tx_date: {lost + sum(v for k_, v in cnt.items() if 'tx_date' in k_)} of 140.")
    # strict rubric by scenario on the 180 floor pairs (cache-only, same function as the run)
    import core as _core
    FIDS = sorted(set(LABS) | set(range(200, 300))); rows = defaultdict(list); rows140 = defaultdict(list)
    for i in FIDS:
        p = raw_p[i]; k = _core.keep_all(p)
        a0, a3 = E.note_c(p, k, 0), E.note_c(p, k, 3); r1, r2 = E.note_c(p, k, 1), E.note_c(p, k, 2)
        if not (a0 and a3 and r1 and r2): continue
        s_ = E.strict([a0["text"], r1["text"], r2["text"]], a3["text"]); rows[p["scn"]].append(s_)
        if i in LABS: rows140[p["scn"]].append(s_)
    P("\nStrict rubric by scenario on the 180 floor pairs (cache-only):\n\n| scenario | n | strict raw | strict confirmed |\n|---|---|---|---|")
    tr = tc = tn = 0
    for sc in ("dup", "first", "noinv", "normal", "spike"):
        rs = rows[sc]; kr = sum(r["raw"] == "DIFFERENT" for r in rs); kc = sum(r["cons"] == "DIFFERENT" for r in rs); n = len(rs); tr += kr; tc += kc; tn += n
        P(f"| {sc} | {n} | {kr}/{n} = {pc(kr/n)} | {kc}/{n} = {pc(kc/n)} |")
    P(f"| all | {tn} | {tr}/{tn} = {pc(tr/tn)} | {tc}/{tn} = {pc(tc/tn)} |")
    P("\nSame, 140 labelled pairs only (the rows the log's by-scenario new-equivalence lines use):\n\n| scenario | n | strict raw | strict confirmed |\n|---|---|---|---|")
    for sc in ("dup", "first", "noinv", "normal", "spike"):
        rs = rows140[sc]; kr = sum(r["raw"] == "DIFFERENT" for r in rs); kc = sum(r["cons"] == "DIFFERENT" for r in rs); n = len(rs)
        P(f"| {sc} | {n} | {kr}/{n} = {pc(kr/n)} | {kc}/{n} = {pc(kc/n)} |")
except Exception as e:  # cache-only extras need the caches and scikit-learn; the main tables above do not
    P(f"\n[cache-only extras skipped: {type(e).__name__}: {e}]")
Path(HERE / "tables_b2.md").write_text("\n".join(out) + "\n", encoding="utf-8")
