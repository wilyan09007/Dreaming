"""S17 report: tables from results.csv (+ hypernetwork json files) -> tables.md. Usage: python -I report.py [results.csv]"""
import sys, json, glob, os
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
TINY = bool(os.environ.get("S17_TINY"))
MODEL_DIRS = [HERE / "models_json", HERE.parents[0] / ".data" / ("s17_tiny" if TINY else "s17_store") / "models"]
src = sys.argv[1] if len(sys.argv) > 1 else ("tiny_results.csv" if TINY else "results.csv")
df = pd.read_csv(HERE / src); df["cov"] = df["cov"].fillna(0.0)
FAMS = ["clinc", "banking", "txn"]; KS = [0, 1, 2, 3, 5, 10, 20]; KMEAN = {f: df[df.family == f].K.mean() for f in FAMS}
out = []
def P(s=""): out.append(s)

# per-seed macro over tasks within family -> then mean/std over seeds; 'all' = macro over the three families (TabPFN only covers banking, so its 'all' column is absent)
per = df.groupby(["arm", "family", "k", "seed"])[["acc", "cov"]].mean().reset_index()
allf = per.groupby(["arm", "k", "seed"])[["acc", "cov"]].agg(["mean", "count"]); allf = allf[allf[("acc", "count")] == len(FAMS)][[("acc", "mean"), ("cov", "mean")]]
allf.columns = ["acc", "cov"]; allf = allf.reset_index(); allf["family"] = "all"
per = pd.concat([per, allf])
agg = per.groupby(["arm", "family", "k"])[["acc", "cov"]].agg(["mean", "std"])
def cell(arm, fam, k, m, std=False):
    try:
        r = agg.loc[(arm, fam, k)]
    except KeyError:
        return "-"
    v = r[(m, "mean")]; s = r[(m, "std")]
    return f"{v:.3f}" + (f" ±{s:.3f}" if std and not np.isnan(s) else "")
ARM_ORDER = ["A1 baseline", "A6 nearest-prototype", "A6 desc+prototype", "A2 synthetic prior", "A2b synthetic prior (2nd LLM draw)", "A3 ft-all", "A3 ft-all+synth", "A3 ft-clinc7", "A3 ft-clinc7+synth",
             "A3 ft-nobank", "A3 ft-nobank+synth", "A3 ft-all_s1", "A3 ft-all_s1+synth", "A4 meta-learned start (FOMAML)", "A7 active learning (synth student)", "A8 ft-all+synth+active learning", "A6 TabPFN (real rows)", "A6 TabPFN (synth rows)"]
arms = [a for a in ARM_ORDER if a in set(df.arm)] + sorted(a for a in set(df.arm) if a not in ARM_ORDER)

def table(fam, m, title, std=False):
    P(f"\n#### {title}\n"); P("| arm | " + " | ".join(f"k={k}" for k in KS) + " |"); P("|---|" + "---|" * len(KS))
    for a in arms: P(f"| {a} | " + " | ".join(cell(a, fam, k, m, std) for k in KS) + " |")
for fam, nm in (("all", "macro-average of the three task families"), ("clinc", "held-out CLINC150 domains (15-way, 3 tasks)"), ("banking", "Banking77 12-label sub-tasks (5 tasks)"), ("txn", "bank transaction descriptors (17 labels)")):
    P(f"\n### Accuracy and certified coverage: {nm}")
    table(fam, "acc", f"Accuracy (mean ± std over the 5 row draws), {nm}", std=True); table(fam, "cov", f"Certified coverage at a 5% error budget (realized on the test rows), {nm}")

# rows needed for 50% certified coverage
P("\n### Real rows needed to reach 50% certified coverage at a 5% budget\n")
P("First k (rows per label) on the grid {0,1,2,3,5,10,20} whose mean certified coverage is >= 0.5 (interpolated rows per label in brackets; total rows = rows per label x labels). '>20' = not reached at 20 rows per label; '-' = arm not run for that family.\n")
P("| arm | " + " | ".join(FAMS + ["all"]) + " |"); P("|---|---|---|---|---|")
def need(arm, fam):
    cv = [(k, agg.loc[(arm, fam, k)][("cov", "mean")]) for k in KS if (arm, fam, k) in agg.index]
    if not cv: return "-"
    for j, (k, c) in enumerate(cv):
        if c >= 0.5:
            if j == 0: return f"{k}"
            k0, c0 = cv[j - 1]; ki = k0 + (0.5 - c0) / (c - c0) * (k - k0)
            nl = KMEAN.get(fam, np.mean(list(KMEAN.values())))
            return f"{k} ({ki:.1f}; ~{int(round(ki * nl))} rows)"
    return ">20"
for a in arms: P(f"| {a} | " + " | ".join(need(a, f) for f in FAMS + ["all"]) + " |")

# certificate validity
P("\n### Certificate check (realized test error above the cutoff)\n")
c = df[df.err.notna()].copy(); c["viol"] = c.err > 0.05
v = c.groupby("arm").agg(units=("viol", "size"), violations=("viol", "sum"), mean_err=("err", "mean"), max_err=("err", "max"))
P("| arm | units with a cutoff | realized error > 5% | mean realized error | max |"); P("|---|---|---|---|---|")
for a in arms:
    if a in v.index: r = v.loc[a]; P(f"| {a} | {int(r.units)} | {int(r.violations)} ({r.violations / r.units:.1%}) | {r.mean_err:.3f} | {r.max_err:.3f} |")

# hypernetwork scaling
P("\n### Hypernetwork: unseen-task accuracy versus number of pretraining tasks\n")
rows = []
for d in MODEL_DIRS:
    fs = sorted(glob.glob(str(d / "*.json")))
    if fs: rows = [j for j in (json.load(open(f)) for f in fs) if "model" in j]; break
if rows:
    P("| generator | pretraining tasks | pretraining labels | held-out TREC val (mean over k) | TREC val desc-cos | " + " | ".join(f"eval acc k={k}" for k in (0, 1, 3, 10)) + " | " + " | ".join(f"cert. cov k={k}" for k in (1, 3, 10)) + " |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|")
    order = {"clinc7": 0, "25": 1, "100": 2, "800": 3, "800nb": 4, "3000": 5}
    for j in sorted(rows, key=lambda j: (j["model"], order.get(str(j["scale"]), 9), j.get("seed", 0))):
        nm = f"A5 hyper {j['model']}_{j['scale']}" + (f"_s{j['seed']}" if j.get("seed") else "")
        dc = np.mean(list(j.get("val_desc_cos", {}).values())) if j.get("val_desc_cos") else float("nan")
        P(f"| {j['model']} ({j['scale']}{', seed ' + str(j['seed']) if j.get('seed') else ''}) | {j['n_tasks']} | {j['n_labels']} | {j['best_val']:.3f} | {dc:.3f} | " + " | ".join(cell(nm, "all", k, "acc") for k in (0, 1, 3, 10)) + " | " + " | ".join(cell(nm, "all", k, "cov") for k in (1, 3, 10)) + " |")

# noise floor
P("\n### Noise floor: the same recipe with a different random draw (macro-average over families)\n")
P("| pair | k | acc A | acc B | abs diff | cov A | cov B |"); P("|---|---|---|---|---|---|---|")
pairs = [("A2 synthetic prior", "A2b synthetic prior (2nd LLM draw)"), ("A3 ft-all", "A3 ft-all_s1"), ("A3 ft-all+synth", "A3 ft-all_s1+synth"),
         ("A5 hyper SETTR_800", "A5 hyper SETTR_800_s1"), ("A5 hyper LINGEN_800", "A5 hyper LINGEN_800_s1")]
for a, b in pairs:
    if a in arms and b in arms:
        for k in KS:
            try: x, y = agg.loc[(a, "all", k)], agg.loc[(b, "all", k)]
            except KeyError: continue
            P(f"| {a} vs {b} | {k} | {x[('acc','mean')]:.3f} | {y[('acc','mean')]:.3f} | {abs(x[('acc','mean')] - y[('acc','mean')]):.3f} | {x[('cov','mean')]:.3f} | {y[('cov','mean')]:.3f} |")
(HERE / ("tables_tiny.md" if TINY else "tables.md")).write_text("\n".join(out), encoding="utf-8"); print("\n".join(out)[:3000])
