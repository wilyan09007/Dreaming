"""S24 salvaged report: salvaged_results.json -> markdown tables on stdout. Usage: python3 -I report_salvaged.py
Means are over tasks x draws (5 tasks x 3 seeds = 15 cells per arm and k); '+-' is the s.d. over the 3 draws of the task-averaged value."""
import json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
R = json.load(open(HERE / "salvaged_results.json")); df = pd.DataFrame(R["records"]); KS = [1, 3, 10]
for c in ("acc", "cov", "err", "auroc", "abstain", "acc_fired", "fit_s", "pred_ms_1000", "params", "llm_cost", "code_chars"): df[c] = pd.to_numeric(df.get(c), errors="coerce")
ARMS = sorted(df.arm.unique()); BASE = ARMS[0]
def P(s=""): print(s)
P(f"records {len(df)}; arms {len(ARMS)}; cells per arm {df.groupby('arm').size().unique().tolist()}; scored (acc not NaN) per arm: {df[df.acc.notna()].groupby('arm').size().to_dict()}")
sc = df[df.acc.notna()]
def mat(col, d=sc, sd=True):
    g = d.groupby(["arm", "seed", "k"])[col].mean().reset_index().groupby(["arm", "k"])[col].agg(["mean", "std"])
    return g
def cell(g, a, k, sd=True, f="{:.3f}"):
    if (a, k) not in g.index: return "-"
    r = g.loc[(a, k)]; return f.format(r["mean"]) + (f" +- {r['std']:.3f}" if sd and not np.isnan(r["std"]) else "")
ga, gc = mat("acc"), mat("cov")
P("\n### Main table (accuracy | certified coverage at 5%)\n")
P("| arm | acc k=1 | acc k=3 | acc k=10 | cov k=1 | cov k=3 | cov k=10 |"); P("|---|---|---|---|---|---|---|")
for a in ARMS: P(f"| {a} | " + " | ".join(cell(ga, a, k) for k in KS) + " | " + " | ".join(cell(gc, a, k, f="{:.3f}") for k in KS) + " |")
def need50(c):
    pts = [(k, c[k]) for k in KS if k in c]
    if pts[0][1] >= .5: return "<=1"
    for (k0, c0), (k1, c1) in zip(pts, pts[1:]):
        if c1 >= .5: return f"{np.exp(np.log(k0) + (.5 - c0) / (c1 - c0) * (np.log(k1) - np.log(k0))):.1f}"
    return ">10"
P("\n### Rows per label to reach 50% certified coverage (log-k interpolation of the all-task mean)\n"); P("| arm | all | CLINC | Banking77 |"); P("|---|---|---|---|")
for a in ARMS:
    out = []
    for sel in (sc, sc[sc.family == "clinc"], sc[sc.family == "banking"]):
        g = mat("cov", sel); out.append(need50({k: g.loc[(a, k)]["mean"] for k in KS if (a, k) in g.index}))
    P(f"| {a} | " + " | ".join(out) + " |")
P("\n### Per family: accuracy / certified coverage (CLINC | Banking77)\n"); P("| arm | k=1 | k=3 | k=10 |"); P("|---|---|---|---|")
for a in ARMS:
    cs = []
    for k in KS:
        p = []
        for fam in ("clinc", "banking"):
            d = sc[(sc.arm == a) & (sc.k == k) & (sc.family == fam)]; p.append(f"{d.acc.mean():.2f}/{d['cov'].mean():.2f}")
        cs.append(" | ".join(p) if False else " ; ".join(p))
    P(f"| {a} | " + " | ".join(cs) + " |")
P("\n### AUROC of the confidence score for 'prediction is correct'\n"); ga_ = mat("auroc"); P("| arm | k=1 | k=3 | k=10 |"); P("|---|---|---|---|")
for a in ARMS: P(f"| {a} | " + " | ".join(cell(ga_, a, k, sd=False) for k in KS) + " |")
P("\n### Paired difference to baseline LR (same task, k, draw): mean +- s.e. (n cells)\n"); P("| arm | d-cov k=1 | d-cov k=3 | d-cov k=10 | d-acc k=1 | d-acc k=3 | d-acc k=10 |"); P("|---|---|---|---|---|---|---|")
key = ["task", "k", "seed"]; b = sc[sc.arm == BASE].set_index(key)
for a in ARMS[1:]:
    j = sc[sc.arm == a].set_index(key).join(b, rsuffix="_b", how="inner"); cs = []
    for col in ("cov", "acc"):
        for k in KS:
            m = j.index.get_level_values("k") == k; d = (j[m][col] - j[m][col + "_b"]).dropna()
            cs.append(f"{d.mean():+.3f} +- {d.std(ddof=1)/np.sqrt(len(d)):.3f} ({len(d)})" if len(d) > 1 else "-")
    P(f"| {a} | " + " | ".join(cs) + " |")
P("\n### Per task: accuracy / coverage (mean over draws)")
for t in sorted(sc.task.unique()):
    g = sc[sc.task == t].groupby(["arm", "k"])[["acc", "cov"]].mean(); P(f"\n{t}\n"); P("| arm | k=1 | k=3 | k=10 |"); P("|---|---|---|---|")
    for a in ARMS: P(f"| {a} | " + " | ".join(f"{g.loc[(a,k)]['acc']:.3f} / {g.loc[(a,k)]['cov']:.2f}" if (a, k) in g.index else "-" for k in KS) + " |")
P("\n### Certificate check (cells with a certified cutoff)\n"); P("| arm | certified cells (of scored) | realized error > 5% | mean realized error | max |"); P("|---|---|---|---|---|")
for a in ARMS:
    d = sc[(sc.arm == a) & sc.err.notna()]; n = (sc.arm == a).sum()
    P(f"| {a} | {len(d)} ({n}) | {int((d.err > .05).sum())} | {d.err.mean():.3f} | {d.err.max():.3f} |" if len(d) else f"| {a} | 0 ({n}) | - | - | - |")
P("\n### Hyper-parameters chosen (means)\n")
for a, c in (("1b Hopfield read-out", "beta"), ("1c Hopfield + LLM-written rows", "beta"), ("2b Laplace last layer (block-diag GGN, own code)", "alpha"), ("2c LR + temperature (fit on calibration set)", "T"), ("2a' GP classifier (fixed RBF, length-scale from calibration set)", "length_scale")):
    d = df[df.arm == a]; P(f"- {a}: {c} by k = " + ", ".join(f"k={k}: median {pd.to_numeric(d[d.k==k][c], errors='coerce').median():.3g}" for k in KS))
P("\n### Cost / latency at k=10 (mean of 15 cells)\n"); P("| arm | params | fit s | predict ms per 1000 rows |"); P("|---|---|---|---|")
for a in ARMS:
    d = df[(df.arm == a) & (df.k == 10)]; P(f"| {a} | {d.params.median():.0f} | {d.fit_s.mean():.3f} | {d.pred_ms_1000.mean():.1f} |")
s = df[df.arm.str.startswith("4 ")]
P("\n### Symbolic rules\n"); P("status counts: " + json.dumps(s.status.value_counts().to_dict())); P("status by k: " + json.dumps({int(k): v for k, v in s.groupby("k").status.apply(lambda x: x.value_counts().to_dict()).to_dict().items()}) if False else "")
P(s.groupby(["k", "status"]).size().unstack(fill_value=0).to_string())
so = s[s.status == "ok"]
P("\n| task | k | rule acc (abstain=wrong) | abstain share | acc when fired | rule cert. cov | LR acc | LR cov |"); P("|---|---|---|---|---|---|---|---|")
gb = df[df.arm == BASE].groupby(["task", "k"])[["acc", "cov"]].mean()
for (t, k), r in so.groupby(["task", "k"])[["acc", "abstain", "acc_fired", "cov"]].mean().iterrows(): P(f"| {t} | {k} | {r.acc:.3f} | {r.abstain:.2f} | {r.acc_fired:.3f} | {r['cov']:.2f} | {gb.loc[(t,k)].acc:.3f} | {gb.loc[(t,k)]['cov']:.2f} |")
P(f"\nrule acc over ok cells by k: " + ", ".join(f"k={k}: {so[so.k==k].acc.mean():.3f}" for k in KS) + f"; overall mean {so.acc.mean():.3f}, range {so.acc.min():.3f}-{so.acc.max():.3f}; if the {int((s.status!='ok').sum())} syntax-error cells count as accuracy 0: {(so.acc.sum())/len(s):.3f}")
P(f"max rule certified coverage {so['cov'].max():.3f}; cells with coverage > 0: {int((so['cov']>0).sum())} of {len(so)}; mean acc when fired {so.acc_fired.mean():.3f}; mean abstain {so.abstain.mean():.2f}; code chars {so.code_chars.mean():.0f}; LLM cost of the 45 symbolic cells ${s.llm_cost.sum():.4f}")
P("\n### Meta\n"); P(json.dumps({k: R['meta'][k] for k in ('gpu', 'started', 'prep', 'tasks', 'prep_s')}))
