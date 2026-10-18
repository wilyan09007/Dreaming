"""Recompute every S23 table from salvaged_results.json (partial: the pod was cut off). Usage: python3 report_salvaged.py
Means over the rows present (tasks x draws). 'cov' = realized certified coverage on the test rows at a 5% budget (common.certify, stored per cell);
raw confidences are not in the JSON, so certification itself is not re-run here, only aggregated."""
import json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
d = json.load(open(HERE / "salvaged_results.json")); df = pd.DataFrame(d["rows"])
df["lr"] = df["lr"].fillna(5e-4); df["pool"] = df["pool"].fillna("-")
BASE = "LR baseline (k random rows)"
def fm(x, p=2): return "-" if pd.isna(x) else f"{x:.{p}f}"
print(f"n_rows in JSON header = {d['n_rows']}, rows loaded = {len(df)}, rows with error field = {df['error'].notna().sum() if 'error' in df else 0}\n")
print("status:", json.dumps(d["status"]), "\n")

def table(title, sel, ks=(1, 3, 10), pool=None, lr=5e-4):
    print(f"### {title}\n")
    print("| arm | method | pool | " + " | ".join(f"k={k} acc | k={k} cov | k={k} n" for k in ks) + " |")
    print("|---|---|---|" + "---|" * (3 * len(ks)))
    for (a, m, p), g in df[sel & (df.lr == lr)].groupby(["arm", "method", "pool"]):
        c = []
        for k in ks:
            r = g[g.k == k]; c += [fm(r.acc.mean()), fm(r["cov"].mean()), str(len(r))] if len(r) else ["-", "-", "0"]
        print(f"| {a} | {m} | {p} | " + " | ".join(c) + " |")
    print()

table("Arm 1 and 2 (CPU): baseline, medoids, semi-supervised probes", df.arm.isin(["1", "2"]) & (df.k > 0) | (df.k == -1) & (df.arm == "1"))
table("Arm 4: TSDAE adaptation", df.arm == "4")
table("Arm 3: PAW (all rows present, any task/draw)", df.arm == "3", ks=(0, 1, 3, 10))
print("### Ceiling (LR on all pool labels, k=-1)\n"); c = df[df.k == -1]; print(f"acc {c.acc.mean():.3f} cov {c['cov'].mean():.3f} n={len(c)}\n")

# PAW: matched comparison, restricted to the exact (task,k,draw) cells that have a PAW fine-tune row
print("### PAW cells present (fine-tune rows) and matched comparisons\n")
ft = df[(df.method == "PAW compiled init + fine-tune") & (df.lr == 5e-4)]
cells = ft[["task", "k", "draw"]].drop_duplicates()
print("fine-tune cells present:", [tuple(x) for x in cells.values.tolist()], f"({len(cells)} of 45 planned = 5 tasks x 3 draws x 3 k)\n")
def matched(method, lr=5e-4):
    return df[(df.method == method) & (df.lr == lr)].merge(cells, on=["task", "k", "draw"])
print("| method | task | k | draw | acc | cov |\n|---|---|---|---|---|---|")
for m in ["PAW compiled init + fine-tune", "random LoRA init + fine-tune (same prompt)", "PAW compiled adapter, zero-shot (spec with k examples)", "no adapter, same prompt (spec with k examples)", BASE, "TSDAE 1000 steps + frozen LR"]:
    for _, r in matched(m).sort_values(["task", "k"]).iterrows():
        print(f"| {m} | {r.task} | {r.k} | {r.draw} | {fm(r.acc)} | {fm(r['cov'])} |")
print("\n| matched cells (same task,k,draw as the PAW fine-tune cells) | k | n | acc | cov |\n|---|---|---|---|---|")
for m in ["PAW compiled init + fine-tune", "random LoRA init + fine-tune (same prompt)", "PAW compiled adapter, zero-shot (spec with k examples)", "no adapter, same prompt (spec with k examples)", BASE, "LR on k*K k-means medoid rows", "TSDAE 1000 steps + frozen LR"]:
    g = matched(m)
    for k in (1, 3, 10):
        r = g[g.k == k]
        if len(r): print(f"| {m} | {k} | {len(r)} | {fm(r.acc.mean())} | {fm(r['cov'].mean())} |")
print()
print("### PAW zero-shot k=0 (per task)\n\n| method | task | acc | valid |\n|---|---|---|---|")
for _, r in df[df.k == 0].sort_values(["method", "task"]).iterrows(): print(f"| {r.method} | {r.task} | {fm(r.acc)} | {fm(r.valid) if 'valid' in r else '-'} |")
print("\n### lr sensitivity rows (k=10, draw 0)\n\n| method | lr | task | acc | cov |\n|---|---|---|---|---|")
for _, r in df[(df.lr != 5e-4)].sort_values(["method", "lr", "task"]).iterrows(): print(f"| {r.method} | {r.lr} | {r.task} | {fm(r.acc)} | {fm(r['cov'])} |")

# paired deltas vs baseline (same task,k,draw), all cells
print("\n### Paired change vs LR baseline (same task,k,draw), mean +- s.e. of the cell differences\n\n| method | pool | k | n | d-acc | d-cov |\n|---|---|---|---|---|---|")
b = df[df.method == BASE].set_index(["task", "k", "draw"])
for (m, p), g in df[(df.arm.isin(["2", "4"])) | (df.method == "LR on k*K k-means medoid rows")].groupby(["method", "pool"]):
    for k in (1, 3, 10):
        r = g[g.k == k]; da = [x.acc - b.loc[(x.task, k, x.draw)].acc for x in r.itertuples()]; dc = [x.cov - b.loc[(x.task, k, x.draw)]["cov"] for x in r.itertuples()]
        se = lambda v: np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
        print(f"| {m} | {p} | {k} | {len(r)} | {np.mean(da):+.3f} +- {se(da):.3f} | {np.mean(dc):+.3f} +- {se(dc):.3f} |")

# error on answered rows and per-task accuracy
print("\n### Realized error on answered rows (max over cells with coverage>0), per method\n\n| method | pool | max err | mean err |\n|---|---|---|---|")
for (m, p), g in df[df.k > 0].groupby(["method", "pool"]):
    e = g.err.dropna(); print(f"| {m} | {p} | {fm(e.max(),3)} | {fm(e.mean(),3)} |")
print("\n### Per task accuracy / coverage at k=1,3,10 (mean over draws): baseline, best probe, TSDAE\n")
for m, p in [(BASE, "-"), ("TK-KNN-style self-training", "full"), ("Sinkhorn pseudo-labels + LR (uniform prior)", "full"), ("LabelSpreading -> LR on reached rows", "full"), ("TSDAE 1000 steps + frozen LR", "-")]:
    g = df[(df.method == m) & (df.pool == p)]
    for t, gt in g.groupby("task"):
        print(f"{m}[{p}] {t}: " + "  ".join(f"k={k}: {gt[gt.k==k].acc.mean():.2f}/{gt[gt.k==k]['cov'].mean():.2f}" for k in (1, 3, 10)))
print("\n### Pseudo-label accuracy per round (mean over tasks, draws)\n")
for (m, p, k), g in df[df.pl.apply(lambda x: isinstance(x, list))].groupby(["method", "pool", "k"]):
    L = min(len(x) for x in g.pl); print(f"{m} [{p}] k={k}: " + ", ".join(f"{np.nanmean([np.nan if x[i] is None else x[i] for x in g.pl]):.3f}" for i in range(L)))
