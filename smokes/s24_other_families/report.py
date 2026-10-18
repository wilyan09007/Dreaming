"""S24 report: results.json -> tables.md. Usage (laptop, cheap): python -I report.py"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
R = json.load(open(HERE / (sys.argv[1] if len(sys.argv) > 1 else "results.json"))); meta = R["meta"]
df = pd.DataFrame(R["records"]); KS = [1, 3, 10]; out = []
def P(s=""): out.append(s)
for c in ("acc", "cov", "err", "auroc", "cal_cov", "fit_s", "pred_ms_1000", "params", "abstain", "acc_fired"):
    df[c] = pd.to_numeric(df.get(c), errors="coerce")
bad = df[df.acc.isna()]; df = df[df.acc.notna()].copy()
ARMS = sorted(df.arm.unique()); BASE = [a for a in ARMS if a.startswith("0 ")][0]
FAM = {"clinc": df[df.family == "clinc"].task.nunique(), "banking": df[df.family == "banking"].task.nunique()}

def macro(d, col):  # mean over tasks within a draw, then mean / std over draws
    per = d.groupby(["arm", "k", "seed"])[col].mean().reset_index(); return per.groupby(["arm", "k"])[col].agg(["mean", "std"])
def table(d, col, title, fmt="{:.3f}", std=True):
    m = macro(d, col); P(f"\n#### {title}\n"); P("| arm | " + " | ".join(f"k={k}" for k in KS) + " |"); P("|---|" + "---|" * len(KS))
    for a in ARMS:
        cells = []
        for k in KS:
            try: r = m.loc[(a, k)]; cells.append(fmt.format(r["mean"]) + (f" ± {r['std']:.3f}" if std and not np.isnan(r["std"]) else ""))
            except KeyError: cells.append("-")
        P(f"| {a} | " + " | ".join(cells) + " |")

P(f"Records: {len(df)} scored cells ({len(bad)} cells without a score, e.g. LLM/rule failures). Tasks: {sorted(df.task.unique())}; k in {KS}; draws: {df.seed.nunique()}.")
P("\n### Accuracy and certified coverage (macro over the 5 tasks, mean ± std over the 3 draws)")
table(df, "acc", "Accuracy on the test rows"); table(df, "cov", "Certified coverage at a 5% error budget (share of test rows answered, cutoff from the 300 calibration rows)")
e = df[df.err.notna()]; table(e, "err", "Realized error on the answered test rows (cells with a certified cutoff only)"); table(df, "auroc", "AUROC of the confidence score for 'prediction is correct' (test rows)", std=False)

# ---- rows needed for 50% certified coverage
def need50(c):
    pts = [(k, c[k]) for k in KS if k in c and not np.isnan(c[k])]
    if not pts: return "-"
    if pts[0][1] >= 0.5: return "<=1"
    for (k0, c0), (k1, c1) in zip(pts, pts[1:]):
        if c1 >= 0.5: return f"{np.exp(np.log(k0) + (0.5 - c0) / (c1 - c0) * (np.log(k1) - np.log(k0))):.1f}"
    return ">10"
P("\n### Rows per label needed to reach 50% certified coverage at a 5% budget\n")
P("Linear interpolation in log k between the measured k = 1, 3, 10 of the mean certified coverage ('<=1' = already above 50% at 1 row per label; '>10' = not reached at 10). Total rows = rows per label x labels (12 or 15).\n")
P("| arm | all 5 tasks | CLINC tasks (15 labels) | Banking77 tasks (12 labels) | coverage at k=1 / 3 / 10 (all tasks) |"); P("|---|---|---|---|---|")
for a in ARMS:
    cells = []
    for sel in (df, df[df.family == "clinc"], df[df.family == "banking"]):
        m = macro(sel[sel.arm == a], "cov")["mean"] if (sel.arm == a).any() else None
        cells.append(need50({k: m.loc[(a, k)] for k in KS if (a, k) in m.index}) if m is not None else "-")
    m = macro(df[df.arm == a], "cov")["mean"]; P(f"| {a} | " + " | ".join(cells) + " | " + " / ".join(f"{m.loc[(a, k)]:.2f}" if (a, k) in m.index else "-" for k in KS) + " |")

# ---- per task
P("\n### Per task: accuracy / certified coverage (mean over the 3 draws)")
for t in sorted(df.task.unique()):
    d = df[df.task == t]; g = d.groupby(["arm", "k"])[["acc", "cov"]].mean(); P(f"\n#### {t}\n"); P("| arm | " + " | ".join(f"k={k}" for k in KS) + " |"); P("|---|" + "---|" * len(KS))
    for a in ARMS: P(f"| {a} | " + " | ".join(f"{g.loc[(a, k)]['acc']:.3f} / {g.loc[(a, k)]['cov']:.2f}" if (a, k) in g.index else "-" for k in KS) + " |")

# ---- paired difference to the baseline (same task, k, draw)
P("\n### Paired difference to the baseline LR (same task, same k, same draw; mean ± standard error over the cells)\n")
P("| arm | " + " | ".join(f"d-coverage k={k}" for k in KS) + " | " + " | ".join(f"d-accuracy k={k}" for k in KS) + " |"); P("|---|" + "---|" * (2 * len(KS)))
key = ["task", "k", "seed"]; b = df[df.arm == BASE].set_index(key)
for a in ARMS:
    if a == BASE: continue
    x = df[df.arm == a].set_index(key); j = x.join(b, rsuffix="_b", how="inner"); cells = []
    for col in ("cov", "acc"):
        for k in KS:
            d = (j[j.index.get_level_values("k") == k][col] - j[j.index.get_level_values("k") == k][col + "_b"]).dropna()
            cells.append(f"{d.mean():+.3f} ± {d.std(ddof=1) / np.sqrt(len(d)):.3f} (n={len(d)})" if len(d) > 1 else "-")
    P(f"| {a} | " + " | ".join(cells) + " |")

# ---- cost / latency / parameters
P("\n### Parameters trained, fit time and predict latency (k = 10, mean over the 15 cells)\n")
P("| arm | parameters trained per task | fit seconds | predict ms per 1000 rows | note |"); P("|---|---|---|---|---|")
notes = {"1": "no training; stores the k rows (+10 LLM-written rows per label for 1c)", "2a": "kernel hyper-parameters only (2 per one-vs-rest problem); stores the rows", "3": f"per-task params 0; meta-trained once: {meta.get('pfn', {}).get('params', '?')} params, {meta.get('pfn', {}).get('secs', '?')} s",
         "4": "fit = LLM call seconds; 0 params; predict = regex execution in a subprocess", "0": "", "2b": "", "2c": ""}
for a in ARMS:
    d = df[(df.arm == a) & (df.k == 10)]
    if d.empty: continue
    note = next((v for kk, v in notes.items() if a.startswith(kk)), ""); P(f"| {a} | {d.params.median():.0f} | {d.fit_s.mean():.3f} | {d.pred_ms_1000.mean():.1f} | {note} |")

# ---- symbolic student
s = df[df.arm.str.startswith("4 ")]
if len(s):
    P("\n### Symbolic student (LLM-written regex rules) against the baseline LR, per task and k (mean over draws)\n")
    P("| task | k | rule acc (abstain = wrong) | abstains | rule acc when fired | rule cert. cov | LR acc | LR cert. cov | rule beats LR on acc? |"); P("|---|---|---|---|---|---|---|---|---|")
    g = s.groupby(["task", "k"])[["acc", "abstain", "acc_fired", "cov"]].mean(); gb = df[df.arm == BASE].groupby(["task", "k"])[["acc", "cov"]].mean()
    for (t, k), r in g.iterrows():
        lb = gb.loc[(t, k)]; P(f"| {t} | {k} | {r.acc:.3f} | {r.abstain:.2f} | {r.acc_fired:.3f} | {r['cov']:.2f} | {lb.acc:.3f} | {lb['cov']:.2f} | {'yes' if r.acc > lb.acc else 'no'} |")
    P(f"\nRule programs: mean {s.code_chars.mean():.0f} characters, {s.n_regex.mean():.1f} regex calls; per-text runtime errors/timeouts total {int(s.text_errs.sum())}; LLM cost over scored cells ${s.llm_cost.sum():.4f}.")
    allsym = pd.DataFrame([r for r in R["records"] if str(r.get("arm", "")).startswith("4 ")]); P("Cell status counts: " + json.dumps(allsym.status.value_counts().to_dict()))

# ---- certificate validity
P("\n### Certificate check: realized error of the answered test rows against the 5% budget\n")
c = df[df.err.notna()].copy(); c["viol"] = c.err > 0.05
P("| arm | cells with a certified cutoff (of all) | realized error > 5% | mean realized error | max |"); P("|---|---|---|---|---|")
for a in ARMS:
    d = c[c.arm == a]; n = (df.arm == a).sum()
    if len(d): P(f"| {a} | {len(d)} ({n}) | {int(d.viol.sum())} ({d.viol.mean():.0%}) | {d.err.mean():.3f} | {d.err.max():.3f} |")
    else: P(f"| {a} | 0 ({n}) | - | - | - |")

# ---- meta
pf = meta.get("pfn")
if pf:
    P("\n### Prior-fitted network training\n"); P(f"{pf['steps']} steps in {pf['secs']} s on {pf.get('device')} (batch {pf.get('B')} tasks), {pf['params']:,} parameters; best validation mean accuracy {pf['best_val']:.3f} at step {pf['best_step']} (validation = held-out CLINC 'meta' domain, k=1/3/10; LR on the same episodes: {pf['val_lr_reference']}).")
    P(f"Batches by source: {pf['batches_by_source']}. Synthetic tasks (K=10, k=3): LR accuracy quantiles 10/25/50/75/90% = {pf.get('synth_lr_acc_quantiles')}.\n")
    P("| step | seconds | train loss (ema) | val acc k=1 | k=3 | k=10 |"); P("|---|---|---|---|---|---|")
    hist = pf["hist"]; keep = hist[:: max(1, len(hist) // 12)] + ([hist[-1]] if hist else [])
    for h in keep: P(f"| {h['step']} | {h['sec']} | {h['loss']} | {h['val']['1']:.3f} | {h['val']['3']:.3f} | {h['val']['10']:.3f} |" if "1" in h["val"] else f"| {h['step']} | {h['sec']} | {h['loss']} | {h['val'][1]:.3f} | {h['val'][3]:.3f} | {h['val'][10]:.3f} |")
P(f"\nlaplace-torch: {meta.get('laplace_torch')}. Failures: {json.dumps(meta.get('failures'))}. LLM spend (all cached calls in this folder): ${meta.get('llm_spend_usd')}. GPU: {meta.get('gpu')}; wall {meta.get('wall_s')} s; prep {meta.get('prep_s')} s.")
P(f"Tasks: {json.dumps(meta.get('tasks'))}\nPool: {json.dumps(meta.get('prep'))}")
(HERE / "tables.md").write_text("\n".join(out), encoding="utf-8"); print("\n".join(out))
