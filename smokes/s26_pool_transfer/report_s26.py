"""Recompute every S26 table from results.json (read only). Usage: python report_s26.py > report_s26_out.md"""
import json, numpy as np, pandas as pd
from pathlib import Path
H = Path(__file__).resolve().parent
df = pd.DataFrame(json.loads((H / "results.json").read_text()))
SHORT = {"1 baseline (frozen bge-small + LR)": "1 baseline", "2 synthetic prior (LLM examples + k rows)": "2 synthetic prior", "3 hypernetwork (pool-trained, one forward pass)": "3 hypernetwork",
         "4 encoder pretrained on pool (33M) + LR": "4 encoder pretrained", "4b encoder pretrained + synthetic prior": "4b encoder pretrained + synth",
         "5 shared LoRA (pool-pretrained) + k rows": "5 shared LoRA", "5c LoRA from base + k rows": "5c LoRA from base", "5b shared LoRA + synthetic prior": "5b shared LoRA + synth", "5d LoRA from base + synthetic prior": "5d LoRA from base + synth"}
ORDER = list(SHORT.values()); df["arm"] = df.arm.map(SHORT); TASKS = ["bank_us", "amazon", "clinc_travel"]; KS = [1, 3, 10]
print(f"records {len(df)}; arms {df.arm.nunique()}; tasks {df.task.nunique()}; k {sorted(df.k.unique())}; draws {sorted(df.seed.unique())}\n")
def per_draw(a, k, key, tasks=TASKS):   # task-average for each draw
    s = df[(df.arm == a) & (df.k == k) & df.task.isin(tasks)].groupby("seed")[key].mean(); return s.mean(), (s.std(ddof=1) if len(s) > 1 else np.nan)
def table(key, tasks, title):
    print(f"### {title}\n"); print("| arm | " + " | ".join(f"{k=}".replace("k=", "k=") if False else f"k={k}" for k in KS) + " |"); print("|---|---|---|---|")
    for a in ORDER: print(f"| {a} | " + " | ".join("%.3f +- %.3f" % per_draw(a, k, key, tasks) for k in KS) + " |")
    print()
for key, nm in (("acc", "Accuracy"), ("cov", "Certified coverage at 5%")):
    table(key, TASKS, f"{nm}, mean over the 3 tasks (+- s.d. over 3 draws of the task-averaged value)")
    table(key, ["clinc_travel"], f"{nm}, clinc_travel only (+- s.d. over draws)")
    table(key, ["bank_us", "amazon"], f"{nm}, mean of bank_us and amazon (+- s.d. over draws)") if key == "acc" else None
for t in TASKS:
    print(f"### Per task {t}: accuracy / coverage / realized error on answered rows (n certified of 3 draws), mean over draws\n"); print("| arm | " + " | ".join(f"k={k}" for k in KS) + " |"); print("|---|---|---|---|")
    for a in ORDER:
        c = []
        for k in KS:
            s = df[(df.arm == a) & (df.task == t) & (df.k == k)]; e = s.err.dropna(); c.append(f"{s.acc.mean():.3f} / {s["cov"].mean():.3f} / " + (f"{e.mean():.3f} ({len(e)})" if len(e) else "- (0)"))
        print(f"| {a} | " + " | ".join(c) + " |")
    print()
print("### Share of cells (task x k x draw, 27 per arm) where the 5% certificate succeeded at all, and worst realized error among certified cells\n"); print("| arm | certified cells | of which on clinc_travel | max realized error |"); print("|---|---|---|---|")
for a in ORDER:
    s = df[df.arm == a]; c = s[s["cov"] > 0]; print(f"| {a} | {len(c)} / {len(s)} | {(c.task == 'clinc_travel').sum()} | {c.err.max():.3f} |" if len(c) else f"| {a} | 0 / {len(s)} | 0 | - |")
print()
byk = df.set_index(["arm", "task", "k", "seed"])
def paired(A, B, tasks, key, k):
    d = [byk.loc[(A, t, k, s), key] - byk.loc[(B, t, k, s), key] for t in tasks for s in range(3)]; w = sum(x > 0 for x in d); l = sum(x < 0 for x in d); return np.mean(d), np.std(d, ddof=1), w, l, len(d)
PAIRS = [("4 encoder pretrained", "1 baseline"), ("4b encoder pretrained + synth", "2 synthetic prior"), ("5 shared LoRA", "5c LoRA from base"), ("5b shared LoRA + synth", "5d LoRA from base + synth"),
         ("2 synthetic prior", "1 baseline"), ("3 hypernetwork", "1 baseline"), ("3 hypernetwork", "2 synthetic prior"), ("4 encoder pretrained", "5 shared LoRA"), ("4b encoder pretrained + synth", "5b shared LoRA + synth"),
         ("1 baseline", "5c LoRA from base + k rows".replace(" + k rows", "")), ("2 synthetic prior", "5d LoRA from base + synth")]
for key in ("acc", "cov"):
    for tasks, tn in ((TASKS, "all 3 tasks (9 cells per k)"), (["bank_us", "amazon"], "bank_us + amazon (6 cells per k)"), (["clinc_travel"], "clinc_travel (3 cells per k)")):
        if key == "cov" and tn.startswith("bank"): continue
        print(f"### Paired differences in {'accuracy' if key=='acc' else 'certified coverage'} (A - B, same task / k / draw), {tn}: mean +- s.d. over cells [A better / A worse]\n"); print("| A - B | k=1 | k=3 | k=10 |"); print("|---|---|---|---|")
        for A, B in PAIRS:
            print(f"| {A} - {B} | " + " | ".join("%+.3f +- %.3f [%d/%d]" % paired(A, B, tasks, key, k)[:4] for k in KS) + " |")
        print()
print("### Per-draw transfer gain in accuracy, decoder (5 - 5c), by task and k, mean over draws\n"); print("| task | k=1 | k=3 | k=10 |"); print("|---|---|---|---|")
for t in TASKS: print(f"| {t} | " + " | ".join("%+.3f" % paired("5 shared LoRA", "5c LoRA from base", [t], "acc", k)[0] for k in KS) + " |")
print("\n### Encoder transfer (4 - 1) accuracy by task and k, mean over draws\n"); print("| task | k=1 | k=3 | k=10 |"); print("|---|---|---|---|")
for t in TASKS: print(f"| {t} | " + " | ".join("%+.3f" % paired("4 encoder pretrained", "1 baseline", [t], "acc", k)[0] for k in KS) + " |")
print("\n### Seconds per run (mean)\n"); print("| arm | sec per cell |"); print("|---|---|")
for a in ORDER: print(f"| {a} | {df[df.arm == a].sec.mean():.2f} |")
print("\n### Certificate arithmetic (Clopper-Pearson one-sided, delta 0.05 / 40 grid levels, budget 5%): fewest answered calibration rows needed\n")
from scipy.stats import beta
def cpu(e, n, d): return 1.0 if e >= n else float(beta.ppf(1 - d, e + 1, n - e))
print("| errors among answered cal rows | min rows answered | share of the 300 | min accuracy of the answered rows |"); print("|---|---|---|---|")
for e in range(0, 7):
    n = e + 1
    while cpu(e, n, 0.05 / 40) > 0.05: n += 1
    print(f"| {e} | {n} | {n/300:.2f} | {1-e/n:.3f} |")
print("\n(The table above ignores the grid: it is the fewest rows at the single level delta/40. `common.certify` only tests levels round(n*j/40), so the real minimum is the first grid level that passes, below.)\n")
print("### Certificate arithmetic with the 40-level grid actually used by common.certify (first passing grid level, zero-padded to the top e errors)\n")
print("| errors | n_cal=300: level (rows) | share | n_cal=1000: level (rows) | share |"); print("|---|---|---|---|---|")
def grid_min(e, n_cal):
    for j in range(1, 41):
        k = max(1, int(round(n_cal * j / 40)))
        if cpu(e, k, 0.05 / 40) <= 0.05: return k
    return None
for e in range(0, 7):
    a, b = grid_min(e, 300), grid_min(e, 1000)
    print(f"| {e} | {a} | {a/300:.3f} | {b} | {b/1000:.3f} |" if a and b else f"| {e} | {a} | - | {b} | - |")
print("\n### Overall accuracy versus certification (what decides it is the top-2-gap ranking, not overall accuracy)\n")
cc = df[df["cov"] > 0]; nc = df[df["cov"] == 0]
print(f"- lowest overall accuracy among the {len(cc)} certified cells: {cc.acc.min():.3f}; highest overall accuracy among the {len(nc)} uncertified cells: {nc.acc.max():.3f}")
print(f"- highest accuracy on bank_us / amazon (any cell): {df[df.task != 'clinc_travel'].acc.max():.3f}; certified cells with accuracy below 0.90: {(cc.acc < 0.90).sum()}")
g = df[(df.task == 'clinc_travel')].groupby(['arm', 'k']).agg(acc=('acc', 'mean'), cert=('cov', lambda x: int((x > 0).sum())), cov=('cov', 'mean')).reset_index()
print("\n| travel cell (arm, k) | mean accuracy | draws certified (of 3) | mean coverage |"); print("|---|---|---|---|")
for _, r in g[(g.acc < 0.90)].sort_values('acc').iterrows(): print(f"| {r.arm}, k={r.k} | {r.acc:.3f} | {r.cert} | {r['cov']:.3f} |")
x = df[(df.task == 'clinc_travel') & (df.acc >= 0.84) & (df["cov"] == 0)][['arm', 'k', 'seed', 'acc']]
print("\nTravel cells with accuracy >= 0.84 that did not certify:\n"); 
for _, r in x.iterrows(): print(f"- {r.arm}, k={r.k}, seed {r.seed}: accuracy {r.acc:.3f}")
print("\n### Compact per-task tables (mean over 3 draws)\n")
for key, nm in (("acc", "Accuracy"), ("cov", "Certified coverage at 5%")):
    print(f"{nm}\n"); print("| arm | " + " | ".join(f"{t} k={k}" for t in TASKS for k in KS) + " |"); print("|---|" + "---|" * 9)
    for a in ORDER: print(f"| {a} | " + " | ".join("%.3f" % df[(df.arm == a) & (df.task == t) & (df.k == k)][key].mean() for t in TASKS for k in KS) + " |")
    print()
