"""Recompute every S22 table from salvaged_results.json (partial run). Usage: python3 report_salvaged.py [path]
Prints markdown. Arms: F frozen bge-small+LR, A LoRA from base, B LoRA from shared pool LoRA, D_comp composition of per-domain
pool LoRAs (head + mixing weights only), D_comp_ft composition then +100 LoRA steps. Means over all available (task, seed) cells."""
import json, sys, collections
from pathlib import Path
import numpy as np
path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "salvaged_results.json"
R = json.load(open(path)); recs = R["records"]; calib = R.get("calib") or {}
ARMS = ["F", "A", "B", "D_comp", "D_comp_ft"]; KS = [1, 3, 10]; ALLT = ["travel", "work", "bank0", "bank1", "bank2"]
NAME = {"F": "F frozen bge-small + LR", "A": "A LoRA from base", "B": "B LoRA from shared pool", "D_comp": "D composition (head + mixing only)", "D_comp_ft": "D+ composition + 100 LoRA steps"}
FAMS = {"CLINC (travel, work)": ("clinc",), "Banking77 (bank0-2)": ("bank",), "All five tasks": ("clinc", "bank")}
def get(fams, arm, k): return {(r["task"], r["seed"]): r for r in recs if r["fam"] in fams and r["arm"] == arm and r["k"] == k}
def mean(v): return 100 * float(np.mean(v)) if v else None
def f1(x): return "-" if x is None else f"{x:.1f}"
print(f"<!-- {path.name}: {len(recs)} records, {len(R.get('complete_cells', []))} complete cells (all 5 arms), errors {len(R.get('errors', []))}, wall {R.get('wall_s')} s, final={R.get('final')} -->")
print("\n### Cells present (task x seed; 15 per k planned, 5 tasks x 3 seeds), records per arm\n\n| k | " + " | ".join(NAME[a].split(" ")[0] for a in ARMS) + " | cells with all 5 arms |\n|---|---|---|---|---|---|---|")
cc = collections.Counter(tuple(c)[2] for c in R["complete_cells"])
for k in KS: print(f"| {k} | " + " | ".join(str(len(get(('clinc', 'bank'), a, k))) for a in ARMS) + f" | {cc.get(k, 0)} |")
print(f"\nTotal records {len(recs)} of 225 planned (45 cells x 5 arms); complete cells {len(R['complete_cells'])} of 45.")
miss = collections.defaultdict(list)
for k in (10,):
    for a in ARMS:
        have = set(get(('clinc', 'bank'), a, k))
        miss[a] = sorted(f"{t}/s{s}" for t in ALLT for s in (0, 1, 2) if (t, s) not in have)
print("\nMissing k=10 cells per arm:\n")
for a in ARMS: print(f"- {a} ({len(miss[a])} missing): " + ", ".join(miss[a]))
def table(title, key, fams, fn=lambda v: mean(v), matched=False):
    print(f"\n**{title}**\n\n| arm | " + " | ".join(f"k={k}" for k in KS) + " |\n|---|---|---|---|")
    for a in ARMS:
        cols = []
        for k in KS:
            g = get(fams, a, k)
            if matched:
                keep = {tuple(c[:2]) for c in R["complete_cells"] if c[2] == k and (c[0] in {r['task'] for r in g.values()})}
                g = {c: r for c, r in g.items() if c in keep}
            v = [r[key] for r in g.values() if r.get(key) is not None]
            m = fn(v); s = 100 * np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 and key in ("acc", "cov") else None
            cols.append("-" if m is None else f1(m) + (f" ± {s:.1f}" if s is not None else "") + f" (n={len(v)})")
        print(f"| {NAME[a]} | " + " | ".join(cols) + " |")
for fn_, fams in FAMS.items():
    table(f"Accuracy (%), mean ± s.e. over available cells, {fn_}", "acc", fams)
    table(f"Certified coverage at 5% budget (%), 0 when nothing certified, {fn_}", "cov", fams)
table("Accuracy (%), only the 6 k=10 cells where all five arms finished (k=1,3 unchanged: all cells), all tasks", "acc", ("clinc", "bank"), matched=True)
table("Certified coverage (%), same matched cells, all tasks", "cov", ("clinc", "bank"), matched=True)
table("Share of cells where the 5% certificate succeeded at all (%)", "certified", ("clinc", "bank"), fn=lambda v: mean([1.0 if x else 0.0 for x in v]))
print("\n**Realized error (%) among answered test rows, cells that certified (budget 5)**\n\n| arm | " + " | ".join(f"k={k}" for k in KS) + " |\n|---|---|---|---|")
for a in ARMS:
    cols = []
    for k in KS:
        v = [r["err"] for r in get(("clinc", "bank"), a, k).values() if r.get("err") is not None]
        cols.append(f"mean {100*np.mean(v):.1f}, max {100*np.max(v):.1f} (n={len(v)})" if v else "-")
    print(f"| {NAME[a]} | " + " | ".join(cols) + " |")
print("\n### Paired differences in accuracy / coverage points (same task, seed, k), all five tasks: mean (s.e.); wins/ties/losses\n")
PAIRS = [("B", "F"), ("A", "F"), ("D_comp", "F"), ("D_comp_ft", "F"), ("B", "A"), ("D_comp", "A"), ("D_comp", "B"), ("D_comp_ft", "B"), ("D_comp_ft", "D_comp")]
for key, tl in (("acc", "accuracy"), ("cov", "certified coverage")):
    print(f"\n**{tl}**\n\n| pair | " + " | ".join(f"k={k}" for k in KS) + " |\n|---|---|---|---|")
    for x, y in PAIRS:
        cols = []
        for k in KS:
            gx, gy = get(("clinc", "bank"), x, k), get(("clinc", "bank"), y, k); d = [100 * (gx[c][key] - gy[c][key]) for c in gx if c in gy]
            if not d: cols.append("-"); continue
            se = np.std(d, ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
            cols.append(f"{np.mean(d):+.1f} ({se:.1f}); {sum(v > .05 for v in d)}/{sum(abs(v) <= .05 for v in d)}/{sum(v < -.05 for v in d)} of {len(d)}")
        print(f"| {x} - {y} | " + " | ".join(cols) + " |")
print("\n### Per-task accuracy (%) / coverage (%), mean over available seeds (n seeds)\n\n| task | arm | " + " | ".join(f"k={k}" for k in KS) + " |\n|---|---|---|---|---|")
for t in ALLT:
    for a in ARMS:
        cols = []
        for k in KS:
            g = [r for r in recs if r["task"] == t and r["arm"] == a and r["k"] == k]
            cols.append(f"{100*np.mean([r['acc'] for r in g]):.1f} / {100*np.mean([r['cov'] for r in g]):.1f} (n={len(g)})" if g else "-")
        print(f"| {t} | {a} | " + " | ".join(cols) + " |")
rl, rc = calib.get("lora_steps_per_s"), calib.get("comp_steps_per_s")
solo = {"A": lambda s: s / rl, "B": lambda s: s / rl, "D_comp": lambda s: s / rc, "D_comp_ft": lambda s: 300 / rc + 100 / rl}
print(f"\n### Trainable parameters and seconds per run ({calib.get('gpu')}, solo calibration: LoRA {rl} steps/s, composition {rc} steps/s; 3 workers shared the GPU in the real run)\n")
print("| arm | params CLINC (15 labels) | params Banking (12 labels) | steps | measured s (mean, median, n) | solo-equivalent s |\n|---|---|---|---|---|---|")
for a in ARMS:
    pc = sorted({r["params"] for r in recs if r["arm"] == a and r["fam"] == "clinc"}); pb = sorted({r["params"] for r in recs if r["arm"] == a and r["fam"] == "bank"})
    ts = [r["train_s"] for r in recs if r["arm"] == a]; st = sorted({r["steps"] for r in recs if r["arm"] == a})
    sol = "<0.1" if a == "F" else f"{solo[a](300):.1f}" if a != "D_comp_ft" else f"{solo[a](0):.1f}"
    print(f"| {NAME[a]} | {pc[0]:,} | {pb[0]:,} | {st[0]} | {np.mean(ts):.1f}, {np.median(ts):.1f}, {len(ts)} | {sol} |")
print("\n### Pool adapters (built once per pod)\n\n| job | rows | labels | steps | measured s | solo-equivalent s | accuracy on pool sample |\n|---|---|---|---|---|---|---|")
tot = collections.defaultdict(lambda: [0.0, 0.0])
for nm, p in sorted(R["pools"].items()):
    se = p["steps"] / rl; tot[p["fam"]][0] += p["secs"]; tot[p["fam"]][1] += se
    print(f"| {nm} | {p['rows']} | {p['labels']} | {p['steps']} | {p['secs']} | {se:.0f} | {p['pool_acc']} |")
print("\n" + "; ".join(f"{f}: {v[0]:.0f} s measured, {v[1]:.0f} s solo-equivalent" for f, v in tot.items()))
byfam = collections.defaultdict(list)
for r in recs:
    if r["arm"] == "D_comp" and r.get("mix"): byfam[r["fam"]].append(r["mix"])
print("\n### Learned mixing weights (D)\n\n| family | adapters | mean largest weight | mean effective number of adapters | uniform = 1/K |\n|---|---|---|---|---|")
for f, m in byfam.items():
    K = len(m[0]); print(f"| {f} | {K} | {np.mean([max(x) for x in m]):.2f} | {np.mean([np.exp(-sum(w*np.log(w+1e-12) for w in x)) for x in m]):.1f} | {1/K:.2f} |")
print(f"\nWall time at salvage: {R.get('wall_s')} s; final flag {R.get('final')}; last record at {max(r['at_s'] for r in recs)} s")
