"""Print the markdown tables for RESULTS.md from results.json (S22-lite). Usage: python smokes/s22_architectures/report.py [results.json]"""
import json, sys, collections
from pathlib import Path
import numpy as np

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results.json"
R = json.load(open(path)); recs = R["records"]; calib = R.get("calib") or {}
ARMS = ["F", "A", "B", "D_comp", "D_comp_ft"]; KS = sorted({r["k"] for r in recs})
NAME = {"F": "F  frozen bge-small + LR (baseline)", "A": "A  LoRA from base", "B": "B  LoRA from shared pool LoRA", "D_comp": "D  composition only (head + mixing weights)", "D_comp_ft": "D+ composition, then +100 LoRA steps"}
FAMS = {"CLINC held-out (travel, work)": ("clinc",), "Banking77 12-label (bank0-2)": ("bank",), "All five tasks": ("clinc", "bank")}
def get(fams, arm, k): return {(r["task"], r["seed"]): r for r in recs if r["fam"] in fams and r["arm"] == arm and r["k"] == k}
def ms(v, pct=True):
    if not v: return "-"
    m, s = np.mean(v), (np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0); f = 100 if pct else 1
    return f"{f*m:.1f} ± {f*s:.1f}"
print(f"<!-- {path.name}: {len(recs)} records, complete cells {len(R.get('complete_cells', []))}, errors {len(R.get('errors', []))}, wall {R.get('wall_s')} s -->")
print("\n### Cells completed (task x seed per k; 5 tasks x 3 seeds = 15 possible)\n\n| k | " + " | ".join(FAMS) + " |\n|---|" + "---|" * len(FAMS))
for k in KS: print(f"| {k} | " + " | ".join(str(len({(r['task'], r['seed']) for r in recs if r['fam'] in f and r['k'] == k and r['arm'] == 'D_comp_ft'})) for f in FAMS.values()) + " |")
for title, key, pct in (("Accuracy (%) on the full test split, mean ± s.e. over (task, seed) cells", "acc", True),
                        ("Certified coverage (%) at a 5% error budget: fraction of the test split above the cutoff certified on the 300 calibration rows (0 when nothing certifies)", "cov", True),
                        ("Share of cells where the 5% certificate succeeded at all (%)", "certified", True),
                        ("Realized error (%) among test rows above the certified cutoff (budget 5), cells that certified", "err", True)):
    for fname, fams in FAMS.items():
        print(f"\n**{title} — {fname}**\n\n| arm | " + " | ".join(f"k={k}" for k in KS) + " |\n|---|" + "---|" * len(KS))
        for arm in ARMS:
            cols = []
            for k in KS:
                g = get(fams, arm, k); v = [float(r[key]) for r in g.values() if r.get(key) is not None]
                cols.append(ms(v, pct) + (f" (n={len(v)})" if v and key != "acc" and key != "cov" else ""))
            print(f"| {NAME[arm]} | " + " | ".join(cols) + " |")
print("\n### Paired accuracy difference in points (same task, seed and rows); mean (s.e.), wins/ties/losses over cells\n")
PAIRS = [("D_comp", "B"), ("D_comp_ft", "B"), ("D_comp_ft", "D_comp"), ("B", "A"), ("D_comp", "A"), ("B", "F"), ("D_comp_ft", "F"), ("A", "F")]
for key, tl in (("acc", "accuracy"), ("cov", "certified coverage")):
    print(f"\n**{tl}, all five tasks**\n\n| pair | " + " | ".join(f"k={k}" for k in KS) + " |\n|---|" + "---|" * len(KS))
    for x, y in PAIRS:
        cols = []
        for k in KS:
            gx, gy = get(("clinc", "bank"), x, k), get(("clinc", "bank"), y, k); d = [100 * (gx[c][key] - gy[c][key]) for c in gx if c in gy]
            if not d: cols.append("-"); continue
            se = np.std(d, ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0; eps = 0.05
            cols.append(f"{np.mean(d):+.1f} ({se:.1f}); {sum(v > eps for v in d)}/{sum(abs(v) <= eps for v in d)}/{sum(v < -eps for v in d)} of {len(d)}")
        print(f"| {NAME[x].split('  ')[0]} - {NAME[y].split('  ')[0]} | " + " | ".join(cols) + " |")
print("\n**D_comp (composition only) minus B, accuracy points, by family**\n\n| family | " + " | ".join(f"k={k}" for k in KS) + " |\n|---|" + "---|" * len(KS))
for fname, fams in list(FAMS.items())[:2]:
    for x in ("D_comp", "D_comp_ft"):
        cols = []
        for k in KS:
            gx, gy = get(fams, x, k), get(fams, "B", k); d = [100 * (gx[c]["acc"] - gy[c]["acc"]) for c in gx if c in gy]; cols.append(f"{np.mean(d):+.1f} (n={len(d)})" if d else "-")
        print(f"| {fname}, {x} - B | " + " | ".join(cols) + " |")
# per-task means (so a reader sees travel vs work vs each bank task)
print("\n### Per-task accuracy (%), mean over seeds\n\n| task | arm | " + " | ".join(f"k={k}" for k in KS) + " |\n|---|---|" + "---|" * len(KS))
for tn in sorted({r["task"] for r in recs}):
    for arm in ARMS:
        cols = []
        for k in KS:
            v = [r["acc"] for r in recs if r["task"] == tn and r["arm"] == arm and r["k"] == k]; cols.append(f"{100*np.mean(v):.1f} (n={len(v)})" if v else "-")
        print(f"| {tn} | {arm} | " + " | ".join(cols) + " |")
# params and seconds
rl, rc = calib.get("lora_steps_per_s"), calib.get("comp_steps_per_s")
solo = {"A": lambda s: s / rl, "B": lambda s: s / rl, "D_comp": lambda s: s / rc, "D_comp_ft": lambda s: 300 / rc + 100 / rl} if rl and rc else {}
print(f"\n### Trainable parameters and pod seconds per run (solo calibration on {calib.get('gpu')}: LoRA {rl} steps/s, composition {rc} steps/s; {calib.get('cpus')} cpus)\n")
print("| arm | trainable params CLINC (15 labels) | trainable params Banking (12 labels) | steps | measured s per run (workers shared the GPU) | solo-equivalent s per run |\n|---|---|---|---|---|---|")
for arm in ARMS:
    pc = sorted({r["params"] for r in recs if r["arm"] == arm and r["fam"] == "clinc"}); pb = sorted({r["params"] for r in recs if r["arm"] == arm and r["fam"] == "bank"})
    ts = [r["train_s"] for r in recs if r["arm"] == arm]; st = sorted({r["steps"] for r in recs if r["arm"] == arm})
    sol = f"{solo[arm](st[0]):.1f}" if arm in solo and st else ("<0.1" if arm == "F" else "-")
    print(f"| {NAME[arm]} | {pc[0] if pc else '-':,} | {pb[0] if pb else '-':,} | {st[0] if st else '-'} | {np.mean(ts):.1f} (median {np.median(ts):.1f}, n={len(ts)}) | {sol} |")
print("\n### Pool build (done once per pod, amortised over every new step)\n\n| job | rows | labels | steps | measured s | solo-equivalent s | train accuracy on 300 pool rows |\n|---|---|---|---|---|---|---|")
tot = collections.defaultdict(lambda: [0.0, 0.0])
for nm, pi in sorted(R.get("pools", {}).items()):
    se = pi["steps"] / rl if rl else float("nan"); tot[pi["fam"]][0] += pi["secs"]; tot[pi["fam"]][1] += se
    print(f"| {nm} | {pi['rows']} | {pi['labels']} | {pi['steps']} | {pi['secs']} | {se:.0f} | {pi['pool_acc']} |")
print("\n" + "; ".join(f"{f}: {v[0]:.0f} s measured, {v[1]:.0f} s solo-equivalent for the whole pool" for f, v in tot.items()))
# what the mixing weights look like
mix = [r["mix"] for r in recs if r["arm"] == "D_comp" and r.get("mix")]
if mix:
    byfam = collections.defaultdict(list)
    for r in recs:
        if r["arm"] == "D_comp" and r.get("mix"): byfam[r["fam"]].append(r["mix"])
    print("\n### Learned mixing weights (D, composition only)\n\n| family | adapters | mean largest weight | mean effective number of adapters (exp entropy) | uniform = 1/K |\n|---|---|---|---|---|")
    for f, ms_ in byfam.items():
        K = len(ms_[0]); mx = np.mean([max(m) for m in ms_]); ne = np.mean([np.exp(-sum(w * np.log(w + 1e-12) for w in m)) for m in ms_]); print(f"| {f} | {K} | {mx:.2f} | {ne:.1f} | {1/K:.2f} |")
# wall and cost
cs = R.get("complete_cells", [])
print(f"\nWall time of the run script: {R.get('wall_s')} s; complete cells: {len(cs)}; errors: {len(R.get('errors', []))}")
at = [r["at_s"] for r in recs]
if at: print(f"Last record at {max(at)} s after the script started.")
