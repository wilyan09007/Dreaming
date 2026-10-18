"""S25 aggregation. `python report.py` prints the markdown tables used in RESULTS.md and rewrites results.json + tables.md; run.py calls dump() every few seconds so a
timeout still leaves data. Macro averages weight every (task, draw) run equally. cov = realized coverage on the test split above the cutoff certified on the 300
calibration rows (5% budget, delta 0.05); err = realized error among those covered rows."""
import json, math, os
from pathlib import Path
from collections import defaultdict
import numpy as np
HERE = Path(__file__).resolve().parent
ORDER = ["fast", "ncd", "tm", "torch", "gpu"]; BASE = "Baseline: bge-small + LR"

def load():
    rows = []
    for g in ORDER + ["bench"]:
        f = HERE / f"rows_{g}.jsonl"
        for ln in (f.read_text(encoding="utf-8").splitlines() if f.exists() else []):
            try: rows.append(json.loads(ln))
            except Exception: pass   # a line still being written
    return rows

def mean(v): v = [x for x in v if x is not None]; return float(np.mean(v)) if v else None
def f3(x, p=3): return "-" if x is None else f"{x:.{p}f}"
def parts(rows):
    ok = [r for r in rows if "acc" in r and r.get("group") != "bench"]; bench = {r["method"]: r for r in rows if r.get("group") is not None and "pred_ms_per_1000" in r and "task" not in r}
    meths = list(dict.fromkeys(r["method"] for r in ok)); tasks = list(dict.fromkeys(r["task"] for r in ok)); ks = sorted({r["k"] for r in ok})
    bad = [r for r in rows if r.get("status")]; return ok, bench, meths, tasks, ks, bad

def cell(ok, m, k, task=None): return [r for r in ok if r["method"] == m and r["k"] == k and (task is None or r["task"] == task)]

def main_table(ok, meths, ks, task=None):
    out = ["| method | " + " | ".join(f"acc k={k} | cov k={k}" for k in ks) + " | realized err k=" + "/".join(map(str, ks)) + " | runs |", "|---|" + "---|" * (2 * len(ks) + 2)]
    for m in meths:
        c = [cell(ok, m, k, task) for k in ks]
        out.append(f"| {m} | " + " | ".join(f"{f3(mean([r['acc'] for r in rs]))} | {f3(mean([r['cov'] for r in rs]),2)}" for rs in c) + " | "
                   + "/".join(f3(mean([r["err"] for r in rs]), 3) for rs in c) + f" | {min(len(rs) for rs in c)} |")
    return "\n".join(out)

def first_k(cov):   # cov: {k: macro coverage}
    for k in sorted(cov):
        if cov[k] is not None and cov[k] >= 0.5: return str(k)
    return f">{max(cov)}"
def interp(cov):
    ks = sorted(cov)
    if cov[ks[0]] is not None and cov[ks[0]] >= 0.5: return f"<={ks[0]}"
    for a, b in zip(ks, ks[1:]):
        if cov[a] is not None and cov[b] is not None and cov[a] < 0.5 <= cov[b]: return f"~{math.exp(math.log(a) + (0.5 - cov[a]) / (cov[b] - cov[a]) * (math.log(b) - math.log(a))):.1f}"
    return f">{ks[-1]}"

def rows_to_50(ok, meths, tasks, ks):
    out = ["| method | macro cov " + " / ".join(f"k={k}" for k in ks) + " | first k on grid | interpolated k (log-linear) | first k per task: " + ", ".join(t.split("/")[1] for t in tasks) + " |", "|---|---|---|---|---|"]
    for m in meths:
        cov = {k: mean([r["cov"] for r in cell(ok, m, k)]) for k in ks}
        per = [first_k({k: mean([r["cov"] for r in cell(ok, m, k, t)]) for k in ks}) for t in tasks]
        out.append(f"| {m} | " + " / ".join(f3(cov[k], 2) for k in ks) + f" | {first_k(cov)} | {interp(cov)} | " + ", ".join(per) + " |")
    return "\n".join(out)

def det_table(ok, meths):
    out = ["| method | predictions identical (tasks) | predictions + confidence bit-identical (tasks) |", "|---|---|---|"]
    for m in meths:
        d = [r for r in ok if r["method"] == m and r.get("det") is not None]
        out.append(f"| {m} | {sum(bool(r['det_pred']) for r in d)}/{len(d)} | {sum(bool(r['det']) for r in d)}/{len(d)} |")
    return "\n".join(out)

def cost_table(ok, bench, meths):
    out = ["| method | params at inference | fit s (k=3) | predict ms / 1000 rows | one-off prep s | device |", "|---|---|---|---|---|---|"]
    for m in meths:
        b = bench.get(m.split(" [")[0]) or bench.get(m); r3 = cell(ok, m, 3)
        out.append(f"| {m} | {f3(mean([r['params'] for r in r3]), 0)} | {f3(b['fit_secs'] if b else None)} | {f3(b['pred_ms_per_1000'] if b else None, 1)} | {f3(b['prep_secs'] if b else None, 1)} | {b['device'] if b else '-'} |")
    return "\n".join(out)

def delta_table(ok, meths, ks):
    base = {(r["task"], r["draw"], r["k"]): r for r in ok if r["method"] == BASE}
    out = ["| method | " + " | ".join(f"d acc k={k} | d cov k={k}" for k in ks) + " |", "|---|" + "---|" * (2 * len(ks))]
    for m in meths:
        if m == BASE: continue
        cs = []
        for k in ks:
            p = [(r["acc"] - base[(r["task"], r["draw"], k)]["acc"], r["cov"] - base[(r["task"], r["draw"], k)]["cov"]) for r in cell(ok, m, k) if (r["task"], r["draw"], k) in base]
            cs += [f"{np.mean([a for a, _ in p]):+.3f}" if p else "-", f"{np.mean([c for _, c in p]):+.2f}" if p else "-"]
        out.append(f"| {m} | " + " | ".join(cs) + " |")
    return "\n".join(out)

def tables():
    rows = load(); ok, bench, meths, tasks, ks, bad = parts(rows)
    if not ok: return "no rows yet"
    s = ["### Macro average over 5 tasks x 3 draws (acc = test accuracy, cov = certified coverage at 5% budget on test, err = realized error above the cutoff)", main_table(ok, meths, ks)]
    for t in tasks: s += [f"\n### {t}", main_table(ok, meths, ks, t)]
    s += ["\n### Rows per label to reach 50% certified coverage", rows_to_50(ok, meths, tasks, ks), "\n### Paired difference to the LR baseline (same task, draw, k)", delta_table(ok, meths, ks),
          "\n### Determinism (same seed, features and pretraining recomputed, k=3 draw 0, one check per task)", det_table(ok, meths), "\n### Cost (single-thread CPU unless noted; task clinc/travel, k=3)", cost_table(ok, bench, meths)]
    if bad: s += ["\n### Skipped / failed", "\n".join(f"- {r['method']} ({r.get('task','')}, k={r.get('k','')}): {r['status']}: {r['reason']}" for r in list({(r['method'], r['reason'][:80]): r for r in bad}.values()))]
    return "\n".join(s)

def dump():
    rows = load(); ok, bench, meths, tasks, ks, bad = parts(rows)
    summ = [dict(method=m, k=k, runs=len(cell(ok, m, k)), acc=mean([r["acc"] for r in cell(ok, m, k)]), cov=mean([r["cov"] for r in cell(ok, m, k)]), err=mean([r["err"] for r in cell(ok, m, k)])) for m in meths for k in ks]
    tmp = HERE / "results.json.tmp"; tmp.write_text(json.dumps(dict(n_rows=len(rows), summary=summ, bench=bench, rows=rows), default=float), encoding="utf-8"); os.replace(tmp, HERE / "results.json")

if __name__ == "__main__":
    t = tables(); print(t); (HERE / "tables.md").write_text(t + "\n", encoding="utf-8"); dump()
