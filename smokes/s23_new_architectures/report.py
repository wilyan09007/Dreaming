"""S23-lite aggregation. `dump()` is called by run.py every few seconds (results.json is rewritten atomically so a timeout still leaves data);
`python report.py` prints the markdown tables used in RESULTS.md. ponytail: plain dict grouping, no pandas."""
import json, os, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
HERE = Path(__file__).resolve().parent
BASE = "LR baseline (k random rows)"; MAIN_LR = 5e-4

def load_rows():
    rows = []
    for f in sorted(HERE.glob("rows_*.jsonl")):
        for ln in f.read_text(encoding="utf-8").splitlines():
            try: rows.append(json.loads(ln))
            except Exception: pass  # a line still being appended
    return rows

def label(r):
    s = r.get("method", "?")
    if r.get("pool") not in (None, "-"): s += f" [pool {r['pool']}]"
    if r.get("lr") not in (None, MAIN_LR): s += f" [lr {r['lr']}]"
    return s

def groups(rows):
    g = defaultdict(list)
    for r in rows:
        if "acc" in r and r.get("error") is None: g[(r["arm"], label(r), r.get("k"))].append(r)
    return g

def mean(rs, f):
    v = [r[f] for r in rs if r.get(f) is not None]; return float(np.mean(v)) if v else None

def summary(rows):
    return [dict(arm=a, method=m, k=k, n=len(rs), acc=mean(rs, "acc"), cov=mean(rs, "cov"), err=mean(rs, "err"), params=mean(rs, "params"), secs=mean(rs, "secs") or mean(rs, "train_secs"))
            for (a, m, k), rs in sorted(groups(rows).items(), key=lambda x: (str(x[0][0]), x[0][1], x[0][2]))]

def dump(path=HERE / "results.json"):
    rows = load_rows(); status = {}
    for f in ("paw_status.json", "status.json"):
        if (HERE / f).exists():
            try: status[f] = json.loads((HERE / f).read_text(encoding="utf-8"))
            except Exception: pass
    tmp = Path(str(path) + ".tmp"); tmp.write_text(json.dumps(dict(n_rows=len(rows), status=status, summary=summary(rows), rows=rows), default=float), encoding="utf-8"); os.replace(tmp, path)

def f(x, p=3): return "-" if x is None else f"{x:.{p}f}"

def tables(rows=None):
    rows = rows or load_rows(); g = groups(rows); out = []
    ks = sorted({k for (_, _, k) in g if k is not None and k > 0}); keyorder = sorted({(a, m) for (a, m, k) in g if k in ks or k == 0 or k == -1}, key=lambda x: (x[0], x[1]))
    base = {(r["task"], k, r["draw"]): r for k in ks for r in g.get(("1", BASE, k), [])}
    out.append("### Accuracy / certified coverage at 5% budget, mean over tasks and draws (paired delta vs the LR baseline, mean +- s.e.)\n")
    out.append("| arm | method | " + " | ".join(f"k={k} acc | k={k} cov | k={k} d-acc" for k in ks) + " | trainable params | s per fit |"); out.append("|---|---|" + "---|" * (3 * len(ks) + 2))
    for a, m in keyorder:
        cells, par, sec = [], None, None
        for k in ks:
            rs = g.get((a, m, k), [])
            if not rs: cells += ["-", "-", "-"]; continue
            d = [r["acc"] - base[(r["task"], k, r["draw"])]["acc"] for r in rs if (r["task"], k, r["draw"]) in base]
            ds = "-" if not d or m == BASE else f"{np.mean(d):+.3f} +- {np.std(d) / max(1, np.sqrt(len(d))):.3f}"
            cells += [f"{f(mean(rs, 'acc'))} (n={len(rs)})", f(mean(rs, "cov")), ds]; par = par or mean(rs, "params"); sec = sec or mean(rs, "secs") or mean(rs, "train_secs")
        for k in (0, -1):
            rs = g.get((a, m, k), [])
            if rs and not any(g.get((a, m, kk)) for kk in ks): cells = [f"k={k}: acc {f(mean(rs, 'acc'))}, cov {f(mean(rs, 'cov'))}"] + ["-"] * (3 * len(ks) - 1); par = mean(rs, "params")
        out.append(f"| {a} | {m} | " + " | ".join(cells) + f" | {f(par, 0)} | {f(sec, 1)} |")
    # per-task accuracy for the headline methods
    tasks = sorted({r["task"] for r in rows if "task" in r}); heads = [BASE, "LR on k*K k-means medoid rows", "TK-KNN-style self-training [pool full]", "LabelSpreading -> LR on reached rows [pool full]",
                                                                    "Sinkhorn pseudo-labels + LR (uniform prior) [pool full]", "TSDAE 1000 steps + frozen LR", "PAW compiled init + fine-tune", "random LoRA init + fine-tune (same prompt)"]
    out.append("\n### Accuracy per task (mean over draws)\n"); out.append("| method | k | " + " | ".join(tasks) + " |"); out.append("|---|---|" + "---|" * len(tasks))
    for m in heads:
        for k in ks:
            vals = []
            for tk in tasks:
                rs = [r for (a, mm, kk), x in g.items() if mm == m and kk == k for r in x if r["task"] == tk]; vals.append(f(mean(rs, "acc")))
            if any(v != "-" for v in vals): out.append(f"| {m} | {k} | " + " | ".join(vals) + " |")
    # pseudo-label accuracy per round
    out.append("\n### Pseudo-label accuracy per round (mean over tasks and draws; TK-KNN: accuracy of the selected rows in rounds 1..4; Sinkhorn: round 0 = plain LR on the pool, rounds 1..3 = balanced labels; LabelSpreading: [accuracy on reached rows, fraction reached])\n")
    out.append("| method | pool | k | per-round pseudo-label accuracy |"); out.append("|---|---|---|---|")
    pl = defaultdict(list)
    for r in rows:
        if r.get("pl") and r.get("arm") == "2" and "LR on reached" not in r["method"]: pl[(r["method"], r["pool"], r["k"])].append(r["pl"])
    for (m, p, k), v in sorted(pl.items()):
        n = min(len(x) for x in v); out.append(f"| {m} | {p} | {k} | " + ", ".join(f(float(np.mean([x[i] for x in v if x[i] is not None])), 3) for i in range(n)) + " |")
    return "\n".join(out)

if __name__ == "__main__":
    print(tables())
