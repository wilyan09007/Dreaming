"""Builds tables.md from results.json (python3 report.py)."""
import json, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
BASE = "LR bge-small (baseline)"

def main():
    R = json.load(open(HERE / "results.json")); C = R["cells"]; KS = R["meta"]["ks"]
    arms = list(dict.fromkeys(c["arm"] for c in C)); tasks = list(dict.fromkeys(c["task"] for c in C))
    g = lambda arm, k, key: {(c["task"], c["draw"]): c[key] for c in C if c["arm"] == arm and c["k"] == k}
    def mean_sd(arm, k, key):
        d = g(arm, k, key); per = [np.mean([v for (t, dr), v in d.items() if dr == dd]) for dd in range(R["meta"]["draws"])]
        return np.mean(list(d.values())), np.std(per, ddof=1)
    def err(arm, k):
        v = [c["err"] for c in C if c["arm"] == arm and c["k"] == k and c["err"] is not None]; n0 = sum(1 for c in C if c["arm"] == arm and c["k"] == k and c["cov"] == 0)
        return (np.mean(v) if v else float("nan")), n0
    L = []
    L += ["### Main table (mean over 5 tasks x 3 draws; +- = s.d. over the 3 draws of the task-averaged value)", "",
          "| arm | acc k=1 | acc k=3 | acc k=10 | cert. cov k=1 | cov k=3 | cov k=10 |", "|---|---|---|---|---|---|---|"]
    for a in arms:
        r = [f"{mean_sd(a, k, 'acc')[0]:.3f} +- {mean_sd(a, k, 'acc')[1]:.3f}" for k in KS] + [f"{mean_sd(a, k, 'cov')[0]:.3f} +- {mean_sd(a, k, 'cov')[1]:.3f}" for k in KS]
        L.append(f"| {a} | " + " | ".join(r) + " |")
    L += ["", "### Paired difference to the baseline (same task, k, draw; mean +- standard error over 15 cells)", "",
          "| arm | d-cov k=1 | k=3 | k=10 | d-acc k=1 | k=3 | k=10 |", "|---|---|---|---|---|---|---|"]
    for a in arms:
        if a == BASE: continue
        row = []
        for key in ("cov", "acc"):
            for k in KS:
                x, b = g(a, k, key), g(BASE, k, key); d = np.array([x[i] - b[i] for i in b]); row.append(f"{d.mean():+.3f} +- {d.std(ddof=1) / np.sqrt(len(d)):.3f}")
        L.append(f"| {a} | " + " | ".join(row) + " |")
    L += ["", "### Realized error above the cutoff on the test rows (mean over cells that certified something; budget 0.05) and cells with nothing certified", "",
          "| arm | err k=1 | err k=3 | err k=10 | zero-coverage cells k=1 / 3 / 10 (of 15) |", "|---|---|---|---|---|"]
    for a in arms:
        e = [err(a, k) for k in KS]; L.append(f"| {a} | " + " | ".join(f"{x[0]:.3f}" for x in e) + " | " + " / ".join(str(x[1]) for x in e) + " |")
    L += ["", "### Cost: fit time per cell (s, mean over the 15 cells at each k) and parameters", "",
          "| arm | fit s k=1 | k=3 | k=10 | trained params (head) | frozen encoder params (M) | embed ms per text (CPU) |", "|---|---|---|---|---|---|---|"]
    emb_of = {"bge-small": "bge_small", "LFM2.5-350M": "lfm25_350m", "bge-base": "bge_base"}
    for a in arms:
        ft = [np.mean([c["fit_s"] for c in C if c["arm"] == a and c["k"] == k]) for k in KS]
        p = int(np.mean([c["params"] for c in C if c["arm"] == a and c["task"] == tasks[0] and c["k"] == 1]))
        en = [v for kk, v in emb_of.items() if kk in a]; info = R["meta"]["embedders"][en[0] if en else "bge_small"]
        L.append(f"| {a} | " + " | ".join(f"{x:.3f}" for x in ft) + f" | {p} | {info['params_m']:.0f} | {info['ms_per_text']:.1f} |")
    L += ["", "### Per-task accuracy / coverage at k=1, 3, 10", "", "| arm | task | acc k=1 | k=3 | k=10 | cov k=1 | k=3 | k=10 |", "|---|---|---|---|---|---|---|---|"]
    for a in arms:
        for t in tasks:
            f = lambda k, key: np.mean([c[key] for c in C if c["arm"] == a and c["task"] == t and c["k"] == k])
            L.append(f"| {a} | {t} | " + " | ".join(f"{f(k, 'acc'):.3f}" for k in KS) + " | " + " | ".join(f"{f(k, 'cov'):.3f}" for k in KS) + " |")
    hp = [c for c in C if c["arm"].startswith("delta")]
    if hp:
        L += ["", "### Chosen hyper-parameters", "", "| arm | k | chosen values (count over 15 cells) |", "|---|---|---|"]
        from collections import Counter
        for a, keys in (("delta-rule head (bge-small)", ("lr", "epochs")), ("ridge least squares (bge-small)", ("lam",)), ("Hopfield read (bge-small)", ("beta",))):
            for k in KS:
                cnt = Counter(tuple(c[x] for x in keys) for c in C if c["arm"] == a and c["k"] == k)
                L.append(f"| {a} | {k} | " + ", ".join(f"{'/'.join(map(str, v))}: {n}" for v, n in sorted(cnt.items())) + " |")
    (HERE / "tables.md").write_text("\n".join(L) + "\n"); print("\n".join(L))

if __name__ == "__main__":
    main()
