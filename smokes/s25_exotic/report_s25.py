"""S25 recomputation for RESULTS.md. Reads results.json (the 600 raw rows), rows_bench.jsonl and timing.json; recomputes every number from the raw rows (does not read tables.md or
the `summary` block). Run: python -I report_s25.py   (numpy and scipy, a few seconds).
Conventions: a cell = (task, draw); 5 tasks x 3 draws = 15 cells per (method, k). "+-" = sample s.d. (ddof=1) over the 3 draws of the task-averaged value (same as S24/S27).
Paired difference = method minus baseline on the same (task, draw, k); "s.e." = s.d. of the 15 cell differences / sqrt(15); W/L/T = cells where the method has higher / lower / equal
certified coverage than the baseline. Task-level test (the unit used in RESULTS.md's verdict): per task, average the 3 draws' paired differences; t = mean of the 5 task means / (s.d. of the 5 / sqrt(5)),
4 d.f., two-sided p, uncorrected for the many comparisons; "+tasks" = tasks (of 5) with a positive mean difference. Draws within a task share the pool, calibration and test sets (and JEPA's pretrained network), so the 15 cells
are not independent and the cell-level s.e. overstates precision. Realized error is pooled over covered test rows (sum of err*cov*n_test / sum of cov*n_test), not a mean of cell means."""
import json, math, collections
from pathlib import Path
import numpy as np
from scipy import stats
HERE = Path(__file__).resolve().parent
BASE = "Baseline: bge-small + LR"; KS = [1, 3, 10]; BUDGET = 0.05
NTEST = {"clinc/travel": 450, "clinc/work": 450, "banking/sub0": 480, "banking/sub1": 480, "banking/sub2": 479}   # test rows per task (pod log, "tasks (name, pool, calibration, test)")

D = json.loads((HERE / "results.json").read_text(encoding="utf-8")); rows = D["rows"]
bench = [json.loads(l) for l in (HERE / "rows_bench.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
timing = json.loads((HERE / "timing.json").read_text(encoding="utf-8"))
ok = [r for r in rows if "acc" in r]; bad = [r for r in rows if r.get("status")]; nb = [r for r in rows if "acc" not in r and not r.get("status")]
M = list(dict.fromkeys(r["method"] for r in ok)); T = list(dict.fromkeys(r["task"] for r in ok))
C = {(r["method"], r["task"], r["draw"], r["k"]): r for r in ok}
assert len(M) == 13 and len(ok) == 13 * 5 * 3 * 3 and len(C) == len(ok), (len(M), len(ok), len(C))
sd = lambda v: float(np.std(v, ddof=1))
def draws(m, k, f): return [np.mean([f(C[m, t, d, k]) for t in T]) for d in range(3)]   # task-average per draw
def cells(m, k, f): return [f(C[m, t, d, k]) for t in T for d in range(3)]
def pm(v, p=3): return f"{np.mean(v):.{p}f} +- {sd(v):.{p}f}"
acc, cov = (lambda r: r["acc"]), (lambda r: r["cov"])
def short(m): return m.replace("Baseline: bge-small + LR", "Baseline LR").replace("GPT-2 L8 dense residual, last token (control)", "GPT-2 L8 dense residual (control)")

print("## Integrity")
print(f"rows {len(rows)} = scored {len(ok)} + error rows {len(bad)} + bench rows {len(nb)}; scored = {len(M)} methods x {len(T)} tasks x 3 draws x {len(KS)} k = {len(M)*5*3*3}")
print("error rows by method:", dict(collections.Counter(r["method"] for r in bad)), "| groups with scored rows:", dict(collections.Counter(r["group"] for r in ok)))
print("methods with no rows at all: Tsetlin machine (BoW) [group tm: no rows_tm.jsonl, no error row], SAE concepts last/mean [8 error rows then silent skip]")
print("timing.json:", {k: (round(v) if isinstance(v, float) else v) for k, v in timing.items()})

print("\n## Main table: accuracy and certified coverage (mean over 15 cells, +- s.d. over 3 draws of the task mean)")
print("| method | acc k=1 | acc k=3 | acc k=10 | cov k=1 | cov k=3 | cov k=10 |\n|---|---|---|---|---|---|---|")
for m in M: print(f"| {short(m)} | " + " | ".join(pm(draws(m, k, acc)) for k in KS) + " | " + " | ".join(pm(draws(m, k, cov), 3) for k in KS) + " |")

print("\n## Paired difference to the baseline, certified coverage: mean +- s.d. over draws [W/L/T of 15 cells]")
print("| method | d-cov k=1 | d-cov k=3 | d-cov k=10 |\n|---|---|---|---|")
def paired(m, k, f): return [f(C[m, t, d, k]) - f(C[BASE, t, d, k]) for t in T for d in range(3)]
def pdraw(m, k, f): return [np.mean([f(C[m, t, d, k]) - f(C[BASE, t, d, k]) for t in T]) for d in range(3)]
def wlt(v): return f"{sum(x > 1e-12 for x in v)}/{sum(x < -1e-12 for x in v)}/{sum(abs(x) <= 1e-12 for x in v)}"
for m in M[1:]: print(f"| {short(m)} | " + " | ".join(f"{np.mean(paired(m,k,cov)):+.3f} +- {sd(pdraw(m,k,cov)):.3f} [{wlt(paired(m,k,cov))}]" for k in KS) + " |")
print("\n## Paired difference to the baseline, accuracy: mean +- s.d. over draws [W/L/T of 15 cells]")
print("| method | d-acc k=1 | d-acc k=3 | d-acc k=10 |\n|---|---|---|---|")
for m in M[1:]: print(f"| {short(m)} | " + " | ".join(f"{np.mean(paired(m,k,acc)):+.3f} +- {sd(pdraw(m,k,acc)):.3f} [{wlt(paired(m,k,acc))}]" for k in KS) + " |")
print("\n## Paired difference, mean +- s.e. over 15 cells (t = mean/s.e.; CELL-LEVEL: treats the 15 cells as independent, which they are not - draws within a task share pool, calibration, test set and the JEPA network - so it overstates precision; the task-level test below is the one quoted in the verdict)")
print("| method | measure | k=1 | k=3 | k=10 |\n|---|---|---|---|---|")
for m in M[1:]:
    for nm, f in (("cov", cov), ("acc", acc)):
        out = []
        for k in KS:
            v = paired(m, k, f); se = sd(v) / math.sqrt(len(v)); out.append(f"{np.mean(v):+.3f} +- {se:.3f} (t={np.mean(v)/se:+.1f})" if se > 0 else f"{np.mean(v):+.3f} +- 0")
        print(f"| {short(m)} | {nm} | " + " | ".join(out) + " |")

print("\n## Task-level paired test (task = unit, n=5, 4 d.f.): mean of the task means (t, p, tasks with a positive difference)")
print("| method | measure | k=1 | k=3 | k=10 |\n|---|---|---|---|---|")
def tdraw(m, k, f): return [np.mean([f(C[m, t, d, k]) - f(C[BASE, t, d, k]) for d in range(3)]) for t in T]   # one paired difference per task (3 draws averaged)
def tl(v):
    mu = float(np.mean(v)); se = sd(v) / math.sqrt(len(v)); pos = sum(x > 0 for x in v)
    return f"{mu:+.3f} (t n/a, {pos}+ of 5)" if se == 0 else f"{mu:+.3f} (t {mu/se:+.1f}, p {2*stats.t.sf(abs(mu/se), len(v)-1):.2f}, {pos}+ of 5)"
for m in M[1:]:
    for nm, f in (("cov", cov), ("acc", acc)): print(f"| {short(m)} | {nm} | " + " | ".join(tl(tdraw(m, k, f)) for k in KS) + " |")
print("task means behind the headline arms, in task order " + str(T) + ", per k=1 / 3 / 10:")
for m in ("LR on [embedding + JEPA hidden]", "HDC bundle (one-shot)", "HDC bundle, pool-centered", "HDC online (OnlineHD, 1 pass)", "Spiking head (2-layer LIF) [membrane gap]", "Spiking head (2-layer LIF) [spike-count gap]"):
    for nm, f in (("cov", cov), ("acc", acc)): print(f"  {short(m)} {nm}: " + " | ".join("[" + " ".join(f"{x:+.3f}" for x in tdraw(m, k, f)) + "]" for k in KS))
jacc = [tdraw("LR on [embedding + JEPA hidden]", k, acc) for k in KS]
print(f"JEPA accuracy: {sum(x > 0 for v in jacc for x in v)} of {3*len(T)} task x k means positive ({', '.join(str(sum(x > 0 for x in v)) for v in jacc)} of 5 tasks at k=1/3/10)")

print("\n## Certificate behaviour: cells that certified anything (of 15) / pooled realized error among covered test rows / worst cell error / cells above the 5% budget")
print("| method | k=1 | k=3 | k=10 |\n|---|---|---|---|")
tot = collections.Counter()
for m in M:
    out = []
    for k in KS:
        cs = [C[m, t, d, k] for t in T for d in range(3)]; cert = [r for r in cs if r["err"] is not None]
        num = sum(r["err"] * r["cov"] * NTEST[r["task"]] for r in cert); den = sum(r["cov"] * NTEST[r["task"]] for r in cert)
        over = sum(r["err"] > BUDGET for r in cert); tot["cert"] += len(cert); tot["over"] += over; tot["cells"] += len(cs)
        out.append(f"{len(cert)} / " + (f"{num/den:.4f} / {max(r['err'] for r in cert):.3f} / {over}" if cert and den else "- / - / -"))
    print(f"| {short(m)} | " + " | ".join(out) + " |")
print(f"all methods: {tot['cert']} of {tot['cells']} cells certified something; {tot['over']} of those had realized test error above 0.05; worst cell error {max(r['err'] for r in ok if r['err'] is not None):.3f}")
b = [C[BASE, t, d, k] for t in T for d in range(3) for k in KS]; print("calibration-promised vs realized test coverage, baseline (mean over cells):", round(np.mean([r['cal_cov'] for r in b]), 3), "vs", round(np.mean([r['cov'] for r in b]), 3))

print("\n## Per task: certified coverage at k=1 / 3 / 10")
print("| method | " + " | ".join(t for t in T) + " |\n|---|" + "---|" * len(T))
for m in M: print(f"| {short(m)} | " + " | ".join("/".join(f"{np.mean([C[m,t,d,k]['cov'] for d in range(3)]):.2f}" for k in KS) for t in T) + " |")
print("\n## Per task: accuracy at k=1 / 3 / 10")
print("| method | " + " | ".join(t for t in T) + " |\n|---|" + "---|" * len(T))
for m in M: print(f"| {short(m)} | " + " | ".join("/".join(f"{np.mean([C[m,t,d,k]['acc'] for d in range(3)]):.3f}" for k in KS) for t in T) + " |")

print("\n## Tasks on which the method's coverage (mean of 3 draws) beats / loses to the baseline's, per k (k=1/3/10)")
for m in M[1:]:
    s = []
    for k in KS:
        d = [np.mean([C[m, t, dr, k]["cov"] - C[BASE, t, dr, k]["cov"] for dr in range(3)]) for t in T]; s.append(f"{sum(x > 0.005 for x in d)}+/{sum(x < -0.005 for x in d)}-")
    print(f"  {short(m)}: " + "  ".join(s))

print("\n## Rows per label to 50% certified coverage (macro coverage over 15 cells; log-linear interpolation between grid points)")
def interp(c):
    if c[1] >= .5: return "<=1"
    for a, b2 in ((1, 3), (3, 10)):
        if c[a] < .5 <= c[b2]: return f"{math.exp(math.log(a) + (.5 - c[a]) / (c[b2] - c[a]) * (math.log(b2) - math.log(a))):.1f}"
    return ">10"
for m in M: c = {k: float(np.mean(cells(m, k, cov))) for k in KS}; print(f"  {short(m)}: cov {c[1]:.3f}/{c[3]:.3f}/{c[10]:.3f} -> {interp(c)}")

print("\n## Determinism (same seed, features and pretraining recomputed in the same process on the same pod; one check per task at k=3, draw 0)")
for m in M:
    d = [r for r in ok if r["method"] == m and r.get("det") is not None]; print(f"  {short(m)}: predictions identical {sum(bool(r['det_pred']) for r in d)}/{len(d)}, predictions+confidence bit-identical {sum(bool(r['det']) for r in d)}/{len(d)}")

print("\n## Cost. NOTE: the field `pred_ms_per_1000` is seconds / n_test_rows * 1000 = milliseconds PER ROW (= seconds per 1000 rows), mislabelled in the code and in tables.md")
print("quiet single-thread bench (clinc/travel, k=3, draw 0, min of 3; predict includes featurization with the cache cleared; the bge encoder itself is not included for any method):")
print("| method | params | fit s | predict ms/row | one-off prep s |\n|---|---|---|---|---|")
for r in bench: print(f"| {r['method']} | {r['params']} | {r['fit_secs']:.4f} | {r['pred_ms_per_1000']:.4f} | {r['prep_secs']:.3f} |")
print("not benched (the bench pass was killed at the deadline while it was on the Tsetlin arm, which precedes torch and gpu groups in arms.GROUPS): PKM, SNN, JEPA, GPT-2 control")
print("\nmain run (5 groups in parallel on one 128-CPU pod, torch 1 thread, JEPA/GPT-2 on the GPU; contended, indicative only), k=3, mean over 5 tasks x 3 draws:")
print("| method | params at inference (task mean) | fit s | predict ms/row | prep s per task (when logged) |\n|---|---|---|---|---|")
for m in M:
    r3 = [C[m, t, d, 3] for t in T for d in range(3)]; pr = [r["prep_secs"] for r in r3 if r.get("prep_secs") and r["draw"] == 0]
    print(f"| {short(m)} | {np.mean([r['params'] for r in r3]):.0f} | {np.mean([r['fit_secs'] for r in r3]):.4f} | {np.mean([r['pred_ms_per_1000'] for r in r3]):.4f} | " + (f"{np.mean(pr):.1f}" if pr else "-") + " |")
print("params by label count (k=3, one value per task): " + "; ".join(f"{short(m)} 12-label {C[m,'banking/sub0',0,3]['params']} / 15-label {C[m,'clinc/travel',0,3]['params']}" for m in M if m in (BASE, "HDC bundle (one-shot)", "NTK kernel ridge", "Product-key memory head", "LR on [embedding + JEPA hidden]", "GPT-2 L8 dense residual, last token (control)")))
print("group wall times (s since T0):", {k: round(v) for k, v in timing.items() if k.endswith("_secs")})

print("\n## Facts quoted in the text (counts that depend on how rows are grouped)")
SP = [m for m in M if m.startswith("Spiking")]; GZ = [m for m in M if m.startswith("gzip")]
cs = {m: {(t, d, k) for t in T for d in range(3) for k in KS if C[m, t, d, k]["err"] is not None} for m in M}
print(f"certified cells per spiking confidence: " + " + ".join(f"{len(cs[m])} ({m.split(chr(91))[1][:-1]})" for m in SP) + f"; cells where at least one of the two certified: {len(cs[SP[0]] | cs[SP[1]])}; spiking predictions are identical in {sum(C[SP[0], t, d, k]['acc'] == C[SP[1], t, d, k]['acc'] for t in T for d in range(3) for k in KS)}/45 cells by accuracy")
tot_c = sum(len(v) for v in cs.values()); print(f"certified cells over all {len(M)} rows: {tot_c} of {len(M)*45}; counting the spiking head once (union of its two confidences): {tot_c - len(cs[SP[0]]) - len(cs[SP[1]]) + len(cs[SP[0]] | cs[SP[1]])} of {(len(M)-1)*45}")
low = [m for m in M if m.startswith(("gzip", "Product", "GPT-2"))]
print(f"gzip, product-key memory, GPT-2 control: {sum(len(cs[m]) for m in low)} certified cells of {len(low)*45} ({len(low)} rows x 45; gzip's two rows count separately: " + ", ".join(f"{short(m)} {len(cs[m])}" for m in low) + ")")
eq = {k: sum(C[GZ[0], t, d, k]["acc"] == C[GZ[1], t, d, k]["acc"] for t in T for d in range(3)) for k in KS}
print(f"gzip vote-margin (3-NN vote) vs NCD-gap (1-NN class): equal accuracy in {sum(eq.values())}/45 cells, by k {eq}; NCD-gap certified cells by k {[sum(1 for (t, d, kk) in cs[GZ[1]] if kk == k) for k in KS]}")
print("spiking spike-count minus membrane gap coverage (the only same-prediction confidence comparison):", {k: round(float(np.mean(cells(SP[0], k, cov)) - np.mean(cells(SP[1], k, cov))), 3) for k in KS})
sdv = {(m, k): sd(draws(m, k, cov)) for m in M for k in KS}; nz = {key: v for key, v in sdv.items() if v > 0}
lo, hi = min(nz, key=nz.get), max(nz, key=nz.get)
print(f"s.d. over draws of the task-mean coverage, rows that certify something: min {nz[lo]:.3f} ({short(lo[0])} k={lo[1]}), max {nz[hi]:.3f} ({short(hi[0])} k={hi[1]}); NTK ridge k=10: {sdv['NTK kernel ridge', 10]:.3f}")
print("task with the lowest mean accuracy / lowest mean coverage, per method at k=1 / 3 / 10 (first task listed on ties):")
for m in M:
    la = [min(T, key=lambda t: np.mean([C[m, t, d, k]["acc"] for d in range(3)])) for k in KS]; lc = [min(T, key=lambda t: np.mean([C[m, t, d, k]["cov"] for d in range(3)])) for k in KS]
    print(f"  {short(m)}: acc {' / '.join(x.split('/')[1] for x in la)}; cov {' / '.join(x.split('/')[1] for x in lc)}")
