"""S1: frozen-embedding + logistic-regression student on Banking77 / CLINC150. Learning curve, top-2-gap
confidence, SGR certificate (budget 5% / 2%), realized holdout error, calibration-set size, day-zero label-name baseline."""
import sys, time; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.linear_model import LogisticRegression
from common import *

OUT = ROOT / "s1_student_certificate" / "RESULTS.md"
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)

def fmt(x, nd=3):
    return "-" if x is None or x == float("inf") else f"{x:.{nd}f}"

def run(name, tr_text, tr_y, te_text, te_y, label_names, sizes):
    Xtr, Xte = embed(tr_text), embed(te_text)
    K = len(np.unique(tr_y))
    rng = np.random.default_rng(1); perm = rng.permutation(len(te_y)); cal, hold = perm[: len(perm)//2], perm[len(perm)//2:]
    # day-zero: cosine to the label name embeddings
    L = embed([n.replace("_", " ") for n in label_names])
    zs = (Xte @ L.T).argmax(1)
    say(f"\n## {name}: {K} labels, train={len(tr_y)}, test={len(te_y)}\n")
    say(f"Day-zero label-name cosine accuracy (no rows): **{(zs == te_y).mean():.3f}**\n")
    say("| rows | acc | cert@5% cutoff | coverage | holdout err above cutoff | cert@2% coverage | holdout err | train s |")
    say("|---|---|---|---|---|---|---|---|")
    for n in sizes:
        idx = stratified_subsample(tr_y, n, seed=0)
        t = time.time(); clf = LogisticRegression(C=10, max_iter=2000).fit(Xtr[idx], tr_y[idx]); dt = time.time() - t
        P = clf.predict_proba(Xte); pred = clf.classes_[P.argmax(1)]; conf = confidence(P); ok = pred == te_y
        row = [f"{len(idx)}", f"{ok.mean():.3f}"]
        c5 = certify(conf[cal], ok[cal], budget=0.05); h5 = apply_certificate(conf[hold], ok[hold], c5["cutoff"])
        c2 = certify(conf[cal], ok[cal], budget=0.02); h2 = apply_certificate(conf[hold], ok[hold], c2["cutoff"])
        row += [fmt(c5["cutoff"]), f"{c5['coverage']:.2f}", fmt(h5["error"]), f"{c2['coverage']:.2f}", fmt(h2["error"]), f"{dt:.1f}"]
        say("| " + " | ".join(row) + " |")
    # calibration-set size study with the full-train student at budget 5%
    clf = LogisticRegression(C=10, max_iter=2000).fit(Xtr, tr_y); P = clf.predict_proba(Xte); conf = confidence(P); ok = clf.classes_[P.argmax(1)] == te_y
    say(f"\nCalibration-set size vs certified coverage (full-train student, budget 5%, delta 5%), holdout = other half of test:\n")
    say("| cal rows | cutoff | coverage | holdout err |"); say("|---|---|---|---|")
    for m in (100, 200, 500, 1000, len(cal)):
        c = certify(conf[cal[:m]], ok[cal[:m]], 0.05); h = apply_certificate(conf[hold], ok[hold], c["cutoff"])
        say(f"| {m} | {fmt(c['cutoff'])} | {c['coverage']:.2f} | {fmt(h['error'])} |")
    mx = P.max(1); c1 = certify(conf[cal], ok[cal], 0.05); c2 = certify(mx[cal], ok[cal], 0.05)
    say(f"\nConfidence = top-2 gap: coverage {c1['coverage']:.2f}; confidence = max prob: coverage {c2['coverage']:.2f} (budget 5%).")

if __name__ == "__main__":
    t0 = time.time()
    b = load_banking77()
    run("Banking77", *b["train"], *b["test"], b["names"], sizes=[77, 154, 385, 770, 1540, 3080, 6000, len(b["train"][1])])
    c = load_clinc150()
    run("CLINC150 (in-scope)", *c["train"], *c["test"], c["names"], sizes=[150, 300, 750, 1500, 3000, 7500, len(c["train"][1])])
    say(f"\nTotal wall time {time.time()-t0:.0f}s on CPU, embedding model {EMB_MODEL}.")
    write_results(OUT, "S1 — student + certificate (frozen bge-small + logistic regression)", lines)
