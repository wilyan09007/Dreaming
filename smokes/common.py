"""Shared helpers for smokes. ponytail: frozen sentence embeddings + sklearn; no training framework."""
import hashlib, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
CACHE = ROOT / ".cache"; CACHE.mkdir(exist_ok=True)
EMB_MODEL = "BAAI/bge-small-en-v1.5"  # 33M params, 384-d, runs fine on CPU

def embed(texts, model=EMB_MODEL):
    texts = list(texts)
    key = hashlib.sha1((model + "\x00".join(texts)).encode()).hexdigest()[:16]
    f = CACHE / f"emb_{key}.npy"
    if f.exists():
        return np.load(f)
    from sentence_transformers import SentenceTransformer
    t = time.time()
    e = SentenceTransformer(model, device="cpu").encode(texts, batch_size=128, normalize_embeddings=True,
                                                        show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    print(f"[embed] {len(texts)} texts in {time.time()-t:.1f}s ({model})")
    np.save(f, e)
    return e

def load_banking77():
    import datasets
    d = datasets.load_dataset("mteb/banking77")
    names = [None] * 77
    for y, s in zip(d["train"]["label"], d["train"]["label_text"]):
        names[y] = s
    return {s: (d[s]["text"], np.array(d[s]["label"])) for s in ("train", "test")} | {"names": names}

def load_clinc150():
    """In-scope CLINC150 only (oos dropped). Returns train/validation/test as (texts, labels) + names."""
    import datasets
    d = datasets.load_dataset("clinc/clinc_oos", "plus")
    names = d["train"].features["intent"].names
    oos = names.index("oos")
    out = {}
    for s in ("train", "validation", "test"):
        keep = [i for i, y in enumerate(d[s]["intent"]) if y != oos]
        out[s] = ([d[s]["text"][i] for i in keep], np.array([d[s]["intent"][i] for i in keep]))
    out["names"] = names
    return out

def confidence(proba):
    """Gap between top-2 scores: the missions' definition of confidence."""
    s = np.sort(proba, axis=1)
    return s[:, -1] - s[:, -2]

def cp_upper(errors, n, delta):
    """Clopper-Pearson one-sided upper bound on error rate."""
    from scipy.stats import beta
    if n == 0 or errors >= n:
        return 1.0
    return float(beta.ppf(1 - delta, errors + 1, n - errors))

def certify(conf, correct, budget, delta=0.05, grid=40):
    """Fixed-grid certificate (replaces the SGR binary search below, which S4 showed is path dependent: the
    Clopper-Pearson bound is U-shaped in coverage, so a failed probe at n/2 says nothing about other coverages).
    Evaluate `grid` coverage levels (top j/grid rows by confidence, ties closed), each with delta/grid (union bound),
    and keep the largest level whose bound is <= budget. Returns dict(cutoff, coverage, emp_error, bound);
    cutoff=inf means nothing certifiable."""
    conf, correct = np.asarray(conf), np.asarray(correct, bool)
    order = np.argsort(-conf, kind="stable"); conf, correct = conf[order], correct[order]
    n = len(conf); d = delta / grid; cum_err = np.cumsum(~correct); best = None
    for j in range(grid, 0, -1):
        k = max(1, int(round(n * j / grid)))
        while k < n and conf[k] == conf[k - 1]:  # close ties so the cutoff is a real threshold
            k += 1
        ub = cp_upper(int(cum_err[k - 1]), k, d)
        if ub <= budget:
            best = (k, ub); break
    if best is None:
        return dict(cutoff=float("inf"), coverage=0.0, emp_error=None, bound=None)
    k, ub = best
    return dict(cutoff=float(conf[k - 1]), coverage=k / n, emp_error=float(cum_err[k - 1] / k), bound=float(ub))

def certify_sgr(conf, correct, budget, delta=0.05):
    """Original SGR (Geifman & El-Yaniv 2017) binary search, kept for comparison. Path dependent: can report
    'nothing certifiable' when a feasible coverage exists (see s4_tool_predictor/RESULTS.md, Revision)."""
    conf, correct = np.asarray(conf), np.asarray(correct, bool)
    order = np.argsort(-conf); conf, correct = conf[order], correct[order]
    n = len(conf); m = int(np.ceil(np.log2(n))) + 1; d = delta / m
    cum_err = np.cumsum(~correct)
    lo, hi, best = 1, n, None
    for _ in range(m):
        if lo > hi:
            break
        k = (lo + hi) // 2
        ub = cp_upper(int(cum_err[k - 1]), k, d)
        if ub <= budget:
            best = k; lo = k + 1
        else:
            hi = k - 1
    if best is None:
        return dict(cutoff=float("inf"), coverage=0.0, emp_error=None, bound=None)
    return dict(cutoff=float(conf[best - 1]), coverage=best / n, emp_error=float(cum_err[best - 1] / best),
                bound=float(cp_upper(int(cum_err[best - 1]), best, d)))

def apply_certificate(conf, correct, cutoff):
    """Realized coverage and error on a fresh holdout above the certified cutoff."""
    conf, correct = np.asarray(conf), np.asarray(correct, bool)
    above = conf >= cutoff
    return dict(coverage=float(above.mean()), error=float((~correct[above]).mean()) if above.any() else None, n_above=int(above.sum()))

def stratified_subsample(labels, n, seed=0):
    """About n rows with at least one per class when n >= #classes."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels); idx = rng.permutation(len(labels))
    classes = np.unique(labels)
    first = [int(idx[labels[idx] == c][0]) for c in classes] if n >= len(classes) else []
    fs = set(first)
    rest = [int(i) for i in idx if int(i) not in fs]
    return np.array(first + rest[: max(0, n - len(first))])

def write_results(path, title, lines):
    Path(path).write_text(f"# {title}\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"[wrote] {path}")

if __name__ == "__main__":  # self-check of certificate math
    rng = np.random.default_rng(0)
    conf = rng.random(2000); correct = rng.random(2000) < 0.5 + 0.5 * conf  # error falls with confidence
    c = certify(conf, correct, budget=0.05)
    assert c["coverage"] > 0 and c["emp_error"] <= 0.05 and c["bound"] <= 0.05, c
    h = apply_certificate(conf, correct, c["cutoff"])
    assert abs(h["coverage"] - c["coverage"]) < 1e-9, (h, c)
    # S4's minimal path-dependence case: top 480 correct, 5 errors at 480-484, rest correct, budget 2%
    conf2 = np.linspace(1, 0, 1000); ok2 = np.ones(1000, bool); ok2[480:485] = False
    assert certify_sgr(conf2, ok2, 0.02)["coverage"] == 0.0, "sgr should fail here (documented path dependence)"
    assert certify(conf2, ok2, 0.02)["coverage"] == 1.0, certify(conf2, ok2, 0.02)
    print("certify self-check ok", c, "| grid fixes the SGR path-dependence case")
