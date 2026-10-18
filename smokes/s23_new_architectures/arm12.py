"""S23-lite arms 1 + 2, CPU only, frozen bge-small embeddings + LogisticRegression.
 arm 1: baseline LR on the k random rows; free variant: k*K rows chosen as k-means medoids of the unlabelled pool (labels revealed afterwards).
 arm 2: semi-supervised probes on the unlabelled pool (all train rows minus calibration minus the k labelled rows, labels hidden):
        TK-KNN-style top-k class-balanced self-training, Sinkhorn-balanced pseudo-labels + LR (uniform and estimated class prior), LabelSpreading on a cosine kNN graph.
        Two pool sizes: the full pool and a random 24 rows/label subset. Pseudo-label accuracy per round is reported (pl).
Usage: python arm12.py [--tiny] [--deadline-min N]"""
import argparse
from lib import *

OUT = HERE / "rows_arm12.jsonl"; R_TK, T_SK = 4, 3

def tkknn(XL, yL, XU, yU, K):
    """Top-k class-balanced self-training: pseudo-label the pool with the current LR, keep the q most confident per predicted class (q grows to half an
    expected class over R_TK rounds), refit on labelled + selected. Returns final classifier and pseudo-label accuracy of the selection in each round."""
    per, sel, accs = len(XU) / K, None, []
    for r in range(R_TK + 1):
        X, y = (XL, yL) if sel is None else (np.r_[XL, XU[sel[0]]], np.r_[yL, sel[1]])
        clf = lr_fit(X, y)
        if r == R_TK: break
        P = full_proba(clf, XU, K); pred = P.argmax(1); q = max(1, int(per * 0.5 * (r + 1) / R_TK)); idx = []
        for c in range(K):
            m = np.flatnonzero(pred == c); idx += list(m[np.argsort(-P[m, c])[:q]])
        idx = np.array(idx, int); sel = (idx, pred[idx]); accs.append(float((pred[idx] == yU[idx]).mean()) if len(idx) else None)
    return clf, accs

def sinkhorn(P, r, it=50):
    Q = P + 1e-9
    for _ in range(it):
        Q = Q * (r / Q.mean(0)); Q = Q / Q.sum(1, keepdims=True)
    return Q

def sink_st(XL, yL, XU, yU, K, est, alpha=0.5):
    """SS-Text-U-style block-coordinate loop: LR scores -> Sinkhorn (class marginals uniform, or the current mean prediction when est) -> hard pseudo-labels
    on the whole pool (weight alpha) -> refit LR. accs[0] = plain LR pool accuracy (round 0), accs[t] = balanced pseudo-label accuracy in round t."""
    clf = lr_fit(XL, yL); P = full_proba(clf, XU, K); accs = [float((P.argmax(1) == yU).mean())]
    for _ in range(T_SK):
        r = np.maximum(P.mean(0), 1e-6) if est else np.ones(K); Q = sinkhorn(P, r / r.sum()); pl = Q.argmax(1); accs.append(float((pl == yU).mean()))
        clf = lr_fit(np.r_[XL, XU], np.r_[yL, pl], np.r_[np.ones(len(XL)), np.full(len(XU), alpha)]); P = full_proba(clf, XU, K)
    return clf, accs

def spread(XL, yL, XU, yU, K, Xq, Xc):
    """LabelSpreading on a kNN graph over labelled + pool (unit vectors, so Euclidean kNN = cosine kNN; test/calibration rows are never in the graph).
    Returns test/calibration probabilities from the spread labels, an LR refit on labelled + reached pool rows (weight 0.5), and [pseudo-label acc, fraction reached]."""
    from sklearn.semi_supervised import LabelSpreading
    ls = LabelSpreading(kernel="knn", n_neighbors=10, alpha=0.2, max_iter=50).fit(np.r_[XL, XU], np.r_[yL, -np.ones(len(XU), int)])
    D = np.zeros((len(XL) + len(XU), K)); D[:, ls.classes_] = ls.label_distributions_; D = D[len(XL):]; m = D.sum(1) > 0; pl = D.argmax(1)
    Pq, Pc = np.zeros((len(Xq), K)), np.zeros((len(Xc), K)); Pq[:, ls.classes_] = np.nan_to_num(ls.predict_proba(Xq)); Pc[:, ls.classes_] = np.nan_to_num(ls.predict_proba(Xc))
    clf = lr_fit(np.r_[XL, XU[m]], np.r_[yL, pl[m]], np.r_[np.ones(len(XL)), np.full(m.sum(), 0.5)])
    return Pq, Pc, clf, [float((pl[m] == yU[m]).mean()) if m.any() else None, float(m.mean())]

def medoids(X, n, seed):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n, n_init=1, random_state=seed).fit(X); out = []
    for c in range(n):
        mem = np.flatnonzero(km.labels_ == c)
        if len(mem): out.append(mem[np.argmin(((X[mem] - km.cluster_centers_[c]) ** 2).sum(1))])
    return np.array(out, int)

def run_unit(ti, dr, ks, deadline, pool_per):
    """one (task, draw): all k, all methods. Rows are appended to OUT as soon as they exist."""
    from threadpoolctl import threadpool_limits
    t = load_tasks()[ti]; K = len(t["labels"]); Xs, ys, Xc, yc, Xq, yq = t["Xs"], t["ys"], t["Xc"], t["yc"], t["Xq"], t["yq"]; NP = (Xs.shape[1] + 1) * K
    base = dict(task=t["name"], family=t["family"], K=K, draw=dr, params=NP)
    with threadpool_limits(1):
        if dr == 0:
            t0 = time.time(); clf = lr_fit(Xs, ys)
            emit(OUT, arm="1", method="LR on all pool labels (ceiling)", sel="all", k=-1, pool="-", secs=time.time() - t0, **base, **score(full_proba(clf, Xq, K), yq, full_proba(clf, Xc, K), yc))
        for k in ks:
            if time.time() > deadline: log("deadline"); return
            t1 = time.time()
            def put(arm, method, sel, pool, P, Pc_, secs, pl=None, **kw):
                emit(OUT, arm=arm, method=method, sel=sel, k=k, pool=pool, secs=secs, pl=pl, **base, **kw, **score(P, yq, Pc_, yc))
            # arm 1: random rows, and k-means medoid rows
            L = draw(ys, k, dr, K); t0 = time.time(); clf = lr_fit(Xs[L], ys[L])
            put("1", "LR baseline (k random rows)", "random", "-", full_proba(clf, Xq, K), full_proba(clf, Xc, K), time.time() - t0, classes_covered=1.0)
            M = medoids(Xs, k * K, dr); t0 = time.time(); clf = lr_fit(Xs[M], ys[M])
            put("1", "LR on k*K k-means medoid rows", "medoid", "-", full_proba(clf, Xq, K), full_proba(clf, Xc, K), time.time() - t0, classes_covered=len(np.unique(ys[M])) / K)
            # arm 2: pseudo-labelling on the pool (random labelled rows only)
            U = np.setdiff1d(np.arange(len(ys)), L); XL, yL = Xs[L], ys[L]
            for pool in ("full", f"{pool_per}/label"):
                Ui = U if pool == "full" else np.sort(np.random.default_rng(5000 + dr).choice(U, min(len(U), pool_per * K), replace=False)); XU, yU = Xs[Ui], ys[Ui]
                t0 = time.time(); c, a = tkknn(XL, yL, XU, yU, K); put("2", "TK-KNN-style self-training", "random", pool, full_proba(c, Xq, K), full_proba(c, Xc, K), time.time() - t0, pl=a)
                for est in (False, True):
                    t0 = time.time(); c, a = sink_st(XL, yL, XU, yU, K, est); put("2", f"Sinkhorn pseudo-labels + LR ({'estimated' if est else 'uniform'} prior)", "random", pool, full_proba(c, Xq, K), full_proba(c, Xc, K), time.time() - t0, pl=a)
                t0 = time.time(); Pq, Pc_, c, a = spread(XL, yL, XU, yU, K, Xq, Xc); secs = time.time() - t0
                put("2", "LabelSpreading (kNN graph, direct)", "random", pool, Pq, Pc_, secs, pl=a)
                put("2", "LabelSpreading -> LR on reached rows", "random", pool, full_proba(c, Xq, K), full_proba(c, Xc, K), secs, pl=a)
            log(f"{t['name']} draw{dr} k{k} {time.time()-t1:.1f}s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tiny", action="store_true"); ap.add_argument("--deadline-min", type=float, default=30); a = ap.parse_args()
    if hasattr(os, "nice"): os.nice(10)  # CPU arms yield to the GPU-feeding processes (workers inherit this)
    deadline = time.time() + a.deadline_min * 60; ks, draws, pool_per = ([1, 3], 1, 8) if a.tiny else (KS, DRAWS, 24)
    if OUT.exists(): OUT.unlink()
    n_tasks = len(load_tasks()); units = [(ti, dr) for dr in range(draws) for ti in range(n_tasks)]  # draw-major so a timeout leaves whole draws
    t0 = time.time()
    if a.tiny: [run_unit(ti, dr, ks, deadline, pool_per) for ti, dr in units]
    else:
        from joblib import Parallel, delayed
        Parallel(n_jobs=min(4, max(1, (os.cpu_count() or 2) - 1)))(delayed(run_unit)(ti, dr, ks, deadline, pool_per) for ti, dr in units)
    log("arm12 done", f"{time.time()-t0:.0f}s")
