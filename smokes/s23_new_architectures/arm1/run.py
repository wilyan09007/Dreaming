"""S23 arms 1 + 5: semi-supervised probes on the unlabelled pool (TK-KNN-style, Sinkhorn-balanced, label spreading) and k-means-medoid selection
of the k labelled rows, all on frozen bge-small embeddings + LogisticRegression. CPU only. Usage: python run.py [--tiny]"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import *

OUT = Path(__file__).resolve().parent / "rows.jsonl"

def tkknn(XL, yL, XU, yU, K, R=5):
    """Top-k class-balanced self-training: pseudo-label the pool with the current LR, keep the top-q most confident per class (q grows to
    half an expected class), retrain on labelled + selected, R rounds. Returns final classifier and pseudo-label accuracy per round."""
    per, sel, accs = len(XU) / K, None, []
    for r in range(R + 1):
        X, y = (XL, yL) if sel is None else (np.r_[XL, XU[sel[0]]], np.r_[yL, sel[1]])
        clf = lr_fit(X, y)
        if r == R: break
        P = full_proba(clf, XU, K); pred = P.argmax(1); q = max(1, int(per * 0.5 * (r + 1) / R)); idx = []
        for c in range(K):
            m = np.flatnonzero(pred == c); idx += list(m[np.argsort(-P[m, c])[:q]])
        idx = np.array(idx, int); sel = (idx, pred[idx]); accs.append(float((pred[idx] == yU[idx]).mean()) if len(idx) else None)
    return clf, accs

def sinkhorn(P, r, it=50):
    Q = P + 1e-9
    for _ in range(it):
        Q = Q * (r / Q.mean(0)); Q = Q / Q.sum(1, keepdims=True)
    return Q

def sink_st(XL, yL, XU, yU, K, est, T=3, alpha=0.5):
    """SS-Text-U-style block coordinate loop: LR scores -> Sinkhorn (marginals uniform, or the current mean prediction when est) -> hard pseudo-labels
    on the whole pool (weight alpha) -> refit LR. accs[0] = plain LR pool accuracy, accs[t] = pseudo-label accuracy after balancing in round t."""
    clf = lr_fit(XL, yL); P = full_proba(clf, XU, K); accs = [float((P.argmax(1) == yU).mean())]
    for _ in range(T):
        r = np.maximum(P.mean(0), 1e-6) if est else np.full(K, 1.0); Q = sinkhorn(P, r / r.sum()); pl = Q.argmax(1); accs.append(float((pl == yU).mean()))
        clf = lr_fit(np.r_[XL, XU], np.r_[yL, pl], np.r_[np.ones(len(XL)), np.full(len(XU), alpha)]); P = full_proba(clf, XU, K)
    return clf, accs

def spread(XL, yL, XU, yU, K, Xq):
    """LabelSpreading on a cosine kNN graph over labelled + pool (test rows never enter the graph). Returns test proba from kNN smoothing,
    and an LR refit on labelled + the pool rows the graph reached (weight 0.5)."""
    from sklearn.semi_supervised import LabelSpreading
    ls = LabelSpreading(kernel="knn", n_neighbors=10, alpha=0.2, max_iter=50).fit(np.r_[XL, XU], np.r_[yL, -np.ones(len(XU), int)])
    D = np.zeros((len(XL) + len(XU), K)); D[:, ls.classes_] = ls.label_distributions_; D = D[len(XL):]
    m = D.sum(1) > 0; pl = D.argmax(1)
    Pq = np.zeros((len(Xq), K)); Pq[:, ls.classes_] = np.nan_to_num(ls.predict_proba(Xq))
    clf = lr_fit(np.r_[XL, XU[m]], np.r_[yL, pl[m]], np.r_[np.ones(len(XL)), np.full(m.sum(), 0.5)])
    return Pq, clf, [float((pl[m] == yU[m]).mean()) if m.any() else None, float(m.mean())]

def medoids(X, n, seed):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n, n_init=1, random_state=seed).fit(X); out = []
    for c in range(n):
        mem = np.flatnonzero(km.labels_ == c)
        if len(mem): out.append(mem[np.argmin(((X[mem] - km.cluster_centers_[c]) ** 2).sum(1))])
    return np.array(out, int)

def run_task(t, ks, draws):
    from threadpoolctl import threadpool_limits
    rows = []; K = len(t["labels"]); Xs, ys, Xq, yq = t["Xs"], t["ys"], t["Xq"], t["yq"]
    base = dict(arm="1+5", task=t["name"], family=t["family"], K=K)
    with threadpool_limits(1):
        clf = lr_fit(Xs, ys); rows.append(dict(base, method="LR on all pool labels (ceiling)", sel="all", k=-1, draw=0, **score(full_proba(clf, Xq, K), yq)))
        for dr in range(draws):
            for k in ks:
                for sel in ("random", "medoid"):
                    t0 = time.time(); L = draw(ys, k, dr, K) if sel == "random" else medoids(Xs, k * K, dr)
                    U = np.setdiff1d(np.arange(len(ys)), L); XL, yL, XU, yU = Xs[L], ys[L], Xs[U], ys[U]; cov_cls = len(np.unique(yL)) / K
                    def put(method, P, extra=None):
                        rows.append(dict(base, method=method, sel=sel, k=k, draw=dr, classes_covered=cov_cls, pl=extra, **score(P, yq)))
                    put("LR baseline", full_proba(lr_fit(XL, yL), Xq, K))
                    c, a = tkknn(XL, yL, XU, yU, K); put("TK-KNN self-training", full_proba(c, Xq, K), a)
                    c, a = sink_st(XL, yL, XU, yU, K, est=False); put("Sinkhorn (uniform prior)", full_proba(c, Xq, K), a)
                    c, a = sink_st(XL, yL, XU, yU, K, est=True); put("Sinkhorn (estimated prior)", full_proba(c, Xq, K), a)
                    Pq, c, a = spread(XL, yL, XU, yU, K, Xq); put("Label spreading (kNN smoothing)", Pq, a); put("Label spreading -> LR", full_proba(c, Xq, K), a)
                    print(f"{t['name']} draw{dr} k{k} {sel} {time.time()-t0:.1f}s", flush=True)
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tiny", action="store_true"); a = ap.parse_args()
    ks, draws = ([1, 3], 1) if a.tiny else (KS, DRAWS)
    if OUT.exists(): OUT.unlink()
    T = tasks(tiny=a.tiny); log("tasks", [(t["name"], len(t["ts"]), len(t["tq"])) for t in T])
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=min(len(T), os.cpu_count() or 1))(delayed(run_task)(t, ks, draws) for t in T)
    for rows in res:
        for r in rows: emit(OUT, **r)
    log("wrote", OUT, sum(len(r) for r in res), "rows")
