"""S24 arms 1 and 2: memory students (kNN vote, Hopfield read-out, with / without LLM-written rows) and Bayesian few-shot heads (GP, Laplace last layer, temperature-scaled LR).
Every function takes the k support rows and returns class scores for [calibration rows ; test rows] stacked, plus the confidence it defines."""
import time, warnings
import numpy as np
from lib import *

warnings.filterwarnings("ignore")
GPJOBS = 1 if TINY else 4

def softmax(z, axis=-1):
    z = z - z.max(axis, keepdims=True); e = np.exp(z); return e / e.sum(axis, keepdims=True)
def gap(P):
    s = np.sort(P, 1); return s[:, -1] - s[:, -2]

# ------------------------------------------------------------------ baseline
def lr_fit(X, y, C=10):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, max_iter=1000).fit(X, y)

# ------------------------------------------------------------------ 1a kNN vote (no training): confidence = vote margin
def knn(M, my, Xq, K, kn):
    """Cosine kNN (rows are unit norm). Returns (pred, conf): pred = most votes (ties broken by summed similarity), conf = (top1 - top2 votes) / kn."""
    S = Xq @ M.T; kn = min(kn, len(M)); top = np.argsort(-S, 1)[:, :kn]; n = len(Xq); r = np.arange(n)
    votes = np.zeros((n, K)); sim = np.zeros((n, K))
    for j in range(kn):
        votes[r, my[top[:, j]]] += 1; sim[r, my[top[:, j]]] += S[r, top[:, j]]
    return (votes + 1e-6 * sim).argmax(1), gap(votes / kn)

# ------------------------------------------------------------------ 1b modern-Hopfield / attention read-out: softmax(beta cos) weighted label vote; beta on the calibration set
BETAS = (5, 10, 20, 40, 80, 160, 320, 640)
def hopfield(M, my, Xq, K, beta):
    return softmax(beta * (Xq @ M.T)) @ np.eye(K)[my]
def hopfield_fit_beta(M, my, Xc, yc, K):
    nll = [-np.log(hopfield(M, my, Xc, K, b)[np.arange(len(yc)), yc] + 1e-9).mean() for b in BETAS]; return BETAS[int(np.argmin(nll))]

# ------------------------------------------------------------------ 2a Gaussian-process classifier on the frozen embeddings (RBF, one-vs-rest; Laplace-approximated GP posterior, hyper-parameters by marginal likelihood)
def gp_fit_predict(X, y, Xq):
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck
    t = time.perf_counter(); gp = GaussianProcessClassifier(kernel=Ck(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-1, 1e1)), multi_class="one_vs_rest", n_jobs=GPJOBS, random_state=0).fit(X, y)
    fit = time.perf_counter() - t; t = time.perf_counter(); P = gp.predict_proba(Xq); return P, fit, time.perf_counter() - t

def gp_fixed(X, y, Xc, yc, XA):
    """2a': same GP classifier with a fixed RBF kernel (amplitude 3), length-scale picked on the calibration set (marginal-likelihood fits degenerate with one positive row per one-vs-rest problem)."""
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck
    t = time.perf_counter(); best = None
    for ls in (0.3, 0.5, 0.8, 1.2):
        gp = GaussianProcessClassifier(kernel=Ck(3.0, "fixed") * RBF(ls, "fixed"), optimizer=None, multi_class="one_vs_rest", n_jobs=GPJOBS).fit(X, y)
        nll = -np.log(gp.predict_proba(Xc)[np.arange(len(yc)), yc] + 1e-9).mean()
        if best is None or nll < best[0]: best = (nll, ls, gp)
    fit = time.perf_counter() - t; t = time.perf_counter(); P = best[2].predict_proba(XA); return P, fit, time.perf_counter() - t, best[1]

# ------------------------------------------------------------------ 2b Laplace-approximated last layer on top of the LR MAP (block-diagonal GGN, one (d+1)x(d+1) block per class; prior precision by Laplace evidence)
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
def laplace_fit(clf, X, y):
    """Posterior over each class's weight vector (incl. bias) = N(MAP, (X^T diag(p_c (1-p_c)) X + alpha I)^-1); alpha maximises the Laplace evidence. Returns state for laplace_predict."""
    n, K = len(X), len(clf.classes_); Xa = np.hstack([X, np.ones((n, 1))]).astype(np.float64); W = np.hstack([clf.coef_, clf.intercept_[:, None]]).astype(np.float64)
    P = softmax(Xa @ W.T); nll = -np.log(P[np.arange(n), y] + 1e-12).sum(); eig = []
    for c in range(K):
        G = (Xa * (P[:, c] * (1 - P[:, c]))[:, None]).T @ Xa; lam, Q = np.linalg.eigh(G); eig.append((np.clip(lam, 0, None), Q))
    dim = Xa.shape[1]; w2 = (W ** 2).sum()
    ev = [-nll - 0.5 * a * w2 + 0.5 * K * dim * np.log(a) - 0.5 * sum(np.log(lam + a).sum() for lam, _ in eig) for a in ALPHAS]
    return dict(W=W, eig=eig, alpha=ALPHAS[int(np.argmax(ev))])
def laplace_predict(st, Xq):
    """Predictive: logit_c / sqrt(1 + pi/8 var_c) (probit approximation), var_c = x^T Sigma_c x."""
    Xqa = np.hstack([Xq, np.ones((len(Xq), 1))]).astype(np.float64); Z = Xqa @ st["W"].T
    var = np.stack([(((Xqa @ Q) ** 2) / (lam + st["alpha"])).sum(1) for lam, Q in st["eig"]], 1)
    return softmax(Z / np.sqrt(1 + np.pi / 8 * var))

# laplace-torch (optional: installed on the pod by run.py in a background thread, 5 minute cap)
def laplace_torch(clf, X, y, Xq):
    """Returns (probs, prior precision, fit seconds, predict seconds)."""
    import torch; from torch.utils.data import DataLoader, TensorDataset; from laplace import Laplace
    t = time.perf_counter(); K = len(clf.classes_); lin = torch.nn.Linear(X.shape[1], K)
    with torch.no_grad(): lin.weight.copy_(torch.tensor(clf.coef_, dtype=torch.float32)); lin.bias.copy_(torch.tensor(clf.intercept_, dtype=torch.float32))
    model = torch.nn.Sequential(torch.nn.Identity(), lin).eval()
    la = Laplace(model, "classification", subset_of_weights="last_layer", hessian_structure="kron")
    la.fit(DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y)), batch_size=len(X)))
    la.optimize_prior_precision(method="marglik"); fit = time.perf_counter() - t; t = time.perf_counter()
    P = la(torch.tensor(Xq, dtype=torch.float32), pred_type="glm", link_approx="probit").detach().cpu().numpy(); return P, float(la.prior_precision.mean()), fit, time.perf_counter() - t

# ------------------------------------------------------------------ 2c LR + temperature fitted on the calibration set (the spec's fallback; also answers "is any gain just calibration?")
TEMPS = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0)
def temperature(Zc, yc):
    nll = [-np.log(softmax(Zc / T)[np.arange(len(yc)), yc] + 1e-9).mean() for T in TEMPS]; return TEMPS[int(np.argmin(nll))]
