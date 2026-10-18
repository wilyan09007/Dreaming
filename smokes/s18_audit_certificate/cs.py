"""S18 helpers: anytime-valid upper/lower bounds on the realized error rate of served traffic from an audit stream.
Setting: served call t has error e_t in {0,1} (fixed by the input). It is audited with known probability p (independent coin A_t);
Y_t = A_t * e_t. Everything is vectorised over S independent audit-coin sequences: Y has shape (S, T). Bounds are on the error RATE.

  wsr()      predictable-plug-in betting CS of Waudby-Smith & Ramdas (JRSSB 2024, arXiv 2010.09686), PrPl variant, one-sided, on Y
             (mean p*error). confseq would not build here (sdist needs CMake + C++), so this is my ~25-line implementation of eq. for
             lambda_t = min(c, sqrt(2 log(1/a) / (sigma2_{t-1} t log(1+t)))), K_t(m) = prod(1 - lambda_i (Y_i - m)). Formally valid for a
             CONSTANT conditional mean (stationary stream).
  chernoff() design-based e-process (my construction, Chernoff/Poisson-binomial form of the same idea): for any lambda>0,
             exp(-lambda*sum Y + (1-e^-lambda) * p * sum e) is a nonnegative supermartingale for ANY error sequence (drifting, clustered,
             adaptive) because only the audit coins are random. Union over a lambda grid + Ville => time-uniform bound on the realized
             running error. Lower bound from the mirror image exp(+lambda sum Y - (e^lambda - 1) p sum e).
"""
import numpy as np
from scipy.stats import beta, binom

LAM = np.array([0.05, 0.15, 0.4, 0.8, 1.5, 2.5, 4., 6., 9., 13.])


def chernoff(Y, p, delta=0.05, g=None):
    """Returns (U, L): running upper/lower bound on error rate at every t, shape (S,T). With g (group id per call, shape (T,)) the bound is the
    sum over groups of per-group bounds (each at delta/G; U also capped by the group's call count); also returns per-group lower bounds Lg (S,T,G)."""
    S, T = Y.shape; g = np.zeros(T, int) if g is None else np.asarray(g); G = int(g.max()) + 1
    a = np.log(len(LAM) * G / delta); t = np.arange(1, T + 1)
    Ub = np.zeros((S, T)); Lg = np.zeros((S, T, G))
    for j in range(G):
        mask = (g == j); k = np.cumsum(Y * mask, 1); n = np.cumsum(mask)
        B = np.min([(l * k + a) / (p * (1 - np.exp(-l))) for l in LAM], 0)
        Ub += np.minimum(B, n)
        LB = np.max([(l * k - a) / (p * np.expm1(l)) for l in LAM], 0).clip(0)
        Lg[:, :, j] = LB / np.maximum(n, 1)
    U = np.minimum(Ub / t, 1.)
    if G == 1:
        L = Lg[:, :, 0]
    else:  # pooled lower bound from the group lower bounds
        L = sum(Lg[:, :, j] * np.cumsum(g == j) for j in range(G)) / t
    return U, L, Lg


def wsr(Y, p, delta=0.05, g=None, grid=110, c=0.75):
    """One-sided betting upper CS, grid on the error-rate scale (geometric 5e-4..1, 7% steps; bound rounded UP to the next grid point). With g: one CS per
    group (delta/G each), combined as sum_g n_g(t) U_g(t) / t. Returns U (S,T)."""
    S, T = Y.shape; g = np.zeros(T, int) if g is None else np.asarray(g); G = int(g.max()) + 1
    thr = np.log(G / delta); gr = np.geomspace(5e-4, 1., grid); m = p * gr
    logK = np.zeros((G, S, grid)); cs = np.zeros((G, S)); ss = np.zeros((G, S)); cnt = np.zeros(G); Ug = np.ones((G, S)); out = np.empty((S, T))
    for t in range(T):
        j = g[t]; cnt[j] += 1; n = cnt[j]; y = Y[:, t]
        lam = np.minimum(c, np.sqrt(2 * thr / ((.25 + ss[j]) / n * n * np.log1p(n))))   # sigma2_{n-1} = (1/4 + sum)/n
        logK[j] += np.log1p(-lam[:, None] * (y[:, None] - m[None]))
        cs[j] += y; mu = (.5 + cs[j]) / (n + 1); ss[j] += (y - mu) ** 2
        rej = logK[j] >= thr; any_ = rej.any(1)
        Ug[j] = np.where(any_, gr[rej.argmax(1)], 1.)
        out[:, t] = (cnt[:, None] * Ug).sum(0) / (t + 1)
    return out


def naive(Y, A):
    """Running empirical error on the audit stream (point estimate, 1.0 before the first audit)."""
    a = np.cumsum(A, 1); k = np.cumsum(Y, 1)
    return np.where(a > 0, k / np.maximum(a, 1), 1.)


def cp_peek(Y, A, delta=0.05, every=20):
    """Fixed-sample Clopper-Pearson bounds on the audit stream recomputed at every `every`-th call (continuous peeking, NOT valid). Returns (U upper, idx checkpoints, L lower), both one-sided at delta."""
    idx = np.arange(every - 1, Y.shape[1], every); a = np.cumsum(A, 1)[:, idx]; k = np.cumsum(Y, 1)[:, idx]
    U = np.where((a == 0) | (k >= a), 1., beta.ppf(1 - delta, k + 1, np.maximum(a - k, 1)))
    L = np.where(k == 0, 0., beta.ppf(delta, np.maximum(k, 1), np.maximum(a - k, 0) + 1))
    return U, idx, L


def ltt_fst(conf, ok, budget, delta=0.05, steps=60, seed=0):
    """Learn-then-Test, fixed-sequence testing (Angelopoulos et al.) with data splitting. Calibration rows are split in halves A/B. Candidate cutoffs are
    the confidence quantiles for coverage 2%..100% (labels unused). Half A picks where the sequence starts (largest coverage whose empirical error on A is
    <= budget/2); half B walks UP in coverage from there, testing each cutoff at the full delta with the exact binomial test of 'selective risk > budget',
    and stops at the first non-rejection. No union bound, but half the rows test and a bad start certifies nothing."""
    conf, ok = np.asarray(conf), np.asarray(ok, bool); n = len(conf)
    perm = np.random.default_rng(seed).permutation(n); A, B = perm[:n // 2], perm[n // 2:]
    nmin = int(np.ceil(np.log(delta) / np.log(1 - budget)))
    cuts = np.quantile(conf, 1 - np.linspace(.02, 1., steps))
    def cnt(ix, c): acc = conf[ix] >= c; return int(acc.sum()), int((~ok[ix][acc]).sum())
    j0 = 0
    for j, c in enumerate(cuts):
        na, ka = cnt(A, c)
        if na and ka / na <= budget / 2: j0 = j
    best = None
    for c in cuts[j0:]:
        nb, kb = cnt(B, c)
        if nb < nmin: continue
        if binom.cdf(kb, nb, budget) > delta: break
        best = float(c)
    if best is None: return dict(cutoff=float("inf"), coverage=0., emp_error=None)
    acc = conf >= best
    return dict(cutoff=best, coverage=float(acc.mean()), emp_error=float((~ok[acc]).mean()))


def first(cond):
    """First index (1-based call count) where cond[s, :] is true, NaN if never. cond: (S,T) bool."""
    return np.where(cond.any(1), cond.argmax(1) + 1., np.nan)


if __name__ == "__main__":  # self-check: coverage of the CS on a stationary stream and on an adversarial (all errors early) stream, + a hand-checkable case
    rng = np.random.default_rng(0); S, T, p = 300, 4000, .1
    t = np.arange(1, T + 1)
    for name, e in (("iid 1% errors", rng.random(T) < .01), ("clustered: 40 errors in the first 60 calls", np.r_[np.ones(40), np.zeros(20), np.zeros(T - 60)].astype(bool))):
        Y = ((rng.random((S, T)) < p) & e).astype(float)
        run_e = np.cumsum(e) / t; Uc, Lc, _ = chernoff(Y, p); Uw = wsr(Y, p)
        vc = (Uc < run_e - 1e-12).any(1).mean(); vw = (Uw < run_e - 1e-12).any(1).mean(); vl = (Lc > run_e + 1e-12).any(1).mean()
        print(f"{name}: P(ever U<realized) chernoff {vc:.3f} wsr {vw:.3f}; P(ever L>realized) chernoff {vl:.3f}; final U chernoff {np.median(Uc[:, -1]):.4f} wsr {np.median(Uw[:, -1]):.4f}, realized {run_e[-1]:.4f}")
        assert vc <= .06 and vl <= .06, "chernoff must be valid for any error sequence"
        if name.startswith("iid"): assert vw <= .06
    # zero errors: chernoff bound is ~ ln(J/delta)/(p t); rule of three would be 3/(p t)
    U0, _, _ = chernoff(np.zeros((1, 1000)), .1); print("zero errors, t=1000: chernoff", U0[0, -1], "vs rule-of-three 3/(p t) =", 3 / (.1 * 1000))
    # LTT picks something sensible on a toy: error falls with confidence
    cf = rng.random(3000); ok = rng.random(3000) < .5 + .5 * cf; print("ltt toy", ltt_fst(cf, ok, .05))
    print("cs self-check ok")
