"""S24 arm 3: a small prior-fitted network (PFN) for embedding space. A 4-layer transformer reads (k labelled embeddings, query embedding) and outputs the query's label distribution;
meta-trained on episodes from (a) synthetic Gaussian-mixture tasks in 384-d (shared low-rank structure, random per-class covariance, support label noise) and (b) the pretraining pool.
At evaluation it is applied in-context to unseen tasks with no gradient step. Class identity is a random slot id per episode (so the net must read the labels, not memorise them)."""
import math, time, copy
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from lib import *

CMAX = 16
DEV = "cuda" if torch.cuda.is_available() else "cpu"

class PFN(nn.Module):
    def __init__(s, d=D, h=256, layers=4, heads=8, ff=1024):
        super().__init__()
        s.inp = nn.Linear(d, h); s.lab = nn.Embedding(CMAX, h); s.qtok = nn.Parameter(torch.zeros(h))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(h, heads, ff, dropout=0.0, activation="gelu", batch_first=True, norm_first=True), layers, enable_nested_tensor=False)
        s.norm = nn.LayerNorm(h); s.out = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, CMAX))
    def forward(s, Xs, ys, Xq, K):
        """Xs (B,S,d) support, ys (B,S) support slot ids, Xq (B,Q,d) queries -> (B,Q,CMAX) logits over the first K slots. Supports see supports; each query sees all supports and itself."""
        S, Q = Xs.shape[1], Xq.shape[1]; z = torch.cat([s.inp(Xs) + s.lab(ys), s.inp(Xq) + s.qtok], 1)
        m = torch.zeros(S + Q, S + Q, dtype=torch.bool, device=Xs.device); m[:, S:] = True; m[S:, S:] = ~torch.eye(Q, dtype=torch.bool, device=Xs.device)
        lg = s.out(s.norm(s.enc(z, mask=m)[:, S:]))
        return lg.masked_fill(torch.arange(CMAX, device=Xs.device) >= K, float("-inf"))

# ------------------------------------------------------------------ episode sources
def synth_cls(B, K, n, d=D):
    """(B,K,n,d): B random Gaussian-mixture tasks. Class centres and noise share one random R-dim subspace (hard, overlapping), per-class random covariance, standardised scale."""
    R = int(np.random.randint(6, 41)); U = torch.linalg.qr(torch.randn(B, d, R, device=DEV))[0]
    a = torch.randn(B, K, R, device=DEV) / math.sqrt(R)
    lu = lambda lo, hi, *sh: torch.exp(torch.empty(*sh, device=DEV).uniform_(math.log(lo), math.log(hi)))
    tau, g = lu(0.07, 0.8, B, 1, 1, 1), lu(0.7, 1.4, B, K, 1, 1); M = torch.randn(B, K, R, R, device=DEV) / math.sqrt(R)
    zin = a[:, :, None, :] + tau * g * torch.einsum("bkrs,bkns->bknr", M, torch.randn(B, K, n, R, device=DEV))
    x = torch.einsum("bdr,bknr->bknd", U, zin) + lu(0.02, 0.3, B, 1, 1, 1) * torch.randn(B, K, n, d, device=DEV) / math.sqrt(d)
    return x * torch.sqrt(d / x.pow(2).sum(-1).mean((1, 2))[:, None, None, None])  # mean squared norm = d, like the standardised pool

def build_sources(X, key, ds, grp):
    """X (N,d) standardised pool embeddings -> per-dataset padded row table on the device."""
    Xt = torch.tensor(X, dtype=torch.float32, device=DEV); src = {}
    for d_ in sorted(set(ds.tolist())):
        labs = sorted(set(key[ds == d_].tolist())); rows = [np.flatnonzero(key == l) for l in labs]; mr = max(len(r) for r in rows); rm = np.zeros((len(labs), mr), np.int64)
        for i, r in enumerate(rows): rm[i, :len(r)] = r
        gs = sorted({grp[l] for l in labs}) if d_ == "clinc" else None
        src[d_] = dict(X=Xt, rm=torch.tensor(rm, device=DEV), cnt=torch.tensor([len(r) for r in rows], device=DEV), nmin=min(len(r) for r in rows),
                       groups=[torch.tensor([i for i, l in enumerate(labs) if grp[l] == g_], device=DEV) for g_ in gs] if gs else None)
    return src

def pool_cls(s, B, K, n, within=0.6):
    """(B,Kr,n,d): B tasks of Kr<=K labels (clinc: within one domain with prob `within`), n distinct rows per label."""
    labs = torch.arange(len(s["cnt"]), device=DEV)
    if s["groups"] is not None and np.random.rand() < within:
        gs = [g_ for g_ in s["groups"] if len(g_) >= 2]; labs = gs[np.random.randint(len(gs))]
    Kr = min(K, len(labs)); lid = labs[torch.rand(B, len(labs), device=DEV).argsort(1)[:, :Kr]]  # (B,Kr)
    rm, c = s["rm"][lid], s["cnt"][lid]; r = torch.rand(rm.shape, device=DEV); r[torch.arange(rm.shape[-1], device=DEV) >= c[..., None]] = 2.0
    return s["X"][rm.gather(-1, r.argsort(-1)[..., :n])]

def assemble(Xc, k, noise=False):
    """(B,Kr,k+q,d) -> support / query tensors with random class->slot maps (slots 0..Kr-1)."""
    B, Kr, n, d = Xc.shape; q = n - k
    Xs, Xq = Xc[:, :, :k].reshape(B, Kr * k, d), Xc[:, :, k:].reshape(B, Kr * q, d)
    perm = torch.rand(B, Kr, device=DEV).argsort(1); cs = torch.arange(Kr, device=DEV).repeat_interleave(k)[None].expand(B, -1); cq = torch.arange(Kr, device=DEV).repeat_interleave(q)[None].expand(B, -1)
    ys, yq = perm.gather(1, cs), perm.gather(1, cq)
    if noise:  # label noise on the support labels of half the tasks, up to 20% of the rows
        eps = (torch.rand(B, 1, device=DEV) < 0.5) * torch.rand(B, 1, device=DEV) * 0.2
        ys = torch.where(torch.rand(ys.shape, device=DEV) < eps, torch.randint(0, Kr, ys.shape, device=DEV), ys)
    return Xs, ys, Xq, yq, Kr

# ------------------------------------------------------------------ evaluation of a trained net on a task, in context
@torch.no_grad()
def pfn_probs(model, mu, sd, Xs, ys, Xq, K, nperm=1, chunk=128, seed=0):
    """Class probabilities for Xq given k labelled rows (Xs, ys) averaged over `nperm` random class->slot assignments (the first uses the identity)."""
    model.eval(); r = np.random.default_rng(seed); out = np.zeros((len(Xq), K))
    Xs_t = torch.tensor((Xs - mu) / sd, dtype=torch.float32, device=DEV)[None]; Xq_t = torch.tensor((Xq - mu) / sd, dtype=torch.float32, device=DEV)
    for p in range(nperm):
        perm = np.arange(K) if p == 0 else r.permutation(K); slot = torch.tensor(perm[ys], device=DEV)[None]
        for i in range(0, len(Xq), chunk):
            pr = F.softmax(model(Xs_t, slot, Xq_t[i:i + chunk][None], K)[0][:, :K], -1).float().cpu().numpy(); out[i:i + chunk] += pr[:, perm]
    if DEV == "cuda": torch.cuda.synchronize()
    return out / nperm

def val_episodes(vX, vkey, seed=7):
    """Fixed validation tasks from the held-out CLINC "meta" domain: for k in (1,3,10) 8 episodes with all its labels; also the LR accuracy on the same episodes as a reference."""
    from sklearn.linear_model import LogisticRegression
    r = np.random.default_rng(seed); labs = sorted(set(vkey.tolist())); rows = {l: np.flatnonzero(vkey == l) for l in labs}; eps = []
    for k in (1, 3, 10):
        for _ in range(8):
            Xs, ys, Xq, yq = [], [], [], []
            for c, l in enumerate(labs):
                kk = min(k, max(len(rows[l]) - 3, 1)); p = r.permutation(rows[l]); Xs.append(vX[p[:kk]]); ys += [c] * kk; Xq.append(vX[p[kk:kk + 5]]); yq += [c] * len(p[kk:kk + 5])
            eps.append((k, np.concatenate(Xs), np.array(ys), np.concatenate(Xq), np.array(yq)))
    lr = {k: float(np.mean([(LogisticRegression(C=10, max_iter=1000).fit(Xs, ys).predict(Xq) == yq).mean() for kk, Xs, ys, Xq, yq in eps if kk == k])) for k in (1, 3, 10)}
    return eps, lr, len(labs)

def val_acc(model, eps, K, mu, sd):
    acc = {1: [], 3: [], 10: []}
    for k, Xs, ys, Xq, yq in eps: acc[k].append(float((pfn_probs(model, mu, sd, Xs, ys, Xq, K).argmax(1) == yq).mean()))
    model.train(); return {k: float(np.mean(v)) for k, v in acc.items()}

def synth_difficulty(n_tasks=40, K=10, k=3, q=20):
    """LR accuracy quantiles on freshly drawn synthetic tasks (sanity check that the synthetic prior spans easy to hard)."""
    from sklearn.linear_model import LogisticRegression
    Xc = synth_cls(n_tasks, K, k + q).cpu().numpy(); accs = []
    for b in range(n_tasks):
        Xs, Xq = Xc[b, :, :k].reshape(-1, Xc.shape[-1]), Xc[b, :, k:].reshape(-1, Xc.shape[-1]); ys, yq = np.repeat(np.arange(K), k), np.repeat(np.arange(K), q)
        accs.append(float((LogisticRegression(C=10, max_iter=300).fit(Xs, ys).predict(Xq) == yq).mean()))
    return [round(float(v), 3) for v in np.quantile(accs, [0.1, 0.25, 0.5, 0.75, 0.9])]

# ------------------------------------------------------------------ meta-training
def train_pfn(pool, val, seconds, B=48, lr=4e-4, log=print):
    X = pool["X"]; mu = X.mean(0); sd = float(np.sqrt(((X - mu) ** 2).mean()))  # standardisation constants (from the pool only)
    src = build_sources(((X - mu) / sd).astype(np.float32), pool["key"], pool["ds"], pool["grp"])
    veps, vlr, vK = val_episodes(val["X"], val["key"])  # raw embeddings; pfn_probs standardises
    torch.manual_seed(0); np.random.seed(0)
    model = PFN().to(DEV); npar = sum(p.numel() for p in model.parameters()); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    names = ["synth"] + sorted(src); w = {"synth": 0.35, "clinc": 0.25, "banking": 0.2, "trec": 0.08, "emotion": 0.06, "ag_news": 0.06}; pr = np.array([w.get(n, 0.05) for n in names]); pr /= pr.sum()
    kmax = 3 if TINY else 12; ktr = [k for k in (1, 2, 3, 4, 5, 6, 8, 10, 12) if k <= kmax]; q = 2 if TINY else 4
    sdiff = synth_difficulty(); log(f"[pfn] synthetic tasks (K=10,k=3): LR accuracy quantiles 10/25/50/75/90% = {sdiff}")
    log(f"[pfn] params {npar:,} on {DEV}; sources {names} p={np.round(pr, 2).tolist()}; val LR reference {vlr}; budget {seconds}s")
    t0 = time.time(); step = 0; hist = []; best = (-1.0, None, 0); ema = None; used = {n: 0 for n in names}
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    model.train()
    while True:
        el = time.time() - t0
        if el > seconds: break
        for g_ in opt.param_groups: g_["lr"] = lr * min(1.0, (step + 1) / 200) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * el / seconds)))
        nm = names[np.random.choice(len(names), p=pr)]; K = int(np.random.randint(2, 16)); k = int(np.random.choice(ktr))
        if nm != "synth": k = min(k, src[nm]["nmin"] - q)
        Xc = synth_cls(B, K, k + q) if nm == "synth" else pool_cls(src[nm], B, K, k + q)
        Xs, ys, Xq, yq, Kr = assemble(Xc, k, noise=(nm == "synth"))
        loss = F.cross_entropy(model(Xs, ys, Xq, Kr).reshape(-1, CMAX), yq.reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1; used[nm] += 1; ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % (20 if TINY else 500) == 0:
            va = val_acc(model, veps, vK, mu, sd); m = float(np.mean(list(va.values()))); hist.append(dict(step=step, sec=round(time.time() - t0), loss=round(ema, 4), val=va, val_mean=m))
            if m > best[0]: best = (m, copy.deepcopy(model.state_dict()), step)
            log(f"[pfn] step {step} {el:.0f}s loss {ema:.3f} val(meta domain) acc {va} mean {m:.3f} (LR {vlr})")
    va = val_acc(model, veps, vK, mu, sd); m = float(np.mean(list(va.values()))); hist.append(dict(step=step, sec=round(time.time() - t0), loss=round(ema or 0, 4), val=va, val_mean=m))
    if m > best[0]: best = (m, copy.deepcopy(model.state_dict()), step)
    model.load_state_dict(best[1]); log(f"[pfn] done: {step} steps in {time.time()-t0:.0f}s, best val {best[0]:.3f} at step {best[2]}, final val {m:.3f}")
    return model, dict(mu=mu, sd=sd), dict(params=npar, steps=step, secs=round(time.time() - t0), best_val=best[0], best_step=best[2], val_lr_reference=vlr, hist=hist, batches_by_source=used, B=B, device=DEV, synth_lr_acc_quantiles=sdiff)
