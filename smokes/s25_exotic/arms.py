"""S25 arms. Interface (all arms share it, so the runner is generic):
  prep(t)           label-free per-task work (feature maps, pretraining on the unlabelled pool). Cached; clear(t) drops it (determinism re-runs).
  fit(t, L, seed)   fit on pool rows L (k real rows per label) -> model
  predict(m, t, sp) sp in {"q": test, "c": calibration}; -> {variant: (pred, conf)}   (conf = that arm's own confidence, stated per arm)
  params(m)         parameters touched at inference
ponytail: hyper-parameters are fixed, untuned defaults (like the LR baseline's C=10); nothing here uses the calibration or test rows except to predict."""
import re, zlib, math
from lib import *
import torch, torch.nn as nn, torch.nn.functional as F

SPL = {"s": ("Xs", "ts"), "c": ("Xc", "tc"), "q": ("Xq", "tq")}
def gap(S): s = np.sort(np.asarray(S, float), 1); return s[:, -1] - s[:, -2]
def _dev(): return "cuda" if torch.cuda.is_available() else "cpu"

class Arm:
    name = "?"
    def __init__(s): s.cache = {}
    def prep(s, t): [s.feats(t, sp) for sp in "scq"]
    def clear(s, t): [s.cache.pop(k) for k in [k for k in s.cache if k[0] == t["name"]]]
    def feats(s, t, sp): return t[SPL[sp][0]]
    def params(s, m): return 0
    def explain(s, m, t): return None

# ------------------------------------------------------------------ baseline
class LR(Arm):
    name = "Baseline: bge-small + LR"   # confidence = top-2 probability gap
    def fit(s, t, L, seed): return lr_fit(t["Xs"][L], t["ys"][L])
    def predict(s, m, t, sp): P = full_proba(m, s.feats(t, sp), len(t["labels"])); return {"": (P.argmax(1), confidence(P))}
    def params(s, m): return m.coef_.size + m.intercept_.size

# ------------------------------------------------------------------ 1. hyperdimensional computing
_R = {}
def _proj(D):   # fixed random projection, regenerable from seed 0
    if D not in _R: _R[D] = np.random.default_rng(0).standard_normal((384, D), dtype=np.float32)
    return _R[D]
class HDC(Arm):
    """bge embedding -> sign(random projection) = 10k-d bipolar hypervector. bundle: class vector = sum of its rows' hypervectors (one-shot, order independent);
    online: single OnlineHD pass, each row adds only its novel part (1 - cos to its class vector) and subtracts from a wrongly winning class; centered: bundle after
    subtracting the label-free pool mean. Confidence = top-2 cosine gap between query and class vectors."""
    D = 10000
    def __init__(s, mode): super().__init__(); s.mode = mode; s.name = {"bundle": "HDC bundle (one-shot)", "online": "HDC online (OnlineHD, 1 pass)", "centered": "HDC bundle, pool-centered"}[mode]
    def feats(s, t, sp):
        key = (t["name"], sp)
        if key not in s.cache:
            X = t[SPL[sp][0]] - (t["Xs"].mean(0) if s.mode == "centered" else 0)
            s.cache[key] = np.where(X @ _proj(s.D) >= 0, 1, -1).astype(np.int8)
        return s.cache[key]
    def fit(s, t, L, seed):
        H = s.feats(t, "s")[L].astype(np.float32); y = t["ys"][L]; K = len(t["labels"]); C = np.zeros((K, s.D), np.float32)
        if s.mode != "online":
            for c in range(K): C[c] = H[y == c].sum(0)
            return C
        for i in np.random.default_rng(seed).permutation(len(L)):
            n = np.linalg.norm(C, axis=1); sim = np.where(n > 0, C @ H[i] / (np.maximum(n, 1e-9) * math.sqrt(s.D)), 0.0); p = int(sim.argmax()); c = int(y[i])
            C[c] += (1 - sim[c]) * H[i]
            if p != c and n[p] > 0 and n[c] > 0: C[p] -= (1 - sim[p]) * H[i]   # a class seen for the first time is not a "mistake" to punish
        return C
    def predict(s, m, t, sp):
        S = s.feats(t, sp).astype(np.float32) @ m.T / (math.sqrt(s.D) * np.maximum(np.linalg.norm(m, axis=1), 1e-9)); return {"": (S.argmax(1), gap(S))}
    def params(s, m): return m.size   # class memory only; the 384 x 10000 projection is regenerated from seed 0

# ------------------------------------------------------------------ 2. compression distance (gzip NCD + kNN)
def _clen(b): o = zlib.compressobj(9, zlib.DEFLATED, 31); return len(o.compress(b) + o.flush())   # wbits=31 = gzip container
class NCD(Arm):
    """'Low-resource text classification: a parameter-free method' (ACL 2023): normalized compression distance on raw text, 3-NN vote.
    Variant 'vote margin' (spec): confidence = (top votes - runner-up votes) / 3. Variant 'NCD gap': 1-NN class, confidence = gap between the nearest and the
    second-nearest class distance."""
    name = "gzip NCD + 3-NN"
    def feats(s, t, sp):
        key = (t["name"], sp)
        if key not in s.cache: tb = [x.encode("utf-8") for x in t[SPL[sp][1]]]; s.cache[key] = (tb, np.array([_clen(b) for b in tb]))
        return s.cache[key]
    def fit(s, t, L, seed): tb, cl = s.feats(t, "s"); return dict(tb=[tb[i] for i in L], cl=cl[L], y=t["ys"][L])
    def predict(s, m, t, sp):
        tb, cl = s.feats(t, sp); K = len(t["labels"]); n, nt = len(tb), len(m["tb"]); D = np.empty((n, nt))
        for i in range(n):
            for j in range(nt): D[i, j] = (_clen(tb[i] + b" " + m["tb"][j]) - min(cl[i], m["cl"][j])) / max(cl[i], m["cl"][j])
        dmin = np.stack([D[:, m["y"] == c].min(1) for c in range(K)], 1); kk = min(3, nt)
        V = np.zeros((n, K)); nn_ = np.argsort(D, 1, kind="stable")[:, :kk]
        for r in range(kk): V[np.arange(n), m["y"][nn_[:, r]]] += 1
        return {"vote margin": ((V - 1e-6 * dmin).argmax(1), gap(V) / kk), "NCD gap": ((-dmin).argmax(1), gap(-dmin))}

# ------------------------------------------------------------------ 3. Tsetlin machine
class Tsetlin(Arm):
    """tmu TMClassifier on boolean bag-of-words (top-2000 pool tokens, label-free). 200 clauses per class, T=50, s=5, weighted clauses, <=16 literals per clause,
    ~1500 sample updates (>=10 epochs). Confidence = top-2 gap of clause-vote sums clipped to [-T, T] and divided by T."""
    name = "Tsetlin machine (BoW)"; CL, T, S, LIT = 200, 50, 5.0, 16
    def __init__(s):
        super().__init__(); import importlib.util
        f = Path(importlib.util.find_spec("tmu").submodule_search_locations[0]) / "clause_bank" / "clause_bank.py"; src = f.read_text(encoding="utf-8")
        if "np.uint32(~0)" in src: f.write_text(src.replace("np.uint32(~0)", "np.uint32(0xFFFFFFFF)"), encoding="utf-8")   # tmu 0.8.3 overflows under numpy 2: one-line patch
        from tmu.models.classification.vanilla_classifier import TMClassifier; s.TM = TMClassifier
    def feats(s, t, sp):
        key = (t["name"], sp); vk = (t["name"], "vec")
        if vk not in s.cache:
            from sklearn.feature_extraction.text import CountVectorizer
            s.cache[vk] = CountVectorizer(binary=True, max_features=2000, token_pattern=r"(?u)\b\w+\b").fit(t["ts"])
        if key not in s.cache: s.cache[key] = s.cache[vk].transform(t[SPL[sp][1]]).toarray().astype(np.uint32)
        return s.cache[key]
    def fit(s, t, L, seed):
        X, y = s.feats(t, "s")[L], t["ys"][L].astype(np.uint32)
        m = s.TM(number_of_clauses=s.CL, T=s.T, s=s.S, platform="CPU", weighted_clauses=True, max_included_literals=s.LIT, seed=int(seed))
        for _ in range(max(10, 1500 // len(L))): m.fit(X, y)
        return m
    def predict(s, m, t, sp):
        _, cs = m.predict(s.feats(t, sp), clip_class_sum=True, return_class_sums=True); S = np.asarray(cs, float).reshape(len(cs), -1) / s.T
        return {"": (S.argmax(1), gap(S))}
    def params(s, m): return int(sum(m.number_of_include_actions(c, j) for c in range(len(m.weight_banks)) for j in range(s.CL)))   # included literals = the rule set
    def explain(s, m, t):
        vocab = s.cache[(t["name"], "vec")].get_feature_names_out(); out = {}
        for c in range(min(5, len(t["labels"]))):
            f = np.asarray(m.literal_importance(c))[:len(vocab)]; top = np.argsort(-f)[:8]
            out[t["labels"][c]] = [(str(vocab[i]), int(f[i])) for i in top if f[i] > 0]
        return dict(top_literals_per_class=out, included_literals=s.params(m), clauses=len(m.weight_banks) * s.CL)

# ------------------------------------------------------------------ 4. NTK kernel machines
def ntk(A, B, depth=3):
    """Infinite-width ReLU NTK (Jacot et al.) for unit-norm inputs; kappa0, kappa1 are the arc-cosine kernels of order 0 and 1."""
    S = np.clip(A @ B.T, -1, 1).astype(np.float64); Th = S.copy()
    for _ in range(depth):
        th = np.arccos(S); k0 = (math.pi - th) / math.pi; k1 = (np.sin(th) + (math.pi - th) * S) / math.pi; Th = Th * k0 + k1; S = k1
    return Th
class Kernel(Arm):
    """ReLU-NTK (depth 3) kernel ridge (lambda=0.1, one-hot targets) or one-vs-rest kernel SVM (C=10). Confidence = top-2 gap of the decision scores."""
    def __init__(s, mode): super().__init__(); s.mode = mode; s.name = {"krr": "NTK kernel ridge", "svm": "NTK kernel SVM (OvR)"}[mode]
    def fit(s, t, L, seed):
        X, y = t["Xs"][L], t["ys"][L]; G = ntk(X, X); K = len(t["labels"])
        if s.mode == "krr": return dict(X=X, a=np.linalg.solve(G + 0.1 * np.eye(len(X)), np.eye(K)[y]))
        from sklearn.svm import SVC; from sklearn.multiclass import OneVsRestClassifier
        return dict(X=X, svm=OneVsRestClassifier(SVC(kernel="precomputed", C=10)).fit(G, y))
    def predict(s, m, t, sp):
        G = ntk(s.feats(t, sp), m["X"]); S = G @ m["a"] if s.mode == "krr" else m["svm"].decision_function(G); return {"": (S.argmax(1), gap(S))}
    def params(s, m): return int(m["a"].size) if s.mode == "krr" else int(len(m["X"]) * len(m["svm"].classes_))   # dual coefficients

# ------------------------------------------------------------------ 5. product-key memory head
class _PKM(nn.Module):
    """Lample et al. 2019: query -> two half-queries -> top-8 of 32 sub-keys each -> top-8 of the 64 pairs among 32x32 = 1024 memory slots -> softmax-weighted sum of values."""
    def __init__(s, K, d_in=384, ns=32, dk=64, dv=64, topk=8):
        super().__init__(); s.ns, s.topk = ns, topk; s.q = nn.Linear(d_in, 2 * dk)
        s.k1, s.k2 = nn.Parameter(torch.randn(ns, dk) / dk ** .5), nn.Parameter(torch.randn(ns, dk) / dk ** .5); s.v = nn.Embedding(ns * ns, dv); s.out = nn.Linear(dv, K)
    def forward(s, x):
        q1, q2 = s.q(x * 19.6).chunk(2, -1)   # unit-norm embedding -> unit-variance entries
        (a, i), (b, j) = (q1 @ s.k1.T).topk(s.topk), (q2 @ s.k2.T).topk(s.topk)
        sc = (a[:, :, None] + b[:, None, :]).flatten(1); ix = (i[:, :, None] * s.ns + j[:, None, :]).flatten(1); w, p = sc.topk(s.topk); ix = ix.gather(1, p)
        return s.out((s.v(ix) * w.softmax(-1)[..., None]).sum(1))
class PKM(Arm):
    """Frozen bge embedding -> product-key memory (1024 slots, 8 active per input) -> linear to classes. 300 full-batch Adam steps, lr 1e-2. Confidence = top-2 softmax gap."""
    name = "Product-key memory head"; STEPS, LR_ = 300, 1e-2
    def fit(s, t, L, seed):
        torch.manual_seed(int(seed)); net = _PKM(len(t["labels"])); X, y = torch.tensor(t["Xs"][L]), torch.tensor(t["ys"][L]); opt = torch.optim.Adam(net.parameters(), s.LR_)
        for _ in range(s.STEPS): opt.zero_grad(); F.cross_entropy(net(X), y).backward(); opt.step()
        return net.eval()
    @torch.no_grad()
    def predict(s, m, t, sp): P = m(torch.tensor(s.feats(t, sp))).softmax(-1).numpy(); return {"": (P.argmax(1), gap(P))}
    def params(s, m): return sum(p.numel() for p in m.parameters())

# ------------------------------------------------------------------ 6. spiking head (snnTorch)
class _SNN(nn.Module):
    def __init__(s, d, K, h=256, beta=0.9):
        super().__init__(); import snntorch as snn; from snntorch import surrogate
        sg = surrogate.fast_sigmoid(slope=25); s.fc1, s.fc2 = nn.Linear(d, h), nn.Linear(h, K); s.l1, s.l2 = snn.Leaky(beta=beta, spike_grad=sg), snn.Leaky(beta=beta, spike_grad=sg)
    def forward(s, x):   # x: (T, B, d) spikes -> output spikes, membrane (T, B, K)
        m1, m2, sp, mm = s.l1.init_leaky(), s.l2.init_leaky(), [], []
        for xt in x: s1, m1 = s.l1(s.fc1(xt), m1); s2, m2 = s.l2(s.fc2(s1), m2); sp.append(s2); mm.append(m2)
        return torch.stack(sp), torch.stack(mm)
class SNN(Arm):
    """Frozen bge embedding -> z-score with pool stats -> rate code (p = sigmoid(2z), deterministic regular spike trains over T=16 steps) -> 256 LIF -> K LIF.
    snnTorch Leaky(beta=0.9) + fast-sigmoid surrogate, ce_rate_loss on the output spikes, 300 full-batch Adam steps (lr 1e-3). (ce_count_loss at lr 5e-3 collapsed into a dead
    network in the first pilot; loss and lr were switched after a 5-config check on held-out POOL rows of the tiny travel task, never test or calibration rows.)
    Variant 'spike-count gap' (spec): prediction = most output spikes (membrane breaks ties), confidence = count gap / T.
    Variant 'membrane gap': same prediction, confidence = top-2 gap of softmax(time-mean membrane)."""
    name = "Spiking head (2-layer LIF)"; T, STEPS, LR_ = 16, 300, 1e-3
    def __init__(s):
        super().__init__(); import snntorch; from snntorch import functional as SF; s.SF = SF
    def code(s, t, X):
        mu, sd = t["Xs"].mean(0), t["Xs"].std(0) + 1e-6; p = torch.sigmoid(2 * torch.tensor((X - mu) / sd, dtype=torch.float32)); st = torch.arange(s.T).view(-1, 1, 1)
        return torch.floor(p[None] * (st + 1)) - torch.floor(p[None] * st)   # (T, B, d): round(pT) evenly spaced spikes
    def fit(s, t, L, seed):
        torch.manual_seed(int(seed)); net = _SNN(384, len(t["labels"])); x, y = s.code(t, t["Xs"][L]), torch.tensor(t["ys"][L]); opt = torch.optim.Adam(net.parameters(), s.LR_); lf = s.SF.ce_rate_loss()
        for _ in range(s.STEPS): opt.zero_grad(); lf(net(x)[0], y).backward(); opt.step()
        return net.eval()
    @torch.no_grad()
    def predict(s, m, t, sp):
        spk, mem = m(s.code(t, s.feats(t, sp))); cnt, mm = spk.sum(0).numpy(), mem.mean(0).numpy(); pred = (cnt + 1e-3 * mm).argmax(1)
        P = torch.tensor(mm).softmax(-1).numpy(); return {"spike-count gap": (pred, gap(cnt) / s.T), "membrane gap": (pred, gap(P))}
    def params(s, m): return sum(p.numel() for p in m.parameters())

# ------------------------------------------------------------------ 7. SAE concept bottleneck (GPT-2 small + public jbloom SAE, block 8 resid_pre, 24576 features)
SAE_REPO, SAE_HOOK = "jbloom/GPT2-Small-SAEs-Reformatted", "blocks.8.hook_resid_pre"
_SAE_CACHE = {}; _NP = {}
class _SAEX:
    inst = None
    @classmethod
    def get(c):
        if c.inst is None: c.inst = c()
        return c.inst
    def __init__(s):
        from transformers import AutoTokenizer, AutoModel; from huggingface_hub import hf_hub_download; from safetensors.torch import load_file
        s.dev = _dev(); torch.backends.cuda.matmul.allow_tf32 = False; s.tok = AutoTokenizer.from_pretrained("gpt2"); s.lm = AutoModel.from_pretrained("gpt2").to(s.dev).eval()
        w = {k: v.to(s.dev).float() for k, v in load_file(hf_hub_download(SAE_REPO, f"{SAE_HOOK}/sae_weights.safetensors")).items()}
        s.We, s.be, s.bd = w["W_enc"], w["b_enc"], w["b_dec"]
        if s.We.shape[0] != 768: s.We = s.We.T.contiguous()
        assert tuple(s.We.shape) == (768, 24576) and s.bd.shape[0] == 768, {k: tuple(v.shape) for k, v in w.items()}
    @torch.no_grad()
    def encode(s, texts, bs=64, maxlen=64):
        """BOS-prefixed GPT-2 forward; resid_pre of block 8 = hidden_states[8]; SAE code = relu((h - b_dec) W_enc + b_enc). Returns CSR (last-token codes), CSR (mean over
        non-BOS token codes), dense last-token residual."""
        import scipy.sparse as sp; eos = s.tok.eos_token_id; Ls, Ms, Ds = [], [], []
        for i in range(0, len(texts), bs):
            ids = [[eos] + s.tok.encode(x)[:maxlen - 1] for x in texts[i:i + bs]]; n = max(map(len, ids))
            inp = torch.tensor([r + [eos] * (n - len(r)) for r in ids], device=s.dev); mask = torch.tensor([[1] * len(r) + [0] * (n - len(r)) for r in ids], device=s.dev)
            h = s.lm(input_ids=inp, attention_mask=mask, output_hidden_states=True).hidden_states[8]; f = torch.relu((h - s.bd) @ s.We + s.be)
            last = mask.sum(1) - 1; ar = torch.arange(len(ids), device=s.dev); m = mask[:, 1:, None].float()
            Ls.append(sp.csr_matrix(f[ar, last].cpu().numpy())); Ms.append(sp.csr_matrix(((f[:, 1:] * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())); Ds.append(h[ar, last].cpu().numpy())
        return sp.vstack(Ls).tocsr(), sp.vstack(Ms).tocsr(), np.vstack(Ds)
def _np_lookup(j):
    """Best-effort Neuronpedia explanation for GPT-2 small 8-res-jb feature j (public GET, short timeout, at most 25 calls)."""
    if j in _NP: return _NP[j]
    if len(_NP) >= 25: return None
    try:
        import requests; r = requests.get(f"https://www.neuronpedia.org/api/feature/gpt2-small/8-res-jb/{j}", timeout=8).json(); e = r.get("explanations") or []
        _NP[j] = e[0].get("description") if e else None
    except Exception as ex: _NP[j] = f"lookup failed: {type(ex).__name__}"
    return _NP[j]
class SAEArm(Arm):
    """Learn from concepts, not tokens: LR (C=10) on L2-normalized SAE concept codes of GPT-2 small (last token, or mean over tokens), vs the dense last-token
    residual (pool-standardized, L2-normalized) as a same-LM control. Confidence = top-2 probability gap."""
    def __init__(s, kind):
        super().__init__(); s.kind = kind; s.cache = _SAE_CACHE; s.x = _SAEX.get()
        s.name = {"last": "SAE concepts, last token (GPT-2 L8)", "mean": "SAE concepts, mean over tokens", "dense": "GPT-2 L8 dense residual, last token (control)"}[kind]
    def feats(s, t, sp):
        key = (t["name"], sp)
        if key not in s.cache: s.cache[key] = s.x.encode(t[SPL[sp][1]])
        return s.cache[key]
    def X(s, t, sp):
        from sklearn.preprocessing import normalize
        f = s.feats(t, sp)
        if s.kind == "last": return normalize(f[0])
        if s.kind == "mean": return normalize(f[1])
        P = s.feats(t, "s")[2]; return normalize((f[2] - P.mean(0)) / (P.std(0) + 1e-6))
    def fit(s, t, L, seed): return lr_fit(s.X(t, "s")[L], t["ys"][L])
    def predict(s, m, t, sp): P = full_proba(m, s.X(t, sp), len(t["labels"])); return {"": (P.argmax(1), confidence(P))}
    def params(s, m): return m.coef_.size + m.intercept_.size
    def explain(s, m, t):
        if s.kind == "dense" or t["name"] != "clinc/travel": return None
        f = s.feats(t, "s"); out = dict(L0_last_token=float(np.diff(f[0].indptr).mean()), L0_mean_pooled=float(np.diff(f[1].indptr).mean()), n_features=24576,
                                         pool_features_ever_active=int((np.diff(f[0].tocsc().indptr) > 0).sum()), audited_predictions=[])
        fp, Xq = f[0].tocsc(), s.X(t, "q"); fq = s.feats(t, "q")[0]
        for i in range(6):
            row = Xq[i]; ci = int(m.predict(row)[0]); idx, val = row.indices, row.data * m.coef_[list(m.classes_).index(ci), row.indices]; feats = []
            for j in idx[np.argsort(-val)[:4]]:
                col = fp[:, int(j)]; best = col.indices[np.argsort(-col.data)[:2]]
                feats.append(dict(feature=int(j), contribution=float(val[list(idx).index(j)]), top_pool_texts=[t["ts"][r] for r in best], neuronpedia=_np_lookup(int(j))))
            out["audited_predictions"].append(dict(text=t["tq"][i], predicted=t["labels"][ci], truth=t["labels"][int(t["yq"][i])], active_concepts=int(fq[i].nnz), top_concepts=feats))
        return out

# ------------------------------------------------------------------ 8. JEPA-style predictor on the unlabelled pool
_ST = {}
class JEPA(Arm):
    """Self-supervised, label-free: MLP 384 -> 1024 (GELU) -> 384 predicts the clean-text embedding from the embedding of a 30%-word-dropout view (4 views per pool
    text, loss 1 - cosine, 2000 AdamW steps, batch 256), trained on the task's pool texts only (no labels, no calibration/test rows). Features = [raw embedding,
    L2-normalized predictor hidden layer]; LR (C=10) on the k rows. Confidence = top-2 probability gap."""
    name = "LR on [embedding + JEPA hidden]"; STEPS, V = int(os.environ.get("S25_JEPA_STEPS", 2000)), 4   # env knob only for the local --tiny check
    def net(s, t):
        key = (t["name"], "net")
        if key not in s.cache:
            from sentence_transformers import SentenceTransformer
            dev = _dev(); torch.manual_seed(0); rng = np.random.default_rng(0)
            if "st" not in _ST: _ST["st"] = SentenceTransformer(EMB_MODEL, device=dev); _ST["st"].max_seq_length = MAXLEN
            def view(x):
                w = x.split() or [x]; keep = rng.random(len(w)) > 0.3
                if not keep.any(): keep[rng.integers(len(w))] = True
                return " ".join(a for a, b in zip(w, keep) if b)
            Z = _ST["st"].encode([view(x) for _ in range(s.V) for x in t["ts"]], batch_size=256, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
            Y = torch.tensor(np.tile(t["Xs"], (s.V, 1)), device=dev); Z = torch.tensor(Z, device=dev); mlp = nn.Sequential(nn.Linear(384, 1024), nn.GELU(), nn.Linear(1024, 384)).to(dev)
            opt = torch.optim.AdamW(mlp.parameters(), 1e-3, weight_decay=1e-2); g = torch.Generator(device=dev).manual_seed(0); loss = None
            for _ in range(s.STEPS):
                ix = torch.randint(0, len(Z), (256,), generator=g, device=dev); loss = (1 - F.cosine_similarity(mlp(Z[ix]), Y[ix])).mean(); opt.zero_grad(); loss.backward(); opt.step()
            s.cache[(t["name"], "jepa_loss")] = float(loss.detach()); s.cache[key] = mlp.eval()
        return s.cache[key]
    def feats(s, t, sp):
        key = (t["name"], sp)
        if key not in s.cache:
            mlp = s.net(t); X = t[SPL[sp][0]]
            with torch.no_grad(): h = F.gelu(mlp[0](torch.tensor(X, device=next(mlp.parameters()).device))); h = (h / h.norm(dim=1, keepdim=True).clamp(min=1e-9)).cpu().numpy()
            s.cache[key] = np.concatenate([X, h], 1).astype(np.float32)
        return s.cache[key]
    def fit(s, t, L, seed): return lr_fit(s.feats(t, "s")[L], t["ys"][L])
    def predict(s, m, t, sp): P = full_proba(m, s.feats(t, sp), len(t["labels"])); return {"": (P.argmax(1), confidence(P))}
    def params(s, m): return m.coef_.size + m.intercept_.size + 384 * 1024 + 1024   # LR + first predictor layer needed at inference

# ------------------------------------------------------------------ registry: (label, factory), cheapest first inside a group; groups run as parallel processes
GROUPS = {
    "fast": [(LR.name, LR), ("HDC bundle (one-shot)", lambda: HDC("bundle")), ("HDC bundle, pool-centered", lambda: HDC("centered")), ("HDC online (OnlineHD, 1 pass)", lambda: HDC("online")),
             ("NTK kernel ridge", lambda: Kernel("krr")), ("NTK kernel SVM (OvR)", lambda: Kernel("svm"))],
    "ncd": [(NCD.name, NCD)],
    "tm": [(Tsetlin.name, Tsetlin)],
    "torch": [(PKM.name, PKM), (SNN.name, SNN)],
    "gpu": [(JEPA.name, JEPA), ("SAE concepts, last token (GPT-2 L8)", lambda: SAEArm("last")), ("SAE concepts, mean over tokens", lambda: SAEArm("mean")),
            ("GPT-2 L8 dense residual, last token (control)", lambda: SAEArm("dense"))],
}

if __name__ == "__main__":  # self-check: pure functions only (no data)
    a = np.random.default_rng(0).standard_normal((5, 384)); a /= np.linalg.norm(a, axis=1, keepdims=True); G = ntk(a, a)
    assert np.allclose(np.diag(G), 4) and np.allclose(G, G.T) and np.linalg.eigvalsh(G).min() > -1e-9, "NTK must be PSD with diagonal depth+1"
    import gzip; s_ = b"cancel my flight to boston"; assert _clen(s_) == len(gzip.compress(s_, 9, mtime=0)), "wbits=31 must equal the gzip container length"
    print("arms self-check ok")
