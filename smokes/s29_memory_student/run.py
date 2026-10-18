"""S29 memory student: is an associative memory over stored (embedding, label) rows, read by a modern-Hopfield softmax, a safe per-call-site student at 1e2-1e6 rows?
All compute on one GPU pod (python smokes/pod.py ...); no API calls. Frozen bge-small embeddings (GPU, fp32), unit norm.
Sections (results.json is rewritten after every cell, cheapest / most informative first):
  A  rows ladder (Banking77 77 labels, CLINC150 in-scope 150 labels, CLINC45 = the 45 CLINC labels that have cached LLM-written examples), 5 splits, budgets 2/5/10%
     arms: LR, LR + LLM-written rows, Hopfield read, Hopfield seeded with LLM-written rows, kNN vote k=10, class prototype cosine, HDC bundle (10,000 bit)
  B  class imbalance (Zipf vs uniform class counts at 3080 rows)     C  deletion (classes / random rows / a frequent token) + recertification
  E  determinism (S7/S8 protocol on CPU 1 thread and GPU)             D  query cost (flat read at 1k..1M rows, hnswlib top-50) incl. the 1M index
Env: S29_TINY=1 builds a fake world with random embeddings; S29_DRY=1 uses the real labels / texts / synthetic rows with RANDOM embeddings and tiny sizes (both are local wiring checks only);
S29_OUT redirects outputs; argv = sections to run (default all). Run order: A_banking, B, C, E, D, A_clinc (CLINC last so a pod timeout costs only its largest cells).
Confidence everywhere = top-2 gap of the arm's class scores. Certificate = common.certify (fixed 40-point grid, delta 0.05) on a 500-row calibration slice of the test set,
applied to the remaining test rows. Accuracy is on those remaining rows."""
import os, sys, json, time, math, re, subprocess, threading, traceback, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
TINY = bool(os.environ.get("S29_TINY")); DRY = bool(os.environ.get("S29_DRY")); LOCAL = TINY or DRY
OUT = Path(os.environ.get("S29_OUT", str(HERE))); OUT.mkdir(parents=True, exist_ok=True)
RES, LOG = OUT / "results.json", OUT / "log.txt"
T0 = time.time()
BUDGETS = (0.02, 0.05, 0.10); BETAS = (5, 10, 20, 40, 80, 160, 320, 640); HC = 3  # HC: number of CUDA attempts (fresh subprocess each)
CFG = dict(splits=2, ncal=60, hd=1000, reps=60, n_mid=160, d_sizes=[100, 160, 320, 2000, 5000], d_q=40) if TINY else \
      dict(splits=2, ncal=500, hd=500, reps=24, n_mid=770, d_sizes=[1000, 770, 9993], d_q=20) if DRY else \
      dict(splits=5, ncal=500, hd=10000, reps=1000, n_mid=3080, d_sizes=[1000, 3080, 9993, 100000, 1000000], d_q=300)
SPLITS, NCAL, HD, REPS, NMID = CFG["splits"], CFG["ncal"], CFG["hd"], CFG["reps"], CFG["n_mid"]
R = {"meta": dict(tiny=TINY, cfg=CFG), "A": [], "B": [], "C": [], "D": {}, "E": {}, "errors": []}

def log(*a):
    s = f"[{time.time() - T0:7.0f}s] " + " ".join(str(x) for x in a); print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f: f.write(s + "\n")
def jd(o): return o.item() if hasattr(o, "item") else str(o)
def save():
    tmp = RES.with_suffix(".tmp"); tmp.write_text(json.dumps(R, default=jd), encoding="utf-8"); os.replace(tmp, RES)

# ------------------------------------------------------------------ CUDA gate: fresh subprocess, 3 tries, drop the image's forward-compat libcuda on the first miss (error 999 on new drivers), exit 2 if absent
def cuda_ok():
    return subprocess.run([sys.executable, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"], capture_output=True).returncode == 0
if not LOCAL and not os.environ.get("S29_CPU"):  # S29_CPU=1: CPU-only fallback host (modal_ak.py), no CUDA gate
    for i in range(HC):
        if cuda_ok(): log(f"cuda ok on attempt {i + 1}"); break
        log(f"cuda NOT available on attempt {i + 1}")
        if i == 0:
            subprocess.run("ls -d /usr/local/cuda*/compat* ; rm -rf /usr/local/cuda*/compat*; ldconfig", shell=True)
            os.environ["LD_LIBRARY_PATH"] = ":".join(p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if "compat" not in p.lower())
        time.sleep(5)
    else:
        log("no CUDA after 3 attempts; exit 2"); sys.exit(2)
    threading.Thread(target=lambda: subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hnswlib"], capture_output=True, timeout=900), daemon=True, name="pip-hnswlib").start()

import torch, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision("highest")
import common
from common import certify, apply_certificate, stratified_subsample
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu"); CPU = torch.device("cpu")
R["meta"].update(torch=torch.__version__, cuda=torch.version.cuda, device=str(DEV), gpu=torch.cuda.get_device_name(0) if DEV.type == "cuda" else "cpu", cpus=os.cpu_count(), numpy=np.__version__)
T = lambda x, dt=None, d=None: torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x, dtype=dt, device=d or DEV)
def sync(d=None):
    if (d or DEV).type == "cuda": torch.cuda.synchronize()

# ------------------------------------------------------------------ data and embeddings
_ST = None
def embed(texts):
    global _ST
    if _ST is None and not os.environ.get("S29_FAKE_EMBED"):
        from sentence_transformers import SentenceTransformer
        _ST = SentenceTransformer(common.EMB_MODEL, device=str(DEV))
    if os.environ.get("S29_FAKE_EMBED"):
        e = np.random.default_rng(len(texts)).normal(size=(len(texts), 384)).astype(np.float32); return e / np.linalg.norm(e, axis=1, keepdims=True)
    t = time.time(); e = _ST.encode(list(texts), batch_size=256, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    log(f"[embed] {len(texts)} texts in {time.time() - t:.1f}s"); return e

def fake(K, ntr, nte, seed, tag):  # TINY world: class centres + noise, texts built from a small vocabulary
    r = np.random.default_rng(seed); C = r.normal(size=(K, 384)); C /= np.linalg.norm(C, axis=1, keepdims=True)
    def mk(n):
        y = np.arange(n) % K; r.shuffle(y); X = C[y] + 0.9 * r.normal(size=(n, 384)) / math.sqrt(384) * 3; X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X.astype(np.float32), y, [f"{tag}{c} w{r.integers(12)} w{r.integers(12)} common" for c in y]
    Xtr, ytr, ttr = mk(ntr); Xte, yte, tte = mk(nte); return Xtr, ytr, ttr, Xte, yte, tte

def make_world(name, Xtr, ytr, ttr, Xte, yte, tte, names, Xsyn, syn_y, seeded):
    K = len(names); w = dict(name=name, K=K, names=names, Xtr=Xtr, ytr=np.asarray(ytr), ttr=ttr, Xte=Xte, yte=np.asarray(yte), tte=tte, Xsyn=None, ysyn=None, seeded=np.asarray(seeded, bool))
    if len(syn_y): w["Xsyn"] = Xsyn; w["ysyn"] = np.asarray(syn_y)
    return w

def build_worlds():
    W = {}
    if TINY:
        a = fake(8, 320, 240, 1, "b"); b = fake(10, 400, 300, 2, "c")
        r = np.random.default_rng(3)
        def syn(K, labs, per):
            ys = np.repeat(labs, per); C = np.random.default_rng(5).normal(size=(K, 384)); X = C[ys] + r.normal(size=(len(ys), 384)); return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32), ys
        Xs, ys = syn(8, np.arange(8), 4); W["banking77"] = make_world("banking77", *a, [f"b{i}" for i in range(8)], Xs, ys, np.ones(8, bool))
        Xs, ys = syn(10, np.arange(5), 4); sm = np.arange(10) < 5; W["clinc150"] = make_world("clinc150", *b, [f"c{i}" for i in range(10)], Xs, ys, sm)
    else:
        b = common.load_banking77(); c = common.load_clinc150()
        DESC = json.load(open(common.CACHE / "label_descriptions.json")); SYN = json.load(open(common.CACHE / "synth_examples.json"))
        collide = set(DESC["clinc150"]) & set(DESC["banking77"])
        t0 = time.time(); btr, bte = [str(t) for t in b["train"][0]], [str(t) for t in b["test"][0]]; ctr, cte = [str(t) for t in c["train"][0]], [str(t) for t in c["test"][0]]
        syn_b = [(i, s) for i, n in enumerate(b["names"]) for s in SYN.get(n, [])]
        cls = np.unique(c["train"][1]); cname = [c["names"][i] for i in cls]; remap = {int(o): i for i, o in enumerate(cls)}
        syn_c = [(i, s) for i, n in enumerate(cname) if n not in collide for s in SYN.get(n, [])]
        E = embed(btr + bte + ctr + cte + [s for _, s in syn_b] + [s for _, s in syn_c]); o = 0
        def take(n):
            nonlocal o; x = E[o:o + n]; o += n; return x
        Xb, Xbt, Xc, Xct = take(len(btr)), take(len(bte)), take(len(ctr)), take(len(cte)); Xsb, Xsc = take(len(syn_b)), take(len(syn_c))
        log("embedded everything in", f"{time.time() - t0:.0f}s", "synthetic rows banking/clinc:", len(syn_b), len(syn_c), "banking labels with synthetic rows:", len({i for i, _ in syn_b}), "clinc labels:", len({i for i, _ in syn_c}))
        W["banking77"] = make_world("banking77", Xb, b["train"][1], btr, Xbt, b["test"][1], bte, b["names"], Xsb, [i for i, _ in syn_b], np.ones(77, bool))
        ytr_c = np.array([remap[int(y)] for y in c["train"][1]]); yte_c = np.array([remap[int(y)] for y in c["test"][1]])
        seeded = np.zeros(len(cname), bool); seeded[sorted({i for i, _ in syn_c})] = True
        W["clinc150"] = make_world("clinc150", Xc, ytr_c, ctr, Xct, yte_c, cte, cname, Xsc, [i for i, _ in syn_c], seeded)
        R["meta"]["clinc_seeded_labels"] = [cname[i] for i in np.flatnonzero(seeded)]
    # CLINC45: only the labels that have cached LLM-written examples, so seeding covers every class
    c = W["clinc150"]; keep = np.flatnonzero(c["seeded"]); mp = -np.ones(c["K"], int); mp[keep] = np.arange(len(keep))
    mtr, mte = mp[c["ytr"]] >= 0, mp[c["yte"]] >= 0; ms = mp[c["ysyn"]] >= 0
    W["clinc45"] = make_world("clinc45", c["Xtr"][mtr], mp[c["ytr"][mtr]], [t for t, m in zip(c["ttr"], mtr) if m], c["Xte"][mte], mp[c["yte"][mte]], [t for t, m in zip(c["tte"], mte) if m],
                              [c["names"][i] for i in keep], c["Xsyn"][ms], mp[c["ysyn"][ms]], np.ones(len(keep), bool))
    for w in W.values():
        w["Xtr_t"], w["Xte_t"] = T(w["Xtr"], torch.float32), T(w["Xte"], torch.float32)
        w["Xsyn_t"] = T(w["Xsyn"], torch.float32) if w["Xsyn"] is not None else None
        R["meta"].setdefault("worlds", {})[w["name"]] = dict(K=w["K"], train=len(w["ytr"]), test=len(w["yte"]), synthetic=0 if w["ysyn"] is None else len(w["ysyn"]), seeded_labels=int(w["seeded"].sum()))
    log("worlds", R["meta"]["worlds"]); return W

# ------------------------------------------------------------------ HDC projection (S25: sign of a fixed random projection, class vector = sum of the rows' hypervectors)
PM = None
def hv(X_t):
    global PM
    if PM is None: PM = T(np.random.default_rng(0).standard_normal((384, HD), dtype=np.float32), torch.float32)
    return torch.cat([torch.where(X_t[i:i + 4096] @ PM >= 0, 1, -1).to(torch.int8) for i in range(0, len(X_t), 4096)])

# ------------------------------------------------------------------ arms: each returns class scores (n_query, K) as float64 numpy; confidence = top-2 gap
def gap(P): s = np.sort(P, 1); return s[:, -1] - s[:, -2]

def lr_fit(X, y, K, C=10.0, sw=None, iters=1000):
    """The S1 recipe (multinomial LR, L2 penalty C=10, unpenalised bias) solved to tight tolerance by L-BFGS in float64 on the GPU: mean CE + 0.5/(C n) |W|^2."""
    Xd = X.double(); n = len(y); y = T(y, torch.long); W = torch.zeros(K, X.shape[1], dtype=torch.float64, device=DEV, requires_grad=True); b = torch.zeros(K, dtype=torch.float64, device=DEV, requires_grad=True)
    sw = None if sw is None else T(sw, torch.float64)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, max_eval=int(iters * 1.5), history_size=20, tolerance_grad=1e-7, tolerance_change=1e-13, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); ce = F.cross_entropy(Xd @ W.T + b, y, reduction="none")
        loss = (ce.mean() if sw is None else (sw * ce).sum() / n) + 0.5 / (C * n) * (W * W).sum(); loss.backward(); return loss
    opt.step(closure); closure(); gn = max(W.grad.abs().max().item(), b.grad.abs().max().item())
    return W.detach(), b.detach(), dict(iters=int(opt.state_dict()["state"][0]["n_iter"]), gnorm=gn)
def lr_proba(W, b, Xq): return torch.softmax(Xq.double() @ W.T + b, 1).cpu().numpy()

def hopfield(M, y, Xq, K, cal, yq, norm=None, cap=None):
    """scores_c = sum_{i: y_i=c} softmax_i(beta cos(q, x_i)); beta from BETAS picked on calibration accuracy (ties: lower NLL). norm: per-class row counts -> divide, renormalise.
    cap: per query keep only the cap most similar rows of each class (missing slots -inf), then the same read."""
    S = Xq @ M.T; Y = F.one_hot(T(y, torch.long), K).float(); yc = T(yq[cal], torch.long); best = None; cal_t = T(cal, torch.long)
    if cap:
        V = torch.full((len(Xq), K, cap), float("-inf"), device=DEV); yt = T(y, torch.long)
        for c in torch.unique(yt).tolist(): Sc = S[:, yt == c]; k = min(cap, Sc.shape[1]); V[:, c, :k] = Sc.topk(k, 1).values
        S = V.reshape(len(Xq), -1); Y = F.one_hot(torch.arange(K, device=DEV).repeat_interleave(cap), K).float()
    for b in BETAS:
        P = torch.softmax(b * S, 1) @ Y
        if norm is not None: P = P / T(np.maximum(norm, 1), torch.float32); P = P / P.sum(1, keepdim=True)
        pc = P[cal_t]; acc = (pc.argmax(1) == yc).float().mean().item(); nll = -torch.log(pc.gather(1, yc[:, None]).squeeze(1) + 1e-9).mean().item()
        if best is None or (acc, -nll) > best[0]: best = ((acc, -nll), b, P)
    return best[2].double().cpu().numpy(), best[1]

def knn(M, y, Xq, K, kn=10):
    S = Xq @ M.T; kn = min(kn, len(M)); v, ix = S.topk(kn, 1); yt = T(y, torch.long)[ix]; n = len(Xq)
    votes = torch.zeros(n, K, device=DEV).scatter_add_(1, yt, torch.ones_like(v)); sim = torch.zeros(n, K, device=DEV).scatter_add_(1, yt, v)
    return (votes + 1e-6 * sim).double().cpu().numpy(), (votes / kn).double().cpu().numpy()

def proto(M, y, Xq, K):
    yt = T(y, torch.long); C = torch.zeros(K, M.shape[1], device=DEV).index_add_(0, yt, M); cnt = torch.bincount(yt, minlength=K)
    S = Xq @ (C / C.norm(dim=1, keepdim=True).clamp_min(1e-12)).T; S[:, cnt == 0] = -1.0; return S.double().cpu().numpy()

def hdc_fit(HVm, y, K):
    C = torch.zeros(K, HD, device=DEV).index_add_(0, T(y, torch.long), HVm.float()); return C  # float32 holds the exact integer sums (|C| <= rows per class)
def hdc_scores(HVq, C):
    nrm = C.double().pow(2).sum(1).sqrt(); S = (HVq.float() @ C.T).double() / (math.sqrt(HD) * nrm.clamp_min(1e-12)); S[:, nrm == 0] = -1.0; return S.cpu().numpy()

# ------------------------------------------------------------------ scoring
def cert_block(conf, ok, cal, hold, budgets=BUDGETS):
    out = {}
    for b in budgets:
        c = certify(conf[cal], ok[cal], b)
        if np.isfinite(c["cutoff"]):
            h = apply_certificate(conf[hold], ok[hold], c["cutoff"]); out[str(b)] = dict(cov=h["coverage"], err=h["error"], cal_cov=c["coverage"], bound=c["bound"], cutoff=c["cutoff"])
        else: out[str(b)] = dict(cov=0.0, err=None, cal_cov=0.0, bound=None, cutoff=None)
    return out
def evaluate(S, y, cal, hold, conf=None):
    pred = S.argmax(1); ok = pred == y; conf = gap(S) if conf is None else conf
    return dict(acc=float(ok[hold].mean()), cal_acc=float(ok[cal].mean()), cert=cert_block(conf, ok, cal, hold)), pred
def split_idx(n, s, base=1000): p = np.random.default_rng(base + s).permutation(n); return p[:NCAL], p[NCAL:]

# ------------------------------------------------------------------ A. rows ladder
def section_A(w, sizes, exact=False):  # exact: exactly n // K rows of every label (A_k) instead of n rows with >= 1 per label
    K, ytr, yte = w["K"], w["ytr"], w["yte"]; Xte = w["Xte_t"]; HVtr, HVte = hv(w["Xtr_t"]), hv(Xte); has_syn = w["Xsyn_t"] is not None
    for n in sizes:
        n = min(n, len(ytr)); t0 = time.time()
        for s in range(SPLITS):
            r = np.random.default_rng(s); idx = np.concatenate([r.choice(np.flatnonzero(ytr == c), n // K, replace=False) for c in range(K)]) if exact else \
                  stratified_subsample(ytr, n, seed=s) if n > 0 else np.array([], int); cal, hold = split_idx(len(yte), s)
            M, y = w["Xtr_t"][T(idx, torch.long)] if n else w["Xtr_t"][:0], ytr[idx]
            def rec(arm, S, conf=None, **extra):
                e, pred = evaluate(S, yte, cal, hold, conf)
                if w["name"].startswith("clinc150"): sm = w["seeded"][yte[hold]]; e.update(acc_seeded=float((pred[hold] == yte[hold])[sm].mean()), acc_unseeded=float((pred[hold] == yte[hold])[~sm].mean()))
                R["A"].append(dict(world=w["name"], n=int(n), split=s, arm=arm, **e, **extra))
            if has_syn:
                Ms, ys = torch.cat([M, w["Xsyn_t"]]), np.concatenate([y, w["ysyn"]])
                S, b = hopfield(Ms, ys, Xte, K, cal, yte); rec("Hopfield+synth", S, beta=b)
                Wt, bt, info = lr_fit(Ms, ys, K); rec("LR+synth", lr_proba(Wt, bt, Xte), **info)
            if n == 0: continue
            S, b = hopfield(M, y, Xte, K, cal, yte); rec("Hopfield", S, beta=b)
            S, V = knn(M, y, Xte, K); rec("kNN10", S, conf=gap(V))
            rec("Prototype", proto(M, y, Xte, K))
            rec("HDC", hdc_scores(HVte, hdc_fit(HVtr[T(idx, torch.long)], y, K)))
            t1 = time.time(); Wt, bt, info = lr_fit(M, y, K); rec("LR", lr_proba(Wt, bt, Xte), fit_s=time.time() - t1, **info)
        save(); log(f"A {w['name']} n={n} done in {time.time() - t0:.0f}s")

def lr_check(w, ns):
    """Cross-check of the GPU L-BFGS LR against sklearn LogisticRegression(C=10, max_iter=2000) on identical rows (split 0)."""
    from sklearn.linear_model import LogisticRegression
    out = []
    for n in ns:
        idx = stratified_subsample(w["ytr"], n, seed=0); cal, hold = split_idx(len(w["yte"]), 0); y = w["ytr"][idx]
        t = time.time(); Wt, bt, info = lr_fit(w["Xtr_t"][T(idx, torch.long)], y, w["K"]); sync(); tt = time.time() - t
        P = lr_proba(Wt, bt, w["Xte_t"]); t = time.time(); sk = LogisticRegression(C=10, max_iter=2000).fit(w["Xtr"][idx], y); ts = time.time() - t
        Q = np.zeros_like(P); Q[:, sk.classes_] = sk.predict_proba(w["Xte"])
        a, _ = evaluate(P, w["yte"], cal, hold); b, _ = evaluate(Q, w["yte"], cal, hold)
        out.append(dict(world=w["name"], n=int(n), torch_s=tt, sklearn_s=ts, sklearn_iter=int(np.max(sk.n_iter_)), torch_iter=info["iters"], agree=float((P.argmax(1) == Q.argmax(1)).mean()), max_dp=float(np.abs(P - Q).max()),
                        acc_torch=a["acc"], acc_sklearn=b["acc"], cov5_torch=a["cert"]["0.05"]["cov"], cov5_sklearn=b["cert"]["0.05"]["cov"]))
        log("lr_check", out[-1])
    R["meta"].setdefault("lr_check", []).extend(out); save()

# ------------------------------------------------------------------ B. imbalance
def waterfill(avail, rank, total):
    """class counts proportional to 1/rank, each class at least 1 and at most its available rows, summing to total (uniform: rank = 1)."""
    f = lambda s: np.minimum(avail, np.maximum(1.0, s / rank)); lo, hi = 0.0, 1e8
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(mid).sum() < total: lo = mid
        else: hi = mid
    c = f(hi); cnt = np.floor(c).astype(int); rem = int(total - cnt.sum())
    for i in np.argsort(-(c - cnt)):
        if rem <= 0: break
        if cnt[i] < avail[i]: cnt[i] += 1; rem -= 1
    return cnt

def section_B(w):
    from sklearn.metrics import f1_score
    K, ytr, yte, Xte = w["K"], w["ytr"], w["yte"], w["Xte_t"]; avail = np.bincount(ytr, minlength=K); HVtr, HVte = hv(w["Xtr_t"]), hv(Xte)
    for r in range(SPLITS):
        cal, hold = split_idx(len(yte), r, 2000); seed = 29 + r
        for design in ("zipf", "uniform"):
            rng = np.random.default_rng(seed); rank = (rng.permutation(K) + 1).astype(float) if design == "zipf" else np.ones(K)
            cnt = waterfill(avail, rank, NMID); idx = np.concatenate([rng.choice(np.flatnonzero(ytr == c), cnt[c], replace=False) for c in range(K)]); y = ytr[idx]
            rare = np.argsort(cnt, kind="stable")[:20]; top = np.argsort(-cnt, kind="stable")[:20]; M = w["Xtr_t"][T(idx, torch.long)]
            sw = (len(y) / (K * cnt[y])).astype(np.float64)
            arms = {"Hopfield": lambda: hopfield(M, y, Xte, K, cal, yte), "Hopfield/count": lambda: hopfield(M, y, Xte, K, cal, yte, norm=cnt),
                    "Hopfield cap1": lambda: hopfield(M, y, Xte, K, cal, yte, cap=1), "Hopfield cap5": lambda: hopfield(M, y, Xte, K, cal, yte, cap=5),
                    "Prototype": lambda: (proto(M, y, Xte, K), None), "HDC": lambda: (hdc_scores(HVte, hdc_fit(HVtr[T(idx, torch.long)], y, K)), None),
                    "LR": lambda: (lr_proba(*lr_fit(M, y, K)[:2], Xte), None), "LR balanced": lambda: (lr_proba(*lr_fit(M, y, K, sw=sw)[:2], Xte), None)}
            for arm, fn in arms.items():
                S, b = fn(); e, pred = evaluate(S, yte, cal, hold); ph, yh = pred[hold], yte[hold]; rm = np.isin(yh, rare)
                R["B"].append(dict(design=design, split=r, arm=arm, beta=b, acc=e["acc"], macro_f1=float(f1_score(yh, ph, average="macro", labels=np.arange(K), zero_division=0)),
                                   rare20_recall=float((ph[rm] == yh[rm]).mean()), top20_recall=float((ph[np.isin(yh, top)] == yh[np.isin(yh, top)]).mean()), pred_in_top20=float(np.isin(ph, top).mean()),
                                   cert=e["cert"], rows_max=int(cnt.max()), rows_min=int(cnt.min()), rows_rare20_mean=float(cnt[rare].mean()), rows_top20_mean=float(cnt[top].mean())))
        save(); log(f"B split {r} done")

# ------------------------------------------------------------------ C. deletion
STOP = set("a an the i my me is it to of and in on for do how can what why when where which who this that be am are was were have has had not no with at by from as or if you your we our they their its i'm i've don't can't there so but about get got would should could will just than then them these those into out up down over any some all more most also".split())
tok = lambda t: set(re.findall(r"[a-z0-9']+", t.lower()))
def pick_token(texts, hi=0.12):
    df = {}
    for t in texts:
        for k in tok(t):
            if k not in STOP and len(k) > 2: df[k] = df.get(k, 0) + 1
    ok = [(c, k) for k, c in df.items() if c <= hi * len(texts)] or [min((c, k) for k, c in df.items())]; c, k = max(ok); return k, c / len(texts)

def section_C(w):
    K, ytr, yte, Xte = w["K"], w["ytr"], w["yte"], w["Xte_t"]; token, share = pick_token(w["ttr"]); R["meta"]["delete_token"] = dict(token=token, share_of_train=share); log("deletion token", token, share)
    for s in range(SPLITS):
        idx = stratified_subsample(ytr, NMID, seed=s); y = ytr[idx]; txt = [w["ttr"][i] for i in idx]; Mfull = w["Xtr_t"][T(idx, torch.long)]
        p = np.random.default_rng(3000 + s).permutation(len(yte)); A, Bc, H = p[:NCAL], p[NCAL:2 * NCAL], p[2 * NCAL:]
        has_tok = np.array([token in tok(t) for t in txt]); te_tok = np.array([token in tok(t) for t in w["tte"]])
        gone = np.random.default_rng(4000 + s).choice(K, min(10, K // 2), replace=False)
        cases = {"classes": np.isin(y, gone), "random20": np.random.default_rng(5000 + s).random(len(y)) < 0.2, "token": has_tok}
        # before: Hopfield and LR on the full memory, certificate from slice A
        Sb, bb = hopfield(Mfull, y, Xte, K, A, yte); eb, pb = evaluate(Sb, yte, A, H); cut_b = eb["cert"]["0.05"]["cutoff"]
        t = time.time(); Wf, bf, _ = lr_fit(Mfull, y, K); Lb = lr_proba(Wf, bf, Xte); sync(); t_fit_full = time.time() - t; elr, plr = evaluate(Lb, yte, A, H)
        for case, drop in cases.items():
            keep = ~drop; dead = np.isin(np.arange(K), gone) if case == "classes" else np.zeros(K, bool); dead_rows = dead[yte]
            def row(arm, stage, e, pred, **x):
                R["C"].append(dict(split=s, case=case, arm=arm, stage=stage, acc=e["acc"], cov=e["cert"]["0.05"]["cov"], err=e["cert"]["0.05"]["err"], cal_cov=e["cert"]["0.05"]["cal_cov"],
                                   pred_deleted_label=float(dead[pred[H]].mean()), acc_token_rows=float((pred[H] == yte[H])[te_tok[H]].mean()) if te_tok[H].any() else None,
                                   acc_other_rows=float((pred[H] == yte[H])[~te_tok[H]].mean()), rows_kept=int(keep.sum()), rows_dropped=int(drop.sum()), **x))
            row("Hopfield", "before", eb, pb, beta=bb); row("LR", "before", elr, plr)
            # stale LR: the model was not refit, so it still answers with the deleted labels / rows' influence
            row("LR", "after_no_refit", elr, plr)
            # memory after deletion: remove the rows, re-pick beta and recertify on the fresh slice B
            sync(); t = time.time(); Mk, yk = Mfull[T(np.flatnonzero(keep), torch.long)], y[keep]; sync(); t_del = time.time() - t
            Sa, ba = hopfield(Mk, yk, Xte, K, Bc, yte); ea, pa = evaluate(Sa, yte, Bc, H); sync(); t_all = time.time() - t
            conf_a = gap(Sa); stale = apply_certificate(conf_a[H], (pa == yte)[H], cut_b) if cut_b is not None else dict(coverage=0.0, error=None)
            row("Hopfield", "after_recert", ea, pa, beta=ba, delete_ms=1000 * t_del, delete_plus_recert_ms=1000 * t_all)
            R["C"].append(dict(split=s, case=case, arm="Hopfield", stage="after_stale_cert", cov=stale["coverage"], err=stale["error"], pred_deleted_label=float(dead[pa[H]].mean())))
            if dead.any():  # view restricted to traffic whose label still exists
                live = ~dead_rows; hl = np.intersect1d(H, np.flatnonzero(live)); cl = np.intersect1d(Bc, np.flatnonzero(live)); e2, _ = evaluate(Sa, yte, cl, hl)
                R["C"].append(dict(split=s, case=case, arm="Hopfield", stage="after_recert_remaining_traffic", acc=e2["acc"], cov=e2["cert"]["0.05"]["cov"], err=e2["cert"]["0.05"]["err"]))
            # LR after deletion: refit on the kept rows over the classes that still have rows
            ck = np.unique(yk); mp = -np.ones(K, int); mp[ck] = np.arange(len(ck)); sync(); t = time.time()
            Wk, bk, _ = lr_fit(Mk, mp[yk], len(ck)); Pk = lr_proba(Wk, bk, Xte); sync(); t_fit = time.time() - t
            Lk = np.zeros((len(Pk), K)); Lk[:, ck] = Pk; elk, plk = evaluate(Lk, yte, Bc, H)
            row("LR", "after_refit", elk, plk, refit_s_gpu=t_fit)
            if s == 0:
                from sklearn.linear_model import LogisticRegression
                t = time.time(); LogisticRegression(C=10, max_iter=2000).fit(Mk.cpu().numpy(), yk); R["C"][-1]["refit_s_sklearn_cpu"] = time.time() - t
        R["C"].append(dict(split=s, case="all", arm="LR", stage="fit_full_s_gpu", refit_s_gpu=t_fit_full, gone_classes=[int(g) for g in gone]))
        save(); log(f"C split {s} done")

# ------------------------------------------------------------------ D / E helpers
try: import hnswlib
except ImportError: hnswlib = None
class StubIndex:  # TINY wiring only: brute-force stand-in with hnswlib's calls
    def __init__(s, space, dim): pass
    def init_index(s, max_elements, ef_construction=0, M=0, random_seed=0): pass
    def add_items(s, X, ids=None, num_threads=1): s.X = np.asarray(X)
    def set_ef(s, ef): pass
    def knn_query(s, Q, k=1, num_threads=1):
        d = 1 - np.asarray(Q) @ s.X.T; ix = np.argsort(d, 1, kind="stable")[:, :k]; return ix, np.take_along_axis(d, ix, 1)
def get_hnsw():
    global hnswlib
    for th in threading.enumerate():
        if th.name == "pip-hnswlib": th.join(timeout=900)
    if hnswlib is None:
        try: import hnswlib as h; hnswlib = h
        except ImportError: pass
    if hnswlib is None and LOCAL: return type("stub", (), {"Index": StubIndex})
    return hnswlib
def hq(ix, Q, k):
    i, d = ix.knn_query(Q, k=k, num_threads=1); return np.asarray(i).astype(np.int64), np.asarray(d)
def hbuild(h, X, threads, seed=29, M=16, efc=100):
    ix = h.Index(space="ip", dim=X.shape[1]); ix.init_index(max_elements=len(X), ef_construction=efc, M=M, random_seed=seed); t = time.time(); ix.add_items(X, np.arange(len(X)), num_threads=threads); return ix, time.time() - t
def ann_read(ids, dist, y, beta, K):
    """top-k neighbours (inner-product distance 1-ip) -> softmax(beta cos) over them -> class scores (float64)."""
    z = -beta * (dist - dist.min(1, keepdims=True)); e = np.exp(z); e /= e.sum(1, keepdims=True); S = np.zeros((len(ids), K))
    for j in range(ids.shape[1]): S[np.arange(len(ids)), y[ids[:, j]]] += e[:, j]
    return S
def mid_memory(w, seed=0):
    idx = stratified_subsample(w["ytr"], NMID, seed=seed); return idx
def pstats(v): v = np.sort(np.asarray(v) * 1000); return dict(p50=float(np.percentile(v, 50)), p95=float(np.percentile(v, 95)), mean=float(v.mean()), n=len(v))  # v in seconds -> ms

DIGEST = r"""
import sys, json, hashlib, numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False; z = np.load(sys.argv[1]); o, got = {}, {}
for d in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
    M, Q = torch.as_tensor(z["M"], device=d), torch.as_tensor(z["Q"], device=d); Y = F.one_hot(torch.as_tensor(z["y"], device=d), int(z["K"])).float()
    P = (torch.softmax(float(z["beta"]) * (Q @ M.T), 1) @ Y).cpu().numpy(); s = np.sort(P, 1); g = s[:, -1] - s[:, -2]; got[d] = (P.argmax(1), g)
    o[d] = dict(raw=hashlib.sha256(P.tobytes()).hexdigest(), label_conf4=hashlib.sha256(P.argmax(1).tobytes() + np.round(g, 4).tobytes()).hexdigest())
if "cuda" in got:
    (la, ga), (lb, gb) = got["cpu"], got["cuda"]
    o["cpu_vs_cuda"] = dict(n=len(la), label_diff=int((la != lb).sum()), conf4_diff=int((np.round(ga, 4) != np.round(gb, 4)).sum()), max_abs_dconf=float(np.abs(ga - gb).max()))
print(json.dumps(o))
"""
BETA_MID = None
def section_E(w):
    global BETA_MID
    torch.set_num_threads(1); K, ytr, yte = w["K"], w["ytr"], w["yte"]; idx = mid_memory(w); y = ytr[idx]; cal, hold = split_idx(len(yte), 0)
    Mn = w["Xtr"][idx]; _, beta = hopfield(w["Xtr_t"][T(idx, torch.long)], y, w["Xte_t"], K, cal, yte); BETA_MID = beta
    HVm, HVq = hv(w["Xtr_t"][T(idx, torch.long)]).cpu(), hv(w["Xte_t"]).cpu(); Wl, bl, _ = lr_fit(w["Xtr_t"][T(idx, torch.long)], y, K)
    probe = 17 % len(yte); E = dict(beta=beta, memory_rows=int(len(idx)), probe=probe, reps=REPS, tests=[]); R["E"] = E
    def hdc_fit_d(H, yt, d): return torch.zeros(K, HD, device=d).index_add_(0, yt, H.float())
    def heads(d):
        Mt, Xq, yt = T(Mn, torch.float32, d), T(w["Xte"], torch.float32, d), T(y, torch.long, d); Y1 = F.one_hot(yt, K).float(); Pn = torch.zeros(K, 384, device=d).index_add_(0, yt, Mt); Pn = Pn / Pn.norm(dim=1, keepdim=True)
        C = hdc_fit_d(T(HVm, torch.int8, d), yt, d); nrm = C.double().pow(2).sum(1).sqrt().cpu().numpy(); Mq = torch.round(Mt.double() * 16384); PMd = T(PM.cpu(), torch.float32, d)
        W32, b32 = Wl.float().to(d), bl.float().to(d)
        def hop_dense(Q): return torch.softmax(beta * (Q @ Mt.T), 1) @ Y1
        def hop_scatter(Q): return torch.zeros(len(Q), K, device=d).index_add_(1, yt, torch.softmax(beta * (Q @ Mt.T), 1))
        def hop_fixed(Q):
            S = (torch.round(Q.double() * 16384) @ Mq.T) / (16384.0 ** 2); e = torch.exp(beta * (S - S.max(1, keepdim=True).values))
            return torch.zeros(len(Q), K, dtype=torch.int64, device=d).index_add_(1, yt, torch.round(e * 2.0 ** 40).to(torch.int64))
        def hdc_head(H): return (H.double() @ C.double().T).long()
        def hdc_full(Q): return hdc_head(torch.where(Q @ PMd >= 0, 1, -1).to(torch.int8))
        pr = lambda v: (int(v.argmax()), float(np.sort(v)[-1] - np.sort(v)[-2]))
        hdc_post = lambda v: pr(v.astype(np.float64) / (math.sqrt(HD) * np.maximum(nrm, 1e-12)))
        return [("Hopfield (dense onehot read)", "emb", hop_dense, lambda v: pr(v.astype(np.float64))), ("Hopfield (index_add scatter)", "emb", hop_scatter, lambda v: pr(v.astype(np.float64))),
                ("Prototype cosine", "emb", lambda Q: Q @ Pn.T, lambda v: pr(v.astype(np.float64))), ("HDC head (integer, from hypervector)", "hv", hdc_head, hdc_post),
                ("HDC projection+sign+head (from embedding)", "emb", hdc_full, hdc_post), ("LR (softmax of float32 logits)", "emb", lambda Q: torch.softmax(Q @ W32.T + b32, 1), lambda v: pr(v.astype(np.float64))),
                ("Hopfield fixed-point (int16 dot, int64 sums)", "emb", hop_fixed, lambda v: pr(v / max(v.sum(), 1)))]
    rng = np.random.default_rng(29)
    def run(d, name):
        sync(d); rng = np.random.default_rng(29); dev_heads = heads(d); inp = {"emb": T(w["Xte"], torch.float32, d), "hv": T(HVq, torch.int8, d)}
        for hn, kind, f, post in dev_heads:
            X = inp[kind]
            def tally(fn, n, label):
                raws, lcs = set(), set()
                for i in range(n):
                    v = fn(i); raws.add(v.tobytes()); lcs.add("%d:%.3f" % post(v))
                E["tests"].append(dict(device=name, head=hn, protocol=label, n=n, distinct_scores=len(raws), distinct_label_conf=len(lcs)))
            with torch.no_grad():
                tally(lambda i: f(X[probe:probe + 1])[0].cpu().numpy(), REPS, "same query x N, batch 1, fixed shape")
                def rb(i, fixed):
                    b = 8 if fixed else int(rng.integers(1, 9)); others = rng.choice(np.setdiff1d(np.arange(len(X)), [probe]), b - 1, replace=False); pos = int(rng.integers(b))
                    ids = np.insert(others, pos, probe); return f(X[T(ids, torch.long, d)])[pos].cpu().numpy()
                tally(lambda i: rb(i, False), REPS, "same query inside random batches of size 1-8 (random slot)")
                tally(lambda i: rb(i, True), REPS, "same query inside random batches of fixed size 8 (random slot)")
            log("E", name, hn, [t for t in E["tests"] if t["head"] == hn and t["device"] == name][-3:])
        save()
        from concurrent.futures import ThreadPoolExecutor  # S7/S8 protocol: 8 threads, own CUDA stream each, batch 1
        for hn, kind, f, post in dev_heads[:2]:
            X = inp[kind]
            def work(k):
                st = torch.cuda.Stream() if d.type == "cuda" else None; out = []
                for _ in range(REPS // 8):
                    if st is not None:
                        with torch.cuda.stream(st): v = f(X[probe:probe + 1])[0].cpu().numpy()
                    else: v = f(X[probe:probe + 1])[0].cpu().numpy()
                    out.append(v)
                return out
            with torch.no_grad(), ThreadPoolExecutor(8) as ex: vs = [v for part in ex.map(work, range(8)) for v in part]
            E["tests"].append(dict(device=name, head=hn, protocol="same query x N from 8 threads (own CUDA stream), batch 1", n=len(vs), distinct_scores=len({v.tobytes() for v in vs}), distinct_label_conf=len({"%d:%.3f" % post(v) for v in vs})))
        save()
    run(CPU, "cpu (1 thread)")
    if DEV.type == "cuda": run(DEV, "gpu")
    # CPU thread count: same batch-1 query and a batch of 8, bytes vs the 1-thread reference
    Mt, q8 = T(Mn, torch.float32, CPU), T(w["Xte"][:8], torch.float32, CPU); Y1 = F.one_hot(T(y, torch.long, CPU), K).float(); rd = lambda Q: (torch.softmax(beta * (Q @ Mt.T), 1) @ Y1).numpy().tobytes()
    ref1, ref8 = None, None; rows = {}
    for th in (1, 2, 4, 8):
        torch.set_num_threads(th); a, b8 = rd(q8[:1]), rd(q8)
        if th == 1: ref1, ref8 = a, b8
        rows[th] = dict(batch1_same_bytes_as_1_thread=a == ref1, batch8_same_bytes_as_1_thread=b8 == ref8)
    torch.set_num_threads(1); E["thread_count"] = rows; log("E thread count", rows)
    # across processes: the same dense read of every test query in 3 fresh interpreters (raw bytes, and label + confidence to 4 dp, hashed)
    import tempfile; f = Path(tempfile.gettempdir()) / "s29_e.npz"; np.savez(f, M=Mn, y=y, Q=w["Xte"], K=K, beta=beta); runs = []
    for _ in range(3):
        p = subprocess.run([sys.executable, "-c", DIGEST, str(f)], capture_output=True, text=True); runs.append(json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else p.stderr[-800:])
    E["cross_process"] = runs; save(); log("E cross-process", runs)
    # approximate index: repeat, batch, rebuild invariance
    h = get_hnsw()
    if h is None: E["ann"] = "hnswlib not installed"; save(); return
    Q = w["Xte"][:200].astype(np.float32); ix1, tb = hbuild(h, Mn, 1); ix1.set_ef(100); ann = dict(build_1thread_s=tb)
    one = lambda ix, q: hq(ix, q[None], 50)
    s1, s2 = set(), set()
    for i in range(REPS):
        ids, dist = one(ix1, Q[0]); s1.add(ids.tobytes() + dist.tobytes()); s2.add("%d:%.3f" % pr_label(ann_read(ids, dist, y, beta, K)[0]))
    ann["same_query_distinct_neighbour_bytes"] = len(s1); ann["same_query_distinct_label_conf"] = len(s2)
    sb = set()
    for i in range(REPS):
        b = int(rng.integers(1, 9)); pos = int(rng.integers(b)); others = rng.choice(np.arange(1, len(Q)), b - 1, replace=False); ids_ = np.insert(others, pos, 0); ii, dd = hq(ix1, Q[ids_], 50); sb.add(ii[pos].tobytes() + dd[pos].tobytes())
    ann["random_batch_distinct_neighbour_bytes"] = len(sb)
    allq = w["Xte"].astype(np.float32)
    def reb(threads, n=3):
        outs = []
        for _ in range(n): ix, _ = hbuild(h, Mn, threads); ix.set_ef(100); outs.append(hq(ix, allq, 50)[0])
        return [float((outs[0] == o).all(1).mean()) for o in outs[1:]]
    ann["rebuild_same_order_1thread_identical_neighbour_lists"] = reb(1); ann["rebuild_4threads_identical_neighbour_lists"] = reb(4)
    E["ann"] = ann; save(); log("E ann", ann)
def pr_label(v): s = np.sort(v); return int(v.argmax()), float(s[-1] - s[-2])

# ------------------------------------------------------------------ D. query cost
def section_D(w, big=True):
    torch.set_num_threads(1); K = w["K"]; ytr = w["ytr"]; nb = len(ytr); beta = BETA_MID or 40; Qall = w["Xte"].astype(np.float32); D = R["D"]; D.update(flat=[], ann=[], const_heads={}, beta=beta, noise_sd=0.005)
    h = get_hnsw(); nq = CFG["d_q"]; qrng = np.random.default_rng(29)
    def memory(N):
        if N <= nb: idx = stratified_subsample(ytr, N, seed=0); return w["Xtr_t"][T(idx, torch.long)], ytr[idx]
        g = torch.Generator(device=DEV); g.manual_seed(29); parts = []; ix = np.arange(N) % nb
        for i in range(0, N, 100000):
            sl = T(ix[i:i + 100000], torch.long); X = w["Xtr_t"][sl] + 0.005 * torch.randn(len(sl), 384, generator=g, device=DEV); parts.append(X / X.norm(dim=1, keepdim=True))
        return torch.cat(parts), ytr[ix]
    qi = qrng.choice(len(Qall), nq, replace=False); Qs = Qall[qi]
    for N in CFG["d_sizes"]:
        if N > nb and not big: continue
        t0 = time.time(); Mg, yn = memory(N); Mc = Mg.cpu().contiguous(); yg, yc = T(yn, torch.long), torch.as_tensor(yn, dtype=torch.long); Y1 = F.one_hot(yg, K).float(); row = dict(N=int(N), memory_mb=N * 384 * 4 / 1e6)
        def cpu_read(q):
            S = Mc @ q; wts = torch.softmax(beta * S, 0); sc = torch.zeros(K).index_add_(0, yc, wts); v, i = sc.topk(2); return int(i[0]), float(v[0] - v[1])
        def gpu_dense(q): S = Mg @ q; sc = torch.softmax(beta * S, 0) @ Y1; v, i = sc.topk(2); return int(i[0]), float((v[0] - v[1]).item())
        def gpu_scatter(q): S = Mg @ q; sc = torch.zeros(K, device=DEV).index_add_(0, yg, torch.softmax(beta * S, 0)); v, i = sc.topk(2); return int(i[0]), float((v[0] - v[1]).item())
        reps = {1000: 400, 3080: 400, 9993: 300, 100000: 100, 1000000: 15}.get(N, 40) if not TINY else 20
        qc, qg = torch.as_tensor(Qs), T(Qs, torch.float32)
        def timeit(fn, qs, n):
            for k in range(3): fn(qs[k % nq])
            sync(); ts = []
            for k in range(n): t = time.perf_counter(); fn(qs[k % nq]); sync(); ts.append(time.perf_counter() - t)
            return pstats(ts)
        with torch.no_grad():
            row["cpu_1thread"] = timeit(cpu_read, qc, reps)
            if DEV.type == "cuda":
                row["gpu_batch1_dense"] = timeit(gpu_dense, qg, reps * 2); row["gpu_batch1_scatter"] = timeit(gpu_scatter, qg, reps * 2)
                Qb = qg[:min(64, nq)]; fb = lambda: torch.softmax(beta * (Qb @ Mg.T), 1) @ Y1
                fb(); sync(); ts = []
                for _ in range(max(5, reps // 10)): t = time.perf_counter(); fb().topk(2, 1); sync(); ts.append(time.perf_counter() - t)
                row["gpu_batch64_per_query"] = pstats([x / len(Qb) for x in ts])
        D["flat"].append(row); save(); log("D flat", row)
        # approximate index: build, query time (1 thread), recall of the exact top-50, share of predictions changed vs the exact full read
        if h is None: D["ann_note"] = "hnswlib not installed"; del Mg, Mc; continue
        if N >= 1000000:
            est = D.get("_build100k_s", 0) * 12
            if est > 240 or time.time() - T0 > 25 * 60: D["ann"].append(dict(N=int(N), skipped=f"estimated build {est:.0f}s / elapsed {time.time() - T0:.0f}s over the cap")); save(); del Mg, Mc; continue
        q_ann = Qall if N == NMID else Qs; Mn = Mc.numpy(); ix, tb = hbuild(h, Mn, min(8, os.cpu_count() or 1))
        if N == 100000: D["_build100k_s"] = tb
        with torch.no_grad():
            qg_ = T(q_ann, torch.float32); S = qg_ @ Mg.T; ex_top = S.topk(min(50, N), 1).indices.cpu().numpy(); full = (torch.softmax(beta * S, 1) @ Y1).double().cpu().numpy(); del S
        sims_top = np.einsum("qd,qkd->qk", q_ann, Mn[ex_top]); ex_sc = ann_read(ex_top, 1 - sims_top, yn, beta, K); pf = full.argmax(1)
        for ef in (50, 100, 200):
            ix.set_ef(ef); ids, dist = hq(ix, q_ann, min(50, N)); sc = ann_read(ids, dist, yn, beta, K)
            rec_ = float(np.mean([len(set(a) & set(b)) / len(b) for a, b in zip(ids, ex_top)])); ts = []
            for k in range(min(len(q_ann), 300)): t = time.perf_counter(); hq(ix, q_ann[k][None], min(50, N)); ts.append(time.perf_counter() - t)
            D["ann"].append(dict(N=int(N), ef=ef, build_s=tb, build_threads=min(8, os.cpu_count() or 1), query=pstats(ts), recall50=rec_, queries=len(q_ann), changed_vs_exact_full=float((sc.argmax(1) != pf).mean()),
                                 trunc_exact_top50_changed_vs_full=float((ex_sc.argmax(1) != pf).mean()), ann_vs_exact_top50_read_changed=float((sc.argmax(1) != ex_sc.argmax(1)).mean())))
            log("D ann", D["ann"][-1])
        save(); del ix, Mg, Mc, Mn
        if DEV.type == "cuda": torch.cuda.empty_cache()
    # constant-cost heads for reference (CPU 1 thread, batch 1)
    q = torch.as_tensor(Qs); Wl = torch.randn(K, 384); Hc = torch.randint(-50, 50, (K, HD)).double(); hvq = torch.where(torch.randn(HD) >= 0, 1.0, -1.0).double()
    with torch.no_grad():
        for nm, fn in (("LR / prototype matvec (K x 384)", lambda k: Wl @ q[k % nq]), ("HDC integer head (K x 10000)", lambda k: Hc @ hvq)):
            for _ in range(5): fn(0)
            ts = []
            for k in range(500): t = time.perf_counter(); fn(k); ts.append(time.perf_counter() - t)
            D["const_heads"][nm] = pstats(ts)
    D.pop("_build100k_s", None); save()

# ------------------------------------------------------------------ main
def main():
    secs = sys.argv[1:] or ["A_banking", "B", "C", "A_clinc", "E", "D", "A_k"]
    W = build_worlds(); b, c, c45 = W["banking77"], W["clinc150"], W["clinc45"]; save()
    if "worlds" in secs: return
    ladders = {"banking77": [0, 77, 231, 385, 770, 3080, 9993], "clinc150": [150, 450, 750, 1500, 7500, 15000], "clinc45": [45, 135, 225, 450, 2250, 4500]}
    if TINY: ladders = {"banking77": [0, 8, 24, 80, 320], "clinc150": [10, 30, 100, 400], "clinc45": [5, 15, 50, 200]}
    if DRY: ladders = {"banking77": [0, 77], "clinc150": [150], "clinc45": [45]}
    def guard(name, fn, *a):
        t = time.time()
        try: fn(*a)
        except Exception: R["errors"].append(dict(section=name, trace=traceback.format_exc()[-3000:])); log("SECTION FAILED", name, traceback.format_exc()[-1500:])
        R["meta"].setdefault("section_secs", {})[name] = time.time() - t; save()
    if "A_banking" in secs:
        guard("A_banking", section_A, b, ladders["banking77"]); guard("lr_check_banking", lr_check, b, [min(770, len(b["ytr"])), min(NMID, len(b["ytr"]))] if not LOCAL else [77 if DRY else 24])
    if "B" in secs: guard("B", section_B, b)
    if "C" in secs: guard("C", section_C, b)
    if "E" in secs: guard("E", section_E, b)
    if "D" in secs: guard("D", section_D, b, True)
    if "A_clinc" in secs:
        guard("A_clinc45", section_A, c45, ladders["clinc45"]); guard("A_clinc150", section_A, c, ladders["clinc150"]); guard("lr_check_clinc", lr_check, c, [1500] if not LOCAL else [150 if DRY else 30])
    if "A_k" in secs:  # exactly k = 1 / 3 / 10 rows per label (the 5-task protocol's k); the ladder's small sizes are n rows with >= 1 per label, so their class counts are uneven
        for w in (b, c45, c):
            nm = w["name"] + " k-exact"; R["meta"]["worlds"][nm] = R["meta"]["worlds"][w["name"]]; guard("A_k_" + w["name"], section_A, dict(w, name=nm), [k * w["K"] for k in ((1, 3) if DRY else (1, 3, 10))], True)
    log("all sections finished; errors:", len(R["errors"]))

if __name__ == "__main__":
    main()
