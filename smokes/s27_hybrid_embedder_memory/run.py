"""S27 hybrid embedder swap + memory heads on the S17/S23/S25 new-step protocol. CPU only, no API calls.
Stages:  python run.py embed   -> builds the five tasks (text), embeds with every model, caches .npy under smokes/.cache
         python run.py arms    -> fits all arms, writes results.json incrementally, tables.md
Protocol copied from smokes/s25_exotic/lib.py (build/draw/score); only the embedder and the heads change."""
import sys, os, json, time, hashlib, shutil, warnings
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import CACHE, confidence, certify, apply_certificate, load_banking77, load_clinc150  # noqa
warnings.filterwarnings("ignore")

KS = [1, 3, 10]; DRAWS = 3; BUDGET = 0.05; N_CAL = 300; HELD = ["travel", "work"]; N_BANK = 3
MODELS = {  # name -> (hub id, params in millions filled at embed time)
    "bge_small": "BAAI/bge-small-en-v1.5",
    "lfm25_350m": "LiquidAI/LFM2.5-Embedding-350M",
    "bge_base": "BAAI/bge-base-en-v1.5",
}
LFM_PATCHED = Path(os.environ.get("S27_LFM_DIR", str(CACHE / "lfm_patched")))
RES = HERE / "results.json"; LOG = HERE / "log.txt"

def log(*a):
    s = f"[{time.strftime('%H:%M:%S')}] " + " ".join(str(x) for x in a); print(s, flush=True)
    with open(LOG, "a") as f: f.write(s + "\n")

# ------------------------------------------------------------------ tasks (text only; same split logic as s25 lib.build)
def build_text():
    DESC = None
    DOM = json.load(open(CACHE / "clinc_domains.json"))
    c = load_clinc150(); cn = c["names"]; raw = []
    for dom in HELD:
        labs = [n for n in cn if n != "oos" and DOM[n] == dom]; li = {cn.index(n): i for i, n in enumerate(labs)}
        pick = lambda s: [i for i, y in enumerate(c[s][1]) if y in li]
        itr, ite = pick("train"), pick("test")
        raw.append(dict(name=f"clinc/{dom}", K=len(labs), ts=[c["train"][0][i] for i in itr], ys=np.array([li[c["train"][1][i]] for i in itr]),
                        tq=[c["test"][0][i] for i in ite], yq=np.array([li[c["test"][1][i]] for i in ite])))
    b = load_banking77(); perm = np.random.default_rng(0).permutation(77); used = set()
    for j in range(N_BANK):
        ids = sorted(perm[j * 12:(j + 1) * 12].tolist()); assert not used & set(ids); used |= set(ids)
        li = {y: i for i, y in enumerate(ids)}
        itr = [i for i, y in enumerate(b["train"][1]) if y in li]; ite = [i for i, y in enumerate(b["test"][1]) if y in li]
        raw.append(dict(name=f"banking/sub{j}", K=12, ts=[b["train"][0][i] for i in itr], ys=np.array([li[b["train"][1][i]] for i in itr]),
                        tq=[b["test"][0][i] for i in ite], yq=np.array([li[b["test"][1][i]] for i in ite])))
    out = []
    for i, t in enumerate(raw):
        p = np.random.default_rng(100 + i).permutation(len(t["ts"])); cal, rest = np.sort(p[:N_CAL]), np.sort(p[N_CAL:])
        out.append(dict(name=t["name"], K=t["K"], ts=[t["ts"][j] for j in rest], ys=t["ys"][rest], tc=[t["ts"][j] for j in cal], yc=t["ys"][cal], tq=t["tq"], yq=t["yq"]))
    return out

def all_texts(tasks):
    return [x for t in tasks for x in t["ts"] + t["tc"] + t["tq"]]

# ------------------------------------------------------------------ embedding with .npy cache (key scheme of common.embed)
def embed_model(name, texts, bs=32):
    hub = MODELS[name]; key = hashlib.sha1((hub + "\x00".join(texts)).encode()).hexdigest()[:16]
    f = CACHE / f"emb_{key}.npy"; meta = CACHE / f"emb_{key}.json"
    if f.exists():
        log(f"[cache] {name} {f.name}"); return np.load(f), json.load(open(meta))
    import torch
    torch.set_num_threads(4)
    from sentence_transformers import SentenceTransformer
    if name == "lfm25_350m":
        if not LFM_PATCHED.exists():
            from huggingface_hub import snapshot_download
            shutil.copytree(snapshot_download(hub), LFM_PATCHED)
            g = LFM_PATCHED / "modeling_lfm2_bidirectional.py"; s = g.read_text()
            old = "    attention_mask: Optional[torch.Tensor] = None,\n) -> torch.Tensor:\n    # eager/sdpa"
            assert old in s; g.write_text(s.replace(old, "    attention_mask: Optional[torch.Tensor] = None,\n    **_ignored,\n) -> torch.Tensor:\n    # eager/sdpa"))
        m = SentenceTransformer(str(LFM_PATCHED), device="cpu", trust_remote_code=True)
    else:
        m = SentenceTransformer(hub, device="cpu")
    m.max_seq_length = 128
    npar = sum(p.numel() for p in m.parameters()) / 1e6
    t = time.time()
    e = m.encode(texts, batch_size=bs, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    dt = time.time() - t; info = dict(model=hub, params_m=npar, dim=int(e.shape[1]), n_texts=len(texts), seconds=dt, ms_per_text=1000 * dt / len(texts))
    log(f"[embed] {name}: {len(texts)} texts in {dt:.0f}s ({info['ms_per_text']:.1f} ms/text), {npar:.0f}M params, dim {e.shape[1]}")
    np.save(f, e); json.dump(info, open(meta, "w")); return e, info

def split(E, tasks):
    o = 0; out = []
    for t in tasks:
        n1, n2, n3 = len(t["ts"]), len(t["tc"]), len(t["tq"])
        out.append((E[o:o + n1], E[o + n1:o + n1 + n2], E[o + n1 + n2:o + n1 + n2 + n3])); o += n1 + n2 + n3
    return out

# ------------------------------------------------------------------ protocol pieces (from s25 lib)
def draw(ys, k, seed, K):
    r = np.random.default_rng(1000 + seed); idx = []
    for c in range(K):
        pool = np.flatnonzero(ys == c); idx += list(r.choice(pool, min(k, len(pool)), replace=False))
    return np.array(idx, int)

def score(Pq, yq, Pc, yc):
    Pq = np.nan_to_num(np.asarray(Pq, float)); Pc = np.nan_to_num(np.asarray(Pc, float))
    okq = Pq.argmax(1) == yq; okc = Pc.argmax(1) == yc
    cert = certify(confidence(Pc), okc, BUDGET); out = dict(acc=float(okq.mean()), cal_acc=float(okc.mean()), cal_cov=cert["coverage"], cov=0.0, err=None)
    if np.isfinite(cert["cutoff"]):
        a = apply_certificate(confidence(Pq), okq, cert["cutoff"]); out.update(cov=a["coverage"], err=a["error"])
    return out

def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

# ------------------------------------------------------------------ heads
def lr_head(X, y, Xc, Xq, yc, K):
    from sklearn.linear_model import LogisticRegression
    t = time.perf_counter(); clf = LogisticRegression(C=10, max_iter=1000).fit(X, y); ft = time.perf_counter() - t
    def pp(Z):
        p = np.zeros((len(Z), K)); p[:, clf.classes_] = clf.predict_proba(Z); return p
    return pp(Xq), pp(Xc), dict(fit_s=ft, params=K * (X.shape[1] + 1), note="")

BETAS = (5, 10, 20, 40, 80, 160, 320, 640)  # S24 grid
def hopfield_head(X, y, Xc, Xq, yc, K):
    t = time.perf_counter(); Y = np.eye(K)[y]
    read = lambda Z, b: softmax(b * (Z @ X.T)) @ Y
    nll = [-np.log(read(Xc, b)[np.arange(len(yc)), yc] + 1e-9).mean() for b in BETAS]; b = BETAS[int(np.argmin(nll))]
    ft = time.perf_counter() - t
    return read(Xq, b), read(Xc, b), dict(fit_s=ft, params=1, stored=int(X.size), beta=b, note="no trained weights; memory = k*K rows")

LRS = (0.03, 0.1, 0.3, 1.0); EPOCHS = (1, 3, 10, 30)  # delta-rule grid
def delta_train(X, y, K, lr, epochs, seed=0):
    """Online least squares: W <- W + lr (onehot - W x) x^T over the k*K rows, shuffled each pass (bias via appended constant)."""
    Xa = np.hstack([X, np.ones((len(X), 1), np.float32)]); W = np.zeros((K, Xa.shape[1]), np.float64); Y = np.eye(K)[y]; r = np.random.default_rng(seed)
    for _ in range(epochs):
        for i in r.permutation(len(Xa)):
            W += lr * np.outer(Y[i] - W @ Xa[i], Xa[i])
    return W
def delta_head(X, y, Xc, Xq, yc, K):
    t = time.perf_counter(); app = lambda Z: np.hstack([Z, np.ones((len(Z), 1), np.float32)]); best = None
    for lr in LRS:
        for ep in EPOCHS:
            W = delta_train(X, y, K, lr, ep); S = app(Xc) @ W.T
            key = ((S.argmax(1) == yc).mean(), -((S - np.eye(K)[yc]) ** 2).mean())  # calibration accuracy, then squared error
            if best is None or key > best[0]: best = (key, lr, ep, W)
    _, lr, ep, W = best; ft = time.perf_counter() - t
    return app(Xq) @ W.T, app(Xc) @ W.T, dict(fit_s=ft, params=K * (X.shape[1] + 1), lr=lr, epochs=ep, note="grid 4 lr x 4 epoch settings on calibration accuracy")

LAMS = (1e-3, 1e-2, 1e-1, 1, 10)
def ridge_head(X, y, Xc, Xq, yc, K):
    """Regularised closed-form least squares (ridge), lambda on calibration; approaches the delta-rule limit only as lambda -> 0."""
    t = time.perf_counter(); app = lambda Z: np.hstack([Z, np.ones((len(Z), 1), np.float32)]).astype(np.float64); Xa = app(X); Y = np.eye(K)[y]; best = None
    for lam in LAMS:
        W = np.linalg.solve(Xa.T @ Xa + lam * np.eye(Xa.shape[1]), Xa.T @ Y).T; S = app(Xc) @ W.T
        key = ((S.argmax(1) == yc).mean(), -((S - np.eye(K)[yc]) ** 2).mean())
        if best is None or key > best[0]: best = (key, lam, W)
    _, lam, W = best
    return app(Xq) @ W.T, app(Xc) @ W.T, dict(fit_s=time.perf_counter() - t, params=K * (X.shape[1] + 1), lam=lam, note="regularised closed-form least squares")

ARMS = [  # (arm id, embedder, head)
    ("LR bge-small (baseline)", "bge_small", lr_head),
    ("LR LFM2.5-350M", "lfm25_350m", lr_head),
    ("LR bge-base", "bge_base", lr_head),
    ("delta-rule head (bge-small)", "bge_small", delta_head),
    ("ridge least squares (bge-small)", "bge_small", ridge_head),
    ("Hopfield read (bge-small)", "bge_small", hopfield_head),
]

def run_arms():
    tasks = build_text(); texts = all_texts(tasks); embs = {}; infos = {}
    for m in MODELS:
        E, info = embed_model(m, texts); embs[m] = split(E, tasks); infos[m] = info
    cells = []; res = dict(meta=dict(tasks=[(t["name"], t["K"], len(t["ts"]), len(t["tc"]), len(t["tq"])) for t in tasks], ks=KS, draws=DRAWS, budget=BUDGET, embedders=infos, note="LLM-written-examples prior not included"), cells=cells)
    for ti, t in enumerate(tasks):
        K = t["K"]
        for k in KS:
            for d in range(DRAWS):
                L = draw(t["ys"], k, d, K)
                for arm, emb, head in ARMS:
                    Xs, Xc, Xq = embs[emb][ti]
                    Pq, Pc, info = head(Xs[L], t["ys"][L], Xc, Xq, t["yc"], K)
                    s = score(Pq, t["yq"], Pc, t["yc"]); s.update(arm=arm, task=t["name"], k=k, draw=d, **{kk: v for kk, v in info.items() if kk != "note"})
                    cells.append(s)
        json.dump(res, open(RES, "w"), default=float); log(f"task {t['name']} done ({len(cells)} cells)")
    return res

if __name__ == "__main__":
    if sys.argv[1] == "embed":
        tasks = build_text(); texts = all_texts(tasks); log(f"{len(tasks)} tasks, {len(texts)} texts")
        for m in (sys.argv[2:] or MODELS): embed_model(m, texts)
    else:
        run_arms(); import report; report.main()
