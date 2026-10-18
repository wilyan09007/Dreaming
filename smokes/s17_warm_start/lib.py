"""S17 helper: LLM caller (cached, cost-tracked), data paths, hypernet/Reptile model classes, evaluation primitives.
ponytail: everything on frozen bge-small embeddings except the encoder arm."""
import sys, os, json, re, time, hashlib, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import common
from common import *  # noqa  (certify, apply_certificate, confidence, ...)

HERE = Path(__file__).resolve().parent
TINY = bool(os.environ.get("S17_TINY"))  # local smoke check: a handful of rows, no LLM calls, nothing written to the real stores
DATA = ROOT / ".data"; HFDIR = DATA / "s17_hf"
WORK = DATA / ("s17_tiny" if TINY else "s17_store"); WORK.mkdir(parents=True, exist_ok=True)  # big intermediates: never bundled or returned
if TINY: CACHE = ROOT / ".cache" / "tiny"; CACHE.mkdir(parents=True, exist_ok=True)
else: CACHE = common.CACHE
D = 384

# same cache key as common.embed, but the model is loaded once and the GPU is used when there is one (the pod)
_ST = {}
def embed(texts, model=common.EMB_MODEL):
    texts = list(texts); key = hashlib.sha1((model + "\x00".join(texts)).encode()).hexdigest()[:16]; f = CACHE / f"emb_{key}.npy"
    if f.exists(): return np.load(f)
    if model not in _ST:
        import torch; from sentence_transformers import SentenceTransformer
        _ST[model] = SentenceTransformer(model, device="cuda" if torch.cuda.is_available() else "cpu")
    t = time.time(); e = _ST[model].encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    print(f"[embed] {len(texts)} texts in {time.time()-t:.1f}s", flush=True); np.save(f, e); return e

KS = [0, 1, 2, 3, 5, 10, 20]
SEEDS = 5
HELD_CLINC = ["meta", "work", "small_talk"]

# ---------------------------------------------------------------- LLM (ASI:One first, OpenRouter nano fallback), cached
_lock = threading.Lock()
LLMCACHE = HERE / "responses.jsonl"
_cache = {}
if LLMCACHE.exists():
    for ln in LLMCACHE.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(ln); _cache[r["key"]] = r
        except Exception:
            pass
_client = {}
def _get_client():
    if _client: return _client
    from openai import OpenAI
    env = (ROOT.parent / ".env").read_text()
    try:
        cl = OpenAI(base_url="https://api.asi1.ai/v1", api_key=re.search(r"ASI1KEY=(\S+)", env).group(1), timeout=120)
        cl.chat.completions.create(model="asi1-mini", messages=[{"role": "user", "content": "Reply OK"}], max_tokens=3)
        _client.update(cl=cl, model="asi1-mini", pin=0.18e-6, pout=0.45e-6, extra={}, name="asi1-mini")
    except Exception as e:
        print("ASI:One unavailable, using OpenRouter:", repr(e)[:120], flush=True)
        cl = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=re.search(r"OPENROUTERKEY=(\S+)", env).group(1), timeout=120)
        _client.update(cl=cl, model="openai/gpt-5.4-nano", pin=0.2e-6, pout=1.25e-6, extra={"reasoning": {"effort": "minimal"}}, name="gpt-5.4-nano")
    return _client

def llm_json(prompt, max_tokens=8000, temperature=0.7, tag=""):
    """Returns parsed JSON object (dict) or {}; every raw response cached in responses.jsonl keyed by a hash of the request."""
    if TINY: return {}  # no network in the local smoke check; callers fall back to label names
    c = _get_client()
    key = hashlib.sha1(json.dumps([c["model"], prompt, max_tokens, temperature, tag]).encode()).hexdigest()
    if key in _cache:
        txt = _cache[key]["text"]
    else:
        for attempt in range(4):
            try:
                r = c["cl"].chat.completions.create(model=c["model"], messages=[{"role": "user", "content": prompt}], temperature=temperature,
                                                    max_tokens=max_tokens, extra_body=c["extra"])
                txt = r.choices[0].message.content or ""
                cost = r.usage.prompt_tokens * c["pin"] + r.usage.completion_tokens * c["pout"]
                with _lock:
                    rec = dict(key=key, model=c["model"], text=txt, cost=cost, prompt_head=prompt[:80])
                    _cache[key] = rec
                    with open(LLMCACHE, "a", encoding="utf-8") as f: f.write(json.dumps(rec) + "\n")
                break
            except Exception as e:
                print("llm retry", attempt, repr(e)[:100], flush=True); time.sleep(2 ** attempt)
        else:
            return {}
    try:
        return json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
    except Exception:
        return {}

def llm_spend():
    return sum(r.get("cost", 0) for r in _cache.values())

# ---------------------------------------------------------------- shared evaluation primitives
def softmax_cos(X, W, s=30.0):
    W = W / np.linalg.norm(W, axis=1, keepdims=True).clip(1e-9)  # cosine: prototypes / summed descriptions are not unit length
    z = s * X @ W.T; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def full_proba(clf, X, K):
    """LR proba padded to K classes (classes missing from the training rows get 0)."""
    p = np.zeros((len(X), K)); p[:, clf.classes_] = clf.predict_proba(X); return p

def score(pc, yc, pq, yq, budget=0.05):
    """accuracy on test + certificate from the 300-row calibration set applied to the test rows."""
    conf_c, ok_c = confidence(pc), pc.argmax(1) == yc
    conf_q, ok_q = confidence(pq), pq.argmax(1) == yq
    cert = certify(conf_c, ok_c, budget)
    if not np.isfinite(cert["cutoff"]):
        return dict(acc=float(ok_q.mean()), cov=0.0, err=None, cal_cov=0.0)
    h = apply_certificate(conf_q, ok_q, cert["cutoff"])
    return dict(acc=float(ok_q.mean()), cov=h["coverage"], err=h["error"], cal_cov=cert["coverage"])

def draw_support(ys, k, seed, K):
    """indices of k rows per label (a different draw per seed)."""
    r = np.random.default_rng(1000 + seed); idx = []
    for c in range(K):
        pool = np.flatnonzero(ys == c); idx += list(r.choice(pool, min(k, len(pool)), replace=False))
    return np.array(idx, int)

# ---------------------------------------------------------------- hypernetworks (S2b classes) + Reptile model; torch imported lazily
def _torch():
    import torch, torch.nn as nn, torch.nn.functional as F
    return torch, nn, F

def make_models():
    torch, nn, F = _torch()
    class LinGen(nn.Module):
        def __init__(s):
            super().__init__(); s.A = nn.Linear(D, D, bias=False); s.B = nn.Linear(D, D, bias=False); nn.init.zeros_(s.A.weight); nn.init.zeros_(s.B.weight)
            s.scale = nn.Parameter(torch.tensor(10.0)); s.mix = nn.Parameter(torch.tensor(1.0))
        def forward(s, desc, proto, k):
            W = F.normalize(desc + s.A(desc) + (s.mix * proto + s.B(proto) if k else 0), dim=-1); return W, torch.zeros(len(desc))
        def logits(s, X, W, b): return s.scale * X @ W.T + b
    class Metric(nn.Module):
        def __init__(s):
            super().__init__(); s.M = nn.Linear(D, D, bias=False); nn.init.eye_(s.M.weight); s.scale = nn.Parameter(torch.tensor(10.0))
        def forward(s, desc, proto, k): return F.normalize(s.M(desc + proto), dim=-1), torch.zeros(len(desc))
        def logits(s, X, W, b): return s.scale * F.normalize(s.M(X), dim=-1) @ W.T + b
    class SetTr(nn.Module):
        def __init__(s, h=256, drop=0.2, layers=2):
            super().__init__(); s.inp = nn.Linear(2 * D + 2, h); s.drop = nn.Dropout(drop)
            s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(h, 4, 2 * h, dropout=drop, batch_first=True), layers)
            s.w, s.b = nn.Linear(h, D), nn.Linear(h, 1); s.scale = nn.Parameter(torch.tensor(10.0))
        def forward(s, desc, proto, k):
            K = desc.shape[0]; z = s.enc(s.drop(s.inp(torch.cat([desc, proto, torch.full((K, 1), float(k > 0)), torch.full((K, 1), float(np.log1p(k)))], -1)))[None])[0]
            return F.normalize(s.w(z) + desc + proto, dim=-1), s.b(z)[:, 0]
        def logits(s, X, W, b): return s.scale * X @ W.T + b
    return dict(LINGEN=LinGen, METRIC=Metric, SETTR=SetTr)

def reptile_adapt(M0, s0, desc, Xs, ys, K, steps, lrR, lrM=0.0):
    """Fast weights for the FOMAML arm: params (M, r), logits = s * cos(M x, M desc_c + r_c); Adam inner loop on the support rows (M only moves if lrM > 0)."""
    torch, nn, F = _torch()
    M = M0.detach().clone().requires_grad_(lrM > 0); r = torch.zeros(K, D, requires_grad=True)
    groups = [{"params": [r], "lr": lrR}] + ([{"params": [M], "lr": lrM}] if lrM > 0 else [])
    opt = torch.optim.Adam(groups)
    for _ in range(steps):
        lg = s0 * F.normalize(Xs @ M.T, dim=-1) @ F.normalize(desc @ M.T + r, dim=-1).T
        loss = F.cross_entropy(lg, ys); opt.zero_grad(); loss.backward(); opt.step()
    return M.detach(), r.detach()

def reptile_logits(M, r, s, desc, X):
    torch, nn, F = _torch()
    return s * F.normalize(X @ M.T, dim=-1) @ F.normalize(desc @ M.T + r, dim=-1).T
