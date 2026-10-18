"""S23-lite shared protocol (same text as the sibling lite smokes).
Tasks: CLINC150 held-out domains travel + work (15 labels each) and three 12-label Banking77 sub-tasks (seed 0 permutation, perm[0:12], [12:24], [24:36]).
Per task: 300 random TRAIN rows are set aside as the fixed calibration set (never labelled-draw rows, never in the unlabelled pool); the remaining train
rows are the pool (labels hidden except k per label); the dataset's TEST split is the evaluation set the certificate is applied to.
Metrics: accuracy + certified coverage at a 5% budget (common.certify, confidence = top-2 gap), realized error above the cutoff on the test split.
ponytail: frozen bge-small + sklearn LR for every baseline."""
import os, sys, json, time, pickle, tempfile
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # smokes/
from common import CACHE, EMB_MODEL, confidence, certify, apply_certificate, load_banking77, load_clinc150  # noqa

KS = [1, 3, 10]; DRAWS = 3; BUDGET = 0.05; N_CAL = 300; MAXLEN = 128
HELD = ["travel", "work"]; N_BANK = 3
DATA = Path(os.environ.get("S23_DATA", str(Path(tempfile.gettempdir()) / "s23_tasks.pkl")))

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

def emit(path, **row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=float) + "\n")

def embed_all(texts, bs=256):
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMB_MODEL, device="cuda" if torch.cuda.is_available() else "cpu"); m.max_seq_length = MAXLEN
    return m.encode(list(texts), batch_size=bs, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)

def build(tiny=False):
    """Build the five tasks (or one tiny CLINC task), embed once, pickle to DATA."""
    DESC = json.load(open(CACHE / "label_descriptions.json")); DOM = json.load(open(CACHE / "clinc_domains.json"))
    c = load_clinc150(); cn = c["names"]; raw = []
    for dom in HELD[:1] if tiny else HELD:
        labs = [n for n in cn if n != "oos" and DOM[n] == dom]; li = {cn.index(n): i for i, n in enumerate(labs)}
        pick = lambda s: [i for i, y in enumerate(c[s][1]) if y in li]
        itr, ite = pick("train"), pick("test")
        raw.append(dict(name=f"clinc/{dom}", family="clinc", labels=[n.replace("_", " ") for n in labs], desc=[DESC["clinc150"].get(n, n) for n in labs],
                        ts=[c["train"][0][i] for i in itr], ys=np.array([li[c["train"][1][i]] for i in itr]),
                        tq=[c["test"][0][i] for i in ite], yq=np.array([li[c["test"][1][i]] for i in ite])))
    if not tiny:
        b = load_banking77(); bn = b["names"]; perm = np.random.default_rng(0).permutation(77); used = set()
        for j in range(N_BANK):
            ids = sorted(perm[j * 12:(j + 1) * 12].tolist()); assert not used & set(ids), "sub-task label sets must be disjoint"; used |= set(ids)
            li = {y: i for i, y in enumerate(ids)}
            itr = [i for i, y in enumerate(b["train"][1]) if y in li]; ite = [i for i, y in enumerate(b["test"][1]) if y in li]
            raw.append(dict(name=f"banking/sub{j}", family="banking", labels=[bn[y].replace("_", " ") for y in ids], desc=[DESC["banking77"][bn[y]] for y in ids],
                            ts=[b["train"][0][i] for i in itr], ys=np.array([li[b["train"][1][i]] for i in itr]),
                            tq=[b["test"][0][i] for i in ite], yq=np.array([li[b["test"][1][i]] for i in ite])))
    if tiny:  # 20 train rows / label, 10 test rows / label, 60-row calibration
        t = raw[0]; r = np.random.default_rng(0); K = len(t["labels"])
        si = np.sort(np.concatenate([r.permutation(np.flatnonzero(t["ys"] == k))[:20] for k in range(K)])); qi = np.sort(np.concatenate([r.permutation(np.flatnonzero(t["yq"] == k))[:10] for k in range(K)]))
        t.update(ts=[t["ts"][i] for i in si], ys=t["ys"][si], tq=[t["tq"][i] for i in qi], yq=t["yq"][qi])
    n_cal = 60 if tiny else N_CAL; allt = [x for t in raw for x in t["ts"] + t["tq"]]; E = embed_all(allt); o = 0; out = []
    for i, t in enumerate(raw):
        Xt, Xq = E[o:o + len(t["ts"])], E[o + len(t["ts"]):o + len(t["ts"]) + len(t["tq"])]; o += len(t["ts"]) + len(t["tq"])
        p = np.random.default_rng(100 + i).permutation(len(t["ts"])); cal, rest = np.sort(p[:n_cal]), np.sort(p[n_cal:])
        assert not set(cal) & set(rest)
        out.append(dict(name=t["name"], family=t["family"], labels=t["labels"], desc=t["desc"],
                        ts=[t["ts"][j] for j in rest], ys=t["ys"][rest], Xs=Xt[rest], tc=[t["ts"][j] for j in cal], yc=t["ys"][cal], Xc=Xt[cal],
                        tq=t["tq"], yq=t["yq"], Xq=Xq))
    DATA.parent.mkdir(parents=True, exist_ok=True); pickle.dump(out, open(DATA, "wb")); return out

def load_tasks():
    return pickle.load(open(DATA, "rb"))

def draw(ys, k, seed, K):
    """k labelled rows per label; draw `seed` (0-2). Same rows for every arm so arms are paired."""
    r = np.random.default_rng(1000 + seed); idx = []
    for c in range(K):
        pool = np.flatnonzero(ys == c); idx += list(r.choice(pool, min(k, len(pool)), replace=False))
    return np.array(idx, int)

def lr_fit(X, y, w=None):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=10, max_iter=1000).fit(X, y, sample_weight=w)

def full_proba(clf, X, K):
    """LR proba padded to K classes (classes absent from the training rows get 0)."""
    p = np.zeros((len(X), K)); p[:, clf.classes_] = clf.predict_proba(X); return p

def score(Pq, yq, Pc=None, yc=None):
    """accuracy on test; with calibration probabilities: certificate fitted on the 300 calibration rows, applied to the test rows
    (cov = fraction of test rows above the cutoff, err = realized error among them, cal_cov = coverage the certificate promised on calibration)."""
    Pq = np.nan_to_num(np.asarray(Pq, float)); okq = Pq.argmax(1) == yq; out = dict(acc=float(okq.mean()))
    if Pc is None: return out
    Pc = np.nan_to_num(np.asarray(Pc, float)); okc = Pc.argmax(1) == yc; cert = certify(confidence(Pc), okc, BUDGET)
    out.update(cal_acc=float(okc.mean()), cal_cov=cert["coverage"], cov=0.0, err=None)
    if np.isfinite(cert["cutoff"]):
        a = apply_certificate(confidence(Pq), okq, cert["cutoff"]); out.update(cov=a["coverage"], err=a["error"])
    return out

def lr_row(t, L, Xs=None, Xc=None, Xq=None):
    """Fit LR on rows L of the (possibly re-embedded) pool and score. Returns (score dict, classifier)."""
    Xs, Xc, Xq = (t["Xs"] if Xs is None else Xs), (t["Xc"] if Xc is None else Xc), (t["Xq"] if Xq is None else Xq); K = len(t["labels"])
    clf = lr_fit(Xs[L], t["ys"][L]); return score(full_proba(clf, Xq, K), t["yq"], full_proba(clf, Xc, K), t["yc"]), clf

if __name__ == "__main__":  # self-check of score() on a synthetic confident-when-right classifier
    r = np.random.default_rng(0); y = r.integers(0, 5, 900); P = np.full((900, 5), 0.02)
    good = r.random(900) < 0.95; P[np.arange(900), np.where(good, y, (y + 1) % 5)] = np.where(good, 0.6 + 0.35 * r.random(900), 0.3 + 0.2 * r.random(900))
    s = score(P[300:], y[300:], P[:300], y[:300]); assert s["acc"] > 0.9 and s["cov"] > 0.3 and (s["err"] or 0) < 0.1, s; print("score ok", s)
