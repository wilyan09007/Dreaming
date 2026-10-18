"""S23 arm 2: unsupervised encoder adaptation before any labels (SimCSE, TSDAE; AdaSent-style) on each task's unlabelled TRAIN texts (never test rows),
then frozen encoder + LogisticRegression on the k rows. Evaluated after 0 / 250 / 500 / 1000 steps. Usage: python run.py [--tiny]"""
import sys, argparse, random, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import *
import torch, torch.nn.functional as F
from sentence_transformers import SentenceTransformer

OUT = Path(__file__).resolve().parent / "rows.jsonl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def lr_rows(t, Xs, Xq, method, steps, ks, draws, secs):
    K = len(t["labels"]); n = 0
    for dr in range(draws):
        for k in ks:
            L = draw(t["ys"], k, dr, K); clf = lr_fit(Xs[L], t["ys"][L])
            emit(OUT, arm="2", method=method, steps=steps, task=t["name"], family=t["family"], K=K, k=k, draw=dr, params=33_360_000 if steps else 0, train_secs=secs, **score(full_proba(clf, Xq, K), t["yq"])); n += 1
    return n

def encode(m, texts):
    return m.encode(texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

def dev(f):
    return {k_: (v.to(DEV) if hasattr(v, "to") else v) for k_, v in f.items()}

def noise(text, p=0.6):  # TSDAE deletion noise without nltk
    w = text.split(); keep = [x for x in w if random.random() > p]
    return " ".join(keep or [random.choice(w)]) if w else text

def adapt(t, method, ckpts, ks, draws, bs, lr):
    random.seed(0); torch.manual_seed(0)
    m = SentenceTransformer(EMB_MODEL, device=DEV); m.max_seq_length = 64
    texts = t["ts"]; rng = np.random.default_rng(0)
    if method == "tsdae":
        from sentence_transformers.losses import DenoisingAutoEncoderLoss
        tr = SentenceTransformer(modules=[m[0], m[1]], device=DEV)  # same weights, no Normalize (as in the TSDAE recipe)
        loss_model = DenoisingAutoEncoderLoss(tr, tie_encoder_decoder=True).to(DEV); params = list(loss_model.parameters())
        ids = {id(p) for p in tr.parameters()}  # the decoder head / cross-attention are new (bge has no LM head): give them a larger lr so they do not wreck the shared encoder
        groups = [{"params": [p for p in params if id(p) in ids], "lr": lr}, {"params": [p for p in params if id(p) not in ids], "lr": 5e-4}]
    else:
        tr, loss_model, params = m, None, list(m.parameters()); groups = [{"params": params, "lr": lr}]
    opt = torch.optim.AdamW(groups, weight_decay=0.0); t0 = time.time(); (loss_model or m).train(); hist = []
    for step in range(1, max(ckpts) + 1):
        b = [texts[i] for i in rng.choice(len(texts), min(bs, len(texts)), replace=False)]
        if method == "simcse":
            f = dev(m.tokenize(b))
            z1, z2 = m(f)["sentence_embedding"], m(f)["sentence_embedding"]  # two dropout views of the same batch
            loss = F.cross_entropy(z1 @ z2.T / 0.05, torch.arange(len(b), device=DEV))
        else:
            src, tgt = dev(tr.tokenize([noise(x) for x in b])), dev(tr.tokenize(b))
            loss = loss_model([src, tgt], None)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); hist.append(loss.item())
        if step in ckpts:
            secs = time.time() - t0; Xs, Xq = encode(m, t["ts"]), encode(m, t["tq"]); (loss_model or m).train()
            lr_rows(t, Xs, Xq, method, step, ks, draws, secs); log(t["name"], method, step, f"loss {np.mean(hist[-50:]):.3f}", f"{secs:.0f}s")
    del m, tr, loss_model, opt
    if DEV == "cuda": torch.cuda.empty_cache()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tiny", action="store_true"); a = ap.parse_args()
    if OUT.exists(): OUT.unlink()
    T = tasks(tiny=a.tiny); ks, draws = ([1, 3], 1) if a.tiny else (KS, DRAWS); ckpts = [2] if a.tiny else [250, 500, 1000]; bs = 8 if a.tiny else 64
    log("device", DEV, [(t["name"], len(t["ts"])) for t in T])
    for t in T:
        lr_rows(t, t["Xs"], t["Xq"], "unadapted", 0, ks, draws, 0.0)
        for method in ("simcse", "tsdae"):
            try: adapt(t, method, ckpts, ks, draws, bs, 2e-5)
            except Exception:
                traceback.print_exc(); emit(OUT, arm="2", method=method, task=t["name"], error=traceback.format_exc()[-600:])
    log("done", OUT)
