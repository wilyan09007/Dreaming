"""S23-lite arm 4 (GPU): TSDAE (sentence-transformers' DenoisingAutoEncoderLoss, 60% word deletion, tied decoder) adaptation of bge-small on each task's
UNLABELLED train texts (the pool; never calibration or test rows), 1000 steps, then the encoder is frozen and LR is fit on the k labelled rows.
Checkpoints at 250/500/1000 steps (1000 is the headline). Usage: python tsdae.py [--tiny] [--deadline-min N]"""
import argparse, random, traceback
from lib import *
import torch
from sentence_transformers import SentenceTransformer
try: from sentence_transformers.sentence_transformer.losses import DenoisingAutoEncoderLoss
except ImportError: from sentence_transformers.losses import DenoisingAutoEncoderLoss

OUT = HERE / "rows_tsdae.jsonl"; DEV = "cuda" if torch.cuda.is_available() else "cpu"

def noise(text, p=0.6):  # TSDAE deletion noise without nltk; keeps at least one word
    w = text.split(); keep = [x for x in w if random.random() > p]
    return " ".join(keep or [random.choice(w)]) if w else text

def encode(m, texts):
    return m.encode(texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

def to_dev(f): return {k: (v.to(DEV) if hasattr(v, "to") else v) for k, v in f.items()}

def adapt(t, ckpts, ks, draws, bs, lr, deadline):
    random.seed(0); torch.manual_seed(0); rng = np.random.default_rng(0); K = len(t["labels"])
    m = SentenceTransformer(EMB_MODEL, device=DEV); m.max_seq_length = MAXLEN
    tr = SentenceTransformer(modules=[m[0], m[1]], device=DEV)  # same weights, without Normalize (as in the TSDAE recipe)
    loss_model = DenoisingAutoEncoderLoss(tr, tie_encoder_decoder=True).to(DEV); params = list(loss_model.parameters()); enc_ids = {id(p) for p in tr.parameters()}
    n_enc, n_all = sum(p.numel() for p in tr.parameters()), sum(p.numel() for p in params)
    # bge has no LM head / cross-attention: those new decoder-only params get a larger lr so they do not wreck the shared encoder
    opt = torch.optim.AdamW([{"params": [p for p in params if id(p) in enc_ids], "lr": lr}, {"params": [p for p in params if id(p) not in enc_ids], "lr": 5e-4}], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50)); loss_model.train(); hist, train_secs, texts = [], 0.0, t["ts"]
    for step in range(1, max(ckpts) + 1):
        if time.time() > deadline: log(t["name"], "deadline in TSDAE at step", step); break
        t0 = time.time(); b = [texts[i] for i in rng.choice(len(texts), min(bs, len(texts)), replace=False)]
        loss = loss_model([to_dev(tr.tokenize([noise(x) for x in b])), to_dev(tr.tokenize(b))], None)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); hist.append(loss.item())
        if DEV == "cuda": torch.cuda.synchronize()
        train_secs += time.time() - t0
        if step in ckpts:
            Xs, Xc, Xq = encode(m, t["ts"]), encode(m, t["tc"]), encode(m, t["tq"]); loss_model.train()
            for dr in range(draws):
                for k in ks:
                    L = draw(t["ys"], k, dr, K); s, _ = lr_row(t, L, Xs, Xc, Xq)
                    emit(OUT, arm="4", method=f"TSDAE {step} steps + frozen LR", steps=step, task=t["name"], family=t["family"], K=K, k=k, draw=dr, params=(Xs.shape[1] + 1) * K,
                         adapt_params=n_all, enc_params=n_enc, train_secs=train_secs, loss_last50=float(np.mean(hist[-50:])), **s)
            log(t["name"], "TSDAE", step, f"loss {np.mean(hist[-50:]):.3f} train {train_secs:.0f}s")
    del m, tr, loss_model, opt
    if DEV == "cuda": torch.cuda.empty_cache()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tiny", action="store_true"); ap.add_argument("--deadline-min", type=float, default=30); a = ap.parse_args()
    deadline = time.time() + a.deadline_min * 60; ks, draws = ([1, 3], 1) if a.tiny else (KS, DRAWS); ckpts = [2] if a.tiny else [250, 500, 1000]; bs = 8 if a.tiny else 32
    if OUT.exists(): OUT.unlink()
    T = load_tasks(); log("device", DEV, [(t["name"], len(t["ts"])) for t in T]); t0 = time.time()
    for t in T:
        try: adapt(t, ckpts, ks, draws, bs, 2e-5, deadline)
        except Exception:
            traceback.print_exc(); emit(OUT, arm="4", method="TSDAE", task=t["name"], error=traceback.format_exc()[-600:])
    log("tsdae done", f"{time.time()-t0:.0f}s")
