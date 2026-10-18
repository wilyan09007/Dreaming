"""S17 arm 3 (runs on the pod GPU; reads WORK/store.pkl written by `run.py prep`).
Fine-tune bge-small on the pretraining pool with a supervised-contrastive loss (a label's description is one more member of its class, so the encoder also learns
description -> example alignment), three pool variants + a repeat of the full pool with another seed (noise floor); the checkpoint is chosen on held-out TREC tasks
(TREC is never in the fine-tune pool), then every eval text is re-embedded. Outputs WORK/gpu_out/out_enc_<variant>.npz and results_enc.json (next to this file)."""
import os, sys, json, time, pickle, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import *
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
T0 = time.time(); MODEL = common.EMB_MODEL; dev = "cuda" if torch.cuda.is_available() else "cpu"
OUT = WORK / "gpu_out"; OUT.mkdir(exist_ok=True)
STEPS, CKPTS = (4, (2, 4)) if TINY else (1200, (400, 800, 1200))
R = {"log": [], "device": dev}
def log(*a):
    s = f"[{time.time()-T0:.0f}s] " + " ".join(str(x) for x in a); print(s, flush=True); R["log"].append(s)
def save():
    json.dump(R, open(HERE / ("results_enc_tiny.json" if TINY else "results_enc.json"), "w"), indent=1)
st = pickle.load(open(WORK / "store.pkl", "rb")); pool, ev = st["pool"], st["evals"]; DOM = json.load(open(common.CACHE / "clinc_domains.json"))
pk = np.asarray(pool["key"]); pt = pool["text"]; desc = {k: v["desc"] for k, v in pool["labels"].items()}; ds_of = {k: v["ds"] for k, v in pool["labels"].items()}
tok = AutoTokenizer.from_pretrained(MODEL)

def encode(model, texts, train=False, maxlen=48):
    b = tok(texts, padding=True, truncation=True, max_length=maxlen, return_tensors="pt").to(dev)
    if train and dev == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16): h = model(**b).last_hidden_state[:, 0]
    else: h = model(**b).last_hidden_state[:, 0]  # inference in fp32, like sentence-transformers
    return F.normalize(h.float(), dim=-1)

@torch.no_grad()
def embed_all(model, texts, bs=256):
    model.eval(); order = np.argsort([len(t) for t in texts]); out = np.zeros((len(texts), 384), np.float32)
    for i in range(0, len(texts), bs):
        idx = order[i:i + bs]; out[idx] = encode(model, [texts[j] for j in idx], maxlen=128).cpu().numpy()
    return out

# ---------------------------------------------------------------- held-out validation on TREC tasks: 3-shot LR and description-cosine with the current encoder
trec_keys = sorted(k for k in desc if ds_of[k] == "trec"); rng0 = np.random.default_rng(5)
val_tasks = [list(rng0.choice(trec_keys, min(10, len(trec_keys)), replace=False)) for _ in range(8)]
val_rows = {k: np.flatnonzero(pk == k) for k in trec_keys}
val_idx = np.unique(np.concatenate([val_rows[k] for k in trec_keys]))
@torch.no_grad()
def val_score(model):
    X = embed_all(model, [pt[i] for i in val_idx]); pos = {int(i): j for j, i in enumerate(val_idx)}
    D_ = embed_all(model, [desc[k] for k in trec_keys]); dpos = {k: j for j, k in enumerate(trec_keys)}
    a0, a3 = [], []
    for t in val_tasks:
        for rep in range(3):
            r = np.random.default_rng(rep); tr, tr_y, te, te_y = [], [], [], []
            for c, k in enumerate(t):
                rows = r.permutation(val_rows[k]); tr += list(rows[:3]); tr_y += [c] * len(rows[:3]); te += list(rows[3:33]); te_y += [c] * len(rows[3:33])
            Xtr, Xte = X[[pos[i] for i in tr]], X[[pos[i] for i in te]]
            if rep == 0: a0.append(float(((Xte @ D_[[dpos[k] for k in t]].T).argmax(1) == np.array(te_y)).mean()))
            clf = LogisticRegression(C=10, max_iter=1000).fit(Xtr, tr_y); a3.append(float((clf.predict(Xte) == np.array(te_y)).mean()))
    return float(np.mean(a0)), float(np.mean(a3))

def fit_variant(name, keep, steps=STEPS, lr=3e-5, ckpts=CKPTS, seed=0):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    labs = [k for k in sorted(desc) if ds_of[k] != "trec" and keep(k)]
    by_ds = {}
    for k in labs: by_ds.setdefault(ds_of[k], []).append(k)
    dss = sorted(by_ds); w = np.sqrt([len(by_ds[d]) for d in dss]); w /= w.sum()
    rows = {k: np.flatnonzero(pk == k) for k in labs}
    log(f"[enc {name}] labels={len(labs)} datasets={ {d: len(v) for d, v in by_ds.items()} }")
    model = AutoModel.from_pretrained(MODEL).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / steps))
    v0 = val_score(model); log(f"[enc {name}] step 0 TREC val desc-cos {v0[0]:.3f} 3-shot {v0[1]:.3f}")
    best = (v0[1], 0, None); hist = [(0, v0[0], v0[1])]; v = v0
    for step in range(1, steps + 1):
        model.train(); d = dss[rng.choice(len(dss), p=w)]; L = min(16, len(by_ds[d])); sel = rng.choice(by_ds[d], L, replace=False)
        texts, ys = [], []
        for c, k in enumerate(sel):
            idx = rng.choice(rows[k], min(4, len(rows[k])), replace=False); texts += [pt[i] for i in idx] + [desc[k]]; ys += [c] * (len(idx) + 1)
        z = encode(model, texts, train=True); y = torch.tensor(ys, device=dev); sim = z @ z.T / 0.1
        eye = torch.eye(len(y), dtype=torch.bool, device=dev); sim = sim.masked_fill(eye, -1e9)
        logp = sim - torch.logsumexp(sim, dim=1, keepdim=True); pos = (y[:, None] == y[None, :]) & ~eye
        loss = -(logp * pos).sum(1) / pos.sum(1).clamp(min=1); loss = loss[pos.sum(1) > 0].mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % 100 == 0: log(f"[enc {name}] step {step} loss {float(loss):.3f}")
        if step in ckpts:
            v = val_score(model); hist.append((step, v[0], v[1])); log(f"[enc {name}] step {step} TREC val desc-cos {v[0]:.3f} 3-shot {v[1]:.3f}")
            if v[1] >= best[0]: best = (v[1], step, {k_: p.detach().cpu().clone() for k_, p in model.state_dict().items()})
    if best[2] is None: log(f"[enc {name}] no checkpoint beat the base encoder on TREC val; keeping the last one"); best = (v[1], steps, {k_: p.detach().cpu().clone() for k_, p in model.state_dict().items()})
    model.load_state_dict(best[2]); log(f"[enc {name}] chosen step {best[1]} (val 3-shot {best[0]:.3f})")
    out = {}
    for i, e in enumerate(ev):
        for nm, key in (("Xs", "ts"), ("Xc", "tc"), ("Xq", "tq"), ("desc", "desc_t"), ("name", "name_t"), ("synth", "synth_t")):
            out[f"{nm}{i}"] = embed_all(model, e[key])
    np.savez_compressed(OUT / f"out_enc_{name}.npz", **out)
    R.setdefault("enc", {})[name] = dict(n_labels=len(labs), n_datasets=len(by_ds), hist=hist, chosen_step=best[1], chosen_val_3shot=best[0], base_val_3shot=v0[1], base_val_desccos=v0[0], seconds=time.time() - T0); save()
    del model, opt; torch.cuda.empty_cache()

try:
    base = AutoModel.from_pretrained(MODEL).to(dev); chk = embed_all(base, ev[0]["ts"][:64]); cs = float((chk * ev[0]["Xs"][:64]).sum(1).min())
    log(f"base-encoder sanity: min cosine to the stored sentence-transformers embeddings = {cs:.5f}"); R["base_sanity_min_cos"] = cs; del base
    variants = [("all", lambda k: True, 0), ("clinc7", lambda k: ds_of[k] == "clinc", 0), ("nobank", lambda k: ds_of[k] != "banking" and not (ds_of[k] == "clinc" and DOM.get(k.split("::", 1)[1], "") in ("banking", "credit_cards")), 0), ("all_s1", lambda k: True, 1)]
    for nm, keep, sd in (variants[:2] if TINY else variants):
        try: fit_variant(nm, keep, seed=sd)
        except Exception: log(f"VARIANT {nm} FAILED", traceback.format_exc()[-800:]); save()
except Exception:
    log("PHASE E FAILED", traceback.format_exc()); save()
R["total_seconds"] = time.time() - T0; save(); log("finished")
