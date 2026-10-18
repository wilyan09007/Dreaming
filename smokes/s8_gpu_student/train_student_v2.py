"""S8 v2: the student ladder on GPU after the first LoRA pass under-performed frozen embeddings + LR.
Rungs: (1) full fine-tune of the 33M bge-small encoder + linear head; (2) SetFit (contrastive fine-tune of bge-small,
the literature's strongest few-shot recipe); (3) LoRA on Qwen2.5-0.5B seq-cls with a tuned recipe (fixed step budget,
separate head LR, r=32). Banking77 at 770 / 3080 / 9993 rows; accuracy, certified coverage @5%, train seconds.
Usage: python train_student_v2.py --out results_v2.json
"""
import argparse, json, time
import numpy as np, torch, torch.nn.functional as F
from scipy.stats import beta as _beta
p = argparse.ArgumentParser(); p.add_argument("--out", default="results_v2.json"); p.add_argument("--maxlen", type=int, default=64)
a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); np.random.seed(0)
R = {"device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"}
def log(*x): print(*x, flush=True)
import datasets
b = datasets.load_dataset("mteb/banking77")
TR_X, TR_Y = b["train"]["text"], np.array(b["train"]["label"]); TE_X, TE_Y = b["test"]["text"], np.array(b["test"]["label"])
def stratified(labels, n, seed=0):
    r = np.random.default_rng(seed); labels = np.asarray(labels); idx = r.permutation(len(labels)); cls = np.unique(labels)
    first = [int(idx[labels[idx] == k][0]) for k in cls] if n >= len(cls) else []
    fs = set(first); rest = [int(i) for i in idx if int(i) not in fs]
    return np.array(first + rest[: max(0, n - len(first))])
def cp_upper(e, n, d): return 1.0 if n == 0 or e >= n else float(_beta.ppf(1 - d, e + 1, n - e))
def certify(conf, ok, budget=0.05, delta=0.05):
    o = np.argsort(-conf); ok = ok[o]; conf = conf[o]; n = len(conf); m = int(np.ceil(np.log2(n))) + 1; d = delta / m
    ce = np.cumsum(~ok); lo, hi, best = 1, n, None
    for _ in range(m):
        if lo > hi: break
        k = (lo + hi) // 2
        if cp_upper(int(ce[k - 1]), k, d) <= budget: best, lo = k, k + 1
        else: hi = k - 1
    return (float("inf"), 0.0) if best is None else (float(conf[best - 1]), best / n)
def evaluate(P, y):
    pred = P.argmax(1); ok = pred == y; s = np.sort(P, 1); conf = s[:, -1] - s[:, -2]
    r = np.random.default_rng(1); perm = r.permutation(len(y)); cal, hold = perm[: len(y) // 2], perm[len(y) // 2:]
    cut, cov = certify(conf[cal], ok[cal]); above = conf[hold] >= cut
    return dict(acc=round(float(ok.mean()), 4), cert5_coverage=round(cov, 3),
                holdout_err_above=round(float((~ok[hold][above]).mean()), 4) if above.any() else None)
ROWS = [770, 3080, 9993]; R["runs"] = []
def record(**kw): R["runs"].append(kw); log(kw); json.dump(R, open(a.out, "w"), indent=1)

# ---------------- rung 1: full fine-tune bge-small + linear head
from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
def ft_encoder(idx, steps, lr=5e-5, bs=32):
    enc = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").to(dev); head = torch.nn.Linear(384, 77).to(dev)
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": lr}, {"params": head.parameters(), "lr": 1e-3}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, (steps - s) / steps))
    X = [TR_X[i] for i in idx]; Y = torch.tensor(TR_Y[idx]); t0 = time.time(); step = 0; enc.train()
    def fwd(texts):
        t = tok(texts, padding=True, truncation=True, max_length=a.maxlen, return_tensors="pt").to(dev)
        h = enc(**t).last_hidden_state[:, 0]; return head(F.normalize(h, dim=-1) * 20)
    while step < steps:
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            if step >= steps: break
            j = perm[i:i + bs]; loss = F.cross_entropy(fwd([X[k] for k in j]), Y[j].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
    dt = time.time() - t0; enc.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(TE_X), 256): out.append(torch.softmax(fwd(TE_X[i:i + 256]), -1).cpu().numpy())
    return np.concatenate(out), dt
for n in ROWS:
    idx = stratified(TR_Y, n); steps = max(300, 8 * ((len(idx) + 31) // 32))
    P, dt = ft_encoder(idx, steps); record(method="finetune-bge-small", rows=int(len(idx)), steps=steps, train_s=round(dt, 1), **evaluate(P, TE_Y))

# ---------------- rung 2: SetFit (contrastive) on bge-small
try:
    from setfit import SetFitModel, Trainer, TrainingArguments
    from datasets import Dataset
    for n in ROWS[:2]:  # SetFit's pair sampling is quadratic-ish; 9993 rows would take too long, and SetFit is a few-shot recipe
        idx = stratified(TR_Y, n); ds = Dataset.from_dict({"text": [TR_X[i] for i in idx], "label": [int(TR_Y[i]) for i in idx]})
        model = SetFitModel.from_pretrained("BAAI/bge-small-en-v1.5"); t0 = time.time()
        args = TrainingArguments(batch_size=32, num_epochs=1, num_iterations=10 if n <= 770 else 3, seed=0, report_to="none")
        Trainer(model=model, args=args, train_dataset=ds).train(); dt = time.time() - t0
        P = model.predict_proba(TE_X, as_numpy=True); record(method="setfit-bge-small", rows=int(len(idx)), train_s=round(dt, 1), **evaluate(P, TE_Y))
        del model; torch.cuda.empty_cache()
except Exception as e:
    log("setfit failed:", repr(e)); R["setfit_error"] = repr(e)

# ---------------- rung 3: LoRA Qwen2.5-0.5B seq-cls, tuned recipe
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
qtok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B"); qtok.padding_side = "right"
if qtok.pad_token is None: qtok.pad_token = qtok.eos_token
fmt = lambda t: f"Step: categorize bank support query\nQuery: {t}\nLabel:"
def lora_run(idx, steps, bs=32):
    m = AutoModelForSequenceClassification.from_pretrained("Qwen/Qwen2.5-0.5B", num_labels=77, dtype=torch.float32); m.config.pad_token_id = qtok.pad_token_id
    m = get_peft_model(m, LoraConfig(task_type="SEQ_CLS", r=32, lora_alpha=64, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], modules_to_save=["score"])).to(dev)
    head = [q for nme, q in m.named_parameters() if q.requires_grad and "score" in nme]; lora = [q for nme, q in m.named_parameters() if q.requires_grad and "score" not in nme]
    opt = torch.optim.AdamW([{"params": lora, "lr": 3e-4}, {"params": head, "lr": 2e-3}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, (steps - s) / steps))
    X = [fmt(TR_X[i]) for i in idx]; Y = torch.tensor(TR_Y[idx]); t0 = time.time(); step = 0; m.train()
    while step < steps:
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            if step >= steps: break
            j = perm[i:i + bs]; t = qtok([X[k] for k in j], padding=True, truncation=True, max_length=a.maxlen, return_tensors="pt").to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")): loss = F.cross_entropy(m(**t).logits.float(), Y[j].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
    dt = time.time() - t0; m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(TE_X), 128):
            t = qtok([fmt(x) for x in TE_X[i:i + 128]], padding=True, truncation=True, max_length=a.maxlen, return_tensors="pt").to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")): out.append(torch.softmax(m(**t).logits.float(), -1).cpu().numpy())
    del m; torch.cuda.empty_cache(); return np.concatenate(out), dt
for n in ROWS:
    idx = stratified(TR_Y, n); steps = max(600, 6 * ((len(idx) + 31) // 32))
    P, dt = lora_run(idx, steps); record(method="lora-qwen2.5-0.5b-tuned", rows=int(len(idx)), steps=steps, train_s=round(dt, 1), **evaluate(P, TE_Y))
log("wrote", a.out)
