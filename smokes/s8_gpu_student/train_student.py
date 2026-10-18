"""S8 (+S7 GPU part): LoRA students on a small causal LM with a classification head (all labels scored in ONE forward pass).
A. per-step student (2a) on Banking77 at several row counts: accuracy, certified coverage, time/throughput/memory.
B. shared workflow student (2b): one LoRA trained on 7 CLINC150 domains (step name in the prompt), then a NEW step
   (held-out domain) fine-tuned from the shared LoRA vs from the base model, at tiny row counts -> rows needed.
C. determinism: 1000 batch-1 forward passes under 8 concurrent threads, and the same input inside random batches.
Usage: python train_student.py --model Qwen/Qwen2.5-0.5B --out results.json [--tiny]
"""
import argparse, json, os, time, threading, hashlib
import numpy as np, torch, torch.nn.functional as F
from scipy.stats import beta as _beta

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-0.5B"); p.add_argument("--out", default="results.json")
p.add_argument("--tiny", action="store_true", help="CPU pipeline check: tiny slices, 1 epoch")
p.add_argument("--bs", type=int, default=32); p.add_argument("--lr", type=float, default=2e-4); p.add_argument("--maxlen", type=int, default=64)
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
R = {"model": a.model, "device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "tiny": a.tiny}
def log(*x): print(*x, flush=True)

# ---------------- data
import datasets
b = datasets.load_dataset("mteb/banking77"); b_names = [None] * 77
for y, s in zip(b["train"]["label"], b["train"]["label_text"]): b_names[y] = s
B_TR = (b["train"]["text"], np.array(b["train"]["label"])); B_TE = (b["test"]["text"], np.array(b["test"]["label"]))
c = datasets.load_dataset("clinc/clinc_oos", "plus"); cn = c["train"].features["intent"].names
DOM = {}  # intent -> domain from the clinc-domain dataset (majority vote over shared texts)
w = datasets.load_dataset("willcb/clinc-domain")["train"]; d_of = dict(zip(w["input"], w["label"]))
from collections import Counter, defaultdict
votes = defaultdict(Counter)
for t, y in zip(c["train"]["text"], c["train"]["intent"]):
    if t in d_of: votes[cn[y]][d_of[t]] += 1
DOM = {k: v.most_common(1)[0][0] for k, v in votes.items()}
HELD = ["travel", "work"]; SHARED = sorted({d for d in DOM.values() if d not in HELD})
def clinc_split(split, doms):
    xs, ys, steps = [], [], []
    for t, y in zip(c[split]["text"], c[split]["intent"]):
        n = cn[y]
        if n != "oos" and DOM[n] in doms: xs.append(t); ys.append(n); steps.append(DOM[n])
    return xs, ys, steps

def stratified(labels, n, seed=0):
    r = np.random.default_rng(seed); labels = np.asarray(labels); idx = r.permutation(len(labels)); cls = np.unique(labels)
    first = [int(idx[labels[idx] == k][0]) for k in cls] if n >= len(cls) else []
    fs = set(first); rest = [int(i) for i in idx if int(i) not in fs]
    return np.array(first + rest[: max(0, n - len(first))])

# ---------------- model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "right"

def make(num_labels, lora_from=None):
    m = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=num_labels, torch_dtype=torch.float32)
    m.config.pad_token_id = tok.pad_token_id
    cfg = LoraConfig(task_type="SEQ_CLS", r=16, lora_alpha=32, lora_dropout=0.05,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], modules_to_save=["score"])
    m = get_peft_model(m, cfg)
    if lora_from is not None:  # copy LoRA (not the head) from a trained shared model
        sd = {k: v for k, v in lora_from.state_dict().items() if "lora_" in k}
        missing = m.load_state_dict(sd, strict=False); assert not [k for k in missing.unexpected_keys], missing.unexpected_keys
    return m.to(dev)

def enc(texts):
    return tok(texts, padding=True, truncation=True, max_length=a.maxlen, return_tensors="pt").to(dev)

def train(m, texts, y, epochs, min_steps=0):
    m.train(); opt = torch.optim.AdamW([q for q in m.parameters() if q.requires_grad], lr=a.lr)
    y = torch.tensor(y); n = len(texts); steps = max(min_steps, epochs * ((n + a.bs - 1) // a.bs)); t0 = time.time(); seen = 0; step = 0
    if dev == "cuda": torch.cuda.reset_peak_memory_stats()
    while step < steps:
        perm = torch.randperm(n)
        for i in range(0, n, a.bs):
            if step >= steps: break
            idx = perm[i:i + a.bs]; batch = enc([texts[j] for j in idx])
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
                loss = F.cross_entropy(m(**batch).logits.float(), y[idx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); step += 1; seen += len(idx)
    dt = time.time() - t0
    return dict(train_s=round(dt, 1), samples_per_s=round(seen / dt, 1), steps=steps,
                peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev == "cuda" else None)

@torch.no_grad()
def predict(m, texts, bs=128):
    m.eval(); out = []
    for i in range(0, len(texts), bs):
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            out.append(torch.softmax(m(**enc(texts[i:i + bs])).logits.float(), -1).cpu().numpy())
    return np.concatenate(out)

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

def evaluate(m, texts, y):
    P = predict(m, texts); pred = P.argmax(1); ok = pred == y; s = np.sort(P, 1); conf = s[:, -1] - s[:, -2]
    r = np.random.default_rng(1); perm = r.permutation(len(y)); cal, hold = perm[: len(y) // 2], perm[len(y) // 2:]
    cut, cov = certify(conf[cal], ok[cal]); above = conf[hold] >= cut
    return dict(acc=round(float(ok.mean()), 4), cert5_cutoff=cut if cut != float("inf") else None, cert5_coverage=round(cov, 3),
                holdout_err_above=round(float((~ok[hold][above]).mean()), 4) if above.any() else None)

# ---------------- A. per-step student on Banking77
fmt = lambda t: f"Step: categorize bank support query\nQuery: {t}\nLabel:"
rows = [77, 770] if a.tiny else [770, 3080, 9993]; epochs = 1 if a.tiny else 3
btr_x = [fmt(t) for t in B_TR[0]]; bte_x = [fmt(t) for t in B_TE[0]]; bte_y = B_TE[1]
if a.tiny: bte_x, bte_y = bte_x[:200], bte_y[:200]
R["A_per_step"] = []
for n in rows:
    idx = stratified(B_TR[1], n); m = make(77)
    tr = train(m, [btr_x[i] for i in idx], B_TR[1][idx], epochs, min_steps=60)
    ev = evaluate(m, bte_x, bte_y); R["A_per_step"].append(dict(rows=int(len(idx)), epochs=epochs, **tr, **ev)); log("A", R["A_per_step"][-1])
    last = m
# batch-1 latency
x1 = enc([bte_x[0]]); m = last; m.eval()
with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
    for _ in range(10): m(**x1)
    if dev == "cuda": torch.cuda.synchronize()
    t0 = time.time(); N = 100 if dev == "cuda" else 10
    for _ in range(N): m(**x1)
    if dev == "cuda": torch.cuda.synchronize()
R["batch1_latency_ms"] = round((time.time() - t0) / N * 1000, 2); log("batch-1 latency ms", R["batch1_latency_ms"])

# ---------------- C. determinism on the Banking77 student
def logits_bytes(m, texts, i=0):
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
        return m(**enc(texts)).logits[i].float().cpu().numpy().tobytes()
probe = bte_x[0]; others = bte_x[1:9]
# C1: batch-1, 1000 passes from 8 threads concurrently (each thread on its own CUDA stream)
seen = set(); lock = threading.Lock()
def worker(k):
    stream = torch.cuda.Stream() if dev == "cuda" else None
    for _ in range(1000 // 8 if not a.tiny else 5):
        if stream is not None:
            with torch.cuda.stream(stream): h = hashlib.sha1(logits_bytes(m, [probe])).hexdigest()
        else: h = hashlib.sha1(logits_bytes(m, [probe])).hexdigest()
        with lock: seen.add(h)
ths = [threading.Thread(target=worker, args=(k,)) for k in range(8)]; [t.start() for t in ths]; [t.join() for t in ths]
R["det_batch1_concurrent_distinct"] = len(seen)
# C2: same input as row 0 of batches of random size 1..8 (padding changes with the batch)
seen2 = set(); r = np.random.default_rng(0)
for _ in range(200 if not a.tiny else 10):
    k = int(r.integers(1, 9)); seen2.add(hashlib.sha1(logits_bytes(m, [probe] + list(r.choice(others, k - 1)))).hexdigest())
R["det_random_batch_distinct"] = len(seen2)
# C3: same but padded to a fixed length (fixed input shape) -> does fixing the shape restore determinism?
seen3 = set()
old = tok.padding_side
for _ in range(200 if not a.tiny else 10):
    k = int(r.integers(1, 9)); batch = tok([probe] + list(r.choice(others, k - 1)), padding="max_length", max_length=a.maxlen, truncation=True, return_tensors="pt").to(dev)
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
        seen3.add(hashlib.sha1(m(**batch).logits[0].float().cpu().numpy().tobytes()).hexdigest())
R["det_random_batch_fixed_shape_distinct"] = len(seen3)
log("determinism distinct outputs: batch1-concurrent", len(seen), "random-batch", len(seen2), "random-batch-fixed-shape", len(seen3))
del m, last; torch.cuda.empty_cache() if dev == "cuda" else None

# ---------------- B. shared workflow student (2b) vs per-step (2a) on a NEW step
sx, sy, ss = clinc_split("train", SHARED); labels = sorted(set(sy)); lid = {l: i for i, l in enumerate(labels)}
fmt2 = lambda step, t: f"Step: {step}\nQuery: {t}\nLabel:"
sx_f = [fmt2(s, t) for s, t in zip(ss, sx)]; sy_i = np.array([lid[l] for l in sy])
if a.tiny: keep = stratified(sy_i, 2 * len(labels)); sx_f = [sx_f[i] for i in keep]; sy_i = sy_i[keep]
shared = make(len(labels)); R["B_shared_train"] = train(shared, sx_f, sy_i, 1 if a.tiny else 2); log("B shared", R["B_shared_train"], "labels", len(labels))
R["B_new_step"] = []
for dom in HELD[:1] if a.tiny else HELD:
    tx, ty, _ = clinc_split("train", [dom]); ex, ey, _ = clinc_split("test", [dom]); dl = sorted(set(ty)); dmap = {l: i for i, l in enumerate(dl)}
    tx_f = [fmt2(dom, t) for t in tx]; ty_i = np.array([dmap[l] for l in ty]); ex_f = [fmt2(dom, t) for t in ex]; ey_i = np.array([dmap[l] for l in ey])
    for n in ([15, 45] if a.tiny else [15, 45, 150, 450, 1500]):
        idx = stratified(ty_i, n)
        for init in ("base", "shared"):
            m = make(len(dl), lora_from=shared if init == "shared" else None)
            tr = train(m, [tx_f[i] for i in idx], ty_i[idx], 3, min_steps=60); ev = evaluate(m, ex_f, ey_i)
            R["B_new_step"].append(dict(step=dom, rows=int(len(idx)), init=init, **tr, **ev)); log("B", R["B_new_step"][-1]); del m
json.dump(R, open(a.out, "w"), indent=1); log("wrote", a.out)
