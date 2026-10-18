"""S7 (CPU part): is a hosted student byte-for-byte replayable on CPU? Student = bge-small embedding -> logistic
regression (the S1 student). We hash the embedding bytes and the probability vector across: 1000 batch-1 calls from
8 concurrent threads; the same text inside random batches (padding differs); different torch thread counts.
GPU counterpart runs inside s8_gpu_student/train_student.py."""
import sys, time, hashlib, threading; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, torch
from sentence_transformers import SentenceTransformer
from common import *

lines = []
def say(s=""): print(s, flush=True); lines.append(s)
m = SentenceTransformer(EMB_MODEL, device="cpu")
b = load_banking77(); texts = b["test"][0]; probe = texts[0]; others = texts[1:9]
def h(x): return hashlib.sha1(np.ascontiguousarray(x, dtype=np.float32).tobytes()).hexdigest()
def emb(ts): return m.encode(list(ts), batch_size=len(ts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

t0 = time.time()
# 1. batch-1, 8 threads, 1000 calls
seen, lock = set(), threading.Lock()
def worker():
    for _ in range(125):
        v = h(emb([probe])[0])
        with lock: seen.add(v)
ths = [threading.Thread(target=worker) for _ in range(8)]; [t.start() for t in ths]; [t.join() for t in ths]
say(f"1000 batch-1 calls from 8 threads: distinct embeddings = **{len(seen)}** ({time.time()-t0:.0f}s)")
# 2. same text as row 0 of random batches (dynamic padding)
r = np.random.default_rng(0); seen2 = set()
for _ in range(200):
    k = int(r.integers(1, 9)); seen2.add(h(emb([probe] + list(r.choice(others, k - 1)))[0]))
say(f"200 random batches (size 1-8, dynamic padding): distinct embeddings of the probe = **{len(seen2)}**")
# 3. thread count
ref = h(emb([probe])[0]); out = {}
for n in (1, 2, 4, 8):
    torch.set_num_threads(n); out[n] = h(emb([probe])[0]) == ref
say(f"torch.set_num_threads 1/2/4/8 vs reference: identical = {out}")
# 4. how different are the 'different' outputs? (max abs diff across the distinct variants)
vs = [emb([probe] + list(r.choice(others, k)))[0] for k in range(0, 8)]
say(f"max abs diff of probe embedding across batch sizes 1..8: {max(np.abs(v - vs[0]).max() for v in vs):.2e}")
# 5. downstream: does the LR decision/confidence change? (fit S1 student, compare argmax and top-2 gap)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=10, max_iter=2000).fit(embed(b["train"][0]), b["train"][1])
P = np.stack([clf.predict_proba(v[None])[0] for v in vs]); gaps = confidence(P)
say(f"LR argmax identical across batch variants: {len(set(P.argmax(1)))==1}; top-2 gap spread: {gaps.max()-gaps.min():.2e}; distinct proba byte-strings: {len({h(p) for p in P})}")
say(f"\nWall {time.time()-t0:.0f}s. Interpretation: with batch size 1 and a fixed thread count the CPU path is byte-identical; "
    f"dynamic padding (batch composition) and thread count are the two things that change bytes, so the runtime must pin both "
    f"(fixed input shape, fixed intra-op threads) or hash the environment including them.")
write_results(ROOT / "s7_determinism" / "RESULTS.md", "S7 — determinism / replayability (CPU)", lines)
