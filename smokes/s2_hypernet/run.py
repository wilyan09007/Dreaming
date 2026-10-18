"""S2: weight generator (hypernetwork, mission 2d). A small set-transformer reads a step's schema (one label
description embedding per allowed output, plus k example embeddings per output if any) and writes the weights of a
linear classifier over frozen bge-small embeddings. Meta-trained on 7 CLINC150 domains; tested on 3 held-out domains,
on Banking77 (77-way) and on 12-label Banking77 sub-tasks it never saw. Baselines: label-description cosine,
label-name cosine, k-shot prototypes, desc+proto, k-shot logistic regression. Part 2: does starting a step's student
from generated weights cut the rows needed to certify?"""
import sys, json, time; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from common import *

torch.manual_seed(0); rng = np.random.default_rng(0)
OUT = ROOT / "s2_hypernet" / "RESULTS.md"; lines = []
def say(s=""): print(s, flush=True); lines.append(s)
DESC = json.load(open(CACHE / "label_descriptions.json")); DOM = json.load(open(CACHE / "clinc_domains.json"))
D = 384; KS = [0, 1, 2, 5, 10]; SEEDS = 5

# ---------------- tasks: dict(desc[K,D], name[K,D], Xs,ys (support pool = train split), Xq,yq (test split))
def task(label_ids, desc_all, name_all, Xa, ya, Xb, yb):
    lid = {l: i for i, l in enumerate(label_ids)}; ma, mb = np.isin(ya, label_ids), np.isin(yb, label_ids)
    return dict(desc=desc_all[label_ids], name=name_all[label_ids], Xs=Xa[ma], ys=np.array([lid[v] for v in ya[ma]]),
                Xq=Xb[mb], yq=np.array([lid[v] for v in yb[mb]]))

c = load_clinc150(); cn = c["names"]
c_desc = embed([DESC["clinc150"].get(n, n) for n in cn]); c_name = embed([n.replace("_", " ") for n in cn])
cXtr, cytr, cXte, cyte = embed(c["train"][0]), c["train"][1], embed(c["test"][0]), c["test"][1]
domains = sorted(set(DOM.values())); held = list(rng.choice(domains, 3, replace=False)); train_dom = [d for d in domains if d not in held]
ctasks = {d: task([i for i, n in enumerate(cn) if n != "oos" and DOM[n] == d], c_desc, c_name, cXtr, cytr, cXte, cyte) for d in domains}
b = load_banking77(); bn = b["names"]
b_desc = embed([DESC["banking77"][n] for n in bn]); b_name = embed([n.replace("_", " ") for n in bn])
bXtr, bytr, bXte, byte_ = embed(b["train"][0]), b["train"][1], embed(b["test"][0]), b["test"][1]
btask = task(list(range(77)), b_desc, b_name, bXtr, bytr, bXte, byte_)
b12 = [task(sorted(rng.choice(77, 12, replace=False).tolist()), b_desc, b_name, bXtr, bytr, bXte, byte_) for _ in range(5)]

def support(t, k, seed):
    """k examples per class from the support pool; returns proto[K,D] (zeros if k=0), and the (X,y) rows used."""
    r = np.random.default_rng(seed); K = len(t["desc"]); P = np.zeros((K, D), np.float32); rows = []
    for ci in range(K):
        idx = np.flatnonzero(t["ys"] == ci); pick = r.choice(idx, min(k, len(idx)), replace=False); rows += list(pick)
        if k: P[ci] = t["Xs"][pick].mean(0)
    return P, t["Xs"][rows], t["ys"][rows]

# ---------------- generator
class Gen(nn.Module):
    def __init__(s, h=256):
        super().__init__()
        s.inp = nn.Linear(2 * D + 2, h)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(h, 4, 2 * h, dropout=0.0, batch_first=True), 2)
        s.w, s.b = nn.Linear(h, D), nn.Linear(h, 1); s.scale = nn.Parameter(torch.tensor(10.0))
    def forward(s, desc, proto, k):  # desc/proto [K,D]; k examples per class
        K = desc.shape[0]; has = torch.full((K, 1), float(k > 0)); logk = torch.full((K, 1), float(np.log1p(k)))
        z = s.enc(s.inp(torch.cat([desc, proto, has, logk], -1))[None])[0]
        W = F.normalize(s.w(z) + desc + proto, dim=-1)  # residual on the obvious baseline (desc + proto)
        return W, s.b(z)[:, 0]
    def logits(s, X, W, bias): return s.scale * X @ W.T + bias

gen = Gen(); opt = torch.optim.Adam(gen.parameters(), 1e-3)
t0 = time.time()
for ep in range(4000):
    t = ctasks[train_dom[ep % len(train_dom)]]; K = int(rng.integers(5, 16)); cls = rng.choice(15, K, replace=False)
    sub = dict(desc=t["desc"][cls], Xs=t["Xs"], ys=t["ys"]); lid = {int(c): i for i, c in enumerate(cls)}
    k = KS[ep % len(KS)]; r = np.random.default_rng(ep); P = np.zeros((K, D), np.float32); qX, qy = [], []
    for ci, c_ in enumerate(cls):
        idx = r.permutation(np.flatnonzero(t["ys"] == c_)); P[ci] = t["Xs"][idx[:k]].mean(0) if k else 0
        qX.append(t["Xs"][idx[k:k + 10]]); qy += [ci] * len(idx[k:k + 10])
    W, bias = gen(torch.tensor(sub["desc"]), torch.tensor(P), k)
    loss = F.cross_entropy(gen.logits(torch.tensor(np.concatenate(qX)), W, bias), torch.tensor(qy))
    opt.zero_grad(); loss.backward(); opt.step()
    if ep % 1000 == 0: print(f"ep {ep} loss {loss.item():.3f}", flush=True)
train_s = time.time() - t0; gen.eval()

# ---------------- evaluation
def acc(pred, y): return float((pred == y).mean())
@torch.no_grad()
def evaluate(t, k, seed):
    P, Xk, yk = support(t, k, seed); Xq, yq = t["Xq"], t["yq"]; out = {}
    out["desc-cos"] = acc((Xq @ t["desc"].T).argmax(1), yq); out["name-cos"] = acc((Xq @ t["name"].T).argmax(1), yq)
    if k:
        out["proto"] = acc((Xq @ P.T).argmax(1), yq)
        out["desc+proto"] = acc((Xq @ (t["desc"] + P).T).argmax(1), yq)
        out["LR k-shot"] = acc(LogisticRegression(C=10, max_iter=1000).fit(Xk, yk).predict(Xq), yq)
    W, bias = gen(torch.tensor(t["desc"]), torch.tensor(P), k)
    out["GENERATOR"] = acc(gen.logits(torch.tensor(Xq), W, bias).argmax(1).numpy(), yq)
    return out

METHODS = ["desc-cos", "name-cos", "proto", "desc+proto", "LR k-shot", "GENERATOR"]
def table(title, tasks):
    say(f"\n### {title}\n"); say("| method | " + " | ".join(f"k={k}" for k in KS) + " |"); say("|---|" + "---|" * len(KS))
    res = {m: {k: [] for k in KS} for m in METHODS}
    for t in tasks:
        for k in KS:
            for s in range(SEEDS if k else 1):
                for m, v in evaluate(t, k, s).items(): res[m][k].append(v)
    for m in METHODS:
        say(f"| {m} | " + " | ".join(f"{np.mean(res[m][k]):.3f}" if res[m][k] else "-" for k in KS) + " |")
    return res

say(f"Generator: set-transformer, {sum(p.numel() for p in gen.parameters())/1e6:.2f}M params, 4000 episodes on CPU in {train_s:.0f}s. "
    f"Train domains: {train_dom}. Held-out domains: {held}. k = examples per label; accuracies on the test split, mean over {SEEDS} support draws.")
r_seen = table("Seen CLINC domains (sanity; generator trained here)", [ctasks[d] for d in train_dom])
r_held = table("Held-out CLINC domains (15-way, never seen)", [ctasks[d] for d in held])
r_b77 = table("Banking77 (77-way, never seen, different dataset)", [btask])
r_b12 = table("Banking77 12-label sub-tasks (x5, never seen)", b12)

# ---------------- part 2: rows to certify when the student starts from generated weights
say("\n### Part 2 — Banking77 student trained on n rows with an L2 prior toward W0 (lambda picked on a 20% validation slice)\n")
say("W0 = zero (plain LR) vs W0 = label-description embeddings vs W0 = GENERATOR(desc, prototypes from the same n rows). cert = SGR certificate at 5% budget, calibration = half of test, realized error on the other half.\n")
say("| rows | prior | acc | cert coverage | holdout err |"); say("|---|---|---|---|---|")
perm = np.random.default_rng(1).permutation(len(byte_)); cal, hold = perm[: len(perm) // 2], perm[len(perm) // 2:]
def fit_prior(X, y, W0, lam, steps=300):
    W = torch.tensor(W0.copy(), requires_grad=True); bb = torch.zeros(77, requires_grad=True); s = torch.tensor(10.0)
    o = torch.optim.Adam([W, bb], 0.05); Xt, yt = torch.tensor(X), torch.tensor(y)
    for _ in range(steps):
        loss = F.cross_entropy(s * Xt @ W.T + bb, yt) + lam * ((W - torch.tensor(W0)) ** 2).sum()
        o.zero_grad(); loss.backward(); o.step()
    return lambda Xn: torch.softmax(s * torch.tensor(Xn) @ W.T + bb, -1).detach().numpy()
for n in (77, 154, 385, 770, 1540):
    idx = stratified_subsample(bytr, n, seed=0); r = np.random.default_rng(n); r.shuffle(idx); nv = max(len(idx) // 5, 1)
    va, tr = idx[:nv], idx[nv:]
    k = max(1, len(tr) // 77); P, _, _ = support(dict(desc=b_desc, Xs=bXtr[tr], ys=bytr[tr]), k, 0)
    with torch.no_grad(): Wg, _ = gen(torch.tensor(b_desc), torch.tensor(P), k)
    for name, W0 in (("zero", np.zeros((77, D), np.float32)), ("desc", b_desc), ("GENERATOR", Wg.numpy())):
        best = max(((acc(fit_prior(bXtr[tr], bytr[tr], W0, lam)(bXtr[va]).argmax(1), bytr[va]), lam) for lam in (0.001, 0.01, 0.1, 1.0)))
        f = fit_prior(bXtr[idx], bytr[idx], W0, best[1]); Pq = f(bXte); ok = Pq.argmax(1) == byte_; conf = confidence(Pq)
        cert = certify(conf[cal], ok[cal], 0.05); h = apply_certificate(conf[hold], ok[hold], cert["cutoff"])
        he = "-" if h["error"] is None else f"{h['error']:.3f}"
        say(f"| {len(idx)} | {name} (lam={best[1]}) | {ok.mean():.3f} | {cert['coverage']:.2f} | {he} |")
say(f"\nWall time {time.time()-t0:.0f}s on CPU.")
write_results(OUT, "S2 — weight generator (hypernetwork) over frozen embeddings", lines)
