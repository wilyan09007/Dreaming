"""S2b: second pass after the set-transformer generator over-fit its 105 training labels (96% seen, 49% unseen).
Orthogonal ideas, all evaluated on tasks never seen in training:
  LINGEN   low-capacity generator: W_c = normalize(desc_c + A desc_c + B proto_c), A,B 384x384 zero-init, weight decay.
  METRIC   learned metric space: one projection M (init identity) applied to inputs and to (desc+proto) class vectors.
  SETTR-REG the S2 set-transformer with dropout, weight decay, 1500 episodes, early stopping on a validation domain.
  SYNTH    schema -> data instead of schema -> weights: an LLM writes 10 example queries per label from the label
           description; LR trained on them (+ k real shots). Day-zero student with no trace rows.
Part 2 (rows-to-certify with a prior) re-run with the NaN bug fixed (classes with no support rows got NaN prototypes)."""
import sys, json, time, re; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from common import *

torch.manual_seed(0); rng = np.random.default_rng(0)
OUT = ROOT / "s2_hypernet" / "RESULTS2.md"; lines = []
def say(s=""): print(s, flush=True); lines.append(s)
DESC = json.load(open(CACHE / "label_descriptions.json")); DOM = json.load(open(CACHE / "clinc_domains.json"))
D = 384; KS = [0, 1, 2, 5, 10]; SEEDS = 5

def task(label_ids, desc_all, name_all, Xa, ya, Xb, yb, names):
    lid = {l: i for i, l in enumerate(label_ids)}; ma, mb = np.isin(ya, label_ids), np.isin(yb, label_ids)
    return dict(desc=desc_all[label_ids], name=name_all[label_ids], Xs=Xa[ma], ys=np.array([lid[v] for v in ya[ma]]),
                Xq=Xb[mb], yq=np.array([lid[v] for v in yb[mb]]), labels=[names[i] for i in label_ids])
c = load_clinc150(); cn = c["names"]
c_desc = embed([DESC["clinc150"].get(n, n) for n in cn]); c_name = embed([n.replace("_", " ") for n in cn])
cXtr, cytr, cXte, cyte = embed(c["train"][0]), c["train"][1], embed(c["test"][0]), c["test"][1]
domains = sorted(set(DOM.values())); held = [str(x) for x in rng.choice(domains, 3, replace=False)]
train_dom = [d for d in domains if d not in held]; val_dom = train_dom[-1]; fit_dom = train_dom[:-1]
ctasks = {d: task([i for i, n in enumerate(cn) if n != "oos" and DOM[n] == d], c_desc, c_name, cXtr, cytr, cXte, cyte, cn) for d in domains}
b = load_banking77(); bn = b["names"]
b_desc = embed([DESC["banking77"][n] for n in bn]); b_name = embed([n.replace("_", " ") for n in bn])
bXtr, bytr, bXte, byte_ = embed(b["train"][0]), b["train"][1], embed(b["test"][0]), b["test"][1]
btask = task(list(range(77)), b_desc, b_name, bXtr, bytr, bXte, byte_, bn)
b12 = [task(sorted(rng.choice(77, 12, replace=False).tolist()), b_desc, b_name, bXtr, bytr, bXte, byte_, bn) for _ in range(5)]
DESC_OF = {**{n: DESC["clinc150"].get(n, n) for n in cn}, **DESC["banking77"]}

def support(t, k, seed):
    r = np.random.default_rng(seed); K = len(t["desc"]); P = np.zeros((K, D), np.float32); rows = []
    for ci in range(K):
        idx = np.flatnonzero(t["ys"] == ci); pick = r.choice(idx, min(k, len(idx)), replace=False); rows += list(pick)
        if k and len(pick): P[ci] = t["Xs"][pick].mean(0)
    return P, t["Xs"][rows], t["ys"][rows]

def episode(ep, doms):
    t = ctasks[doms[ep % len(doms)]]; K = int(rng.integers(5, 16)); cls = rng.choice(15, K, replace=False)
    k = KS[ep % len(KS)]; r = np.random.default_rng(ep); P = np.zeros((K, D), np.float32); qX, qy = [], []
    for ci, c_ in enumerate(cls):
        idx = r.permutation(np.flatnonzero(t["ys"] == c_)); P[ci] = t["Xs"][idx[:k]].mean(0) if k else 0
        qX.append(t["Xs"][idx[k:k + 10]]); qy += [ci] * len(idx[k:k + 10])
    return torch.tensor(t["desc"][cls]), torch.tensor(P), k, torch.tensor(np.concatenate(qX)), torch.tensor(qy)

# ---------------- generators
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
    def __init__(s, h=256, drop=0.2):
        super().__init__(); s.inp = nn.Linear(2 * D + 2, h); s.drop = nn.Dropout(drop)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(h, 4, 2 * h, dropout=drop, batch_first=True), 2)
        s.w, s.b = nn.Linear(h, D), nn.Linear(h, 1); s.scale = nn.Parameter(torch.tensor(10.0))
    def forward(s, desc, proto, k):
        K = desc.shape[0]; z = s.enc(s.drop(s.inp(torch.cat([desc, proto, torch.full((K, 1), float(k > 0)), torch.full((K, 1), float(np.log1p(k)))], -1)))[None])[0]
        return F.normalize(s.w(z) + desc + proto, dim=-1), s.b(z)[:, 0]
    def logits(s, X, W, b): return s.scale * X @ W.T + b

@torch.no_grad()
def gen_acc(g, t, k, seed):
    g.eval(); P, _, _ = support(t, k, seed); W, bias = g(torch.tensor(t["desc"]), torch.tensor(P), k)
    return float((g.logits(torch.tensor(t["Xq"]), W, bias).argmax(1).numpy() == t["yq"]).mean())

def train_gen(g, episodes, wd, patience=3):
    opt = torch.optim.AdamW(g.parameters(), 1e-3, weight_decay=wd); best, best_state, bad = -1, None, 0
    for ep in range(episodes):
        g.train(); desc, P, k, qX, qy = episode(ep, fit_dom); W, bias = g(desc, P, k)
        loss = F.cross_entropy(g.logits(qX, W, bias), qy); opt.zero_grad(); loss.backward(); opt.step()
        if ep % 250 == 249:  # early stopping on the validation domain (never in fit_dom)
            v = np.mean([gen_acc(g, ctasks[val_dom], k, 0) for k in KS])
            if v > best: best, bad, best_state = v, 0, {n: p.clone() for n, p in g.state_dict().items()}
            else: bad += 1
            if bad >= patience: break
    g.load_state_dict(best_state); return best

t0 = time.time()
gens = {}
g = LinGen(); v = train_gen(g, 4000, 1e-3); gens["LINGEN"] = g; say(f"LINGEN val acc {v:.3f}")
g = Metric(); v = train_gen(g, 4000, 1e-3); gens["METRIC"] = g; say(f"METRIC val acc {v:.3f}")
g = SetTr(); v = train_gen(g, 3000, 1e-2); gens["SETTR-REG"] = g; say(f"SETTR-REG val acc {v:.3f}")
say(f"generators trained in {time.time()-t0:.0f}s; fit domains {fit_dom}, validation domain {val_dom}, held-out {held}")

# ---------------- SYNTH: schema -> examples via a cheap LLM (cached)
SYN = CACHE / "synth_examples.json"; synth = json.loads(SYN.read_text()) if SYN.exists() else {}
def synth_examples(labels):
    need = [l for l in labels if l not in synth]
    if need:
        from openai import OpenAI
        env = (ROOT.parent / ".env").read_text()
        # track service first: ASI:One (OpenAI-compatible, asi1-mini $0.18/$0.45 per M); fall back to OpenRouter gpt-5.4-nano
        try:
            cl = OpenAI(base_url="https://api.asi1.ai/v1", api_key=re.search(r"ASI1KEY=(\S+)", env).group(1)); MODEL, PIN, POUT, EXTRA = "asi1-mini", 0.18e-6, 0.45e-6, {}
            cl.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Reply OK"}], max_tokens=3)
        except Exception as e:
            print("ASI:One unavailable, using OpenRouter:", repr(e)[:120], flush=True)
            cl = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=re.search(r"OPENROUTERKEY=(\S+)", env).group(1)); MODEL, PIN, POUT, EXTRA = "openai/gpt-5.4-nano", 0.2e-6, 1.25e-6, {"reasoning": {"effort": "minimal"}}
        cost = 0.0
        for i in range(0, len(need), 15):
            chunk = need[i:i + 15]
            prompt = ("For each label below, write 10 short, varied, realistic user messages that would get that label (different phrasings, "
                      "lengths, some typos, some indirect). Reply with a JSON object mapping label -> list of 10 strings, nothing else.\n"
                      + "\n".join(f"{l}: {DESC_OF[l]}" for l in chunk))
            for attempt in range(3):
                r = cl.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.7, max_tokens=8000, extra_body=EXTRA)
                txt = r.choices[0].message.content; cost += r.usage.prompt_tokens * PIN + r.usage.completion_tokens * POUT
                try:
                    d = json.loads(txt[txt.find("{"): txt.rfind("}") + 1]); break
                except Exception: d = {}
            for l in chunk: synth[l] = [str(x) for x in d.get(l, [])][:10] or [DESC_OF[l]]
            print(f"synth {i+len(chunk)}/{len(need)} labels, ${cost:.3f}", flush=True)
        SYN.write_text(json.dumps(synth, indent=1), encoding="utf-8")
        with open(ROOT / "COSTS.md", "a", encoding="utf-8") as f: f.write(f"s2b_synth,{cost:.4f},10 synthetic examples per label via gpt-5.4-nano\n")
    return {l: synth[l] for l in labels}

def synth_acc(t, k, seed):
    ex = synth_examples(t["labels"]); X = [s for l in t["labels"] for s in ex[l]]; y = [i for i, l in enumerate(t["labels"]) for _ in ex[l]]
    E = embed(X); P, Xk, yk = support(t, k, seed)
    if k: E, y = np.concatenate([E, Xk]), list(y) + list(yk)
    clf = LogisticRegression(C=10, max_iter=2000).fit(E, y); return float((clf.predict(t["Xq"]) == t["yq"]).mean())

def evaluate(t, k, seed):
    P, Xk, yk = support(t, k, seed); Xq, yq = t["Xq"], t["yq"]; out = {}
    out["desc-cos"] = float(((Xq @ t["desc"].T).argmax(1) == yq).mean()); out["name-cos"] = float(((Xq @ t["name"].T).argmax(1) == yq).mean())
    if k:
        out["proto"] = float(((Xq @ P.T).argmax(1) == yq).mean()); out["desc+proto"] = float(((Xq @ (t["desc"] + P).T).argmax(1) == yq).mean())
        out["LR k-shot"] = float((LogisticRegression(C=10, max_iter=1000).fit(Xk, yk).predict(Xq) == yq).mean())
    for n, g in gens.items(): out[n] = gen_acc(g, t, k, seed)
    out["SYNTH(+k real)"] = synth_acc(t, k, seed)
    return out
METHODS = ["desc-cos", "name-cos", "proto", "desc+proto", "LR k-shot", "LINGEN", "METRIC", "SETTR-REG", "SYNTH(+k real)"]
def table(title, tasks):
    say(f"\n### {title}\n"); say("| method | " + " | ".join(f"k={k}" for k in KS) + " |"); say("|---|" + "---|" * len(KS))
    res = {m: {k: [] for k in KS} for m in METHODS}
    for t in tasks:
        for k in KS:
            for s in range(SEEDS if k else 1):
                for m, v in evaluate(t, k, s).items(): res[m][k].append(v)
    for m in METHODS: say(f"| {m} | " + " | ".join(f"{np.mean(res[m][k]):.3f}" if res[m][k] else "-" for k in KS) + " |")
say("Accuracy on the test split, mean over 5 support draws; k = real examples per label. All tasks below were never used to fit a generator.")
table("Held-out CLINC domains (15-way)", [ctasks[d] for d in held])
table("Banking77 (77-way, different dataset)", [btask])
table("Banking77 12-label sub-tasks (x5)", b12)

# ---------------- Part 2 (fixed): rows to certify with a prior
say("\n### Part 2 — Banking77 student on n rows with an L2 prior toward W0 (NaN bug fixed; lambda picked on a 20% validation slice)\n")
say("| rows | prior | acc | cert coverage @5% | holdout err |"); say("|---|---|---|---|---|")
perm = np.random.default_rng(1).permutation(len(byte_)); cal, hold = perm[: len(perm) // 2], perm[len(perm) // 2:]
def fit_prior(X, y, W0, lam, steps=300):
    W = torch.tensor(W0.copy(), requires_grad=True); bb = torch.zeros(77, requires_grad=True); s = torch.tensor(10.0)
    o = torch.optim.Adam([W, bb], 0.05); Xt, yt = torch.tensor(X), torch.tensor(y)
    for _ in range(steps):
        loss = F.cross_entropy(s * Xt @ W.T + bb, yt) + lam * ((W - torch.tensor(W0)) ** 2).sum(); o.zero_grad(); loss.backward(); o.step()
    return lambda Xn: torch.softmax(s * torch.tensor(Xn) @ W.T + bb, -1).detach().numpy()
synth_E = embed([s for l in bn for s in synth_examples(bn)[l]]); synth_y = np.array([i for i, l in enumerate(bn) for _ in synth[l]])
for n in (77, 154, 385, 770, 1540):
    idx = stratified_subsample(bytr, n, seed=0); r = np.random.default_rng(n); r.shuffle(idx); nv = max(len(idx) // 5, 1); va, tr = idx[:nv], idx[nv:]
    k = max(1, len(tr) // 77); P, _, _ = support(dict(desc=b_desc, Xs=bXtr[tr], ys=bytr[tr]), k, 0)
    with torch.no_grad(): Wl, _ = gens["LINGEN"](torch.tensor(b_desc), torch.tensor(P), k)
    Wsyn = LogisticRegression(C=10, max_iter=2000).fit(synth_E, synth_y).coef_.astype(np.float32); Wsyn = Wsyn / np.linalg.norm(Wsyn, axis=1, keepdims=True)
    for name, W0 in (("zero", np.zeros((77, D), np.float32)), ("desc", b_desc), ("LINGEN", Wl.numpy()), ("SYNTH-LR", Wsyn)):
        assert np.isfinite(W0).all(), name
        best = max(((float((fit_prior(bXtr[tr], bytr[tr], W0, lam)(bXtr[va]).argmax(1) == bytr[va]).mean()), lam) for lam in (0.001, 0.01, 0.1, 1.0)))
        Pq = fit_prior(bXtr[idx], bytr[idx], W0, best[1])(bXte); ok = Pq.argmax(1) == byte_; conf = confidence(Pq)
        cert = certify(conf[cal], ok[cal], 0.05); h = apply_certificate(conf[hold], ok[hold], cert["cutoff"]); he = "-" if h["error"] is None else f"{h['error']:.3f}"
        say(f"| {len(idx)} | {name} (lam={best[1]}) | {ok.mean():.3f} | {cert['coverage']:.2f} | {he} |")
# synthetic rows alone + n real rows, plain LR (the simplest day-zero recipe)
say("\n### Synthetic-example student (LR on 770 synthetic rows + n real rows, Banking77)\n"); say("| real rows | acc | cert coverage @5% | holdout err |"); say("|---|---|---|---|")
for n in (0, 77, 154, 385, 770, 1540):
    idx = stratified_subsample(bytr, n, seed=0) if n else np.array([], int)
    E = np.concatenate([synth_E, bXtr[idx]]) if n else synth_E; y = np.concatenate([synth_y, bytr[idx]]) if n else synth_y
    Pq = LogisticRegression(C=10, max_iter=2000).fit(E, y).predict_proba(bXte); ok = Pq.argmax(1) == byte_; conf = confidence(Pq)
    cert = certify(conf[cal], ok[cal], 0.05); h = apply_certificate(conf[hold], ok[hold], cert["cutoff"]); he = "-" if h["error"] is None else f"{h['error']:.3f}"
    say(f"| {n} | {ok.mean():.3f} | {cert['coverage']:.2f} | {he} |")
say(f"\nWall time {time.time()-t0:.0f}s on CPU.")
write_results(OUT, "S2b — weight generator, second pass (low-capacity generators, metric learning, synthetic examples)", lines)
