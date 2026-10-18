"""S22: learn-from-less architectures for a NEW step, one shared protocol (Qwen2.5-0.5B seq-cls head, bf16, bs 32, max_len 64,
steps = min(600, 10 epochs)). Arms: A LoRA from base | B LoRA from shared pool LoRA | C LoReFT (pyreft, else hand-written) |
D LoraHub-style composition of per-step pool LoRAs (comp only / comp + 100-step LoRA ft) | E1 SimCSE on the unlabelled stream then A |
E2 self-training on the unlabelled stream | F bge-small + LogisticRegression | A_s8/B_s8 = the S8 phase-B recipe (replication).
Usage: python s22_arch.py --tasks travel,work --out results.json [--tiny]   (tiny = CPU pipeline check with a random 4-layer Qwen2)."""
import argparse, copy, gc, json, math, sys, time, traceback
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import beta as _beta

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-0.5B"); p.add_argument("--out", default="results.json")
p.add_argument("--tasks"); p.add_argument("--ks"); p.add_argument("--seeds"); p.add_argument("--s8ks")
p.add_argument("--tiny", action="store_true"); p.add_argument("--hand", action="store_true", help="skip pyreft, use the hand-written LoReFT")
p.add_argument("--bs", type=int, default=32); p.add_argument("--maxlen", type=int, default=64)
p.add_argument("--lr", type=float, default=2e-4, help="LoRA"); p.add_argument("--hlr", type=float, default=2e-3, help="head")
p.add_argument("--rlr", type=float, default=2e-3, help="ReFT"); p.add_argument("--wlr", type=float, default=5e-2, help="mixing logits")
a = p.parse_args()
if a.tiny: a.model = "tiny"
a.tasks = a.tasks or ("travel,bank0" if a.tiny else "travel,work,bank0,bank1,bank2,bank3,bank4"); a.ks = a.ks or ("1,3" if a.tiny else "1,3,10,30")
a.seeds = a.seeds or ("0" if a.tiny else "0,1,2"); a.s8ks = a.s8ks or ("3" if a.tiny else "3,10")
dev = "cuda" if torch.cuda.is_available() else "cpu"
assert dev == "cuda" or a.tiny, "no CUDA on this pod: abort instead of crawling on CPU"
KS, SEEDS, S8KS = [[int(x) for x in s.split(",") if x] for s in (a.ks, a.seeds, a.s8ks)]
POOL_STEPS = {"clinc": 750, "bank": 600}; AD_STEPS, FT_STEPS, SIM_STEPS, SIM_LR, SIM_DROP = 150, 100, 300, 5e-4, 0.1
RANK, NPOS = 4, 3
if a.tiny: POOL_STEPS = {"clinc": 3, "bank": 3}; AD_STEPS = FT_STEPS = SIM_STEPS = 2
T0 = time.time()
R = dict(args=vars(a), gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", records=[], pools={}, errors=[], reft={})
def log(*x): print(f"[{time.time()-T0:.0f}s]", *x, flush=True)
def save(): json.dump(R, open(a.out, "w"))
def sync():
    if dev == "cuda": torch.cuda.synchronize()

# ---------------- data (copied from train_student.py)
import datasets
from collections import Counter, defaultdict
b = datasets.load_dataset("mteb/banking77")
B_TR = (list(b["train"]["text"]), np.array(b["train"]["label"])); B_TE = (list(b["test"]["text"]), np.array(b["test"]["label"]))
c = datasets.load_dataset("clinc/clinc_oos", "plus"); cn = c["train"].features["intent"].names
w = datasets.load_dataset("willcb/clinc-domain")["train"]; d_of = dict(zip(w["input"], w["label"]))
votes = defaultdict(Counter)
for t, y in zip(c["train"]["text"], c["train"]["intent"]):
    if t in d_of: votes[cn[y]][d_of[t]] += 1
DOM = {k: v.most_common(1)[0][0] for k, v in votes.items()}
HELD = ["travel", "work"]; SHARED = sorted({d for d in DOM.values() if d not in HELD})
def clinc_split(split, doms):
    xs, ys, st = [], [], []
    for t, y in zip(c[split]["text"], c[split]["intent"]):
        n = cn[y]
        if n != "oos" and DOM[n] in doms: xs.append(t); ys.append(n); st.append(DOM[n])
    return xs, ys, st
fmt = lambda step, t: f"Step: {step}\nQuery: {t}\nLabel:"

def mk(name):
    if name in HELD:
        tx, ty, _ = clinc_split("train", [name]); ex, ey, _ = clinc_split("test", [name]); mp = {l: i for i, l in enumerate(sorted(set(ty)))}
        t = dict(step=name, pool="clinc", trx=tx, tr_y=np.array([mp[l] for l in ty]), tex=ex, te_y=np.array([mp[l] for l in ey]))
    else:  # five 12-label Banking77 sub-tasks, seed-0 draws (same style as S2b)
        j = int(name[4:]); rg = np.random.default_rng(0); labs = [sorted(rg.choice(77, 12, replace=False).tolist()) for _ in range(5)][j]; mp = {l: i for i, l in enumerate(labs)}
        t = dict(step=f"bank task {j}", pool=name, labs=labs, trx=[x for x, y in zip(*B_TR) if y in mp], tr_y=np.array([mp[y] for y in B_TR[1] if y in mp]),
                 tex=[x for x, y in zip(*B_TE) if y in mp], te_y=np.array([mp[y] for y in B_TE[1] if y in mp]))
    if a.tiny:  # keep <=12 train and <=4 test rows per label
        keep = lambda y, n: np.concatenate([np.flatnonzero(y == l)[:n] for l in range(len(mp))])
        i, e = keep(t["tr_y"], 12), keep(t["te_y"], 4)
        t.update(trx=[t["trx"][q] for q in i], tr_y=t["tr_y"][i], tex=[t["tex"][q] for q in e], te_y=t["te_y"][e])
    t.update(name=name, L=len(mp), fam="clinc" if name in HELD else "bank")
    t["X"] = [fmt(t["step"], x) for x in t["trx"]]; t["TX"] = [fmt(t["step"], x) for x in t["tex"]]
    return t

_perm = {}
def sample_idx(t, seed, k):  # nested across k: first k of a per-label permutation
    if (t["name"], seed) not in _perm:
        r = np.random.default_rng(100 + seed); _perm[(t["name"], seed)] = [r.permutation(np.flatnonzero(t["tr_y"] == l)) for l in range(t["L"])]
    return np.concatenate([q[:k] for q in _perm[(t["name"], seed)]])

# ---------------- model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Qwen2Config
from peft import LoraConfig, get_peft_model
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B" if a.tiny else a.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "right"
if a.tiny:
    BASE = AutoModelForSequenceClassification.from_config(Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                                                        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128))
else:
    try: BASE = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2, dtype=torch.float32)
    except TypeError: BASE = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2, torch_dtype=torch.float32)
BASE.config.pad_token_id = tok.pad_token_id; BASE = BASE.to(dev)
H, NL = BASE.config.hidden_size, BASE.config.num_hidden_layers
REFT_LAYERS = [int(NL * f) for f in (0.125, 0.375, 0.625, 0.875)]
LORA = lambda: LoraConfig(task_type="SEQ_CLS", r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], modules_to_save=["score"])

def new_net(L, seed, lora=True, init_sd=None):
    torch.manual_seed(seed); m = copy.deepcopy(BASE); m.score = nn.Linear(H, L, bias=False).to(dev); nn.init.normal_(m.score.weight, std=0.02); m.config.num_labels = L
    if not lora: return m
    m = get_peft_model(m, LORA())
    if init_sd is not None: miss = m.load_state_dict(init_sd, strict=False); assert not miss.unexpected_keys, miss.unexpected_keys
    return m
lora_sd = lambda m: {k: v.detach().clone() for k, v in m.state_dict().items() if "lora_" in k}
nparams = lambda m: sum(q.numel() for q in m.parameters() if q.requires_grad)
def groups(m, lr=None):  # LoRA params at lr, everything else trainable (the head) at hlr
    ps = [(n, q) for n, q in m.named_parameters() if q.requires_grad]
    g = [dict(params=[q for n, q in ps if "lora_" in n], lr=lr or a.lr), dict(params=[q for n, q in ps if "lora_" not in n], lr=a.hlr)]
    return [x for x in g if x["params"]]
enc = lambda texts: tok(texts, padding=True, truncation=True, max_length=a.maxlen, return_tensors="pt").to(dev)
def lg(net, bt):
    o = net(**bt); return o.logits if hasattr(o, "logits") else o
ac = lambda: torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda"))
steps_for = lambda n: min(600, 10 * math.ceil(n / a.bs))

def fit(net, X, y, steps, grp):
    net.train(); opt = torch.optim.AdamW(grp); y = torch.as_tensor(np.asarray(y)); n = len(X); step = 0; sync(); t0 = time.time()
    while step < steps:
        perm = torch.randperm(n)
        for i in range(0, n, a.bs):
            if step >= steps: break
            idx = perm[i:i + a.bs]
            with ac(): loss = F.cross_entropy(lg(net, enc([X[j] for j in idx.tolist()])).float(), y[idx].to(dev))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1
    sync(); return time.time() - t0

@torch.no_grad()
def probs(net, X, bs=128):
    net.eval(); out = []
    for i in range(0, len(X), bs):
        with ac(): out.append(torch.softmax(lg(net, enc(X[i:i + bs])).float(), -1).cpu().numpy())
    return np.concatenate(out)

# ---------------- certificates: train_student.py's search (cert_sgr) and common.py's fixed grid (cert_grid), both Clopper-Pearson
def cp_upper(e, n, d): return 1.0 if n == 0 or e >= n else float(_beta.ppf(1 - d, e + 1, n - e))
def cert_sgr(conf, ok, budget=0.05, delta=0.05):
    o = np.argsort(-conf); ok = ok[o]; conf = conf[o]; n = len(conf); m = int(np.ceil(np.log2(n))) + 1; d = delta / m
    ce = np.cumsum(~ok); lo, hi, best = 1, n, None
    for _ in range(m):
        if lo > hi: break
        k = (lo + hi) // 2
        if cp_upper(int(ce[k - 1]), k, d) <= budget: best, lo = k, k + 1
        else: hi = k - 1
    return (float("inf"), 0.0) if best is None else (float(conf[best - 1]), best / n)
def cert_grid(conf, ok, budget=0.05, delta=0.05, grid=40):
    o = np.argsort(-conf, kind="stable"); ok = ok[o]; conf = conf[o]; n = len(conf); ce = np.cumsum(~ok)
    for j in range(grid, 0, -1):
        k = max(1, int(round(n * j / grid)))
        while k < n and conf[k] == conf[k - 1]: k += 1
        if cp_upper(int(ce[k - 1]), k, delta / grid) <= budget: return float(conf[k - 1]), k / n
    return float("inf"), 0.0
def metrics(P, y):  # certify on a random half of the test rows, apply on the other half
    ok = P.argmax(1) == y; s = np.sort(P, 1); conf = s[:, -1] - s[:, -2]
    perm = np.random.default_rng(1).permutation(len(y)); cal, hold = perm[: len(y) // 2], perm[len(y) // 2:]
    out = dict(acc=round(float(ok.mean()), 4))
    for nm, fn in (("sgr", cert_sgr), ("grid", cert_grid)):
        cut, cov = fn(conf[cal], ok[cal]); ab = conf[hold] >= cut
        out.update({f"cov_{nm}": round(cov, 3), f"hold_cov_{nm}": round(float(ab.mean()), 3), f"hold_err_{nm}": round(float((~ok[hold][ab]).mean()), 4) if ab.any() else None})
    return out

# ---------------- arm C: LoReFT. pyreft if it works (probe below), else a hand-written LoReFT-style edit of the last NPOS token states at REFT_LAYERS
def last_pos(mask): return (mask.sum(1)[:, None] - 1 - torch.arange(NPOS - 1, -1, -1, device=mask.device)[None]).clamp(min=0)
class HandReft(nn.Module):  # h' = h + R^T(Wh + b - Rh), R (r x d) with orthonormal rows; per layer; edits the last NPOS positions
    def __init__(s, base):
        super().__init__(); s.base = base; [q.requires_grad_(False) for q in base.parameters()]; base.score.weight.requires_grad_(True)
        s.R = nn.ModuleList([nn.utils.parametrizations.orthogonal(nn.Linear(H, RANK, bias=False)) for _ in REFT_LAYERS]).to(dev)
        s.W = nn.ModuleList([nn.Linear(H, RANK) for _ in REFT_LAYERS]).to(dev)
        for i, l in enumerate(REFT_LAYERS): base.model.layers[l].register_forward_hook(s._hook(i))
    def _hook(s, i):
        def f(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            with torch.autocast(dev, enabled=False):
                hs = h[s.bi, s.pos].float(); Rw = s.R[i].weight; new = hs + (s.W[i](hs) - hs @ Rw.T) @ Rw
                h2 = h.clone(); h2[s.bi, s.pos] = new.to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        return f
    def forward(s, input_ids, attention_mask):
        s.pos = last_pos(attention_mask); s.bi = torch.arange(len(s.pos), device=s.pos.device)[:, None]
        return s.base(input_ids=input_ids, attention_mask=attention_mask).logits
class PyReft(nn.Module):
    def __init__(s, base):
        super().__init__(); import pyreft
        reps = [dict(layer=l, component="block_output", low_rank_dimension=RANK, intervention=pyreft.LoreftIntervention(embed_dim=H, low_rank_dimension=RANK, dtype=torch.float32)) for l in REFT_LAYERS]
        s.rm = pyreft.get_reft_model(base, pyreft.ReftConfig(representations=reps), set_device=True, disable_model_grads=True); s.rm.float()
        base.score.weight.requires_grad_(True)
    def forward(s, input_ids, attention_mask):
        loc = last_pos(attention_mask)[None].expand(len(REFT_LAYERS), -1, -1).tolist()
        return s.rm({"input_ids": input_ids, "attention_mask": attention_mask}, unit_locations={"sources->base": (None, loc)})[1].logits
REFT_IMPL = "hand"
def reft_net(L, seed, impl=None):
    base = new_net(L, seed, lora=False); n = (PyReft if (impl or REFT_IMPL) == "pyreft" else HandReft)(base)
    ps = [(nm, q) for nm, q in n.named_parameters() if q.requires_grad]
    return n, [dict(params=[q for nm, q in ps if "score" not in nm], lr=a.rlr), dict(params=[q for nm, q in ps if "score" in nm], lr=a.hlr)]
def probe_pyreft():
    sys.path.append("/workspace/reftlibs")
    n, g = reft_net(3, 0, "pyreft"); bt = enc([fmt("x", "hello there"), fmt("x", "a rather longer query about my money")])
    with ac(): out = lg(n, bt).float(); ref = n.rm.model(**bt).logits.float()
    assert torch.isfinite(out).all() and (out - ref).abs().max() > 1e-4, "pyreft intervention did not change the logits"
    out.sum().backward(); gn = sum(float(q.grad.abs().sum()) for nm, q in n.named_parameters() if q.grad is not None and "score" not in nm and "rm.model" not in nm)
    assert gn > 0, "no gradient reached the pyreft intervention parameters"
    return nparams(n)

# ---------------- arm D: LoraHub-style composition. All pool LoRAs share the A init (seed 777); mix = softmax(w) over the K adapters, applied to A and B
class Comp(nn.Module):
    def __init__(s, pm, stacks):
        super().__init__(); s.pm = pm; s.st = stacks; s.w = nn.Parameter(torch.zeros(len(next(iter(stacks.values()))), device=dev))
        for n, q in pm.named_parameters():
            if "lora_" in n: q.requires_grad_(False)
    def hats(s): wt = torch.softmax(s.w, 0); return {n: torch.tensordot(wt, t, dims=1) for n, t in s.st.items()}
    def forward(s, input_ids, attention_mask):
        return torch.func.functional_call(s.pm, s.hats(), args=(), kwargs=dict(input_ids=input_ids, attention_mask=attention_mask)).logits

# ---------------- arm E: SimCSE-style unsupervised contrastive LoRA adaptation (dropout positives), no labels
def simcse(X_unl, seed):
    pm = new_net(2, seed); hs = []
    def drop(mod, inp, out): return (F.dropout(out[0], SIM_DROP, True),) + tuple(out[1:]) if isinstance(out, tuple) else F.dropout(out, SIM_DROP, True)
    for l in pm.base_model.model.model.layers: hs.append(l.register_forward_hook(drop))
    pm.train(); opt = torch.optim.AdamW([q for n, q in pm.named_parameters() if q.requires_grad and "lora_" in n], lr=SIM_LR)
    inner = pm.base_model.model.model; rg = np.random.default_rng(seed); sync(); t0 = time.time()
    def pooled(bt):
        h = inner(input_ids=bt["input_ids"], attention_mask=bt["attention_mask"]).last_hidden_state
        return F.normalize(h[torch.arange(len(h), device=dev), bt["attention_mask"].sum(1) - 1].float(), dim=-1)
    for _ in range(SIM_STEPS):
        bt = enc([X_unl[i] for i in rg.choice(len(X_unl), a.bs, replace=False)])
        with ac(): z1, z2 = pooled(bt), pooled(bt)
        loss = F.cross_entropy(z1 @ z2.T / 0.05, torch.arange(len(z1), device=dev)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    sync(); [h.remove() for h in hs]; return lora_sd(pm), time.time() - t0

# ---------------- pools (sibling steps) for B and D
def build_pool(t):
    if t["pool"] == "clinc":
        sx, sy, ss = clinc_split("train", SHARED); lid = {l: i for i, l in enumerate(sorted(set(sy)))}; rows = [(s, x, lid[y]) for s, x, y in zip(ss, sx, sy)]
        info = dict(kind="8 non-held-out CLINC domains", disjoint="held-out domains travel/work excluded by construction (SHARED = domains not in HELD)")
    else:
        excl = set(t["labs"]); pl = [l for l in range(77) if l not in excl]; assert not (set(pl) & excl) and len(pl) == 65
        lid = {l: i for i, l in enumerate(pl)}; parts = np.array_split(np.random.default_rng(1000 + int(t["name"][4:])).permutation(pl), 4)
        gname = {int(l): f"bank group {g}" for g, part in enumerate(parts) for l in part}
        rows = [(gname[int(y)], x, lid[int(y)]) for x, y in zip(*B_TR) if int(y) in lid]
        info = dict(kind="65 Banking77 labels outside the sub-task, 4 random label groups as sibling steps", disjoint=f"assert pool_labels & task_labels == empty; task labels {t['labs']}")
    if a.tiny:
        seen = Counter(); rows = [r for r in rows if (seen.update([r[0]]) or seen[r[0]] <= 40)]
    X = [fmt(s, x) for s, x, _ in rows]; Y = np.array([y for *_, y in rows]); S = np.array([s for s, *_ in rows]); L = len(lid)
    sh = new_net(L, 777); ts = fit(sh, X, Y, POOL_STEPS[t["fam"]], groups(sh)); sd = lora_sd(sh)
    sub = np.random.default_rng(0).choice(len(X), min(300, len(X)), replace=False); pacc = float((probs(sh, [X[i] for i in sub]).argmax(1) == Y[sub]).mean())
    ads, tad = [], 0.0
    for s in sorted(set(S.tolist())):
        ix = np.flatnonzero(S == s); ul = {l: i for i, l in enumerate(sorted(set(Y[ix].tolist())))}
        net = new_net(len(ul), 777); tad += fit(net, [X[i] for i in ix], [ul[y] for y in Y[ix]], AD_STEPS, groups(net)); ads.append(lora_sd(net)); del net
    pool = dict(sd=sd, stacks={k: torch.stack([d[k] for d in ads]) for k in ads[0]}, K=len(ads))
    R["pools"][t["pool"]] = dict(info, rows=len(X), labels=L, K=len(ads), shared_steps=POOL_STEPS[t["fam"]], shared_s=round(ts, 1), shared_train_acc=round(pacc, 3),
                                 adapter_steps=AD_STEPS, adapters_s=round(tad, 1)); log("pool", t["pool"], R["pools"][t["pool"]]); save()
    del sh; gc.collect(); return pool

# ---------------- one (task, seed, k) cell
EMB = {}
def embed(t):
    if t["name"] not in EMB:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer("BAAI/bge-small-en-v1.5", device=dev)
        EMB[t["name"]] = (st.encode(t["trx"], batch_size=128, normalize_embeddings=True, show_progress_bar=False), st.encode(t["tex"], batch_size=128, normalize_embeddings=True, show_progress_bar=False))
    return EMB[t["name"]]
SIM = {}
def run_cell(t, pool, seed, k):
    idx = sample_idx(t, seed, k); n = len(idx); L = t["L"]; steps = steps_for(n); Xk = [t["X"][i] for i in idx]; yk = t["tr_y"][idx]
    unl = np.setdiff1d(np.arange(len(t["X"])), idx); Xu = [t["X"][i] for i in unl]; yu = t["tr_y"][unl]
    def rec(arm, P, secs, params, st, **kw):
        r = dict(task=t["name"], fam=t["fam"], seed=seed, k=k, rows=n, arm=arm, steps=st, train_s=round(secs, 1), params=params, **metrics(P, t["te_y"]), **kw)
        R["records"].append(r); log({x: r[x] for x in ("task", "seed", "k", "arm", "steps", "train_s", "params", "acc", "cov_sgr", "cov_grid")})
    def safe(arm, f):
        try: f()
        except Exception: R["errors"].append(dict(task=t["name"], seed=seed, k=k, arm=arm, tb=traceback.format_exc()[-1500:])); log("ERROR", arm, traceback.format_exc()[-600:])
    out = {}
    def lora_arm(arm, init_sd, **kw):  # A / B / E1: LoRA + new head on the k rows
        m = new_net(L, seed, init_sd=init_sd); s = fit(m, Xk, yk, steps, groups(m)); rec(arm, probs(m, t["TX"]), s, nparams(m), steps, **kw)
        if arm == "A": out["A"], out["A_s"] = m, s
    safe("A", lambda: lora_arm("A", None))
    safe("B", lambda: lora_arm("B", pool["sd"]))
    def armC():
        net, g = reft_net(L, seed); s = fit(net, Xk, yk, steps, g); rec("C", probs(net, t["TX"]), s, nparams(net), steps, impl=REFT_IMPL)
    safe("C", armC)
    def armD():
        pm = new_net(L, seed); comp = Comp(pm, pool["stacks"]); hp = [q for nm, q in comp.named_parameters() if q.requires_grad and nm != "w"]
        s1 = fit(comp, Xk, yk, steps, [dict(params=hp, lr=a.hlr), dict(params=[comp.w], lr=a.wlr)])
        rec("D_comp", probs(comp, t["TX"]), s1, nparams(comp), steps, mix=[round(float(x), 3) for x in torch.softmax(comp.w.detach(), 0)])
        with torch.no_grad():
            hats = comp.hats()
            for nm, q in pm.named_parameters():
                if nm in hats: q.copy_(hats[nm]); q.requires_grad_(True)
        s2 = fit(pm, Xk, yk, FT_STEPS, groups(pm)); rec("D_comp_ft", probs(pm, t["TX"]), s1 + s2, nparams(pm), steps + FT_STEPS, ft_s=round(s2, 1))
    safe("D", armD)
    def armE1():
        if (t["name"], seed) not in SIM:  # unlabelled stream = train texts minus this seed's largest-k labelled sample (no labels used)
            SIM[(t["name"], seed)] = simcse([t["X"][i] for i in np.setdiff1d(np.arange(len(t["X"])), sample_idx(t, seed, max(KS)))], seed)
        sd, ss = SIM[(t["name"], seed)]; lora_arm("E1", sd, simcse_s=round(ss, 1), simcse_steps=SIM_STEPS)
    safe("E1", armE1)
    def pseudo(m):
        P = probs(m, Xu); s = np.sort(P, 1); gap = s[:, -1] - s[:, -2]; sel = np.flatnonzero(gap >= np.percentile(gap, 80)); lab = P.argmax(1)[sel]
        return sel, lab, float((lab == yu[sel]).mean())
    def armE2():
        m = out["A"]; tot = out["A_s"]
        for rnd in (1, 2):
            sel, lab, pacc = pseudo(m); X2 = Xk + [Xu[i] for i in sel]; y2 = np.concatenate([yk, lab]); st2 = steps_for(len(X2))
            m = new_net(L, seed); s = fit(m, X2, y2, st2, groups(m)); tot += s
            rec(f"E2_r{rnd}", probs(m, t["TX"]), tot, nparams(m), st2, n_pseudo=int(len(sel)), pseudo_acc=round(pacc, 3), train_s_incl_A=True)
    safe("E2", armE2)
    def armF():
        from sklearn.linear_model import LogisticRegression
        Etr, Ete = embed(t); t0 = time.time(); clf = LogisticRegression(C=10, max_iter=1000).fit(Etr[idx], yk)
        rec("F", clf.predict_proba(Ete), time.time() - t0, int(clf.coef_.size + clf.intercept_.size), 0)
    safe("F", armF)
    if k in S8KS and t["fam"] == "clinc":  # S8 phase-B recipe: one lr 2e-4 for everything, 3 epochs, >= 60 steps
        for arm, sd in (("A_s8", None), ("B_s8", pool["sd"])):
            def f(arm=arm, sd=sd):
                m = new_net(L, seed, init_sd=sd); st8 = max(60, 3 * math.ceil(n / a.bs)); s = fit(m, Xk, yk, st8, [dict(params=[q for q in m.parameters() if q.requires_grad], lr=a.lr)])
                rec(arm, probs(m, t["TX"]), s, nparams(m), st8)
            safe(arm, f)
    out.clear(); gc.collect()
    if dev == "cuda": torch.cuda.empty_cache()
    save()

if __name__ == "__main__":
    tasks = [mk(nm) for nm in a.tasks.split(",")]; pools = {}
    log("tasks", [(x["name"], x["L"], len(x["X"]), len(x["TX"])) for x in tasks], "reft layers", REFT_LAYERS)
    try:
        assert not a.hand, "--hand: pyreft skipped"; R["reft"] = dict(impl="pyreft", params=probe_pyreft()); REFT_IMPL = "pyreft"
    except Exception: R["reft"] = dict(impl="hand", pyreft_error=traceback.format_exc()[-1500:]); REFT_IMPL = "hand"
    log("ReFT impl:", REFT_IMPL, R["reft"].get("pyreft_error", "")[-400:]); save()
    for t in tasks:
        if t["pool"] not in pools: pools[t["pool"]] = build_pool(t)
    for seed in SEEDS:
        for t in tasks:
            for k in KS: log("cell", t["name"], seed, k); run_cell(t, pools[t["pool"]], seed, k)
    R["done"] = True; R["total_s"] = round(time.time() - T0); save(); log("wrote", a.out)
