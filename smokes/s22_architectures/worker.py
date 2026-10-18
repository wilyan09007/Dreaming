"""S22-lite worker (started by run.py, several at once on one GPU). Qwen2.5-0.5B seq-cls head, bf16 autocast, bs 32, max_len 64, 300 steps per run.
Arms per cell (task, seed, k): F frozen bge-small + LR | B LoRA from the shared pool LoRA | D composition of per-domain pool LoRAs
(D_comp = new head + softmax mixing weights only, D_comp_ft = then +100 LoRA steps) | A LoRA from base.
Pools: clinc (8 non-held-out CLINC domains) for travel/work; bank (41 Banking77 labels outside the three sub-tasks, 4 label groups) for bank0-2.
Jobs and cells are claimed through O_EXCL files so workers share the load without a queue server. ponytail: no framework, fixed recipe, no tuning per arm."""
import argparse, copy, gc, json, math, os, sys, time, traceback
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
p = argparse.ArgumentParser()
p.add_argument("--i", type=int, default=0); p.add_argument("--workers", type=int, default=1); p.add_argument("--t0", type=float, default=0.0)
p.add_argument("--budget-min", type=float, default=32.0); p.add_argument("--scr", required=True); p.add_argument("--out", required=True)
p.add_argument("--ks", default="3,1,10"); p.add_argument("--seeds", default="0,1,2"); p.add_argument("--tiny", action="store_true")
a = p.parse_args()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(2)
dev = "cuda" if torch.cuda.is_available() else "cpu"
assert dev == "cuda" or a.tiny, "no CUDA on this pod: exit instead of crawling on CPU"
OUT, SCR = Path(a.out), Path(a.scr); (OUT / "rec").mkdir(parents=True, exist_ok=True)
T0 = a.t0 or time.time()
def log(*x): print(f"[w{a.i} +{time.time()-T0:.0f}s]", *x, flush=True)
def sync():
    if dev == "cuda": torch.cuda.synchronize()

BS, MAXLEN, STEPS, FT_STEPS, LR, HLR, WLR = 32, 64, 300, 100, 2e-4, 2e-3, 5e-2
POOL_STEPS = {"clinc": 750, "bank": 600}; AD_STEPS = 150; NCAL = 300; SEED_POOL = 777
TASK_NAMES = ["travel", "work", "bank0", "bank1", "bank2"]
if a.tiny: STEPS = FT_STEPS = AD_STEPS = 2; POOL_STEPS = {"clinc": 2, "bank": 2}; NCAL = 30; TASK_NAMES = ["travel", "bank0"]
KS, SEEDS = [[int(x) for x in s.split(",") if x] for s in (a.ks, a.seeds)]

# ---------------- data and tasks
import datasets; datasets.disable_progress_bars()
from common import CACHE, load_clinc150, load_banking77, confidence, certify, apply_certificate
DOM = json.load(open(CACHE / "clinc_domains.json"))
C = load_clinc150(); CN = C["names"]; B = load_banking77(); BN = B["names"]
HELD = ["travel", "work"]; SHARED = sorted(set(DOM.values()) - set(HELD))
PERM = np.random.default_rng(0).permutation(77)
BANK_IDS = [sorted(PERM[j * 12:(j + 1) * 12].tolist()) for j in range(3)]; POOL_BANK = sorted(PERM[36:].tolist())
# disjointness of every pretraining pool from every evaluation task (enforced, not assumed)
held_lab = {n for n, d in DOM.items() if d in HELD}; pool_lab = {n for n, d in DOM.items() if d in SHARED}
assert not (held_lab & pool_lab) and len(held_lab) == 30 and len(pool_lab) == 120 and len(SHARED) == 8, (len(held_lab), len(pool_lab), SHARED)
assert not (set(POOL_BANK) & set(sum(BANK_IDS, []))) and len(POOL_BANK) == 41 and len(set(sum(BANK_IDS, []))) == 36
fmt = lambda step, t: f"Step: {step}\nQuery: {t}\nLabel:"

def clinc_rows(split, names):
    li = {CN.index(n): i for i, n in enumerate(names)}; xs, ys = C[split]; sel = [j for j, y in enumerate(ys) if int(y) in li]
    return [xs[j] for j in sel], np.array([li[int(ys[j])] for j in sel])
def bank_rows(split, ids):
    li = {int(y): i for i, y in enumerate(ids)}; xs, ys = B[split]; sel = [j for j, y in enumerate(ys) if int(y) in li]
    return [xs[j] for j in sel], np.array([li[int(ys[j])] for j in sel])
def trim(xs, ys, L, n):  # tiny mode: first n rows per label
    keep = np.concatenate([np.flatnonzero(ys == l)[:n] for l in range(L)]); return [xs[i] for i in keep], ys[keep]

def make_task(name):
    if name in HELD:
        labels = [n for n in CN if n != "oos" and DOM[n] == name]; trx, tr_y = clinc_rows("train", labels); tex, te_y = clinc_rows("test", labels); step, fam = name, "clinc"
    else:
        j = int(name[4:]); ids = BANK_IDS[j]; labels = [BN[y] for y in ids]; trx, tr_y = bank_rows("train", ids); tex, te_y = bank_rows("test", ids); step, fam = f"bank task {j}", "bank"
    L = len(labels)
    if a.tiny: trx, tr_y = trim(trx, tr_y, L, 14); tex, te_y = trim(tex, te_y, L, 4)
    perm = np.random.default_rng(0).permutation(len(trx)); cal, rest = perm[:NCAL], perm[NCAL:]  # fixed calibration rows, never drawn as training rows
    t = dict(name=name, fam=fam, step=step, L=L, labels=labels, trx=trx, tr_y=tr_y, tex=tex, te_y=te_y, cal=cal, rest=rest)
    t["X"] = [fmt(step, x) for x in trx]; t["TX"] = [fmt(step, x) for x in tex]; t["EV"] = [t["X"][i] for i in cal] + t["TX"]; t["y_cal"] = tr_y[cal]
    return t

_perm = {}
def sample_idx(t, seed, k):  # k rows per label from the non-calibration train rows; nested across k for a given seed
    if (t["name"], seed) not in _perm:
        r = np.random.default_rng(100 + seed); rest = t["rest"]; _perm[(t["name"], seed)] = [r.permutation(rest[t["tr_y"][rest] == l]) for l in range(t["L"])]
    return np.concatenate([q[:k] for q in _perm[(t["name"], seed)]])

# ---------------- model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Qwen2Config
from peft import LoraConfig, get_peft_model
MODEL = "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "right"
if a.tiny:
    BASE = AutoModelForSequenceClassification.from_config(Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                                                        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128))
else:
    try: BASE = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2, dtype=torch.float32)
    except TypeError: BASE = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2, torch_dtype=torch.float32)
BASE.config.pad_token_id = tok.pad_token_id; BASE = BASE.to(dev); H = BASE.config.hidden_size
LORA = lambda: LoraConfig(task_type="SEQ_CLS", r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], modules_to_save=["score"])

def seed_lora_A(m, seed):  # same lora_A init for every pool adapter, so averaging A and B across adapters is meaningful (independent of head size)
    g = torch.Generator().manual_seed(seed)
    for n, q in m.named_parameters():
        if "lora_A" in n:
            w = torch.empty(q.shape); nn.init.kaiming_uniform_(w, a=math.sqrt(5), generator=g)
            with torch.no_grad(): q.copy_(w.to(q.device, q.dtype))
def new_net(L, seed, init_sd=None, lora_seed=None):
    # copy the module tree but share the frozen base weights (a full 2 GB copy per run plus BASE itself gave ~8 GB per worker and an OOM with 3 workers on a 24 GB card)
    torch.manual_seed(seed); m = copy.deepcopy(BASE, {id(q): q for q in BASE.parameters()}); m.score = nn.Linear(H, L, bias=False).to(dev); nn.init.normal_(m.score.weight, std=0.02); m.config.num_labels = L
    m = get_peft_model(m, LORA())
    if lora_seed is not None: seed_lora_A(m, lora_seed)
    if init_sd is not None:
        r = m.load_state_dict(init_sd, strict=False); assert not r.unexpected_keys and not [k for k in r.missing_keys if "lora_" in k], r
    return m
lora_sd = lambda m: {k: v.detach().clone() for k, v in m.state_dict().items() if "lora_" in k}
nparams = lambda m: sum(q.numel() for q in m.parameters() if q.requires_grad)
def groups(m):  # LoRA params at LR, the new head at HLR
    ps = [(n, q) for n, q in m.named_parameters() if q.requires_grad]
    g = [dict(params=[q for n, q in ps if "lora_" in n], lr=LR), dict(params=[q for n, q in ps if "lora_" not in n], lr=HLR)]
    return [x for x in g if x["params"]]
enc = lambda texts: tok(texts, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(dev)
def lg(net, bt):
    o = net(**bt); return o.logits if hasattr(o, "logits") else o
ac = lambda: torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda"))

def fit(net, X, y, steps, grp):
    net.train(); opt = torch.optim.AdamW(grp, fused=(dev == "cuda")); y = torch.as_tensor(np.asarray(y)); n = len(X); step = 0; sync(); t0 = time.time()
    while step < steps:
        perm = torch.randperm(n)
        for i in range(0, n, BS):
            if step >= steps: break
            idx = perm[i:i + BS]
            with ac(): loss = F.cross_entropy(lg(net, enc([X[j] for j in idx.tolist()])).float(), y[idx].to(dev))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1
    sync(); return time.time() - t0

@torch.no_grad()
def probs(net, X, bs=256):
    net.eval(); out = []
    for i in range(0, len(X), bs):
        with ac(): out.append(torch.softmax(lg(net, enc(X[i:i + bs])).float(), -1).cpu().numpy())
    return np.concatenate(out)

class Comp(nn.Module):  # LoraHub-style: one softmax over the K frozen pool adapters, applied to every lora_A and lora_B tensor; only the head and K logits train
    def __init__(s, pm, stacks):
        super().__init__(); s.pm = pm; s.st = stacks; s.w = nn.Parameter(torch.zeros(len(next(iter(stacks.values()))), device=dev))
        for n, q in pm.named_parameters():
            if "lora_" in n: q.requires_grad_(False)
    def hats(s): wt = torch.softmax(s.w, 0); return {n: torch.tensordot(wt, t, dims=1) for n, t in s.st.items()}
    def forward(s, input_ids, attention_mask):
        return torch.func.functional_call(s.pm, s.hats(), args=(), kwargs=dict(input_ids=input_ids, attention_mask=attention_mask)).logits

def free():
    gc.collect()
    if dev == "cuda": torch.cuda.empty_cache()

# ---------------- pools (sibling steps) built once per pod and shared between workers through files
ROWS = {}
def pool_rows(fam):
    if fam not in ROWS:
        if fam == "clinc":
            xs, ys = C["train"]; rows = [(DOM[CN[int(y)]], x, CN[int(y)]) for x, y in zip(xs, ys) if DOM[CN[int(y)]] in SHARED]
        else:
            g_of = {int(l): g for g, part in enumerate(np.array_split(np.random.default_rng(1000).permutation(POOL_BANK), 4)) for l in part}
            xs, ys = B["train"]; rows = [(f"bank group {g_of[int(y)]}", x, int(y)) for x, y in zip(xs, ys) if int(y) in g_of]
        if a.tiny:
            keep = sorted({r[0] for r in rows})[:2]; cnt = {}; out = []
            for r in rows:
                if r[0] in keep and cnt.get(r[0], 0) < 24: cnt[r[0]] = cnt.get(r[0], 0) + 1; out.append(r)
            rows = out
        ROWS[fam] = rows
    return ROWS[fam]
JOBS = [(f, "shared") for f in ("clinc", "bank")] + [(f, s) for f in ("clinc", "bank") for s in sorted({r[0] for r in pool_rows(f)})]
jname = lambda fam, which: f"{fam}__{which.replace(' ', '_')}"
def claim(name):
    try: os.close(os.open(SCR / "claims" / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY)); return True
    except FileExistsError: return False

def run_pool_job(fam, which):
    rows = pool_rows(fam) if which == "shared" else [r for r in pool_rows(fam) if r[0] == which]
    keys = sorted({r[2] for r in rows}, key=str); lid = {k: i for i, k in enumerate(keys)}
    X = [fmt(s, x) for s, x, _ in rows]; Y = np.array([lid[r[2]] for r in rows]); steps = POOL_STEPS[fam] if which == "shared" else AD_STEPS
    net = new_net(len(keys), SEED_POOL, lora_seed=SEED_POOL); secs = fit(net, X, Y, steps, groups(net))
    sub = np.random.default_rng(0).choice(len(X), min(300, len(X)), replace=False); acc = float((probs(net, [X[i] for i in sub]).argmax(1) == Y[sub]).mean())
    info = dict(job=jname(fam, which), fam=fam, which=which, rows=len(X), labels=len(keys), steps=steps, secs=round(secs, 1), pool_acc=round(acc, 3), params=nparams(net), worker=a.i,
                disjoint=("pool intents = 8 CLINC domains, asserted disjoint from travel/work intents" if fam == "clinc" else "pool labels = 41 Banking77 ids, asserted disjoint from the 36 ids of bank0-2"))
    tmp = SCR / "pools" / (jname(fam, which) + ".tmp"); torch.save(dict(sd={k: v.cpu() for k, v in lora_sd(net).items()}, info=info), tmp); os.replace(tmp, SCR / "pools" / (jname(fam, which) + ".pt"))
    json.dump(info, open(OUT / "rec" / f"pool_{jname(fam, which)}.json", "w")); log("pool job", info)
    del net; free()

def wait_pool(fam):
    need = [jname(f, w) for f, w in JOBS if f == fam]; t_end = time.time() + 25 * 60
    while True:
        if all((SCR / "pools" / (n + ".pt")).exists() for n in need): return
        if any((SCR / "pools" / (n + ".err")).exists() for n in need): raise RuntimeError(f"pool {fam} failed (see .err in {SCR/'pools'})")
        if time.time() > t_end: raise TimeoutError(f"pool {fam} not ready")
        time.sleep(2)
POOLS = {}
def get_pool(fam):
    if fam not in POOLS:
        wait_pool(fam); shared = torch.load(SCR / "pools" / f"{fam}__shared.pt", map_location=dev)["sd"]
        ads = [torch.load(f, map_location=dev)["sd"] for f in sorted((SCR / "pools").glob(f"{fam}__*.pt")) if not f.name.endswith("__shared.pt")]
        POOLS[fam] = dict(shared=shared, stacks={k: torch.stack([d[k] for d in ads]) for k in ads[0]}, K=len(ads))
    return POOLS[fam]

# ---------------- metrics: 300 fixed calibration rows -> cutoff (common.certify, 5% budget, top-2 gap) -> applied to the whole test split
def metrics(P, nc, y_cal, y_te):
    P = np.nan_to_num(np.asarray(P, float)); Pc, Pt = P[:nc], P[nc:]; okc, okt = Pc.argmax(1) == y_cal, Pt.argmax(1) == y_te
    c = certify(confidence(Pc), okc, 0.05); ap = apply_certificate(confidence(Pt), okt, c["cutoff"])
    return dict(acc=round(float(okt.mean()), 4), cal_acc=round(float(okc.mean()), 4), certified=bool(np.isfinite(c["cutoff"])), cal_cov=round(c["coverage"], 4),
                cov=round(ap["coverage"], 4), err=None if ap["error"] is None else round(ap["error"], 4))

EMBS, ST = {}, []
def get_emb(t):
    if t["name"] not in EMBS:
        if not ST:
            from sentence_transformers import SentenceTransformer
            ST.append(SentenceTransformer("BAAI/bge-small-en-v1.5", device=dev))
        e = lambda xs: ST[0].encode(xs, batch_size=128, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        EMBS[t["name"]] = (e(t["trx"]), e(t["tex"]))
    return EMBS[t["name"]]

def run_cell(t, pool, seed, k):
    idx = sample_idx(t, seed, k); n = len(idx); L = t["L"]; nc = len(t["cal"]); Xk = [t["X"][i] for i in idx]; yk = t["tr_y"][idx]
    def rec(arm, P, secs, params, steps, **kw):
        r = dict(task=t["name"], fam=t["fam"], seed=seed, k=k, rows=n, arm=arm, steps=steps, train_s=round(secs, 2), params=int(params), worker=a.i, at_s=round(time.time() - T0),
                 **metrics(P, nc, t["y_cal"], t["te_y"]), **kw)
        with open(OUT / "rec" / f"rec_w{a.i}.jsonl", "a", encoding="utf-8") as f: f.write(json.dumps(r) + "\n")
        log({x: r[x] for x in ("task", "seed", "k", "arm", "train_s", "params", "acc", "cov", "err")})
    def safe(arm, f):
        try: f()
        except Exception:
            tb = traceback.format_exc()[-1500:]; log("ERROR", arm, t["name"], seed, k, tb[-500:])
            with open(OUT / "rec" / f"err_w{a.i}.jsonl", "a", encoding="utf-8") as fh: fh.write(json.dumps(dict(task=t["name"], seed=seed, k=k, arm=arm, tb=tb)) + "\n")
        finally: free()
    def armF():
        from sklearn.linear_model import LogisticRegression
        Etr, Ete = get_emb(t); t0 = time.time(); clf = LogisticRegression(C=10, max_iter=1000).fit(Etr[idx], yk); P = np.concatenate([clf.predict_proba(Etr[t["cal"]]), clf.predict_proba(Ete)])
        rec("F", P, time.time() - t0, clf.coef_.size + clf.intercept_.size, 0)
    def armB():
        m = new_net(L, seed, init_sd=pool["shared"]); s = fit(m, Xk, yk, STEPS, groups(m)); rec("B", probs(m, t["EV"]), s, nparams(m), STEPS); del m
    def armD():
        pm = new_net(L, seed); comp = Comp(pm, pool["stacks"]); hp = [q for nm, q in comp.named_parameters() if q.requires_grad and nm != "w"]
        s1 = fit(comp, Xk, yk, STEPS, [dict(params=hp, lr=HLR), dict(params=[comp.w], lr=WLR)])
        rec("D_comp", probs(comp, t["EV"]), s1, nparams(comp), STEPS, mix=[round(float(x), 3) for x in torch.softmax(comp.w.detach(), 0)])
        with torch.no_grad():
            hats = comp.hats()
            for nm, q in pm.named_parameters():
                if nm in hats: q.copy_(hats[nm]); q.requires_grad_(True)
        del comp; s2 = fit(pm, Xk, yk, FT_STEPS, groups(pm)); rec("D_comp_ft", probs(pm, t["EV"]), s1 + s2, nparams(pm), STEPS + FT_STEPS, ft_s=round(s2, 2)); del pm
    def armA():
        m = new_net(L, seed); s = fit(m, Xk, yk, STEPS, groups(m)); rec("A", probs(m, t["EV"]), s, nparams(m), STEPS); del m
    for arm, f in (("F", armF), ("B", armB), ("D", armD), ("A", armA)): safe(arm, f)  # most informative first: reference, shared, composition, from-base

def calibrate():  # worker 0 alone, before the others start: uncontended steps/s of a LoRA run and of a composition run
    t = TASKS[TASK_NAMES[0]]; idx = sample_idx(t, 0, 3); Xk = [t["X"][i] for i in idx]; yk = t["tr_y"][idx]; w, nt = (1, 2) if a.tiny else (4, 20)
    net = new_net(t["L"], 0); fit(net, Xk, yk, w, groups(net)); r1 = nt / fit(net, Xk, yk, nt, groups(net))
    pm = new_net(t["L"], 0); stacks = {k: torch.stack([v] * 8) for k, v in lora_sd(pm).items()}; comp = Comp(pm, stacks)
    hp = [q for nm, q in comp.named_parameters() if q.requires_grad and nm != "w"]; grp = lambda: [dict(params=hp, lr=HLR), dict(params=[comp.w], lr=WLR)]
    fit(comp, Xk, yk, w, grp()); r2 = nt / fit(comp, Xk, yk, nt, grp())
    c = dict(lora_steps_per_s=round(r1, 2), comp_steps_per_s=round(r2, 2), gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", cpus=os.cpu_count(), torch=torch.__version__,
             peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev == "cuda" else None)
    json.dump(c, open(OUT / "calib.json", "w")); log("solo calibration", c); del net, pm, comp; free()

if __name__ == "__main__":
    TASKS = {nm: make_task(nm) for nm in TASK_NAMES}
    log("tasks", [(t["name"], t["L"], len(t["X"]), len(t["TX"]), len(t["cal"])) for t in TASKS.values()], "pool jobs", len(JOBS), "device", dev)
    if a.i == 0:
        try: calibrate()
        except Exception: log("calibration failed (continuing):", traceback.format_exc()[-600:])
        try: get_emb(TASKS[TASK_NAMES[0]])  # downloads bge-small once, before the other workers need it
        except Exception: log("bge preload failed (continuing):", traceback.format_exc()[-400:])
        open(SCR / "calib.done", "w").close()
    for fam, which in JOBS:  # longest jobs first (shared pools), then the per-domain adapters
        if claim("pool_" + jname(fam, which)):
            for attempt in (1, 2):  # one retry (e.g. a transient OOM while the other workers peak)
                try: run_pool_job(fam, which); break
                except Exception:
                    tb = traceback.format_exc(); log("POOL ERROR attempt", attempt, fam, which, tb[-600:]); free()
                    if attempt == 2: open(SCR / "pools" / (jname(fam, which) + ".err"), "w").write(tb)
                    else: time.sleep(15)
    done = {tuple(x) for x in json.load(open(SCR / "done_cells.json"))} if (SCR / "done_cells.json").exists() else set()
    for k in KS:
        for s in SEEDS:
            for tn in TASK_NAMES:
                if (tn, s, k) in done: continue
                if time.time() - T0 > a.budget_min * 60: log("time budget reached, no new cells"); sys.exit(0)
                try: pool = get_pool(TASKS[tn]["fam"])
                except Exception: log("pool unavailable, skipping", tn, s, k, traceback.format_exc()[-300:]); continue
                if not claim(f"cell_{tn}_{s}_{k}"): continue
                log("cell", tn, s, k); run_cell(TASKS[tn], pool, s, k)
    log("all cells claimed or done")
