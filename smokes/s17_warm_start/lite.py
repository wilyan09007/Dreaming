"""S17-lite: ways to warm-start a NEW step, scored on the shared protocol (5 unseen tasks x k in {1,3,10} x 3 draws, accuracy + certified coverage @5%).
Arms: 1 baseline (frozen bge-small + LR) | 2 synthetic prior | 3 many-task hypernetwork (old.meta = the S2b set-transformer) | 4 workflow-pretrained encoder |
      5 Reptile projection + cosine head | 6 active selection of the rows.
Stages (python lite.py <stage>): prep | enc | hyper | reptile | eval cheap | eval trained | pod | tiny | agg.   `tiny` = everything in-process on a handful of rows.
Reuses lib.py (LLM cache, score, draw_support, models) and run.py (PoolIdx, sample_tasks, episode, val_episodes, meta) from the earlier S17 attempt."""
import sys, os, json, re, time, pickle, subprocess, hashlib
from pathlib import Path
TINY = "tiny" in sys.argv[1:2] or "--tiny" in sys.argv
if TINY: os.environ["S17_TINY"] = "1"
os.environ.setdefault("S17_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import *      # np, common, TINY, HERE, WORK, CACHE, D, embed, llm_json, llm_spend, score, full_proba, draw_support, make_models, confidence, load_clinc150, load_banking77
import run as old      # PoolIdx, sample_tasks, episode, val_episodes, meta, MODELS
old.KS = [1, 2, 3, 5, 10]                                  # hypernetwork training shots = the shots we evaluate (+2)
old.val_episodes.__defaults__ = (30, (1, 3, 10), 99)       # early stopping on held-out TREC episodes at k = 1, 3, 10

STORE = WORK; MODELS = old.MODELS; GPU_OUT = STORE / "gpu_out"; GPU_OUT.mkdir(exist_ok=True)
RES = HERE / ("tiny_results.json" if TINY else "results.json")
KS3, NSEED, HELD = (1, 3, 10), 3, ["travel", "work"]
CAP, MINROWS, NCAL, SN = (4, 3, 9, 3) if TINY else (100, 20, 300, 10)   # rows per pool label | min rows | calibration rows | synthetic examples per label
SCALE_R = 20.0
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:.0f}s]", *a, flush=True)
def load_store(): return pickle.load(open(STORE / "store.pkl", "rb"))
def load_res(): return json.loads(RES.read_text()) if RES.exists() else {"records": [], "meta": {}}
def save_res(R):
    tmp = RES.with_suffix(".tmp"); tmp.write_text(json.dumps(R)); os.replace(tmp, RES)

# ================================================================== prep: pool + 5 unseen eval tasks
def synth_for(family, labels, desc):
    """10 LLM-written practice examples per label: S2b cache first (not for names shared by CLINC and Banking77), else one LLM call per 15 labels; own cache synth_lite.json."""
    DESC = json.load(open(common.CACHE / "label_descriptions.json")); SYN = json.load(open(common.CACHE / "synth_examples.json"))
    collide = set(DESC["clinc150"]) & set(DESC["banking77"]); path = HERE / "synth_lite.json"; own = json.loads(path.read_text()) if path.exists() else {}
    have = lambda l: own.get(f"{family}::{l}") or (SYN.get(l) if l not in collide else None)
    need = [l for l in labels if not have(l)]
    if need and not TINY:
        for i in range(0, len(need), 15):
            chunk = need[i:i + 15]
            prompt = ("For each label below, write 10 short, varied, realistic user messages that would get that label (different phrasings, lengths, some typos, some indirect). "
                      "Reply with a JSON object mapping label -> list of 10 strings, nothing else.\n" + "\n".join(f"{l}: {desc[l]}" for l in chunk))
            for att in range(3):
                d = llm_json(prompt, tag=f"s17lite-synth-{att}", temperature=0.7)
                if all(c in d and len(d[c]) >= 5 for c in chunk): break
            for c in chunk: own[f"{family}::{c}"] = [str(x) for x in d.get(c, [])][:10] or [desc[c]]
        path.write_text(json.dumps(own, indent=1), encoding="utf-8")
    return {l: (have(l) or [desc[l]])[:SN] for l in labels}

CONTEXT = {"massive": "voice-assistant utterances; the label is the intent name (scenario_action, e.g. alarm_set)",
           "trec": "English questions classified by the kind of answer they expect (TREC question types; label shown as code: name)",
           "emotion": "short English tweets labelled with the emotion they express", "ag_news": "news articles labelled by topic",
           "20ng": "Usenet newsgroup posts labelled by the newsgroup they were posted to"}
def describe(ds, names):
    out = {}
    for i in range(0, len(names), 40):
        chunk = names[i:i + 40]
        prompt = (f"Task: classify {CONTEXT[ds]}.\nFor each label below write ONE short sentence (max 25 words) describing what an input with that label looks like or is about, "
                  "in the style of: 'User wants help activating their newly received bank card.' Reply with a JSON object mapping label -> sentence, nothing else.\nLabels:\n" + "\n".join(chunk))
        d = llm_json(prompt, tag="desc")
        for n in chunk: out[n] = str(d.get(n) or n)
    return out
clean = lambda t, n=300: re.sub(r"\s+", " ", str(t)).strip()[:n]
def shrink(texts, y, keep_ids, per):  # TINY only
    y = np.asarray(y); cnt = {}; idx = []
    for i, yy in enumerate(y.tolist()):
        if yy in keep_ids and cnt.get(yy, 0) < per: idx.append(i); cnt[yy] = cnt.get(yy, 0) + 1
    return [texts[i] for i in idx], y[idx]

def prep():
    import datasets
    rng = np.random.default_rng(17); info = {}
    if not TINY: log("LLM ping", llm_json('Reply with exactly this JSON object and nothing else: {"ok": 1}', max_tokens=50, temperature=0, tag="ping"))
    DESC = json.load(open(common.CACHE / "label_descriptions.json")); DOM = json.load(open(common.CACHE / "clinc_domains.json"))
    ld = lambda *a, **k: datasets.load_dataset(*a, **({} if TINY else dict(cache_dir=str(STORE / "hf"))), **k)
    c = load_clinc150(); cn = c["names"]; b = load_banking77(); bn = b["names"]
    clinc_pool = [n for n in cn if n != "oos" and DOM[n] not in HELD]
    perm = np.random.default_rng(0).permutation(77); b_eval = [sorted(perm[12 * j:12 * j + 12].tolist()) for j in range(3)]; b_pool = sorted(perm[36:].tolist())
    if TINY:
        hl = {d: [n for n in cn if n != "oos" and DOM[n] == d][:3] for d in HELD}
        keep = {cn.index(n) for n in clinc_pool[:4] + sum(hl.values(), [])}; clinc_pool = clinc_pool[:4]
        c = dict(names=cn, **{s: shrink(*c[s], keep, per) for s, per in (("train", 6), ("validation", 4), ("test", 4))})
        b_eval = [e[:3] for e in b_eval]; b_pool = b_pool[:3]; keep = {y for e in b_eval for y in e} | set(b_pool)
        b = dict(names=bn, train=shrink(*b["train"], keep, 12), test=shrink(*b["test"], keep, 4))
    cXtr, cXva, cXte = embed(c["train"][0]), embed(c["validation"][0]), embed(c["test"][0]); bXtr, bXte = embed(b["train"][0]), embed(b["test"][0])
    evals = []
    def mk(name, family, labels, desc, ts, ys, tc, yc, tq, yq, Xs, Xc, Xq):
        syn = synth_for(family, labels, desc)
        e = dict(name=name, family=family, labels=labels, desc_t=[desc[l] for l in labels], synth_t=[s for l in labels for s in syn[l]],
                 synth_y=np.array([i for i, l in enumerate(labels) for _ in syn[l]]), ts=ts, ys=np.asarray(ys), tc=tc, yc=np.asarray(yc), tq=tq, yq=np.asarray(yq))
        e["desc_X"], e["synth_X"] = embed(e["desc_t"]), embed(e["synth_t"]); e["Xs"], e["Xc"], e["Xq"] = Xs, Xc, Xq; evals.append(e)
    for dom in HELD:   # CLINC domain: support = train, calibration = validation (20/label = 300 for 15 labels), test = test
        labs = [n for n in cn if n != "oos" and DOM[n] == dom][:3 if TINY else 15]; li = {cn.index(n): i for i, n in enumerate(labs)}
        sel = lambda split: [i for i, y in enumerate(c[split][1]) if y in li]; itr, iva, ite = sel("train"), sel("validation"), sel("test")
        mk(f"clinc/{dom}", "clinc", labs, {n: DESC["clinc150"].get(n, n) for n in labs}, [c["train"][0][i] for i in itr], [li[c["train"][1][i]] for i in itr],
           [c["validation"][0][i] for i in iva], [li[c["validation"][1][i]] for i in iva], [c["test"][0][i] for i in ite], [li[c["test"][1][i]] for i in ite], cXtr[itr], cXva[iva], cXte[ite])
    for j, lab_ids in enumerate(b_eval):   # Banking77 12-label sub-tasks: calibration = NCAL random train rows, support = the rest, test = official test rows
        li = {y: i for i, y in enumerate(lab_ids)}; labs = [bn[y] for y in lab_ids]
        itr = np.array([i for i, y in enumerate(b["train"][1]) if y in li]); ite = np.array([i for i, y in enumerate(b["test"][1]) if y in li])
        pm = np.random.default_rng(100 + j).permutation(len(itr)); ic, isup = itr[pm[:NCAL]], itr[pm[NCAL:]]
        g = lambda a, i: [a[0][x] for x in i]; yl = lambda a, i: [li[a[1][x]] for x in i]
        mk(f"banking/sub{j}", "banking", labs, DESC["banking77"], g(b["train"], isup), yl(b["train"], isup), g(b["train"], ic), yl(b["train"], ic), g(b["test"], ite), yl(b["test"], ite), bXtr[isup], bXtr[ic], bXte[ite])
    log("eval tasks", [(e["name"], len(e["labels"]), len(e["ts"]), len(e["tc"]), len(e["tq"])) for e in evals]); info["eval_labels"] = {e["name"]: e["labels"] for e in evals}
    # ---------------- pool: CLINC train domains (not travel / work), Banking77 labels outside the 3 sub-tasks, MASSIVE, TREC-QC (held-out for early stopping), emotion, ag_news, 20NG
    P = dict(text=[], key=[], ds=[], X=[]); LAB = {}
    def add(ds, keys, texts, X, name_of, desc_of):
        keys = np.asarray(keys)
        for k in sorted(set(keys.tolist())):
            idx = np.flatnonzero(keys == k)
            if len(idx) < MINROWS: continue
            idx = rng.permutation(idx)[:CAP]; LAB[f"{ds}::{k}"] = dict(ds=ds, name=name_of(k), desc=desc_of(k))
            P["text"] += [texts[i] for i in idx]; P["key"] += [f"{ds}::{k}"] * len(idx); P["ds"] += [ds] * len(idx); P["X"].append(X[idx])
    m = np.isin(c["train"][1], [cn.index(n) for n in clinc_pool])
    add("clinc", [cn[y] for y in c["train"][1][m]], [t for t, f in zip(c["train"][0], m) if f], cXtr[m], lambda k: k.replace("_", " "), lambda k: DESC["clinc150"].get(k, k))
    m = np.isin(b["train"][1], b_pool)
    add("banking", [bn[y] for y in b["train"][1][m]], [t for t, f in zip(b["train"][0], m) if f], bXtr[m], lambda k: k.replace("_", " "), lambda k: DESC["banking77"][k])
    def capped(df, labcol, fn):
        keep = []; groups = sorted(df.groupby(labcol).indices.items(), key=lambda kv: str(kv[0]))
        for k, idx in (groups[:3] if TINY else groups):
            if len(idx) >= MINROWS: keep += list(rng.permutation(idx)[:CAP])
        sub = df.iloc[sorted(keep)]; return sub[labcol].values, [clean(fn(r)) for r in sub.itertuples()]
    specs = [("massive", ("mteb/amazon_massive_intent", "en"), "label_text"), ("emotion", ("SetFit/emotion",), "label_text"), ("ag_news", ("SetFit/ag_news",), "label_text"),
             ("20ng", ("SetFit/20_newsgroups",), "label_text"), ("trec", ("SetFit/TREC-QC",), "label_original")]
    for ds, dargs, lc in specs:
        if TINY and ds not in ("emotion", "trec"): continue
        try:
            df_ = ld(*dargs)["train"].to_pandas()
            if ds == "massive" and TINY: raise RuntimeError("skip")
        except Exception as ex:
            log("SKIP pool dataset", ds, repr(ex)[:200])
            if ds == "massive":   # fallback hint from the task: parquet branch of the original dataset
                try: df_ = ld("AmazonScience/massive", "en-US", revision="refs/convert/parquet")["train"].to_pandas(); df_["label_text"] = df_["intent"].astype(str); df_["text"] = df_["utt"]
                except Exception as ex2: log("SKIP massive fallback", repr(ex2)[:200]); continue
            elif ds == "trec": raise
            else: continue
        if ds == "20ng": df_ = df_[df_.text.str.len() > 40]
        labs, texts = capped(df_, lc, lambda r: r.text); labs = np.asarray(labs)
        nmap = df_.groupby("label_original").label_text.first().to_dict() if ds == "trec" else {k: k for k in set(labs.tolist())}
        keys = sorted(set(labs.tolist())); nm = lambda k: f"{k}: {nmap[k]}" if ds == "trec" else str(nmap[k]); dn = describe(ds, [nm(k) for k in keys]); X = embed(texts)
        for k in keys:
            idx = np.flatnonzero(labs == k); key = f"{ds}::{k}"; LAB[key] = dict(ds=ds, name=str(nmap[k]).replace("_", " "), desc=dn[nm(k)])
            P["text"] += [texts[i] for i in idx]; P["key"] += [key] * len(idx); P["ds"] += [ds] * len(idx); P["X"].append(X[idx])
    # ---------------- disjointness: label sets by construction (domains / label ids / other datasets), plus exact-text dedupe and a label-name overlap report
    ev_clinc = {l.replace("_", " ") for e in evals if e["family"] == "clinc" for l in e["labels"]}; ev_bank = {l.replace("_", " ") for e in evals if e["family"] == "banking" for l in e["labels"]}
    assert not {v["name"] for v in LAB.values() if v["ds"] == "clinc"} & ev_clinc and not {v["name"] for v in LAB.values() if v["ds"] == "banking"} & ev_bank
    ev_texts = {t.lower() for e in evals for t in e["ts"] + e["tc"] + e["tq"]}
    keep = np.array([i for i, t in enumerate(P["text"]) if t.lower() not in ev_texts]); info["dups_removed"] = len(P["text"]) - len(keep)
    info["name_overlaps"] = sorted({v["name"].lower() for v in LAB.values()} & (ev_clinc | ev_bank))
    Xall = np.concatenate(P["X"]); keys = np.array(P["key"]); dss = np.array(P["ds"]); texts = np.array(P["text"], dtype=object)
    labkeys = sorted(set(keys[keep].tolist())); ds_labels = {d_: [k_ for k_ in labkeys if LAB[k_]["ds"] == d_] for d_ in sorted(set(dss[keep].tolist()))}
    pool = dict(X=Xall[keep], key=keys[keep], ds=dss[keep], text=list(texts[keep]), labels={k_: LAB[k_] for k_ in labkeys}, lab_keys=labkeys,
                lab_desc_X=embed([LAB[k_]["desc"] for k_ in labkeys]), lab_name_X=embed([LAB[k_]["name"] for k_ in labkeys]))
    info.update(pool_labels={d_: len(v) for d_, v in ds_labels.items()}, pool_rows=len(keep), desc_fallback=sum(v["desc"] == v["name"] for v in LAB.values()), n_labels=len(labkeys))
    log("pool", info); pickle.dump(dict(pool=pool, evals=evals, ds_labels=ds_labels), open(STORE / "store.pkl", "wb"))
    json.dump(info, open(HERE / ("tiny_prep_info.json" if TINY else "prep_info.json"), "w"), indent=1); log(f"LLM spend (all cached calls) ${llm_spend():.4f}")

# ================================================================== arm 4: classification fine-tune of bge-small on the pool, head discarded
def enc_stage():
    import torch, torch.nn as nn, torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    from sklearn.linear_model import LogisticRegression
    dev = "cuda" if torch.cuda.is_available() else "cpu"; t_all = time.time(); STEPS, CK, BS = (4, (2, 4), 8) if TINY else (1000, (250, 500, 750, 1000), 64)
    st = load_store(); pool, ev = st["pool"], st["evals"]; pk = np.asarray(pool["key"]); pt = pool["text"]; ds_of = {k: v["ds"] for k, v in pool["labels"].items()}
    tok = AutoTokenizer.from_pretrained(common.EMB_MODEL); R = {"device": dev, "variants": {}}
    def enc(model, texts, train=False, maxlen=48):
        bt = tok(texts, padding=True, truncation=True, max_length=maxlen, return_tensors="pt").to(dev)
        if train and dev == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16): return model(**bt).last_hidden_state[:, 0].float()
        return model(**bt).last_hidden_state[:, 0].float()
    @torch.no_grad()
    def embed_all(model, texts, bs=256):
        model.eval(); order = np.argsort([len(t) for t in texts]); out = np.zeros((len(texts), 384), np.float32)
        for i in range(0, len(texts), bs):
            idx = order[i:i + bs]; out[idx] = F.normalize(enc(model, [texts[j] for j in idx], maxlen=128), dim=-1).cpu().numpy()
        return out
    trec = sorted(k for k in ds_of if ds_of[k] == "trec"); rows = {k: np.flatnonzero(pk == k) for k in trec}; vidx = np.unique(np.concatenate([rows[k] for k in trec])); pos = {int(i): j for j, i in enumerate(vidx)}
    r0 = np.random.default_rng(5); vtasks = [list(r0.choice(trec, min(10, len(trec)), replace=False)) for _ in range(8)]
    def val3(model):   # 3-shot LR accuracy on held-out TREC tasks (TREC is never in the fine-tune pool)
        X = embed_all(model, [pt[i] for i in vidx]); accs = []
        for t in vtasks:
            for rep in range(3):
                r = np.random.default_rng(rep); tr, ytr, te, yte = [], [], [], []
                for c, k in enumerate(t):
                    rr = r.permutation(rows[k]); tr += list(rr[:3]); ytr += [c] * len(rr[:3]); te += list(rr[3:33]); yte += [c] * len(rr[3:33])
                clf = LogisticRegression(C=10, max_iter=1000).fit(X[[pos[i] for i in tr]], ytr); accs.append(float((clf.predict(X[[pos[i] for i in te]]) == np.array(yte)).mean()))
        return float(np.mean(accs))
    def variant(name, keep):
        t_v = time.time(); labs = sorted(k for k in ds_of if ds_of[k] != "trec" and keep(k)); lid = {k: i for i, k in enumerate(labs)}
        ridx = np.array([i for i, k in enumerate(pk) if k in lid]); yl = np.array([lid[pk[i]] for i in ridx]); rng = np.random.default_rng(0); torch.manual_seed(0)
        model = AutoModel.from_pretrained(common.EMB_MODEL).to(dev); head = nn.Linear(384, len(labs)).to(dev)
        opt = torch.optim.AdamW([{"params": model.parameters(), "lr": 3e-5}, {"params": head.parameters(), "lr": 1e-3}], weight_decay=0.01)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / STEPS))
        v0 = val3(model); log(f"[enc {name}] labels {len(labs)} rows {len(ridx)}; step 0 (base encoder) TREC 3-shot {v0:.3f}"); best = (-1, 0, None); hist = [(0, v0)]
        for step in range(1, STEPS + 1):
            model.train(); bi = rng.choice(len(ridx), min(BS, len(ridx)), replace=False); h = enc(model, [pt[ridx[j]] for j in bi], train=True)
            loss = F.cross_entropy(head(F.dropout(h, 0.1, training=True)), torch.tensor(yl[bi], device=dev)); opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            if step % 100 == 0: log(f"[enc {name}] step {step} loss {float(loss):.3f}")
            if step in CK:
                v = val3(model); hist.append((step, v)); log(f"[enc {name}] step {step} TREC 3-shot {v:.3f}")
                if v >= best[0]: best = (v, step, {k_: p.detach().cpu().clone() for k_, p in model.state_dict().items()})   # only fine-tuned checkpoints compete, never the untouched base
        if best[2] is None: best = (v0, STEPS, {k_: p.detach().cpu().clone() for k_, p in model.state_dict().items()})   # no usable validation score: keep the last model
        model.load_state_dict(best[2]); log(f"[enc {name}] chosen step {best[1]}"); out = {}
        for i, e in enumerate(ev):
            for nm, texts in (("Xs", e["ts"]), ("Xc", e["tc"]), ("Xq", e["tq"]), ("synth", e["synth_t"])): out[f"{nm}{i}"] = embed_all(model, texts)
        np.savez_compressed(GPU_OUT / f"out_enc_{name}.npz", **out)
        R["variants"][name] = dict(n_labels=len(labs), n_rows=int(len(ridx)), hist=hist, chosen_step=best[1], base_val=v0, seconds=time.time() - t_v, params=int(sum(p.numel() for p in model.parameters())))
        json.dump(R, open(HERE / ("tiny_results_enc.json" if TINY else "results_enc.json"), "w"), indent=1); del model, opt; torch.cuda.empty_cache()
    base = AutoModel.from_pretrained(common.EMB_MODEL).to(dev); chk = embed_all(base, ev[0]["ts"][:64]); R["base_sanity_min_cos"] = float((chk * ev[0]["Xs"][:64]).sum(1).min())
    log(f"base encoder via transformers vs stored sentence-transformers embeddings: min cosine {R['base_sanity_min_cos']:.5f}"); del base
    variant("all", lambda k: True); variant("nosib", lambda k: ds_of[k] not in ("clinc", "banking"))   # nosib: no CLINC / Banking77 siblings in the pool
    R["total_seconds"] = time.time() - t_all; json.dump(R, open(HERE / ("tiny_results_enc.json" if TINY else "results_enc.json"), "w"), indent=1)

# ================================================================== arm 5: Reptile for a linear projection M + cosine head over description-initialised class vectors
def adapt(M_init, dsc, K, get_batch, T, lrR, lrM):
    """fast weights: (M, r), logits = S * cos(M x, M desc_c + r_c); Adam on the batches get_batch(t) yields."""
    import torch, torch.nn.functional as F
    M = M_init.detach().clone().requires_grad_(True); r = torch.zeros(K, D, requires_grad=True); opt = torch.optim.Adam([{"params": [r], "lr": lrR}, {"params": [M], "lr": lrM}])
    for t in range(T):
        X, y = get_batch(t); loss = F.cross_entropy(SCALE_R * F.normalize(X @ M.T, dim=-1) @ F.normalize(dsc @ M.T + r, dim=-1).T, y); opt.zero_grad(); loss.backward(); opt.step()
    return M.detach(), r.detach()
def rep_logits(M, r, dsc, X):
    import torch.nn.functional as F
    return SCALE_R * F.normalize(X @ M.T, dim=-1) @ F.normalize(dsc @ M.T + r, dim=-1).T

def reptile_stage(steps=1500, E=4, T=8, lrM=5e-4, eps=0.5, ve=250, tag=""):
    import torch
    if TINY: steps, E, ve = 4, 2, 2
    torch.set_num_threads(int(os.environ.get("S17_THREADS", "2"))); torch.manual_seed(0); t0 = time.time()
    st = load_store(); P = old.PoolIdx(st); Xp = torch.tensor(P.X); tasks = old.sample_tasks(P.ds_labels, P.fit_ds, 800, 1); val = old.val_episodes(P, st); rng = np.random.default_rng(3)
    def val_acc(M, lrR):
        a = []
        for t, k, (dsc, pr, si, sy, qi, qy) in val:
            dt = torch.tensor(dsc); Xs_, ys_ = Xp[si], torch.tensor(sy); Ma, r = adapt(M, dt, len(t), lambda _: (Xs_, ys_), T, lrR, lrM)
            with torch.no_grad(): a.append(float((rep_logits(Ma, r, dt, Xp[qi]).argmax(1).numpy() == qy).mean()))
        return float(np.mean(a))
    I = torch.eye(D); grid = {lr: val_acc(I, lr) for lr in ((0.02,) if TINY else (0.005, 0.02, 0.05))}; lrR = max(grid, key=grid.get)
    log(f"[reptile] identity-init control on held-out TREC episodes by lrR {grid} -> lrR {lrR}")
    M = I.clone(); best, bestM, bad, hist = -1, None, 0, []
    for step in range(steps):
        delta = torch.zeros(D, D)
        for _ in range(E):
            task = tasks[int(rng.integers(len(tasks)))]; k = (1, 3, 10)[int(rng.integers(3))]; dsc = torch.tensor(np.stack([P.dX[l] for l in task])); K = len(task)
            def batch(_t, task=task, k=k):   # a fresh k rows per label at every inner step (Reptile's cross-batch signal)
                _, _, si, sy, _, _ = old.episode(P, task, k, rng, nq=0); return Xp[si], torch.tensor(sy)
            Ma, _ = adapt(M, dsc, K, batch, T, lrR, lrM); delta += (Ma - M) / E
        M = M + eps * delta
        if step % ve == ve - 1:
            v = val_acc(M, lrR); hist.append((step + 1, v)); log(f"[reptile] step {step+1} TREC val {v:.3f} (identity control {grid[lrR]:.3f}) [{time.time()-t0:.0f}s]")
            if v > best: best, bad, bestM = v, 0, M.clone()
            else:
                bad += 1
                if bad >= 3: break
    torch.save(dict(M=bestM, lrR=lrR, lrM=lrM, T=T, eps=eps), MODELS / f"REPTILE{tag}.pt")
    json.dump(dict(best_val=best, control_val=grid[lrR], grid=grid, hist=hist, seconds=time.time() - t0, params_M=D * D), open(HERE / (("tiny_" if TINY else "") + f"results_reptile{tag}.json"), "w"), indent=1)

def hyper_stage():
    import shutil
    for scale in (["clinc7"] if TINY else ["800", "clinc7", "100", "25", "3000"]):   # headline (800 tasks) first; CLINC-only = the S2b recipe
        try: old.meta("SETTR", scale)
        except Exception as ex: log("hyper", scale, "FAILED", repr(ex)[:300])
    (HERE / "models_json").mkdir(exist_ok=True)
    for f in MODELS.glob("SETTR_*.json"): shutil.copy(f, HERE / "models_json" / f.name)

def hyper2(scale):
    """second validation signal for the hypernetwork: early stopping on held-out MASSIVE intents (TREC joins the training pool), checked every 100 steps."""
    import shutil
    old.VAL_DS = "massive"; old.MODELS = STORE / "models_massive"; old.MODELS.mkdir(exist_ok=True); old.meta("SETTR", scale, ve=100)
    (HERE / "models_json2").mkdir(exist_ok=True)
    for f in old.MODELS.glob("SETTR_*.json"): shutil.copy(f, HERE / "models_json2" / f.name)

# ================================================================== evaluation: every arm on every (task, k, draw); results.json rewritten after each record
def lr_fit(X, y, C=10):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, max_iter=1000).fit(X, y)

def al_pick(Xp, yp, K, k, seed, student_X=None, student_y=None):
    """active selection: budget k*K rows from the unlabelled pool; the student (LR) picks the smallest top-2 gap; gold labels stand in for the teacher.
    student_X None -> baseline student: one random pool row per label to start, then (k-1)*K picked in up to 3 rounds. Else the synthetic-prior student picks all k*K in up to 4 rounds."""
    rg = np.random.default_rng(3000 + seed); chosen = []
    if student_X is None:
        chosen = [int(rg.choice(np.flatnonzero(yp == c))) for c in range(K)]; budget, rounds = (k - 1) * K, min(3, k - 1)
    else: budget, rounds = k * K, min(4, k)
    for r in range(rounds):
        sz = budget // rounds + (1 if r < budget % rounds else 0)
        ci = np.array(chosen, int); X = np.concatenate(([student_X] if student_X is not None else []) + [Xp[ci]]); y = np.concatenate(([student_y] if student_X is not None else []) + [yp[ci]])
        conf = confidence(full_proba(lr_fit(X, y), Xp, K)); conf[chosen] = np.inf; chosen += [int(i) for i in np.argsort(conf, kind="stable")[:sz]]
    return np.array(chosen, int)

def eval_stage(group):
    import torch; torch.set_num_threads(1)
    st = load_store(); EV = st["evals"]; R = load_res(); done = {(r["arm"], r["task"], r["k"], r["seed"]) for r in R["records"]}; secs = R["meta"].setdefault("arm_seconds", {}); params = R["meta"].setdefault("arm_params", {})
    views = {}; hyp = {}; reps = {}
    if group == "trained":
        for d, suffix in ((MODELS, ""), (STORE / "models_massive", ", early-stop on MASSIVE")):
            for f in sorted(d.glob("SETTR_*.pt")):
                try: G = make_models()["SETTR"](); G.load_state_dict(torch.load(f)); G.eval(); hyp[f.stem.split("_", 1)[1] + suffix] = G; params["3 hypernet"] = int(sum(p.numel() for p in G.parameters()))
                except Exception as ex: log("skip hypernet", f.name, repr(ex)[:100])
        for v in ("all", "nosib"):
            f = GPU_OUT / f"out_enc_{v}.npz"
            if f.exists(): z = np.load(f); views[v] = [{n: z[f"{n}{i}"] for n in ("Xs", "Xc", "Xq", "synth")} for i in range(len(EV))]
        for tag, suffix in (("", ""), ("2", " (gentle, eps 0.05)")):
            try: reps[suffix] = torch.load(MODELS / f"REPTILE{tag}.pt")
            except Exception as ex: log("skip reptile", tag, repr(ex)[:100])
        log("trained artifacts: hypernets", sorted(hyp), "encoders", sorted(views), "reptile", sorted(reps)); params["4 encoder"] = 33_360_000; params["5 reptile"] = D * D
    n_unit = 0
    def emit(arm, i, k, seed, pc, pq, dt):
        e = EV[i]; R["records"].append(dict(arm=arm, task=e["name"], family=e["family"], k=k, seed=seed, K=len(e["labels"]), **score(pc, e["yc"], pq, e["yq"]), sec=dt))
        secs[arm] = secs.get(arm, 0.0) + dt; done.add((arm, e["name"], k, seed)); save_res(R)
    def run(arm, i, k, seed, fn):   # fn() -> (pc, pq); skipped when already recorded; a failing arm is logged and does not stop the rest
        if (arm, EV[i]["name"], k, seed) in done: return
        t = time.time()
        try: pc, pq = fn()
        except Exception as ex: log(f"ARM FAILED {arm} {EV[i]['name']} k={k} seed={seed}: {repr(ex)[:300]}"); return
        emit(arm, i, k, seed, pc, pq, time.time() - t)
    def lr_arm(Xs_, Xc_, Xq_, idx, K, ys, pre=None):
        X, y = (Xs_[idx], ys[idx]) if pre is None else (np.concatenate([pre[0], Xs_[idx]]), np.concatenate([pre[1], ys[idx]]))
        clf = lr_fit(X, y); return full_proba(clf, Xc_, K), full_proba(clf, Xq_, K)
    for seed in range(NSEED):
        for i, e in enumerate(EV):
            K = len(e["labels"]); ys, Xs, Xc, Xq, SX, sy = e["ys"], e["Xs"], e["Xc"], e["Xq"], e["synth_X"], e["synth_y"]; desc = torch.tensor(e["desc_X"]); torch.manual_seed(seed)
            cand = np.random.default_rng(2000 + seed).choice(len(ys), min(len(ys), 40 * K), replace=False)   # the unlabelled pool for active selection: 40 random support rows per label
            for k in KS3:
                idx = draw_support(ys, k, seed, K)
                if group == "cheap":
                    run("1 baseline (frozen bge-small + LR)", i, k, seed, lambda: lr_arm(Xs, Xc, Xq, idx, K, ys))
                    run("2 synthetic prior", i, k, seed, lambda: lr_arm(Xs, Xc, Xq, idx, K, ys, (SX, sy)))
                    def al1():
                        sel = cand[al_pick(Xs[cand], ys[cand], K, k, seed)]; return lr_arm(Xs, Xc, Xq, sel, K, ys)
                    def al2():
                        sel = cand[al_pick(Xs[cand], ys[cand], K, k, seed, SX, sy)]; return lr_arm(Xs, Xc, Xq, sel, K, ys, (SX, sy))
                    run("6 active selection (baseline student)", i, k, seed, al1); run("6b active selection (synthetic-prior student)", i, k, seed, al2)
                    continue
                P = np.zeros((K, D), np.float32)
                for c in range(K): P[c] = Xs[idx][ys[idx] == c].mean(0)
                for scale, G in hyp.items():
                    def hy():
                        with torch.no_grad():
                            W, b = G(desc, torch.tensor(P), k); return [torch.softmax(G.logits(torch.tensor(X), W, b), -1).numpy().astype(np.float64) for X in (Xc, Xq)]
                    run(f"3 hypernetwork ({scale})", i, k, seed, hy)
                for v, V in views.items():
                    run(f"4 encoder pretrained on pool ({v})", i, k, seed, lambda: lr_arm(V[i]["Xs"], V[i]["Xc"], V[i]["Xq"], idx, K, ys))
                    if v == "all": run("4b encoder + synthetic prior", i, k, seed, lambda: lr_arm(V[i]["Xs"], V[i]["Xc"], V[i]["Xq"], idx, K, ys, (V[i]["synth"], sy)))
                for arm, M0, lrR, lrM, Tn in [x for sf, rp_ in reps.items() for x in ((f"5 Reptile start{sf}", rp_["M"], rp_["lrR"], rp_["lrM"], rp_["T"]), (f"5c control: identity start{sf}", torch.eye(D), rp_["lrR"], rp_["lrM"], rp_["T"]))]:
                    def rp(M0=M0, lrR=lrR, lrM=lrM, Tn=Tn):
                        Xk, yk = torch.tensor(Xs[idx]), torch.tensor(ys[idx]); Ma, r = adapt(M0, desc, K, lambda _: (Xk, yk), Tn, lrR, lrM)
                        with torch.no_grad(): return [torch.softmax(rep_logits(Ma, r, desc, torch.tensor(X)), -1).numpy().astype(np.float64) for X in (Xc, Xq)]
                    run(arm, i, k, seed, rp)
            n_unit += 1
        log(f"eval {group}: seed {seed} done, {len(R['records'])} records")
    save_res(R)

# ================================================================== aggregation (markdown tables)
def rows_to(cov_by_k, target=0.5):
    ks = sorted(cov_by_k)
    if cov_by_k[ks[0]] >= target: return "<=1"
    for a, b in zip(ks, ks[1:]):
        if cov_by_k[b] >= target:
            t = (target - cov_by_k[a]) / (cov_by_k[b] - cov_by_k[a]); return f"{np.exp(np.log(a) + t * (np.log(b) - np.log(a))):.1f}"
    return ">10"

def agg(path=None):
    R = json.loads(Path(path or RES).read_text()); rec = R["records"]; arms = sorted({r["arm"] for r in rec}); out = []
    def cell(arm, k, key, fam=None):
        v = [r[key] for r in rec if r["arm"] == arm and r["k"] == k and (fam is None or r["family"] == fam) and r[key] is not None]; return np.mean(v) if v else float("nan")
    def seed_sd(arm, k, key):   # s.d. over the 3 draws of the task-averaged value
        per = [np.mean(v) for s in range(NSEED) if (v := [r[key] for r in rec if r["arm"] == arm and r["k"] == k and r["seed"] == s and r[key] is not None])]
        return float(np.std(per, ddof=1)) if len(per) > 1 else float("nan")
    for title, key in (("Accuracy (test rows)", "acc"), ("Certified coverage at 5% budget (share of test rows answered by the student)", "cov"), ("Realized error on the answered rows (budget 5%)", "err")):
        out += [f"\n### {title}, mean over 5 tasks x 3 draws (+- s.d. over draws)\n", "| arm | k=1 | k=3 | k=10 |", "|---|---|---|---|"]
        for a in arms: out.append(f"| {a} | " + " | ".join(f"{cell(a, k, key):.3f} +- {seed_sd(a, k, key):.3f}" for k in KS3) + " |")
    out += ["\n### Certified coverage by family (CLINC 15-label tasks | Banking77 12-label tasks)\n", "| arm | k=1 | k=3 | k=10 |", "|---|---|---|---|"]
    for a in arms: out.append(f"| {a} | " + " | ".join(f"{cell(a, k, 'cov', 'clinc'):.2f} \\| {cell(a, k, 'cov', 'banking'):.2f}" for k in KS3) + " |")
    out += ["\n### Real rows per label needed to reach 50% certified coverage at 5% (linear interpolation in log k between k = 1, 3, 10)\n", "| arm | all 5 tasks | CLINC tasks | Banking77 tasks |", "|---|---|---|---|"]
    for a in arms: out.append(f"| {a} | " + " | ".join(rows_to({k: cell(a, k, "cov", f) for k in KS3}) for f in (None, "clinc", "banking")) + " |")
    out += ["\n### Pod seconds and parameters\n", "arm seconds (eval fit + predict, summed over all cells): " + json.dumps({k: round(v, 1) for k, v in R["meta"].get("arm_seconds", {}).items()}),
            "trainable parameters: " + json.dumps(R["meta"].get("arm_params", {}))]
    print("\n".join(out)); (HERE / ("tiny_summary.md" if TINY else "summary.md")).write_text("\n".join(out), encoding="utf-8")

# ================================================================== pod orchestration
def pod():
    import torch
    if not torch.cuda.is_available(): print("NO CUDA ON THIS POD - aborting so the launcher can retry on another GPU type", flush=True); sys.exit(3)
    LOGS = HERE / "logs"; LOGS.mkdir(exist_ok=True); env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    EVAL_AT = float(os.environ.get("S17_EVAL_AT_MIN", "26")) * 60
    def start(args, name): return subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), *args], env=env, stdout=open(LOGS / f"{name}.log", "w", encoding="utf-8"), stderr=subprocess.STDOUT, cwd=str(HERE))
    def tail(name, n=500): return (LOGS / f"{name}.log").read_text(encoding="utf-8", errors="replace")[-n:]
    log("cpus", os.cpu_count(), "gpu", torch.cuda.get_device_name(0))
    p = start(["prep"], "prep"); p.wait(); log("prep exit", p.returncode, tail("prep", 900))
    if p.returncode: sys.exit(4)
    bg = {n: start([n], n) for n in ("enc", "hyper", "reptile")}   # GPU job + two CPU jobs in parallel
    p = start(["eval", "cheap"], "eval_cheap"); p.wait(); log("eval cheap exit", p.returncode, tail("eval_cheap", 300))
    for n, p in bg.items():
        try: p.wait(timeout=max(1, EVAL_AT - (time.time() - T0)))
        except subprocess.TimeoutExpired: p.kill(); log(n, "cut off at the stage deadline")
        log(n, "exit", p.returncode, tail(n, 300))
    p = start(["eval", "trained"], "eval_trained"); p.wait(); log("eval trained exit", p.returncode, tail("eval_trained", 300)); agg(); log("pod done")

def pod2():
    """launch 2: second hypernetwork validation signal (4 task counts in parallel) + gentler Reptile; evaluates only the new artifacts (results.json from launch 1 is bundled and kept)."""
    import torch
    if not torch.cuda.is_available(): print("NO CUDA ON THIS POD - aborting", flush=True); sys.exit(3)
    LOGS = HERE / "logs"; LOGS.mkdir(exist_ok=True); env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    def start(args, name): return subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), *args], env=env, stdout=open(LOGS / f"{name}.log", "w", encoding="utf-8"), stderr=subprocess.STDOUT, cwd=str(HERE))
    log("cpus", os.cpu_count(), "gpu", torch.cuda.get_device_name(0)); p = start(["prep"], "prep"); p.wait(); log("prep exit", p.returncode)
    if p.returncode: sys.exit(4)
    bg = {f"hyper2_{s_}": start(["hyper2", s_], f"hyper2_{s_}") for s_ in ("25", "100", "800", "3000")}; bg["reptile2"] = start(["reptile2"], "reptile2")
    for n, p in bg.items():
        try: p.wait(timeout=max(1, 20 * 60 - (time.time() - T0)))
        except subprocess.TimeoutExpired: p.kill(); log(n, "cut off")
        log(n, "exit", p.returncode)
    p = start(["eval", "trained"], "eval_trained"); p.wait(); log("eval trained exit", p.returncode); agg(); log("pod2 done")

def tiny():
    for f in (prep, enc_stage, hyper_stage, reptile_stage): f();
    eval_stage("cheap"); eval_stage("trained"); agg()

if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--tiny"]; st = a[0]
    {"prep": prep, "enc": enc_stage, "hyper": hyper_stage, "reptile": reptile_stage, "pod": pod, "tiny": tiny, "agg": agg, "eval": lambda: eval_stage(a[1]), "hyper2": lambda: hyper2(a[1]), "reptile2": lambda: reptile_stage(eps=0.05, ve=50, lrM=2e-4, tag="2"), "pod2": pod2}[st]()
