"""S24 helpers: embeddings, LLM caller (cached, cost-capped), scoring, and the data prep (5 unseen tasks + pretraining pool).
ponytail: frozen bge-small embeddings + numpy/sklearn everywhere; torch only for the prior-fitted network."""
import sys, os, json, re, time, hashlib, threading
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
import common
from common import certify, apply_certificate, confidence, load_clinc150, load_banking77, ROOT

TINY = bool(os.environ.get("S24_TINY"))  # local plumbing check: a handful of rows, no network LLM calls, nothing real written
CACHE = ROOT / ".cache" / "tiny" if TINY else common.CACHE  # embedding cache (label_descriptions.json etc. always come from common.CACHE)
CACHE.mkdir(parents=True, exist_ok=True)
DATA = ROOT / ".data"; DATA.mkdir(exist_ok=True)
HFDIR = DATA / ("s17_hf" if (DATA / "s17_hf").exists() else "s24_hf")
D = 384; BUDGET = 0.05
KS = [1, 3, 10]; SEEDS = 3; NCAL = 300
EVAL_CLINC = ["travel", "work"]; VAL_CLINC = "meta"
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:.0f}s]", *a, flush=True)

# ------------------------------------------------------------------ embeddings (same cache key as common.embed; GPU when there is one)
_ST = {}
def embed(texts, model=common.EMB_MODEL):
    texts = list(texts); key = hashlib.sha1((model + "\x00".join(texts)).encode()).hexdigest()[:16]; f = CACHE / f"emb_{key}.npy"
    if f.exists(): return np.load(f)
    if model not in _ST:
        import torch; from sentence_transformers import SentenceTransformer
        _ST[model] = SentenceTransformer(model, device="cuda" if torch.cuda.is_available() else "cpu")
    t = time.time(); e = _ST[model].encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    log(f"[embed] {len(texts)} texts in {time.time()-t:.1f}s"); np.save(f, e); return e

# ------------------------------------------------------------------ LLM (ASI:One first, OpenRouter nano fallback), cached, hard cost cap
LLM_CAP = 0.45  # USD; the task cap is 0.5
_lock = threading.Lock(); LLMCACHE = HERE / "responses.jsonl"; _cache = {}; _client = {}
if LLMCACHE.exists() and not TINY:
    for ln in LLMCACHE.read_text(encoding="utf-8").splitlines():
        try: r = json.loads(ln); _cache[r["key"]] = r
        except Exception: pass
def _openrouter():
    from openai import OpenAI
    env = (ROOT.parent / ".env").read_text(); cl = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=re.search(r"OPENROUTERKEY=(\S+)", env).group(1), timeout=120)
    _client.clear(); _client.update(cl=cl, model="openai/gpt-5.4-nano", pin=0.2e-6, pout=1.25e-6, extra={"reasoning": {"effort": "minimal"}})
def _get_client():
    with _lock:
        if _client: return _client
        from openai import OpenAI
        env = (ROOT.parent / ".env").read_text()
        try:
            cl = OpenAI(base_url="https://api.asi1.ai/v1", api_key=re.search(r"ASI1KEY=(\S+)", env).group(1), timeout=120)
            cl.chat.completions.create(model="asi1-mini", messages=[{"role": "user", "content": "Reply OK"}], max_tokens=3)
            _client.update(cl=cl, model="asi1-mini", pin=0.18e-6, pout=0.45e-6, extra={})
        except Exception as e:
            log("ASI:One unavailable, using OpenRouter:", repr(e)[:120]); _openrouter()
        return _client
def llm_spend(): return sum(r.get("cost", 0) for r in _cache.values())
def llm(prompt, max_tokens=4000, temperature=0.2, tag=""):
    """Returns dict(text, cost, secs, model) or None (tiny mode, cost cap reached, or all retries failed). Every raw response is cached in responses.jsonl."""
    if TINY: return None
    c = _get_client(); m0 = c["model"]; key = hashlib.sha1(json.dumps([m0, prompt, max_tokens, temperature, tag]).encode()).hexdigest()
    if key in _cache: return _cache[key]
    if llm_spend() >= LLM_CAP: return None
    for attempt in range(4):
        try:
            t = time.time(); r = c["cl"].chat.completions.create(model=c["model"], messages=[{"role": "user", "content": prompt}], temperature=temperature, max_tokens=max_tokens, extra_body=c["extra"])
            rec = dict(key=key, model=c["model"], text=r.choices[0].message.content or "", secs=time.time() - t,
                       cost=r.usage.prompt_tokens * c["pin"] + r.usage.completion_tokens * c["pout"], prompt_head=prompt[:80])
            with _lock:
                _cache[key] = rec
                with open(LLMCACHE, "a", encoding="utf-8") as f: f.write(json.dumps(rec) + "\n")
            return rec
        except Exception as e:
            log("llm retry", attempt, repr(e)[:100]); time.sleep(2 ** attempt)
    with _lock:  # ASI:One kept failing: switch every later call to OpenRouter gpt-5.4-nano (once, whichever thread gets here first), then retry this prompt
        if m0 == "asi1-mini" and _client.get("model") == "asi1-mini": log("switching LLM to OpenRouter gpt-5.4-nano"); _openrouter()
    return llm(prompt, max_tokens, temperature, tag) if m0 == "asi1-mini" else None
def llm_json(prompt, **kw):
    r = llm(prompt, **kw)
    try: return json.loads(r["text"][r["text"].find("{"): r["text"].rfind("}") + 1])
    except Exception: return {}

# ------------------------------------------------------------------ scoring: accuracy + fixed-grid certificate (300 calibration rows, 5% budget) applied to the test rows
def auroc(conf, ok):
    from sklearn.metrics import roc_auc_score
    ok = np.asarray(ok, bool)
    return float(roc_auc_score(ok, conf)) if 0 < ok.sum() < len(ok) else None
def score_conf(conf_c, ok_c, conf_q, ok_q, budget=BUDGET):
    cert = certify(conf_c, ok_c, budget); out = dict(acc=float(np.mean(ok_q)), auroc=auroc(conf_q, ok_q))
    if not np.isfinite(cert["cutoff"]): return dict(out, cov=0.0, err=None, cal_cov=0.0)
    h = apply_certificate(conf_q, ok_q, cert["cutoff"]); return dict(out, cov=h["coverage"], err=h["error"], cal_cov=cert["coverage"])
def score(pc, yc, pq, yq): return score_conf(confidence(pc), pc.argmax(1) == yc, confidence(pq), pq.argmax(1) == yq)
def draw_support(ys, k, seed, K):
    r = np.random.default_rng(1000 + seed); idx = []
    for c in range(K): idx += list(r.choice(np.flatnonzero(ys == c), min(k, int((ys == c).sum())), replace=False))
    return np.array(idx, int)

# ------------------------------------------------------------------ data prep
clean = lambda t, n=300: re.sub(r"\s+", " ", str(t)).strip()[:n]
def shrink(texts, y, keep_ids, per):  # TINY only
    y = np.asarray(y); cnt = {}; idx = []
    for i, yy in enumerate(y.tolist()):
        if yy in keep_ids and cnt.get(yy, 0) < per: idx.append(i); cnt[yy] = cnt.get(yy, 0) + 1
    return [texts[i] for i in idx], y[idx]

def synth_for(family, labels, desc):
    """10 LLM-written practice examples per label: the shared S2b cache first (shared file is never modified; not used for names CLINC and Banking77 share);
    labels it lacks (all CLINC travel labels) are written by one LLM call per 15 labels into synth_extra.json."""
    SYN = json.load(open(common.CACHE / "synth_examples.json")); path = HERE / "synth_extra.json"; own = json.loads(path.read_text()) if path.exists() else {}
    DS = json.load(open(common.CACHE / "label_descriptions.json")); collide = set(DS["clinc150"]) & set(DS["banking77"])
    have = lambda l: own.get(f"{family}::{l}") if len(own.get(f"{family}::{l}", [])) >= 5 else (SYN[l] if l not in collide and len(SYN.get(l, [])) >= 5 else None)
    need = [l for l in labels if not have(l)]
    if need and not TINY:
        for i in range(0, len(need), 15):
            chunk = need[i:i + 15]
            prompt = ("For each label below, write 10 short, varied, realistic user messages that would get that label (different phrasings, lengths, some typos, some indirect). "
                      "Reply with a JSON object mapping label -> list of 10 strings, nothing else.\n" + "\n".join(f"{l}: {desc[l]}" for l in chunk))
            for att in range(3):
                d = llm_json(prompt, max_tokens=8000, temperature=0.7, tag=f"s24-synth-{att}")
                if all(c in d and len(d[c]) >= 5 for c in chunk): break
            for c in chunk: own[f"{family}::{c}"] = [str(x) for x in d.get(c, [])][:10] or [desc[c]]
        path.write_text(json.dumps(own, indent=1), encoding="utf-8")
    return {l: (have(l) or [desc[l]])[:10] for l in labels}

def prep():
    """Returns dict(evals=[5 tasks], pool=dict(X,key,ds,grp), val=dict(X,key)). Evaluation labels are disjoint from every pool label (CLINC domains / Banking77 label ids / other datasets)."""
    import datasets
    rng = np.random.default_rng(24); CAP, MINROWS, ncal = (12, 8, 20) if TINY else (100, 20, NCAL)
    DESC = json.load(open(common.CACHE / "label_descriptions.json")); DOM = json.load(open(common.CACHE / "clinc_domains.json"))
    ld = lambda *a, **k: datasets.load_dataset(*a, cache_dir=str(HFDIR), **k)
    c = load_clinc150(); cn = c["names"]; b = load_banking77(); bn = b["names"]
    held = EVAL_CLINC[:1] if TINY else EVAL_CLINC
    dom_labs = lambda ds: [n for n in cn if n != "oos" and DOM[n] in ds]
    eval_labs = {d: dom_labs([d])[:4 if TINY else None] for d in held}
    clinc_pool = dom_labs([d for d in sorted(set(DOM.values())) if d not in EVAL_CLINC + [VAL_CLINC]]); val_labs = dom_labs([VAL_CLINC])
    perm = np.random.default_rng(0).permutation(77); b_eval = [sorted(perm[12 * j:12 * j + 12].tolist()) for j in range(3)]; b_pool = sorted(perm[36:].tolist())
    if TINY:
        clinc_pool = clinc_pool[::17][:9]; val_labs = val_labs[:3]
        keep = {cn.index(n) for n in clinc_pool + val_labs + sum(eval_labs.values(), [])}
        c = dict(names=cn, **{s: shrink(*c[s], keep, per) for s, per in (("train", 14), ("validation", 5), ("test", 4))})
        b_eval = [b_eval[0][:4]]; b_pool = b_pool[:3]; keep = set(b_eval[0]) | set(b_pool)
        b = dict(names=bn, train=shrink(*b["train"], keep, 14 + 5), test=shrink(*b["test"], keep, 4))
    cid = {n: i for i, n in enumerate(cn)}
    evals = []
    def mk(name, family, labels, desc, ts, ys, tc, yc, tq, yq):
        syn = synth_for(family, labels, desc)
        e = dict(name=name, family=family, labels=labels, K=len(labels), desc_t=[desc[l] for l in labels], synth_t=[s for l in labels for s in syn[l]],
                 synth_y=np.array([i for i, l in enumerate(labels) for _ in syn[l]]), ts=list(ts), ys=np.asarray(ys), tc=list(tc), yc=np.asarray(yc), tq=list(tq), yq=np.asarray(yq))
        e["synth_X"] = embed(e["synth_t"]); e["Xs"], e["Xc"], e["Xq"] = embed(e["ts"]), embed(e["tc"]), embed(e["tq"]); evals.append(e)
    for dom in held:  # CLINC domain: support = train, calibration = validation (20/label = 300 for 15 labels), test = test
        labs = eval_labs[dom]; li = {cid[n]: i for i, n in enumerate(labs)}
        sel = lambda split: [i for i, y in enumerate(c[split][1]) if y in li]; itr, iva, ite = sel("train"), sel("validation"), sel("test")
        mk(f"clinc/{dom}", "clinc", labs, {n: DESC["clinc150"].get(n, n) for n in labs}, [c["train"][0][i] for i in itr], [li[c["train"][1][i]] for i in itr],
           [c["validation"][0][i] for i in iva], [li[c["validation"][1][i]] for i in iva], [c["test"][0][i] for i in ite], [li[c["test"][1][i]] for i in ite])
    for j, lab_ids in enumerate(b_eval):  # Banking77 12-label sub-tasks: calibration = ncal random train rows, support = the rest, test = official test rows
        li = {y: i for i, y in enumerate(lab_ids)}; labs = [bn[y] for y in lab_ids]
        itr = np.array([i for i, y in enumerate(b["train"][1]) if y in li]); ite = np.array([i for i, y in enumerate(b["test"][1]) if y in li])
        pm = np.random.default_rng(100 + j).permutation(len(itr)); ic, isup = itr[pm[:ncal]], itr[pm[ncal:]]
        g = lambda a, i: [a[0][x] for x in i]; yl = lambda a, i: [li[a[1][x]] for x in i]
        mk(f"banking/sub{j}", "banking", labs, DESC["banking77"], g(b["train"], isup), yl(b["train"], isup), g(b["train"], ic), yl(b["train"], ic), g(b["test"], ite), yl(b["test"], ite))
    log("eval tasks", [(e["name"], e["K"], len(e["ts"]), len(e["tc"]), len(e["tq"]), len(e["synth_t"])) for e in evals])
    # ---------------- pretraining pool (for the prior-fitted network): CLINC train domains, Banking77 labels outside the 3 sub-tasks, TREC-QC, emotion, ag_news; CLINC "meta" = validation
    P = dict(text=[], key=[], ds=[], X=[]); grp = {}; V = dict(text=[], key=[], X=[])
    def add(store, ds, keys, texts, X, group=None):
        keys = np.asarray(keys)
        for k in sorted(set(keys.tolist())):
            idx = np.flatnonzero(keys == k)
            if len(idx) < MINROWS: continue
            idx = rng.permutation(idx)[:CAP]; key = f"{ds}::{k}"; grp[key] = group(k) if group else ds
            store["text"] += [texts[i] for i in idx]; store["key"] += [key] * len(idx); store["X"].append(X[idx])
            if "ds" in store: store["ds"] += [ds] * len(idx)
    for store, names in ((P, clinc_pool), (V, val_labs)):
        m = np.isin(c["train"][1], [cid[n] for n in names]); texts = [t for t, f in zip(c["train"][0], m) if f]
        add(store, "clinc", [cn[y] for y in c["train"][1][m]], texts, embed(texts), group=lambda k: DOM[k])
    m = np.isin(b["train"][1], b_pool); texts = [t for t, f in zip(b["train"][0], m) if f]
    add(P, "banking", [bn[y] for y in b["train"][1][m]], texts, embed(texts))
    for ds, dargs, lc in (("trec", ("SetFit/TREC-QC",), "label_original"), ("emotion", ("SetFit/emotion",), "label_text"), ("ag_news", ("SetFit/ag_news",), "label_text")):
        if TINY and ds == "ag_news": continue
        df = ld(*dargs)["train"]
        if TINY: df = df.select(range(min(len(df), 3000)))
        df = df.to_pandas(); keep = []; groups = sorted(df.groupby(lc).indices.items(), key=lambda kv: str(kv[0]))
        for k, idx in (groups[:3] if TINY else groups):
            if len(idx) >= MINROWS: keep += list(rng.permutation(idx)[:CAP])
        sub = df.iloc[sorted(keep)]; texts = [clean(t) for t in sub.text]; add(P, ds, sub[lc].values, texts, embed(texts))
    ev_texts = {t.lower() for e in evals for t in e["ts"] + e["tc"] + e["tq"]}
    keep = np.array([i for i, t in enumerate(P["text"]) if t.lower() not in ev_texts])
    pool = dict(X=np.concatenate(P["X"])[keep], key=np.array(P["key"])[keep], ds=np.array(P["ds"])[keep], grp=grp)
    val = dict(X=np.concatenate(V["X"]), key=np.array(V["key"]))
    ev_names = {f: {l for e in evals if e["family"] == f for l in e["labels"]} for f in ("clinc", "banking")}
    assert not any(k.split("::", 1)[1] in ev_names[k.split("::")[0]] for k in set(pool["key"].tolist()) if k.split("::")[0] in ev_names), "pool/eval label overlap"
    info = {d: len({k for k, dd in zip(pool["key"], pool["ds"]) if dd == d}) for d in sorted(set(pool["ds"].tolist()))}
    log("pool labels per dataset", info, "rows", len(pool["X"]), "dups removed", len(P["text"]) - len(keep), "| val rows", len(val["X"]))
    return dict(evals=evals, pool=pool, val=val, info=dict(pool_labels=info, pool_rows=len(pool["X"]), dups_removed=len(P["text"]) - len(keep), val_rows=len(val["X"])))
