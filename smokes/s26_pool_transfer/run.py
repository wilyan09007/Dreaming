"""S26 - does transfer from STRUCTURALLY SIMILAR call sites change the warm-start verdict (S17) and which base transfers better (S17 33M encoder vs S8 0.5B decoder)?
Pool = 8 public "categorize this short record" datasets (+1 held-out pool dataset for early stopping). Eval tasks = WHOLE held-out datasets (US bank descriptors,
Amazon product titles) + CLINC travel domain. Hold-out is by dataset (and by merchant/store inside an eval dataset), never by row.
Arms: 1 LR baseline | 2 synthetic prior | 3 hypernetwork | 4 encoder pretrained on pool (+4b synth) | 5 shared LoRA 0.5B vs 5c LoRA from base (+5b/5d with synth).
Stages (python run.py <stage>): prep | enc | hyper | decpre | eval cheap|arm4|arm3 | dec <worker> <nworkers> | agg | pod | tiny."""
import sys, os, json, re, time, pickle, subprocess, hashlib, threading
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; sys.path.insert(0, str(HERE.parent))
import numpy as np, pandas as pd
import common
TINY = "tiny" in sys.argv[1:2] or "--tiny" in sys.argv
WORK = ROOT / ".data" / ("s26_tiny" if TINY else "s26_store"); WORK.mkdir(parents=True, exist_ok=True)   # big intermediates: never bundled or returned
OUT = HERE / "tiny_out" if TINY else HERE; OUT.mkdir(exist_ok=True)
NSEED, KS3, D = (1 if TINY else 3), (1, 3, 10), 384
CAP, MINROWS, NCAL, NTEST, SN = (4, 3, 9, 12, 2) if TINY else (100, 20, 300, 800, 10)   # rows/pool label | min rows | calibration rows | test rows | synthetic per label
PRE_STEPS, PRE_BS, CKPTS = (4, 4, (2, 4)) if TINY else (600, 32, (200, 400, 600))        # SAME pretraining budget for encoder and decoder
FT_STEPS, FT_BS, MAXLEN = (3, 4, 32) if TINY else (60, 32, 80)                            # decoder fine-tune on a new step (S8 recipe)
QWEN, VAL_DS = "Qwen/Qwen2.5-0.5B", "bitext"
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:.0f}s]", *a, flush=True)
def dev():
    import torch; return "cuda" if torch.cuda.is_available() else "cpu"
def jload(p, default=None):
    return json.loads(Path(p).read_text()) if Path(p).exists() else default
def jsave(p, obj):
    tmp = Path(str(p) + ".tmp"); tmp.write_text(json.dumps(obj)); os.replace(tmp, p)
def load_store(): return pickle.load(open(WORK / "store.pkl", "rb"))

# ---------------------------------------------------------------- dataset descriptions: step name (decoder prompt) and context (LLM prompts)
STEP = {"bank_us": "categorize US bank transaction", "amazon": "categorize product listing by department", "clinc_travel": "travel assistant request",
        "dikshith": "categorize bank transaction descriptor", "sumeet": "categorize Indian bank transaction narration", "rajesh": "categorize card merchant by spending",
        "tickets": "route customer support ticket to queue", "lezh": "categorize marketplace listing by product category", "flip": "categorize product name by department",
        "banking77": "classify banking support query", "clinc_fin": "classify banking or credit card request", "bitext": "classify customer support request intent"}
CTX = {"bank_us": "raw US bank-statement transaction descriptors as shown in a bank feed (a '[debit]' or '[credit]' prefix, then merchant text with store numbers, city, processor codes such as PPD ID or WEB ID); the label is the spending category",
       "amazon": "Amazon product listings written as 'Title: ... | Store: ... | Manufacturer: ...'; the label is the top-level department",
       "clinc_travel": "short spoken requests to a travel voice assistant; the label is the intent",
       "dikshith": "bank transaction descriptors from a card statement; the label is the spending category", "sumeet": "Indian bank transaction narrations (UPI, IMPS, POS, ECS); the label is the spending category",
       "rajesh": "merchant names from card transactions; the label is the spending category", "tickets": "English customer-support tickets (subject and start of the body) routed to the support queue named by the label",
       "lezh": "online marketplace listings (title plus short description); the label is the product category", "flip": "Flipkart product names; the label is the top-level product category",
       "bitext": "customer-service chat messages to an online retailer; the label is the intent"}

# ---------------------------------------------------------------- LLM (ASI:One first, pinned OpenRouter fallback), every raw response cached
_lock, _cache, _client, LLMCACHE, CAP_USD = threading.Lock(), {}, {}, HERE / "responses.jsonl", 0.5
if LLMCACHE.exists():
    for ln in LLMCACHE.read_text(encoding="utf-8").splitlines():
        try: r = json.loads(ln); _cache[r["key"]] = r
        except Exception: pass
def _get_client():
    with _lock:
        if _client: return _client
        from openai import OpenAI
        env = (ROOT / ".env").read_text()
        try:
            cl = OpenAI(base_url="https://api.asi1.ai/v1", api_key=re.search(r"ASI1KEY=(\S+)", env).group(1), timeout=120)
            cl.chat.completions.create(model="asi1-mini", messages=[{"role": "user", "content": "Reply OK"}], max_tokens=3)
            _client.update(cl=cl, model="asi1-mini", pin=0.18e-6, pout=0.45e-6, extra={})
        except Exception as e:
            print("ASI:One unavailable, OpenRouter fallback:", repr(e)[:120], flush=True)
            cl = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=re.search(r"OPENROUTERKEY=(\S+)", env).group(1), timeout=120)
            _client.update(cl=cl, model="openai/gpt-5.4-nano", pin=0.2e-6, pout=1.25e-6, extra={"reasoning": {"effort": "minimal"}, "provider": {"order": ["openai"], "allow_fallbacks": False}})
        return _client
def llm_spend(): return sum(r.get("cost", 0) for r in _cache.values())
def llm_json(prompt, max_tokens=8000, temperature=0.7, tag=""):
    if TINY: return {}                       # no network in the local check; callers fall back to label names
    c = _get_client(); key = hashlib.sha1(json.dumps([c["model"], prompt, max_tokens, temperature, tag]).encode()).hexdigest()
    if key not in _cache:
        if llm_spend() > CAP_USD: log("LLM cap reached"); return {}
        for attempt in range(4):
            try:
                r = c["cl"].chat.completions.create(model=c["model"], messages=[{"role": "user", "content": prompt}], temperature=temperature, max_tokens=max_tokens, extra_body=c["extra"])
                rec = dict(key=key, model=c["model"], text=r.choices[0].message.content or "", cost=r.usage.prompt_tokens * c["pin"] + r.usage.completion_tokens * c["pout"], prompt_head=prompt[:80])
                with _lock:
                    _cache[key] = rec
                    with open(LLMCACHE, "a", encoding="utf-8") as f: f.write(json.dumps(rec) + "\n")
                break
            except Exception as e: print("llm retry", attempt, repr(e)[:100], flush=True); time.sleep(2 ** attempt)
        else: return {}
    txt = _cache[key]["text"]
    try: return json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
    except Exception: return {}

# ---------------------------------------------------------------- embeddings: frozen bge-small via transformers (CLS pooling + L2 norm = the sentence-transformers pipeline of this model)
_bge = {}
def bge():
    if not _bge:
        from transformers import AutoTokenizer, AutoModel
        _bge["tok"] = AutoTokenizer.from_pretrained(common.EMB_MODEL); _bge["m"] = AutoModel.from_pretrained(common.EMB_MODEL).to(dev()).eval()
    return _bge["tok"], _bge["m"]
def embed_all(model, tok, texts, bs=256, maxlen=128):
    import torch, torch.nn.functional as F
    model.eval(); texts = list(texts); order = np.argsort([len(t) for t in texts]); out = np.zeros((len(texts), D), np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            idx = order[i:i + bs]; bt = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=maxlen, return_tensors="pt").to(dev())
            out[idx] = F.normalize(model(**bt).last_hidden_state[:, 0].float(), dim=-1).cpu().numpy()
    return out
def embed(texts):
    texts = list(texts); f = WORK / f"emb_{hashlib.sha1(chr(0).join(texts).encode()).hexdigest()[:16]}.npy"
    if f.exists(): return np.load(f)
    tok, m = bge(); e = embed_all(m, tok, texts); np.save(f, e); return e

# ---------------------------------------------------------------- scoring: accuracy + certificate fixed on the 300 calibration rows, applied to the test rows
def full_proba(clf, X, K):
    p = np.zeros((len(X), K)); p[:, clf.classes_] = clf.predict_proba(X); return p
def score(pc, yc, pq, yq, budget=0.05):
    cc, okc, cq, okq = common.confidence(pc), pc.argmax(1) == yc, common.confidence(pq), pq.argmax(1) == yq
    cert = common.certify(cc, okc, budget)
    if not np.isfinite(cert["cutoff"]): return dict(acc=float(okq.mean()), cov=0.0, err=None, cal_cov=0.0)
    h = common.apply_certificate(cq, okq, cert["cutoff"]); return dict(acc=float(okq.mean()), cov=h["coverage"], err=h["error"], cal_cov=cert["coverage"])
def draw_support(ys, k, seed, K):
    r = np.random.default_rng(1000 + seed); idx = []
    for c in range(K):
        p = np.flatnonzero(ys == c); idx += list(r.choice(p, min(k, len(p)), replace=False))
    return np.array(idx, int)
def lr_fit(X, y):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=10, max_iter=1000).fit(X, y)

# ================================================================== data
def hf_file(repo, f):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, f, repo_type="dataset")
def rd_csv(repo, f, **kw):   # TINY reads only the first rows over HTTP
    return pd.read_csv(f"https://huggingface.co/datasets/{repo}/resolve/main/{f}", nrows=2500, **kw) if TINY else pd.read_csv(hf_file(repo, f), **kw)
def rd_pq(repo, f, columns=None):
    if TINY:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem
        with HfFileSystem().open(f"datasets/{repo}/{f}", "rb") as fh: return next(pq.ParquetFile(fh).iter_batches(batch_size=2500)).to_pandas()
    return pd.read_parquet(hf_file(repo, f), columns=columns)
clean = lambda t, n=300: re.sub(r"\s+", " ", str(t)).strip()[:n]
STOP = set("withdrawal from to preauthorized preapproved payment pmt bill user ppd id web inst xfer paypal pypl orig name entry descr sec express checkout purchase debit credit card pos the and for online transfer deposit ach recurring autopay".split())
def merchant_key(t):   # first brand-like token of a bank descriptor = the "merchant" used to group rows
    w = [x for x in re.findall(r"[a-z]{3,}", t.lower()) if x not in STOP]; return w[0] if w else t.lower()[:10]
def ld_bank_us():
    d = rd_csv("DoDataThings/us-bank-transaction-categories-v2", "transactions-synthetic.csv").rename(columns={"description": "text", "category": "label"})[["text", "label"]]; d["group"] = d.text.map(merchant_key); return d
def ld_amazon():
    d = rd_pq("SurAyush/ProductTitle-To-Category", "data/train-00000-of-00001.parquet", columns=["input", "target"])
    if not TINY: d = d.sample(80000, random_state=0)
    t = d["input"]; store = t.str.extract(r"Store: (.*)")[0]; lab = d["target"].str.extract(r'"L0_category":\s*"([^"]*)"')[0]
    text = "Title: " + t.str.extract(r"Title: (.*)")[0].fillna("") + " | Store: " + store.fillna("") + " | Manufacturer: " + t.str.extract(r"Manufacturer: (.*)")[0].fillna("")
    vc = lab.value_counts(); keep = vc[vc >= (6 if TINY else 500)].index[:15]
    o = pd.DataFrame(dict(text=text, label=lab, group=store.fillna("").str.lower().str.strip())); o = o[o.label.isin(keep)].reset_index(drop=True)
    m = o.group.isin(["", "none", "nan"]).to_numpy(); o.loc[m, "group"] = ["row%d" % i for i in np.flatnonzero(m)]; return o
def ld_dikshith(): return rd_pq("Dikshith4500/bank-transaction-categories-50k", "data/train-00000-of-00001.parquet")[["text", "category"]].rename(columns={"category": "label"})
def ld_sumeet(): return rd_csv("Sumeetgpt/indian-transaction-categorization-synthetic", "synthetic_indian_transactions.csv").rename(columns={"category": "label"})[["text", "label"]]
def ld_rajesh():
    names = ["Shopping", "Dining Out", "Entertainment", "Transportation", "Housing", "Payments/Credits", "Utilities", "Service Subscriptions"]
    d = pd.concat([rd_pq("rajeshradhakrishnan/fin-transaction-category", f"data/{s}-00000-of-00001.parquet") for s in ("train", "test")])
    return pd.DataFrame(dict(text=d["merchant"].astype(str), label=[names[int(i)] for i in d["category"]]))
def ld_tickets():
    d = rd_csv("Tobi-Bueck/customer-support-tickets", "dataset-tickets-multi-lang-4-20k.csv", usecols=["subject", "body", "queue", "language"]); d = d[d.language == "en"]
    return pd.DataFrame(dict(text=d.subject.fillna("") + ". " + d.body.fillna(""), label=d.queue))
def ld_lezh(): return rd_pq("Lezh1n/ecommerce-product-classification-by-categories", "data/train-00000-of-00001.parquet")[["text", "label"]]
def ld_flip():
    d = rd_csv("jason1966/PromptCloudHQ_flipkart-products", "flipkart_com-ecommerce_sample.csv", usecols=["product_name", "product_category_tree"])
    lab = d.product_category_tree.map(lambda s: (json.loads(s)[0].split(" >> ")[0] if isinstance(s, str) and s.startswith("[") else None))
    o = pd.DataFrame(dict(text=d.product_name, label=lab)).dropna(); vc = o.label.value_counts(); return o[o.label.isin(vc[vc >= (6 if TINY else 100)].index[:15])]
def ld_bitext(): return rd_csv("bitext/Bitext-customer-support-llm-chatbot-training-dataset", "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv", usecols=["instruction", "intent"]).rename(columns={"instruction": "text", "intent": "label"})
def ld_banking77():
    b = common.load_banking77(); return pd.DataFrame(dict(text=b["train"][0], label=[b["names"][y] for y in b["train"][1]]))
def ld_clinc(domains, split="train"):
    c = common.load_clinc150(); DOM = json.load(open(common.CACHE / "clinc_domains.json")); n = c["names"]; t, y = c[split]
    keep = [i for i, v in enumerate(y) if DOM[n[v]] in domains]; return pd.DataFrame(dict(text=[t[i] for i in keep], label=[n[y[i]] for i in keep]))
POOL_LOADERS = {"dikshith": ld_dikshith, "sumeet": ld_sumeet, "rajesh": ld_rajesh, "tickets": ld_tickets, "lezh": ld_lezh, "flip": ld_flip, "banking77": ld_banking77,
                "clinc_fin": lambda: ld_clinc({"banking", "credit_cards"}), VAL_DS: ld_bitext}

def group_eval(name, df, rng):
    """WHOLE held-out dataset; support / calibration / test rows come from DISJOINT groups (merchant or store). Rows per group capped so no merchant dominates."""
    df = df.dropna().copy(); df["text"] = df.text.map(clean); df = df.drop_duplicates("text").sample(frac=1, random_state=int(rng.integers(1 << 30))).reset_index(drop=True)
    df = df[df.groupby("group").cumcount() < (4 if TINY else 15)]; df = df[df.groupby("label").cumcount() < (12 if TINY else 400)].reset_index(drop=True)
    g = df.groupby("group").size(); order = rng.permutation(len(g)); gn, gs = g.index.to_numpy()[order], g.to_numpy()[order]; cum = np.cumsum(gs) / gs.sum()
    pv = np.where(cum <= 0.5, 0, np.where(cum <= 0.7, 1, 2)); part = dict(zip(gn, pv)); p = df.group.map(part).to_numpy()
    minsup = 3 if TINY else 10; lab_arr = df.label.to_numpy()
    labs = sorted(l for l in df.label.unique() if (((p == 0) & (lab_arr == l)).sum() >= minsup) and (((p == 2) & (lab_arr == l)).sum() >= 1))
    labs = labs[:4] if TINY else labs; li = {l: i for i, l in enumerate(labs)}; inl = df.label.isin(labs).to_numpy()
    sup, cal, tst = [np.flatnonzero((p == j) & inl) for j in range(3)]; cal = rng.choice(cal, min(NCAL, len(cal)), replace=False); tst = rng.choice(tst, min(NTEST, len(tst)), replace=False)
    f = lambda ix: ([df.text.iloc[i] for i in ix], np.array([li[df.label.iloc[i]] for i in ix]))
    (ts, ys), (tc, yc), (tq, yq) = f(sup), f(cal), f(tst); grp = df.group.to_numpy()
    info = dict(n_labels=len(labs), n_groups=int(len(g)), groups_sup=int((pv == 0).sum()), groups_cal=int((pv == 1).sum()), groups_test=int((pv == 2).sum()), n_support_pool=len(ts), n_cal=len(tc), n_test=len(tq),
                cal_groups_overlap_support=len(set(grp[cal]) & set(grp[sup])), test_groups_overlap_support=len(set(grp[tst]) & set(grp[sup])), cal_groups_overlap_test=len(set(grp[cal]) & set(grp[tst])))
    return dict(name=name, labels=labs, ts=ts, ys=ys, tc=tc, yc=yc, tq=tq, yq=yq), info

def prep():
    rng = np.random.default_rng(26); info = {"datasets": {}}; DESC = json.load(open(common.CACHE / "label_descriptions.json"))
    if not TINY: log("LLM ping", llm_json('Reply with exactly this JSON object and nothing else: {"ok": 1}', max_tokens=50, temperature=0, tag="ping"))
    P = dict(text=[], key=[], ds=[]); LAB = {}
    for ds, fn in POOL_LOADERS.items():
        t = time.time(); df = fn().dropna(subset=["text", "label"]).copy(); df["text"] = df.text.map(clean); df["label"] = df.label.astype(str); df = df[df.text.str.len() > 2].drop_duplicates("text")
        vc = df.label.value_counts(); labs = sorted(vc[vc >= MINROWS].index); labs = labs[:5] if TINY else labs; per = CAP if TINY else min(CAP, max(30, 3000 // max(1, len(labs))))
        for l in labs:
            sub = df[df.label == l]; ix = rng.permutation(len(sub))[:per]; key = f"{ds}::{l}"; LAB[key] = dict(ds=ds, raw=l, name=l.replace("_", " "))
            P["text"] += sub.text.iloc[ix].tolist(); P["key"] += [key] * len(ix); P["ds"] += [ds] * len(ix)
        info["datasets"][ds] = dict(rows_loaded=len(df), labels=len(labs), rows_in_pool=int(sum(1 for d_ in P["ds"] if d_ == ds)), val=(ds == VAL_DS)); log("loaded", ds, info["datasets"][ds], f"{time.time()-t:.0f}s")
    evals = []; ev_info = {}
    for name, fn in (("bank_us", ld_bank_us), ("amazon", ld_amazon)):
        e, ev_info[name] = group_eval(name, fn(), rng); evals.append(e); log("eval", name, ev_info[name])
    c = common.load_clinc150(); DOM = json.load(open(common.CACHE / "clinc_domains.json")); cn = c["names"]
    labs = [n for n in cn if n != "oos" and DOM[n] == "travel"]; labs = labs[:4] if TINY else labs; li = {cn.index(n): i for i, n in enumerate(labs)}
    sel = lambda s: [i for i, y in enumerate(c[s][1]) if y in li]; sp = {s: sel(s) for s in ("train", "validation", "test")}
    ev = dict(name="clinc_travel", labels=labs)
    for s, a in (("train", "s"), ("validation", "c"), ("test", "q")): ev["t" + a] = [c[s][0][i] for i in sp[s]]; ev["y" + a] = np.array([li[c[s][1][i]] for i in sp[s]])
    evals.append(ev); ev_info["clinc_travel"] = dict(n_labels=len(labs), n_support_pool=len(ev["ts"]), n_cal=len(ev["tc"]), n_test=len(ev["tq"]), note="official CLINC splits, no group field")
    # ---- label descriptions (LLM; CLINC / Banking77 from the S2b cache) and synthetic examples for the eval labels
    from concurrent.futures import ThreadPoolExecutor
    desc = {}; jobs = []
    for ds in list(POOL_LOADERS) + [e["name"] for e in evals]:
        raws = [v["raw"] for v in LAB.values() if v["ds"] == ds] if ds in POOL_LOADERS else [l for e in evals if e["name"] == ds for l in e["labels"]]
        if ds == "banking77": desc.update({(ds, r): DESC["banking77"].get(r, r) for r in raws}); continue
        if ds in ("clinc_fin", "clinc_travel"): desc.update({(ds, r): DESC["clinc150"].get(r, r) for r in raws}); continue
        jobs += [(ds, raws[i:i + 40]) for i in range(0, len(raws), 40)]
    def one(job):
        ds, chunk = job; d = llm_json(f"Task: classify {CTX[ds]}.\nFor each label below write ONE short sentence (max 25 words) describing what an input record with that label looks like or is about. Reply with a JSON object mapping label -> sentence, nothing else.\nLabels:\n" + "\n".join(chunk), tag="s26desc")
        return {(ds, r): str(d.get(r) or r) for r in chunk}
    with ThreadPoolExecutor(8) as ex:
        for r in ex.map(one, jobs): desc.update(r)
    sjobs = [(e["name"], e["labels"][i:i + 12]) for e in evals for i in range(0, len(e["labels"]), 12)]
    def syn(job):
        ds, chunk = job; out = {}
        for att in range(3):
            d = llm_json(f"Task: classify {CTX[ds]}.\nFor each label below, write {SN} short, varied, realistic input records that would get that label (different phrasings, lengths and formats, some noisy). Reply with a JSON object mapping label -> list of {SN} strings, nothing else.\n" + "\n".join(f"{l}: {desc[(ds, l)]}" for l in chunk), tag=f"s26syn{att}", temperature=0.7)
            if all(isinstance(d.get(l), list) and len(d[l]) >= min(5, SN) for l in chunk): break
        for l in chunk: out[(ds, l)] = [clean(x) for x in (d.get(l) or [])][:SN] or [desc[(ds, l)]]
        return out
    SYN = {}
    with ThreadPoolExecutor(8) as ex:
        for r in ex.map(syn, sjobs): SYN.update(r)
    for e in evals:
        e["desc_t"] = [desc[(e["name"], l)] for l in e["labels"]]; e["synth_t"] = [s for l in e["labels"] for s in SYN[(e["name"], l)]]; e["synth_y"] = np.array([i for i, l in enumerate(e["labels"]) for _ in SYN[(e["name"], l)]])
    # ---- pool arrays; drop pool rows whose text equals an eval text
    ev_texts = {t.lower() for e in evals for t in e["ts"] + e["tc"] + e["tq"]}; keep = np.array([i for i, t in enumerate(P["text"]) if t.lower() not in ev_texts]); info["pool_rows_dropped_equal_to_eval_text"] = len(P["text"]) - len(keep)
    keys, dss, texts = np.array(P["key"])[keep], np.array(P["ds"])[keep], [P["text"][i] for i in keep]; labkeys = sorted(set(keys.tolist()))
    for k_ in labkeys: LAB[k_]["desc"] = desc[(LAB[k_]["ds"], LAB[k_]["raw"])]
    pool = dict(text=texts, key=keys, ds=dss, X=embed(texts), lab_keys=labkeys, labels={k_: LAB[k_] for k_ in labkeys}, lab_desc_X=embed([LAB[k_]["desc"] for k_ in labkeys]))
    ds_labels = {d_: [k_ for k_ in labkeys if LAB[k_]["ds"] == d_] for d_ in sorted(set(dss.tolist()))}
    for e in evals:
        for a in ("s", "c", "q"): e["X" + a] = embed(e["t" + a])
        e["synth_X"], e["desc_X"] = embed(e["synth_t"]), embed(e["desc_t"])
    info.update(eval_tasks=ev_info, pool_labels={d_: len(v) for d_, v in ds_labels.items()}, pool_rows=len(keep), n_pool_train_labels=sum(len(v) for d_, v in ds_labels.items() if d_ != VAL_DS), eval_labels={e["name"]: e["labels"] for e in evals},
                llm_spend_usd=round(llm_spend(), 5), step_names=STEP)
    pickle.dump(dict(pool=pool, evals=evals, ds_labels=ds_labels), open(WORK / "store.pkl", "wb")); jsave(OUT / "prep_info.json", info); log("prep done", {k: v for k, v in info.items() if k != "eval_labels"})

# ================================================================== hypernetwork (S17 recipe: SetTr set-transformer, pool tasks, early stop on the held-out pool dataset)
def make_settr():
    import torch, torch.nn as nn, torch.nn.functional as F
    class SetTr(nn.Module):
        def __init__(s, h=256, drop=0.2, layers=2):
            super().__init__(); s.inp = nn.Linear(2 * D + 2, h); s.drop = nn.Dropout(drop)
            s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(h, 4, 2 * h, dropout=drop, batch_first=True), layers)
            s.w, s.b = nn.Linear(h, D), nn.Linear(h, 1); s.scale = nn.Parameter(torch.tensor(10.0))
        def forward(s, desc, proto, k):
            K = desc.shape[0]; z = s.enc(s.drop(s.inp(torch.cat([desc, proto, torch.full((K, 1), float(k > 0)), torch.full((K, 1), float(np.log1p(k)))], -1)))[None])[0]
            return F.normalize(s.w(z) + desc + proto, dim=-1), s.b(z)[:, 0]
        def logits(s, X, W, b): return s.scale * X @ W.T + b
    return SetTr
class Pool:
    def __init__(s, st):
        p = st["pool"]; s.X = p["X"].astype(np.float32); s.ds_labels = st["ds_labels"]; keys = p["key"]
        s.rows = {k: np.flatnonzero(keys == k) for k in p["lab_keys"]}; s.dX = dict(zip(p["lab_keys"], p["lab_desc_X"].astype(np.float32)))
def sample_tasks(ds_labels, dsets, n, seed):
    rng = np.random.default_rng(seed); w = np.sqrt([len(ds_labels[d]) for d in dsets]); w = w / w.sum(); seen, tasks, tries = set(), [], 0
    while len(tasks) < n and tries < n * 40:
        tries += 1; d = dsets[rng.choice(len(dsets), p=w)]; labs = ds_labels[d]; nl = len(labs); lo = min(5, nl) if nl > 6 else min(3, nl); K = int(rng.integers(lo, min(15, nl) + 1))
        sel = tuple(sorted(rng.choice(nl, K, replace=False).tolist()))
        if (d, sel) in seen: continue
        seen.add((d, sel)); tasks.append([labs[i] for i in sel])
    return tasks
def episode(P, task, k, rng, nq=10):
    K = len(task); desc = np.stack([P.dX[l] for l in task]); proto = np.zeros((K, D), np.float32); si, sy, qi, qy = [], [], [], []
    for c, l in enumerate(task):
        r = rng.permutation(P.rows[l]); kk = min(k, max(len(r) - nq, 0)); sup, qry = r[:kk], r[kk:kk + nq]
        if kk: proto[c] = P.X[sup].mean(0)
        si += list(sup); sy += [c] * kk; qi += list(qry); qy += [c] * len(qry)
    return desc, proto, np.array(si, int), np.array(sy, int), np.array(qi, int), np.array(qy, int)
def hyper_stage():
    import torch, torch.nn.functional as F
    torch.set_num_threads(2); torch.manual_seed(0); st = load_store(); P = Pool(st); fit_ds = [d for d in P.ds_labels if d != VAL_DS]
    steps, E, ve, ntask = (6, 2, 3, 20) if TINY else (2000, 8, 100, 800); tasks = sample_tasks(P.ds_labels, fit_ds, ntask, 1); KSH = [1, 2, 3, 5, 10]
    vt = sample_tasks(P.ds_labels, [VAL_DS], 3 if TINY else 30, 99); vr = np.random.default_rng(99); val = [(t, k, episode(P, t, k, vr)) for t in vt for k in (1, 3, 10)]
    G = make_settr()(); opt = torch.optim.AdamW(G.parameters(), 1e-3, weight_decay=1e-2); Xp = torch.tensor(P.X)
    @torch.no_grad()
    def val_acc():
        G.eval(); out = {}
        for t, k, (dsc, pr, si, sy, qi, qy) in val:
            W, b = G(torch.tensor(dsc), torch.tensor(pr), k); out.setdefault(k, []).append(float((G.logits(Xp[qi], W, b).argmax(1).numpy() == qy).mean()))
        return {k: float(np.mean(v)) for k, v in out.items()}
    rng = np.random.default_rng(7); best, bstate, bad, hist, t0 = -1, None, 0, [], time.time()
    for step in range(steps):
        G.train(); loss = 0
        for _ in range(E):
            task = tasks[int(rng.integers(len(tasks)))]; k = KSH[int(rng.integers(len(KSH)))]; dsc, pr, si, sy, qi, qy = episode(P, task, k, rng); W, b = G(torch.tensor(dsc), torch.tensor(pr), k)
            loss = loss + F.cross_entropy(G.logits(Xp[qi], W, b), torch.tensor(qy)) / E
        opt.zero_grad(); loss.backward(); opt.step()
        if time.time() > float(os.environ.get("S26_DEADLINE", "1e18")) - 240: log("[hyper] deadline near, stopping"); break
        if step % ve == ve - 1:
            v = val_acc(); m = float(np.mean(list(v.values()))); hist.append((step + 1, m, v)); log(f"[hyper] step {step+1} loss {float(loss):.3f} val {({k: round(a, 3) for k, a in v.items()})}")
            if m > best:
                best, bad, bstate = m, 0, {n_: p.clone() for n_, p in G.state_dict().items()}; torch.save(bstate, WORK / "SETTR.pt")   # saved as soon as it improves, so a kill at the deadline still leaves a usable generator
            else:
                bad += 1
                if bad >= 4: break
    bstate = bstate or {n_: p.clone() for n_, p in G.state_dict().items()}; G.load_state_dict(bstate); torch.save(bstate, WORK / "SETTR.pt")
    dcos = {k: float(np.mean([float(((P.X[ep[4]] @ ep[0].T).argmax(1) == ep[5]).mean()) for t, kk, ep in val if kk == k])) for k in (1, 3, 10)}
    jsave(OUT / "results_hyper.json", dict(n_tasks=len(tasks), n_labels=len({l for t in tasks for l in t}), fit_datasets=fit_ds, val_dataset=VAL_DS, best_val=best, hist=hist, val_desc_cos=dcos, seconds=time.time() - t0, params=int(sum(p.numel() for p in G.parameters()))))

# ================================================================== arm 4: encoder pretrained on the pool (classification over the union of pool labels, head discarded)
def enc_stage():
    import torch, torch.nn as nn, torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    d = dev(); t0 = time.time(); st = load_store(); pool, EV = st["pool"], st["evals"]; pk, pt = pool["key"], pool["text"]; ds_of = {k: v["ds"] for k, v in pool["labels"].items()}
    tok = AutoTokenizer.from_pretrained(common.EMB_MODEL); labs = sorted(k for k in ds_of if ds_of[k] != VAL_DS); lid = {k: i for i, k in enumerate(labs)}
    ridx = np.array([i for i, k in enumerate(pk) if k in lid]); yl = np.array([lid[pk[i]] for i in ridx]); vk = sorted(k for k in ds_of if ds_of[k] == VAL_DS)
    vrows = {k: np.flatnonzero(pk == k) for k in vk}; vi = np.unique(np.concatenate(list(vrows.values()))); pos = {int(i): j for j, i in enumerate(vi)}; r0 = np.random.default_rng(5)
    vtasks = [list(r0.choice(vk, min(10, len(vk)), replace=False)) for _ in range(3 if TINY else 8)]
    def val3(model):   # 3-shot LR accuracy on held-out-dataset tasks
        X = embed_all(model, tok, [pt[i] for i in vi]); accs = []
        for t in vtasks:
            for rep in range(1 if TINY else 3):
                r = np.random.default_rng(rep); tr, ytr, te, yte = [], [], [], []
                for c, k in enumerate(t):
                    rr = r.permutation(vrows[k]); tr += list(rr[:3]); ytr += [c] * len(rr[:3]); te += list(rr[3:33]); yte += [c] * len(rr[3:33])
                clf = lr_fit(X[[pos[i] for i in tr]], ytr); accs.append(float((clf.predict(X[[pos[i] for i in te]]) == np.array(yte)).mean()))
        return float(np.mean(accs))
    model = AutoModel.from_pretrained(common.EMB_MODEL).to(d); head = nn.Linear(D, len(labs)).to(d); torch.manual_seed(0); rng = np.random.default_rng(0)
    opt = torch.optim.AdamW([{"params": model.parameters(), "lr": 3e-5}, {"params": head.parameters(), "lr": 1e-3}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / PRE_STEPS)); v0 = val3(model); hist = [(0, v0)]; best = (-1, 0, None); losses = []
    log(f"[enc] labels {len(labs)} rows {len(ridx)}; step 0 (base encoder) val 3-shot {v0:.3f}")
    for step in range(1, PRE_STEPS + 1):
        model.train(); bi = rng.choice(len(ridx), min(PRE_BS, len(ridx)), replace=False); bt = tok([pt[ridx[j]] for j in bi], padding=True, truncation=True, max_length=48, return_tensors="pt").to(d)
        with torch.autocast(d, dtype=torch.bfloat16, enabled=(d == "cuda")): h = model(**bt).last_hidden_state[:, 0].float()
        loss = F.cross_entropy(head(F.dropout(h, 0.1, training=True)), torch.tensor(yl[bi], device=d)); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); losses.append(float(loss))
        if step in CKPTS:
            v = val3(model); hist.append((step, v)); log(f"[enc] step {step} loss {np.mean(losses[-50:]):.3f} val 3-shot {v:.3f}")
            if v >= best[0]: best = (v, step, {k_: p.detach().cpu().clone() for k_, p in model.state_dict().items()})   # only trained checkpoints compete, never the untouched base
    if best[2] is None: best = (v0, PRE_STEPS, {k_: p.detach().cpu().clone() for k_, p in model.state_dict().items()})
    model.load_state_dict(best[2]); out = {}
    for i, e in enumerate(EV):
        for nm, texts in (("Xs", e["ts"]), ("Xc", e["tc"]), ("Xq", e["tq"]), ("synth", e["synth_t"])): out[f"{nm}{i}"] = embed_all(model, tok, texts)
    np.savez_compressed(WORK / "enc.npz", **out)
    jsave(OUT / "results_enc.json", dict(n_labels=len(labs), n_rows=int(len(ridx)), steps=PRE_STEPS, batch=PRE_BS, hist=hist, chosen_step=best[1], val_base=v0, seconds=time.time() - t0, params=int(sum(p.numel() for p in model.parameters())), final_loss=float(np.mean(losses[-50:]))))

# ================================================================== arm 5: decoder (Qwen2.5-0.5B + LoRA + seq-cls head, S8 recipe), shared pool pretraining vs from base
class Dec:
    def __init__(s):
        import torch
        from transformers import AutoTokenizer
        s.torch = torch; s.d = dev(); s.tok = AutoTokenizer.from_pretrained(QWEN)
        if s.tok.pad_token is None: s.tok.pad_token = s.tok.eos_token
        s.tok.padding_side = "right"
    fmt = staticmethod(lambda step, t: f"Step: {step}\nRecord: {t}\nLabel:")
    def make(s, K, lora_state=None):
        from transformers import AutoModelForSequenceClassification
        from peft import LoraConfig, get_peft_model
        if TINY:
            from transformers import Qwen2Config
            m = AutoModelForSequenceClassification.from_config(Qwen2Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, num_labels=K, pad_token_id=s.tok.pad_token_id))
        else:
            try: m = AutoModelForSequenceClassification.from_pretrained(QWEN, num_labels=K, dtype=s.torch.float32)
            except TypeError: m = AutoModelForSequenceClassification.from_pretrained(QWEN, num_labels=K, torch_dtype=s.torch.float32)
        m.config.pad_token_id = s.tok.pad_token_id
        m = get_peft_model(m, LoraConfig(task_type="SEQ_CLS", r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], modules_to_save=["score"]))
        if lora_state is not None: miss = m.load_state_dict(lora_state, strict=False); assert not miss.unexpected_keys, miss.unexpected_keys
        return m.to(s.d)
    def enc(s, texts): return s.tok(texts, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(s.d)
    def fit(s, m, texts, y, steps, bs=FT_BS, lr=2e-4, opt=None, rng=None):
        torch = s.torch; m.train(); opt = opt or torch.optim.AdamW([q for q in m.parameters() if q.requires_grad], lr=lr); y = torch.tensor(np.asarray(y)); n = len(texts); step = 0; rng = rng or np.random.default_rng(0)
        while step < steps:
            perm = rng.permutation(n)
            for i in range(0, n, bs):
                if step >= steps: break
                idx = perm[i:i + bs]; batch = s.enc([texts[j] for j in idx])
                with torch.autocast(s.d, dtype=torch.bfloat16, enabled=(s.d == "cuda")): loss = torch.nn.functional.cross_entropy(m(**batch).logits.float(), y[idx].to(s.d))
                opt.zero_grad(); loss.backward(); opt.step(); step += 1
        return opt, float(loss)
    def predict(s, m, texts, bs=128):
        torch = s.torch; m.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                with torch.autocast(s.d, dtype=torch.bfloat16, enabled=(s.d == "cuda")): out.append(torch.softmax(m(**s.enc(texts[i:i + bs])).logits.float(), -1).cpu().numpy())
        return np.concatenate(out).astype(np.float64)
def lora_of(m): return {k: v.detach().cpu().clone() for k, v in m.state_dict().items() if "lora_" in k}
def decpre_stage():
    torch = __import__("torch"); t0 = time.time(); dec = Dec(); st = load_store(); pool = st["pool"]; pk, pt, pds = pool["key"], pool["text"], pool["ds"]
    labs = sorted(k for k in pool["labels"] if pool["labels"][k]["ds"] != VAL_DS); lid = {k: i for i, k in enumerate(labs)}; ridx = np.array([i for i, k in enumerate(pk) if k in lid])
    texts = [dec.fmt(STEP[pds[i]], pt[i]) for i in ridx]; y = np.array([lid[pk[i]] for i in ridx]); log(f"[decpre] labels {len(labs)} rows {len(ridx)}")
    vk = sorted(k for k in pool["labels"] if pool["labels"][k]["ds"] == VAL_DS); vr = np.random.default_rng(5); vt = []   # 2 held-out-dataset tasks, 3-shot, to choose the checkpoint
    for _ in range(2):
        t = list(vr.choice(vk, min(10, len(vk)), replace=False)); sup, ys, qry, yq = [], [], [], []
        for c, k in enumerate(t):
            rr = vr.permutation(np.flatnonzero(pk == k)); sup += [dec.fmt(STEP[VAL_DS], pt[i]) for i in rr[:3]]; ys += [c] * len(rr[:3]); qry += [dec.fmt(STEP[VAL_DS], pt[i]) for i in rr[3:23]]; yq += [c] * len(rr[3:23])
        vt.append((len(t), sup, ys, qry, np.array(yq)))
    def val_dec(state):
        a = []
        for K, sup, ys, qry, yq in vt:
            torch.manual_seed(0); m = dec.make(K, state); dec.fit(m, sup, ys, FT_STEPS); a.append(float((dec.predict(m, qry).argmax(1) == yq).mean())); del m
        return float(np.mean(a))
    v0 = val_dec(None); log(f"[decpre] LoRA-from-base 3-shot val {v0:.3f}")
    torch.manual_seed(0); m = dec.make(len(labs)); rng = np.random.default_rng(0); opt = None; hist = [(0, v0)]; best = (-1, 0, None); done = 0; losses = []
    for ck in CKPTS:
        opt, loss = dec.fit(m, texts, y, ck - done, bs=PRE_BS, opt=opt, rng=rng); done = ck; losses.append(loss)
        state = lora_of(m); v = val_dec(state); hist.append((ck, v)); log(f"[decpre] step {ck} loss {loss:.3f} val 3-shot (60-step fine-tune) {v:.3f} [{time.time()-t0:.0f}s]")
        if v >= best[0]: best = (v, ck, state)
    if best[2] is None: best = (v0, CKPTS[-1], lora_of(m))
    torch.save(best[2], WORK / "shared_lora.pt")
    jsave(OUT / "results_decpre.json", dict(n_labels=len(labs), n_rows=int(len(ridx)), steps=PRE_STEPS, batch=PRE_BS, hist=hist, chosen_step=best[1], val_base=v0, seconds=time.time() - t0, last_losses=losses))

def dec_worker(w, n):
    torch = __import__("torch"); st = load_store(); EV = st["evals"]; dec = Dec(); path = OUT / f"results_dec_{w}.json"; R = jload(path, {"records": []}); deadline = float(os.environ.get("S26_DEADLINE", "1e18"))
    done = {(r["arm"], r["task"], r["k"], r["seed"]) for r in R["records"]}; shared = torch.load(WORK / "shared_lora.pt"); units = [(s, i) for s in range(NSEED) for i in range(len(EV))][w::n]
    VAR = [("5 shared LoRA (pool-pretrained) + k rows", shared, False), ("5c LoRA from base + k rows", None, False), ("5b shared LoRA + synthetic prior", shared, True), ("5d LoRA from base + synthetic prior", None, True)]
    for seed, i in units:
        e = EV[i]; K = len(e["labels"]); stp = STEP[e["name"]]; F_ = lambda ts: [dec.fmt(stp, t) for t in ts]; ts, tc, tq, tsyn = F_(e["ts"]), F_(e["tc"]), F_(e["tq"]), F_(e["synth_t"])
        for k in KS3:
            idx = draw_support(e["ys"], k, seed, K)
            for arm, state, syn in VAR:
                if (arm, e["name"], k, seed) in done: continue
                if time.time() > deadline: log("deadline reached, worker stops"); jsave(path, R); return
                t = time.time(); texts = [ts[j] for j in idx] + (tsyn if syn else []); y = np.concatenate([e["ys"][idx]] + ([e["synth_y"]] if syn else []))
                try:
                    torch.manual_seed(1000 * seed + k); m = dec.make(K, state); dec.fit(m, texts, y, FT_STEPS, rng=np.random.default_rng(seed)); pc, pq = dec.predict(m, tc), dec.predict(m, tq); del m
                except Exception as ex: log(f"RUN FAILED {arm} {e['name']} k={k} seed={seed}: {repr(ex)[:300]}"); continue
                R["records"].append(dict(arm=arm, task=e["name"], k=k, seed=seed, K=K, **score(pc, e["yc"], pq, e["yq"]), sec=time.time() - t)); done.add((arm, e["name"], k, seed)); jsave(path, R)
        log(f"[dec {w}] unit seed {seed} task {e['name']} done, {len(R['records'])} records")
    jsave(path, R)

# ================================================================== arms 1-4b on frozen / pretrained embeddings and the hypernetwork
def eval_stage(group):
    import torch; torch.set_num_threads(1)
    st = load_store(); EV = st["evals"]; path = OUT / f"results_{group}.json"; R = jload(path, {"records": []}); done = {(r["arm"], r["task"], r["k"], r["seed"]) for r in R["records"]}
    G = Z = None
    if group == "arm3":
        try: G = make_settr()(); G.load_state_dict(torch.load(WORK / "SETTR.pt")); G.eval()
        except Exception as ex: log("no hypernetwork:", repr(ex)[:100])
    if group == "arm4":
        try: Z = np.load(WORK / "enc.npz")
        except Exception as ex: log("no pretrained encoder:", repr(ex)[:100])
    def run(arm, i, k, seed, fn):
        e = EV[i]
        if (arm, e["name"], k, seed) in done: return
        t = time.time()
        try: pc, pq = fn()
        except Exception as ex: log(f"ARM FAILED {arm} {e['name']} k={k} seed={seed}: {repr(ex)[:300]}"); return
        R["records"].append(dict(arm=arm, task=e["name"], k=k, seed=seed, K=len(e["labels"]), **score(pc, e["yc"], pq, e["yq"]), sec=time.time() - t)); done.add((arm, e["name"], k, seed)); jsave(path, R)
    def lr_arm(Xs, Xc, Xq, idx, K, ys, pre=None):
        X, y = (Xs[idx], ys[idx]) if pre is None else (np.concatenate([pre[0], Xs[idx]]), np.concatenate([pre[1], ys[idx]])); clf = lr_fit(X, y); return full_proba(clf, Xc, K), full_proba(clf, Xq, K)
    for seed in range(NSEED):
        for i, e in enumerate(EV):
            K = len(e["labels"]); ys, Xs, Xc, Xq, SX, sy = e["ys"], e["Xs"], e["Xc"], e["Xq"], e["synth_X"], e["synth_y"]
            for k in KS3:
                idx = draw_support(ys, k, seed, K)
                if group == "cheap":
                    run("1 baseline (frozen bge-small + LR)", i, k, seed, lambda: lr_arm(Xs, Xc, Xq, idx, K, ys)); run("2 synthetic prior (LLM examples + k rows)", i, k, seed, lambda: lr_arm(Xs, Xc, Xq, idx, K, ys, (SX, sy)))
                    continue
                if G is not None:
                    def hy():
                        Pm = np.zeros((K, D), np.float32)
                        for c in range(K): Pm[c] = Xs[idx][ys[idx] == c].mean(0)
                        with torch.no_grad(): W, b = G(torch.tensor(e["desc_X"]), torch.tensor(Pm), k); return [torch.softmax(G.logits(torch.tensor(X), W, b), -1).numpy().astype(np.float64) for X in (Xc, Xq)]
                    run("3 hypernetwork (pool-trained, one forward pass)", i, k, seed, hy)
                if Z is not None:
                    V = {n: Z[f"{n}{i}"] for n in ("Xs", "Xc", "Xq", "synth")}
                    run("4 encoder pretrained on pool (33M) + LR", i, k, seed, lambda: lr_arm(V["Xs"], V["Xc"], V["Xq"], idx, K, ys))
                    run("4b encoder pretrained + synthetic prior", i, k, seed, lambda: lr_arm(V["Xs"], V["Xc"], V["Xq"], idx, K, ys, (V["synth"], sy)))

# ================================================================== aggregation
def rows_to(cov_by_k, target=0.5):
    ks = sorted(cov_by_k)
    if any(np.isnan(cov_by_k[k]) for k in ks): return "n/a"
    if cov_by_k[ks[0]] >= target: return "<=1"
    for a, b in zip(ks, ks[1:]):
        if cov_by_k[b] >= target: t = (target - cov_by_k[a]) / (cov_by_k[b] - cov_by_k[a]); return f"{np.exp(np.log(a) + t * (np.log(b) - np.log(a))):.1f}"
    return ">10"
def agg(quiet=False):
    rec = []
    for f in sorted(OUT.glob("results_*.json")):
        r = jload(f)
        if isinstance(r, dict) and "records" in r: rec += r["records"]
    if not rec: print("no records"); return
    (OUT / "results.json").write_text(json.dumps(rec))
    order = ["1 baseline", "2 synthetic", "3 hyper", "4 encoder pretrained on pool", "4b", "5 shared", "5c", "5b", "5d"]; arms = sorted({r["arm"] for r in rec}, key=lambda a: next((j for j, p in enumerate(order) if a.startswith(p)), 99)); tasks = sorted({r["task"] for r in rec})
    cell = lambda a, k, key, t=None: (lambda v: float(np.mean(v)) if v else float("nan"))([r[key] for r in rec if r["arm"] == a and r["k"] == k and (t is None or r["task"] == t) and r[key] is not None])
    sd = lambda a, k, key: (lambda per: float(np.std(per, ddof=1)) if len(per) > 1 else float("nan"))([np.mean(v) for s in range(NSEED) if (v := [r[key] for r in rec if r["arm"] == a and r["k"] == k and r["seed"] == s and r[key] is not None])])
    cnt = lambda a: len([r for r in rec if r["arm"] == a])
    out = [f"records: {len(rec)}; cells per arm (full = {len(tasks) * len(KS3) * NSEED}): " + json.dumps({a: cnt(a) for a in arms})]
    for title, key in (("Certified coverage at a 5% budget (share of test rows answered), by task", "cov"), ("Accuracy on the test rows, by task", "acc"), ("Realized error on the answered rows (budget 5%), by task", "err")):
        out += [f"\n### {title}; mean over draws\n", "| arm | " + " | ".join(f"{t} k={k}" for t in tasks for k in KS3) + " |", "|---|" + "---|" * (len(tasks) * len(KS3))]
        for a in arms: out.append(f"| {a} | " + " | ".join(f"{cell(a, k, key, t):.3f}" for t in tasks for k in KS3) + " |")
    out += ["\n### Mean over tasks (+- s.d. over draws of the task-averaged value)\n", "| arm | cov k=1 | cov k=3 | cov k=10 | acc k=1 | acc k=3 | acc k=10 | rows per label to 50% coverage (all tasks) | " + " | ".join(f"50% rows {t}" for t in tasks) + " |", "|---|" + "---|" * (7 + len(tasks) + 1)]
    for a in arms:
        out.append(f"| {a} | " + " | ".join(f"{cell(a, k, key):.3f} +- {sd(a, k, key):.3f}" for key in ("cov", "acc") for k in KS3) + f" | {rows_to({k: cell(a, k, 'cov') for k in KS3})} | " + " | ".join(rows_to({k: cell(a, k, 'cov', t) for k in KS3}) for t in tasks) + " |")
    comb = {a: np.nanmean([cell(a, k, "cov") for k in KS3]) for a in arms if a.startswith(("4b", "5b"))}
    if comb: out.append(f"\nArm 6 (best combination = higher mean coverage over k, chosen on these same tasks so slightly optimistic): {max(comb, key=comb.get)} ({json.dumps({a: round(v, 3) for a, v in comb.items()})})")
    # paired transfer gains: same task / k / draw, arm minus its no-transfer control
    PAIRS = [("4 encoder pretrained on pool (33M) + LR", "1 baseline (frozen bge-small + LR)", "encoder transfer (4 - 1)"), ("4b encoder pretrained + synthetic prior", "2 synthetic prior (LLM examples + k rows)", "encoder transfer with synthetic prior (4b - 2)"),
             ("5 shared LoRA (pool-pretrained) + k rows", "5c LoRA from base + k rows", "decoder transfer (5 - 5c)"), ("5b shared LoRA + synthetic prior", "5d LoRA from base + synthetic prior", "decoder transfer with synthetic prior (5b - 5d)"),
             ("3 hypernetwork (pool-trained, one forward pass)", "1 baseline (frozen bge-small + LR)", "hypernetwork vs baseline (3 - 1)"), ("3 hypernetwork (pool-trained, one forward pass)", "2 synthetic prior (LLM examples + k rows)", "hypernetwork vs synthetic prior (3 - 2)"),
             ("4 encoder pretrained on pool (33M) + LR", "5 shared LoRA (pool-pretrained) + k rows", "encoder-transfer vs decoder-transfer, absolute (4 - 5)")]
    byk = {(r["arm"], r["task"], r["k"], r["seed"]): r for r in rec}
    out += ["\n### Paired transfer gains (mean over tasks and draws of arm A minus arm B on the same task / k / draw; +- s.d. over the paired cells)\n", "| comparison | cov k=1 | cov k=3 | cov k=10 | acc k=1 | acc k=3 | acc k=10 | cells per k |", "|---|---|---|---|---|---|---|---|"]
    for A, B, nm in PAIRS:
        row, nn = [], 0
        for key in ("cov", "acc"):
            for k in KS3:
                d = [byk[(A, t, kk, sd)][key] - byk[(B, t, kk, sd)][key] for (a_, t, kk, sd) in byk if a_ == A and kk == k and (B, t, kk, sd) in byk and byk[(A, t, kk, sd)][key] is not None and byk[(B, t, kk, sd)][key] is not None]
                row.append(f"{np.mean(d):+.3f} +- {np.std(d, ddof=1) if len(d) > 1 else float('nan'):.3f}" if d else "n/a"); nn = max(nn, len(d))
        out.append(f"| {nm} | " + " | ".join(row) + f" | {nn} |")
    for nm in ("prep_info", "results_enc", "results_decpre", "results_hyper"):
        j = jload(OUT / f"{nm}.json")
        if j: out.append(f"\n{nm}: " + json.dumps({k: v for k, v in j.items() if k not in ("eval_labels",)})[:3000])
    (OUT / "summary.md").write_text("\n".join(out), encoding="utf-8")
    if not quiet: print("\n".join(out))

# ================================================================== pod orchestration
def pod():
    import torch
    assert torch.cuda.is_available(), "NO CUDA ON THIS POD"
    LOGS = HERE / "logs"; LOGS.mkdir(exist_ok=True); budget = float(os.environ.get("S26_BUDGET_MIN", "40")) * 60; end = T0 + budget; ncpu = os.cpu_count() or 4; NW = 3 if ncpu >= 5 else 2
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1", S26_DEADLINE=str(end))
    def start(args, name, nice=0, **e): return subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), *args], env={**env, **e}, stdout=open(LOGS / f"{name}.log", "w", encoding="utf-8"), stderr=subprocess.STDOUT, cwd=str(HERE), preexec_fn=(lambda: os.nice(nice)) if nice else None)
    def tail(name, n=250): return (LOGS / f"{name}.log").read_text(encoding="utf-8", errors="replace")[-n:].replace("\n", " | ")
    log("cpus", ncpu, "gpu", torch.cuda.get_device_name(0), "budget min", budget / 60, "decoder workers", NW)
    p = start(["prep"], "prep")
    try: p.wait(timeout=max(1, end - time.time()))
    except subprocess.TimeoutExpired: p.kill()
    log("prep exit", p.returncode, tail("prep"))
    if p.returncode: sys.exit(4)
    # most informative first: encoder and decoder pretraining (GPU) start together; the hypernetwork (CPU, low priority) only uses spare cores
    P = {"enc": start(["enc"], "enc"), "decpre": start(["decpre"], "decpre"), "hyper": start(["hyper"], "hyper", nice=10, OMP_NUM_THREADS="2"), "eval_cheap": start(["eval", "cheap"], "eval_cheap")}
    later = {"decpre": [(f"dec{w}", ["dec", str(w), str(NW)]) for w in range(NW)], "enc": [("eval_arm4", ["eval", "arm4"])], "hyper": [("eval_arm3", ["eval", "arm3"])]}; fired, done_logged, last_agg = set(), set(), time.time()
    while time.time() < end + 60:
        for nm, pr in list(P.items()):
            if pr.poll() is None: continue
            if nm not in done_logged: done_logged.add(nm); log(nm, "exit", pr.returncode, tail(nm))
            if nm in later and nm not in fired:
                fired.add(nm)
                if pr.returncode == 0:
                    for n2, a2 in later[nm]: P[n2] = start(a2, n2)
                else: log(nm, "FAILED, dependent stages skipped")
        if all(pr.poll() is not None for pr in P.values()) and all(k in fired for k in later): break
        if time.time() - last_agg > 120:
            last_agg = time.time()
            try: agg(quiet=True)
            except Exception as ex: log("agg failed", repr(ex)[:150])
        time.sleep(3)
    for nm, pr in P.items():
        if pr.poll() is None: pr.kill(); log(nm, "killed at the deadline")
    agg(); log("pod done")

def tiny():
    prep(); enc_stage(); hyper_stage(); decpre_stage(); eval_stage("cheap"); eval_stage("arm4"); eval_stage("arm3"); dec_worker(0, 1); agg()

if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--tiny"]
    {"prep": prep, "enc": enc_stage, "hyper": hyper_stage, "decpre": decpre_stage, "eval": lambda: eval_stage(a[1]), "dec": lambda: dec_worker(int(a[1]), int(a[2])), "agg": agg, "pod": pod, "tiny": tiny}[a[0]]()
