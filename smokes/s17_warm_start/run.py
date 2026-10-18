"""S17 warm start. Everything real runs on a RunPod pod (python smokes/pod.py ... --cmd "python smokes/s17_warm_start/run.py pod").
Stages (python -I run.py <stage>):
  prep                 load pool + eval datasets, LLM label descriptions / txn synthetic examples, embed with bge-small  -> WORK/store.pkl
  synth2               second independent LLM draw of the practice examples (noise floor for the synthetic-prior arm)
  meta MODEL SCALE [seed]   train a hypernetwork (LINGEN | METRIC | SETTR) on SCALE tasks (clinc7 | N | Nnb), early stopping on held-out TREC tasks
  fomaml TAG lrM lrR T [steps]   first-order MAML metric + class residuals
  eval [nproc]         all arms x eval tasks x k x draws -> results.csv   (evalmod.py)
  pod                  the whole pipeline on the pod: prep, synth2, then encoder fine-tune + TabPFN (GPU) alongside hypernet/FOMAML training (CPU), then eval + report
S17_TINY=1 shrinks every stage to a few rows for a local smoke check (no LLM calls, nothing written to the real stores)."""
import sys, os, json, re, hashlib, time, pickle
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from lib import *

STORE = WORK
CAP = 4 if TINY else 100       # rows per pool label
MINROWS = 3 if TINY else 20    # pool labels with fewer rows are dropped
NCAL = 9 if TINY else 300      # calibration rows per eval task
SN = 2 if TINY else 10         # synthetic examples kept per label

CONTEXT = {
    "massive": "voice-assistant utterances; the label is the intent name (scenario_action, e.g. alarm_set)",
    "trec": "English questions classified by the kind of answer they expect (TREC question types; label shown as code: name)",
    "emotion": "short English tweets labelled with the emotion they express",
    "ag_news": "news articles labelled by topic",
    "20ng": "Usenet newsgroup posts labelled by the newsgroup they were posted to",
    "dbpedia": "Wikipedia article abstracts labelled by their DBpedia ontology class",
    "sst5": "movie review sentences labelled by sentiment",
    "hate": "tweets labelled hate speech / offensive language / neither",
    "txn": "raw US bank-statement transaction descriptors (merchant strings with codes) labelled with a spending category",
}

def describe(ds, names):
    """one-sentence descriptions for labels, cached via llm_json."""
    out = {}
    for i in range(0, len(names), 40):
        chunk = names[i:i + 40]
        prompt = (f"Task: classify {CONTEXT[ds]}.\nFor each label below write ONE short sentence (max 25 words) describing what an input with that label "
                  "looks like or is about, in the style of: 'User wants help activating their newly received bank card.' "
                  "Reply with a JSON object mapping label -> sentence, nothing else.\nLabels:\n" + "\n".join(chunk))
        d = llm_json(prompt, tag="desc")
        for n in chunk:
            out[n] = str(d.get(n) or n)
    return out

def txn_synth(cats, desc):
    path = HERE / "synth_txn.json"
    syn = json.loads(path.read_text()) if path.exists() else {}
    need = [c for c in cats if c not in syn]
    if need and not TINY:
        prompt = ("For each spending category below write 10 short, varied, realistic raw bank-statement transaction descriptors as they appear on a US checking "
                  "account statement (merchant names mostly in capitals, store numbers, city/state, payment-rail words such as POS, ACH, PPD, WEB, ID numbers; mix debits "
                  "and credits where plausible; do not copy real account numbers) that belong to that category. Reply with a JSON object mapping category -> list of 10 strings, nothing else.\n"
                  + "\n".join(f"{c}: {desc[c]}" for c in need))
        for att in range(3):
            d = llm_json(prompt, tag=f"txnsyn{att}", temperature=0.8)
            if all(c in d and len(d[c]) >= 5 for c in need): break
        for c in need: syn[c] = [str(x) for x in d.get(c, [])][:10] or [desc[c]]
        path.write_text(json.dumps(syn, indent=1), encoding="utf-8")
    return {c: syn.get(c) or [desc[c]] for c in cats}

def clean(t, n=300):
    return re.sub(r"\s+", " ", str(t)).strip()[:n]

def gkey(s):
    s = re.sub(r"^\[(debit|credit)\]\s*", "", s.lower())
    toks = [t for t in re.findall(r"[a-z][a-z&'\.]+", s) if t not in {"ppd", "web", "id", "ccd", "pos", "ach", "pmt", "the", "of", "to", "payment", "debit", "credit", "orig", "co", "name"}]
    return " ".join(toks[:2]) or s[:10]

def shrink(texts, y, keep_ids, per):
    """TINY only: keep `per` rows for each label id in keep_ids."""
    y = np.asarray(y); cnt = {}; idx = []
    for i, yy in enumerate(y.tolist()):
        if yy in keep_ids and cnt.get(yy, 0) < per: idx.append(i); cnt[yy] = cnt.get(yy, 0) + 1
    return [texts[i] for i in idx], y[idx]

def prep():
    import datasets, pandas as pd
    rng = np.random.default_rng(17)
    if not TINY:
        ping = llm_json('Reply with exactly this JSON object and nothing else: {"ok": 1}', max_tokens=50, temperature=0, tag="ping")
        print("LLM ping", ping, "(label descriptions fall back to label names if the LLM is unreachable)", flush=True)
    DESC = json.load(open(common.CACHE / "label_descriptions.json")); DOM = json.load(open(common.CACHE / "clinc_domains.json"))
    SYN = json.load(open(common.CACHE / "synth_examples.json"))
    ld = lambda *a, **k: datasets.load_dataset(*a, cache_dir=str(HFDIR), **k)
    # ------------------------------------------------------------ pool containers
    P = dict(text=[], key=[], ds=[], X=[]); LAB = {}  # key -> dict(ds, name, desc)
    def add_rows(ds, keys, texts, X, name_of, desc_of):
        """cap CAP rows per label; X are embeddings aligned with texts."""
        keys = np.asarray(keys)
        for k in sorted(set(keys.tolist())):
            idx = np.flatnonzero(keys == k)
            if len(idx) < MINROWS: continue
            idx = rng.permutation(idx)[:CAP]
            LAB[f"{ds}::{k}"] = dict(ds=ds, name=name_of(k), desc=desc_of(k))
            P["text"] += [texts[i] for i in idx]; P["key"] += [f"{ds}::{k}"] * len(idx); P["ds"] += [ds] * len(idx); P["X"].append(X[idx])
    # ------------------------------------------------------------ eval tasks + CLINC/Banking pool
    c = load_clinc150(); cn = c["names"]
    held = HELD_CLINC[:1] if TINY else HELD_CLINC
    clinc_pool = [n for n in cn if n != "oos" and DOM[n] not in HELD_CLINC]
    if TINY:
        clinc_pool = clinc_pool[:4]; keep = {cn.index(n) for n in clinc_pool + [n for n in cn if n != "oos" and DOM[n] == held[0]][:3]}
        c = dict(names=cn, **{s: shrink(*c[s], keep, per) for s, per in (("train", 6), ("validation", 4), ("test", 4))})
    cXtr, cXva, cXte = embed(c["train"][0]), embed(c["validation"][0]), embed(c["test"][0])
    m = np.isin(c["train"][1], [cn.index(n) for n in clinc_pool])
    add_rows("clinc", [cn[y] for y in c["train"][1][m]], [t for t, f in zip(c["train"][0], m) if f], cXtr[m], lambda k: k.replace("_", " "), lambda k: DESC["clinc150"].get(k, k))
    b = load_banking77(); bn = b["names"]
    perm = np.random.default_rng(0).permutation(77); b_eval = [sorted(perm[i * 12:(i + 1) * 12].tolist()) for i in range(5)]; b_pool = sorted(perm[60:].tolist())
    if TINY:
        b_eval = [b_eval[0][:3]]; b_pool = b_pool[:3]; keep = set(b_eval[0]) | set(b_pool)
        b = dict(names=bn, train=shrink(*b["train"], keep, 12), test=shrink(*b["test"], keep, 4))
    bXtr, bXte = embed(b["train"][0]), embed(b["test"][0])
    m = np.isin(b["train"][1], b_pool)
    add_rows("banking", [bn[y] for y in b["train"][1][m]], [t for t, f in zip(b["train"][0], m) if f], bXtr[m], lambda k: k.replace("_", " "), lambda k: DESC["banking77"][k])

    evals = []
    def mk(name, family, labels, desc, nm, syn, ts, ys, tc, yc, tq, yq, Xs=None, Xc=None, Xq=None):
        syn = {l: syn[l][:SN] for l in labels}
        e = dict(name=name, family=family, labels=labels, desc_t=[desc[l] for l in labels], name_t=[nm[l] for l in labels], synth_t=[s for l in labels for s in syn[l]],
                 synth_y=np.array([i for i, l in enumerate(labels) for _ in syn[l]]), ts=ts, ys=np.asarray(ys), tc=tc, yc=np.asarray(yc), tq=tq, yq=np.asarray(yq))
        e["desc_X"], e["name_X"], e["synth_X"] = embed(e["desc_t"]), embed(e["name_t"]), embed(e["synth_t"])
        e["Xs"] = Xs if Xs is not None else embed(ts); e["Xc"] = Xc if Xc is not None else embed(tc); e["Xq"] = Xq if Xq is not None else embed(tq)
        evals.append(e)
    # held-out CLINC domains: support = train, calibration = validation (300), test = test
    for dom in held:
        labs = [n for n in cn if n != "oos" and DOM[n] == dom]
        if TINY: labs = labs[:3]
        li = {cn.index(n): i for i, n in enumerate(labs)}
        sel = lambda split: [i for i, y in enumerate(c[split][1]) if y in li]
        itr, iva, ite = sel("train"), sel("validation"), sel("test")
        mk(f"clinc/{dom}", "clinc", labs, {n: DESC["clinc150"].get(n, n) for n in labs}, {n: n.replace("_", " ") for n in labs}, SYN,
           [c["train"][0][i] for i in itr], [li[c["train"][1][i]] for i in itr], [c["validation"][0][i] for i in iva], [li[c["validation"][1][i]] for i in iva],
           [c["test"][0][i] for i in ite], [li[c["test"][1][i]] for i in ite], cXtr[itr], cXva[iva], cXte[ite])
    # Banking77: disjoint 12-label sub-tasks; calibration = NCAL random train rows of the sub-task, support = the rest, test = official test rows
    for j, lab_ids in enumerate(b_eval):
        li = {y: i for i, y in enumerate(lab_ids)}; labs = [bn[y] for y in lab_ids]
        itr = np.array([i for i, y in enumerate(b["train"][1]) if y in li]); ite = np.array([i for i, y in enumerate(b["test"][1]) if y in li])
        pm = np.random.default_rng(100 + j).permutation(len(itr)); ic, isup = itr[pm[:NCAL]], itr[pm[NCAL:]]
        g = lambda a, i: [a[0][x] for x in i]; yl = lambda a, i: [li[a[1][x]] for x in i]
        mk(f"banking/sub{j}", "banking", labs, DESC["banking77"], {n: n.replace("_", " ") for n in labs}, SYN, g(b["train"], isup), yl(b["train"], isup), g(b["train"], ic), yl(b["train"], ic),
           g(b["test"], ite), yl(b["test"], ite), bXtr[isup], bXtr[ic], bXte[ite])
    # transactions: split by merchant-ish group so no merchant string crosses support / calibration / test
    df = ld("DoDataThings/us-bank-transaction-categories-v2")["train"].to_pandas()
    cats = sorted(df.category.unique())
    if TINY: cats = cats[:3]; df = df[df.category.isin(cats)].sample(3000, random_state=0)
    df["g"] = df.description.map(gkey); df["bucket"] = df.g.map(lambda s: int(hashlib.md5(s.encode()).hexdigest()[:8], 16) % 10)
    print("txn groups", df.g.nunique(), "groups per category min", df.groupby("category").g.nunique().min(), flush=True)
    tdesc = describe("txn", cats); tsyn = txn_synth(cats, tdesc)
    lbl = {c_: i for i, c_ in enumerate(cats)}
    sup = df[df.bucket <= 4].sample(frac=1, random_state=1).groupby("category").head(8 if TINY else 200).reset_index(drop=True)
    cal = df[df.bucket.isin([5, 6])].sample(NCAL, random_state=2).reset_index(drop=True)
    tst = df[df.bucket >= 7].sample(frac=1, random_state=3).groupby("category").head(4 if TINY else 60).reset_index(drop=True)
    assert not (set(sup.g) & set(cal.g)) and not (set(sup.g) & set(tst.g)) and not (set(cal.g) & set(tst.g))
    mk("txn/us-bank-v2", "txn", cats, tdesc, {c_: c_.lower() for c_ in cats}, tsyn, sup.description.tolist(), sup.category.map(lbl).values, cal.description.tolist(), cal.category.map(lbl).values,
       tst.description.tolist(), tst.category.map(lbl).values)
    print("eval tasks", [(e["name"], len(e["labels"]), len(e["ts"]), len(e["tc"]), len(e["tq"])) for e in evals], flush=True)

    # ------------------------------------------------------------ remaining pool datasets
    def lds(name, cfg=None, split="train", **k):
        d = (ld(name, cfg) if cfg else ld(name))[split]
        if TINY: d = d.select(range(min(len(d), 6000)))
        return d.to_pandas()
    # (datasets below are capped BEFORE embedding to save time)
    def capped(df, labcol, textcol_fn):
        keep = []
        groups = sorted(df.groupby(labcol).indices.items(), key=lambda kv: str(kv[0]))
        for k, idx in (groups[:3] if TINY else groups):
            if len(idx) >= MINROWS: keep += list(rng.permutation(idx)[:CAP])
        sub = df.iloc[sorted(keep)]; return sub[labcol].values, [clean(textcol_fn(r)) for r in sub.itertuples()]
    specs = [("massive", ("mteb/amazon_massive_intent", "en"), "label_text", lambda r: r.text),
             ("emotion", ("SetFit/emotion",), "label_text", lambda r: r.text),
             ("ag_news", ("SetFit/ag_news",), "label_text", lambda r: r.text),
             ("20ng", ("SetFit/20_newsgroups",), "label_text", lambda r: r.text),
             ("dbpedia", ("fancyzhx/dbpedia_14",), "label", lambda r: r.title + ". " + r.content),
             ("sst5", ("SetFit/sst5",), "label_text", lambda r: r.text),
             ("hate", ("SetFit/hate_speech_offensive",), "label_text", lambda r: r.text),
             ("trec", ("SetFit/TREC-QC",), "label_original", lambda r: r.text)]
    dbp = ["Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace", "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"]
    for ds, dargs, lc, fn in specs:
        try: df_ = lds(*dargs)
        except Exception as ex:  # one unavailable pool dataset must not sink the run (TREC is the held-out validation dataset, so it is required)
            print("SKIP pool dataset", ds, repr(ex)[:200], flush=True)
            if ds == "trec": raise
            continue
        if ds == "20ng": df_ = df_[df_.text.str.len() > 40]
        labs, texts = capped(df_, lc, fn)
        if ds == "dbpedia": nmap = {i: dbp[i] for i in range(14)}
        elif ds == "trec": nmap = df_.groupby("label_original").label_text.first().to_dict()
        else: nmap = {k: k for k in set(labs.tolist())}
        keys = sorted(set(labs.tolist())); dn = describe(ds, [f"{k}: {nmap[k]}" if ds == "trec" else str(nmap[k]) for k in keys])
        dd = {k: dn[f"{k}: {nmap[k]}" if ds == "trec" else str(nmap[k])] for k in keys}
        X = embed(texts); labs = np.asarray(labs)
        for k in keys:  # rows are already capped, so register directly
            idx = np.flatnonzero(labs == k); key = f"{ds}::{k}"
            LAB[key] = dict(ds=ds, name=str(nmap[k]).replace("_", " "), desc=dd[k])
            P["text"] += [texts[i] for i in idx]; P["key"] += [key] * len(idx); P["ds"] += [ds] * len(idx); P["X"].append(X[idx])
    # ------------------------------------------------------------ disjointness enforcement
    pool_label_names = {v["name"].lower() for v in LAB.values()}
    ev_label_names = {l.replace("_", " ").lower() for e in evals for l in e["labels"]}
    overlap_names = sorted(pool_label_names & ev_label_names)
    ev_texts = {t.lower() for e in evals for t in e["ts"] + e["tc"] + e["tq"]}
    keep = [i for i, t in enumerate(P["text"]) if t.lower() not in ev_texts]
    print(f"pool rows {len(P['text'])}, exact-text duplicates with eval removed: {len(P['text']) - len(keep)}; label-name overlaps pool/eval: {overlap_names}", flush=True)
    Xall = np.concatenate(P["X"]); keys = np.array(P["key"]); dss = np.array(P["ds"]); texts = np.array(P["text"], dtype=object)
    keep = np.array(keep)
    pool = dict(X=Xall[keep], key=keys[keep], ds=dss[keep], text=list(texts[keep]), labels=LAB)
    labkeys = sorted(set(pool["key"].tolist()))
    pool["labels"] = {k_: pool["labels"][k_] for k_ in labkeys}
    ds_labels = {d_: [k_ for k_ in labkeys if pool["labels"][k_]["ds"] == d_] for d_ in sorted(set(pool["ds"].tolist()))}
    print("pool labels per dataset", {d_: len(v) for d_, v in ds_labels.items()}, "rows", len(pool["X"]), flush=True)
    pool["lab_keys"] = labkeys; pool["lab_desc_X"] = embed([pool["labels"][k_]["desc"] for k_ in labkeys]); pool["lab_name_X"] = embed([pool["labels"][k_]["name"] for k_ in labkeys])
    # label sets of the evaluation tasks are disjoint from every pretraining label by construction (Pool CLINC excludes the 3 held-out domains; pool Banking77 is the 17 labels outside the 5 eval sub-tasks;
    # txn / TREC / MASSIVE / ... are different datasets). Asserted here per family on label names.
    assert not {lab["name"] for lab in LAB.values() if lab["ds"] == "banking"} & {l.replace("_", " ") for e in evals if e["family"] == "banking" for l in e["labels"]}
    assert not {lab["name"] for lab in LAB.values() if lab["ds"] == "clinc"} & {l.replace("_", " ") for e in evals if e["family"] == "clinc" for l in e["labels"]}
    pickle.dump(dict(pool=pool, evals=evals, ds_labels=ds_labels), open(STORE / "store.pkl", "wb"))
    nfall = sum(v["desc"] == v["name"] for v in LAB.values()); print(f"pool labels whose description fell back to the label name: {nfall}/{len(LAB)}", flush=True)
    print(f"LLM spend so far ${llm_spend():.4f}", flush=True)


# ================================================================== shared pool helpers
MODELS = STORE / "models"; MODELS.mkdir(exist_ok=True)
VAL_DS = "trec"

def load_store():
    return pickle.load(open(STORE / "store.pkl", "rb"))

class PoolIdx:
    def __init__(s, st):
        p = st["pool"]; s.X = p["X"].astype(np.float32); s.ds_labels = st["ds_labels"]; keys = p["key"]
        s.rows = {k: np.flatnonzero(keys == k) for k in p["lab_keys"]}
        s.dX = dict(zip(p["lab_keys"], p["lab_desc_X"].astype(np.float32))); s.nX = dict(zip(p["lab_keys"], p["lab_name_X"].astype(np.float32)))
        s.fit_ds = [d for d in s.ds_labels if d != VAL_DS]

def sample_tasks(ds_labels, dsets, n, seed, labels_filter=None):
    """n distinct (dataset, label-subset) tasks of 5-15 labels (3-6 for the tiny datasets); sampling weight ~ sqrt(#labels)."""
    rng = np.random.default_rng(seed); w = np.sqrt([len(ds_labels[d]) for d in dsets]); w = w / w.sum(); seen, tasks, tries = set(), [], 0
    while len(tasks) < n and tries < n * 40:
        tries += 1; d = dsets[rng.choice(len(dsets), p=w)]; labs = ds_labels[d]
        if labels_filter is not None:
            f = labels_filter(d, labs)
            if f is None: continue
            labs = f
        nl = len(labs); lo = min(5, nl) if nl > 6 else min(3, nl); hi = min(15, nl)
        K = int(rng.integers(lo, hi + 1)); sel = tuple(sorted(rng.choice(nl, K, replace=False).tolist()))
        if (d, sel) in seen: continue
        seen.add((d, sel)); tasks.append([labs[i] for i in sel])
    return tasks

def episode(P, task, k, rng, nq=10):
    """desc (K,D), proto (K,D) (0 when k=0), support rows (idx,y), query rows (idx,y) from the pool."""
    K = len(task); desc = np.stack([P.dX[l] for l in task]); proto = np.zeros((K, D), np.float32); si, sy, qi, qy = [], [], [], []
    for c, l in enumerate(task):
        r = rng.permutation(P.rows[l]); kk = min(k, max(len(r) - nq, 0)); sup, qry = r[:kk], r[kk:kk + nq]
        if kk: proto[c] = P.X[sup].mean(0)
        si += list(sup); sy += [c] * kk; qi += list(qry); qy += [c] * len(qry)
    return desc, proto, np.array(si, int), np.array(sy, int), np.array(qi, int), np.array(qy, int)

def val_episodes(P, st, n_tasks=30, ks=(0, 1, 3, 10), seed=99):
    tasks = sample_tasks(P.ds_labels, [VAL_DS], 3 if TINY else n_tasks, seed); rng = np.random.default_rng(seed)
    return [(t, k, episode(P, t, k, rng)) for t in tasks for k in ks]

NTHREADS = int(os.environ.get("S17_THREADS", "2"))

# ================================================================== hypernetwork stage
def meta(model_name, scale, seed=0, steps=3000, E=8, ve=250):
    import torch, torch.nn.functional as F
    if TINY: steps, E, ve = 6, 2, 3
    torch.set_num_threads(NTHREADS); torch.manual_seed(seed)
    st = load_store(); P = PoolIdx(st); DOM = json.load(open(common.CACHE / "clinc_domains.json"))
    if scale == "clinc7":   # S2b recipe: only the CLINC train domains, tasks = label subsets inside one domain
        doms = sorted({DOM[l.split("::")[1]] for l in P.ds_labels["clinc"]})
        tasks = []
        rr = np.random.default_rng(seed)
        for i in range(2000):
            d = doms[i % len(doms)]; labs = [l for l in P.ds_labels["clinc"] if DOM[l.split("::")[1]] == d]; K = min(int(rr.integers(5, 16)), len(labs)); tasks.append([labs[j] for j in sorted(rr.choice(len(labs), K, replace=False))])
        desc = f"clinc7: {len(doms)} domains, {len(P.ds_labels['clinc'])} labels, episodes over label subsets inside a domain"
    elif scale.endswith("nb"):   # pool without any bank-like source: no Banking77 labels, no CLINC banking / credit_cards domains
        nbf = lambda d, labs: [l for l in labs if DOM.get(l.split("::")[1], "") not in ("banking", "credit_cards")] if d == "clinc" else labs
        tasks = sample_tasks(P.ds_labels, [d for d in P.fit_ds if d != "banking"], int(scale[:-2]), seed + 1, labels_filter=nbf); desc = f"pool without bank-like sources, {len(tasks)} distinct tasks"
    else:
        n = int(scale); tasks = sample_tasks(P.ds_labels, P.fit_ds, n, seed + 1); desc = f"pool, {len(tasks)} distinct tasks"
    nlab = len({l for t in tasks for l in t}); name = f"{model_name}_{scale}" + (f"_s{seed}" if seed else "")
    print(f"[meta {name}] tasks={len(tasks)} labels={nlab} ({desc})", flush=True)
    G = make_models()[model_name](); opt = torch.optim.AdamW(G.parameters(), 1e-3, weight_decay=1e-3 if model_name != "SETTR" else 1e-2)
    Xp = torch.tensor(P.X); val = val_episodes(P, st)
    @torch.no_grad()
    def val_acc(G):
        G.eval(); out = {}
        for t, k, (dsc, pr, si, sy, qi, qy) in val:
            W, b = G(torch.tensor(dsc), torch.tensor(pr), k); acc = float((G.logits(Xp[qi], W, b).argmax(1).numpy() == qy).mean()); out.setdefault(k, []).append(acc)
        return {k: float(np.mean(v)) for k, v in out.items()}
    dcos = {}
    for t, k, (dsc, pr, si, sy, qi, qy) in val: dcos.setdefault(k, []).append(float(((P.X[qi] @ dsc.T).argmax(1) == qy).mean()))
    print("   val desc-cos baseline", {k: round(float(np.mean(v)), 3) for k, v in dcos.items()}, flush=True)
    rng = np.random.default_rng(seed + 7); best, best_state, bad, hist = -1, None, 0, []; t0 = time.time()
    for step in range(steps):
        G.train(); loss = 0
        for _ in range(E):
            task = tasks[int(rng.integers(len(tasks)))]; k = KS[int(rng.integers(len(KS)))]
            dsc, pr, si, sy, qi, qy = episode(P, task, k, rng); W, b = G(torch.tensor(dsc), torch.tensor(pr), k)
            loss = loss + F.cross_entropy(G.logits(Xp[qi], W, b), torch.tensor(qy)) / E
        opt.zero_grad(); loss.backward(); opt.step()
        if step % ve == ve - 1:
            v = val_acc(G); m = float(np.mean(list(v.values()))); hist.append((step + 1, m)); print(f"   step {step+1} loss {float(loss):.3f} val {({k: round(a, 3) for k, a in v.items()})} mean {m:.3f} [{time.time()-t0:.0f}s]", flush=True)
            if m > best: best, bad, best_state = m, 0, {n_: p.clone() for n_, p in G.state_dict().items()}
            else:
                bad += 1
                if bad >= 4: break
    G.load_state_dict(best_state); torch.save(best_state, MODELS / f"{name}.pt"); v = val_acc(G)
    json.dump(dict(model=model_name, scale=scale, seed=seed, n_tasks=len(tasks), n_labels=nlab, best_val=best, val_by_k=v, hist=hist, seconds=time.time() - t0, desc=desc,
                   val_desc_cos={k: float(np.mean(x)) for k, x in dcos.items()}), open(MODELS / f"{name}.json", "w"))
    print(f"[meta {name}] best val {best:.3f} in {time.time()-t0:.0f}s", flush=True)


# ================================================================== arm 4: first-order MAML on a linear projection + cosine head
SCALE_R = 20.0
def fomaml(tag, lrM, lrR, T, steps=1500, E=8, seed=0, ve=250):
    import torch, torch.nn.functional as F
    if TINY: steps, E, ve = 4, 2, 2
    torch.set_num_threads(NTHREADS); torch.manual_seed(seed)
    st = load_store(); P = PoolIdx(st); tasks = sample_tasks(P.ds_labels, P.fit_ds, 800, seed + 1); Xp = torch.tensor(P.X); val = val_episodes(P, st)
    M0 = torch.eye(D).requires_grad_(True); outer = torch.optim.Adam([M0], 1e-3); rng = np.random.default_rng(seed + 3)
    def adapt(M_init, desc, Xs, ys, K):
        if T == 0 or len(ys) == 0: return M_init.detach().clone(), torch.zeros(K, D)
        return reptile_adapt(M_init, SCALE_R, desc, Xs, ys, K, T, lrR, lrM)
    def val_acc(M_init):
        out = {}
        for t, k, (dsc, pr, si, sy, qi, qy) in val:
            dsc_t = torch.tensor(dsc); M, r = adapt(M_init, dsc_t, Xp[si], torch.tensor(sy), len(t)) if k else (M_init.detach(), torch.zeros(len(t), D))
            with torch.no_grad(): out.setdefault(k, []).append(float((reptile_logits(M, r, SCALE_R, dsc_t, Xp[qi]).argmax(1).numpy() == qy).mean()))
        return {k: float(np.mean(v)) for k, v in out.items()}
    best, best_M, bad, hist = -1, None, 0, []; t0 = time.time()
    for step in range(steps):
        gM = torch.zeros(D, D); lq = 0
        for _ in range(E):
            task = tasks[int(rng.integers(len(tasks)))]; k = KS[int(rng.integers(len(KS)))]; dsc, pr, si, sy, qi, qy = episode(P, task, k, rng); dsc_t = torch.tensor(dsc); K = len(task)
            if k and T > 0:
                M, r = adapt(M0, dsc_t, Xp[si], torch.tensor(sy), K)
            else:
                M, r = M0.detach().clone(), torch.zeros(K, D)
            M = M.requires_grad_(True)
            loss = F.cross_entropy(reptile_logits(M, r, SCALE_R, dsc_t, Xp[qi]), torch.tensor(qy)); loss.backward(); gM += M.grad / E; lq += float(loss) / E
        M0.grad = gM; outer.step()
        if step % ve == ve - 1:
            with torch.no_grad(): v = val_acc(M0.detach())
            m = float(np.mean(list(v.values()))); hist.append((step + 1, m)); print(f"   [{tag}] step {step+1} q-loss {lq:.3f} val {({k: round(a, 3) for k, a in v.items()})} mean {m:.3f} [{time.time()-t0:.0f}s]", flush=True)
            if m > best: best, bad, best_M = m, 0, M0.detach().clone()
            else:
                bad += 1
                if bad >= 4: break
    torch.save(dict(M=best_M, T=T, lrR=lrR, lrM=lrM), MODELS / f"FOMAML_{tag}.pt")
    json.dump(dict(tag=tag, lrM=lrM, lrR=lrR, T=T, best_val=best, hist=hist, seconds=time.time() - t0), open(MODELS / f"FOMAML_{tag}.json", "w"))
    print(f"[fomaml {tag}] best val {best:.3f} in {time.time()-t0:.0f}s", flush=True)


# ================================================================== noise floor for the synthetic prior: a second, independent LLM draw of the practice examples
def synth2():
    st = load_store(); evals = st["evals"]; done = 0
    for e in evals:
        if "synth2_X" in e: continue
        labs = e["labels"]; desc = dict(zip(labs, e["desc_t"])); syn = {}
        for i in range(0, len(labs), 15):
            chunk = labs[i:i + 15]
            if e["family"] == "txn":
                prompt = ("For each spending category below write 10 short, varied, realistic raw bank-statement transaction descriptors as they appear on a US checking account statement "
                          "(merchant names mostly in capitals, store numbers, city/state, payment-rail words such as POS, ACH, PPD, WEB, ID numbers; mix debits and credits where plausible; "
                          "do not copy real account numbers) that belong to that category. Reply with a JSON object mapping category -> list of 10 strings, nothing else.\n" + "\n".join(f"{c}: {desc[c]}" for c in chunk))
            else:
                prompt = ("For each label below, write 10 short, varied, realistic user messages that would get that label (different phrasings, lengths, some typos, some indirect). "
                          "Reply with a JSON object mapping label -> list of 10 strings, nothing else.\n" + "\n".join(f"{l}: {desc[l]}" for l in chunk))
            for att in range(3):
                d = llm_json(prompt, tag=f"synth2-{att}", temperature=0.9)
                if all(c in d and len(d[c]) >= 5 for c in chunk): break
            for c in chunk: syn[c] = [str(x) for x in d.get(c, [])][:SN] or [desc[c]]
        e["synth2_t"] = [t for l in labs for t in syn[l]]; e["synth2_y"] = np.array([j for j, l in enumerate(labs) for _ in syn[l]]); e["synth2_X"] = embed(e["synth2_t"]); done += 1
    pickle.dump(st, open(STORE / "store.pkl", "wb"))
    print(f"synth2 done for {done} tasks; LLM spend total ${llm_spend():.4f}", flush=True)


# ================================================================== the whole pipeline on the pod
def n_cpus():
    """CPUs this container can really use (affinity and cgroup quota), capped at 8."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4)
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if q != "max": n = min(n, max(1, int(int(q) / int(p))))
    except Exception: pass
    try:
        q, p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read()), int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0: n = min(n, max(1, q // p))
    except Exception: pass
    return min(n, 8)

def pod():
    import subprocess, shutil
    t0 = time.time(); ncpu = n_cpus(); py = sys.executable; LOGS = HERE / "logs"; LOGS.mkdir(exist_ok=True)
    EVAL_AT = float(os.environ.get("S17_EVAL_AT_MIN", "44")) * 60   # background stages are cut off here
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", S17_THREADS="1", PYTHONUNBUFFERED="1")
    mainlog = open(HERE / "log.txt", "a", encoding="utf-8")
    def log(*a):
        s = f"[pod +{time.time()-t0:.0f}s] " + " ".join(str(x) for x in a); print(s, flush=True); mainlog.write(s + "\n"); mainlog.flush()
    def start(cmd, name, e=env):
        if os.environ.get("S17_DRYRUN"): cmd = [py, "-c", "import time; time.sleep(0.3)"]  # control-flow check only
        return subprocess.Popen(cmd, stdout=open(LOGS / f"{name}.log", "w", encoding="utf-8"), stderr=subprocess.STDOUT, env=e, cwd=str(HERE))
    try: gpu_name = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except Exception: gpu_name = "no nvidia-smi"
    log(f"cpus={ncpu} gpu={gpu_name}")
    for st in ("prep", "synth2"):
        p = start([py, "-u", str(HERE / "run.py"), st], st); p.wait(); log(st, "exit", p.returncode, (LOGS / f"{st}.log").read_text(encoding="utf-8")[-600:])
        if p.returncode: raise SystemExit(f"{st} failed")
    # GPU jobs (own processes, share the 4090); TabPFN lives in an isolated venv so its pip dependencies cannot disturb the other jobs
    bg = {"enc": start([py, "-u", str(HERE / "s17_enc.py")], "enc")}
    venv = "/workspace/job/tabvenv"
    mkvenv = f"(python -m venv --system-site-packages {venv} || (pip install -q virtualenv && python -m virtualenv --system-site-packages {venv}))"
    tab_cmd = f"{mkvenv} && {venv}/bin/pip install -q tabpfn==9.2.0 tabpfn-extensions==0.6.3 && {venv}/bin/python -u {HERE / 's17_tabpfn.py'}"
    bg["tabpfn"] = start(["bash", "-c", tab_cmd], "tabpfn")
    jobs = [["fomaml", "m0", "0.0", "0.05", "10"], ["fomaml", "m1", "0.001", "0.05", "10"],   # most important first: anything unfinished at EVAL_AT is cut
            ["meta", "SETTR", "800"], ["meta", "LINGEN", "800"], ["meta", "SETTR", "clinc7"], ["meta", "LINGEN", "clinc7"], ["meta", "SETTR", "100"], ["meta", "SETTR", "25"],
            ["meta", "LINGEN", "100"], ["meta", "LINGEN", "25"], ["meta", "SETTR", "800nb"], ["meta", "LINGEN", "800nb"], ["meta", "SETTR", "800", "1"], ["meta", "LINGEN", "800", "1"],
            ["meta", "METRIC", "800"], ["meta", "SETTR", "3000"], ["meta", "METRIC", "clinc7"]]
    W = max(2, ncpu - 2); running = []
    while True:
        for j, (p, nm) in list(enumerate(running))[::-1]:
            if p.poll() is not None: log("done", nm, "exit", p.returncode); running.pop(j)
        for k_, p in list(bg.items()):
            if p.poll() is not None: log("done", k_, "exit", p.returncode); bg.pop(k_)
        if time.time() - t0 > EVAL_AT:
            for p, nm in running: log("cut off", nm); p.kill()
            for k_, p in bg.items(): log("cut off", k_); p.kill()
            break
        while jobs and len(running) < W:
            a = jobs.pop(0); nm = "_".join(a); running.append((start([py, "-u", str(HERE / "run.py")] + a, nm), nm)); log("start", nm)
        if not jobs and not running and not bg: break
        time.sleep(0.5 if os.environ.get("S17_DRYRUN") else 5)
    # choose the FOMAML configuration and the prior-LR models from the held-out pool dataset, then evaluate everything
    fm = {p.stem[7:]: json.load(open(p)) for p in MODELS.glob("FOMAML_*.json")}
    e = dict(env); e.pop("S17_THREADS", None)
    if fm: e["S17_REP"] = max(fm, key=lambda t: fm[t]["best_val"]); log("FOMAML configs", {t: round(v["best_val"], 3) for t, v in fm.items()}, "-> using", e["S17_REP"])
    e["S17_PRIOR"] = json.dumps({n: 0.01 for n in ("SETTR_800", "LINGEN_800") if (MODELS / f"{n}.pt").exists()})
    p = start([py, "-u", str(HERE / "run.py"), "eval", str(ncpu)], "eval", e); p.wait(); log("eval exit", p.returncode, (LOGS / "eval.log").read_text(encoding="utf-8")[-400:])
    (HERE / "models_json").mkdir(exist_ok=True)
    for f in MODELS.glob("*.json"): shutil.copy(f, HERE / "models_json" / f.name)
    p = start([py, "-I", str(HERE / "report.py")], "report"); p.wait(); log("report exit", p.returncode)
    log("finished; spend on LLM $%.4f" % llm_spend())


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "eval":
        import evalmod
        evalmod.evalrun(nproc=int(sys.argv[2]) if len(sys.argv) > 2 else 6, tasks=json.loads(sys.argv[3]) if len(sys.argv) > 3 else None, seeds=json.loads(sys.argv[4]) if len(sys.argv) > 4 else None,
                        outname=sys.argv[5] if len(sys.argv) > 5 else "results.csv")
    else:
        {"prep": prep, "meta": lambda: meta(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 0), "synth2": synth2, "pod": pod,
         "fomaml": lambda: fomaml(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]) if len(sys.argv) > 6 else 1500)}[st]()
