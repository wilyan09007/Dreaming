"""S17 evaluation harness: all arms x eval tasks x k x draws. Imported by run.py (so multiprocessing 'spawn' workers can import it).
One unit = (eval task, draw seed); every arm is scored with the same accuracy + fixed-grid certificate (300 calibration rows, 5% budget) on the test rows."""
import os, sys, json, time, pickle, traceback
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from lib import *

STORE = WORK; MODELS = STORE / "models"; GPU_OUT = STORE / "gpu_out"; SCALE_R = 20.0
EV = VIEWS = TAB = HYP = REP = None; PRIOR = {}

def load_store():
    return pickle.load(open(STORE / "store.pkl", "rb"))

def _init(hyp_names, rep_tag, prior):
    global EV, VIEWS, TAB, HYP, REP, PRIOR
    import torch; torch.set_num_threads(1)
    st = load_store(); EV = st["evals"]; PRIOR = prior; del st
    VIEWS = {"base": [dict(Xs=e["Xs"], Xc=e["Xc"], Xq=e["Xq"], desc=e["desc_X"], synth=e["synth_X"]) for e in EV]}
    nz = lambda a: (lambda x: x / np.linalg.norm(x, axis=1, keepdims=True))(a.astype(np.float32))
    for v in ("all", "clinc7", "nobank", "all_s1"):
        f = GPU_OUT / f"out_enc_{v}.npz"
        if f.exists():
            z = np.load(f); VIEWS["ft-" + v] = [{kk: nz(z[f"{src}{i}"]) for kk, src in (("Xs", "Xs"), ("Xc", "Xc"), ("Xq", "Xq"), ("desc", "desc"), ("synth", "synth"))} for i in range(len(EV))]
    TAB = {}
    for sd in range(SEEDS):
        f = GPU_OUT / f"out_tabpfn_seed{sd}.npz"
        if f.exists(): TAB[sd] = dict(np.load(f))
    HYP = {}
    for nm in hyp_names:  # a model file cut off by the stage time limit must not take the pool down
        try: G = make_models()[nm.split("_")[0]](); G.load_state_dict(torch.load(MODELS / f"{nm}.pt")); G.eval(); HYP[nm] = G
        except Exception as ex: print("skip hyper model", nm, repr(ex)[:100], flush=True)
    try: REP = torch.load(MODELS / f"FOMAML_{rep_tag}.pt") if rep_tag else None
    except Exception as ex: REP = None; print("skip FOMAML", repr(ex)[:100], flush=True)

def fit_prior(X, y, W0, lam, scale, K, steps=200):
    """linear classifier trained on the k rows with an L2 pull toward the generated weights W0."""
    import torch
    W0t = torch.tensor(W0); W = W0t.clone().requires_grad_(True); b = torch.zeros(K, requires_grad=True); Xt, yt = torch.tensor(X), torch.tensor(y)
    opt = torch.optim.Adam([W, b], 0.05); F = torch.nn.functional
    for _ in range(steps):
        loss = F.cross_entropy(scale * Xt @ W.T + b, yt) + lam * ((W - W0t) ** 2).sum(); opt.zero_grad(); loss.backward(); opt.step()
    return lambda Xn: torch.softmax(scale * torch.tensor(Xn) @ W.T + b, -1).detach().numpy()

def lr_fit(X, y, K, C=10):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, max_iter=1000).fit(X, y)

def al_select(Xc_, yc_, SX, sy, K, B, rounds):
    """uncertainty sampling: the current student (synthetic prior + rows labelled so far) picks the pool rows with the smallest top-2 gap."""
    chosen = []; sizes = [B // rounds + (1 if r < B % rounds else 0) for r in range(rounds)]
    for sz in sizes:
        X = np.concatenate([SX] + ([Xc_[chosen]] if chosen else [])); y = np.concatenate([sy] + ([yc_[chosen]] if chosen else []))
        p = full_proba(lr_fit(X, y, K), Xc_, K); conf = confidence(p); conf[chosen] = np.inf
        chosen += list(np.argsort(conf, kind="stable")[:sz])
    return np.array(chosen)

def run_unit(arg):
    import torch
    i, seed = arg; e = EV[i]; K = len(e["labels"]); yc, yq, ys, sy = e["yc"], e["yq"], e["ys"], e["synth_y"]; out = []
    def rec(arm, k, pc, pq):
        out.append(dict(arm=arm, task=e["name"], family=e["family"], k=k, seed=seed, K=K, **score(pc, yc, pq, yq)))
    def safe(tag, k, fn):  # an arm that fails is logged and skipped; the rest of the unit still runs
        try: fn()
        except Exception: print(f"ARM FAILED {tag} {e['name']} seed={seed} k={k}: {traceback.format_exc()[-600:]}", flush=True)
    def views_block(k, idx, cand):
        for vname, V in VIEWS.items():
            V = V[i]
            Xs, Xc, Xq, desc, SX = V["Xs"], V["Xc"], V["Xq"], V["desc"], V["synth"]
            XR, yR = Xs[idx], ys[idx]
            if vname == "base":
                if k == 0: rec("A1 baseline", 0, softmax_cos(Xc, desc), softmax_cos(Xq, desc))
                else:
                    clf = lr_fit(XR, yR, K); rec("A1 baseline", k, full_proba(clf, Xc, K), full_proba(clf, Xq, K))
                P = np.zeros((K, D), np.float32)
                if k:
                    for c in range(K): P[c] = XR[yR == c].mean(0)
                    rec("A6 nearest-prototype", k, softmax_cos(Xc, P), softmax_cos(Xq, P))
                rec("A6 desc+prototype", k, softmax_cos(Xc, desc + P), softmax_cos(Xq, desc + P))
            else:  # fine-tuned encoder view: frozen encoder + LR
                if k == 0: rec(f"A3 {vname}", 0, softmax_cos(Xc, desc), softmax_cos(Xq, desc))
                else:
                    clf = lr_fit(XR, yR, K); rec(f"A3 {vname}", k, full_proba(clf, Xc, K), full_proba(clf, Xq, K))
            clf = lr_fit(np.concatenate([SX, XR]), np.concatenate([sy, yR]), K)
            rec("A2 synthetic prior" if vname == "base" else f"A3 {vname}+synth", k, full_proba(clf, Xc, K), full_proba(clf, Xq, K))
            if vname == "base" and "synth2_X" in e:  # noise floor: same recipe, second independent LLM draw of the practice examples
                clf = lr_fit(np.concatenate([e["synth2_X"], XR]), np.concatenate([e["synth2_y"], yR]), K); rec("A2b synthetic prior (2nd LLM draw)", k, full_proba(clf, Xc, K), full_proba(clf, Xq, K))
            if k and vname in ("base", "ft-all"):  # active learning with the synthetic-prior student; unlabelled pool = 40 random support rows per label
                rounds = min(4, k); sel = al_select(Xs[cand], ys[cand], SX, sy, K, k * K, rounds); XA, yA = Xs[cand][sel], ys[cand][sel]
                clf = lr_fit(np.concatenate([SX, XA]), np.concatenate([sy, yA]), K)
                rec("A7 active learning (synth student)" if vname == "base" else "A8 ft-all+synth+active learning", k, full_proba(clf, Xc, K), full_proba(clf, Xq, K))
    def fomaml_block(k, idx):
        base = VIEWS["base"][i]; desc, Xs = base["desc"], base["Xs"]; dt = torch.tensor(desc)
        M, r = reptile_adapt(REP["M"], SCALE_R, dt, torch.tensor(Xs[idx]), torch.tensor(ys[idx]), K, REP["T"], REP["lrR"], REP["lrM"]) if k else (REP["M"], torch.zeros(K, D))
        with torch.no_grad(): rec("A4 meta-learned start (FOMAML)", k, *[torch.softmax(reptile_logits(M, r, SCALE_R, dt, torch.tensor(X)), -1).numpy() for X in (base["Xc"], base["Xq"])])
    def hyper_block(k, idx, nm, G):
        base = VIEWS["base"][i]; desc, Xs, Xc, Xq = base["desc"], base["Xs"], base["Xc"], base["Xq"]; XR, yR = Xs[idx], ys[idx]
        P = np.zeros((K, D), np.float32)
        if k:
            for c in range(K): P[c] = XR[yR == c].mean(0)
        with torch.no_grad():
            W, b = G(torch.tensor(desc), torch.tensor(P), k)
            rec(f"A5 hyper {nm}", k, *[torch.softmax(G.logits(torch.tensor(X), W, b), -1).numpy() for X in (Xc, Xq)])
        if nm in PRIOR and k:  # generated weights as the prior of a regularised classifier
            W0 = W.detach().numpy().astype(np.float32); f = fit_prior(XR, yR, W0, PRIOR[nm], float(G.scale), K); rec(f"A5 hyper {nm} -> prior LR", k, f(Xc), f(Xq))
    def tab_block(k):
        for variant in ("real", "synth"):
            kc, kq = f"pc|{i}|{variant}|{k}", f"pq|{i}|{variant}|{k}"
            if kc in TAB[seed]: rec(f"A6 TabPFN ({variant} rows)", k, TAB[seed][kc].astype(np.float64), TAB[seed][kq].astype(np.float64))
    for k in KS:
        if k == 0 and seed > 0: continue
        idx = draw_support(ys, k, seed, K) if k else np.array([], int)
        cand = np.random.default_rng(2000 + seed).choice(len(ys), min(len(ys), 40 * K), replace=False)
        safe("views", k, lambda: views_block(k, idx, cand))
        if REP is not None: safe("fomaml", k, lambda: fomaml_block(k, idx))
        for nm, G in HYP.items(): safe(f"hyper {nm}", k, lambda: hyper_block(k, idx, nm, G))
        if seed in TAB: safe("tabpfn", k, lambda: tab_block(k))
    return out

def evalrun(tasks=None, seeds=None, nproc=6, outname="results.csv"):
    import multiprocessing as mp, pandas as pd
    if TINY: outname = "tiny_" + outname
    hyp = sorted(p.stem for p in MODELS.glob("*.pt") if not p.stem.startswith("FOMAML")); rep = os.environ.get("S17_REP", "")
    prior = json.loads(os.environ.get("S17_PRIOR", "{}"))
    n = len(load_store()["evals"]); tasks = tasks if tasks is not None else list(range(n)); seeds = seeds if seeds is not None else list(range(1 if TINY else SEEDS))
    units = [(i, sd) for sd in seeds for i in tasks]; print("units", len(units), "hyper models", hyp, "rep", rep, "prior", prior, flush=True)
    rows = []; t0 = time.time()
    with mp.get_context("spawn").Pool(nproc, initializer=_init, initargs=(hyp, rep, prior)) as pool:
        for j, r in enumerate(pool.imap_unordered(run_unit, units)):
            rows += r; print(f"unit {j+1}/{len(units)} done [{time.time()-t0:.0f}s]", flush=True)
            if (j + 1) % 10 == 0: pd.DataFrame(rows).to_csv(HERE / ("partial_" + outname), index=False)
    df = pd.DataFrame(rows); df.to_csv(HERE / outname, index=False); print("wrote", outname, len(df), flush=True)
    p = HERE / ("partial_" + outname)
    if p.exists(): p.unlink()
