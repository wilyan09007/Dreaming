"""S24: student families that are not a transformer adapter. Everything real runs on a RunPod pod:
  python smokes/pod.py --tag s24 --bundle smokes/common.py smokes/s24_other_families smokes/.cache/label_descriptions.json smokes/.cache/synth_examples.json smokes/.cache/clinc_domains.json \
      --cmd "python smokes/s24_other_families/run.py" --max-min 50 --gpu "NVIDIA RTX A5000"
Arms (in this order; results.json is rewritten after every (arm, task, k, draw)): 0 baseline LR | 1 memory students (kNN, Hopfield, +synthetic rows) | 2 Bayesian heads (GP, Laplace, temperature LR)
| 3 prior-fitted network (meta-trained here, in-context at test time) | 2b' laplace-torch (if it installs) | 4 symbolic rule student (runs in a thread from the start, LLM + subprocess).
S24_TINY=1 shrinks everything to a plumbing check (no LLM, 25 s of PFN training, stub rule code)."""
import sys, os, json, re, time, threading, subprocess, traceback
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
import arms, symbolic

RES = HERE / ("tiny_results.json" if TINY else "results.json")
R = dict(meta={}, records=[]); RLOCK = threading.RLock()
def save():
    with RLOCK: tmp = RES.with_suffix(".tmp"); tmp.write_text(json.dumps(R, default=lambda o: o.item() if hasattr(o, "item") else str(o))); os.replace(tmp, RES)
def rec(**kw):
    with RLOCK: R["records"].append(kw); save()
def emit(arm, e, k, seed, sc, fit_s, pred_s, nrows, params, **extra):
    rec(arm=arm, task=e["name"], family=e["family"], k=k, seed=seed, K=e["K"], **sc, fit_s=fit_s, pred_ms_1000=1e6 * pred_s / nrows, params=params, **extra)
def guarded(tag, fn):  # an arm that crashes is logged and recorded as a failure; the other arms still run
    try: fn()
    except Exception:
        msg = traceback.format_exc()[-800:]; log(f"ARM FAILED {tag}:\n{msg}")
        with RLOCK: R["meta"].setdefault("failures", {})[tag] = msg; save()

FAILED = set()
def tryarm(tag, fn):  # one arm failing (API drift, degenerate fit) must not take the other arms of the pass down; recorded once per arm
    try: fn()
    except Exception:
        if tag in FAILED: return
        FAILED.add(tag); msg = traceback.format_exc()[-800:]; log(f"ARM FAILED {tag}:\n{msg}")
        with RLOCK: R["meta"].setdefault("failures", {})[tag] = msg; save()

# ------------------------------------------------------------------ optional: laplace-torch, installed in a background thread (5 minute cap, --no-deps so torch is never touched)
LAP = dict(ok=False, msg="not tried")
def install_laplace():
    if TINY: LAP["msg"] = "skipped in tiny mode"; return
    t = time.perf_counter(); py = sys.executable
    try:
        subprocess.run([py, "-m", "pip", "install", "-q", "--no-deps", "laplace-torch"], timeout=150, capture_output=True)
        for _ in range(10):
            p = subprocess.run([py, "-c", "from laplace import Laplace"], capture_output=True, text=True, timeout=120)
            if p.returncode == 0: break
            m = re.search(r"No module named '([\w\.\-]+)'", p.stderr)
            if not m or time.perf_counter() - t > 270: LAP["msg"] = "import failed: " + p.stderr[-300:].replace("\n", " "); return
            mod = m.group(1).split(".")[0]; pkg = {"backpack": "backpack-for-pytorch", "curvlinops": "curvlinops-for-pytorch"}.get(mod, mod)
            subprocess.run([py, "-m", "pip", "install", "-q", "--no-deps", pkg], timeout=120, capture_output=True)
        else: LAP["msg"] = "dependency loop did not converge"; return
        import importlib; importlib.invalidate_caches()
        r = np.random.default_rng(0); X = r.standard_normal((30, D)).astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True); y = np.repeat(np.arange(3), 10)
        P = arms.laplace_torch(arms.lr_fit(X, y), X, y, X[:5])[0]; assert P.shape == (5, 3) and np.isfinite(P).all()
        LAP.update(ok=True, msg=f"installed and self-tested in {time.perf_counter()-t:.0f}s")
    except Exception as ex: LAP["msg"] = repr(ex)[:300]
    log("laplace-torch:", LAP)

# ------------------------------------------------------------------ arms 0, 1, 2
def support(e, k, seed):
    idx = draw_support(e["ys"], k, seed, e["K"]); return e["Xs"][idx], e["ys"][idx]
def pass_memory(evals, ks, seeds):
    for e in evals:
        K, Xc, Xq, yc, yq = e["K"], e["Xc"], e["Xq"], e["yc"], e["yq"]; XA = np.vstack([Xc, Xq]); nc, nr = len(Xc), len(Xc) + len(Xq)
        for k in ks:
            for seed in range(seeds):
                X, y = support(e, k, seed); kw = lambda arm, sc, fit, pr, params, **x: emit(arm, e, k, seed, sc, fit, pr, nr, params, **x)
                M2, y2 = np.vstack([X, e["synth_X"]]), np.concatenate([y, e["synth_y"]])
                def base():
                    t = time.perf_counter(); clf = arms.lr_fit(X, y); fit = time.perf_counter() - t; t = time.perf_counter(); P = clf.predict_proba(XA); pr = time.perf_counter() - t
                    kw("0 baseline LR (frozen bge-small)", score(P[:nc], yc, P[nc:], yq), fit, pr, K * D + K)
                def knn(M, my, kn, arm):
                    t = time.perf_counter(); pred, conf = arms.knn(M, my, XA, K, kn); pr = time.perf_counter() - t
                    kw(arm, score_conf(conf[:nc], pred[:nc] == yc, conf[nc:], pred[nc:] == yq), 0.0, pr, 0)
                def hop(M, my, arm):
                    t = time.perf_counter(); beta = arms.hopfield_fit_beta(M, my, Xc, yc, K); fit = time.perf_counter() - t; t = time.perf_counter(); P = arms.hopfield(M, my, XA, K, beta); pr = time.perf_counter() - t
                    kw(arm, score(P[:nc], yc, P[nc:], yq), fit, pr, 0, beta=beta)
                for tag, fn in (("0", base), ("1a", lambda: knn(X, y, min(5, k), "1a kNN vote (k_nn=min(5,k))")), ("1b", lambda: hop(X, y, "1b Hopfield read-out")),
                                ("1c-knn", lambda: knn(M2, y2, 5, "1c kNN + LLM-written rows")), ("1c-hop", lambda: hop(M2, y2, "1c Hopfield + LLM-written rows"))): tryarm(tag, fn)
        log(f"memory arms done for {e['name']}")

def pass_bayes(evals, ks, seeds):
    for e in evals:
        K, Xc, Xq, yc, yq = e["K"], e["Xc"], e["Xq"], e["yc"], e["yq"]; XA = np.vstack([Xc, Xq]); nc, nr = len(Xc), len(Xc) + len(Xq)
        for k in ks:
            for seed in range(seeds):
                X, y = support(e, k, seed); kw = lambda arm, sc, fit, pr, params, **x: emit(arm, e, k, seed, sc, fit, pr, nr, params, **x)
                def gp():
                    P, fit, pr = arms.gp_fit_predict(X, y, XA); kw("2a GP classifier (RBF, one-vs-rest)", score(P[:nc], yc, P[nc:], yq), fit, pr, 2 * K)
                def gpf():
                    P, fit, pr, ls = arms.gp_fixed(X, y, Xc, yc, XA); kw("2a' GP classifier (fixed RBF, length-scale from calibration set)", score(P[:nc], yc, P[nc:], yq), fit, pr, 0, length_scale=ls)
                t = time.perf_counter(); clf = arms.lr_fit(X, y); fl = time.perf_counter() - t
                def lap():
                    t = time.perf_counter(); st = arms.laplace_fit(clf, X, y); fit = fl + time.perf_counter() - t; t = time.perf_counter(); P = arms.laplace_predict(st, XA); pr = time.perf_counter() - t
                    kw("2b Laplace last layer (block-diag GGN, own code)", score(P[:nc], yc, P[nc:], yq), fit, pr, K * (D + 1), alpha=st["alpha"])
                def temp():
                    t = time.perf_counter(); T = arms.temperature(clf.decision_function(Xc), yc); fit = fl + time.perf_counter() - t
                    t = time.perf_counter(); P = arms.softmax(clf.decision_function(XA) / T); pr = time.perf_counter() - t
                    kw("2c LR + temperature (fit on calibration set)", score(P[:nc], yc, P[nc:], yq), fit, pr, K * (D + 1) + 1, T=T)
                for tag, fn in (("2b", lap), ("2c", temp), ("2a", gp), ("2a'", gpf)): tryarm(tag, fn)
        log(f"bayes arms done for {e['name']}")

def pass_laplace_torch(evals, ks, seeds):
    LAP_THREAD.join(timeout=120)
    if not LAP["ok"]: log("laplace-torch not available:", LAP["msg"]); return
    for e in evals:
        K, Xc, Xq, yc, yq = e["K"], e["Xc"], e["Xq"], e["yc"], e["yq"]; XA = np.vstack([Xc, Xq]); nc, nr = len(Xc), len(Xc) + len(Xq)
        for k in ks:
            for seed in range(seeds):
                X, y = support(e, k, seed); clf = arms.lr_fit(X, y); P, prior, fit, pr = arms.laplace_torch(clf, X, y, XA)
                emit("2b' Laplace last layer (laplace-torch, kron)", e, k, seed, score(P[:nc], yc, P[nc:], yq), fit, pr, nr, K * (D + 1), prior=prior)

# ------------------------------------------------------------------ arm 3
def run_pfn(S, ks, seeds):
    import pfn
    secs = int(os.environ.get("S24_PFN_SEC", 25 if TINY else 840))
    model, nz, info = pfn.train_pfn(S["pool"], S["val"], secs, B=16 if TINY else 48, log=log); R["meta"]["pfn"] = info; save()
    for e in S["evals"]:
        K, Xc, Xq, yc, yq = e["K"], e["Xc"], e["Xq"], e["yc"], e["yq"]; XA = np.vstack([Xc, Xq]); nc, nr = len(Xc), len(Xc) + len(Xq)
        pfn.pfn_probs(model, nz["mu"], nz["sd"], e["Xs"][:K], e["ys"][:K], XA[:8], K)  # warm-up (CUDA init)
        for k in ks:
            for seed in range(seeds):
                X, y = support(e, k, seed)
                for nperm, arm in ((1, "3 PFN in-context (1 pass)"), (4, "3 PFN in-context (4 slot permutations)")):
                    t = time.perf_counter(); P = pfn.pfn_probs(model, nz["mu"], nz["sd"], X, y, XA, K, nperm=nperm); pr = time.perf_counter() - t
                    emit(arm, e, k, seed, score(P[:nc], yc, P[nc:], yq), 0.0, pr, nr, 0, meta_params=info["params"])
        log(f"PFN eval done for {e['name']}")

# ------------------------------------------------------------------ arm 4
def stub_code(e):
    return "import re\nLAB = %r\ndef classify(text):\n    t = text.lower()\n    for l in LAB:\n        if re.search(r'\\b' + re.escape(l.lower().split('_')[0][:4]), t):\n            return l\n    return None\n" % (e["labels"],)
def run_symbolic(evals, ks, seeds):
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(e, k, s) for k in sorted(ks, reverse=True) for s in range(seeds) for e in evals]
    def one(a):
        e, k, s = a
        try: r = symbolic.job(e, k, s, stub=stub_code(e) if TINY else None)
        except Exception: r = dict(arm="4 symbolic rules (LLM-written)", task=e["name"], family=e["family"], k=k, seed=s, K=e["K"], status="exception: " + traceback.format_exc()[-300:])
        rec(**r)
    with ThreadPoolExecutor(1 if TINY else 4) as ex: list(ex.map(one, jobs))
    log(f"symbolic arm done; LLM spend so far ${llm_spend():.4f}")

def main():
    t0 = time.perf_counter()
    if not TINY:
        import torch; assert torch.cuda.is_available(), "no CUDA on this pod"; R["meta"]["gpu"] = torch.cuda.get_device_name(0)
    R["meta"].update(tiny=TINY, cpus=os.cpu_count(), started=time.strftime("%Y-%m-%d %H:%M:%S"))
    global LAP_THREAD; LAP_THREAD = threading.Thread(target=install_laplace, daemon=True); LAP_THREAD.start()
    S = prep(); R["meta"]["prep"] = S["info"]; R["meta"]["tasks"] = {e["name"]: dict(K=e["K"], support_rows=len(e["ts"]), cal_rows=len(e["tc"]), test_rows=len(e["tq"]), synth_rows=len(e["synth_t"])) for e in S["evals"]}
    R["meta"]["prep_s"] = round(time.perf_counter() - t0); save()
    ks, seeds = ([1, 3], 2) if TINY else (KS, SEEDS)
    sym = threading.Thread(target=lambda: guarded("symbolic", lambda: run_symbolic(S["evals"], ks, seeds))); sym.start()
    guarded("memory", lambda: pass_memory(S["evals"], ks, seeds))
    guarded("bayes", lambda: pass_bayes(S["evals"], ks, seeds))
    guarded("pfn", lambda: run_pfn(S, ks, seeds))
    guarded("laplace_torch", lambda: pass_laplace_torch(S["evals"], ks, seeds))
    sym.join()
    R["meta"].update(laplace_torch=LAP, llm_spend_usd=round(llm_spend(), 5), wall_s=round(time.perf_counter() - t0)); save()
    log(f"all done in {time.perf_counter()-t0:.0f}s; {len(R['records'])} records; LLM ${llm_spend():.4f}; laplace-torch: {LAP['msg']}")

if __name__ == "__main__":
    main()
