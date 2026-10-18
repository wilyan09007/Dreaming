"""S17 arm 6 (TabPFN v2 in-context classifier). Runs on the pod GPU inside an isolated venv (pip tabpfn==9.2.0 tabpfn-extensions==0.6.3 on top of the system torch).
Input: WORK/store.pkl. Features: PCA to 32 dims of the frozen bge-small embeddings (PCA fitted on unlabelled pool rows only).
Tasks: the five 12-label Banking77 sub-tasks (TabPFN v2 handles at most 10 classes natively, so tabpfn_extensions' ManyClassClassifier is used: alphabet 10, 3x redundancy).
Variants: 'real' = k real rows per label; 'synth' = the LLM-written practice examples + k real rows. Seed-major loop; a seed is saved only when complete; time-boxed.
Output: WORK/gpu_out/out_tabpfn_seed<s>.npz (class probabilities for the calibration and test rows) and results_tabpfn.json."""
import os, sys, json, time, pickle, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import *
T0 = time.time(); R = {"log": []}
LIMIT = float(os.environ.get("S17_TABPFN_MAX_MIN", "30")) * 60
def log(*a):
    s = f"[{time.time()-T0:.0f}s] " + " ".join(str(x) for x in a); print(s, flush=True); R["log"].append(s)
def save():
    json.dump(R, open(HERE / ("results_tabpfn_tiny.json" if TINY else "results_tabpfn.json"), "w"), indent=1)
OUT = WORK / "gpu_out"; OUT.mkdir(exist_ok=True)
try:
    import torch
    from sklearn.decomposition import PCA
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    from tabpfn_extensions.many_class import ManyClassClassifier
    dev = "cuda" if torch.cuda.is_available() else "cpu"; R["device"] = dev; log("device", dev, "torch", torch.__version__)
    st = pickle.load(open(WORK / "store.pkl", "rb")); pool, ev = st["pool"], st["evals"]
    pca = PCA(32, random_state=0).fit(pool["X"][np.random.default_rng(0).permutation(len(pool["X"]))[:20000]]); R["pca_explained_variance"] = float(pca.explained_variance_ratio_.sum())
    tasks = [i for i, e in enumerate(ev) if e["family"] == "banking"]; log("tasks", [ev[i]["name"] for i in tasks], "pca var", R["pca_explained_variance"])
    def fit_predict(Xtr, ytr, Xte, K):
        base = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device=dev, n_estimators=2 if TINY else 4)
        clf = ManyClassClassifier(base, alphabet_size=10, n_estimators_redundancy=3, random_state=0) if K > 10 or TINY else base
        clf.fit(Xtr, ytr); p = np.zeros((len(Xte), K), np.float32); p[:, clf.classes_.astype(int)] = clf.predict_proba(Xte); return p
    t_fit = time.time(); n_fit = 0
    for seed in range(1 if TINY else SEEDS):
        if time.time() - T0 > LIMIT: log(f"time box reached before seed {seed}; stopping"); break
        res = {}
        for i in tasks:
            e = ev[i]; K = len(e["labels"]); ys = e["ys"]; Xs, Xc, Xq, Sx = (pca.transform(e[k]).astype(np.float32) for k in ("Xs", "Xc", "Xq", "synth_X")); sy = e["synth_y"]
            for k in (KS[:3] if TINY else KS):
                if k == 0 and seed > 0: continue
                idx = draw_support(ys, k, seed, K) if k else np.array([], int)
                for variant in ("real", "synth"):
                    if variant == "real" and k == 0: continue
                    Xtr = np.concatenate([Sx, Xs[idx]]) if variant == "synth" else Xs[idx]; ytr = np.concatenate([sy, ys[idx]]) if variant == "synth" else ys[idx]
                    p = fit_predict(Xtr, ytr, np.concatenate([Xc, Xq]), K); n_fit += 1
                    res[f"pc|{i}|{variant}|{k}"] = p[:len(Xc)].astype(np.float16); res[f"pq|{i}|{variant}|{k}"] = p[len(Xc):].astype(np.float16)
            log(f"seed {seed} task {i} done, fits so far {n_fit}, {(time.time()-t_fit)/max(n_fit,1):.2f}s/fit")
            if time.time() - T0 > LIMIT: break
        else:
            np.savez_compressed(OUT / f"out_tabpfn_seed{seed}.npz", **res); R.setdefault("seeds_done", []).append(seed); save(); continue
        log(f"time box hit inside seed {seed}; that seed is dropped"); break
    R["fits"] = n_fit
except Exception:
    R["error"] = traceback.format_exc()[-1500:]; log("TABPFN FAILED", R["error"])
R["total_seconds"] = time.time() - T0; save(); log("finished")
