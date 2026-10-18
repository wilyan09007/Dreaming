"""S20: pretrained in-context classifiers (TabICL v2, TabPFN v2 + many-class wrapper) as the few-row head over frozen
bge-small embeddings (PCA to 32/64 dims, fit on the training rows only), vs LR / nearest prototype / S2b synthetic-prior LR.
  python smokes/s20_tabicl_head/run.py            (pod, from the project root) everything -> results.npz, meta.json, tables.md
  python smokes/s20_tabicl_head/run.py --tiny     laptop pipeline check on cached embeddings (a handful of rows, writes to ./tiny)
  python smokes/s20_tabicl_head/run.py report     re-print the tables from results.npz / meta.json
Tasks are S2b's (s2_hypernet/run2.py): rng(0) -> 3 held-out CLINC domains, then five 12-label Banking77 subsets, plus 77-way Banking77.
Per (task, seed, k): calibration = 300 random rows of the task's train split (fixed per task/seed), training rows = k per label
from the REST of the train split, evaluation = the task's test split (Banking77: fixed 1000-row subset, same for every method).
Metrics come from (correct, confidence = top-2 gap) arrays so common.certify is the one certificate code for every method."""
import os, sys, json, time, io, argparse, warnings
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import load_banking77, load_clinc150, embed, certify, apply_certificate, confidence as top2, CACHE
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
KS = [1, 3, 5, 10, 20]; NCAL = 300; QCAP = 0; BUDGET = 0.05; B77_Q = 1000
ARMS = "tabicl:32,tabicl:64,tabpfn:32,tabpfn:64,tabicl_synth:64"
GROUPS = {"Banking77 (77-way)": lambda n: n == "banking77", "Banking77 12-label x5": lambda n: n.startswith("b12_"),
          "CLINC150 held-out domains x3 (15-way)": lambda n: n.startswith("clinc_")}

# ---------------------------------------------------------------- tasks and draws
def make_spec():
    DOM = json.load(open(CACHE / "clinc_domains.json", encoding="utf-8")); cn = load_clinc150()["names"]; bn = load_banking77()["names"]
    rng = np.random.default_rng(0); domains = sorted(set(DOM.values()))
    held = [str(x) for x in rng.choice(domains, 3, replace=False)]
    b12 = [sorted(rng.choice(77, 12, replace=False).tolist()) for _ in range(5)]
    T = {"banking77": dict(ds="b77", labels=list(range(77)))}
    T.update({f"b12_{i}": dict(ds="b77", labels=l) for i, l in enumerate(b12)})
    for d in held:
        T[f"clinc_{d}"] = dict(ds="clinc", labels=[i for i, n in enumerate(cn) if n != "oos" and DOM[n] == d])
    for t in T.values(): t["names"] = [(bn if t["ds"] == "b77" else cn)[i] for i in t["labels"]]
    return dict(tasks=T, held=held)

def build_task(st, D):
    Xa, ya, Xb, yb = D[st["ds"]]; L = st["labels"]; lid = {l: i for i, l in enumerate(L)}
    ma, mb = np.isin(ya, L), np.isin(yb, L); Xq, yq = Xb[mb], np.array([lid[v] for v in yb[mb]])
    cap = QCAP or (B77_Q if len(L) == 77 else 10**9)
    if len(yq) > cap: sub = np.sort(np.random.default_rng(7).choice(len(yq), cap, replace=False)); Xq, yq = Xq[sub], yq[sub]
    return dict(Xs=Xa[ma], ys=np.array([lid[v] for v in ya[ma]]), Xq=Xq, yq=yq, names=st["names"], K=len(L))

def draw(t, k, seed):
    """-> (support idx, calibration idx) into the task's train rows; disjoint; deterministic in (task, k, seed)."""
    n = len(t["ys"]); cal = np.random.default_rng(10_000 + seed).choice(n, NCAL, replace=False)
    pool = np.setdiff1d(np.arange(n), cal); rs = np.random.default_rng(seed * 1000 + k)
    sup = np.concatenate([rs.choice(pool[t["ys"][pool] == c], k, replace=False) for c in range(t["K"])])
    return sup, cal

def pca_feats(ctx, others, dim):
    from sklearn.decomposition import PCA
    nc = min(dim, len(ctx) - 1); p = PCA(nc, random_state=0).fit(ctx)
    return p.transform(ctx).astype(np.float32), [p.transform(o).astype(np.float32) for o in others], nc

def load_data(spec, tiny, dev):
    """-> D {ds: (Xtrain, ytrain, Xtest, ytest)} and enc(texts). tiny: cached full-split embeddings (no embedding work); pod: embed only used labels."""
    use = {t["ds"] for t in spec["tasks"].values()}; src = {"b77": load_banking77, "clinc": load_clinc150}
    if tiny: enc = embed
    else:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("BAAI/bge-small-en-v1.5", device=dev)
        enc = lambda T: m.encode(list(T), batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    D = {}
    for ds in use:
        d = src[ds](); need = sorted({l for t in spec["tasks"].values() if t["ds"] == ds for l in t["labels"]}); out = []
        for s in ("train", "test"):
            txt, y = d[s]
            if not tiny: keep = np.flatnonzero(np.isin(y, need)); txt, y = [txt[i] for i in keep], y[keep]
            out += [enc(txt), y]
        D[ds] = tuple(out); print(ds, [x.shape for x in out], flush=True)
    return D, enc

# ---------------------------------------------------------------- baselines (CPU, sklearn)
def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def baselines(tasks, SE, SL, seeds, ks, R):
    from sklearn.linear_model import LogisticRegression
    for s in range(seeds):
        for k in ks:
            for tn, t in tasks.items():
                sup, cal = draw(t, k, s); X, y = t["Xs"][sup], t["ys"][sup]; Q = np.concatenate([t["Xs"][cal], t["Xq"]]); yq = np.concatenate([t["ys"][cal], t["yq"]])
                idx = {l: j for j, l in enumerate(t["names"])}; sel = [j for j, l in enumerate(SL) if l in idx]
                Xsy = np.concatenate([SE[sel], X]); ysy = np.concatenate([[idx[SL[j]] for j in sel], y]); outs = {}
                outs["LR-384"] = LogisticRegression(C=10, max_iter=1000).fit(X, y).predict_proba(Q)
                outs["proto"] = softmax(10 * Q @ np.stack([X[y == c].mean(0) for c in range(t["K"])]).T)
                outs["SYNTH+k"] = LogisticRegression(C=10, max_iter=2000).fit(Xsy, ysy).predict_proba(Q)
                for dim in (32, 64):  # control: the same PCA features the ICL heads see, fed to LR (rescaled to unit mean row norm)
                    A, (Qp,), nc = pca_feats(X, [Q], dim); sc = 1 / np.linalg.norm(A, axis=1).mean()
                    outs[f"LR-pca{dim}"] = LogisticRegression(C=10, max_iter=1000).fit(A * sc, y).predict_proba(Qp * sc)
                for m_, Pr in outs.items():
                    ok = Pr.argmax(1) == yq; c = top2(Pr); R[(m_, tn, k, s)] = dict(okc=ok[:NCAL], cc=c[:NCAL].astype(np.float32), okt=ok[NCAL:], ct=c[NCAL:].astype(np.float32))
        print(f"baselines seed {s} done", flush=True)
    for s in range(seeds):  # day-zero reference (k=0): LR on the synthetic examples only, scored on the same calibration set as seed s
        for tn, t in tasks.items():
            idx = {l: j for j, l in enumerate(t["names"])}; sel = [j for j, l in enumerate(SL) if l in idx]
            clf = LogisticRegression(C=10, max_iter=2000).fit(SE[sel], [idx[SL[j]] for j in sel]); _, cal = draw(t, 1, s)
            Q = np.concatenate([t["Xs"][cal], t["Xq"]]); yq = np.concatenate([t["ys"][cal], t["yq"]]); Pr = clf.predict_proba(Q); ok = Pr.argmax(1) == yq; c = top2(Pr)
            R[("SYNTH k=0", tn, 0, s)] = dict(okc=ok[:NCAL], cc=c[:NCAL].astype(np.float32), okt=ok[NCAL:], ct=c[NCAL:].astype(np.float32))

# ---------------------------------------------------------------- ICL heads
def make_head(name, dev, ck):
    if name == "tabicl":
        from tabicl import TabICLClassifier
        return TabICLClassifier(model_path=ck["icl"], allow_auto_download=True, device=dev, random_state=0)  # v2 default; support_many_classes=True
    from tabpfn import TabPFNClassifier
    from tabpfn_extensions.many_class import ManyClassClassifier  # TabPFN v2 caps at 10 classes; code-book wrapper for 12/15/77
    return ManyClassClassifier(TabPFNClassifier(model_path=ck["pfn"], device=dev, n_estimators=4, random_state=0), alphabet_size=10, random_state=0)

def sync():
    import torch
    if torch.cuda.is_available(): torch.cuda.synchronize()

def run_head(clf, A, y, Q):
    t = time.time(); clf.fit(A, y); sync(); tf = time.time() - t
    t = time.time(); P = clf.predict_proba(Q); sync(); tp = time.time() - t
    return P, tf, tp

def run_job(t, k, s, name, dim, ctx):
    sup, cal = draw(t, k, s); X, y = t["Xs"][sup], t["ys"][sup]
    if name.endswith("_synth"):
        idx = {l: j for j, l in enumerate(t["names"])}; sel = [j for j, l in enumerate(ctx["SL"]) if l in idx]
        X = np.concatenate([ctx["SE"][sel], X]); y = np.concatenate([[idx[ctx["SL"][j]] for j in sel], y])
    A, (Qc, Qt), nc = pca_feats(X, [t["Xs"][cal], t["Xq"]], dim)
    P, tf, tp = run_head(make_head(name.replace("_synth", ""), ctx["dev"], ctx["ck"]), A, y, np.concatenate([Qc, Qt]))
    res = dict(okc=P[:NCAL].argmax(1) == t["ys"][cal], cc=top2(P[:NCAL]).astype(np.float32), okt=P[NCAL:].argmax(1) == t["yq"], ct=top2(P[NCAL:]).astype(np.float32))
    return dict(method=f"{name}-{dim}", task=None, k=k, seed=s, nctx=len(y), ncomp=nc, nq=len(P), fit_s=round(tf, 3), pred_s=round(tp, 3), acc=float(res["okt"].mean())), res

# ---------------------------------------------------------------- IO
def dump(R, meta, extra, out):
    arrs = {f"{m}|{tn}|{k}|{s}|{f}": a for (m, tn, k, s), v in R.items() for f, a in v.items()}
    np.savez_compressed(out / "results_tmp.npz", **arrs); os.replace(out / "results_tmp.npz", out / "results.npz")
    (out / "meta.json").write_text(json.dumps(dict(meta=meta, extra=extra)), encoding="utf-8")

def load(out):
    z = np.load(out / "results.npz"); R = {}
    for key in z.files:
        m, tn, k, s, f = key.split("|"); R.setdefault((m, tn, int(k), int(s)), {})[f] = z[key]
    return R, json.loads((out / "meta.json").read_text(encoding="utf-8"))

# ---------------------------------------------------------------- report
def metrics(r):
    cert = certify(r["cc"], r["okc"], BUDGET); h = apply_certificate(r["ct"], r["okt"], cert["cutoff"])
    return dict(acc=float(r["okt"].mean()), cert=cert["coverage"] > 0, cov=h["coverage"], err=h["error"], n_above=h["n_above"])

def summarize(R):
    S = {}
    for g, f in GROUPS.items():
        for m in sorted({m for m, *_ in R}):
            for k in sorted({k for mm, tn, k, s in R if mm == m and f(tn)}):
                ms = [metrics(v) for (mm, tn, kk, s), v in R.items() if mm == m and kk == k and f(tn)]
                above = sum(x["n_above"] for x in ms if x["err"] is not None); errs = sum(x["err"] * x["n_above"] for x in ms if x["err"] is not None)
                S[(g, m, k)] = dict(n=len(ms), acc=np.mean([x["acc"] for x in ms]), cov=np.mean([x["cov"] for x in ms]), frac_cert=np.mean([x["cert"] for x in ms]),
                                    err=(errs / above if above else None), viol=sum(1 for x in ms if x["err"] is not None and x["err"] > BUDGET), ncert=sum(1 for x in ms if x["err"] is not None))
    return S

def paired(R, a, b, k, f):
    d = [R[(a, tn, kk, s)]["okt"].mean() - R[(b, tn, kk, s)]["okt"].mean() for (aa, tn, kk, s) in R if aa == a and kk == k and f(tn) and (b, tn, kk, s) in R]
    return (float(np.mean(d)), float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan"), len(d)) if d else None

def report(R, meta, extra):
    L = []; say = lambda s="": (print(s, flush=True), L.append(s))
    S = summarize(R); ref = ["SYNTH k=0", "proto", "LR-384", "LR-pca32", "LR-pca64", "SYNTH+k"]; ms = sorted({m for m, *_ in R})
    order = [m for m in ref if m in ms] + [m for m in ms if m not in ref]; kk = [0] + KS; hdr = "| method | " + " | ".join(f"k={k}" for k in kk) + " |\n|---|" + "---|" * len(kk)
    for title, key, fmt in (("Accuracy (mean over draws; (n=..) marks cells with fewer draws than the group maximum)", "acc", "{:.3f}"),
                            (f"Certified coverage at a {BUDGET:.0%} budget, applied to the test split (mean over draws; 0 where nothing certifies)", "cov", "{:.2f}"),
                            ("Share of draws where the 300-row calibration set certifies anything", "frac_cert", "{:.2f}")):
        say(f"\n## {title}")
        for g in GROUPS:
            nmax = {k: max([S[(g, m, k)]["n"] for m in order if (g, m, k) in S] or [0]) for k in kk}
            say(f"\n### {g}\n\n{hdr}")
            for m in order:
                row = [S.get((g, m, k)) for k in kk]
                if any(row): say(f"| {m} | " + " | ".join((fmt.format(r[key]) + ("" if r["n"] == nmax[k] else f" (n={r['n']})")) if r else "-" for r, k in zip(row, kk)) + " |")
    say(f"\n## Realized error above the certified cutoff, pooled over draws that certified (violations = draws with error > {BUDGET:.0%}): err (violations/certified draws)")
    for g in GROUPS:
        say(f"\n### {g}\n\n{hdr}")
        for m in order:
            row = [S.get((g, m, k)) for k in kk]
            if any(row): say(f"| {m} | " + " | ".join((f"{r['err']:.3f} ({r['viol']}/{r['ncert']})" if r and r["err"] is not None else "-") for r in row) + " |")
    for ref_m, subset in (("LR-384", None), ("SYNTH+k", ["tabicl_synth-64"])):
        say(f"\n## Paired accuracy difference vs {ref_m} (mean +- SE over matched task x seed draws)")
        for g, f in GROUPS.items():
            say(f"\n### {g}\n\n| method | " + " | ".join(f"k={k}" for k in KS) + " |\n|---|" + "---|" * len(KS))
            for m in (subset or order):
                if m in (ref_m, "SYNTH k=0"): continue
                row = [paired(R, m, ref_m, k, f) for k in KS]
                if any(row): say(f"| {m} | " + " | ".join(f"{r[0]:+.3f} +- {r[1]:.3f} (n={r[2]})" if r else "-" for r, k in zip(row, KS)) + " |")
    ok = [m for m in meta if "error" not in m]
    say("\n## Inference time on the pod GPU, seconds per 1000 query rows (median over draws; predict only, fit excluded; ctx = median context rows)")
    for g, f in GROUPS.items():
        say(f"\n### {g}\n\n| head | " + " | ".join(f"k={k}" for k in KS) + " |\n|---|" + "---|" * len(KS))
        for arm in sorted({m["method"] for m in ok}):
            row = []
            for k in KS:
                v = [m for m in ok if m["method"] == arm and m["k"] == k and f(m["task"])]
                row.append(f"{np.median([m['pred_s'] / m['nq'] * 1000 for m in v]):.2f} (ctx {int(np.median([m['nctx'] for m in v]))})" if v else "-")
            say(f"| {arm} | " + " | ".join(row) + " |")
    if extra.get("cpu"):
        say("\n## Inference time on the pod CPU (4 threads), 100 query rows, k=5, PCA 64")
        say("\n| head | task | ctx rows | fit s | predict s | s per 1000 rows |\n|---|---|---|---|---|---|")
        for c in extra["cpu"]: say(f"| {c['head']} | {c['task']} | {c['nctx']} | {c['fit_s']} | {c['pred_s']} | {c['s_per_1000']} |")
    if extra.get("repeat"):
        say("\n## Same-config repeat (noise floor of the ICL heads themselves: same draw, same seed, run twice)")
        say("\n| method | task | k | seed | acc run 1 | acc run 2 | test rows whose prediction flipped | max abs change in top-2 gap |\n|---|---|---|---|---|---|---|---|")
        for c in extra["repeat"]: say(f"| {c['method']} | {c['task']} | {c['k']} | {c['seed']} | {c['acc1']:.4f} | {c['acc2']:.4f} | {c['flips']} / {c['n']} | {c['max_dconf']:.2e} |")
    say(f"\nICL jobs {len(ok)}, errors {len(meta) - len(ok)}, total fit s {sum(m['fit_s'] for m in ok):.0f}, total predict s {sum(m['pred_s'] for m in ok):.0f}")
    errs = [m for m in meta if "error" in m]
    for m in errs[:8]: say(f"ERROR {m['method']} {m['task']} k={m['k']} seed={m['seed']}: {m['error']}")
    say("PCA width actually used (ncomp) at dim 64 by k: " + str({k: sorted({m['ncomp'] for m in ok if m['method'].endswith('-64') and m['k'] == k}) for k in KS}))
    return L

# ---------------------------------------------------------------- main
def main(a):
    global NCAL, QCAP
    T0 = time.time()
    if a.tiny: NCAL, QCAP = 60, 40
    import torch
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    dev = "cpu" if a.tiny or not torch.cuda.is_available() else "cuda"; print("device", dev, flush=True)
    out = HERE / "tiny" if a.tiny else HERE; out.mkdir(exist_ok=True)
    if a.tiny:
        torch.set_num_threads(4); D0 = HERE.parents[0] / ".data" / "s20_tabicl"
        ck = dict(icl=str(D0 / "tabicl" / "tabicl-classifier-v2-20260212.ckpt"), pfn=str(D0 / "tabpfn_v2" / "tabpfn-v2-classifier.ckpt"))
    else:
        from huggingface_hub import hf_hub_download
        ck = dict(icl=None, pfn=hf_hub_download("Prior-Labs/TabPFN-v2-clf", "tabpfn-v2-classifier.ckpt", local_dir="w_pfn"))
    spec = make_spec(); print("held-out CLINC domains", spec["held"], flush=True)
    if a.tasks: spec["tasks"] = {n: t for n, t in spec["tasks"].items() if n in a.tasks.split(",")}
    D, enc = load_data(spec, a.tiny, dev)
    synth = json.load(open(CACHE / "synth_examples.json", encoding="utf-8")); labels = {l for t in spec["tasks"].values() for l in t["names"]}
    flat = [(l, s) for l, v in synth.items() if l in labels for s in v]; SE = enc([s for _, s in flat]); SL = [l for l, _ in flat]
    tasks = {n: build_task(st, D) for n, st in spec["tasks"].items()}
    for n, t in tasks.items(): print(n, "train rows", len(t["ys"]), "test rows", len(t["yq"]), "K", t["K"], flush=True)
    ks = [int(x) for x in a.ks.split(",")]; arms = [(x.split(":")[0], int(x.split(":")[1])) for x in a.arms.split(",")]
    R, meta, extra = {}, [], {}; t1 = time.time(); baselines(tasks, SE, SL, a.seeds, ks, R); print(f"baselines {time.time()-t1:.0f}s", flush=True)
    dump(R, meta, extra, out)
    ctx = dict(dev=dev, ck=ck, SE=SE, SL=SL)
    jobs = [(s, k, tn, arm) for s in range(a.seeds) for k in ks for tn in tasks for arm in arms]
    jobs.sort(key=lambda j: (tasks[j[2]]["K"] == 77, tasks[j[2]]["K"] == 77 and j[3][0] == "tabpfn", j[0], j[1]))  # Banking77 last, its TabPFN jobs very last
    done, last = [], time.time()
    for i, (s, k, tn, (name, dim)) in enumerate(jobs):
        if time.time() - T0 > a.deadline_min * 60: print("deadline reached, stopping", flush=True); break
        try:
            rec, res = run_job(tasks[tn], k, s, name, dim, ctx); rec["task"] = tn; R[(rec["method"], tn, k, s)] = res; done.append(jobs[i])
        except Exception as e:
            import traceback; traceback.print_exc(); rec = dict(method=f"{name}-{dim}", task=tn, k=k, seed=s, error=repr(e)[:300])
            if dev == "cuda": torch.cuda.empty_cache()
        meta.append(rec); print(f"[{i+1}/{len(jobs)} {time.time()-T0:.0f}s]", {x: rec[x] for x in rec if x != "nq"}, flush=True)
        if time.time() - last > 120: dump(R, meta, extra, out); last = time.time()
    dump(R, meta, extra, out)
    # same-config repeat: first tabicl and first tabpfn job at k=5 (else first), run again, compare
    extra["repeat"] = []
    for fam in ("tabicl", "tabpfn"):
        c = [j for j in done if j[3][0] == fam]; c = [j for j in c if j[1] == 5] or c
        if not c: continue
        s, k, tn, (name, dim) = c[0]; rec, res = run_job(tasks[tn], k, s, name, dim, ctx); old = R[(rec["method"], tn, k, s)]
        extra["repeat"].append(dict(method=rec["method"], task=tn, k=k, seed=s, acc1=float(old["okt"].mean()), acc2=float(res["okt"].mean()), flips=int((old["okt"] != res["okt"]).sum()),
                                    n=len(res["okt"]), max_dconf=float(np.abs(old["ct"] - res["ct"]).max())))
    if not a.tiny and dev == "cuda" and time.time() - T0 < (a.deadline_min + 5) * 60:  # CPU inference time on the pod (4 threads), small samples
        torch.set_num_threads(4); extra["cpu"] = []; ctx["dev"] = "cpu"
        for tn, name in [("b12_0", "tabicl"), ("b12_0", "tabpfn"), ("clinc_meta", "tabicl"), ("banking77", "tabicl")]:
            if tn not in tasks or time.time() - T0 > (a.deadline_min + 9) * 60: continue
            try:
                t = tasks[tn]; sup, _ = draw(t, 5, 0); A, (Q,), nc = pca_feats(t["Xs"][sup], [t["Xq"][:100]], 64)
                P, tf, tp = run_head(make_head(name, "cpu", ck), A, t["ys"][sup], Q)
                extra["cpu"].append(dict(head=name, task=tn, nctx=len(sup), fit_s=round(tf, 2), pred_s=round(tp, 2), s_per_1000=round(tp / len(Q) * 1000, 1)))
            except Exception as e: print("cpu timing failed", tn, name, repr(e)[:200], flush=True)
        ctx["dev"] = dev
    dump(R, meta, extra, out); L = report(R, meta, extra)
    (out / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print(f"total {time.time()-T0:.0f}s", flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("mode", nargs="?", default="run", choices=["run", "report"]); p.add_argument("--tiny", action="store_true")
    p.add_argument("--seeds", type=int, default=5); p.add_argument("--ks", default=",".join(map(str, KS))); p.add_argument("--arms", default=ARMS)
    p.add_argument("--tasks", default=""); p.add_argument("--deadline-min", type=float, default=55)
    a = p.parse_args()
    if a.mode == "report": R, d = load(HERE); print("\n".join(report(R, d["meta"], d["extra"])))
    else:
        if a.tiny: a.seeds, a.ks, a.arms, a.tasks = 1, "1,3", "tabicl:32,tabpfn:32,tabicl_synth:32", "b12_0"
        main(a)
