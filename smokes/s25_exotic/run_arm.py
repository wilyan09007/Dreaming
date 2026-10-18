"""S25 runner. python run_arm.py GROUP [--tiny] [--deadline-min N]   (GROUP in fast|ncd|tm|torch|gpu), or `bench` for the single-thread timing pass.
Loop order is draw-major (all tasks and k for draw 0 first) so a timeout leaves whole draws; rows are appended to rows_<group>.jsonl as they are produced.
Per (arm, task) at draw 0 and k=3 the fit+predict is repeated from scratch (features and any pretraining redone, same seed) and compared bit for bit (`det`)."""
import argparse, re, traceback
from lib import *
import arms

def slug(x): return re.sub(r"\W+", "_", x)[:48]

def same(a, b): return bool(np.array_equal(a[0], b[0]) and np.asarray(a[1]).tobytes() == np.asarray(b[1]).tobytes())

def run(group, tiny, deadline):
    out = HERE / f"rows_{group}.jsonl"; out.unlink(missing_ok=True); T = load_tasks(); ks, draws = ([1, 3], 1) if tiny else (KS, DRAWS)
    for label, mk in arms.GROUPS[group]:
        try: arm = mk(); assert arm.name == label, (arm.name, label)
        except Exception as e: emit(out, method=label, status="skipped", reason=f"{type(e).__name__}: {str(e)[:300]}"); log("SKIP", label, repr(e)[:200]); continue
        prepped, prep_s, errs, explained = set(), {}, 0, {}
        for dr in range(draws):
            for t in T:
                K = len(t["labels"])
                for k in ks:
                    if time.time() > deadline: log("deadline reached in", group); return
                    if errs >= 4: break
                    try:
                        if t["name"] not in prepped: t0 = time.time(); arm.prep(t); prep_s[t["name"]] = time.time() - t0; prepped.add(t["name"])
                        L = draw(t["ys"], k, dr, K); t0 = time.time(); m = arm.fit(t, L, dr); fit_s = time.time() - t0
                        t0 = time.time(); rq = arm.predict(m, t, "q"); pq_s = time.time() - t0; rc = arm.predict(m, t, "c"); det = {}
                        if dr == 0 and k == 3:   # determinism: redo everything with the same seed
                            arm.clear(t); arm.prep(t); m2 = arm.fit(t, L, dr); r2 = arm.predict(m2, t, "q")
                            det = {v: same(rq[v], r2[v]) for v in rq}; detp = {v: bool(np.array_equal(rq[v][0], r2[v][0])) for v in rq}
                        for v, (pq, cq) in rq.items():
                            pc, cc = rc[v]; emit(out, method=label + (f" [{v}]" if v else ""), group=group, task=t["name"], family=t["family"], K=K, draw=dr, k=k, params=arm.params(m), fit_secs=fit_s,
                                                 pred_ms_per_1000=pq_s / len(t["yq"]) * 1000, prep_secs=prep_s.get(t["name"]), det=det.get(v), det_pred=detp.get(v) if det else None,
                                                 **score_pc(pq, cq, pc, cc, t["yq"], t["yc"]))
                        if dr == 0 and k == max(ks) and t["name"] not in explained:
                            explained[t["name"]] = 1; ex = arm.explain(m, t)
                            if ex is not None:
                                f = HERE / f"explain_{slug(label)}.json"; cur = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}; cur[t["name"]] = ex; f.write_text(json.dumps(cur, indent=1, default=float), encoding="utf-8")
                        errs = 0
                    except Exception as e:
                        errs += 1; emit(out, method=label, group=group, task=t["name"], draw=dr, k=k, status="error", reason=f"{type(e).__name__}: {str(e)[:200]} | {traceback.format_exc()[-300:]}"); log("ERROR", label, t["name"], k, repr(e)[:200])
            log(f"{group}: {label} draw {dr} done")
        log(f"{group}: {label} finished")

def bench(tiny):
    """Single-thread timing at task 0, draw 0, k=3, nothing else running: min of 3 fits; predict time = featurize (cache cleared) + predict on the test split."""
    out = HERE / "rows_bench.jsonl"; out.unlink(missing_ok=True); t = load_tasks()[0]; L = draw(t["ys"], 3, 0, len(t["labels"]))
    for g, items in arms.GROUPS.items():
        for label, mk in items:
            try:
                arm = mk(); t0 = time.time(); arm.prep(t); prep = time.time() - t0; fits, prs = [], []
                for _ in range(3): t0 = time.time(); m = arm.fit(t, L, 0); fits.append(time.time() - t0)
                for _ in range(3): arm.cache.pop((t["name"], "q"), None); t0 = time.time(); arm.predict(m, t, "q"); prs.append(time.time() - t0)
                emit(out, method=label, group=g, params=arm.params(m), fit_secs=min(fits), pred_ms_per_1000=min(prs) / len(t["yq"]) * 1000, prep_secs=prep, device="cuda" if g == "gpu" else "cpu 1 thread")
                log("bench", label, f"fit {min(fits):.3f}s predict {min(prs)/len(t['yq'])*1000:.1f} ms/1000")
            except Exception as e: emit(out, method=label, group=g, status="error", reason=repr(e)[:200]); log("bench error", label, repr(e)[:200])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("group"); ap.add_argument("--tiny", action="store_true"); ap.add_argument("--deadline-min", type=float, default=30); a = ap.parse_args()
    import torch; torch.use_deterministic_algorithms(True, warn_only=True); torch.backends.cudnn.deterministic = True; torch.set_num_threads(1)
    if a.group == "bench": bench(a.tiny)
    else: run(a.group, a.tiny, time.time() + a.deadline_min * 60)
