"""S25 orchestrator (one pod launch): python smokes/s25_exotic/run.py [--tiny] [--deadline-min 32]
1. CUDA check (3 tries; none -> exit 2 so the pod can be relaunched on another GPU type). 2. pip install snntorch, tmu in the background (each isolated, 5 min cap) while the five tasks
are built once (lib.build) and pickled. 3. Five arm groups run as parallel processes (fast numpy arms, gzip NCD, Tsetlin, torch CPU heads, GPU arms JEPA + SAE); each writes rows_<group>.jsonl
after every (arm, task, k, draw). 4. In the last 4 minutes a quiet single-thread timing pass (`bench`). results.json + tables.md are rewritten every 15 s."""
import argparse, os, subprocess, sys, threading
from lib import *
import report

GROUP_NAMES = ["gpu", "fast", "ncd", "torch", "tm"]   # gpu first: it has the longest pole
def wait(p, name, until):
    try: p.wait(timeout=max(1, until - time.time()))
    except subprocess.TimeoutExpired:
        log(name, "past deadline, terminating"); p.terminate()
        try: p.wait(timeout=20)
        except subprocess.TimeoutExpired: p.kill()
    return p.returncode

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tiny", action="store_true"); ap.add_argument("--deadline-min", type=float, default=32); a = ap.parse_args()
    T0 = time.time(); END = T0 + a.deadline_min * 60; py = sys.executable; fl = ["--tiny"] if a.tiny else []; timing = {}; pipres = {}
    for f in list(HERE.glob("rows_*.jsonl")) + list(HERE.glob("explain_*.json")) + [HERE / n for n in ("results.json", "tables.md", "timing.json")]: f.unlink(missing_ok=True)
    if not a.tiny:
        for i in range(3):   # torch caches a failed CUDA init, so retry in fresh processes
            q = subprocess.run([py, "-c", "import torch; print(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else exit(1)"], capture_output=True, text=True)
            if q.returncode == 0: log("cuda ok:", q.stdout.strip(), "cpus", os.cpu_count()); break
            log(f"NO CUDA (try {i+1}/3): {q.stderr.strip()[-200:]}"); time.sleep(20)
        else: log("NO CUDA on this pod: aborting (relaunch on another GPU type)"); sys.exit(2)
        def pip():
            for pkg in ("snntorch", "tmu"):
                t0 = time.time()
                try: r = subprocess.run([py, "-m", "pip", "install", "-q", pkg], capture_output=True, text=True, timeout=300); rc, err = r.returncode, r.stderr.strip()[-200:]
                except Exception as e: rc, err = -1, repr(e)[:200]
                pipres[pkg] = rc; log(f"pip install {pkg}: rc={rc} {time.time()-t0:.0f}s {err if rc else ''}")
        th = threading.Thread(target=pip, daemon=True); th.start()
    else: th = None
    t0 = time.time(); T = load_tasks() if (a.tiny and DATA.exists()) else build(a.tiny); timing["build_secs"] = time.time() - t0
    log("tasks (name, pool, calibration, test):", [(x["name"], len(x["ts"]), len(x["tc"]), len(x["tq"])) for x in T], f"{timing['build_secs']:.0f}s")
    stop = threading.Event()
    def agg():
        while not stop.is_set():
            try: report.dump()
            except Exception as e: log("dump failed", repr(e)[:100])
            stop.wait(15)
    threading.Thread(target=agg, daemon=True).start()
    reserve = 0.7 if a.tiny else 4.0; env = dict(os.environ, S25_JEPA_STEPS="40" if a.tiny else "2000", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    def launch(g): return subprocess.Popen([py, "run_arm.py", g, *fl, "--deadline-min", str((END - time.time()) / 60 - reserve)], cwd=HERE, env=env)
    procs = {g: launch(g) for g in GROUP_NAMES if g not in ("torch", "tm")}
    if th: th.join(timeout=330)   # snntorch / tmu installs finish while the first groups run
    for g in ("torch", "tm"): procs[g] = launch(g)
    for g, p in procs.items(): wait(p, g, END - reserve * 60); timing[f"{g}_done_secs"] = time.time() - T0; log(g, "rc", p.returncode)
    if END - time.time() > 60:
        b = subprocess.Popen([py, "run_arm.py", "bench", *fl], cwd=HERE, env=env); wait(b, "bench", END); log("bench rc", b.returncode)
    timing["total_secs"] = time.time() - T0; timing["pip"] = pipres; (HERE / "timing.json").write_text(json.dumps(timing, indent=1)); stop.set()
    report.dump(); t = report.tables(); (HERE / "tables.md").write_text(t + "\n", encoding="utf-8"); print(t); log("timing", timing)
