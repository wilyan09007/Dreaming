"""S22-lite orchestrator (the pod command). Checks CUDA, starts worker 0 alone (uncontended calibration), then the other workers,
merges their per-record jsonl files into results.json every 30 s, and stops handing out cells after --budget-min.
Resume: records of cells that already have all 5 arm rows in results.json are kept and skipped (pools are rebuilt).
Usage on the pod: python smokes/s22_architectures/run.py        Local plumbing check: python smokes/s22_architectures/run.py --tiny --workers 1"""
import argparse, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--tiny", action="store_true"); p.add_argument("--workers", type=int, default=3); p.add_argument("--budget-min", type=float, default=32.0)
p.add_argument("--ks", default="3,1,10"); p.add_argument("--seeds", default="0,1,2")
a = p.parse_args()
OUT = HERE / "tiny_out" if a.tiny else HERE
SCR = Path(tempfile.gettempdir()) / ("s22_tiny" if a.tiny else "s22_work")  # pools (~100 MB) stay out of the folder that pod.py tars back
t0 = time.time()
def log(*x): print(f"[orch +{time.time()-t0:.0f}s]", *x, flush=True)

if not a.tiny:
    import torch
    log("nvidia-smi:", subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip(), "| cpus", os.cpu_count(), "| torch", torch.__version__)
    if not torch.cuda.is_available(): log("NO CUDA on this pod: aborting"); sys.exit(3)

ARMS = {"F", "B", "D_comp", "D_comp_ft", "A"}
OUT.mkdir(exist_ok=True); RES = OUT / "results.json"; prior = json.load(open(RES)) if RES.exists() else {}
cells = {}
for r in prior.get("records", []): cells.setdefault((r["task"], r["seed"], r["k"]), set()).add(r["arm"])
done = [list(c) for c, arms in cells.items() if arms >= ARMS]
prior_recs = [r for r in prior.get("records", []) if [r["task"], r["seed"], r["k"]] in done]; prior_pools = prior.get("pools", {})
shutil.rmtree(SCR, ignore_errors=True); (SCR / "claims").mkdir(parents=True); (SCR / "pools").mkdir()
shutil.rmtree(OUT / "rec", ignore_errors=True); json.dump(done, open(SCR / "done_cells.json", "w"))
log("start; resumed cells:", len(done))

def load_jsonl(pattern):
    out = []
    for f in sorted((OUT / "rec").glob(pattern)):
        for line in f.read_text(encoding="utf-8").splitlines():
            try: out.append(json.loads(line))
            except Exception: pass  # a line being written
    return out
def merge(final=False):
    recs = prior_recs + load_jsonl("rec_w*.jsonl"); errs = load_jsonl("err_w*.jsonl"); pools = dict(prior_pools)
    for f in sorted((OUT / "rec").glob("pool_*.json")): pools[f.stem[5:]] = json.load(open(f))
    calib = json.load(open(OUT / "calib.json")) if (OUT / "calib.json").exists() else prior.get("calib")
    c = {}
    for r in recs: c.setdefault((r["task"], r["seed"], r["k"]), set()).add(r["arm"])
    full = sorted([list(x) for x, s in c.items() if s >= ARMS], key=lambda x: (x[2], x[1], x[0]))
    res = dict(args=vars(a), calib=calib, pools=pools, records=recs, errors=errs, complete_cells=full, wall_s=round(time.time() - t0), final=final)
    json.dump(res, open(OUT / "results.json.tmp", "w")); os.replace(OUT / "results.json.tmp", RES); return res

common = ["--workers", str(a.workers), "--t0", str(t0), "--budget-min", str(a.budget_min), "--scr", str(SCR), "--out", str(OUT), "--ks", a.ks, "--seeds", a.seeds] + (["--tiny"] if a.tiny else [])
env = dict(os.environ, OMP_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
spawn = lambda i: subprocess.Popen([sys.executable, "-u", str(HERE / "worker.py"), "--i", str(i)] + common, env=env)
procs = [spawn(0)]
while not (SCR / "calib.done").exists():  # worker 0 runs alone until its solo calibration is written
    if procs[0].poll() is not None: log("worker 0 exited before calibration, rc =", procs[0].returncode); merge(True); sys.exit(1)
    if time.time() - t0 > 900: log("calibration wait timed out, starting the other workers anyway"); break
    time.sleep(2)
procs += [spawn(i) for i in range(1, a.workers)]
last = time.time()
while any(q.poll() is None for q in procs):
    time.sleep(5)
    if time.time() - last > 30: res = merge(); last = time.time(); log("cells complete:", len(res["complete_cells"]), "records:", len(res["records"]))
    if time.time() - t0 > a.budget_min * 60 + 12 * 60: log("hard timeout: killing workers"); [q.kill() for q in procs]; break
res = merge(True); log("worker exit codes", [q.returncode for q in procs], "| complete cells", len(res["complete_cells"]), "| records", len(res["records"]), "| errors", len(res["errors"]))
sys.exit(0 if res["complete_cells"] else 1)
