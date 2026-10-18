"""S23-lite orchestrator (one pod launch). Build tasks + embeddings once, then run concurrently:
  arm12.py (CPU, niced: baseline, medoids, semi-supervised probes), tsdae.py (GPU, small), paw.py (GPU: compile + zero-shot + fine-tune cells until the deadline).
TSDAE overlaps PAW's weight download / pseudo-program generation so the GPU is not idle; fine-tune cells start after TSDAE has normally finished.
results.json is rewritten every few seconds from the per-arm rows_*.jsonl files. Usage: python smokes/s23_new_architectures/run.py [--tiny] [--deadline-min 38]"""
import argparse, subprocess, threading, sys
from lib import *
import report

def wait(p, name, until):
    try: p.wait(timeout=max(1, until - time.time()))
    except subprocess.TimeoutExpired:
        log(name, "past deadline, terminating"); p.terminate()
        try: p.wait(timeout=20)
        except subprocess.TimeoutExpired: p.kill()
    return p.returncode

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tiny", action="store_true"); ap.add_argument("--deadline-min", type=float, default=38); a = ap.parse_args()
    T0 = time.time(); END = T0 + a.deadline_min * 60; py = sys.executable; fl = ["--tiny"] if a.tiny else []; timing = {}
    for f in list(HERE.glob("rows_*.jsonl")) + [HERE / n for n in ("results.json", "timing.json", "paw_status.json", "pseudo_programs.jsonl", "paw_sdk_install.log")]:
        if f.exists(): f.unlink()  # stale outputs of earlier (tiny) runs
    gpu = True
    if not a.tiny:  # torch caches a failed CUDA init, so retry in fresh processes (an earlier pod had nvidia-smi but "CUDA unknown error")
        for i in range(3):
            q = subprocess.run([py, "-c", "import torch; print(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else exit(1)"], capture_output=True, text=True)
            if q.returncode == 0: log("cuda ok:", q.stdout.strip(), "cpus", os.cpu_count()); break
            log(f"NO CUDA (try {i+1}/3): {q.stderr.strip()[-200:]}"); time.sleep(20)
        else:
            gpu = False; log("NO CUDA on this pod: GPU arms (TSDAE, PAW) skipped, CPU arms still run; this run exits non-zero")
    t = time.time(); T = build(a.tiny); timing["build_secs"] = time.time() - t
    log("tasks (name, pool rows, calibration rows, test rows):", [(x["name"], len(x["ts"]), len(x["tc"]), len(x["tq"])) for x in T], f"{timing['build_secs']:.0f}s")
    stop = threading.Event()
    def agg():
        while not stop.is_set():
            try: report.dump()
            except Exception as e: log("dump failed", repr(e)[:100])
            stop.wait(10)
    threading.Thread(target=agg, daemon=True).start()
    def mins(): return max(1.0, (END - time.time()) / 60 - 1.5)
    t_cpu = time.time(); cpu = subprocess.Popen([py, "arm12.py", *fl, "--deadline-min", str(mins())], cwd=HERE)
    t_ts = time.time(); ts = subprocess.Popen([py, "tsdae.py", *fl, "--deadline-min", str(min(14.0, mins()))], cwd=HERE) if gpu else None
    t_pw = time.time(); pw = subprocess.Popen([py, "paw.py", *(["--selftest"] if a.tiny else []), "--deadline-min", str(mins())], cwd=HERE) if gpu else None
    if ts: r = wait(ts, "tsdae", END); timing["tsdae_secs"] = time.time() - t_ts; log("tsdae rc", r)
    sdk = None
    if not a.tiny:  # record whether the public SDK installs (only the package itself: --no-deps keeps the pod's torch/transformers untouched)
        sdk = subprocess.Popen("nice -n 19 timeout 600 pip install --no-deps --target /tmp/pawsdk programasweights --extra-index-url https://pypi.programasweights.com/simple/ > paw_sdk_install.log 2>&1; echo rc=$? >> paw_sdk_install.log; "
                               "cat /tmp/pawsdk/programasweights*/METADATA 2>/dev/null | grep -i -E '^(Version|License|Requires-Dist)' >> paw_sdk_install.log", shell=True, cwd=HERE)
    if pw: r = wait(pw, "paw", END); timing["paw_secs"] = time.time() - t_pw; log("paw rc", r)
    r = wait(cpu, "arm12", END); timing["arm12_secs"] = time.time() - t_cpu; log("arm12 rc", r)
    if sdk is not None: wait(sdk, "sdk install", time.time() + 60)
    timing["total_secs"] = time.time() - T0; (HERE / "timing.json").write_text(json.dumps(timing, indent=1)); stop.set(); report.dump(); log("timing", timing)
    print(report.tables()); sys.exit(0 if gpu else 2)
