"""Run any smoke on a RunPod GPU pod instead of the laptop, with no SSH and no S3.
The pod boots a tiny HTTP server (inline script); we PUT a tar.gz bundle of the project files through RunPod's HTTP proxy,
the pod runs the command, serves live log and a results tar, and terminates itself (we also terminate it).
Usage (from the project root):
  python smokes/pod.py --tag s14 --bundle smokes/common.py smokes/s14_tool_traces smokes/.cache/label_descriptions.json \
      --cmd "cd smokes/s14_tool_traces && python run.py" [--pip "tabicl confseq"] [--max-min 60] [--gpu "NVIDIA GeForce RTX 4090"] [--cpu]
Results: everything under the bundled smoke folders that changed on the pod is tarred back and extracted over the local copy
(RESULTS.md, log.txt, responses.jsonl, results.json ...). Cost is appended to smokes/COSTS.md."""
import argparse, base64, io, json, os, re, sys, tarfile, time, pathlib, requests
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--tag", required=True); p.add_argument("--bundle", nargs="+", required=True); p.add_argument("--cmd", required=True)
p.add_argument("--pip", default=""); p.add_argument("--max-min", type=int, default=60); p.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
p.add_argument("--image", default="runpod/pytorch:1.4.1-cu1281-torch280-ubuntu2204"); p.add_argument("--out", nargs="*", default=None, help="folders to bring back (default: bundled folders)")
p.add_argument("--pull", nargs="*", default=[], help="project-relative files to fetch from the pod on every poll (partial results survive a vanished pod)")
p.add_argument("--attach", default=None, help="existing pod id (already booted with this launcher's job script, waiting for a bundle): skip creation and upload to it")
a = p.parse_args()
ENVT = (ROOT / ".env").read_text() if (ROOT / ".env").exists() else ""  # a git worktree has no .env: key from the environment (runenv.py), nothing shipped
KEY = os.environ.get("RUNPODKEY") or re.search(r"RUNPODKEY=(\S+)", ENVT).group(1)
GQL = f"https://api.runpod.io/graphql?api_key={KEY}"
def gql(q):
    r = requests.post(GQL, json={"query": q}, timeout=60).json()
    if "errors" in r: raise RuntimeError(r["errors"])
    return r["data"]

# ---- bundle (skip big caches and virtualenvs)
SKIP = re.compile(r"(\.venv|__pycache__|\.data[\\/]|emb_[0-9a-f]+\.npy|\.git[\\/]|bin[\\/]cloudflared)")
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    for b in a.bundle:
        path = ROOT / b
        for f in ([path] if path.is_file() else path.rglob("*")):
            if f.is_file() and not SKIP.search(str(f)):
                tar.add(f, arcname=str(f.relative_to(ROOT)).replace("\\", "/"))
bundle = buf.getvalue(); print(f"bundle {len(bundle)/1e6:.2f} MB", flush=True)
out_dirs = a.out or [b for b in a.bundle if (ROOT / b).is_dir()]

SERVER = r'''
import http.server, os, pathlib, json
ROOT = "/workspace/job"
class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *x, **k): super().__init__(*x, directory=ROOT, **k)
    def do_PUT(self):
        n = int(self.headers.get("Content-Length", 0)); data = self.rfile.read(n)
        dest = pathlib.Path(ROOT) / self.path.strip("/"); dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data); self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def log_message(self, *x): pass
http.server.ThreadingHTTPServer(("0.0.0.0", 8000), H).serve_forever()
'''
job = f"""#!/bin/bash
mkdir -p /workspace/job/proj && cd /workspace/job
echo "$SERVER_B64" | base64 -d > server.py
(python server.py > server.log 2>&1 &)
echo "$ENV_B64" | base64 -d > proj/.env
nvidia-smi --query-gpu=name,memory.total --format=csv > log.txt 2>&1 || echo "no gpu" >> log.txt
echo "waiting for bundle" >> log.txt
while [ ! -f bundle.tar.gz ]; do sleep 2; done
sleep 2; tar -xzf bundle.tar.gz -C proj && echo "bundle extracted $(find proj -type f | wc -l) files" >> log.txt
export HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
pip install -q sentence-transformers scikit-learn scipy datasets openai pandas requests transformers peft {a.pip} 2>&1 | tail -2 >> log.txt
cd proj && (timeout {a.max_min * 60 - 240} bash -c '{a.cmd.replace("'", "'\\''")}' >> ../log.txt 2>&1; echo "exit=$?" >> ../log.txt); cd ..
tar -czf out.tar.gz -C proj {" ".join(out_dirs)} 2>> log.txt
echo done > DONE
sleep 300
curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" -H 'Content-Type: application/json' -d "{{\\"query\\":\\"mutation {{ podTerminate(input: {{podId: \\\\\\"$RUNPOD_POD_ID\\\\\\"}}) }}\\"}}"
sleep 300
"""
b64 = lambda s: base64.b64encode(s.encode()).decode()
env = {"JOB_B64": b64(job), "SERVER_B64": b64(SERVER), "ENV_B64": b64(ENVT), "RUNPOD_API_KEY": KEY}
env_gql = ", ".join('{key: "%s", value: "%s"}' % (k, v) for k, v in env.items())
cmd = "bash -c 'echo $JOB_B64 | base64 -d > /job.sh && bash /job.sh'"
q = f"""mutation {{ podFindAndDeployOnDemand(input: {{ cloudType: COMMUNITY, gpuCount: 1, volumeInGb: 0, containerDiskInGb: 40,
  minVcpuCount: 4, minMemoryInGb: 16, gpuTypeId: "{a.gpu}", name: "dreaming-{a.tag}", imageName: "{a.image}",
  dockerArgs: "{cmd}", env: [{env_gql}], ports: "8000/http" }}) {{ id machineId costPerHr }} }}"""
if a.attach:  # reuse a pod that is already up and waiting for its bundle (its job script was shipped at creation)
    info = gql(f'query {{ pod(input: {{podId: "{a.attach}"}}) {{ id costPerHr desiredStatus }} }}')["pod"]
    assert info and info["desiredStatus"] == "RUNNING", f"pod {a.attach} not running: {info}"
    pod = {"id": a.attach, "costPerHr": info["costPerHr"], "attached": True}
else:
    try:
        pod = gql(q)["podFindAndDeployOnDemand"]
    except Exception as e:  # community capacity can be empty; fall back to any cloud
        print("community launch failed, trying ALL:", str(e)[:200]); pod = gql(q.replace("cloudType: COMMUNITY", "cloudType: ALL"))["podFindAndDeployOnDemand"]
pid = pod["id"]; base = f"https://{pid}-8000.proxy.runpod.net/"; print("pod", pod, flush=True)
t0 = time.time(); uploaded = False; last = 0; ok = False
try:
    while time.time() - t0 < a.max_min * 60:
        time.sleep(30)
        try: st = gql(f'query {{ pod(input: {{podId: "{pid}"}}) {{ desiredStatus runtime {{ uptimeInSeconds }} }} }}')["pod"]
        except Exception as e: st = {"desiredStatus": "?"}
        if st is None: print("pod gone"); break
        if not uploaded:
            try:
                r = requests.put(base + "bundle.tar.gz", data=bundle, timeout=120)
                if r.status_code == 200: uploaded = True; print(f"[{int(time.time()-t0)}s] bundle uploaded", flush=True)
            except Exception: pass
            continue
        try: txt = requests.get(base + "log.txt", timeout=20).text
        except Exception: txt = ""
        if len(txt) > last: print(txt[last:], end="", flush=True); last = len(txt)
        for rel in a.pull:   # partial results: fetch on every poll so a vanished pod loses at most 30 s of work
            try:
                rr = requests.get(base + "proj/" + rel, timeout=30)
                if rr.status_code == 200 and rr.content: (ROOT / rel).parent.mkdir(parents=True, exist_ok=True); (ROOT / rel).write_bytes(rr.content)
            except Exception: pass
        print(f"[{int(time.time()-t0)}s status={st['desiredStatus']} log={len(txt)}B]", flush=True)
        try:
            if requests.get(base + "DONE", timeout=20).status_code == 200:
                r = requests.get(base + "out.tar.gz", timeout=300)
                if r.status_code == 200:
                    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tar: tar.extractall(ROOT)
                    print(f"results extracted ({len(r.content)/1e6:.2f} MB) into {out_dirs}", flush=True); ok = True
                (ROOT / "smokes" / f"pod_log_{a.tag}.txt").write_text(txt, encoding="utf-8"); break
        except Exception as e: print("fetch error", e)
finally:
    try: print("terminate:", gql(f'mutation {{ podTerminate(input: {{podId: "{pid}"}}) }}'))
    except Exception as e: print("terminate failed (pod may have self-terminated):", e)
hrs = (time.time() - t0) / 3600; print(f"wall {hrs*60:.1f} min, est cost ${hrs*pod['costPerHr']:.3f} at ${pod['costPerHr']}/h, ok={ok}")
with open(ROOT / "smokes" / "COSTS.md", "a", encoding="utf-8") as f: f.write(f"pod_{a.tag},{hrs*pod['costPerHr']:.3f},RunPod {a.gpu} {hrs*60:.0f} min ok={ok}\n")
sys.exit(0 if ok else 1)
