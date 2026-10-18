"""Launch train_student.py on a RunPod GPU pod with no SSH and no S3: the script and job ship inline as base64 env vars,
the pod serves its working dir over RunPod's HTTP proxy (port 8000) so we can read live logs and results, and the pod
terminates itself (controller terminates it too as a backstop). Usage: python launch_runpod.py [--gpu ...] [--max-min 50]"""
import argparse, base64, json, re, sys, time, pathlib, requests
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parents[1]
p = argparse.ArgumentParser(); p.add_argument("--gpu", default="NVIDIA GeForce RTX 4090"); p.add_argument("--max-min", type=int, default=50)
p.add_argument("--image", default="runpod/pytorch:1.4.1-cu1281-torch280-ubuntu2204"); p.add_argument("--script", default="train_student.py"); p.add_argument("--args", default="--model Qwen/Qwen2.5-0.5B --out results.json"); p.add_argument("--tag", default=""); p.add_argument("--cuda", default="", help="comma list of allowed host CUDA versions, e.g. 12.8 (driver 580 hosts fail cuInit with error 999)")
a = p.parse_args()
KEY = re.search(r"RUNPODKEY=(\S+)", (ROOT / ".env").read_text()).group(1)
GQL = f"https://api.runpod.io/graphql?api_key={KEY}"
def gql(q):
    r = requests.post(GQL, json={"query": q}, timeout=60).json()
    if "errors" in r: raise RuntimeError(r["errors"])
    return r["data"]

job = f"""#!/bin/bash
set -x
export HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p /workspace/job && cd /workspace/job
echo "$SCRIPT_B64" | base64 -d > script.py
(python -m http.server 8000 --directory /workspace/job > /dev/null 2>&1 &)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > log.txt
# newer host drivers (580) break against the image's CUDA forward-compat libcuda (error 999): drop the compat libs if torch cannot see the GPU
if ! python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  echo "cuda not available at first try; removing cuda compat libs" >> log.txt; ls -d /usr/local/cuda*/compat* >> log.txt 2>&1
  rm -rf /usr/local/cuda*/compat*; ldconfig 2>/dev/null
  export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '
' | grep -vi compat | paste -sd: -)
  python -c 'import torch; print("cuda after compat removal:", torch.cuda.is_available())' >> log.txt 2>&1
else echo "cuda ok at first try" >> log.txt; fi
pip install -q transformers peft datasets accelerate scipy scikit-learn sentence-transformers 2>&1 | tail -2 >> log.txt
pip install -q pyvene 2>&1 | tail -2 >> log.txt
pip install -q --no-deps pyreft 2>&1 | tail -2 >> log.txt
pip list 2>/dev/null | grep -i -E '^(torch|transformers|peft|numpy|pyvene|pyreft|sentence-transformers) ' >> log.txt
timeout {a.max_min * 60 - 300} python script.py {a.args} >> log.txt 2>&1
echo "exit=$?" >> log.txt
echo done > DONE
sleep 420
curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" -H 'Content-Type: application/json' -d "{{\\"query\\":\\"mutation {{ podTerminate(input: {{podId: \\\\\\"$RUNPOD_POD_ID\\\\\\"}}) }}\\"}}"
sleep 300
"""
b64 = lambda s: base64.b64encode(s.encode()).decode()
env = {"JOB_B64": b64(job), "SCRIPT_B64": b64((HERE / a.script).read_text(encoding="utf-8")), "RUNPOD_API_KEY": KEY}
env_gql = ", ".join('{key: "%s", value: "%s"}' % (k, v) for k, v in env.items())
cmd = "bash -c 'echo $JOB_B64 | base64 -d > /job.sh && bash /job.sh'"
ALLOWED = ("allowedCudaVersions: [%s], " % ", ".join('"%s"' % v for v in a.cuda.split(","))) if a.cuda else ""
q = f"""mutation {{ podFindAndDeployOnDemand(input: {{ cloudType: COMMUNITY, gpuCount: 1, volumeInGb: 0, containerDiskInGb: 40,
  minVcpuCount: 4, minMemoryInGb: 16, gpuTypeId: "{a.gpu}", {ALLOWED}name: "dreaming-s22{a.tag}", imageName: "{a.image}",
  dockerArgs: "{cmd}", env: [{env_gql}], ports: "8000/http" }}) {{ id machineId costPerHr }} }}"""
pod = gql(q)["podFindAndDeployOnDemand"]; pid = pod["id"]; print("pod", pod, flush=True)
base = f"https://{pid}-8000.proxy.runpod.net/"
t0 = time.time(); last_len = 0
try:
    while time.time() - t0 < a.max_min * 60:
        time.sleep(60)
        try:
            st = gql(f'query {{ pod(input: {{podId: "{pid}"}}) {{ desiredStatus runtime {{ uptimeInSeconds }} }} }}')["pod"]
        except Exception as e: st = {"desiredStatus": f"? {e}"}
        try:
            lg = requests.get(base + "log.txt", timeout=20); txt = lg.text if lg.status_code == 200 else ""
        except Exception: txt = ""
        if len(txt) > last_len: print(txt[last_len:], end="", flush=True); last_len = len(txt)
        print(f"[{int(time.time()-t0)}s status={st and st['desiredStatus']} uptime={((st or {}).get('runtime') or {}).get('uptimeInSeconds')} log={len(txt)}B]", flush=True)
        try:
            if requests.get(base + "DONE", timeout=20).status_code == 200:
                r = requests.get(base + "results.json", timeout=30)
                if r.status_code == 200: (HERE / f"results{a.tag}.json").write_text(r.text, encoding="utf-8"); print("downloaded results.json")
                (HERE / f"log{a.tag}.txt").write_text(txt, encoding="utf-8"); print("downloaded log.txt"); break
        except Exception: pass
        if st is None: print("pod gone"); break
finally:
    if not (HERE / f"results{a.tag}.json").exists() or time.time() - t0 >= a.max_min * 60:
        try:
            r = requests.get(base + "results.json", timeout=30)
            if r.status_code == 200: (HERE / f"results{a.tag}.json").write_text(r.text, encoding="utf-8"); print("downloaded partial results.json")
            lg = requests.get(base + "log.txt", timeout=30)
            if lg.status_code == 200: (HERE / f"log{a.tag}.txt").write_text(lg.text, encoding="utf-8")
        except Exception as e: print("partial download failed:", e)
    try: print("terminate:", gql(f'mutation {{ podTerminate(input: {{podId: "{pid}"}}) }}'))
    except Exception as e: print("terminate failed (pod may have self-terminated):", e)
hrs = (time.time() - t0) / 3600; print(f"wall {hrs*60:.1f} min, est cost ${hrs * pod['costPerHr']:.3f} at ${pod['costPerHr']}/h")
with open(ROOT / "smokes" / "COSTS.md", "a") as f: f.write(f"s22_arch{a.tag},{hrs * pod['costPerHr']:.3f},RunPod {a.gpu} {hrs*60:.0f} min\n")
