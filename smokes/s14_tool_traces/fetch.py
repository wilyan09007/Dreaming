"""S14 data download, meant to run ON THE POD (fast network). Each dataset goes to its own folder under $S14_DATA (default /workspace/job/data,
outside the bundled smoke folder, so nothing large comes back). Prints name, bytes and sha256 so the files can be matched to the earlier local download.
  tau-bench historical trajectories (MIT, 51 MB) | TraceLab v0.0.2 trace (code Apache-2.0, data CC BY 4.0, 101 MB) | Toucan-1.5M OSS shard 10 (Apache-2.0, 147 MB)"""
import os, hashlib, time, requests
from pathlib import Path

D = Path(os.environ.get("S14_DATA", "/workspace/job/data"))
TAU = "https://raw.githubusercontent.com/sierra-research/tau-bench/main/"
TRACE = "https://github.com/uw-syfi/TraceLab/releases/download/v0.0.2/syfi_coding_trace.jsonl.gz"


def get(url, dest):
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        for k in range(4):
            try:
                with requests.get(url, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(dest.with_suffix(dest.suffix + ".part"), "wb") as f:
                        for ch in r.iter_content(1 << 20): f.write(ch)
                os.replace(dest.with_suffix(dest.suffix + ".part"), dest); break
            except Exception as e:
                print("retry", k, url, repr(e)[:120], flush=True); time.sleep(5 * (k + 1))
    h = hashlib.sha256(dest.read_bytes()).hexdigest()[:16]
    print(f"{dest.relative_to(D)}  {dest.stat().st_size} bytes  sha256 {h}", flush=True)


for n in ("gpt-4o-airline", "gpt-4o-retail", "sonnet-35-new-airline", "sonnet-35-new-retail"):
    get(TAU + f"historical_trajectories/{n}.json", D / "tau_bench" / f"{n}.json")
get(TAU + "LICENSE", D / "tau_bench" / "LICENSE")
get(TRACE, D / "tracelab" / "syfi_coding_trace_v0.0.2.jsonl.gz")
from huggingface_hub import hf_hub_download
p = hf_hub_download("Agent-Ark/Toucan-1.5M", "OSS/train-00010-of-00047.parquet", repo_type="dataset", local_dir=str(D / "toucan"))
print(f"toucan/OSS/train-00010-of-00047.parquet  {Path(p).stat().st_size} bytes", flush=True)
