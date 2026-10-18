"""S14b data download (github.com only). Big files go to <repo>/.data (git-ignored). Day 1 (2026-06-01) of the Microsoft GitHub Copilot coding-agent
traces (CC BY 4.0) is expected to be already extracted under .data/ghcp/date=2026-06-01/; this script fetches day 2 (chronological test set) and the TraceLab trace
(used for the cross-dataset latency check). Never downloads more than two Copilot days. Usage: python -I fetch.py [day2] [tracelab]"""
import sys, os, tarfile, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2] / ".data"
REL = "https://github.com/Azure/AzurePublicDataset/releases/download/ghcp-coding-agent-2026/date.{d}.tar.gz"
TRACE = "https://github.com/uw-syfi/TraceLab/releases/download/v0.0.2/syfi_coding_trace.jsonl.gz"

def get(url, dest):
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists(): print("have", dest); return
    subprocess.run(["curl", "-fsSL", "--retry", "4", "-o", str(dest) + ".part", url], check=True)
    os.replace(str(dest) + ".part", dest); print("got", dest, dest.stat().st_size, flush=True)

what = sys.argv[1:] or ["day2", "tracelab"]
if "tracelab" in what: get(TRACE, ROOT / "tracelab" / "syfi_coding_trace_v0.0.2.jsonl.gz")
if "day2" in what:
    d = "2026-06-02"; tgz = ROOT / "ghcp_dl" / f"date.{d}.tar.gz"; get(REL.format(d=d), tgz)
    if not (ROOT / "ghcp" / f"date={d}" / "manifest.json").exists():
        with tarfile.open(tgz) as t: t.extractall(ROOT / "ghcp", filter="data")
    print("extracted", sorted(p.name for p in (ROOT / "ghcp").iterdir()))
