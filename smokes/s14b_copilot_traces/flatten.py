"""S14b step 1: stream the gz shards of one Copilot-trace day and flatten to three compact parquet tables under .data/ghcp_flat/ (never holds a day of raw JSON in memory).
  calls   one row per tool call   (session, turn, batch, position, tool, status, duration, batch start in epoch ms, + model/tokens of the LLM call that issued it)
  llm     one row per LLM call    (end time in epoch ms, duration, initiator, model, result, tokens)
  batches one row per tool batch
Timestamps in this corpus are COMPLETION times (checked in run.py: batch start = timestamp - duration lies after the issuing LLM call's timestamp in 99.9% of pairs),
so LLM start = ts - duration and batch start = ts - duration. Usage: python -I flatten.py 2026-06-01"""
import sys, gzip, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2] / ".data"


def ms(s):  # '2026-06-01T07:19:20.220000000Z' -> epoch ms (float)
    return np.datetime64(s[:23]).astype("datetime64[ms]").astype("int64") + 0.0


def shard(path):
    C, L, B = [], [], []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line); sid = r["session_id"]
            for ti, t in enumerate(r["turns"]):
                ls = []
                for li, c in enumerate(t["llm_calls"]):
                    tk = c.get("tokens") or {}
                    e = (sid, ti, li, ms(c["timestamp"]), c["duration_ms"], c["initiator_type"], c["model"], c["result"], c["status_code"], tk.get("prompt"), tk.get("completion"), tk.get("cached"))
                    L.append(e); ls.append(e)
                lend = np.array([x[3] for x in ls]) if ls else np.array([])
                for bi, b in enumerate(t["tool_batches"]):
                    bend = ms(b["timestamp"]); bd = b["duration_ms"]; bs = bend - bd
                    fcs = b["function_calls"] or []
                    # issuing LLM call = latest LLM call that ended before the batch started (5 ms slack; first calls of a turn have second-truncated stamps)
                    j = int(np.searchsorted(lend, bs + 5, side="right")) - 1 if len(lend) else -1
                    lc = ls[j] if j >= 0 else None
                    B.append((sid, ti, bi, bs, bd, len(fcs), sum(1 for f in fcs if f["status"] != 1), b["auto_continue"], b["result"], j))
                    for k, f in enumerate(fcs):
                        C.append((sid, ti, bi, k, f["name"], f["status"], f["duration_ms"], bs, bd, len(fcs), j,
                                  lc[6] if lc else None, lc[5] if lc else None, lc[9] if lc else None, lc[11] if lc else None, lc[4] if lc else None))
    return C, L, B


def flatten(day):
    out = ROOT / "ghcp_flat"; out.mkdir(exist_ok=True)
    files = sorted((ROOT / "ghcp" / f"date={day}").glob("shard-*.jsonl.gz")); C, L, B = [], [], []
    with Pool(4) as p:
        for c, l, b in p.imap(shard, files, chunksize=1): C += c; L += l; B += b
    pd.DataFrame(C, columns=["sid", "turn", "bidx", "k", "tool", "status", "dur", "bstart", "bdur", "bsize", "lidx", "model", "init", "prompt", "cached", "ldur"]).to_parquet(out / f"{day}_calls.parquet")
    pd.DataFrame(L, columns=["sid", "turn", "lidx", "end", "dur", "init", "model", "result", "status_code", "prompt", "completion", "cached"]).to_parquet(out / f"{day}_llm.parquet")
    pd.DataFrame(B, columns=["sid", "turn", "bidx", "bstart", "bdur", "n", "nfail", "auto", "result", "lidx"]).to_parquet(out / f"{day}_batches.parquet")
    print(day, "shards", len(files), "calls", len(C), "llm", len(L), "batches", len(B), flush=True)


if __name__ == "__main__":
    for d in sys.argv[1:]: flatten(d)
