"""Export the veri cell's training rows: the bookkeeper `categorize` prompts joined with the teacher's recorded labels.

    uv run python experiments/veri_finetune/rows.py [--recorded data/recorded.jsonl] [--out experiments/veri_finetune/rows.jsonl]

The prompts are re-rendered from `bookkeeper.tools.ledger(2000, 0)` with the registered step function and joined on
`prompt_sha` with the StepFun answers `just record` wrote. The split is by merchant token (the text before " *"), the
training-plan.md rule: `sha(token) mod 10` sends buckets 0-1 to `calib`, 2-3 to `test`, the rest to `train`, so the
held-out merchants were never seen by the encoder. One line per row: {"text", "label", "merchant", "split"}.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import bookkeeper.workflow  # noqa: F401  # registers the sites
from bookkeeper import tools
from dreaming import registry
from dreaming.hashing import prompt_sha, sha_bytes

SITE = "bookkeeper.categorize"


def split_of(merchant: str) -> str:
    bucket = int(sha_bytes(merchant.encode()), 16) % 10
    return "calib" if bucket < 2 else "test" if bucket < 4 else "train"


def labels_by_prompt(recorded: Path) -> dict[str, str]:
    """prompt_sha -> label for this site's recorded answers (the key is 'site:model:params8:prompt_sha')."""
    out: dict[str, str] = {}
    for line in recorded.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        site, _model, _params, sha = row["key"].split(":")
        if site == SITE and row["answer"].get("label"):
            out[sha] = row["answer"]["label"]
    return out


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--recorded", type=Path, default=here.parents[1] / "data" / "recorded.jsonl")
    ap.add_argument("--out", type=Path, default=here / "rows.jsonl")
    args = ap.parse_args()
    fn = registry.get(SITE).fn
    labels = labels_by_prompt(args.recorded)
    rows = []
    for txn in tools.ledger(2000, 0):
        text = fn(txn)
        if (label := labels.get(prompt_sha(text))) is not None:
            merchant = str(txn["descriptor"]).split(" *")[0]
            rows.append({"text": text, "label": label, "merchant": merchant, "split": split_of(merchant)})
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    by_split = collections.Counter(r["split"] for r in rows)
    print(f"{len(rows)} of 2000 ledger rows have a recorded label ({len(labels)} categorize answers on file); {dict(by_split)}; "
          f"{len({r['merchant'] for r in rows})} merchants; labels {dict(collections.Counter(r['label'] for r in rows))}")


if __name__ == "__main__":
    main()
