"""Before / after for the veri cell, offline on the laptop CPU: the frozen bge-small + `fit_head` against the downloaded artifact.

    uv run python experiments/veri_finetune/compare.py <artifact.tar from the job, or a dir with encoder/ and head.npz> [--out results.json]

Both students run the CPU inference path (batch 1, 1 thread) over the held-out merchants of rows.jsonl. `calib` rows fit the
certificate (`gate.certify`, 5% budget, delta 0.05), `test` rows check it (`gate.realized`); accuracy is over calib + test.
The artifact goes through `finetune.pack` / `finetune.load`, the path the candidate uses for weights from any trainer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from dreaming.gate import certify, realized, smallest_budget
from dreaming.hashing import sha_bytes
from dreaming.predictors import Predictor
from dreaming.predictors.finetune import load, pack
from dreaming.predictors.student import SentenceEncoder, Student, fit_head

BUDGET = 0.05


def evaluate(model: Predictor, rows: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    conf: dict[str, np.ndarray[Any, Any]] = {}
    ok: dict[str, np.ndarray[Any, Any]] = {}
    for split in ("calib", "test"):
        preds = [model.predict(r["text"]) for r in rows[split]]
        conf[split] = np.array([p.confidence for p in preds])
        ok[split] = np.array([p.label == r["label"] for p, r in zip(preds, rows[split], strict=True)])
    gate = certify(conf["calib"], ok["calib"], BUDGET)
    cov, err = realized(conf["test"], ok["test"], gate.cutoff)
    return {
        "accuracy_held_out": round(float(np.concatenate([ok["calib"], ok["test"]]).mean()), 4),
        "accuracy_calib": round(float(ok["calib"].mean()), 4), "accuracy_test": round(float(ok["test"].mean()), 4),
        "certified": gate.certified, "certified_coverage_calib": round(gate.coverage, 4), "bound": round(gate.bound, 4),
        "cutoff": None if not gate.certified else gate.cutoff, "realized_coverage_test": round(cov, 4), "realized_error_test": round(err, 4),
        "smallest_certifiable_budget": smallest_budget(conf["calib"], ok["calib"]),
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact", type=Path)
    ap.add_argument("--rows", type=Path, default=here / "rows.jsonl")
    ap.add_argument("--out", type=Path, default=here / "results.json")
    args = ap.parse_args()
    allrows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines()]
    rows = {s: [r for r in allrows if r["split"] == s] for s in ("train", "calib", "test")}
    labels = sorted({r["label"] for r in allrows})

    enc = SentenceEncoder()  # the pinned hub bge-small, CPU, batch 1
    head = fit_head(np.stack([enc.embed(r["text"]) for r in rows["train"]]), [r["label"] for r in rows["train"]], labels, enc.model_sha)
    before = Student(enc, head, sha_bytes(head.to_bytes()))

    data = args.artifact.read_bytes() if args.artifact.is_file() else pack(args.artifact / "encoder", (args.artifact / "head.npz").read_bytes())
    after = load(cast(Any, None), data, None)  # `site` and the shared encoder are unused by this candidate

    held = rows["calib"] + rows["test"]
    top = max({r["label"] for r in held}, key=[r["label"] for r in held].count)
    result = {
        "rows": {s: len(v) for s, v in rows.items()}, "budget": BUDGET,
        "majority_label_accuracy_held_out": round(float(np.mean([r["label"] == top for r in held])), 4),
        "frozen_bge_small_plus_fit_head": evaluate(before, rows), "finetuned_artifact": evaluate(after, rows),
        "artifact_sha256": sha_bytes(data), "encoder_sha256": cast(Any, after).encoder.model_sha,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
