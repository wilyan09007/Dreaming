"""Record the teacher's answers for the demo ledger once, with the live model, so fast-forward can replay them without a key.

    OPENROUTERKEY=... uv run python -m bookkeeper.record 2000 [--out answers.jsonl]

The first N transactions of each ledger seed (0, 1, 2) go through `workflow.run_transaction` in fast-forward mode on one
Runtime, seeds in order, the way apps/api draws its batches. The runtime learns as it goes (MemoryStore, LocalTrainer,
BUDGET as the API reads it), so the questions put to the teacher are the ones the demo's own run will put. Every (site,
rendered input) pair lands once in RECORDED_ANSWERS (RecordedTeacher over OpenAICompatTeacher, provider pinned by the
workflow's spec); a rerun resumes, serving what is on file and paying only for the rest. With the `student` extra the
SentenceEncoder also fills EMBEDDING_CACHE (the runtime saves it after each seed). Without torch the student sites never
train, so every typed step is asked: record on the machine that runs the demo, or a dreaming run may ask what the file lacks.
Prints one line per seed; exits 1 if any run failed (the first errors go to stderr), because a failed run records nothing.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from dreaming.contracts import CallSite, EventKind, TeacherAnswer
from dreaming.predictors import Loader
from dreaming.predictors.train import LocalTrainer
from dreaming.runtime import Runtime
from dreaming.store import MemoryStore
from dreaming.teachers import Teacher
from dreaming.teachers.openai_compat import PROVIDERS, OpenAICompatTeacher
from dreaming.teachers.recorded import RecordedTeacher

from . import tools, workflow

SEEDS = (0, 1, 2)  # must equal dreaming_api.wiring.RECORDED_SEEDS: the demo's stream draws only these seeds, in this order


class Counting(RecordedTeacher):
    """A RecordedTeacher that tallies what the per-seed line reports: served from the file, asked live, and what that cost."""

    served, new, cost = 0, 0, 0.0

    def ask(self, site: CallSite, rendered: str) -> TeacherAnswer:
        ans = super().ask(site, rendered)
        if ans.recorded:
            self.served += 1
        else:
            self.new, self.cost = self.new + 1, self.cost + ans.cost_usd
        return ans


def _encoder() -> object | None:
    """The student encoder with its cache, or None (with a note) when the `student` extra is not installed. Same default as apps/api."""
    try:
        from dreaming.predictors.student import SentenceEncoder

        return SentenceEncoder(cache_path=Path(os.environ.get("EMBEDDING_CACHE", "data/embeddings.npz")))
    except ImportError:
        print("note: torch/sentence-transformers not installed; the embedding cache is not filled and student sites stay awake", file=sys.stderr)
        return None


def main(argv: Sequence[str] | None = None, live: Teacher | None = None) -> int:
    """`live` is the teacher that answers misses; tests pass a fake, the command builds the OpenRouter one from OPENROUTERKEY."""
    ap = argparse.ArgumentParser(prog="python -m bookkeeper.record", description=(__doc__ or "").split("\n")[0])
    ap.add_argument("n", type=int, help="transactions to record per ledger seed (the API draws 2000)")
    ap.add_argument("--out", default=os.environ.get("RECORDED_ANSWERS", "recorded_answers.jsonl"), help="answers file (JSON lines, append-only)")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)), help="ledger seeds to record, in order (default: all three; one seed is ~2 h on StepFun)")
    args = ap.parse_args(argv)
    seeds = [int(x) for x in args.seeds.split(",") if x]
    if not set(seeds) <= set(SEEDS):
        raise SystemExit(f"--seeds must be among {SEEDS}: the demo's stream draws only these")
    if live is None:
        provider = workflow.TEACHER.provider  # the row the workflow pins (TEACHER env: openrouter | asi1 | stepfun)
        if not (key := os.environ.get(PROVIDERS[provider].key)):
            raise SystemExit(f"{PROVIDERS[provider].key} is not set: recording needs the live teacher ({provider})")
        live = OpenAICompatTeacher({provider: key})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)  # the first paid answer must not be lost to a missing directory
    store, teacher = MemoryStore(), Counting(out, live)
    loader = Loader(encoder=_encoder(), teacher=teacher)
    rt = Runtime(store, teacher, LocalTrainer(store, loader), loader, budget=float(os.environ.get("BUDGET", "0.05")))
    rt.bind()
    for seed in seeds:
        teacher.served = teacher.new = 0
        teacher.cost = 0.0
        rt.fast_forward(workflow.W, tools.ledger(args.n, seed), workflow.run_transaction)
        print(f"seed {seed}: {args.n} transactions, {teacher.new} new answers, {teacher.served} already recorded, ${teacher.cost:.4f}")
    errors = [e for e in store.events(limit=100_000) if e.kind == EventKind.component_error]
    for e in errors[:3]:
        print(f"component.error: {e.message}: {e.error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
