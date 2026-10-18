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

`--wide N` first asks the teacher N transactions at a time (`prefetch`): the typed steps' prompts are functions of the
transaction and the teacher's own upstream labels (the mock tools are pure), so every question the sequential pass will
ask is on file before it runs, and the pass itself only pays for the few prompts a student's upstream guess changed.
Sequential StepFun is ~14 answers/min, so one seed is ~7 h; 8-wide it is ~1.5 h (smokes/RESULTS.md).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dreaming import registry
from dreaming.contracts import CallSite, EventKind, Json, TeacherAnswer
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


def prefetch(teacher: Teacher, txns: list[dict[str, Json]], width: int) -> int:
    """Ask the three typed sites about every transaction, `width` at a time, along the teacher's own labels. Returns the failures."""
    cat, inv, odd = (registry.get(f"{workflow.W}.{s}") for s in ("categorize", "match_invoice", "anything_odd"))
    cheap = odd.site.model_copy(update={"teacher": workflow.CHEAP}) if workflow.CHEAP else None  # the router's trial config is asked too

    def chain(txn: dict[str, Json]) -> bool:
        try:
            category = teacher.ask(cat.site, cat.fn(txn)).label
            merchant = str(txn["descriptor"]).split(" *")[0]
            vendor = tools.MERCHANTS[merchant].name if merchant in tools.MERCHANTS else "unknown"  # vendor_api without the clock
            if category is None:
                return True  # unparsed label: the downstream prompts cannot be formed; the sequential pass asks live
            invoice = teacher.ask(inv.site, inv.fn(txn, vendor, category, tools.open_invoices(vendor))).label
            if invoice is None:
                return True
            rendered = odd.fn(txn, vendor, category, invoice, tools.history(vendor))
            teacher.ask(odd.site, rendered)
            if cheap is not None:
                teacher.ask(cheap, rendered)
            return False
        except Exception as e:  # noqa: BLE001  # any provider error: one more live question later, not a dead pass
            print(f"prefetch: {txn.get('id')}: {type(e).__name__}: {e}", file=sys.stderr)
            return True

    failed, t0 = 0, time.perf_counter()
    with ThreadPoolExecutor(max_workers=width) as pool:
        for i, bad in enumerate(pool.map(chain, txns), 1):
            failed += bad
            if i % 100 == 0 or i == len(txns):
                print(f"prefetch: {i}/{len(txns)} transactions, {failed} failed, {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return failed


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
    ap.add_argument("--wide", type=int, default=1, help="prefetch the teacher's answers this many transactions at a time before the sequential pass")
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
        if args.wide > 1:
            prefetch(teacher, list(tools.ledger(args.n, seed)), args.wide)
            print(f"prefetch seed {seed}: {teacher.new} new answers, {teacher.served} already recorded, ${teacher.cost:.4f}", flush=True)
            teacher.served = teacher.new = 0
            teacher.cost = 0.0
        rt.fast_forward(workflow.W, tools.ledger(args.n, seed), workflow.run_transaction)
        print(f"seed {seed}: {args.n} transactions, {teacher.new} new answers, {teacher.served} already recorded, ${teacher.cost:.4f}", flush=True)
    errors = [e for e in store.events(limit=100_000) if e.kind == EventKind.component_error]
    for e in errors[:3]:
        print(f"component.error: {e.message}: {e.error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
