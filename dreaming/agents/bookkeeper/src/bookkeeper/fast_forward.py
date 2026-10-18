"""Fast-forward the bookkeeper offline: N synthetic transactions through the real workflow, no network, no keys.

    uv run python -m bookkeeper.fast_forward 1500 [--seed 0]

Builds the same Runtime the tests do (MemoryStore, RuleTeacher, LocalTrainer, Loader with the CharEncoder), runs
`Runtime.fast_forward` and prints JSON lines: the final Metrics, then one Policy per call site.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from dreaming.contracts import EventKind
from dreaming.store import MemoryStore

from . import offline, tools, workflow


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bookkeeper.fast_forward", description=(__doc__ or "").split("\n")[0])
    ap.add_argument("n", type=int, help="number of transactions to run")
    ap.add_argument("--seed", type=int, default=0, help="ledger seed (default 0)")
    args = ap.parse_args(argv)
    store = MemoryStore()
    rt = offline.build_runtime(store)
    metrics = rt.fast_forward(workflow.W, tools.ledger(args.n, args.seed), workflow.run_transaction)
    print(metrics.model_dump_json())
    for policy in rt.policies.values():
        print(policy.model_dump_json())
    errors = [e for e in store.events(limit=100_000) if e.kind == EventKind.component_error]
    for e in errors[:5]:
        print(f"component.error: {e.message}: {e.error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
