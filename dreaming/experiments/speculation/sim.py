"""How much latency can a DAG runner with speculation take off one bookkeeper run? (docs/speculation.md)

Three executors over the same four call sites and two effects, with the per-site coverage measured on the deployed stack
(smokes/RESULTS.md, 2,000-run batch at a 5% budget, rule teacher) and the live latencies seen there (asi1-mini 0.3-0.7 s per
step, the mock vendor API 1.2 s). A predicted call costs ~5 ms on a free core and ~300 ms on Fly's shared CPU (both measured);
the table is printed for both. Speculation: a node whose inputs are still in flight starts on the predictor's top guess; the
work is kept when the real value matches, redone otherwise. A wasted guess into a real teacher call costs one teacher call
($0.0042 per 4-step live run with 3 teacher calls, so ~$0.0014 each).

    uv run python experiments/speculation/sim.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass

TEACHER_MS, VENDOR_MS, CHEAP_MS = 500, 1200, 300
TEACHER_USD = 0.0014


@dataclass(frozen=True)
class Site:
    name: str
    deps: tuple[str, ...]
    coverage: float  # share of calls the certified predictor answers (skipped real call)
    real_ms: int
    effect: bool = False


SITES = [
    Site("categorize", (), 0.40, TEACHER_MS),
    Site("vendor_lookup", (), 0.67, VENDOR_MS),
    Site("match_invoice", ("vendor_lookup", "categorize"), 0.60, TEACHER_MS),
    Site("anything_odd", ("vendor_lookup", "categorize", "match_invoice"), 0.0, CHEAP_MS),  # router: the cheap config is a real call
    Site("post_or_wake", ("anything_odd",), 0.0, 0, effect=True),
]


def run(rng: random.Random, mode: str, hit: float, pred_ms: int) -> tuple[float, float]:
    """(latency_ms, wasted_usd) for one run. mode: sequential | dag | speculate."""
    guess_at: dict[str, float] = {}  # when a top guess is available
    done_at: dict[str, float] = {}  # when the verified value is available
    clock, wasted = 0.0, 0.0
    for s in SITES:
        predicted = rng.random() < s.coverage
        own = pred_ms if predicted else s.real_ms
        verified_in = max((done_at[d] for d in s.deps), default=0.0)
        if mode == "sequential":
            start = clock
        elif mode == "dag" or s.effect:  # effects never run on a guess
            start = verified_in
        else:  # speculate: start on the guesses as soon as every input has one
            start = max((guess_at[d] for d in s.deps), default=0.0)
        finish = start + own
        if mode == "speculate" and not s.effect and start < verified_in:  # ran on a guess
            if rng.random() < hit:
                finish = max(finish, verified_in)  # kept: valid once the inputs are verified
            else:
                finish = verified_in + own  # redone on the real inputs
                wasted += 0 if predicted else TEACHER_USD
        guess_at[s.name] = start + pred_ms if (mode == "speculate" and not predicted and not s.effect) else finish
        done_at[s.name] = finish
        clock = finish if mode == "sequential" else max(clock, finish)
    return clock, wasted


def table(pred_ms: int, n: int = 20_000) -> None:
    rng = random.Random(0)
    print(f"\npredicted call = {pred_ms} ms")
    print(f"{'executor':<12}{'hit':>5}{'mean s':>9}{'p95 s':>8}{'wasted $/run':>14}")
    for mode, hits in (("sequential", (None,)), ("dag", (None,)), ("speculate", (0.3, 0.6, 0.9))):
        for hit in hits:
            res = [run(rng, mode, hit or 0.0, pred_ms) for _ in range(n)]
            lat = sorted(r[0] for r in res)
            print(f"{mode:<12}{(f'{hit:.1f}' if hit else '-'):>5}{sum(lat) / n / 1000:>9.2f}{lat[int(0.95 * n)] / 1000:>8.2f}{sum(r[1] for r in res) / n:>14.5f}")


if __name__ == "__main__":
    table(5)  # a free core
    table(300)  # Fly shared-cpu-1x, measured 2026-10-17
    # self-check: dag never slower than sequential; speculate at hit=1.0 never slower than dag
    r = random.Random(1)
    for _ in range(200):
        seed = r.random()
        a, b, c = (run(random.Random(seed), m, 1.0, 5)[0] for m in ("sequential", "dag", "speculate"))
        assert b <= a + 1e-9 and c <= b + 1e-9, (a, b, c)
