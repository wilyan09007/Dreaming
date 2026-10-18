"""The trivial baselines of the benchmark table (docs/tracks.md §4, training-plan.md §5), on the same rows and split as the students.

    customer_rule(rows)        invoice_paid: the customer's usual outcome so far, read from the row's own history line (what was known that day)
    vendor_rule(rows)          bill_account: this vendor's most frequent account among the company's earlier bills (an account is known once posted)
    closest_document(rows)     bank_match: the open document closest in amount (the first one listed)
    exact_cache(train, rows)   any site: the label memorized for an identical rendered input among the training rows
    evaluate(rows, preds, budget)   certify on the calib slice, measure on test: answered share, accuracy when answering, certified coverage
    report(site, rows, budget)      both baselines of a site, one call

A prediction is (label, confidence) or (None, 0.0) to abstain; confidence is the winning label's share of the votes, so the certificate has
something to cut on. Rows are data.rows() dicts: {rendered, label, group, ts}.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from dreaming.gate import CONF_DECIMALS, certify, realized
from dreaming.hashing import sha

Rows = Sequence[Mapping[str, str]]
Pred = tuple[str | None, float]
ABSTAIN: Pred = (None, 0.0)


def _vote(c: Counter[str]) -> Pred:
    if not c:
        return ABSTAIN
    label, n = c.most_common(1)[0]
    return label, round(n / c.total(), CONF_DECIMALS)


def _by_time(rows: Rows) -> list[int]:
    return sorted(range(len(rows)), key=lambda i: (rows[i]["ts"], rows[i]["rendered"]))


def vendor_rule(rows: Rows) -> list[Pred]:
    seen: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    out: list[Pred] = [ABSTAIN] * len(rows)
    for i in _by_time(rows):
        c = seen[rows[i]["group"], rows[i]["rendered"].split("\n")[0]]
        out[i] = _vote(c)
        c[rows[i]["label"]] += 1
    return out


def customer_rule(rows: Rows) -> list[Pred]:
    out: list[Pred] = []
    for r in rows:
        h = {k: int(n) for n, k in re.findall(r"(\d+) paid (on time|late)", r["rendered"])}
        out.append(_vote(Counter({"on_time": h.get("on time", 0), "late": h.get("late", 0)}) - Counter()))  # `- Counter()` drops zeros
    return out


def closest_document(rows: Rows) -> list[Pred]:
    out: list[Pred] = []
    for r in rows:
        bank, _, docs = r["rendered"].partition("\nOpen documents:\n")
        first = docs.split("\n")[0].split("  ")
        if len(first) < 4:  # "none"
            out.append(("none", 0.5))
            continue
        out.append((first[0], 1.0 if first[2] == bank.split("  ")[1] else 0.5))  # an exact amount match is the confident case
    return out


def exact_cache(train: Rows, rows: Rows) -> list[Pred]:
    memo: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for r in train:
        memo[r["rendered"]][r["label"]] += 1
    return [_vote(memo.get(r["rendered"], Counter())) for r in rows]


def split(rows: Rows) -> dict[str, list[int]]:
    """docs/training-plan.md §3: the last 20% by time is forward; earlier rows go whole by group, sha(group) mod 10: 0-1 calib, 2-3 test.
    ponytail: a copy of dreaming_trainer.campaign.split (the trainer imports this package, not the reverse); a test pins them equal.
    Move both into `dreaming` when either changes."""
    order = _by_time(rows)
    cut = len(order) * 4 // 5
    out: dict[str, list[int]] = {"train": [], "calib": [], "test": [], "forward": order[cut:]}
    for i in order[:cut]:
        b = int(sha(rows[i]["group"]), 16) % 10
        out["calib" if b < 2 else "test" if b < 4 else "train"].append(i)
    return out


def evaluate(rows: Rows, preds: Sequence[Pred], budget: float = 0.05, delta: float = 0.05, sp: Mapping[str, Sequence[int]] | None = None) -> dict[str, Any]:
    sp = sp or split(rows)

    def arrays(ix: Sequence[int]) -> tuple[Any, Any, Any]:
        conf = np.array([preds[i][1] for i in ix], dtype=np.float64)
        ok = np.array([preds[i][0] == rows[i]["label"] for i in ix], dtype=bool)
        return conf, ok, np.array([preds[i][0] is not None for i in ix], dtype=bool)

    cc, cok, cans = arrays(sp["calib"])
    tc, tok, tans = arrays(sp["test"])
    gate = certify(cc[cans], cok[cans], budget, delta)  # an abstention is never served, so it is never coverable: certify the answers only
    coverage, error = realized(np.where(tans, tc, -1.0), tok, gate.cutoff)  # coverage over all test rows; -1 sits below every cutoff
    return {"n_calib": len(cc), "n_test": len(tc), "answered": round(float(tans.mean()), 4) if len(tc) else 0.0,
            "accuracy": round(float(tok[tans].mean()), 4) if tans.any() else 0.0, "budget": budget, "cutoff": gate.cutoff,
            "bound": round(gate.bound, 4), "certified_coverage": round(coverage, 4), "realized_error": round(error, 4)}


RULES: dict[str, Callable[[Rows], list[Pred]]] = {"invoice_paid": customer_rule, "bill_account": vendor_rule, "bank_match": closest_document}


def report(site: str, rows: Rows, budget: float = 0.05) -> dict[str, dict[str, Any]]:
    sp = split(rows)
    return {"rule": evaluate(rows, RULES[site](rows), budget, sp=sp),
            "exact_cache": evaluate(rows, exact_cache([rows[i] for i in sp["train"]], rows), budget, sp=sp)}
