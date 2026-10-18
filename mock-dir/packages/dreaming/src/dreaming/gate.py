"""Cutoff and error bound. Pure math, no I/O.

A certificate is a fixed-grid search (not a binary search: the Clopper-Pearson bound is U-shaped in coverage,
so a failed probe at one coverage says nothing about the others; smokes/FINDINGS.md plan change 7). Evaluate
`grid` coverage levels, each with delta/grid (union bound), and keep the largest level whose bound clears the budget.
Realized holdout error stayed under budget in every smoke (S1, S3, S4, S8, S9).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta

_beta: Any = beta  # scipy ships no types

CONF_DECIMALS = 4  # confidence is quantized before it is thresholded or hashed (S7: raw floats differ by 1e-7 across shapes)


def confidence(proba: NDArray[np.floating]) -> NDArray[np.float64]:
    """Gap between the top two scores per row: the missions' definition of confidence. Quantized."""
    s = np.sort(np.asarray(proba, dtype=np.float64), axis=1)
    return np.round(s[:, -1] - s[:, -2], CONF_DECIMALS)


def cp_upper(errors: int, n: int, delta: float) -> float:
    """Clopper-Pearson one-sided upper bound on the error rate: P(true error > bound) <= delta."""
    if n == 0 or errors >= n:
        return 1.0
    return float(_beta.ppf(1 - delta, errors + 1, n - errors))


@dataclass(frozen=True)
class Gate:
    cutoff: float  # answer when confidence >= cutoff; math.inf means nothing is certifiable
    coverage: float
    emp_error: float
    bound: float
    n: int  # rows at or above the cutoff
    errors: int  # errors among them
    total: int = 0  # calibration rows the gate looked at

    @property
    def certified(self) -> bool:
        return self.cutoff != float("inf")


def certify(conf: NDArray[np.floating], correct: NDArray[np.bool_], budget: float, delta: float = 0.05, grid: int = 40) -> Gate:
    """Largest coverage whose Clopper-Pearson bound is <= budget. `conf` and `correct` are the calibration rows."""
    c = np.round(np.asarray(conf, dtype=np.float64), CONF_DECIMALS)
    ok = np.asarray(correct, dtype=bool)
    if len(c) == 0:
        return Gate(float("inf"), 0.0, 0.0, 1.0, 0, 0, 0)
    order = np.argsort(-c, kind="stable")
    c, ok = c[order], ok[order]
    n = len(c)
    cum_err = np.cumsum(~ok)
    levels: list[int] = []
    for j in range(grid, 0, -1):
        k = max(1, round(n * j / grid))
        while k < n and c[k] == c[k - 1]:  # close ties so the cutoff is a real threshold
            k += 1
        if k not in levels:  # a constant-confidence predictor has one level, not forty: pay delta once
            levels.append(k)
    d = delta / len(levels) if levels else delta
    for k in levels:
        ub = cp_upper(int(cum_err[k - 1]), k, d)
        if ub <= budget:
            return Gate(float(c[k - 1]), k / n, float(cum_err[k - 1] / k), ub, k, int(cum_err[k - 1]), n)
    return Gate(float("inf"), 0.0, 0.0, 1.0, n, int(cum_err[-1]) if n else 0, n)


def smallest_budget(conf: NDArray[np.floating], correct: NDArray[np.bool_], delta: float = 0.05, upto: float = 0.20) -> float | None:
    """The first budget on a 1% grid at which `certify` finds a cutoff: the page's "allow 7% and it switches" hint."""
    if len(conf) == 0:
        return None
    for b in range(1, round(upto * 100) + 1):
        if certify(conf, correct, b / 100, delta).certified:
            return b / 100
    return None


def audits_needed(budget: float, delta: float) -> int:
    """Audits with zero errors needed before their bound can clear the budget: below this, audits cannot revoke anything."""
    return int(np.ceil(np.log(delta) / np.log(1 - budget)))


def realized(conf: NDArray[np.floating], correct: NDArray[np.bool_], cutoff: float) -> tuple[float, float]:
    """(coverage, error) a cutoff actually achieves on another set of rows. The audit monitor compares this with the bound."""
    c = np.round(np.asarray(conf, dtype=np.float64), CONF_DECIMALS)
    above = c >= cutoff
    if not above.any():
        return 0.0, 0.0
    return float(above.mean()), float((~np.asarray(correct, dtype=bool))[above].mean())
