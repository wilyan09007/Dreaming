"""Prefetch the likely next call (mission 3, hidden-state tools; missions.md › speculation).

When a tool's return depends on state we cannot see, the predictor gives a distribution; the runner starts the
next read-only step on the most likely return while the real call runs and keeps the work only if they match.
Opt-in through `@tool(..., speculate_next="step_name")`. S4: hit rate 0.63 on 71% of calls, ~20% latency saved;
S19: worth it only when the tool takes seconds. shortcut: not in the demo path; the hint is logged, nothing is prefetched.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .contracts import Prediction


def prefetch(prediction: Prediction, next_step: Callable[[str], Any]) -> Callable[[str], Any]:
    """Return a function that yields the prefetched result when the real return equals the predicted label, else recomputes."""
    raise NotImplementedError("speculation is logged only in the demo")
