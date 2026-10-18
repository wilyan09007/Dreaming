"""Tool-return predictors for tools whose return is a function of the input we can see (mission 3, case 1).

MemoryPredictor: exact memory of (rendered input -> return) from the trace log. No model, no training: the
confidence is the share of past calls that agreed, so an input seen once with one return is confident and an
input never seen is not. Byte-replayable by construction. S4: certifiable at a 1% budget on known merchants.

TfidfPredictor: char n-gram TF-IDF + logistic head for near-miss inputs (S4: 70% of calls skipped at 0.1% error).
shortcut: contract only; MemoryPredictor is enough for the demo's vendor lookup, upgrade when inputs vary in form.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

from ..contracts import Hashes, Prediction
from ..gate import CONF_DECIMALS
from ..hashing import prompt_sha
from . import profile_sha


class MemoryPredictor:
    """The memory is fixed for the life of a certificate; new pairs mean new weights (a retrain), never an edit in place."""

    def __init__(self, pairs: Iterable[tuple[str, str]], weights_sha: str) -> None:
        self._seen: dict[str, Counter[str]] = defaultdict(Counter)
        for rendered, label in pairs:
            self._seen[prompt_sha(rendered)][label] += 1
        self.env_sha = profile_sha(predictor="memory")
        self.weights_sha = weights_sha

    def predict(self, rendered: str, *, fresh: bool = False) -> Prediction:
        key = prompt_sha(rendered)
        votes = self._seen.get(key)
        if not votes:
            return Prediction(label="", confidence=0.0, hashes=Hashes(prompt=key, weights=self.weights_sha, env=self.env_sha))
        (label, n), total = votes.most_common(1)[0], sum(votes.values())
        return Prediction(
            label=label, confidence=round(n / total, CONF_DECIMALS),
            scores={lab: round(c / total, 4) for lab, c in votes.items()},
            hashes=Hashes(prompt=key, weights=self.weights_sha, env=self.env_sha),
        )


class TfidfPredictor:
    weights_sha: str
    env_sha: str

    def __init__(self) -> None:
        raise NotImplementedError("port from smokes/s4_tool_predictor/run.py when exact memory is not enough")

    def predict(self, rendered: str, *, fresh: bool = False) -> Prediction:
        raise NotImplementedError
