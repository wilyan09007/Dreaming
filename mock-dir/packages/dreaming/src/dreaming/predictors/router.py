"""Mission 1 as a predictor: the cheaper configuration of the frontier model (a smaller tier, thinking off).

Its "prediction" is a real call to the cheap configuration, so it costs money and is not replayable; its
confidence is constant, so a certificate either covers the whole call site or nothing (S5: per-call-site routing
with a measured match rate beats prompt-only routers, which do no better than random mixing). The weights are
the cheap TeacherSpec itself, hashed.
"""

from __future__ import annotations

from ..contracts import CallSite, Hashes, Prediction, TeacherSpec
from ..hashing import prompt_sha
from ..teachers import Teacher
from . import profile_sha


class ConfigPredictor:
    def __init__(self, site: CallSite, cheap: TeacherSpec, teacher: Teacher, weights_sha: str) -> None:
        self.site = site.model_copy(update={"teacher": cheap})
        self.teacher = teacher
        self.weights_sha = weights_sha
        self.env_sha = profile_sha(predictor="config", provider=cheap.provider)

    def predict(self, rendered: str, *, fresh: bool = False) -> Prediction:
        ans = self.teacher.ask(self.site, rendered)
        return Prediction(
            label=ans.label or "", confidence=1.0, hashes=Hashes(prompt=prompt_sha(rendered), weights=self.weights_sha, env=self.env_sha),
            cost_usd=ans.cost_usd, latency_ms=ans.latency_ms,
        )
