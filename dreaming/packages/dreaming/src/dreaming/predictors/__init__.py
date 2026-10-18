"""Predictors: anything that scores every allowed output of a call site in one pass and reports a confidence.

    student.py   encoder (33M bge-small, frozen or fine-tuned) + linear head, for typed model steps (missions 2a/2b)
    tool.py      tool-return predictors: exact memory and char n-gram head, for tools whose return is a function of input
    router.py    the cheaper frontier configuration as a predictor (mission 1)
    memory.py    Hopfield read over stored (embedding, label) rows: no weights, forgets by refitting without a row
    seeds.py     LLM-written example inputs per label, cached under data/seeds/: the day-zero rows for the head and the memory
    train.py     rows in, weights bytes out; shared by the local trainer and the Modal job

Every predictor carries weights_sha and env_sha and runs at batch size one with a fixed input shape and a pinned
thread count: that is what makes (label, quantized confidence) byte-identical on replay (S7, S8).
"""

from __future__ import annotations

import os
import platform
from typing import Any, Protocol

from ..contracts import Prediction
from ..hashing import env_sha as _env_sha


class Predictor(Protocol):
    weights_sha: str
    env_sha: str

    def predict(self, rendered: str, *, fresh: bool = False) -> Prediction: ...  # fresh: bypass every cache (replay)


def env_profile(**extra: Any) -> dict[str, Any]:
    """The execution profile that is hashed into env_sha. Add the fields that change bytes for your predictor."""
    import numpy

    return {
        "image": os.environ.get("MODAL_IMAGE_ID") or os.environ.get("DREAMING_IMAGE_DIGEST", "dev"),  # container id; "dev" locally
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "threads": int(os.environ.get("OMP_NUM_THREADS", "1")),
        "batch": 1,
        "device": os.environ.get("DREAMING_DEVICE", "cpu"),
        "machine": platform.machine(),
        **extra,
    }


def profile_sha(**extra: Any) -> str:
    return _env_sha(env_profile(**extra))


class Loader:
    """weights bytes -> Predictor, by call-site kind. One per process; the encoder is loaded once and shared."""

    def __init__(self, encoder: Any | None = None, teacher: Any | None = None) -> None:
        self.encoder, self.teacher = encoder, teacher

    def load(self, site: Any, data: bytes, candidate: str | None = None) -> Predictor:
        """`candidate` is the family that produced the bytes (weights.meta["candidate"]); None = the kind's first default."""
        from ..contracts import CallSite
        from .candidates import CANDIDATES, for_site

        assert isinstance(site, CallSite)
        c = CANDIDATES[candidate] if candidate else for_site(site)[0]
        return c.load(site, data, self.teacher if c.name == "config" else self.encoder)
