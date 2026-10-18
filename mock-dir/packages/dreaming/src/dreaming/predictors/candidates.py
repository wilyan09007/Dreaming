"""The candidate registry: the model slot (DESIGN.md §5, docs/training-plan.md §1).

A candidate is one family of predictor: two pure functions and two flags. The learner trains every candidate a site lists
(`CallSite.candidates`, or the kind's default), keeps all of them in shadow, certifies each at delta/N and promotes the best;
the campaign runs the same `fit` per cell. Nothing here reads the environment or touches a Policy.

    CANDIDATES["head"].fit(site, rows, encoder) -> Fitted      the serialized weights plus what the TrainResult reports
    CANDIDATES["head"].load(site, data, encoder) -> Predictor  the same bytes back as a predictor

Built: head (frozen encoder + linear head), exact (tool memory), config (a router site's cheap configuration),
memory (Hopfield read over stored rows).
To port from the smokes: online (delta-rule head), hdc (integer bundles),
finetune (bge-small unfrozen, GPU; its weights also arrive from veri.studio in the same artifact layout).
Each is one module registering itself here; the registry is the only list to edit.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from ..contracts import CallSite, Kind, TraceRow
from ..hashing import sha_bytes
from . import Predictor


@dataclass(frozen=True)
class Fitted:
    data: bytes  # the weights, serialized; the store hashes exactly these bytes
    n_train: int
    holdout_accuracy: float  # on a held-out slice of the training rows, for the record; 1.0 where there is nothing to hold out
    seconds: float

    @property
    def weights_sha(self) -> str:
        return sha_bytes(self.data)


Fit = Callable[[CallSite, Sequence[TraceRow], Any], Fitted]  # Any: the Encoder (None for candidates that need none)
Load = Callable[[CallSite, bytes, Any], Predictor]


@dataclass(frozen=True)
class Candidate:
    name: str
    fit: Fit
    load: Load
    needs_encoder: bool = True
    needs_gpu: bool = False  # True: the learner submits it as a GPU job and it lands on a later tick


CANDIDATES: dict[str, Candidate] = {}
DEFAULT = {Kind.model: ["head", "memory"], Kind.tool: ["exact"], Kind.router: ["config"], Kind.a2a: ["exact"], Kind.effect: []}


def register(c: Candidate) -> Candidate:
    if c.name in CANDIDATES:
        raise ValueError(f"candidate {c.name!r} registered twice")
    CANDIDATES[c.name] = c
    return c


def for_site(site: CallSite) -> list[Candidate]:
    """The candidates to train for a site: its own list, else the kind's default. Unknown names are loud."""
    return [CANDIDATES[n] for n in (site.candidates or DEFAULT[site.kind])]


# ---------------------------------------------------------------- the three built ones, from the modules that already exist


def _fit_head(site: CallSite, rows: Sequence[TraceRow], encoder: Any) -> Fitted:
    from .train import fit_head_from_rows

    return fit_head_from_rows(site, rows, encoder)


def _load_head(site: CallSite, data: bytes, encoder: Any) -> Predictor:
    from .student import Head, Student

    if encoder is None:
        raise ValueError(f"{site.id}: the head candidate needs an encoder")
    return Student(encoder, Head.from_bytes(data), sha_bytes(data))


def _fit_exact(site: CallSite, rows: Sequence[TraceRow], encoder: Any) -> Fitted:
    from .train import labelled

    t0 = time.perf_counter()
    pairs = labelled(rows)
    if not pairs:
        raise ValueError(f"{site.id}: no labelled rows to train on (did every real call fail?)")  # seen live: a 402 teacher
    return Fitted(json.dumps(pairs).encode(), len(pairs), 1.0, time.perf_counter() - t0)


def _load_exact(site: CallSite, data: bytes, encoder: Any) -> Predictor:
    from .tool import MemoryPredictor

    return MemoryPredictor(((text, label) for text, label in json.loads(data)), sha_bytes(data))


def _fit_config(site: CallSite, rows: Sequence[TraceRow], encoder: Any) -> Fitted:
    if site.cheap is None:
        raise ValueError(f"{site.id}: a router site needs a cheap TeacherSpec")
    return Fitted(site.cheap.model_dump_json().encode(), 0, 1.0, 0.0)  # nothing to fit: the cheap configuration is the predictor


def _load_config(site: CallSite, data: bytes, teacher: Any) -> Predictor:
    from ..contracts import TeacherSpec
    from .router import ConfigPredictor

    if teacher is None:
        raise ValueError("a router predictor needs a teacher to call the cheap configuration")
    return ConfigPredictor(site, TeacherSpec.model_validate_json(data), teacher, sha_bytes(data))


register(Candidate("head", _fit_head, _load_head))
register(Candidate("exact", _fit_exact, _load_exact, needs_encoder=False))
register(Candidate("config", _fit_config, _load_config, needs_encoder=False))  # `load` takes the teacher where others take the encoder

import_module(".memory", __package__)  # registers "memory"; at the bottom because memory.py imports Candidate from here
