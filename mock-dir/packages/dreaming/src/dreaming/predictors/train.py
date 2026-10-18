"""Fit a predictor from trace rows. Pure: rows in, weights bytes out. The local trainer and the Modal job both call this.

    LocalTrainer.train(req, site, rows)   CANDIDATES[req.candidate].fit(site, rows, encoder) -> weights by sha -> TrainResult
    fit_head_from_rows(site, rows, encoder)   the `head` candidate: embed the rendered inputs, fit the linear head on the
                                              teacher's labels (S1), hold out 20%

Certification is not here: a certificate needs shadow rows these weights never saw (learning.issue).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from ..contracts import CallSite, TraceRow, TrainRequest, TrainResult
from ..hashing import canonical
from ..store import Store
from . import Loader
from .candidates import CANDIDATES, Fitted
from .student import Encoder, fit_head


class Trainer(Protocol):
    """A job: TrainRequest + the rows it names in, TrainResult out. Never touches a Policy; never issues a certificate."""

    def train(self, req: TrainRequest, site: CallSite, rows: Sequence[TraceRow]) -> TrainResult: ...


class LocalTrainer:
    """Train in this process on the CPU. The fallback when Modal is unreachable, and the test double."""

    def __init__(self, store: Store, loader: Loader) -> None:
        self.store, self.loader = store, loader

    def train(self, req: TrainRequest, site: CallSite, rows: Sequence[TraceRow]) -> TrainResult:
        c = CANDIDATES[req.candidate]
        f = c.fit(site, rows, self.loader.teacher if c.name == "config" else self.loader.encoder)
        self.store.put_weights(f.weights_sha, f.data, {"call_site": site.id, "rows_sha": req.rows_sha, "candidate": c.name, "where": "local:cpu"})
        return TrainResult(call_site=site.id, candidate=c.name, weights_sha=f.weights_sha, env_sha=self.loader.load(site, f.data, c.name).env_sha,
                           n_train=f.n_train, holdout_accuracy=f.holdout_accuracy, seconds=f.seconds, where="local:cpu")


def labelled(rows: Sequence[TraceRow]) -> list[tuple[str, str]]:
    """(rendered input, teacher label) for every row where the real call answered with a parsable label."""
    out: list[tuple[str, str]] = []
    for r in rows:
        if r.teacher is None or r.teacher.label is None:
            continue
        out.append((r.request if isinstance(r.request, str) else canonical(r.request).decode(), r.teacher.label))
    return out


def fit_head_from_rows(site: CallSite, rows: Sequence[TraceRow], encoder: Encoder | None, holdout: float = 0.2, seed: int = 0) -> Fitted:
    t0 = time.perf_counter()
    pairs = labelled(rows)
    if not pairs:
        raise ValueError(f"{site.id}: no labelled rows to train on (did every real call fail?)")  # seen live: a 402 teacher
    if encoder is None or site.labels is None:
        raise ValueError(f"{site.id}: a model step needs an encoder and a label set")
    rng = np.random.default_rng(seed)
    idx = [int(i) for i in rng.permutation(len(pairs))]
    cut = max(1, int(len(pairs) * (1 - holdout))) if len(pairs) > 5 else len(pairs)
    train, held = [pairs[i] for i in idx[:cut]], [pairs[i] for i in idx[cut:]]
    X = np.stack([encoder.embed(text) for text, _ in train])
    head = fit_head(X, [lab for _, lab in train], site.labels, encoder.model_sha)
    acc = 1.0
    if held:
        Xh = np.stack([encoder.embed(text) for text, _ in held])
        pred = np.asarray(Xh @ head.W + head.b).argmax(axis=1)
        acc = float(np.mean([head.labels[int(k)] == lab for k, (_, lab) in zip(pred, held, strict=True)]))
    return Fitted(head.to_bytes(), len(train), acc, time.perf_counter() - t0)
