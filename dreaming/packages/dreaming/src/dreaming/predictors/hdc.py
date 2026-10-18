"""The `hdc` candidate: one integer bundle per class over 10,000-bit codes. Compaction under the memory (training-plan §4.4).

    fit(site, rows, encoder) -> Fitted          code every labelled row, sum the codes per class, store bundles in one .npz
    load(site, data, encoder) -> Predictor      the same bytes back as an HDC predictor
    HDC.predict(rendered) -> Prediction         argmax of bundle . code / rows; confidence = top-2 gap / (|top| + |second|)

Port of S25 `HDC` (smokes/s25_exotic/arms.py) and S29 `hdc_fit` (smokes/s29_memory_student/run.py). A row's code is
sign(x . P) as int8, P a fixed random ±1 matrix (dim x 10,000) regenerated from numpy seed 0 and never stored. Everything
after the encoder and that one projection is integer math, so the same embedding gives the same bytes on any hardware;
a class's memory is 10,000 integers however many rows it has, and a row is added or removed in 10,000 integer ops.
The score divides by the class's row count, as an exact fraction: the raw sum favours the biggest class (0.67 vs 0.97
leave-one-out on the 60-row test fakes) and the normalized bundle needs a float sqrt.
"""

from __future__ import annotations

import io
import time
from collections.abc import Sequence
from fractions import Fraction
from functools import cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..contracts import CallSite, Hashes, Prediction, TraceRow
from ..gate import CONF_DECIMALS
from ..hashing import prompt_sha, sha_bytes
from . import profile_sha
from .candidates import Candidate, Fitted, register
from .student import Encoder

D = 10_000
I64 = NDArray[np.int64]


@cache
def _proj(dim: int) -> NDArray[np.float32]:
    """The fixed ±1 projection, dim x D (384 x 10,000 for bge-small), regenerated from seed 0."""
    return np.where(np.random.default_rng(0).standard_normal((dim, D), dtype=np.float32) >= 0, 1.0, -1.0).astype(np.float32)


def code(v: NDArray[np.floating[Any]]) -> NDArray[np.int8]:
    """Embedding(s) -> bipolar code(s), +1/-1 as int8."""
    x = np.asarray(v, dtype=np.float32)
    return np.where(x @ _proj(x.shape[-1]) >= 0, 1, -1).astype(np.int8)


def _scores(s: I64, n: I64) -> list[Fraction | None]:
    """Bundle . code per class over the class's row count, exactly; None for a class with no rows (it never wins)."""
    return [Fraction(int(a), int(c)) if c > 0 else None for a, c in zip(s, n, strict=True)]


def _gap(f: list[Fraction | None]) -> tuple[int, float]:
    """Exact scores of one query -> (argmax, first index on ties; top-2 gap normalized to [0, 1] and quantized)."""
    live = sorted(((v, -k) for k, v in enumerate(f) if v is not None), reverse=True)
    if len(live) < 2:
        return (-live[0][1], 1.0) if live else (0, 0.0)
    (a, k), (b, _) = live[0], live[1]
    den = abs(a) + abs(b)
    return -k, round(float((a - b) / den), CONF_DECIMALS) if den else 0.0


def fit(site: CallSite, rows: Sequence[TraceRow], encoder: Encoder | None) -> Fitted:
    from .train import labelled  # train imports candidates, which imports this module: import it late

    t0 = time.perf_counter()
    if encoder is None or site.labels is None:
        raise ValueError(f"{site.id}: a model step needs an encoder and a label set")
    idx = {lab: k for k, lab in enumerate(site.labels)}
    pairs = [p for p in labelled(rows) if p[1] in idx]  # rows labelled outside the set are dropped, like fit_head
    if not pairs:
        raise ValueError(f"{site.id}: no labelled rows to train on (did every real call fail?)")
    H = np.stack([code(encoder.embed(text)) for text, _ in pairs]).astype(np.int64)  # one row at a time, as predict does
    y = np.array([idx[lab] for _, lab in pairs], dtype=np.int64)
    K = len(site.labels)
    B = np.zeros((K, D), np.int64)
    np.add.at(B, y, H)  # the bundle: an integer sum, so row order cannot change the bytes
    n = np.bincount(y, minlength=K)
    # leave-one-out, exactly: take each row's own code out of its bundle (code . code = D) and drop classes left empty
    S, N = H @ B.T, np.broadcast_to(n, (len(y), K)).copy()
    S[np.arange(len(y)), y] -= D
    N[np.arange(len(y)), y] -= 1
    acc = float(np.mean([_gap(_scores(S[i], N[i]))[0] == y[i] for i in range(len(y))])) if len(y) > 1 else 1.0
    buf = io.BytesIO()
    np.savez(buf, bundles=B.astype(np.int32), counts=n, labels=np.array(site.labels), encoder_sha=np.array(encoder.model_sha))
    return Fitted(buf.getvalue(), len(pairs), acc, time.perf_counter() - t0)


class HDC:
    def __init__(self, encoder: Encoder, data: bytes) -> None:
        z = np.load(io.BytesIO(data))
        if str(z["encoder_sha"]) != encoder.model_sha:
            raise ValueError(f"hdc bundles stored with encoder {z['encoder_sha']}, got {encoder.model_sha}: certificate void")
        self.encoder, self.labels = encoder, [str(x) for x in z["labels"]]
        self.B: I64 = z["bundles"].astype(np.int64)
        self.n: I64 = z["counts"].astype(np.int64)
        self.weights_sha = sha_bytes(data)  # the stored bytes' sha (npz is not byte-stable across writes)
        self.env_sha = profile_sha(encoder=encoder.env_sha, candidate="hdc", dim=D)

    def predict(self, rendered: str, *, fresh: bool = False) -> Prediction:
        s = self.B @ code(self.encoder.embed(rendered, fresh=fresh)).astype(np.int64)
        f = _scores(s, self.n)
        k, conf = _gap(f)
        return Prediction(
            label=self.labels[k],
            confidence=conf,
            scores={lab: round(float(v or 0), 4) for lab, v in zip(self.labels, f, strict=True)},
            hashes=Hashes(prompt=prompt_sha(rendered), weights=self.weights_sha, env=self.env_sha),
        )


def load(site: CallSite, data: bytes, encoder: Any) -> HDC:
    if encoder is None:
        raise ValueError(f"{site.id}: the hdc candidate needs an encoder")
    return HDC(encoder, data)


register(Candidate("hdc", fit, load))
