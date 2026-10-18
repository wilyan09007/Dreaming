"""The `memory` candidate: a Hopfield read over stored (embedding, label) rows. No weights, only the rows and a beta.

    fit(site, rows, encoder) -> Fitted          embed every labelled row, store unit vectors + labels + beta in one .npz
    load(site, data, encoder) -> Predictor      the same bytes back as a Memory predictor
    Memory.predict(rendered) -> Prediction      softmax(beta * q.X^T) over the stored rows, summed per label

Port of S27 `hopfield_head` (smokes/s27_hybrid_embedder_memory). Rows are stored sorted by (label, text) and read in
float64, so the same rows give the same bytes and the same (label, quantized confidence) wherever they are embedded.
"""

from __future__ import annotations

import io
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..contracts import CallSite, Hashes, Prediction, TraceRow
from ..gate import confidence
from ..hashing import prompt_sha, sha_bytes
from . import profile_sha
from .candidates import Candidate, Fitted, register
from .student import Encoder

F64 = NDArray[np.float64]
BETAS = (5, 10, 20, 40, 80, 160, 320, 640)  # S24's grid; the winner is picked per site on leave-one-out NLL


def _unit(v: NDArray[np.floating[Any]]) -> F64:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n else v


def _read(S: F64, Y: F64, beta: float) -> F64:
    """S (queries, stored) similarities, Y (stored, labels) one-hot -> (queries, labels) probabilities; no rows: 0."""
    z = beta * S
    p = np.exp(z - z.max(axis=1, keepdims=True)) @ Y  # = per-label logsumexp of beta*s, normalized
    return p / p.sum(axis=1, keepdims=True)


def fit(site: CallSite, rows: Sequence[TraceRow], encoder: Encoder | None) -> Fitted:
    from .train import labelled  # train imports candidates, which imports this module: import it late

    t0 = time.perf_counter()
    if encoder is None or site.labels is None:
        raise ValueError(f"{site.id}: a model step needs an encoder and a label set")
    idx = {lab: k for k, lab in enumerate(site.labels)}
    # rows labelled outside the set are dropped, like fit_head; sorted so the same rows give the same bytes
    pairs = sorted((p for p in labelled(rows) if p[1] in idx), key=lambda p: (p[1], p[0]))
    if not pairs:
        raise ValueError(f"{site.id}: no labelled rows to train on (did every real call fail?)")
    X = np.stack([_unit(encoder.embed(text)).astype(np.float32) for text, _ in pairs])
    y = np.array([idx[lab] for _, lab in pairs], dtype=np.int64)
    Y, S = np.eye(len(site.labels))[y], X.astype(np.float64) @ X.astype(np.float64).T
    acc = 1.0
    beta = float(BETAS[0])
    if len(pairs) > 1:
        # shortcut: leave-one-out on the stored rows, N x N floats (fine to ~5k rows); use a 20% holdout above that.
        # Duplicate inputs flatter it, as they do any random holdout.
        np.fill_diagonal(S, -np.inf)
        nll = [-np.log(_read(S, Y, b)[np.arange(len(y)), y] + 1e-9).mean() for b in BETAS]
        beta = float(BETAS[int(np.argmin(nll))])
        acc = float((_read(S, Y, beta).argmax(axis=1) == y).mean())
    buf = io.BytesIO()
    np.savez(buf, X=X, y=y, labels=np.array(site.labels), beta=np.array(beta), encoder_sha=np.array(encoder.model_sha))
    return Fitted(buf.getvalue(), len(pairs), acc, time.perf_counter() - t0)


class Memory:
    def __init__(self, encoder: Encoder, data: bytes) -> None:
        z = np.load(io.BytesIO(data))
        if str(z["encoder_sha"]) != encoder.model_sha:
            msg = f"memory stored with encoder {z['encoder_sha']}, got {encoder.model_sha}: certificate void"
            raise ValueError(msg)
        self.encoder, self.labels, self.beta = encoder, [str(x) for x in z["labels"]], float(z["beta"])
        self.X, self.Y = z["X"].astype(np.float64), np.eye(len(self.labels))[z["y"]]
        self.weights_sha = sha_bytes(data)  # the stored bytes' sha (npz is not byte-stable across writes)
        self.env_sha = profile_sha(encoder=encoder.env_sha, candidate="memory", beta=self.beta)

    def predict(self, rendered: str, *, fresh: bool = False) -> Prediction:
        q = _unit(self.encoder.embed(rendered, fresh=fresh))
        p = _read((self.X @ q)[None, :], self.Y, self.beta)[0]
        return Prediction(
            label=self.labels[int(np.argmax(p))],
            confidence=float(confidence(p[None, :])[0]) if len(p) > 1 else 1.0,
            scores={lab: round(float(s), 4) for lab, s in zip(self.labels, p, strict=True)},
            hashes=Hashes(prompt=prompt_sha(rendered), weights=self.weights_sha, env=self.env_sha),
        )


def load(site: CallSite, data: bytes, encoder: Any) -> Memory:
    if encoder is None:
        raise ValueError(f"{site.id}: the memory candidate needs an encoder")
    return Memory(encoder, data)


register(Candidate("memory", fit, load))
