"""The `memory` candidate: a Hopfield read over stored (embedding, label) rows. No trained weights: the rows and one beta.

    fit(site, rows, encoder) -> Fitted          embed every labelled row, pick beta, serialize rows sorted by row id
    load(site, data, encoder) -> Memory         the same bytes back; refuses a foreign encoder
    Memory.predict(rendered) -> Prediction      p_c = sum_{i in c} softmax(beta * cos(q, x_i)) / n_c, renormalized; confidence = top-2 gap

Port of S27 `hopfield_head` / S24 `hopfield` to docs/training-plan.md §4.1 (root training-plan.md §4.1):
- beta from {5..640} by NLL. fit has no calibration rows (those arrive after it, while the site shadows), so the NLL is
  leave-one-out over the stored rows (at most 2,000 of them as queries, every k-th, so the matrix stays O(2,000 x rows)).
  shortcut: a repeated merchant flatters leave-one-out toward a high beta (80-320 picked; the best beta on unseen merchants reads
  0.01-0.06 more accurately); repeat-heavy live traffic wants exactly that. Upgrade: re-pick on the calibration rows at checkpoints.
- Imbalance: the class sum is divided by the class's row count (S29's "Hopfield/count"; S29 never ran). Measured on bookkeeper
  `categorize`, bge-small, 10 merchant holdouts: accuracy on unseen merchants 0.30 vs 0.25 plain vs 0.25 top-5 cap (StepFun labels,
  s.e. 0.025), 0.28 / 0.27 / 0.27 (rule labels); tied within s.e. once seed rows are in.
- Bytes: a sorted-key JSON header (row ids sorted, labels, beta, encoder_sha) + the unit float32 vectors in row-id order. The
  same rows give the same bytes in any order, so weights_sha = sha(sorted row ids, beta, encoder_sha, the vectors): appending or
  deleting a row (= refit without it; seconds, no training) is new weights that learning re-gates. The vectors are stored, not
  re-embedded at load, so a Modal fit loads on Fly without thousands of encoder passes; they change only if the encoder's output
  does, which voids a certificate anyway.
- Read in float64 in row-id order; confidence quantized by gate.confidence. Seed rows (predictors/seeds.py) are ordinary rows
  with id "seed:<label>:<i>"; the caller passes them in with the trace rows.
"""

from __future__ import annotations

import json
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
BETAS = (5, 10, 20, 40, 80, 160, 320, 640)  # S24's grid
LOO_QUERIES = 2000


def _read(S: F64, Y: F64, beta: float) -> F64:
    """S (queries, rows) cosines, Y (rows, labels) one-hot -> (queries, labels): softmax mass per label over its row count."""
    z = beta * S
    n = Y.sum(axis=0)
    p = (np.exp(z - z.max(axis=1, keepdims=True)) @ Y) / np.maximum(n, 1.0)  # a label with no rows scores 0
    return p / p.sum(axis=1, keepdims=True)


def _pick_beta(X: F64, Y: F64, y: NDArray[np.int64]) -> tuple[float, float]:
    """(beta, leave-one-out accuracy at it). Every k-th row is a query against all the others."""
    if len(y) < 2:
        return float(BETAS[0]), 1.0
    q = np.arange(0, len(y), -(-len(y) // LOO_QUERIES))
    S = X[q] @ X.T
    S[np.arange(len(q)), q] = -np.inf  # leave the query's own row out
    reads = [_read(S, Y, b) for b in BETAS]
    nll = [-np.log(r[np.arange(len(q)), y[q]] + 1e-9).mean() for r in reads]
    k = int(np.argmin(nll))
    return float(BETAS[k]), float((reads[k].argmax(axis=1) == y[q]).mean())


def fit(site: CallSite, rows: Sequence[TraceRow], encoder: Encoder | None) -> Fitted:
    from .train import labelled  # train imports candidates, which imports this module: import it late

    t0 = time.perf_counter()
    if encoder is None or site.labels is None:
        raise ValueError(f"{site.id}: a model step needs an encoder and a label set")
    idx = {lab: k for k, lab in enumerate(site.labels)}
    # rows labelled outside the set are dropped, like fit_head; one row per id, in id order
    by_id = {r.id: pair for r in rows for pair in labelled([r]) if pair[1] in idx}
    if not by_id:
        raise ValueError(f"{site.id}: no labelled rows to train on (did every real call fail?)")
    ids = sorted(by_id)
    raw = np.stack([encoder.embed(by_id[i][0]) for i in ids]).astype(np.float64)
    X = (raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)).astype(np.float32)  # stored as float32, read as float64
    y = np.array([idx[by_id[i][1]] for i in ids], dtype=np.int64)
    beta, acc = _pick_beta(X.astype(np.float64), np.eye(len(site.labels))[y], y)
    head = {"ids": ids, "y": y.tolist(), "labels": site.labels, "beta": beta, "encoder_sha": encoder.model_sha, "dim": int(X.shape[1])}
    data = json.dumps(head, sort_keys=True, separators=(",", ":")).encode() + b"\n" + X.astype("<f4").tobytes()
    return Fitted(data, len(ids), acc, time.perf_counter() - t0)


class Memory:
    def __init__(self, encoder: Encoder, data: bytes) -> None:
        raw, _, vecs = data.partition(b"\n")  # json.dumps escapes newlines, so the first one ends the header
        h: dict[str, Any] = json.loads(raw)
        if h["encoder_sha"] != encoder.model_sha:
            raise ValueError(f"memory stored with encoder {h['encoder_sha']}, got {encoder.model_sha}: certificate void")
        self.encoder, self.labels, self.beta, self.ids = encoder, [str(x) for x in h["labels"]], float(h["beta"]), [str(i) for i in h["ids"]]
        self.X = np.frombuffer(vecs, dtype="<f4").reshape(len(self.ids), int(h["dim"])).astype(np.float64)
        self.Y = np.eye(len(self.labels))[np.array(h["y"], dtype=np.int64)]
        self.weights_sha = sha_bytes(data)
        self.env_sha = profile_sha(encoder=encoder.env_sha, candidate="memory", beta=self.beta)

    def predict(self, rendered: str, *, fresh: bool = False) -> Prediction:
        q = np.asarray(self.encoder.embed(rendered, fresh=fresh), dtype=np.float64)
        q /= max(float(np.linalg.norm(q)), 1e-12)
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
