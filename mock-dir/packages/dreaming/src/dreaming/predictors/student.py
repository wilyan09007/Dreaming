"""The student: an encoder plus a linear head. Picks one of the site's labels in a single forward pass.

Weights file = the head (W, b, labels) plus the encoder's weights sha, serialized as one .npz, so weights_sha
names the whole student. The encoder embeds one text at a time (batch 1, fixed max length, 1 thread) and caches
vectors by prompt_sha: fast-forward re-uses them, replay recomputes and compares bytes.
"""

from __future__ import annotations

import importlib
import io
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from ..contracts import Hashes, Prediction
from ..gate import CONF_DECIMALS
from ..hashing import prompt_sha, sha_bytes
from . import profile_sha

Vec = NDArray[np.float32]
ENCODER = "BAAI/bge-small-en-v1.5"  # 33M params, 384-d, the student ladder's winner (S8 v2)
ENCODER_REV = os.environ.get("ENCODER_REV", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")  # hub commit checked 2026-10-17 (smokes/RESULTS.md); one setting for every process


class Encoder(Protocol):
    model_sha: str  # sha256 of the encoder's safetensors file: the same file hashes the same on Modal and on Fly
    env_sha: str

    def embed(self, text: str, *, fresh: bool = False) -> Vec: ...


class SentenceEncoder:
    """sentence-transformers on CPU, batch size 1, pinned threads. Vectors are L2-normalized float32.

    `model` is a hub id (pinned by `revision`) or a local directory (the trainer image bakes one). Either way the
    weights file itself is hashed. Vectors are cached by prompt sha and, with `cache_path`, persisted across restarts:
    fast-forward never embeds a text twice, and `just record` can fill the cache for the whole ledger ahead of a demo.
    """

    def __init__(self, model: str = ENCODER, revision: str | None = ENCODER_REV, max_length: int = 128, threads: int = 1,
                 cache_path: Path | None = None) -> None:
        torch: Any = importlib.import_module("torch")  # optional extra "student"; untyped
        st: Any = importlib.import_module("sentence_transformers")

        torch.set_num_threads(threads)
        self.model: Any = st.SentenceTransformer(model, device="cpu", revision=revision)
        self.model.max_seq_length = max_length
        local = Path(model) / "model.safetensors"
        if local.is_file():
            weights = local
        else:  # the hub copy sentence-transformers just loaded, from the local HF cache (no second download)
            hub: Any = importlib.import_module("huggingface_hub")
            weights = Path(hub.hf_hub_download(model, "model.safetensors", revision=revision))
        self.model_sha = sha_bytes(weights.read_bytes())
        self.env_sha = profile_sha(encoder=self.model_sha, revision=revision, max_length=max_length, torch=torch.__version__, threads=threads)
        self.cache_path = cache_path
        self._cache: dict[str, Vec] = {}
        if cache_path and cache_path.is_file():
            with np.load(cache_path) as z:
                self._cache = {k: z[k] for k in z.files}

    def embed(self, text: str, *, fresh: bool = False) -> Vec:
        key = prompt_sha(text)
        v = None if fresh else self._cache.get(key)
        if v is None:
            out: Any = self.model.encode([text], batch_size=1, normalize_embeddings=True, convert_to_numpy=True)
            v = np.asarray(out, dtype=np.float32)[0]
            if not fresh:
                self._cache[key] = v
        return v

    def save_cache(self) -> None:
        if self.cache_path:
            cast(Any, np.savez)(self.cache_path, **self._cache)


@dataclass(frozen=True)
class Head:
    labels: list[str]
    W: NDArray[np.float32]  # (dim, n_labels)
    b: NDArray[np.float32]  # (n_labels,)
    encoder_sha: str

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        np.savez(buf, W=self.W, b=self.b, labels=np.array(self.labels), encoder_sha=np.array(self.encoder_sha))
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> Head:
        z = np.load(io.BytesIO(data))
        return cls([str(x) for x in z["labels"]], z["W"].astype(np.float32), z["b"].astype(np.float32), str(z["encoder_sha"]))


class Student:
    def __init__(self, encoder: Encoder, head: Head, weights_sha: str) -> None:
        if head.encoder_sha != encoder.model_sha:
            raise ValueError("head was trained on a different encoder; a certificate on it would be meaningless")
        self.encoder, self.head = encoder, head
        self.weights_sha = weights_sha  # the sha of the stored bytes (npz is not byte-stable across writes; the store's copy is)
        self.env_sha = encoder.env_sha

    def predict(self, rendered: str, *, fresh: bool = False) -> Prediction:
        x = self.encoder.embed(rendered, fresh=fresh)
        z = x @ self.head.W + self.head.b
        p = np.exp(z - z.max())
        p /= p.sum()
        order = np.argsort(-p)
        conf = round(float(p[order[0]] - p[order[1]]), CONF_DECIMALS) if len(p) > 1 else 1.0
        return Prediction(
            label=self.head.labels[int(order[0])], confidence=conf,
            scores={lab: round(float(s), 4) for lab, s in zip(self.head.labels, p, strict=True)},
            hashes=Hashes(prompt=prompt_sha(rendered), weights=self.weights_sha, env=self.env_sha),
        )


def fit_head(X: NDArray[np.float32], y: Sequence[str], labels: Sequence[str], encoder_sha: str, l2: float = 0.1) -> Head:
    """Multinomial logistic regression on frozen embeddings (S1), fitted with L-BFGS. Rows with labels outside `labels` are dropped.

    scipy is already a dependency (the gate), so no sklearn: same objective as sklearn's lbfgs solver, C = 1/l2.
    """
    _minimize: Any = importlib.import_module("scipy.optimize").minimize  # scipy ships no types
    idx = {lab: k for k, lab in enumerate(labels)}
    keep = [i for i, lab in enumerate(y) if lab in idx]
    Xk = np.asarray(X[keep], dtype=np.float64)
    yk = np.array([idx[y[i]] for i in keep])
    n, d, k = Xk.shape[0], Xk.shape[1], len(labels)
    onehot = np.zeros((n, k))
    onehot[np.arange(n), yk] = 1.0

    def objective(theta: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        W, b = theta[: d * k].reshape(d, k), theta[d * k :]
        z = Xk @ W + b
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        loss = -np.log(p[np.arange(n), yk] + 1e-12).sum() + 0.5 * l2 * (W * W).sum()
        g = p - onehot
        return float(loss), np.concatenate([(Xk.T @ g + l2 * W).ravel(), g.sum(axis=0)])

    res: Any = _minimize(objective, np.zeros(d * k + k), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    theta: NDArray[np.float32] = np.asarray(res.x, dtype=np.float32)
    return Head(list(labels), theta[: d * k].reshape(d, k), theta[d * k :], encoder_sha)
