"""The `finetune` candidate: bge-small unfrozen and trained end to end with a linear head (S8 v2 rung 1). A GPU job.

    fit(site, rows, encoder) -> Fitted      tune a copy of the encoder on the teacher's labels (GPU), then fit the Head on its
                                            CPU embeddings, the same inference path `load` serves; hold out 20%
    load(site, data, encoder) -> Predictor  the bytes back as a Student over the unpacked encoder (CPU, batch 1, 1 thread)
    pack(encoder_dir, head) -> bytes        the artifact: a tar of `encoder/` (Hugging Face layout) and `head.npz` (Head.to_bytes)
    unpack(data, dest)                      the reverse; the veri.studio job writes the same layout (docs/training-plan.md §3)

Importing this module registers the candidate. torch is imported inside `fit` (and by SentenceEncoder in `load`), so the
import works without it. GPU training is not bitwise deterministic; what is certified is the CPU path of the saved weights.
"""

from __future__ import annotations

import copy
import importlib
import io
import tarfile
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import CallSite, TraceRow
from ..hashing import sha_bytes
from . import Predictor
from .candidates import Candidate, Fitted, register
from .student import Head, SentenceEncoder, Student, fit_head
from .train import labelled

BATCH, MAX_LEN, MIN_STEPS, MAX_STEPS = 32, 128, 300, 600  # S8 v2: max(300, 8 epochs); capped so 2,000 rows stay under ~2 min on a T4


def _tune(torch: Any, model: Any, train: list[tuple[str, str]], labels: list[str]) -> None:
    """S8 v2 `ft_encoder`: AdamW, encoder lr 5e-5, head lr 1e-3, warmup then linear decay, cross-entropy on unit embedding x 20."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"  # shortcut: fp32, one device; amp or two GPUs when a cell outgrows a T4
    torch.manual_seed(0)
    model.max_seq_length = MAX_LEN
    model.to(dev).train()
    head = torch.nn.Linear(model.get_sentence_embedding_dimension(), len(labels)).to(dev)
    per_epoch = -(-len(train) // BATCH)
    steps = min(MAX_STEPS, max(MIN_STEPS, 8 * per_epoch))
    opt = torch.optim.AdamW([{"params": model.parameters(), "lr": 5e-5}, {"params": head.parameters(), "lr": 1e-3}], weight_decay=0.01)

    def scale(s: int) -> float:
        return min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, (steps - s) / steps)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, scale)
    y = torch.tensor([labels.index(lab) for _, lab in train], device=dev)
    perm: list[int] = []
    for step in range(steps):
        if step % per_epoch == 0:
            perm = torch.randperm(len(train)).tolist()
        j = perm[(step % per_epoch) * BATCH :][:BATCH]
        feats = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in model.tokenize([train[i][0] for i in j]).items()}  # v6 adds str entries
        loss = torch.nn.functional.cross_entropy(head(model(feats)["sentence_embedding"] * 20), y[j])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    model.eval().to("cpu")  # the head above is thrown away: the CPU path refits the linear head on the saved encoder's embeddings


def pack(encoder_dir: Path, head: bytes) -> bytes:
    """One blob. Entries sorted, mtime 0, no owner, GNU format (no pax timestamps): the same weights give the same bytes."""
    files = [(f"encoder/{p.relative_to(encoder_dir).as_posix()}", p.read_bytes()) for p in encoder_dir.rglob("*") if p.is_file()]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for name, blob in sorted([*files, ("head.npz", head)]):  # "head.npz" sorts after "encoder/": `load` relies on it being last
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(blob), 0o644
            tar.addfile(info, io.BytesIO(blob))
    return buf.getvalue()


def unpack(data: bytes, dest: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(dest, filter="data")  # the "data" filter refuses absolute paths and links: veri's download is not ours


def fit(site: CallSite, rows: Sequence[TraceRow], encoder: Any) -> Fitted:
    try:
        torch: Any = importlib.import_module("torch")
    except ImportError as e:
        raise RuntimeError("the finetune candidate needs torch and sentence-transformers (the 'student' extra; the trainer image has them)") from e
    t0 = time.perf_counter()
    labels = site.labels or []
    pairs = [(text, lab) for text, lab in labelled(rows) if lab in labels]
    model: Any = getattr(encoder, "model", None)
    if not pairs or model is None:
        raise ValueError(f"{site.id}: finetune needs labelled rows and a SentenceEncoder to start from")
    idx = [int(i) for i in np.random.default_rng(0).permutation(len(pairs))]  # the split fit_head_from_rows draws
    cut = max(1, int(len(pairs) * 0.8)) if len(pairs) > 5 else len(pairs)
    train, held = [pairs[i] for i in idx[:cut]], [pairs[i] for i in idx[cut:]]
    model = copy.deepcopy(model)  # the learner's encoder stays frozen and shared
    _tune(torch, model, train, labels)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # ignore: Windows keeps the safetensors mapped
        model.save(f"{tmp}/encoder", create_model_card=False)  # the card embeds versions and a date: not byte-stable
        enc = SentenceEncoder(f"{tmp}/encoder", revision=None, max_length=MAX_LEN)  # CPU, batch 1: what `load` runs
        head = fit_head(np.stack([enc.embed(t) for t, _ in train]), [lab for _, lab in train], labels, enc.model_sha)
        acc = 1.0
        if held:
            pred = np.asarray(np.stack([enc.embed(t) for t, _ in held]) @ head.W + head.b).argmax(axis=1)
            acc = float(np.mean([labels[int(k)] == lab for k, (_, lab) in zip(pred, held, strict=True)]))
        data = pack(Path(tmp) / "encoder", head.to_bytes())
    return Fitted(data, len(train), acc, time.perf_counter() - t0)


def load(site: CallSite, data: bytes, encoder: Any) -> Predictor:
    """`encoder` (the shared base) is unused: this candidate carries its own."""
    root = Path(tempfile.gettempdir()) / "dreaming-finetune" / sha_bytes(data)  # shortcut: never evicted; ~130 MB per version
    if not (root / "head.npz").is_file():  # the last entry in the tar, so its presence means the unpack finished
        unpack(data, root)
    enc = SentenceEncoder(str(root / "encoder"), revision=None, max_length=MAX_LEN)
    return Student(enc, Head.from_bytes((root / "head.npz").read_bytes()), sha_bytes(data))


register(Candidate("finetune", fit, load, needs_encoder=True, needs_gpu=True))
