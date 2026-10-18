"""The finetune candidate, offline: it registers as a GPU job, refuses to run without torch, and its artifact is byte-stable."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from dreaming.contracts import CallSite, Kind
from dreaming.predictors import finetune
from dreaming.predictors.candidates import CANDIDATES
from dreaming.predictors.student import Head


def test_registered_as_a_gpu_job():
    c = CANDIDATES["finetune"]
    assert (c.fit, c.load, c.needs_encoder, c.needs_gpu) == (finetune.fit, finetune.load, True, True)


def test_fit_without_torch_says_so(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # what a missing install looks like to `import`
    site = CallSite(id="w.a", workflow="w", name="a", kind=Kind.model, labels=["x", "y"])
    with pytest.raises(RuntimeError, match="needs torch"):
        finetune.fit(site, [], None)


def fake_encoder(root: Path) -> Path:
    for name, blob in {"model.safetensors": b"\x00weights\xff" * 100, "config.json": b"{}", "1_Pooling/config.json": b'{"cls": true}'}.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(blob)
    return root


def test_artifact_round_trip_is_byte_stable(tmp_path: Path):
    head = Head(["x", "y"], np.arange(8, dtype=np.float32).reshape(4, 2), np.zeros(2, np.float32), "sha").to_bytes()
    enc = fake_encoder(tmp_path / "enc")
    data = finetune.pack(enc, head)
    for p in enc.rglob("*"):
        os.utime(p, (1, 1))  # file times must not reach the bytes
    assert finetune.pack(enc, head) == data

    out = tmp_path / "out"
    finetune.unpack(data, out)
    assert (out / "head.npz").read_bytes() == head
    assert sorted(p.relative_to(out / "encoder").as_posix() for p in (out / "encoder").rglob("*") if p.is_file()) == [
        "1_Pooling/config.json", "config.json", "model.safetensors"]
    assert finetune.pack(out / "encoder", (out / "head.npz").read_bytes()) == data  # unpack then pack is the identity
