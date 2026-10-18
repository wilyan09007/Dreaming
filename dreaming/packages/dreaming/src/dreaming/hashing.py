"""Canonical JSON and the three hashes every predicted row carries: prompt, weights, environment.

The rendered prompt is hashed, never the template. Weights are hashed as the one file that answered.
The environment is a dict of the things that change bytes (S7, determinism.md): image digest or package versions,
thread count, batch size, hardware class. Changing any of them sends a call site back through the gate.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical(obj: Any) -> bytes:
    """Deterministic JSON: sorted keys, no whitespace, NaN rejected, non-JSON values via str()."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode()


def sha(obj: Any) -> str:
    """sha256 of canonical JSON. The request_sha of a trace row and the calib_sha of a certificate use this."""
    return hashlib.sha256(canonical(obj)).hexdigest()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prompt_sha(rendered: str) -> str:
    return sha_bytes(rendered.encode())


def env_sha(profile: dict[str, Any]) -> str:
    """profile example: {"image": "sha256:…", "numpy": "2.1.0", "torch": "2.5.0", "threads": 1, "batch": 1, "device": "cpu"}"""
    return sha(profile)


def short(h: str | None) -> str:
    """Six characters, what the dashboard prints next to a decision."""
    return (h or "")[:6]
