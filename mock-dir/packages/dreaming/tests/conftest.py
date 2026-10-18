"""Fakes that make the whole loop run offline in a second: a keyword teacher, a bag-of-characters encoder."""

from __future__ import annotations

import zlib

import numpy as np
import pytest
from dreaming import registry
from dreaming.contracts import CallSite, TeacherAnswer
from dreaming.predictors.student import Vec

LABELS = ["Software", "Meals", "Travel"]
WORDS = {"Software": ["ADOBE", "NOTION", "FIGMA", "GOOGLE"], "Meals": ["BLUE BOTTLE", "DOORDASH", "SWEETGREEN"], "Travel": ["UBER", "LYFT", "DELTA"]}


class KeywordTeacher:
    """Deterministic 'frontier model': the label whose keyword appears. Costs a cent and 800 ms, like Haiku."""

    def ask(self, site: CallSite, rendered: str) -> TeacherAnswer:
        label = next((lab for lab, ws in WORDS.items() if any(w in rendered for w in ws)), "Meals")
        return TeacherAnswer(text=label, label=label, provider="fake", model="keyword", cost_usd=0.01, latency_ms=800)


class CharEncoder:
    """Bag of character trigrams, hashed into 256 dims, L2-normalized. Byte-deterministic by construction."""

    model_sha = "charenc-v1"
    env_sha = "test-env"

    def embed(self, text: str, *, fresh: bool = False) -> Vec:
        v = np.zeros(256, np.float32)
        t = text.upper()
        for k in range(len(t) - 2):
            v[zlib.crc32(t[k : k + 3].encode()) % 256] += 1.0  # crc32, not hash(): stable across processes
        n = float(np.linalg.norm(v))
        return v / n if n else v


def transactions(n: int, seed: int = 0) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    merchants = [w for ws in WORDS.values() for w in ws]
    return [{"d": f"{merchants[int(rng.integers(len(merchants)))]} *{int(rng.integers(1000, 9999))}", "amt": round(float(rng.uniform(5, 500)), 2)} for _ in range(n)]


@pytest.fixture(autouse=True)
def clean_registry():
    registry._SITES.clear()  # pyright: ignore[reportPrivateUsage]
    registry.bind(None)
    yield
    registry._SITES.clear()  # pyright: ignore[reportPrivateUsage]
    registry.bind(None)
