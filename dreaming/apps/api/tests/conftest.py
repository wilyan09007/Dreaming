"""Offline fixtures for the API tests.

The fakes and `build()` follow packages/dreaming/tests (conftest.py, test_runtime.py): that directory is on sys.path only
when pytest runs there, and a second module called `conftest` cannot be imported by name from here. Differences:
the teacher can say `recorded=True`, as RecordedTeacher does in fast-forward mode, and the encoder reads only the
merchant words, so categorize is certified well inside 600 runs whatever the library's own fake does.
"""

from __future__ import annotations

import re
import threading
import zlib
from collections.abc import Callable, Generator, Iterable

import numpy as np
import pytest
from dreaming import Runtime, registry
from dreaming.contracts import CallSite, Component, Json, TeacherAnswer, Thresholds
from dreaming.predictors import Loader
from dreaming.predictors.student import Vec
from dreaming.predictors.train import LocalTrainer
from dreaming.store import MemoryStore
from dreaming_api.main import create_app
from fastapi.testclient import TestClient

LABELS = ["Software", "Meals", "Travel"]
WORDS = {"Software": ["ADOBE", "NOTION", "FIGMA", "GOOGLE"], "Meals": ["BLUE BOTTLE", "DOORDASH", "SWEETGREEN"], "Travel": ["UBER", "LYFT", "DELTA"]}


class KeywordTeacher:
    """Deterministic 'frontier model': the label whose keyword appears. Costs a cent and 800 ms, like Haiku."""

    def __init__(self, recorded: bool = False) -> None:
        self.recorded = recorded

    def ask(self, site: CallSite, rendered: str) -> TeacherAnswer:
        label = next((lab for lab, ws in WORDS.items() if any(w in rendered for w in ws)), "Meals")
        return TeacherAnswer(text=label, label=label, provider="fake", model="keyword", cost_usd=0.01, latency_ms=800, recorded=self.recorded)


class CharEncoder:
    """Bag of character trigrams of the merchant words (no ids, no amounts), hashed into 64 dims, L2-normalized."""

    model_sha = "charenc-v1"
    env_sha = "test-env"

    def embed(self, text: str, *, fresh: bool = False) -> Vec:
        v = np.zeros(256, np.float32)
        t = re.sub(r"[^A-Z ]", "", text.upper().split(",")[0])
        for k in range(len(t) - 2):
            v[zlib.crc32(t[k : k + 3].encode()) % 256] += 1.0  # crc32, not hash(): stable across processes
        n = float(np.linalg.norm(v))
        return v / n if n else v


def transactions(n: int, seed: int = 0) -> list[dict[str, Json]]:
    rng = np.random.default_rng(seed)
    merchants = [w for ws in WORDS.values() for w in ws]
    return [{"d": f"{merchants[int(rng.integers(len(merchants)))]} *{int(rng.integers(1000, 9999))}", "amt": round(float(rng.uniform(5, 500)), 2)} for _ in range(n)]


DOWN = {"vendor": False}  # the fake vendor API's fault switch; create_app's `inject` flips it


def inject(site: str, down: bool) -> None:
    DOWN["vendor"] = down  # nobody tells the runtime: it detects the outage from the failures


def build(store: MemoryStore, teacher: KeywordTeacher) -> Runtime:
    """The same two-site workflow as packages/dreaming/tests/test_runtime.py::build (fallback always registered)."""
    DOWN["vendor"] = False
    loader = Loader(encoder=CharEncoder())
    rt = Runtime(store, teacher, LocalTrainer(store, loader), loader, component=Component.api, budget=0.05)

    @registry.step("categorize", workflow="books", labels=LABELS, thresholds=Thresholds(watch_after=60, calibrate_after=200))
    def categorize(txn: dict[str, object]) -> str:
        return f"Transaction: {txn['d']}, ${txn['amt']}"

    def snapshot_lookup(d: str) -> str:  # the registered fallback: last night's snapshot
        return {"ADOBE": "Adobe Inc."}.get(d.split()[0], "unknown")

    @registry.tool("vendor_lookup", workflow="books", fallback=snapshot_lookup, thresholds=Thresholds(watch_after=60, calibrate_after=200))
    def vendor_lookup(descriptor: str) -> str:
        if DOWN["vendor"]:
            raise ConnectionError("vendor API down")
        return descriptor.split()[0].title() + " Inc."

    def workflow(txn: Json) -> None:
        assert isinstance(txn, dict)
        categorize(txn)
        vendor_lookup(str(txn["d"]).split(" *")[0])

    rt.bind()
    rt.workflow = workflow
    return rt


@pytest.fixture(autouse=True)
def clean_registry() -> Generator[None, None, None]:
    registry._SITES.clear()  # pyright: ignore[reportPrivateUsage]
    registry.bind(None)
    yield
    registry._SITES.clear()  # pyright: ignore[reportPrivateUsage]
    registry.bind(None)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whatever the developer's shell exports must not leak into /healthz or the app version."""
    for name in ("DREAMING_VERSION", "MODAL_IMAGE_ID", "DREAMING_IMAGE_DIGEST", "SUPABASE_URL", "SUPABASE_PUBLISHABLE_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def txns() -> Callable[..., list[dict[str, Json]]]:
    """`transactions(n, seed=0)`, for tests that build their own inputs."""
    return transactions


@pytest.fixture
def recorded() -> bool:
    """Whether the fake teacher answers like RecordedTeacher (`recorded=True`). Override with `parametrize("recorded", …)`."""
    return True


@pytest.fixture
def rt(recorded: bool) -> Runtime:
    """The workflow, built once per test (the registry is process-global)."""
    return build(MemoryStore(), KeywordTeacher(recorded=recorded))


@pytest.fixture
def down() -> dict[str, bool]:
    """The fake vendor API's fault switch (`down["vendor"]`), as create_app's `inject` leaves it."""
    return DOWN


Clients = Callable[..., TestClient]


@pytest.fixture
def make_client(rt: Runtime) -> Generator[Clients, None, None]:
    """`create_app(runtime=rt, inputs=…, inject=…)` behind a TestClient; no fast-forward thread may outlive the test."""

    def make(inputs: Iterable[Json] | None, *, faults: bool = True, raise_server_exceptions: bool = True) -> TestClient:
        return TestClient(create_app(runtime=rt, inputs=inputs, inject=inject if faults else None), raise_server_exceptions=raise_server_exceptions)

    yield make
    for t in threading.enumerate():
        if t.name == "fast-forward":
            t.join(timeout=15)


@pytest.fixture
def client(make_client: Clients) -> TestClient:
    """1000 transactions, drawn once across batches: enough for categorize to watch (60), shadow (200) and be promoted in 600."""
    return make_client(transactions(1000))
