"""Build one Runtime from the environment. The only place that reads env vars and picks implementations.

    DATABASE_URL        Supabase session-pooler URL (IPv4). Unset: MemoryStore (tests, `just dev`)
    TEACHER             openrouter (default) | asi1 | stepfun | rule: which TeacherSpec the bookkeeper workflow pins (workflow._teachers).
                        `rule` is the offline rule teacher from agents/bookkeeper/offline.py, labelled provider="rule" in every row:
                        for `just dev` and tests, never for a recording
    OPENROUTERKEY, ASI1KEY, STEPFUNKEY, GEMINIKEY
                        one key per provider row (teachers.PROVIDERS); every key that is set is usable, the site's spec picks the row.
                        OPENROUTERKEY is also the upstream of the OpenAI-compatible proxy at /v1 (mounted only when set)
    RECORDED_ANSWERS    path of the pre-recorded teacher file; set => fast-forward reads it, misses go live (and are counted)
    TRAINER             modal | local (default: modal when MODAL_TOKEN_ID is set, else local). Chosen at startup, not a live fallback
    ENCODER             sentence (default: bge-small, torch) | char (the hashed-trigram fake for `just dev`; must match the trainer's)
    ENCODER_REV         the encoder's pinned revision; the same value must be set for the Modal trainer (predictors/student.py)
    LOGFIRE_TOKEN       optional
    SENTRY_DSN          optional: errors, traces (gen_ai spans per run and routed call) and logs to Sentry (telemetry.configure and
                        telemetry.instrument; SENTRY_ENV, SENTRY_RELEASE tune it; SENTRY_TRACES is the sample rate, 1.0, 0.05 for batches)
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
from dreaming.connectors.proxy import proxy_router
from dreaming.contracts import Component, Json
from dreaming.predictors import Loader
from dreaming.predictors.train import LocalTrainer, Trainer
from dreaming.runtime import Runtime
from dreaming.store import MemoryStore, PostgresStore, Store
from dreaming.teachers import Teacher
from dreaming.teachers.openai_compat import OPENROUTER, PROVIDERS, OpenAICompatTeacher
from dreaming.teachers.recorded import RecordedTeacher
from dreaming.telemetry import configure, instrument
from starlette.routing import Mount


def build_store() -> Store:
    dsn = os.environ.get("DATABASE_URL")
    return PostgresStore(dsn) if dsn else MemoryStore()


def build_teacher() -> Teacher:
    if os.environ.get("TEACHER") == "rule":
        from bookkeeper.offline import RuleTeacher

        return RuleTeacher()
    keys = {name: os.environ[row.key] for name, row in PROVIDERS.items() if os.environ.get(row.key)}  # every provider with a key
    live: Teacher | None = OpenAICompatTeacher(keys) if keys else None  # the workflow's TeacherSpec.provider picks the row per call
    if rec := os.environ.get("RECORDED_ANSWERS"):
        return RecordedTeacher(Path(rec), live)
    if live is None:
        raise RuntimeError("no teacher: set a provider key (OPENROUTERKEY, ASI1KEY, STEPFUNKEY, GEMINIKEY), or RECORDED_ANSWERS for offline fast-forward")
    return live


def build_trainer(store: Store, loader: Loader) -> Trainer:
    if os.environ.get("TRAINER", "modal" if os.environ.get("MODAL_TOKEN_ID") else "local") == "modal":
        from .trainer import ModalTrainer

        return ModalTrainer(store, loader)
    return LocalTrainer(store, loader)


def build_encoder() -> object | None:
    """The student encoder, loaded once per process. None when the `student` extra is not installed (tools-only runs)."""
    if os.environ.get("ENCODER") == "char":  # offline dev: the hashed-trigram encoder, no torch (never with TRAINER=modal: Modal uses the real one)
        from bookkeeper.offline import CharEncoder

        return CharEncoder()
    try:
        from dreaming.predictors.student import SentenceEncoder

        return SentenceEncoder(cache_path=Path(os.environ.get("EMBEDDING_CACHE", "data/embeddings.npz")))
    except ImportError:
        return None


def build_runtime(component: Component) -> Runtime:
    configure(component)
    store = build_store()
    loader = Loader(encoder=build_encoder(), teacher=None)
    teacher = build_teacher()
    loader.teacher = teacher
    rt = Runtime(store, teacher, build_trainer(store, loader), loader, component=component, budget=float(os.environ.get("BUDGET", "0.05")))
    import bookkeeper.workflow  # declares the call sites into the registry

    rt.bind()
    instrument(rt)  # Sentry's gen_ai spans; a no-op without SENTRY_DSN
    rt.workflow = bookkeeper.workflow.run_transaction  # type: ignore[attr-defined]
    return rt


RECORDED_SEEDS = (0, 1, 2)  # `just record` records these; wiring cycles through them


def ledger_inputs() -> Iterator[Json]:
    """The stream fast-forward draws from: the seeded ledger, endless."""
    from bookkeeper.tools import ledger

    for seed in itertools.cycle(RECORDED_SEEDS):  # only seeds `just record` recorded: a fresh seed would send every call live
        yield from ledger(2000, seed)


def inject(site: str, down: bool) -> None:
    """Fault injection for the demo's one unpluggable tool. The runtime detects the outage; the page sees tool.outage."""
    from bookkeeper.tools import DOWN

    if site.endswith(".vendor_lookup"):
        DOWN["vendor_api"] = down


def app() -> Any:
    """The uvicorn factory: `uvicorn dreaming_api.wiring:app --factory`. Serves the OpenAI-compatible proxy when OPENROUTERKEY is set."""
    from .main import create_app

    rt = build_runtime(Component.api)
    api = create_app(rt, ledger_inputs(), inject)
    if key := os.environ.get("OPENROUTERKEY"):
        api.include_router(proxy_router(rt, httpx.AsyncClient(base_url=OPENROUTER, headers={"Authorization": f"Bearer {key}"}, timeout=120)))
        api.router.routes.sort(key=lambda route: isinstance(route, Mount))  # create_app mounted the dashboard at "/" and it must stay last
    return api


def checks(rt: Runtime) -> dict[str, str]:
    """One line per dependency: "ok", "ok (detail)" or the error. The page shows the red ones."""
    out: dict[str, str] = {}
    try:
        rt.store.sites()
        out["db"] = "ok"
    except Exception as e:  # noqa: BLE001
        out["db"] = f"{type(e).__name__}: {e}"
    out["trainer"] = f"configured ({type(rt.trainer).__name__})"  # not probed: a probe would be a paid job
    out["teacher"] = f"configured ({type(rt.teacher).__name__})"  # not probed: a probe would be a paid call
    out["realtime"] = "configured (supabase)" if os.environ.get("SUPABASE_URL") else "configured (polling)"
    return out
