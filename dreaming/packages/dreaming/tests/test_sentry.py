"""The Sentry sink against a capturing transport (nothing leaves the process): gen_ai spans per run and routed call, the JSON
lines as Sentry Logs with queryable attributes, one grouped issue per failure place, and nothing at all without a DSN."""

from __future__ import annotations

import logging
import threading
from collections.abc import Generator, Iterator
from typing import Any

import pytest
from conftest import KeywordTeacher
from dreaming import telemetry
from dreaming.contracts import CallSite, Component, EventKind, TeacherAnswer
from dreaming.store import MemoryStore
from test_runtime import build

sentry_sdk = pytest.importorskip("sentry_sdk")
from sentry_sdk.transport import Transport

TXN = {"d": "ADOBE *1234", "amt": 5.0}


class Sink(Transport):
    def __init__(self) -> None:
        super().__init__()
        self.items: list[tuple[str, Any]] = []

    def capture_envelope(self, envelope: Any) -> None:
        self.items += [(i.type, i.payload.json) for i in envelope.items]

    def spans(self) -> Iterator[tuple[str, dict[str, Any]]]:
        """(name, attributes) of every transaction and span sent, whichever of the SDK's two wire formats carried it."""
        for kind, p in self.items:
            if kind == "transaction":
                yield p["transaction"], {**p["contexts"]["trace"]["data"], "status": p["contexts"]["trace"].get("status")}
                yield from ((s["description"], s["data"]) for s in p["spans"])
            elif kind == "span":
                yield from ((s["name"], {k: v["value"] for k, v in s["attributes"].items()}) for s in p["items"])


@pytest.fixture
def sink(monkeypatch: pytest.MonkeyPatch) -> Generator[Sink, None, None]:
    sink, init = Sink(), sentry_sdk.init
    root, handlers, level = logging.getLogger(), logging.getLogger().handlers[:], logging.getLogger().level

    def init_into_sink(**kw: Any) -> None:
        init(transport=sink, **kw)

    monkeypatch.setattr(sentry_sdk, "init", init_into_sink)
    monkeypatch.setenv("SENTRY_DSN", "http://key@localhost/1")
    telemetry.configure(Component.api)  # the real init path, with the transport swapped
    yield sink
    sentry_sdk.get_client().close()
    sentry_sdk.get_global_scope().set_client(None)
    root.handlers[:], root.level = handlers, level


def test_a_run_and_its_routed_calls_are_gen_ai_spans_with_the_routers_decision(sink: Sink) -> None:
    rt = build(MemoryStore())
    telemetry.instrument(rt)
    assert rt.workflow
    with rt.run("books", TXN):
        rt.workflow(TXN)
    sentry_sdk.flush()
    spans = dict(sink.spans())
    assert spans["invoke_agent books"]["gen_ai.agent.name"] == "books"
    assert spans["chat categorize"]["dreaming.decision"] == "called" and spans["chat categorize"]["gen_ai.operation.name"] == "chat"
    assert spans["execute_tool vendor_lookup"]["gen_ai.tool.name"] == "vendor_lookup"


def test_a_failed_run_marks_its_span(sink: Sink) -> None:
    rt = build(MemoryStore())
    telemetry.instrument(rt)
    with rt.run("books", TXN) as r:
        raise ValueError("boom")
    sentry_sdk.flush()
    assert r.failed and dict(sink.spans())["invoke_agent books"]["status"] == "internal_error"


def test_the_json_lines_are_sentry_logs_and_errors_group_by_place_not_by_message(sink: Sink) -> None:
    store = MemoryStore()
    for n in (1, 2):
        telemetry.emit(store, Component.api, EventKind.component_error, f"run run_{n} failed", error="X: y", data={"dependency": "demo", "n": n})
    sentry_sdk.flush()
    logs = [row for kind, p in sink.items if kind == "log" for row in p["items"]]
    assert {r["body"] for r in logs} >= {"run run_1 failed", "run run_2 failed"}
    attrs = next(r["attributes"] for r in logs if r["body"] == "run run_2 failed")
    assert attrs["component"]["value"] == "api" and attrs["data.dependency"]["value"] == "demo" and attrs["data.n"]["value"] == 2
    events = [p for kind, p in sink.items if kind == "event"]
    assert len(events) == 2 and events[0]["fingerprint"] == events[1]["fingerprint"] and events[0]["tags"]["dependency"] == "demo"


def test_without_sentry_nothing_is_wrapped() -> None:
    rt = build(MemoryStore())
    telemetry.instrument(rt)
    assert "run" not in vars(rt) and "_route" not in vars(rt)
    with telemetry.span("gen_ai.chat", "chat x") as sp:
        assert sp is None


def test_a_malformed_dsn_is_a_warning_not_a_dead_process(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    root, handlers, level = logging.getLogger(), logging.getLogger().handlers[:], logging.getLogger().level
    monkeypatch.setenv("SENTRY_DSN", "not a dsn")
    try:
        telemetry.configure(Component.api)
    finally:
        root.handlers[:], root.level = handlers, level
    assert "no Sentry" in capsys.readouterr().out


def test_a_thread_started_by_a_request_still_gets_its_own_transaction(sink: Sink) -> None:
    rt = build(MemoryStore())
    telemetry.instrument(rt)
    request_done, ran = threading.Event(), threading.Event()

    def batch() -> None:  # like POST /batch: the thread inherits the request's scope, then the request's transaction is sent
        request_done.wait()
        with rt.run("books", TXN):
            pass
        ran.set()

    with sentry_sdk.start_transaction(op="http.server", name="/batch"):
        threading.Thread(target=batch).start()
    request_done.set()
    assert ran.wait(10)
    sentry_sdk.flush()
    assert "invoke_agent books" in dict(sink.spans()) and any(kind == "transaction" and p["transaction"] == "invoke_agent books" for kind, p in sink.items)


def test_a_recorded_answers_tokens_and_cost_ride_on_its_chat_span(sink: Sink) -> None:
    class Recorded(KeywordTeacher):  # what RecordedTeacher serves in fast-forward: no live SDK call, so no other span counts the tokens
        def ask(self, site: CallSite, rendered: str) -> TeacherAnswer:
            return super().ask(site, rendered).model_copy(update={"recorded": True, "usage": {"prompt_tokens": 120, "completion_tokens": 3}})

    rt = build(MemoryStore())
    rt.teacher = Recorded()
    telemetry.instrument(rt)
    assert rt.workflow
    with rt.run("books", TXN):
        rt.workflow(TXN)
    sentry_sdk.flush()
    chat = dict(sink.spans())["chat categorize"]
    assert (chat["gen_ai.usage.input_tokens"], chat["gen_ai.usage.output_tokens"], chat["gen_ai.cost.total_tokens"]) == (120, 3, 0.01)
