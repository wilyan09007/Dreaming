"""The routes added for Intuit's System One format, the Modal campaign and the budget floor, plus the Sentry sink, all offline.

The app is the one test_api.py builds (conftest `rt`: `books.categorize` with labels Software/Meals/Travel, and the tool
`books.vendor_lookup`). A trained student is a fake in `rt.predictors`; Modal is the `remote` function of campaign_routes,
replaced per test; Sentry is a fake `sentry_sdk` module in sys.modules.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from typing import Any

import pytest
from dreaming.contracts import Component, EventKind, Hashes, Json, Prediction
from dreaming.runtime import Runtime
from dreaming.telemetry import emit
from dreaming_api import campaign_routes
from fastapi.testclient import TestClient

CAT = "books.categorize"
OPTIONS = {"Software": "software subscriptions", "Meals": "food and drink", "Travel": "rides and flights"}
Clients = Callable[..., TestClient]


class FakeStudent:
    """A trained predictor: always Meals at 0.7 (Travel 0.2, Software 0.1), or with no scores at all."""

    weights_sha, env_sha = "w1", "e1"

    def __init__(self, scores: bool = True) -> None:
        self.scores = scores
        self.seen: list[str] = []

    def predict(self, rendered: str, *, fresh: bool = False) -> Prediction:
        self.seen.append(rendered)
        return Prediction(label="Meals", confidence=0.5, scores={"Software": 0.1, "Meals": 0.7, "Travel": 0.2} if self.scores else {},
                          hashes=Hashes(prompt="p", weights=self.weights_sha, env=self.env_sha))


def ask(client: TestClient, qid: str = "categorize", *, state: Json = "ADOBE *1234, $12.00", criteria: Json = OPTIONS, type_: str = "choice") -> Any:
    body = {"model": "typesafe/jev-1.13", "state": state, "questions": {qid: {"type": type_, "instructions": "category", "criteria": criteria}}}
    return client.post("/v1/systemone", json=body)


@pytest.fixture
def c(make_client: Clients) -> TestClient:
    return make_client(None)


# ------------------------------------------------------------------ /v1/systemone


def test_systemone_rejects_what_the_student_cannot_answer(c: TestClient) -> None:
    assert ask(c, "nope").status_code == 422  # no such call site
    assert ask(c, criteria={"Software": "x", "Meals": "y"}).status_code == 422  # not the site's label set
    assert ask(c, criteria=["low", "high"], type_="score").status_code == 422  # only choice is served
    assert "choice" in ask(c, type_="noul").json()["detail"]
    assert ask(c, "vendor_lookup", criteria={}).status_code == 422  # a tool has no label set


def test_systemone_without_a_trained_student_is_409(c: TestClient) -> None:
    r = ask(c)
    assert r.status_code == 409 and "nothing trained for books.categorize yet" in r.json()["detail"]


@pytest.mark.parametrize("qid", ["categorize", CAT])  # name or id
def test_systemone_answers_in_the_wire_format(c: TestClient, rt: Runtime, qid: str) -> None:
    student = FakeStudent()
    rt.predictors[CAT] = student
    r = ask(c, qid, state={"d": "ADOBE", "amt": 12})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "typesafe/jev-1.13"
    a = body["answers"][qid]
    assert a["type"] == "choice" and a["choice"] == "Meals" and a["confidence"] == 0.5  # 0.7 - 0.2
    assert sum(a["probabilities"].values()) == pytest.approx(1.0) and set(a["probabilities"]) == set(OPTIONS)
    assert body["usage"]["input_tokens"] == 1 and body["usage"]["output_tokens"] == 1
    assert student.seen == ['{"amt":12,"d":"ADOBE"}']  # an object state is its canonical JSON


def test_systemone_without_scores_is_one_hot(c: TestClient, rt: Runtime) -> None:
    rt.predictors[CAT] = FakeStudent(scores=False)
    a = ask(c).json()["answers"]["categorize"]
    assert a["probabilities"] == {"Software": 0.0, "Meals": 1.0, "Travel": 0.0} and a["confidence"] == 1.0


# ------------------------------------------------------------------ /campaign


def test_campaign_routes_proxy_modal(c: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, tuple[Json, ...]]] = []

    def fake(fn_name: str, *args: Json) -> Json:
        calls.append((fn_name, args))
        return [{"cell": "a"}] if fn_name == "launch_cells" else {"done": 3}

    monkeypatch.setattr(campaign_routes, "remote", fake)
    r = c.post("/campaign/run", json={"campaign": "x", "sites": ["a"], "limit": 5})
    assert r.status_code == 200 and r.json() == {"cells": [{"cell": "a"}]}
    assert c.post("/campaign/run", json={"campaign": "x"}).status_code == 200  # sites [] = all, limit 0 = all
    assert c.get("/campaign/x").json() == {"campaign": "x", "table": {"done": 3}}
    assert calls == [("launch_cells", ("x", ["a"], 5)), ("launch_cells", ("x", [], 0)), ("progress", ("x",))]


def test_campaign_failure_is_503_with_the_error(c: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(fn_name: str, *args: Json) -> Json:
        raise RuntimeError("app not deployed")

    monkeypatch.setattr(campaign_routes, "remote", boom)
    for r in (c.post("/campaign/run", json={"campaign": "x"}), c.get("/campaign/x")):
        assert r.status_code == 503 and "app not deployed" in r.json()["detail"]


# ------------------------------------------------------------------ POST /budget


def test_budget_below_the_teacher_floor_is_422(c: TestClient, rt: Runtime) -> None:
    rt.sites[CAT].teacher.floor = 0.035
    r = c.post("/budget", json={"budget": 0.02})
    assert r.status_code == 422 and r.json()["detail"] == (
        "budget 2% is below the teacher's measured self-disagreement 3.5%: no certificate at that level is measurable")
    assert c.post("/budget", json={"budget": 0.05}).json() == {"budget": 0.05}
    assert rt.budget == 0.05


def test_budget_without_a_floor_is_unchanged(c: TestClient) -> None:
    assert c.post("/budget", json={"budget": 0.01}).status_code == 200


# ------------------------------------------------------------------ the Sentry sink


def fake_sentry(monkeypatch: pytest.MonkeyPatch, *, initialized: bool = True, raises: bool = False) -> list[tuple[str, dict[str, Any]]]:
    sent: list[tuple[str, dict[str, Any]]] = []

    def capture(message: str, **kw: Any) -> None:
        if raises:
            raise RuntimeError("sentry down")
        sent.append((message, kw))

    monkeypatch.setitem(sys.modules, "sentry_sdk", types.SimpleNamespace(is_initialized=lambda: initialized, capture_message=capture))
    return sent


def test_emit_error_without_sentry_does_not_raise() -> None:
    ev = emit(None, Component.api, EventKind.component_error, "x failed", error="Boom: no", data={"dependency": "store"})
    assert ev.error == "Boom: no"


def test_emit_error_goes_to_sentry_tagged(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = fake_sentry(monkeypatch)
    ev = emit(None, Component.api, EventKind.component_error, "api: store failed", call_site=CAT, data={"dependency": "store"}, error="Boom: no")
    emit(None, Component.api, EventKind.batch_started, "fine")  # no error: nothing sent
    assert len(sent) == 1 and sent[0][0] == "api: store failed"
    assert sent[0][1]["level"] == "error"
    assert sent[0][1]["tags"] == {"component": "api", "kind": "component.error", "call_site": CAT, "dependency": "store"}
    assert sent[0][1]["extras"] == {"error": "Boom: no", "event_id": ev.id}


def test_sentry_that_is_off_or_broken_never_breaks_emit(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = fake_sentry(monkeypatch, initialized=False)
    emit(None, Component.api, EventKind.component_error, "off", error="e")
    assert sent == []
    fake_sentry(monkeypatch, raises=True)
    assert emit(None, Component.api, EventKind.component_error, "broken", error="e").message == "broken"
