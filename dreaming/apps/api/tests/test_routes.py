"""The routes added for Intuit's System One format, the Modal campaign and the budget floor, plus the Sentry sink, all offline.

The app is the one test_api.py builds (conftest `rt`: `books.categorize` with labels Software/Meals/Travel, and the tool
`books.vendor_lookup`). A trained student is a fake in `rt.predictors`; Modal is a fake `modal.Function.from_name`; the
TypeSafe client is the real typesafe-sdk over an in-process transport; Sentry is a fake `sentry_sdk` module in sys.modules.
"""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Callable
from typing import Any

import pytest
from dreaming.contracts import Component, EventKind, Hashes, Json, Prediction
from dreaming.runtime import Runtime
from dreaming.telemetry import emit
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
    assert body["model"] == "dreaming/books.categorize@w1"  # what answered, not the model the client named
    a = body["answers"][qid]
    assert a["type"] == "choice" and a["choice"] == "Meals" and a["confidence"] == 0.5  # 0.7 - 0.2
    assert sum(a["probabilities"].values()) == pytest.approx(1.0) and set(a["probabilities"]) == set(OPTIONS)
    assert a["certified"] is False and a["certificate"] is None  # no certificate on the site yet
    assert body["usage"]["input_tokens"] == 1 and body["usage"]["output_tokens"] == 1
    assert student.seen == ['{"amt":12,"d":"ADOBE"}']  # an object state is its canonical JSON


def test_systemone_says_when_the_certificate_covers_the_answer(c: TestClient, rt: Runtime) -> None:
    rt.predictors[CAT] = FakeStudent()
    rt.policies[CAT] = rt.policies[CAT].model_copy(update={"cert_id": "cert-1", "cutoff": 0.5})
    a = ask(c).json()["answers"]["categorize"]
    assert a["certified"] is True and a["certificate"] == "cert-1"  # confidence 0.5 >= cutoff 0.5
    rt.policies[CAT] = rt.policies[CAT].model_copy(update={"cutoff": 0.6})
    assert ask(c).json()["answers"]["categorize"]["certified"] is False


def test_systemone_without_scores_is_one_hot(c: TestClient, rt: Runtime) -> None:
    rt.predictors[CAT] = FakeStudent(scores=False)
    a = ask(c).json()["answers"]["categorize"]
    assert a["probabilities"] == {"Software": 0.0, "Meals": 1.0, "Travel": 0.0} and a["confidence"] == 1.0


def test_typesafe_sdk_parses_our_reply(c: TestClient, rt: Runtime) -> None:
    """The compatibility proof (devtools.md): typesafe-sdk 0.7.5's own client posts to our route and validates the reply."""
    sdk: Any = pytest.importorskip("typesafe_sdk")
    httpx2: Any = pytest.importorskip("httpx2")
    rt.predictors[CAT] = FakeStudent()
    sent: list[Any] = []

    def forward(req: Any) -> Any:
        sent.append(json.loads(req.content))
        r = c.request(req.method, req.url.path, content=req.content, headers={"content-type": req.headers["content-type"]})
        return httpx2.Response(r.status_code, content=r.content, headers={"content-type": r.headers["content-type"]})

    client = sdk.TypeSafeClient(api_key="unused", base_url="http://dreaming.test", model="typesafe/jev-1.13", transport=httpx2.MockTransport(forward),
                                compression=False)
    res = client.system_one("ADOBE *1234, $12.00", {"categorize": sdk.Choice(criteria=OPTIONS, instructions="category")})
    assert sent[0]["questions"]["categorize"]["type"] == "choice" and sent[0]["state"] == "ADOBE *1234, $12.00"
    a = res.answers["categorize"]
    assert isinstance(a, sdk.ChoiceAnswer) and a.choice == "Meals" and a.confidence == 0.5
    assert res.model == "dreaming/books.categorize@w1" and res.usage.input_tokens == 3
    with pytest.raises(sdk.TypeSafeUnprocessableEntityError):  # our 422 maps to the SDK's own error
        client.system_one("x", {"nope": sdk.Choice(criteria=OPTIONS)})


# ------------------------------------------------------------------ /campaign

Calls = list[tuple[str, str, tuple[Json, ...]]]
KEY = {"Authorization": "Bearer hackathon-key"}


class FakeFunction:
    """modal.Function.from_name(app, name) as the routes use it, `await fn.remote.aio(*args)`; `error` plays an undeployed app."""

    def __init__(self, calls: Calls, app: str, name: str, error: Exception | None) -> None:
        async def aio(*args: Json) -> Json:
            if error:
                raise error
            calls.append((app, name, args))
            return [{"cell": "a"}] if name == "launch_cells" else {"done": 3}

        self.remote = types.SimpleNamespace(aio=aio)


@pytest.fixture
def modal_fn(monkeypatch: pytest.MonkeyPatch) -> Callable[[Exception | None], Calls]:
    import modal

    def install(error: Exception | None) -> Calls:
        calls: Calls = []
        monkeypatch.setattr(modal.Function, "from_name", lambda app, name: FakeFunction(calls, app, name, error))  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
        monkeypatch.setenv("DREAMING_API_KEY", "hackathon-key")
        return calls

    return install


def test_campaign_routes_call_modal_by_name(c: TestClient, modal_fn: Callable[[Exception | None], Calls]) -> None:
    calls = modal_fn(None)
    r = c.post("/campaign/run", json={"provider": "stepfun", "site": "invoice_paid"}, headers=KEY)
    assert r.status_code == 200 and r.json() == {"campaign_id": "stepfun-invoice_paid", "cells": [{"cell": "a"}]}
    assert c.post("/campaign/run", json={"provider": "books", "site": "bill_account", "campaign": "night-1018", "limit": 3}, headers=KEY).status_code == 200
    assert c.get("/campaign/night-1018").json() == {"campaign": "night-1018", "table": {"done": 3}}
    assert calls == [("dreaming-campaign", "launch_cells", ("stepfun-invoice_paid", ["invoice_paid"], 0)),
                     ("dreaming-campaign", "launch_cells", ("night-1018", ["bill_account"], 3)),
                     ("dreaming-campaign", "progress", ("night-1018",))]


def test_campaign_launch_needs_the_key(c: TestClient, monkeypatch: pytest.MonkeyPatch, modal_fn: Callable[[Exception | None], Calls]) -> None:
    calls = modal_fn(None)
    body = {"provider": "stepfun", "site": "invoice_paid"}
    assert c.post("/campaign/run", json=body).status_code == 401
    assert c.post("/campaign/run", json=body, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.post("/campaign/run", json={"provider": "a b", "site": "x"}, headers=KEY).status_code == 422  # names go into run ids and URLs
    monkeypatch.delenv("DREAMING_API_KEY")
    assert c.post("/campaign/run", json=body, headers=KEY).status_code == 503  # no key configured: launching is off, not open
    assert calls == []


def test_campaign_undeployed_is_503_with_the_error(c: TestClient, modal_fn: Callable[[Exception | None], Calls]) -> None:
    from modal.exception import NotFoundError

    modal_fn(NotFoundError("Lookup failed for Function 'launch_cells' from the 'dreaming-campaign' app"))
    for r in (c.post("/campaign/run", json={"provider": "stepfun", "site": "invoice_paid"}, headers=KEY), c.get("/campaign/x")):
        assert r.status_code == 503 and "NotFoundError" in r.json()["detail"] and "dreaming-campaign" in r.json()["detail"]


def test_campaign_unknown_site_is_422_not_503(c: TestClient, modal_fn: Callable[[Exception | None], Calls]) -> None:
    modal_fn(ValueError("unknown campaign site 'nope'; choose from bookkeeper.categorize"))  # what launch_cells raises
    r = c.post("/campaign/run", json={"provider": "stepfun", "site": "nope"}, headers=KEY)
    assert r.status_code == 422 and "unknown campaign site 'nope'" in r.json()["detail"]


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
