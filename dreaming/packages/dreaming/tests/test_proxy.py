"""The OpenAI-compatible proxy, offline: an httpx.MockTransport is the upstream, a TestClient is the agent."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Callable, MutableMapping
from typing import Any

import httpx
import pytest
from conftest import LABELS, CharEncoder, KeywordTeacher
from dreaming import Runtime
from dreaming.connectors.openai_client import site_of
from dreaming.connectors.proxy import proxy_router
from dreaming.contracts import CallSite, Decision, Hashes, Kind, Phase, Prediction, Thresholds, TraceRow
from dreaming.predictors import Loader
from dreaming.predictors.train import LocalTrainer
from dreaming.store import MemoryStore
from fastapi import FastAPI
from fastapi.testclient import TestClient

SITE = "agent.triage"
TAG = {"X-Dreaming-Call-Site": SITE}
BODY: dict[str, Any] = {"model": "gpt-x", "messages": [{"role": "system", "content": "Categorize."}, {"role": "user", "content": "ADOBE *1234"}]}
TOOLS: dict[str, Any] = {"tools": [{"type": "function", "function": {"name": "vendor_lookup", "parameters": {"type": "object"}}}], "tool_choice": "required"}
USAGE = {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8}
COMPLETION = {"id": "c1", "object": "chat.completion", "created": 1, "model": "gpt-x", "usage": USAGE,
              "choices": [{"index": 0, "message": {"role": "assistant", "content": "Software"}, "finish_reason": "stop"}]}
TOOL_TURN = {**COMPLETION, "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [
    {"id": "call_1", "type": "function", "function": {"name": "vendor_lookup", "arguments": '{"d": "ADOBE"}'}}]}}]}
RESPONSE = {"id": "resp_1", "object": "response", "status": "completed", "model": "gpt-x", "usage": {"input_tokens": 7, "output_tokens": 1},
            "output": [{"type": "message", "id": "msg_1", "role": "assistant", "content": [{"type": "output_text", "text": "Software"}]}]}


def _chunk(delta: dict[str, Any], **more: Any) -> bytes:
    return b"data: " + json.dumps({"id": "c1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta}], **more}).encode() + b"\n\n"


CHAT_PARTS = [_chunk({"role": "assistant", "content": "Soft"}), _chunk({"content": "ware"}), _chunk({}, usage=USAGE), b"data: [DONE]\n\n"]


def _event(type: str, **more: Any) -> bytes:
    return f"event: {type}\n".encode() + b"data: " + json.dumps({"type": type, **more}).encode() + b"\n\n"


RESPONSES_PARTS = [_event("response.created", response={}), _event("response.output_text.delta", delta="Soft"),
                   _event("response.output_text.delta", delta="ware"), _event("response.completed", response={"usage": USAGE})]
SSE = {"content-type": "text/event-stream"}


def upstream(req: httpx.Request) -> httpx.Response:
    body = json.loads(req.content)
    chat = req.url.path == "/chat/completions"
    if body.get("stream"):
        return httpx.Response(200, content=b"".join(CHAT_PARTS if chat else RESPONSES_PARTS), headers=SSE)
    return httpx.Response(200, json=(TOOL_TURN if body.get("tools") else COMPLETION) if chat else RESPONSE)


class Rig:
    """A Runtime on MemoryStore (the fakes of conftest.py), the proxy on a FastAPI app, and an upstream that records what it is sent."""

    def __init__(self, handler: Callable[[httpx.Request], httpx.Response] = upstream) -> None:
        self.sent: list[httpx.Request] = []

        def record(req: httpx.Request) -> httpx.Response:
            self.sent.append(req)
            return handler(req)

        store = MemoryStore()
        loader = Loader(encoder=CharEncoder())
        self.rt = Runtime(store, KeywordTeacher(), LocalTrainer(store, loader), loader)
        self.app = FastAPI()
        up = httpx.AsyncClient(transport=httpx.MockTransport(record), base_url="https://up.test")
        self.app.include_router(proxy_router(self.rt, up))
        self.client = TestClient(self.app)

    def post(self, body: dict[str, Any], path: str = "/v1/chat/completions", *, tag: bool = True, **headers: str) -> Any:
        """The TestClient's response (a starlette-bundled httpx type, not this module's `httpx.Response`)."""
        return self.client.post(path, json=body, headers={**(TAG if tag else {}), **headers})

    def rows(self, site: str) -> list[TraceRow]:
        self.rt.flush()
        return self.rt.store.rows(site)


class Sure:
    """A predictor that is always confident: stands in for a certified student."""

    weights_sha: str = "w1"
    env_sha: str = "e1"

    def predict(self, rendered: str, *, fresh: bool = False) -> Prediction:
        return Prediction(label="Software", confidence=1.0, hashes=Hashes(prompt="p"))


def dreaming_rig() -> Rig:
    """SITE declared with labels, a confident predictor installed, phase dreaming, cutoff 0, no audits."""
    rig = Rig()
    rig.rt.declare(CallSite(id=SITE, workflow="agent", name="triage", kind=Kind.model, labels=LABELS, thresholds=Thresholds(audit_rate=0.0)))
    rig.rt.predictors[SITE] = Sure()
    policy = rig.rt.policies[SITE]
    policy.phase, policy.cutoff = Phase.dreaming, 0.0
    return rig


def test_tagged_call_is_forwarded_and_logged_under_the_headers_site():
    rig = Rig()
    r = rig.post(BODY, Authorization="Bearer the-agents-key")
    assert r.status_code == 200 and r.json() == COMPLETION
    (sent,) = rig.sent
    assert sent.url.path == "/chat/completions" and json.loads(sent.content) == BODY
    assert "the-agents-key" not in sent.headers.get("authorization", "")  # the upstream client's key is used, never the caller's
    (row,) = rig.rows(SITE)
    assert row.decision == Decision.called and row.kind == Kind.model and row.error is None
    assert isinstance(row.request, dict) and set(row.request) == {"model", "messages", "tools", "tool_choice", "response_format"}
    assert row.request["messages"] == BODY["messages"] and row.request["tools"] is None
    assert row.response == COMPLETION and row.teacher is not None and row.teacher.label == "Software"


def test_untagged_call_lands_under_a_sys_site_shared_by_the_same_system_prompt():
    rig = Rig()
    other = {**BODY, "messages": [BODY["messages"][0], {"role": "user", "content": "UBER *9"}]}
    assert all(rig.post(b, tag=False).status_code == 200 for b in (BODY, other))
    site = site_of(BODY)
    assert site == "sys:" + hashlib.sha1(b"Categorize.").hexdigest()[:12] and len(rig.rows(site)) == 2


def test_request_key_covers_response_format_and_tools():
    rig = Rig()
    schema = {"type": "json_schema", "json_schema": {"name": "label", "schema": {"type": "object"}}}
    rig.post({**BODY, "response_format": schema})
    rig.post(BODY)
    a, b = rig.rows(SITE)
    assert isinstance(a.request, dict) and a.request["response_format"] == schema and a.request_sha != b.request_sha


def test_tool_call_turn_is_logged_as_an_effect_and_passed_through_unchanged():
    rig = Rig()
    body = {**BODY, **TOOLS}
    r = rig.post(body)
    assert r.status_code == 200 and r.json() == TOOL_TURN and json.loads(rig.sent[0].content) == body
    assert rig.rows(SITE) == []  # never under the site that predicts
    (row,) = rig.rows(SITE + "#forward")
    assert row.kind == Kind.effect and row.decision == Decision.called and row.response == TOOL_TURN
    assert row.teacher is not None and row.teacher.label == ""  # a tool call is not an answer
    assert isinstance(row.request, dict) and row.request["tool_choice"] == "required" and row.request["tools"] == TOOLS["tools"]


def test_stream_passes_through_byte_for_byte_and_logs_one_row():
    rig = Rig()
    r = rig.post({**BODY, "stream": True})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert r.content == b"".join(CHAT_PARTS)
    (row,) = rig.rows(SITE + "#forward")
    assert row.kind == Kind.effect and row.decision == Decision.called and row.error is None
    assert row.response == {"text": "Software", "usage": USAGE, "tool_calls": False, "stream": True}
    assert row.teacher is not None and row.teacher.label == "Software"
    assert rig.rows(SITE) == []


def test_stream_row_latency_is_the_streams():
    class Slow(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            for part in CHAT_PARTS:
                await asyncio.sleep(0.05)
                yield part

    rig = Rig(lambda req: httpx.Response(200, stream=Slow(), headers=SSE))
    assert rig.post({**BODY, "stream": True}).content == b"".join(CHAT_PARTS)
    (row,) = rig.rows(SITE + "#forward")
    assert row.latency_ms >= 150


def test_stream_row_is_written_before_the_terminal_line_goes_out():
    """Clients stop reading at [DONE]: a row logged after it races whoever reads the trace next (S10)."""
    rig = Rig()
    seen: list[int] = []
    raw = json.dumps({**BODY, "stream": True}).encode()
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "path": "/v1/chat/completions", "raw_path": b"",
             "query_string": b"", "root_path": "", "scheme": "http", "server": ("t", 80), "client": ("c", 1),
             "headers": [(b"content-type", b"application/json"), (b"x-dreaming-call-site", SITE.encode())]}

    first = True

    async def receive() -> MutableMapping[str, Any]:
        nonlocal first
        if first:
            first = False
            return {"type": "http.request", "body": raw, "more_body": False}
        await asyncio.Event().wait()  # the client stays connected until the response is done
        raise AssertionError("unreachable")

    async def send(msg: MutableMapping[str, Any]) -> None:
        if msg["type"] == "http.response.body" and b"[DONE]" in msg.get("body", b""):
            seen.append(len(rig.rows(SITE + "#forward")))

    asyncio.run(rig.app(scope, receive, send))
    assert seen == [1]


def test_responses_api_is_forwarded_and_logs_one_row():
    rig = Rig()
    body = {"model": "gpt-x", "instructions": "Categorize.", "input": "ADOBE *1234"}
    r = rig.post(body, "/v1/responses", tag=False)
    assert r.status_code == 200 and r.json() == RESPONSE
    assert rig.sent[0].url.path == "/responses" and json.loads(rig.sent[0].content) == body
    site = "sys:" + hashlib.sha1(b"Categorize.").hexdigest()[:12]  # no header: the site is the hash of `instructions`
    (row,) = rig.rows(site + "#forward")
    assert row.kind == Kind.effect and row.response == RESPONSE and row.teacher is not None and row.teacher.label == "Software"
    assert isinstance(row.request, dict) and row.request["instructions"] == "Categorize." and row.request["input"] == "ADOBE *1234"


def test_responses_stream_is_forwarded_and_logs_one_row():
    rig = Rig()
    r = rig.post({"model": "gpt-x", "instructions": "Categorize.", "input": "x", "stream": True}, "/v1/responses")
    assert r.content == b"".join(RESPONSES_PARTS)
    (row,) = rig.rows(SITE + "#forward")
    assert row.response == {"text": "Software", "usage": USAGE, "tool_calls": False, "stream": True}


@pytest.mark.parametrize("stream", [False, True])
def test_upstream_error_is_relayed_and_is_the_rows_error(stream: bool):
    err = {"error": {"message": "slow down", "code": 429}}
    rig = Rig(lambda req: httpx.Response(429, json=err))
    r = rig.post({**BODY, "stream": stream})
    assert r.status_code == 429 and r.json() == err
    (row,) = rig.rows(SITE + "#forward" if stream else SITE)
    assert row.error is not None and row.error.startswith("Upstream: 429") and row.decision == Decision.called


def test_upstream_500_with_a_body_that_is_not_json_is_relayed_as_it_came():
    rig = Rig(lambda req: httpx.Response(500, text="bad gateway", headers={"content-type": "text/plain"}))
    r = rig.post(BODY)
    assert r.status_code == 500 and r.text == "bad gateway" and r.headers["content-type"].startswith("text/plain")
    (row,) = rig.rows(SITE)
    assert row.error is not None and "bad gateway" in row.error


def test_unreachable_upstream_is_a_502_and_a_row_with_the_error():
    def down(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    rig = Rig(down)
    r = rig.post(BODY)
    assert r.status_code == 502 and "no route" in r.json()["error"]["message"]
    (row,) = rig.rows(SITE)
    assert row.error == "ConnectError: no route"


def test_site_flagged_down_still_forwards_and_streams():
    """Three failures in a row set Policy.outage; the router then raises Stuck for most calls. A proxy stays a proxy."""
    codes = iter([500, 500, 500, 200])
    rig = Rig(lambda req: httpx.Response(next(codes), json=COMPLETION))
    assert [rig.post(BODY).status_code for _ in range(3)] == [500, 500, 500]
    assert rig.rt.policies[SITE].outage
    assert rig.post(BODY).json() == COMPLETION
    rig = Rig()
    assert rig.post({**BODY, "stream": True}).content == b"".join(CHAT_PARTS)
    rig.rt.policies[SITE + "#forward"].outage = True
    assert rig.post({**BODY, "stream": True}).content == b"".join(CHAT_PARTS)


def test_predicted_answer_is_served_without_the_upstream():
    rig = dreaming_rig()
    r = rig.post(BODY)
    out = r.json()
    assert r.status_code == 200 and rig.sent == []
    assert out["dreaming"] == {"predicted": True} and out["model"] == "gpt-x" and out["object"] == "chat.completion"
    assert out["choices"][0]["message"] == {"role": "assistant", "content": "Software"} and out["usage"]["total_tokens"] == 0
    (row,) = rig.rows(SITE)
    assert row.decision == Decision.predicted and row.cost_usd == 0.0


def test_what_cannot_be_predicted_still_reaches_the_upstream_at_a_dreaming_site():
    """Tool-call turns, streams and /v1/responses at a site that predicts: forwarded and logged, never a phantom predicted row (S10)."""
    rig = dreaming_rig()
    plain = {"model": "gpt-x", "instructions": "Categorize.", "input": "x"}
    calls = [(rig.post({**BODY, **TOOLS}), TOOL_TURN), (rig.post({**BODY, "stream": True}), None),
             (rig.post(plain, "/v1/responses"), RESPONSE), (rig.post({**plain, "stream": True}, "/v1/responses"), None)]
    assert all(r.status_code == 200 and "dreaming" not in r.text for r, _ in calls) and len(rig.sent) == 4
    assert [r.json() for r, want in calls if want] == [TOOL_TURN, RESPONSE]
    rows = rig.rows(SITE + "#forward")
    assert len(rows) == 4 and all(r.decision == Decision.called and r.kind == Kind.effect for r in rows)
    assert rig.rows(SITE) == []
