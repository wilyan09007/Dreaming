"""The ASI:One front door offline: the real chat Protocol, a fake uagents Context, and a fake API (httpx.MockTransport)
that answers POST /run with the RunView of a real offline run. No network, no agent process, no keys."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import uuid
from collections.abc import Callable
from datetime import date
from typing import Any, cast

import httpx
import pytest
from bookkeeper import agent, offline, tools, workflow
from dreaming.connectors.uagents import chat_handler
from dreaming.contracts import Decision, Kind, RunView
from dreaming.hashing import short
from uagents import Protocol
from uagents_core.contrib.protocols.chat import ChatAcknowledgement, ChatMessage, EndSessionContent, TextContent
from uagents_core.models import Model

LINE = "AIRBNB *WEB 583.24"  # in the recorded world's mix: some steps dream, some still call (the guard below says so)
API = "http://api.test"


@functools.cache
def live_run() -> RunView:
    """What POST /run returns: one live run on a runtime that has learned from 1500 offline transactions."""
    rt = offline.build_runtime()
    rt.fast_forward(workflow.W, tools.ledger(1500, seed=0), workflow.run_transaction)
    txn = agent.to_input(LINE)
    with rt.run(workflow.W, txn, "live") as run:
        workflow.run_transaction(txn)
    return RunView(run=run, rows=rt.store.run_rows(run.id))


class FakeCtx:
    """The three things the handler touches on a uagents Context; `log` is shared with the fake API to show the order."""

    def __init__(self, log: list[str]) -> None:
        self.sent: list[Model] = []
        self.log, self.logger, self.session = log, logging.getLogger("test-agent"), uuid.uuid4()

    async def send(self, destination: str, message: Model) -> None:
        self.sent.append(message)
        self.log.append(type(message).__name__)


def client(handle: Callable[[httpx.Request], httpx.Response], log: list[str], seen: list[httpx.Request] | None = None) -> httpx.Client:
    def logged(request: httpx.Request) -> httpx.Response:
        log.append("POST " + request.url.path)
        if seen is not None:
            seen.append(request)
        return handle(request)

    return httpx.Client(transport=httpx.MockTransport(logged))


def deliver(proto: Protocol, ctx: FakeCtx, msg: ChatMessage) -> None:
    """Hand a ChatMessage to the handler the way uagents does: by schema digest, on a Context."""
    handler = proto.signed_message_handlers[Model.build_schema_digest(ChatMessage)]

    async def turn() -> None:
        await handler(cast(Any, ctx), "agent1qsender", msg)

    asyncio.run(turn())


def chat(text: str) -> ChatMessage:
    return ChatMessage(content=[TextContent(text=text)])


def raising(exc: Exception) -> Callable[[httpx.Request], httpx.Response]:
    def handle(request: httpx.Request) -> httpx.Response:
        raise exc

    return handle


def replying(status: int, body: dict[str, Any]) -> Callable[[httpx.Request], httpx.Response]:
    return lambda request: httpx.Response(status, json=body)


def test_a_chat_line_is_acked_relayed_to_the_api_and_answered_in_words() -> None:
    view = live_run()
    split = {r.decision for r in view.rows if r.kind != Kind.effect}
    assert {Decision.predicted, Decision.called} <= split, split  # the guard: this run shows both a dream and a real call
    log: list[str] = []
    seen: list[httpx.Request] = []
    proto = chat_handler(API, workflow.W, agent.to_input, agent.to_reply,
                         http=client(replying(200, view.model_dump(mode="json")), log, seen))
    ctx, msg = FakeCtx(log), chat(LINE)
    deliver(proto, ctx, msg)

    ack, final = ctx.sent  # no "working on it": the planner would take it as the answer
    assert isinstance(ack, ChatAcknowledgement) and ack.acknowledged_msg_id == msg.msg_id
    assert log == ["ChatAcknowledgement", "POST /run", "ChatMessage"]  # ack, the call, the answer
    assert isinstance(final, ChatMessage) and isinstance(final.content[-1], EndSessionContent) and final.text() == agent.to_reply(view)

    (request,) = seen
    assert request.method == "POST" and str(request.url) == f"{API}/run"
    txn = json.loads(request.content)["input"]
    assert json.loads(request.content).keys() == {"input"}
    assert (txn["descriptor"], txn["amount"], txn["merchant"], txn["date"]) == ("AIRBNB *WEB", 583.24, "", date.today().isoformat())
    assert txn["id"].startswith("chat-")

    posted = view.rows[-1]  # the last effect row is what was posted
    assert posted.kind == Kind.effect and f"{posted.response} (post_to_books)" in final.text()
    for row in (r for r in view.rows if r.kind != Kind.effect):
        line = next(ln for ln in final.text().splitlines() if ln.startswith(f"- {row.call_site.split('.')[-1]}:"))
        if row.decision == Decision.predicted:
            assert row.prediction is not None
            assert "a dream answered" in line and short(row.prediction.hashes.weights) in line and str(row.prediction.cert_id) in line
        else:
            assert "the real call answered" in line and "certificate" not in line
    dreamed = sum(r.decision == Decision.predicted for r in view.rows)
    assert f"{dreamed} of 4 steps were answered by a dream" in final.text() and f"${view.run.cost_usd:.4f}" in final.text()


def test_the_reply_says_what_failed_when_the_run_did() -> None:
    view = live_run()
    stuck = view.model_copy(update={"run": view.run.model_copy(update={"failed": True, "stuck_at": "bookkeeper.vendor_lookup"}),
                                    "rows": [r for r in view.rows if r.kind != Kind.effect]})
    assert "Result: the run failed at bookkeeper.vendor_lookup, nothing was posted." in agent.to_reply(stuck)


@pytest.mark.parametrize("text", ["hello", "ADOBE *WEB", "ADOBE *WEB lots", "nan"])
def test_text_that_is_not_a_bank_feed_line_gets_the_usage_and_never_reaches_the_api(text: str) -> None:
    log: list[str] = []
    proto = chat_handler(API, workflow.W, agent.to_input, agent.to_reply, http=client(replying(500, {}), log))
    ctx = FakeCtx(log)
    deliver(proto, ctx, chat(text))
    ack, final = ctx.sent
    assert isinstance(ack, ChatAcknowledgement) and isinstance(final, ChatMessage) and final.text() == agent.USAGE
    assert isinstance(final.content[-1], EndSessionContent) and log == ["ChatAcknowledgement", "ChatMessage"]


def test_a_message_without_text_is_acked_and_nothing_else() -> None:
    log: list[str] = []
    ctx = FakeCtx(log)
    deliver(chat_handler(API, workflow.W, agent.to_input, agent.to_reply, http=client(replying(500, {}), log)), ctx,
            ChatMessage(content=[EndSessionContent()]))
    assert log == ["ChatAcknowledgement"]


FAILURES = [
    (raising(httpx.ConnectError("connection refused")), "ConnectError: connection refused"),
    (raising(httpx.ReadTimeout("")), "ReadTimeout"),
    (replying(503, {"detail": "no workflow configured on this API"}), "503"),
    (replying(200, {"detail": "a body that is not a RunView"}), "RunView"),
]


@pytest.mark.parametrize("handle, named", FAILURES)
def test_an_api_failure_becomes_a_reply_naming_it_not_a_crash(handle: Callable[[httpx.Request], httpx.Response], named: str) -> None:
    log: list[str] = []
    proto = chat_handler(API, workflow.W, agent.to_input, agent.to_reply, http=client(handle, log))
    ctx = FakeCtx(log)
    deliver(proto, ctx, chat(LINE))  # must not raise
    ack, final = ctx.sent
    assert isinstance(ack, ChatAcknowledgement) and isinstance(final, ChatMessage)
    assert named in final.text() and "may not have posted" in final.text()
    assert isinstance(final.content[-1], EndSessionContent)


def test_to_input_reads_descriptor_and_amount() -> None:
    txn = cast(dict[str, Any], agent.to_input("  UBER   TRIP  *WEB  $1,023.50 "))
    assert (txn["descriptor"], txn["amount"], txn["merchant"]) == ("UBER TRIP *WEB", 1023.5, "")
    assert txn["id"] != cast(dict[str, Any], agent.to_input("AWS 12"))["id"]  # every chat is its own transaction
    asked = cast(dict[str, Any], agent.to_input("Process this bank-feed line: `NOTION *WEB 96.00`"))  # how ASI:One's planner relays it
    assert (asked["descriptor"], asked["amount"]) == ("NOTION *WEB", 96.0)


def test_main_attaches_the_chat_protocol_to_a_mailbox_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    from uagents import Agent

    async def no_publish(self: Agent, manifest: dict[str, Any]) -> None:
        """include(publish_manifest=True) schedules a POST to the Almanac; the test stays offline."""

    built: list[Agent] = []
    monkeypatch.setenv("AGENT_SEED", "offline test seed, not a real agent")
    monkeypatch.setattr(Agent, "publish_manifest", no_publish)
    def run(self: Agent) -> None:
        built.append(self)

    monkeypatch.setattr(Agent, "run", run)  # everything but the blocking event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)  # asyncio.run() in the tests above left no current loop, and an Agent takes the current one
    try:
        agent.main()
        loop.run_until_complete(asyncio.sleep(0))  # lets the scheduled (stubbed) publish finish
    finally:
        loop.close()
    (a,) = built
    assert a.name == "dreaming-bookkeeper" and a.address.startswith("agent1q")
    assert [p.name for p in a.protocols.values()] == ["AgentChatProtocol"]
    assert "tag:innovationlab" in a._build_registration_profile().readme  # pyright: ignore[reportPrivateUsage] - /connect publishes the README
