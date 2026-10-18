"""uAgents: the chat handler is the front door (done), `ctx.send_and_receive` as an a2a call site is not (stub).

The demo agent runs locally with an Agentverse mailbox (outbound polling only) and is reachable from ASI:One
through the Agent Chat Protocol. A user's message becomes `POST /run` on the API, which owns the policies and
certificates; the reply carries which calls were predicted and under which certificates. Proven in
smokes/s11_asi_one (mailbox connect is one POST to the agent's own /connect) and smokes/s12_wrappers
(send_and_receive wrapper, outage answered from the fallback in 2 s).
shortcut: `wrap_ctx` is not ported; the demo agent never calls another agent, so nothing in the demo path needs it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast

from ..contracts import Json, RunView

if TYPE_CHECKING:
    import httpx
    from uagents import Protocol

    from ..runtime import Runtime

TIMEOUT_S = 40.0  # ASI:One's planner waits 60 s for the first reply (S11): a turn that fails must still answer inside that


def wrap_ctx(ctx: Any, runtime: Runtime) -> Any:
    """Route `ctx.send_and_receive(dest, msg, response_type)` through the runtime; `ctx.send` is an effect, logged only."""
    raise NotImplementedError("port from smokes/s12_wrappers/dreaming_wrap.py::wrap_send_and_receive")


def chat_handler(api_base: str, workflow: str, to_input: Callable[[str], Json], to_reply: Callable[[RunView], str],
                 http: httpx.Client | None = None) -> Protocol:
    """The Agent Chat Protocol for one workflow: ack, `POST {api_base}/run`, reply with the API's own RunView in words.

    The agent only relays: the API holds the policies and runs the workflow, so this process needs no store and can never
    serve a stale certificate. `to_input` turns the user's text into the workflow input (a ValueError is the reply: the
    usage hint); `to_reply` turns the run into the answer. Any failure after the interim message becomes a short reply
    naming it, because ASI:One waits for one and a crashed handler leaves the user with silence. `workflow` only labels
    the messages: POST /run has no workflow selector.
    """
    import httpx
    from uagents import Context, Protocol
    from uagents_core.contrib.protocols.chat import (
        ChatAcknowledgement,
        ChatMessage,
        EndSessionContent,
        TextContent,
        chat_protocol_spec,
    )

    client = http or httpx.Client(timeout=TIMEOUT_S)
    proto = Protocol(spec=chat_protocol_spec)
    # uagents' decorator is untyped: say so once here instead of ignoring it at every handler
    on = cast("Callable[[type[Any]], Callable[[Callable[..., Awaitable[None]]], Any]]", proto.on_message)  # pyright: ignore[reportUnknownMemberType]

    def say(text: str, *, end: bool = True) -> ChatMessage:
        return ChatMessage(content=[TextContent(text=text), *([EndSessionContent()] if end else [])])

    @on(ChatMessage)
    async def on_message(ctx: Context, sender: str, msg: ChatMessage) -> None:
        await ctx.send(sender, ChatAcknowledgement(acknowledged_msg_id=msg.msg_id))  # ack first, as the chat protocol expects
        text = msg.text().strip()
        ctx.logger.info(f"chat from {sender} (session {ctx.session}): {text!r}")
        if not text:
            return
        try:
            body: Json = {"input": to_input(text)}
        except ValueError as e:
            await ctx.send(sender, say(str(e)))
            return
        await ctx.send(sender, say(f"Working on it: running the {workflow} workflow, this can take up to half a minute.", end=False))
        try:
            r = await asyncio.to_thread(client.post, f"{api_base}/run", json=body, timeout=TIMEOUT_S)  # off the loop: the mailbox keeps polling
            r.raise_for_status()
            out = to_reply(RunView.model_validate(r.json()))
        except Exception as e:  # noqa: BLE001 - the edge: HTTP errors, timeouts, a body that is not a RunView
            ctx.logger.error(f"{workflow} run failed: {type(e).__name__}: {e}")
            why = f"{type(e).__name__}: {e}".split("\n")[0] if str(e) else type(e).__name__
            out = f"Sorry, I could not get the outcome of the {workflow} run ({why}). It may not have posted; please try again."
        await ctx.send(sender, say(out))

    @on(ChatAcknowledgement)
    async def on_ack(ctx: Context, sender: str, msg: ChatAcknowledgement) -> None:  # the spec lists it, so include() demands a handler
        ctx.logger.info(f"ack from {sender}")

    return proto
