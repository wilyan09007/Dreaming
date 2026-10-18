# Uploaded verbatim to Agentverse as agent.py (hosted). Echo + optional asi1-mini call.
import os
from datetime import datetime, timezone
from uuid import uuid4
import requests
from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement, ChatMessage, EndSessionContent, TextContent, chat_protocol_spec)

agent = Agent()
proto = Protocol(spec=chat_protocol_spec)


def asi(text):
    k = os.getenv("ASI1KEY")
    if not k:
        return "(no ASI1KEY in env)"
    r = requests.post("https://api.asi1.ai/v1/chat/completions",
                      headers={"Authorization": "Bearer " + k},
                      json={"model": "asi1-mini", "max_tokens": 40,
                            "messages": [{"role": "user", "content": text}]}, timeout=30)
    return f"[{r.status_code}] " + (r.json()["choices"][0]["message"]["content"] if r.ok else r.text[:200])


@agent.on_event("startup")
async def up(ctx: Context):
    ctx.logger.info(f"dreaming hosted up; ASI1KEY present={bool(os.getenv('ASI1KEY'))}")


@proto.on_message(ChatMessage)
async def on_msg(ctx: Context, sender: str, msg: ChatMessage):
    await ctx.send(sender, ChatAcknowledgement(timestamp=datetime.now(timezone.utc), acknowledged_msg_id=msg.msg_id))
    text = " ".join(c.text for c in msg.content if isinstance(c, TextContent))
    ctx.logger.info(f"got chat from {sender}: {text!r}")
    if not text:
        return
    out = "dreaming: " + text
    if text.startswith("llm "):
        out += " | asi1-mini " + asi(text[4:])
    await ctx.send(sender, ChatMessage(timestamp=datetime.now(timezone.utc), msg_id=uuid4(),
                                       content=[TextContent(type="text", text=out), EndSessionContent(type="end-session")]))


@proto.on_message(ChatAcknowledgement)
async def on_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"ack from {sender}")


agent.include(proto, publish_manifest=True)
