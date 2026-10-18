"""Second local uAgent (not on Agentverse). usage: client_agent.py <target_address> "<text>"
Sends one chat message, prints the reply, exits. Its reply path is http://127.0.0.1:8001 (works because target runs on this machine)."""
import os, pathlib, re, sys
from datetime import datetime, timezone
from uuid import uuid4

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import h
from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import ChatAcknowledgement, ChatMessage, TextContent, chat_protocol_spec

SEED = dict(re.findall(r"^(\w+)=(.*)$", (h.HERE / "seeds.txt").read_text(), re.M))["client"]
agent = Agent(name="dreaming-smoke-client", seed=SEED, port=8001, network="testnet", endpoint=["http://127.0.0.1:8001/submit"], enable_agent_inspector=False)
proto = Protocol(spec=chat_protocol_spec)


@proto.on_message(ChatMessage)
async def on_msg(ctx: Context, sender: str, m: ChatMessage):
    await ctx.send(sender, ChatAcknowledgement(timestamp=datetime.now(timezone.utc), acknowledged_msg_id=m.msg_id))
    ctx.logger.info("REPLY: " + repr(" ".join(c.text for c in m.content if isinstance(c, TextContent))))
    os._exit(0)


@proto.on_message(ChatAcknowledgement)
async def on_ack(ctx: Context, sender: str, m: ChatAcknowledgement):
    ctx.logger.info("ACK received")


agent.include(proto)


@agent.on_event("startup")
async def go(ctx: Context):
    import asyncio
    await asyncio.sleep(8)  # let almanac registration land
    ctx.logger.info(f"sending to {sys.argv[1]}")
    await ctx.send(sys.argv[1], ChatMessage(timestamp=datetime.now(timezone.utc), msg_id=uuid4(), content=[TextContent(type="text", text=sys.argv[2])]))
    await asyncio.sleep(40)
    ctx.logger.info("TIMEOUT no reply")
    os._exit(1)


if __name__ == "__main__":
    agent.run()
