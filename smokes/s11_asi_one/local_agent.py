"""Local uAgent speaking the chat protocol. usage: local_agent.py endpoint|mailbox [send_to_address]
Replies 'dreaming: <echo>' + a demo certificate line; 'llm <q>' also calls asi1-mini."""
import asyncio, hashlib, pathlib, re, sys
from datetime import datetime, timezone
from uuid import uuid4
import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import h
from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement, ChatMessage, EndSessionContent, TextContent, chat_protocol_spec)
from uagents_core.utils.registration import RegistrationRequestCredentials, register_chat_agent

MODE = sys.argv[1]
SEND_TO = sys.argv[2] if len(sys.argv) > 2 else None
SEED = dict(re.findall(r"^(\w+)=(.*)$", (h.HERE / "seeds.txt").read_text(), re.M))["local"]
TUNNEL = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", (h.HERE / "scratch/cf.log").read_text())[0] if MODE == "endpoint" else None
README = (h.HERE / "README_agent.md").read_text()

kw = dict(endpoint=[f"{TUNNEL}/submit"]) if MODE == "endpoint" else dict(mailbox=True)
agent = Agent(name="dreaming-smoke-local", seed=SEED, port=8000, network="testnet",
              description="Dreaming runtime smoke agent (local).", readme_path=str(h.HERE / "README_agent.md"), **kw)
proto = Protocol(spec=chat_protocol_spec)


def ask_asi(q):
    r = requests.post("https://api.asi1.ai/v1/chat/completions", headers={"Authorization": "Bearer " + h.ASI},
                      json={"model": "asi1-mini", "max_tokens": 40, "messages": [{"role": "user", "content": q}]}, timeout=30)
    return f"[{r.status_code}] " + (r.json()["choices"][0]["message"]["content"] if r.ok else r.text[:200])


def msg(text, end=True):
    c = [TextContent(type="text", text=text)] + ([EndSessionContent(type="end-session")] if end else [])
    return ChatMessage(timestamp=datetime.now(timezone.utc), msg_id=uuid4(), content=c)


@proto.on_message(ChatMessage)
async def on_msg(ctx: Context, sender: str, m: ChatMessage):
    await ctx.send(sender, ChatAcknowledgement(timestamp=datetime.now(timezone.utc), acknowledged_msg_id=m.msg_id))
    text = " ".join(c.text for c in m.content if isinstance(c, TextContent))
    ctx.logger.info(f"CHAT from {sender}: {text!r}")
    if not text or text.startswith("dreaming:"):  # never answer our own echoes (avoids ping-pong)
        return
    out = "dreaming: " + text
    if text.startswith("interim "):  # revision: send a non-final progress message first, then the final reply after N s
        await ctx.send(sender, msg("dreaming: step running, final result follows", end=False))
        await asyncio.sleep(int(text.split()[1]))
    if text.startswith("slow "):  # revision: how long does ASI:One wait for a reply? (ack already sent above)
        await asyncio.sleep(int(text.split()[1]))
    if text.startswith("llm "):
        out += " | asi1-mini " + await asyncio.to_thread(ask_asi, text[4:])
    out += "\ncertificate: demo-sha256:" + hashlib.sha256(out.encode()).hexdigest()[:16]
    await ctx.send(sender, msg(out))


@proto.on_message(ChatAcknowledgement)
async def on_ack(ctx: Context, sender: str, m: ChatAcknowledgement):
    ctx.logger.info(f"ACK from {sender}")


agent.include(proto, publish_manifest=True)


@agent.on_event("startup")
async def up(ctx: Context):
    ctx.logger.info(f"MODE={MODE} address={agent.address} tunnel={TUNNEL if MODE == 'endpoint' else '-'}")
    if MODE == "endpoint":
        ok = await asyncio.to_thread(
            register_chat_agent, "dreaming-smoke-local", f"{TUNNEL}/submit", True,
            RegistrationRequestCredentials(agentverse_api_key=h.AV, agent_seed_phrase=SEED),
            description="Dreaming runtime smoke agent (local, tunnel).", readme=README,
            starter_prompts=["hello dreaming", "llm say hi in 3 words"])
        ctx.logger.info(f"register_chat_agent -> {ok}")
    if SEND_TO:
        await asyncio.sleep(20)
        ctx.logger.info(f"sending to {SEND_TO}")
        await ctx.send(SEND_TO, msg("llm say hi in 3 words", end=False))


if __name__ == "__main__":
    agent.run()
