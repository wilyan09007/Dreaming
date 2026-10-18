"""T2c chat agent: an ASI:One-style chat-protocol handler (ChatMessage in; ChatAcknowledgement + ChatMessage out) with BOTH an agent-to-agent lookup (ctx.send_and_receive)
and a model call (OpenAI-compatible async client; ASI:One's API is OpenAI-compatible, OpenRouter stands in) inside one run. The Dreaming lines are marked."""
import httpx, openai
import dreaming_wrap as dw
from uagents import Protocol
from uagents_core.contrib.protocols.chat import ChatAcknowledgement, ChatMessage, TextContent, chat_protocol_spec
from t2_common import *
from _t import GEN, HERE, KEY

TR = HERE / "t2c_traces.jsonl"
SYS = "Reply with exactly one line: 'Payment terms for <vendor>: <terms>.'"
KNOWN, SAID = {}, {}  # stand-ins for trained predictors: learn every live answer
def predictor(msg): return (VendorInfo(vendor=msg.vendor, terms=KNOWN[msg.vendor]), 0.99) if msg.vendor in KNOWN else (None, 0.0)
RESP = f"{ADDR['responder']}|VendorLookup"
dw.Recorder("record", TR, predictors={RESP: predictor, "reply": SAID}, cutoffs={RESP: 0.9},
            fallbacks={RESP: lambda m: VendorInfo(vendor=m.vendor, terms="unknown (fallback)")})                       # <- Dreaming line 0: recorder
async def _hook(req): print("MODEL HTTP request", req.url.path, flush=True)  # proof of every request that really leaves this process
llm = dw.wrap_openai(openai.AsyncOpenAI(api_key=KEY, base_url="https://openrouter.ai/api/v1", http_client=httpx.AsyncClient(event_hooks={"request": [_hook]}, timeout=60)))  # <- line 1: model client
agent = make("chat")
chat = Protocol(spec=chat_protocol_spec)

@chat.on_message(ChatMessage)
@dw.run_boundary                                                                                                      # <- line 2: run boundary (also wraps ctx.send / send_and_receive)
async def on_chat(ctx, sender, msg: ChatMessage):
    await ctx.send(sender, ChatAcknowledgement(acknowledged_msg_id=msg.msg_id))                                      # effect: the ack carries datetime.now()
    vendor = msg.text().strip()
    info, st = await ctx.send_and_receive(ADDR["responder"], VendorLookup(vendor=vendor), response_type=VendorInfo, timeout=4)  # a2a call site
    if info and "fallback" not in info.terms and info.terms != "unknown": KNOWN.setdefault(info.vendor, info.terms)
    messages = [{"role": "system", "content": SYS}, {"role": "user", "content": f"{vendor}: {info.terms if info else 'lookup failed'}"}]
    with dw.call_site("reply"): r = await llm.chat.completions.create(**GEN, messages=messages)                       # model call site
    text = r.choices[0].message.content.strip(); SAID[dw.sha(messages)] = text
    await ctx.send(sender, ChatMessage(content=[TextContent(type="text", text=text)]))
    return text

@chat.on_message(ChatAcknowledgement)
async def on_ack(ctx, sender, msg: ChatAcknowledgement): pass

agent.include(chat, publish_manifest=False)  # publish_manifest=True would POST to Agentverse
if __name__ == "__main__": agent.run()
