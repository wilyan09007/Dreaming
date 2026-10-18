"""T2 requester: a chat-style handler (Ask -> Answer) that calls the responder through ctx.send_and_receive. The Dreaming change is the marked lines."""
import dreaming_wrap as dw
from t2_common import *
HERE = dw.Path(__file__).parent
TR = HERE / "t2_traces.jsonl"
KNOWN = {}  # stand-in for a trained predictor: learns every live answer, serves it at 0.99 confidence
def predictor(msg): return (VendorInfo(vendor=msg.vendor, terms=KNOWN[msg.vendor]), 0.99) if msg.vendor in KNOWN else (None, 0.0)
RESP = f"{ADDR['responder']}|VendorLookup"  # call site = destination|message type
dw.Recorder("record", TR, predictors={RESP: predictor}, cutoffs={RESP: 0.9},
            fallbacks={RESP: lambda m: VendorInfo(vendor=m.vendor, terms="unknown (fallback)")})  # <- Dreaming line 0: recorder + registered predictor/cutoff/fallback
agent = make("requester")

@agent.on_message(Ask, replies={Answer})
@dw.run_boundary                                                                                  # <- Dreaming line 1 (also wraps ctx.send / ctx.send_and_receive)
async def chat(ctx, sender, msg: Ask):
    info, status = await ctx.send_and_receive(ADDR["responder"], VendorLookup(vendor=msg.text), response_type=VendorInfo, timeout=4)
    if info is None: text = f"{msg.text}: lookup failed ({status.detail})"
    else:
        if "fallback" not in info.terms and info.terms != "unknown": KNOWN.setdefault(info.vendor, info.terms)
        text = f"{info.vendor}: {info.terms}"
    await ctx.send(sender, Answer(text=text))
    return text

if __name__ == "__main__": agent.run()
