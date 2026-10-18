"""T2 responder process: VendorLookup -> VendorInfo. Prints one line per request it actually receives (stdout is the proof that a send happened or was skipped)."""
import sys
from t2_common import *
DB = {"acme": "net30", "globex": "net60", "initech": "net15"}
agent = make("responder")

@agent.on_message(VendorLookup, replies={VendorInfo})
async def lookup(ctx, sender, msg: VendorLookup):
    print(f"RESPONDER got VendorLookup({msg.vendor})", flush=True)
    await ctx.send(sender, VendorInfo(vendor=msg.vendor, terms=DB.get(msg.vendor, "unknown")))

if __name__ == "__main__": agent.run()
