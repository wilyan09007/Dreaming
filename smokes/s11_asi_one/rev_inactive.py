"""Revision: mark agents inactive via uagents_core update_agent_status (signed POST {almanac}/agents/{addr}/status) and read back
the status from the Almanac GET and from Agentverse search. usage: rev_inactive.py local|client"""
import re, sys
sys.path.insert(0, ".")
import h
from uagents_core.identity import Identity
from uagents_core.utils import registration as reg
who = sys.argv[1]
seed = dict(re.findall(r"^(\w+)=(.*)$", (h.HERE / "seeds.txt").read_text(), re.M))[who]
ident = Identity.from_seed(seed, 0)
try:
    reg._update_agent_status(False, ident)  # raises with the HTTP error, unlike the bool wrapper
    h.log(f"update_agent_status({who}, active=False) -> OK (no exception, HTTP 2xx)")
except Exception as e:
    h.log(f"update_agent_status({who}, active=False) -> FAILED {type(e).__name__}: {str(e)[:200]}")
s, b = h.av("GET", f"/v1/almanac/agents/{ident.address}", _show=False)
h.log(f"almanac GET {who}: HTTP {s} status={b.get('status') if isinstance(b, dict) else b} type={b.get('type') if isinstance(b, dict) else None} expiry={b.get('expiry') if isinstance(b, dict) else None} endpoints={b.get('endpoints') if isinstance(b, dict) else None}")
