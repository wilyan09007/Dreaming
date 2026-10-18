"""Revision: poll the Agentverse index until our agent shows up for a query. usage: rev_poll.py "<query>" <max_seconds>"""
import json, sys, time
sys.path.insert(0, ".")
import h
q, mx = sys.argv[1], int(sys.argv[2])
loc = open(h.HERE / "local_addr.txt").read().strip()
t0 = time.time()
while time.time() - t0 < mx:
    s, b = h.av("POST", "/v1/search/agents", json={"search_text": q, "limit": 10}, _show=False)
    hit = [i for i, a in enumerate(b.get("agents", [])) if a.get("address") == loc] if isinstance(b, dict) else []
    if hit:
        h.log(f"POLL {q!r}: visible at rank {hit[0]+1} after {time.time()-t0:.0f}s of polling (UTC {time.strftime('%H:%M:%SZ', time.gmtime())})")
        break
    time.sleep(10)
else:
    h.log(f"POLL {q!r}: NOT visible after {mx}s")
