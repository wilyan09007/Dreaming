"""Revision: Agentverse agent search (free). usage: rev_search.py "<query>" [N]. Prints rank of our two agents among results."""
import json, sys
sys.path.insert(0, ".")
import h
q, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 30
ours = {open(h.HERE / "local_addr.txt").read().strip(): "LOCAL", json.load(open(h.HERE / "hosted.json"))["address"]: "HOSTED"}
s, b = h.av("POST", "/v1/search/agents", json={"search_text": q, "limit": n, "sort": "relevancy", "direction": "desc"}, _show=False)
ag = b.get("agents", []) if isinstance(b, dict) else []
h.log(f"SEARCH {q!r} -> {s}, {len(ag)} results, total={b.get('total') if isinstance(b, dict) else None}")
for i, a in enumerate(ag):
    if a.get("address") in ours:
        h.log(f"  OURS {ours[a['address']]} rank {i+1} name={a.get('name')} status={a.get('status')} type={a.get('type')} last_updated={a.get('last_updated')}")
h.log("  top3:", [(a.get("name"), a.get("address", "")[:12]) for a in ag[:3]])
