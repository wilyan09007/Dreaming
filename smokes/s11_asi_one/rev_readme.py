"""Revision: push README_agent.md to the Agentverse agent record by REST (mailbox start does not sync it)."""
import sys
sys.path.insert(0, ".")
import h
loc = open(h.HERE / "local_addr.txt").read().strip()
s, b = h.av("PUT", f"/v1/agents/{loc}", json={"name": "dreaming-smoke-local", "readme": (h.HERE / "README_agent.md").read_text(), "short_description": "Dreaming runtime smoke agent (local)."})
s, b = h.av("GET", f"/v1/agents/{loc}")
h.log("readme now ends with quokka:", "quokka-7731" in str(b))
