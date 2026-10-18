"""Q1 step 1: create hosted agent, set secret, upload code. Prints address; saves to hosted.json."""
import json, sys
sys.path.insert(0, ".")
import h

code = open(h.HERE / "hosted_agent.py").read()
s, a = h.av("POST", "/v1/hosting/agents", json={"name": "dreaming-smoke-hosted"})
addr = a["address"]
json.dump({"address": addr}, open(h.HERE / "hosted.json", "w"))
# secret for the ASI:One key (value never logged)
s, _ = h.av("POST", "/v1/hosting/secrets", json={"address": addr, "name": "ASI1KEY", "secret": h.ASI})
h.av("GET", f"/v1/hosting/secrets?address={addr}")
h.av("PUT", f"/v1/hosting/agents/{addr}/code", json={"code": [{"id": 0, "name": "agent.py", "value": code, "language": "python"}]})
h.av("GET", f"/v1/hosting/agents/{addr}")
