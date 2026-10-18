"""Q1 step 2: upload code (code field is a JSON-encoded string of the file list), start, read logs."""
import json, sys, time
sys.path.insert(0, ".")
import h

addr = json.load(open(h.HERE / "hosted.json"))["address"]
code = open(h.HERE / "hosted_agent.py").read()
files = [{"id": 0, "name": "agent.py", "value": code, "language": "python"}]
s, b = h.av("PUT", f"/v1/hosting/agents/{addr}/code", json={"code": json.dumps(files)})
if s == 200:
    h.av("POST", f"/v1/hosting/agents/{addr}/start")
    for i in range(8):
        time.sleep(5)
        s, b = h.av("GET", f"/v1/hosting/agents/{addr}/logs/latest")
        s2, a = h.av("GET", f"/v1/hosting/agents/{addr}")
        if isinstance(b, list) and any("up" in str(x) or "rror" in str(x) for x in b):
            break
