import json, sys
sys.path.insert(0, ".")
import h
a = json.load(open(h.HERE / "hosted.json"))["address"]
h.av("POST", "/v1/hosting/secrets", json={"address": a, "name": "ASI1KEY", "secret": "REVOKED-PLACEHOLDER"})
h.av("GET", f"/v1/hosting/secrets?address={a}")
