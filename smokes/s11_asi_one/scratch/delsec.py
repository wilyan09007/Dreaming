import json, sys
sys.path.insert(0, ".")
import h
a = json.load(open(h.HERE / "hosted.json"))["address"]
h.av("DELETE", "/v1/hosting/secrets", json={"address": a, "name": "ASI1KEY"})
h.av("DELETE", f"/v1/hosting/secrets/ASI1KEY?address={a}")
h.av("GET", f"/v1/hosting/secrets?address={a}")
