"""Revision: print live state (no keys printed). usage: rev_state.py"""
import base64, json, re, sys, datetime as dt
sys.path.insert(0, ".")
import h
loc = open(h.HERE / "local_addr.txt").read().strip()
hosted = json.load(open(h.HERE / "hosted.json"))["address"]
seeds = dict(re.findall(r"^(\w+)=(.*)$", (h.HERE / "seeds.txt").read_text(), re.M))
def jwt(tok):
    try:
        p = json.loads(base64.urlsafe_b64decode(tok.split(".")[1] + "=="))
        return {k: dt.datetime.fromtimestamp(p[k], dt.timezone.utc).isoformat() for k in ("iat", "exp") if k in p}
    except Exception as e:
        return f"not a decodable JWT ({type(e).__name__})"
h.log("AVKEY jwt:", jwt(h.AV))
h.log("ASIKEY jwt:", jwt(h.ASI))
for n, a in (("local", loc), ("hosted", hosted)):
    s, b = h.av("GET", f"/v1/almanac/agents/{a}")
s, b = h.av("GET", f"/v1/hosting/agents/{hosted}")
s, b = h.av("GET", f"/v1/agents/{loc}")
