"""Revision: hosted agent start/stop/logs/secret-delete. usage: rev_hosted.py start|stop|logs|delsecret"""
import json, sys
sys.path.insert(0, ".")
import h
a = json.load(open(h.HERE / "hosted.json"))["address"]
cmd = sys.argv[1]
if cmd in ("start", "stop"):
    h.av("POST", f"/v1/hosting/agents/{a}/{cmd}", _show=False)
    s, b = h.av("GET", f"/v1/hosting/agents/{a}", _show=False)
    h.log(f"hosted running={b.get('running')} compiled={b.get('compiled')}")
elif cmd == "logs":
    s, b = h.av("GET", f"/v1/hosting/agents/{a}/logs/latest", _show=False)
    for e in b[:12][::-1]:
        h.log("HOSTED-LOG", e["log_timestamp"], e["log_entry"][:160])
elif cmd == "delsecret":
    h.av("GET", f"/v1/hosting/secrets?address={a}")
    h.av("DELETE", f"/v1/hosting/secrets?address={a}&name=ASI1KEY")
    h.av("GET", f"/v1/hosting/secrets?address={a}")
