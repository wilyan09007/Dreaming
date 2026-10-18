"""Tiny helpers shared by the s11 scripts. Keys are read from ../../.env and never printed."""
import json, os, re, sys, time, pathlib
import requests

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
_env = dict(re.findall(r"^(\w+)=(.*)$", (ROOT / ".env").read_text(), re.M))
AV, ASI = _env["AGENTVERSEKEY"].strip(), _env["ASI1KEY"].strip()
AVBASE = "https://agentverse.ai"


def log(*a):
    s = " ".join(str(x) for x in a)
    for k in (AV, ASI):
        s = s.replace(k, "<key>")
    print(s, flush=True)


def av(method, path, **kw):
    show = kw.pop("_show", True)
    r = requests.request(method, AVBASE + path, headers={"Authorization": f"Bearer {AV}"}, timeout=60, **kw)
    try:
        body = r.json()
    except Exception:
        body = r.text[:500]
    log(f"AV {method} {path} -> {r.status_code}", json.dumps(body)[:600] if show else "")
    return r.status_code, body


def asi(payload, headers=None, path="/chat/completions", method="POST"):
    h = {"Authorization": f"Bearer {ASI}", **(headers or {})}
    r = requests.request(method, "https://api.asi1.ai/v1" + path, headers=h, json=payload if method == "POST" else None, timeout=180)
    try:
        body = r.json()
    except Exception:
        body = r.text[:500]
    log(f"ASI {method} {path} -> {r.status_code}", json.dumps(body)[:800])
    return r.status_code, body
