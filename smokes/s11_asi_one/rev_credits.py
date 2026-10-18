"""Revision: read-only balance checks for the other keys in .env. Prints only numeric/boolean fields, never keys."""
import json, re, sys, requests
sys.path.insert(0, ".")
import h
env = dict(re.findall(r"^(\w+)=(.*)$", (h.ROOT / ".env").read_text(), re.M))
OR, RP = env["OPENROUTERKEY"].strip(), env["RUNPODKEY"].strip()
def show(tag, r, keep):
    try:
        b = r.json()
    except Exception:
        b = {}
    d = b.get("data", b) if isinstance(b, dict) else {}
    h.log(f"{tag} -> {r.status_code}", json.dumps({k: d.get(k) for k in keep if isinstance(d, dict) and k in d}))
H = {"Authorization": f"Bearer {OR}"}
show("OpenRouter GET /api/v1/credits", requests.get("https://openrouter.ai/api/v1/credits", headers=H, timeout=30), ["total_credits", "total_usage", "error"])
show("OpenRouter GET /api/v1/key", requests.get("https://openrouter.ai/api/v1/key", headers=H, timeout=30), ["usage", "limit", "limit_remaining", "is_free_tier", "expires_at", "error"])
r = requests.post("https://api.runpod.io/graphql", headers={"Authorization": f"Bearer {RP}"}, json={"query": "query { myself { clientBalance currentSpendPerHr spendLimit } }"}, timeout=30)
try:
    m = r.json().get("data", {}).get("myself") or r.json()
except Exception:
    m = {}
h.log(f"RunPod POST graphql myself -> {r.status_code}", json.dumps({k: m.get(k) for k in ("clientBalance", "currentSpendPerHr", "spendLimit", "errors", "message") if isinstance(m, dict) and k in m}))
