"""Revision: ASI:One /usage summary (numbers only)."""
import json, sys
sys.path.insert(0, ".")
import h
s, b = h.asi(None, path="/usage", method="GET") if False else (None, None)
import requests
r = requests.get("https://api.asi1.ai/v1/usage", headers={"Authorization": f"Bearer {h.ASI}"}, timeout=30)
b = r.json()
h.log(f"ASI GET /usage -> {r.status_code} exceeded_limits={b['exceeded_limits']} exceeded_token_limits={b['exceeded_token_limits']} daily={json.dumps(b['daily_usage'])} 15min_windows={json.dumps(b['usage'])} eon_resets_at={b['eon_resets_at']}")
