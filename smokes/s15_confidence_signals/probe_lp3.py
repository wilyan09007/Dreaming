import json, re, sys, httpx
sys.stdout.reconfigure(encoding="utf-8")
key = re.search(r"^OPENROUTERKEY=(.*)$", open(r"C:\Users\<user>\Desktop\personalprojects\Dreaming\.env", encoding="utf-8").read(), re.M).group(1).strip()
H = {"Authorization": "Bearer " + key}
# models advertising logprobs, cheap
ms = httpx.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]
c = []
for m in ms:
    sp = m.get("supported_parameters") or []
    try: pr = float(m["pricing"]["prompt"]) * 1e6; co = float(m["pricing"]["completion"]) * 1e6
    except Exception: continue
    if "logprobs" in sp and "top_logprobs" in sp and pr <= 0.5 and co <= 1.5: c.append((pr, co, m["id"]))
for x in sorted(c): print("%.3f %.3f %s" % x)
# raw tenstorrent response
body = {"model": "qwen/qwen3-32b", "provider": {"order": ["Tenstorrent"], "allow_fallbacks": False}, "max_tokens": 40, "temperature": 0, "logprobs": True, "top_logprobs": 3,
        "messages": [{"role": "system", "content": "Reply with exactly one label from the list, nothing else.\nactivate_my_card\nage_limit\ncard_arrival"}, {"role": "user", "content": "How old do I have to be? /no_think"}]}
r = httpx.post("https://openrouter.ai/api/v1/chat/completions", headers=H, json=body, timeout=60).json()
print(json.dumps(r["choices"][0], ensure_ascii=False)[:1800])
body["messages"][1]["content"] = "How old do I have to be?"
r = httpx.post("https://openrouter.ai/api/v1/chat/completions", headers=H, json=body | {"max_tokens": 400}, timeout=60).json()
print(json.dumps(r["choices"][0], ensure_ascii=False)[:1500])
