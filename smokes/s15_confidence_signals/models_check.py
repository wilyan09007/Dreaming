import httpx, json, re, sys
sys.stdout.reconfigure(encoding="utf-8")
for m in ["openai/gpt-5.4-nano", "openai/gpt-4.1-nano", "qwen/qwen3-32b", "openai/gpt-5.4"]:
    r = httpx.get(f"https://openrouter.ai/api/v1/models/{m}/endpoints", timeout=30)
    if r.status_code != 200: print(m, r.status_code, r.text[:200]); continue
    d = r.json()["data"]
    print(m, "pricing", d["endpoints"][0].get("pricing"))
    for e in d["endpoints"]:
        sp = e.get("supported_parameters", [])
        print("   ", e.get("provider_name"), e.get("tag"), "logprobs" in sp, "top_logprobs" in sp, "temperature" in sp, "reasoning" in sp)
