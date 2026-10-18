import json, re, sys, httpx
from openai import OpenAI
sys.stdout.reconfigure(encoding="utf-8")
key = re.search(r"^OPENROUTERKEY=(.*)$", open(r"C:\Users\<user>\Desktop\personalprojects\Dreaming\.env", encoding="utf-8").read(), re.M).group(1).strip()
cl = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, timeout=120, max_retries=0)
# all endpoints' full supported parameter lists
for m in ["openai/gpt-5.4-nano", "openai/gpt-4.1-nano", "openai/gpt-4.1-mini", "openai/gpt-4o-mini", "qwen/qwen3-32b"]:
    r = httpx.get(f"https://openrouter.ai/api/v1/models/{m}/endpoints", timeout=30)
    for e in r.json()["data"]["endpoints"]:
        print(m, e["provider_name"], e.get("tag"), sorted(e.get("supported_parameters", [])))
msgs = [{"role": "system", "content": "Reply with exactly one label from the list, nothing else.\nactivate_my_card\nage_limit\ncard_arrival"}, {"role": "user", "content": "How old do I have to be to open an account?"}]
tests = [
 ("openai/gpt-5.4-nano", {"reasoning": {"effort": "none"}}, {}),
 ("openai/gpt-5.4-nano", {"reasoning": {"effort": "none"}, "provider": {"order": ["OpenAI"], "allow_fallbacks": False, "require_parameters": True}}, {}),
 ("openai/gpt-4.1-nano", {}, {}),
 ("openai/gpt-4.1-nano", {"provider": {"require_parameters": True}}, {}),
 ("qwen/qwen3-32b", {"provider": {"order": ["Tenstorrent"], "allow_fallbacks": False}}, {}),
 ("qwen/qwen3-32b", {"provider": {"order": ["DeepInfra"], "allow_fallbacks": False}}, {}),
]
for model, eb, kw in tests:
    try:
        r = cl.chat.completions.create(model=model, messages=msgs, max_tokens=30, logprobs=True, top_logprobs=5, temperature=0, extra_body=eb)
        lp = r.choices[0].logprobs
        n = len(lp.content) if lp and lp.content else 0
        print(model, json.dumps(eb)[:90], "-> content", repr(r.choices[0].message.content), "provider", getattr(r, "provider", None), "logprob tokens", n, "usage.cost", getattr(r.usage, "cost", None))
        if n: print("   first token:", lp.content[0].token, lp.content[0].logprob, [(t.token, round(t.logprob, 3)) for t in lp.content[0].top_logprobs])
    except Exception as e:
        print(model, json.dumps(eb)[:90], "ERR", str(e)[:300])
