import json, re, sys, httpx
from openai import OpenAI
sys.stdout.reconfigure(encoding="utf-8")
key = re.search(r"^OPENROUTERKEY=(.*)$", open(r"C:\Users\<user>\Desktop\personalprojects\Dreaming\.env", encoding="utf-8").read(), re.M).group(1).strip()
cl = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, timeout=120, max_retries=0)
for e in httpx.get("https://openrouter.ai/api/v1/models/qwen/qwen3-32b/endpoints", timeout=30).json()["data"]["endpoints"]:
    print(e["provider_name"], e.get("tag"), "logprobs" in e["supported_parameters"], "top_logprobs" in e["supported_parameters"], "temperature" in e["supported_parameters"], e["pricing"]["prompt"], e["pricing"]["completion"])
sysm = "You are a bank support intent classifier. Reply with exactly one label from the list, nothing else.\nactivate_my_card\nage_limit\ncard_arrival\nrefund_not_showing_up"
for prov in ["DeepInfra", "Tenstorrent", "Groq", "Cerebras", "Nebius", "Together", "Fireworks", "SambaNova"]:
    try:
        r = cl.chat.completions.create(model="qwen/qwen3-32b", messages=[{"role":"system","content":sysm},{"role":"user","content":"How old do I have to be to open an account? /no_think"}], max_tokens=40, logprobs=True, top_logprobs=5, temperature=0, extra_body={"provider":{"order":[prov],"allow_fallbacks":False}})
        lp = r.choices[0].logprobs; n = len(lp.content) if lp and lp.content else 0
        print(prov, repr(r.choices[0].message.content), "served", getattr(r,"provider",None), "lp_tokens", n, "cost", getattr(r.usage,"cost",None))
        if n: print("   ", [(t.token, round(t.logprob,3), [(a.token, round(a.logprob,2)) for a in t.top_logprobs][:3]) for t in lp.content[:4]])
    except Exception as ex:
        print(prov, "ERR", str(ex)[:160])
