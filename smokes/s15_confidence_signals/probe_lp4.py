import json, re, sys, httpx
from openai import OpenAI
sys.path.insert(0, r"C:\Users\<user>\Desktop\personalprojects\Dreaming\smokes")
from common import *
sys.stdout.reconfigure(encoding="utf-8")
key = re.search(r"^OPENROUTERKEY=(.*)$", open(r"C:\Users\<user>\Desktop\personalprojects\Dreaming\.env", encoding="utf-8").read(), re.M).group(1).strip()
cl = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, timeout=120, max_retries=0)
b = load_banking77(); names = b["names"]; texts, labels = b["test"]
idx = stratified_subsample(labels, 200, seed=0)
sysmsg = "You are a bank support intent classifier. Reply with exactly one label from the list, nothing else.\n" + "\n".join(names)
for model in ["qwen/qwen3-30b-a3b-instruct-2507", "meta-llama/llama-3.3-70b-instruct", "mistralai/mistral-small-3.2-24b-instruct", "qwen/qwen3-next-80b-a3b-instruct"]:
    r0 = httpx.get(f"https://openrouter.ai/api/v1/models/{model}/endpoints", timeout=30).json()["data"]["endpoints"]
    print(model, [(e["provider_name"], "logprobs" in e["supported_parameters"], "temperature" in e["supported_parameters"], e["pricing"]["prompt"]) for e in r0])
    for k in (0, 2):
        q = texts[idx[k]]
        try:
            r = cl.chat.completions.create(model=model, messages=[{"role":"system","content":sysmsg},{"role":"user","content":q}], max_tokens=40, logprobs=True, top_logprobs=5, temperature=0)
            lp = r.choices[0].logprobs
            print("  ", repr(r.choices[0].message.content), "prov", getattr(r,"provider",None), "cost", r.usage.cost, "ntok", len(lp.content) if lp and lp.content else 0)
            for t in (lp.content if lp and lp.content else [])[:6]:
                print("      ", repr(t.token), round(t.logprob,4), [(x.token, round(x.logprob,2)) for x in t.top_logprobs][:4])
        except Exception as e:
            print("  ERR", str(e)[:200])
