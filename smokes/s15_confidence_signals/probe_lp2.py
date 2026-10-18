import json, re, sys
from openai import OpenAI
sys.path.insert(0, r"C:\Users\<user>\Desktop\personalprojects\Dreaming\smokes")
from common import *
sys.stdout.reconfigure(encoding="utf-8")
key = re.search(r"^OPENROUTERKEY=(.*)$", open(r"C:\Users\<user>\Desktop\personalprojects\Dreaming\.env", encoding="utf-8").read(), re.M).group(1).strip()
cl = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, timeout=120, max_retries=0)
b = load_banking77(); names = b["names"]; texts, labels = b["test"]
idx = stratified_subsample(labels, 200, seed=0)
sysmsg = "You are a bank support intent classifier. Reply with exactly one label from the list, nothing else.\n" + "\n".join(names)
def show(model, eb, q, suffix="", **kw):
    try:
        r = cl.chat.completions.create(model=model, messages=[{"role":"system","content":sysmsg},{"role":"user","content":q+suffix}], max_tokens=60, logprobs=True, top_logprobs=5, temperature=0, extra_body=eb, **kw)
        lp = r.choices[0].logprobs
        print(model, eb.get("provider"), "->", repr(r.choices[0].message.content), "prov", getattr(r,"provider",None), "cost", r.usage.cost)
        for t in (lp.content if lp and lp.content else []):
            print("    ", repr(t.token), round(t.logprob,4), [(x.token, round(x.logprob,3)) for x in t.top_logprobs])
    except Exception as e:
        print(model, "ERR", str(e)[:200])
for k in range(3):
    q = texts[idx[k]]; print("Q:", q, "| gold", names[labels[idx[k]]])
    show("openai/gpt-5.4-nano", {"reasoning":{"effort":"none"}}, q)
    show("openai/gpt-4.1-nano", {}, q)
    show("qwen/qwen3-32b", {"provider":{"order":["Tenstorrent"],"allow_fallbacks":False}}, q, " /no_think")
    show("qwen/qwen3-32b", {"provider":{"order":["DeepInfra"],"allow_fallbacks":False}}, q, " /no_think")
