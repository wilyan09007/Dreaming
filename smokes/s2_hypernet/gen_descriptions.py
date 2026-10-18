"""One-line label descriptions for CLINC150 intents and Banking77 labels (the 'schema' a step declares).
Cached to smokes/.cache/label_descriptions.json. ~$0.01 on openai/gpt-5.4-nano."""
import json, os, sys, re
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from common import ROOT, CACHE, load_banking77, load_clinc150
from openai import OpenAI

OUT = CACHE / "label_descriptions.json"
key = re.search(r"OPENROUTERKEY=(\S+)", (ROOT.parent / ".env").read_text()).group(1)
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
MODEL = "openai/gpt-5.4-nano"

def describe(domain, names):
    prompt = (f"These are output labels of a {domain} intent classifier. For each label write one short description "
              f"(6-14 words) of what user messages get this label. Reply with a JSON object mapping each label to its description, nothing else.\n"
              + "\n".join(names))
    r = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0,
                                       max_tokens=6000, extra_body={"reasoning": {"effort": "minimal"}})
    txt = r.choices[0].message.content.strip()
    txt = txt[txt.find("{"): txt.rfind("}") + 1]
    d = json.loads(txt)
    missing = [n for n in names if n not in d]
    if missing:  # model occasionally drops one; ask again just for those
        head = prompt.split("\n")[0]
        r2 = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": head + "\n" + "\n".join(missing)}],
                                            temperature=0, max_tokens=2000, extra_body={"reasoning": {"effort": "minimal"}})
        t2 = r2.choices[0].message.content; d.update(json.loads(t2[t2.find("{"): t2.rfind("}") + 1]))
        for n in missing: d.setdefault(n, n.replace("_", " "))
    u = r.usage
    print(domain, len(d), "labels; tokens", u.prompt_tokens, u.completion_tokens, flush=True)
    return {n: d[n] for n in names}, u.prompt_tokens * 0.2e-6 + u.completion_tokens * 1.25e-6

if __name__ == "__main__":
    out, cost = {}, 0.0
    b = load_banking77(); c = load_clinc150()
    out["banking77"], c1 = describe("bank customer support", b["names"])
    clinc_names = [n for n in c["names"] if n != "oos"]
    out["clinc150"], c2 = describe("general-purpose voice assistant", clinc_names)
    cost = c1 + c2
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    with open(ROOT / "COSTS.md", "a", encoding="utf-8") as f:
        f.write(f"s2_descriptions,{cost:.4f},label descriptions via {MODEL}\n")
    print("wrote", OUT, "cost $%.4f" % cost)
