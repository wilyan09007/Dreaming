import sys; sys.path.insert(0, r"C:\Users\<user>\Desktop\personalprojects\Dreaming\smokes\s16_pruner_note")
from core import *
import llm
pick = []
for s in ("normal", "noinv", "spike", "dup", "first"): pick.append(next(p["id"] for p in P if p["scn"] == s and p["id"] not in pick))
c0 = llm.SPENT[0]
labs = pmap(label_prompt, pick)
for lab in labs:
    p = P[lab["i"]]; sl = span_labels(p, lab)
    print("=" * 100); print(p["scn"], "| tok_full", lab["tok_full"], "| n_hist", len(p["hist"])); print(render(p, keep_all(p))[:700])
    for k, t in enumerate(lab["refs"]): print(f"A{k}:", t)
    for name, v in lab["var"].items(): print(f"  {name:4s} raw={v['raw'][:4]} cons={v['cons'][:4]} n={v['n']} tok={v['tok']} trunc={v['trunc']} | {v['note'][:230]}")
    print("  labels:", sl["fixed"], "H0", sl["h0"], "dist", sl["dist"], "hist pos", sum(sl["hist"]), "/", len(sl["hist"]))
print("spent", llm.SPENT[0] - c0, "new calls", llm.NEW[0])
