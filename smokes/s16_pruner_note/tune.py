"""Pilot on 20 separate tuning prompts (seed 99, NOT in the 300): note length / truncation, judge format, floor, label rates, cost per prompt."""
import sys, json; sys.path.insert(0, r"C:\Users\<user>\Desktop\personalprojects\Dreaming\smokes\s16_pruner_note")
import core, llm
from core import *
core.P = json.load(open(HERE / "prompts_tune.json"))
n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
ids = list(range(n)); c0 = llm.SPENT[0]; k0 = llm.NEW[0]
labs = pmap(label_prompt, ids)
from collections import Counter
tr = Counter(); fl = Counter(); fx = Counter(); jc = Counter(); toks = []
for lab in labs:
    p = core.P[lab["i"]]; sl = span_labels(p, lab)
    for name, v in lab["var"].items(): tr[v["trunc"]] += 1; jc[v["n"]] += 1; fx[(name[0] if name[0] == "B" else name, v["cons"][:4], v["raw"][:4])] += 1
    toks.append(lab["tok_full"])
    print(p["scn"], "| tok", lab["tok_full"], "| fixed", sl["fixed"], "H0", sl["h0"], "dist", sl["dist"], "hist+", sum(sl["hist"]), "/", len(sl["hist"]), "| A0:", lab["refs"][0][:200])
print("trunc", tr, "| judge calls per variant", jc, "| (variant, cons, raw):", sorted(fx.items()))
print("spent", round(llm.SPENT[0] - c0, 4), "new calls", llm.NEW[0] - k0, "per prompt $", round((llm.SPENT[0] - c0) / n, 4))
