import sys, json; sys.path.insert(0, r"C:\Users\<user>\Desktop\personalprojects\Dreaming\smokes\s16_pruner_note")
import core, llm
from core import *
core.P = json.load(open(HERE / "prompts_tune.json"))
from collections import Counter
c0 = llm.SPENT[0]
res = Counter(); pairs = []
for i in range(10):
    R = refs_of(i); t = [r["text"] for r in R]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        pairs.append((t[a], t[b]))
out = pmap(lambda ab: (J(*ab), J(*ab, system=JSYS_CORE)), pairs)
v2 = sum(o[0] != "SAME" for o in out) / len(out); v3 = sum(o[1] != "SAME" for o in out) / len(out)
print("n pairs", len(out), "floor DIFF rate rubric v2 (strict)", round(v2, 3), "| core-facts rubric", round(v3, 3), "| agreement of rubrics", round(sum((o[0] == o[1]) for o in out) / len(out), 3))
print("spent", round(llm.SPENT[0] - c0, 4))
