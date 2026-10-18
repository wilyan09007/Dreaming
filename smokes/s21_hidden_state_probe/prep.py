"""S21 prep: build data.json (inputs for the extractor, no keys) and targets.json (labels for the analysis).
Inputs: Banking77 200 (S5, same order/prompt), 1000 train (proxy), 1000 new test (stage 2), RouterBench 2000 (subset of S5's 5k embedding subsample)."""
import sys, json, zlib, base64, numpy as np
sys.dont_write_bytecode = True
from pathlib import Path as _P
_SM = _P(__file__).resolve().parents[1]; sys.path.insert(0, str(_SM)); sys.path.insert(0, str(_SM / "s5_tier_router"))
from common import *  # noqa
import run as s5
from pathlib import Path
HERE = Path(__file__).parent
sys.stdout.reconfigure(encoding="utf-8")
SEED = 21
names, t200, y200 = s5.b_data()
gold200 = [names[k] for k in y200]
# E1 / E1b answers from S5's cached responses (no re-run)
rows = [json.loads(l) for l in s5.RESP.read_text(encoding="utf-8").splitlines() if l.strip()]
last = {}
for r in rows:
    if r["err"] is None: last.setdefault((r["cfg"], r["i"]), r)
e1 = [s5.parse(last[("E1", i)]["content"], names)[0] for i in range(200)]
e1b = [s5.parse(last[("E1b", i)]["content"], names)[0] for i in range(200)]
print("E1 gold acc", np.mean([a == b for a, b in zip(e1, gold200)]), "E1 vs E1b match", np.mean([a == b for a, b in zip(e1, e1b)]))
d = load_banking77()
trX, trY = d["train"]; teX, teY = d["test"]
i200 = set(int(i) for i in stratified_subsample(teY, 200, seed=0)); assert [teX[i] for i in sorted(i200)] != [] and len(i200) == 200
assert [teX[i] for i in stratified_subsample(teY, 200, seed=0)] == t200
rest = np.array([i for i in range(len(teX)) if i not in i200])
j = stratified_subsample(teY[rest], 1000, seed=SEED); new_idx = rest[j]
tr_idx = stratified_subsample(trY, 1000, seed=SEED)
b1000new = [teX[i] for i in new_idx]; b1000tr = [trX[i] for i in tr_idx]
assert not (set(b1000new) & set(t200)), "overlap texts new vs 200 (duplicate texts)" if False else True
print("dup texts new/200:", len(set(b1000new) & set(t200)), "train/200:", len(set(b1000tr) & set(t200)), "train/new:", len(set(b1000tr) & set(b1000new)))
# RouterBench
df, M, fam = s5.load_a()
from sklearn.model_selection import train_test_split
sub = np.sort(train_test_split(np.arange(len(df)), train_size=s5.SUB, stratify=fam, random_state=0)[0])
rb_idx = np.sort(np.random.default_rng(SEED).choice(sub, 2000, replace=False))
G, E = "gpt-3.5-turbo-1106", "gpt-4-1106-preview"
def trunc(p, n=1200): return p if len(p) <= n else p[: n // 2] + "\n...\n" + p[-n // 2:]
rbp = df.prompt.astype(str).values[rb_idx]
rb = dict(idx=rb_idx.tolist(), fam=fam[rb_idx].tolist(), sc=df[G].astype(float).values[rb_idx].tolist(), se=df[E].astype(float).values[rb_idx].tolist(),
          cc=df[G + "|total_cost"].astype(float).values[rb_idx].tolist(), ce=df[E + "|total_cost"].astype(float).values[rb_idx].tolist(),
          prompt400=[p[:400] for p in rbp], text=[trunc(p) for p in rbp])
print("RB family counts", dict(zip(*np.unique(rb["fam"], return_counts=True))), "mean len", np.mean([len(t) for t in rb["text"]]))
data = dict(names=names, sys_prompt=s5.sys_prompt(names), b200=t200, b1000tr=b1000tr, b1000new=b1000new, rb2000=rb["text"])
tg = dict(names=names, gold200=gold200, e1=e1, e1b=e1b, tr_gold=[names[k] for k in trY[tr_idx]], new_gold=[names[k] for k in teY[new_idx]], new_idx=new_idx.tolist(), tr_idx=tr_idx.tolist(), rb=rb)
(HERE / "data.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
(HERE / "targets.json").write_text(json.dumps(tg, ensure_ascii=False), encoding="utf-8")
raw = json.dumps(data, ensure_ascii=False).encode("utf-8"); z = base64.b64encode(zlib.compress(raw, 9))
print(f"data.json raw {len(raw)/1e3:.0f} KB -> zlib+b64 {len(z)/1e3:.0f} KB")
