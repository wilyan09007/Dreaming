"""Cache-only stand-in for s16_pruner_note/llm.py: same key construction, never touches the network, raises on a miss."""
import json, hashlib, sys, types
from pathlib import Path
S16 = Path(__file__).resolve().parents[1] / "s16_pruner_note"
R = {}
for l in (S16 / "responses.jsonl").open(encoding="utf-8"):
    try: r = json.loads(l); R[r["key"]] = r
    except Exception: pass
MISSING = []; HITS = [0]
class Miss(KeyError): pass
def chat(model, system, user, max_tokens, extra=None, rep=0):
    extra = dict(extra or {}); extra["usage"] = {"include": True}
    key = hashlib.sha1(json.dumps([model, system, user, max_tokens, extra, rep], sort_keys=True).encode()).hexdigest()
    if key not in R:
        MISSING.append((model, rep, user[:60])); raise Miss(key)
    HITS[0] += 1
    d = R[key]["resp"]; u = d.get("usage") or {}; ch = d["choices"][0]
    return dict(text=(ch["message"].get("content") or ""), provider=d.get("provider"), usage=u, finish=ch.get("finish_reason"), reasoning_tokens=0, ms=None, key=key)
def install():
    m = types.ModuleType("llm"); m.chat = chat; m.R = R; sys.modules["llm"] = m
    sys.path.insert(0, str(S16)); sys.path.insert(0, str(S16.parent))
