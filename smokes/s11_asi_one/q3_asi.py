"""Q3: call ASI:One planner mode with our agent addresses. usage: q3_asi.py <model> <agent|hosted|both|none> "<prompt>" """
import json, sys, time, uuid
sys.path.insert(0, ".")
import h

model, which, prompt = sys.argv[1:4]
hosted = json.load(open(h.HERE / "hosted.json"))["address"]
local = open(h.HERE / "local_addr.txt").read().strip()
body = {"model": model, "stream": False, "messages": [{"role": "user", "content": prompt}], "planner_mode": True}
ag = {"local": [local], "hosted": [hosted], "both": [local, hosted]}.get(which)
if ag:
    body["agents"] = ag
h.log(f"PROMPT {time.strftime('%H:%M:%SZ', time.gmtime())} model={model} agents={which} planner_mode=True: {prompt!r}")
t = time.time()
s, r = h.asi(body, headers={"x-session-id": str(uuid.uuid4())})
h.log(f"elapsed {time.time()-t:.0f}s usage={r.get('usage') if isinstance(r, dict) else None}")
