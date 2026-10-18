import sys, time, torch; sys.path.insert(0, "smokes")
from common import embed, load_banking77
t = time.time(); b = load_banking77(); E = embed(b["train"][0][:2000]); print("embedded", E.shape, "in", round(time.time()-t, 1), "s; cuda", torch.cuda.is_available())
open("smokes/_podtest/RESULTS.md", "w").write(f"pod test ok: {E.shape} cuda={torch.cuda.is_available()}\n")
