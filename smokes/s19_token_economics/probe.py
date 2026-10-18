"""S19 helper: tiny paid probes (<$0.01 total) that replace assumptions with measurements.
 A  tokens   : Claude (haiku-5.5) token counts of the S16 draft_note prompts and of a 12-label categorize prefix (max_tokens=1)
 B  cache    : does OpenRouter pass cache_control through to Claude, what do write/read cost, is the 512-token minimum real
 C  openai   : gpt-5.4-nano automatic caching below / above 1,024 tokens
Usage: python probe.py A5 | A | B | C   (A5 = pilot of 5 prompts). Every raw response is appended to responses.jsonl (cache + ledger).
ponytail: one function, key = sha1 of the request, so re-runs cost nothing."""
import re, sys, json, time, hashlib
from pathlib import Path
import openai
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
KEY = re.search(r"OPENROUTERKEY\s*=\s*(\S+)", (ROOT / ".env").read_text()).group(1).strip("\"'")
client = openai.OpenAI(api_key=KEY, base_url="https://openrouter.ai/api/v1", max_retries=0, timeout=120)
RESP = HERE / "responses.jsonl"; CAP = 0.05
R = {}
if RESP.exists():
    for l in RESP.open(encoding="utf-8"):
        r = json.loads(l); R[r["key"]] = r
SPENT = [sum((r["resp"].get("usage") or {}).get("cost") or 0 for r in R.values())]
HAIKU = "anthropic/claude-haiku-5.5"
PIN_AWS = {"provider": {"order": ["Claude Platform on AWS"], "allow_fallbacks": False}}

def chat(tag, model, messages, max_tokens=1, extra=None, rep=0):
    extra = dict(extra or {}); extra["usage"] = {"include": True}
    key = hashlib.sha1(json.dumps([model, messages, max_tokens, extra, rep], sort_keys=True).encode()).hexdigest()
    if key not in R:
        if SPENT[0] > CAP: raise RuntimeError(f"spend cap hit {SPENT[0]:.4f}")
        t = time.time()
        d = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, extra_body=extra).model_dump()
        d["_ms"] = (time.time() - t) * 1000
        R[key] = dict(key=key, tag=tag, model=model, rep=rep, resp=d); SPENT[0] += (d.get("usage") or {}).get("cost") or 0
        with RESP.open("a", encoding="utf-8") as f: f.write(json.dumps(R[key]) + "\n")
    d = R[key]["resp"]; u = d.get("usage") or {}; pd = u.get("prompt_tokens_details") or {}
    return dict(tag=tag, pt=u.get("prompt_tokens"), ct=u.get("completion_tokens"), cached=pd.get("cached_tokens"), write=pd.get("cache_write_tokens"),
                cost=u.get("cost"), provider=d.get("provider"), ms=d.get("_ms"))

# ---- prompts -----------------------------------------------------------------------------------------------
SYS_NOTE = "Write a 1-3 sentence bookkeeping note for the reviewer covering the transaction, the invoice match, and why it was or was not flagged. Keep it under 40 words."
CATS = [("Office supplies", "paper, toner, desk items"), ("Software", "subscriptions, licenses, app stores"), ("Meals", "restaurants, coffee, food delivery"),
        ("Travel", "flights, hotels, ride share, fuel"), ("Utilities", "electricity, water, internet, phone"), ("Rent", "office or storage lease payments"),
        ("Payroll", "salaries, contractor payouts, benefits"), ("Insurance", "business, liability and health premiums"), ("Marketing", "ads, sponsorships, printing, events"),
        ("Professional services", "legal, accounting, consulting fees"), ("Equipment", "computers, tools, machinery, repairs"), ("Other", "anything that fits none of the above")]
SYS_CAT12 = "Pick exactly one label.\n" + "\n".join(f"  {a:<22}— {b}" for a, b in CATS)
USER_CAT = "Transaction: ADOBE *CREATIVE CLD, $54.99\nLabel:"
def render(p, hist=True):  # same layout as s16_pruner_note/core.py render(keep_all)
    L = [p[f] for f in ("tx", "inv", "flag", "bank")]
    if hist and p["hist"]: L += ["Recent ledger history (newest first):"] + [h["text"] for h in p["hist"]]
    return "\n".join(L)
def S13_SYSTEM():
    from common import load_banking77
    return "You are a bank support intent classifier. Reply with exactly one label from the list, nothing else.\n" + "\n".join(load_banking77()["names"])

def part_A(n):
    P = json.load(open(HERE.parent / "s16_pruner_note" / "prompts.json", encoding="utf-8"))
    scn = {}
    for p in P: scn.setdefault(p["scn"], p)  # one per scenario first, then fill by id
    pick = (list(scn.values()) + [p for p in P if p not in scn.values()])[:n]
    out = []
    for p in pick:
        for hist in (True, False):
            r = chat("A:note", HAIKU, [{"role": "system", "content": SYS_NOTE}, {"role": "user", "content": render(p, hist)}], 1, {"reasoning": {"enabled": False}, **PIN_AWS})
            r.update(id=p["id"], hist=hist, chars=len(render(p, hist)), n_hist=len(p["hist"]) if hist else 0, scn=p["scn"]); out.append(r); print(r)
    r = chat("A:cat12", HAIKU, [{"role": "system", "content": SYS_CAT12}, {"role": "user", "content": USER_CAT}], 1, {"reasoning": {"enabled": False}, **PIN_AWS}); r.update(chars=len(SYS_CAT12) + len(USER_CAT)); out.append(r); print(r)
    r = chat("A:sys_note_only", HAIKU, [{"role": "system", "content": SYS_NOTE}, {"role": "user", "content": "x"}], 1, {"reasoning": {"enabled": False}, **PIN_AWS}); out.append(r); print(r)
    r = chat("A:s13", HAIKU, [{"role": "system", "content": S13_SYSTEM()}, {"role": "user", "content": "I want to top up my card"}], 1, {"reasoning": {"enabled": False}, **PIN_AWS}); out.append(r); print(r)
    json.dump(out, open(HERE / 'probe_A.json', 'w'), indent=1)

def part_B():
    sysm = S13_SYSTEM(); qs = ["I want to top up my card", "Why was my transfer declined?", "How do I change my PIN?"]
    for label, text in (("77labels", sysm), ("12labels", SYS_CAT12)):
        for k, q in enumerate(qs):
            msgs = [{"role": "system", "content": [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]}, {"role": "user", "content": q}]
            print(label, k, chat(f"B:{label}", HAIKU, msgs, 1, {"reasoning": {"enabled": False}, **PIN_AWS}, rep=0)); time.sleep(2)
    msgs = [{"role": "system", "content": sysm}, {"role": "user", "content": qs[0]}]  # control: no cache_control, same text
    print("nocache", chat("B:nocache", HAIKU, msgs, 1, {"reasoning": {"enabled": False}, **PIN_AWS}))

def part_C():
    sysm = S13_SYSTEM(); big = sysm + "\n\n" + sysm + "\n\n" + sysm
    for label, text in (("439tok", sysm), ("1300tok", big)):
        for k, q in enumerate(["I want to top up my card", "Why was my transfer declined?"]):
            print(label, k, chat(f"C:{label}", "openai/gpt-5.4-nano", [{"role": "system", "content": text}, {"role": "user", "content": q}], 1, {"reasoning": {"effort": "none"}, "provider": {"order": ["OpenAI"], "allow_fallbacks": False}}))
            time.sleep(3)

def part_C2():  # 5 identical calls, same user message, sticky session id: does OpenAI automatic caching show up at >1,024 tokens?
    sysm = S13_SYSTEM(); big = "\n\n".join([sysm, sysm, sysm])
    for k in range(5):
        print("1300tok-identical", k, chat("C2:1300tok", "openai/gpt-5.4-nano", [{"role": "system", "content": big}, {"role": "user", "content": "I want to top up my card"}], 1,
              {"reasoning": {"effort": "none"}, "session_id": "s19-cache-test", "provider": {"order": ["OpenAI"], "allow_fallbacks": False}}, rep=k)); time.sleep(4)

if __name__ == "__main__":
    a = sys.argv[1]
    if a == "A5": part_A(5)
    elif a == "A": part_A(10)
    elif a == "B": part_B()
    elif a == "C": part_C()
    elif a == "C2": part_C2()
    print(f"[spent so far ${SPENT[0]:.5f}]")
