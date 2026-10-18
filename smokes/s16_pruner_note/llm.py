"""S16 helper: cached, pinned, cost-tracked OpenRouter chat calls. responses.jsonl is cache + spend ledger.
ponytail: one function; key = sha1 of the full request (model, messages, max_tokens, extra_body, rep)."""
import re, json, time, hashlib, threading
from pathlib import Path
import openai
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
KEY = re.search(r"OPENROUTERKEY\s*=\s*(\S+)", (ROOT / ".env").read_text()).group(1).strip("\"'")
client = openai.OpenAI(api_key=KEY, base_url="https://openrouter.ai/api/v1", max_retries=0, timeout=120)
RESP = HERE / "responses.jsonl"; lock = threading.Lock(); CAP = 5.0
R = {}
if RESP.exists():
    for l in RESP.open(encoding="utf-8"):
        try: r = json.loads(l); R[r["key"]] = r
        except Exception: pass   # a line interleaved by two concurrent processes; that response is simply re-fetched
SPENT = [sum((r["resp"].get("usage") or {}).get("cost") or 0 for r in R.values())]; NEW = [0]

def chat(model, system, user, max_tokens, extra=None, rep=0):
    """returns dict(text, provider, usage, reasoning_tokens, finish, cached). extra = extra_body (provider pin, reasoning)."""
    extra = dict(extra or {}); extra["usage"] = {"include": True}
    key = hashlib.sha1(json.dumps([model, system, user, max_tokens, extra, rep], sort_keys=True).encode()).hexdigest()
    if key not in R:
        if SPENT[0] > CAP: raise RuntimeError(f"spend cap hit: {SPENT[0]:.3f}")
        for a in range(6):
            try:
                t = time.time()
                d = client.chat.completions.create(model=model, max_tokens=max_tokens, extra_body=extra,
                    messages=([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]).model_dump()
                d["_ms"] = (time.time() - t) * 1000; break
            except (openai.BadRequestError, openai.NotFoundError): raise
            except Exception:
                if a == 5: raise
                time.sleep(2 ** a)
        rec = dict(key=key, model=model, rep=rep, resp=d)
        with lock:
            R[key] = rec; SPENT[0] += (d.get("usage") or {}).get("cost") or 0; NEW[0] += 1
            with RESP.open("a", encoding="utf-8") as f: f.write(json.dumps(rec) + "\n")
    d = R[key]["resp"]; u = d.get("usage") or {}
    ch = d["choices"][0]
    return dict(text=(ch["message"].get("content") or ""), provider=d.get("provider"), usage=u, finish=ch.get("finish_reason"),
                reasoning_tokens=((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0), ms=d.get("_ms"), key=key)
