"""Second-judge check of the primary judge (gemini-3.5-flash-lite): 80 strict-different pairs per teacher, balanced on the primary verdict,
re-judged by anthropic/claude-haiku-4.5 with the same prompt. Reads pairs_<tag>.json written by run.py. ponytail: one-off audit, prints agreement + Cohen's kappa."""
import json, random, sys
from pathlib import Path
import openai
from concurrent.futures import ThreadPoolExecutor
HERE = Path(__file__).parent
key = next(l.split("=", 1)[1].strip().strip("'\"") for l in open(HERE.parents[1] / ".env") if l.startswith("OPENROUTERKEY"))
c = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, timeout=90)
M, PIN, POUT = "anthropic/claude-haiku-4.5", 1.0, 5.0
JPROMPT = ("You compare two answers to one question.\nQuestion: {q}\nAnswer A: {a}\nAnswer B: {b}\n\nSame answer? Extra words, punctuation and formatting do not matter. "
           "Different entities, values or claims, or 'unknown/none' versus a real answer, mean NO.\nOutput only the single word YES or NO, with no explanation.")
def ask(p):
    for a in range(5):
        try:
            r = c.chat.completions.create(model=M, messages=[{"role": "user", "content": JPROMPT.format(**{k: p[k] for k in "qab"})}], temperature=0, max_tokens=64)
            return r.choices[0].message.content.strip(), r.usage.prompt_tokens, r.usage.completion_tokens
        except Exception:
            if a == 4: raise
            import time; time.sleep(2 ** a)
out, usd, rows = {}, 0.0, []
for tag in ("nano_t1", "qwen_t0", "gemini_t0"):
    P = json.load(open(HERE / f"pairs_{tag}.json")); rnd = random.Random(0)
    yes = [p for p in P if p["judge"].upper().startswith("YES")]; no = [p for p in P if not p["judge"].upper().startswith("YES")]
    S = rnd.sample(yes, min(40, len(yes))) + rnd.sample(no, min(40, len(no)))
    with ThreadPoolExecutor(8) as ex: R = list(ex.map(ask, S))
    usd += sum(i * PIN + o * POUT for _, i, o in R) / 1e6
    a = [p["judge"].upper().startswith("YES") for p in S]; b = [r[0].upper().startswith("YES") for r in R]
    n, po = len(S), sum(x == y for x, y in zip(a, b)) / len(S)
    pe = (sum(a) / n) * (sum(b) / n) + (1 - sum(a) / n) * (1 - sum(b) / n)
    out[tag] = dict(n=n, agree=po, kappa=(po - pe) / (1 - pe), primary_yes=sum(a), second_yes=sum(b), unparsed_second=sum(not r[0].upper().startswith(("YES", "NO")) for r in R))
    rows += [(tag, p, r[0]) for p, r in zip(S, R) if p["judge"].upper().startswith("YES") != r[0].upper().startswith("YES")]
    print(tag, out[tag], flush=True)
print(f"spend ${usd:.4f}")
json.dump(dict(per_teacher=out, usd=usd, disagreements=[dict(tag=t, q=p["q"], a=p["a"], b=p["b"], primary=p["judge"], second=s) for t, p, s in rows]), open(HERE / "judge_audit.json", "w"), indent=1)
for t, p, s in rows: print(ascii(f"[{t}] primary={p['judge'][:6]} second={s[:6]} | A={p['a'][:70]} | B={p['b'][:70]}"))
