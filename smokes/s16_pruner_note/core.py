"""S16 core: prompt rendering, teacher notes (haiku), judges (sonnet primary, gpt-5.4-mini second), confirmed-difference verdict, ablation labels.
ponytail: shared by pilot.py / run.py so the pilot and the full run use the exact same code paths (all calls are cached)."""
import sys, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import llm
from llm import chat
HAIKU, SONNET, MINI = "anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5", "openai/gpt-5.4-mini"
pin = lambda name: {"provider": {"order": [name], "allow_fallbacks": False}}
EX_T = {"reasoning": {"enabled": False}, **pin("Claude Platform on AWS")}
EX_S = {"reasoning": {"effort": "minimal"}, **pin("Claude Platform on AWS")}   # sonnet: reasoning cannot be disabled; minimal is the lowest accepted
EX_M = {"reasoning": {"effort": "none"}, **pin("OpenAI")}
SYS_T = "Write a 1-3 sentence bookkeeping note for the reviewer covering the transaction, the invoice match, and why it was or was not flagged. State only facts from the record; do not speculate. Keep it under 40 words."
JSYS = ("You compare two bookkeeping notes written for the same transaction. Say SAME if a reviewer would come away from either note with the same facts "
        "and the same conclusion: the same transaction details (vendor, amount, date), the same invoice-match outcome and figures, the same flag decision "
        "with the same reason (including the specific evidence given for that reason, such as the earlier charge a duplicate matches, or how the amount compares "
        "with the vendor's usual amounts when that is the reason), and the same verdict on what needs follow-up. Wording, order, length, and extra background "
        "that does not change those points do not matter. Say DIFFERENT if the notes conflict on any of those points, or if one note gives evidence or raises "
        "a concern that changes the reviewer's conclusion and the other leaves it out.")
JSYS_CORE = ("You compare two bookkeeping notes written for the same transaction. Say SAME if both notes state the same core facts: the transaction (vendor, amount, date), "
             "the invoice-match outcome (which invoice and its amount, or no match), any bank-feed amount they mention, and the flag decision with the same stated reason. "
             "Ignore wording, length, commentary, background about the vendor's history, and follow-up suggestions unless they contradict the other note. "
             "Say DIFFERENT if the notes conflict on any of those core facts.")
JTAIL = "\n\nAnswer with the single word SAME or DIFFERENT and nothing else."
NB = 3   # history ablation blocks: thirds, newest first
P = json.load(open(HERE / "prompts.json"))
FIELDS = ("tx", "inv", "flag", "bank")
def keep_all(p): return dict(tx=1, inv=1, flag=1, bank=1, hist=tuple(range(len(p["hist"]))))
def minus(p, field=None, hist=None):  # drop one fixed field, or a set of history indices ('all' = whole history)
    k = keep_all(p)
    if field: k[field] = 0
    if hist == "all": k["hist"] = ()
    elif hist is not None: k["hist"] = tuple(j for j in k["hist"] if j not in hist)
    return k
def render(p, k):
    L = [p[f] for f in FIELDS if k[f]]
    if k["hist"]: L += ["Recent ledger history (newest first):"] + [p["hist"][j]["text"] for j in k["hist"]]
    return "\n".join(L)
def note(p, k, rep=0):
    r = chat(HAIKU, SYS_T, render(p, k), 200, EX_T, rep)
    return dict(text=r["text"].strip(), tok=r["usage"].get("prompt_tokens"), trunc=r["finish"] == "length", prov=r["provider"], ptxt=render(p, k))
def J(a, b, rep=0, model=SONNET, swap=False, system=None):
    """SAME / DIFFERENT / FAIL (unparseable twice -> FAIL, counted as DIFFERENT by callers). identical strings are SAME without a call."""
    if a == b: return "SAME"
    if swap: a, b = b, a
    u = f"Note 1:\n{a}\n\nNote 2:\n{b}" + JTAIL
    for t in range(3):
        r = chat(model, system or JSYS, u, 40 if model == SONNET else 30, EX_S if model == SONNET else EX_M, rep + 10 * t)
        w = r["text"].strip().upper().strip(". \n*")
        if w in ("SAME", "DIFFERENT"): return w
    return "FAIL"
PAIRS = []   # (A0 text, x text, source, sonnet raw verdict) for the judge-validation sample
def verdict(refs, x, confirm=True, src=""):
    """x vs refs=[A0,A1,A2]. raw = J(A0,x). confirmed (cons) = DIFFERENT only if A0 says DIFFERENT and a second full-prompt note agrees
    (j1 DIFFERENT, or j1 SAME and the third note j2 says DIFFERENT): majority of 3, stopping at the first SAME from A0."""
    j = [J(refs[0], x)]; PAIRS.append((refs[0], x, src, j[0]))
    if j[0] == "SAME" or not confirm: return dict(raw=j[0], cons=j[0], n=1, j=j)
    j.append(J(refs[1], x))
    if j[1] != "SAME": return dict(raw=j[0], cons="DIFFERENT", n=2, j=j)
    j.append(J(refs[2], x))
    return dict(raw=j[0], cons=j[2] if j[2] == "SAME" else "DIFFERENT", n=3, j=j)
isdiff = lambda w: w != "SAME"   # FAIL counts as DIFFERENT

def refs_of(i):
    p = P[i]; return [note(p, keep_all(p), r) for r in range(3)]
def label_prompt(i):
    """Ablation labels for prompt i: 4 fixed fields, whole history (H0), and 5-record history blocks only when H0 changed the note."""
    p = P[i]; R = refs_of(i); rt = [r["text"] for r in R]; out = dict(i=i, refs=rt, tok_full=R[0]["tok"], var={})
    def run(name, k, confirm=True):
        n = note(p, k); v = verdict(rt, n["text"], confirm, src="abl:" + name); out["var"][name] = dict(note=n["text"], tok=n["tok"], trunc=n["trunc"], **v)
        return v
    for f in FIELDS: run(f, minus(p, field=f), confirm=f == "bank")   # tx / inv / flag are almost always needed: one judgment each, no confirmation (bank is the uncertain field)
    h0 = run("H0", minus(p, hist="all"))
    out["nb"] = NB
    if isdiff(h0["cons"]):
        for b in range(NB): run(f"B{b}", minus(p, hist=set(blk(p, b))))
    return out
def blk(p, b):  # history indices of block b (thirds of the record list, newest first)
    n = len(p["hist"]); return range(b * n // NB, (b + 1) * n // NB)
def block_of(p, j): return next(b for b in range(NB) if j in blk(p, b))
def span_labels(p, lab):
    """Per-span 0/1 from ablation labels. Fixed fields: confirmed change. History record: 1 iff its block changed the note, or H0 changed it
    and no block did (evidence is spread over the history: every record may be needed)."""
    v = lab["var"]; fx = {f: int(isdiff(v[f]["cons"])) for f in FIELDS}; n = len(p["hist"])
    hz = int(isdiff(v["H0"]["cons"])); bl = [int(isdiff(v[f"B{b}"]["cons"])) for b in range(lab["nb"])] if hz else []
    dist = int(hz and not any(bl))
    hist = [int(hz and (dist or bl[block_of(p, j)])) for j in range(n)]
    lit = [int(hz and not dist and bl[block_of(p, j)]) for j in range(n)]   # literal "this span's ablation changed the note" (blocks)
    return dict(fixed=fx, hist=hist, hist_lit=lit, h0=hz, dist=dist)
def pmap(fn, xs, w=12):
    with ThreadPoolExecutor(w) as ex: return list(ex.map(fn, xs))
