"""S19 token economics: where do the dollars sit, which lever saves what, and is speculation worth it. $0 (arithmetic on logged tokens).
Stages: tokens | trace | econ | sens | spec | all (v2: econ runs both thinking scenarios; sens = sensitivity tables; spec = exact TraceLab arrays).  Inputs: probe_A.json (Claude token probe), models.json (OpenRouter /models, 2026-10-17), S5/S13/S16 logs, TraceLab.
ponytail: costs are linear in tokens, so means over the logged prompts give exact totals; no simulation needed except the speculation Monte-Carlo."""
import sys, os, json, gzip, time, statistics as st, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent; SM = HERE.parent
sys.path.insert(0, str(SM))
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(SM / ".data" / "s19_tiktoken"))
import numpy as np

# ------------------------------------------------------------------------------------------------ parameters
CALLS_PER_DAY = 10_000          # demo workflow runs/day; each run = 1 categorize + 1 flag_anomaly + 1 draft_note LLM call (vendor_lookup, match_invoice are tools)
H_SYNC, H_BATCH = 0.95, 0.65    # cache hit rate on cacheable prefix: steady sync traffic / Batch (Anthropic: "30% to 98%", best effort)
AUDIT = 0.05                    # share of student-answered (or pruned) calls re-run on the teacher / full prompt, async -> Batch  (missions.md: one call in twenty)
THINK_DELTA = 157               # extra reasoning tokens when thinking is on; econ runs both scenarios in THINK_SCEN (this is the module default that the stage loop overwrites)
THINK_SCEN = {"low": 32, "high": 157}   # low = S5 E1 gpt-5.4 effort low (reasoning 31.4 / 32.6 tokens over two 200-input runs); high = S5 Ton gpt-5.4-mini effort high (156.5). Measured elsewhere on a typed step: sonnet-5.5 15 (S13, reasoning cannot be disabled), gemini-lite 0 (mandatory), qwen 1
THINK_OFF_OK = {"sonnet-5.5": False, "gemini-3.5-flash-lite": False}   # S13: Sonnet 5.5 reasoning cannot be disabled; Gemini 3.5 Flash-Lite reasoning is mandatory (400 if disabled). Others can be switched off (S5, S13).
PRUNE = (0.6, 0.8)              # mission range: share of history tokens removed
PRUNE_S6B = 0.50                # what S6b supports: learned k=2 with selective gating certifies only at a 15% budget: 61.3% coverage x 82% removal = 50.2% effective; 0 at 5% and 10% budgets
COVER = (0.6, 0.75, 0.9)        # certified student coverage on typed steps (S1/S8: 59/89/94% Banking77 at 770/3080/9993 rows)
OVERHEAD_OAI = 10               # chat-template tokens added by OpenAI-style tokenizers (S13: 439.2 observed vs 417 + 12.2 counted by tiktoken)
TEACHERS = {  # short name -> (OpenRouter id, tokenizer family, min cacheable prefix tokens, S13/S5 p50 latency ms, note)
    "sonnet-5.5": ("anthropic/claude-sonnet-5.5", "claude", 512, 1149),
    "haiku-5.5": ("anthropic/claude-haiku-5.5", "claude", 512, 828),
    "gpt-5.4": ("openai/gpt-5.4", "oai", 1024, 2228),
    "gpt-5.4-mini": ("openai/gpt-5.4-mini", "oai", 1024, 762),
    "gpt-5.4-nano": ("openai/gpt-5.4-nano", "oai", 1024, 831),
    "gemini-3.5-flash-lite": ("google/gemini-3.5-flash-lite", "gem", 2048, 574),   # Google caching page (2026-10-17, updated 2026-10-13): 2,048 for Gemini 2.5, 4,096 for 3.x; 3.5 Flash-Lite not listed -> lowest listed figure (v1 used a guessed 1,024)
    "qwen3-32b": ("qwen/qwen3-32b", "oai", None, 390),
}
FLAG_SYS = ("You review one bookkeeping transaction for anomalies. Using the matched invoice, the bank-feed record and the vendor's recent ledger history, "
            "answer yes if the amount, date, vendor or invoice match looks anomalous (unusual amount, duplicate, missing invoice, bank mismatch), otherwise no. Reply with one word.")
S16 = SM / "s16_pruner_note"
def jl(path):
    out = []
    for l in open(path, encoding="utf-8"):
        try: out.append(json.loads(l))
        except Exception: pass   # another agent may be appending
    return out

# ------------------------------------------------------------------------------------------------ prices
def prices():
    M = {m["id"]: m["pricing"] for m in json.load(open(HERE / "models.json", encoding="utf-8"))["data"]}
    out = {}
    for t, (mid, fam, minc, lat) in TEACHERS.items():
        p, b = M[mid], M.get(mid + ":batch")
        f = lambda d, k, dflt=None: float(d[k]) if d and k in d else dflt
        pi, po = f(p, "prompt"), f(p, "completion")
        read, write = f(p, "input_cache_read"), f(p, "input_cache_write")
        if fam != "claude": write = pi   # OpenAI/Gemini implicit caching: no write fee (OpenAI pre-5.6 per pricing guide; Gemini implicit); Gemini listing's write field is 5-min storage of explicit caches
        out[t] = dict(i=pi, o=po, read=read, write=write, minc=minc, lat=lat,
                      b_i=f(b, "prompt"), b_o=f(b, "completion"), b_read=f(b, "input_cache_read"), b_write=(f(b, "input_cache_write") if fam == "claude" else f(b, "prompt")))
    return out
PR = None
LIVE_BATCH = False   # lever "Batch for every call" (only valid if every step tolerates up to 24 h)

# ------------------------------------------------------------------------------------------------ token profiles
def tokens():
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base"); n = lambda s: len(enc.encode(s))
    P = json.load(open(S16 / "prompts.json", encoding="utf-8"))
    A = json.load(open(HERE / "probe_A.json", encoding="utf-8"))
    sys.path.insert(0, str(HERE)); import probe   # only for the prompt strings; importing makes no API call
    rec_fields = lambda p, fs: "\n".join(p[f] for f in fs)
    hist_text = lambda p: "\n".join(["Recent ledger history (newest first):"] + [h["text"] for h in p["hist"]])
    # o200k means over all prompts
    o = dict(rec_note=np.mean([n(rec_fields(p, ("tx", "inv", "flag", "bank"))) for p in P]), rec_flag=np.mean([n(rec_fields(p, ("tx", "inv", "bank"))) for p in P]),
             hist=np.mean([n(hist_text(p)) for p in P]), tx=np.mean([n(p["tx"]) for p in P]), n_hist=np.mean([len(p["hist"]) for p in P]))
    # Claude/o200k ratios from the paid probe (10 prompts, with and without history)
    by = collections.defaultdict(dict)
    for r in A:
        if r["tag"] == "A:note": by[r["id"]][r["hist"]] = r
    sysnote_cl = next(r["pt"] for r in A if r["tag"] == "A:sys_note_only")   # system prompt + 1-token user + template
    Pm = {p["id"]: p for p in P}
    rec_cl = sum(v[False]["pt"] - sysnote_cl for v in by.values()); rec_o = sum(n(rec_fields(Pm[i], ("tx", "inv", "flag", "bank"))) for i in by)
    his_cl = sum(v[True]["pt"] - v[False]["pt"] for v in by.values()); his_o = sum(n(hist_text(Pm[i])) for i in by)
    k_rec, k_his = rec_cl / rec_o, his_cl / his_o
    cat_cl = next(r["pt"] for r in A if r["tag"] == "A:cat12")
    cat_pref_o, cat_user_o = n(probe.SYS_CAT12), n(probe.USER_CAT)
    cat_pref_cl = cat_cl * cat_pref_o / (cat_pref_o + cat_user_o)
    # completion lengths of the S16 haiku notes that ended normally
    R16 = [r for r in jl(S16 / "responses.jsonl") if r.get("model", "").endswith("haiku-5.5") and r["resp"]["choices"][0]["finish_reason"] == "stop"]
    out_cl = st.mean(r["resp"]["usage"]["completion_tokens"] for r in R16); out_o = st.mean(n(r["resp"]["choices"][0]["message"]["content"]) for r in R16)
    lab = st.mean(n(a) for a, _ in probe.CATS)
    # S13 measured 77-label prefix per teacher: pt = fixed + slope*chars(query); intercept = fixed prefix
    from common import load_banking77, stratified_subsample
    b = load_banking77(); texts, labels = b["test"]; IDX = stratified_subsample(labels, 100, seed=0); QS = [texts[i] for i in IDX]
    s13 = collections.defaultdict(dict)
    for r in jl(SM / "s13_teacher_noise" / "responses.jsonl"):
        if r["cond"] == "A" and r["rep"] == 0: s13[r["model"]][r["i"]] = (r["resp"]["usage"]["prompt_tokens"], r["resp"]["usage"]["completion_tokens"])
    c77 = {}
    for m, d in s13.items():
        ii = sorted(d); pts = np.array([d[i][0] for i in ii]); ch = np.array([len(QS[i]) for i in ii]); cts = np.array([d[i][1] for i in ii])
        coef = np.linalg.lstsq(np.vstack([np.ones_like(ch), ch]).T, pts, rcond=None)[0]
        c77[m] = dict(P=float(coef[0]), V=float(pts.mean() - coef[0]), O=float(cts.mean()))
    T = dict(o200k=o, k_rec=k_rec, k_his=k_his, sysnote_claude=sysnote_cl, cat12_total_claude=cat_cl, cat12_prefix_claude=cat_pref_cl, out_claude=out_cl, out_o200k=out_o,
             n_prompts=len(P), n_notes=len(R16), c77=c77, label_tokens=lab, flag_sys_o200k=n(FLAG_SYS), note_sys_o200k=n(probe.SYS_NOTE),
             probe_pairs=len(by), pairs=[dict(id=i, n_hist=len(Pm[i]["hist"]), pt_full=v[True]["pt"], pt_nohist=v[False]["pt"]) for i, v in by.items()])
    json.dump(T, open(HERE / "tokens.json", "w"), indent=1)
    return T

def profiles(T):
    """tokens per call by class: P prefix (cacheable), V variable non-prunable, H history (prunable), O output."""
    o = T["o200k"]; S = {}
    for t, (mid, fam, minc, lat) in TEACHERS.items():
        S[t] = {}
        if fam == "claude":
            k = T["k_rec"], T["k_his"]
            P_note = T["sysnote_claude"]; P_flag = round(T["sysnote_claude"] * T["flag_sys_o200k"] / T["note_sys_o200k"])
            V_note, V_flag, H = k[0] * o["rec_note"], k[0] * o["rec_flag"], k[1] * o["hist"]
            O_note, O_cat = T["out_claude"], T["label_tokens"] * T["out_claude"] / T["out_o200k"] + 1
            P_cat12, V_cat12 = T["cat12_prefix_claude"], T["cat12_total_claude"] - T["cat12_prefix_claude"]
            c77 = T["c77"]["anthropic/claude-haiku-5.5" if t == "haiku-5.5" else "anthropic/claude-sonnet-5.5"]
        else:
            g = 1.0
            P_note = T["note_sys_o200k"] + OVERHEAD_OAI; P_flag = T["flag_sys_o200k"] + OVERHEAD_OAI
            V_note, V_flag, H = o["rec_note"] * g, o["rec_flag"] * g, o["hist"] * g
            O_note, O_cat = T["out_o200k"], T["label_tokens"] + 1
            import tiktoken; enc = tiktoken.get_encoding("o200k_base")
            sys.path.insert(0, str(HERE)); import probe
            P_cat12 = len(enc.encode(probe.SYS_CAT12)) + OVERHEAD_OAI; V_cat12 = len(enc.encode(probe.USER_CAT))
            key = {"gpt-5.4": "openai/gpt-5.4-mini", "gpt-5.4-nano": "openai/gpt-5.4-mini", "gpt-5.4-mini": "openai/gpt-5.4-mini", "gemini-3.5-flash-lite": "google/gemini-3.5-flash-lite", "qwen3-32b": "qwen/qwen3-32b"}[t]
            c77 = T["c77"][key]
        S[t]["cat12"] = dict(P=P_cat12, V=V_cat12, H=0.0, O=O_cat, thinks=False)
        S[t]["cat77"] = dict(P=c77["P"], V=c77["V"], H=0.0, O=c77["O"], thinks=False)
        S[t]["flag"] = dict(P=P_flag, V=V_flag, H=H, O=2.0, thinks=True)
        S[t]["note"] = dict(P=P_note, V=V_note, H=H, O=O_note, thinks=False)
    return S

# ------------------------------------------------------------------------------------------------ cost model
def call_cost(t, pr, cache=False, prune=0.0, think=True, batch=False):
    """USD for one call. Prefix is cacheable only if cache=True and len(prefix) >= the model's minimum; misses pay the write rate."""
    batch = batch or LIVE_BATCH
    p = PR[t]; i, o, rd, wr = (p["b_i"], p["b_o"], p["b_read"], p["b_write"]) if batch and p["b_i"] else (p["i"], p["o"], p["read"], p["write"])
    h = H_BATCH if batch else H_SYNC
    if cache and p["minc"] and pr["P"] >= p["minc"] and rd: pre = pr["P"] * (h * rd + (1 - h) * wr)
    else: pre = pr["P"] * i
    out = pr["O"] + (THINK_DELTA if (pr["thinks"] and think) else 0)
    return pre + (pr["V"] + pr["H"] * (1 - prune)) * i + out * o

def step_cost(t, step, S, cache=False, prune=0.0, think=True, student=0.0, abatch=True):
    """Expected USD per call of this step. prune: share of history tokens removed (adds a batched full-prompt audit on AUDIT of pruned calls).
    student: certified coverage c; the student answers c of calls for free, AUDIT*c of them are re-asked async (Batch); (1-c) go to the teacher in real time."""
    pr = S[t][step]
    live = call_cost(t, pr, cache, prune, think)
    if not think and pr["thinks"]: live += AUDIT * call_cost(t, pr, cache, prune, True, batch=abatch)   # v2: thinking-off carries the same 1-in-20 audit (re-run with thinking on, Batch) as the student and the pruner
    if prune and pr["H"]: live += AUDIT * call_cost(t, pr, cache, 0.0, think, batch=abatch)   # no history -> nothing to prune, no audit
    if student:
        aud = call_cost(t, pr, cache, prune, think, batch=abatch)
        return (1 - student) * live + student * AUDIT * aud
    return live

def table_per_step(S):
    rows = {}
    for t in TEACHERS:
        for step in ("cat12", "cat77", "flag", "note"):
            pr = S[t][step]; k = 1000
            d = dict(no_cache=call_cost(t, pr, think=True), cache=call_cost(t, pr, cache=True),
                     cache_prune60=step_cost(t, step, S, cache=True, prune=0.6), cache_prune80=step_cost(t, step, S, cache=True, prune=0.8),
                     batch=call_cost(t, pr, batch=True), batch_cache=call_cost(t, pr, cache=True, batch=True),
                     think_on=call_cost(t, pr, think=True), think_off=call_cost(t, pr, think=False),
                     student60=step_cost(t, step, S, cache=True, student=0.6), student75=step_cost(t, step, S, cache=True, student=0.75), student90=step_cost(t, step, S, cache=True, student=0.9),
                     cacheable=bool(PR[t]["minc"] and pr["P"] >= PR[t]["minc"]), P=pr["P"], V=pr["V"], H=pr["H"], O=pr["O"])
            rows[(t, step)] = {k_: (v * k if isinstance(v, float) and k_ in ("no_cache", "cache", "cache_prune60", "cache_prune80", "batch", "batch_cache", "think_on", "think_off", "student60", "student75", "student90") else v) for k_, v in d.items()}
    return rows

def shares(t, S, step):
    """share of the no-cache cost by token class"""
    pr = S[t][step]; p = PR[t]; out = pr["O"] + (THINK_DELTA if pr["thinks"] else 0)
    parts = dict(prefix=pr["P"] * p["i"], record=pr["V"] * p["i"], history=pr["H"] * p["i"], output=out * p["o"]); tot = sum(parts.values())
    return {k: v / tot for k, v in parts.items()}, tot

def pair_cost(t, S, student, prune, think, abatch=True):
    """Cache-friendly LAYOUT (an extra lever, not in the task's definition of prefix caching): [system][record + history][step instruction last], so the
    flag_anomaly call writes the cache and the draft_note call of the same run (seconds later) reads it. Only possible when both reach the teacher:
    if the student answered flag_anomaly there is no writer and the note call pays plain price. Returns (flag, note) USD per run."""
    p = PR[t]; f, n = S[t]["flag"], S[t]["note"]; i, o, rd, wr = p["i"], p["o"], p["read"], p["write"]; h = H_SYNC
    X = f["V"] + f["H"] * (1 - prune)                      # shared context: record without the flag line + (pruned) history
    ok = bool(p["minc"] and rd and X >= p["minc"])
    fo = (f["O"] + (THINK_DELTA if think else 0)) * o
    f_plain = (f["P"] + X) * i + fo; f_write = f["P"] * i + X * (wr if ok else i) + fo
    n_extra = n["P"] * i + (n["V"] - f["V"]) * i + n["O"] * o
    n_plain = n_extra + X * i; n_hit = n_extra + X * ((h * rd + (1 - h) * wr) if ok else i)
    q = 1 - student
    flag = q * f_write + student * AUDIT * call_cost(t, f, False, prune, think, batch=abatch)
    note = q * n_hit + (1 - q) * n_plain
    if not think: flag += q * AUDIT * call_cost(t, f, False, prune, True, batch=abatch)   # thinking-off audit
    if prune:  # pruner audit: AUDIT of pruned live calls re-run on the full prompt, async
        flag += q * AUDIT * call_cost(t, f, False, 0.0, think, batch=abatch); note += AUDIT * call_cost(t, n, False, 0.0, True, batch=abatch)
    return flag, note

def workflow(t, S, cache=False, prune=0.0, think=True, student=0.0, cat="cat12", prune_steps=("flag", "note"), layout=False, abatch=True):
    """USD/day for the demo workflow (cat, flag, note per run). student applies to the two typed steps."""
    tot = {}
    tot["cat"] = step_cost(t, cat, S, cache=cache or layout, student=student, abatch=abatch)
    if layout:
        tot["flag"], tot["note"] = pair_cost(t, S, student, prune, think, abatch)
    else:
        tot["flag"] = step_cost(t, "flag", S, cache=cache, prune=prune if "flag" in prune_steps else 0, think=think, student=student, abatch=abatch)
        tot["note"] = step_cost(t, "note", S, cache=cache, prune=prune if "note" in prune_steps else 0, abatch=abatch)
    return {k: v * CALLS_PER_DAY for k, v in tot.items()}

def levers(t, S):
    """standalone saving of each lever vs ITS OWN baseline: rows are (name, $/day saved, own baseline $/day). Not additive."""
    global LIVE_BATCH
    W = lambda **kw: sum(workflow(t, S, **kw).values()); b = W(); b77 = W(cat="cat77")
    rows = [("certified student, 90% coverage on the two typed steps", b - W(student=0.9), b), ("certified student, 75%", b - W(student=0.75), b),
            ("certified student, 60%", b - W(student=0.6), b), ("certified student, 50%", b - W(student=0.5), b),
            ("pruning 80% of history (mission range)", b - W(prune=0.8), b), ("pruning 60% of history (mission range)", b - W(prune=0.6), b),
            ("pruning 50% (what S6b supports at a 15% budget)", b - W(prune=PRUNE_S6B), b), ("pruning at a 5% or 10% budget (S6b: nothing certifies)", 0.0, b),
            ("thinking off on flag_anomaly (incl. 1-in-20 audit)", b - W(think=False), b), ("cache-friendly layout (flag writes, note reads)", b - W(layout=True), b),
            ("prefix caching as specified (demo prompts)", b - W(cache=True), b), ("prefix caching, if categorize carried the 77-label list", b77 - W(cat="cat77", cache=True), b77),
            ("Batch for the audit calls only (inside student 75%)", W(student=0.75, abatch=False) - W(student=0.75), b)]
    LIVE_BATCH = True; vb = W(); LIVE_BATCH = False   # ponytail: toggles a module flag instead of threading a parameter through five functions
    rows.append(("Batch for ALL calls (only if every step tolerates up to 24 h)", b - vb, b))
    return b, sorted(rows, key=lambda r: -r[1])

COMBOS = [  # name, kwargs
    ("0 baseline: no cache, flag thinking on, all real time", dict()),
    ("1 + prefix caching", dict(cache=True)),
    ("2 + pruning 70% (no cache)", dict(prune=0.7)),
    ("2f pruning 70% on flag_anomaly only (typed step, S6b)", dict(prune=0.7, prune_steps=("flag",))),
    ("2n pruning 70% on draft_note only (free text, S16 pending)", dict(prune=0.7, prune_steps=("note",))),
    ("3 caching + pruning 70%  [= build the pruner first]", dict(cache=True, prune=0.7)),
    ("1c + cache-friendly layout (flag writes, note reads the shared record+history)", dict(layout=True)),
    ("3c layout + pruning 70%", dict(layout=True, prune=0.7)),
    ("4 thinking off on flag_anomaly (mission 1)", dict(think=False)),
    ("5 certified student 75% on typed steps (mission 2)", dict(student=0.75)),
    ("5a student 59% only (S1 at 770 rows, 5% budget: the one lever with a 5% certificate)", dict(student=0.59)),
    ("5b student 75% + pruning 50% (no thinking change)", dict(student=0.75, prune=PRUNE_S6B)),
    ("5c student 75% + pruning 70% (no thinking change)", dict(student=0.75, prune=0.7)),
    ("6 missions 1+2 (thinking off + student 75%)", dict(think=False, student=0.75)),
    ("7 missions 1+2 + caching", dict(think=False, student=0.75, cache=True)),
    ("7c missions 1+2 + layout", dict(think=False, student=0.75, layout=True)),
    ("8 missions 1+2+5 + caching (+ pruning 70% on what still goes to the teacher)", dict(think=False, student=0.75, cache=True, prune=0.7)),
    ("2s pruning 50% (S6b-anchored: certifies only at a 15% budget)", dict(prune=PRUNE_S6B)),
    ("8s missions 1+2+5 with the S6b-anchored 50% pruner + caching", dict(think=False, student=0.75, cache=True, prune=PRUNE_S6B)),
    ("9 mission 5 on top of 1+2 without caching", dict(think=False, student=0.75, prune=0.7)),
]
def sub_prune_flag_only(t, S):  # pruning only where S6b proved it (typed step), S16 (free-text note) still open
    return sum(workflow(t, S, think=False, student=0.75, cache=True, prune=0.7, prune_steps=("flag",)).values())

def prefix_sweep(t, S, sizes=(100, 250, 500, 1000, 2000, 4000, 8000), minc=None):
    """Same demo workflow with a fixed prefix of X tokens (instructions + labels + few-shot) on every step: $/day each lever saves. Below the model's cache minimum caching saves 0.
    minc overrides the provider minimum (to test Gemini at 4,096 or OpenAI at 1,024)."""
    old = PR[t]["minc"]
    if minc: PR[t]["minc"] = minc
    try:
        rows = []
        for X in sizes:
            S2 = {t: {s_: dict(S[t][s_], P=float(X)) for s_ in S[t]}}
            W = lambda **kw: sum(workflow(t, S2, **kw).values()); b = W()
            rows.append(dict(P=X, base=b, cache=b - W(cache=True), prune70=b - W(prune=0.7), prune50=b - W(prune=PRUNE_S6B), student75=b - W(student=0.75), cache_prune70=b - W(cache=True, prune=0.7)))
        def cross(key):   # smallest prefix (50-token grid) where caching saves at least as much as `key`
            for X in range(50, 40001, 50):
                S2 = {t: {s_: dict(S[t][s_], P=float(X)) for s_ in S[t]}}
                W = lambda **kw: sum(workflow(t, S2, **kw).values())
                if W() - W(cache=True) >= W() - W(**key): return X
            return None
        return rows, dict(minc=PR[t]["minc"], cache_beats_prune70=cross(dict(prune=0.7)), cache_beats_prune50=cross(dict(prune=PRUNE_S6B)), cache_beats_student75=cross(dict(student=0.75)))
    finally: PR[t]["minc"] = old

# ------------------------------------------------------------------------------------------------ TraceLab + speculation
def trace_stats():
    import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)   # keep the laptop responsive; single streaming pass over the 101 MB gz
    t0 = time.time()
    f = SM / ".data" / "tracelab" / "syfi_coding_trace_v0.0.2.jsonl.gz"
    lat, names, tok = [], [], collections.Counter(); rounds = 0
    for l in gzip.open(f, "rt", encoding="utf-8"):
        r = json.loads(l); rounds += 1
        if r.get("provider") == "claude":
            tok["read"] += r.get("claude_cache_read_input_tokens") or 0; tok["create"] += r.get("claude_cache_creation_input_tokens") or 0; tok["uncached"] += r.get("claude_uncached_input_tokens") or 0
            tok["output"] += r.get("output_tokens") or 0
        for t in r.get("tools") or []:
            v = t.get("tool_wall_latency_ms")
            if v is not None: lat.append(v / 1000.0); names.append(t.get("tool_name", "?") if r.get("provider") == "claude" else "codex:" + str(t.get("tool_name", "?")))
    lat = np.array(lat); np.save(HERE / "tracelab_latency_s.npy", lat.astype(np.float32))
    isc = np.array([not x.startswith("codex:") for x in names]); lc = lat[isc]; np.save(HERE / "tracelab_latency_claude_s.npy", lc.astype(np.float32))
    qs = {q: float(np.quantile(lat, q)) for q in (0.5, 0.75, 0.9, 0.95, 0.99)}
    by = collections.defaultdict(list)
    for a, b in zip(names, lat): by[a].append(b)
    top = sorted(by.items(), key=lambda kv: -len(kv[1]))[:8]
    S = dict(rounds=rounds, n_tools=int(len(lat)), mean_s=float(lat.mean()), q=qs, share_ge1s=float((lat >= 1).mean()), time_share_ge1s=float(lat[lat >= 1].sum() / lat.sum()),
             share_gt60s=float((lat > 60).mean()), time_share_gt60s=float(lat[lat > 60].sum() / lat.sum()), share_lt50ms=float((lat < 0.05).mean()),
             claude_only=dict(n=int(len(lc)), mean_s=float(lc.mean()), p50=float(np.median(lc)), p90=float(np.quantile(lc, .9)), p99=float(np.quantile(lc, .99)), share_gt60s=float((lc > 60).mean()), time_share_gt60s=float(lc[lc > 60].sum() / lc.sum())),
             tool_types=[dict(tool=k, n=len(v), p50=float(np.median(v)), p90=float(np.quantile(v, .9)), mean=float(np.mean(v))) for k, v in top], claude_input_tokens=dict(tok))
    S["parse_seconds"] = time.time() - t0
    json.dump(S, open(HERE / "trace_stats.json", "w"), indent=1)
    return S

def predictor_latency():
    """single-row latency (ms) of the predictors S4/S14 use, on this laptop CPU, batch 1"""
    from common import load_banking77, embed
    import time as _t
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    b = load_banking77(); X, y = b["train"][0][:3000], b["train"][1][:3000]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2).fit(X); lr = LogisticRegression(max_iter=300).fit(vec.transform(X), y)
    q = b["test"][0][0]
    rng = np.random.default_rng(0); Xt = rng.normal(size=(3000, 12)); yt = (Xt[:, 0] + rng.normal(size=3000) > 0).astype(int) + (Xt[:, 1] > 1)
    hgb = HistGradientBoostingClassifier(max_iter=100).fit(Xt, yt)
    table = {("ledger", "a", "b"): "read_x"}; ng = ("get_user_details", "get_order_details")
    def tm(fn, k=300):
        fn(); a = []
        for _ in range(k):
            t0 = _t.perf_counter(); fn(); a.append((_t.perf_counter() - t0) * 1000)
        return float(np.median(a))
    out = dict(ngram_lookup=tm(lambda: table.get(("ledger", "a", "b")) or ng), tfidf_lr_77class=tm(lambda: lr.predict_proba(vec.transform([q]))), hgb_12feat=tm(lambda: hgb.predict_proba(Xt[:1])))
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu"); out["bge_small_embed_cpu"] = tm(lambda: m.encode([q], normalize_embeddings=True, show_progress_bar=False), 60)
    out["json_compare_verify"] = tm(lambda: json.dumps({"status": "refunded"}, sort_keys=True) == json.dumps({"status": "refunded"}, sort_keys=True), 1000)
    return out

def hit_rates():
    """precision of launched speculations = useful% / launched% from S14 tables_E.md (S14 RESULTS.md was still pending when this ran)"""
    out = []
    for l in open(SM / "s14_tool_traces" / "tables_E.md", encoding="utf-8"):
        c = [x.strip() for x in l.strip().strip("|").split("|")]
        if len(c) >= 15 and c[0] in ("W", "X"):
            out.append(dict(regime=c[0], domain=c[1], agent=c[2], useful=float(c[-2]), launched=float(c[-1]), precision=float(c[-2]) / float(c[-1])))
    return out

HITS = (0.19, 0.47, 0.73, 0.79)   # S14 launched-speculation precision (useful / launched, tables_E.md): cold airline 0.19/0.20, cold retail 0.47/0.48, warm 0.68-0.79; the grid takes 0.19, 0.47, 0.73, 0.79
NEXT = (("haiku step 0.83 s", 0.828), ("sonnet step 1.15 s", 1.149), ("thinking step 2.1 s", 2.131))
def predictors(pl):   # name -> latency ms
    return {"n-gram / trajectory cache": pl["ngram_lookup"], "TF-IDF + LR (77-class)": pl["tfidf_lr_77class"], "gradient boosting (12 features)": pl["hgb_12feat"],
            "bge-small embed (+LR)": pl["bge_small_embed_cpu"], "qwen3-32b LLM prediction (S13 p50)": 390.0, "haiku-5.5 LLM prediction (S13 p50)": 828.0}

def spec_gain(L, L2, h, p, v, serial=False):
    """Seconds saved per speculation vs running tool (L) and next step (L2) in sequence. ONE model for every table.
    parallel (default): the predictor runs alongside the previous step and only delays the launch by p, so a hit hides min(L, L2 - p); every call pays the verify v; a miss costs no latency.
    serial (worst case): the predictor blocks the critical path on every call: a hit hides min(L, L2) and every call pays p + v."""
    return (h * np.minimum(L, L2) - (p + v)) if serial else (h * np.minimum(L, np.maximum(L2 - p, 0)) - v)

def spec_table(arrs, pl):
    """arrs: pool name -> exact latency array (every value, no resampling)."""
    v = pl["json_compare_verify"] / 1000; PRED = predictors(pl)
    res = dict(stats={}, ceiling=[], saving=[], breakeven=[], min_hit=[], fixed=[])
    for pool, L in arrs.items():
        q = np.quantile(L, [.5, .9, .99])
        res["stats"][pool] = dict(n=int(len(L)), mean=float(L.mean()), std=float(L.std()), max=float(L.max()), p50=float(q[0]), p90=float(q[1]), p99=float(q[2]), share_ge1s=float((L >= 1).mean()), time_share_ge1s=float(L[L >= 1].sum() / L.sum()))
        for nm, L2 in NEXT:
            res["ceiling"].append(dict(pool=pool, next=nm, pct=float(np.minimum(L, L2).mean() / (L + L2).mean())))   # perfect predictor, zero overhead
            for pn in ("gradient boosting (12 features)", "qwen3-32b LLM prediction (S13 p50)"):
                p = PRED[pn] / 1000
                for serial in (False, True):
                    for h in HITS:
                        g = spec_gain(L, L2, h, p, v, serial); m = np.median(L)
                        res["saving"].append(dict(pool=pool, next=nm, predictor=pn, mode="serial" if serial else "parallel", hit=h, pct=float(g.mean() / (L + L2).mean()),
                                                  pct_median_call=float(spec_gain(m, L2, h, p, v, serial) / (m + L2)), pct_calls_ge1s=float(g[L >= 1].mean() / (L[L >= 1] + L2).mean())))
        if pool.startswith("pooled (exact)") or pool.startswith("claude"):
            L2 = 1.149   # break-even against a Sonnet-length next step
            for pn, ms in PRED.items():
                for serial in (False, True):
                    for h in HITS:
                        Ls = (ms / 1000 + v) / h if serial else v / h
                        win = L2 if serial else L2 - ms / 1000     # time the next step leaves for overlap
                        ok = bool(win > Ls)
                        res["breakeven"].append(dict(pool=pool, predictor=pn, overhead_ms=ms, mode="serial" if serial else "parallel", hit=h, L_star_s=float(Ls), window_s=float(win), possible=ok,
                                                     calls_above=float((L > Ls).mean()) if ok else 0.0, time_above=float(L[L > Ls].sum() / L.sum()) if ok else 0.0))
    for pn, ms in PRED.items():   # hit rate needed when the next step is a 0.83 s Haiku call
        for serial in (False, True):
            for Lt in (0.03, 0.1, 0.3, 1.0, 3.0):
                p = ms / 1000; win = min(Lt, 0.828) if serial else min(Lt, max(0.828 - p, 0)); need = (p + v) if serial else v
                res["min_hit"].append(dict(predictor=pn, mode="serial" if serial else "parallel", tool_s=Lt, h_star=float(need / win) if win > 0 else None))
    p = PRED["gradient boosting (12 features)"] / 1000   # fixed-latency regime (bookkeeping-style API tools, no heavy tail)
    res["fixed"] = [dict(tool_s=Lt, next_s=L2, hit=h, mode="serial" if sr else "parallel", pct=float(spec_gain(Lt, L2, h, p, v, sr) / (Lt + L2))) for Lt in (0.1, 0.3, 1.0, 3.0) for L2 in (0.828, 1.149, 2.131) for h in HITS for sr in (False, True)]
    return res

class Set:   # temporarily set module globals
    def __init__(self, **kw): self.kw = kw
    def __enter__(self): self.old = {k: globals()[k] for k in self.kw}; globals().update(self.kw)
    def __exit__(self, *a): globals().update(self.old)

def solve(f, lo, hi):
    """root of an increasing f on [lo, hi]; None if f(hi) < 0, lo if f(lo) >= 0"""
    if f(hi) < 0: return None
    if f(lo) >= 0: return lo
    for _ in range(50):
        mid = (lo + hi) / 2; lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return hi

def sens(S):
    """Sensitivity of every lever to the assumptions v1 fixed: thinking delta, student coverage, pruner removal, audit rate, cache hit rate, history size, extra LLM steps."""
    out = {}
    for t in ("sonnet-5.5", "haiku-5.5", "gpt-5.4"):
        o = out[t] = {}; W = lambda **kw: sum(workflow(t, S, **kw).values())
        o["think"] = []
        for d in (0, 15, 32, 64, 100, 157, 300):
            with Set(THINK_DELTA=d): b = W(); o["think"].append(dict(delta=d, base=b, off=b - W(think=False)))
        o["student"], o["prune"], o["breakeven"], o["matrix"] = {}, {}, {}, []
        for sc, d in THINK_SCEN.items():
            with Set(THINK_DELTA=d):
                b = W(); o["student"][sc] = [dict(cov=c, base=b, saving=b - W(student=c)) for c in (0.25, 0.4, 0.5, 0.6, 0.75, 0.9)]
                o["prune"][sc] = [dict(removed=r, saving=b - W(prune=r)) for r in (0.3, 0.5, 0.7, 0.8)]
                o["breakeven"][sc] = {}
                for r in (0.5, 0.7, 0.8):
                    tgt = b - W(prune=r)
                    def f(dl):
                        with Set(THINK_DELTA=dl): return (W() - W(think=False)) - tgt
                    o["breakeven"][sc][f"prune{r}"] = dict(target=tgt, student_cov=solve(lambda c: (b - W(student=c)) - tgt, 0, 1), think_delta=solve(f, 0, 5000))
                for c in (0.5, 0.75, 0.9):
                    for r in (0.0, PRUNE_S6B, 0.7):
                        m1, m2, m12, pr = b - W(think=False), b - W(student=c), b - W(think=False, student=c), b - W(prune=r)
                        inc = W(think=False, student=c, cache=True) - W(think=False, student=c, cache=True, prune=r)
                        o["matrix"].append(dict(scen=sc, delta=d, base=b, cov=c, prune=r, m1=m1, m2=m2, m12=m12, pruner=pr, pruner_on_top_of_12=inc))
        o["bands"] = {}
        for sc, d in THINK_SCEN.items():   # parameter bands, each under both thinking scenarios
          with Set(THINK_DELTA=d):
            bd = o["bands"][sc] = dict(audit=[], hit=[], hist=[])
            for au in (0.02, 0.05, 0.10):
                with Set(AUDIT=au): b = W(); bd["audit"].append(dict(audit=au, base=b, student75=b - W(student=0.75), prune50=b - W(prune=PRUNE_S6B), prune70=b - W(prune=0.7), think_off=b - W(think=False)))
            for hh in (0.5, 0.8, 0.95, 0.99):
                with Set(H_SYNC=hh): b = W(); b77 = W(cat="cat77"); bd["hit"].append(dict(hit=hh, layout=b - W(layout=True), cache77=b77 - W(cat="cat77", cache=True), base=b))
            for sc_ in (1.0, 2.2):   # S6b prompts average 1,989 tokens vs ~1,000 here: scale the history
                S2 = {t: {s_: dict(S[t][s_], H=S[t][s_]["H"] * sc_) for s_ in S[t]}}; W2 = lambda **kw: sum(workflow(t, S2, **kw).values()); b = W2()
                bd["hist"].append(dict(scale=sc_, base=b, student75=b - W2(student=0.75), prune50=b - W2(prune=PRUNE_S6B), prune70=b - W2(prune=0.7), think_off=b - W2(think=False)))
            # vendor_lookup and match_invoice priced as two extra LLM calls the size of flag_anomaly without history (an upper bound; missions.md calls vendor_lookup a tool and never mentions match_invoice)
            f = S[t]["flag"]; S3 = {t: dict(S[t], x=dict(P=f["P"], V=f["V"], H=0.0, O=2.0, thinks=False))}
            xb = 2 * CALLS_PER_DAY * step_cost(t, "x", S3); xs = 2 * CALLS_PER_DAY * step_cost(t, "x", S3, student=0.75); b = W()
            bd["extra_steps"] = dict(base=b, base_with_extra=b + xb, extra_cost=xb, student75_extra_saving=xb - xs, student75_saving=b - W(student=0.75), prune50=b - W(prune=PRUNE_S6B), think_off=b - W(think=False))
    return out


# ------------------------------------------------------------------------------------------------ main
def econ_block(S):
    """everything that depends on THINK_DELTA (set by the caller): per-step table, shares, daily bills, levers, cache sweeps, one-time costs"""
    o = {}
    rows = table_per_step(S)
    o["per_step_per_1000"] = {f"{t}|{s}": d for (t, s), d in rows.items()}
    o["shares"] = {}
    for t in ("sonnet-5.5", "haiku-5.5", "gpt-5.4-mini"):
        for step in ("cat12", "cat77", "flag", "note"):
            sh, tot = shares(t, S, step); o["shares"][f"{t}|{step}"] = dict(sh, usd_per_1000=tot * 1000)
    o["daily"] = {}
    for t in TEACHERS:
        o["daily"][t] = {}
        for name, kw in COMBOS:
            w = workflow(t, S, **kw); o["daily"][t][name] = dict(total=sum(w.values()), **w)
        o["daily"][t]["8b prune only flag_anomaly (typed, S6b step) + 1+2 + caching"] = dict(total=sub_prune_flag_only(t, S))
        o["daily"][t]["0b baseline with the 77-label categorize (measured S13 prompt)"] = dict(total=sum(workflow(t, S, cat="cat77").values()))
        o["daily"][t]["1b caching with the 77-label categorize"] = dict(total=sum(workflow(t, S, cache=True, cat="cat77").values()))
    o["levers"] = {}
    for t in ("sonnet-5.5", "haiku-5.5", "gpt-5.4-mini", "gemini-3.5-flash-lite", "gpt-5.4"):
        b, L = levers(t, S); o["levers"][t] = dict(baseline_day=b, rows=L)
    wb = lambda t, **kw: sum(workflow(t, S, **kw).values())
    o["tier_down"] = {"sonnet->haiku": (wb("sonnet-5.5"), wb("haiku-5.5")), "gpt-5.4->mini": (wb("gpt-5.4"), wb("gpt-5.4-mini")), "mini->nano": (wb("gpt-5.4-mini"), wb("gpt-5.4-nano"))}
    o["prefix_sweep"] = {}
    for key, t, mc in (("sonnet-5.5", "sonnet-5.5", None), ("haiku-5.5", "haiku-5.5", None), ("gpt-5.4-mini (min 1,024: documented for GPT-5.6+, unverified for 5.4)", "gpt-5.4-mini", None),
                       ("gemini-3.5-flash-lite (min 2,048)", "gemini-3.5-flash-lite", None), ("gemini-3.5-flash-lite (min 4,096)", "gemini-3.5-flash-lite", 4096)):
        rows_, cr = prefix_sweep(t, S, minc=mc); o["prefix_sweep"][key] = dict(rows=rows_, crossover=cr)
    o["hit_sens"] = {}
    for hh in (0.0, 0.5, 0.8, 0.95, 0.99):
        with Set(H_SYNC=hh): o["hit_sens"][hh] = {t: call_cost(t, S[t]["cat77"], cache=True) * 1000 for t in ("sonnet-5.5", "haiku-5.5")}
    # one-time build costs (teacher labels): 1000 rows per typed site, 2 sites; pruner needs ~400 calibration rows x (full + pruned) per site, S6b ablation labels extra
    o["onetime"] = {}
    for t in TEACHERS:
        lab = 1000 * (call_cost(t, S[t]["cat12"], think=False, batch=True) + call_cost(t, S[t]["flag"], think=True, batch=True))
        prn = 400 * (call_cost(t, S[t]["flag"], batch=True) + call_cost(t, S[t]["flag"], prune=0.8, batch=True)) * 3
        o["onetime"][t] = dict(student_labels_usd=lab, pruner_calibration_usd=prn)
    return o

def main(stage):
    global PR
    PR = prices()
    out = {}
    if stage == "trace": print(json.dumps(trace_stats())[:1500]); return
    if stage in ("tokens", "all", "econ", "sens"):
        T = json.load(open(HERE / "tokens.json", encoding="utf-8")) if (HERE / "tokens.json").exists() and "--retokens" not in sys.argv else tokens()   # v2: reuse the frozen profile so every number is reproducible from the saved file
        S = profiles(T)
        out["tokens"] = T; out["profiles"] = {t: S[t] for t in S}
    if stage in ("econ", "all"):
        out["econ"] = {}
        for sc, d in THINK_SCEN.items():
            with Set(THINK_DELTA=d):
                out["econ"][sc] = o = econ_block(S)
                print(f"\n=== econ, THINK_DELTA={d} ({sc}) ===")
                for t in ("sonnet-5.5", "haiku-5.5", "gpt-5.4", "gpt-5.4-mini", "gemini-3.5-flash-lite"):
                    print(" ", t)
                    for name, dd in o["daily"][t].items(): print(f"    {name:86s} ${dd['total']:9.2f}/day")
                for t in ("sonnet-5.5", "haiku-5.5", "gpt-5.4", "gemini-3.5-flash-lite"):
                    print(f"  levers {t}: baseline ${o['levers'][t]['baseline_day']:.2f}/day")
                    for n_, usd, base in o["levers"][t]["rows"]: print(f"     {n_:68s} saves ${usd:8.3f}/day {usd / base * 100:6.1f}% of ${base:.2f}")
                for k, v_ in o["prefix_sweep"].items(): print("  crossover", k, v_["crossover"])
    if stage in ("sens", "all"):
        out["sens"] = sn = sens(S)
        for t, o in sn.items():
            print("\n=== sens", t)
            print("  thinking-off saving by delta:", [(r["delta"], round(r["off"], 2)) for r in o["think"]], " baseline at each:", [round(r["base"], 2) for r in o["think"]])
            for sc in THINK_SCEN:
                print(" ", sc, "student:", [(r["cov"], round(r["saving"], 2)) for r in o["student"][sc]], "prune:", [(r["removed"], round(r["saving"], 2)) for r in o["prune"][sc]])
                print("   break-even:", json.dumps(o["breakeven"][sc], default=float))
            for sc, bd in o["bands"].items(): print(" ", sc, "audit:", [{k: round(v_, 2) for k, v_ in r.items()} for r in bd["audit"]], "hist:", [{k: round(v_, 2) for k, v_ in r.items()} for r in bd["hist"]], "extra:", {k: round(v_, 2) for k, v_ in bd["extra_steps"].items()})
    if stage in ("spec", "all"):
        TS = trace_stats() if not ((HERE / "trace_stats.json").exists() and (HERE / "tracelab_latency_claude_s.npy").exists()) else json.load(open(HERE / "trace_stats.json"))
        lat = np.load(HERE / "tracelab_latency_s.npy").astype(float); latc = np.load(HERE / "tracelab_latency_claude_s.npy").astype(float)
        prev = HERE / "results_spec.json"; pl = json.load(open(prev))["predictor_ms"] if prev.exists() else predictor_latency(); print("predictor latency ms:", pl)   # ponytail: timing is hardware-bound, measure once and reuse
        hr = hit_rates(); print("S14 launched-speculation precision:", sorted(round(h["precision"], 3) for h in hr))
        arrs = {"pooled (exact)": lat, "claude only (exact)": latc, "pooled, capped at 1 h": np.minimum(lat, 3600.0)}   # v2: every value, no resampling (v1 drew 200k samples from a tail with sd 789 s and max 440,698 s)
        sp = spec_table(arrs, pl); out["trace"] = TS; out["predictor_ms"] = pl; out["hit_rates"] = hr; out["spec"] = sp
        for k, v_ in sp["stats"].items(): print(k, {a: round(b, 3) for a, b in v_.items()})
        for r in sp["ceiling"]: print(f"  ceiling {r['pool']:24s} {r['next']:20s} {r['pct'] * 100:5.2f}%")
        for r in sp["saving"]:
            if r["predictor"].startswith("gradient"): print(f"  saving {r['pool']:24s} {r['next']:20s} {r['mode']:8s} hit {r['hit']:.2f}: {r['pct'] * 100:5.2f}% (median call {r['pct_median_call'] * 100:5.1f}%, calls>=1s {r['pct_calls_ge1s'] * 100:5.1f}%)")
    json.dump(out, open(HERE / f"results_{stage}.json", "w"), indent=1, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
