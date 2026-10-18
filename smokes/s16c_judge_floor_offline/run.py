"""S16c: cache-only re-analysis of the draft_note floor (no network, no LLM call, $0).
Usage: python3 run.py            (reads ../s16_pruner_note/{responses.jsonl,prompts.json,labels.json,results.json}; writes log.txt results.json tables.md)
Notes and judge verdicts are re-derived with s16_pruner_note/core.py's own functions (note(), J(), verdict()) on top of cacheonly.py, a stand-in for llm.py
that builds the same sha1 cache key and raises on a miss instead of calling the API. Missing keys are counted and reported."""
import sys, json, re, math, itertools, collections
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cacheonly; cacheonly.install()
import core
from core import P, note, keep_all, J, JSYS_CORE, MINI, isdiff, verdict
from common import cp_upper
import canon as C
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
S16 = HERE.parent / "s16_pruner_note"
LOG = open(HERE / "log.txt", "w", encoding="utf-8"); RES = {}; TAB = []
def say(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def tab(s=""): TAB.append(s)
ub = lambda k, n: cp_upper(int(k), int(n), 0.05) if n else float("nan")
pc = lambda x: "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"
def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)): return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    return o
def save(): json.dump(jsonable(RES), open(HERE / "results.json", "w"), indent=1)
def n_needed(rate, b):
    """smallest n whose one-sided 95% Clopper-Pearson bound, at the observed rate (k = round(rate*n)), is <= budget b; None if rate >= b"""
    if rate >= b: return None
    if rate == 0: return int(math.ceil(math.log(0.05) / math.log(1 - b)))
    n = 20
    while n <= 200000:
        if ub(round(rate * n), n) <= b: return n
        n += 1 if n < 3000 else 25
    return None
def stats_row(k, n):
    r = k / n; return dict(k=int(k), n=int(n), rate=r, ub=ub(k, n), ub_at={m: ub(round(r * m), m) for m in (100, 300, 1000)}, n5=n_needed(r, .05), n10=n_needed(r, .10))

# ------------------------------------------------------------------ 0. recover notes and judge verdicts from the cache
LAB = sorted(int(k) for k in json.load(open(S16 / "labels.json")))
assert LAB == list(range(80)) + list(range(200, 260))
FLOOR = list(range(200, 300))      # run.py: HELD = TRAIN..300 -> HELD = range(200,300); floor_one(i) uses note(p, keep_all(p), rep=3) vs the three reps 0-2
NOTE = {}; miss_notes = collections.Counter()
for i in sorted(set(LAB) | set(FLOOR)):
    NOTE[i] = []
    for r in range(4):
        if r == 3 and i not in FLOOR: NOTE[i].append(None); continue
        try: NOTE[i].append(note(P[i], keep_all(P[i]), r)["text"])
        except cacheonly.Miss: NOTE[i].append(None); miss_notes[r] += 1
lab_json = json.load(open(S16 / "labels.json"))
assert all(NOTE[i][:3] == lab_json[str(i)]["refs"] for i in LAB), "re-derived refs differ from labels.json"
say(f"[cache] responses.jsonl entries {len(cacheonly.R)}; notes re-derived for {len(NOTE)} prompts; missing note keys by rep: {dict(miss_notes)}; re-derived A0-A2 == labels.json refs for all {len(LAB)} labelled prompts")
RES["cache"] = dict(entries=len(cacheonly.R), prompts=len(NOTE), missing_note_keys=dict(miss_notes), floor_ids="200-299 (n=100)", labelled_ids="0-79,200-259 (n=140)")

tryj = lambda a, b, **kw: (lambda: None)  # placeholder
def TJ(a, b, **kw):
    """cached judge verdict or None when the key is not in the cache (identical strings -> 'SAME' without a call, as in core.J)"""
    if a is None or b is None: return None
    try: return J(a, b, **kw)
    except cacheonly.Miss: return None

# ------------------------------------------------------------------ 1. canonicalizer validation against prompts.json ground truth
CN = {(i, r): C.canon(t, P[i]["tx"]) for i in NOTE for r, t in enumerate(NOTE[i]) if t}
GT = {i: C.gt_of(P[i]) for i in NOTE}
allnotes = list(CN)
say(f"\n=== 1. canonicalizer vs ground truth on {len(allnotes)} full-prompt notes ({len(NOTE)} prompts; A0-A2 on 180 prompts incl. 260-299 where cached, A3 on 100)")
V = {}
def val(name, fn, applicable=lambda i: True):
    """fn(canon, gt, p) -> (stated: bool, correct: bool). stated share = extractor produced a value; accuracy = correct among stated; strict = correct / all applicable"""
    n = st = ok = 0
    for (i, r) in allnotes:
        if not applicable(i): continue
        s, c = fn(CN[i, r], GT[i], P[i]); n += 1; st += s; ok += (s and c)
    V[name] = dict(n=n, stated=st / n, unclear=1 - st / n, acc_stated=(ok / st if st else None), acc_all=ok / n)
    say(f"[field] {name:34s} n={n:4d} stated {pc(st/n):>6s} unclear {pc(1-st/n):>6s} accuracy among stated {pc(ok/st if st else None):>6s} correct/all {pc(ok/n):>6s}")
close = lambda a, b: a is not None and b is not None and abs(a - b) < 0.005
val("vendor name (fuzzy vs tx string)", lambda c, g, p: (c["vendor_ok"] is True, c["vendor_ok"] is True))
val("transaction amount", lambda c, g, p: (c["tx_amt"] is not None, close(c["tx_amt"], g["tx_amt"])))
val("transaction date (first date)", lambda c, g, p: (c["tx_date"] is not None, c["tx_date"] == g["tx_date"]))
val("invoice outcome (id or NONE)", lambda c, g, p: (c["inv_id"] is not None, c["inv_id"] == g["inv_id"]))
val("invoice amount (prompts with invoice)", lambda c, g, p: (c["inv_amt"] is not None, close(c["inv_amt"], g["inv_amt"])), lambda i: GT[i]["inv_id"] != "NONE")
val("invoice issue date (prompts w/ invoice)", lambda c, g, p: (c["inv_date"] is not None, c["inv_date"] == g["inv_date"]), lambda i: GT[i]["inv_id"] != "NONE")
val("bank amount, bank != tx (mismatch)", lambda c, g, p: (c["bank_amt"] is not None, close(c["bank_amt"], g["bank_amt"])), lambda i: P[i]["bank_mismatch"])
val("bank amount, bank == tx (no mismatch)", lambda c, g, p: (c["bank_amt"] is not None, close(c["bank_amt"], g["bank_amt"])), lambda i: not P[i]["bank_mismatch"] and "-$" in P[i]["bank"])
val("flag decision (flagged/not flagged)", lambda c, g, p: (c["flag"] is not None, c["flag"] == g["flag"]))
val("flag reason category (set)", lambda c, g, p: (c["reason"] is not None and c["reason"] != ("other",), c["reason"] == g["reason"]))
val("flag reason, flagged prompts only", lambda c, g, p: (c["reason"] is not None and c["reason"] != ("other",), c["reason"] == g["reason"]), lambda i: GT[i]["flag"] == "yes")
fu = [CN[k]["followup"] for k in allnotes]; hi = [CN[k]["history"] for k in allnotes]
V["followup_requested_rate"] = float(np.mean(fu)); V["history_mention_rate"] = float(np.mean(hi))
say(f"[field] follow-up requested (no ground truth in prompts.json): yes in {pc(np.mean(fu))} of notes ({sum(fu)}/{len(fu)}); history mention: yes in {pc(np.mean(hi))} ({sum(hi)}/{len(hi)})")
bk = [CN[k]["bank_amt"] is not None for k in allnotes if P[k[0]]["bank_mismatch"]]
say(f"[field] bank-feed amount stated in notes for bank-mismatch prompts: {pc(np.mean(bk))} ({sum(bk)}/{len(bk)}); whole-note exact-field match (flag, reason, invoice, tx amount all right): "
    f"{pc(np.mean([CN[k]['flag']==GT[k[0]]['flag'] and CN[k]['reason']==GT[k[0]]['reason'] and CN[k]['inv_id']==GT[k[0]]['inv_id'] and close(CN[k]['tx_amt'],GT[k[0]]['tx_amt']) for k in allnotes]))}")
V["bank_stated_when_mismatch"] = float(np.mean(bk))
V["typed_fields_all_right"] = float(np.mean([CN[k]['flag']==GT[k[0]]['flag'] and CN[k]['reason']==GT[k[0]]['reason'] and CN[k]['inv_id']==GT[k[0]]['inv_id'] and close(CN[k]['tx_amt'],GT[k[0]]['tx_amt']) for k in allnotes]))
RES["validation"] = V; save()
tab("## Table 1. Canonicalizer vs prompts.json ground truth (full-prompt notes, n = %d)\n" % len(allnotes))
tab("| field | n | stated | unclear / not stated | accuracy among stated | correct / all |"); tab("|---|---|---|---|---|---|")
for k, v in V.items():
    if isinstance(v, dict): tab(f"| {k} | {v['n']} | {pc(v['stated'])} | {pc(v['unclear'])} | {pc(v['acc_stated'])} | {pc(v['acc_all'])} |")
tab(f"\nFollow-up requested in {pc(V['followup_requested_rate'])} of notes; history mentioned in {pc(V['history_mention_rate'])}; bank amount stated in {pc(V['bank_stated_when_mismatch'])} of bank-mismatch notes.\n")

# ------------------------------------------------------------------ 2. pairs and equivalence definitions
ALLF = ["vendor", "tx_amt", "tx_date", "inv", "inv_amt", "bank_amt", "flag", "reason", "followup"]
CORE = ["vendor", "tx_amt", "inv", "inv_amt", "flag"]       # (d): vendor, amount, invoice outcome (id + amount), flag decision
TYPED = ["flag", "reason", "inv", "tx_amt"]                   # fields a typed draft_note step would emit (vendor/dates are copied from the record)
def fv(c, f):
    return {"vendor": True if c["vendor_ok"] else None, "tx_amt": c["tx_amt"], "tx_date": c["tx_date"], "inv": c["inv_id"], "inv_amt": c["inv_amt"], "bank_amt": c["bank_amt"],
            "flag": c["flag"], "reason": c["reason"] if c["reason"] != ("other",) else None, "followup": c["followup"]}[f]
def same_v(x, y): return abs(x - y) < 0.005 if isinstance(x, float) and isinstance(y, float) else x == y
EV = ["hist_count", "hist_lo", "hist_hi", "hist_top", "other_dates"]
def compare(ca, cb):
    conf, ind, asym = set(), set(), set()
    for f in ALLF:
        x, y = fv(ca, f), fv(cb, f)
        if x is None and y is None: continue
        if x is None or y is None:
            (ind if f in ("flag", "tx_amt", "inv", "reason", "tx_date") else asym).add(f); continue
        if f == "inv_amt" and fv(ca, "inv") != fv(cb, "inv"): continue
        if not same_v(x, y): conf.add(f)
    cev, cevt, aev = set(), set(), set()
    for f in EV:
        x, y = ca[f], cb[f]
        if x is None and y is None: continue
        if x is None or y is None: aev.add(f); continue
        if x != y: cev.add(f)
        if isinstance(x, float) and abs(x - y) > max(1.0, 0.05 * max(x, y)) or not isinstance(x, float) and x != y: cevt.add(f)
    ident = not conf and not ind and not asym and not cev and not aev and ca["history"] == cb["history"] and ca["followup"] == cb["followup"]
    return conf, ind, asym, ident, cev, cevt, aev
def norm(t):
    t = t.lower().replace("$", " "); t = re.sub(r"(?<=\d),(?=\d{3})", "", t); t = re.sub(r"(\d\.\d*?)0+(?!\d)", r"\1", t); t = re.sub(r"(\d)\.(?!\d)", r"\1", t)
    return " ".join(re.sub(r"[^a-z0-9.*#]+", " ", t).split())
pairs = []
def addpair(i, ka, kb, ptype):
    a, b = NOTE[i][ka], NOTE[i][kb]
    if a is None or b is None: return
    conf, ind, asym, ident, cev, cevt, aev = compare(CN[i, ka], CN[i, kb]); d = dict(i=i, cev=cev, cevt=cevt, aev=aev, ka=ka, kb=kb, type=ptype, a=a, b=b, conf=conf, ind=ind, asym=asym, ident=ident, scn=P[i]["scn"],
        exact=norm(a) == norm(b), strict=None, strict_src=None, cons=None, core=None, swap=None, mini=None, hist_cat=("both" if CN[i, ka]["history"] and CN[i, kb]["history"] else "one" if CN[i, ka]["history"] or CN[i, kb]["history"] else "neither"))
    d["c"] = bool(conf); d["c_nofu"] = bool(conf - {"followup"}); d["d"] = bool(conf & set(CORE)); d["cplus"] = bool(conf or cev); d["cplus_tol"] = bool(conf or cevt); d["typed"] = bool(conf & set(TYPED)); d["typed_ind"] = bool((conf | ind) & set(TYPED))
    pairs.append(d)
for i in LAB:
    for ka, kb in ((0, 1), (0, 2), (1, 2)): addpair(i, ka, kb, f"A{ka}-A{kb}")
for i in FLOOR:
    addpair(i, 0, 3, "A0-A3")
    for ka, kb in ((1, 3), (2, 3)): addpair(i, ka, kb, f"A{ka}-A{kb}")
# cached judge verdicts. strict (JSYS) for every pair if the key exists; identical strings are SAME without a call (kept apart: 'ident')
miss_j = collections.Counter(); have_j = collections.Counter()
for d in pairs:
    for nm, kw in (("strict", {}), ("core", dict(system=JSYS_CORE)), ("swap", dict(swap=True)), ("mini", dict(model=MINI))):
        v = TJ(d["a"], d["b"], **kw)
        if d["a"] == d["b"] and v is not None: d[nm] = "ident"; continue
        if v is None: miss_j[(d["type"], nm)] += 1
        else: have_j[(d["type"], nm)] += 1
        d[nm] = v
refs_of_i = {i: NOTE[i][:3] for i in FLOOR}
for d in pairs:
    if d["type"] == "A0-A3":
        try: v = verdict(refs_of_i[d["i"]], d["b"], True); d["cons"] = v["cons"]; d["chain"] = v["j"]; assert v["raw"] == (d["strict"] if d["strict"] != "ident" else "SAME")
        except cacheonly.Miss: d["cons"] = None
say("\n=== 2. pairs: " + ", ".join(f"{t}: {sum(1 for d in pairs if d['type']==t)}" for t in ("A0-A1", "A0-A2", "A1-A2", "A0-A3", "A1-A3", "A2-A3")))
say("[cache] strict judge verdicts present (non-identical pairs only counted as real calls) / missing, by pair type:")
for t in ("A0-A1", "A0-A2", "A1-A2", "A0-A3", "A1-A3", "A2-A3"):
    n = sum(1 for d in pairs if d["type"] == t); ni = sum(1 for d in pairs if d["type"] == t and d["strict"] == "ident")
    say(f"   {t}: {n} pairs; identical strings {ni}; strict cached {have_j[(t,'strict')]}, missing {miss_j[(t,'strict')]}; core-rubric cached {have_j[(t,'core')]}; order-swapped cached {have_j[(t,'swap')]}; second judge cached {have_j[(t,'mini')]}")
A03 = [d for d in pairs if d["type"] == "A0-A3"]
raw_d = np.array([d["strict"] == "ident" and False or (d["strict"] not in ("SAME", "ident")) for d in A03]); cons_d = np.array([d["cons"] not in ("SAME", None) for d in A03])
assert sum(d["strict"] is None for d in A03) == 0 and sum(d["cons"] is None for d in A03) == 0
r16 = json.load(open(S16 / "results.json"))["floor"]
say(f"[check] A0-A3 re-derived: strict raw DIFFERENT {raw_d.sum()}/100 (results.json floor.raw={r16['raw']}), confirmed {cons_d.sum()}/100 (floor.cons={r16['cons']}); FAIL counted as DIFFERENT")
assert abs(raw_d.mean() - r16["raw"]) < 1e-9 and abs(cons_d.mean() - r16["cons"]) < 1e-9
RES["judge_cache"] = dict(have={f"{k[0]}|{k[1]}": v for k, v in have_j.items()}, missing={f"{k[0]}|{k[1]}": v for k, v in miss_j.items()}, floor_recomputed=dict(raw=int(raw_d.sum()), cons=int(cons_d.sum())))

# rows of the floor table
PRIM = ["A0-A1", "A0-A2", "A1-A2", "A0-A3"]
def sel(t): return [d for d in pairs if d["type"] == t]
def boot_ub(ds, key, B=2000, seed=0):
    """prompt-cluster bootstrap 95th percentile of the pooled rate (pairs of one prompt move together)"""
    rng = np.random.default_rng(seed); by = collections.defaultdict(list)
    for d in ds: by[d["i"]].append(float(key(d)))
    ids = list(by); out = []
    for _ in range(B):
        s = rng.integers(0, len(ids), len(ids)); v = [x for j in s for x in by[ids[j]]]; out.append(np.mean(v))
    return float(np.percentile(out, 95))
DEFS = [  # name, fn(d) -> True/False/None (None = not available for that pair)
 ("(a) strict judge, raw (as cached)", lambda d: None if d["type"] != "A0-A3" else d["strict"] not in ("SAME", "ident")),
 ("(b) strict judge, confirmed (as cached)", lambda d: None if d["type"] != "A0-A3" else d["cons"] not in ("SAME",)),
 ("(a') strict judge, raw, FAIL (cut-off answer) not counted as DIFFERENT", lambda d: None if d["type"] != "A0-A3" else d["strict"] == "DIFFERENT"),
 ("(b') strict judge, confirmed, backed by >=1 real DIFFERENT vote", lambda d: None if d["type"] != "A0-A3" else (d["cons"] != "SAME" and "DIFFERENT" in d["chain"])),
 ("(c) canonical-facts conflict (all fields)", lambda d: d["c"]),
 ("(c') as (c) without the follow-up flag", lambda d: d["c_nofu"]),
 ("(c+) (c) plus history/earlier-charge evidence (count, range, top, dates)", lambda d: d["cplus"]),
 ("(c+tol) as (c+), numbers within max(1, 5%) count as equal", lambda d: d["cplus_tol"]),
 ("(d) canonical conflict, core fields only", lambda d: d["d"]),
 ("(f) exact match after normalisation", lambda d: not d["exact"]),
]
FL = {}
say("\n=== 2. floor under several equivalence definitions (DIFFERENT rate, CP 95% upper bound, budget n rows could certify at the same rate)")
tab("## Table 2. Floor under several equivalence definitions\n")
tab("UB@n = one-sided 95% Clopper-Pearson bound if n rows showed the same rate (k = round(rate*n)); n for 5% / 10% = fewest rows whose bound is below that budget at the observed rate ('never' when the rate itself is at or above the budget). n/a = verdict not in the cache for that pair type.\n")
tab("| definition | pair set | k/n | DIFFERENT | UB95 | UB@100 | UB@300 | UB@1000 | n for 5% | n for 10% |"); tab("|---|---|---|---|---|---|---|---|---|---|")
for nm, fn in DEFS:
    for t in PRIM + ["pooled A0-A1,A0-A2,A1-A2", "pooled A0-A1,A0-A2,A1-A2,A0-A3"]:
        ds = sel(t) if t in PRIM else [d for d in pairs if d["type"] in (["A0-A1", "A0-A2", "A1-A2"] if t.count("A0-A3") == 0 else PRIM)]
        v = [fn(d) for d in ds]
        if any(x is None for x in v): tab(f"| {nm} | {t} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |") if t in PRIM else None; continue
        k, n = sum(v), len(v); s = stats_row(k, n); FL[(nm, t)] = s
        extra = f" (cluster-bootstrap UB {pc(boot_ub(ds, fn))})" if t not in PRIM else ""
        say(f"[floor] {nm:46s} {t:34s} {k:3d}/{n:3d} = {pc(s['rate']):>6s}  UB95 {pc(s['ub']):>6s}{extra}  UB@100/300/1000 {pc(s['ub_at'][100])}/{pc(s['ub_at'][300])}/{pc(s['ub_at'][1000])}  n for 5%: {s['n5']}  n for 10%: {s['n10']}")
        tab(f"| {nm} | {t}{' (pairs not independent: bootstrap UB ' + pc(boot_ub(ds, fn)) + ')' if t not in PRIM else ''} | {k}/{n} | {pc(s['rate'])} | {pc(s['ub'])} | {pc(s['ub_at'][100])} | {pc(s['ub_at'][300])} | {pc(s['ub_at'][1000])} | {s['n5'] or 'never'} | {s['n10'] or 'never'} |")
# (e) canonical + core judge on A0-A3
core_av = [d for d in A03 if d["core"] not in (None, "ident")]
ident_pairs = [d for d in A03 if d["strict"] == "ident"]
res_pairs = [d for d in A03 if not d["c"] and not d["ident"]]       # canonical residual: no conflict but facts not identical
unres = [d for d in res_pairs if d["core"] in (None,) and d["strict"] != "ident"]
conf_n = sum(d["c"] for d in A03)
e_core = [d for d in A03 if d["core"] not in (None,)]   # has a core verdict, incl. 'ident' (= identical strings)
e_k = sum(d["c"] or d["core"] not in ("SAME", "ident") for d in e_core)
say(f"\n[e] canonical + core-rubric judge: A0-A3 has {len(A03)} pairs; canonical conflict {conf_n}; identical-string pairs {len(ident_pairs)}; core verdict cached for {len(core_av)} non-identical pairs (+{len(ident_pairs)} identical strings, SAME by definition); "
    f"canonical-residual pairs (no conflict, facts not identical) {len(res_pairs)} of which core verdict cached {sum(1 for d in res_pairs if d['core'] not in (None,'ident'))}, unresolved {len(unres)}")
lo = conf_n; hi = conf_n + len(unres)
say(f"[e] bounds on all 100 pairs: DIFFERENT rate between {lo}% (every unresolved residual pair SAME) and {hi}% (every unresolved residual pair DIFFERENT); on the {len(e_core)} pairs with a core verdict: DIFFERENT {e_k}/{len(e_core)}; core rubric alone on those: {sum(d['core'] not in ('SAME','ident') for d in e_core)}/{len(e_core)}")
RES["e_canon_plus_core"] = dict(n_pairs=len(A03), canon_conflict=conf_n, core_cached_nonident=len(core_av), ident=len(ident_pairs), residual=len(res_pairs), unresolved=len(unres), lo=lo, hi=hi,
    on_core_subset=dict(n=len(e_core), diff=int(e_k)), core_alone=int(sum(d['core'] not in ('SAME', 'ident') for d in e_core)))
cc = [d for d in A03 if d["core"] not in (None, "ident")]
say(f"[e] on the {len(cc)} non-identical A0-A3 pairs that have a core-rubric verdict: strict raw DIFFERENT {sum(d['strict'] not in ('SAME','ident') for d in cc)}, core-rubric DIFFERENT {sum(d['core']!='SAME' for d in cc)}; order-swapped strict DIFFERENT {sum(d['swap'] not in ('SAME','ident',None) for d in cc)}")
RES["e_canon_plus_core"]["strict_vs_core_on_core_cached"] = dict(n=len(cc), strict=int(sum(d['strict'] not in ('SAME','ident') for d in cc)), core=int(sum(d['core']!='SAME' for d in cc)))
res_s = [d for d in A03 if not d["cplus_tol"] and not d["ident"]]
say(f"[e] residual after (c+tol) (no conflict, facts not identical): {len(res_s)} pairs; the strict judge calls {sum(d['strict'] not in ('SAME','ident') for d in res_s)} of them DIFFERENT; strict judge on pairs with identical facts: {sum(d['strict'] not in ('SAME','ident') for d in A03 if d['ident'])}/{sum(d['ident'] for d in A03)}")
RES["e_canon_plus_core"]["residual_tol"] = len(res_s); RES["e_canon_plus_core"]["residual_tol_strict_diff"] = int(sum(d['strict'] not in ('SAME','ident') for d in res_s))
# --- revision: what the cache really measures for route (ii) (residual judging)
res_ni = [d for d in res_pairs if d["strict"] != "ident"]
res_core = [d for d in res_ni if d["core"] not in (None, "ident")]
res_core_diff = sum(d["core"] != "SAME" for d in res_core)
res_strict_diff = sum(d["strict"] == "DIFFERENT" for d in res_ni); res_strict_fail = sum(d["strict"] == "FAIL" for d in res_ni)
ident_str = sum(1 for d in A03 if d["strict"] == "ident"); ident_facts = sum(1 for d in A03 if d["ident"] and d["strict"] != "ident")
say(f"[e-rev] (c)-residual pairs (no (c) conflict, facts not identical): {len(res_pairs)} (non-byte-identical {len(res_ni)}); core-rubric verdict cached for {len(res_core)} of them (prompts {sorted(d['i'] for d in res_core)}), DIFFERENT {res_core_diff}/{len(res_core)} -> UB95 {pc(ub(res_core_diff, len(res_core)))}; "
    f"strict judge on the residual: DIFFERENT {res_strict_diff}, FAIL (cut-off) {res_strict_fail}; byte-identical pairs {ident_str}, identical-canonical-facts but different strings {ident_facts} (SAME by construction, not residual)")
say(f"[e-rev] so the '15 pairs, 0 DIFFERENT' of the first version = {len(e_core)} pairs with a core verdict = {len(res_core)} residual + {len(e_core)-len(res_core)} byte-identical or identical-facts (SAME by construction)")
RES["e_rev"] = dict(residual=len(res_pairs), residual_nonident=len(res_ni), residual_core_cached=len(res_core), residual_core_ids=sorted(d["i"] for d in res_core), residual_core_diff=int(res_core_diff),
    residual_core_ub=ub(res_core_diff, len(res_core)), residual_strict_diff=int(res_strict_diff), residual_strict_fail=int(res_strict_fail), byte_identical=ident_str, identical_facts_other=ident_facts, canonical_only_k=conf_n, canonical_only_ub=ub(conf_n, 100),
    range_lo=lo, range_hi=hi)
core_all = [d for d in A03 if d["core"] not in (None, "ident") ]
say(f"[e-rev] core rubric vs strict on the {len(core_all)} non-identical A0-A3 pairs where both are cached: core DIFFERENT {sum(d['core']=='DIFFERENT' for d in core_all)}, strict DIFFERENT {sum(d['strict']=='DIFFERENT' for d in core_all)} (strict FAIL {sum(d['strict']=='FAIL' for d in core_all)})")
sE = stats_row(e_k, len(e_core)) if e_core else None
tab(f"| (e) canonical conflict OR core-rubric judge, only pairs with a cached core verdict | A0-A3 subset | {e_k}/{len(e_core)} | {pc(e_k/len(e_core))} | {pc(sE['ub'])} | {pc(sE['ub_at'][100])} | {pc(sE['ub_at'][300])} | {pc(sE['ub_at'][1000])} | {sE['n5'] or 'never'} | {sE['n10'] or 'never'} |")
tab(f"| (e-lo) canonical conflict, unresolved residual counted SAME | A0-A3 all 100 | {lo}/100 | {pc(lo/100)} | {pc(ub(lo,100))} | | | | | |")
tab(f"| (e-hi) canonical conflict + every unresolved residual pair counted DIFFERENT | A0-A3 all 100 | {hi}/100 | {pc(hi/100)} | {pc(ub(hi,100))} | | | | | |")
tab(f"\n(e) has a core-rubric verdict for only {len(core_av)} non-identical A0-A3 pairs (these are the floor pairs in the 30-pair rubric-comparison sample of S16, plus {len(ident_pairs)} pairs whose two notes are byte-identical); {len(unres)} canonical-residual pairs have none, so (e) is reported as a range.\n")
RES["floor_table"] = {f"{k[0]} | {k[1]}": v for k, v in FL.items()}
# judge availability table
tab("### Table 2b. Which judge verdicts exist in the cache\n"); tab("| pair type | pairs | identical strings | strict | core rubric | order swap | second judge (mini) |"); tab("|---|---|---|---|---|---|---|")
for t in PRIM + ["A1-A3", "A2-A3"]:
    n = sum(1 for d in pairs if d["type"] == t); ni = sum(1 for d in pairs if d["type"] == t and d["strict"] == "ident")
    tab(f"| {t} | {n} | {ni} | {have_j[(t,'strict')]} | {have_j[(t,'core')]} | {have_j[(t,'swap')]} | {have_j[(t,'mini')]} |")
tab("\nA1-A3 and A2-A3 strict verdicts exist only where A0-A3 was DIFFERENT (they are the confirmation calls of the S16 confirmed verdict), so they are a biased subset.\n")
save()

# ------------------------------------------------------------------ 3. cross-tab judge vs canonical conflict
say("\n=== 3. judge verdict vs canonical facts")
JP = [d for d in pairs if d["strict"] not in (None, "ident") and d["type"] in ("A0-A3", "A1-A3", "A2-A3")]
def cat(d): return "conflict" if d["c"] else ("identical" if d["ident"] else "residual")
def xt(ds, label):
    c = collections.Counter((("DIFFERENT" if d["strict"] not in ("SAME", "ident") else "SAME"), cat(d)) for d in ds)
    say(f"[crosstab {label}] n={len(ds)}: " + ", ".join(f"judge {j} / canonical {k}: {c[(j,k)]}" for j in ("DIFFERENT", "SAME") for k in ("conflict", "residual", "identical")))
    return c
x1 = xt([d for d in A03 if d["strict"] != "ident"] + [d for d in A03 if d["strict"] == "ident" and False], "A0-A3, non-identical strings with a judge call")
xall = xt(A03, "A0-A3, all 100 (identical strings counted SAME)")
x3 = xt(JP, "A0-A3 + A1-A3 + A2-A3 with a cached strict verdict (biased subset)")
xcore = collections.Counter((("DIFFERENT" if d["strict"] not in ("SAME", "ident") else "SAME"), ("conflict" if d["d"] else "no core conflict")) for d in A03)
say(f"[crosstab vs core-field conflict, A0-A3 all 100] " + ", ".join(f"judge {k[0]} / {k[1]}: {v}" for k, v in sorted(xcore.items())))
RES["crosstab"] = dict(a03_all={f"{k[0]}|{k[1]}": v for k, v in xall.items()}, a03_core={f"{k[0]}|{k[1]}": v for k, v in xcore.items()}, a03_a13_a23={f"{k[0]}|{k[1]}": v for k, v in x3.items()})
tab("## Table 3. Judge verdict (strict, raw) vs canonical facts, pair A0-A3 (n = 100, identical-string pairs count as SAME)\n")
tab("canonical status: conflict = some field stated in both notes with different values; residual = no conflict but the notes are not field-for-field identical (a field or the history/follow-up mention is stated in only one note); identical = all extracted facts and mention flags equal.\n")
tab("| judge \\ canonical | conflict | residual | identical | total |"); tab("|---|---|---|---|---|")
for j in ("DIFFERENT", "SAME"): tab(f"| {j} | {xall[(j,'conflict')]} | {xall[(j,'residual')]} | {xall[(j,'identical')]} | {sum(xall[(j,k)] for k in ('conflict','residual','identical'))} |")
tab("\nSame cross-tab on all pairs with a cached strict verdict (A0-A3, plus A1-A3 and A2-A3 which exist only when A0-A3 was DIFFERENT; biased):\n")
tab("| judge \\ canonical | conflict | residual | identical | total |"); tab("|---|---|---|---|---|")
for j in ("DIFFERENT", "SAME"): tab(f"| {j} | {x3[(j,'conflict')]} | {x3[(j,'residual')]} | {x3[(j,'identical')]} | {sum(x3[(j,k)] for k in ('conflict','residual','identical'))} |")
tab("\nAgainst core-field conflict (vendor, amount, invoice outcome, flag) on A0-A3:\n"); tab("| judge \\ canonical | core conflict | no core conflict |"); tab("|---|---|---|")
for j in ("DIFFERENT", "SAME"): tab(f"| {j} | {xcore[(j,'conflict')]} | {xcore[(j,'no core conflict')]} |")
tab("")
def pdesc(d):
    return f"prompt {d['i']} ({d['scn']}), pair {d['type']}, judge strict={d['strict']}" + (f", confirmed={d['cons']}" if d['cons'] else "") + f"; canonical conflict={sorted(d['conf']) or 'none'}; one-sided/unstated={sorted(d['asym'] | d['ind']) or 'none'}; history mention A/B={CN[d['i'],d['ka']]['history']}/{CN[d['i'],d['kb']]['history']}"
def ex_block(title, ds, n=8):
    tab(f"### {title}\n"); tab(f"{len(ds)} pairs in this category in the cache; showing {min(n, len(ds))}.\n")
    for k, d in enumerate(ds[:n]):
        tab(f"**{k+1}. {pdesc(d)}**\n"); tab(f"> A ({d['type'].split('-')[0]}): {d['a']}\n>\n> B ({d['type'].split('-')[1]}): {d['b']}\n")
# category lists. priority: A0-A3 pairs, then A1-A3/A2-A3
catA = [d for d in JP if d["strict"] != "SAME" and not d["c"]]; catA.sort(key=lambda d: (d["strict"] == "FAIL", d["type"] != "A0-A3", not d["ident"], d["i"]))
def why2(d):
    w = []
    if d["conf"]: w.append("(c) conflict " + str(sorted(d["conf"])))
    if d["cev"]: w.append("history-evidence conflict " + str(sorted(d["cev"])))
    if d["ind"] & {"flag", "tx_amt", "inv", "reason", "tx_date"}: w.append("required field unstated in one note " + str(sorted(d["ind"])))
    if "bank_amt" in d["asym"]: w.append("bank amount stated in one note only")
    if "vendor" in d["asym"]: w.append("vendor name not recognised in one note")
    return "; ".join(w)
catB_c = [d for d in JP if d["strict"] == "SAME" and d["c"]]
catB_ev = [d for d in JP if d["strict"] == "SAME" and not d["c"] and d["cev"]]
catB_wk = [d for d in JP if d["strict"] == "SAME" and not d["c"] and not d["cev"] and why2(d)]
catB = sorted(catB_c, key=lambda d: (d["type"] != "A0-A3", d["i"])) + sorted(catB_ev, key=lambda d: (d["type"] != "A0-A3", d["i"])) + sorted(catB_wk, key=lambda d: (d["type"] != "A0-A3", d["i"]))
catA_id = [d for d in catA if d["ident"]]
catA_nocplus = [d for d in catA if not d["cplus"]]
say(f"[crosstab] judge-DIFFERENT with NO (c) conflict: {len(catA)} pairs ({len(catA_id)} with field-for-field identical canonical facts incl. evidence; {len(catA_nocplus)} with no (c+) conflict); "
    f"judge-SAME with a (c) conflict: {len(catB_c)}; with only a history-evidence conflict: {len(catB_ev)}; with only a core-field one-sided/unstated item: {len(catB_wk)}")
xc = collections.Counter((("DIFFERENT" if d["strict"] not in ("SAME", "ident") else "SAME"), "conflict (c+)" if d["cplus"] else ("identical" if d["ident"] else "residual")) for d in A03)
say("[crosstab vs (c+), A0-A3 all 100] " + ", ".join(f"judge {k[0]} / canonical {k[1]}: {v}" for k, v in sorted(xc.items())))
RES["crosstab"]["a03_cplus"] = {f"{k[0]}|{k[1]}": v for k, v in xc.items()}
RES["crosstab"]["cat_counts"] = dict(judgeDIFF_noC=len(catA), judgeDIFF_identical=len(catA_id), judgeDIFF_no_cplus=len(catA_nocplus), judgeSAME_C=len(catB_c), judgeSAME_evidence_only=len(catB_ev), judgeSAME_weak=len(catB_wk))
tab("\nAgainst (c+) (adds history / earlier-charge evidence) on A0-A3:\n"); tab("| judge \\ canonical | conflict (c+) | residual | identical |"); tab("|---|---|---|---|")
for j in ("DIFFERENT", "SAME"): tab(f"| {j} | {xc[(j,'conflict (c+)')]} | {xc[(j,'residual')]} | {xc[(j,'identical')]} |")
tab("")
tab("## Examples\n")
ex_block("Category 1: judge DIFFERENT, canonical facts do not conflict under (c) (presentation / emphasis / omission differences)", catA)
tab("### Category 2: judge SAME, canonical facts conflict or are unstated\n")
tab(f"Strictly under (c) there are {len(catB_c)} such pairs in the cache. To give examples anyway the list is filled in this order: (c) conflicts ({len(catB_c)}), history-evidence conflicts ({len(catB_ev)}), a core field unstated or the bank amount stated in only one note ({len(catB_wk)}); each example says which.\n")
for k, d in enumerate(catB[:8]):
    tab(f"**{k+1}. {pdesc(d)}; why listed: {why2(d)}**\n"); tab(f"> A ({d['type'].split('-')[0]}): {d['a']}\n>\n> B ({d['type'].split('-')[1]}): {d['b']}\n")
# hand inspection of judge-SAME & conflict
say("\n[hand inspection] judge SAME but canonical conflict / unstated (up to 10, ordered (c), evidence, weak):")
HAND = catB[:10]
for k, d in enumerate(HAND):
    say(f"--- #{k+1} {pdesc(d)} | why: {why2(d)}"); say(f"    A: {d['a']}"); say(f"    B: {d['b']}")
    for f in sorted(d["conf"] | d["cev"] | d["asym"] | d["ind"]): say(f"    field {f}: A={(CN[d['i'],d['ka']].get(f) if f in EV else fv(CN[d['i'],d['ka']],f))!r} B={(CN[d['i'],d['kb']].get(f) if f in EV else fv(CN[d['i'],d['kb']],f))!r}")
say("\n[judge-DIFFERENT A0-A3 pairs, raw 16: verdict chain, canonical status, history mention A/B]")
for d in sorted([d for d in A03 if d["strict"] not in ("SAME", "ident")], key=lambda d: d["i"]):
    say(f"   prompt {d['i']:3d} ({d['scn']:6s}) chain {d['chain']} cons {d['cons']}; canonical {cat(d)}, (c+) {d['cplus']}, one-sided {sorted(d['asym'] | d['ind'])}, hist {CN[d['i'],0]['history']}/{CN[d['i'],3]['history']}, ev conflict {sorted(d['cev'])}, ev one-sided {sorted(d['aev'])}")
RES["judge_diff_chains"] = {str(d["i"]): dict(chain=d["chain"], cons=d["cons"], canon=cat(d)) for d in A03 if d["strict"] not in ("SAME", "ident")}
RES["hand_inspection_n"] = len(HAND)
MANUAL = json.load(open(HERE / "manual_classification.json")) if (HERE / "manual_classification.json").exists() else {}
if MANUAL:
    say("[hand inspection] classification (written after reading the pairs above, manual_classification.json):")
    cc = collections.Counter(v["class"] for v in MANUAL.values())
    for key, v in MANUAL.items(): say(f"   {key}: {v['class']} - {v['why']}")
    say(f"   counts: {dict(cc)}"); RES["hand_classes"] = dict(cc)
    tab("### Hand classification of the judge-SAME / canonical-conflict pairs\n"); tab("| pair | class | why |"); tab("|---|---|---|")
    for key, v in MANUAL.items(): tab(f"| {key} | {v['class']} | {v['why']} |")
    tab(f"\nCounts: {dict(cc)}\n")
save()

# ------------------------------------------------------------------ 4. decomposition
say("\n=== 4. where do the differences come from")
BASE = [d for d in pairs if d["type"] in PRIM]
tab("## Table 4a. By scenario (canonical conflict rate over the 4 primary pair types; judge columns are A0-A3 only)\n")
tab("| scenario | prompts (lab./floor) | pairs | (c) conflict | (d) core conflict | residual (no conflict, facts differ) | judge raw DIFFERENT (A0-A3) | judge confirmed (A0-A3) |"); tab("|---|---|---|---|---|---|---|---|")
for s in ("normal", "noinv", "dup", "first", "spike"):
    ds = [d for d in BASE if d["scn"] == s]; a3 = [d for d in A03 if d["scn"] == s]
    tab(f"| {s} | {len({d['i'] for d in ds if d['i'] in LAB})}/{len(a3)} | {len(ds)} | {pc(np.mean([d['c'] for d in ds]))} | {pc(np.mean([d['d'] for d in ds]))} | {pc(np.mean([(not d['c']) and (not d['ident']) for d in ds]))} | {sum(d['strict'] not in ('SAME','ident') for d in a3)}/{len(a3)} | {sum(d['cons']!='SAME' for d in a3)}/{len(a3)} |")
    say(f"[by scn] {s:7s} pairs {len(ds):3d} conflict(c) {pc(np.mean([d['c'] for d in ds]))} core {pc(np.mean([d['d'] for d in ds]))} residual {pc(np.mean([(not d['c']) and (not d['ident']) for d in ds]))} | A0-A3 n={len(a3)} judge raw {sum(d['strict'] not in ('SAME','ident') for d in a3)} conf {sum(d['cons']!='SAME' for d in a3)}")
tab("\n## Table 4b. By whether the notes mention the vendor history\n")
tab("| history mentioned in | pairs | (c) conflict | residual | identical | A0-A3 pairs | judge raw DIFFERENT (A0-A3) |"); tab("|---|---|---|---|---|---|---|")
for h in ("neither", "one", "both"):
    ds = [d for d in BASE if d["hist_cat"] == h]; a3 = [d for d in A03 if d["hist_cat"] == h]
    tab(f"| {h} note(s) | {len(ds)} | {pc(np.mean([d['c'] for d in ds]) if ds else None)} | {pc(np.mean([(not d['c']) and (not d['ident']) for d in ds]) if ds else None)} | {pc(np.mean([d['ident'] for d in ds]) if ds else None)} | {len(a3)} | {sum(d['strict'] not in ('SAME','ident') for d in a3)}/{len(a3)} |")
    say(f"[by history] {h:8s} pairs {len(ds):3d} conflict {pc(np.mean([d['c'] for d in ds]) if ds else None)} identical {pc(np.mean([d['ident'] for d in ds]) if ds else None)} | A0-A3 n={len(a3)} judge raw DIFF {sum(d['strict'] not in ('SAME','ident') for d in a3)}")
# history-mismatch inside A0-A3 x judge
hm = collections.Counter((d["hist_cat"] == "one", d["strict"] not in ("SAME", "ident")) for d in A03)
say(f"[history asymmetry x judge, A0-A3] one-note-only-mentions-history: judge DIFF {hm[(True,True)]}/{hm[(True,True)]+hm[(True,False)]}; otherwise judge DIFF {hm[(False,True)]}/{hm[(False,True)]+hm[(False,False)]}")
RES["hist_asym_a03"] = {str(k): v for k, v in hm.items()}
# by field
tab("\n## Table 4c. Which canonical field conflicts, and which fields are stated in only one note\n")
tab("Share of the 4 primary pair types (%d pairs). 'conflict' = stated in both with different values; 'one-sided' = stated in one note only (optional detail); 'unstated' = required field (flag, amount, invoice, reason, date) not extractable in one note.\n" % len(BASE))
tab("| field | conflict | one-sided / unstated |"); tab("|---|---|---|")
fc = {}
for f in ALLF:
    c1 = sum(f in d["conf"] for d in BASE); c2 = sum(f in d["asym"] or f in d["ind"] for d in BASE); fc[f] = (c1, c2)
    tab(f"| {f} | {c1} ({pc(c1/len(BASE))}) | {c2} ({pc(c2/len(BASE))}) |"); say(f"[by field] {f:9s} conflict {c1:3d} ({pc(c1/len(BASE))})  one-sided/unstated {c2:3d} ({pc(c2/len(BASE))})")
hdiff = sum(CN[d['i'], d['ka']]['history'] != CN[d['i'], d['kb']]['history'] for d in BASE)
tab(f"| history mentioned in exactly one note | n/a | {hdiff} ({pc(hdiff/len(BASE))}) |"); say(f"[by field] history mentioned in exactly one note: {hdiff} ({pc(hdiff/len(BASE))})")
RES["by_field"] = {f: dict(conflict=v[0], one_sided=v[1]) for f, v in fc.items()}; RES["by_field"]["history_one_note"] = hdiff
bm = [d for d in BASE if P[d["i"]]["bank_mismatch"]]; bmo = [d for d in bm if (CN[d["i"], d["ka"]]["bank_amt"] is None) != (CN[d["i"], d["kb"]]["bank_amt"] is None)]
say(f"[by field] pairs on bank-mismatch prompts {len(bm)}; real bank mismatch stated in exactly one of the two notes (a substantive one-sided omission, not a conflict): {len(bmo)}"); RES["bank_mismatch_one_sided"] = dict(pairs=len(bm), one_sided=len(bmo))
conf_pairs = [d for d in BASE if d["c"]]; say(f"[by field] (c) conflicts after the follow-up fix: {len(conf_pairs)} pairs " + str([(d["i"], d["type"], sorted(d["conf"])) for d in conf_pairs]))
fcnt = collections.Counter(f for d in BASE for f in d["conf"]); say(f"[by field] conflicts per field among conflicting pairs: {dict(fcnt)}")
# clustering by prompt. Pairs of one prompt share notes (one odd note makes 2 of the 3 pairs differ), so the test is done on NOTE properties: do the 3 notes of a prompt agree on a
# property more often than independent notes with the same (scenario-specific) rate would?
NF = {"history mentioned": lambda c: c["history"], "bank amount stated": lambda c: c["bank_amt"] is not None, "history range stated": lambda c: c["hist_lo"] is not None,
      "history count stated": lambda c: c["hist_count"] is not None, "earlier-charge date stated": lambda c: c["other_dates"] is not None}
tab("\n## Table 4d. Do the 3 notes of a prompt share their optional details more often than chance? (140 labelled prompts, notes A0-A2)\n")
tab("Per property of a note: share of prompts where all 3 notes agree, observed vs expected if each note drew the property independently with the scenario's own rate. Expected is the mean over prompts of p^3 + (1-p)^3.\n")
tab("| note property | rate | all 3 notes agree (observed) | expected if independent, global rate | expected if independent, per-scenario rate |"); tab("|---|---|---|---|---|")
RES["cluster"] = {}
for nm, f in NF.items():
    v = {i: [bool(f(CN[i, r])) for r in range(3)] for i in LAB}
    pg = np.mean([x for i in LAB for x in v[i]]); ps = {s: np.mean([x for i in LAB if P[i]["scn"] == s for x in v[i]]) for s in ("normal", "noinv", "dup", "first", "spike")}
    obs = np.mean([len(set(v[i])) == 1 for i in LAB]); eg = pg ** 3 + (1 - pg) ** 3; es = np.mean([ps[P[i]["scn"]] ** 3 + (1 - ps[P[i]["scn"]]) ** 3 for i in LAB])
    RES["cluster"][nm] = dict(rate=float(pg), observed=float(obs), expected_global=float(eg), expected_scn=float(es))
    say(f"[cluster, note property] {nm:28s} rate {pc(pg)}: prompts where all 3 notes agree observed {pc(obs)} vs independent-global {pc(eg)} vs independent-per-scenario {pc(es)}")
    tab(f"| {nm} | {pc(pg)} | {pc(obs)} | {pc(eg)} | {pc(es)} |")
cp_by_prompt = collections.Counter(sum(d["cplus"] for d in BASE if d["i"] == i and d["type"] in ("A0-A1", "A0-A2", "A1-A2")) for i in LAB)
sc_cp = collections.Counter(P[i]["scn"] for i in LAB if any(d["cplus"] for d in BASE if d["i"] == i and d["type"] in ("A0-A1", "A0-A2", "A1-A2")))
say(f"[cluster] prompts by number of their 3 pairs with a (c+) conflict (0/1/2/3): {[cp_by_prompt[k] for k in range(4)]}; prompts with any (c+) conflict by scenario: {dict(sc_cp)} (labelled prompts per scenario: {dict(collections.Counter(P[i]['scn'] for i in LAB))})")
RES["cluster"]["cplus_pairs_per_prompt"] = [cp_by_prompt[k] for k in range(4)]; RES["cluster"]["cplus_prompts_by_scn"] = dict(sc_cp)
tab(f"\nPrompts by number of their 3 pairs with a (c+) conflict (0/1/2/3): {[cp_by_prompt[k] for k in range(4)]}; prompts with any (c+) conflict by scenario: {dict(sc_cp)}.\n")
# prompt-level predictor
def feats(i):
    p = P[i]; return [p["scn"] == s for s in ("noinv", "dup", "first", "spike")] + [p["spike"], p["has_inv"], p["bank_mismatch"], len(p["hist"]), p["flag"].startswith("Anomaly flag: yes")]
FN = ["scn=noinv", "scn=dup", "scn=first", "scn=spike", "spike", "has_inv", "bank_mismatch", "n_hist", "flagged"]
def prompt_auc(ids, y, label, reps=20):
    X = np.array([feats(i) for i in ids], float); y = np.array(y, int)
    if y.sum() < 5 or y.sum() > len(y) - 5: say(f"[AUC {label}] too few positives ({y.sum()}/{len(y)}), skipped"); return None
    aucs = []; oof = np.zeros(len(y))
    for r in range(reps):
        sk = StratifiedKFold(5, shuffle=True, random_state=r); pr = np.zeros(len(y))
        for tr, te in sk.split(X, y):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X[tr], y[tr]); pr[te] = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y, pr)); oof += pr / reps
    rb = np.random.default_rng(0); bs = []
    for _ in range(1000):
        s = rb.integers(0, len(y), len(y))
        if 0 < y[s].sum() < len(y): bs.append(roc_auc_score(y[s], oof[s]))
    ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    uni = {FN[j]: roc_auc_score(y, X[:, j]) for j in range(X.shape[1]) if X[:, j].std() > 0}
    say(f"[AUC {label}] n={len(y)} positives {y.sum()}: 5-fold (stratified, {reps} repeats, by prompt; one row per prompt) logistic-regression AUC {np.mean(aucs):.3f} (sd over CV repeats {np.std(aucs):.3f}; bootstrap-over-prompts 95% CI {ci[0]:.2f}-{ci[1]:.2f}); univariate AUC " + ", ".join(f"{k} {v:.2f}" for k, v in uni.items()))
    return dict(n=len(y), pos=int(y.sum()), auc=float(np.mean(aucs)), sd=float(np.std(aucs)), ci=ci, uni=uni)
def anyp(i, key, types=("A0-A1", "A0-A2", "A1-A2")): return int(any(key(d) for d in BASE if d["i"] == i and d["type"] in types))
AUC = {}
AUC["any_conflict_c"] = prompt_auc(LAB, [anyp(i, lambda d: d["c"]) for i in LAB], "target: any canonical conflict among the prompt's 3 pairs, 140 prompts")
AUC["any_cplus"] = prompt_auc(LAB, [anyp(i, lambda d: d["cplus"]) for i in LAB], "target: any (c+) conflict incl. history evidence, 140 prompts")
AUC["any_conflict_core"] = prompt_auc(LAB, [anyp(i, lambda d: d["d"]) for i in LAB], "target: any core-field conflict, 140 prompts")
AUC["any_nonident"] = prompt_auc(LAB, [anyp(i, lambda d: not d["ident"]) for i in LAB], "target: any non-identical facts, 140 prompts")
AUC["judge_a0a3"] = prompt_auc(FLOOR, [int(d["strict"] not in ("SAME", "ident")) for d in sorted(A03, key=lambda d: d["i"])], "target: strict judge raw DIFFERENT on A0-A3, 100 floor prompts")
AUC["judge_cons_a0a3"] = prompt_auc(FLOOR, [int(d["cons"] != "SAME") for d in sorted(A03, key=lambda d: d["i"])], "target: strict judge confirmed DIFFERENT on A0-A3, 100 floor prompts")
# does another pair of the same prompt predict the judge's A0-A3 verdict?  Reference pairs (A0-A1, A0-A2, A1-A2) exist only for the labelled prompts 200-259, so the score is
# defined on those 60 prompts only (prompts 260-299 have no reference pairs; an earlier version scored them 0 by mistake).
FL60 = [i for i in FLOOR if i in LAB]; a3by = {d["i"]: d for d in A03}
def rep_auc(ids, y, s, label):
    y = np.array(y); s = np.array(s)
    if len(set(y)) < 2: say(f"[AUC reproducibility {label}] one class only"); return None
    a = float(roc_auc_score(y, s)); say(f"[AUC reproducibility {label}] n={len(y)} positives {y.sum()}: AUC {a:.3f}; mean score judge-DIFF {s[y==1].mean():.2f} vs SAME {s[y==0].mean():.2f}"); return a
REPRO = {}
for nm, key in (("n_c", lambda i: sum(d["c"] for d in BASE if d["i"] == i and d["type"] in ("A0-A1", "A0-A2", "A1-A2"))),
                ("n_cplus", lambda i: sum(d["cplus"] for d in BASE if d["i"] == i and d["type"] in ("A0-A1", "A0-A2", "A1-A2"))),
                ("n_nonident", lambda i: sum(not d["ident"] for d in BASE if d["i"] == i and d["type"] in ("A0-A1", "A0-A2", "A1-A2")))):
    for tg, ids in (("strict raw (FAIL=DIFFERENT)", FL60), ("strict raw (FAIL prompts dropped)", [i for i in FL60 if a3by[i]["strict"] != "FAIL"])):
        y = [int(a3by[i]["strict"] != "SAME" and a3by[i]["strict"] != "ident") for i in ids]
        REPRO[f"{nm}|{tg}"] = rep_auc(ids, y, [key(i) for i in ids], f"{nm} -> {tg}")
AUC["repro_60"] = REPRO
# model-based AUC with the cut-off (FAIL) prompts dropped
fids = [i for i in FLOOR if a3by[i]["strict"] != "FAIL"]
AUC["judge_a0a3_noFAIL"] = prompt_auc(fids, [int(a3by[i]["strict"] == "DIFFERENT") for i in fids], "target: strict judge raw DIFFERENT on A0-A3, FAIL prompts dropped")
RES["prompt_auc"] = AUC
tab("\n## Table 4e. Can prompt-level fields predict 'this prompt's notes will disagree'? (AUC, 5-fold CV by prompt, logistic regression on scenario, spike, has_inv, bank_mismatch, history length, flagged)\n")
tab("| target | prompts | positives | AUC (5-fold, 20 repeats; bootstrap 95% CI over prompts) | strongest single feature (AUC, in-sample) |"); tab("|---|---|---|---|---|")
for k, nm in (("any_conflict_c", "any canonical conflict among 3 pairs"), ("any_cplus", "any (c+) conflict (history evidence included)"), ("any_conflict_core", "any core-field conflict among 3 pairs"), ("any_nonident", "any non-identical facts among 3 pairs"), ("judge_a0a3", "strict judge raw DIFFERENT (A0-A3)"), ("judge_cons_a0a3", "strict judge confirmed DIFFERENT (A0-A3)")):
    r = AUC[k]
    if r is None: tab(f"| {nm} | | | too few positives | |"); continue
    best = max(r["uni"].items(), key=lambda kv: abs(kv[1] - .5)); tab(f"| {nm} | {r['n']} | {r['pos']} | {r['auc']:.3f} (CI {r['ci'][0]:.2f}-{r['ci'][1]:.2f}) | {best[0]} {best[1]:.2f} |")
save()

# ------------------------------------------------------------------ 4b. what the new equivalence would do to the S16 ablation labels (variant notes are in labels.json, no cache lookup needed)
say("\n=== 4b. S16 ablation labels re-derived with canonical facts (cached variant notes vs A0)")
REQ = {"flag", "tx_amt", "inv", "reason", "tx_date"}
def core_change(i, name):
    v = lab_json[str(i)]["var"][name]; cb = C.canon(v["note"], P[i]["tx"]); conf_, ind_, asym_, ident_, cev_, cevt_, aev_ = compare(CN[i, 0], cb)
    return bool((conf_ - {"followup"}) or (ind_ & REQ)), bool(cevt_ or aev_), bool(conf_ - {"followup"}), cb
names = ["tx", "inv", "flag", "bank", "H0", "B0", "B1", "B2"]
tab("\n## Table 6. S16 ablation labels: strict confirmed verdict (as used for the span labels) vs canonical facts\n")
tab("core change = a core field conflicts, or a required field (flag, amount, invoice, reason, date) stated in A0 is not stated in the variant or vice versa; evidence change = history count/range/top/earlier-charge date differs or is stated in one note only.\n")
tab("| ablation | prompts | strict confirmed DIFFERENT | canonical core change | both | strict only | canonical only | evidence change |"); tab("|---|---|---|---|---|---|---|---|")
RES["relabel"] = {}
for nm in names:
    ids = [i for i in LAB if nm in lab_json[str(i)]["var"]]
    if not ids: continue
    st = np.array([isdiff(lab_json[str(i)]["var"][nm]["cons"]) for i in ids]); cr = [core_change(i, nm) for i in ids]; cc = np.array([x[0] for x in cr]); ev = np.array([x[1] for x in cr])
    RES["relabel"][nm] = dict(n=len(ids), strict=int(st.sum()), canon=int(cc.sum()), both=int((st & cc).sum()), strict_only=int((st & ~cc).sum()), canon_only=int((~st & cc).sum()), evidence=int(ev.sum()))
    say(f"[relabel] {nm:5s} n={len(ids):3d} strict-confirmed DIFFERENT {st.sum():3d} ({pc(st.mean())}); canonical core change {cc.sum():3d} ({pc(cc.mean())}); both {int((st&cc).sum())}, strict only {int((st&~cc).sum())}, canonical only {int((~st&cc).sum())}; evidence change {ev.sum()}")
    tab(f"| {nm} | {len(ids)} | {st.sum()} ({pc(st.mean())}) | {cc.sum()} ({pc(cc.mean())}) | {int((st&cc).sum())} | {int((st&~cc).sum())} | {int((~st&cc).sum())} | {ev.sum()} |")
h0s = [i for i in LAB if isdiff(lab_json[str(i)]["var"]["H0"]["cons"])]; h0c = [i for i in LAB if core_change(i, "H0")[0]]
bysc = {s: (sum(P[i]["scn"] == s for i in h0s), sum(P[i]["scn"] == s for i in h0c), sum(P[i]["scn"] == s for i in LAB)) for s in ("normal", "noinv", "dup", "first", "spike")}
say(f"[relabel] dropping the whole history: strict labels it needed on {len(h0s)}/140 prompts, canonical core facts change on {len(h0c)}/140; by scenario (strict/canonical/prompts): {bysc}")
RES["relabel"]["H0_by_scn"] = bysc
tab(f"\nDropping the whole history changes the note's strict verdict on {len(h0s)}/140 prompts but its canonical core facts on {len(h0c)}/140. By scenario (strict / canonical / prompts): {bysc}.\n")
save()

# ------------------------------------------------------------------ 5. typed floor and calibration n
say("\n=== 5. floors reachable")
tab("\n## Table 5. Floors by route (DIFFERENT rate between two independent teacher outputs)\n")
tab("| route | pair set | k/n | rate | UB95 | UB@300 | UB@1000 | n for 5% | n for 10% |"); tab("|---|---|---|---|---|---|---|---|---|")
rows5 = []
def r5(label, ds, fn, note_=""):
    k, n = sum(fn(d) for d in ds), len(ds); s = stats_row(k, n); rows5.append((label, s))
    tab(f"| {label} | {note_ or ds[0]['type'] if len({d['type'] for d in ds})==1 else note_} | {k}/{n} | {pc(s['rate'])} | {pc(s['ub'])} | {pc(s['ub_at'][300])} | {pc(s['ub_at'][1000])} | {s['n5'] or 'never'} | {s['n10'] or 'never'} |")
    say(f"[floor route] {label:62s} {note_ or ''} {k}/{n} = {pc(s['rate'])} UB95 {pc(s['ub'])}  n5 {s['n5']} n10 {s['n10']}")
    return s
r5("(i) judge only: strict raw", A03, lambda d: d["strict"] not in ("SAME", "ident"), "A0-A3 (n=100)")
r5("(i) judge only: strict raw, cut-off (FAIL) answers not counted", A03, lambda d: d["strict"] == "DIFFERENT", "A0-A3 (n=100)")
r5("(i) judge only: strict confirmed, needs >=1 real DIFFERENT vote", A03, lambda d: d["cons"] != "SAME" and "DIFFERENT" in d["chain"], "A0-A3 (n=100)")
r5("(i) judge only: strict confirmed", A03, lambda d: d["cons"] != "SAME", "A0-A3 (n=100)")
A3s = A03
r5("(ii) canonical conflict (c), residual counted SAME (lower end)", A3s, lambda d: d["c"], "A0-A3 (n=100)")
r5("(ii) canonical conflict incl. history evidence, numbers within 5% equal (c+tol)", A3s, lambda d: d["cplus_tol"], "A0-A3 (n=100)")
r5("(ii) canonical conflict incl. history evidence, exact numbers (c+)", A3s, lambda d: d["cplus"], "A0-A3 (n=100)")
r5("(ii) core-field conflict only", A3s, lambda d: d["d"], "A0-A3 (n=100)")
POOL3 = [d for d in pairs if d["type"] in ("A0-A1", "A0-A2", "A1-A2")]
r5("(ii) canonical conflict, all fields", POOL3, lambda d: d["c"], "3 pair types, 420 pairs (not independent)")
r5("(iii) typed: conflict on flag, reason, invoice outcome, tx amount", A3s, lambda d: d["typed"], "A0-A3 (n=100)")
for t in ("A0-A1", "A0-A2", "A1-A2"): r5("(iii) typed: conflict on flag, reason, invoice outcome, tx amount", sel(t), lambda d: d["typed"], f"{t} (n=140)")
r5("(iii) typed, counting unstated required fields as disagreement", A3s, lambda d: d["typed_ind"], "A0-A3 (n=100)")
r5("(iii) typed, 3 pair types pooled", POOL3, lambda d: d["typed"], "420 pairs (not independent)")
# typed floor expressed vs ground truth
nt = [all(CN[k][f] == GT[k[0]][g] for f, g in (("flag", "flag"), ("reason", "reason"), ("inv_id", "inv_id"))) and close(CN[k]["tx_amt"], GT[k[0]]["tx_amt"]) for k in allnotes]
say(f"[typed vs ground truth] a note's typed fields (flag, reason, invoice outcome, tx amount) all equal the record's ground truth in {sum(nt)}/{len(nt)} = {pc(np.mean(nt))} of notes")
RES["floor_routes"] = {l: s for l, s in rows5}; RES["typed_all_right_vs_gt"] = float(np.mean(nt))
tab(f"\nTyped fields all equal to the record's ground truth in {sum(nt)}/{len(nt)} notes ({pc(np.mean(nt))}).\n")
# typed-conflict pairs by hand: what are they
tc = [d for d in pairs if d["typed"]]
say(f"[typed conflicts] {len(tc)} pairs: " + "; ".join(f"{d['type']} p{d['i']} {sorted(d['conf'] & set(TYPED))}" for d in tc[:12]))
save()
say(f"\n[cache] total cache hits {cacheonly.HITS[0]}; required keys missing: notes {dict(miss_notes)}; strict A0-A3 judge calls 0 missing; optional probes missing (strict on A0-A1/A0-A2/A1-A2: "
    f"{sum(miss_j[(t,'strict')] for t in ('A0-A1','A0-A2','A1-A2'))}, core on A0-A3: {miss_j[('A0-A3','core')]}, swap on A0-A3: {miss_j[('A0-A3','swap')]}, mini on A0-A3: {miss_j[('A0-A3','mini')]})")
(HERE / "tables.md").write_text("# S16c tables (generated by run.py)\n\n" + "\n".join(TAB) + "\n", encoding="utf-8")
save()
