"""S16b v2 equivalence for draft_note (helper for run_b2.py; nothing here edits core.py / llm.py / run_b.py).
equiv(p, ref, x, evidence) -> DIFFERENT / SAME / FAIL / None
  1. canonical conflict, or a lost required field relative to the reference  -> DIFFERENT (no judge)
  2. canonical facts identical under the core fields (+ evidence policy)       -> SAME (no judge)
  3. else the residual judge in BOTH presentation orders (judge1, pinned as in core.py); equal answers decide; any other combination
     (disagreement or an unparsed answer) goes to the second-family judge (second-family judge) as adjudicator; adjudicator unparsed -> FAIL.
FAIL is never counted as DIFFERENT: callers exclude it from the rate and report the count.
evidence policy: core  = history evidence (counts, ranges, earlier-charge dates, whether history is mentioned) is NOT a core fact.
                 loose = evidence must match for the canonical SAME shortcut; pairs that fail it go to the residual judge.
                 The residual judge rubric is JSYS_CORE under BOTH policies (S16c Verdict: only the evidence policy changes), reported as RUBRIC next to every rate.
Canonicalizer fixes (canon2/canon_decide): posting dates are not transaction-date candidates; a negative or bank-adjacent amount counts as a stated transaction amount when it equals the
reference value; an invoice amount is ignored when the outcome is 'no invoice' or the invoice ids differ. Typed fields are normalised (categories, yes/no, abs amounts).
Cache: judge calls now use max_tokens 200; verdicts already in responses.jsonl under the old key (max_tokens 40/30) are reused when they parse, never when cut off.
`STATE['net']` False = cache-only: a missing call raises Miss (callers count it). setup(dry) must run before core is imported."""
import sys, re, json, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent; S16C = HERE.parent / "s16c_judge_floor_offline"
class Miss(KeyError): pass
STATE = dict(net=False, dry=True, capped=False); MISSES = (Miss,); MT = 200; RUBRIC = 'core (JSYS_CORE)'
class CapHit(Miss): pass
llm = core = C = None; EVMEMO = {}

def setup(dry):
    global llm, core, C, MISSES
    sys.path.insert(0, str(S16C))
    if dry:
        import cacheonly; cacheonly.install(); MISSES = (Miss, cacheonly.Miss)
    else:
        sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
    import llm as _l, core as _c, canon as _k
    llm, core, C = _l, _c, _k; STATE.update(dry=dry, net=not dry)

# ------------------------------------------------------------------ cache-aware judge call
def ckey(model, system, user, mt, extra, rep):
    ex = dict(extra); ex["usage"] = {"include": True}
    return hashlib.sha1(json.dumps([model, system, user, mt, ex, rep], sort_keys=True).encode()).hexdigest()
def chat_safe(*a):
    """llm.chat, but the spend-cap RuntimeError becomes CapHit (a Miss: the cell is simply unanswerable) and sets STATE['capped']"""
    try: return llm.chat(*a)
    except RuntimeError as e:
        if "spend cap" in str(e): STATE["capped"] = True; raise CapHit(str(e))
        raise
def cached(model, system, user, mt, extra, rep):
    r = llm.R.get(ckey(model, system, user, mt, extra, rep))
    return None if r is None else (r["resp"]["choices"][0]["message"].get("content") or "")
def parse(text):
    """first SAME / DIFFERENT word of the judge output (None if neither appears)"""
    m = re.search(r"\b(SAME|DIFFERENT)\b", (text or "").upper()); return m.group(1) if m else None
def note_cached(p, k, rep=0):
    return ckey(core.HAIKU, core.SYS_T, core.render(p, k), 200, core.EX_T, rep) in llm.R
def note_c(p, k, rep=0):
    """core.note, or None when the note is not cached and the network is off"""
    if not STATE["net"] and not note_cached(p, k, rep): return None
    if STATE["capped"] and not note_cached(p, k, rep): return None
    try: return core.note(p, k, rep)
    except RuntimeError as e:
        if "spend cap" in str(e): STATE["capped"] = True; return None
        raise

def J2(a, b, system, model=None, swap=False, rep=0):
    """-> (SAME | DIFFERENT | FAIL, source in ident/cache/legacy/new/fail). Raises Miss when a call is needed and the network is off."""
    model = model or core.SONNET
    if a == b: return "SAME", "ident"
    if swap: a, b = b, a
    u = f"Note 1:\n{a}\n\nNote 2:\n{b}" + core.JTAIL
    ex, leg = (core.EX_S, (40,)) if model == core.SONNET else (core.EX_M, (30,))
    for t in range(3):
        r_ = rep + 10 * t; txt = cached(model, system, u, MT, ex, r_)
        if txt is not None:
            w = parse(txt)
            if w: return w, "cache"
            continue                                     # a stored 200-token answer without a verdict: next retry slot
        for m in leg:                                    # old key (max_tokens 40/30): reuse only a verdict that parsed (a cut-off answer is not evidence)
            w = parse(cached(model, system, u, m, ex, r_))
            if w: return w, "legacy"
        if not STATE["net"]: raise Miss(("judge", model))
        w = parse(chat_safe(model, system, u, MT, ex, r_)["text"])
        if w: return w, "new"
    return "FAIL", "fail"

# ------------------------------------------------------------------ canonical comparison (same logic as s16c run.py compare/core_change)
def canon2(text, tx_line):
    """C.canon plus: posting dates removed from the transaction-date candidates; all amounts kept (c['amts']); invoice amount dropped when the outcome is 'no invoice'."""
    c = dict(C.canon(text, tx_line)); rem = list(c["dates"])
    for m in re.finditer(r"(?:posted|pending|bank|feed)[^0-9$]{0,25}(\d{4}-\d\d-\d\d)", text, re.I):
        if m.group(1) in rem: rem.remove(m.group(1))
    c["tx_date"] = rem[0] if rem else None
    c["other_dates"] = frozenset(d for d in rem if d != c["tx_date"] and d != c["inv_date"]) or None
    c["amts"] = [round(C.money(m.group(2)), 2) for m in C.AMT.finditer(text)]
    if c["inv_id"] == "NONE": c["inv_amt"] = None
    return c
REQUIRED = ("flag", "tx_amt", "inv", "inv_amt", "reason", "tx_date")
ALLF = ["vendor", "tx_amt", "tx_date", "inv", "inv_amt", "bank_amt", "flag", "reason", "followup"]
EV = ["hist_count", "hist_lo", "hist_hi", "hist_top", "other_dates"]
def fv(c, f):
    return {"vendor": True if c["vendor_ok"] else None, "tx_amt": c["tx_amt"], "tx_date": c["tx_date"], "inv": c["inv_id"], "inv_amt": c["inv_amt"], "bank_amt": c["bank_amt"],
            "flag": c["flag"], "reason": c["reason"] if c["reason"] != ("other",) else None, "followup": c["followup"]}[f]
def same_v(x, y): return abs(x - y) < 0.005 if isinstance(x, float) and isinstance(y, float) else x == y
def compare(ca, cb):
    conf, one = set(), set()
    for f in ALLF:
        x, y = fv(ca, f), fv(cb, f)
        if x is None and y is None: continue
        if x is None or y is None: one.add(f); continue
        if f == "inv_amt" and fv(ca, "inv") != fv(cb, "inv"): continue
        if not same_v(x, y): conf.add(f)
    cevt, aev = False, False
    for f in EV:
        x, y = ca[f], cb[f]
        if x is None and y is None: continue
        if x is None or y is None: aev = True; continue
        if isinstance(x, float) and abs(x - y) > max(1.0, 0.05 * max(x, y)) or not isinstance(x, float) and x != y: cevt = True
    return conf - {"followup"}, one - {"followup"}, cevt, aev, ca["history"] == cb["history"]
def canon_decide(ca, cb, evidence):
    """-> (verdict or 'JUDGE', path). ca = reference facts, cb = candidate facts."""
    ca, cb = dict(ca), dict(cb)
    for a_, b_ in ((ca, cb), (cb, ca)):            # a transaction amount is stated if its value appears among ANY of the other note's amounts (sign / bank-adjacent wording)
        if a_["tx_amt"] is not None and any(abs(a_["tx_amt"] - v) < 0.005 for v in b_.get("amts", ())):
            was_none = b_["tx_amt"] is None; b_["tx_amt"] = a_["tx_amt"]
            if was_none and b_["bank_amt"] is not None and abs(b_["bank_amt"] - a_["tx_amt"]) < 0.005: b_["bank_amt"] = None   # the 'bank' amount was the transaction amount itself
    conf, one, cevt, aev, hsame = compare(ca, cb)
    if conf: return "DIFFERENT", "canon-conflict"
    def lost(f):
        if fv(ca, f) is None or fv(cb, f) is not None: return False
        if f == "inv_amt" and (fv(ca, "inv") == "NONE" or fv(ca, "inv") != fv(cb, "inv")): return False
        return True
    ll = [f for f in REQUIRED if lost(f)]
    if ll: return "DIFFERENT", "canon-lost:" + ",".join(ll)
    ev_ok = True if evidence == "core" else (not cevt and not aev and hsame)
    if not one and ev_ok: return "SAME", "canon-identical"
    return "JUDGE", "judge-onesided" if one else "judge-evidence"

ADJ_P = 0.13       # share of pairs whose two presentation orders disagree (S16 order-swap agreement 86.7%), used only for dry-run cost expectations
def equiv(p, ref, x, evidence="core"):
    """-> dict(v, path, orders, adj, src, need=dict(judge1, judge2, adj_exp)). v None = not answerable from the cache (net off)."""
    key = (ref, x, evidence, p["id"])
    if key in EVMEMO and (EVMEMO[key]["v"] is not None or not STATE["net"]): return EVMEMO[key]
    out = dict(v=None, path=None, orders=None, adj=None, src=[], need=dict(judge1=0, judge2=0, adj_exp=0.0))
    if ref == x: out.update(v="SAME", path="exact-text")
    else:
        d, path = canon_decide(canon2(ref, p["tx"]), canon2(x, p["tx"]), evidence); out["path"] = path
        if d != "JUDGE": out["v"] = d
        else:
            sysm = core.JSYS_CORE; o = []
            for sw in (False, True):
                try: w, s = J2(ref, x, sysm, swap=sw); o.append(w); out["src"].append(s)
                except MISSES: o.append(None); out["need"]["judge1"] += 1
            out["orders"] = o
            if None in o: out["need"]["adj_exp"] = ADJ_P   # outcome unknown offline; adjudicator needed with probability ADJ_P
            elif o[0] == o[1] and o[0] in ("SAME", "DIFFERENT"): out["v"] = o[0]
            else:
                try: w, s = J2(ref, x, core.JSYS_CORE, model=core.MINI); out["adj"] = w; out["src"].append("adj-" + s); out["v"] = w
                except MISSES: out["need"]["judge2"] += 1
    if out["v"] is not None or STATE["net"]: EVMEMO[key] = out
    return out

def strict(refs, x):
    """the strict-rubric verdict (core.verdict logic: A0 raw, confirmed by the two other reference notes) with the fixed judge parse; FAIL is kept as FAIL.
    -> dict(raw, cons, need) ; None entries when not answerable offline"""
    need = 0
    def one(r):
        nonlocal need
        try: return J2(r, x, core.JSYS)[0]
        except MISSES: need += 1; return None
    j0 = one(refs[0])
    if j0 is None: return dict(raw=None, cons=None, need=need, exp=0.3)
    if j0 != "DIFFERENT": return dict(raw=j0, cons=j0, need=need, exp=0)
    j1 = one(refs[1])
    if j1 == "DIFFERENT": return dict(raw=j0, cons="DIFFERENT", need=need, exp=0)
    if j1 is None: return dict(raw=j0, cons=None, need=need, exp=0)
    j2 = one(refs[2])
    if j2 is None: return dict(raw=j0, cons=None, need=need, exp=0)
    cons = "DIFFERENT" if j2 == "DIFFERENT" else "SAME" if (j1 == "SAME" and j2 == "SAME") else "FAIL"
    return dict(raw=j0, cons=cons, need=need, exp=0)

# ------------------------------------------------------------------ typed arm
SYS_TY = ("Return one JSON object for the reviewer and nothing else (no prose, no code fence). Use only facts in the record. Keys: "
          '"flag" ("yes" or "no": the anomaly flag decision), '
          '"reasons" (list; each item one of "duplicate", "first purchase", "unusually high", "no invoice", "other"; empty list when not flagged), '
          '"invoice_outcome" ("matched" or "none"), "invoice_id" (string or null), "invoice_amount" (number or null), '
          '"tx_amount" (number), "tx_date" ("YYYY-MM-DD", the transaction date), '
          '"bank_mismatch" ("yes" only if a bank-feed line is present and its amount differs from the transaction amount, otherwise "no").')
TFIELDS = ("flag", "reasons", "invoice_outcome", "invoice_id", "invoice_amount", "tx_amount", "tx_date", "bank_mismatch")
def _num(v):
    try: return None if v is None else round(float(str(v).replace(",", "").replace("$", "")), 2)
    except Exception: return None
RCAT = [("duplicate", r"duplic|repeat|double"), ("first purchase", r"first|new vendor|never"), ("unusually high", r"high|large|unusual|spike|above|exceed"), ("no invoice", r"invoice|unmatch|missing")]
def norm_reasons(rs):
    out = set()
    for r in rs:
        r = str(r).strip().lower()
        if r in ("", "none", "null", "n/a"): continue
        hit = {c for c, rx in RCAT if re.search(rx, r)}
        out |= hit or {"other"}
    return tuple(sorted(out))
def norm_yn(v, default="no"):
    s = str(v).strip().lower()
    return "yes" if s in ("yes", "true", "y", "1", "flagged") else "no" if s in ("no", "false", "n", "0", "not flagged", "none", "") else s if s else default
def norm_outcome(v, iid):
    s = str(v or "").strip().lower()
    if not s: return "none" if iid is None else "matched"
    return "none" if re.search(r"\b(no|none|missing|unmatched|not|without|absent)\b", s) else "matched"
def parse_record(text):
    """normalised typed record: reasons mapped onto the 5 categories, flag / bank_mismatch yes/no, outcome matched/none, amounts absolute, date = first ISO date"""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m: return None
    try: d = json.loads(m.group(0))
    except Exception: return None
    if not isinstance(d, dict) or "flag" not in d or "tx_amount" not in d: return None
    rs = d.get("reasons", d.get("reason") or [])
    rs = [rs] if isinstance(rs, str) else list(rs or [])
    iid = d.get("invoice_id"); iid = None if iid in (None, "", "null", "None") else str(iid).strip()
    out = norm_outcome(d.get("invoice_outcome"), iid); flag = norm_yn(d["flag"])
    ia = _num(d.get("invoice_amount")); ta = _num(d["tx_amount"])
    if out == "none": iid = None; ia = None
    dt = re.search(r"\d{4}-\d\d-\d\d", str(d.get("tx_date") or ""))
    return dict(flag=flag, reasons=(norm_reasons(rs) if flag == "yes" else ()), invoice_outcome=out, invoice_id=iid, invoice_amount=None if ia is None else abs(ia),
                tx_amount=None if ta is None else abs(ta), tx_date=dt.group(0) if dt else "", bank_mismatch=norm_yn(d.get("bank_mismatch") or "no"))
def rec_diff(a, b):
    """fields on which two typed records differ (equivalence for the typed arm = no differing field)"""
    return [f for f in TFIELDS if (abs(a[f] - b[f]) > 0.005 if isinstance(a[f], float) and isinstance(b[f], float) else a[f] != b[f])]
def gt_record(p):
    g = C.gt_of(p)
    return dict(flag=g["flag"], reasons=tuple(r for r in g["reason"] if r != "none"), invoice_outcome="none" if g["inv_id"] == "NONE" else "matched", invoice_id=None if g["inv_id"] == "NONE" else g["inv_id"],
                invoice_amount=g["inv_amt"], tx_amount=round(float(g["tx_amt"]), 2), tx_date=g["tx_date"], bank_mismatch="yes" if p["bank_mismatch"] else "no")
def render_template(r):
    s = f"Transaction of ${r['tx_amount']:.2f} on {r['tx_date']}. "
    s += (f"Matched invoice {r['invoice_id']} (${r['invoice_amount']:.2f}). " if r["invoice_outcome"] == "matched" and r["invoice_amount"] is not None else
          f"Matched invoice {r['invoice_id']}. " if r["invoice_outcome"] == "matched" else "No matching invoice. ")
    s += ("Flagged: " + ", ".join(r["reasons"] or ("other",)) + ". ") if r["flag"] == "yes" else "Not flagged. "
    return s + ("Bank feed amount differs from the transaction." if r["bank_mismatch"] == "yes" else "No bank-feed mismatch noted.")
def typed_c(p, k, rep=0):
    """typed teacher record for prompt p with keep-set k (teacher, pinned as the free-text notes). None = not cached and net off. dict(rec=None) = unparseable (FAIL)."""
    ptxt = core.render(p, k)
    if not STATE["net"] and ckey(core.HAIKU, SYS_TY, ptxt, 200, core.EX_T, rep) not in llm.R: return None
    if STATE["capped"] and ckey(core.HAIKU, SYS_TY, ptxt, 200, core.EX_T, rep) not in llm.R: return None
    try: r = llm.chat(core.HAIKU, SYS_TY, ptxt, 200, core.EX_T, rep)
    except RuntimeError as e:
        if "spend cap" in str(e): STATE["capped"] = True; return None
        raise
    return dict(rec=parse_record(r["text"]), tok=r["usage"].get("prompt_tokens"), raw=r["text"])
