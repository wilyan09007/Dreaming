"""Deterministic canonicalizer for draft_note notes (no model calls). canon(note) -> dict of canonical facts; None = not stated / unclear."""
import re
from difflib import SequenceMatcher

MONTHS = {m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}
NEG_SIGN = "-\u2212"
AMT = re.compile(r"(?<![\w$])([" + NEG_SIGN + r"])?\s?\$\s?([\d,]+(?:\.\d+)?)")
INV = re.compile(r"INV-\d{6}")
ISO = re.compile(r"\b(\d{4})-(\d\d)-(\d\d)\b")
MDY = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? (\d{1,2}),? (\d{4})\b", re.I)
NOISE = {"com", "net", "org", "pos", "sq", "tst", "paypal", "checkcard", "debit", "purchase", "card", "the", "inc", "llc", "co", "and"}

RANGE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*(?:to|through|and|[-\u2013\u2014])\s*\$?\s*([\d,]+(?:\.\d+)?)")

def money(s): return float(s.replace(",", ""))

def vendor_tokens(tx_line):
    """distinctive tokens of the vendor string in the transaction line: alphabetic, >=3 letters, not a processor prefix; first two kept"""
    v = tx_line.split("|")[1] if "|" in tx_line else tx_line
    toks = [t for t in re.findall(r"[A-Za-z]+", v) if len(t) >= 3 and t.lower() not in NOISE]
    return [t.lower() for t in toks[:2]]

def tok_match(vt, words):
    for w in words:
        if w == vt: return True
        if len(w) >= 4 and len(vt) >= 4 and (w.startswith(vt) or vt.startswith(w)): return True
        if len(vt) >= 5 and SequenceMatcher(None, vt, w).ratio() >= 0.85: return True
    return False

def vendor_ok(text, tx_line):
    vts = vendor_tokens(tx_line)
    if not vts: return None
    words = re.findall(r"[a-z]+", text.lower())
    return all(tok_match(vt, words) for vt in vts)

def dates(t):
    out = [(m.start(), m.group(0)) for m in ISO.finditer(t)]
    for m in MDY.finditer(t): out.append((m.start(), f"{m.group(3)}-{MONTHS[m.group(1)[:3].lower()]:02d}-{int(m.group(2)):02d}"))
    return [d for _, d in sorted(out)]

CAT = {
 "duplicate": r"duplicate|repeat(ed)? charge|double[- ]charg|same (charge|amount) (as|was)",
 "first purchase": r"first[- ]time|first (purchase|transaction|charge|payment|order|time)|new vendor|never (purchased|bought|used|paid)",
 "unusually high": r"unusually (high|large)|higher than (the )?(usual|typical|normal)|exceeds? (the )?(usual|typical)|well above|spike|abnormally|above (the )?(vendor'?s? )?(usual|typical)|out of (the )?(usual|typical)",
 "no invoice": r"no (matching|matched|corresponding|linked|associated|invoice)|without (a |an )?(matching )?invoice|invoice (was )?(missing|not found|unmatched)|unmatched|missing invoice|lack(s|ing)?( of)? (a |an )?(matching )?invoice|not matched",
}
NOINV_ANY = re.compile(CAT["no invoice"], re.I)
NEGFLAG = re.compile(r"\b(not|n't|never)\s+(been\s+|be\s+)?(flagged( as (an |a )?(anomaly|anomalous))?|an anomaly|anomalous)|\bno (anomaly|flag)\b|\bunflagged\b|without (a |an )?(flag|anomaly)|anomaly flag:?\s*no\b|wasn't flagged|\bnot flag", re.I)
POSFLAG = re.compile(r"\bflagged\b|\braised a flag\b|\bflag (raised|set)\b|marked (as )?(an )?anomal|anomaly flag:?\s*yes|\b(is|was) (an )?anomal|\banomalous\b", re.I)
FOLLOW = re.compile(r"follow[- ]?up|recommend|should (be )?(review|verif|check|confirm|investigat|look)|needs? (further |manual |to be )?(review|verif|check|confirm|investigat|attention)|requires? (further |manual )?(review|verif|attention|follow)|\bplease\b|\bverify\b|investigat|warrants?|worth (checking|reviewing)|suggest|confirm (with|that|whether)|reviewer (should|to|may) (review|verif|check|confirm|investigat|follow|look|contact|inspect)", re.I)
HIST = re.compile(r"\b(prior|previous|previously|historical|history|earlier|past|ledger (shows|history|records)|recent (charges|transactions|purchases|ledger|history|payments)|highest|lowest|typically|mostly|vendor'?s? (usual|typical|prior|recent)|usual amounts?|average|median|range of|between about|ranging)\b", re.I)

def sentences(t): return [s for s in re.split(r"(?<=[.;])\s+|\s*;\s*", t) if s.strip()]

def canon(text, tx_line):
    t = text.strip(); out = {}
    ids = [(m.start(), m.group(0)) for m in INV.finditer(t)]
    ams = [(m.start(), bool(m.group(1)), money(m.group(2)), m.end()) for m in AMT.finditer(t)]
    # invoice amount: first $ after an invoice id, within 80 chars, no second id or sentence end between
    inv_am, tag = None, set()
    if ids:
        s0 = ids[0][0]; nxt = ids[1][0] if len(ids) > 1 else len(t)
        for k, (pos, neg, v, end) in enumerate(ams):
            if s0 < pos < min(nxt, s0 + 80) and not re.search(r"[.;]\s", t[s0:pos]) and not neg: inv_am = v; tag.add(k); break
    # bank amount: negative-signed amount, or the first $ after bank/feed/posted within 45 chars
    bank_am = None
    for k, (pos, neg, v, end) in enumerate(ams):
        if k in tag: continue
        if neg: bank_am = v; tag.add(k); break
    if bank_am is None:
        for k, (pos, neg, v, end) in enumerate(ams):
            if k in tag: continue
            pre = t[max(0, pos - 45):pos]
            if re.search(r"(bank|feed|posted)[^$.;]*$", pre, re.I) and not re.search(r"(bank|feed|posted)\s*\d{4}-\d\d-\d\d[^$]*\$", pre + "$", re.I) or re.search(r"(bank|feed)[^$.;]*$", pre, re.I):
                bank_am = v; tag.add(k); break
    tx_am = next((v for k, (pos, neg, v, end) in enumerate(ams) if k not in tag), None)
    out["tx_amt"] = tx_am; out["inv_amt"] = inv_am; out["bank_amt"] = bank_am
    out["vendor_ok"] = vendor_ok(t, tx_line)
    ds = dates(t); out["tx_date"] = ds[0] if ds else None; out["dates"] = ds
    m = re.search(r"issued\s+(?:on\s+)?(\d{4}-\d\d-\d\d)", t, re.I); out["inv_date"] = m.group(1) if m else None
    # invoice outcome
    nm = NOINV_ANY.search(t)
    if nm and (not ids or nm.start() < ids[0][0]): out["inv_id"] = "NONE"
    elif ids: out["inv_id"] = ids[0][1]
    elif nm: out["inv_id"] = "NONE"
    else: out["inv_id"] = None
    # flag decision
    neg = NEGFLAG.search(t); rest = NEGFLAG.sub(" ", t); pos = POSFLAG.search(rest)
    out["flag"] = "no" if neg and not pos else "yes" if pos and not neg else None
    # reason category set
    rs = None
    if out["flag"] == "no": rs = ("none",)
    elif out["flag"] == "yes":
        ss = sentences(rest); keep = []
        for k, s_ in enumerate(ss):
            if POSFLAG.search(s_):
                if k and re.search(r"\b(that|this|the same|those) (reason|issue)|for that|for this", s_, re.I): keep.append(ss[k - 1])
                keep.append(s_)
        fs = " ".join(keep) or rest
        found = {c for c, rx in CAT.items() if c != "no invoice" and re.search(rx, fs, re.I)}
        if re.search(r"(because|due to|since|on account of|flagged for|reason[: ]+|\bfor\b)[^.;]{0,25}(" + CAT["no invoice"] + ")", fs, re.I) or (not found and re.search(CAT["no invoice"], fs, re.I)): found.add("no invoice")
        rs = tuple(sorted(found)) or ("other",)
    out["reason"] = rs
    out["followup"] = bool(FOLLOW.search(t))
    # history / earlier-charge evidence (not in the first field list; the strict judge treats a different count, range or earlier-charge date as a conflict)
    m = re.search(r"(\d{1,3})\+?\s+(?:prior|previous|recent|earlier|listed|historical|past|ledger)\b", t, re.I); out["hist_count"] = float(m.group(1)) if m else None
    out["hist_lo"] = out["hist_hi"] = None
    for m in RANGE.finditer(t):
        if re.search(r"range|between|from|ranged|ranging|mostly|about|roughly|typical|history|prior|usual|charges", t[max(0, m.start() - 70):m.start()], re.I):
            out["hist_lo"], out["hist_hi"] = money(m.group(1)), money(m.group(2)); break
    m = re.search(r"(?:highest|largest|maximum|max)\b[^$.;]{0,50}\$\s?([\d,]+(?:\.\d+)?)|\bone at \$\s?([\d,]+(?:\.\d+)?)", t, re.I)
    out["hist_top"] = money(m.group(1) or m.group(2)) if m else None
    bd = {x.group(1) for x in re.finditer(r"(?:posted|pending|bank|feed)[^0-9$]{0,25}(\d{4}-\d\d-\d\d)", t, re.I)}
    out["other_dates"] = frozenset(d for d in ds if d != out["tx_date"] and d != out["inv_date"] and d not in bd) or None
    hs = HIST.sub(lambda m: m.group(0), t)
    out["history"] = bool(HIST.search(re.sub(r"usual range|usual amounts? range", "", t, flags=re.I)))
    return out

def gt_of(p):
    g = {"tx_amt": p["tx_amt"], "tx_date": re.search(r"\d{4}-\d\d-\d\d", p["tx"]).group(0), "flag": "yes" if p["flag"].startswith("Anomaly flag: yes") else "no"}
    m = re.search(r"(INV-\d{6}).*?\$([\d,]+\.\d\d), issued (\d{4}-\d\d-\d\d)", p["inv"])
    g["inv_id"], g["inv_amt"], g["inv_date"] = (m.group(1), money(m.group(2)), m.group(3)) if m else ("NONE", None, None)
    mb = re.search(r"-\$([\d,]+\.\d\d)", p["bank"]); g["bank_amt"] = money(mb.group(1)) if mb else None
    g["bank_date"] = (re.search(r"posted (\d{4}-\d\d-\d\d)", p["bank"]) or [None, None])[1]
    f = p["flag"].lower()
    if g["flag"] == "no": r = ("none",)
    else:
        r = []
        if "duplicate" in f: r.append("duplicate")
        if "first purchase" in f: r.append("first purchase")
        if "unusually high" in f: r.append("unusually high")
        if "no matching invoice" in f: r.append("no invoice")
        r = tuple(sorted(r)) or ("other",)
    g["reason"] = r
    return g
