"""S16 prompt generator: 300 draft_note prompts from the synthetic bookkeeping world (imports s4 env; does not edit it).
Each prompt = tx line, matched invoice or NONE, anomaly flag + reason, bank-feed line, 15-30 history records (one span each)."""
import sys, json, math, datetime
import numpy as np
sys.path.insert(0, r"C:\Users\<user>\Desktop\personalprojects\Dreaming\smokes\s4_tool_predictor")
from env import Env
SEED, N = 16, 300
RECUR = {"software", "utilities", "telecom", "insurance", "advertising"}
MIX = [("normal", 105), ("noinv", 60), ("spike", 45), ("dup", 45), ("first", 45)]
D0 = datetime.date(2025, 1, 1)
fmt = lambda d: d.isoformat()

def build(seed=SEED, mix=MIX):
    env = Env(seed=seed); rng = np.random.default_rng(seed)
    scn = [s for s, k in mix for _ in range(k)]; rng.shuffle(scn)
    out = []
    for idx, sc in enumerate(scn):
        for _ in range(100000):
            t = env.new_txn("train"); mi = t["_m"]; vid = env.vid[mi]; m = env.m[mi]
            cand = dict(env.invoices.get(vid, []))
            inv, _ = env.match_invoice(vid, t["amount"]); flag, _ = env.flag_anomaly(vid, t["amount"], inv)
            spike = t["amount"] > 3 * env.vmean[vid]; a = t["amount"]
            if a < 8: continue
            if sc == "normal" and inv != "NONE" and not spike: break
            if sc == "noinv" and inv == "NONE" and not spike: break
            if sc == "spike" and spike and a >= 15: break
            if sc in ("dup", "first") and not spike: break
        txd = D0 + datetime.timedelta(days=int(rng.integers(0, 700)))
        n = int(rng.integers(15, 31)); rec = m["cat"] in RECUR; sig = .25 if rec else .5
        render = lambda mm: env._render(mm["tpl"][int(rng.integers(len(mm["tpl"])))])
        recs = []  # (days_before, amount, descriptor, invoice_text, vendor_is_same)
        if sc == "first":
            others = [int(x) for x in rng.choice([j for j in range(len(env.m)) if j != mi], int(rng.integers(3, 6)), replace=False)]
            days = np.sort(rng.choice(np.arange(1, 400), n, replace=False))
            for dd in days:
                o = env.m[int(rng.choice(others))]
                recs.append((int(dd), round(o["med"] * math.exp(rng.normal(0, .5)), 2), render(o), None, False))
        else:
            gaps = (rng.normal(30, 2.5, n).round().clip(24, 38) if rec else rng.exponential(14, n).round().clip(1, 90) + 1).astype(int)
            days = np.cumsum(gaps)
            if sc == "dup": days = days + 4  # keep a free slot for the duplicate 1-4 days back (days are >= 5)
            for dd in days:
                recs.append((int(dd), round(m["med"] * math.exp(rng.normal(0, sig)), 2), render(m), None, True))
            if sc == "dup":
                dd = int(rng.integers(1, 5)); desc = t["descriptor"] if rng.random() < .5 else render(m)
                recs.append((dd, a, desc, None, True)); recs.sort(key=lambda r: r[0])
        inv_ok = bool(env.invoiced[mi])
        lines = []
        for dd, amt, desc, _, same in recs:
            st = f"INV-{int(rng.integers(100000, 999999))}" if (inv_ok and same and rng.random() < .9) or (not same and rng.random() < .4) else "no invoice"
            lines.append(dict(days=dd, amt=amt, text=f"{fmt(txd - datetime.timedelta(days=dd))} | {desc} | ${amt:,.2f} | {st}"))
        # fixed fields
        if inv == "NONE": invl = "Matched invoice: NONE"; invamt = None
        else:
            invamt = cand[inv]; invl = f"Matched invoice: {inv} (vendor {vid}, ${invamt:,.2f}, issued {fmt(txd - datetime.timedelta(days=int(rng.integers(2, 20))))}; amount within 2% of the transaction, counts as a match)"
        reasons = []
        if sc == "dup": fl = "yes"; reasons = ["possible duplicate of a recent charge"]
        elif sc == "first": fl = "yes"; reasons = ["first purchase from this vendor"]
        else:
            fl = flag
            if inv == "NONE": reasons.append("no matching invoice")
            if spike: reasons.append("amount unusually high for this vendor")
        flagl = f"Anomaly flag: {fl} (reason: {'; '.join(reasons) if reasons else 'invoice matched and amount in the usual range'})"
        bam = round(a * 1.03, 2) if rng.random() < .12 else a  # 12%: bank posted a different amount (fee / FX)
        bdesc = t["descriptor"].upper()[:24]; pend = rng.random() < .06
        bankl = f"Bank feed: {'pending' if pend else 'posted ' + fmt(txd + datetime.timedelta(days=int(rng.integers(1, 3))))} | {bdesc} | -${bam:,.2f} | card ****{int(rng.integers(1000, 9999))} | {'PENDING' if pend else 'POSTED'}"
        txl = f"Transaction: {fmt(txd)} | {t['descriptor']} | ${a:,.2f}"
        out.append(dict(id=idx, scn=sc, vendor=vid, cat=m["cat"], tx_amt=a, spike=bool(spike), tx=txl, inv=invl, flag=flagl, bank=bankl,
                        flag_reasons=reasons, bank_mismatch=bool(bam != a), hist=lines, has_inv=inv != "NONE"))
    return out

if __name__ == "__main__":
    from pathlib import Path
    json.dump(build(99, [("normal", 6), ("noinv", 4), ("spike", 4), ("dup", 3), ("first", 3)]), open(Path(__file__).with_name("prompts_tune.json"), "w"), indent=0)  # 20 prompts, different seed, for pilot / judge-prompt tuning only
    P = build(); json.dump(P, open(Path(__file__).with_name("prompts.json"), "w"), indent=0)
    from collections import Counter
    print(len(P), Counter(p["scn"] for p in P), "flag yes", sum("yes" in p["flag"].split(":")[1][:5] for p in P))
    for k in (0, 1, 2, 3, 4):
        p = P[k]; print("---", p["scn"], "n_hist", len(p["hist"])); print(p["tx"]); print(p["inv"]); print(p["flag"]); print(p["bank"]); print("\n".join(h["text"] for h in p["hist"][:6]), "...")
