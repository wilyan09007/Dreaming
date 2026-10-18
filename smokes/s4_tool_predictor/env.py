"""Mock bookkeeping tool environment. SYNTHETIC: it only exercises the mechanics (certify / skip / speculate / rollout).
Seeded, in-process, virtual clock (latencies are numbers, never slept). Every tool call appends a row to the trace file.
Tools: vendor_lookup, lookup_order, issue_refund (S4) and teacher_category, match_invoice, flag_anomaly (S9 workflow).
Every tool returns (ret, latency_ms). Names starting with '_' / the 'hid' trace field are for evaluation only."""
import json, math
from collections import deque
import numpy as np

CATS = ["meals", "software", "office", "travel", "utilities", "telecom", "fuel", "groceries", "advertising", "insurance", "professional", "shipping"]
PRE = ["Blue", "Golden", "Urban", "Maple", "Summit", "Harbor", "Pioneer", "Cedar", "Lotus", "Nova", "Atlas", "Redwood", "Silver", "Prime",
       "Metro", "Sunrise", "Iron", "Coastal", "Bright", "Evergreen", "Orchid", "Falcon", "Granite", "Willow", "Echo", "Juniper", "Keystone",
       "Lakeside", "Mosaic", "Northern", "Onyx", "Pacific", "Quartz", "Riverside", "Sterling", "Titan", "Union", "Vista", "Wild", "Zenith"]
SUF = dict(
    meals="Bistro,Cafe,Grill,Kitchen,Diner,Pizza,Taqueria,Bakery,Sushi,Noodle House,Burger Co,Coffee",
    software="Cloud,Labs,Software,Systems,Analytics,Apps,Soft,Data,Stack,Logic,Platform,AI",
    office="Office Supply,Paper Co,Stationers,Print Shop,Furniture,Desk Co,Supply Depot,Toner Hub,Copy Center,Business Supply,Workspace,Equipment",
    travel="Airlines,Hotel,Inn,Suites,Car Rental,Travel,Rail,Resort,Lodge,Tours,Cruises,Motel",
    utilities="Electric,Water Co,Gas & Power,Utilities,Energy,Waste Services,Power & Light,Sewer District,Municipal Water,Solar,Heating,Power Co",
    telecom="Wireless,Mobile,Telecom,Fiber,Broadband,Communications,Cellular,Internet,Networks,Cable,Telephone,Connect",
    fuel="Fuel,Gas Station,Petroleum,Oil,Fuel Stop,Travel Plaza,Service Station,Gasoline,Pump,Auto Fuel,Truck Stop,Energy Mart",
    groceries="Market,Grocery,Foods,Supermarket,Farms,Fresh Market,Produce,Pantry,Wholesale Foods,Mart,Organics,Co-op",
    advertising="Media,Ads,Marketing,Advertising,Digital,Promotions,Creative,Outreach,Print Media,Social,Campaigns,Brand Studio",
    insurance="Insurance,Assurance,Mutual,Underwriters,Indemnity,Coverage,Risk Partners,Casualty,Life & Health,Benefits,Protect,Surety",
    professional="Law Group,Consulting,Accounting,Advisors,Partners,Legal,CPA,Associates,Engineering,Architects,Staffing,Solutions",
    shipping="Freight,Logistics,Courier,Express,Shipping,Delivery,Parcel,Cargo,Movers,Postal,Transport,Fulfillment")
SUF = {k: v.split(",") for k, v in SUF.items()}
CITIES = ["SEATTLE WA", "AUSTIN TX", "DENVER CO", "BOSTON MA", "PORTLAND OR", "CHICAGO IL", "MIAMI FL", "ATLANTA GA", "PHOENIX AZ", "SAN JOSE CA",
          "NEW YORK NY", "DALLAS TX", "RALEIGH NC", "MADISON WI", "TAMPA FL", "OMAHA NE", "BOISE ID", "RENO NV", "TUCSON AZ", "NASHVILLE TN"]
SQ_CATS = {"meals", "groceries", "office", "professional", "fuel", "shipping"}
MEDIAN = dict(meals=25, software=60, office=80, travel=250, utilities=120, telecom=90, fuel=55, groceries=70, advertising=400, insurance=300, professional=600, shipping=45)
INV_P = dict(meals=.2, software=.95, office=.7, travel=.6, utilities=.9, telecom=.9, fuel=.1, groceries=.1, advertising=.9, insurance=.95, professional=.95, shipping=.8)
RISK = dict(meals=0, software=-.3, office=.2, travel=.8, utilities=-.4, telecom=0, fuel=.5, groceries=.1, advertising=.6, insurance=-.2, professional=.4, shipping=.9)
DELIV = dict(software=3, advertising=3, insurance=3, utilities=3, telecom=3, groceries=1.5, shipping=.7)  # others 1
REFUND = dict(meals=.5, office=1.5, software=.7, travel=1.3, groceries=.6)  # others 1
CHANNELS = ["web", "app", "marketplace", "phone"]; CH_SPEED = [1.0, 1.1, .6, .45]
STATUS = ["pending", "shipped", "delivered", "refunded", "cancelled"]
OUT = ["ok", "insufficient_funds", "vendor_error", "timeout"]
ALNUM = list("0123456789ABCDEFGHJKLMNPQRSTUVWXYZ")
AMT_SIGMA = .5

def _p25(med):  # P(amount >= 25) for a merchant with median amount med
    return 1 - .5 * (1 + math.erf((math.log(25 / med)) / (AMT_SIGMA * math.sqrt(2))))

class Env:
    def __init__(self, seed=0, trace_path=None, n_unseen=60, target_inv=.70):
        r = self.r = np.random.default_rng(seed)
        m = []
        for c in CATS:  # 25 merchants per category = 300 canonical merchants
            pairs = [(p, s) for p in PRE for s in SUF[c]]
            for k in r.choice(len(pairs), 25, replace=False):
                m.append(dict(cat=c, name=f"{pairs[k][0]} {pairs[k][1]}"))
        amz = CATS.index("office") * 25
        m[amz]["name"] = "Amazon Marketplace"
        ids = r.permutation(len(m)) + 1
        for k, d in enumerate(m):
            d["vid"] = f"V{ids[k]:04d}"
            d["med"] = MEDIAN[d["cat"]] * math.exp(r.normal(0, .4))
            d["tpl"] = self._templates(d, r)
        self.m = m; self.vid = [d["vid"] for d in m]
        self.vmean = {d["vid"]: d["med"] * math.exp(AMT_SIGMA ** 2 / 2) for d in m}  # the vendor's mean amount (hidden master data)
        # seen / unseen split: S = merchants that can appear in training; U_cal / U_test = never in training, disjoint from each other
        perm = r.permutation([i for i in range(len(m)) if i != amz])
        zipf = lambda g: (np.array(g), (lambda w: w / w.sum())(1 / np.arange(1, len(g) + 1) ** .7))
        S = [amz] + [int(i) for i in perm[n_unseen:]]  # Amazon is the most frequent merchant
        self.grp = dict(S=zipf(S), Ucal=zipf(perm[:n_unseen // 2]), Utest=zipf(perm[n_unseen // 2:n_unseen]))
        self.group_of = {int(i): g for g, (ix, _) in self.grp.items() for i in ix}
        # which vendors send invoices; greedily balanced so ~70% of transactions (mix 80% S / 10% each U) have a matching invoice
        wt = np.zeros(len(m))
        for g, share in (("S", .8), ("Ucal", .1), ("Utest", .1)):
            ix, w = self.grp[g]; wt[ix] = share * w
        inv = np.array([r.random() < INV_P[d["cat"]] for d in m])
        pm = lambda i: (.99 * _p25(m[i]["med"]) + .15 * (1 - _p25(m[i]["med"]))) if inv[i] else .01
        rate = lambda: sum(wt[i] * pm(i) for i in range(len(m))) / wt.sum()
        for i in r.permutation(len(m)):
            if abs(rate() - target_inv) < .01: break
            inv[i] = rate() < target_inv
        self.invoiced = inv; self.pm = pm
        self.desc2m = {}; self.orders = {}; self.invoices = {}
        self.t = 1.7e9; self.regime = 0; self.outage = False
        self.n_calls = 0; self.n_tx = 0; self.n_inv = 0
        self.recent = {}
        self.f = open(trace_path, "w", encoding="utf-8") if trace_path else None

    # ---- world construction -------------------------------------------------
    def _templates(self, d, r):
        if d["name"] == "Amazon Marketplace":
            return [dict(pre=p, name="", tag=1, store=0, city=0, trail=0, trunc=0) for p in ("AMZN Mktp US*", "Amazon.com*", "AMAZON PRIME*")]
        nm = d["name"].upper(); w = nm.split()
        forms = [nm, " ".join([w[0]] + [x[:5] for x in w[1:]]), nm.replace(" ", ""), nm[:12]]
        pool = [""] * 5 + ["POS ", "CHECKCARD ", "PAYPAL *"] + ["SQ *"] * 2 * (d["cat"] in SQ_CATS) + ["TST* "] * (d["cat"] == "meals")
        out = []
        for _ in range(int(r.choice([1, 2, 3, 4], p=[.4, .3, .2, .1]))):
            pre = pool[r.integers(len(pool))]
            f = forms[r.choice(4, p=[.5, .2, .15, .15])]
            if pre == "PAYPAL *": f = forms[2][:10]
            out.append(dict(pre=pre, name=f, store=r.random() < .35, city=r.random() < .45, trail=r.random() < .25, trunc=r.random() < .3, tag=0))
        return out

    def _render(self, t):
        r = self.r
        s = t["pre"] + t["name"]
        if t["tag"]: s += "".join(r.choice(ALNUM, 5))
        if t["store"]: s += f" #{r.integers(1, 9999)}"
        if t["city"]: s += " " + CITIES[r.integers(len(CITIES))]
        if t["trail"]: s += f" {r.integers(1000, 999999)}"
        u = r.random(); s = s.lower() if u < .1 else s.title() if u < .2 else s
        return s[:22] if t["trunc"] else s

    def new_txn(self, split):
        """split: 'train' (only seen merchants) | 'cal' | 'test' (80% seen / 20% merchants never in training; cal and test use disjoint unseen sets)."""
        r = self.r
        self.t += r.exponential(480); self.n_tx += 1
        g = "S" if split == "train" or r.random() < .8 else "U" + split
        ix, w = self.grp[g]; mi = int(r.choice(ix, p=w)); d = self.m[mi]
        desc = self._render(d["tpl"][r.integers(len(d["tpl"]))])
        self.desc2m.setdefault(desc, mi)
        amt = d["med"] * math.exp(r.normal(0, AMT_SIGMA))
        if r.random() < .04: amt *= r.uniform(3.5, 8)  # spike
        amt = round(amt, 2); age = int(r.integers(0, 31)); oid = f"O{self.n_tx:05d}"
        self.orders[oid] = dict(created=self.t - age * 86400 - r.uniform(0, 86400), s="pending", done=0, f=math.exp(r.normal(0, .5)),
                                cat=d["cat"], amount=amt, ch=int(r.choice(4, p=[.4, .3, .2, .1])), vid=d["vid"])
        has = bool(r.random() < self.pm_amt(mi, amt))
        if has:
            self.n_inv += 1; self.invoices.setdefault(d["vid"], []).append((f"INV-{self.n_inv:06d}", amt * (1 + r.uniform(-.019, .019))))
        if r.random() < .5:  # distractor: same vendor, amount clearly outside the 2% window
            self.n_inv += 1; self.invoices.setdefault(d["vid"], []).append((f"INV-{self.n_inv:06d}", amt * (r.uniform(1.1, 2) if r.random() < .5 else r.uniform(.4, .9))))
        return dict(order_id=oid, descriptor=desc, amount=amt, hour=int(self.t / 3600) % 24, channel=CHANNELS[self.orders[oid]["ch"]], age_days=age,
                    ts=self.t, _m=mi, _group=g, _has_invoice=has)

    def pm_amt(self, mi, amt):
        return (.99 if amt >= 25 else .15) if self.invoiced[mi] else .01

    # ---- trace --------------------------------------------------------------
    def _log(self, tool, args, feat, ret, lat, hid=None):
        q = self.recent.setdefault(tool, deque(maxlen=30))
        row = dict(i=self.n_calls, tool=tool, args=args, feat=feat, ret=ret, latency_ms=round(float(lat), 1), ts=round(self.t, 1),
                   recent_error_rate=round(sum(q) / len(q), 4) if q else 0.0, hid=hid or {})
        q.append(int(isinstance(ret, str) and (ret.startswith("ERROR") or ret in OUT[1:])))
        self.n_calls += 1
        if self.f: self.f.write(json.dumps(row, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
        return ret, row["latency_ms"]

    def close(self):
        if self.f: self.f.close(); self.f = None

    # ---- tools --------------------------------------------------------------
    def vendor_lookup(self, descriptor):
        r = self.r; mi = self.desc2m.get(descriptor)
        lat = 1500 + r.lognormal(math.log(800), .6) if r.random() < .03 else r.lognormal(math.log(120), .35)  # 3% tail > 1.5 s
        if self.outage: ret, lat = "ERROR:outage", 3000.0
        elif r.random() < .02: ret, lat = "ERROR:timeout", 5000.0
        elif mi is None: ret = "ERROR:not_found"
        else: ret = self.vid[mi]
        return self._log("vendor_lookup", dict(descriptor=descriptor), dict(n_chars=len(descriptor)), ret, lat,
                         dict(m=mi, group=self.group_of.get(mi), outage=self.outage))

    def truth_vendor(self, descriptor):  # evaluation only
        return self.vid[self.desc2m[descriptor]]

    def lookup_order(self, order_id):
        r = self.r; o = self.orders[order_id]
        days = int((self.t - o["created"]) // 86400)
        while o["done"] < days:  # hidden Markov chain, one step per day of age
            o["done"] += 1; o["s"] = STATUS[int(r.choice(5, p=self._P(o["f"], o["ch"], o["amount"], o["cat"])[STATUS.index(o["s"])]))]
        lat = r.lognormal(math.log(200), .5)
        return self._log("lookup_order", dict(order_id=order_id), dict(age_days=days, channel=CHANNELS[o["ch"]], amount=o["amount"], category=o["cat"]), o["s"], lat)

    def _P(self, f, ch, amount, cat):  # daily transition matrix over STATUS = pending, shipped, delivered, refunded, cancelled
        pc = .03 + .03 * (amount > 300); ps = min(.9, .35 * CH_SPEED[ch] * f); pd = min(.9, .3 * DELIV.get(cat, 1) * f); pr = .02 * REFUND.get(cat, 1) * (1 + amount / 500)
        P = np.eye(5); P[0] = [1 - pc - ps, ps, 0, 0, pc]; P[1] = [0, 1 - pd, pd, 0, 0]; P[2] = [0, 0, 1 - pr, pr, 0]
        return P

    def order_dist(self, age, channel, amount, cat):  # evaluation only: exact P(status | visible features), integrating over the hidden speed factor f
        from scipy.stats import norm
        ps = []
        for f in np.exp(.5 * norm.ppf((np.arange(30) + .5) / 30)):
            ps.append(np.linalg.matrix_power(self._P(f, CHANNELS.index(channel), amount, cat), int(age))[0])
        return np.mean(ps, axis=0)

    def refund_probs(self, amount, cat, hour, regime):
        z = np.array([0., -3.8 + math.log(amount / 50),
                      -3.2 + RISK[cat] + 1.4 * (2 <= hour < 5) + 2.2 * regime,
                      -4.0 + .5 * RISK[cat] + .5 * (12 <= hour < 14) + 2.8 * regime])
        e = np.exp(z - z.max()); return e / e.sum()

    def issue_refund(self, order_id, amount):
        r = self.r; o = self.orders[order_id]; hour = int(self.t / 3600) % 24
        self.regime = int(r.random() < .005) if self.regime == 0 else int(r.random() < .97)  # hidden degraded-bank regime
        p = self.refund_probs(amount, o["cat"], hour, self.regime)
        out = OUT[int(r.choice(4, p=p))]
        med = dict(ok=400, insufficient_funds=250, vendor_error=700, timeout=8000)[out]
        mult = (1 + .15 * math.log1p(amount / 100)) * (1 + .5 * RISK[o["cat"]]) * (1 + .3 * max(0, math.sin(math.pi * (hour - 6) / 12))) * (1.6 if self.regime and out != "timeout" else 1)  # slower vendor APIs by category, busier in the day
        lat = r.lognormal(math.log(med * mult), .35 if out != "timeout" else .05)
        if out == "ok": o["refunded"] = True  # side effect on the world
        return self._log("issue_refund", dict(order_id=order_id, amount=amount), dict(amount=amount, category=o["cat"], hour=hour), out, lat,
                         dict(regime=self.regime, p_err=float(1 - p[0])))

    # ---- S9 workflow steps --------------------------------------------------
    def teacher_category(self, descriptor):  # stands in for the frontier model: reads the hidden category
        return self._log("categorize", dict(descriptor=descriptor), {}, self.m[self.desc2m[descriptor]]["cat"], 900.0)

    def match_invoice(self, vendor_id, amount):
        c = [(abs(a - amount), i) for i, a in self.invoices.get(vendor_id, []) if abs(a - amount) <= .02 * amount]
        ret = min(c)[1] if c else "NONE"
        if c: self.invoices[vendor_id] = [x for x in self.invoices[vendor_id] if x[0] != ret]  # an invoice is consumed once matched
        return self._log("match_invoice", dict(vendor_id=vendor_id, amount=amount), {}, ret, self.r.lognormal(math.log(80), .3))

    def flag_anomaly(self, vendor_id, amount, invoice):  # rule: no invoice OR amount > 3x the vendor's mean
        ret = "yes" if invoice == "NONE" or amount > 3 * self.vmean[vendor_id] else "no"
        return self._log("flag_anomaly", dict(vendor_id=vendor_id, amount=amount, invoice=invoice), {}, ret, 600.0)


if __name__ == "__main__":  # self-check: determinism, outage, tail, shapes
    def run(seed):
        e = Env(seed); rows = []
        for _ in range(300):
            t = e.new_txn("test")
            rows += [e.vendor_lookup(t["descriptor"]), e.lookup_order(t["order_id"]), e.issue_refund(t["order_id"], t["amount"])]
        return e, rows
    (e1, a), (_, b) = run(0), run(0)
    assert a == b, "same seed must give identical returns and latencies"
    g = {k: set(map(int, v[0])) for k, v in e1.grp.items()}; assert len(g["S"]) == 240 and len(g["Ucal"]) == 30 and len(g["Utest"]) == 30 and not (g["S"] & g["Ucal"] | g["S"] & g["Utest"] | g["Ucal"] & g["Utest"])
    assert len(e1.m) == 300 and len({d["vid"] for d in e1.m}) == 300 and len({d["name"] for d in e1.m}) == 300
    v = [x for x in a[0::3] if not x[0].startswith("ERROR")]
    assert all(e1.truth_vendor(d) == e1.vid[e1.desc2m[d]] for d in list(e1.desc2m)[:50])
    lat = np.array([x[1] for x in a[0::3]]); print("vendor_lookup latency median %.0f ms, share >1.5s %.3f, errors %.3f" % (np.median(lat), (lat > 1500).mean(), 1 - len(v) / 300))
    e1.outage = True; t = e1.new_txn("test"); assert e1.vendor_lookup(t["descriptor"])[0].startswith("ERROR")
    print("sample descriptors:", [e1.new_txn("train")["descriptor"] for _ in range(6)]); print("env self-check ok")
