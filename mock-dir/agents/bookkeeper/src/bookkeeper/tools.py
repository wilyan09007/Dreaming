"""The mock world: a seeded ledger with merchants, invoices and history, plus the three externals the workflow calls.

Synthetic by design (bookkeeping-data.md: nothing public links transactions, vendors, invoices and anomalies), and the
page says so. Nothing sleeps: fast-forward stays fast and byte-replayable. The world (120 merchants, their popularity,
their usual amounts) is fixed; `seed` only picks the stream drawn from it. Ideas ported from smokes/s4_tool_predictor/env.py.
"""

from __future__ import annotations

import hashlib
from bisect import bisect_left
from collections.abc import Iterator
from datetime import date, timedelta
from functools import cache
from itertools import accumulate
from typing import NamedTuple

import numpy as np
from dreaming.context import charge
from dreaming.contracts import Json

DOWN = {"vendor_api": False}  # fault injection: the API's outage endpoint flips it, the tool raises, the runtime detects the outage
SNAPSHOT_ROWS, UNSEEN_P, ANOMALY_P, ZIPF = 500, 0.08, 0.02, 1.1  # the snapshot knows the first 500 rows; 8% of rows from the rare tail
VENDOR_API_MS = 1200  # the one slow external: its latency is charged to the virtual clock, never slept
TAGS = ("SQ", "POS", "WEB", "ACH", "RECUR", "EMEA")  # the bank feed's channel word after "MERCHANT *"
_RAW = {  # category: (usual amount in $, "TOKEN=Canonical name;..."), TOKEN is what comes before " *" in a descriptor
    "Software": (60, ("ADOBE=Adobe Inc.;NOTION=Notion Labs;FIGMA=Figma Inc.;SLACK=Slack Technologies;ZOOM=Zoom Video;DROPBOX=Dropbox Inc.;"
                     "GITHUB=GitHub Inc.;ATLASSIAN=Atlassian Corp;1PASSWORD=AgileBits Inc.;LINEAR=Linear Orbit;CANVA=Canva Pty;JETBRAINS=JetBrains")),
    "Meals": (25, ("BLUE BOTTLE=Blue Bottle Coffee;STARBUCKS=Starbucks Corp;SWEETGREEN=Sweetgreen;CHIPOTLE=Chipotle Grill;DOORDASH=DoorDash;"
                  "UBER EATS=Uber Eats;GRUBHUB=Grubhub;PANERA=Panera Bread;SHAKE SHACK=Shake Shack;PEETS=Peet's Coffee;LA TAQUERIA=La Taqueria;"
                  "JOE JUICE=Joe & The Juice")),
    "Travel": (250, ("UNITED AIR=United Airlines;DELTA AIR=Delta Air Lines;UBER TRIP=Uber Rides;LYFT=Lyft Inc.;MARRIOTT=Marriott Intl.;"
                    "HILTON=Hilton Hotels;AIRBNB=Airbnb Inc.;HERTZ=Hertz Corp;AMTRAK=Amtrak;SOUTHWEST AIR=Southwest Airlines;EXPEDIA=Expedia Group;"
                    "HYATT=Hyatt Hotels")),
    "Hosting": (150, ("AWS=Amazon Web Services;GCP=Google Cloud;AZURE=Microsoft Azure;DIGITALOCEAN=DigitalOcean;CLOUDFLARE=Cloudflare Inc.;"
                     "HEROKU=Heroku Inc.;VERCEL=Vercel Inc.;FLY IO=Fly.io;NETLIFY=Netlify Inc.;LINODE=Akamai Linode;HETZNER=Hetzner Online;"
                     "RENDER=Render Services")),
    "Office supplies": (75, ("STAPLES=Staples Inc.;OFFICE DEPOT=Office Depot;ULINE=Uline Inc.;AMZN MKTP=Amazon Marketplace;QUILL=Quill Corp;"
                            "PAPER SOURCE=Paper Source;BLICK ART=Blick Art Materials;WB MASON=W.B. Mason;COSTCO=Costco Wholesale;TARGET=Target Corp;"
                            "MICHAELS=Michaels Stores;IKEA=IKEA US")),
    "Equipment": (400, ("APPLE STORE=Apple Inc.;BEST BUY=Best Buy Co;B AND H=B&H Photo Video;DELL=Dell Technologies;LENOVO=Lenovo US;"
                       "HOME DEPOT=The Home Depot;NEWEGG=Newegg Inc.;MICRO CENTER=Micro Center;GRAINGER=W.W. Grainger;LOGITECH=Logitech Inc.;"
                       "SAMSUNG=Samsung Electronics;CDW=CDW Corp")),
    "Payroll": (2500, ("GUSTO=Gusto Payroll;ADP=ADP Inc.;RIPPLING=Rippling People Center;DEEL=Deel Inc.;PAYCHEX=Paychex Inc.;TRINET=TriNet Group;"
                      "JUSTWORKS=Justworks Inc.;UPWORK=Upwork Global;FIVERR=Fiverr Intl.;TOPTAL=Toptal LLC;REMOTE COM=Remote Technology;OYSTER=Oyster HR")),
    "Utilities": (140, ("PG AND E=Pacific Gas & Electric;CON EDISON=Con Edison;COMCAST BIZ=Comcast Business;VERIZON FIOS=Verizon Communications;"
                       "ATT BUSINESS=AT&T Inc.;SPECTRUM=Charter Spectrum;CITY WATER=City Water Dept;DUKE ENERGY=Duke Energy;XFINITY=Xfinity Internet;"
                       "TMOBILE=T-Mobile US;CITY UTIL=City Utilities;WASTE MGMT=Waste Management")),
    "Advertising": (500, ("GOOGLE ADS=Google Ads;META ADS=Meta Platforms;LINKEDIN ADS=LinkedIn Corp;X ADS=X Corp;MAILCHIMP=Mailchimp Inc.;"
                         "HUBSPOT=HubSpot Inc.;TIKTOK ADS=TikTok Inc.;REDDIT ADS=Reddit Inc.;SPOTIFY ADS=Spotify AB;YELP ADS=Yelp Inc.;"
                         "SEMRUSH=Semrush Inc.;BING ADS=Microsoft Advertising")),
    "Other": (90, ("USPS=US Postal Service;FEDEX=FedEx Corp;UPS STORE=The UPS Store;NOTARY=Notary Public;STATE FILING=State Filing Fee;"
                  "LEGALZOOM=LegalZoom;STRIPE FEE=Stripe Inc.;BANK FEE=Bank Service Fee;WEWORK=WeWork;DONATE=Charity Donation;"
                  "IRS EFTPS=US Treasury;PARKING=City Parking")),
}


class Merchant(NamedTuple):
    token: str  # what the descriptor starts with: the stable key the workflow looks up
    name: str  # canonical vendor name
    category: str
    usual: float  # the usual amount in $ (the median of its lognormal)


_SPREAD = iter(np.random.default_rng(1).normal(0, 0.4, 120))  # each merchant's own usual amount, fixed for every seed
MERCHANTS = {t: Merchant(t, n, cat, med * float(np.exp(next(_SPREAD)))) for cat, (med, raw) in _RAW.items()
             for t, n in (kv.split("=") for kv in raw.split(";"))}
_RANK = [list(MERCHANTS)[int(i)] for i in np.random.default_rng(99).permutation(len(MERCHANTS))]  # most popular first
_POOLS = {False: _RANK[:-16], True: _RANK[-16:]}  # True: the rare tail, drawn at UNSEEN_P from the first row on (S4: novel inputs at the deployment's rate)


_CDF = {k: list(accumulate(1.0 / r**ZIPF for r in range(1, len(v) + 1))) for k, v in _POOLS.items()}  # unnormalized Zipf


def ledger(n: int, seed: int = 0) -> Iterator[dict[str, Json]]:
    """n transactions: {"id", "descriptor", "amount", "date", "merchant"}; repeat merchants, a long tail, a few planted anomalies."""
    rng = np.random.default_rng(seed)
    for i in range(n):
        odd = bool(rng.random() < ANOMALY_P)  # a known merchant at 20-100x its usual amount
        tail = not odd and bool(rng.random() < UNSEEN_P)
        m = MERCHANTS[_POOLS[tail][int(bisect_left(_CDF[tail], rng.random() * _CDF[tail][-1]))]]
        amount = m.usual * (rng.uniform(20, 100) if odd else float(np.exp(rng.normal(0, 0.5))))
        yield {"id": f"txn_{seed}_{i:05d}", "date": (date(2026, 1, 1) + timedelta(days=i // 10)).isoformat(), "merchant": m.name,
               "descriptor": f"{m.token} *{TAGS[int(rng.integers(len(TAGS)))]}", "amount": round(float(amount), 2)}


@cache
def _snapshot() -> frozenset[str]:
    return frozenset(str(t["descriptor"]).split(" *")[0] for t in ledger(SNAPSHOT_ROWS, 0))


def vendor_api(merchant: str) -> str:
    """Return the canonical vendor name. Raises ConnectionError while DOWN["vendor_api"]."""
    charge(VENDOR_API_MS)
    if DOWN["vendor_api"]:
        raise ConnectionError("vendor API unreachable")
    return MERCHANTS[merchant].name if merchant in MERCHANTS else "unknown"


def vendor_snapshot(merchant: str) -> str:
    """Last night's snapshot: the registered fallback. Knows the merchants seen before the snapshot, 'unknown' otherwise."""
    return MERCHANTS[merchant].name if merchant in _snapshot() else "unknown"


def _h(vendor: str) -> int:
    return int.from_bytes(hashlib.sha256(vendor.encode()).digest()[:8], "big")


def open_invoices(vendor: str) -> list[str]:
    h = _h(vendor)
    return [f"INV-{(h >> (9 * k)) % 9000 + 1000}" for k in range(h % 4)]  # 0-3 ids


def history(vendor: str) -> list[str]:
    h = _h(vendor) >> 8
    return [f"2025-{(h >> (6 * k)) % 12 + 1:02d} ${(h >> (7 * k)) % 400 + 10}.00 posted" for k in range(h % 6)]  # 0-5 lines


def ledger_post(txn: dict[str, Json], category: str, vendor: str) -> str:
    return "posted"


def review_queue(txn: dict[str, Json], reason: str) -> str:
    return "sent for review"
