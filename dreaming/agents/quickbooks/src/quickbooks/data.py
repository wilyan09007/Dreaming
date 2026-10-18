"""Intuit's QuickBooks-Hackathon-2026 as rows for the three call sites (docs/training-plan.md §2.2-§3). The books are the teacher.

    download(root) -> Path                               the dataset dir (holds index.json): one 26.7 MB tarball, unpacked once; idempotent
    rows(site, root) -> [{rendered, label, group, ts}]   group = company id, ts = ISO date; site: invoice_paid | bill_account | bank_match
    company_rows(site, company_dir)                      the same for one company (rows() is these, chained; a caller may pool them)
    render_invoice / render_bill / render_bank           the text each site sees; quickbooks.workflow's @steps call the same functions

invoice_paid: on_time (receivable paid in full by the due day), late (paid in full after it), never (still unpaid on the last day of the
books). Right-censoring: an invoice due within HORIZON days of the last day is dropped whatever its outcome, so `never` means at least 90
days past due (measured 2026-10-18: 99.9% of paid invoices are paid within 87 days of the due day; the max is 170). The history line holds
only what was known on the invoice's day: earlier invoices of the customer already paid (on time / late) or already past due and open.
An invoice with nothing owed on its receivable (a deposit plus the withheld retainage cover the total; measured 82 of 53,644) has no
outcome: it is neither a row nor part of any history.

bill_account: the account the bill posts to, by name: account_lines[0] or, for a product bill, the item's purchase account (every bill in
the set has exactly one line). The line's own account name is never rendered.

bank_match (tool-shaped: the label is the source_document_id of the bank line, or `none`). DISCLOSURE, the data we changed: the dataset's
bank feed is clean by construction (DATA_README: "the feed matches the books exactly -- inject your own mess"), so bank_match rows carry
three injected faults, drawn from random.Random(company id) in feed order (sim_day, id), deterministic: (1) amount noise: 10% of bank lines
show the document amount minus a fee of $0.01-$25.00, capped at 3% of the amount (at least $0.01); (2) memo truncation: 30% of bank lines
keep only the first 12 characters of the memo (a memo of 12 or fewer is unchanged); (3) duplicates: 2% of bank lines are repeated once, same day, memo and amount, and the
repeat's correct answer is `none` because its document was already matched by the first copy. The open documents shown with a line are
the book records (one per clean feed line: id, date, amount, kind) dated within 3 days of it and not yet matched, the 5 closest in amount
(measured 2026-10-18: the correct document is among the five for 99.1% of the lines that have one, and first for 93.4%).
Labels and every other file (invoices, bills, payments, journal entries) are unchanged.
"""

from __future__ import annotations

import random
import shutil
import tarfile
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import yaml

REPO, TARBALL = "CL251215/QuickBooks-Hackathon-2026", "hackathon-100-participant.tar.gz"
LAST_DAY = 365  # every company: 365 days over one window (DATA_README); statuses are as of this day
HORIZON = 90
NOISE, TRUNC, DUP = 0.10, 0.30, 0.02
WINDOW, SHOWN = 3, 5
OUTCOMES = {"on_time": "paid in full by the due date", "late": "paid in full after the due date", "never": "not paid 90 days past due"}
# bill_account's labels: every account a bill posts to across the 100 companies (measured 2026-10-18; the rest of the 36-account chart is never billed)
ACCOUNTS = ["Shipping and Delivery", "Work in Progress", "Operating Expense", "Utilities", "Inventory Asset", "Rent and Lease", "Insurance",
            "Accrued Interest", "Taxes and Licenses", "Income Tax Expense", "Legal and Professional Fees", "Interest Expense",
            "Repairs and Maintenance", "Advertising and Marketing", "Equipment"]

Row = dict[str, str]
Doc = dict[str, Any]
_LOADER = yaml.CSafeLoader  # libyaml: 401 MB of YAML is minutes with it, far slower without; a wheel lacking it fails here instead of crawling


def download(root: Path) -> Path:
    out = root / "hackathon-100"
    if not (out / "index.json").is_file():
        # plain HTTPS, not huggingface_hub: the campaign image runs with HF_HUB_OFFLINE=1 (its encoder is baked in), which made
        # hf_hub_download raise OfflineModeIsEnabled on Modal (2026-10-18); the dataset is public, no login
        root.mkdir(parents=True, exist_ok=True)
        tgz = root / TARBALL  # fetched again on every attempt: a truncated download must never be unpacked twice
        urllib.request.urlretrieve(f"https://huggingface.co/datasets/{REPO}/resolve/main/{TARBALL}", tgz)
        partial = root / ".partial"
        shutil.rmtree(partial, ignore_errors=True)
        with tarfile.open(tgz) as t:
            t.extractall(partial, filter="data")
        (partial / "hackathon-100").rename(out)  # whole or absent: a killed unpack never looks done
        partial.rmdir()
    return out


def companies(root: Path) -> list[Path]:
    return sorted(root.glob("company-*/rwm-v4/world-*"))


def rows(site: str, root: Path) -> list[Row]:
    return [r for d in companies(root) for r in company_rows(site, d)]


def company_rows(site: str, d: Path) -> list[Row]:
    return _SITES[site](d, d.parents[1].name)


def _load(d: Path, name: str) -> list[Doc]:
    return yaml.load((d / f"{name}.yaml").read_text(encoding="utf-8"), Loader=_LOADER) or []


def _cents(s: str) -> int:
    return int(s.replace(".", ""))  # amounts are two-decimal strings, signed in the feed: exact


def _money(c: int) -> str:
    return f"{'-' if c < 0 else ''}{abs(c) // 100}.{abs(c) % 100:02d}"


# ---------------------------------------------------------------- invoice_paid


def render_invoice(customer: str, terms: str, amount: str, date: str, due: str, history: Counter[str], last: str) -> str:
    h = ", ".join(f"{n} {k}" for k, n in history.items() if n) or "none yet"
    return f"Customer: {customer}\nTerms: {terms}\nAmount: ${amount}\nIssued {date}, due {due}\nHistory: {h}; last: {last or 'none'}"


def _owed(inv: Doc) -> int:
    return _cents(inv["total_amount"]) - _cents(inv["deposit_applied_amount"]) - _cents(inv["retainage_amount"])


def _paid_day(inv: Doc, paid: Sequence[tuple[int, int]]) -> int | None:
    owed, got = _owed(inv), 0
    for day, c in sorted(paid):
        got += c
        if got >= owed:
            return day
    return None


def _invoice_rows(d: Path, group: str) -> list[Row]:
    terms = {c["id"]: c["payment_terms"] for c in _load(d, "customers")}
    paid: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    for p in _load(d, "customer_payments"):
        if p["settles"] == "receivable":  # retainage releases settle the withheld part, not the receivable
            for a in p["applications"]:
                paid[a["document_id"]].append((p["txn_sim_day"], _cents(a["amount"])))
    invs = sorted(_load(d, "invoices"), key=lambda i: (i["txn_sim_day"], i["id"]))
    day = {i["id"]: _paid_day(i, paid[i["id"]]) for i in invs}
    seen: defaultdict[str, list[Doc]] = defaultdict(list)
    out: list[Row] = []
    for inv in invs:
        if _owed(inv) <= 0:  # a deposit and the withheld retainage cover it: no receivable, so no outcome (82 of 147 "never" rows before this)
            continue
        t, prior = inv["txn_sim_day"], seen[inv["customer_id"]]
        seen[inv["customer_id"]].append(inv)
        if inv["due_sim_day"] > LAST_DAY - HORIZON:
            continue
        history: Counter[str] = Counter({"paid on time": 0, "paid late": 0, "open past due": 0})
        known: list[tuple[int, str]] = []
        for p in prior:
            pd = day[p["id"]]
            if pd is not None and pd < t:
                known.append((pd, "paid on time" if pd <= p["due_sim_day"] else "paid late"))
            elif p["due_sim_day"] < t:
                known.append((p["due_sim_day"], "open past due"))
        history.update(k for _, k in known)
        pd = day[inv["id"]]
        out.append({"rendered": render_invoice(inv["customer_name"], terms.get(inv["customer_id"], ""), inv["total_amount"], inv["txn_date"],
                                               inv["due_date"], history, max(known)[1] if known else ""),
                    "label": "never" if pd is None else "on_time" if pd <= inv["due_sim_day"] else "late", "group": group, "ts": inv["txn_date"]})
    return out


# ---------------------------------------------------------------- bill_account


def render_bill(vendor: str, kind: str, amount: str, memo: str, line: str) -> str:
    return f"Vendor: {vendor}\nKind: {kind}\nAmount: ${amount}\nMemo: {memo}\nLine: {line}"


def _bill_rows(d: Path, group: str) -> list[Row]:
    names = {a["id"]: a["name"] for a in _load(d, "accounts")}
    buys = {i["id"]: i["purchase_account_id"] for i in _load(d, "items")}
    out: list[Row] = []
    for b in sorted(_load(d, "bills"), key=lambda b: (b["txn_sim_day"], b["id"])):
        line = (b["account_lines"] or b["item_lines"])[0]
        account = line.get("account_id") or buys[line["item_id"]]
        out.append({"rendered": render_bill(b["vendor_name"], b["expense_kind"], b["total_amount"], b["memo"], line["description"]),
                    "label": names[account], "group": group, "ts": b["txn_date"]})
    return out


# ---------------------------------------------------------------- bank_match (injected mess: see the module docstring)


def render_bank(date: str, amount: str, memo: str, open_docs: Sequence[tuple[str, str, str, str]]) -> str:
    docs = "\n".join("  ".join(doc) for doc in open_docs) or "none"
    return f"Bank line: {date}  {amount}  {memo}\nOpen documents:\n{docs}"


def _messy(feed: Sequence[Doc], rng: random.Random) -> Iterator[tuple[int, int, str, str]]:
    """(index of the clean line, bank amount in cents, bank memo, correct answer) in feed order, faults drawn per line in a fixed order."""
    for k, b in enumerate(feed):
        amt, memo = _cents(b["amount"]), b["memo"]
        if rng.random() < NOISE:
            amt -= min(rng.randint(1, 2500), max(1, abs(amt) * 3 // 100))
        if rng.random() < TRUNC:
            memo = memo[:12]
        yield k, amt, memo, b["source_document_id"]
        if rng.random() < DUP:
            yield k, amt, memo, "none"


def _bank_rows(d: Path, group: str) -> list[Row]:
    feed = sorted(_load(d, "bank_transactions"), key=lambda b: (b["sim_day"], b["id"]))
    by_day: defaultdict[int, list[int]] = defaultdict(list)  # book record index by day; record k is feed[k]
    for k, b in enumerate(feed):
        by_day[b["sim_day"]].append(k)
    matched: set[int] = set()
    out: list[Row] = []
    for k, amt, memo, answer in _messy(feed, random.Random(group)):
        b, day = feed[k], feed[k]["sim_day"]
        near = [j for t in range(day - WINDOW, day + WINDOW + 1) for j in by_day.get(t, ()) if j not in matched]
        near.sort(key=lambda j: (abs(_cents(feed[j]["amount"]) - amt), feed[j]["source_document_id"]))
        shown = [(feed[j]["source_document_id"], feed[j]["date"], feed[j]["amount"], feed[j]["kind"]) for j in near[:SHOWN]]
        if answer != "none":
            matched.add(k)
        out.append({"rendered": render_bank(b["date"], _money(amt), memo, shown), "label": answer, "group": group, "ts": b["date"]})
    return out


_SITES: dict[str, Callable[[Path, str], list[Row]]] = {"invoice_paid": _invoice_rows, "bill_account": _bill_rows, "bank_match": _bank_rows}
