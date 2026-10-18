"""The QuickBooks call sites (docs/training-plan.md §2.2), declared the way a customer would: each @step renders its input with the
render function quickbooks.data uses for the campaign's rows, so a live call and a training row see the same text.

    invoice_paid   step, on_time / late / never          a collections workflow's question per open invoice (System One shape)
    bill_account   step, the 15 accounts bills post to   Intuit-shaped categorize
    bank_match     tool, the document id a bank line settles, or none   benchmark-only (below)

The books are the teacher: labels come from what happened (zero label noise, so the certificate is against the truth); nothing here
calls an LLM. Importing this module registers the sites (the campaign's call_site() relies on that).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from dreaming import step, tool
from dreaming.contracts import TeacherSpec

from . import data

W = "quickbooks"
BOOKS = TeacherSpec(provider="books", model=data.REPO, floor=0.0)


@step("invoice_paid", workflow=W, labels=list(data.OUTCOMES), descriptions=data.OUTCOMES, teacher=BOOKS)
def invoice_paid(customer: str, terms: str, amount: str, date: str, due: str, history: Counter[str], last: str) -> str:
    return data.render_invoice(customer, terms, amount, date, due, history, last)


@step("bill_account", workflow=W, labels=data.ACCOUNTS, teacher=BOOKS)
def bill_account(vendor: str, kind: str, amount: str, memo: str, line: str) -> str:
    return data.render_bill(vendor, kind, amount, memo, line)


@tool("bank_match", workflow=W)
def bank_match(date: str, amount: str, memo: str, open_docs: Sequence[tuple[str, str, str, str]]) -> str:
    # ponytail: no live ledger to reconcile against; the answers are the books' source_document_id (data.rows). A ledger API goes here.
    raise NotImplementedError("bank_match is benchmark-only: its answers come from the books via quickbooks.data.rows")
