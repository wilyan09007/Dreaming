"""The nightly-books workflow, written the way a customer would write it: four call sites, two effects, no Dreaming logic.

    bank feed → categorize → vendor lookup → match invoice → anything odd? → post to books | wake a human

Only the public API is used (`step`, `tool`, `effect`). The functions under `@step` render the input text; the library
makes the call. Tools are the mock environment in tools.py (seeded, in-process, virtual latencies: S4's env).
"""

from __future__ import annotations

import os

from dreaming import effect, step, tool
from dreaming.contracts import Json, TeacherSpec, Thresholds
from dreaming.speculate import result

from . import tools

W = "bookkeeper"


def _teachers() -> tuple[TeacherSpec, TeacherSpec | None]:
    """(the teacher, the cheaper configuration on trial for mission 1) for the provider TEACHER names."""
    if os.environ.get("TEACHER") == "asi1":
        # The sponsor's model: no provider pin (one provider), no thinking toggle (so no router step), unknown body fields rejected.
        return TeacherSpec(provider="asi1", model="asi1-mini", price_in=0.18, price_out=0.45), None  # $/M per docs.asi1.ai; not in usage
    if os.environ.get("TEACHER") == "stepfun":
        # StepFun (docs/devtools.md): step-5-preview cannot switch thinking off, so mission 1's cheaper configuration is the
        # cheaper tier step-3.5-flash; reasoning tokens bill as output; temperature defaults to 0.5, so it is set.
        # floor: 200 identical calls on one ambiguous input disagreed 3.5% (193 / 2 / 5 unparsed), measured 2026-10-17 (smokes/RESULTS.md)
        teacher = TeacherSpec(provider="stepfun", model="step-5-preview", params={"reasoning_effort": "low", "temperature": 0}, price_in=1.00, price_out=2.70, floor=0.035)
        cheap = TeacherSpec(provider="stepfun", model="step-3.5-flash", params={"temperature": 0, "reasoning_effort": "low"}, price_in=0.10, price_out=0.30)
        # reasoning_effort=low: without it step-3.5-flash spent the whole 1024-token budget thinking on 58% of anything_odd prompts
        # (empty content, no label; data/recorded.jsonl 03:30); with it 61 tokens and a parsed label (measured 2026-10-18)
        return teacher, cheap
    # Pinned to Anthropic's endpoint; no temperature/seed/require_parameters: Anthropic ignores or 404s them (S13, SERVICES.md).
    pin = {"provider": {"order": ["anthropic"], "allow_fallbacks": False}}
    model = "anthropic/claude-haiku-5.5"  # $0.10/$0.50 per M on OpenRouter 2026-10-17 (smokes/RESULTS.md); usage.cost overrides when present
    teacher = TeacherSpec(provider="openrouter", model=model, params={**pin, "reasoning": {"effort": "low"}}, price_in=0.10, price_out=0.50)
    cheap = TeacherSpec(provider="openrouter", model=model, params={**pin, "reasoning": {"enabled": False}}, price_in=0.10, price_out=0.50)
    return teacher, cheap  # the same model with thinking off: mission 1's cheaper configuration


TEACHER, CHEAP = _teachers()
CATEGORIES = {
    "Software": "subscriptions, licenses, app stores", "Meals": "restaurants, coffee, food delivery", "Travel": "flights, rides, hotels",
    "Hosting": "cloud and servers", "Office supplies": "paper, toner, desk items", "Equipment": "hardware and tools",
    "Payroll": "salaries and contractors", "Utilities": "power, water, internet", "Advertising": "ads and promotion", "Other": "anything else",
}
T = Thresholds(watch_after=60, calibrate_after=500, audit_rate=0.05)


@step("categorize", workflow=W, labels=list(CATEGORIES), descriptions=CATEGORIES, teacher=TEACHER, thresholds=T)
def categorize(txn: dict[str, Json]) -> str:
    return f"Transaction: {txn['descriptor']}, ${float(str(txn['amount'])):.2f}"


@tool("vendor_lookup", workflow=W, fallback=tools.vendor_snapshot, thresholds=Thresholds(watch_after=60, calibrate_after=500),
      price_call=0.004)  # a paid lookup API: what the dream saves
def vendor_lookup(merchant: str) -> str:
    return tools.vendor_api(merchant)


@step("match_invoice", workflow=W, labels=["match", "none", "no invoice expected"],
      descriptions={"match": "an open invoice fits this vendor and amount", "none": "an invoice was expected but none fits", "no invoice expected": "this kind of spend has no invoice"},
      teacher=TEACHER, thresholds=T)
def match_invoice(txn: dict[str, Json], vendor: str, category: str, open_invoices: list[str]) -> str:
    return f"Vendor: {vendor}\nCategory: {category}\nAmount: ${float(str(txn['amount'])):.2f}\nOpen invoices: {', '.join(open_invoices) or 'none'}"


@step("anything_odd", workflow=W, labels=["yes", "no"], descriptions={"yes": "unusual amount, vendor or timing: a bookkeeper should look", "no": "routine"},
      teacher=TEACHER, cheap=CHEAP, thresholds=T)
def anything_odd(txn: dict[str, Json], vendor: str, category: str, invoice: str, history: list[str]) -> str:
    """Mission 1: the cheap configuration (CHEAP, thinking off) is this site's predictor; see predictors/router.py."""
    return f"Vendor: {vendor} ({category}), ${float(str(txn['amount'])):.2f}, invoice: {invoice}\nRecent history: {'; '.join(history[-5:]) or 'none'}"


@effect("post_to_books", workflow=W)
def post_to_books(txn: dict[str, Json], category: str, vendor: str) -> str:
    return tools.ledger_post(txn, category, vendor)


@effect("wake_a_human", workflow=W)
def wake_a_human(txn: dict[str, Json], reason: str) -> str:
    return tools.review_queue(txn, reason)


def run_transaction(txn: Json) -> str:
    """One run. Called inside a run boundary by the API (fast-forward) and by the uAgent (live chat).
    Plain Python takes `result(x)`, the verified value: under Runtime(speculate=True) calls return Pendings; otherwise a no-op."""
    assert isinstance(txn, dict)
    category = categorize(txn)
    vendor = vendor_lookup(str(txn["descriptor"]).split(" *")[0])  # runs alongside categorize under Runtime(speculate=True)
    invoice = match_invoice(txn, vendor, category, tools.open_invoices(result(vendor)))
    odd = anything_odd(txn, vendor, category, invoice, tools.history(result(vendor)))
    return wake_a_human(txn, f"{result(category)}/{result(invoice)}") if result(odd) == "yes" else post_to_books(txn, category, vendor)
