"""quickbooks.data, .workflow and .baselines offline, on a two-company dataset written in the real layout (tmp_path, no network)."""

from __future__ import annotations

import importlib
import json
import shutil
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml
from dreaming import registry
from quickbooks import baselines, data


def bill(i: int, vendor: str, day: int, account: str | None = None, item: str | None = None) -> dict[str, Any]:
    acct = [{"line_num": 1, "account_id": account, "account_name": "LEAK", "description": "running costs", "amount": "500.00"}] if account else []
    items = [{"line_num": 1, "item_id": item, "item_name": "Buns", "description": "flat of buns", "quantity": 1, "amount": "9.00"}] if item else []
    return {"id": f"bill-{i}", "vendor_id": vendor, "vendor_name": f"Vendor {vendor}", "expense_kind": "operating" if account else "product",
            "txn_sim_day": day, "txn_date": f"2026-01-{day:02d}", "item_lines": items, "account_lines": acct, "total_amount": "500.00", "memo": "m"}


def inv(i: str, day: int, due: int, total: str = "100.00", deposit: str = "0.00", retainage: str = "0.00") -> dict[str, Any]:
    return {"id": i, "customer_id": "c1", "customer_name": "Acme", "txn_sim_day": day, "txn_date": f"d{day:03d}", "due_sim_day": due,
            "due_date": f"d{due:03d}", "total_amount": total, "deposit_applied_amount": deposit, "retainage_amount": retainage}


def pay(doc: str, day: int, amount: str) -> dict[str, Any]:
    return {"txn_sim_day": day, "settles": "receivable", "applications": [{"document_id": doc, "amount": amount}]}


def bank(i: int, day: int, amount: str, doc: str, kind: str = "customer_payment") -> dict[str, Any]:
    return {"id": f"bank-{i:06d}", "sim_day": day, "date": f"2026-02-{day:02d}", "amount": amount, "source_document_id": doc, "memo": "Payment received from customer", "kind": kind}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    files: dict[str, list[dict[str, Any]]] = {
        "customers": [{"id": "c1", "payment_terms": "Net 7"}],
        "invoices": [inv("a", 1, 8), inv("b", 2, 9), inv("f", 10, 17, deposit="60.00", retainage="40.00"), inv("c", 20, 27), inv("d", 30, 37), inv("e", 200, 300)],
        "customer_payments": [pay("a", 5, "60.00"), pay("a", 8, "40.00"), pay("b", 25, "100.00"), pay("c", 26, "100.00"), pay("x", 9, "1.00")],
        "accounts": [{"id": "acct-rent", "name": "Rent and Lease"}, {"id": "acct-inventory", "name": "Inventory Asset"}],
        "items": [{"id": "item-1", "purchase_account_id": "acct-inventory"}],
        "bills": [bill(1, "v1", 3, account="acct-rent"), bill(2, "v1", 4, account="acct-rent"), bill(3, "v2", 5, item="item-1")],
        "bank_transactions": [bank(k, 1 + k // 3, f"{100 + k}.00", f"rcpt-{k}") for k in range(300)],
    }
    for company in ("company-001", "company-002"):
        d = tmp_path / "hackathon-100" / company / "rwm-v4" / "world-x"
        d.mkdir(parents=True)
        for name, docs in files.items():
            (d / f"{name}.yaml").write_text(yaml.safe_dump(docs), encoding="utf-8")
    (tmp_path / "hackathon-100" / "index.json").write_text("[]")
    return tmp_path / "hackathon-100"


def test_download_is_idempotent_without_network(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request

    def fail(*a: object, **k: object) -> None:
        pytest.fail("downloaded again")

    monkeypatch.setattr(urllib.request, "urlretrieve", fail)
    assert data.download(root.parent) == root


def test_download_unpacks_whole_or_not_at_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request

    src = tmp_path / "src" / "hackathon-100"
    src.mkdir(parents=True)
    (src / "index.json").write_text("[]")
    tgz = tmp_path / "ok.tar.gz"
    with tarfile.open(tgz, "w:gz") as t:
        t.add(src, arcname="hackathon-100")
    (bad := tmp_path / "bad.tar.gz").write_bytes(tgz.read_bytes()[:40])  # a download cut short
    root = tmp_path / "root"

    def serve(path: Path) -> None:
        monkeypatch.setattr(urllib.request, "urlretrieve", lambda url, dst: shutil.copy(path, dst))  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]

    serve(bad)
    with pytest.raises((tarfile.TarError, EOFError)):
        data.download(root)
    assert not (root / "hackathon-100").exists()  # a killed unpack never looks done
    serve(tgz)
    assert (data.download(root) / "index.json").is_file() and not (root / ".partial").exists()


def test_invoice_labels_censoring_and_history_known_that_day(root: Path) -> None:
    rows = data.company_rows("invoice_paid", data.companies(root)[0])
    by_ts = {r["ts"]: r for r in rows}
    assert [r["label"] for r in rows] == ["on_time", "late", "on_time", "never"]  # e (due day 300 > 365 - 90) is censored, whatever happens; f owes nothing: no row, no history
    assert by_ts["d020"]["rendered"].endswith("History: 1 paid on time, 1 open past due; last: open past due")  # b paid on day 25: unknown on day 20
    assert "1 paid on time, 2 paid late" not in by_ts["d030"]["rendered"] and "2 paid on time, 1 paid late" in by_ts["d030"]["rendered"]
    assert by_ts["d001"]["rendered"] == "Customer: Acme\nTerms: Net 7\nAmount: $100.00\nIssued d001, due d008\nHistory: none yet; last: none"
    assert {r["group"] for r in data.rows("invoice_paid", root)} == {"company-001", "company-002"}


def test_bill_account_from_the_line_or_the_items_purchase_account(root: Path) -> None:
    rows = data.company_rows("bill_account", data.companies(root)[0])
    assert [r["label"] for r in rows] == ["Rent and Lease", "Rent and Lease", "Inventory Asset"]
    assert all("LEAK" not in r["rendered"] for r in rows) and rows[2]["rendered"].endswith("Line: flat of buns")
    assert {r["label"] for r in data.rows("bill_account", root)} <= set(data.ACCOUNTS)


def test_bank_match_mess_is_deterministic_bounded_and_disclosed(root: Path) -> None:
    d = data.companies(root)[0]
    rows = data.company_rows("bank_match", d)
    assert rows == data.company_rows("bank_match", d)  # seeded by company id
    dups = [r for r in rows if r["label"] == "none"]
    assert len(rows) == 300 + len(dups) and dups  # 2% of 300 lines repeat (seeded: at least one here)
    for r in rows:
        head, _, docs = r["rendered"].partition("\nOpen documents:\n")
        listed = [line.split("  ")[0] for line in docs.split("\n")]
        assert len(listed) <= data.SHOWN and (r["label"] in listed) == (r["label"] != "none")  # a repeat's document is already matched
        amount, k = float(head.split("  ")[1]), int(r["label"].split("-")[1]) if r["label"] != "none" else None
        if k is not None:
            assert 0 <= (100 + k) - amount <= max(0.01, (100 + k) * 0.03) + 1e-9  # noise: a fee, at most 3% of the amount
    truncated = sum(r["rendered"].split("\n")[0].endswith("Payment rece") for r in rows)
    assert 0 < truncated < len(rows)
    for fault in ("amount noise", "memo truncation", "duplicates"):
        assert fault in (data.__doc__ or "")


def test_workflow_steps_render_with_the_data_functions() -> None:
    from quickbooks import workflow  # the first import (here or earlier in the session) declares the sites

    for sid in [s for s in list(registry._SITES) if s.startswith("quickbooks.")]:  # pyright: ignore[reportPrivateUsage]
        del registry._SITES[sid]  # pyright: ignore[reportPrivateUsage]
    importlib.reload(workflow)  # other suites clear the registry, so re-declare into a known state
    sites = {s.id: s for s in registry.sites() if s.workflow == "quickbooks"}
    assert set(sites) == {"quickbooks.invoice_paid", "quickbooks.bill_account", "quickbooks.bank_match"}
    assert sites["quickbooks.invoice_paid"].labels == ["on_time", "late", "never"] and sites["quickbooks.bill_account"].labels == data.ACCOUNTS
    h: Counter[str] = Counter({"paid late": 2})
    assert registry.get("quickbooks.invoice_paid").fn("Acme", "Net 7", "1.00", "d1", "d8", h, "paid late") == data.render_invoice("Acme", "Net 7", "1.00", "d1", "d8", h, "paid late")


def rows_of(*specs: tuple[str, str, str, str]) -> list[dict[str, str]]:
    return [{"group": g, "ts": ts, "rendered": rendered, "label": label} for g, ts, rendered, label in specs]


def test_vendor_rule_votes_on_earlier_bills_of_the_same_company() -> None:
    rows = rows_of(("A", "1", "Vendor: v\nx", "Rent"), ("A", "2", "Vendor: v\ny", "Rent"), ("A", "3", "Vendor: v\nz", "Utilities"),
                   ("A", "4", "Vendor: v\nw", "Rent"), ("B", "5", "Vendor: v\nx", "Rent"))
    assert baselines.vendor_rule(rows) == [(None, 0.0), ("Rent", 1.0), ("Rent", 1.0), ("Rent", 0.6667), (None, 0.0)]  # B is another company


def test_customer_rule_reads_the_history_line_and_closest_document_the_list() -> None:
    rows = rows_of(("A", "1", "History: 3 paid late, 1 paid on time, 2 open past due; last: paid late", "late"), ("A", "2", "History: none yet; last: none", "late"))
    assert baselines.customer_rule(rows) == [("late", 0.75), (None, 0.0)]
    bank = data.render_bank("2026-02-01", "99.00", "Pay", [("rcpt-1", "2026-02-01", "99.00", "customer_payment"), ("rcpt-2", "2026-02-01", "98.00", "customer_payment")])
    assert baselines.closest_document(rows_of(("A", "1", bank, "rcpt-1"), ("A", "2", data.render_bank("d", "1.00", "m", []), "none"))) == [("rcpt-1", 1.0), ("none", 0.5)]


def test_exact_cache_and_evaluate_certify_only_answers() -> None:
    train = rows_of(("A", "1", "same", "x"), ("A", "2", "same", "x"), ("A", "3", "same", "y"))
    assert baselines.exact_cache(train, rows_of(("B", "1", "same", "x"), ("B", "2", "new", "x"))) == [("x", 0.6667), (None, 0.0)]
    rows = rows_of(*[(f"g{k}", f"{k:04d}", f"r{k}", "x") for k in range(400)])
    preds: list[baselines.Pred] = [("x", 1.0) if k % 2 else (None, 0.0) for k in range(400)]
    sp = {"calib": list(range(200)), "test": list(range(200, 400))}
    res = baselines.evaluate(rows, preds, 0.05, sp=sp)
    assert res["answered"] == 0.5 and res["accuracy"] == 1.0 and res["certified_coverage"] == 0.5 and res["realized_error"] == 0.0  # abstentions never covered


def test_split_is_the_campaigns() -> None:
    campaign: Any = pytest.importorskip("dreaming_trainer.campaign")
    rows = rows_of(*[(f"company-{k % 17:03d}", f"2026-01-{k % 28 + 1:02d}", json.dumps(k), "x") for k in range(500)])
    assert baselines.split(rows) == campaign.split(rows)
