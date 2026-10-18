"""Everything the bookkeeper needs to run with no network: a rule-based teacher, a hashed-trigram encoder, one Runtime builder.

The RuleTeacher stands in for the frontier model in tests and in `python -m bookkeeper.fast_forward`. It answers from the
mock world's ground truth (tools.MERCHANTS), reading only the rendered text a real teacher would read, so a student can
learn the same function from the same text. It is labelled provider="rule" in every row; nothing here pretends to be a model.
"""

from __future__ import annotations

import re
import zlib

import numpy as np
from dreaming.contracts import CallSite, TeacherAnswer
from dreaming.predictors import Loader
from dreaming.predictors.student import Vec
from dreaming.predictors.train import LocalTrainer
from dreaming.runtime import Runtime
from dreaming.store import MemoryStore, Store

from . import tools, workflow

NO_INVOICE = {"Meals", "Travel", "Payroll", "Other"}  # spend that does not come with an invoice
ODD_X = 10  # a planted anomaly is 20-100x the usual amount; routine spend never reaches 10x (lognormal sigma 0.5)
_BY_NAME = {m.name: m for m in tools.MERCHANTS.values()}
_ODD = re.compile(r"Vendor: (?P<vendor>.+) \((?P<category>[^()]*)\), \$(?P<amount>[\d.]+), invoice: ")


def _categorize(rendered: str) -> str:
    token = rendered.removeprefix("Transaction: ").split(" *")[0]
    return tools.MERCHANTS[token].category if token in tools.MERCHANTS else "Other"


def _match_invoice(rendered: str) -> str:
    fields = dict(line.split(": ", 1) for line in rendered.splitlines())
    if fields["Category"] in NO_INVOICE:
        return "no invoice expected"
    return "none" if fields["Open invoices"] == "none" else "match"


def _anything_odd(rendered: str) -> str:
    m = _ODD.match(rendered)
    assert m is not None, rendered
    vendor = _BY_NAME.get(m["vendor"])
    return "yes" if vendor is None or float(m["amount"]) > ODD_X * vendor.usual else "no"  # unknown vendor, or a spike


RULES = {"categorize": _categorize, "match_invoice": _match_invoice, "anything_odd": _anything_odd}


class RuleTeacher:
    """Deterministic 'frontier model' for the three typed sites. The router site's cheap configuration is answered by the
    same rule (ConfigPredictor calls ask() with the same site id and the CHEAP spec), so the cheap config always agrees."""

    answers = "offline rule teacher answers"  # what Runtime.fast_forward's batch.started message says instead of "live teacher answers"

    def ask(self, site: CallSite, rendered: str) -> TeacherAnswer:
        label = RULES[site.name](rendered)
        return TeacherAnswer(text=label, label=label, provider="rule", model="rule", cost_usd=0.01, latency_ms=800)


class CharEncoder:
    """Bag of character trigrams, hashed into 64 dims, L2-normalized. Byte-deterministic across processes (crc32, not hash())."""

    model_sha = "charenc-v1"
    env_sha = "test-env"

    def embed(self, text: str, *, fresh: bool = False) -> Vec:
        v = np.zeros(64, np.float32)
        t = text.upper()
        for k in range(len(t) - 2):
            v[zlib.crc32(t[k : k + 3].encode()) % 64] += 1.0
        n = float(np.linalg.norm(v))
        return v / n if n else v


def build_runtime(store: Store | None = None, budget: float = 0.05) -> Runtime:
    """MemoryStore + RuleTeacher + LocalTrainer + Loader(CharEncoder), bound to the sites workflow.py declared."""
    store = store or MemoryStore()
    teacher = RuleTeacher()
    loader = Loader(encoder=CharEncoder(), teacher=teacher)  # a router site's predictor needs the teacher
    rt = Runtime(store, teacher, LocalTrainer(store, loader), loader, budget=budget)
    rt.bind()
    rt.workflow = workflow.run_transaction  # same wiring as apps/api: the function one run executes
    return rt
