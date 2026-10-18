"""The `memory` candidate offline: fit on keyword-labelled rows, load, read back byte-stably, forget a row, refuse a foreign encoder;
and seed rows written once by a fake LLM, cached by schema, read in as ordinary rows."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from conftest import LABELS, WORDS, CharEncoder, KeywordTeacher, transactions
from dreaming.contracts import CallSite, Decision, Kind, TraceRow
from dreaming.hashing import sha, sha_bytes
from dreaming.predictors import memory, profile_sha, seeds
from dreaming.predictors.candidates import CANDIDATES, for_site
from dreaming.predictors.memory import Memory

SITE = CallSite(id="books.categorize", workflow="books", name="categorize", kind=Kind.model,
                labels=[*LABELS, "Other"], descriptions={"Software": "apps", "Meals": "food", "Travel": "trips", "Other": "the rest"})  # "Other" has no rows


def make_rows(n: int) -> list[TraceRow]:
    teacher, rows = KeywordTeacher(), list[TraceRow]()
    for i, txn in enumerate(transactions(n), 1):
        text = f"Transaction: {txn['d']}, ${txn['amt']}"
        ans = teacher.ask(SITE, text)
        rows.append(TraceRow(
            id=f"r{i}", run_id="run", i=i, call_site=SITE.id, kind=Kind.model, request=text, request_sha=sha(text),
            response=ans.text, decision=Decision.called, teacher=ans, latency_ms=800, cost_usd=0.01,
            ts=datetime(2026, 10, 17, tzinfo=UTC),
        ))
    return rows


def test_memory_reads_back_its_rows_byte_stably_and_refuses_a_foreign_encoder():
    rows, enc = make_rows(60), CharEncoder()
    assert [c.name for c in for_site(SITE.model_copy(update={"labels": LABELS}))] == ["head", "memory"]

    f = CANDIDATES["memory"].fit(SITE, rows, enc)
    assert f.n_train == 60 and f.holdout_accuracy >= 0.9 and f.weights_sha == sha_bytes(f.data)
    again = CANDIDATES["memory"].fit(SITE, [*reversed(rows), rows[0]], enc)  # another order, a duplicate id: the same rows
    assert again.data == f.data  # same rows -> same bytes -> same weights_sha (training-plan §4.1)
    m = CANDIDATES["memory"].load(SITE, f.data, enc)
    assert isinstance(m, Memory) and m.weights_sha == f.weights_sha and m.ids == sorted(r.id for r in rows) and m.beta in (5, 10, 20, 40, 80, 160, 320, 640)
    assert m.env_sha == profile_sha(encoder=enc.env_sha, candidate="memory", beta=m.beta)

    preds = [m.predict(str(r.request)) for r in rows]
    assert sum(p.label == r.teacher.label for p, r in zip(preds, rows, strict=True) if r.teacher) >= 54  # >= 90% of 60
    assert preds == [m.predict(str(r.request)) for r in rows] == [m.predict(str(r.request), fresh=True) for r in rows]
    assert preds == [CANDIDATES["memory"].load(SITE, f.data, enc).predict(str(r.request)) for r in rows]  # a fresh load, same answers
    assert all(p.scores["Other"] == 0.0 and 0.0 <= p.confidence <= 1.0 and p.confidence == round(p.confidence, 4) for p in preds)

    forgot = memory.load(SITE, memory.fit(SITE, rows[1:], enc).data, enc)  # delete = refit without the row
    assert forgot.weights_sha != m.weights_sha and "r1" not in forgot.ids and len(forgot.ids) == 59

    other = CharEncoder()
    other.model_sha = "charenc-v2"
    with pytest.raises(ValueError, match="charenc-v1.*charenc-v2|charenc-v2.*charenc-v1"):
        CANDIDATES["memory"].load(SITE, f.data, other)


def test_seed_rows_are_written_once_per_schema_and_seed_a_memory_with_no_real_rows(tmp_path: Path):
    prompts: list[str] = []

    def fake_llm(prompt: str) -> str:  # asi1-mini's real habit: keys echo "Label: description"; "Other" is left out
        prompts.append(prompt)
        examples = {lab: [f"Transaction: {ws[k % len(ws)]} *{1000 + k}, $12.00" for k in range(12)] for lab, ws in WORDS.items()}  # 12: capped at 10
        return "Sure:\n" + json.dumps({f"{lab}: {SITE.descriptions[lab]}": v for lab, v in examples.items()})

    rows = seeds.write(SITE, fake_llm, "Transaction: ADOBE *WEB, $60.00", root=tmp_path)
    assert len(prompts) == 3  # "Other" was missing every time: three tries, then no rows for it rather than a placeholder
    assert "Transaction: ADOBE *WEB, $60.00" in prompts[0] and '"Software", "Meals", "Travel", "Other"' in prompts[0]
    assert len(rows) == 30 and {r.teacher.label for r in rows if r.teacher} == set(LABELS)
    assert all(r.id.startswith("seed:") and r.run_id == "seed" and r.phase.value == "awake" for r in rows)  # trains, never calibrates
    assert seeds.write(SITE, fake_llm, root=tmp_path) == rows == seeds.rows(SITE, tmp_path) and len(prompts) == 3  # cached
    assert seeds.rows(SITE.model_copy(update={"labels": LABELS}), tmp_path) == []  # another schema: not these rows

    enc = CharEncoder()
    day_zero = CANDIDATES["memory"].load(SITE, CANDIDATES["memory"].fit(SITE, rows, enc).data, enc)
    real = make_rows(60)
    assert sum(day_zero.predict(str(r.request)).label == r.teacher.label for r in real if r.teacher) >= 54
    both = CANDIDATES["memory"].fit(SITE, [*rows, *real], enc)
    assert both.n_train == 90 and memory.load(SITE, both.data, enc).ids[:2] == ["r1", "r10"]  # sorted ids: "r..." < "seed:..."
