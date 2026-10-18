"""Offline checks for the campaign (campaign.py) and the monitor (monitor.py): the split, the matrix, one rung end to end, the pruning rules.

No Modal, no Postgres: `run_rung` takes the encoder and the rows directly, so the head candidate runs on a bag-of-trigrams fake in a second.
"""

from __future__ import annotations

import json
import zlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from dreaming.contracts import CallSite, Hashes, Kind, Prediction, TeacherAnswer
from dreaming.hashing import sha
from dreaming.predictors.candidates import CANDIDATES, Candidate, Fitted
from dreaming.predictors.student import Vec
from dreaming.teachers.recorded import RecordedTeacher
from dreaming_trainer import campaign, monitor  # installed by `uv sync --all-packages`

LABELS = ["Software", "Meals", "Travel"]
MERCHANTS = {"ADOBE": "Software", "NOTION": "Software", "FIGMA": "Software", "BLUE BOTTLE": "Meals", "DOORDASH": "Meals", "SWEETGREEN": "Meals",
             "UBER": "Travel", "LYFT": "Travel", "DELTA": "Travel", "AMTRAK": "Travel"}
T0 = datetime(2026, 1, 1, tzinfo=UTC)


class CharEncoder:
    """Bag of character trigrams hashed into 256 dims (the same fake as packages/dreaming/tests/conftest.py, which cannot be imported from here)."""

    model_sha = "charenc-v1"
    env_sha = "test-env"

    def embed(self, text: str, *, fresh: bool = False) -> Vec:
        v = np.zeros(256, np.float32)
        t = text.upper()
        for k in range(len(t) - 2):
            v[zlib.crc32(t[k : k + 3].encode()) % 256] += 1.0
        n = float(np.linalg.norm(v))
        return v / n if n else v


def ledger(n: int) -> list[dict[str, str]]:
    names = list(MERCHANTS)
    return [{"rendered": f"Transaction: {names[i % len(names)]} *{1000 + i}, ${i % 90 + 5}.00", "label": MERCHANTS[names[i % len(names)]],
             "group": names[i % len(names)], "ts": (T0 + timedelta(hours=i)).isoformat()} for i in range(n)]


def site(kind: Kind = Kind.model) -> CallSite:
    return CallSite(id="w.cat", workflow="w", name="cat", kind=kind, labels=LABELS if kind == Kind.model else None)


# ---------------------------------------------------------------- split and the matrix


def test_split_is_a_deterministic_partition_by_group_and_time() -> None:
    rows = [{**r, "group": f"m{i % 37}"} for i, r in enumerate(ledger(400))]
    a, b = campaign.split(rows), campaign.split(rows)
    assert a == b
    assert sorted(i for ids in a.values() for i in ids) == list(range(len(rows)))  # every row exactly once
    groups = {name: {rows[i]["group"] for i in a[name]} for name in ("train", "calib", "test")}
    assert not (groups["train"] & groups["calib"]) and not (groups["train"] & groups["test"]) and not (groups["calib"] & groups["test"])
    assert len(a["forward"]) == len(rows) // 5
    cut = min(rows[i]["ts"] for i in a["forward"])
    assert all(rows[i]["ts"] < cut for name in ("train", "calib", "test") for i in a[name])  # forward is later than everything else
    assert a["train"] and a["calib"] and a["test"]


def test_split_ignores_input_order() -> None:
    rows = ledger(200)
    shuffled = list(reversed(rows))
    one, two = campaign.split(rows), campaign.split(shuffled)
    assert {k: {rows[i]["rendered"] for i in v} for k, v in one.items()} == {k: {shuffled[i]["rendered"] for i in v} for k, v in two.items()}


def test_eval_sets_are_frozen_by_content() -> None:
    rows = ledger(100)
    sp = campaign.split(rows)
    sets = campaign.eval_sets("w.cat", rows, sp)
    assert [s.split for s in sets] == ["calib", "test", "forward"]
    assert sets == campaign.eval_sets("w.cat", rows, sp) and len({s.sha for s in sets}) == 3
    assert sets[1].row_ids == [str(i) for i in sp["test"]] and sets[1].labels == [rows[i]["label"] for i in sp["test"]]
    relabelled = [{**r, "label": "Other"} for r in rows]  # a re-prepared site: same line numbers, new labels -> new eval sets
    assert {s.sha for s in campaign.eval_sets("w.cat", relabelled, sp)}.isdisjoint(s.sha for s in sets)
    assert campaign.eval_sets("w.other", rows, sp)[0].sha != sets[0].sha  # line numbers repeat across sites; the sha must not


@pytest.mark.parametrize("finetune", [True, False])
def test_cells_matrix(monkeypatch: pytest.MonkeyPatch, finetune: bool) -> None:
    monkeypatch.setattr(campaign, "_finetune_ported", lambda: finetune)
    got = campaign.cells("night", [])
    assert len(got) == (34 if finetune else 30)
    assert len({c["id"] for c in got}) == len(got)
    first = campaign.cells("night", ["categorize"])[:3]
    assert [c["id"] for c in first] == [f"night:bookkeeper.categorize:head:real:s{k}" for k in range(3)]
    assert {c["candidate"] for c in got if c["call_site"] in campaign.TOOL_SITES} == {"exact"}
    gpu = [c for c in got if c["where"] == "modal:T4"]
    assert len(gpu) == (4 if finetune else 0) and all(c["candidate"] == "finetune" and c["seed"] == 0 for c in gpu)
    assert {c["call_site"] for c in campaign.cells("night", ["bank_match", "quickbooks.bill_account"])} == {"quickbooks.bank_match", "quickbooks.bill_account"}
    with pytest.raises(ValueError, match="unknown campaign site"):
        campaign.cells("night", ["nope"])
    # seed rows on the Volume for one site add its seed+real and seed variants: 6 CPU cells each, plus a finetune cell per variant
    seeded = campaign.cells("t", ["categorize", "match_invoice", "vendor_lookup"], seeded={"bookkeeper.categorize"})
    by_site = {s: [c for c in seeded if c["call_site"] == s] for s in campaign.SITES[:3]}
    assert [len(by_site[s]) for s in campaign.SITES[:3]] == ([21, 7, 3] if finetune else [18, 6, 3])
    assert {c["variant"] for c in by_site["bookkeeper.categorize"]} == {"real", "seed+real", "seed"} and {c["variant"] for c in by_site["bookkeeper.match_invoice"]} == {"real"}
    assert all(c["config"]["rungs"] == ([0] if c["variant"] == "seed" else [60, 200, 500, 2000]) for c in seeded)
    assert len({c["id"] for c in seeded}) == len(seeded) and len(campaign.cells("t", [], seeded=set(campaign.SITES))) == (90 if finetune else 78)


def test_labels_for_joins_the_stepfun_record_and_drops_the_rest(tmp_path: Path) -> None:
    cs, rows = site(), ledger(4)
    trace = campaign.as_trace_rows(cs, rows)  # their own teacher says MERCHANTS; the record says otherwise for two of them
    lines = [{"key": RecordedTeacher.key(cs, rows[0]["rendered"]), "answer": TeacherAnswer(text="Travel", label="Travel", provider="stepfun", model="m").model_dump(mode="json")},
             {"key": RecordedTeacher.key(cs, rows[1]["rendered"]), "answer": TeacherAnswer(text="??", label=None, provider="stepfun", model="m").model_dump(mode="json")}]
    (tmp_path / "r.jsonl").write_text("".join(json.dumps(d) + "\n" for d in lines), encoding="utf-8")
    got = campaign.labels_for(cs, trace, RecordedTeacher(tmp_path / "r.jsonl"))
    assert got == [{**rows[0], "label": "Travel", "group": campaign.group_of(cs.id, rows[0]["rendered"])}]  # row 1 unparsed, 2 and 3 unrecorded
    assert [r["label"] for r in campaign.labels_for(site(Kind.tool), trace, None)] == [r["label"] for r in rows]  # a tool site keeps its exact answers


def test_seed_rows_reads_b1s_file_for_the_current_schema(tmp_path: Path) -> None:
    cs = site()
    key = sha([cs.labels, cs.descriptions])
    lines = [{"key": key, "id": "seed:Meals:0", "label": "Meals", "text": "Transaction: CAFE, $4.00", "model": "asi1-mini"},
             {"key": "stale", "id": "seed:Meals:0", "label": "Meals", "text": "old schema", "model": "asi1-mini"}]
    (tmp_path / f"{cs.id}.jsonl").write_text("".join(json.dumps(d) + "\n" for d in lines), encoding="utf-8")
    assert campaign.seed_rows(cs, tmp_path) == [{"rendered": "Transaction: CAFE, $4.00", "label": "Meals"}]
    assert campaign.seed_rows(site(Kind.tool), tmp_path) == [] and campaign.seed_rows(CallSite(id="w.none", workflow="w", name="none", kind=Kind.model, labels=["a"]), tmp_path) == []


def test_site_data_refuses_a_site_re_prepared_under_the_campaign(tmp_path: Path) -> None:
    rows, seeds = ledger(5), [{"rendered": "Transaction: CAFE, $4.00", "label": "Meals"}]
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "seeds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in seeds))
    np.save(tmp_path / "embeddings.npy", np.zeros((6, 4), np.float32))
    meta = json.dumps({"site_id": "w.cat", "labels": LABELS, "encoder_sha": "", "env_sha": "", "n": 5, "n_seeds": 1})
    (tmp_path / "meta.json").write_text(meta)
    want = campaign.sha_bytes(meta.encode())
    got = campaign.site_data(tmp_path, want)
    assert got[0]["n"] == 5 and got[1] == rows and got[2] == seeds and campaign.site_data(tmp_path, None)[1] == rows  # no expectation: counts alone
    with pytest.raises(RuntimeError, match="changed since launch"):
        campaign.site_data(tmp_path, "someone else's prepare")
    np.save(tmp_path / "embeddings.npy", np.zeros((5, 4), np.float32))  # the mixed state of 2026-10-18: old vectors, new seeds
    with pytest.raises(RuntimeError, match="disagree"):
        campaign.site_data(tmp_path, want)


def test_group_and_rungs() -> None:
    assert campaign.group_of("bookkeeper.categorize", "Transaction: ADOBE *1234, $5.00") == "Transaction: ADOBE"
    assert campaign.group_of("bookkeeper.match_invoice", "Vendor: Adobe\nCategory: Software") == "Vendor: Adobe"
    assert campaign.rungs_for(10_000) == [60, 200, 500, 2000] and campaign.rungs_for(300) == [60, 200] and campaign.rungs_for(40) == [40]
    assert campaign.rungs_for(10_000, gpu=True) == [500, 2000] and campaign.rungs_for(300, gpu=True) == [200]


# ---------------------------------------------------------------- the summary math and one rung


def test_summarize_on_a_hand_made_eval() -> None:
    code = np.array([0] * 40 + [1] * 10 + [2] * 4, np.int8)
    conf = np.array([0.9] * 40 + [0.9] * 10 + [0.5] * 4, np.float64)
    ok = np.array([True] * 40 + [True] * 9 + [False] + [True, True, False, False], bool)
    s = campaign.summarize(conf, ok, code, n_train=60)
    assert (s["acc_test"], s["acc_forward"]) == (0.9, 0.5)
    assert (s["n_calib"], s["n_test"], s["n_forward"], s["n_train"]) == (40, 10, 4, 60)
    assert s["coverage_02"] == 0.0 and s["coverage_05"] == 0.0 and s["bound_05"] == 1.0  # 40 clean rows bound the error at 7.2% > 5%: nothing certifies
    assert s["coverage_10"] == 1.0 and s["err_10"] == pytest.approx(0.1)  # at 10% the one cutoff (0.9) certifies; the test slice shows its one miss
    assert s["coverage_20"] == 1.0 and s["ece"] == pytest.approx(0.0)  # test: confidence 0.9 and accuracy 0.9
    assert (s["cov_fwd_05"], s["err_fwd_05"]) == (0.0, 0.0)  # nothing certified at 5%, so the forward slice gets no cutoff either


def test_run_rung_head_end_to_end() -> None:
    rows = ledger(600)
    sp = campaign.split(rows)
    cs = site()
    trace = campaign.as_trace_rows(cs, [rows[i] for i in sp["train"]])
    r = campaign.run_rung(cs, CANDIDATES["head"], trace[:60], rows, sp, LABELS, CharEncoder())
    n_eval = len(sp["calib"]) + len(sp["test"]) + len(sp["forward"])
    assert r.stable and r.fitted.n_train > 0 and r.head_ms >= 0.0 and r.embed_ms == 0.0
    assert r.arrays["scores"].shape == (n_eval, 3) and r.arrays["scores"].dtype == np.float16
    assert r.arrays["idx"].dtype == np.int32 and list(r.arrays["split"][:1]) == [0] and r.arrays["split"][-1] == 2
    assert np.array_equal(r.arrays["conf"], r.arrays["conf2"]) and np.array_equal(r.arrays["pred"], r.arrays["pred2"])
    assert r.summary["n_test"] == len(sp["test"]) and all(0.0 <= r.summary[k] <= 1.0 for k in ("acc_test", "coverage_05", "ece"))
    assert r.summary["coverage_02"] <= r.summary["coverage_05"] <= r.summary["coverage_10"] <= r.summary["coverage_20"]


def test_run_rung_exact_on_a_tool_site() -> None:
    rows = ledger(300)
    sp = campaign.split(rows)
    cs = site(Kind.tool)
    trace = campaign.as_trace_rows(cs, [rows[i] for i in sp["train"]])
    r = campaign.run_rung(cs, CANDIDATES["exact"], trace, rows, sp, sorted(set(LABELS)), None)
    assert r.stable and r.fitted.n_train == len(sp["train"])
    assert r.arrays["scores"].shape[1] == 3  # one-hot of the shares the memory reports


def test_an_unstable_candidate_is_flagged() -> None:
    class Flaky:
        weights_sha, env_sha = "w", "e"

        def predict(self, rendered: str, *, fresh: bool = False) -> Prediction:
            return Prediction(label="Meals", confidence=0.5 if fresh else 0.25, hashes=Hashes(prompt="p"))

    flaky = Candidate("flaky", lambda s, r, e: Fitted(b"x", len(r), 1.0, 0.0), lambda s, d, e: Flaky(), needs_encoder=False)
    rows = ledger(100)
    r = campaign.run_rung(site(), flaky, campaign.as_trace_rows(site(), rows[:10]), rows, campaign.split(rows), LABELS, None)
    assert not r.stable


def test_volume_encoder_looks_vectors_up() -> None:
    emb = np.eye(3, dtype=np.float32)
    enc = campaign.VolumeEncoder(["a", "b", "c"], emb, "sha", "env")
    assert np.array_equal(enc.embed("b", fresh=True), emb[1]) and enc.model_sha == "sha"


# ---------------------------------------------------------------- the monitor


def row(run_id: str, cand: str, seed: int, rung: int | None, *, site_: str = "s", variant: str = "real", status: str = "done", stable: bool = True,
        cov: float | None = 0.5, err: float = 0.0, where: str = "modal:cpu", fc: str | None = None) -> dict[str, Any]:
    return {"run_id": run_id, "call_site": site_, "candidate": cand, "variant": variant, "seed": seed, "where": where, "status": status, "note": None,
            "seconds": 1.0, "fc_id": fc, "rung": rung, "bytes": 10, "stable": stable if rung else None, "coverage_05": cov if rung else None,
            "coverage_20": 0.9, "err_05": err, "acc_test": 0.8}


def test_rule_1_hard_fails_at_any_rung() -> None:
    rows = [row("a", "head", 0, 60), row("a", "head", 0, 200, stable=False), row("b", "head", 1, 60, err=0.08), row("c", "head", 2, 60),
            row("d", "head", 0, 60, status="pruned", stable=False), row("e", "head", 0, None, status="failed")]
    got = dict(monitor.decide(rows))
    assert set(got) == {"a", "b"} and "not byte-stable at rung 200" in got["a"] and "0.080" in got["b"]


def test_rule_2_prunes_the_family_by_its_best_seed_and_waits_for_two_seeds() -> None:
    def rung500(head: list[float], mem: list[float]) -> list[dict[str, Any]]:
        return ([row(f"h{k}", "head", k, 500, cov=c) for k, c in enumerate(head)] + [row(f"m{k}", "memory", k, 500, cov=c) for k, c in enumerate(mem)])

    got = dict(monitor.decide(rung500([0.60, 0.40, 0.55], [0.10, 0.30, 0.20])))  # best seeds 0.60 and 0.30: median 0.45, memory is 0.15 under
    assert set(got) == {"m0", "m1", "m2"} and "rule 2" in got["m1"]
    assert monitor.decide(rung500([0.60, 0.40, 0.55], [0.40, 0.30, 0.20])) == []  # best seed 0.40 is within 0.10 of 0.5
    one_seed = rung500([0.60, 0.40, 0.55], [0.10]) + [row("m1", "memory", 1, None, status="running"), row("m2", "memory", 2, 200)]
    assert monitor.decide(one_seed) == []  # one seed alone never prunes a family that was launched with three
    assert monitor.decide(rung500([0.60], [])) == [] and monitor.decide([row("h0", "head", 0, 200, cov=0.9), row("m0", "memory", 0, 200, cov=0.0)]) == []


def test_rule_2_can_prune_a_one_seed_gpu_family_and_stays_inside_a_site() -> None:
    rows = [row("h0", "head", 0, 2000, cov=0.8), row("m0", "memory", 0, 2000, cov=0.7), row("f0", "finetune", 0, 2000, cov=0.4, where="modal:T4", fc="fc-1"),
            row("x0", "head", 0, 2000, site_="other", cov=0.0)]
    assert [r for r, _ in monitor.decide(rows)] == ["f0"]  # median 0.7, finetune 0.3 under; `other` has one candidate and is left alone


def test_latest_keeps_the_newest_rung_per_run_and_render_prints_it() -> None:
    rows = [row("a", "head", 0, 60, cov=0.1), row("a", "head", 0, 200, cov=0.3), row("b", "memory", 0, None, status="queued", site_="t")]
    t = monitor.latest(rows)
    assert [r["rung"] for r in t["s"]] == [200] and t["s"][0]["coverage_05"] == 0.3 and t["t"][0]["rung"] is None
    text = monitor.render(t)
    assert "s\n" in text and "0.300" in text and "queued" in text
