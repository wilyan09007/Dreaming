"""bookkeeper.record offline: the rule teacher stands in for OpenRouter and counts what it is asked; the answers file is a
temp file. No network, no keys, no torch (the encoder is stubbed either way, so an installed torch changes nothing)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from bookkeeper import offline, record
from dreaming.contracts import CallSite, TeacherAnswer
from dreaming.teachers.recorded import RecordedTeacher

N = 80  # per seed: past watch_after (60), so the recording passes the point where students would start to train
SEED_LINE = re.compile(r"seed (\d): 80 transactions, (\d+) new answers, (\d+) already recorded, \$([\d.]+)")


class FakeLive:
    """The rule teacher, remembering every question: what OpenRouter would have been paid for."""

    def __init__(self) -> None:
        self.asked: list[tuple[CallSite, str]] = []

    def ask(self, site: CallSite, rendered: str) -> TeacherAnswer:
        self.asked.append((site, rendered))
        return offline.RuleTeacher().ask(site, rendered)


class NoLive:
    def ask(self, site: CallSite, rendered: str) -> TeacherAnswer:
        raise AssertionError(f"asked the live teacher about {site.id}")


class DownLive:
    def ask(self, site: CallSite, rendered: str) -> TeacherAnswer:
        raise ConnectionError("openrouter unreachable")


class NoTorch:
    def __init__(self, cache_path: Path | None = None) -> None:
        raise ImportError("No module named 'torch'")


@pytest.fixture(autouse=True)
def without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dreaming.predictors.student.SentenceEncoder", NoTorch)


def seed_lines(text: str) -> list[tuple[str, ...]]:
    """(seed, new, already recorded, dollars) from each per-seed line the command printed."""
    matches = (SEED_LINE.fullmatch(ln) for ln in text.splitlines())
    return [m.groups() for m in matches if m]


def lines(out: Path) -> list[dict[str, object]]:
    return [json.loads(ln) for ln in out.read_text().splitlines()]


def test_one_answer_per_distinct_key_then_a_rerun_needs_no_live_teacher(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out, live = tmp_path / "answers.jsonl", FakeLive()
    assert record.main([str(N), "--out", str(out)], live=live) == 0
    printed = capsys.readouterr()

    keys = [str(row["key"]) for row in lines(out)]
    asked = {RecordedTeacher.key(site, rendered) for site, rendered in live.asked}
    assert len(keys) == len(set(keys)) == len(asked) == len(live.asked)  # (site, params, prompt): each paid for once, none twice
    assert {k.split(":")[0] for k in keys} == {f"bookkeeper.{s}" for s in ("categorize", "match_invoice", "anything_odd")}
    assert len(keys) > 2 * N * len(record.SEEDS)  # a thousand-odd questions, not a handful: the ledger really went through the workflow

    per_seed = seed_lines(printed.out)
    assert [seed for seed, *_ in per_seed] == ["0", "1", "2"]
    assert sum(int(new) for _, new, _, _ in per_seed) == len(keys)
    assert sum(float(cost) for *_, cost in per_seed) == pytest.approx(0.01 * len(keys))  # the rule teacher bills a cent an answer
    assert "torch" in printed.err  # the note: no embedding cache without the student extra

    before = out.read_bytes()
    assert record.main([str(N), "--out", str(out)], live=NoLive()) == 0  # NoLive would fail the run on its first question
    assert out.read_bytes() == before
    assert [new for _, new, _, _ in seed_lines(capsys.readouterr().out)] == ["0", "0", "0"]  # no new answers in any seed
    replay = RecordedTeacher(out)  # and with no live teacher at all, every question the first run asked is answered from the file
    assert all(replay.ask(site, rendered).recorded for site, rendered in live.asked)


def test_a_dead_live_teacher_is_a_failed_command_not_an_empty_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "answers.jsonl"
    assert record.main(["5", "--out", str(out)], live=DownLive()) == 1  # Runtime.run turns each failure into a failed run: look at the events
    assert "ConnectionError: openrouter unreachable" in capsys.readouterr().err
    assert not out.exists() or out.read_text() == ""


def test_no_key_and_no_injected_teacher_says_so_before_touching_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTERKEY", raising=False)
    out = tmp_path / "deep" / "answers.jsonl"
    with pytest.raises(SystemExit, match="OPENROUTERKEY"):
        record.main(["5", "--out", str(out)])
    assert not out.parent.exists()


def test_the_student_encoder_is_built_over_the_embedding_cache_and_saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                       capsys: pytest.CaptureFixture[str]) -> None:
    built: list[Path | None] = []
    saves: list[str] = []

    class FakeStudentEncoder(offline.CharEncoder):
        def __init__(self, cache_path: Path | None = None) -> None:
            built.append(cache_path)

        def save_cache(self) -> None:  # what Runtime.fast_forward calls when it finishes a seed
            saves.append("saved")

    cache = tmp_path / "embeddings.npz"
    monkeypatch.setenv("EMBEDDING_CACHE", str(cache))
    monkeypatch.setattr("dreaming.predictors.student.SentenceEncoder", FakeStudentEncoder)
    assert record.main([str(N), "--out", str(tmp_path / "answers.jsonl")], live=FakeLive()) == 0
    assert built == [cache] and len(saves) == len(record.SEEDS)
    assert "torch" not in capsys.readouterr().err
