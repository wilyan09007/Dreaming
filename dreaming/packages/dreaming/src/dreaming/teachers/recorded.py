"""Pre-recorded teacher answers for fast-forward mode.

Two thousand runs in eighteen seconds cannot make eight thousand frontier calls, so the ledger's teacher answers
are recorded once (with the pinned live teacher) and served from a file. Every served row says `recorded=True`
and the dashboard labels the batch as pre-recorded ("nothing shown is faked"). Misses go to the live teacher
when one is given and are appended, so the file only grows.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from ..contracts import CallSite, TeacherAnswer
from ..hashing import prompt_sha, sha
from . import Teacher


class RecordedTeacher:
    recorded = True  # Runtime.recorded reads this: the page labels the batch as pre-recorded

    def __init__(self, path: Path, live: Teacher | None = None) -> None:
        self.path, self.live = path, live
        self._answers: dict[str, TeacherAnswer] = {}
        self._lock = threading.Lock()  # bookkeeper.record asks N-wide; one line per append, whole
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                d = json.loads(line)
                self._answers[d["key"]] = TeacherAnswer.model_validate(d["answer"])

    @staticmethod
    def key(site: CallSite, rendered: str) -> str:
        return f"{site.id}:{site.teacher.model}:{sha(site.teacher.params)[:8]}:{prompt_sha(rendered)}"

    def ask(self, site: CallSite, rendered: str) -> TeacherAnswer:
        k = self.key(site, rendered)
        hit = self._answers.get(k)
        if hit is not None:
            return hit.model_copy(update={"recorded": True})
        if self.live is None:
            raise LookupError(f"no recorded answer for {site.id} and no live teacher")
        ans = self.live.ask(site, rendered)
        with self._lock:
            self._answers[k] = ans
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": k, "answer": ans.model_dump(mode="json")}) + "\n")
        return ans
