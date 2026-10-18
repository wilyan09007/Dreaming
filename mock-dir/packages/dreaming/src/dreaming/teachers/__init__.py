"""Teachers: whatever produces the true answer for a call site. A frontier model, a tool, or a human.

    openai_compat.py   OpenRouter and ASI:One through the openai SDK, provider pinned, cost from usage
    recorded.py        pre-recorded answers for fast-forward mode (labelled as such in every row)

The teacher's prompt for a typed step is rendered here from the site's labels and descriptions, so the student
and the teacher read the same input text and the prompt hash means one thing.
"""

from __future__ import annotations

from typing import Protocol

from ..contracts import CallSite, TeacherAnswer


class Teacher(Protocol):
    def ask(self, site: CallSite, rendered: str) -> TeacherAnswer: ...


def messages(site: CallSite, rendered: str) -> list[dict[str, str]]:
    """The chat request for a typed step: the label menu as the system prompt, the rendered input as the user turn."""
    menu = "\n".join(f"  {lab} — {site.descriptions.get(lab, '')}".rstrip(" —") for lab in site.labels or [])
    system = (
        f"You are the `{site.name}` step of the `{site.workflow}` workflow.\n"
        f"Pick exactly one label. Answer with the label name only, nothing else: no description, no punctuation.\n{menu}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": rendered}]


def parse_label(text: str, labels: list[str] | None) -> str | None:
    """The label on the first non-empty line, case-insensitive; None counts as a wrong answer.

    Exact match first; else the line may start with a label followed by a separator ("Meals — restaurants, coffee"): step-5-preview
    echoed the menu line in 4 of 10 calls (measured 2026-10-17), and a lenient reader beats a prompt the model ignores. Longest
    label first, so "Office supplies" is never read as "Office". Anything else (two labels, prose) stays None.
    """
    if labels is None:
        return text
    first = next((ln.strip().strip("`*\"'. ") for ln in text.splitlines() if ln.strip()), "").lower()
    by_lower = {lab.lower(): lab for lab in labels}
    if first in by_lower:
        return by_lower[first]
    for low in sorted(by_lower, key=len, reverse=True):
        rest = first[len(low) :].lstrip() if first.startswith(low) else None
        if rest is not None and (rest == "" or rest[0] in "—-:(,"):  # a separator, never a second word
            return by_lower[low]
    return None
