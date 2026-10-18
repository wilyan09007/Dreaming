"""Seed rows: LLM-written example inputs per label, written once from a typed site's schema (docs/training-plan.md §2.1).

    write(site, ask, example=None) -> [TraceRow]   one prompt per 15 labels, 10 inputs per label; cached in <root>/<site id>.jsonl
    rows(site) -> [TraceRow]                       the cached rows for the site's current labels and descriptions; [] if none
    python -m dreaming.predictors.seeds bookkeeper.workflow bookkeeper.categorize "Transaction: ADOBE *WEB, $60.00"

Port of S2b (smokes/s2_hypernet/run2.py `synth_examples`): ~$0.005 per site on asi1-mini; with these rows the memory's certified
coverage at k=1 went 0.10 -> 0.50 and the head's 0.14 -> 0.80 (S24, S17). A seed row is a TraceRow with id "seed:<label>:<i>" and
run_id "seed" (the source tag), phase awake: it trains, it never calibrates. The cache key is the site id plus the sha of its labels
and descriptions, so a changed schema writes new rows. `ask` is prompt -> reply text (`asi1` here; the tests pass a fake).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path

from ..contracts import CallSite, Decision, TeacherAnswer, TraceRow
from ..hashing import sha

log = logging.getLogger(__name__)
ROOT = Path("data/seeds")  # relative to the process's working directory, like data/embeddings.npz
PER_LABEL, CHUNK = 10, 15  # S2b: 10 inputs per label, 15 labels per call


def _key(site: CallSite) -> str:
    return sha([site.labels, site.descriptions])


def rows(site: CallSite, root: Path = ROOT) -> list[TraceRow]:
    path = root / f"{site.id}.jsonl"
    if site.labels is None or not path.is_file():
        return []
    lines = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [TraceRow(id=d["id"], run_id="seed", i=k + 1, call_site=site.id, kind=site.kind, request=d["text"], request_sha=sha(d["text"]),
                     response=d["label"], decision=Decision.called, teacher=TeacherAnswer(text=d["label"], label=d["label"], provider="seed", model=d["model"]),
                     latency_ms=0, cost_usd=0.0, ts=datetime(2026, 1, 1, tzinfo=UTC))
            for k, d in enumerate(x for x in lines if x["key"] == _key(site))]


def write(site: CallSite, ask: Callable[[str], str], example: str | None = None, model: str = "asi1-mini", root: Path = ROOT) -> list[TraceRow]:
    """Ask for PER_LABEL inputs per label (up to 3 tries per chunk when the reply is not JSON or misses a label) and cache them.
    A label the model never covered gets no rows, not a placeholder. Cached rows for the same schema are returned without a call."""
    if site.labels is None:
        raise ValueError(f"{site.id}: seed rows need a label set")
    if have := rows(site, root):
        return have
    shape = f" Write every input in exactly the format of this example input, changing only its content: {example!r}." if example else ""
    out: list[dict[str, str]] = []
    for c in range(0, len(site.labels), CHUNK):
        chunk = site.labels[c : c + CHUNK]
        prompt = (f"These are labels of the `{site.name}` step of the `{site.workflow}` workflow. For each label write {PER_LABEL} short, "
                  f"varied, realistic inputs this step would receive that should get that label.{shape} Reply with a JSON object whose keys "
                  f"are exactly these label strings {json.dumps(chunk)} and whose values are lists of {PER_LABEL} strings, nothing else.\n"
                  "What each label means:\n" + "\n".join(f"{lab}: {site.descriptions.get(lab, lab)}" for lab in chunk))
        by_lower = {lab.lower(): lab for lab in chunk}
        got: dict[str, list[str]] = {}
        for _ in range(3):
            text = ask(prompt)
            try:
                reply = json.loads(text[text.find("{") : text.rfind("}") + 1])
            except json.JSONDecodeError:
                continue
            # asi1-mini echoed "Label: description" as the key in 3 of 3 replies (measured); a retry fills in what an earlier reply missed
            got |= {lab: v for k, v in reply.items() if isinstance(v, list)
                    and (lab := by_lower.get(k.strip().lower()) or by_lower.get(k.split(":")[0].strip().lower()))}
            if len(got) == len(chunk):
                break
        if missing := [lab for lab in chunk if lab not in got]:
            log.warning("%s: the model gave no usable inputs for %s after 3 tries; those labels have no seed rows", site.id, missing)
        out += [{"key": _key(site), "id": f"seed:{lab}:{i}", "label": lab, "text": str(t), "model": model}
                for lab in chunk for i, t in enumerate(got.get(lab, [])[:PER_LABEL])]
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{site.id}.jsonl").write_text("".join(json.dumps(d) + "\n" for d in out), encoding="utf-8", newline="\n")  # LF on Windows too
    return rows(site, root)


def asi1(prompt: str, model: str = "asi1-mini") -> str:
    from openai import OpenAI

    from ..teachers.openai_compat import ASI1

    r = OpenAI(api_key=os.environ["ASI1KEY"], base_url=ASI1, timeout=120).chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], temperature=0.7, max_tokens=4000)
    return r.choices[0].message.content or ""


if __name__ == "__main__":  # <workflow module> <site id> [example input]
    from .. import registry

    import_module(sys.argv[1])  # declaring the workflow registers its sites
    seeded = write(registry.get(sys.argv[2]).site, asi1, sys.argv[3] if len(sys.argv) > 3 else None)
    print(f"{sys.argv[2]}: {len(seeded)} seed rows in {ROOT / sys.argv[2]}.jsonl")
