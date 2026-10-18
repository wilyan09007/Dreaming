"""API smoke: /healthz, /state parsed as the State model, a POST /budget round trip, GET /events?limit=5.

DREAMING_API defaults to http://127.0.0.1:8080 (`just dev`); point it at the Fly app for the deployed check. One line per check,
exit 1 on failure. POST /budget re-certifies every site, so the budget the API had is put back afterwards.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

import httpx
from dreaming.contracts import Event, Health
from dreaming_api.main import State
from pydantic import TypeAdapter, ValidationError

failed: list[str] = []


def step(name: str, fn: Callable[[], tuple[bool, str]]) -> None:
    try:
        ok, detail = fn()
    except (httpx.HTTPError, ValidationError, ValueError) as e:
        ok, detail = False, f"{type(e).__name__}: {' '.join(str(e).split())[:200]}"
    print(f"{'ok  ' if ok else 'FAIL'} {name}  {detail}")
    if not ok:
        failed.append(name)


def main() -> int:
    base = os.environ.get("DREAMING_API", "http://127.0.0.1:8080").rstrip("/")
    print(f"api  {base}")
    with httpx.Client(base_url=base, timeout=30) as c:
        def state() -> State:
            return State.model_validate_json(c.get("/state").raise_for_status().text)

        def healthz() -> tuple[bool, str]:
            r = c.get("/healthz")
            h = Health.model_validate_json(r.text)
            return r.is_success and h.ok, f"{h.version} " + " ".join(f"{k}={v}" for k, v in h.checks.items())

        def parses() -> tuple[bool, str]:
            s = state()
            return bool(s.sites), f"{len(s.sites)} sites, budget {s.budget}, {len(s.series)} series points, {len(s.runs)} runs, recorded={s.batch.recorded}"

        def budget() -> tuple[bool, str]:
            before = state().budget
            r = c.post("/budget", json={"budget": 0.05}).raise_for_status()
            after = state().budget
            if before != 0.05:
                c.post("/budget", json={"budget": before}).raise_for_status()
            return r.json() == {"budget": 0.05} and after == 0.05, f"was {before}, set 0.05, read back {after}"

        def events() -> tuple[bool, str]:
            evs = TypeAdapter(list[Event]).validate_json(c.get("/events", params={"limit": 5}).raise_for_status().text)
            return len(evs) <= 5, f"{len(evs)} events, newest {evs[-1].kind.value if evs else 'none'}"

        step("GET /healthz ok", healthz)
        step("GET /state parses as State", parses)
        step("POST /budget 0.05 round trip", budget)
        step("GET /events?limit=5 parse as Event", events)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
