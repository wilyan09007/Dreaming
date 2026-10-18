"""ASI:One -> Agentverse mailbox -> our uAgent -> POST /run on the API -> reply, end to end, from the outside.

    AGENT_ADDRESS=agent1q... ASI1KEY=... uv run python smokes/smoke_agent.py

The planner is told the agent's address explicitly (S11: cold discovery by a vague prompt is unreliable; it also lags a
fresh registration by a few minutes). One line per check; exit 1 on failure. ~20-40 s when it works.
"""

from __future__ import annotations

import os
import sys
import time
import uuid

import httpx

LINE = "NOTION *WEB 96.00"


def main() -> int:
    key, addr = os.environ.get("ASI1KEY"), os.environ.get("AGENT_ADDRESS")
    if not key or not addr:
        print("FAIL ASI1KEY and AGENT_ADDRESS must be set (the address is in the agent machine's log)")
        return 1
    body = {"model": "asi1-mini", "stream": False, "planner_mode": True, "agents": [addr],
            "messages": [{"role": "user", "content": f"Ask the agent at {addr} to process this bank-feed line and show me its reply verbatim: {LINE}"}]}
    t0 = time.perf_counter()
    r = httpx.post("https://api.asi1.ai/v1/chat/completions", json=body, timeout=280,
                   headers={"Authorization": f"Bearer {key}", "x-session-id": str(uuid.uuid4())})
    ms = round((time.perf_counter() - t0) * 1000)
    if r.status_code != 200:
        print(f"FAIL planner call: {r.status_code} {r.text[:200]}")
        return 1
    text: str = r.json()["choices"][0]["message"]["content"]
    print(f"ok   planner call 200 in {ms} ms, {r.json().get('usage', {}).get('total_tokens')} tokens")
    relayed = "Result:" in text and "steps were answered" in text  # the agent's receipt (agents/bookkeeper agent.py to_reply)
    print(f"{'ok  ' if relayed else 'FAIL'} the agent's receipt came back through the planner\n{text}")
    return 0 if relayed else 1


if __name__ == "__main__":
    sys.exit(main())
