"""ASI:One smoke: one short chat completion on asi1-mini over raw HTTP (needs ASI1KEY).

ASI:One rejects unknown top-level request fields (DESIGN.md section 9), so the body is exactly model, messages and max_tokens:
no `provider`, `metadata` or `extra_body`, which the OpenRouter teacher sends. One line per check, exit 1 on failure.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

URL = "https://api.asi1.ai/v1/chat/completions"
failed: list[str] = []


class Message(BaseModel):
    content: str | None = None


class Choice(BaseModel):
    message: Message


class Reply(BaseModel):
    model: str = ""
    choices: list[Choice] = []
    usage: dict[str, Any] = {}


def check(name: str, ok: bool, detail: object = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}  {detail}".rstrip())
    if not ok:
        failed.append(name)


def main() -> int:
    key = os.environ.get("ASI1KEY", "")
    check("ASI1KEY is set", bool(key))
    if not key:
        return 1
    body = {"model": "asi1-mini", "messages": [{"role": "user", "content": "Reply with one word: pong"}], "max_tokens": 64}
    t0 = time.perf_counter()
    try:
        r = httpx.post(URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=60)
        ms = (time.perf_counter() - t0) * 1000
        check("POST /v1/chat/completions returns 200", r.status_code == 200, f"{ms:.0f} ms" if r.is_success else f"HTTP {r.status_code}: {r.text[:200]}")
        reply = Reply.model_validate_json(r.text) if r.is_success else Reply()
    except (httpx.HTTPError, ValidationError) as e:
        check("POST /v1/chat/completions returns 200", False, f"{type(e).__name__}: {e}")
        return 1
    content = reply.choices[0].message.content if reply.choices else None
    check("reply has non-empty content", bool(content and content.strip()), repr(content)[:80])
    check("latency and usage", r.is_success, f"{ms:.0f} ms, model {reply.model}, usage {reply.usage}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
