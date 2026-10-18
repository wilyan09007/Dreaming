"""Wrap `client.chat.completions.create` in place. The call site comes from the `X-Dreaming-Call-Site` header,
else from the hash of the system prompt (untagged sites are logged only; a label set makes them predictable).

Proven in smokes/s12_wrappers/dreaming_wrap.py (sync and async clients, with_options/copy, replay guard against
leaked HTTP). shortcut: the port here covers non-streaming chat completions; streams pass through unlogged.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from ..hashing import canonical

if TYPE_CHECKING:
    from ..runtime import Runtime


def site_of(kwargs: dict[str, Any]) -> str:
    hdr: dict[str, Any] = kwargs.get("extra_headers") or {}
    if "X-Dreaming-Call-Site" in hdr:
        return str(hdr["X-Dreaming-Call-Site"])
    system = next((str(m.get("content")) for m in kwargs.get("messages", []) if m.get("role") in ("system", "developer")), "")
    return "sys:" + hashlib.sha1(system.encode()).hexdigest()[:12]


def request_of(kwargs: dict[str, Any]) -> dict[str, Any]:
    """The predictor key covers everything that changes the answer: model, messages, tools, tool_choice, response_format (S10)."""
    return {k: kwargs.get(k) for k in ("model", "messages", "tools", "tool_choice", "response_format")}


def wrap_openai(client: Any, runtime: Runtime) -> Any:
    comp = client.chat.completions
    create = comp.create

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return create(*args, **kwargs)
        req = request_of(kwargs)
        return runtime.external(
            site_of(kwargs), req, canonical(req["messages"]).decode(), real=lambda: create(*args, **kwargs),
            label_of=lambda r: r.choices[0].message.content or "",
        )

    comp.create = wrapped
    return client
