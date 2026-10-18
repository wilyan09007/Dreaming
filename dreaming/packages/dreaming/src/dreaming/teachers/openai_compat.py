"""OpenAI-compatible teachers: one class, one table of providers (the teacher slot, DESIGN.md §5).

A call site's `TeacherSpec.provider` names a row of PROVIDERS; the teacher keeps one client per provider and dispatches
per call, so two sites in one process can pin two providers. Adding a provider is one row (base URL, key env var, the
max_tokens a one-label reply needs) when the API speaks the OpenAI chat shape.

Pinning (S13): four of five teachers silently drop temperature 0 under default routing and endpoints of the same model
disagree with each other more than with themselves, so every request names its provider and every answer records the
provider that served it (OpenRouter reports the upstream in `openrouter_metadata`; elsewhere it is the row's name).
OpenRouter always returns usage.cost; the others do not, so the site's prices apply.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..contracts import CallSite, TeacherAnswer
from . import messages, parse_label

OPENROUTER = "https://openrouter.ai/api/v1"
ASI1 = "https://api.asi1.ai/v1"
MAX_TOKENS_TEXT = 1024  # untyped connector sites (free text)


@dataclass(frozen=True)
class Provider:
    base_url: str
    key: str  # the env var that holds the API key
    max_tokens_label: int = 64  # a label plus a stray word; raised where reasoning tokens share the budget (StepFun, Gemini)
    metadata: bool = False  # OpenRouter: ask which upstream served (ASI:One rejects unknown headers/fields)


PROVIDERS: dict[str, Provider] = {
    "openrouter": Provider(OPENROUTER, "OPENROUTERKEY", metadata=True),
    "asi1": Provider(ASI1, "ASI1KEY"),
    # reasoning tokens count toward max_tokens on these two: at 256, step-5-preview returned finish_reason=length with empty content
    # on 1 of 10 label calls (measured 2026-10-17, reasoning_effort=low); 1024 leaves room and costs ~$0.0005 per call
    "stepfun": Provider("https://api.stepfun.ai/v1", "STEPFUNKEY", max_tokens_label=1024),  # thinking cannot be off; docs/devtools.md
    "gemini": Provider("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINIKEY", max_tokens_label=1024),
}


class OpenAICompatTeacher:
    def __init__(self, keys: Mapping[str, str], timeout: float = 60.0) -> None:
        """`keys`: provider name -> API key, for the providers this process may call (wiring reads the env vars once)."""
        self.keys, self.timeout = dict(keys), timeout
        self._clients: dict[str, Any] = {}

    def providers(self) -> list[str]:
        return sorted(self.keys)

    def _client(self, name: str) -> Any:
        if name not in self._clients:
            from openai import OpenAI

            if name not in PROVIDERS:
                raise ValueError(f"unknown teacher provider {name!r}; rows: {sorted(PROVIDERS)}")
            if name not in self.keys:
                raise RuntimeError(f"no key for teacher provider {name!r}: set {PROVIDERS[name].key}")
            self._clients[name] = OpenAI(api_key=self.keys[name], base_url=PROVIDERS[name].base_url, timeout=self.timeout, max_retries=2)
        return self._clients[name]

    def ask(self, site: CallSite, rendered: str) -> TeacherAnswer:
        spec = site.teacher
        row, client = PROVIDERS[spec.provider], self._client(spec.provider)
        body: dict[str, Any] = dict(spec.params)  # temperature, reasoning, provider: {...}: passed through as-is
        headers = {"X-OpenRouter-Metadata": "enabled"} if row.metadata else {}
        # A typed step answers with one label. Without max_tokens OpenRouter reserves the model's whole output window (65k on
        # haiku-5.5) against the account's credit and answers 402 once the balance is below that (seen live, smokes/RESULTS.md).
        max_tokens = row.max_tokens_label if site.labels is not None else MAX_TOKENS_TEXT
        t0 = time.perf_counter()
        r: Any = client.chat.completions.create(
            model=spec.model, messages=messages(site, rendered), max_tokens=max_tokens, extra_body=body, extra_headers=headers
        )
        text: str = r.choices[0].message.content or ""
        usage: dict[str, Any] = r.usage.model_dump() if r.usage else {}
        extra: dict[str, Any] = r.model_extra or {}
        meta: dict[str, Any] = extra.get("openrouter_metadata") or {}  # absent on cache hits; GET /generation?id= is the fallback
        attempts: list[dict[str, Any]] = meta.get("attempts") or []
        cost = float(usage.get("cost") or 0.0) or (
            usage.get("prompt_tokens", 0) * spec.price_in + usage.get("completion_tokens", 0) * spec.price_out
        ) / 1e6
        return TeacherAnswer(
            text=text, label=parse_label(text, site.labels),
            provider=str((attempts[-1].get("provider") if attempts else None) or spec.provider),
            model=r.model or spec.model, cost_usd=cost, latency_ms=round((time.perf_counter() - t0) * 1000),
            usage={k: int(v) for k, v in usage.items() if isinstance(v, int)},
        )
