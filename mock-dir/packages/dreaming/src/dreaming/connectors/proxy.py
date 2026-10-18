"""OpenAI-compatible proxy: `/v1/chat/completions` and `/v1/responses`, forwarded upstream, every call routed.

For agents whose client cannot be wrapped: point `base_url` at the API. The site is the `X-Dreaming-Call-Site` header, else
`sys:<sha1 of the system prompt>`. Proven in smokes/s10_connectors/proxy.py (LangGraph, Agents SDK, pydantic-ai, raw SDK).

`proxy_router(runtime, upstream)`: `upstream` is an `httpx.AsyncClient` with base_url and Authorization set, not an openai
client: bodies go out and answers come back as they are, so no SDK model sits between, and the caller's own key and
headers never reach the upstream. Every call is routed through `runtime.external`.

Only a non-streaming chat completion without tools can be predicted: it is routed under its site and, once that site has
labels and a certified predictor, answered without the upstream (`dreaming: {"predicted": true}`, zero usage). Everything
else is forwarded and logged under `<site>#forward`, a logged-only effect site: tool-call turns (a tool call is not a label,
and the predictor never reads the tools), streams (the row is written when the stream ends, before its terminal line goes
out) and /v1/responses. Why a second site and not `kind=effect` on the first: `external(kind=)` only counts when the site
is first declared, so a site declared with labels would still predict, and a streamed hit would log a phantom predicted
row (S10). The logged request carries model/messages/tools/tool_choice/response_format: the predictor key covers them (S10).
shortcut: a predicted answer is never streamed (a stream pays the upstream); streamed tool calls are flagged, not
reassembled; a live stream parks one worker thread until it ends, because its row's latency is the stream's.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from collections.abc import AsyncIterator, Callable
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, cast

import httpx
from anyio import from_thread
from fastapi import APIRouter, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse

from ..context import run_boundary
from ..contracts import Json, Kind
from ..hashing import canonical
from ..router import Stuck
from .openai_client import request_of, site_of

if TYPE_CHECKING:
    from ..runtime import Runtime

HEADER = "x-dreaming-call-site"
FORWARD = "#forward"
RESPONSES_KEYS = ("model", "instructions", "input", "tools", "tool_choice", "text", "previous_response_id")
END_EVENTS = ("response.completed", "response.failed", "response.incomplete")


class Upstream(Exception):
    """A non-2xx upstream answer: relayed to the caller as it came, and the error of the row."""

    def __init__(self, r: httpx.Response) -> None:
        super().__init__(f"{r.status_code} {r.text[:200]}")
        self.status, self.body, self.media = r.status_code, r.content, r.headers.get("content-type")


def fake(model: Any, text: str) -> dict[str, Any]:
    """The completion a predicted answer is delivered as: OpenAI-shaped, zero usage, flagged. Raw-SDK clients can read the
    flag (`model_extra`); LangChain, the Agents SDK and pydantic-ai drop it, so for them the trace is the proof (S10)."""
    return {
        "id": f"chatcmpl-dreaming-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "dreaming": {"predicted": True},
    }


def _obj(x: Any) -> dict[str, Any]:
    """`x` if it is a JSON object, else {}: an upstream body is not trusted to have the shape."""
    return cast("dict[str, Any]", x) if isinstance(x, dict) else {}


def _items(x: Any) -> list[dict[str, Any]]:
    return [_obj(i) for i in cast("list[Any]", x)] if isinstance(x, list) else []


def _chat_text(d: dict[str, Any]) -> str:
    """The assistant text of a completion, "" for a tool-call turn: a tool call is logged, never an answer to learn.
    Lenient on purpose: OpenRouter can answer 200 with an error object, which is relayed untouched."""
    m = _obj((_items(d.get("choices")) or [{}])[0].get("message"))
    return "" if m.get("tool_calls") else str(m.get("content") or "")


def _responses_text(d: dict[str, Any]) -> str:
    out = _items(d.get("output"))
    if any(o.get("type") == "function_call" for o in out):
        return ""
    return "".join(str(c.get("text", "")) for o in out if o.get("type") == "message" for c in _items(o.get("content")))


def _tally(line: str, acc: dict[str, Any]) -> bool:
    """Fold one SSE line into the running tally; True on the terminal one (chat: `[DONE]`; responses: completed/failed/incomplete)."""
    if line == "data: [DONE]":
        return True
    if not line.startswith("data: {"):
        return False
    try:
        e = _obj(json.loads(line[6:]))
    except ValueError:
        return False
    kind = str(e.get("type"))
    acc["usage"] = e.get("usage") or _obj(e.get("response")).get("usage") or acc["usage"]
    delta = _obj((_items(e.get("choices")) or [{}])[0].get("delta"))
    acc["text"] += str(delta.get("content") or "") + (str(e.get("delta", "")) if kind == "response.output_text.delta" else "")
    acc["tool_calls"] = acc["tool_calls"] or bool(delta.get("tool_calls")) or kind.startswith("response.function_call")
    return kind in END_EVENTS


def _failed(e: Exception) -> Response:
    if isinstance(e, Upstream):
        return Response(e.body, e.status, media_type=e.media)
    return JSONResponse({"error": {"message": f"upstream: {e}", "type": "bad_gateway"}}, 502)


def proxy_router(runtime: Runtime, upstream: httpx.AsyncClient) -> APIRouter:
    """One proxy, one upstream (OpenRouter/ASI:One); the key lives in `upstream` and never leaves the API."""
    r = APIRouter()

    async def post(path: str, body: dict[str, Any]) -> dict[str, Any]:
        resp = await upstream.post(path, json=body)
        if resp.is_error:
            raise Upstream(resp)
        out: dict[str, Any] = resp.json()
        return out

    async def routed(site: str, request: Json, rendered: str, real: Callable[[], Any], label_of: Callable[[Any], str], *,
                     kind: Kind = Kind.model, decode: Callable[[str], Any] = lambda s: s) -> Any:
        """`runtime.external` off the event loop, inside a run: without one the router never draws audits or outage probes."""
        with run_boundary(site, seed=random.getrandbits(32)):
            return await run_in_threadpool(runtime.external, site, request, rendered, real=real, label_of=label_of, decode=decode, kind=kind)

    async def serve(path: str, body: dict[str, Any], site: str, request: Json, rendered: str, label_of: Callable[[Any], str], *,
                    kind: Kind = Kind.model, decode: Callable[[str], Any] = lambda s: s) -> Response:
        """A non-streaming call. An upstream error is the row's error and is relayed as it came."""
        try:
            try:
                out = await routed(site, request, rendered, lambda: from_thread.run(post, path, body), label_of, kind=kind, decode=decode)
            except Stuck:  # three failures in a row flagged the site down: a proxy that stops forwarding is worse than one that stops logging
                out = await post(path, body)
            return JSONResponse(out)
        except (Upstream, httpx.HTTPError, ValueError) as e:  # ValueError: a 200 that is not JSON
            return _failed(e)

    async def stream(path: str, body: dict[str, Any], site: str, request: Json, rendered: str) -> Response:
        """Pass an SSE stream through, tally text and usage, log one row when it ends. The router starts at once and its
        `real` waits for the end, so the row's latency is the stream's; the upstream status is read first so errors keep it."""
        acc: dict[str, Any] = {"text": "", "usage": None, "tool_calls": False, "stream": True}
        end: Future[dict[str, Any]] = Future()
        logged = asyncio.ensure_future(
            routed(site + FORWARD, request, rendered, end.result, lambda d: "" if d["tool_calls"] else d["text"], kind=Kind.effect))

        async def close() -> None:
            if not end.done():
                end.set_result(acc)
            await asyncio.gather(logged, return_exceptions=True)  # waits for the row; Stuck only means there is none

        try:
            up = await upstream.send(upstream.build_request("POST", path, json=body), stream=True)
            if up.is_error:
                await up.aread()
                await up.aclose()
                raise Upstream(up)
        except BaseException as e:  # whatever happens here, release the router's thread: it is waiting on `end`
            end.set_exception(e)
            if not isinstance(e, (Upstream, httpx.HTTPError)):
                raise
            await asyncio.gather(logged, return_exceptions=True)  # the row carries the error
            return _failed(e)

        async def gen() -> AsyncIterator[bytes]:
            try:
                async for line in up.aiter_lines():
                    if _tally(line, acc):
                        await close()  # before the terminal line goes out: clients stop reading right after it
                    yield (line + "\n").encode()
                await close()  # a body with no terminal line
            finally:
                if not end.done():
                    end.set_result(acc)  # the client hung up: log what arrived
                await up.aclose()

        return StreamingResponse(gen(), status_code=up.status_code, media_type="text/event-stream")

    @r.post("/v1/chat/completions")
    async def chat(req: Request) -> Response:
        body: dict[str, Any] = await req.json()
        site = req.headers.get(HEADER) or site_of({"messages": body.get("messages") or []})
        request = request_of(body)
        rendered = canonical(request["messages"]).decode()
        if body.get("stream"):
            return await stream("chat/completions", body, site, request, rendered)
        if body.get("tools") or body.get("tool_choice"):
            return await serve("chat/completions", body, site + FORWARD, request, rendered, _chat_text, kind=Kind.effect)
        return await serve("chat/completions", body, site, request, rendered, _chat_text, decode=lambda label: fake(body.get("model"), label))

    @r.post("/v1/responses")
    async def responses(req: Request) -> Response:
        body: dict[str, Any] = await req.json()
        site = req.headers.get(HEADER) or site_of({"messages": [{"role": "system", "content": body.get("instructions") or ""}]})
        request = {k: body.get(k) for k in RESPONSES_KEYS}
        rendered = canonical([body.get("instructions"), body.get("input")]).decode()
        if body.get("stream"):
            return await stream("responses", body, site, request, rendered)
        return await serve("responses", body, site + FORWARD, request, rendered, _responses_text, kind=Kind.effect)

    return r
