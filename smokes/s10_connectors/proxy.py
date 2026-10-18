"""Dreaming connector proxy: OpenAI-compatible /v1/chat/completions and /v1/responses -> OpenRouter.
Logs every call to traces.jsonl; answers from predictors.json {call_site: {"answers": {sha256(messages): text}}}.
Run: python proxy.py   (127.0.0.1:8765)"""
import hashlib, json, time, uuid
from pathlib import Path
import httpx, uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

HERE = Path(__file__).parent
TRACES, PRED = HERE / "traces.jsonl", HERE / "predictors.json"
KEY = dict(l.split("=", 1) for l in (HERE.parents[1] / ".env").read_text().splitlines() if "=" in l)["OPENROUTERKEY"].strip().strip('"')
UP = "https://openrouter.ai/api/v1"
app, http = FastAPI(), httpx.AsyncClient(timeout=120)

def sha(x): return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def parts(api, b):  # (rendered messages, system prompt, upstream path)
    if api == "chat":
        return b["messages"], next((str(m["content"]) for m in b["messages"] if m["role"] in ("system", "developer")), ""), "chat/completions"
    return [b.get("instructions"), b.get("input")], b.get("instructions") or "", "responses"

def text_of(api, r):  # (text, tool_calls): tool-call turns are logged but must never become predictor answers
    if api == "chat":
        m = r["choices"][0]["message"]; return m.get("content") or "", m.get("tool_calls")
    out = r.get("output", [])
    return "".join(c.get("text", "") for o in out if o.get("type") == "message" for c in o.get("content") or []), [o for o in out if o.get("type") == "function_call"] or None

def fake(api, b, text):  # OpenAI-shaped response, zero usage, plus the dreaming flag
    i, t = uuid.uuid4().hex[:12], int(time.time())
    if api == "chat":
        return {"id": f"chatcmpl-dreaming-{i}", "object": "chat.completion", "created": t, "model": b.get("model"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "dreaming": {"predicted": True}}
    return {"id": f"resp_dreaming_{i}", "object": "response", "created_at": t, "status": "completed", "model": b.get("model"),
            "output": [{"type": "message", "id": f"msg_{i}", "status": "completed", "role": "assistant",
                        "content": [{"type": "output_text", "text": text, "annotations": []}]}],
            "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, "dreaming": {"predicted": True}}

async def serve(api, req):
    b, t0 = await req.json(), time.time()
    msgs, system, path = parts(api, b)
    site = req.headers.get("x-dreaming-call-site") or "sha1:" + hashlib.sha1(system.encode()).hexdigest()[:12]
    h = sha(msgs)

    def log(text, usage, predicted, status=200, tool_calls=None):
        row = dict(ts=t0, api=api, call_site=site, model=b.get("model"), msg_sha256=h, response=text, tool_calls=tool_calls, status=status,
                   latency_ms=round((time.time() - t0) * 1000), usage=usage, predicted=predicted, stream=bool(b.get("stream")))
        with TRACES.open("a", encoding="utf-8") as f: f.write(json.dumps(row) + "\n")

    ans = (json.loads(PRED.read_text()) if PRED.exists() else {}).get(site, {}).get("answers", {}).get(h)
    if ans is not None:
        log(ans, None, True)
        if b.get("stream") and api == "chat":  # ponytail: predicted streaming only for chat; /responses streams are always forwarded
            ck = lambda d, fin: "data: " + json.dumps({"id": "chatcmpl-dreaming", "object": "chat.completion.chunk", "model": b.get("model"),
                                                       "choices": [{"index": 0, "delta": d, "finish_reason": fin}], "dreaming": {"predicted": True}}) + "\n\n"
            return StreamingResponse(iter([ck({"role": "assistant", "content": ans}, None), ck({}, "stop"), "data: [DONE]\n\n"]), media_type="text/event-stream")
        if not b.get("stream"):
            return JSONResponse(fake(api, b, ans))

    hdr = {"Authorization": f"Bearer {KEY}"}  # client key is dropped; client headers are not forwarded; body is forwarded as-is
    if not b.get("stream"):
        r = await http.post(f"{UP}/{path}", json=b, headers=hdr)
        j = r.json()  # ponytail: a non-JSON upstream error body would 500 here
        if r.status_code == 200: text, tc = text_of(api, j); log(text, j.get("usage"), False, 200, tc)
        else: log(json.dumps(j), None, False, r.status_code)
        return JSONResponse(j, r.status_code)

    up = await http.send(http.build_request("POST", f"{UP}/{path}", json=b, headers=hdr), stream=True)  # send first so the upstream status code is relayed

    async def gen():  # SSE passthrough, accumulate text + usage for the trace
        text, usage, tc, done = "", None, False, False
        try:
            async for line in up.aiter_lines():
                end = line == "data: [DONE]"
                if line.startswith("data: {"):
                    e = json.loads(line[6:])
                    usage = e.get("usage") or (e.get("response") or {}).get("usage") or usage
                    if e.get("choices"):
                        d = e["choices"][0]["delta"]; text += d.get("content") or ""; tc = tc or bool(d.get("tool_calls"))
                    else:
                        text += e.get("delta", "") if e.get("type") == "response.output_text.delta" else ""; tc = tc or str(e.get("type")).startswith("response.function_call")
                    end = e.get("type") in ("response.completed", "response.failed", "response.incomplete")
                if end and not done: done = True; log(text, usage, False, up.status_code, tc or None)  # log BEFORE the terminal line goes out: clients stop reading right after it
                yield (line + "\n").encode()
        finally:
            if not done: log(text, usage, False, up.status_code, tc or None)  # non-SSE error body, or client hung up mid-stream
            await up.aclose()  # ponytail: streamed tool calls are flagged (tool_calls=true), not reassembled
    return StreamingResponse(gen(), status_code=up.status_code, media_type="text/event-stream")

@app.post("/v1/chat/completions")
async def chat(req: Request): return await serve("chat", req)

@app.post("/v1/responses")
async def responses(req: Request): return await serve("responses", req)

if __name__ == "__main__": uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
