"""dreaming_wrap (mission 6): run boundary, OpenAI client wrapper, tool decorator, uAgents send_and_receive wrapper, record/replay.
Rows in traces.jsonl: {run_id, i, kind: openai|tool|a2a|effect|run, call_site, req_sha, request, response, predicted, fallback, error, exc, latency_ms, ...}.
Build a Recorder first (it becomes the process-global one), then wrap. Non-streaming, n=1, chat.completions only."""
import asyncio, base64, contextlib, contextvars, dataclasses, datetime, functools, hashlib, importlib, inspect, json, threading, time, uuid
from pathlib import Path

RUN = contextvars.ContextVar("dreaming_run", default=None)  # {"id", "n", "closed"}: a dict, so copied contexts share one call counter
SITE = contextvars.ContextVar("dreaming_site", default=None)
CUR = None  # ponytail: one process-global Recorder, last constructed wins; per-client recorders if two ever coexist
MISS = object()
INF = float("inf")
TOOLS = {}  # call_site -> the undecorated tool function (recertify re-executes it)

class Divergence(Exception): pass
class ReplayLeak(Divergence): pass  # an HTTP request left the process in replay mode

def sha(x): return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
def _cur():  # the open run of this context (a generator finalised in another context can leave a closed one behind)
    r = RUN.get(); return r if r and not r.get("closed") else None

# ---- type-preserving JSON for tool/a2a returns: plain JSON passes through unchanged; the rest gets a {"$tag": ...} form that unpack() reverses
def _js(m): return json.loads(m.model_dump_json())  # uAgents Models are pydantic v1 (model_dump takes no mode): model_dump_json works on v1 and v2
def _cls(x): t = x if isinstance(x, type) else type(x); return f"{t.__module__}:{t.__qualname__}"
def _load(name, ok):  # import a class named in a trace row; refuse anything that is not the expected kind (rows are data, not code)
    mod, qual = name.split(":"); o = importlib.import_module(mod)
    for p in qual.split("."): o = getattr(o, p)
    if not (isinstance(o, type) and ok(o)): raise TypeError(f"{name} is not a {ok.__name__}")
    return o
def _dc(c): return dataclasses.is_dataclass(c)
def _model(c): return hasattr(c, "model_validate")
def _exc_cls(c): return issubclass(c, BaseException)

def pack(v):
    if v is None or isinstance(v, (str, int, float, bool)): return v
    if isinstance(v, list): return [pack(x) for x in v]
    if isinstance(v, dict):
        if all(isinstance(k, str) and not k.startswith("$") for k in v): return {k: pack(x) for k, x in v.items()}
        return {"$dict": [[pack(k), pack(x)] for k, x in v.items()]}
    if type(v) is tuple: return {"$tuple": [pack(x) for x in v]}
    if isinstance(v, (set, frozenset)): return {"$frozenset" if isinstance(v, frozenset) else "$set": sorted((pack(x) for x in v), key=json.dumps)}
    if isinstance(v, bytes): return {"$bytes": base64.b64encode(v).decode()}
    if type(v) in (datetime.datetime, datetime.date): return {"$" + type(v).__name__: v.isoformat()}
    if isinstance(v, uuid.UUID): return {"$uuid": str(v)}
    if dataclasses.is_dataclass(v) and not isinstance(v, type): return {"$dc": _cls(v), "v": {f.name: pack(getattr(v, f.name)) for f in dataclasses.fields(v)}}
    if hasattr(v, "model_dump"): return {"$model": _cls(v), "v": _js(v)}
    return {"$lossy": _cls(v), "v": str(v)}  # replay refuses these instead of returning a string

def unpack(v):
    if isinstance(v, list): return [unpack(x) for x in v]
    if not isinstance(v, dict): return v
    t = next((k for k in v if k.startswith("$")), None)
    if t is None: return {k: unpack(x) for k, x in v.items()}
    x = v[t]
    if t == "$tuple": return tuple(map(unpack, x))
    if t in ("$set", "$frozenset"): return (set if t == "$set" else frozenset)(map(unpack, x))
    if t == "$dict": return {unpack(k): unpack(y) for k, y in x}
    if t == "$bytes": return base64.b64decode(x)
    if t in ("$datetime", "$date"): return getattr(datetime, t[1:]).fromisoformat(x)
    if t == "$uuid": return uuid.UUID(x)
    if t == "$dc": return _load(x, _dc)(**unpack(v["v"]))
    if t == "$model": return _load(x, _model).model_validate(v["v"])
    raise Divergence(f"{x} return value is not JSON-reproducible")

def _ok(x):
    try: json.dumps(x); return True
    except (TypeError, ValueError): return False
def _exc(e): return dict(cls=_cls(e), args=[a for a in e.args if _ok(a)], attrs={k: x for k, x in vars(e).items() if _ok(x)})  # attrs: status_code etc.
def _rebuild(info, error):  # the logged exception class, without running its __init__ (openai errors need an httpx response)
    try: c = _load(info["cls"], _exc_cls); e = c.__new__(c); e.args = tuple(info["args"]); e.__dict__.update(info["attrs"]); return e
    except Exception: return RuntimeError("replayed error: " + error)  # class not importable in this process

@contextlib.contextmanager
def call_site(name):
    t = SITE.set(name)
    try: yield
    finally: SITE.reset(t)

class Recorder:
    """record: append rows to path. replay: serve run_id's rows in order, nothing executes live (an HTTP request from a wrapped openai client raises ReplayLeak).
    predictors {site: {sha(messages): text}} (openai) or {site: fn(*args) -> (value, confidence)} (tool, a2a); fallbacks {site: value|fn}; cutoffs {site: certified confidence}."""
    def __init__(self, mode="record", path="traces.jsonl", run_id=None, predictors=None, fallbacks=None, cutoffs=None):
        global CUR; CUR = self
        self.mode, self.path, self.src, self.lock, self.divergence, self.leaks = mode, Path(path), run_id, threading.Lock(), None, []
        self.predictors, self.fallbacks, self.cutoffs = predictors or {}, fallbacks or {}, cutoffs or {}
        if mode == "replay":
            rows = (json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines())
            self.rows = sorted((r for r in rows if r["run_id"] == run_id and r["kind"] != "run"), key=lambda r: r["i"])
            if not self.rows: raise ValueError(f"no rows for run {run_id}")
    def write(self, row):
        if self.mode == "record":
            with self.lock, self.path.open("a", encoding="utf-8") as f: f.write(json.dumps(row, default=str) + "\n")
    def diverge(self, row, expected, actual):  # row is 1-based: the Nth wrapped call of the run
        self.divergence = dict(row=row, expected=expected, actual=actual)
        return Divergence(f"replay diverged at row {row}: expected {expected}, got {actual}")

class Call:
    """One wrapped call. hit() serves it without executing (replay row or predictor); otherwise the caller runs it live, then done()/fail() log it."""
    def __init__(self, kind, site, req, h=None, enc=pack, dec=lambda row: unpack(row["response"]), predict=None, fb=None, bad=lambda out: None, meta=lambda out: {}, extra=None):
        self.rec, self.run, self.kind, self.site, self.req, self.h = CUR, _cur(), kind, site, req, h or sha(req)
        self.enc, self.dec, self.predict, self.fb, self.bad, self.meta, self.t0 = enc, dec, predict, fb, bad, meta, time.time()
        self.extra, self.i = extra if extra is not None else {}, 0
        if self.run:
            with self.rec.lock: self.run["n"] += 1; self.i = self.run["n"]
    def hit(self):
        r = self.rec
        if r.mode == "replay":
            row = r.rows[self.i - 1] if 0 < self.i <= len(r.rows) else None
            if not row or (row["call_site"], row["req_sha"]) != (self.site, self.h):
                raise r.diverge(self.i, row and [row["call_site"], row["req_sha"]], [self.site, self.h])
            if row["exc"]: raise _rebuild(row["exc"], row["error"])  # the live call raised: raise the same class again
            try: return True, self.dec(row)
            except Divergence as e: raise r.diverge(self.i, "a reproducible response", str(e)) from e
        v = self.predict() if self.predict else MISS
        if v is MISS: return False, None
        self.log(v, predicted=True); return True, v
    def log(self, v, **more):
        row = dict(run_id=self.run and self.run["id"], i=self.i, kind=self.kind, call_site=self.site, req_sha=self.h, request=self.req, response=None,
                   predicted=False, fallback=False, error=None, exc=None, latency_ms=round((time.time() - self.t0) * 1000), ts=self.t0)
        if v is not MISS: row.update(response=self.enc(v), **self.meta(v))
        self.rec.write({**row, **self.extra, **more})
    def done(self, out):
        err = self.bad(out)  # soft failure (uAgents returns (None, status) instead of raising): replays as the same return value
        if err and self.fb: return self.fail(RuntimeError(err))
        self.log(out, error=err); return out
    def fail(self, e):
        if not self.fb: self.log(MISS, error=repr(e), exc=_exc(e)); raise e
        v = self.fb(); self.log(v, error=repr(e), fallback=True); return v

def _async(f): return inspect.iscoroutinefunction(inspect.unwrap(f))  # openai's AsyncCompletions.create is a sync-looking wrapper around an async def: unwrap first

def _wrap(f, mk):
    """Sync or async wrapper around f; mk(args, kwargs) -> Call, or None to pass the call through untouched."""
    if _async(f):
        @functools.wraps(f)
        async def w(*a, **kw):
            c = mk(a, kw)
            if c is None: return await f(*a, **kw)
            ok, v = c.hit()
            if ok: return v
            try: out = await f(*a, **kw)
            except Exception as e: return c.fail(e)
            return c.done(out)
    else:
        @functools.wraps(f)
        def w(*a, **kw):
            c = mk(a, kw)
            if c is None: return f(*a, **kw)
            ok, v = c.hit()
            if ok: return v
            try: out = f(*a, **kw)
            except Exception as e: return c.fail(e)
            return c.done(out)
    w._orig = f  # not __wrapped__: the SDK's own create has one too
    return w

@contextlib.contextmanager
def _run(name):
    rec, outer = CUR, _cur()
    if outer: yield outer; return  # a nested boundary joins the open run
    run = {"id": rec.src if rec.mode == "replay" else uuid.uuid4().hex[:8], "n": 0}
    tok, t0, err = RUN.set(run), time.time(), None
    leak = lambda: ReplayLeak(f"replay made {len(rec.leaks)} live HTTP request(s): {rec.leaks[:3]}")
    try:
        yield run
        if rec.mode == "replay":
            if run["n"] < len(rec.rows):  # the handler stopped calling before the log did
                nxt = rec.rows[run["n"]]; raise rec.diverge(run["n"] + 1, [nxt["call_site"], nxt["req_sha"]], None)
            if rec.leaks: raise leak()  # a leak the handler swallowed
    except BaseException as e:
        err = repr(e)
        if rec.leaks and not isinstance(e, Divergence): raise leak() from e  # a leak the SDK or handler re-wrapped
        raise
    finally:
        run["closed"] = True
        try: RUN.reset(tok)
        except ValueError: pass  # a generator finalised in another context; _cur() ignores the closed run
        rec.write(dict(kind="run", run_id=run["id"], fn=name, calls=run["n"], error=err, latency_ms=round((time.time() - t0) * 1000), ts=t0))

def run_boundary(fn):
    """Decorate a graph.invoke / message handler (sync or async, or a stream generator: the run stays open until it is consumed): opens a run id in a ContextVar, closes it on return or exception."""
    ctxs = lambda a: tuple(wrap_send_and_receive(x) if hasattr(x, "send_and_receive") else x for x in a)  # uAgents ctx gets wrapped for free
    name, g = getattr(fn, "__name__", repr(fn)), inspect.unwrap(fn)
    if inspect.isasyncgenfunction(g):
        @functools.wraps(fn)
        async def w(*a, **kw):
            with _run(name):
                async for x in fn(*ctxs(a), **kw): yield x
    elif inspect.isgeneratorfunction(g):
        @functools.wraps(fn)
        def w(*a, **kw):
            with _run(name): yield from fn(*ctxs(a), **kw)
    elif _async(fn):
        @functools.wraps(fn)
        async def w(*a, **kw):
            with _run(name): return await fn(*ctxs(a), **kw)
    else:
        @functools.wraps(fn)
        def w(*a, **kw):
            with _run(name): return fn(*ctxs(a), **kw)
    return w

def completion(choice, model):
    from openai.types.chat import ChatCompletion
    return ChatCompletion.model_validate(dict(id="chatcmpl-dreaming", object="chat.completion", created=int(time.time()), model=model or "dreaming",
                                              choices=[choice], usage=dict(prompt_tokens=0, completion_tokens=0, total_tokens=0)))

class Raw:  # what with_raw_response callers (langchain_openai) get back for a served call: .parse() -> completion, .headers
    def __init__(self, c): self.c, self.headers = c, {}
    def parse(self): return self.c

def _guard(hc):  # replay mode: any request through the client's httpx transport that is not a served wrapped call (responses, embeddings, parse, streams) fails loudly
    if getattr(hc, "_dreaming", False): return
    send = hc.send
    def g(req, *a, **kw):
        if CUR and CUR.mode == "replay":
            CUR.leaks.append(f"{req.method} {req.url}"); raise ReplayLeak(f"replay mode: live HTTP request {req.method} {req.url} (not a wrapped call)")
        return send(req, *a, **kw)  # sync or async: an async send's coroutine goes straight back to the caller's await
    hc.send, hc._dreaming = g, True

def wrap_openai(client):
    """Patch client.chat.completions.create in place (OpenAI or AsyncOpenAI) before first use; clients made by copy()/with_options() are wrapped too.
    Site: X-Dreaming-Call-Site header > call_site() > hash of system prompt."""
    comp = client.chat.completions; is_async = _async(comp.create)
    def mk(a, kw):
        hdr = kw.get("extra_headers") or {}; raw = "X-Stainless-Raw-Response" in hdr  # with_raw_response.create (langchain_openai) routes through this same create
        if kw.get("stream") or (raw and is_async):  # ponytail: streams and async raw-response calls pass through unlogged
            if CUR.mode == "replay": raise NotImplementedError("streaming / async raw responses are not supported in replay")
            return None
        msgs, model = kw["messages"], kw.get("model")
        sysp = next((str(m.get("content")) for m in msgs if m.get("role") in ("system", "developer")), "")
        site = hdr.get("X-Dreaming-Call-Site") or SITE.get() or "sys:" + hashlib.sha1(sysp.encode()).hexdigest()[:12]
        h, preds = sha(msgs), CUR.predictors.get(site)
        text = preds.get(h) if isinstance(preds, dict) else None
        req = dict(model=model, messages=msgs, tools=kw.get("tools"), params={k: v for k, v in kw.items() if k not in ("model", "messages", "tools", "extra_headers")})
        P, W = (lambda r: r.parse() if raw else r), (lambda c: Raw(c) if raw else c)
        return Call("openai", site, req, h, enc=lambda r: P(r).choices[0].model_dump(exclude_none=True), dec=lambda row: W(completion(row["response"], model)),
                    predict=lambda: W(completion(dict(index=0, finish_reason="stop", message=dict(role="assistant", content=text)), model)) if text is not None else MISS,
                    meta=lambda r: dict(text=P(r).choices[0].message.content, usage=P(r).usage and P(r).usage.model_dump(), model=P(r).model))
    comp.create = _wrap(comp.create, mk)
    _guard(client._client)
    for n in ("copy", "with_options"):  # these return a NEW client whose create is not patched
        setattr(client, n, lambda *a, f=getattr(client, n), **kw: wrap_openai(f(*a, **kw)))
    return client

def _predict(site, extra, a, kw, post=lambda v: v):  # predictor(*args) -> (value, confidence); skip only at/above the site's certified cutoff
    def predict():
        p = CUR.predictors.get(site)
        if not callable(p): return MISS
        try: v, conf = p(*a, **kw)
        except Exception: return MISS  # ponytail: a broken predictor is a miss, not an outage
        extra["pred"] = dict(value=pack(v), conf=conf)  # logged even when the real call runs: free calibration data
        return post(v) if conf >= CUR.cutoffs.get(site, INF) else MISS
    return predict

def _fb(site, fallback, a, kw, post=lambda v: v):
    fb = fallback if fallback is not None else CUR.fallbacks.get(site)
    return None if fb is None else (lambda: post(fb(*a, **kw) if callable(fb) else fb))

def tool(call_site, fallback=None, speculate_next=None):
    """Decorate a tool function (sync or async): log args/return/latency/errors; predictor+cutoff may skip it; fallback on exception; speculate_next is a logged hint."""
    def deco(f):
        TOOLS[call_site] = f
        def mk(a, kw):
            extra = {"speculate_next": speculate_next}
            return Call("tool", call_site, pack(dict(args=list(a), kwargs=kw)), predict=_predict(call_site, extra, a, kw), fb=_fb(call_site, fallback, a, kw), extra=extra)
        return _wrap(f, mk)
    return deco

def _body(m): return dict(type=type(m).__name__, body=_js(m))

def wrap_send_and_receive(ctx):
    """Wrap one uAgents ctx: send_and_receive (site = destination|MessageType, tool-style predictor/fallback) and send (effect, logged only: replay checks destination and type, not the body)."""
    if getattr(ctx, "_dreaming", False): return ctx
    from uagents_core.types import DeliveryStatus, MsgStatus
    sar, send = ctx.send_and_receive, ctx.send
    s_sar, s_send = inspect.signature(sar), inspect.signature(send)  # bind, so keyword and positional calls both work
    ok = lambda dest: MsgStatus(status=DeliveryStatus.DELIVERED, detail="dreaming", destination=dest, endpoint="", session=ctx.session)
    def mk_sar(a, kw):
        b = s_sar.bind(*a, **kw).arguments; dest, msg, rt = b["destination"], b["message"], b["response_type"]
        types, site = ({rt} if isinstance(rt, type) else set(rt)), f"{dest}|{type(msg).__name__}"
        extra = dict(msg_cls=_cls(msg), response_types=sorted(map(_cls, types)))  # recertify rebuilds the call from these
        return Call("a2a", site, dict(destination=dest, **_body(msg)), extra=extra,
                    predict=_predict(site, extra, (msg,), {}, lambda v: (v, ok(dest))), fb=_fb(site, None, (msg,), {}, lambda v: (v, ok(dest))),
                    enc=lambda out: None if out[0] is None else _body(out[0]), meta=lambda out: dict(status=_js(out[1])),
                    dec=lambda row: (row["response"] and next(t for t in types if t.__name__ == row["response"]["type"]).model_validate(row["response"]["body"]), MsgStatus.model_validate(row["status"])),
                    bad=lambda out: (out[0] is None and (out[1].detail or "no response")) or None)
    def mk_send(a, kw):
        b = s_send.bind(*a, **kw).arguments; dest, msg = b["destination"], b["message"]
        return Call("effect", f"{dest}|{type(msg).__name__}", dict(destination=dest, **_body(msg)), h=sha(dict(destination=dest, type=type(msg).__name__)),
                    enc=lambda st: st.status.value, dec=lambda row: ok(dest))
    ctx.send_and_receive, ctx.send, ctx._dreaming = _wrap(sar, mk_sar), _wrap(send, mk_send), True
    return ctx

async def arecertify(path, run_id, client=None, ctx=None):
    """Re-execute every predicted row of a run live on its logged request (record mode) and compare with the logged answer.
    openai rows need the (sync or async) client, tool rows the registered tool function, a2a rows a live uAgents ctx."""
    out = []
    for r in map(json.loads, Path(path).read_text(encoding="utf-8").splitlines()):
        if r["run_id"] != run_id or not r.get("predicted"): continue
        q, d = r["request"], dict(row=r["i"], kind=r["kind"], call_site=r["call_site"])
        if r["kind"] == "openai":
            got = getattr(client.chat.completions.create, "_orig", client.chat.completions.create)(model=q["model"], messages=q["messages"], **({"tools": q["tools"]} if q["tools"] else {}), **q["params"])
            got = await got if inspect.isawaitable(got) else got
            d.update(logged=r["text"], got=got.choices[0].message.content, usage=got.usage and got.usage.model_dump())
        elif r["kind"] == "tool":
            got = TOOLS[r["call_site"]](*unpack(q["args"]), **unpack(q["kwargs"]))
            got = await got if inspect.isawaitable(got) else got
            d.update(logged=r["response"], got=pack(got))
        elif r["kind"] == "a2a":
            f = getattr(ctx.send_and_receive, "_orig", ctx.send_and_receive)
            got, _ = await f(q["destination"], _load(r["msg_cls"], _model).model_validate(q["body"]), response_type={_load(c, _model) for c in r["response_types"]})
            d.update(logged=r["response"], got=got and _body(got))
        out.append({**d, "same": d["logged"] == d["got"]})
    return out

def recertify(*a, **kw): return asyncio.run(arecertify(*a, **kw))  # from async code: await arecertify(...)
