"""T3: @tool decorator on vendor_lookup: log, skip above the certified cutoff, exception -> fallback, speculate_next hint, async variant, replay."""
import asyncio, json
import dreaming_wrap as dw
from _t import HERE

TR = HERE / "t3_traces.jsonl"; TR.unlink(missing_ok=True)
DB = {"acme": {"id": "V1", "terms": "net30"}, "globex": {"id": "V2", "terms": "net60"}, "initech": {"id": "V3", "terms": "net15"}}
EXEC = []  # every REAL execution of the tool body
FALLBACK = {"id": None, "terms": "unknown"}

@dw.tool("vendor_lookup", fallback=FALLBACK, speculate_next="match_invoice")
def vendor_lookup(vendor):
    EXEC.append(vendor)
    if vendor == "boom": raise TimeoutError("vendor db down")
    return DB[vendor]

@dw.tool("vendor_lookup_nofb")
def vendor_lookup_nofb(vendor):
    EXEC.append(vendor)
    raise TimeoutError("vendor db down")

@dw.tool("vendor_lookup_async", fallback=lambda vendor: {"id": None, "terms": "unknown", "vendor": vendor})  # callable fallback gets the tool's args
async def vendor_lookup_async(vendor):
    EXEC.append(vendor); await asyncio.sleep(0.01)
    if vendor == "boom": raise TimeoutError("vendor db down")
    return DB[vendor]

CONF = {"acme": 0.99, "globex": 0.97, "initech": 0.5}  # predictor knows acme, globex well; initech barely
def predictor(vendor): return (DB[vendor], CONF[vendor]) if vendor in CONF else (None, 0.0)

def rows(): return [r for r in map(json.loads, TR.read_text(encoding="utf-8").splitlines()) if r["kind"] == "tool"]
def run_ids(): return [r for r in map(json.loads, TR.read_text(encoding="utf-8").splitlines()) if r["kind"] == "run"]
def show(n):
    for r in rows()[-n:]:
        print(f"  i={r['i']} site={r['call_site']} args={r['request']['args']} predicted={r['predicted']} fallback={r['fallback']} error={r['error']} "
              f"pred={r.get('pred') and r['pred']['conf']} spec_next={r['speculate_next']} resp={r['response']}")

dw.Recorder("record", TR, predictors={"vendor_lookup": predictor, "vendor_lookup_async": predictor}, cutoffs={"vendor_lookup": 0.9, "vendor_lookup_async": 0.9})

@dw.run_boundary
def handler():
    out = []
    out.append(vendor_lookup("acme"))     # conf .99 >= .9  -> skipped
    out.append(vendor_lookup("globex"))   # .97             -> skipped
    out.append(vendor_lookup("initech"))  # .5 < .9         -> real call, prediction logged next to it
    out.append(vendor_lookup("boom"))     # no prediction; real call raises -> fallback
    return out

print("== sync tool in a run")
res = handler(); show(4); print("  real executions:", EXEC)
assert EXEC == ["initech", "boom"] and res[0] == DB["acme"] and res[3] == FALLBACK
r = rows()
assert [x["predicted"] for x in r] == [True, True, False, False] and r[3]["fallback"] and "TimeoutError" in r[3]["error"] and r[2]["pred"]["conf"] == 0.5
assert r[0]["speculate_next"] == "match_invoice" and len({x["run_id"] for x in r}) == 1 and r[0]["run_id"] is not None

print("== no fallback: exception propagates, still logged")
try: vendor_lookup_nofb("x"); raise SystemExit("should raise")
except TimeoutError: show(1)
assert rows()[-1]["error"] and rows()[-1]["response"] is None

print("== broken predictor is a miss, not an outage")
dw.Recorder("record", TR, predictors={"vendor_lookup": lambda v: 1 / 0}, cutoffs={"vendor_lookup": 0.9}); EXEC.clear()
assert vendor_lookup("acme") == DB["acme"] and EXEC == ["acme"]

print("== async tool")
async def amain():
    dw.Recorder("record", TR, predictors={"vendor_lookup_async": predictor}, cutoffs={"vendor_lookup_async": 0.9}); EXEC.clear()
    @dw.run_boundary
    async def ahandler(): return [await vendor_lookup_async("acme"), await vendor_lookup_async("initech"), await vendor_lookup_async("boom")]
    out = await ahandler(); show(3); print("  real executions:", EXEC)
    assert EXEC == ["initech", "boom"] and out[0] == DB["acme"] and out[2]["terms"] == "unknown" and out[2]["vendor"] == "boom"
    return out
aout = asyncio.run(amain())

print("== recertify: predicted tool rows re-executed live on their logged args (sync run, async run); a wrong predictor is caught")
first = run_ids()[0]["run_id"]; EXEC.clear(); rc = dw.recertify(TR, first)
print("  sync run :", [(x["call_site"], x["same"]) for x in rc], "| re-executed:", EXEC); assert [x["same"] for x in rc] == [True, True] and EXEC == ["acme", "globex"]
rca = dw.recertify(TR, run_ids()[-1]["run_id"]); print("  async run:", [(x["call_site"], x["same"]) for x in rca]); assert [x["same"] for x in rca] == [True]
dw.Recorder("record", TR, predictors={"vendor_lookup": lambda v: ({"id": "V1", "terms": "net99"}, 0.99)}, cutoffs={"vendor_lookup": 0.9})
dw.run_boundary(lambda: vendor_lookup("acme"))(); rw = dw.recertify(TR, run_ids()[-1]["run_id"])
print("  wrong predictor:", rw[0]["logged"], "vs real", rw[0]["got"], "-> same =", rw[0]["same"]); assert rw[0]["same"] is False

print("== replay the first run: no tool body executes, same results incl. the fallback")
EXEC.clear()
rec = dw.Recorder("replay", TR, run_id=first); out = handler()
print("  replay results equal:", out == res, "| real executions:", EXEC); assert out == res and EXEC == [] and rec.divergence is None
print("== replay with a different arg -> divergence at row 2")
@dw.run_boundary
def other():
    vendor_lookup("acme"); vendor_lookup("globex2")
EXEC.clear()
try: other(); raise SystemExit("expected divergence")
except dw.Divergence as e: print("  ", str(e)[:100], "|", rec.divergence["row"]); assert rec.divergence["row"] == 2 and EXEC == []
print("T3 OK")
