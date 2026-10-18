"""T1c probe: LangGraph node using langchain_openai.ChatOpenAI (the usual way people call a model in LangGraph). Wrap its inner OpenAI client; no node change."""
import json
from langchain_openai import ChatOpenAI
import dreaming_wrap as dw
from _t import HERE, KEY, M, NET, _hook
import httpx

TR = HERE / "t1c_traces.jsonl"; TR.unlink(missing_ok=True)
dw.Recorder("record", TR)
llm = ChatOpenAI(model=M, api_key=KEY, base_url="https://openrouter.ai/api/v1", temperature=0, max_tokens=60, reasoning_effort="minimal",
                 http_client=httpx.Client(event_hooks={"request": [_hook]}, timeout=60))
dw.wrap_openai(llm.root_client)  # <- the one line; llm.client is root_client.chat.completions, so the patch is picked up
def node():
    with dw.call_site("categorize"): return llm.invoke([("system", "Classify the transaction as one of: travel, software, meals, other. Reply with one word."), ("human", "Delta Air Lines $412.80 flight SFO-JFK")]).content
h = dw.run_boundary(node)
a = h(); r = [json.loads(l) for l in TR.read_text(encoding="utf-8").splitlines()]; run = r[-1]["run_id"]
print("live:", repr(a), "| row:", r[0]["call_site"], r[0]["request"]["messages"], r[0]["usage"] and r[0]["usage"]["total_tokens"], "| net:", NET["n"])
assert r[0]["kind"] == "openai" and r[0]["call_site"] == "categorize" and NET["n"] == 1
dw.Recorder("record", TR, predictors={"categorize": {r[0]["req_sha"]: r[0]["text"]}}); n0 = NET["n"]
b = h(); print("predicted:", repr(b), "| net delta:", NET["n"] - n0); assert b == a and NET["n"] == n0
dw.Recorder("replay", TR, run_id=run); c = h(); print("replay:", repr(c), "| net delta:", NET["n"] - n0); assert c == a and NET["n"] == n0
print("T1c OK")
