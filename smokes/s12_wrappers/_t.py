"""Test helpers: OpenRouter client with a network-request counter, tiny cost ledger."""
import json, os, re
from pathlib import Path
import httpx, openai

HERE = Path(__file__).parent
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
KEY = re.search(r"^OPENROUTERKEY\s*=\s*\"?([^\"\r\n]+)", (HERE.parents[1] / ".env").read_text(), re.M).group(1).strip()
M = "openai/gpt-5.4-nano"
GEN = dict(model=M, temperature=0, max_tokens=60, extra_body={"reasoning": {"effort": "minimal"}})
NET = {"n": 0}  # every HTTP request that actually leaves the process
COST = HERE / "cost_s12.json"  # running token totals across the t*.py scripts

def _hook(req): NET["n"] += 1
async def _ahook(req): NET["n"] += 1

def client(async_=False):
    if async_: return openai.AsyncOpenAI(api_key=KEY, base_url="https://openrouter.ai/api/v1", http_client=httpx.AsyncClient(event_hooks={"request": [_ahook]}, timeout=60))
    return openai.OpenAI(api_key=KEY, base_url="https://openrouter.ai/api/v1", http_client=httpx.Client(event_hooks={"request": [_hook]}, timeout=60))

def spend(traces):
    """Sum usage of live (non-predicted) openai rows in a traces file -> usd at 0.2/1.25 per M; append to cost_s12.json."""
    pin = pout = 0
    for l in Path(traces).read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        if r["kind"] == "openai" and not r["predicted"] and r.get("usage"):
            pin += r["usage"]["prompt_tokens"]; pout += r["usage"]["completion_tokens"]
    return pin, pout, (pin * 0.2 + pout * 1.25) / 1e6
