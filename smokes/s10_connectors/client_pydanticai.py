"""pydantic-ai through the Dreaming proxy. The change: OpenAIProvider(base_url=...) + extra_headers in model settings. Optional argv: stream"""
import asyncio, json, sys
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

stream, P = "stream" in sys.argv, "The checkout flow was fast and painless."
model = OpenAIChatModel("openai/gpt-5.4-nano", provider=OpenAIProvider(base_url="http://127.0.0.1:8765/v1", api_key="x"))
agent = Agent(model, instructions="Reply with exactly one word: Positive, Negative or Neutral.",
              model_settings={"temperature": 0, "max_tokens": 60, "extra_headers": {"X-Dreaming-Call-Site": "sentiment" + "-stream" * stream}})

async def go():
    async with agent.run_stream(P) as r: return [t async for t in r.stream_text()][-1]
print(json.dumps({"output": asyncio.run(go()) if stream else agent.run_sync(P).output}))
