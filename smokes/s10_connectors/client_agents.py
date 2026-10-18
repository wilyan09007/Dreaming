"""OpenAI Agents SDK through the Dreaming proxy. Usage: client_agents.py chat|global|responses [stream]
  chat      OpenAIChatCompletionsModel(openai_client=AsyncOpenAI(base_url, default_headers))   <- the recommended change
  global    set_default_openai_client(client) + set_default_openai_api("chat_completions"); agents then take a bare model name
  responses OpenAIResponsesModel(openai_client=client): the SDK default API, goes to the proxy's /v1/responses"""
import asyncio, json, sys
from openai import AsyncOpenAI
from agents import (Agent, ModelSettings, OpenAIChatCompletionsModel, OpenAIResponsesModel, Runner,
                    set_default_openai_api, set_default_openai_client, set_tracing_disabled)

mode, stream, M = sys.argv[1], "stream" in sys.argv, "openai/gpt-5.4-nano"
set_tracing_disabled(True)  # else the SDK tries to export traces to api.openai.com with the (fake) key
client = AsyncOpenAI(base_url="http://127.0.0.1:8765/v1", api_key="x", default_headers={"X-Dreaming-Call-Site": f"triage-{mode}" + "-stream" * stream})
model = M
if mode == "chat": model = OpenAIChatCompletionsModel(model=M, openai_client=client)
if mode == "responses": model = OpenAIResponsesModel(model=M, openai_client=client)
if mode == "global": set_default_openai_client(client, use_for_tracing=False); set_default_openai_api("chat_completions")
# max_tokens is sent as max_output_tokens on /responses (OpenAI minimum 16)
agent = Agent(name="triage", instructions="Reply with exactly one word: Billing, Technical or Other.", model=model,
              model_settings=ModelSettings(temperature=0, max_tokens=60))
P = "My invoice shows a double charge for March."

async def go():
    r = Runner.run_streamed(agent, P)
    async for _ in r.stream_events(): pass
    return r.final_output
print(json.dumps({"output": asyncio.run(go()) if stream else Runner.run_sync(agent, P).final_output}))
