"""The same workflow on LangGraph, untouched except for one line per node: the client points at the API's proxy and
names its call site in a header (mission 6, proxy path; smokes/s10_connectors proves LangGraph, Agents SDK, pydantic-ai).

    llm = ChatOpenAI(base_url=f"{DREAMING_API}/v1", api_key="unused", model="anthropic/claude-haiku-5.5",
                     default_headers={"X-Dreaming-Call-Site": "langgraph_bookkeeper.categorize"})

Untagged nodes are logged under a hash of their system prompt; declaring the site's labels (POST /sites, or in code
through the API's registry) is what makes them predictable. shortcut: contract only; not in the demo's recorded path.
"""

from __future__ import annotations


def build_graph(api_base: str) -> object:
    raise NotImplementedError("StateGraph with four nodes; each node's ChatOpenAI carries its X-Dreaming-Call-Site header")
