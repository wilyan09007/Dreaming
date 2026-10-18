"""Connectors: Dreaming never takes over orchestration. It wraps the four places an agent already makes a call.

    openai_client.py   wrap an OpenAI-compatible client in place (model calls, in-process)        S12
    proxy.py           OpenAI-compatible HTTP proxy for clients nobody can wrap (base_url change)  S10
    uagents.py         uAgents: send_and_receive as a call site, the chat handler as the run boundary
    mcp.py             FastMCP middleware: tool calls over MCP go through the router

All four build a router.Call and hand it to Runtime.external(); none of them has its own logging or gate.
"""
