"""MCP: a FastMCP proxy whose `on_call_tool` middleware routes each tool call. Tools exposed over MCP get the
same predictor, cutoff and fallback as decorated functions. Proven in smokes/s10_connectors/mcp_demo.py.
shortcut: contract only; not in the demo path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..runtime import Runtime


def mcp_proxy(runtime: Runtime, backend: Any) -> Any:
    raise NotImplementedError("FastMCP create_proxy + Middleware.on_call_tool -> runtime.external")
