"""Dreaming: log every call an agent makes, learn to predict it, certify the predictor, answer from it, replay it.

Public API (what a customer imports):
    step, tool, effect   declare call sites (registry.py)
    run_boundary         open a run around a handler or a graph invocation (context.py)
    wrap_openai          wrap an OpenAI-compatible client in place (connectors/openai_client.py)
    Runtime              bind a store, a teacher and a trainer; run workflows; fast-forward; replay (runtime.py)
"""

from .connectors.openai_client import wrap_openai
from .context import run_boundary
from .registry import effect, step, tool
from .runtime import Runtime

__all__ = ["Runtime", "effect", "run_boundary", "step", "tool", "wrap_openai"]
