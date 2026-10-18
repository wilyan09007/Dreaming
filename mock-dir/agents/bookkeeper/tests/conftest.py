"""Isolation for the bookkeeper tests: a clean registry and a re-declared workflow for every test, the vendor API plugged in.

Declaring the sites is an import side effect of bookkeeper.workflow, so "clear the registry" (as packages/dreaming/tests/conftest.py
does) must be followed by a reload of that module, or the next Runtime would bind to nothing.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator

import pytest
from bookkeeper import tools, workflow
from dreaming import registry


def _reset() -> None:
    registry._SITES.clear()  # pyright: ignore[reportPrivateUsage]
    registry.bind(None)
    tools.DOWN["vendor_api"] = False


@pytest.fixture(autouse=True)
def clean_world() -> Iterator[None]:
    _reset()
    importlib.reload(workflow)  # re-declares the four call sites and two effects into the cleared registry
    yield
    _reset()
