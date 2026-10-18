"""The campaign's two doors, proxied to the deployed Modal app `dreaming-campaign` (apps/trainer; the cells run there, not here).

    POST /campaign/run {"campaign", "sites": [], "limit": 0}   -> {"cells": [...]}: Modal's launch_cells(campaign, sites, limit); [] = all sites, 0 = all rows
    GET  /campaign/{name}                                      -> {"campaign": name, "table": {...}}: Modal's progress(name)
    remote(fn_name, *args)                                     the one place that touches Modal (tests replace it)

Any failure of the Modal call (no token, app not deployed, function raised) is a 503 carrying the error text.
shortcut: `.remote()` blocks one worker thread until the function returns; launch_cells only starts the cells, so that is
seconds. A long-running call would want `.spawn()` and a poll.
"""

from __future__ import annotations

from typing import Any

from dreaming.contracts import Json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


class CampaignRunIn(BaseModel):
    campaign: str
    sites: list[str] = Field(default_factory=list)
    limit: int = 0


def remote(fn_name: str, *args: Json) -> Any:
    import modal  # lazy: the API image carries the light client, tests and `just dev` never reach it

    fn: Any = modal.Function.from_name("dreaming-campaign", fn_name)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return fn.remote(*args)


def call(fn_name: str, *args: Json) -> Any:
    try:
        return remote(fn_name, *args)
    except Exception as e:  # whatever Modal raises (auth, not deployed, remote error) means the campaign is unavailable
        raise HTTPException(503, f"campaign {fn_name} failed: {type(e).__name__}: {e}") from e


@router.post("/campaign/run")
def run(body: CampaignRunIn) -> dict[str, Any]:
    return {"cells": call("launch_cells", body.campaign, body.sites, body.limit)}


@router.get("/campaign/{name}")
def progress(name: str) -> dict[str, Any]:
    return {"campaign": name, "table": call("progress", name)}
