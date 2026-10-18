"""The campaign's two doors for the CREAO benchmark agent (docs/devtools.md §CREAO), proxied by name to the deployed Modal app
`dreaming-campaign` (apps/trainer campaign.py; the cells run there, not here).

    POST /campaign/run {"provider", "site", "campaign"?, "limit"?}   -> {"campaign_id", "cells"}: Modal's launch_cells(campaign_id, [site], limit)
    GET  /campaign/{name}                                           -> {"campaign": name, "table"}: Modal's progress(name), polled until done

`campaign_id` defaults to "<provider>-<site>"; launch_cells takes no provider, so the provider is recorded in the name and the cells train
on the labels prepare_site wrote for the site (the books for QuickBooks, the pinned teacher's trace rows for the bookkeeper).
POST spends Modal compute, so it needs `Authorization: Bearer $DREAMING_API_KEY` (401 otherwise; 503 while the API has no key set).
Any failure of the Modal call (no token, app not deployed, the function raised) is a 503 carrying the error text. Both routes await
`.remote.aio`, so a slow Modal call holds no worker thread.
"""

from __future__ import annotations

import os
import secrets
from typing import Any

from dreaming.contracts import Json
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()
NAME = r"^[A-Za-z0-9_.-]{1,64}$"  # it becomes part of TrainRun ids and a URL path


class CampaignRunIn(BaseModel):
    provider: str = Field(pattern=NAME)
    site: str = Field(pattern=NAME)  # "invoice_paid" or "quickbooks.invoice_paid"; launch_cells rejects unknown sites
    campaign: str | None = Field(default=None, pattern=NAME)
    limit: int = Field(default=0, ge=0)  # 0 = every cell of the site


async def call(fn_name: str, *args: Json) -> Any:
    import modal  # lazy: tests and `just dev` never reach Modal

    try:
        fn: Any = modal.Function.from_name("dreaming-campaign", fn_name)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        return await fn.remote.aio(*args)
    except Exception as e:  # whatever Modal raises (auth, NotFoundError when not deployed, a remote error) means the campaign is unavailable
        raise HTTPException(503, f"campaign {fn_name} failed: {type(e).__name__}: {e}") from e


def _authorized(authorization: str | None) -> None:
    key = os.environ.get("DREAMING_API_KEY", "")
    if not key:
        raise HTTPException(503, "campaign launch is disabled: DREAMING_API_KEY is not set on this API")
    if not secrets.compare_digest((authorization or "").encode(), f"Bearer {key}".encode()):
        raise HTTPException(401, "POST /campaign/run needs Authorization: Bearer <DREAMING_API_KEY>")


@router.post("/campaign/run")
async def run(body: CampaignRunIn, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _authorized(authorization)
    name = body.campaign or f"{body.provider}-{body.site}"
    return {"campaign_id": name, "cells": await call("launch_cells", name, [body.site], body.limit)}


@router.get("/campaign/{name}")
async def progress(name: str) -> dict[str, Any]:
    return {"campaign": name, "table": await call("progress", name)}
