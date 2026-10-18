"""The caller side of apps/trainer: a Trainer that runs `train` on Modal and waits for the weights.

Needs MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in the environment and `modal deploy apps/trainer` done once.
The job reads rows from Postgres itself (it has its own DATABASE_URL secret) and writes the weights row;
the API only passes the TrainRequest as a plain dict and reads the TrainResult back.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dreaming.contracts import CallSite, TraceRow, TrainRequest, TrainResult
from dreaming.predictors import Loader
from dreaming.predictors.train import LocalTrainer
from dreaming.store import Store

APP, FUNCTION = "dreaming-trainer", "train"


class ModalTrainer:
    def __init__(self, store: Store, loader: Loader, timeout_s: float = 600) -> None:
        import modal

        self.store, self.loader, self.timeout_s = store, loader, timeout_s
        self.fn: Any = modal.Function.from_name(APP, FUNCTION)  # pyright: ignore[reportUnknownMemberType]  # lazy: resolved on first call

    def train(self, req: TrainRequest, site: CallSite, rows: Sequence[TraceRow]) -> TrainResult:
        if req.candidate == "config":  # a router site's weights are its cheap TeacherSpec: fit in-process, the job refuses it (2026-10-18)
            return LocalTrainer(self.store, self.loader).train(req, site, rows)
        call = self.fn.spawn(req.model_dump(), site.model_dump())  # plain dicts: Modal pickles arguments
        return TrainResult.model_validate(call.get(timeout=self.timeout_s))
