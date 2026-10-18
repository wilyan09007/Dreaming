"""The trainer as a Modal app. `modal deploy apps/trainer/src/dreaming_trainer/modal_app.py` once; the API calls it by name.

    train(request: dict, site: dict) -> dict      TrainRequest + CallSite in, TrainResult out (plain dicts, Modal pickles)
    embed(texts: list[str]) -> list[list[float]]  the pinned encoder, batch size 1, for replay from another process

The image pins Python and every package, bakes the encoder weights by revision, and sets one thread everywhere; the
container's MODAL_IMAGE_ID is recorded in env_sha. Weights up to a few MB are written to Postgres as bytea; larger
files (a fine-tuned encoder) go to the Volume and the weights row carries the uri.
Checked against modal 1.6.2 decorator names (infra/SERVICES.md); not deployed from this environment.
"""

from __future__ import annotations

import os
from typing import Any

import modal
from dreaming.predictors.student import (  # one setting for the API and the trainer (ENCODER_REV env)
    ENCODER,
    ENCODER_REV,
)

image = (
    modal.Image.debian_slim(python_version="3.12.10")
    .uv_pip_install("torch==2.14.1", "sentence-transformers==6.1.0", "numpy==2.5.4", "scipy==1.17.1", "psycopg[binary]==3.3.6", "pydantic==2.14.1")
    .run_commands(f'python -c "from huggingface_hub import snapshot_download as d; d(\'{ENCODER}\', revision=\'{ENCODER_REV}\', local_dir=\'/opt/encoder\')"')
    .env({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "PYTHONHASHSEED": "0"})
    .add_local_python_source("dreaming", copy=True)
)
app = modal.App("dreaming-trainer", image=image)
volume = modal.Volume.from_name("dreaming-weights", create_if_missing=True)
db = modal.Secret.from_name("dreaming-db", required_keys=["DATABASE_URL"])


@app.function(cpu=(2.0, 2.0), memory=4096, timeout=15 * 60, retries=modal.Retries(max_retries=1), max_containers=2, volumes={"/weights": volume}, secrets=[db])
def train(request: dict[str, Any], site: dict[str, Any]) -> dict[str, Any]:
    """CPU by default: a linear head on frozen embeddings fits in seconds (S1). gpu="T4" when `finetune` is set (S8 v2)."""
    from dreaming.contracts import CallSite, TrainRequest, TrainResult
    from dreaming.predictors.candidates import CANDIDATES
    from dreaming.predictors.student import SentenceEncoder
    from dreaming.store import PostgresStore

    req, cs = TrainRequest.model_validate(request), CallSite.model_validate(site)
    store = PostgresStore(os.environ["DATABASE_URL"])
    wanted = set(req.row_ids)
    rows = [r for r in store.rows(cs.id, limit=100_000) if r.id in wanted]  # exactly the rows the request names: same request, same weights
    if len(rows) != len(wanted):
        raise LookupError(f"{cs.id}: {len(wanted) - len(rows)} of the requested rows are not in the store yet")
    c = CANDIDATES[req.candidate]
    if c.name == "config":
        raise ValueError("a router site's weights are its cheap TeacherSpec: fit them in-process (LocalTrainer), not as a job")
    encoder = SentenceEncoder("/opt/encoder", revision=ENCODER_REV) if c.needs_encoder else None
    f = c.fit(cs, rows, encoder)
    store.put_weights(f.weights_sha, f.data, {"call_site": cs.id, "rows_sha": req.rows_sha, "candidate": c.name, "image": os.environ.get("MODAL_IMAGE_ID", "")})
    env_sha = encoder.env_sha if encoder else c.load(cs, f.data, None).env_sha
    return TrainResult(call_site=cs.id, candidate=c.name, weights_sha=f.weights_sha, env_sha=env_sha, n_train=f.n_train,
                       holdout_accuracy=f.holdout_accuracy, seconds=f.seconds, where="modal:cpu").model_dump()


@app.function(cpu=(1.0, 1.0), memory=2048, timeout=60, min_containers=0, scaledown_window=300)
def embed(texts: list[str]) -> list[list[float]]:
    from dreaming.predictors.student import SentenceEncoder

    enc = SentenceEncoder("/opt/encoder", revision=ENCODER_REV)
    return [enc.embed(t, fresh=True).tolist() for t in texts]
