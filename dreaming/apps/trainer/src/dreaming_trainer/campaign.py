"""The campaign as a deployed Modal app: data per site on the Volume, one call per cell, a monitor on a cron (docs/training-plan.md §3-§4).

    just deploy-campaign                                   modal deploy: the functions below, the monitor cron armed
    just campaign night-1017 categorize,invoice_paid 3    launch_cells: queue the TrainRun rows, spawn the cells (here: 3) by name, print fc ids
    just monitor night-1017                                print the monitor's table once (the cron prints and prunes every 10 min)

    prepare_site(site) -> {rows, seeds, bytes} the rows a site trains on, written once to the Volume (below). Bookkeeper sites export the
                                               trace_rows from Postgres, relabelled from the StepFun record; QuickBooks sites call quickbooks.data
    train_cell(run: dict) -> dict              one TrainRun on CPU: rungs 60/200/500/2000 of a seeded row order; at each rung fit, score the three
                                               frozen EvalSets twice (stability), write Checkpoint + weights + scores .npz; running -> done | failed
    train_cell_gpu(run: dict) -> dict          the same body on a T4 for candidates with needs_gpu (finetune); the 10-GPU cap queues the rest
    launch_cells(campaign, sites, limit, force)  prepare_site for sites not on the Volume yet (every named site with --force) -> queued train_runs
                                               rows -> spawn each by name (a site that fails to prepare gets its cells as failed rows, the error in `note`)
    progress(campaign) -> dict                 monitor.table as plain rows, for the API
    monitor() [cron */10]                      monitor.prune + monitor.table for every campaign started in the last 24 h; one campaign.progress event
    cells(campaign, sites, seeded) -> [TrainRun dict]  the matrix; split(rows) -> {train, calib, test, forward}  the frozen eval sets

Two files come from the laptop, not from Postgres, and go up by hand (re-upload with -f when they grow, then re-prepare with --force):
    modal volume put dreaming-weights data/recorded.jsonl campaign/_recorded.jsonl       RecordedTeacher's file: the StepFun answers by key
    modal volume put dreaming-weights data/seeds/bookkeeper.categorize.jsonl campaign/_seeds/   predictors.seeds' rows for a site (B1's writer)
    modal run apps/trainer/src/dreaming_trainer/campaign.py --campaign X --sites categorize,match_invoice --force   re-prepare, then launch

Volume layout, per site, written by prepare_site and only read by cells:
    /weights/campaign/<site>/rows.jsonl      {"rendered", "label", "group", "ts"} per line, sorted by (ts, rendered); a row's id is its line number
    /weights/campaign/<site>/seeds.jsonl     {"rendered", "label"} per line: the LLM-written seed rows for the site's current schema, when uploaded
    /weights/campaign/<site>/meta.json       {"site_id", "labels", "encoder_sha", "env_sha", "n", "n_seeds", "prepared"}; launch_cells puts its sha in
                                             every cell's config["meta_sha"] and the cell refuses any other (site_data): no re-prepare under a campaign
    /weights/campaign/<site>/embeddings.npy  float32 [n + seeds, 384], row i = embed(text i) at batch size 1, rows then seeds (model sites only)
    /weights/campaign/<run>/<rung>.npz       idx int32, split int8 (0 calib, 1 test, 2 forward), scores float16 [N, L], conf float32, pred int32,
                                             conf2 / pred2 (the second pass); Checkpoint.eval_sha = sha of the three EvalSet shas (calib, test, forward)

Variants (training-plan §3): "real" trains on the rung's real rows; "seed+real" prepends every seed row; "seed" is day zero, one checkpoint at
rung 0 of seed rows only. Eval rows are always real rows. A site without seeds on the Volume runs "real" only (launch_cells says so).

Rules: a cell never writes a Policy; the same request gives the same weights_sha on one CPU type (not across Modal's: measured below); eval
rows are never training rows (EvalSet by group, and
forward in time); cells never call a teacher; a cell opens Postgres only to write and closes it (the session pooler's pool is small).
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

import modal
import numpy as np
import psycopg
from dreaming import telemetry
from dreaming.contracts import (
    CallSite,
    Checkpoint,
    Component,
    Decision,
    EvalSet,
    EventKind,
    Kind,
    TeacherAnswer,
    TraceRow,
    TrainRun,
)
from dreaming.gate import certify, realized
from dreaming.hashing import canonical, sha, sha_bytes
from dreaming.predictors import Predictor
from dreaming.predictors.candidates import CANDIDATES, Candidate, Fitted
from dreaming.predictors.student import ENCODER_REV, SentenceEncoder, Vec
from dreaming.predictors.train import labelled
from dreaming.store import PostgresStore
from dreaming.teachers.recorded import RecordedTeacher
from numpy.typing import NDArray
from psycopg.types.json import Jsonb

# Absolute imports in both modes: `modal deploy path/to/campaign.py` runs this file as a top-level module, and the package (installed in the
# dev venv, copied into the image below) is where its siblings live.
from dreaming_trainer import monitor as mon
from dreaming_trainer.modal_app import db, volume
from dreaming_trainer.modal_app import image as base_image

# modal_app.image already holds torch, the encoder and `dreaming` (copied); the two workflows and this package come after.
image = (
    base_image.uv_pip_install("huggingface_hub==1.33.0", "pyyaml==6.0.2")  # the hub version sentence-transformers 6.1.0 resolves to; quickbooks.data reads the dataset with it
    # one BLAS thread and one kernel family (numpy's OpenBLAS picks per CPU otherwise). Not enough for the same weights_sha on every Modal CPU:
    # head x 3 seeds x 3 rungs on the same rows matched 4 of 9 before the pin and 6 of 9 after it (smoke-1018e vs 1018c/d); predictions stay stable.
    # TEACHER=stepfun: the workflows declare the StepFun TeacherSpec, so RecordedTeacher.key matches the record file's keys (its model and params)
    .env({"OPENBLAS_NUM_THREADS": "1", "OPENBLAS_CORETYPE": "Haswell", "TEACHER": "stepfun"})
    .add_local_python_source("bookkeeper", "quickbooks", "dreaming_trainer", copy=True)
)
app = modal.App("dreaming-campaign", image=image)  # its own app: the trainer's `train` has max_containers=2

SITES = (
    "bookkeeper.categorize", "bookkeeper.match_invoice", "bookkeeper.vendor_lookup",
    "quickbooks.invoice_paid", "quickbooks.bill_account", "quickbooks.bank_match",
)
TOOL_SITES = frozenset({"bookkeeper.vendor_lookup", "quickbooks.bank_match"})  # the rest are typed model steps
RUNGS = (60, 200, 500, 2000)
BUDGETS = (0.02, 0.05, 0.10, 0.20)
DELTA = 0.05
SPLITS: tuple[Literal["calib", "test", "forward"], ...] = ("calib", "test", "forward")  # the order of the npz's split codes
ROOT = Path("/weights/campaign")
RECORDED, SEEDS = ROOT / "_recorded.jsonl", ROOT / "_seeds"  # uploaded by hand (module docstring)
SEED_TS = "2026-01-01T00:00:00+00:00"  # seed rows predate every real row: they are the prior, never the forward slice
URI = "modal://dreaming-weights/campaign"
MAX_BYTES = 8 * 1024 * 1024  # weights above this go to the Volume, not a bytea
MAX_DENSE = 64  # a label set wider than this (document ids) stores no per-label score matrix
# Modal list prices (docs/devtools.md, modal.com/pricing 2026-10-17): $0.047 per physical core-hour, $0.008 per GiB-hour, T4 $0.59/h. Approximate:
# wall clock x the request (one core, its memory, the T4); good for the table and the order of magnitude, not for billing.
CPU_CORE_S, GIB_S, T4_S = 0.047 / 3600, 0.008 / 3600, 0.59 / 3600

Rows = Sequence[Mapping[str, Any]]


# ---------------------------------------------------------------- the matrix and the frozen eval sets (pure)


def site_id(name: str) -> str:
    """'categorize' or 'bookkeeper.categorize' -> the full id; anything else is loud."""
    for s in SITES:
        if name in (s, s.split(".")[1]):
            return s
    raise ValueError(f"unknown campaign site {name!r}; choose from {', '.join(SITES)}")


def _finetune_ported() -> bool:
    with contextlib.suppress(ImportError):  # the module needs torch to run, not to import; a laptop without it simply has no GPU cells
        importlib.import_module("dreaming.predictors.finetune")
    return "finetune" in CANDIDATES


Variant = Literal["seed+real", "real", "seed"]


def cells(campaign: str, sites: Sequence[str] = (), seeded: Collection[str] = ()) -> list[dict[str, Any]]:
    """The matrix as TrainRun dicts, variant-major, then candidate, then seed, so `limit=3` on one site is three seeds of head on real rows.

    Per site and variant: head and memory x seeds 0,1,2 on CPU, plus finetune x seed 0 on a T4 when it is registered (model sites); exact x
    seeds 0,1,2 (tool sites). Variants: "real" everywhere (the smokes' baseline, S17), "seed+real" and "seed" too for the sites in `seeded`
    (those with seed rows on the Volume): the six-site matrix is 34 cells unseeded, 58 with every model site seeded.
    """
    out: list[dict[str, Any]] = []
    for s in [site_id(n) for n in sites] or SITES:
        tool = s in TOOL_SITES
        variants: list[Variant] = ["real"] if tool or s not in seeded else ["real", "seed+real", "seed"]
        plan = [(c, k, "modal:cpu") for c in (["exact"] if tool else ["head", "memory"]) for k in range(3)]
        if not tool and _finetune_ported():
            plan.append(("finetune", 0, "modal:T4"))
        for variant in variants:
            for cand, seed, where in plan:
                run = TrainRun(id=f"{campaign}:{s}:{cand}:{variant}:s{seed}", campaign=campaign, call_site=s, candidate=cand, variant=variant,
                               seed=seed, where=where, config={"rungs": [0] if variant == "seed" else list(RUNGS)})
                out.append(run.model_dump(mode="json"))
    return out


def split(rows: Rows) -> dict[str, list[int]]:
    """Row indices by role. The last 20% by time is "forward"; of the earlier rows each group goes whole to one bucket, sha(group) mod 10:
    0-1 calib, 2-3 test, the rest the training pool. Deterministic, and shared by every cell of a site (paired comparisons).
    ts strings sort as time because one exporter writes them in one format."""
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["ts"], rows[i]["rendered"]))
    cut = len(order) * 4 // 5
    out: dict[str, list[int]] = {"train": [], "calib": [], "test": [], "forward": order[cut:]}
    for i in order[:cut]:
        b = int(sha(rows[i]["group"]), 16) % 10
        out["calib" if b < 2 else "test" if b < 4 else "train"].append(i)
    return out


def eval_sets(site: str, rows: Rows, sp: Mapping[str, Sequence[int]]) -> list[EvalSet]:
    """The sha covers the rows' content, not only their line numbers: a re-prepared site (new labels) gets new eval sets, not the old ones."""
    out: list[EvalSet] = []
    for s in SPLITS:
        ids, labels, groups = [str(i) for i in sp[s]], [rows[i]["label"] for i in sp[s]], [rows[i]["group"] for i in sp[s]]
        out.append(EvalSet(sha=sha([site, s, ids, [rows[i]["rendered"] for i in sp[s]], labels, groups]), call_site=site, split=s,
                           row_ids=ids, labels=labels, groups=groups))
    return out


def rungs_for(n_pool: int, gpu: bool = False) -> list[int]:
    """60/200/500/2000 that fit the pool; a pool under 60 is one rung of everything. The fine-tune runs 500 and 2000 only (training-plan §3)."""
    out = [r for r in RUNGS if r <= n_pool] or [n_pool]
    return ([r for r in out if r >= 500] or out[-1:]) if gpu else out


def as_trace_rows(site: CallSite, rows: Rows) -> list[TraceRow]:
    """Exported rows as the TraceRows `Candidate.fit` takes: the label is the teacher's answer, the provider says where it came from."""
    return [TraceRow(id=f"campaign:{k}", run_id="campaign", i=k + 1, call_site=site.id, kind=site.kind, request=r["rendered"], request_sha=sha(r["rendered"]),
                     response=r["label"], decision=Decision.called, teacher=TeacherAnswer(text=r["label"], label=r["label"], provider="campaign", model="rows"),
                     latency_ms=0, cost_usd=0.0, ts=datetime.fromisoformat(r["ts"])) for k, r in enumerate(rows)]


def group_of(site: str, rendered: str) -> str:
    """What a site's rows are held out by: the merchant for categorize, otherwise the first line (vendor; the lookup's input is one line)."""
    return rendered.split(" *")[0] if site == "bookkeeper.categorize" else rendered.split("\n")[0]


# ---------------------------------------------------------------- one rung: fit, score twice, summarize (no I/O)


class VolumeEncoder:
    """The Encoder protocol over vectors prepare_site already computed, so a CPU cell never loads torch. `fresh` is ignored: the lookup is the embedding."""

    def __init__(self, texts: Sequence[str], emb: NDArray[np.float32], model_sha: str, env_sha: str) -> None:
        self._v: dict[str, Vec] = {t: emb[i] for i, t in enumerate(texts)}
        self.model_sha, self.env_sha = model_sha, env_sha

    def embed(self, text: str, *, fresh: bool = False) -> Vec:
        return self._v[text]


@dataclass(frozen=True)
class Rung:
    fitted: Fitted
    arrays: dict[str, NDArray[Any]]  # the npz
    stable: bool
    summary: dict[str, float]
    embed_ms: float
    head_ms: float


def _mean(a: NDArray[Any]) -> float:
    return float(a.mean()) if len(a) else 0.0


def _ece(conf: NDArray[np.float64], ok: NDArray[np.bool_], bins: int = 10) -> float:
    """Expected calibration error on `conf` (the top-2 gap the gate thresholds, not a probability), equal-width bins."""
    if not len(conf):
        return 0.0
    b = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum((b == k).mean() * abs(conf[b == k].mean() - ok[b == k].mean()) for k in range(bins) if (b == k).any()))


def summarize(conf: NDArray[np.float64], ok: NDArray[np.bool_], code: NDArray[np.int8], n_train: int) -> dict[str, float]:
    """The numbers the monitor reads in one query. Cutoffs are certified on calib and measured on test, never the same rows."""
    (cc, co), (tc, to), (fc, fo) = ((conf[code == k], ok[code == k]) for k in range(3))
    out = {"acc_test": _mean(to), "acc_forward": _mean(fo)}
    for b in BUDGETS:
        g = certify(cc, co, b, DELTA)
        cov, err = realized(tc, to, g.cutoff) if g.certified else (0.0, 0.0)
        out[f"coverage_{round(b * 100):02d}"], out[f"err_{round(b * 100):02d}"] = cov, err
    g = certify(cc, co, 0.05, DELTA)
    out["cov_fwd_05"], out["err_fwd_05"] = realized(fc, fo, g.cutoff) if g.certified else (0.0, 0.0)  # the same cutoff on the forward slice
    return out | {"bound_05": g.bound, "emp_error_05": g.emp_error, "ece": _ece(tc, to), "n_calib": float(len(cc)), "n_test": float(len(tc)),
                  "n_forward": float(len(fc)), "n_train": float(n_train)}


def run_rung(site: CallSite, cand: Candidate, train: Sequence[TraceRow], rows: Rows, sp: Mapping[str, Sequence[int]], labels: Sequence[str],
             encoder: Any) -> Rung:
    """Fit `cand` on `train`, then score every eval row twice (the second pass bypasses caches) and summarize. `stable` = the passes agree on
    (label, quantized confidence). The whole point of the campaign, in one function so a test can run it on a fake encoder."""
    fitted = cand.fit(site, train, encoder)
    predictor: Predictor = cand.load(site, fitted.data, encoder)
    idx = [i for s in SPLITS for i in sp[s]]
    texts = [rows[i]["rendered"] for i in idx]
    t0 = perf_counter()
    p1 = [predictor.predict(t) for t in texts]
    head_ms = (perf_counter() - t0) / max(len(texts), 1) * 1000
    p2 = [predictor.predict(t, fresh=True) for t in texts]
    embed_ms = 0.0  # shortcut: a VolumeEncoder looks vectors up, so a CPU cell has no embed cost to report (prepare_site paid it once); GPU cells time the real encoder
    if isinstance(encoder, SentenceEncoder) and texts:
        t0 = perf_counter()
        for t in texts[:5]:
            encoder.embed(t, fresh=True)
        embed_ms = (perf_counter() - t0) / min(len(texts), 5) * 1000
    col = {lab: k for k, lab in enumerate(labels)}
    scores = np.zeros((len(idx), len(labels) if len(labels) <= MAX_DENSE else 0), np.float16)  # shortcut: a wide label set (bank_match's document ids) keeps no per-label scores
    for n, p in enumerate(p1):
        if scores.shape[1]:
            for lab, s in (p.scores or {p.label: 1.0}).items():
                if lab in col:
                    scores[n, col[lab]] = s
    code = np.array([k for k, s in enumerate(SPLITS) for _ in sp[s]], np.int8)
    ok = np.array([p.label == rows[i]["label"] for p, i in zip(p1, idx, strict=True)], bool)
    conf, conf2 = (np.array([p.confidence for p in ps], np.float32) for ps in (p1, p2))
    pred, pred2 = (np.array([col.get(p.label, -1) for p in ps], np.int32) for ps in (p1, p2))
    stable = [(p.label, p.confidence) for p in p1] == [(p.label, p.confidence) for p in p2]
    arrays = {"idx": np.array(idx, np.int32), "split": code, "scores": scores, "conf": conf, "pred": pred, "conf2": conf2, "pred2": pred2}
    return Rung(fitted, arrays, stable, summarize(conf.astype(np.float64), ok, code, fitted.n_train), embed_ms, head_ms)


# ---------------------------------------------------------------- prepare_site: rows to the Volume


def _import(module: str, site: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise RuntimeError(f"{site}: cannot import {module} ({e}); is it in the campaign image (add_local_python_source)?") from e


def call_site(site: str) -> CallSite:
    from dreaming import registry

    _import(f"{site.split('.')[0]}.workflow", site)  # declaring the workflow registers its sites
    for s in registry.sites():
        if s.id == site:
            return s
    raise LookupError(f"{site}: not declared by its workflow (registry has {sorted(s.id for s in registry.sites())})")


def labels_for(cs: CallSite, trace: Sequence[TraceRow], recorded: RecordedTeacher | None) -> list[dict[str, Any]]:
    """A site's trace rows as training rows. A tool site (`recorded` None) keeps the tool's exact answer. A model site's label is the StepFun
    answer recorded for the same rendered input (RecordedTeacher.key), whatever teacher the run itself used; a row with no recorded answer,
    or an unparsed one, is dropped: never a rule label."""
    out: list[dict[str, Any]] = []
    for r in trace:
        if recorded is None:
            pairs = labelled([r])
        else:
            text = r.request if isinstance(r.request, str) else canonical(r.request).decode()
            try:
                label = recorded.ask(cs, text).label
            except LookupError:
                continue
            pairs = [(text, label)] if label else []
        out += [{"rendered": text, "label": label, "group": group_of(cs.id, text), "ts": r.ts.isoformat()} for text, label in pairs]
    return out


def _trace_rows(cs: CallSite) -> list[dict[str, Any]]:
    """Every bookkeeper trace row with a label (labels_for), from Postgres and the record file on the Volume."""
    store = PostgresStore(os.environ["DATABASE_URL"])
    try:
        trace = store.rows(cs.id, limit=100_000)
    finally:
        store.close()
    recorded = None
    if cs.id not in TOOL_SITES:
        if not RECORDED.is_file():
            raise FileNotFoundError(f"{RECORDED} is not on the Volume: `modal volume put dreaming-weights data/recorded.jsonl campaign/_recorded.jsonl`")
        recorded = RecordedTeacher(RECORDED)
    rows = labels_for(cs, trace, recorded)
    print(f"{cs.id}: {len(rows)} of {len(trace)} trace rows have a label", flush=True)
    return rows


def seed_rows(cs: CallSite, root: Path) -> list[dict[str, str]]:
    """The LLM-written seed rows for the site's current schema, {"rendered", "label"}, from <root>/<site id>.jsonl; [] when there is none.
    shortcut: the file is predictors.seeds' (B1's branch: {"key": sha([labels, descriptions]), "id", "label", "text", "model"}); read it with
    seeds.rows once that branch is merged."""
    path = root / f"{cs.id}.jsonl"
    if cs.labels is None or not path.is_file():
        return []
    key = sha([cs.labels, cs.descriptions])
    return [{"rendered": d["text"], "label": d["label"]} for d in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if d["key"] == key]


def _export(cs: CallSite) -> list[dict[str, str]]:
    if cs.id.startswith("quickbooks."):
        data = _import("quickbooks.data", cs.id)
        # container disk, not the Volume: the raw dataset is not a campaign artifact, and the three sites prepare in parallel
        raw: list[dict[str, Any]] = data.rows(cs.name, data.download(Path("/tmp/quickbooks")))
    else:
        raw = _trace_rows(cs)
        # the demo stream replays one ledger, so a text recurs with the same label on every batch: identical (text, label) rows
        # carry no information and swamped the memory's vote (acc 0.2 at 9k rows vs 0.9 at 600 unique, 2026-10-18); keep the earliest
        raw = list({(r["rendered"], r["label"]): r for r in sorted(raw, key=lambda r: str(r["ts"]), reverse=True)}.values())
    rows = [{k: str(r[k]) for k in ("rendered", "label", "group", "ts")} for r in raw]
    return sorted(rows, key=lambda r: (r["ts"], r["rendered"]))


@app.function(cpu=(1.0, 1.0), memory=2048, timeout=40 * 60, max_containers=4, volumes={"/weights": volume}, secrets=[db])
def prepare_site(site: str) -> dict[str, Any]:
    """Export a site's rows and its seed rows, embed them once (batch size 1), write the files, commit the Volume. Overwrites a prepared site."""
    site = site_id(site)
    cs = call_site(site)
    rows = _export(cs)
    if not rows:
        raise ValueError(f"{site}: no labelled rows to train on")
    seeds = [] if site in TOOL_SITES else seed_rows(cs, SEEDS)
    d = ROOT / site
    d.mkdir(parents=True, exist_ok=True)
    encoder_sha = env_sha = ""
    if site not in TOOL_SITES:  # `exact` needs no vectors, and a tool site can have tens of thousands of rows
        enc = SentenceEncoder("/opt/encoder", revision=ENCODER_REV)
        encoder_sha, env_sha = enc.model_sha, enc.env_sha
        vecs = [enc.embed(r["rendered"]) for r in rows + seeds]
        # the encoder's half of Checkpoint.stable: CPU cells look these vectors up, so its second forward pass happens here, on a sample
        if not all(np.array_equal(enc.embed(r["rendered"], fresh=True), v) for r, v in zip(rows[:64], vecs[:64], strict=True)):
            raise RuntimeError(f"{site}: the encoder is not byte-stable at batch 1 (a thread setting leaked?); nothing written")
        np.save(d / "embeddings.npy", np.stack(vecs).astype(np.float32))
    labels = cs.labels if cs.labels is not None else sorted({r["label"] for r in rows})
    (d / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "seeds.jsonl").unlink(missing_ok=True)  # a re-prepare without seeds leaves none behind
    if seeds:
        (d / "seeds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in seeds))
    meta = json.dumps({"site_id": site, "labels": labels, "encoder_sha": encoder_sha, "env_sha": env_sha, "n": len(rows), "n_seeds": len(seeds),
                       "prepared": datetime.now(UTC).isoformat()})
    (d / "meta.json").write_text(meta)
    volume.commit()
    return {"site": site, "rows": len(rows), "seeds": len(seeds), "bytes": sum(f.stat().st_size for f in d.iterdir()), "meta_sha": sha_bytes(meta.encode())}


def site_data(d: Path, meta_sha: str | None) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """A site's meta, rows and seeds from its Volume directory, checked as one snapshot: the meta.json launch_cells saw (`meta_sha`, when the
    run carries one) and row counts that agree with it and with embeddings.npy. A re-prepare under a running campaign (seen 2026-10-18:
    another app overwrote categorize mid-launch; 4 cells raised IndexError, one trained on the other labels) fails loudly here instead."""
    raw = (d / "meta.json").read_bytes()
    meta = json.loads(raw)
    if meta_sha is not None and sha_bytes(raw) != meta_sha:
        raise RuntimeError(f"{d.name}: meta.json changed since launch (prepared {meta.get('prepared')}); the site was re-prepared under this campaign")
    rows: list[dict[str, Any]] = [json.loads(line) for line in (d / "rows.jsonl").read_text().splitlines()]
    seeds: list[dict[str, Any]] = [json.loads(line) for line in (d / "seeds.jsonl").read_text().splitlines()] if (d / "seeds.jsonl").is_file() else []
    n_emb = len(np.load(d / "embeddings.npy", mmap_mode="r")) if (d / "embeddings.npy").is_file() else len(rows) + len(seeds)
    if (len(rows), len(seeds), n_emb) != (meta["n"], meta.get("n_seeds", len(seeds)), len(rows) + len(seeds)):
        raise RuntimeError(f"{d.name}: rows.jsonl ({len(rows)}), seeds.jsonl ({len(seeds)}), embeddings.npy ({n_emb}) and meta.json (n {meta['n']}) disagree")
    return meta, rows, seeds


# ---------------------------------------------------------------- the cell


_CHECKPOINT = (
    "insert into checkpoints (run_id, rung, weights_sha, eval_sha, scores_uri, seconds, embed_ms, head_ms, bytes, stable, summary, ts) "
    "values (%(run_id)s, %(rung)s, %(weights_sha)s, %(eval_sha)s, %(scores_uri)s, %(seconds)s, %(embed_ms)s, %(head_ms)s, %(bytes)s, %(stable)s, %(summary)s, %(ts)s) "
    "on conflict (run_id, rung) do update set weights_sha = excluded.weights_sha, eval_sha = excluded.eval_sha, scores_uri = excluded.scores_uri, "
    "seconds = excluded.seconds, embed_ms = excluded.embed_ms, head_ms = excluded.head_ms, bytes = excluded.bytes, stable = excluded.stable, "
    "summary = excluded.summary, ts = excluded.ts"
)


def _pg() -> psycopg.Connection[tuple[Any, ...]]:
    return psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)


def _write_rung(run: TrainRun, rung: int, r: Rung, ck: Checkpoint, sets: Sequence[EvalSet]) -> None:
    """The weights, the eval sets, the checkpoint: Postgres is open for exactly this and closed again."""
    f = r.fitted
    meta = {"call_site": run.call_site, "candidate": run.candidate, "campaign": run.campaign, "run_id": run.id, "rung": str(rung),
            "image": os.environ.get("MODAL_IMAGE_ID", "")}
    data = f.data
    if len(f.data) > MAX_BYTES:  # shortcut: PostgresStore.put_weights has no uri parameter, so the row holds b"" and the uri in meta; the Volume file is the weights
        (ROOT / run.id / f"{rung}.weights").write_bytes(f.data)
        data, meta["uri"] = b"", f"{URI}/{run.id}/{rung}.weights"
        volume.commit()
    store = PostgresStore(os.environ["DATABASE_URL"])
    try:
        store.put_weights(f.weights_sha, data, meta)
    finally:
        store.close()
    with _pg() as conn, conn.cursor() as cur:
        cur.executemany("insert into eval_sets (sha, call_site, split, row_ids, labels, groups) values (%s, %s, %s, %s, %s, %s) on conflict (sha) do nothing",
                        [(s.sha, s.call_site, s.split, Jsonb(s.row_ids), Jsonb(s.labels), Jsonb(s.groups)) for s in sets])
        cur.execute(_CHECKPOINT, {**ck.model_dump(), "summary": Jsonb(ck.summary)})


def _candidate(name: str) -> Candidate:
    if name not in CANDIDATES:
        importlib.import_module(f"dreaming.predictors.{name}")  # a ported candidate registers itself on import; a real ImportError surfaces here
    return CANDIDATES[name]


def _cell(run_dict: dict[str, Any], gpu: bool) -> dict[str, Any]:
    run = TrainRun.model_validate(run_dict)
    t0 = perf_counter()
    with _pg() as conn:  # a cell the monitor pruned before it started never starts (rule 2: pruned for the next rung)
        started = conn.execute('insert into train_runs (id, campaign, call_site, candidate, variant, seed, config, "where", status, started_at) '
                               "values (%s, %s, %s, %s, %s, %s, %s, %s, 'running', now()) "
                               "on conflict (id) do update set status = 'running', started_at = now(), finished_at = null, note = null "
                               "where train_runs.status <> 'pruned' returning id",
                               (run.id, run.campaign, run.call_site, run.candidate, run.variant, run.seed, Jsonb(run.config), run.where)).fetchone()
    if started is None:
        return {"id": run.id, "seconds": 0.0, "skipped": "pruned"}
    try:
        volume.reload()
        d = ROOT / run.call_site
        meta, rows, seeds = site_data(d, cast(str | None, run.config.get("meta_sha")))
        if run.variant != "real" and not seeds:
            raise ValueError(f"{run.call_site}: variant {run.variant} needs seed rows and the Volume has none (upload them, then re-prepare with --force)")
        tool = run.call_site in TOOL_SITES
        site = CallSite(id=run.call_site, workflow=run.call_site.split(".")[0], name=run.call_site.split(".")[1], kind=Kind.tool if tool else Kind.model,
                        labels=None if tool else meta["labels"])
        cand = _candidate(run.candidate)
        encoder: Any = None
        if cand.needs_encoder and gpu:
            encoder = SentenceEncoder("/opt/encoder", revision=ENCODER_REV)
        elif cand.needs_encoder:
            encoder = VolumeEncoder([r["rendered"] for r in rows + seeds], np.load(d / "embeddings.npy"), meta["encoder_sha"], meta["env_sha"])
        sp = split(rows)
        sets = eval_sets(run.call_site, rows, sp)
        eval_sha = sha([s.sha for s in sets])
        pool = sp["train"]
        if not pool:
            raise ValueError(f"{run.call_site}: the split left no training rows ({len(rows)} rows, {len({r['group'] for r in rows})} groups)")
        order = [pool[int(i)] for i in np.random.default_rng(run.seed).permutation(len(pool))]  # shortcut: the seed picks which rows come first; Candidate.fit has no seed, so its holdout draw is fixed
        trace = as_trace_rows(site, [rows[i] for i in order])
        prior = as_trace_rows(site, [{**s, "ts": SEED_TS} for s in seeds]) if run.variant != "real" else []  # the seed rows come first, at every rung
        out = ROOT / run.id
        out.mkdir(parents=True, exist_ok=True)
        for n in [0] if run.variant == "seed" else rungs_for(len(pool), gpu):
            t1 = perf_counter()
            r = run_rung(site, cand, prior + trace[:n], rows, sp, meta["labels"], encoder)
            cast(Any, np.savez)(out / f"{n}.npz", **r.arrays)
            t2 = perf_counter()
            volume.commit()
            t3 = perf_counter()
            ck = Checkpoint(run_id=run.id, rung=n, weights_sha=r.fitted.weights_sha, eval_sha=eval_sha, scores_uri=f"{URI}/{run.id}/{n}.npz",
                            seconds=r.fitted.seconds, embed_ms=r.embed_ms, head_ms=r.head_ms, bytes=len(r.fitted.data), stable=r.stable,
                            summary=r.summary, ts=datetime.now(UTC))
            _write_rung(run, n, r, ck, sets)
            print(f"{run.id} rung {n}: fit+score {t2 - t1:.2f}s, volume commit {t3 - t2:.2f}s, postgres {perf_counter() - t3:.2f}s", flush=True)
    except Exception as e:
        with contextlib.suppress(Exception), _pg() as conn:  # the original error is the one to raise, even if the database is why we failed
            conn.execute("update train_runs set status = 'failed', note = %s, finished_at = now(), seconds = %s where id = %s",
                         (f"{type(e).__name__}: {e}", perf_counter() - t0, run.id))
        raise
    seconds = perf_counter() - t0
    with _pg() as conn:  # a cell the monitor pruned while it ran stays pruned: the finish only records time and money
        conn.execute("update train_runs set status = case when status = 'pruned' then status else 'done' end, finished_at = now(), seconds = %s, cost_usd = %s where id = %s",
                     (seconds, seconds * (CPU_CORE_S + GIB_S * (4 if gpu else 2) + (T4_S if gpu else 0.0)), run.id))
    return {"id": run.id, "seconds": seconds}


@app.function(cpu=(1.0, 1.0), memory=2048, timeout=20 * 60, retries=modal.Retries(max_retries=1), max_containers=12, volumes={"/weights": volume}, secrets=[db])
def train_cell(run: dict[str, Any]) -> dict[str, Any]:
    return _cell(run, gpu=False)


@app.function(gpu="T4", cpu=(1.0, 1.0), memory=4096, timeout=30 * 60, retries=modal.Retries(max_retries=1), max_containers=8, volumes={"/weights": volume}, secrets=[db])
def train_cell_gpu(run: dict[str, Any]) -> dict[str, Any]:
    return _cell(run, gpu=True)


# ---------------------------------------------------------------- launch, progress, monitor


@app.function(memory=1024, secrets=[db], timeout=45 * 60, volumes={"/weights": volume})
def launch_cells(campaign: str, sites: list[str], limit: int, force: bool = False, only: Sequence[str] = ()) -> list[dict[str, Any]]:
    """Prepare the sites the Volume lacks (every named site with `force`: new labels or seed rows were uploaded), queue the cells in
    train_runs, then spawn each by name on the deployed app (they outlive this call). The fc id lands in config["fc_id"].
    `only`: run ids to spawn and nothing else (a rerun of failed cells: the cell's own insert sets them running again and its checkpoints
    overwrite the old ones, rung by rung)."""
    names = [site_id(n) for n in sites] or list(SITES)
    missing = names if force else [s for s in names if not (ROOT / s / "meta.json").is_file()]
    done: list[Any] = list(cast(Any, prepare_site).map(missing, return_exceptions=True)) if missing else []  # in parallel, one container per site
    broken = {s: f"prepare_site: {type(r).__name__}: {r}" for s, r in zip(missing, done, strict=True) if isinstance(r, BaseException)}
    volume.reload()  # see what the prepare containers committed
    seeded = {s for s in names if (ROOT / s / "seeds.jsonl").is_file()}
    for s in names:
        if s not in seeded and s not in TOOL_SITES and s not in broken:
            print(f"{s}: no seed rows on the Volume ({SEEDS / s}.jsonl missing, or uploaded after the site was prepared: --force); "
                  "variants seed+real and seed skipped, real runs", flush=True)
    # the meta.json every cell must find (site_data): the one prepare_site just wrote, else the one on the Volume now
    meta_sha = {s: r["meta_sha"] for s, r in zip(missing, done, strict=True) if not isinstance(r, BaseException)}
    meta_sha |= {s: sha_bytes((ROOT / s / "meta.json").read_bytes()) for s in names if s not in meta_sha and s not in broken}
    todo = [c for c in cells(campaign, names, seeded) if not only or c["id"] in only][: limit or None]
    if only and len(todo) != len(set(only)):
        raise ValueError(f"--only names {sorted(set(only) - {c['id'] for c in todo})} that are not cells of this campaign and sites")
    for c in todo:
        if c["call_site"] in meta_sha:
            c["config"]["meta_sha"] = meta_sha[c["call_site"]]
    by_name: Any = cast(Any, modal.Function).from_name  # Modal's Function is untyped; spawning by name keeps the calls alive after the caller exits
    fn_cpu, fn_gpu = by_name("dreaming-campaign", "train_cell"), by_name("dreaming-campaign", "train_cell_gpu")
    with _pg() as conn:
        with conn.cursor() as cur:
            cur.executemany('insert into train_runs (id, campaign, call_site, candidate, variant, seed, config, "where", status, note) '
                            "values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) on conflict (id) do nothing",
                            [(c["id"], c["campaign"], c["call_site"], c["candidate"], c["variant"], c["seed"], Jsonb(c["config"]), c["where"],
                              "failed" if c["call_site"] in broken else "queued", broken.get(c["call_site"])) for c in todo])
        for c in todo:
            if c["call_site"] in broken:  # no spawn: the monitor's table shows the cells as failed with the prepare error
                c["status"], c["note"], c["config"]["fc_id"] = "failed", broken[c["call_site"]], ""
                continue
            fc = (fn_gpu if c["where"] == "modal:T4" else fn_cpu).spawn(c)
            c["config"]["fc_id"] = fc.object_id
            conn.execute("update train_runs set config = config || %s where id = %s", (Jsonb({"fc_id": fc.object_id}), c["id"]))
    return todo


@app.function(memory=1024, secrets=[db], timeout=60)
def progress(campaign: str) -> dict[str, Any]:
    return mon.table(campaign)


@app.function(schedule=modal.Cron("*/10 * * * *"), memory=1024, secrets=[db], timeout=5 * 60)
def monitor() -> None:
    """Prune, then print and publish the table of every campaign started in the last day (so the table shows what the rules just did)."""
    store = PostgresStore(os.environ["DATABASE_URL"])
    try:
        for name in mon.campaigns():
            pruned = mon.prune(name)
            tbl = mon.table(name)
            print(f"{name}: {len(pruned)} pruned\n{mon.render(tbl)}")
            telemetry.emit(store, Component.trainer, EventKind.campaign_progress, f"campaign {name}: {sum(map(len, tbl.values()))} cells, {len(pruned)} pruned",
                           data={"campaign": name, "table": cast(Any, tbl)})
    finally:
        store.close()


@app.local_entrypoint()
def launch(campaign: str = "night-1017", sites: str = "", limit: int = 0, force: bool = False, only: str = "") -> None:
    """`sites` is comma-separated, full ids or bare names; empty means all six. `--force` re-prepares the named sites first; `--only` is a
    comma-separated list of run ids to (re)spawn."""
    for c in launch_cells.remote(campaign, [s for s in sites.split(",") if s], limit, force, [o for o in only.split(",") if o]):
        print(c["id"], c["config"]["fc_id"])
