"""The provider benchmark (docs/tracks.md §4, docs/training-plan.md §5): one table, recomputed from train_runs, checkpoints and eval_sets.

    table(store, scores, baselines, campaign) -> Benchmark    one row per (call site, candidate, variant), then the baseline rows
    markdown(b) -> str                                         the same table for the README and the Devpost
    benchmark_router(store, scores, baselines)                 GET /benchmark (JSON) and GET /benchmark.md; `create_app` includes it
    volume_scores(uri) -> arrays                               a checkpoint's .npz on the Modal Volume (tests inject a dict instead)

A candidate row is its top rung, averaged over the seeds that reached it. Per budget the cutoff is certified with gate.certify on the calib
split of the stored scores (delta = Thresholds().delta, per candidate: the runtime's delta/N applies when it chooses among N) and measured on
the test split, held out by group: `coverage` is the certificate's (calib rows above the cutoff), `test_coverage` and `test_error` are what that
cutoff does on merchants or companies it never saw (the generalization case). The `forward` slice (the last 20% by time, repeat merchants allowed:
the production-like case, "repeat traffic") gets its own cutoff, certified on its first half by time and measured on its second half
(`forward_coverage`, `forward_error`); empty when the scores hold fewer than 2 forward rows or were not recomputed (a summary keeps only
acc_forward), and the notes say so. Both are quoted: on a site whose group split is dominated by one merchant they differ a lot. Correctness is `labels[pred] == the EvalSet label`, or the npz's `ok` array when the cell
wrote one. Scores that cannot be loaded or scored fall back to the checkpoint's own summary (`source="summary"`, no calib coverage) and say why in `notes`.
Cost and latency are projections from the site's measured teacher calls (trace rows), the test coverage at the 5% budget and the audit rate, plus the
checkpoint's measured head time and, where the cell read precomputed vectors, an estimated embedding time. Audits run off the critical path.
shortcut: seeds are averaged without standard errors or paired differences to `head` (training-plan §5 asks for both over (group, seed) cells).
"""

from __future__ import annotations

import importlib
import io
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from typing import Any, Literal

import numpy as np
from dreaming.contracts import CallSite, Checkpoint, EvalSet, TeacherAnswer, Thresholds, TrainRun
from dreaming.gate import certify, realized
from dreaming.hashing import sha
from dreaming.predictors.candidates import CANDIDATES
from dreaming.store import Store
from fastapi import APIRouter, Response
from numpy.typing import NDArray
from pydantic import BaseModel

BUDGETS = (0.02, 0.05, 0.10, 0.20)
HEAD = BUDGETS.index(0.05)  # the budget the cost and latency columns, the seed-only certificate and the sort use
DELTA = Thresholds().delta
SPLITS = ("calib", "test", "forward")  # the npz's split codes 0, 1, 2 (apps/trainer campaign.py)
EMBED_MS = 5.0  # est.: bge-small on a free core (S7); cells that read precomputed vectors report 0
CPU_USD_S = 0.047 / 3600  # Modal list price per core-second, the constant the campaign bills with
TTL_S = 60.0
TRACE_ROWS = 2000

Scores = Mapping[str, NDArray[Any]]
ScoresLoader = Callable[[str], Scores]
Baselines = Callable[[str], list[dict[str, Any]]]


class Cell(BaseModel):
    budget: float
    coverage: float | None  # calib rows at or above the certified cutoff; 0 when nothing certifies; None from a summary
    test_coverage: float | None  # the same cutoff on the held-out test rows
    test_error: float | None  # realized error above it on test; None when nothing is above it
    forward_coverage: float | None = None  # a cutoff certified on the first half of the forward slice, its coverage on the second half; None without forward scores
    forward_error: float | None = None  # realized error above it on that second half; None when nothing is above it


class Money(BaseModel):  # per 1,000 calls
    usd: float
    p50_ms: float
    p95_ms: float


class Row(BaseModel):
    call_site: str
    campaign: str  # the campaign this row was trained in: the site's latest unless `campaign=` pinned one
    provider: str  # the teacher that labelled the rows ("tool" or "human" when the books are the truth)
    floor: float | None  # TeacherSpec.floor: the provider's self-disagreement; no budget below it is measurable
    candidate: str  # a trained candidate, or a baseline: "rule", "exact", "teacher alone"
    variant: str
    seeds: int  # cells averaged (0 for a baseline)
    rung: int  # rows seen at the checkpoint scored
    stable: bool  # every second forward pass was byte-equal
    source: Literal["scores", "summary", "baseline"]
    cells: list[Cell]  # one per budget
    seed_cert_s: float | None  # fit seconds to the first 5% certificate of this candidate's seed-only cells (variant "seed"), on every row of the candidate
    acc_test: float | None = None  # the picked checkpoints' own summary: right on every unseen (test) row, no cutoff; None for a baseline
    acc_forward: float | None = None  # the same on the repeat (forward) rows
    without_dreaming: Money | None
    with_dreaming: Money | None


class Benchmark(BaseModel):
    campaign: str | None  # the `campaign=` that pinned the table; None: each site's latest (see Row.campaign)
    rows: list[Row]
    notes: list[str]


def volume_scores(uri: str) -> Scores:
    """modal://<volume>/<path>.npz -> its arrays. Needs a Modal token in the environment; tests never reach it."""
    import modal  # lazy, like campaign_routes

    volume, _, path = uri.removeprefix("modal://").partition("/")
    try:
        vol: Any = modal.Volume.from_name(volume)  # pyright: ignore[reportUnknownMemberType]
        with np.load(io.BytesIO(b"".join(vol.read_file(path)))) as z:
            return {k: z[k] for k in z.files}
    except Exception as e:  # Modal's errors (auth, not found) and a truncated file are not OSErrors; the report handles one kind of failure
        raise OSError(f"{uri}: {type(e).__name__}: {e}") from e


def quickbooks_baselines(site: str) -> list[dict[str, Any]]:
    """The rule and exact-cache rows of an Intuit site: quickbooks.baselines.benchmark(site) -> [{"name", "coverage", "error"}] on the test split."""
    if not site.startswith("quickbooks."):
        return []
    mod: Any = importlib.import_module("quickbooks.baselines")  # ImportError: table() notes it and goes on
    return mod.benchmark(site)


# ---------------------------------------------------------------- one checkpoint, one baseline, one cost


def _summary_cells(ck: Checkpoint) -> list[Cell]:
    out: list[Cell] = []
    for b in BUDGETS:
        cov, err = ck.summary.get(f"coverage_{round(b * 100):02d}"), ck.summary.get(f"err_{round(b * 100):02d}")
        out.append(Cell(budget=b, coverage=None, test_coverage=cov, test_error=err if cov else None))
    return out


def _scored_cells(z: Scores, sets: Sequence[EvalSet], labels: Sequence[str] | None) -> list[Cell]:
    conf, code = np.asarray(z["conf"], np.float64), np.asarray(z["split"])
    if "ok" in z:
        ok = np.asarray(z["ok"], bool)
        if not len(ok) == len(conf) == len(code):
            raise ValueError(f"conf, split and ok differ in length ({len(conf)}, {len(code)}, {len(ok)})")
    elif labels is None:
        raise KeyError("no `ok` array in the scores and no label list to read `pred` against")
    else:  # the npz rows are calib, test, forward in the order of the EvalSets; zip(strict) is the check that these are the sets the cell scored
        truth = [t for es in sets for t in es.labels]
        ok = np.array([0 <= p < len(labels) and labels[p] == t for p, t in zip(np.asarray(z["pred"]).tolist(), truth, strict=True)], bool)
    (cc, co), (tc, to), (fc, fo) = ((conf[code == k], ok[code == k]) for k in (0, 1, 2))
    h = len(fc) // 2  # the forward rows are in time order (campaign.split): the first half certifies, the second is what the cutoff is measured on
    out: list[Cell] = []
    for b in BUDGETS:
        g = certify(cc, co, b, DELTA)
        cov, err = realized(tc, to, g.cutoff) if g.certified else (0.0, 0.0)
        fcov = ferr = None
        if h:
            fg = certify(fc[:h], fo[:h], b, DELTA)
            fcov, ferr = realized(fc[h:], fo[h:], fg.cutoff) if fg.certified else (0.0, 0.0)
        out.append(Cell(budget=b, coverage=g.coverage, test_coverage=cov, test_error=err if cov else None, forward_coverage=fcov, forward_error=ferr if fcov else None))
    return out


def _fixed_cells(cov: float, err: float) -> list[Cell]:
    """A baseline has one operating point: it answers `cov` of the rows at `err`, which is a promise only to budgets at or above `err`."""
    return [Cell(budget=b, coverage=cov if err <= b else 0.0, test_coverage=cov if err <= b else 0.0, test_error=err if err <= b else None) for b in BUDGETS]


def _mean(xs: Sequence[float | None]) -> float | None:
    v = [x for x in xs if x is not None]
    return sum(v) / len(v) if v else None


def _money(calls: Sequence[TeacherAnswer], cov: float, audit: float, pred_ms: float) -> tuple[Money, Money]:
    """(alone, with Dreaming) per 1,000 calls: `cov` of the calls are answered in `pred_ms`, the rest by the teacher, and audits still cost a call."""
    usd, lat = float(np.mean([c.cost_usd for c in calls])), np.array([c.latency_ms for c in calls], np.float64)

    def mixed(p: float) -> float:  # the predictor's share sits below every teacher latency, so the mixture's quantile is closed-form
        return pred_ms if p <= cov else float(np.quantile(lat, (p - cov) / (1 - cov)))

    alone = Money(usd=usd * 1000, p50_ms=float(np.quantile(lat, 0.5)), p95_ms=float(np.quantile(lat, 0.95)))
    return alone, Money(usd=1000 * ((1 - cov + cov * audit) * usd + pred_ms / 1000 * CPU_USD_S), p50_ms=mixed(0.5), p95_ms=mixed(0.95))


class _Scorer:
    """Checkpoint -> Cells. Reads the Volume in parallel up front (a read is ~0.7 s; the table needs dozens), resolves each checkpoint's EvalSets."""

    def __init__(self, store: Store, scores: ScoresLoader, sites: Mapping[str, CallSite], cks: Sequence[Checkpoint], notes: list[str]) -> None:
        def fetch(uri: str) -> Scores | Exception:
            try:
                return scores(uri)
            except (OSError, KeyError, ValueError) as e:
                return e

        uris = sorted({c.scores_uri for c in cks})
        with ThreadPoolExecutor(8) as pool:
            self.loaded = dict(zip(uris, pool.map(fetch, uris), strict=True))
        self.store, self.sites, self.notes = store, sites, notes
        self.triples: dict[str, dict[str, tuple[EvalSet, ...]]] = {}  # site -> eval_sha -> (calib, test, forward)

    def cells(self, site: str, ck: Checkpoint) -> tuple[list[Cell], bool]:
        """(cells, recomputed from scores)"""
        if site not in self.triples:  # Checkpoint.eval_sha = sha of the three EvalSet shas, calib first
            by = [[e for e in self.store.eval_sets(site) if e.split == s] for s in SPLITS]
            self.triples[site] = {sha([c.sha, t.sha, f.sha]): (c, t, f) for c, t, f in product(*by)}
        try:
            z = self.loaded[ck.scores_uri]
            if isinstance(z, Exception):
                raise z
            sets = self.triples[site].get(ck.eval_sha)
            if sets is None:
                raise KeyError(f"no calib, test and forward EvalSets of {site} hash to eval_sha {ck.eval_sha[:12]}")
            return _scored_cells(z, sets, self.sites[site].labels if site in self.sites else None), True
        except (OSError, KeyError, ValueError) as e:
            self.notes.append(f"{ck.run_id} rung {ck.rung}: scores not usable ({type(e).__name__}: {e}); its own summary is used")
            return _summary_cells(ck), False


# ---------------------------------------------------------------- the table


def _latest(runs: Sequence[TrainRun]) -> dict[str, str]:
    """site -> the campaign of its newest run (the runs given have a checkpoint); a run with no start time counts as oldest, ties go to the later name."""
    newest: dict[str, tuple[float, str]] = {}
    for r in runs:
        newest[r.call_site] = max(newest.get(r.call_site, (0.0, "")), (r.started_at.timestamp() if r.started_at else 0.0, r.campaign))
    return {site: c for site, (_, c) in newest.items()}


def table(store: Store, scores: ScoresLoader, baselines: Baselines | None = quickbooks_baselines, campaign: str | None = None) -> Benchmark:
    """`campaign` None: each site's latest campaign (QuickBooks trains in its own campaign, the bookkeeper sites in another); a name pins them all. `baselines(site)` -> [{"name", "coverage", "error"}]; if it raises, its rows are skipped with a note."""
    by_run: dict[str, list[Checkpoint]] = defaultdict(list)
    for ck in store.checkpoints(campaign):
        by_run[ck.run_id].append(ck)
    live = [r for r in store.train_runs(campaign) if by_run[r.id]]
    pick = _latest(live)  # site -> campaign, when no campaign pins the table

    def camp(site: str) -> str:
        return campaign or pick[site]

    groups: dict[tuple[str, str, str], list[TrainRun]] = defaultdict(list)  # (site, candidate, variant) -> its cells
    for r in live:
        if r.campaign == camp(r.call_site):
            groups[r.call_site, r.candidate, r.variant].append(r)
    top = {k: max(c.rung for r in rs for c in by_run[r.id]) for k, rs in groups.items()}  # only the cells that reached the top rung are averaged
    sites = {s.id: s for s in store.sites()}
    notes: list[str] = []
    scorer = _Scorer(store, scores, sites, [c for k, rs in groups.items() for r in rs for c in by_run[r.id] if c.rung == top[k] or k[2] == "seed"], notes)
    calls: dict[str, list[TeacherAnswer]] = {}  # the site's real teacher calls, from the live trace rows
    embed_est: set[str] = set()
    rows: list[Row] = []
    seed_s: dict[tuple[str, str], list[float]] = defaultdict(list)  # (site, candidate) -> the seed-only cells' fit seconds to the first rung that certifies 5%
    for (site, cand, variant), rs in groups.items():
        for r in rs if variant == "seed" else []:
            for ck in by_run[r.id]:
                first = scorer.cells(site, ck)[0][HEAD]
                if (first.coverage if first.coverage is not None else first.test_coverage or 0.0) > 0:
                    seed_s[site, cand].append(ck.seconds)
                    break

    def teacher(site: str) -> tuple[str, float | None]:
        t = sites[site].teacher if site in sites else None
        return (f"{t.provider}/{t.model}" if t.model else t.provider, t.floor) if t else ("unknown", None)

    for (site, cand, variant), rs in groups.items():
        provider, floor = teacher(site)
        picked = [ck for r in rs for ck in by_run[r.id] if ck.rung == top[site, cand, variant]]
        scored = [scorer.cells(site, ck) for ck in picked]
        per_seed = [cells for cells, _ in scored]
        merged = [Cell(budget=b, coverage=_mean([c[i].coverage for c in per_seed]), test_coverage=_mean([c[i].test_coverage for c in per_seed]),
                       test_error=_mean([c[i].test_error for c in per_seed]), forward_coverage=_mean([c[i].forward_coverage for c in per_seed]),
                       forward_error=_mean([c[i].forward_error for c in per_seed])) for i, b in enumerate(BUDGETS)]
        alone = dreamed = None
        if site not in calls:
            calls[site] = [r.teacher for r in store.rows(site, limit=TRACE_ROWS) if r.teacher]
        cov5 = merged[HEAD].test_coverage
        if calls[site] and cov5 is not None:
            known = CANDIDATES.get(cand)
            needs_embed = known is None or known.needs_encoder
            if needs_embed and any(not ck.embed_ms for ck in picked):
                embed_est.add(f"{site} {cand}")
            pred_ms = float(np.mean([(ck.embed_ms or (EMBED_MS if needs_embed else 0.0)) + ck.head_ms for ck in picked]))
            audit = sites[site].thresholds.audit_rate if site in sites else Thresholds().audit_rate
            alone, dreamed = _money(calls[site], cov5, audit, pred_ms)
        rows.append(Row(call_site=site, campaign=camp(site), provider=provider, floor=floor, candidate=cand, variant=variant, seeds=len(picked), rung=top[site, cand, variant],
                        stable=all(ck.stable for ck in picked), source="scores" if all(ok for _, ok in scored) else "summary", cells=merged,
                        seed_cert_s=_mean(seed_s[site, cand]), without_dreaming=alone, with_dreaming=dreamed,
                        acc_test=_mean([ck.summary.get("acc_test") for ck in picked]), acc_forward=_mean([ck.summary.get("acc_forward") for ck in picked])))
    notes += [f"{r.call_site} {r.candidate} {r.variant}: no repeat-traffic (forward) numbers: "
              + ("the scores were not recomputed and a checkpoint summary keeps only acc_forward" if r.source == "summary" else "the scores hold fewer than 2 forward rows")
              for r in rows if r.cells[HEAD].forward_coverage is None]
    if embed_est:
        notes.append(f"latency with Dreaming adds est. {EMBED_MS} ms to embed (S7, a free core; Fly's shared CPU measured ~300 ms) for {', '.join(sorted(embed_est))}: "
                     "those cells read precomputed vectors")
    notes += [f"{r.call_site} {r.candidate} {r.variant}: a second forward pass over the same rows differed on some seed (stable is false); treat these numbers as unverified"
              for r in rows if not r.stable]
    if groups and not any(k[2] == "seed" for k in groups):
        notes.append('seconds to the first certificate: no seed-only (variant "seed") runs in this campaign')

    for site in sorted({k[0] for k in groups}):  # baselines on the same held-out rows
        provider, floor = teacher(site)
        extra: list[dict[str, Any]] = []
        try:
            extra = baselines(site) if baselines else []
        except Exception as e:  # noqa: BLE001  an optional table must not take the report down; the note says why it is missing
            notes.append(f"{site}: rule and exact rows skipped ({type(e).__name__}: {e})")
        if site.startswith("quickbooks.") and not {"llm alone", "jev"} & {str(x["name"]) for x in extra}:
            notes.append(f"{site}: LLM-alone and Jev baselines pending (each needs its own labelling pass, docs/tracks.md section 4)")
        if floor:  # a teacher at its floor answers every call and disagrees with itself that often; the books (floor 0) are the truth, not a baseline
            extra.append({"name": "teacher alone", "coverage": 1.0, "error": floor})
        rows += [Row(call_site=site, campaign=camp(site), provider=provider, floor=floor, candidate=str(x["name"]), variant="-", seeds=0, rung=0, stable=True, source="baseline",
                     cells=_fixed_cells(float(x["coverage"]), float(x["error"])), seed_cert_s=None, with_dreaming=None,
                     without_dreaming=_money(calls[site], 0.0, 0.0, 0.0)[0] if x["name"] == "teacher alone" and calls.get(site) else None) for x in extra]

    # candidates by coverage at 5%, the baselines under them
    rows.sort(key=lambda r: (r.call_site, r.source == "baseline", -(r.cells[HEAD].test_coverage or 0.0), r.candidate, r.variant))
    return Benchmark(campaign=campaign, rows=rows, notes=list(dict.fromkeys(notes)))  # a checkpoint scored twice notes once


# ---------------------------------------------------------------- markdown and the routes


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.1%}"


def _ms(x: float) -> str:
    return f"{x:.1f}" if x < 10 else f"{x:.0f}"


def _both(r: Row, field: str, fmt: Callable[[float], str]) -> str:
    return " / ".join("-" if m is None else fmt(getattr(m, field)) for m in (r.without_dreaming, r.with_dreaming))


def markdown(b: Benchmark) -> str:
    head = ["site", "campaign", "provider", "floor", "candidate", "seeds", "rows seen", "repeat cov 5%", "repeat err 5%", *(f"unseen cov {round(x * 100)}%" for x in BUDGETS),
            "unseen err 5%", "seed-only cert s", "$/1k alone / with", "p50 ms alone / with", "p95 ms alone / with"]
    title = (f"Provider benchmark, {f'campaign `{b.campaign}`' if b.campaign else 'latest campaign per site'}. cov = share of rows answered at a certified error budget (Clopper-Pearson, delta {DELTA}), err = realized error on those rows; "
             "repeat = the forward slice (last 20% by time, merchants recur: production-like traffic; cutoff certified on its first half, measured on its second), "
             "unseen = group-held-out test rows (merchants and companies never seen in training).")
    lines = [title, "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in b.rows:
        variant = "" if r.variant in ("seed+real", "-") else f" ({r.variant})"
        under = [r.source != "baseline" and c.budget < (r.floor or 0.0) for c in r.cells]  # a budget under the floor is not measurable
        cells = [r.call_site, r.campaign, r.provider, _pct(r.floor), r.candidate + variant + (" †" if r.source == "summary" else ""), str(r.seeds or "-"), str(r.rung or "-"),
                 _pct(r.cells[HEAD].forward_coverage), _pct(r.cells[HEAD].forward_error),
                 *(_pct(c.test_coverage) + ("*" if u else "") for c, u in zip(r.cells, under, strict=True)), _pct(r.cells[HEAD].test_error), "-" if r.seed_cert_s is None else f"{r.seed_cert_s:.1f}",
                 _both(r, "usd", "{:.2f}".format), _both(r, "p50_ms", _ms), _both(r, "p95_ms", _ms)]
        lines.append("| " + " | ".join(cells) + " |")
    if any(r.source == "summary" for r in b.rows):
        lines += ["", "† from the checkpoint's own summary: its scores could not be recomputed."]
    if any(r.source != "baseline" and BUDGETS[0] < (r.floor or 0.0) for r in b.rows):
        lines += ["", "* a budget below the provider's floor (its self-disagreement on identical inputs) is not measurable: the share is agreement with a noisy label."]
    return "\n".join([*lines, "", *(f"- {n}" for n in b.notes)]) + "\n"


def benchmark_router(store: Store, scores: ScoresLoader, baselines: Baselines | None = quickbooks_baselines) -> APIRouter:
    router = APIRouter()
    cache: dict[str | None, tuple[float, Benchmark]] = {}  # the table reads dozens of Volume files: recompute at most once a minute
    lock = threading.Lock()  # requests run on threads: one recompute, the rest wait and hit the cache

    def get(campaign: str | None) -> Benchmark:
        with lock:
            hit = cache.get(campaign)
            if hit is None or time.monotonic() - hit[0] > TTL_S:
                hit = (time.monotonic(), table(store, scores, baselines, campaign))
                if hit[1].rows:  # an unknown ?campaign= is cheap to recompute and must not grow the cache
                    cache[campaign] = hit
            return hit[1]

    @router.get("/benchmark")
    def benchmark(campaign: str | None = None) -> Benchmark:
        return get(campaign)

    @router.get("/benchmark.md")
    def benchmark_md(campaign: str | None = None) -> Response:
        return Response(markdown(get(campaign)), media_type="text/markdown; charset=utf-8")

    return router
