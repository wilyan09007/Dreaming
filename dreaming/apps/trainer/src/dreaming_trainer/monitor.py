"""The monitor: one table per site, one campaign.progress event, and the pruning rule (docs/training-plan.md §4).

Called by campaign.monitor (a Modal cron every 10 minutes, so it runs whether or not a laptop is awake), by campaign.progress
(the API reads the table by name) and by `just monitor` for a one-off table in the terminal.

    table(campaign) -> dict[site, list[row]]   one SQL query (train_runs x checkpoints), the latest rung per run: coverage at 5% and 20%,
                                               realized error, accuracy, seconds, bytes, stable, status, note
    render(table) -> str                       the same table as text, one block per site
    prune(campaign) -> list[dict]              decide() on every checkpoint, then status -> pruned with the reason in `note`; a pruned
                                               GPU cell is cancelled (FunctionCall.from_id(config.fc_id).cancel())
    decide(rows) -> list[(run_id, note)]       the four rules of §4 as a pure function over flat (run x checkpoint) rows
    recompute(run_id, rung, budget, delta)     any metric from the stored scores .npz, after the fact: the point of storing them

Rules (§4): 1 at any rung: not byte-stable, realized error over 5% on the test slice, or the cell raised. 2 at rungs 500 and 2000: within
a site and variant, a candidate whose best seed's coverage at 5% is below the median of the candidates by more than 0.10 (needs two
candidates at that rung). 3 a pruned GPU cell is cancelled; CPU cells finish on their own. 4 a candidate is judged by its best seed, and
by at least two seeds when it was launched with two or more; calibration numbers are never read, only test.
"""

from __future__ import annotations

import os
import statistics
from collections import defaultdict
from typing import Any

import psycopg
from psycopg.rows import DictRow, dict_row

BUDGET = 0.05  # rule 1's error ceiling: the 5% budget the table headlines
MARGIN = 0.10  # rule 2: two s.e. in the smokes
PRUNE_RUNGS = (500, 2000)
LIVE = ("queued", "running", "done")  # statuses a rule may still change; pruned and failed are final
COLUMNS = ("candidate", "variant", "seed", "where", "status", "rung", "coverage_05", "coverage_20", "err_05", "acc_test", "seconds", "bytes", "stable", "note")

_FETCH = """
select r.id as run_id, r.call_site, r.candidate, r.variant, r.seed, r."where", r.status, r.note, r.seconds, r.config ->> 'fc_id' as fc_id,
       c.rung, c.bytes, c.stable,
       (c.summary ->> 'coverage_05')::float8 as coverage_05, (c.summary ->> 'coverage_20')::float8 as coverage_20,
       (c.summary ->> 'err_05')::float8 as err_05, (c.summary ->> 'acc_test')::float8 as acc_test
from train_runs r left join checkpoints c on c.run_id = r.id
where r.campaign = %s order by r.call_site, r.candidate, r.variant, r.seed, c.rung
"""


def _connect() -> psycopg.Connection[DictRow]:
    return psycopg.Connection[DictRow].connect(os.environ["DATABASE_URL"], autocommit=True, row_factory=dict_row)


def fetch(campaign: str) -> list[dict[str, Any]]:
    """Flat rows, one per (run, checkpoint); a run with no checkpoint yet has rung None. The only read of the database."""
    with _connect() as conn:
        return conn.execute(_FETCH, (campaign,)).fetchall()


def campaigns() -> list[str]:
    """Campaigns with a cell started in the last 24 hours: what the cron watches."""
    with _connect() as conn:
        return [r["campaign"] for r in conn.execute("select distinct campaign from train_runs where started_at > now() - interval '24 hours' order by 1")]


def latest(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """The newest checkpoint of every run, grouped by site: the table's rows."""
    last: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r["run_id"] not in last or (r["rung"] or 0) >= (last[r["run_id"]]["rung"] or 0):
            last[r["run_id"]] = r
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in last.values():
        out[r["call_site"]].append({"run_id": r["run_id"], **{k: r.get(k) for k in COLUMNS}})
    return dict(out)


def table(campaign: str) -> dict[str, list[dict[str, Any]]]:
    return latest(fetch(campaign))


def render(tbl: dict[str, list[dict[str, Any]]]) -> str:
    def cell(v: Any) -> str:
        return "-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)

    lines: list[str] = []
    for site, rows in sorted(tbl.items()):
        lines += [site, "  " + " ".join(f"{c:>11.11}" for c in COLUMNS)]
        lines += ["  " + " ".join(f"{cell(r[c]):>11.11}" for c in COLUMNS) for r in rows]
    return "\n".join(lines)


def decide(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """(run_id, reason) for every run the four rules prune now. Pure: `rows` are fetch()'s rows, nothing else is read."""
    out: dict[str, str] = {}
    live = {r["run_id"] for r in rows if r["status"] in LIVE}
    for r in rows:  # rule 1, at any rung
        if r["run_id"] not in live or r["run_id"] in out or r["rung"] is None:
            continue
        if not r["stable"]:
            out[r["run_id"]] = f"rule 1: not byte-stable at rung {r['rung']}"
        elif (r["err_05"] or 0.0) > BUDGET:
            out[r["run_id"]] = f"rule 1: realized error {r['err_05']:.3f} over {BUDGET} on test at rung {r['rung']}"
    launched: dict[tuple[str, str, str], set[str]] = defaultdict(set)  # (site, variant, candidate) -> its runs
    for r in rows:
        launched[r["call_site"], r["variant"], r["candidate"]].add(r["run_id"])
    for rung in PRUNE_RUNGS:  # rule 2, with rule 4's best seed
        best: dict[tuple[str, str], dict[str, tuple[float, int]]] = defaultdict(dict)  # (site, variant) -> candidate -> (best coverage, seeds seen)
        for r in rows:
            if r["rung"] != rung or r["coverage_05"] is None:
                continue
            old = best[r["call_site"], r["variant"]].get(r["candidate"], (0.0, 0))
            best[r["call_site"], r["variant"]][r["candidate"]] = (max(old[0], r["coverage_05"]), old[1] + 1)
        for (site, variant), cands in best.items():
            if len(cands) < 2:
                continue
            median = statistics.median(c for c, _ in cands.values())
            for cand, (cov, seen) in cands.items():
                runs = launched[site, variant, cand]
                if cov < median - MARGIN and seen >= min(2, len(runs)):
                    for run_id in sorted(runs & live):
                        out.setdefault(run_id, f"rule 2: best seed's coverage_05 {cov:.3f} at rung {rung} is {median - cov:.3f} below the candidates' median {median:.3f}")
    return sorted(out.items())


def prune(campaign: str) -> list[dict[str, Any]]:
    """Apply decide(): status -> pruned, reason in note; cancel the pruned cells that run on a GPU. Returns what it pruned."""
    rows = fetch(campaign)
    meta = {r["run_id"]: r for r in rows}
    out: list[dict[str, Any]] = []
    for run_id, note in decide(rows):
        with _connect() as conn:
            conn.execute("update train_runs set status = 'pruned', note = %s where id = %s and status not in ('pruned', 'failed')", (note, run_id))
        r = meta[run_id]
        cancelled = bool(r["where"] == "modal:T4" and r["fc_id"])
        if cancelled:  # rule 3: a CPU cell is seconds from done, a T4 cell is minutes of money
            import modal

            modal.FunctionCall.from_id(r["fc_id"]).cancel()
        out.append({"run_id": run_id, "note": note, "cancelled": cancelled})
    return out


def recompute(run_id: str, rung: int, budget: float, delta: float) -> dict[str, float]:
    raise NotImplementedError("shortcut: np.load the checkpoint's scores_uri from the Volume and call gate.certify / gate.realized; wire it when a table needs a budget the summary lacks")
