"""Where rows, policies, certificates, runs, weights and events live.

One Protocol, two implementations: MemoryStore (tests, offline fast-forward) and PostgresStore (the deployment;
same tables as db/migrations/0001_init.sql, which is the schema of record). The browser reads the same tables
through Supabase Realtime, so nothing here is duplicated for the dashboard.

Rules: writes are batched by the runtime; a store error is logged and never blocks a call (fail open);
weights up to a few MB live in Postgres as bytea, larger files go to object storage behind the same two methods.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

from .contracts import CallSite, Certificate, Event, Json, Phase, Policy, Run, TraceRow

CALIBRATION = (Phase.watching, Phase.testing)

if TYPE_CHECKING:  # psycopg is the `postgres` extra and every install imports this module: it is imported where it is used
    from psycopg.sql import Composed
    from psycopg.types.json import Jsonb


class Store(Protocol):
    # call sites and their policy (one row each)
    def upsert_sites(self, sites: Sequence[CallSite]) -> None: ...
    def sites(self) -> list[CallSite]: ...
    def policy(self, call_site: str) -> Policy: ...
    def set_policy(self, policy: Policy) -> None: ...

    # trace rows
    def write_rows(self, rows: Sequence[TraceRow]) -> None: ...
    def rows(self, call_site: str, *, shadow_only: bool = False, limit: int = 10_000) -> list[TraceRow]: ...
    # shadow_only: calibration rows only (a prediction, a teacher answer, phase watching/testing), so a long dreaming
    # stream can never page the frozen calibration set out of the window
    def run_rows(self, run_id: str) -> list[TraceRow]: ...

    # runs
    def put_run(self, run: Run) -> None: ...
    def write_runs(self, runs: Sequence[Run]) -> None: ...  # one batch, like write_rows
    def runs(self, *, limit: int = 50) -> list[Run]: ...
    def run_count(self) -> int: ...
    def first_run(self) -> Run | None: ...

    # certificates and weights
    def put_certificate(self, cert: Certificate) -> None: ...
    def certificate(self, cert_id: str) -> Certificate: ...
    def certificates(self, call_site: str) -> list[Certificate]: ...
    def put_weights(self, weights_sha: str, data: bytes, meta: dict[str, str]) -> None: ...
    def weights(self, weights_sha: str) -> bytes: ...

    # events: the live feed and the audit trail
    def emit(self, event: Event) -> None: ...
    def events(self, *, after_id: str | None = None, limit: int = 200) -> list[Event]: ...


class MemoryStore:
    """Everything in dicts. Tests and offline runs; also the reference behaviour the Postgres store must match."""

    def __init__(self) -> None:
        self._sites: dict[str, CallSite] = {}
        self._policies: dict[str, Policy] = {}
        self._rows: dict[str, list[TraceRow]] = defaultdict(list)  # by call site
        self._by_run: dict[str, list[TraceRow]] = defaultdict(list)
        self._runs: dict[str, Run] = {}
        self._certs: dict[str, Certificate] = {}
        self._weights: dict[str, bytes] = {}
        self._events: list[Event] = []

    def upsert_sites(self, sites: Sequence[CallSite]) -> None:
        for s in sites:
            self._sites[s.id] = s
            self._policies.setdefault(s.id, Policy(call_site=s.id, updated_at=datetime.now(UTC)))

    def sites(self) -> list[CallSite]:
        return list(self._sites.values())

    def policy(self, call_site: str) -> Policy:
        return self._policies[call_site]

    def set_policy(self, policy: Policy) -> None:
        self._policies[policy.call_site] = policy

    def write_rows(self, rows: Sequence[TraceRow]) -> None:
        for r in rows:
            self._rows[r.call_site].append(r)
            self._by_run[r.run_id].append(r)

    def rows(self, call_site: str, *, shadow_only: bool = False, limit: int = 10_000) -> list[TraceRow]:
        rs = self._rows[call_site]
        if shadow_only:
            rs = [r for r in rs if r.agree is not None and r.phase in CALIBRATION]
        return rs[-limit:]

    def run_rows(self, run_id: str) -> list[TraceRow]:
        return sorted(self._by_run[run_id], key=lambda r: r.i)

    def put_run(self, run: Run) -> None:
        self._runs[run.id] = run

    def write_runs(self, runs: Sequence[Run]) -> None:
        for r in runs:
            self._runs[r.id] = r

    def runs(self, *, limit: int = 50) -> list[Run]:
        return sorted(self._runs.values(), key=lambda r: (r.started_at, r.n))[-limit:]

    def run_count(self) -> int:
        return len(self._runs)

    def first_run(self) -> Run | None:
        return min(self._runs.values(), key=lambda r: r.n) if self._runs else None

    def put_certificate(self, cert: Certificate) -> None:
        self._certs[cert.id] = cert

    def certificate(self, cert_id: str) -> Certificate:
        return self._certs[cert_id]

    def certificates(self, call_site: str) -> list[Certificate]:
        return [c for c in self._certs.values() if c.call_site == call_site]

    def put_weights(self, weights_sha: str, data: bytes, meta: dict[str, str]) -> None:
        self._weights[weights_sha] = data

    def weights(self, weights_sha: str) -> bytes:
        return self._weights[weights_sha]

    def emit(self, event: Event) -> None:
        self._events.append(event)

    def events(self, *, after_id: str | None = None, limit: int = 200) -> list[Event]:
        if after_id is None:
            return self._events[-limit:]
        ids = [e.id for e in self._events]
        start = ids.index(after_id) + 1 if after_id in ids else 0
        return self._events[start : start + limit]


def _j(v: Json | BaseModel) -> Jsonb | None:
    """A JSON column value. None is SQL NULL, not JSON null, so `prediction is null` in a query means what it says."""
    from psycopg.types.json import Jsonb

    if v is None:
        return None
    return Jsonb(v.model_dump(mode="json") if isinstance(v, BaseModel) else v)


def _insert(table: str, model: type[BaseModel], key: str, *, update: bool) -> Composed:
    """INSERT for a table that mirrors a contract one column per field, so the column list is the model's fields.
    `update` replaces the row on a duplicate key; otherwise the duplicate is ignored (a retried batch must not fail)."""
    from psycopg import sql

    cols = list(model.model_fields)
    sets = sql.SQL(", ").join(sql.SQL("{0} = excluded.{0}").format(sql.Identifier(c)) for c in cols if c != key)
    return sql.SQL("insert into {} ({}) values ({}) on conflict ({}) {}").format(
        sql.Identifier(table), sql.SQL(", ").join(map(sql.Identifier, cols)), sql.SQL(", ").join(map(sql.Placeholder, cols)), sql.Identifier(key),
        sql.SQL("do update set {}").format(sets) if update else sql.SQL("do nothing"))


class PostgresStore:
    """psycopg 3 against db/migrations/0001_init.sql. Construct with the pooler URL; one autocommit connection per store is
    enough for the demo, and there is no reconnect: a store error is the caller's to log (fail open, see the module docstring).

    Same behaviour as MemoryStore, with the differences the schema forces: rows, runs and events come back in (ts, id) order
    where MemoryStore keeps insertion order; sites() is ordered by id; trace rows, policies and certificates need their
    call site declared first (foreign keys); a duplicate trace row or event id is ignored instead of appended.
    """

    def __init__(self, dsn: str) -> None:
        import psycopg
        from psycopg.rows import DictRow, dict_row

        self._conn = psycopg.Connection[DictRow].connect(dsn, autocommit=True, row_factory=dict_row)
        self._conn.execute("set timezone = 'UTC'")  # every datetime we write is UTC; read them back the same way

    def close(self) -> None:
        self._conn.close()

    def upsert_sites(self, sites: Sequence[CallSite]) -> None:
        with self._conn.transaction(), self._conn.cursor() as cur:  # a site never exists without its policy row
            cur.executemany("insert into call_sites (id, spec) values (%s, %s) on conflict (id) do update set spec = excluded.spec, updated_at = now()",
                            [(s.id, _j(s)) for s in sites])
            cur.executemany("insert into policies (call_site) values (%s) on conflict (call_site) do nothing", [(s.id,) for s in sites])

    def sites(self) -> list[CallSite]:
        return [CallSite.model_validate(r["spec"]) for r in self._conn.execute("select spec from call_sites order by id")]

    def policy(self, call_site: str) -> Policy:
        r = self._conn.execute("select * from policies where call_site = %s", (call_site,)).fetchone()
        return Policy.model_validate(r) if r else Policy(call_site=call_site, updated_at=datetime.now(UTC))

    def set_policy(self, policy: Policy) -> None:
        self._conn.execute(_insert("policies", Policy, "call_site", update=True),
                           {**dict(policy), "attempt": _j(policy.attempt), "shadow": _j(policy.model_dump(mode="json")["shadow"])})

    def write_rows(self, rows: Sequence[TraceRow]) -> None:
        """One batch, one transaction. executemany (pipelined), not COPY: COPY has no ON CONFLICT."""
        params = [{**dict(r), "request": _j(r.request), "response": _j(r.response), "prediction": _j(r.prediction), "teacher": _j(r.teacher),
                   "shadow": _j(r.model_dump(mode="json")["shadow"])} for r in rows]
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.executemany(_insert("trace_rows", TraceRow, "id", update=False), params)

    def rows(self, call_site: str, *, shadow_only: bool = False, limit: int = 10_000) -> list[TraceRow]:
        cur = self._conn.execute(
            "select * from (select * from trace_rows where call_site = %s and (not %s or (agree is not null and phase in ('watching', 'testing'))) "
            "order by ts desc, id desc limit %s) t order by ts, id", (call_site, shadow_only, limit))
        return [TraceRow.model_validate(r) for r in cur]

    def run_rows(self, run_id: str) -> list[TraceRow]:
        cur = self._conn.execute("select * from trace_rows where run_id = %s order by i, id", (run_id,))
        return [TraceRow.model_validate(r) for r in cur]

    def put_run(self, run: Run) -> None:
        self._conn.execute(_insert("runs", Run, "id", update=True), {**dict(run), "input": _j(run.input), "down": _j(run.down)})

    def write_runs(self, runs: Sequence[Run]) -> None:
        params = [{**dict(r), "input": _j(r.input), "down": _j(r.down)} for r in runs]
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.executemany(_insert("runs", Run, "id", update=True), params)

    def runs(self, *, limit: int = 50) -> list[Run]:
        cur = self._conn.execute("select * from (select * from runs order by started_at desc, id desc limit %s) r order by started_at, id", (limit,))
        return [Run.model_validate(r) for r in cur]

    def run_count(self) -> int:
        r = self._conn.execute("select count(*) as n from runs").fetchone()
        return int(r["n"]) if r else 0

    def first_run(self) -> Run | None:
        r = self._conn.execute("select * from runs order by n, started_at limit 1").fetchone()
        return Run.model_validate(r) if r else None

    def put_certificate(self, cert: Certificate) -> None:
        self._conn.execute(_insert("certificates", Certificate, "id", update=True), dict(cert))

    def certificate(self, cert_id: str) -> Certificate:
        r = self._conn.execute("select * from certificates where id = %s", (cert_id,)).fetchone()
        if r is None:
            raise KeyError(cert_id)
        return Certificate.model_validate(r)

    def certificates(self, call_site: str) -> list[Certificate]:
        cur = self._conn.execute("select * from certificates where call_site = %s order by created_at, id", (call_site,))
        return [Certificate.model_validate(r) for r in cur]

    def put_weights(self, weights_sha: str, data: bytes, meta: dict[str, str]) -> None:
        self._conn.execute("insert into weights (sha, data, meta) values (%s, %s, %s) on conflict (sha) do update set data = excluded.data, meta = excluded.meta",
                           (weights_sha, data, _j(meta)))

    def weights(self, weights_sha: str) -> bytes:
        r = self._conn.execute("select data from weights where sha = %s", (weights_sha,)).fetchone()
        if r is None or r["data"] is None:  # a row with only a uri (object storage) has no bytes to give here
            raise KeyError(weights_sha)
        return bytes(r["data"])

    def emit(self, event: Event) -> None:
        self._conn.execute(_insert("events", Event, "id", update=False), {**dict(event), "data": _j(event.data)})

    def events(self, *, after_id: str | None = None, limit: int = 200) -> list[Event]:
        if after_id is None:  # the newest `limit`, oldest first
            cur = self._conn.execute("select * from (select * from events order by ts desc, id desc limit %s) e order by ts, id", (limit,))
        else:  # strictly after (ts, id), not ts alone: events share a timestamp. An unknown id pages from the start, as MemoryStore does
            cur = self._conn.execute(
                "select * from events where not exists (select 1 from events where id = %(a)s) "
                "or (ts, id) > (select ts, id from events where id = %(a)s) order by ts, id limit %(n)s", {"a": after_id, "n": limit})
        return [Event.model_validate(r) for r in cur]
