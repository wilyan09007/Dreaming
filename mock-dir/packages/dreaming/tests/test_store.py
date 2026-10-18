"""MemoryStore is the reference behaviour: every test below runs against it and against PostgresStore, so the two cannot drift.

The Postgres half needs the scratch server below (schema re-applied from db/migrations/0001_init.sql, tables truncated before
each test) and skips itself when there is none, so CI without a Postgres still passes.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg
import pytest
from conftest import transactions
from dreaming.contracts import (
    CallSite,
    Certificate,
    Component,
    Decision,
    Event,
    EventKind,
    Hashes,
    Kind,
    Phase,
    Policy,
    Prediction,
    Run,
    TeacherAnswer,
    TeacherSpec,
    Thresholds,
    TraceRow,
)
from dreaming.replay import replay
from dreaming.store import MemoryStore, PostgresStore, Store
from test_runtime import build

PG = "postgresql://dreaming@127.0.0.1:54329/dreaming"
MIGRATION = Path(__file__).parents[3] / "db" / "migrations" / "0001_init.sql"
T0 = datetime(2026, 3, 1, 12, 0, 0, 123_456, tzinfo=UTC)  # with microseconds on purpose: timestamptz must keep every one


# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def pg_ready() -> None:
    """Apply the schema once per module (it is idempotent), or skip every Postgres test."""
    try:
        conn = psycopg.connect(PG, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError as e:
        pytest.skip(f"no scratch Postgres at {PG}: {e}")
    with conn:
        conn.execute(MIGRATION.read_bytes())


@pytest.fixture
def pg_store(pg_ready: None) -> Iterator[PostgresStore]:
    with psycopg.connect(PG, autocommit=True) as conn:
        conn.execute("truncate call_sites, policies, runs, trace_rows, certificates, weights, events")
    store = PostgresStore(PG)
    yield store
    store.close()


@pytest.fixture(params=["memory", "postgres"])
def store(request: pytest.FixtureRequest) -> Store:
    return MemoryStore() if request.param == "memory" else request.getfixturevalue("pg_store")


# ---------------------------------------------------------------- data: every nested part filled, so a JSON round trip has something to lose


def mk_site(name: str = "a") -> CallSite:
    teacher = TeacherSpec(provider="tool", model="m", params={"temperature": 0.0, "stop": ["\n"], "n": {"k": None}}, price_in=0.25, price_call=0.001)
    return CallSite(id=f"w.{name}", workflow="w", name=name, kind=Kind.tool, labels=["x", "y"], descriptions={"x": "first"}, teacher=teacher,
                    fallback="snapshot", replayable=False, thresholds=Thresholds(watch_after=10, calibrate_after=20, audit_rate=0.1, delta=0.01))


def mk_row(n: int, *, site_id: str = "w.a", run: str = "r1", i: int | None = None, agree: bool | None = None,
           phase: Phase = Phase.watching) -> TraceRow:
    """Row n. Ids and timestamps ascend with n, so (ts, id) order is insertion order. agree set: a shadow row with both sides."""
    shadow = agree is not None
    prediction = Prediction(label="Meals", confidence=0.9876, scores={"Meals": 0.9876, "Travel": 0.0124}, cert_id="c1", cost_usd=1e-6, latency_ms=2,
                            hashes=Hashes(prompt="p", weights="w", env="e"))
    teacher = TeacherAnswer(text="Meals", label="Meals", provider="fake", model="m", cost_usd=0.01, latency_ms=800, usage={"in": 5, "out": 1}, recorded=True)
    return TraceRow(
        id=f"row{n:03d}", run_id=run, i=n if i is None else i, call_site=site_id, kind=Kind.model, phase=phase, request={"d": f"txn {n}", "xs": [1, 2.5, None]},
        request_sha=f"sha{n}", response="Meals", decision=Decision.audited if shadow else Decision.called, prediction=prediction if shadow else None,
        teacher=teacher, agree=agree, latency_ms=800, cost_usd=0.01, error="boom" if n % 5 == 0 else None, ts=T0 + timedelta(seconds=n, microseconds=n),
    )


def mk_run(n: int, input: object) -> Run:
    return Run(id=f"run{n}", workflow="w", mode="fast_forward", input=input, started_at=T0 + timedelta(minutes=n))  # type: ignore[arg-type]


def mk_cert(n: int, call_site: str = "w.a") -> Certificate:
    return Certificate(id=f"cert{n}", call_site=call_site, teacher_sha="t" * 64, n_above=0, weights_sha="w" * 64, env_sha="e" * 64, budget=0.05, delta=0.05, cutoff=0.8125, coverage=0.6,
                       emp_error=0.0123, bound=0.0419, n_calib=500, n_errors=6, calib_sha="c" * 64, created_at=T0 + timedelta(minutes=n, microseconds=n))


def mk_event(n: int, seconds: int) -> Event:
    return Event(id=f"evt{n}", ts=T0 + timedelta(seconds=seconds), component=Component.lib, kind=EventKind.site_phase, message=f"event {n}",
                 run_id="r1" if n % 2 else None, call_site="w.a", data={"from": "awake", "to": "watching", "xs": [1, 2.5]},
                 error="boom" if n == 3 else None, trace_id="0af7651916cd43dd8448eb211c80319c" if n == 2 else None)


# ---------------------------------------------------------------- the behaviour both stores share


def test_sites_get_a_default_policy(store: Store):
    a, b = mk_site("a"), mk_site("b")
    store.upsert_sites([b, a])
    assert sorted(store.sites(), key=lambda s: s.id) == [a, b]  # order is the store's own; the spec round-trips whole
    p = store.policy("w.a")
    assert p == Policy(call_site="w.a", updated_at=p.updated_at) and p.phase == Phase.awake


def test_redeclaring_a_site_replaces_its_spec_and_keeps_its_policy(store: Store):
    a = mk_site()
    store.upsert_sites([a])
    store.set_policy(store.policy("w.a").model_copy(update={"phase": Phase.dreaming, "rows": 7}))
    a2 = a.model_copy(update={"labels": ["x", "y", "z"]})
    store.upsert_sites([a2])
    assert store.sites() == [a2]
    assert store.policy("w.a").phase == Phase.dreaming and store.policy("w.a").rows == 7


def test_set_policy_round_trip(store: Store):
    store.upsert_sites([mk_site()])
    p = Policy(call_site="w.a", phase=Phase.down, cert_id="c1", cutoff=0.8125, weights_sha="ab" * 32, rows=12, shadow_rows=7, outage=True, updated_at=T0)
    store.set_policy(p)
    assert store.policy("w.a") == p
    store.set_policy(p.model_copy(update={"phase": Phase.dreaming, "outage": False, "cutoff": None}))  # replaces the row: cutoff is not kept
    got = store.policy("w.a")
    assert (got.phase, got.outage, got.cutoff, got.cert_id) == (Phase.dreaming, False, None, "c1")


def test_rows_come_back_in_order_and_limit_keeps_the_last(store: Store):
    store.upsert_sites([mk_site("a"), mk_site("b")])
    rows = [mk_row(1), mk_row(2, agree=True), mk_row(3, agree=False), mk_row(4), mk_row(5, agree=True)]
    other = mk_row(6, site_id="w.b", run="r2", agree=True)
    store.write_rows(rows[:2])
    store.write_rows([*rows[2:], other])
    store.write_rows([])
    assert store.rows("w.a") == rows  # request JSON, prediction and teacher intact; null prediction and teacher on the unshadowed rows
    assert store.rows("w.a", limit=2) == rows[-2:]
    assert store.rows("w.b") == [other]
    assert store.rows("w.none") == []


def test_shadow_only_means_calibration_rows(store: Store):
    """A prediction, a teacher answer, and a phase before promotion. A dreaming-phase row with both sides is an audit, not calibration."""
    store.upsert_sites([mk_site()])
    rows = [mk_row(1), mk_row(2, agree=True), mk_row(3, agree=False), mk_row(4), mk_row(5, agree=True, phase=Phase.testing),
            mk_row(6, agree=True, phase=Phase.dreaming)]
    store.write_rows(rows)
    assert store.rows("w.a", shadow_only=True) == [rows[1], rows[2], rows[4]]  # a disagreement is calibration data too
    assert store.rows("w.a", shadow_only=True, limit=2) == [rows[2], rows[4]]


def test_run_rows_follow_the_order_of_the_calls(store: Store):
    store.upsert_sites([mk_site()])
    rows = [mk_row(1, i=3), mk_row(2, i=1), mk_row(3, run="r2", i=1), mk_row(4, i=2)]
    store.write_rows(rows)
    assert store.run_rows("r1") == [rows[1], rows[3], rows[0]]
    assert store.run_rows("r2") == [rows[2]]
    assert store.run_rows("none") == []


def test_runs_are_ordered_by_start_and_limit_keeps_the_last(store: Store):
    runs = [mk_run(1, {"d": "ADOBE", "amt": 5.0}), mk_run(2, "a string"), mk_run(3, None), mk_run(4, [1, 2.5, "x"])]
    for r in (runs[2], runs[0], runs[3], runs[1]):  # they arrive in another order than they started
        store.put_run(r)
    assert store.runs() == runs
    assert store.runs(limit=2) == runs[2:]
    done = runs[0].model_copy(update={"finished_at": T0 + timedelta(minutes=1, microseconds=7), "calls": 3, "cost_usd": 0.03, "latency_ms": 2400,
                                      "failed": True, "stuck_at": "w.a", "replayable": 2})
    store.put_run(done)  # same id: replaced, not added
    assert store.runs() == [done, *runs[1:]]


def test_certificate_round_trip(store: Store):
    store.upsert_sites([mk_site("a"), mk_site("b")])
    c1, c2, c3 = mk_cert(1), mk_cert(2), mk_cert(3, "w.b")
    for c in (c1, c2, c3):
        store.put_certificate(c)
    got = store.certificate("cert1")
    assert got == c1 and got.created_at.isoformat() == c1.created_at.isoformat()  # every field; the datetime to the microsecond, in UTC
    assert store.certificates("w.a") == [c1, c2] and store.certificates("w.b") == [c3]
    revoked = c1.model_copy(update={"status": "revoked"})
    store.put_certificate(revoked)  # same id: replaced
    assert store.certificate("cert1") == revoked and store.certificates("w.a") == [revoked, c2]
    with pytest.raises(KeyError):
        store.certificate("none")


def test_weights_bytes_round_trip(store: Store):
    data = bytes(range(256)) * 4096  # 1 MiB, every byte value, NULs included
    store.put_weights("sha1", data, {"call_site": "w.a", "where": "local:cpu"})
    got = store.weights("sha1")
    assert isinstance(got, bytes) and got == data
    store.put_weights("sha1", data, {})  # content-addressed: putting the same file again is fine
    assert store.weights("sha1") == data
    with pytest.raises(KeyError):
        store.weights("none")


def test_events_page_by_cursor(store: Store):
    evs = [mk_event(1, 0), mk_event(2, 1), mk_event(3, 1), mk_event(4, 2), mk_event(5, 3)]  # evt2 and evt3 share a timestamp
    for e in evs:
        store.emit(e)
    assert store.events() == evs
    assert store.events(limit=2) == evs[-2:]  # no cursor: the newest, oldest first
    assert store.events(after_id="evt1", limit=2) == evs[1:3]  # strictly after the cursor, up to limit
    assert store.events(after_id="evt2") == evs[2:]  # evt3 has evt2's timestamp and still comes after it
    assert store.events(after_id="evt5") == []
    assert store.events(after_id="gone", limit=2) == evs[:2]  # a cursor nobody knows pages from the start


# ---------------------------------------------------------------- what only the schema decides


def test_unknown_site_gets_the_default_policy(pg_store: PostgresStore):
    p = pg_store.policy("w.never")
    assert p == Policy(call_site="w.never", updated_at=p.updated_at)


def test_duplicate_ids_are_ignored_not_appended(pg_store: PostgresStore):
    pg_store.upsert_sites([mk_site()])
    first, second = mk_row(1), mk_row(1).model_copy(update={"response": "Travel"})
    pg_store.write_rows([first])
    pg_store.write_rows([first, second])  # a retried batch
    pg_store.emit(mk_event(1, 0))
    pg_store.emit(mk_event(1, 0))
    assert pg_store.rows("w.a") == [first] and len(pg_store.events()) == 1


def test_a_batch_with_an_undeclared_site_writes_nothing(pg_store: PostgresStore):
    pg_store.upsert_sites([mk_site()])
    with pytest.raises(psycopg.errors.ForeignKeyViolation):  # MemoryStore does not check; the schema does
        pg_store.write_rows([mk_row(1), mk_row(2, site_id="w.undeclared")])
    assert pg_store.rows("w.a") == []


def test_weights_meta_is_stored(pg_store: PostgresStore):
    pg_store.put_weights("sha1", b"x", {"call_site": "w.a"})
    with psycopg.connect(PG) as conn:
        assert conn.execute("select meta from weights where sha = 'sha1'").fetchone() == ({"call_site": "w.a"},)


# ---------------------------------------------------------------- the whole loop, on Postgres


def test_loop_reaches_dreaming_on_postgres(pg_store: PostgresStore):
    rt = build(pg_store)  # pyright: ignore[reportArgumentType]  # annotated MemoryStore, needs only the Store protocol
    rt.fast_forward("books", transactions(900), rt.workflow)  # type: ignore[attr-defined]
    cat, ven = rt.policies["books.categorize"], rt.policies["books.vendor_lookup"]
    events = pg_store.events(limit=10_000)
    assert cat.phase == Phase.dreaming and ven.phase == Phase.dreaming, [e.message for e in events]
    assert cat.cert_id and pg_store.certificate(cat.cert_id).bound <= 0.05
    kinds = [e.kind for e in events]  # a store error is swallowed at the boundary and shows up here, not as an exception
    assert kinds.count(EventKind.cert_issued) >= 2 and EventKind.train_finished in kinds and EventKind.component_error not in kinds
    assert pg_store.policy("books.categorize").phase == Phase.dreaming  # the phase change was written, not only held by the runtime
    assert len(pg_store.rows("books.categorize")) == 900 and len(pg_store.runs(limit=10_000)) == 900

    # a predicted decision re-executes byte for byte from rows that made the trip through JSONB
    run = next(r for r in reversed(pg_store.runs()) if r.replayable)
    rep = replay(pg_store, run.id, rt.predictors)
    assert rep.re_executed == run.replayable and rep.same == rep.re_executed
