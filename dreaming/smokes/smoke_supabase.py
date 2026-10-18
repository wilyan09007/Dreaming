"""Supabase smoke: IPv4 session pooler, the seven tables, one Event round trip, the Realtime publication, anon grants.

Needs DATABASE_URL (Dashboard > Connect > Session pooler, port 5432). One line per check, exit 1 on failure; the smoke event is always deleted.
"""

from __future__ import annotations

import os
import socket
import sys
import uuid
from datetime import UTC, datetime
from urllib.parse import urlparse

import psycopg
from dreaming.contracts import Component, Event, EventKind
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

TABLES = {"call_sites", "policies", "runs", "trace_rows", "certificates", "weights", "events"}
LIVE = {"events", "policies", "certificates"}  # not runs (volume) and never trace_rows
failed: list[str] = []


def check(name: str, ok: bool, detail: object = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}  {detail}".rstrip())
    if not ok:
        failed.append(name)


def main() -> int:
    u = urlparse(os.environ.get("DATABASE_URL", ""))
    host, port = u.hostname or "", u.port or 5432
    try:
        ips = sorted({str(a[4][0]) for a in socket.getaddrinfo(host, port, socket.AF_INET)})
    except OSError:
        ips = []
    check("session pooler over IPv4", host.endswith(".pooler.supabase.com") and port == 5432 and bool(ips), f"{host}:{port} -> {', '.join(ips) or 'no A record'}")
    try:
        conn = psycopg.connect(os.environ.get("DATABASE_URL", ""), autocommit=True, connect_timeout=10)
    except psycopg.Error as e:
        check("connect", False, " ".join(str(e).split())[:200])
        return 1
    with conn:
        have = {r[0] for r in conn.execute("select table_name from information_schema.tables where table_schema = 'public'")}
        check("seven tables exist", TABLES <= have, f"missing {sorted(TABLES - have)}" if TABLES - have else "")
        ev = Event(id=f"smoke_{uuid.uuid4().hex[:12]}", ts=datetime.now(UTC), component=Component.web, kind=EventKind.batch_started, message="smoke")
        try:
            conn.execute("insert into events (id, ts, component, kind, message, data) values (%s, %s, %s, %s, %s, %s)",
                         (ev.id, ev.ts, ev.component.value, ev.kind.value, ev.message, Jsonb(ev.data)))
            row = conn.cursor(row_factory=dict_row).execute("select * from events where id = %s", (ev.id,)).fetchone()
            check("event insert, read back as contracts.Event", row is not None and Event.model_validate(row) == ev, ev.id)
        except (psycopg.Error, ValueError) as e:
            check("event insert, read back as contracts.Event", False, " ".join(str(e).split())[:200])
        finally:
            check("event deleted", conn.execute("delete from events where id = %s", (ev.id,)).rowcount == 1)
        pub = {r[0] for r in conn.execute("select tablename from pg_publication_tables where pubname = 'supabase_realtime' and schemaname = 'public'")}
        check("supabase_realtime has events, policies, certificates; not runs, not trace_rows", LIVE <= pub and "trace_rows" not in pub and "runs" not in pub, sorted(pub))
        grants: dict[str, set[str]] = {}
        for t, priv in conn.execute("select table_name, privilege_type from information_schema.role_table_grants where grantee = 'anon' and table_schema = 'public'"):
            grants.setdefault(t, set()).add(priv)
        check("anon can SELECT policies", "SELECT" in grants.get("policies", set()))
        check("anon has no privilege on trace_rows or weights", not (grants.get("trace_rows") or grants.get("weights")))
        check("anon is read-only everywhere", all(p == {"SELECT"} for p in grants.values()), {t: sorted(p) for t, p in grants.items()})
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
