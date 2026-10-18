"""Apply every db/migrations/*.sql, in order, to the Supabase database from a Modal container, then report what is there.

Why from Modal: many laptop networks block outbound 5432/6543 (this one does); Modal containers reach the pooler, and the
trainer will run there anyway. Also the first live Modal + Supabase check in one go.

    modal secret create dreaming-db DATABASE_URL=...     (once; the trainer app reads the same secret)
    modal run smokes/modal_db.py                         apply the schema (idempotent) and report
    modal run smokes/modal_db.py --reset                 also empty every table first: a clean slate before a demo
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import modal

app = modal.App("dreaming-db-smoke")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("psycopg[binary]==3.3.6")
MIGRATIONS = sorted((Path(__file__).resolve().parents[1] / "db" / "migrations").glob("*.sql"))  # 0001_init, 0002_campaign, ...: each idempotent


@app.function(image=image, secrets=[modal.Secret.from_name("dreaming-db", required_keys=["DATABASE_URL"])], timeout=120)
def migrate(sql: str, reset: bool = False) -> dict[str, Any]:
    import os
    import time

    import psycopg

    t0 = time.perf_counter()
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True, connect_timeout=20) as c:
        connect_ms = round((time.perf_counter() - t0) * 1000)
        for one in sql.split("\n-- MIGRATION --\n"):  # each file is its own transaction block; idempotent, safe to run twice
            c.execute(one)
        if reset:
            c.execute("truncate trace_rows, events, runs, certificates, weights, policies, call_sites, memory_rows, checkpoints, train_runs, eval_sets")
        tables = [r[0] for r in c.execute("select tablename from pg_tables where schemaname = 'public' order by 1")]
        realtime = [r[0] for r in c.execute("select tablename from pg_publication_tables where pubname = 'supabase_realtime' order by 1")]
        anon = [r[0] for r in c.execute("select table_name from information_schema.role_table_grants where grantee = 'anon' and privilege_type = 'SELECT' order by 1")]
        version = c.execute("select version()").fetchone()
    return {"connect_ms": connect_ms, "tables": tables, "realtime": realtime, "anon_select": anon, "version": version and version[0][:40]}


@app.local_entrypoint()
def main(reset: bool = False) -> None:
    print([p.name for p in MIGRATIONS])
    print(migrate.remote("\n-- MIGRATION --\n".join(p.read_text(encoding="utf-8") for p in MIGRATIONS), reset))
