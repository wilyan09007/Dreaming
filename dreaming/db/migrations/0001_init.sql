-- Schema of record. Mirrors packages/dreaming/src/dreaming/contracts.py one to one (a column per field, JSON for nested
-- objects). Apply with:  psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f db/migrations/0001_init.sql   (idempotent)
--
-- Who reads what:
--   Python (api, agent, trainer)  connect as the owner through the Supabase session pooler: RLS does not apply to them
--   the browser (dashboard)       Supabase Realtime + REST with the publishable key, role `anon`: read-only, and only
--                                 the four small tables below. Trace rows and weights never leave the server side.
begin;

create table if not exists call_sites (
  id          text primary key,            -- "<workflow>.<name>"
  spec        jsonb not null,              -- contracts.CallSite
  updated_at  timestamptz not null default now()
);

create table if not exists policies (                     -- contracts.Policy: one row per call site, the router's input
  call_site   text primary key references call_sites(id),
  phase       text not null default 'awake',
  cert_id     text,
  cutoff      double precision,
  weights_sha text,
  rows        integer not null default 0,
  shadow_rows integer not null default 0,
  predicted_rows integer not null default 0,
  audit_rows  integer not null default 0,
  audit_errors integer not null default 0,
  attempt     jsonb,                                   -- contracts.Attempt while testing fails
  outage      boolean not null default false,
  updated_at  timestamptz not null default now()
);

create table if not exists runs (                         -- contracts.Run
  id          text primary key,
  n           integer not null default 0,                 -- ordinal within the workflow
  workflow    text not null,
  mode        text not null,                              -- live | fast_forward | replay
  input       jsonb,
  started_at  timestamptz not null,
  finished_at timestamptz,
  calls       integer not null default 0,
  decisions   integer not null default 0,
  cost_usd    double precision not null default 0,
  latency_ms  integer not null default 0,
  failed      boolean not null default false,
  stuck_at    text,
  replayable  integer not null default 0,
  down        jsonb not null default '[]'::jsonb           -- call sites unplugged during the run
);
create index if not exists runs_started_idx on runs (started_at desc);

create table if not exists trace_rows (                   -- contracts.TraceRow: one row per call
  id          text primary key,
  run_id      text not null,
  i           integer not null,
  call_site   text not null references call_sites(id),
  kind        text not null,
  phase       text not null default 'awake',           -- the site's phase at call time; calibration rows are watching/testing
  request     jsonb,
  request_sha text not null,
  response    jsonb,
  decision    text not null,                              -- called | predicted | audited | fallback | replayed
  prediction  jsonb,                                      -- contracts.Prediction (hashes inside)
  teacher     jsonb,                                      -- contracts.TeacherAnswer
  agree       boolean,
  latency_ms  integer not null,
  cost_usd    double precision not null default 0,
  error       text,
  ts          timestamptz not null
);
create index if not exists trace_rows_site_ts_idx on trace_rows (call_site, ts);
create index if not exists trace_rows_run_idx on trace_rows (run_id, i);
create index if not exists trace_rows_shadow_idx on trace_rows (call_site, ((prediction->'hashes'->>'weights'))) where agree is not null and phase in ('watching', 'testing');

create table if not exists certificates (                 -- contracts.Certificate
  id          text primary key,
  call_site   text not null references call_sites(id),
  weights_sha text not null,
  env_sha     text not null,
  teacher_sha text not null,
  budget      double precision not null,
  delta       double precision not null,
  cutoff      double precision not null,
  coverage    double precision not null,
  emp_error   double precision not null,
  bound       double precision not null,
  n_calib     integer not null,
  n_above     integer not null,
  n_errors    integer not null,
  calib_sha   text not null,
  created_at  timestamptz not null,
  status      text not null default 'active'              -- active | superseded | revoked
);

create table if not exists weights (                      -- by hash; bytea for heads (< 5 MB), a Modal Volume path for encoders
  sha         text primary key,
  data        bytea,
  uri         text,                                       -- e.g. "modal://dreaming-weights/<sha>.safetensors" when data is null
  meta        jsonb not null default '{}'::jsonb,
  created_at  timestamptz not null default now()
);

create table if not exists events (                       -- contracts.Event: the live feed and the audit trail
  id          text primary key,
  ts          timestamptz not null,
  component   text not null,                              -- lib | api | agent | trainer | web
  kind        text not null,                              -- contracts.EventKind
  message     text not null,
  run_id      text,
  call_site   text,
  data        jsonb not null default '{}'::jsonb,
  error       text,
  trace_id    text
);
create index if not exists events_ts_idx on events (ts desc);

-- Browser access: read-only on the four tables the dashboard watches, through RLS; nothing on trace rows or weights.
-- Supabase stopped auto-granting privileges to anon on new tables (2026), so the GRANTs are explicit.
do $$
declare t text;
begin
  if exists (select 1 from pg_roles where rolname = 'anon') then
    foreach t in array array['call_sites', 'policies', 'runs', 'certificates', 'events'] loop
      execute format('alter table %I enable row level security', t);
      execute format('revoke all on %I from anon, authenticated', t);
      execute format('grant select on %I to anon, authenticated', t);
      execute format('drop policy if exists "dashboard read" on %I', t);
      execute format('create policy "dashboard read" on %I for select to anon, authenticated using (true)', t);
    end loop;
    foreach t in array array['trace_rows', 'weights'] loop
      execute format('alter table %I enable row level security', t);
      execute format('revoke all on %I from anon, authenticated', t);
    end loop;
  end if;
  -- Realtime: the dashboard subscribes to changes on these. Not runs (100+ inserts/s in fast-forward would hit the
  -- 100 msg/s free cap; the page refetches /state on batch.progress) and not trace_rows.
  if exists (select 1 from pg_publication where pubname = 'supabase_realtime') then
    foreach t in array array['policies', 'certificates', 'events'] loop
      if not exists (select 1 from pg_publication_tables where pubname = 'supabase_realtime' and tablename = t) then
        execute format('alter publication supabase_realtime add table %I', t);
      end if;
    end loop;
  end if;
end $$;

commit;
