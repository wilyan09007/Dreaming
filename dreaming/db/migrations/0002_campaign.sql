-- Candidates, shadows, speculation flags, and the offline campaign (DESIGN.md §5, docs/training-plan.md §3). Idempotent.
-- Apply after 0001:  psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f db/migrations/0002_campaign.sql
begin;

-- the promoted candidate, and every other candidate's prediction / state / certificate
alter table trace_rows   add column if not exists shadow     jsonb   not null default '{}'::jsonb;  -- {candidate: contracts.Prediction}
alter table trace_rows   add column if not exists speculated boolean not null default false;
alter table trace_rows   add column if not exists kept       boolean;
alter table certificates add column if not exists candidate  text    not null default '';
alter table policies     add column if not exists candidate  text    not null default '';
alter table policies     add column if not exists shadow     jsonb   not null default '{}'::jsonb;  -- {candidate: contracts.Spare}
alter table runs         add column if not exists critical_ms integer not null default 0;

-- the memory candidate's rows: the embedding must be durable and queryable (docs/training-plan.md §2.1)
create table if not exists memory_rows (
  call_site   text not null references call_sites(id),
  row_id      text not null,                              -- trace_rows.id, or "seed:<label>:<i>" for LLM-written rows
  label       text not null,
  source      text not null,                              -- seed | shadow | audit | correction | self
  encoder_sha text not null,                              -- rows from another encoder are invalid for this candidate
  embedding   real[] not null,                            -- unit norm, 384 floats; pgvector `vector(384)` when the extension exists
  ts          timestamptz not null default now(),
  primary key (call_site, row_id)
);

-- the campaign: one row per cell, one row per checkpoint, frozen eval slices by sha
create table if not exists train_runs (                   -- contracts.TrainRun
  id          text primary key,
  campaign    text not null,
  call_site   text not null,
  candidate   text not null,
  variant     text not null,                              -- seed+real | real | seed
  seed        integer not null,
  config      jsonb not null default '{}'::jsonb,
  "where"     text not null,                              -- modal:cpu | modal:T4 | veri | local:cpu
  status      text not null default 'queued',             -- queued | running | done | failed | pruned
  started_at  timestamptz,
  finished_at timestamptz,
  seconds     double precision not null default 0,
  cost_usd    double precision not null default 0,
  note        text
);
create index if not exists train_runs_campaign_idx on train_runs (campaign, call_site, candidate);

create table if not exists checkpoints (                  -- contracts.Checkpoint
  run_id      text not null references train_runs(id),
  rung        integer not null,                           -- rows seen: 60 | 200 | 500 | 2000
  weights_sha text not null,
  eval_sha    text not null,
  scores_uri  text not null,                              -- modal://dreaming-weights/campaign/<run>/<rung>.npz
  seconds     double precision not null,
  embed_ms    double precision not null,
  head_ms     double precision not null,
  bytes       integer not null,
  stable      boolean not null,                           -- second forward pass byte-equal: the hard-fail check
  summary     jsonb not null default '{}'::jsonb,         -- acc, coverage_02/05/10/20, bound, emp_error, ece, n_eval
  ts          timestamptz not null,
  primary key (run_id, rung)
);

create table if not exists eval_sets (                    -- contracts.EvalSet
  sha         text primary key,
  call_site   text not null,
  split       text not null,                              -- calib | test | forward
  row_ids     jsonb not null,
  labels      jsonb not null,
  groups      jsonb not null
);

-- the browser never reads these three; nothing to grant. The monitor's table arrives as a campaign.progress event.
do $$
begin
  if exists (select 1 from pg_roles where rolname = 'anon') then
    alter table memory_rows enable row level security;  revoke all on memory_rows from anon, authenticated;
    alter table train_runs  enable row level security;  revoke all on train_runs  from anon, authenticated;
    alter table checkpoints enable row level security;  revoke all on checkpoints from anon, authenticated;
    alter table eval_sets   enable row level security;  revoke all on eval_sets   from anon, authenticated;
  end if;
end $$;

commit;
