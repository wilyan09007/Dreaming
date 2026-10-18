# infra

| File | What | Deploy |
|---|---|---|
| `fly.toml` + `serve.Dockerfile` | API + uAgent, one Fly app, two machines | `fly apps create dreaming-demo && fly secrets set --stage … && fly deploy -c infra/fly.toml --ha=false -y` |
| `../apps/trainer/src/dreaming_trainer/modal_app.py` | the trainer's image is defined in code | `modal secret create dreaming-db DATABASE_URL=… && modal deploy apps/trainer/src/dreaming_trainer/modal_app.py` |
| `../db/migrations/*.sql` | Supabase schema, RLS, Realtime publication | `modal run smokes/modal_db.py` (applies it from a Modal container: laptop networks often block 5432/6543; this one does), or `psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f db/migrations/0001_init.sql` where the port is open |
| `supabase-keepalive.yml` | GitHub Actions cron: one REST ping every 3 days so the free project never pauses | copy to `.github/workflows/` |

Sign-ins a human does once: Supabase (project, pooler URL, publishable key), Modal (token; the Starter credit is $1 until a
payment method is added), Fly (account with card, an org token), Logfire (optional write token). The agent registers its own
Agentverse mailbox at startup from `AGENTVERSEKEY`. Every token was created with a 7-day lifetime on 2026-10-17; recreate them
after that. Keys live in the hosts' secret stores and in `../.env` locally (the full list with links: `../../.env.example`),
never in the repo. Numbers and gotchas: `SERVICES.md`; what ran live: `../smokes/RESULTS.md`.
