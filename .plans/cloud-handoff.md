# Cloud hand-off (2026-10-18 10:40 PDT)

Status: IN_PROGRESS. Moved from the laptop session so the team can work on the move.

## State
- `main` holds everything. The judges' deck is in `dreaming/apps/web/launch/`:
  - `deck.js` (stage, keys, presenter sync, `?only=<id>` solo mode)
  - `numbers.js` (the only source of figures)
  - `screens/*.js`: hook, race, bench, psi, train, build, close
  - Spec: `launch/VISUAL_INTENT.md`
- `launch_demo.py` at the repo root (stdlib only) does three things:
  - Serves the modules from a static thread.
  - Starts the offline API via `uv` (DATABASE_URL empty, rule teacher, char encoder, local trainer).
  - Opens one chromeless Chrome/Edge `--app` window per module plus the live app, and places them with Win32: modules maximized on the external display, the live app on the primary.
  - Closing every window, or Ctrl+C, stops the backends.
- `python launch_demo.py --check` runs headless: it fetches every page and returns exit 0 or 1. Use it in the cloud, where there is no display.
- Verified on the laptop (1 display):
  - `--check` 18/18 pages; the API is up in 3.9 s.
  - Windowed run: 8/8 windows placed.
  - Closing the browser stops the API.

## Open items
1. **Untested in the windowed run:** after the last two fixes (browser output to DEVNULL; module windows open before the API is up), only `--check` was rerun.
2. **Windows may open behind other windows.** Windows blocks focus stealing, so the module windows land behind the user's terminal. A SetForegroundWindow/AllowSetForegroundWindow pass, or the `--start-maximized` / `--kiosk` flags, would fix it.
3. **Not deployed.** Fly (`dreaming-demo`) does not serve `/launch/` until a deploy. Ask the user first: a deploy restarts the live app.
4. **Secrets.** Deploys (Fly), Modal, StepFun and Supabase need the `.env` names below in the cloud environment's variables. Names only; the user must add the values:
   - Deploy and run: FLY_API_TOKEN, STEPFUNKEY, DATABASE_URL, SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY, MODAL_TOKEN_ID, MODAL_TOKEN_SECRET, AGENTVERSEKEY, AGENT_SEED, ASI1KEY, OPENROUTERKEY.
   - Optional: SENTRY_DSN, LOGFIRE_TOKEN, RUNPODKEY, VERIKEY, CREAOKEY, DEVINKEY, GEMINIKEY, BROWSERUSEKEY.

## How to run the demo (on the laptop)
`git pull && python launch_demo.py`. A subset: `python launch_demo.py race bench`. The deployed live app: `--url https://dreaming-demo.fly.dev`.
