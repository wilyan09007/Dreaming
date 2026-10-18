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
1. Windowed run on the laptop (11:05, 1 display): done.
   - 8/8 windows placed, in front, maximized.
   - Closing the windows stops the API and the launcher, and removes the temp profile.
   - Closing them while the API is still starting exits cleanly.
   - Solo mode is checked: arrows are ignored, and focus replays the module.
   - Still untested: the Alt tap + `SetForegroundWindow` focus (merged from the cloud session at 11:15), and a two-display placement.
2. API start no longer uses `--all-packages`.
   - It runs `uv sync --inexact --package dreaming-api --package bookkeeper --package quickbooks`, the server image's set, then `uv run --no-sync`. So torch and CUDA are never installed.
   - Fresh clone on the laptop: `--check` 19/19 (now including `/state`) in 59 s, with 0 torch, sentence-transformers or nvidia packages in the venv.
   - The API wait is 300 s, and `--check` exits 1 when the API is down.
3. **Not deployed.** Fly (`dreaming-demo`) does not serve `/launch/` until a deploy. Ask the user first: a deploy restarts the live app.
4. **Secrets.** Deploys (Fly), Modal, StepFun and Supabase need the `.env` names below in the cloud environment's variables. Names only; the user must add the values:
   - Deploy and run: FLY_API_TOKEN, STEPFUNKEY, DATABASE_URL, SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY, MODAL_TOKEN_ID, MODAL_TOKEN_SECRET, AGENTVERSEKEY, AGENT_SEED, ASI1KEY, OPENROUTERKEY.
   - Optional: SENTRY_DSN, LOGFIRE_TOKEN, RUNPODKEY, VERIKEY, CREAOKEY, DEVINKEY, GEMINIKEY, BROWSERUSEKEY.

## How to run the demo (on the laptop)
`git pull && python launch_demo.py`. A subset: `python launch_demo.py race bench`. The deployed live app: `--url https://dreaming-demo.fly.dev`.

## Cloud session (2026-10-18)
- First `--check` in a fresh container: the API missed its 90 s window because `uv` was downloading torch and the CUDA wheels (about 3 GB). `--check` still said OK. Fixed: `--check` now exits 1 when the API doesn't come up, and the API log is copied to `<tmp>/dreaming-demo-api.log` (it was deleted on exit). After `uv sync --all-packages`: 18/18 pages, API up in 6.4 s, exit 0.
- Deck QA in headless Chromium at 1600x900 and 1280x720: all 7 `?only=` modules load with no console errors and no failed requests. Arrow keys walk hook → close. Build's Run it settles at 11.52 s / 138 ms and $3.38 / $0.67, the same as Race.
- This container has no secrets and no `flyctl`, so a deploy (item 3) has to run from the laptop, or from here once the env vars are added.
