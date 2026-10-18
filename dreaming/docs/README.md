# docs

The engineering docs for the hackathon build. The system design is `../DESIGN.md`; these files say what it is for.

| File | Read it when |
|---|---|
| `tracks.md` | deciding what to build for a sponsor, what evidence to keep, and what sentence the Devpost section can say |
| `training-plan.md` | building or running the campaign: candidates, data, the matrix, the monitor, the pruning rules, the numbers we defend, the night in order |
| `speculation.md` | asked whether nodes can run in parallel on predicted values, and what that buys (7–14%; not tonight) |
| `devtools.md` | sitting down to use a sponsor's tool: install, auth, the smallest real use, what to capture, when to quit it |

## Decided (2026-10-17, Saturday evening)

- Every candidate predictor family is a `Candidate(fit, load)` in `predictors/candidates.py`; the learner will train every
  candidate a site lists, certify each at δ/N, promote the best coverage and keep the rest as spares (`../DESIGN.md` §5; the
  parallel loop itself is build step 1, today one candidate per site runs end to end).
- The pinned teacher of the demo workflow is StepFun `step-5-preview` (`TEACHER=stepfun`) once its key exists; the router
  site's cheap configuration is the `step-3.5-flash` tier. OpenRouter, ASI:One and Gemini are rows of the same provider table.
- The benchmark workflow is QuickBooks over Intuit's data: one pooled student per site, certified separately on each held-out
  company; `invoice_paid` is the headline site, `bill_account` second, `bank_match` only with disclosed injected mess.
- The campaign is a deployed Modal app (`dreaming-campaign`): data per site on the Volume, one spawned call per cell, rungs
  60 / 200 / 500 / 2,000 inside a cell, raw scores stored per checkpoint, a cron monitor every 10 minutes that prints and prunes.
  Tonight's matrix is tier 1 (`head`, `memory`, `exact`, `finetune`); `online` and `hdc` join only if ported by 01:00.
- Speculation is lazy futures in the runtime, lossless, 7–14% on residual latency; built after the hackathon.

## Sign-in status (2026-10-17 21:25 PDT)

Done, keys in the root `.env`: `GEMINIKEY` (AI Studio, key `dreaming-lahacks-7d` on the existing paid project MPCO; no expiry
option, delete it in AI Studio after Oct 24), `CREAOKEY` (the account already has Pro Plus; key `dreaming-lahacks`, no expiry
option, revoke under Settings → API keys after Oct 24). ASI:One's free plan resets its request quota daily at 00:00 UTC
(1,000/day) and showed 0 used, so the chat demo runs again without the Pro code. Agentverse is on the Basic plan with ample limits.

Waiting on a human click, each one unblocks the next automated step:
- StepFun (21:50 PDT): signed in by email code after the slider was solved by hand; the $25 voucher is redeemed (valid to
  2027-01-15); an API key named `dreaming-lahacks` exists on the API Keys page. The auto-mode classifier refuses to copy the
  key's value out of the page ("credential materialization"), so: open platform.stepfun.ai/interface-key, click the copy icon
  on the `dreaming-lahacks` row, and add `STEPFUNKEY=<value>` to the root `.env`.
- Google asks for a phone confirmation before any "Continue with Google" sign-up: tap Yes, then the number shown, on the iPhone
  Gmail prompt.
- Sentry: done. Org `dreaming`, project `dreaming-api` (FastAPI platform, error monitoring on); `SENTRY_DSN` is in `.env` and
  verified: one `capture_message` from the laptop became issue DREAMING-API-1 in the feed. Add Tracing/Logs/Profiling in the
  SDK init, not the project settings. Rotate the DSN at Settings → Projects → dreaming-api → Client Keys after Oct 24.
- veri.studio: done. Personal workspace, key `dreaming-lahacks` in `.env` as `VERIKEY` (shown once; revoke it at Settings → API
  Keys after Oct 24). Credits: $20.00 (code redeemed 2026-10-17); a $0 balance would kill a job after boot, so watch it.
- Modal: done. A card is on file; a T4 function ran on 2026-10-17 22:20 PDT (`Tesla T4, 15360 MiB`), so GPU cells are unlocked.
  OpenRouter: done, about $20 of prepaid credit after the top-up and $30 left on the key limit. veri credits: done, the code from
  @daniung redeemed for $20.00 at Settings → Billing. ASI:One Pro: a checkout with a card (optional now).
- Devin: done. Account exists and a personal API key is in `.env` as `DEVINKEY` (verified: `GET /v1/sessions` and `/v1/secrets`
  return 200). A personal key attributes sessions to the account holder, which is what the evidence line wants; a service-user key is only
  for unattended automation. Still to do in the UI: Settings → Connections → GitHub → this repo; the credits form
  (#cognition on Discord) needs a team name and two member emails. Revoke the key at Settings → API keys after Oct 24.
- Browser Use: done. Email verified; key `dreaming-lahacks` in `.env` as `BROWSERUSEKEY`, created with the platform's own
  **7-day expiration (expires Oct 24, 2026)**, the only provider tonight that offers one. Tier 0 ($0 credits, 10 concurrent):
  the rep's $250 credit is still unverified, so the first run may need a purchase. Keys live at cloud.browser-use.com/api-keys.

## The one sign-in list, by what it unblocks (other docs point here)

1. **Modal payment method** (the matrix; GPU needs it; only $1 of the $30 credit is unlocked without it). Set a workspace budget.
2. **StepFun** voucher + key (the teacher, the record pass). **OpenRouter top-up, ~$10** (the Anthropic row, the Jev baseline, the
   LLM baseline, the fallback teacher). **ASI:One Pro code `LAHACKSAI26`** (the key is at its request quota; the chat demo and the
   fallback teacher need it).
3. **veri.studio**: credit code from @daniung on the LA Hacks Discord, redeemed at Settings → Billing before the first job; `veri login`.
4. **CREAO** account through the hackathon link + per-agent `capi_` key + Secrets (`DREAMING_API_URL`, `DREAMING_API_KEY`).
   **Devin**: GitHub app on this repo + the #cognition credits form. **Sentry**: account + DSN (`SENTRY_DSN`).
5. **Gemini** key; **Paper** code `LAHACKS26`; **GoDaddy** domain at tech.study; **Snowflake** trial; **Browser Use** key;
   **Tiger** signup; a funded **Solana** devnet wallet (faucet, by hand).
6. **Devpost** project with every teammate (the email they applied with) and the CREAO track selected; the **Fetch Submission
   Agent** registration; the two Innovation Lab badges in the README.

## Open

- StepFun's self-disagreement and latency at low effort; whether `max_tokens=256` leaves a parsable label every time (10-call check first).
- Whether a veri `run-script` job's artifacts download (save in Hugging Face layout; test on a tiny job).
- The δ/N selection cost (reasoned, unmeasured; a query over the campaign's stored scores afterwards).
- What Intuit means by a "System-One-compatible API" (we serve TypeSafe's `/v1/systemone` wire format; confirm with the mentor).
- How much mess to inject into the QuickBooks feed for `bank_match`, and the disclosure text.
