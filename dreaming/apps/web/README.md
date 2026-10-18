# apps/web: the dashboard's contract

The page is `demo/index.html` (looks, tabs and copy stay as in `demo/VISUAL_INTENT.md`) with its simulated model (`SITES`, `S`, `phase()`,
`cover()`, `metrics()`, `sample()`) replaced by the API's data: `apps/web/index.html` (the demo's markup and CSS) and `app.js` (its script, rewired),
served by the API's `StaticFiles` mount at `/` (same origin: no CORS, `apiBase = ''`). The page computes nothing; every number is a field below
(the one ratio is a site's "−N% cost" note, `1 − cost_dream_usd / cost_awake_usd`). `demo/index.html` stays the simulation the design was drawn from.

Under each dream-journal row, small Gaegu lines name the promoted candidate with its certified coverage and bound (`Policy.candidate`,
`Certificate`), then each spare (`Policy.shadow`, its certificate from `spare_certificates` when it certified at the budget in force).

## (a) On load
1. `GET /healthz` -> `Health {ok, checks{db, trainer, teacher, realtime}, version, env_sha}`. Show a strip only for a red check.
2. `GET /state` -> `State` (defined in `apps/api/.../main.py`, not in `contracts.py`):
   `sites: SiteView[]` (`{site: CallSite, policy: Policy, certificate: Certificate | null}`), `budget` (fraction 0.01..0.10; slider = budget x 100),
   `metrics: Metrics` (now), `series: Metrics[]` (`batch.progress` snapshots, oldest first), `runs: Run[]` (last 50, newest last), `recorded: boolean`.
3. `subscribe(...)` from `live.js`. Site ids: `bookkeeper.categorize | vendor_lookup | match_invoice | anything_odd` (kinds `model | tool | model | router`)
   and the effects `post_to_books`, `wake_a_human` (`kind: effect`: the two non-clickable nodes). "bank feed" is a static node.

## (b) Live updates (`live.js`)
```js
import { subscribe } from './live.js';
const stop = subscribe(supabaseUrl, publishableKey, '', (m) => apply(m)); // m.table: events | policies | certificates | resync
```
- Supabase Realtime `postgres_changes` with the **publishable key** (`sb_publishable_...`, role `anon`, SELECT only, never on `trace_rows` or `weights`):
  `INSERT` on `events`; `INSERT` and `UPDATE` on `policies`, `runs`, `certificates`. `new` is the raw row: its columns are the contract's field names
  (the SQL mirrors `contracts.py`), jsonb arrives as objects, timestamps as ISO strings. Ignore `DELETE`. `old` holds only the primary key (RLS).
- `supabase-js` is pinned (`@2.117.1`, never `@2` or `@next`); the channel sets `postgres_changes_options: {wait: true}`, so `SUBSCRIBED` means ready.
- **Backfill after every `SUBSCRIBED`**: `GET /events?limit=200` (oldest first). Postgres Changes has no replay, so the gap is re-read, and
  **dedupe by `Event.id`** (a `Set`): backfill, poll and live insert overlap. Ids are `evt_<12 hex>`, random: never sort or compare them, keep arrival order.
  Each `SUBSCRIBED` also emits `resync`: re-`GET /state`, because `policies` and `certificates` rows are not in `/events` (and `runs` come from `/state`).
- **Polling fallback**: on `CHANNEL_ERROR | TIMED_OUT | CLOSED`, `GET /events?after=<last id>&limit=200` every 2 s until `SUBSCRIBED`; every non-empty poll emits `resync`.
  The API must resolve `after` by position (`ts`, then `id`), not by comparing ids.
- Load: free Realtime is 100 msg/s, each row change counted once per viewer. `runs` is therefore not in the publication (2,000 runs in 18 s would be 110 inserts/s); the runtime batches run writes and the page refetches `/state` on every `batch.progress` event (one per 20 runs).

## (c) Commands (all JSON; the page sends only these four)
| Button | Request | Reply |
|---|---|---|
| run 2,000 | `POST /batch {"n": 2000}` | `{"status": "started"}`; 409 while one runs, 503 if the API has no workflow inputs |
| mistakes allowed (on release) | `POST /budget {"budget": 0.07}` | `{"budget": 0.07}`; per-site `cert.issued` / `cert.revoked` follow within ms |
| unplug the API | `POST /tools/bookkeeper.vendor_lookup/outage {"down": true}` | the updated `Policy`; 404 for an unknown site |
| replay this run | `POST /replay/{run_id}` | `ReplayReport {steps[{i, call_site, decision, re_executed, same_bytes, output_sha}], re_executed, same, from_log}`; 404 unknown run, 409 when the run's weights or env are not loaded in the API |

`GET /runs/{run_id}` -> `{run: Run, rows: TraceRow[]}` fills the replay tab's trace card (rows carry `request`, `response`, `decision`, `prediction`, `teacher`; `evidence_texts` maps each memory voter's row id to its rendered input).

## (d) What replaces what in `demo/index.html`
| Demo (simulated) | Contract (real) |
|---|---|
| `SITES[i]`: `id`, `name`, `tool`, `router`, `watchAt`, `teacher` | `State.sites[i].site`: `id`, `kind` (`model`, `tool`, `router`, `effect`), `thresholds.watch_after`, `teacher.provider`. Display names ("vendor lookup", "anything odd?") are a page map keyed by `site.name` |
| `S.seen[id]` | `Policy.rows` |
| `phase(s).p` | `Policy.phase`; `down` comes from `Policy.outage` (g) |
| `ph.counted`, `CHECK` = 500, the sheep | `min(Policy.shadow_rows, thresholds.calibrate_after)`; 1 sheep = `calibrate_after / 10` |
| `ph.cant` ("no promise at 5%") | `phase === 'testing' && !cert_id && shadow_rows >= calibrate_after` (the runtime leaves a site in testing when nothing certifies) |
| `cover(s, b)`, `ph.cov` ("dream answers 89%") | `Certificate.coverage` x 100 of `Policy.cert_id`. No cover tables, no interpolation |
| `measured(s, b)`, `s.bound` | `Certificate.emp_error`, `Certificate.bound` (fractions, x 100). `bound <= budget` is what "certified" means |
| `ph.b`, `S.budget` | "wrong <= b%": `Certificate.budget`; slider and "mistakes allowed": `State.budget` |
| `S.cert[id]` ("cert c-0412") | `Certificate.id` (`cert_` + 12 hex): print `id.slice(5, 11)`. Every re-certify issues a new id. Find the live promise by `Policy.cert_id`, never by `Certificate.status` |
| `s.weights` ("weights 9f3a1c") | `Policy.weights_sha.slice(0, 6)` (= `Certificate.weights_sha`, = `Prediction.hashes.weights` on a row) |
| `AUDIT` ("1 in 20"); `s.replay` | `1 / thresholds.audit_rate`; `site.replayable` (false for the router: "not replayable") |
| `S.down`, the plug pill | `Policy.outage` on `bookkeeper.vendor_lookup`. "circuit breaker: 100%" is page copy, not a measurement |
| `metrics()` `.cost .sec .fail .rep` | `State.metrics`: `cost_per_run` (USD, x 100 for cents), `sec_per_run`, `failed_pct`, `replayable_pct` |
| `S.series`, `record()` | `State.series`, then `data.metrics` of each live `batch.progress`. x = point index x 20 runs (`Metrics` has no x) |
| `S.saved` ("saved so far") | `Metrics.saved_usd` |
| `S.runs`, `S.target`, `S.running` | no cumulative counter (`Metrics.runs` is the total number of runs so far (the x of every series point); the other fields average the last 100 finished runs. `POST /batch/pause` and `/batch/resume` drive the run button.
| `sample(r)`, `S.samples` (receipts) | `State.runs`: `Run.input.descriptor` and `.amount`, `Run.replayable`, `Run.stuck_at`; the trace card reads `GET /runs/{id}` |
| `steps[].by` | `TraceRow.decision` + `kind`: `predicted` = dream; `called` or `audited` (both ran) = the real call: big model (model, router) or vendor API (tool); `fallback` = snapshot; effects = ledger / bookkeeper |
| "voted:" line under a dream's answer (no demo counterpart) | `TraceRow.prediction.evidence` (the memory candidate's top 3 rows: `row_id`, `label`, `weight` = share of the vote) with each row's text from `RunView.evidence_texts[row_id]` (first line, its `Field: ` prefix dropped); `seed:` ids read "seed example", an id with no text shows its label only; weights under 0.005 are left out, rows that read the same merge into "2× ..." with weights summed |
| `steps[].ans`, `.hash`, "same bytes" | `TraceRow.response`; `ReplayStep.output_sha.slice(0, 6)`, `same_bytes`, `re_executed`. The 0.5 s row stepping animates a reply already in hand |
| stuck run | `Run.stuck_at` (a site id); rows after it do not exist: "never ran" |
| narration, `S.note`, `S.pop` | event `message`s: `site.phase` (`data.to`), `cert.issued` / `cert.revoked` (verdict, hold 2 s), `train.*`, `tool.outage`, `budget.changed`. Pacing stays in the page |
| "re-checking..." ripple | `budget.changed`, then one `cert.issued` or `cert.revoked` per site; the stagger only spaces out results already returned |

## (e) Honesty labels
- `State.batch.recorded === true`: the batch narration says **"pre-recorded teacher answers"** ("fast-forwarding 2,000 transactions · pre-recorded teacher answers ·
  every call is logged"), as the server's `batch.started` message does. Never a bare "fast-forwarding". Seconds per run are virtual-clock time; say so in that sentence.
- `TeacherAnswer.recorded` marks each row: the trace card shows "recorded answer" on those rows. Live moments (ASI:One chat, audit calls) have `recorded: false`.
- No simulated constant survives (cover tables, 89%, 10.7%, 60%); a number with no field in (d) is not shown.

## (f) Types
`just schema` writes `schema.json` + `schema.d.ts` from `contracts.py` (`Event`, `Policy`, `Run`, `Certificate`, `Metrics`, `TraceRow`, `ReplayReport`, `Health`;
`CallSite` is emitted as `CallSite3`). Both are generated, not committed by hand. `index.html` and `live.js` start with `// @ts-check` and use JSDoc
(`/** @type {import('./schema').Policy} */`). Check, after `just schema`: `npx -p typescript tsc --noEmit --allowJs --checkJs --strict --target es2022 --module es2022 --moduleResolution bundler --lib es2022,dom apps/web/live.js`.
`State`, `SiteView` and `RunView` are not in the recipe's union yet; declare them as `@typedef`s from (a) until they are.
`app.js` passes the non-strict check `apps/api/tests/test_web.py` runs (`tsc --checkJs` on `app.js` and `live.js`); `live.js` also passes the strict one above.

## (g) The five stickers
`awake` sun, `watching` dashed eye, `testing` half-blue circle, `dreaming` sleeping cloud = `Policy.phase`, one to one. `down` (rain cloud) = `Policy.outage`
on a tool site; it wins over the stored phase, and `Phase.down` is never stored. `testing` lasts milliseconds server-side (one `tick()` goes watching, testing, certify);
the page holds each sticker for at least 0.9 s from the `site.phase` events, which spaces the display out and changes no value.

## (h) The scores tab
`GET /benchmark` -> `Benchmark {campaign, rows: Row[], notes}` (`apps/api/.../benchmark.py`, not in `contracts.py`: `app.js` declares `BRow`/`Bench` typedefs), fetched
each time the tab opens; the latest campaign only. One notebook card per `Row.call_site` (workflow order), one table row per `Row` in the server's order. The 5% cell
(`cells[budget == 0.05]`) gives repeat traffic (`forward_coverage`, `forward_error`) and unseen (`test_coverage`, `test_error`; the 2/10/20% coverages are its tooltip);
`acc_forward` / `acc_test` are the accuracy column; `without_dreaming` / `with_dreaming` give `$ per 1,000 calls` and p50 ms. Coverage 0 reads "no promise yet"
(a baseline: "no promise at 5%"); a missing or null field reads "–". The page's only choices: the highlighted row (highest repeat coverage at 5%, then unseen;
with no coverage anywhere, highest accuracy) and the verdict sentence built from that row's fields. `notes` are listed verbatim. No rows: "no campaign has run yet".

## Open gaps (what the page still cannot show from the API)

1. The spares' certificates come from `spare_certificates`, a field `main.py`'s `PageState` adds to `State`; move it into `contracts.State` with the next `just schema`.
2. "circuit breaker: 100%" is comparison copy, not a measurement; the page prints it as "circuit breaker: 100% (S4)" with the S4 smoke named in its tooltip.
3. The fault switch's own state (unplugged or not) is not in `State`; after a reload the rain cloud comes from `Policy.outage`, which is the detected state and clears after a successful probe.
4. The replay footer ("cpu · batch size 1 · 1 thread") comes from `Health.profile`, not from the row; rows carry only the env hash.
