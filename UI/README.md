# Dashboard spec

Judge-facing and stage-facing. Consumes the event stream in [`../design/CONTRACTS.md`](../design/CONTRACTS.md) section 7. Paper is the brief's suggested bolt-on for the dashboard; the choice of tooling is open.

## 1. Purpose

Make two claims visible in seconds, and survive questions about them:
1. Cost per run falls as steps get promoted.
2. Failure rate falls as the agent routes around flaky tools.

Plus one live control: **kill a tool**.

## 2. Layout

```
+-------------------------------------------+---------------------------+
| HERO CHART: two lines vs run number       | CALL TYPES                |
|  - cost per run (rolling mean)            |  categorize   PROMOTED    |
|  - failure rate (rolling mean)            |  flag_anomaly SHADOW      |
|  promotion / demotion ticks on x-axis     |  recover_...  PROMOTED    |
|                                           |  (bound, coverage, audit) |
+-------------------------------------------+---------------------------+
| LIVE RUN FEED                             | TOOLS                     |
|  step -> real | predicted | audit | routed|  bank_feed  p(fail) bar   |
|  with cost per step                       |  vendor_api p(fail) bar   |
|                                           |  [ KILL TOOL ]            |
+-------------------------------------------+---------------------------+
| SUMMARY STRIP: baseline vs Dreaming (cost, latency, accuracy, frontier calls/run)  |
+----------------------------------------------------------------------------------+
```

## 3. Panels

### Hero chart
- X: run number. Two series on **one chart**, each normalized to its own baseline (baseline = 100%), so both visibly fall from the same starting height. Tooltip shows raw dollars and raw failure percentage.
- Rolling mean with a stated window (for example 20 runs); faint raw points beneath.
- Vertical ticks for `state_change` events (promotion, demotion), labeled with the call type.
- Optional dashed baseline arm for reference when a baseline run is available.

### Call types panel
Per call type: state chip, shadow agreement with its lower bound, coverage at `tau_serve`, current audit rate, live audit agreement, dollars saved. Click for a detail drawer: risk-coverage curve and recent disagreements.

### Live run feed
Newest first; each step is a pill: `real`, `predicted`, `audit`, `routed`, `recovered`. Shows cost and latency per step. Disagreements from audits are shown, not filtered.

### Tools panel
Per endpoint: `p(fail)` bar, status, recent latency. **Kill** and **Revive** buttons call the mock control endpoints. On kill, the panel highlights the next recovery event.

### Summary strip
Baseline vs Dreaming on held-out rows: latency, cost, accuracy, frontier calls per run. Marked "measured" with the run count and seed.

## 4. Behavior details

- Must render sensibly with **replay mode** feeding the same events.
- Update at most ~10 times per second; smooth but not laggy under concurrency 8 to 16.
- Reconnect automatically if the stream drops, and recover state from a snapshot endpoint.
- Keep a terminal fallback view of the same events (see `../demo/README.md` section 7).

## 5. Design notes

- Projector-first: large type, high contrast, no reliance on color alone (state chips carry text).
- Two lines are the point; restrain everything else. Use at most two accent colors on the hero chart and keep state colors consistent across panels.
- Do not smooth away the long-tail floor; the flat portion of the curve is evidence of honesty.
- Light and dark themes; test on a projector-like low-contrast setting.

## 6. API needs from the backend

- Event stream (`run_started`, `step`, `run_finished`, `state_change`, `audit_result`, `tool_health`, `tool_killed`, `recovery`).
- `GET /snapshot`: current registry states, last N runs, rolling aggregates.
- `POST /control/kill`, `POST /control/revive` (proxied to the mocks).
- `GET /summary`: baseline vs Dreaming table for the latest completed evaluation.

## 7. Open items

- Choose implementation (React page vs Paper vs minimal HTML). Prefer the simplest that can be made reliable offline.
- Decide whether the text interface (Photon bolt-on) is worth any time; default no.
