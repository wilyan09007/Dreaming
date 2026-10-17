# Demo video and submission

The brief: **reserve the last four hours for the demo**; every 2026 winner report says the video took longer than expected. Prepare this material early so those hours are for recording and polish, not invention.

## 1. Constraints to confirm

Length limit, format, and required elements (see checklist in `../research/05-sponsors-and-track-fit.md`). The script below assumes about 2 to 3 minutes.

## 2. Story arc

One principle, stated once, then shown:

> "A runtime that learns a model of every call your agent makes, then stops making the ones it can predict."

Do **not** list mechanisms. Show the two lines and the kill.

## 3. Script outline

| Time | Visual | Narration (draft) |
|---|---|---|
| 0:00 to 0:15 | Agent workflow diagram; frontier calls lit up with dollar signs | "Every agent step costs a model call, and every tool call can fail. Today agents pay for both, every time." |
| 0:15 to 0:35 | The principle on screen; the lifecycle loop (observe, shadow, promote, audit) | "Dreaming learns a model of each call. It runs the model in shadow, and only when it agrees with reality does it stop making the call." |
| 0:35 to 1:15 | Hero chart filling in live; promotion ticks; call-type panel flipping to PROMOTED | "Cost per run falls as steps are promoted. Failure rate falls because the tool model routes around flaky endpoints." |
| 1:15 to 1:50 | **Kill a tool**; feed shows recovery with 0 frontier calls | "Now we kill a tool mid-run. The agent reroutes without a frontier call, because its predictor already expected the failure." |
| 1:50 to 2:15 | Baseline vs Dreaming table on held-out rows, tiers 0/1 off | "On held-out rows, with the cache disabled, it matches the frontier baseline within [measured margin] at [measured cost]." |
| 2:15 to 2:30 | Audit panel; honest floor | "It still audits a few percent of its own guesses. If agreement slips, it demotes itself." |
| 2:30 to 2:45 | Sponsor stack, repo, Devin credit | "Built with Devin on veri.studio, Intuit data, Sentry tracing..." (only what is actually used) |

All bracketed numbers must be filled with measured values or the line cut.

## 4. Shot list and assets

- [ ] Screen capture of a full live run at 1080p (and a replay-mode capture as backup)
- [ ] Clean capture of the kill moment, two takes
- [ ] Hero chart final frame, exported as an image for the Devpost page
- [ ] Baseline vs Dreaming table image
- [ ] Architecture diagram (from `../design/DESIGN.md` section 5)
- [ ] Voice-over recorded separately from screen capture so either can be redone

## 5. Devpost page outline

1. One-sentence pitch (the principle)
2. The problem (cost and reliability of agents)
3. What it does (two lines, one moment)
4. How it works (lifecycle in one diagram; three call kinds)
5. Results (measured only; error budget and seed stated)
6. What we did not build or claim (honesty section)
7. Sponsor tech used (only real integrations)
8. Built with Devin (per brief, say so in the README too)
9. Links: repo, video

## 6. Timeline inside the last four hours

| Slot | Task |
|---|---|
| T-4h to T-3h | Code freeze. Final full run; capture charts and table |
| T-3h to T-2h | Record screen takes and the kill moment |
| T-2h to T-1h | Voice-over and edit |
| T-1h to T-30m | Devpost page, README, sponsor registrations verified |
| Final 30m | Submit early; keep buffer for upload problems |
