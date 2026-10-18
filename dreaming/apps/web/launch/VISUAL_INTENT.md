# Launch flow: visual intent

The judges' walk-through on the big monitor. Seven screens, each one idea, each one moving. It is choreographed:
every number comes from `numbers.js` (measured values, best slice shown) and every animation runs on a script, so
it looks the same with or without wifi or the live stack. Style is the Lucid Doodle look of `apps/web` and
`demo/index.html`: dotted paper, crayon blobs, Gaegu hand lettering for headings and numbers, Bricolage for body,
2.5px ink outlines, sticky notes with a slight tilt, a soft offset shadow (`--lift`). Teacher is warm coral
(`--teacher`), Dreaming is purple (`--ours`), mint/green means "certified / served", red means "pruned / wrong".

Open `launch/` (served by the API at `/launch/`, or any static server). Keys: right / PageDown / space = next,
left / PageUp = back, 1-7 jump, R = replay the current screen, F = fullscreen. `?presenter` opens a presenter
window (stage on top, notes below) that drives every display window on the same machine.

## Global
- The stage is a fixed 1600x900 canvas, letterboxed to the window. Check: at 1920x1080, 1440x900 and 1280x800 nothing is cut off or overlaps, and there is no scrollbar.
- Logo top-left on every screen. Check: the cloud logo and "dreaming" are visible on every screen and nothing covers them.
- Screens cross-fade (~0.45 s). Check: fast key mashing (right x10 in 1 s, then left x10) leaves exactly one screen visible, no stray animation from a screen that was left, and no console errors.
- Every screen settles within 12 s of arriving, with no layout jump after it settles. R replays it from the start.
- No text under 16px on the stage (it is shown on a monitor across a room); headline numbers are 56px or larger.
- Frame rate: animations use transform/opacity and rAF; Check: no long tasks over 50 ms after the first paint.

## 1 hook: "Your agent calls a big model on every step"
- Left: kicker "agents, but awake only when it matters", a big hand-lettered title, and four pain sticky notes landing one by one (cost, latency, outages, audit), slightly rotated.
- Right: four headline stat tiles count up (87% cheaper, ~500x faster per call, 0 LLM tokens on 89% of calls, certified <=5% error), each with a one-line plain-English caption.
- Check: all four tiles reach their final numbers by 4 s; the numbers never overflow their tiles.

## 2 race: the example workflow, teacher vs Dreaming
- Two lanes, same bookkeeper workflow (webhook -> categorize -> vendor lookup -> match invoice -> anything odd? -> post to books). Top lane "LLM agent: step-5-preview on every step" (coral), bottom lane "Dreaming: certified stand-ins" (purple).
- Auto-plays on arrival: an invoice token travels each lane. A node pulses while working and glows (ring + check) when it finishes, with its real latency printed under it (2.6 s, 0.9 s, 4.8 s, 3.1 s, 0.12 s for the teacher; 5 ms, 3 ms, 6 ms, 4 ms, 120 ms for Dreaming). The Dreaming lane finishes in a blink and its finish line shows "done in 138 ms"; the teacher crawls for 11.5 s in real time while a stopwatch runs on both lanes.
- Then a "x1,000 invoices" phase: per-lane counters roll smoothly (invoices done, $ spent, LLM calls, hours) to the per-1k values in numbers.js: $3.38 vs $0.67, 3,000 vs 649 calls, 3.2 h vs 0.6 h. A verdict sticky lands: "80% cheaper per invoice, 83x faster end to end".
- Check: counters only go up and stop exactly on the final values; no counter freezes mid-way; the token never jumps backwards; both lanes are fully visible.

## 3 bench: benchmark graphs
- Three hand-drawn charts side by side, bars grow on arrival (staggered): (a) $ per 1,000 calls, teacher vs Dreaming per LLM site (categorize 0.58 -> 0.07, match invoice 1.69 -> 0.38, anything odd 1.11 -> 0.22) with the % saved on each pair; (b) p50 latency per call on a log scale (2.6 s / 4.8 s / 3.1 s vs 5 / 6 / 4 ms) labelled "~500x"; (c) coverage at a certified <=5% error budget (89%, 78%, 80%) with the 5% budget line and the 3.5% "teacher disagrees with itself" floor explained in one line.
- Check: every bar has its value label, no label collides with another or with an axis, the log axis ticks read 1 ms / 10 ms / 100 ms / 1 s / 10 s.

## 4 psi: the Psi model diagram
- Title "Psi: a workflow language model". A left-to-right diagram: a new call ("AMZN MKTP *WEB $63.20") -> encoder (bge-small, 384-d) -> query vector -> memory of 8,219 past traces drawn as a grid of dots -> softmax attention (the nearest few dots light up with weights) -> a weighted vote -> answer "Office supplies" with confidence -> certification gate (Clopper-Pearson, <=5%) -> "served, 0 LLM tokens"; a dashed branch for low confidence goes to the teacher and its answer is appended back into memory as a new dot ("learn = append, forget = delete, no training step").
- The update rule in small type: xi_new = X softmax(beta X^T xi).
- Animated on arrival as a 2-3 s pass; then a second query that is unfamiliar takes the teacher branch and a new dot is appended.
- Check: arrows connect the boxes they point at; the lit dots are the same ones the weights refer to; nothing overlaps.

## 5 train: model training management (the dream campaign)
- A grid of cells: rows = candidates (linear head, Psi, fine-tuned encoder, hdc bundle, exact cache) x columns = sites, small chips per variant/seed. A rung ruler on top: 60 / 200 / 500 / 2,000 rows.
- Plays as a ladder: all cells start at 60; at each rung, cells that fail a hard rule (error above budget, byte-unstable replay) turn red and drop out with a tiny reason tag; survivors climb. A per-site winner gets a "certified" stamp with its bound shrinking (e.g. 9.8% -> 4.1%). A footer: "45 cells, pruned automatically, under $5 of GPU for the whole campaign; repeat traffic = last 20% by time, unseen = whole merchants held out".
- Check: the counts are consistent (cells alive + pruned = 45 at every rung); stamps land on the cells that survived.

## 6 build: build your own workflow
- The judge edits a given workflow. Left: an editable list of steps (text inputs, pre-filled with the bookkeeper steps), with add / remove / move up / move down and a row of quick-add chips. As they type, each step snaps (debounced, 150 ms) to the nearest step we have measured, shown as a pill ("~ categorize, Psi serves 89%"); unknown steps snap to the closest one with a similarity bar, and steps with no decent match say "new step: the teacher answers until a stand-in certifies".
- Right: the workflow re-draws as a small graph of the snapped steps, and "Run it" plays the same teacher vs Dreaming race as screen 2 for their workflow, with totals composed from the snapped sites.
- Check: typing never loses focus or caret position; deleting every step shows a friendly empty state and Run is disabled; 12 steps still fit; weird input (emoji, 200 characters, only spaces) never breaks layout; keys typed into inputs never change screen.

## 7 close: what it means at scale
- A "saved so far" jar style card with a monthly-calls slider (100k to 100M, log steps): $ saved per year, LLM calls avoided, and hours of waiting removed, from numbers.js per-1k values. Below: three proof stickers (certified <=5% error, replay is byte-identical: 3 of 3, 1 in 20 served calls is audited against the real model) and the live URL `dreaming-demo.fly.dev`.
- Check: dragging the slider updates the numbers in the same frame, no layout jump as digit counts change (tabular numbers, fixed widths).
