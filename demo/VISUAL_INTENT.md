# Dreaming demo — visual intent

What every part of `demo/index.html` is for, what it should look like, and how to check it. This is the spec the visual QA pass checks against.

## The story the demo tells

Pitch from `missions.md`: **every saving comes with a certified error bound, and every decision can be replayed.** The three tabs are the three parts of that sentence:

| Tab | Part of the pitch | Live demo moment (`missions.md` › Goal) |
|---|---|---|
| **flows** | "every saving" | one workflow run many times, four lines moving; kill a tool and watch the agent keep working |
| **dreams** | "comes with a certified error bound" | change the error budget and watch every promise re-certify |
| **replay** | "every decision can be replayed" | pick a past decision and watch it re-run byte for byte |
| **scores** | the proof behind all three: how each candidate does on rows it never trained on | the scored benchmark table, read straight from `GET /benchmark` |

The agent is the bookkeeping workflow from `missions.md` (categorize → vendor lookup → match invoice → anything odd? → post / wake a human). The numbers come from `smokes/FINDINGS.md`: coverage at a 5% budget (S1), the vendor tool skipping 70–76% of calls and 89% of runs finishing during an outage (S4), thinking off disagreeing too often to certify at 5% (S5), and batch-1 CPU replay (S7).

**Vocabulary (keep it consistent everywhere):** *awake* = the real call (big model or API) answers. *watching* = a dream runs in shadow, "counting sheep" (collecting 500 real answers). *testing* = certifying. *dreaming* = promoted: the dream answers first, and the real call happens when it is unsure plus 1 in 20 spot checks. A *promise* is a certificate.

## Principles (added 2026-10-18 03:40, from the user)

Follow the style guide below to a T, but the *components and flows* are not sacred: the page is for an ops or platform team that runs agents on infrastructure they already have, so every element must be useful to that reader (what is dreaming, what it saved, what it promised, what can be replayed, which candidate is serving). Uncluttered beats complete. This is a hackathon demo, so flash and pomp are welcome; anything that makes no sense from any angle is not, however pretty.

## Style source of truth

The design is **D · Lucid Doodle** from the "Dreaming UI Explorations" canvas. Reference renders (`doodle-{Flow,Distill,Data}.png`) and their sources (`Doodle-*.dc.html`) were kept outside this repo. The demo should look like the same product.

- **Paper:** `#FFFEFB` ground with a 22px dot grid (`#DCD8E4`), plus 3–4 large blurred pastel blobs (pink, blue, mint, butter) fixed behind everything.
- **Ink:** `#2B2B33` for text and outlines. Muted text `#5C5A66`. Accent purple `#6A4FD8` for handwritten notes and highlights. Green `#1E7A55` for "after" costs and success.
- **Type:** Gaegu 700 for every heading, label, number and handwritten note. Bricolage Grotesque for small body text (sublabels, explanations). Monospace only for bank descriptors and hashes.
- **Shapes:** wobbly asymmetric border radii (no perfect rectangles), 2–2.5px ink borders, hard offset shadows (`4px 5px 0 rgba(43,43,51,.13)`; buttons `3px 4px 0 #2B2B33`), small rotations (−3° to +2.5°) on cards, notes and nodes.
- **Decoration:** washi tape strips (translucent lavender, yellow, blue, coral) on active or important items only. Icons are hand-drawn stroke SVGs passed through the `#rough` displacement filter. No emoji.
- **Minimal:** nothing on screen without a job. No toolbars, zoom controls, node palettes, settings or fake chrome. If an element doesn't serve the story above, it's a bug.

## Global

### Header (all tabs)
- **Logo:** sleeping doodle cloud with purple "z z" (gently bobbing), the word "dreaming" in Gaegu, and a 5-colour crayon squiggle under it. Links to flows.
- **Tabs:** four sticky notes, each with an icon and a label. *flows* yellow (−3°), *dreams* lavender (+2.5°), *replay* blue (−1.5°), *scores* mint (+2°, a hand-drawn bar chart). The active tab is bigger (74px vs 62px), has a tape strip on top and a deeper shadow. Exactly one is active.
- **"saved so far" sticky:** pale yellow sticky note at +2° with a blue tape strip and a jar of stars. It counts up in dollars and cents while runs happen, starts at $0.00, and never goes negative. *Why:* the running total of what dreaming saved.
- Layout is a three-column grid (logo left, tabs dead centre, sticky right), so the tabs never move when the dollar amount gets wider.
- Check: all three header groups (logo, the four tabs, the sticky) sit on one row at ≥1024px with no overlap, tabs dead centre. Tab labels are readable.

## flows tab

### Canvas card
Large card with a wobbly 2.5px ink border, the title **"nightly books"** in Gaegu 40px, and a narration line under it.
- **Narration line** (Gaegu, purple; muted grey when idle). One sentence saying what just happened: the *proof* ("categorize passed its check — wrong 3.1%, 5% allowed"), while the handwritten note on the graph gives the *payoff*. Step names inside it get a yellow highlighter stroke (no quote marks). During an outage the outage message stays up until the API is plugged back in, even through run, pause and resume. Idle: "a bookkeeping agent, wrapped not rewritten · press run to fast-forward 2,000 transactions". During a run it announces each event ("a dream is shadowing categorize…", "checking the categorize dream against 500 real answers", "categorize passed its check — wrong 3.1%, 5% allowed", …). Pacing for a recording: a verdict line stays up at least 2 s, and other event lines at least 0.9 s. Lesser lines wait their turn; when several wait, only the newest is kept, so a shadowing or checking line may be skipped but a verdict never is. Lines caused by the presenter (run, pause, unplug, budget summary, done) show at once. The first batch ends with "done · cost per run 13.1¢ → 7.8¢ · click a step to see the promise behind it". It must stay on one or two lines and must not overlap the run button.
- **Run counter** ("1,240 runs", muted Gaegu) and the **coral Run button**: ▶ "run 2,000" → "pause" (white background while running) → "resume" → "run 2,000 more". `?speed=N` speeds it up for testing; at 1× a batch takes about 18 s.

### Workflow graph (`#stage`)
Seven nodes left to right, joined by crayon-coloured wires with hand-drawn arrowheads. While running, small ink dots travel along the wires.

| Node | Type | Sublabel by state |
|---|---|---|
| bank feed | trigger, not clickable | webhook |
| categorize | call site (model) | big model → dream shadowing → checking the dream → **dream answers 89%** |
| vendor lookup | call site (tool) | vendor API → … → **dream answers 74%** |
| match invoice | call site (model) | big model → … → **dream answers 78%** |
| anything odd? | round branch node (router) | big model, thinking → thinking off on trial → checking thinking off → stays "big model, thinking" at 5% / "big model, thinking off" at ≥7% |
| post to books | write, not clickable | ledger |
| wake a human | human, not clickable | bookkeeper |

- Call-site nodes carry a **round status sticker** at the top-right corner (top-left on the round node) (ink-bordered circle, rotated 10°): sun = awake, dashed eye = watching, half-filled blue circle = testing (pulses), lavender sleeping cloud with a bobbing purple z = dreaming, grey rain cloud = API down.
- A **dreaming** node gets a yellow washi-tape strip across its top edge, a lavender hard shadow (`5px 6px 0 #B39DFF`) and a brief scale "pop" when it gets promoted (as "classify" is drawn on the design board). The round node's sticker sits at its top-left so it never covers the outgoing "no" wire.
- **Handwritten note** (purple Gaegu, −3°, with a hand-drawn purple arrow), as on the design board: one at a time, pointing at whatever just changed. Promotions: "learned to dream! −84% cost" under categorize, "knows what the API will say! −70% cost" above vendor lookup, "learned to dream! −74% cost" under match invoice, "thinks less now! −77% cost" above "anything odd?". Outage: "API down, still answering!" (or "stuck! nothing learned yet") above vendor lookup. The note must not cover any node, label, wire end or sticker, and it disappears when its statement stops being true.
- Branch labels **"no"** (to post to books) and **"yes"** (to wake a human) sit beside their wires, clear of the wires, the badge and the node label.
- Clicking a call-site node opens the dreams tab with that step selected (`#dreams/<id>`). Hovering a node deepens its grey shadow. A dreaming node keeps its lavender shadow on hover.
- **"unplug the API"** pill (yellow, ink border, hard shadow, plug icon) sits under vendor lookup. Pressing it turns the pill mint ("plug it back in", sparking plug icon) and gives the vendor node a dashed border, grey wash and rain-cloud sticker (sublabel "API down · dream + snapshot" or "API down · stuck"). The narration then says either "vendor API is down · the dream and the last snapshot keep 89% of runs going" (vendor lookup already dreaming) or "…nothing has learned it yet, so every run is stuck" (not yet). A batch that ends during the outage reads "vendor API is still down · the dream and the snapshot kept 89% of runs going" (or "…every run is stuck at vendor lookup"). When stuck, the travelling dots stop at vendor lookup, the run button and counter stay on the title row, and per-run cost, seconds and replayable freeze at their last finished-run values.
- Below about 700px wide the graph keeps a readable minimum scale and scrolls sideways inside the canvas card. The page itself never scrolls sideways.
- Check: no node, label, sticker or pill overlaps another. Sublabels fit on one line (outage text may wrap to two). The graph is horizontally centred and fully visible in the canvas at 1280×720, 1440×790, 1440×900 and 1920×1080. Spare height is split evenly above and below the graph.

### "the more it runs" card
One white card under the canvas, four cells split by dashed dividers: **cost per run** (coral line), **seconds per run** (blue), **failed runs** (gold), **replayable decisions** (green). Line colours are the same crayons as the wires. Each cell has a muted Gaegu label, a big Gaegu value, a Gaegu "was X" once runs > 0, and a crayon line over runs on a 2px ink baseline (x axis: "0 → 2,000 runs"). A faint dashed line marks each metric's starting level, so even before the first run the card reads as four lines waiting to move. The x axis leaves a little headroom, and each line ends in a short flat "now" tail, so a change at the latest run reads as a step, not a clipped edge. A budget change at one run draws a single clean step, however many times the slider moved. *Why:* this is the "chart of those lines moving" from `missions.md`.
- Expected after the first 2,000 runs at 5%: cost 13.1¢ → 7.8¢, seconds 9.6 → 5.3 s, failed 2.4% → 1.2%, replayable 0% → 60%. Cost, seconds and failures step down; replayable steps up.
- After raising the budget to 7%: cost drops to about 2.6¢ and seconds to about 2.1 s (a visible step at the current x).
- During an outage: failed runs jumps to 10.7% (100% if nothing had learned the API) and its "was" note becomes red "circuit breaker: 100%", which is the comparison that matters.
- Each line is scaled to its own peak from zero, so a halving looks like a halving and fills the cell's height.
- Check: lines stay inside their cells, the big values never touch the small labels above them, and the four cells have equal width.

## dreams tab

### Dream journal (left)
Lined notebook paper: blue rules every 48px, a red margin line, three punched holes, the title "dream journal" in Gaegu 44px. One row per call site (categorize, vendor lookup, match invoice, anything odd?):
- status pictogram (sun / dashed eye / half circle / sleeping cloud / rain cloud) · name in Gaegu 30px · status line ("dreaming · answers 89% of calls", "watching · counting sheep 320/500", "awake · no promise at 5% yet", "re-checking…") · cost per call on the right ("~~2.1¢~~ → 0.33¢" with a wobbly purple arrow and the new cost in green, or a single cost while awake). A single muted caption "cost per call" sits on the title row above that column.
- Hovering a row gives only a faint grey wash; just the selected row gets the lavender wash.
- The selected row gets a soft lavender wash and a yellow highlighter stroke under its name. Clicking a row selects it.
- Footnote in muted Gaegu: "post to books and wake a human never dream — they change the world." *Why:* it explains why only four of seven nodes appear.

### "mistakes allowed" card (top right)
White card at −0.6°: title, a big Gaegu value (5%), a purple range slider from 1% to 10% with tick labels, one sentence on what the budget means, and a dashed-separated live line: "right now: 7.8¢ per run · 60% replayable". *Why:* this is the operator's error budget and the "change the budget, watch it re-certify" moment.
- Releasing the slider re-certifies every certified step. Rows ripple: each row keeps its old numbers until its turn, shows "re-checking…" (half-moon icon) for about 0.45 s, then settles. Turns start 170 ms apart. While a row waits for or sits in its turn, its old cost stays visible, dimmed. The promise card keeps the old promise dimmed with "re-checking…" in place of the promise, and its cert number goes up only when its row finishes. The live line reads "right now: re-checking…" (same height as the settled line, so nothing below it moves) until the last row settles. The flows narration then summarises: "re-checked at 7% · 4 of 4 steps hold a promise · cost per run now 2.6¢". At **2%**: categorize ~52%, match invoice ~41%. At **7%**: anything odd? switches to dreaming with thinking off, and cost per run falls to ~2.6¢. At **1%**: vendor lookup still answers ~70% (a deterministic tool certifies at tight budgets).

### Promise card (bottom right)
White card at +0.5° with a lavender tape strip and a yellow hard shadow (`6px 7px 0 #FFD166`). For the selected step:
- header: status pictogram + name
- the promise in Gaegu 40px: "wrong ≤ 5%". Otherwise a smaller faint-grey "no promise yet" / "counting sheep…" / "checking…" / "no promise at 5%", which must look clearly weaker than a real promise.
- a row of 10 sheep (lavender when counted, dashed outlines when not), "1 sheep = 50"
- evidence line: "checked against 500 real answers · wrong on 3.1%."
- split bar: striped lavender for the dream's share, pale yellow for the real call's share, with a legend (entries at 0% are hidden)
- cost line "~~2.1¢~~ → 0.33¢ per call"
- dashed-separated fine print: "1 in 20 dream answers still double-checked · cert c-0412 · weights 9f3a1c". The cert number goes up by one on every re-certify.
- For "anything odd?" when promoted, the fine print adds "not replayable: it is still the big model". Thinking off is cheaper but still a frontier call, so replay reads it from the log.
- vendor lookup during an outage: rain-cloud pictogram. If it was dreaming, the promise stays, plus "API down: calls the dream is unsure about use the last snapshot." and the legend says "snapshot" for the remainder. If it had not learned yet: a faint "API down" and "nothing has learned this API yet, so every run is stuck here."
- For "anything odd?" at 5%: "no promise at 5%", "thinking off disagreed on 4.6% … could be as high as 6.4%", plus a purple hint "allow 7% and it switches thinking off".
- Check: nothing overflows the card; the sheep row fits on one line at 1440 wide.

## replay tab

### Run list (left)
Title "replay" in Gaegu 46px, a one-sentence explanation, then up to 7 receipt cards (newest first, with the very first run always kept last for contrast). Each receipt: white, wobbly corners, hard offset shadow, slight alternating rotation, monospace bank descriptor, Gaegu amount on the right, full-width muted meta line ("run 1,250 · 3 of 4 replayable", "· API down" when the API was unplugged). "Replayable" counts the decisions a hosted dream answered. A promoted "anything odd?" (thinking off) is still the big model, so it never counts. The selected receipt has an ink border and a lavender hard shadow.

### Trace card (right)
White card with a blue tape strip: "run 1,250" (Gaegu 36px), descriptor and amount in monospace, and a mint **"replay this run"** pill. Five rows split by dashed lines: step name · answer · who answered (icon + Gaegu: sheep = dream, moon = big model, plug = vendor API, camera = snapshot, book = ledger, clock = bookkeeper) · replay result.
- Pressing replay steps through the rows about 0.5 s apart. The current row gets a lavender wash and "re-running…" or "reading the log…". Dreamed rows end with a green check, "same bytes" and a 6-character hash; everything else reads "from the log" (the ledger row: "from the log, not re-posted", never broken across lines).
- Before replay, the same line previews the outcome in muted grey ("3 of 4 decisions can re-run exactly, press replay"). While replaying it reads "re-running 3 of 4 decisions…". The card keeps its height throughout. This line avoids "·" and "—", because Gaegu lacks them and the fallback glyphs shift the line.
- After replay, the summary line turns purple Gaegu: "3 of 4 decisions re-ran with the same bytes, the rest came back from the log" (one unit with the line before it: the ledger write is a fifth step, not a fifth decision). For run 7: "nothing re-ran: this run came before anything learned to dream" (one line). The pill keeps one width through "replay this run" / "replaying…" / "replay again".
- Stuck runs (API unplugged before vendor lookup learned): vendor lookup reads "no answer · stuck"; later steps, including a fifth "post to books" row, read "never ran" with nothing in the who column and no busy flash during replay (so stuck and normal runs have the same card height); the receipt says "stuck at vendor lookup"; the summary reads "categorize re-ran with the same bytes, then the run got stuck at vendor lookup".
- The selected receipt is always one of the receipts on screen (after a new batch the selection falls back to the newest run).
- A row the memory dream answered carries one muted Bricolage 13px line under its answer, the stored rows that voted for it: "voted: CON EDISON *EMEA, $335.54 (Utilities) 0.31 · CON EDISON *POS, $188.18 (Utilities) 0.08 · …" (a seed row reads "seed example"). It sits inside the row's fixed height and ends in an ellipsis rather than wrapping, so the card keeps its height.
- Footer, muted: "pinned for replay: weights by hash · cpu · batch size 1 · 8 threads".
- Empty state (before any run): "nothing to replay yet" plus a mint "go to flows" pill. No empty list, and no "pick a past run" sentence.
- Repeat merchants in later batches show different amounts. The same descriptor at a very different amount (Adobe $54.99 vs $5,499.00) is deliberate: that one is the anomaly.

## scores tab

*Why:* the hackathon goal is "scored benchmarks with our models showing how they outperform the competition", and the Intuit judges score a benchmark against classical and LLM baselines. Until now that table lived only in `GET /benchmark` (JSON) and `/benchmark.md`. This tab is that table, nothing more: an ops team reads which candidate to trust per call site, how much traffic it would take, how often it would be wrong, and what it would cost.

### Page head
Title **"scores"** in Gaegu 46px (as "replay") and one muted sentence: "how each dream did on rows it never trained on, at a 5% mistake budget. every number comes from the latest campaign's benchmark."
- The data is fetched when the tab opens, and again on every later open (it takes a few hundred ms). No campaign selector: it always shows the latest campaign.

### One notebook card per call site
Lined notebook paper exactly as the dream journal (blue rules, red margin, punched holes, ink border, wobbly corners), in workflow order (categorize, vendor lookup, match invoice; other sites after). Each card has:
- the **site name** in Gaegu 34px;
- a one-line **verdict** in purple Gaegu, computed from the row the table highlights. When some candidate holds a promise at 5%: "the seeded Hopfield memory answers 83% of repeat traffic at 5%, wrong 2.1%" (plus "and 40% of merchants it never saw" when that is non-zero too). When nothing holds a promise but accuracies are in the JSON: "no promise at 5% yet: the seeded Hopfield memory is right on 83% of repeat traffic and 90% of merchants it never saw". When neither: "no promise at 5% yet from any of the 9 candidates" ("from its one candidate"). Accuracy is never called "answers": answering means holding a certificate.
- a **table**, one row per candidate and variant (the server's order: best unseen coverage first), then the baseline rows. Rows are the paper's rules: a thin blue line under each.

| column | shows | from |
|---|---|---|
| who | the candidate in plain words in Gaegu (Hopfield memory, linear head, fine-tuned encoder, HDC bundle, exact cache; baselines: vendor rule, LLM alone, Jev, teacher alone) and, small and muted, the variant ("seed + real rows", "real rows", "seed rows only", "baseline"), "from its summary" when the scores could not be recomputed, "unverified" when not byte-stable, and "first promise after N s" on the seed-only row | `candidate`, `variant`, `source`, `stable`, `seed_cert_s` |
| repeat traffic | share answered at 5% in Gaegu, "wrong N%" small | `cells[5%].forward_coverage`, `forward_error` |
| unseen | the same on whole merchants held out; the 2/10/20% coverages in its tooltip | `cells[5%].test_coverage`, `test_error` |
| right on all calls | accuracy if it answered every call, "repeat / unseen" | `acc_forward`, `acc_test` |
| $ per 1,000 calls | "~~$10.00~~ → $4.12", the after in green when it is cheaper (the dream journal's cost idiom) | `without_dreaming.usd`, `with_dreaming.usd` |
| p50 latency | "800 → 5 ms" | `without_dreaming.p50_ms`, `with_dreaming.p50_ms` |

- The **best row** (the highest repeat coverage at 5%, then unseen coverage; when no candidate covers anything, the highest accuracy) gets a yellow highlighter stroke under its name, the same stroke as the selected journal row. If no candidate is ahead on any number, nothing is highlighted.
- A cell with coverage 0 (nothing certified at 5%) shows a faint "no promise yet" (a baseline: "no promise at 5%"), never "0%". A field that is missing or null in the JSON shows "–".
- **Fine print** (muted, dashed rule above), one per card: "repeat = the last 20% of the stream by time, merchants recur: the production case. unseen = whole merchants held out. $ and ms are projected from this site's logged teacher calls at the unseen 5% coverage. labels: <provider> (a tool site: "the tool's own returns"). campaign <campaign>." Plus the provider's floor when the JSON has one ("the labeller disagrees with itself on N%: no promise below that is measurable").
- Under the last card, a native `details` line "notes from the benchmark (N)" lists the JSON's `notes` verbatim (why a row fell back to its summary, estimated embed times). Closed by default.

### Empty and failed states
- No rows in `/benchmark`: one notebook card, "no campaign has run yet" (Gaegu) and a muted line "a campaign trains every candidate on a site's logged calls and scores it on rows it never saw. its table lands here."
- Before the first reply: the same card reads "reading the scores…". A refused request: "the scores could not be read: <status and detail>". A later failed refetch keeps the last table on screen.

### Nothing simulated
Every number is a field of `GET /benchmark` (README (e)). The page formats percentages, dollars and milliseconds, and picks the best row; it computes no number of its own.

- Check: the header holds four tabs with no overlap at ≥1024px; the table fits the card with no sideways page scroll at 1280×720 and 1440×900; small text ≥13px; the best row has the highlighter; "no promise yet" cells read clearly weaker than real numbers; no console errors.

## Recording sizes
Known limit, measured 2026-10-18 09:40 on the live page: at 1280×720 the flows graph scales to 0.79 (the canvas is height-limited by the metrics card), so node sublabels render at ~9.5 px. Rendering them at 13 px there needs a 16.4 px font, which makes "dream answers 89%" 155 px wide against 132 px of node text area; keeping it on one line needs ~230 px nodes, i.e. 1,234 px of nodes and gaps on the 1,200 px stage. The two rules below (≥13 px, one line) are therefore satisfiable together only from 1440 wide (scale ≥1.08), which is why the recording sizes are 1440×900 and 1920×1080. Not a bug to fix under the demo; a layout change (narrower blobs or a taller stage at short heights) would be the route.

At 1920×1080 the flows graph scales itself up, and the dreams, replay and scores tabs are zoomed 1.18× so all three tabs read at the same scale. Small explanatory text is at least 13px. On short screens (≤780px tall, e.g. 1280×720) the dreams column tightens its spacing, so the promise card fits without scrolling.

## Demo script (what will be recorded)
*Numbers on the API-backed page (`dreaming/apps/web`):* the figures in this file (13.1¢ → 7.8¢, 60%, 10.7%, ~2.6¢, ~70% at 1%) are the simulation's in `demo/index.html`. The API-backed page prints only the backend's fields (`apps/web/README.md` (e)), so check the beats there, not these figures. Cost, seconds, failed and replayable average the last 100 finished runs, so a budget change shows in the lines only once runs follow it. On the offline backend (`TEACHER=rule`, which `agents/bookkeeper/tests` pin) the teacher is an exact rule at 1¢ and 800 ms a call (no failed runs), its thinking-off configuration is the same rule (anything odd? certifies at 5%, so 7% changes nothing), and vendor lookup falls back to last night's snapshot (an outage fails no run and strands none, before or after learning).

1. flows, fresh: every step awake, $0.00, flat values.
2. Press **run 2,000**. Watch the stickers go sun → eye → half moon → sleeping cloud for categorize (~run 600), vendor lookup (~740), match invoice (~940). "anything odd?" starts watching at ~660 and stays awake at 5% (~1,200). The lines step, the savings tick up, and it ends with "done · cost per run 13.1¢ → 7.8¢ · …".
3. Click **categorize** → dreams opens with its promise.
4. Drag **mistakes allowed** to 2% (coverage falls), then 7% (anything odd? starts dreaming, cost per run ~2.6¢).
5. Back to flows: the step is visible in the lines. **Unplug the API**: failed runs 10.7% against the circuit breaker's 100%. **run 2,000 more**: runs keep finishing.
6. replay: pick a recent run → **replay this run** → same bytes. Pick run 7 → nothing could re-run.

## Driving it (for the QA agent)
- Demo: `http://127.0.0.1:8765/` (add `?speed=3` to fast-forward). Reference boards: `http://127.0.0.1:8766/Doodle-Flow.dc.html` (also Doodle-Distill, Doodle-Data). The references load without their design runtime, so a literal `{{rootClass}}` and a 404 for `support.js` are expected there.
- Browser: `bash "$SP/b.sh" <command>` (b2.sh and b3.sh drive two more isolated browsers), where `SP` is the folder holding these browser scripts (not in this repo). `b.sh` wraps gstack browse and talks to a persistent server. Useful commands: `viewport 1440x900`, `goto URL`, `click "<css>"`, `js "<expr>"`, `screenshot <path>`, `console --errors`.
- Slider: `js "const b=document.querySelector('#budget'); b.value=7; b.dispatchEvent(new Event('input')); b.dispatchEvent(new Event('change')); 1"`.
- Reloading the page resets the demo.
