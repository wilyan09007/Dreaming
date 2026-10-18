# Demo video hand-off (2026-10-18 10:40 PDT, deadline 11:30 PDT)

## Goal (the user's words, verbatim)
"I filmed the voiceover can you create a demo video backing it, with an animated mouse showing what you're clicking on and dynamically going around the components? Files are in part1.m4a and part2.m4a in CWD, 'For those of you who have incredibly long sequentially chained workflows' got repeated, at the last line 'Keep on dreaming.' I need the logo reveal, clean subtitles matching our design philosophy (make sure appropriate visual size) would be nice, map actions to text, Verify everything works visually. Use any tools or services that you find that might be useful. ... Keep going until you have something you're really proud of. Needs to be done by 11:30 am at the latest. Also feel free to cut out any dead noise and this clip probably needs to be under 2 minutes. If you need to do any speed-ups or cut out any parts, that's completely fine. Also maybe upload the audio files and your work for this to a new branch."

Branch: `demo-video` (this folder). Deliverable: `demo/video/dreaming-demo.mp4`, 1920x1080, under 2:00, plus the scripts that build it.

## Audio (measured)
- part1.m4a 102.6 s, part2.m4a 39.1 s. Word timestamps (faster-whisper small.en) in `words.json` (keys part1.m4a / part2.m4a; times in seconds of each file).
- Whisper hears "Psi" as "SIE": the subtitles must say **Psi**.
- CUT part1 from 99.14 s to the end ("For those of you who have incredibly..." is a false start; the full line is part2 0.64-8.98).
- Speech totals ~125 s, so: trim inter-phrase gaps to ~0.2 s, cap pauses inside phrases at ~0.25 s, then `atempo` ~1.15-1.2 so the whole video (logo intro + speech + outro) lands at ~115 s. Light `afftdn`, then `loudnorm=I=-16:TP=-1.5`.

## Phrases (file time) and the shot each maps to
| # | file | time | line | shot |
|---|---|---|---|---|
| 0 | - | - | (logo reveal ~2 s) | custom HTML: cloud logo pops, "dreaming" wordmark, rainbow underline draws |
| 1 | p1 | 0.78-9.78 | A lot of AI agents and orchestration platforms today do their job by asking a big, slow language model a question at every step. | deck `#hook` (headline "Your agent calls a big model on every step") |
| 2 | p1 | 10.56-21.62 | Dreaming learns to answer the repetitive questions in milliseconds for almost nothing ... thousands of times a day. | deck `#race` (teacher 11.5 s vs Dreaming 138 ms, x1,000 counters) |
| 3 | p1 | 22.40-34.02 | We can slash those costs by up to 87% while lowering processing latency from 2.6 seconds to 5 milliseconds using our novel workflow language model, Psi. | deck `#bench` (cursor: 87% card, then the 2.6 s vs 5 ms bars) |
| 4 | p1 | 34.64-37.52 | So let's start by running 2,000 workflow runs. | live `#flows`, cursor clicks "run 2,000 more" (INTERCEPT the POST, see below) |
| 5 | p1 | 38.00-45.56 | What's amazing about Psi is that it can pick up on workflow results quickly and with little data with its structural memory capabilities. | deck `#psi` (memory dots, softmax attention) |
| 6 | p1 | 46.26-59.70 | Of course, a lot of companies have different error rates ... statistical bound on the error ... stand-ins take over when that bound is under your budget. | deck `#bench` coverage card (budget ruler) then deck `#train` (certified badges "bound 4.1%") |
| 7 | p1 | 60.28-68.12 | You can adjust that by clicking on the Dreams tab here and adjusting the number of allowed mistakes and watching coverage shrink or grow accordingly. | live: click the dreams tab, drag the "mistakes allowed" slider (INTERCEPT /budget) |
| 8 | p1 | 68.86-74.92 | For those of you who deal with compliance, you've probably noticed a big issue with your LLM providers. | live: click the replay tab |
| 9 | p1 | 75.38-83.28 | That is, even when you set the temperature to zero, these providers will send back different answers ... different machines ... | replay tab: cursor over the "big model / recorded answer" rows |
| 10 | p1 | 83.92-89.84 | We hash every single machine that we use and we attach these hashes to every single response we give back. | replay: pick a run with replayable decisions (e.g. CON EDISON *EMEA, run 1,996, 3 of 4 replayable), cursor to the "pinned for replay: weights by hash" footer |
| 11 | p1 | 90.36-98.26 | That way, anyone can reproduce our model returns with a byte-identical return ... our replay functionality. | click "replay this run" (POST /replay is a read-only re-execution; fine on live) |
| 13 | p2 | 0.64-8.98 | For those of you who have incredibly long sequentially chained workflows, we can run those independent steps in parallel using speculative tool predictions. | deck `#build`: cursor clicks a few chips to make a long chain, then "Run it" |
| 14 | p2 | 9.48-13.54 | That's part of what allows us to lower latency by up to 500 times. | build race finishing (Dreaming lane in ms vs the teacher crawl) |
| 15 | p2 | 14.06-17.46 | Finally, you can always manage your own data and your own model training. | live `#dreams` journal (cursor sweeps the per-step rows) |
| 16 | p2 | 17.92-29.40 | If we go to scores here, we can see exactly how different data points have been categorized ... decide at each node whether to use our stand-ins or not. | live: click the scores tab (it loads slowly: open it once before capture), cursor across the columns |
| 17 | p2 | 30.08-36.04 | For those of you dealing with data handling policies, Dreaming is available in self-hosted and fully managed flavors. | deck `#close` (cursor drags the traffic dial) |
| 18 | p2 | 36.68-37.62 | Keep on dreaming. | outro: logo + "keep on dreaming" + dreaming-demo.fly.dev |

## Screens
- Deck: `dreaming/apps/web/launch/index.html` (works from file://). Screens by hash: hook, race, bench, psi, train, build, close. Key `r` replays the current screen's animation (press it when a shot's capture starts). Stage is 1600x900, letterboxed; at 1920x1080 it fills.
- Live app: https://dreaming-demo.fly.dev/ tabs by hash `#flows #dreams #replay #scores`. Current live state: 2,000 runs, saved $4.12.
- DO NOT mutate the live demo: intercept `POST /batch` and `POST /budget` with Playwright `page.route` (fulfill 200 `{}`) so clicks animate without changing the server. `POST /replay/{id}` is fine.
- Logo SVG + wordmark markup: `dreaming/apps/web/index.html` lines 293-294 (cloud with sleepy face, purple "z"s, rainbow underline in #FF8A80 #FFB84D #7BD8B4 #7FB7FF #B39DFF).
- Design: dotted paper background, Gaegu 700 for hand-lettered headings, Bricolage Grotesque for body, ink #2B2B33, purple #6A4FD8, cards with 2px ink border and an offset shadow. Subtitles: Bricolage Grotesque ~38-40 px at 1080p in a paper card (white, 2px ink border, offset shadow), bottom-center, one line per cue (split phrases into ~6-9 word chunks from the word timestamps), never covering the cursor's target.

## Build approach (planned)
1. Python: compute keep-intervals from words.json -> ffmpeg filter (atrim/concat, afftdn, atempo, loudnorm) -> `voice.m4a`; emit the final-timeline time of every word -> `timeline.json` (phrase starts/ends + subtitle chunks).
2. Node + Playwright (chromium): one shot per row above. For each shot open the page, settle, inject an overlay (animated cursor SVG with eased moves + click ripple, and the subtitle card driven by the global cues), then capture frames with CDP `Page.startScreencast` (jpeg q~92) for exactly the shot's duration while running its choreography; real clicks via `page.mouse` at the cursor's coordinates.
3. ffmpeg: frames -> per-shot CFR 30 fps clips (concat demuxer with per-frame durations), concat shots, mux voice, H.264 CRF 18 + AAC 192k.
4. Verify: extract frames at every phrase midpoint and look at them (cursor on the right target, subtitle legible, nothing blank), check total duration < 120 s and A/V sync at a few cue words.
