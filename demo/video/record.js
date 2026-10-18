// Records every shot of the demo with Playwright (CDP screencast), then builds dreaming-demo.mp4 with ffmpeg.
// Needs timeline.json + voice.m4a from audio.py. Usage: node record.js [captureName ...]  (no args = all, then assemble)
// The live site is never mutated: POST /batch and POST /budget are answered locally; POST /replay is a read-only re-run.
const { chromium } = require('playwright');
const fs = require('fs'), path = require('path'), { execFileSync } = require('child_process');

const HERE = __dirname;
const TL = JSON.parse(fs.readFileSync(path.join(HERE, 'timeline.json'), 'utf8'));
const OVERLAY = fs.readFileSync(path.join(HERE, 'overlay.js'), 'utf8');
const url = (p) => 'file:///' + path.resolve(HERE, p).replace(/\\/g, '/');
const DECK = url('../../dreaming/apps/web/launch/index.html');
const LOGO = url('logo.html');
const LIVE = 'https://dreaming-demo.fly.dev/';
const WORK = path.join(HERE, 'build');
const FPS = 30;
const FFMPEG = process.env.FFMPEG || (fs.existsSync('C:/ffmpeg/bin/ffmpeg.exe') ? 'C:/ffmpeg/bin/ffmpeg.exe' : 'ffmpeg'); // needs libx264
const sleep = (ms) => new Promise((r) => setTimeout(r, Math.max(0, ms)));
const shot = (n) => TL.shots.find((s) => s.name === n);

// final-timeline time of the first word matching `w` (case/punctuation-insensitive) at or after `after` seconds
const norm = (s) => s.toLowerCase().replace(/[^a-z0-9%]/g, '');
function T(w, after = 0) {
  for (const p of TL.phrases) for (const [word, s] of p.words) if (s >= after && norm(word) === norm(w)) return s;
  throw new Error(`word not found: ${w} after ${after}`);
}
// deck coordinates: probe screenshots are the 1600x900 stage at 1.2x; we show it at 1.1x, centered, top-aligned
const D = (x, y) => [80 + (x / 1.2) * 1.1, (y / 1.2) * 1.1];
// __warp(k) runs the page's own timers k times faster (the race needs 12 s; its shot is 9.6 s). The overlay keeps real time.
const WARP = `(() => {
  const rNow = performance.now.bind(performance), rST = setTimeout.bind(window), rSI = setInterval.bind(window), rRAF = requestAnimationFrame.bind(window);
  let k = 1, base = 0, rBase = 0;
  const v = () => base + (rNow() - rBase) * k;
  performance.now = v;
  window.__warp = (nk) => { base = v(); rBase = rNow(); k = nk; };
  window.setTimeout = (fn, ms = 0, ...a) => rST(fn, ms / k, ...a);
  window.setInterval = (fn, ms = 0, ...a) => rSI(fn, ms / k, ...a);
  window.requestAnimationFrame = (fn) => rRAF(() => fn(v()));
})();`;
const DECK_FIT = '#stage{transform:translate(80px,0) scale(1.1)!important}#dots{display:none!important}';

const CAPTURES = [
  { name: 'intro', from: 'intro', to: 'intro', url: LOGO, cursor: false, run: async (c) => { await c.page.evaluate(() => play()); } },
  {
    name: 'deck1', from: 'hook', to: 'bench', url: DECK + '#hook', css: DECK_FIT, start: [1500, 1000],
    run: async (c) => {
      await c.page.keyboard.press('r');
      await c.at(T('agents') - 0.3); await c.move(...D(300, 300), 900);           // the headline
      await c.at(T('big,') - 0.2); await c.move(...D(330, 520), 800);             // "every step"
      await c.at(T('step.') - 0.3); await c.move(...D(700, 670), 700);            // latency sticky: 11.5 s of waiting
      await c.at(shot('race').start); await c.page.evaluate(() => __warp(1.45)); await c.hash('race'); await c.move(...D(470, 330), 900); // teacher lane
      await c.at(T('milliseconds') - 0.3); await c.move(...D(1205, 515), 800);    // Dreaming: done in 138 ms
      await c.at(T('nothing,') - 0.1); await c.move(...D(1675, 595), 800);        // $0.67 spent
      await c.at(T('thousands') - 0.3); await c.move(...D(1510, 522), 700);       // x 1,000 invoices
      await c.at(T('day.') + 0.1); await c.move(...D(960, 800), 800);             // 80% cheaper, 83x faster
      await c.at(shot('bench').start); await c.page.evaluate(() => __warp(1)); await c.hash('bench'); await c.move(...D(600, 220), 700);
      await c.at(T('87%') - 0.4); await c.move(...D(195, 372), 700);              // 87% cheaper card
      await c.at(T('latency', 20) - 0.2); await c.move(...D(822, 545), 800);      // 2.6 s teacher bar
      await c.at(T('5', 24) - 0.2); await c.move(...D(888, 760), 600);            // 5 ms Dreaming bar
      await c.at(T('psi.') - 0.8); await c.move(...D(1760, 135), 900);            // legend: Dreaming
    },
  },
  {
    name: 'run', from: 'run', to: 'run', url: LIVE + '#flows', start: [1180, 420],
    run: async (c) => { await c.click(1595, 153, T('running', 29) - shot('run').start - 0.1); },
  },
  {
    name: 'deck2', from: 'psi', to: 'bound', url: DECK + '#psi', css: DECK_FIT, start: [1500, 700],
    run: async (c) => {
      await c.page.keyboard.press('r');
      await c.at(T('psi', 32) - 0.2); await c.move(...D(195, 470), 800);           // a new call
      await c.at(T('pick') - 0.2); await c.move(...D(900, 545), 900);              // memory dots
      await c.at(T('little') - 0.2); await c.move(...D(1290, 470), 700);           // weighted vote
      await c.at(T('structural') - 0.2); await c.move(...D(1180, 885), 800);       // learn = append
      await c.at(shot('bound').start); await c.hash('bench'); await c.move(...D(1550, 380), 900); // coverage card
      await c.at(T('strong') - 0.3); await c.move(...D(1552, 822), 800);           // the budget ruler
      await c.at(T('ours,') - 0.3); await c.page.evaluate(() => __warp(2.5)); await c.hash('train'); await c.move(...D(650, 293), 900); // bound 7.1%
      await c.at(T('stand-ins') - 0.3); await c.move(...D(1040, 520), 800);        // certified seeds
      await c.at(T('budget.') - 0.4); await c.move(...D(1505, 185), 700);          // alive 11
    },
  },
  {
    name: 'live1', from: 'budget', to: 'replay', url: LIVE + '#flows', start: [1180, 500],
    run: async (c) => {
      const g0 = shot('budget').start;
      await c.click(921, 60, T('dreams') - g0 - 0.05);                             // dreams tab
      await c.at(T('adjusting') - 0.1); await c.drag(1402, 207, 1640, 1300);        // fewer mistakes -> more
      await c.at(T('coverage') + 0.2); await c.drag(1640, 207, 1300, 1200);         // and back down
      await c.click(998, 60, T('compliance,') - g0);                                // replay tab
      await c.at(T('temperature') - 0.3); await c.move(1268, 254, 700);             // big model rows
      await c.at(T('different', 63) - 0.2); await c.move(1000, 392, 600);              // recorded answer
      await c.at(T('machines') - 0.2); await c.move(1268, 461, 600);
      await c.click(425, 652, T('hash') - g0 - 0.1);                                // CON EDISON *EMEA, 3 of 4 replayable
      await c.at(T('hashes') - 0.4); const p = await c.box('#pinned'); await c.move(p[0] - 120, p[1], 800); // pinned: weights by hash
      await c.click(1580, 170, T('reproduce') - g0 - 0.2);                          // replay this run
      await c.at(T('byte-identical') - 0.2); await c.move(1100, 470, 800);
      await c.at(T('replay', 78) - 0.2); await c.move(1268, 600, 700);
    },
  },
  {
    name: 'build', from: 'build', to: 'buildrace', url: DECK + '#build', css: DECK_FIT, start: [1300, 900],
    run: async (c) => {
      const g0 = shot('build').start;
      await c.click(...D(626, 903), T('incredibly') - g0 - 0.2);                    // measured chips make the chain longer
      await c.click(...D(305, 903), T('sequentially') - g0 - 0.1, 450);
      await c.click(...D(458, 903), T('chained') - g0, 450);
      await c.click(...D(980, 228), T('parallel') - g0 - 0.3);                      // Run it
      await c.at(T('predictions.')); await c.move(...D(1500, 588), 900);           // Dreaming lane
      await c.at(T('500') - 0.3); await c.move(...D(1735, 588), 600);              // done in ms
    },
  },
  {
    name: 'live2', from: 'journal', to: 'scores', url: LIVE + '#scores', warm: '#dreams', start: [700, 820],
    run: async (c) => {
      const g0 = shot('journal').start;
      await c.at(T('manage') - 0.3); await c.move(440, 232, 700);
      await c.at(T('data', 93) - 0.1); await c.move(440, 345, 500);
      await c.at(T('model', 95) - 0.2); await c.move(440, 458, 500);
      await c.click(1075, 60, T('scores') - g0 - 0.2);                              // scores tab
      await c.at(T('categorized,') - 0.4); await c.move(360, 410, 800);             // who
      await c.at(T('performed,') - 0.3); await c.move(700, 410, 700);               // repeat traffic
      await c.at(T('costs') - 0.3); await c.move(1370, 425, 800);                   // $ per 1,000 calls
      await c.at(T('latency', 100) - 0.3); await c.move(1590, 425, 600);            // p50 latency
      await c.at(T('node') - 0.3); await c.move(330, 522, 900);                     // a candidate per node
      await c.at(T('not.') - 0.3); await c.move(700, 300, 800);                     // the verdict line
    },
  },
  {
    name: 'close', from: 'close', to: 'close', url: DECK + '#close', css: DECK_FIT, start: [1200, 760],
    run: async (c) => {
      await c.page.keyboard.press('r');
      await c.at(T('policies,') - 0.6); const [x, y] = D(1422, 398); await c.drag(x, y, D(1700, 398)[0], 1500);
      await c.at(T('self-hosted') + 0.1); await c.move(...D(1590, 835), 800);       // live right now
    },
  },
  { name: 'outro', from: 'outro', to: 'outro', url: LOGO + '#outro', cursor: false, run: async (c) => { await c.page.evaluate(() => play()); } },
];

async function capture(browser, cap) {
  const g0 = shot(cap.from).start, g1 = shot(cap.to).end, dur = g1 - g0;
  const ctx = await browser.newContext({ viewport: { width: 1920, height: 1080 } });
  const page = await ctx.newPage();
  const fake = (r) => (r.request().method() === 'POST' ? r.fulfill({ status: 200, body: '{}', contentType: 'application/json' }) : r.continue());
  await page.route('**/batch**', fake);
  await page.route('**/budget**', fake);
  await page.addInitScript(OVERLAY);
  await page.addInitScript(WARP);
  await page.goto(cap.url, { waitUntil: 'load' }).catch(() => {});
  if (cap.css) await page.addStyleTag({ content: cap.css });
  await page.evaluate(() => document.fonts.ready);
  await sleep(cap.url.startsWith(LIVE) ? 6000 : 1500);
  if (cap.warm) { await page.evaluate((h) => (location.hash = h), cap.warm); await sleep(2500); }
  await page.evaluate(([s, show]) => { const [x, y] = s || [1500, 1000]; __ov.set(x, y); __ov.hide(!show); }, [cap.start, cap.cursor !== false]);

  const cdp = await ctx.newCDPSession(page);
  const frames = [];
  cdp.on('Page.screencastFrame', (f) => { frames.push({ t: f.metadata.timestamp, d: f.data }); cdp.send('Page.screencastFrameAck', { sessionId: f.sessionId }).catch(() => {}); });
  await cdp.send('Page.startScreencast', { format: 'jpeg', quality: 92, maxWidth: 1920, maxHeight: 1080, everyNthFrame: 1 });
  await sleep(400);
  const t0 = Date.now();
  await page.evaluate(([g, cues]) => __ov.start(g, cues), [g0, TL.cues]);
  const c = {
    page,
    at: (g) => sleep((g - g0) * 1000 - (Date.now() - t0)),
    hash: (h) => page.evaluate((h) => (location.hash = h), h),
    move: async (x, y, ms = 700) => { await page.evaluate(([x, y, ms]) => __ov.move(x, y, ms), [x, y, ms]); await page.mouse.move(x, y); },
    box: async (sel) => { const b = await page.locator(sel).first().boundingBox(); return [b.x + b.width / 2, b.y + b.height / 2]; },
    // move so the press lands at capture-relative second `when`
    click: async (x, y, when, ms = 650) => { await c.at(g0 + when - ms / 1000); await c.move(x, y, ms); await page.evaluate(() => __ov.press()); await page.mouse.click(x, y); },
    drag: async (x0, y, x1, ms) => {
      await c.move(x0, y, 450); await page.evaluate(() => __ov.press()); await page.mouse.down();
      const n = Math.round(ms / 33);
      for (let k = 1; k <= n; k++) {
        const e = k / n < 0.5 ? 4 * (k / n) ** 3 : 1 - Math.pow(-2 * (k / n) + 2, 3) / 2, x = x0 + (x1 - x0) * e;
        await Promise.all([page.mouse.move(x, y), page.evaluate(([x, y]) => __ov.set(x, y), [x, y])]);
        await sleep(25);
      }
      await page.mouse.up();
    },
  };
  await cap.run(c);
  await sleep(dur * 1000 - (Date.now() - t0) + 300);
  await cdp.send('Page.stopScreencast');
  await ctx.close();

  // frames -> CFR clip of exactly `dur` seconds
  const dir = path.join(WORK, cap.name);
  fs.rmSync(dir, { recursive: true, force: true }); fs.mkdirSync(dir, { recursive: true });
  const rel = frames.map((f) => ({ ...f, t: f.t - t0 / 1000 }));
  let first = rel.findIndex((f) => f.t > 0); if (first < 0) first = rel.length;
  const use = rel.slice(Math.max(0, first - 1)).filter((f) => f.t < dur);
  if (!use.length) throw new Error(`${cap.name}: no frames`);
  use[0].t = 0;
  let list = '';
  use.forEach((f, i) => {
    const file = `f${String(i).padStart(5, '0')}.jpg`;
    fs.writeFileSync(path.join(dir, file), Buffer.from(f.d, 'base64'));
    const next = i + 1 < use.length ? use[i + 1].t : dur;
    list += `file '${file}'\nduration ${(next - f.t).toFixed(4)}\n`;
  });
  list += `file 'f${String(use.length - 1).padStart(5, '0')}.jpg'\n`;
  fs.writeFileSync(path.join(dir, 'list.txt'), list);
  execFileSync(FFMPEG, ['-y', '-hide_banner', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', 'list.txt',
    '-vf', `fps=${FPS},scale=1920:1080:flags=lanczos,format=yuv420p`, '-t', dur.toFixed(3), '-c:v', 'libx264', '-crf', '14',
    '-preset', 'fast', '-r', String(FPS), path.join(WORK, cap.name + '.mp4')], { cwd: dir });
  const fps = (use.length / dur).toFixed(1);
  console.log(`${cap.name}: ${dur.toFixed(2)} s, ${use.length} frames (${fps} fps avg)`);
}

function assemble() {
  const list = CAPTURES.map((c) => `file '${c.name}.mp4'`).join('\n');
  fs.writeFileSync(path.join(WORK, 'shots.txt'), list);
  execFileSync(FFMPEG, ['-y', '-hide_banner', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', 'shots.txt',
    '-i', path.join(HERE, 'voice.m4a'), '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-crf', '18', '-preset', 'medium',
    '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-t', TL.total.toFixed(3), '-movflags', '+faststart',
    path.join(HERE, 'dreaming-demo.mp4')], { cwd: WORK });
  console.log('wrote dreaming-demo.mp4');
}

(async () => {
  fs.mkdirSync(WORK, { recursive: true });
  const only = process.argv.slice(2).filter((a) => a !== 'assemble');
  const browser = await chromium.launch({ args: ['--force-color-profile=srgb', '--hide-scrollbars'] });
  for (const cap of CAPTURES) if (!only.length || only.includes(cap.name)) await capture(browser, cap);
  await browser.close();
  if (!only.length || process.argv.includes('assemble')) assemble();
})().catch((e) => { console.error(e); process.exit(1); });
