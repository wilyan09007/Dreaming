// Record every shot of the demo video in real time: page + animated cursor + subtitle card, CDP screencast frames.
// Usage: node record.mjs [onlyShotIndex]   -> frames/NNNNN.jpg + frames.json (global times)
import { createRequire } from 'node:module';
import fs from 'node:fs';
const require = createRequire('C:/Users/aiden/.claude/skills/gstack/node_modules/');
const { chromium } = require('playwright-core');

const TL = JSON.parse(fs.readFileSync('timeline.json', 'utf8'));
const DECK = 'file:///C:/Users/aiden/Desktop/personalprojects/Dreaming-final/dreaming/apps/web/launch/index.html';
const LIVE = 'https://dreaming-demo.fly.dev/';
const LOGO = 'file:///' + process.cwd().replace(/\\/g, '/') + '/logo.html';
const P = (n) => TL.phrases[n - 1];
const W = (n, text) => { const w = P(n).words.find((x) => x.w.toLowerCase().includes(text.toLowerCase())); if (!w) throw new Error(`no "${text}" in ${n}`); return w.t; };
const mid = (a, b) => (P(a).end + P(b).start) / 2;

// subtitle cues: chunks of <= 8 words / 46 chars, broken after punctuation when possible
const glue = (ws) => ws.reduce((s, w) => (s && !/^[%.,'\-]/.test(w.w) ? s + ' ' : s) + w.w, '');
const cues = [];
for (const p of TL.phrases) {
  let cur = [];
  const flush = () => { if (cur.length) cues.push({ t: cur[0].t - 0.06, text: glue(cur) }); cur = []; };
  for (const w of p.words) {
    cur.push(w);
    const txt = glue(cur);
    if (cur.length >= 8 || txt.length >= 46 || (/[,.]$/.test(w.w) && cur.length >= 4)) flush();
  }
  flush();
  cues.push({ t: p.end + 0.25, text: '' });
}
for (let i = 0; i + 1 < cues.length; i++) if (cues[i].text === '' && cues[i + 1].t < cues[i].t) cues[i].t = cues[i + 1].t - 0.001;

const OVERLAY = String.raw`(() => {
  const css = document.createElement('style');
  css.textContent = '#__ov{position:fixed;inset:0;pointer-events:none;z-index:2147483647}' +
    '#__cur{position:absolute;left:0;top:0;width:38px;height:44px;transform:translate(-200px,-200px);filter:drop-shadow(2px 3px 2px rgba(43,43,51,.28))}' +
    '.__rip{position:absolute;width:20px;height:20px;margin:-10px 0 0 -10px;border-radius:50%;border:3px solid #6A4FD8;animation:__rip .55s ease-out forwards}' +
    '@keyframes __rip{from{transform:scale(.4);opacity:1}to{transform:scale(3.4);opacity:0}}' +
    '#__sub{position:absolute;left:50%;bottom:46px;transform:translateX(-50%);max-width:1560px;white-space:nowrap;background:#FFFEFB;border:2.5px solid #2B2B33;border-radius:16px;box-shadow:4px 5px 0 rgba(43,43,51,.16);padding:12px 26px 14px;font:500 38px/1.2 "Bricolage Grotesque",system-ui,sans-serif;color:#2B2B33;opacity:0;transition:opacity .12s}' +
    '#__sub b{color:#6A4FD8;font-weight:700}';
  document.head.appendChild(css);
  const ov = document.createElement('div'); ov.id = '__ov';
  ov.innerHTML = '<div id="__sub"></div><svg id="__cur" viewBox="0 0 38 44"><path d="M4 3 L4 35 L12.5 27.5 L18.5 41 L24 38.5 L18 25.5 L29.5 25 Z" fill="#2B2B33" stroke="#FFFEFB" stroke-width="2.6" stroke-linejoin="round"/></svg>';
  document.body.appendChild(ov);
  const cur = ov.querySelector('#__cur'), sub = ov.querySelector('#__sub');
  let x = -200, y = -200, anim = null, t0 = 0, base = 0, cues = [], shown = -2;
  const place = () => { cur.style.transform = 'translate(' + (x - 4) + 'px,' + (y - 3) + 'px)'; };
  const ease = (k) => k < .5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
  const fmt = (s) => s.replace(/(\S*\d\S*|Psi[.,]?|Dreaming[.,]?)/g, '<b>$1</b>');
  function frame() {
    const now = performance.now();
    if (anim) { const k = Math.min(1, (now - anim.s) / anim.d), e = ease(k);
      x = anim.x0 + (anim.x1 - anim.x0) * e; y = anim.y0 + (anim.y1 - anim.y0) * e - Math.sin(Math.PI * e) * anim.arc; if (k >= 1) anim = null; }
    place();
    if (t0) { const g = base + (now - t0) / 1000; let i = -1; for (let j = 0; j < cues.length; j++) if (cues[j].t <= g) i = j;
      if (i !== shown) { shown = i; const t = i >= 0 ? cues[i].text : ''; if (t) sub.innerHTML = fmt(t); sub.style.opacity = t ? 1 : 0; } }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  window.__cur = {
    at(nx, ny) { x = nx; y = ny; anim = null; place(); },
    move(nx, ny, d) { const dist = Math.hypot(nx - x, ny - y); anim = { s: performance.now(), d, x0: x, y0: y, x1: nx, y1: ny, arc: Math.min(60, dist * 0.08) }; },
    click() { const r = document.createElement('div'); r.className = '__rip'; r.style.left = x + 'px'; r.style.top = y + 'px'; ov.appendChild(r); setTimeout(() => r.remove(), 600);
      cur.animate([{ scale: 1 }, { scale: .82 }, { scale: 1 }], { duration: 220 }); },
    start(b, c, hide) { base = b; cues = c; t0 = performance.now(); if (hide) cur.style.display = 'none'; return Date.now(); },
  };
  window.__box = (q) => {
    let best = null;
    const els = q.startsWith('css:') ? [...document.querySelectorAll(q.slice(4))] : [...document.querySelectorAll('body *')].filter((e) => e.textContent.includes(q));
    for (const e of els) { const r = e.getBoundingClientRect(); if (r.width < 2 || r.height < 2 || r.bottom < 0 || r.top > innerHeight) continue;
      if (e.closest('#__ov')) continue; if (e.checkVisibility && !e.checkVisibility({ visibilityProperty: true, opacityProperty: true })) continue;
      if (!best || r.width * r.height < best.a) best = { a: r.width * r.height, x: r.left + r.width / 2, y: r.top + r.height / 2, w: r.width, h: r.height }; }
    return best;
  };
})()`;

// ---- shots: [start, end, url, prep(page), choreography(c)] ----
const S = [];
const shot = (end, url, run, opts = {}) => S.push({ end, url, run, ...opts });
const deck = (h) => DECK + '#' + h;
shot(P(1).start - 0.08, LOGO + '#intro', async (c) => { await c.page.evaluate(() => window.__go()); await c.until(99); }, { hideCursor: true });
shot(mid(1, 2), deck('hook'), async (c) => {
  await c.key('r'); c.at(1500, 760);
  await c.until(W(1, 'agents')); await c.move('Your agent calls', 900);
  await c.until(W(1, 'every')); await c.move('every step', 700);
  await c.until(W(1, 'step') + 0.6); await c.move('cost', 900);
});
shot(mid(2, 3), deck('race'), async (c) => {
  await c.key('r'); c.at(900, 980);
  await c.until(W(2, 'milliseconds') - 0.4); await c.move('done in', 800);
  await c.until(W(2, 'workflows') - 0.6); await c.move('1,000 invoices', 900);
});
shot(mid(3, 4), deck('bench'), async (c) => {
  await c.key('r'); c.at(800, 1000);
  await c.until(W(3, '87') - 0.7); await c.move('cheaper, best step', 700);
  await c.until(W(3, 'latency') - 0.3); await c.move('faster per call', 800);
  await c.until(W(3, 'milliseconds') - 0.3); await c.move('Log scale', 700);
  await c.until(W(3, 'Psi') - 0.6); await c.move('served, 0 LLM tokens', 800);
});
shot(mid(4, 5), LIVE + '#flows', async (c) => {
  c.at(1250, 520);
  await c.move('run 2,000 more', 1100); await c.until(W(4, 'running')); await c.click('run 2,000 more');
}, { live: true });
shot(W(6, 'ours') - 0.25, deck('psi'), async (c) => {
  await c.key('r'); c.at(300, 900);
  await c.until(W(5, 'Psi') - 0.2); await c.move('softmax attention', 900);
  await c.until(W(5, 'memory') - 0.6); await c.move('weighted vote', 900);
  await c.until(W(6, 'error') - 0.2); await c.move('Clopper-Pearson', 900);
}, { psiThenBench: true });
shot(mid(6, 7), deck('train'), async (c) => {
  await c.key('r'); c.at(1200, 1000);
  await c.until(W(6, 'statistical') - 0.3); await c.move('pruned for', 900);
  await c.until(W(6, 'take') - 0.4); await c.move('alive', 800);
  await c.until(W(6, 'budget') - 0.2); await c.move('categorize', 800);
});
shot(mid(7, 8), LIVE + '#flows', async (c) => {
  c.at(1300, 600);
  await c.until(W(7, 'Dreams') - 0.9); await c.move('css:a.tab.t-dreams', 800); await c.click('css:a.tab.t-dreams');
  await c.wait(900);
  await c.until(W(7, 'mistakes') - 0.3); await c.drag('css:#budget', [-0.25, 0.3, 0], 2600);
  await c.move('answers 28% of calls', 700);
}, { live: true });
shot(mid(11, 12), LIVE + '#flows', async (c) => {
  c.at(1300, 600);
  await c.until(W(8, 'compliance') - 0.6); await c.move('css:a.tab.t-replay', 800); await c.click('css:a.tab.t-replay');
  await c.until(W(9, 'temperature') - 0.4); await c.move('recorded answer', 900);
  await c.until(W(9, 'machines') - 0.6); await c.move('big model', 800);
  await c.until(W(10, 'hash') - 0.9); await c.move('CON EDISON *EMEA', 900); await c.click('CON EDISON *EMEA');
  await c.wait(1200); await c.move('pinned for replay', 900);
  await c.until(W(11, 'reproduce') - 0.5); await c.move('decisions can re-run', 900);
  await c.until(W(11, 'replay') - 0.9); await c.move('replay this run', 800); await c.click('replay this run');
}, { live: true });
shot(mid(12, 13) + 3.5, deck('build'), async (c) => {
  await c.key('r'); c.at(900, 900);
  await c.until(W(12, 'long') - 0.4);
  for (const chip of ['summarize email', 'wake a human', 'summarize email', 'wake a human']) {
    await c.move(chip, 500); await c.click(chip); await c.wait(250);
  }
  await c.until(W(12, 'parallel') - 0.9); await c.move('Run it', 800); await c.click('Run it');
  await c.until(W(13, 'latency')); await c.move('done in', 900);
});
shot(mid(14, 15), LIVE + '#dreams', async (c) => {
  c.at(1500, 700); await c.move('vendor lookup', 900); await c.wait(300); await c.move('match invoice', 700);
}, { live: true });
shot(mid(15, 16), LIVE + '#dreams', async (c) => {
  c.at(900, 560);
  await c.until(W(15, 'scores') - 0.8); await c.move('css:a.tab.t-scores', 700); await c.click('css:a.tab.t-scores');
  await c.wait(1500); await c.move('categorize', 900);
  await c.until(W(15, 'costs') - 0.4); await c.move('$', 900);
  await c.until(W(15, 'latency') - 0.3); await c.move('ms', 800);
  await c.until(W(15, 'node') - 0.4); await c.move('match invoice', 900);
}, { live: true, preload: '#scores' });
shot(P(17).start - 0.15, deck('close'), async (c) => {
  await c.key('r'); c.at(1300, 950);
  await c.wait(2800); await c.drag('css:#s-close input[type=range]', [0.25, -0.1], 2000);
});
shot(TL.total, LOGO + '#outro', async (c) => { await c.page.evaluate(() => window.__go()); await c.until(999); }, { hideCursor: true });

// ---- runner ----
const only = process.argv[2] !== undefined ? +process.argv[2] : null;
fs.mkdirSync('frames', { recursive: true });
const b = await chromium.launch({ headless: true, args: ['--force-color-profile=srgb', '--hide-scrollbars'] });
const ctx = await b.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
await ctx.route('**/batch**', (r) => r.request().method() === 'POST' ? r.fulfill({ status: 200, contentType: 'application/json', body: '{}' }) : r.continue());
await ctx.route('**/budget', (r) => r.fulfill({ status: 200, contentType: 'application/json', body: '{}' }));
const index = only === null ? [] : JSON.parse(fs.existsSync('frames.json') ? fs.readFileSync('frames.json', 'utf8') : '[]');
let start = 0;
for (let k = 0; k < S.length; k++) {
  const s = S[k]; s.start = start; start = s.end;
  if (only !== null && k !== only) continue;
  for (let i = index.length - 1; i >= 0; i--) if (index[i].shot === k) index.splice(i, 1);
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log(`shot ${k} pageerror`, String(e).slice(0, 160)));
  if (s.preload) { await page.goto(LIVE + s.preload); await page.waitForFunction(() => !document.body.innerText.includes('reading the scores'), null, { timeout: 30000 }).catch(() => console.log('scores never loaded')); }
  await page.goto(s.url); await page.waitForLoadState('networkidle', { timeout: 20000 }).catch(() => {});
  await page.evaluate(() => document.fonts.ready); await page.waitForTimeout(s.live ? 2500 : 600);
  await page.evaluate(OVERLAY);
  const cdp = await ctx.newCDPSession(page);
  const frames = [];
  cdp.on('Page.screencastFrame', (f) => { frames.push({ ts: f.metadata.timestamp, data: f.data }); cdp.send('Page.screencastFrameAck', { sessionId: f.sessionId }).catch(() => {}); });
  await cdp.send('Page.startScreencast', { format: 'jpeg', quality: 92, maxWidth: 1920, maxHeight: 1080, everyNthFrame: 1 });
  await page.waitForTimeout(150);
  const wall0 = await page.evaluate(([b0, c, h]) => window.__cur.start(b0, c, h), [s.start, cues, !!s.hideCursor]);
  const dur = s.end - s.start;
  const rel = (T) => (T - s.start) * 1000 - (Date.now() - wall0);
  const pt = async (q) => (typeof q === 'object' ? q : (await page.evaluate((q) => window.__box(q), q)) || (console.log(`shot ${k}: no box for ${q}`), null));
  let pos = { x: 0, y: 0 };
  const c = {
    page,
    until: async (T) => { const ms = Math.min(rel(T), rel(s.end)); if (ms > 0) await page.waitForTimeout(ms); },
    wait: (ms) => page.waitForTimeout(Math.max(0, Math.min(ms, rel(s.end)))),
    at: (x, y) => { pos = { x, y }; return page.evaluate(([x, y]) => window.__cur.at(x, y), [x, y]); },
    move: async (q, ms) => { const p = await pt(q); if (!p) return; pos = p; await page.evaluate(([x, y, d]) => window.__cur.move(x, y, d), [p.x, p.y, ms]); await c.wait(ms); },
    click: async (q) => { const p = (await pt(q)) || pos; await page.evaluate(() => window.__cur.click()); await page.mouse.click(p.x, p.y).catch(() => {}); },
    key: (k) => page.keyboard.press(k),
    drag: async (q, fracs, ms) => { // fracs: offsets as fractions of the element width from its current thumb (centre start)
      const p = await pt(q); if (!p) return;
      const v = await page.evaluate((q) => { const e = document.querySelector(q.slice(4)); return (e.value - e.min) / (e.max - e.min); }, q);
      const x0 = p.x - p.w / 2 + p.w * v; await c.move({ x: x0, y: p.y }, 600); await page.mouse.move(x0, p.y); await page.mouse.down();
      const step = ms / fracs.length; let last = x0;
      for (const f of fracs) { const nx = x0 + f * p.w; const n = 12; for (let i = 1; i <= n; i++) { const xx = last + (nx - last) * i / n;
        await page.evaluate(([x, y]) => window.__cur.at(x, y), [xx, p.y]); await page.mouse.move(xx, p.y); await page.waitForTimeout(step / n); } last = nx; }
      await page.mouse.up();
    },
  };
  try { await s.run(c); } catch (e) { console.log(`shot ${k} choreography error`, String(e).slice(0, 200)); }
  await c.until(s.end); await page.waitForTimeout(120);
  await cdp.send('Page.stopScreencast');
  let pre = null; // frames captured before the shot clock started: keep only the newest, pinned to the shot start
  const keep = [];
  for (const f of frames) { const t = s.start + (f.ts * 1000 - wall0) / 1000; if (t < s.start) pre = f; else if (t <= s.end) keep.push([t, f]); }
  if (pre || !keep.length || keep[0][0] > s.start) keep.unshift([s.start, pre || keep[0]?.[1] || frames[0]]);
  for (const [t, f] of keep) {
    const name = `frames/${k.toString().padStart(2, '0')}_${(index.length).toString().padStart(5, '0')}.jpg`;
    fs.writeFileSync(name, Buffer.from(f.data, 'base64')); index.push({ shot: k, t, f: name });
  }
  console.log(`shot ${k} ${s.start.toFixed(2)}-${s.end.toFixed(2)} ${s.url.split('/').pop()} frames ${frames.length} (${(frames.length / dur).toFixed(1)} fps)`);
  await page.close();
}
index.sort((a, b) => a.shot - b.shot || a.t - b.t);
fs.writeFileSync('frames.json', JSON.stringify(index));
await b.close();
