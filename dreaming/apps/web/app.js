// @ts-check
// The designed page (demo/index.html; looks, copy and pacing per demo/VISUAL_INTENT.md) on the API's data: README.md (a)-(g).
// Every number is a field of an API reply. The page formats, paces and animates; the only arithmetic is unit conversion and the
// "−N% cost" note (1 − cost_dream_usd / cost_awake_usd of one site). The simulation's model (SITES, phase(), cover(), metrics()) is gone.
/** @typedef {import('./schema').State} State */
/** @typedef {import('./schema').SiteView} SiteView */
/** @typedef {import('./schema').Certificate} Certificate */
/** @typedef {import('./schema').Metrics} Metrics */
/** @typedef {import('./schema').Run} Run */
/** @typedef {import('./schema').RunView} RunView */
/** @typedef {import('./schema').TraceRow} TraceRow */
/** @typedef {import('./schema').ReplayReport} ReplayReport */
/** @typedef {import('./schema').Health} Health */
/** @typedef {import('./schema').Event} DEvent */
/** @typedef {import('./live.js').Live} Live */
/** @typedef {State & {spare_certificates?: Record<string, Certificate>}} PageState  main.py's PageState: State plus the spares' certificates */
/** @typedef {{p: string, raw: string, cant: boolean, counted: number, cov: number, b: number, c: Certificate | null}} Look  what one site shows */
/** @typedef {{name: string, ans: string, by: string, rec?: boolean, i?: number, ev?: string}} Step  one row of the trace card */

const apiBase = ''; // the API serves this page: same origin
const BATCH = 2000; // what the run button sends: POST /batch {n: 2000}
const HOLD_PHASE = 900; // README (g): each sticker stays at least 0.9 s, however fast the server moved
const $ = (/** @type {string} */ s) => /** @type {HTMLElement} */ (document.querySelector(s));
const now = () => performance.now();

/** @type {PageState | null} */ let st = null;
/** @type {Health | null} */ let health = null;
const S = {
  view: 'flows', sel: 'categorize', selRun: null, rp: null, trace: null, loading: null, unplugged: false,
  narr: '', narrIdle: true, narrRank: 0, narrUntil: 0, queued: null, verdicts: [], last: null, quietUntil: 0, budgetAt: -1e9,
  note: null, prev: {}, pop: {}, shown: {}, hold: {}, recheck: {}, lastCert: {}, saved: 0, runs: 0, bnow: '',
  cmdAt: 0, stAt: 0, failQ: {}, failSaid: new Set(), // command reply time, request time of the /state on screen, train failures to judge
};

// ---------- the API: every request goes through req(); a refusal becomes one line of text, never a throw ----------
/** @param {'GET' | 'POST'} method @param {string} path @param {object} [body] @returns {Promise<{ok: boolean, data: any, text: string}>} */
async function req(method, path, body) {
  try {
    const r = await fetch(apiBase + path, { method, headers: body ? { 'content-type': 'application/json' } : {}, body: body && JSON.stringify(body) });
    const data = await r.json().catch(() => null);
    const why = typeof data?.detail === 'string' ? data.detail : JSON.stringify(data?.detail ?? '');
    return { ok: r.ok, data, text: r.ok ? '' : `${r.status} ${why}` };
  } catch (e) {
    return { ok: false, data: null, text: String(e) };
  }
}
/** A command, then /state again. /state reads sent before the reply predate it and are dropped (a newer read follows);
 * `apply` patches the page from the reply at once, so a slow /state never shows the old state.
 * @param {'GET' | 'POST'} method @param {string} path @param {object} [body] @param {(data: any) => void} [apply] */
async function cmd(method, path, body, apply) {
  S.cmdAt = Infinity;
  const r = await req(method, path, body);
  S.cmdAt = now();
  if (r.ok && apply) apply(r.data);
  await refresh();
  return r;
}
// /state, coalesced: one request in flight, at most one queued, 250 ms apart (a batch emits many events). Resolves on the newest.
/** @type {Promise<void> | null} */ let inflight = null;
let again = false;
function refresh() {
  if (inflight) { again = true; return inflight; }
  inflight = (async () => {
    try {
      do {
        again = false;
        const t0 = now(), r = await req('GET', '/state');
        problems.state = r.ok ? '' : `GET /state: ${r.text}`;
        if (r.ok && t0 >= S.cmdAt) { S.stAt = t0; loaded(r.data); }
        if (again) await new Promise((done) => setTimeout(done, 250));
      } while (again);
    } finally { inflight = null; strip(); }
  })();
  return inflight;
}
async function healthz() {
  const r = await req('GET', '/healthz');
  health = r.ok ? r.data : health;
  problems.health = r.ok ? Object.entries(health?.checks ?? {}).filter(([, v]) => !/^(ok|configured)\b/.test(v)).map(([k, v]) => `healthz ${k}: ${v}`) : [`GET /healthz: ${r.text}`];
  const p = health?.profile ?? {};
  setText($('#pinned'), `pinned for replay: weights by hash${p.device ? ` · ${p.device} · batch size ${p.batch} · ${p.threads} thread${p.threads === '1' ? '' : 's'}` : ''}`);
  strip();
}
// red health checks, component errors and a failed /state: one red line under the header (README (a))
const problems = { health: /** @type {string[]} */ ([]), events: /** @type {string[]} */ ([]), state: '' };
function strip() {
  const all = [...problems.health, ...problems.events, ...(problems.state ? [problems.state] : [])];
  $('#errs').hidden = !all.length;
  setText($('#errs'), all.join('\n'));
}

/** @param {PageState} s */
function loaded(s) {
  const first = !st;
  st = s;
  const vid = vendor()?.site.id;
  S.unplugged = !!vid && (s.unplugged ?? []).includes(vid); // the fault switch (README (d)): the pill survives a reload and other tabs
  if (first) {
    buildJournal();
    S.saved = s.metrics.saved_usd; S.runs = s.metrics.runs;
  }
}
// A page opened mid-outage or mid-batch says so, not the idle line (after /healthz: the batch line names the teacher).
function greet() {
  const v = vendor();
  if (!st || !v) return;
  if (S.unplugged) narrate(outageLine(v));
  else if (st.batch.running) narrate(st.batch.paused ? `paused at run ${int(st.metrics.runs)}` : `fast-forwarding ${int(st.batch.target)} transactions · ${teacherWords()}`);
}

// ---------- the sites: names are a page map keyed by site.name (README (d)) ----------
/** @type {Record<string, string>} */
const NAME = { categorize: 'categorize', vendor_lookup: 'vendor lookup', match_invoice: 'match invoice', anything_odd: 'anything odd?', post_to_books: 'post to books', wake_a_human: 'wake a human' };
const short = (/** @type {string} */ id) => id.split('.').pop() || id;
const nameOf = (/** @type {string} */ id) => NAME[short(id)] ?? short(id).replace(/_/g, ' ');
/** @returns {SiteView[]} the four sites that can dream, in workflow order */
const sites = () => (st ? st.sites.filter((v) => v.site.kind !== 'effect') : []);
const byName = (/** @type {string} */ n) => sites().find((v) => v.site.name === n);
const vendor = () => byName('vendor_lookup');
// README (e): the answering side has one name everywhere, the one the trace card reads off each row's teacher.provider
const teacherOf = (/** @type {SiteView} */ v) => ({ tool: 'vendor API', rule: 'rule teacher' })[v.site.teacher.provider] ?? 'big model';
// the offline rule teacher bills a flat stand-in per answer (offline.RuleTeacher), so every ¢ and s on the page is its constant times a count
const ruled = () => sites().some((v) => v.site.teacher.provider === 'rule');
// a tool is down while its fault switch is off (the pill: State.unplugged) or after the runtime saw it fail (Policy.outage)
const downOf = (/** @type {SiteView} */ v) => v.site.kind === 'tool' && (S.unplugged || v.policy.outage);
const outageLine = (/** @type {SiteView} */ v) => (phase(v).p === 'dreaming' ? 'vendor API is down · the dream answers, the last snapshot covers the rest' : 'vendor API is down · nothing has learned it yet, so the last snapshot answers');
const lastRun = () => (st && st.runs.length ? st.runs[st.runs.length - 1] : null);
const stuckNow = () => S.unplugged && !!lastRun()?.stuck_at; // runs are stuck only when the newest one says so

// ---------- formatting ----------
const int = (/** @type {number} */ n) => Math.round(n).toLocaleString('en-US');
const cents = (/** @type {number} */ usd) => { const c = usd * 100; return (c >= 1 ? c.toFixed(1) : c.toFixed(2)) + '¢'; };
const pct = (/** @type {number} */ x) => (x < 1 ? +x.toFixed(2) : +x.toFixed(1)) + '%'; // x already in percent
const pf = (/** @type {number} */ f) => pct(f * 100); // a fraction, as percent
const money = (/** @type {number} */ d) => '$' + d.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const esc = (/** @type {unknown} */ s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const setText = (/** @type {HTMLElement} */ el, /** @type {string} */ t) => { if (el.textContent !== t) el.textContent = t; };
const setHTML = (/** @type {any} */ el, /** @type {string} */ h) => { if (el._h !== h) { el._h = h; el.innerHTML = h; } };
const setAttr = (/** @type {Element} */ el, /** @type {string} */ k, /** @type {string} */ v) => { if (el.getAttribute(k) !== v) el.setAttribute(k, v); };

// ---------- icons ----------
const R = 'style="filter:url(#rough)"';
/** @type {Record<string, string>} */
const PATHS = {
  feed: '<path d="M14 3.5l1.9 3.9 4.3.6-3.1 3 .7 4.3-3.8-2-3.8 2 .7-4.3-3.1-3 4.3-.6z" fill="#FFE58F"/><path d="M3 21l5-5M3 16.5l2.5-2.5M7.5 21l2.5-2.5"/>',
  tag: '<path d="M3.5 12.5V4h8.5l8.5 8.5-8.5 8.5z" fill="#fff"/><circle cx="8" cy="8.5" r="1.4"/>',
  store: '<path d="M5 11v9h14v-9" fill="#fff"/><path d="M3.5 6.5h17l-1.2 4.5H4.7z" fill="#FFB3AE"/><path d="M10 20v-5h4v5"/>',
  invoice: '<path d="M6 3h9l4 4v14H6z" fill="#fff"/><path d="M15 3v4h4M9 14l2 2 4-4"/>',
  branch: '<path d="M12 3v18M12 5.5h6l2.2 2.25L18 10h-6M12 12.5H6l-2.2 2.25L6 17h6" fill="#fff"/>',
  book: '<path d="M12 5C9.7 3.2 6.1 2.7 2 3.2v13c4.1-.5 7.7 0 10 1.8 2.3-1.8 5.9-2.3 10-1.8v-13c-4.1-.5-7.7 0-10 1.8z" fill="#fff"/><path d="M12 5v13M5 7h4.5M5 10h4.5M14.5 7H19M14.5 10H18"/>',
  clock: '<circle cx="12" cy="13" r="7" fill="#fff"/><path d="M12 9.5V13l2.5 1.8M4.5 5.5l2.5-2M19.5 5.5l-2.5-2"/>',
  plug: '<path d="M9 3v4M15 3v4"/><path d="M6.5 7h11v3.5a5.5 5.5 0 0 1-11 0z" fill="#fff"/><path d="M12 16v5"/>',
  unplug: '<g transform="rotate(-24 12 12)"><path d="M9 5v4M15 5v4"/><path d="M6.5 9h11v3.5a5.5 5.5 0 0 1-11 0z" fill="#fff"/><path d="M12 18v4"/></g><path d="M2.5 6l2.2 1.2M2 11h2.6M5 2.5l1 2.2" stroke="#E5534B"/>',
  camera: '<rect x="3" y="7" width="18" height="12" rx="2" fill="#fff"/><path d="M8 7l1.5-2.5h5L16 7"/><circle cx="12" cy="13" r="3.2"/>',
  moon: '<circle cx="12" cy="12" r="8" fill="#FFF3A8"/><circle cx="9.5" cy="10" r="1.3"/><circle cx="14" cy="14.5" r="1.8"/>',
  sheep: '<path d="M5 15c-1.6 0-2.5-1.9-1.4-3.1-.6-1.6.9-3.1 2.5-2.6.6-1.5 2.7-1.8 3.7-.6 1-1.2 3-1 3.6.4 1.6-.3 2.9 1.2 2.3 2.7 1 1.2.1 3.2-1.5 3.2z" fill="#CDBEFF"/><ellipse cx="18.4" cy="11" rx="2.4" ry="2.8" fill="#2B2B33"/><path d="M7 15v4M12 15v4"/>',
  check: '<path d="M4 12.5l5 5L20 6" stroke="#1E7A55" stroke-width="2.6"/>',
  play: '<path d="M7 4.5v15l13-7.5z" fill="#2B2B33" stroke="none"/>',
  pause: '<path d="M6.5 5h4v14h-4zM13.5 5h4v14h-4z" fill="#2B2B33" stroke="none"/>',
};
// the wobbly purple "→" between an old cost and a new one (dream journal, scores)
const WOB = `<svg width="30" height="16" viewBox="0 0 34 16" ${R} aria-hidden="true"><path d="M2 9 C10 3 20 13 30 7 M24 3 L31 7 L25 12" fill="none" stroke="#6A4FD8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
const ico = (/** @type {string} */ k, s = 24) => `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" stroke="#2B2B33" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" ${R} aria-hidden="true">${PATHS[k]}</svg>`;
/** @type {Record<string, (s: number) => string>} */
const STATUS = {
  awake: (s) => `<svg width="${s}" height="${s}" viewBox="0 0 24 24" ${R} aria-hidden="true"><circle cx="12" cy="12" r="5" fill="#FFD166" stroke="#2B2B33" stroke-width="1.3"/><path d="M12 2.5v2.5M12 19v2.5M2.5 12H5M19 12h2.5M5.3 5.3l1.8 1.8M16.9 16.9l1.8 1.8M5.3 18.7l1.8-1.8M16.9 7.1l1.8-1.8" stroke="#2B2B33" stroke-width="1.3" stroke-linecap="round"/></svg>`,
  watching: (s) => `<svg width="${s}" height="${s}" viewBox="0 0 24 24" ${R} aria-hidden="true"><circle cx="12" cy="12" r="8" fill="none" stroke="#2B2B33" stroke-width="1.4" stroke-dasharray="2.4 2.4"/><path d="M8 12.2q4-3.6 8 0q-4 3.6-8 0z" fill="none" stroke="#2B2B33" stroke-width="1.3"/><circle cx="12" cy="12.2" r="1.2" fill="#2B2B33"/></svg>`,
  testing: (s) => `<svg width="${s}" height="${s}" viewBox="0 0 24 24" ${R} aria-hidden="true"><circle cx="12" cy="12" r="8" fill="#fff" stroke="#2B2B33" stroke-width="1.4"/><path d="M12 4a8 8 0 0 1 0 16z" fill="#7FB7FF" stroke="#2B2B33" stroke-width="1.4"/></svg>`,
  dreaming: (s) => `<svg width="${s}" height="${Math.round(s * 46 / 58)}" viewBox="0 0 58 46" ${R} aria-hidden="true"><path d="M12 38h30a9 9 0 0 0 1.5-17.9A13 13 0 0 0 18.6 17 10 10 0 0 0 12 38z" fill="#E2D9FF" stroke="#2B2B33" stroke-width="2.4" stroke-linejoin="round"/><path d="M20 29q2.5 2.4 5 0M30 29q2.5 2.4 5 0" fill="none" stroke="#2B2B33" stroke-width="2.2" stroke-linecap="round"/><path class="bob" d="M44 4h6l-6 7h6" fill="none" stroke="#6A4FD8" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
  down: (s) => `<svg width="${s}" height="${s}" viewBox="0 0 24 24" aria-hidden="true"><path d="M6.5 14h11a3.6 3.6 0 0 0 .4-7.2A5.3 5.3 0 0 0 7.6 6 3.9 3.9 0 0 0 6.5 14z" fill="#B7BDD3" stroke="#2B2B33" stroke-width="1.1"/><g stroke="#3D63D6" stroke-width="1.6" stroke-linecap="round"><path class="drop" d="M8.5 16.5l-1 2.5"/><path class="drop d2" d="M12.5 16.5l-1 2.5"/><path class="drop d3" d="M16.5 16.5l-1 2.5"/></g></svg>`,
};
const sheepSVG = (/** @type {boolean} */ full) => `<svg viewBox="0 0 24 20" aria-hidden="true"><path d="M5 14c-1.6 0-2.5-1.9-1.4-3.1-.6-1.6.9-3.1 2.5-2.6.6-1.5 2.7-1.8 3.7-.6 1-1.2 3-1 3.6.4 1.6-.3 2.9 1.2 2.3 2.7 1 1.2.1 3.2-1.5 3.2z" ${full ? 'fill="#B39DFF" stroke="#2B2B33"' : 'fill="none" stroke="#C9C5D3" stroke-dasharray="2 2"'} stroke-width="1.2"/>${full ? '<ellipse cx="18.4" cy="10" rx="2.4" ry="2.8" fill="#2B2B33"/>' : ''}</svg>`;

// ---------- what a site shows ----------
// The held sticker (README (g)): site.phase events queue per site and each phase stays up HOLD_PHASE ms. A change that arrives
// only through /state (a reply caught between two writes of one tick) is held the same way, so no sticker flashes.
/** @param {string} name @param {string} phase  the Policy's phase now @returns {string | null} */
function held(name, phase) {
  const t = now(), was = S.shown[name];
  if (!S.hold[name] && was && was !== phase && !S.recheck[name]) S.hold[name] = { q: was === 'watching' && phase === 'dreaming' ? ['testing', phase] : [phase], p: was, until: 0 };
  const h = S.hold[name];
  if (!h) return null;
  if (t >= h.until && h.q.length) { h.p = h.q.shift(); h.until = t + HOLD_PHASE; }
  if (t < h.until || h.q.length) return h.p;
  delete S.hold[name];
  return null;
}
/** @param {DEvent} e */
function onPhase(e) {
  const name = short(e.call_site || ''), d = /** @type {any} */ (e.data);
  if (S.recheck[name]) return; // a budget re-check animates this row itself
  const h = S.hold[name] ??= { q: [], p: S.shown[name] ?? String(d.from), until: 0 };
  if ((h.q.length ? h.q[h.q.length - 1] : h.p) === String(d.from)) h.q.push(String(d.to)); // a transition the page already passed is dropped
}
/** @param {SiteView} v @returns {Look} */
function phase(v) {
  const p = v.policy, c = v.certificate ?? null, th = v.site.thresholds;
  // a certificate installed a moment before its phase write (one tick: install, then site.phase) already means dreaming
  const pp = p.phase === 'testing' && p.cert_id && c ? 'dreaming' : p.phase === 'dreaming' && !c ? 'testing' : p.phase, h = held(v.site.name, pp);
  const raw = h && !(h === 'dreaming' && !c) ? h : pp;
  // README (d): the runtime leaves a site in testing when nothing certifies: the page's "no promise at 5%" (shown awake, as designed)
  const cant = !h && raw === 'testing' && !p.cert_id && p.shadow_rows >= th.calibrate_after;
  return { p: cant ? 'awake' : raw, raw, cant, counted: Math.min(p.shadow_rows, th.calibrate_after), cov: c ? Math.round(c.coverage * 100) : 0,
    b: Math.round(100 * (c?.budget ?? p.attempt?.budget ?? st?.budget ?? 0)), c };
}
// A budget change re-checks rows one after another: each keeps its old SiteView until its turn ends (README (d), the ripple).
/** @param {SiteView} cur */
function look(cur) {
  const name = cur.site.name, r = S.recheck[name], t = now();
  if (r && t >= r.to) { delete S.recheck[name]; S.shown[name] = cur.policy.phase; } // the ripple already showed the change: no extra hold
  const pend = !!S.recheck[name], rc = pend && t >= r.from, v = pend ? r.old : cur, ph = phase(v);
  if (rc) Object.assign(ph, { p: 'testing', cant: false });
  return { v, ph, rc, pend };
}
const nm = (/** @type {SiteView} */ v) => `<mark>${nameOf(v.site.id)}</mark>`;

// ---------- narration ----------
// A verdict stays up at least 2 s and verdicts queue in order (several sites can pass in one tick); lesser lines wait their
// turn and only the newest waiting one is kept. Lines from the presenter's own actions show at once.
/** @type {Record<number, number>} */
const HOLD = { 1: 900, 2: 900, 3: 2000 };
function show(/** @type {string} */ msg, /** @type {number} */ rank, idle = false) { Object.assign(S, { narr: msg, narrIdle: idle, narrRank: rank, narrUntil: now() + (HOLD[rank] || 0), queued: null }); }
function narrate(/** @type {string} */ msg, idle = false) { S.verdicts = []; S.last = null; show(msg, 2, idle); }
function narrateLast(/** @type {string} */ msg) { if (S.verdicts.length || (now() < S.narrUntil && S.narrRank === 3)) S.last = msg; else narrate(msg); }
function narrateEvent(/** @type {string} */ msg, /** @type {number} */ rank) {
  const free = !st?.batch.paused && (now() >= S.narrUntil || rank > S.narrRank);
  if (rank === 3) { if (free && !S.verdicts.length) show(msg, 3); else S.verdicts.push(msg); }
  else if (free && !S.verdicts.length) show(msg, rank);
  else if (!S.queued || rank >= S.queued.rank) S.queued = { msg, rank };
}
/** @param {SiteView} v @param {Look} ph */
function eventMsg(v, ph) {
  const s = v.site, router = s.kind === 'router', c = ph.c ?? S.lastCert[s.name], n = c?.n_calib ?? s.thresholds.calibrate_after, a = v.policy.attempt;
  if (ph.p === 'watching') return router ? `trying ${nm(v)} with thinking off, side by side` : `a dream is shadowing ${nm(v)} — same input, answers compared`;
  if (ph.p === 'testing') return router ? `checking thinking off against ${int(n)} real answers` : `checking the ${nm(v)} dream against ${int(n)} real ${s.kind === 'tool' ? 'API returns' : 'answers'}`;
  if (ph.p === 'dreaming' && c) return router ? `${nm(v)} passed with thinking off — wrong ≤ ${pf(c.bound)}, ${pf(c.budget)} allowed` : `${nm(v)} passed its check — wrong ${pf(c.emp_error)}, ${pf(c.budget)} allowed`;
  if (ph.cant && a) return router ? `${nm(v)} keeps thinking — thinking off could be wrong ${pf(a.bound)}, only ${pf(a.budget)} allowed` : `${nm(v)} keeps the real call — no cutoff keeps the dream under ${pf(a.budget)} yet`;
  return '';
}
// The handwritten note beside the graph points at whatever just changed, as on the design board.
/** @type {Record<string, {x: number, y: number, a: string, l: number, w?: number}>} */
const NOTE_AT = {
  categorize: { x: 178, y: 208, a: 'up', l: 60 }, vendor_lookup: { x: 400, y: 62, a: 'down', l: 90, w: 230 },
  match_invoice: { x: 646, y: 212, a: 'up', l: 60 }, anything_odd: { x: 846, y: 30, a: 'down', l: 30, w: 150 },
};
/** @type {Record<string, (l: number) => string>} */
const ARROWS = {
  up: (l) => `<svg width="64" height="34" viewBox="0 0 64 34" style="left:${l}px;top:-34px" aria-hidden="true"><path d="M8 32 C12 18 24 9 46 5 M36 1 L47 5 L39 13" fill="none" stroke="#6A4FD8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
  down: (l) => `<svg width="64" height="34" viewBox="0 0 64 34" style="left:${l}px;bottom:-36px" aria-hidden="true"><path d="M8 2 C12 16 24 25 46 29 M37 33 L47 29 L40 21" fill="none" stroke="#6A4FD8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
};
/** @param {{id: string, kind: string}} n */
function noteText(n) {
  const v = byName(n.id);
  if (!v) return null;
  const ph = phase(v), tool = v.site.kind === 'tool';
  if (n.kind === 'down') return !S.unplugged ? null : ph.p === 'dreaming' ? 'API down, still answering!' : stuckNow() ? 'stuck! nothing learned yet' : 'API down, the snapshot answers!';
  // the saving is the site's own two costs; no predicted call yet means no saving to point at
  if (ph.p !== 'dreaming' || downOf(v) || !v.policy.predicted_rows || !v.cost_awake_usd) return null;
  const save = Math.round((1 - v.cost_dream_usd / v.cost_awake_usd) * 100);
  return v.site.kind === 'router' ? `thinks less now! −${save}% cost` : tool ? `knows what the API will say! −${save}% cost` : `learned to dream! −${save}% cost`;
}
function watchPhases() {
  const msgs = [];
  for (const cur of sites()) {
    const name = cur.site.name, { ph, pend } = look(cur), key = ph.p + (ph.cant ? '!' : '');
    if (pend) continue; // a budget re-check is narrated by its summary; compared after it, a promise it kept is no new promotion
    const was = S.prev[name];
    S.shown[name] = ph.raw;
    if (was && was !== key) {
      if (ph.p === 'dreaming') { S.pop[name] = now() + 650; if (!S.unplugged) S.note = { id: name, kind: 'dream' }; }
      // a verdict outranks a check starting, which outranks a dream starting to watch
      const rank = ph.p === 'dreaming' || ph.cant ? 3 : ph.p === 'testing' ? 2 : 1, msg = eventMsg(cur, ph);
      if (msg) msgs.push({ rank, msg });
    }
    S.prev[name] = key;
  }
  // during an outage the outage message stays up; after a budget change one summary line is shown instead
  if (!S.unplugged && now() > S.quietUntil) {
    // one lesser line per frame (the highest rank, first in workflow order), then every verdict
    const lesser = msgs.reduce((a, m) => (m.rank < 3 && (!a || m.rank > a.rank) ? m : a), /** @type {{rank: number, msg: string} | null} */ (null));
    if (lesser) narrateEvent(lesser.msg, lesser.rank);
    for (const m of msgs) if (m.rank === 3) narrateEvent(m.msg, 3);
  }
  // a training failure is news only when no candidate of that site trained in the same tick (a spare failing is not), once per site
  for (const [name, at] of Object.entries(S.failQ)) {
    if (at >= S.stAt) continue; // judged on a /state read sent after the event
    delete S.failQ[name];
    const v = byName(name);
    if (v && !v.policy.candidate && !S.failSaid.has(name) && !S.unplugged) { S.failSaid.add(name); narrateEvent(`the ${nm(v)} dream failed to train · the real call keeps answering`, 2); }
  }
  if (!S.unplugged && !st?.batch.paused && now() >= S.narrUntil) {
    if (S.verdicts.length) show(S.verdicts.shift(), 3);
    else if (S.last) { show(S.last, 2); S.last = null; }
    else if (S.queued) show(S.queued.msg, S.queued.rank);
  }
}
/** @param {Metrics} m */
function doneLine(m) {
  const v = vendor(), first = st?.series[0];
  if (S.unplugged) {
    if (stuckNow()) return 'vendor API is still down · every run is stuck at <mark>vendor lookup</mark>';
    return `vendor API is still down · the ${v && phase(v).p === 'dreaming' ? 'dream and the snapshot' : 'snapshot'} kept ${pct(100 - m.failed_pct)} of runs going`;
  }
  const live = st?.batch.recorded && st.batch.live_calls ? ` · ${int(st.batch.live_calls)} answers went live` : '';
  return `done · cost per run ${first ? `${cents(first.cost_per_run)} → ` : ''}${cents(m.cost_per_run)}${live} · click a step to see the promise behind it`;
}
// README (e): a fast-forward never says a bare "fast-forwarding": whose answers, and that seconds are virtual-clock time
function teacherWords() {
  const t = health?.checks.teacher ?? '';
  if (st?.batch.recorded || /Recorded/.test(t)) return 'pre-recorded teacher answers, virtual clock';
  return /Rule/.test(t) ? 'offline rule teacher, virtual clock' : 'live teacher answers, virtual clock';
}

// ---------- events: the narration's source, and which of them mean "read /state again" ----------
const seen = new Set(); // Event.id: backfill, poll and live inserts overlap. Ids are random: never compare, keep arrival order
let last = '';
let regating = false; // inside the run of events one budget change emits (budget.changed, then each site's cert.* and site.phase)
/** @param {DEvent} e @param {boolean} fresh  false for backfill: history is not animated @returns {boolean} true when /state changed */
function onEvent(e, fresh) {
  if (seen.has(e.id)) return false;
  seen.add(e.id);
  last = e.id;
  if (e.kind === 'component.error') { problems.events = [...problems.events, `${e.component}: ${e.error ?? e.message}`].slice(-3); strip(); }
  const name = short(e.call_site || ''), v = byName(name), d = /** @type {any} */ (e.data);
  if (e.kind === 'cert.issued' && d.certificate) S.lastCert[name] = d.certificate;
  regating = e.kind === 'budget.changed' || (regating && /^(cert\.|site\.phase|candidate\.)/.test(e.kind));
  if (fresh) {
    // a budget change's phase moves are the ripple's to animate; they reach the page a poll later and must not replay as held stickers
    if (e.kind === 'site.phase') { if (!regating) onPhase(e); }
    else if (e.kind === 'batch.finished' && d.metrics) narrateLast(doneLine(d.metrics));
    else if (e.kind === 'budget.changed') S.budgetAt = now();
    else if (e.kind === 'cert.revoked' && v && !S.recheck[name] && now() - S.budgetAt > 5000 && !S.unplugged) narrateEvent(`${nm(v)} lost its promise · the real call answers again`, 3);
    else if (e.kind === 'train.failed' && v) S.failQ[name] = now();
    else if (e.kind === 'train.finished') { delete S.failQ[name]; S.failSaid.delete(name); }
  }
  return /^(batch\.|site\.phase|cert\.|tool\.|budget\.changed|candidate\.|train\.(finished|failed))/.test(e.kind); // a failure is judged on the next read
}
// Polling when Realtime is not configured: the first read is history, then `after=<last id>`, faster while a batch runs.
// /state is also re-read while the last read failed (API down at load, a 500 mid-batch), so the page never stays stale.
let polled = false;
async function poll() {
  const r = last ? await req('GET', `/events?after=${encodeURIComponent(last)}&limit=200`) : await req('GET', '/events?limit=200');
  if (r.ok) {
    const changed = /** @type {DEvent[]} */ (r.data).map((e) => onEvent(e, polled)).includes(true);
    polled = true; // only a read that arrived is history; a failed first read must not make later history look fresh
    if (changed || !st || problems.state) refresh();
  }
  setTimeout(poll, st?.batch.running ? 700 : 2000);
}
/** @param {Record<string, string>} config */
async function live(config) {
  const url = config.supabase_url, key = config.supabase_publishable_key;
  if (url && key) {
    try {
      const { subscribe } = await import('./live.js'); // dynamic: offline, the CDN import inside live.js must not take the page down
      subscribe(url, key, apiBase, (/** @type {Live} */ m) => { if (m.table !== 'events' || onEvent(m.row, m.type !== 'BACKFILL')) refresh(); });
      return;
    } catch (e) { problems.events = [...problems.events, `realtime unavailable, polling instead: ${e}`].slice(-3); strip(); }
  }
  poll();
}

// ---------- flows stage ----------
/** @type {{id: string, x: number, y: number, w: number, icon: string, blob: string, rot: number, sub?: string, round?: boolean, el?: HTMLElement, who?: HTMLElement, badge?: HTMLElement}[]} */
const NODES = [
  { id: 'feed', x: 0, y: 126, w: 126, sub: 'webhook', icon: 'feed', blob: '#FFF0C2', rot: -2 },
  { id: 'categorize', x: 168, y: 86, w: 192, icon: 'tag', blob: '#FFD6E0', rot: 1.5 },
  { id: 'vendor_lookup', x: 402, y: 148, w: 192, icon: 'store', blob: '#CFE8FF', rot: -1 },
  { id: 'match_invoice', x: 636, y: 90, w: 192, icon: 'invoice', blob: '#DDF5E8', rot: 1.2 },
  { id: 'anything_odd', x: 870, y: 118, w: 96, round: true, icon: 'branch', blob: '#FFE9D6', rot: 0 },
  { id: 'post_to_books', x: 1008, y: 14, w: 192, sub: 'ledger', icon: 'book', blob: '#E2D9FF', rot: -1.5 },
  { id: 'wake_a_human', x: 1008, y: 236, w: 192, sub: 'bookkeeper', icon: 'clock', blob: '#FFF0C2', rot: 2 },
];
const DREAMERS = ['categorize', 'vendor_lookup', 'match_invoice', 'anything_odd']; // the clickable nodes; effects and the feed never dream
const WIRES = [
  { id: 'w1', d: 'M126 168 C147 168 147 128 168 128', c: '#FF8A80' },
  { id: 'w2', d: 'M360 128 C381 128 381 190 402 190', c: '#B39DFF' },
  { id: 'w3', d: 'M594 190 C615 190 615 132 636 132', c: '#7FB7FF', after: true },
  { id: 'w4', d: 'M828 132 C849 132 849 166 870 166', c: '#5CC79F', after: true },
  { id: 'w5', d: 'M960 142 C988 120 990 56 1008 56', c: '#F2B544', after: true },
  { id: 'w6', d: 'M960 190 C988 212 990 278 1008 278', c: '#FF8A80', after: true, rare: true },
];
function buildStage() {
  const arrow = (/** @type {typeof WIRES[0]} */ w) => { const [x, y] = w.d.split(' ').slice(-2).map(Number); return `<path d="M${x - 9} ${y - 6} L${x} ${y} L${x - 9} ${y + 6}" stroke="${w.c}"/>`; };
  const dots = (/** @type {typeof WIRES[0]} */ w) => [0, 1].slice(0, w.rare ? 1 : 2).map((i) => `<circle r="3.6" fill="#2B2B33"><animateMotion dur="${w.rare ? 2.4 : 1.1}s" begin="${-i * .55}s" repeatCount="indefinite"><mpath xlink:href="#${w.id}"/></animateMotion></circle>`).join('');
  const node = (/** @type {typeof NODES[0]} */ n) => {
    const site = DREAMERS.includes(n.id), tag = site ? 'button' : 'div', name = n.id === 'feed' ? 'bank feed' : nameOf(n.id);
    const style = `left:${n.x}px;top:${n.y}px;width:${n.w}px;--rot:${n.rot}deg${n.round ? `;background:${n.blob}` : ''}`;
    const badge = site ? '<span class="badge"></span>' : '';
    if (n.round) return `<${tag} type="button" class="node round" data-id="${n.id}" style="${style}">${ico(n.icon, 42)}<span class="rlabel"><b>${name}</b><span class="who"></span></span>${badge}</${tag}>`;
    return `<${tag} ${site ? 'type="button"' : ''} class="node" data-id="${n.id}" style="${style}"><span class="blob" style="background:${n.blob}">${ico(n.icon, 30)}</span><span class="txt"><b>${name}</b><span class="who">${n.sub || ''}</span></span>${badge}</${tag}>`;
  };
  $('#stage').innerHTML =
    `<svg width="1200" height="330" viewBox="0 0 1200 330" ${R} aria-hidden="true"><g fill="none" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round">${WIRES.map((w) => `<path id="${w.id}" d="${w.d}" stroke="${w.c}"/>${arrow(w)}`).join('')}</g></svg>` +
    `<svg class="dots" width="1200" height="330" viewBox="0 0 1200 330" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true">${WIRES.map((w) => `<g data-w="${w.id}" class="off">${dots(w)}</g>`).join('')}</svg>` +
    `<span class="blbl" style="left:984px;top:124px">no</span><span class="blbl" style="left:982px;top:186px">yes</span>` +
    NODES.map(node).join('') +
    `<button id="plug" class="plug" type="button" aria-pressed="false" style="left:424px;top:248px"></button>` +
    `<div id="note" class="note" hidden aria-hidden="true"><span></span><i></i></div>`;
  for (const n of NODES) { n.el = $(`.node[data-id="${n.id}"]`); n.who = n.el.querySelector('.who'); n.badge = n.el.querySelector('.badge'); }
  $('#stage').addEventListener('click', (e) => { const b = /** @type {HTMLElement} */ (e.target).closest('button.node'); if (b) location.hash = '#dreams/' + /** @type {HTMLElement} */ (b).dataset.id; });
  $('#plug').addEventListener('click', togglePlug);
}
/** @param {SiteView} v @param {Look} ph @param {boolean} down @param {boolean} rc */
function whoText(v, ph, down, rc) {
  const router = v.site.kind === 'router';
  if (down) return ph.p === 'dreaming' ? 'API down · dream + snapshot' : stuckNow() ? 'API down · stuck' : 'API down · snapshot';
  if (ph.p === 'watching') return router ? 'thinking off on trial' : 'dream shadowing';
  if (ph.p === 'testing') return rc ? 're-checking…' : router ? 'checking thinking off' : 'checking the dream';
  if (ph.p === 'dreaming') return router ? (ph.cov >= 100 ? `${teacherOf(v)}, thinking off` : `thinking off ${ph.cov}%`) : `dream answers ${ph.cov}%`;
  return router ? `${teacherOf(v)}, thinking` : teacherOf(v);
}
function renderStage() {
  for (const n of NODES) {
    const cur = byName(n.id);
    if (!cur || !n.el || !n.badge || !n.who) continue;
    const { v, ph, rc } = look(cur), down = downOf(cur), sk = down ? 'down' : ph.p;
    setAttr(n.el, 'class', `node${n.round ? ' round' : ''} ${sk}${S.pop[n.id] > now() ? ' pop' : ''}`);
    setHTML(n.badge, STATUS[sk](sk === 'dreaming' ? 30 : 26));
    const who = whoText(v, ph, down, rc);
    setText(n.who, who);
    setAttr(n.el, 'aria-label', `${nameOf(n.id)}: ${who}. Open its promise.`);
  }
  const flowing = !!st?.batch.running && !st.batch.paused, blocked = stuckNow();
  for (const w of WIRES) $(`.dots g[data-w="${w.id}"]`).classList.toggle('off', !flowing || !!(w.after && blocked));
  const plug = $('#plug');
  setAttr(plug, 'aria-pressed', String(S.unplugged));
  setAttr(plug, 'aria-label', S.unplugged ? 'Plug the vendor API back in' : 'Unplug the vendor API');
  setHTML(plug, S.unplugged ? `${ico('unplug', 26)}plug it back in` : `${ico('plug', 26)}unplug the API`);
  const note = $('#note'), text = S.note && noteText(S.note);
  note.hidden = !text;
  if (!text) return;
  const at = NOTE_AT[S.note.id], key = S.note.id + S.note.kind;
  if (note.dataset.k !== key) { note.dataset.k = key; note.classList.remove('fresh'); void note.offsetWidth; note.classList.add('fresh'); }
  note.style.left = at.x + 'px'; note.style.top = at.y + 'px'; note.style.width = (at.w || 190) + 'px';
  setText(/** @type {HTMLElement} */ (note.firstChild), text);
  setHTML(note.lastChild, ARROWS[at.a](at.l));
}
let plugBusy = false; // one toggle in flight: rapid clicks must not race each other to the switch
async function togglePlug() {
  const v = vendor();
  if (!v || plugBusy) return;
  plugBusy = true;
  const down = !S.unplugged; // shown at once: the presenter's action; reverted if the API refuses
  S.unplugged = down;
  S.note = down ? { id: 'vendor_lookup', kind: 'down' } : null;
  narrate(down ? outageLine(v) : 'vendor API is back');
  const r = await cmd('POST', `/tools/${encodeURIComponent(v.site.id)}/outage`, { down }, (p) => { const cur = vendor(); if (cur) cur.policy = p; plugBusy = false; });
  plugBusy = false;
  if (!r.ok) { S.unplugged = !down; S.note = null; narrate(`the fault switch refused: ${esc(r.text)}`); }
}

// ---------- flows charts ----------
/** @type {{k: 'cost_per_run' | 'sec_per_run' | 'failed_pct' | 'replayable_pct', lbl: string, c: string, f: (v: number) => string, val?: HTMLElement, was?: HTMLElement, svg?: HTMLElement, line?: HTMLElement, ref?: HTMLElement}[]} */
const METRICS = [
  { k: 'cost_per_run', lbl: 'cost per run', c: '#FF8A80', f: cents },
  { k: 'sec_per_run', lbl: 'seconds per run', c: '#7FB7FF', f: (v) => v.toFixed(1) + ' s' },
  { k: 'failed_pct', lbl: 'failed runs', c: '#F2B544', f: pct },
  { k: 'replayable_pct', lbl: 'replayable decisions', c: '#5CC79F', f: (v) => Math.round(v) + '%' },
];
function buildCharts() {
  // dashed line = where this metric started; crayon line = where it is now
  $('#cells').innerHTML = METRICS.map((m) => `<div class="cell" data-k="${m.k}"><span class="lbl">${m.lbl}</span><div class="row"><b class="val"></b><span class="was"></span></div>` +
    `<svg viewBox="0 0 300 56" preserveAspectRatio="none" role="img"><path class="ref" fill="none" stroke="#C9C5D3" stroke-width="1.5" stroke-dasharray="4 5" vector-effect="non-scaling-stroke"/>` +
    `<path d="M0 55H300" stroke="#2B2B33" stroke-width="2" vector-effect="non-scaling-stroke"/>` +
    `<path class="line" fill="none" stroke="${m.c}" stroke-width="3.6" stroke-linecap="round" stroke-linejoin="round" vector-effect="non-scaling-stroke" style="filter:url(#roughl)"/></svg></div>`).join('');
  for (const m of METRICS) {
    const c = $(`.cell[data-k="${m.k}"]`), q = (/** @type {string} */ s) => /** @type {HTMLElement} */ (c.querySelector(s));
    Object.assign(m, { val: q('.val'), was: q('.was'), svg: q('svg'), line: q('.line'), ref: q('.ref') });
  }
}
// "circuit breaker: 100%" is comparison copy, not a measurement (README (d)): every run calls vendor lookup, so a breaker fails them all
const BREAKER = 'a plain circuit breaker fails every run while the vendor API is down, because every run needs it. not measured by this run';
let drawn = '';
function renderCharts() {
  if (!st) return;
  const cur = st.metrics, pts = st.series, first = pts[0] || cur, b = st.batch, ran = cur.runs > 0;
  // x is Metrics.runs; the axis reaches the end of the batch that is running (Metrics.runs counts the run in flight, Batch.done not)
  const xmax = Math.max(BATCH, Math.ceil((cur.runs + (b.running ? b.target - b.done - 1 : 0)) / BATCH) * BATCH);
  const vv = vendor(), down = !!vv && downOf(vv);
  setText($('#xr'), `0 → ${int(xmax)} runs${ruled() ? " — ¢ and s use the rule teacher's flat stand-in, not a model's price" : ''}`);
  for (const m of METRICS) {
    if (!m.val || !m.was || !m.svg || !m.line || !m.ref) continue;
    const alarm = m.k === 'failed_pct' && down;
    setText(m.val, ran ? m.f(cur[m.k]) : '–');
    setText(m.was, alarm ? 'circuit breaker: 100%' : ran ? `was ${m.f(first[m.k])}` : '');
    setAttr(m.was, 'title', alarm ? BREAKER : '');
    m.was.classList.toggle('alarm', alarm);
    setAttr(m.svg, 'aria-label', ran ? `${m.lbl}: was ${m.f(first[m.k])}, now ${m.f(cur[m.k])}${alarm ? `; ${BREAKER}` : ''}` : `${m.lbl}: no runs yet`);
  }
  const lastPt = pts[pts.length - 1], key = `${pts.length}:${lastPt?.runs}:${xmax}`;
  if (key === drawn) return;
  drawn = key;
  const X = (/** @type {number} */ r) => r / xmax * 288;
  for (const m of METRICS) {
    if (!m.line || !m.ref) continue;
    if (!pts.length) { m.ref.setAttribute('d', 'M0 30H300'); m.line.setAttribute('d', ''); continue; } // four lines waiting to move
    // each line gets its own height, from zero, so a halving looks like a halving
    const top = Math.max(...pts.map((p) => p[m.k])) * 1.15 || 1;
    const Y = (/** @type {number} */ v) => (49 - v / top * 45).toFixed(1); // zero sits above the 2px axis, so a flat 0% line still shows
    m.ref.setAttribute('d', `M0 ${Y(pts[0][m.k])}H300`);
    // the short flat tail marks "now", so a change at the latest run reads as a step, not a clipped edge
    m.line.setAttribute('d', pts.map((p, i) => `${i ? 'L' : 'M'}${X(p.runs).toFixed(1)} ${Y(p[m.k])}`).join('') + `H${(X(lastPt.runs) + 10).toFixed(1)}`);
  }
}

// ---------- dreams ----------
/** @type {Record<string, {b: HTMLElement, icon: HTMLElement, status: HTMLElement, cand: HTMLElement, cost: HTMLElement}>} */
const ROWS = {};
function buildJournal() {
  $('#jrows').innerHTML = sites().map((v) => `<li><button class="jrow" type="button" data-id="${v.site.name}" aria-pressed="false"><span class="jicon"></span><span class="jtext"><b>${nameOf(v.site.id)}</b><span class="jstatus"></span></span><span class="jcost"></span><span class="jcand" title="the trained models for this step: the one that answers and the share of calls it takes, its certified error bound, then the spares"></span></button></li>`).join('');
  for (const v of sites()) {
    const b = $(`.jrow[data-id="${v.site.name}"]`), q = (/** @type {string} */ s) => /** @type {HTMLElement} */ (b.querySelector(s));
    ROWS[v.site.name] = { b, icon: q('.jicon'), status: q('.jstatus'), cand: q('.jcand'), cost: q('.jcost') };
  }
  $('#jrows').addEventListener('click', (e) => { const b = /** @type {HTMLElement} */ (e.target).closest('.jrow'); if (b) { S.sel = /** @type {HTMLElement} */ (b).dataset.id || S.sel; history.replaceState(null, '', '#dreams/' + S.sel); } });
}
/** @param {SiteView} v @param {Look} ph @param {boolean} rc @param {boolean} down */
function statusLine(v, ph, rc, down) {
  const n = v.site.thresholds.calibrate_after;
  if (rc) return 're-checking…';
  if (down) return ph.p === 'dreaming' ? `API down · dream ${ph.cov}%, snapshot for the rest` : 'API down · nothing learned yet';
  if (ph.p === 'watching') return `watching · counting sheep ${ph.counted}/${n}`;
  if (ph.p === 'testing') return `testing · checking ${n} real answers`;
  if (ph.p === 'dreaming') return v.site.kind === 'router' ? (ph.cov >= 100 ? 'dreaming · thinking off on every call' : `dreaming · thinking off on ${ph.cov}% of calls`) : `dreaming · answers ${ph.cov}% of calls`;
  return ph.cant ? `awake · no promise at ${ph.b}% yet` : 'awake · logging every call';
}
// The candidate rows (DESIGN.md §5, training-plan §1): the promoted family in plain words with the share it answers and its certified
// error bound (the card's "at most"), then each spare. They follow the held sticker: a row still showing "testing" states no result.
// No "·" here: Gaegu lacks it and the fallback glyph shifts the line.
/** @param {SiteView} v @param {Look} ph */
function candHTML(v, ph) {
  const p = v.policy, c = ph.p === 'dreaming' ? v.certificate : null, certs = st?.spare_certificates ?? {}, err = v.site.kind === 'router' ? 'disagrees' : 'wrong';
  if (!p.candidate) return '';
  const plain = (/** @type {string} */ n) => PLAIN[n] ?? n, waiting = ph.p === 'watching' ? 'shadowing' : ph.p === 'testing' ? 'being checked' : 'not certified';
  const lines = [c && c.candidate === p.candidate ? `${plain(p.candidate)} answers ${Math.round(c.coverage * 100)}%, ${err} at most ${pf(c.bound)}` : `${plain(p.candidate)}, ${waiting}`];
  for (const [name, sp] of Object.entries(p.shadow)) {
    const sc = c && sp.cert_id && certs[sp.cert_id]?.budget === st?.budget ? certs[sp.cert_id] : null; // a promise at an older budget is not one now
    lines.push(sc ? `spare ${plain(name)}: ${Math.round(sc.coverage * 100)}%, ${err} at most ${pf(sc.bound)}` : `spare ${plain(name)}, ${waiting}`);
  }
  return lines.map((l) => `<span>${esc(l)}</span>`).join('');
}
// no real call in the window (only the dream and the snapshot answered, e.g. during an outage): no cost to print
const noCost = (/** @type {SiteView} */ v) => !v.policy.rows || (!v.cost_awake_usd && !v.sec_awake);
/** @param {SiteView} v @param {Look} ph */
function costHTML(v, ph) {
  if (noCost(v)) return '<span>–</span>';
  if (ph.p !== 'dreaming') return `<span>${cents(v.cost_awake_usd)}</span>`;
  return `<s>${cents(v.cost_awake_usd)}</s>${WOB}<em>${cents(v.cost_dream_usd)}</em>`; // cost_dream_usd: the blend the promise in force certifies
}
function renderJournal() {
  for (const cur of sites()) {
    const row = ROWS[cur.site.name];
    if (!row) continue;
    const { v, ph, rc, pend } = look(cur), down = downOf(cur);
    setAttr(row.b, 'aria-pressed', String(S.sel === cur.site.name));
    setHTML(row.icon, STATUS[rc ? 'testing' : down ? 'down' : ph.p](ph.p === 'dreaming' && !down && !rc ? 50 : 46));
    setText(row.status, statusLine(v, ph, rc, down));
    if (!pend) setHTML(row.cand, candHTML(v, ph)); // a row waiting to re-check keeps its old candidate lines: the new spare certificates are already in State
    // while a row re-checks, its old saving stays visible (dimmed) rather than looking like it vanished
    setHTML(row.cost, costHTML(v, phase(v)));
    row.cost.classList.toggle('dim', pend);
  }
  const m = st?.metrics, busy = Object.keys(S.recheck).length > 0;
  // a sticker still held (README (g)) has not stated its result yet, so neither has this line; after a budget change the API re-priced the window
  if (!Object.keys(S.hold).length || !S.bnow) S.bnow = m && m.runs ? `right now: <b>${cents(m.cost_per_run)}</b> per run · <b>${Math.round(m.replayable_pct)}%</b> replayable` : 'right now: <b>no runs yet</b>';
  setHTML($('#bnow'), busy ? 'right now: <b>re-checking…</b>' : S.bnow);
  const bud = /** @type {HTMLInputElement} */ ($('#budget'));
  if (st && document.activeElement !== bud && !busy && !budgetBusy) { bud.value = String(Math.round(st.budget * 100)); setText($('#bout'), bud.value + '%'); }
}
/** @param {string} name */
function certHTML(name) {
  const cur = byName(name);
  if (!cur) return '';
  // in its re-check turn the card keeps the old promise's body, dimmed, with "re-checking…" in its place: phase(v), not look()'s testing
  const { v, rc, pend } = look(cur), ph = phase(v), s = v.site, p = v.policy, th = s.thresholds, router = s.kind === 'router', tool = s.kind === 'tool';
  const down = downOf(cur), real = tool ? 'real API returns' : 'real answers', b = ph.b;
  const icon = rc ? 'testing' : down ? 'down' : ph.p;
  const head = `<div class="chead">${STATUS[icon](icon === 'dreaming' ? 44 : 40)}<h2>${nameOf(s.id)}</h2></div>`;
  // README (d): the sheep count the rows checked, 1 sheep = a tenth of them: calibrate_after while counting, then the rows the gate read
  const total = ph.c?.n_calib ?? (ph.cant && p.attempt?.n ? p.attempt.n : th.calibrate_after), per = total / 10;
  const filled = ph.c || ph.cant ? 10 : Math.floor(ph.counted / per);
  const sheep = `<div class="sheep" role="img" aria-label="${int(filled * per)} of ${int(total)} ${real} checked">${Array.from({ length: 10 }, (_, i) => sheepSVG(i < filled)).join('')}<small>1 sheep = ${int(per)}</small></div>`;
  const helper = router ? 'thinking off' : 'dream', teacher = router ? 'thinking on' : down ? 'snapshot' : teacherOf(v);
  const split = (/** @type {number} */ dream, /** @type {string} */ label) => `<div class="split" role="img" aria-label="${helper} answers ${dream}%"><i style="width:${dream}%"></i></div><div class="legend">${dream ? `<span><i style="background:#B39DFF"></i>${helper} ${dream}%</span>` : ''}${dream < 100 ? `<span><i style="background:#FFF3A8"></i>${label} ${100 - dream}%</span>` : ''}</div>`;
  const promise = (/** @type {string} */ text, soft = false) => `<p class="promise${soft || rc ? ' soft' : ''}">${rc ? 're-checking…' : text}</p>`;
  let body;
  if (down && ph.p !== 'dreaming')
    body = promise('API down', true) + sheep + `<p class="cline">nothing has learned this API yet, so ${stuckNow() ? 'every run is stuck here' : 'the last snapshot answers every call'}.</p>`;
  else if (ph.p === 'awake' && !ph.cant)
    body = promise('no promise yet', true) + sheep + `<p class="cline">logging every call. a dream starts watching after ${th.watch_after} calls.</p>${split(0, teacher)}`;
  else if (ph.p === 'watching')
    body = promise('counting sheep…', true) + sheep + `<p class="cline">${ph.counted} of ${th.calibrate_after} ${real} compared so far. the ${teacherOf(v)} still answers every call.</p>${split(0, teacher)}`;
  else if (ph.p === 'testing' || !ph.c) {
    body = promise('checking…', true) + sheep + `<p class="cline">${router ? `measuring how often thinking off disagrees with thinking on, against a ${b}% limit.` : `sorting ${th.calibrate_after} ${real} by confidence to find where mistakes stay under ${b}%.`}</p>`;
    if (ph.cant) {
      // answering every call is what failed; the hint's budget is where a cutoff leaves out enough unsure calls to stay under it
      const a = p.attempt;
      body = promise(`no promise at ${b}%`, true) + sheep + (a ? `<p class="cline">answering every call, the ${helper} disagreed on ${pf(a.emp_error)} of ${int(a.n)} ${real} (at most ${pf(a.bound)}). no cutoff keeps it under ${b}%.</p>` : '') + split(0, teacher) +
        (a && a.min_budget != null ? `<p class="hint">allow ${pf(a.min_budget)} and it ${router ? 'switches thinking off' : 'starts dreaming'}</p>` : '');
    }
  } else {
    const c = ph.c, every = Math.round(1 / th.audit_rate), cert = `<span class="nw">cert ${c.id.slice(5, 11)}</span>`;
    const checks = down ? 'no double-checks while the API is down' : `1 in ${every} still double-checked${router ? ' with thinking on' : ''}`;
    const fine = !s.replayable
      ? `${checks} · ${cert} · not replayable: it is still the ${teacherOf(v)}`
      : `${checks} · ${cert} · <span class="nw">weights <code>${(p.weights_sha ?? c.weights_sha).slice(0, 6)}</code></span>`;
    const unit = s.teacher.provider === 'rule' ? 'per call at rule-teacher prices' : 'per call'; // a tool site bills its own price_call
    const cost = noCost(v) ? '' : `<p class="ccost"><s>${cents(v.cost_awake_usd)}</s> → <em>${cents(v.cost_dream_usd)}</em> ${unit}</p>`;
    // the three numbers of one promise: allowed (the headline), seen on the calls it answers, and the certified bound on them
    body = promise(`wrong ≤ ${pf(c.budget)}`) + sheep + `<p class="cline">checked against ${int(c.n_calib)} ${real} · ${ph.cov >= 100 ? 'on every call' : `on the ${ph.cov}% it answers`}, ${router ? 'disagreed' : 'wrong'} ${pf(c.emp_error)}, at most ${pf(c.bound)}.</p>` +
      (down ? `<p class="cline">API down: calls the dream is unsure about use the last snapshot.</p>` : '') + split(ph.cov, teacher) + cost + `<p class="fine">${fine}</p>`;
  }
  return head + (pend ? `<div class="rc">${body}</div>` : body);
}
// Releasing the slider: POST /budget, then the rows that hold or tried a promise ripple through the results already returned.
// A release while one is in flight is sent when it returns (the latest release wins), so the slider never shows a budget nobody sent.
let budgetBusy = false, budgetNext = 0, budgetSeq = 0;
async function onBudget(commit = false) {
  const el = /** @type {HTMLInputElement} */ ($('#budget'));
  let b = +el.value;
  setText($('#bout'), b + '%');
  if (!commit || !st) return;
  if (budgetBusy) { budgetNext = b; return; }
  if (b === Math.round(st.budget * 100)) return;
  budgetBusy = true;
  S.budgetAt = now();
  const seq = ++budgetSeq; // a newer release replaces this one's summary
  const done = sites().filter((v) => v.policy.phase === 'testing' || v.policy.phase === 'dreaming'); // what set_budget re-gates (and keeps re-gating)
  const olds = done.map((v) => look(v).v);
  done.forEach((v, i) => { S.recheck[v.site.name] = { from: Infinity, to: Infinity, old: olds[i] }; delete S.hold[v.site.name]; }); // dimmed while in flight
  let r = await cmd('POST', '/budget', { budget: b / 100 });
  while (r.ok && budgetNext && budgetNext !== b) { b = budgetNext; budgetNext = 0; S.budgetAt = now(); r = await cmd('POST', '/budget', { budget: b / 100 }); }
  budgetNext = 0;
  budgetBusy = false;
  const t = now();
  if (!r.ok) {
    done.forEach((v) => delete S.recheck[v.site.name]);
    el.value = String(Math.round(st.budget * 100)); setText($('#bout'), el.value + '%');
    narrate(`the budget did not change: ${esc(r.text)}`);
    return;
  }
  done.forEach((v, i) => { S.recheck[v.site.name] = { from: t + i * 170, to: t + i * 170 + 450, old: olds[i] }; });
  const end = done.length ? (done.length - 1) * 170 + 450 : 0;
  S.quietUntil = t + end + 60;
  // read now, from the reply to this budget: a later release must not change what this summary says
  const n = done.length, kept = done.filter((v) => byName(v.site.name)?.policy.cert_id).length, m = st.metrics;
  const cost = m.runs ? ` · cost per run now ${cents(m.cost_per_run)}` : ''; // the API re-priced the last runs under the new promises
  setTimeout(() => {
    // during an outage the outage line stays up
    if (n && !S.unplugged && seq === budgetSeq) narrate(`re-checked at ${b}% · ${kept} of ${n} step${n > 1 ? 's' : ''} hold${kept === 1 && n === 1 ? 's' : ''} a promise${cost}`);
  }, end);
}

// ---------- replay ----------
/** @type {Record<string, string>} */
const BYICON = { dream: 'sheep', 'big model': 'moon', 'rule teacher': 'moon', 'thinking off': 'moon', 'vendor API': 'plug', snapshot: 'camera', ledger: 'book', bookkeeper: 'clock', stuck: 'unplug' };
/** @returns {Run[]} newest first, the very first run kept last for contrast; a run that was replayed stays on screen */
function listedRuns() {
  if (!st) return [];
  const newest = st.runs.slice(-6).reverse(), f = st.first_run, k = S.rp?.run;
  // during a batch the replayed run leaves the newest six within one /state read: it keeps the sixth place, so it stays selected
  if (k && k.id !== f?.id && !newest.some((r) => r.id === k.id)) newest.splice(5, 1, k);
  if (f && !newest.some((r) => r.id === f.id)) newest.push(f);
  return newest;
}
function currentRun() {
  // the selection must be one of the receipts on screen; otherwise fall back to the newest
  const list = listedRuns();
  if (S.selRun != null && !list.some((x) => x.id === S.selRun)) { S.selRun = null; S.rp = null; }
  return list.find((x) => x.id === S.selRun) || list[0] || null;
}
/** The memory's top voting rows under a dream's answer (README (d)): the row's first line minus its "Field: " prefix, a seed row
 * reads "seed example", a row the API could not find shows its label only. Rows that read the same are one "2× ..." with their
 * weights summed. @param {TraceRow} r @param {Record<string, string>} texts */
function votedOf(r, texts) {
  const ev = r.decision === 'predicted' ? r.prediction?.evidence?.filter((e) => e.weight >= 0.005) : null; // a "0.00" voter says nothing
  if (!ev?.length) return '';
  const by = /** @type {Map<string, {n: number, w: number}>} */ (new Map());
  for (const e of ev) {
    const t = e.row_id.startsWith('seed:') ? 'seed example' : texts[e.row_id]?.split('\n')[0].replace(/^\w+: /, '');
    const k = t ? `${t} (${e.label})` : e.label, g = by.get(k) ?? { n: 0, w: 0 };
    by.set(k, { n: g.n + 1, w: g.w + e.weight });
  }
  return 'voted: ' + [...by].map(([k, g]) => `${g.n > 1 ? `${g.n}× ` : ''}${k} ${g.w.toFixed(2)}`).join(' · ');
}
/** README (d): who answered = decision + kind. @param {TraceRow} r @param {Record<string, string>} [texts] @returns {Step} */
function stepOf(r, texts = {}) {
  const by = r.error?.startsWith('Stuck') ? 'stuck'
    : r.kind === 'effect' ? (short(r.call_site) === 'wake_a_human' ? 'bookkeeper' : 'ledger')
      : r.decision === 'predicted' ? (r.kind === 'router' ? 'thinking off' : 'dream')
        : r.decision === 'fallback' ? 'snapshot'
          : r.kind === 'tool' ? 'vendor API' : r.teacher?.provider === 'rule' ? 'rule teacher' : 'big model';
  const ans = by === 'stuck' ? 'no answer' : typeof r.response === 'string' ? r.response : JSON.stringify(r.response);
  return { name: nameOf(r.call_site), ans, by, rec: !!r.teacher?.recorded, i: r.i, ev: votedOf(r, texts) };
}
/** Five rows: the four decisions, then the effect; rows after a stuck site do not exist and read "never ran". @param {RunView} rv @returns {Step[]} */
function stepsOf(rv) {
  const rows = rv.rows, out = /** @type {Step[]} */ ([]);
  for (const v of sites()) {
    const r = rows.find((x) => x.call_site === v.site.id);
    out.push(r ? stepOf(r, rv.evidence_texts) : rv.run.stuck_at === v.site.id ? { name: nameOf(v.site.id), ans: 'no answer', by: 'stuck' } : { name: nameOf(v.site.id), ans: 'never ran', by: '' });
  }
  const eff = rows.find((x) => x.kind === 'effect');
  out.push(eff ? stepOf(eff) : { name: 'post to books', ans: 'never ran', by: '' });
  return out;
}
// The card shows the last trace that arrived until the selected one lands, so it keeps its rows and height (VISUAL_INTENT).
// A reply for an earlier pick still lands while the card is not on the selected run: on a slow API during a batch it keeps up.
async function loadTrace(/** @type {Run} */ run) {
  S.loading = run.id;
  const r = await req('GET', `/runs/${encodeURIComponent(run.id)}`);
  if (S.loading === run.id) S.loading = null;
  if (S.trace?.id !== currentRun()?.id) S.trace = { id: run.id, run, rv: r.ok ? r.data : null, err: r.ok ? '' : r.text };
}
function renderReplay() {
  const has = !!st && (st.runs.length > 0 || !!st.first_run);
  $('#tfull').hidden = !has; $('#tempty').hidden = has; $('#receipts').hidden = !has; /** @type {HTMLElement} */ ($('.rsub')).hidden = !has;
  if (!has) return;
  const run = currentRun();
  if (!run) return;
  if (S.trace?.id !== run.id && S.loading !== run.id) loadTrace(run);
  const t = S.trace, rv = t?.rv ?? null, shown = /** @type {Run} */ (rv?.run ?? t?.run ?? run); // the card's run; the receipts show the pick at once
  const inp = /** @type {any} */ (shown.input) ?? {}, amt = money(+inp.amount || 0), rot = [-1, .8, -.5, 1.1, -.9, .6, -1.2];
  setHTML($('#receipts'), listedRuns().map((x, i) => {
    const xi = /** @type {any} */ (x.input) ?? {};
    return `<li><button class="receipt" type="button" data-r="${esc(x.id)}" aria-pressed="${x.id === run.id}" style="--rot:${rot[i]}deg"><span class="m">${esc(xi.descriptor ?? '')}</span><span class="a">${money(+xi.amount || 0)}</span>` +
      `<span class="meta">run ${int(x.n)} · ${x.stuck_at ? `stuck at ${nameOf(x.stuck_at)}` : `${x.replayable} of ${x.decisions} replayable${x.down.length ? ' · API down' : ''}`}</span></button></li>`;
  }).join(''));
  setText($('#ttitle'), `run ${int(shown.n)}`);
  setText($('#tsub'), `${inp.descriptor ?? ''} · ${amt}`);
  const rp = S.rp && S.rp.id === shown.id ? S.rp : null;
  const rb = /** @type {HTMLButtonElement} */ ($('#rbtn'));
  rb.disabled = !rv || (S.selRun != null && t?.id !== S.selRun) || !!(rp && !rp.done); // a picked receipt waits for its own trace
  setHTML(rb, `${ico('play', 18)}${rp ? (rp.done ? 'replay again' : 'replaying…') : 'replay this run'}`);
  // before the first trace lands: the five step names and nothing else, the same height as a full card
  const steps = rv ? stepsOf(rv) : [...sites().map((v) => v.site.id), 'post_to_books'].map((id) => /** @type {Step} */ ({ name: nameOf(id), ans: '', by: '' }));
  /** @type {Record<number, import('./schema').ReplayStep>} */
  const res = Object.fromEntries((rp?.rep?.steps ?? []).map((/** @type {import('./schema').ReplayStep} */ x) => [x.i, x]));
  setHTML($('#trows'), steps.map((x, i) => {
    const ran = x.by !== '', r = x.i != null ? res[x.i] : undefined, re = !!r?.re_executed;
    const doneRow = rp && rp.rep && rp.i > i, busy = ran && rp && !rp.done && rp.rep && rp.i === i;
    let out = '';
    if (doneRow && ran) {
      out = re ? (r?.same_bytes ? `<span class="ok">${ico('check', 20)}</span><span class="ok">same bytes</span><code>${esc((r.output_sha ?? '').slice(0, 6))}</code>` : '<span class="bad">different bytes</span>')
        : x.by === 'stuck' ? '<span class="log">API was down</span>' : `<span class="log">from the log${x.by === 'ledger' ? ', not re-posted' : ''}</span>`;
    } else if (busy) out = `<span class="busy">${re ? 're-running…' : 'reading the log…'}</span>`;
    return `<li class="trow${busy ? ' busy' : ''}"><b>${x.name}</b><span class="tans">${esc(x.ans)}${x.rec ? '<small>recorded answer</small>' : ''}</span><span class="tby">${ran ? ico(BYICON[x.by] ?? 'moon', 22) + x.by : ''}</span><span class="tres">${out}</span>${x.ev ? `<small class="tev">${esc(x.ev)}</small>` : ''}</li>`;
  }).join(''));
  // no "·" or "—" here: Gaegu lacks them and the fallback glyphs shift this line between states
  const k0 = shown.replayable, n = shown.decisions, before = !!rv && rv.rows.every((/** @type {TraceRow} */ r) => r.phase === 'awake');
  let sum = !rv ? (!t?.err ? '' : /^404 /.test(t.err) ? 'this run is no longer in the log' : `this run could not be read: ${t.err}`)
    : shown.stuck_at ? `this run got stuck at ${nameOf(shown.stuck_at)}`
      : k0 ? `${k0} of ${n} decisions can re-run exactly, press replay`
        : before ? 'none of these decisions can re-run: nothing had learned to dream yet' : 'none of these decisions can re-run: no dream answered in this run';
  if (rp && !rp.done) sum = rp.err ? '' : k0 ? `re-running ${k0} of ${n} decisions…` : 'reading the log…';
  $('#tsum').classList.toggle('pre', !(rp && rp.done));
  if (rp && rp.done) {
    const p = /** @type {ReplayReport | null} */ (rp.rep);
    if (!p) sum = `replay refused: ${rp.err}`;
    else if (shown.stuck_at) sum = p.same ? `${p.same} decision${p.same > 1 ? 's' : ''} re-ran with the same bytes, then the run got stuck at ${nameOf(shown.stuck_at)}` : `stuck at ${nameOf(shown.stuck_at)}: the API was down and nothing had learned it`;
    else if (!p.re_executed) sum = before ? 'nothing re-ran: this run came before anything learned to dream' : 'nothing re-ran: no dream answered in this run';
    // one unit with the line before it ("2 of 4 decisions can re-run"): ReplayReport.from_log also counts the ledger write, a fifth step
    else sum = `${p.same} of ${n} decisions re-ran with the same bytes${p.re_executed > p.same ? `, ${p.re_executed - p.same} did not match` : ''}, the rest came back from the log`;
  }
  setText($('#tsum'), sum);
}
// The 0.5 s row stepping animates a ReplayReport already in hand (README (d)).
// It replays the run the card shows (the newest one it has, when following the newest during a batch); that run stays selected
// and on screen (listedRuns) while the batch moves on.
async function startReplay() {
  const t = S.trace, rv = t?.rv;
  if (!t || !rv || (S.selRun != null && t.id !== S.selRun)) return;
  S.selRun = t.id;
  const rp = S.rp = { id: t.id, run: rv.run, i: 0, done: false, rep: null, err: '' };
  const r = await req('POST', `/replay/${encodeURIComponent(t.id)}`);
  if (S.rp !== rp) return;
  if (!r.ok) { Object.assign(rp, { done: true, err: r.text }); return; }
  rp.rep = r.data;
  const n = stepsOf(rv).length, tick = () => {
    if (S.rp !== rp) return;
    rp.i++;
    if (rp.i >= n) rp.done = true; else setTimeout(tick, 480);
  };
  setTimeout(tick, 480);
}

// ---------- scores: GET /benchmark (apps/api benchmark.py), fetched every time the tab opens ----------
// The page picks the best row and formats; every number is a field of the reply. A missing or null field reads "–".
/** @typedef {{budget: number, coverage: number | null, test_coverage: number | null, test_error: number | null, forward_coverage?: number | null, forward_error?: number | null}} BCell */
/** @typedef {{usd: number, p50_ms: number, p95_ms: number}} BMoney  per 1,000 calls */
/** @typedef {{call_site: string, provider: string, floor: number | null, candidate: string, variant: string, seeds: number, rung: number, stable: boolean, source: string, cells: BCell[], seed_cert_s: number | null, without_dreaming: BMoney | null, with_dreaming: BMoney | null, acc_test?: number | null, acc_forward?: number | null, campaign?: string}} BRow */
/** @typedef {{campaign: string | null, rows: BRow[], notes: string[]}} Bench */
/** @type {{data: Bench | null, err: string, busy: boolean}} */
const SC = { data: null, err: '', busy: false };
/** @type {Record<string, string>} training-plan.md §1, in plain words; the baselines as the benchmark names them */
const PLAIN = { memory: 'Hopfield memory', head: 'linear head', finetune: 'fine-tuned encoder', hdc: 'HDC bundle', exact: 'exact cache', config: 'thinking off',
  rule: 'vendor rule', 'llm alone': 'LLM alone', jev: 'Jev', 'teacher alone': 'teacher alone' };
/** @type {Record<string, string[]>} variant -> [the rows it learned from, before the name, after the name] */
const VARIANT = { 'seed+real': ['seed + real rows', 'seeded ', ''], seed: ['seed rows only', 'seed-only ', ''], real: ['real rows', '', ' on real rows'] };
const at5 = (/** @type {BRow} */ r) => r.cells.find((c) => Math.abs(c.budget - 0.05) < 1e-9);
const isBase = (/** @type {BRow} */ r) => r.source === 'baseline';
const groupsOf = (/** @type {string} */ site) => (site.startsWith('quickbooks.') ? 'companies' : 'merchants'); // what the held-out split holds out
/** @param {BRow} r */
const phrase = (r) => { const v = VARIANT[r.variant] ?? ['', '', ` (${r.variant})`]; return `the ${v[1]}${PLAIN[r.candidate] ?? r.candidate}${v[2]}`; };
/** The highlighted row: highest repeat coverage at 5%, then unseen; with no coverage anywhere, highest accuracy. Null when nobody has a number. @param {BRow[]} rows */
function bestOf(rows) {
  const cands = rows.filter((r) => !isBase(r)), c5 = (/** @type {BRow} */ r) => at5(r) ?? /** @type {Partial<BCell>} */ ({});
  const cov = cands.some((r) => (c5(r).forward_coverage ?? 0) > 0 || (c5(r).test_coverage ?? 0) > 0);
  const key = (/** @type {BRow} */ r) => (cov ? [c5(r).forward_coverage, c5(r).test_coverage] : [r.acc_forward, r.acc_test]).map((x) => x ?? -1);
  let best = null, bk = [-1, -1];
  for (const r of cands) { const k = key(r); if (k[0] > bk[0] || (k[0] === bk[0] && k[1] > bk[1])) { best = r; bk = k; } } // ties keep the server's order
  return best;
}
/** @param {BRow[]} rows @param {BRow | null} best @param {string} site */
function verdict(rows, best, site) {
  const n = rows.filter((r) => !isBase(r)).length, g = groupsOf(site);
  if (!best) return `no promise at 5% yet from ${n === 1 ? 'its one candidate' : `any of the ${n} candidates`}`;
  const c = at5(best), fc = c?.forward_coverage ?? 0, tc = c?.test_coverage ?? 0, who = phrase(best), wrong = (/** @type {number | null | undefined} */ e) => (e == null ? '' : `, wrong ${pf(e)}`);
  if (fc > 0) return `${who} answers ${pf(fc)} of repeat traffic at 5%${wrong(c?.forward_error)}${tc > 0 ? `, and ${pf(tc)} of ${g} it never saw` : ''}`;
  if (tc > 0) return `${who} answers ${pf(tc)} of ${g} it never saw at 5%${wrong(c?.test_error)}`;
  // accuracy is never "answers": answering means holding a certificate
  const right = [best.acc_forward != null ? `${pf(best.acc_forward)} of repeat traffic` : '', best.acc_test != null ? `${pf(best.acc_test)} of ${g} it never saw` : ''].filter(Boolean);
  return `no promise at 5% yet: ${who} is right on ${right.join(' and ')}`;
}
/** coverage at 5% and the error above the cutoff; 0 means nothing certified @param {number | null | undefined} cov @param {number | null | undefined} err @param {boolean} base */
const share = (cov, err, base) => (cov == null ? '–' : !cov ? `<span class="nope">no promise ${base ? 'at 5%' : 'yet'}</span>` : `<b>${pf(cov)}</b>${err == null ? '' : `<small>wrong ${pf(err)}</small>`}`);
/** without -> with Dreaming, the journal's cost idiom; the slash and the green only when it got cheaper or faster @param {number | undefined} a @param {number | undefined} b @param {(x: number) => string} f */
function pair(a, b, f) {
  if (a == null || b == null) return a == null && b == null ? '–' : `<span class="jcost">${f(a ?? /** @type {number} */ (b))}</span>`;
  return b < a && f(b) !== f(a) ? `<span class="jcost"><s>${f(a)}</s>${WOB}<em>${f(b)}</em></span>` : `<span class="jcost">${f(a)} → ${f(b)}</span>`;
}
const usd = (/** @type {number} */ d) => '$' + d.toFixed(2);
const ms = (/** @type {number} */ x) => (x < 10 ? x.toFixed(1) : String(Math.round(x)));
const HOLES = '<span class="hole" style="top:30px"></span><span class="hole" style="bottom:30px"></span>';
const blank = (/** @type {string} */ h, p = '') => `<section class="journal scard"><span class="hole" style="top:30px"></span><div class="empty"><h2>${h}</h2>${p ? `<p>${p}</p>` : ''}</div></section>`;
/** @param {string} site @param {BRow[]} rows */
function scoreCard(site, rows) {
  const best = bestOf(rows), g = groupsOf(site), floor = rows[0].floor;
  const head = '<tr><th>who<small>candidate, rows it learned from</small></th><th>repeat traffic<small>answers, wrong at 5%</small></th>' +
    `<th>unseen ${g}<small>answers, wrong at 5%</small></th><th>right on all calls<small>repeat / unseen</small></th>` +
    '<th>$ per 1,000 calls<small>without → with dreaming</small></th><th>p50 latency<small>ms, without → with</small></th></tr>';
  const body = rows.map((r) => {
    const base = isBase(r), c = at5(r), w = r.without_dreaming, d = r.with_dreaming;
    const sub = [base ? 'baseline' : VARIANT[r.variant]?.[0] ?? r.variant, r.source === 'summary' ? 'from its summary' : '', r.stable ? '' : 'unverified',
      r.variant === 'seed' && r.seed_cert_s != null ? `first promise after ${r.seed_cert_s.toFixed(1)} s` : ''].filter(Boolean).join(', ');
    const tip = 'unseen coverage ' + r.cells.map((x) => `${pf(x.budget)}: ${x.test_coverage == null ? '–' : pf(x.test_coverage)}`).join(', ');
    const acc = r.acc_forward == null && r.acc_test == null ? '–' : `<b>${r.acc_forward == null ? '–' : pf(r.acc_forward)} / ${r.acc_test == null ? '–' : pf(r.acc_test)}</b>`;
    return `<tr${r === best ? ' class="best"' : ''}><td><b>${esc(PLAIN[r.candidate] ?? r.candidate)}</b><small>${esc(sub)}</small></td>` +
      `<td>${share(c?.forward_coverage, c?.forward_error, base)}</td><td title="${esc(tip)}">${share(c?.test_coverage, c?.test_error, base)}</td><td>${acc}</td>` +
      `<td>${pair(w?.usd, d?.usd, usd)}</td><td>${pair(w?.p50_ms, d?.p50_ms, ms)}</td></tr>`;
  }).join('');
  const fine = `repeat = the last 20% of the stream by time, ${g} recur: the production case. unseen = whole ${g} held out. ` +
    `$ and ms are projected from this site's logged teacher calls at the unseen 5% coverage. labels: ${rows[0].provider === 'tool' ? "the tool's own returns" : esc(rows[0].provider)}. campaign ${esc(rows[0].campaign ?? '–')}.` +
    (floor ? ` the labeller disagrees with itself on ${pf(floor)} of identical inputs: no promise below that is measurable.` : '');
  return `<section class="journal scard" aria-label="${esc(nameOf(site))} scores">${HOLES}<h2>${esc(nameOf(site))}</h2><p class="sverdict">${esc(verdict(rows, best, site))}</p>` +
    `<div class="swrap"><table class="stbl"><thead>${head}</thead><tbody>${body}</tbody></table></div><p class="fine">${fine}</p></section>`;
}
function renderScores() {
  const b = SC.data;
  if (!b) { setHTML($('#scards'), SC.err ? blank('the scores could not be read', esc(SC.err)) : blank('reading the scores…')); return; }
  if (!b.rows.length) { setHTML($('#scards'), blank('no campaign has run yet', "a campaign trains every candidate on a site's logged calls and scores it on rows it never saw. its table lands here.")); return; }
  /** @type {Map<string, BRow[]>} */
  const by = new Map();
  for (const r of b.rows) by.set(r.call_site, [...(by.get(r.call_site) ?? []), r]);
  const order = (/** @type {string} */ s) => { const i = DREAMERS.indexOf(short(s)); return i < 0 ? DREAMERS.length : i; }; // workflow order, other sites after
  const cards = [...by].sort(([a], [z]) => order(a) - order(z) || a.localeCompare(z)).map(([site, rows]) => scoreCard(site, rows));
  const notes = b.notes.length ? `<details class="fine snotes"><summary>notes from the benchmark (${b.notes.length})</summary><ul>${b.notes.map((n) => `<li>${esc(n)}</li>`).join('')}</ul></details>` : '';
  setHTML($('#scards'), cards.join('') + notes);
}
async function loadScores() {
  renderScores();
  if (SC.busy) return;
  SC.busy = true;
  const r = await req('GET', '/benchmark');
  SC.busy = false;
  if (r.ok) SC.data = r.data;
  SC.err = r.ok ? '' : `GET /benchmark: ${r.text}`; // a failed refetch keeps the last table on screen
  renderScores();
}

// ---------- routing, sizing, frame ----------
function route() {
  const [v, arg] = (location.hash.slice(1) || 'flows').split('/');
  const views = ['flows', 'dreams', 'replay', 'scores'];
  S.view = views.includes(v) ? v : 'flows';
  if (S.view === 'dreams' && DREAMERS.includes(arg)) S.sel = arg;
  for (const id of views) $('#v-' + id).hidden = id !== S.view;
  document.querySelectorAll('.tab').forEach((t) => (/** @type {HTMLElement} */ (t).dataset.v === S.view ? t.setAttribute('aria-current', 'page') : t.removeAttribute('aria-current')));
  if (st) renderView(); // the tab is filled before it is first painted, not one frame later
  if (S.view === 'flows') fit();
  if (S.view === 'scores') loadScores();
}
function fit() {
  const wrap = $('#stage-wrap'), w = wrap.clientWidth;
  if (!w || S.view !== 'flows') return;
  const top = wrap.getBoundingClientRect().top + scrollY;
  const below = $('#charts').offsetHeight + 16 + 14 + 19;
  const avail = innerHeight - top - below;
  // on a phone the graph stays readable and scrolls sideways inside the canvas instead
  const s = Math.max(w < 700 ? .7 : .42, Math.min(w / 1200, avail / 330, 1.3));
  // fill the canvas like the design board; nodes sit centred in the spare height
  const h = Math.min(Math.max(330 * s, avail), 330 * s + 220);
  const stg = $('#stage');
  stg.style.transform = `scale(${s})`;
  stg.style.left = Math.max(0, (w - 1200 * s) / 2) + 'px';
  stg.style.top = (h - 330 * s) / 2 + 'px';
  wrap.style.height = h + 'px';
}
// the saved total and the counter glide to the newest field value between /state replies; they never show a value the API did not send
const glide = (/** @type {number} */ cur, /** @type {number} */ to, /** @type {number} */ dt) => (Math.abs(to - cur) < .005 ? to : cur + (to - cur) * Math.min(1, dt * 5));
function renderTop(/** @type {number} */ dt) {
  const m = st?.metrics, b = st?.batch;
  if (m) { S.saved = glide(S.saved, m.saved_usd, dt); S.runs = b?.running && !b.paused ? glide(S.runs, m.runs, dt) : m.runs; }
  const saved = m ? money(Math.max(0, S.saved)) : '–'; // before the first /state: nothing to show yet, not a zero
  const what = ruled() ? 'saved at rule-teacher prices' : 'saved so far';
  setText($('#saved'), saved);
  setText($('#savedbox small'), what);
  setAttr($('#savedbox'), 'aria-label', `${saved} ${what}`);
  setText($('#count'), m ? `${int(S.runs)} runs` : '– runs');
  const rb = /** @type {HTMLButtonElement} */ ($('#run')), running = !!b?.running && !b.paused;
  rb.disabled = !st; // no run button until the page knows whether a batch is running
  const label = b?.running ? (b.paused ? 'resume' : 'pause') : m?.runs ? `run ${int(BATCH)} more` : `run ${int(BATCH)}`;
  setHTML(rb, ico(running ? 'pause' : 'play', 20) + label);
  rb.classList.toggle('on', running);
  const n = $('#narrate');
  n.classList.toggle('idle', S.narrIdle);
  if (n.dataset.t !== S.narr) { n.dataset.t = S.narr; n.innerHTML = `<span>${S.narr}</span>`; }
}

let lastT = now();
function frame(/** @type {number} */ t) {
  const dt = Math.min(.1, (t - lastT) / 1000);
  lastT = t;
  if (st) watchPhases();
  renderTop(dt);
  if (st) renderView();
  requestAnimationFrame(frame);
}
function renderView() {
  if (S.view === 'flows') { renderStage(); renderCharts(); }
  else if (S.view === 'dreams') { renderJournal(); setHTML($('#cert'), certHTML(S.sel)); }
  else if (S.view === 'replay') renderReplay();
}

let runBusy = false; // a double click sends one command (a second POST /batch would be a 409)
$('#run').addEventListener('click', async () => {
  if (runBusy || !st) return;
  runBusy = true;
  const b = st.batch, runs = () => int(st?.metrics.runs ?? 0); // read after the reply: the batch moves ~100 runs/s offline
  /** @param {Partial<import('./schema').Batch>} patch @param {() => void} [then] */
  const replied = (patch, then) => () => { if (st) st.batch = { ...st.batch, ...patch }; runBusy = false; then?.(); };
  // during an outage the outage line stays up: it is what explains the runs that keep going (or don't).
  // pause and resume are shown with show(), not narrate(): verdicts still queued follow them instead of being dropped
  if (b.running) {
    const pause = !b.paused, at = runs();
    const then = replied({ paused: pause }, () => { if (!pause && !S.unplugged) show(`resuming at run ${at}`, 2); });
    const r = pause ? await cmd('POST', '/batch/pause', undefined, then) : await cmd('POST', '/batch/resume', undefined, then);
    runBusy = false;
    if (!r.ok) narrate(`the run did not ${pause ? 'pause' : 'resume'}: ${esc(r.text)}`);
    else if (pause && !S.unplugged) show(`paused at run ${runs()}`, 2);
    return;
  }
  S.rp = null; // a new batch: a replayed run gives up its place, and the selection falls back to the newest once it leaves the screen
  if (!S.unplugged) narrate(st?.metrics.runs ? `fast-forwarding ${int(BATCH)} more transactions · ${teacherWords()}` : `fast-forwarding ${int(BATCH)} transactions · ${teacherWords()} · every call is logged`);
  const r = await cmd('POST', '/batch', { n: BATCH }, replied({ running: true, paused: false, target: BATCH, done: 0 }));
  runBusy = false;
  if (!r.ok) narrate(`the run did not start: ${esc(r.text)}`);
});
$('#budget').addEventListener('input', () => onBudget(false));
$('#budget').addEventListener('change', () => onBudget(true));
const pickRun = (/** @type {Event} */ e) => { const b = /** @type {HTMLElement | null} */ (/** @type {HTMLElement} */ (e.target).closest('.receipt')); if (b && b.dataset.r !== S.selRun) { S.selRun = b.dataset.r; S.rp = null; } };
$('#receipts').addEventListener('pointerdown', pickRun);
$('#receipts').addEventListener('click', pickRun);
$('#rbtn').addEventListener('click', startReplay);
addEventListener('hashchange', route);
addEventListener('resize', fit);
document.fonts && document.fonts.ready.then(fit);
// web fonts change text heights after load; re-fit whenever the pieces around the graph resize
const ro = new ResizeObserver(() => fit());
['.top', '.canvas-head', '#charts'].forEach((sel) => ro.observe($(sel)));

buildStage(); buildCharts();
narrate(`a bookkeeping agent, wrapped not rewritten · press run to fast-forward ${int(BATCH)} transactions`, true);
route();
requestAnimationFrame(frame);
const hz = healthz();
setInterval(healthz, 15000);
refresh().then(() => { live(st?.config ?? {}); hz.then(greet); });
