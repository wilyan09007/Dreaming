// @ts-check
// Developer view: exercises the contract in apps/web/README.md (GET /state, live events or polling, the four commands).
// Every number on the page is a field of an API reply; this file formats and never computes. demo/index.html is the designed page.
/** @typedef {import('./schema').Policy} Policy */
/** @typedef {import('./schema').Run} Run */
/** @typedef {import('./schema').Certificate} Certificate */
/** @typedef {import('./schema').Metrics} Metrics */
/** @typedef {import('./schema').CallSite} CallSite */
/** @typedef {import('./schema').Event} DEvent */
/** @typedef {import('./schema').ReplayReport} ReplayReport */
/** @typedef {import('./schema').Health} Health */
/** @typedef {import('./live.js').Live} Live */
// SiteView, Batch and State are not in schema.json yet (the justfile's union omits them): mirrored from contracts.py, test_web.py checks the field names.
/** @typedef {import('./schema').SiteView} SiteView */
/** @typedef {import('./schema').Batch} Batch */
/** @typedef {import('./schema').State} State */

const apiBase = ''; // the API serves this page: same origin
/** @type {(id: string) => HTMLElement} */
const $ = (id) => /** @type {HTMLElement} */ (document.getElementById(id));
const budget = /** @type {HTMLInputElement} */ ($('budget'));
const unplug = /** @type {HTMLInputElement} */ ($('unplug'));
const esc = (/** @type {unknown} */ s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const pct = (/** @type {number} */ fraction) => `${+(fraction * 100).toFixed(1)}%`;
const cents = (/** @type {number} */ usd) => `${+(usd * 100).toFixed(2)}¢`;
/** @param {unknown[]} cells @param {string} [button] raw HTML of a last cell */
const tr = (cells, button = '') => `<tr>${cells.map((c) => `<td>${esc(c)}</td>`).join('')}${button && `<td>${button}</td>`}</tr>`;

/** @type {State | null} */
let state = null;

// ---- every request goes through here: a failure becomes one line of text (FastAPI's `detail`), never a throw
/** @param {'GET' | 'POST'} method @param {string} path @param {object} [body]
 *  @returns {Promise<{ok: boolean, data: any, text: string}>} */
async function req(method, path, body) {
  try {
    const r = await fetch(apiBase + path, { method, headers: body ? { 'content-type': 'application/json' } : {}, body: body && JSON.stringify(body) });
    const data = await r.json().catch(() => null);
    const why = typeof data?.detail === 'string' ? data.detail : JSON.stringify(data?.detail ?? '');
    return { ok: r.ok, data, text: r.ok ? '' : `${method} ${path}: ${r.status} ${why}` };
  } catch (e) {
    return { ok: false, data: null, text: `${method} ${path}: ${e}` };
  }
}

/** A command: show the API's refusal (409 "a batch is already running", 503, 404), then re-read /state.
 *  @param {'GET' | 'POST'} method @param {string} path @param {object} [body] */
async function cmd(method, path, body) {
  const r = await req(method, path, body);
  $('msg').textContent = r.text;
  refresh();
  return r;
}

// ---- the error strip: /healthz checks that are red, any component.error event, a failed /state
const problems = { health: /** @type {string[]} */ ([]), events: /** @type {string[]} */ ([]), state: '' };
function strip() {
  const all = [...problems.health, ...problems.events, ...(problems.state ? [problems.state] : [])];
  $('errors').hidden = !all.length;
  $('errors').textContent = all.join('\n');
}
async function health() {
  const r = await req('GET', '/healthz');
  const checks = /** @type {Health} */ (r.data)?.checks ?? {};
  problems.health = r.ok ? Object.entries(checks).filter(([, v]) => !/^(ok|configured)\b/.test(v)).map(([k, v]) => `healthz ${k}: ${v}`) : [r.text];
  strip();
}

// ---- /state, coalesced: one request in flight, at most one queued, 250 ms apart (a batch emits many events)
let busy = false, again = false;
async function refresh() {
  if (busy) { again = true; return; }
  busy = true;
  try {
    do {
      again = false;
      const r = await req('GET', '/state');
      problems.state = r.text;
      strip();
      if (r.ok) { state = r.data; render(r.data); }
      if (again) await new Promise((done) => setTimeout(done, 250));
    } while (again);
  } finally { busy = false; }
}

/** @param {State} s */
function render(s) {
  const b = s.batch;
  $('batch').textContent = `batch: ${b.running ? (b.paused ? 'paused' : 'running') : b.target ? 'done' : 'idle'} ${b.done}/${b.target}`
    + (b.recorded ? ' · pre-recorded teacher answers (seconds are virtual-clock time)' : '') + (b.live_calls > 0 ? ` (${b.live_calls} live calls)` : '');
  $('pause').textContent = b.paused ? 'resume' : 'pause';
  /** @type {HTMLButtonElement} */ ($('pause')).disabled = !b.running;

  const m = s.metrics;
  $('metrics').innerHTML = [['runs', m.runs], ['cost per run', cents(m.cost_per_run)], ['seconds per run', +m.sec_per_run.toFixed(2)],
    ['failed', `${+m.failed_pct.toFixed(1)}%`], ['replayable', `${+m.replayable_pct.toFixed(1)}%`], ['saved', `$${m.saved_usd.toFixed(2)}`]]
    .map(([k, v]) => `<div><small>${k}</small><b>${esc(v)}</b></div>`).join('');

  $('sites').innerHTML = s.sites.map(({ site, policy: p, certificate: c, cost_awake_usd: awake, cost_dream_usd: dream }) => {
    const a = p.attempt;
    const promise = c ? `wrong ≤ ${pct(c.budget)}, measured ${pct(c.emp_error)}, bound ${pct(c.bound)}, coverage ${pct(c.coverage)}`
      : a ? `no promise at ${pct(a.budget)}: disagreed ${pct(a.emp_error)}, could be ${pct(a.bound)}, allow ${a.min_budget == null ? 'none up to 20%' : pct(a.min_budget)}`
      : 'no promise yet';
    return tr([site.name, p.outage ? 'down' : p.phase, `${p.rows} / ${p.shadow_rows} / ${p.predicted_rows}`, promise,
      !p.rows ? '–' : p.predicted_rows ? `${cents(awake)} → ${cents(dream)}` : cents(awake), `${p.audit_errors}/${p.audit_rows}`]);
  }).join('');

  $('runs').innerHTML = s.runs.slice(-10).reverse().map((r) => tr([r.n, cents(r.cost_usd ?? 0), `${((r.latency_ms ?? 0) / 1000).toFixed(2)} s`,
    r.stuck_at ? `stuck at ${r.stuck_at}` : r.failed ? 'failed' : 'ok', `${r.replayable}/${r.decisions}`, (r.down ?? []).join(', ')],
    `<button data-run="${esc(r.id)}">replay</button>`)).join('');

  if (document.activeElement !== budget) { budget.value = String(Math.round(s.budget * 100)); $('budget-out').textContent = `${budget.value}%`; }
  unplug.disabled = !vendor();
  if (vendor()?.policy.outage) unplug.checked = true; // the detected outage; the switch itself is not in /state (README, open gap 3)
}

const vendor = () => state?.sites.find((v) => v.site.name === 'vendor_lookup');

// ---- the four commands
$('run').onclick = () => cmd('POST', '/batch', { n: 2000 });
$('pause').onclick = () => (state?.batch.paused ? cmd('POST', '/batch/resume') : cmd('POST', '/batch/pause'));
budget.oninput = () => { $('budget-out').textContent = `${budget.value}%`; };
budget.onchange = () => cmd('POST', '/budget', { budget: Number(budget.value) / 100 });
unplug.onchange = async () => {
  const v = vendor();
  if (v && !(await cmd('POST', `/tools/${encodeURIComponent(v.site.id)}/outage`, { down: unplug.checked })).ok) unplug.checked = !unplug.checked;
};

/** The replay button: re-execute the run's predictions and show whether the bytes are the same. */
$('runs').onclick = async (ev) => {
  const id = /** @type {HTMLElement} */ (ev.target).dataset?.run;
  if (!id) return;
  const r = await cmd('POST', `/replay/${encodeURIComponent(id)}`);
  /** @type {ReplayReport} */
  const p = r.data;
  $('replay').textContent = !r.ok ? r.text : `run ${p.run_id}: ${p.same} of ${p.re_executed} re-executed answers have the same bytes, ${p.from_log} served from the log\n`
    + p.steps.map((x) => `${x.i}. ${x.call_site} ${x.decision} ${x.re_executed ? `re-executed ${x.same_bytes ? 'same bytes' : 'DIFFERENT bytes'} ${x.output_sha?.slice(0, 6) ?? ''}` : 'from the log'}`).join('\n');
};

// ---- events: the feed, and which of them mean "read /state again"
const seen = new Set(); // Event.id: backfill, poll and live inserts overlap. Ids are random: never compare, keep arrival order
let last = '';
/** @param {DEvent} e @returns {boolean} true when the event is new and changes something /state shows */
function addEvent(e) {
  if (seen.has(e.id)) return false;
  seen.add(e.id);
  last = e.id;
  const li = document.createElement('li');
  li.textContent = `${e.ts.slice(11, 19)} ${e.component} ${e.kind}${e.call_site ? ` ${e.call_site}` : ''}: ${e.message}${e.error ? ` [error: ${e.error}]` : ''}`;
  $('feed').prepend(li);
  while ($('feed').children.length > 100) $('feed').lastElementChild?.remove();
  if (e.kind === 'component.error') { problems.events = [...problems.events, `${e.component} ${e.kind}: ${e.error ?? e.message}`].slice(-3); strip(); }
  return /^(batch\.|site\.phase|cert\.|tool\.|budget\.changed)/.test(e.kind);
}

/** Polling: the first call reads the newest events, then `after=<last id>` every 2 s. */
async function poll() {
  const r = last ? await req('GET', `/events?after=${encodeURIComponent(last)}&limit=200`) : await req('GET', '/events?limit=200');
  if (r.ok && /** @type {DEvent[]} */ (r.data).map(addEvent).includes(true)) refresh();
  setTimeout(poll, 2000);
}

/** @param {Record<string, string>} config */
async function live(config) {
  const url = config.supabase_url, key = config.supabase_publishable_key;
  if (url && key) {
    try {
      const { subscribe } = await import('./live.js'); // dynamic: offline, the CDN import inside live.js must not take the page down
      subscribe(url, key, apiBase, (/** @type {Live} */ m) => { if (m.table !== 'events' || addEvent(m.row)) refresh(); }); // resync, policies, certificates: read /state again
      $('mode').textContent = 'live: Supabase Realtime (live.js polls /events if the channel drops)';
      return;
    } catch (e) { problems.events = [...problems.events, `realtime unavailable, polling instead: ${e}`].slice(-3); strip(); }
  }
  $('mode').textContent = 'live: polling GET /events every 2 s';
  poll();
}

health();
setInterval(health, 15000);
refresh().then(() => live(state?.config ?? {}));
