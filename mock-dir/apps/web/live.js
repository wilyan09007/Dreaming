// @ts-check
// Live updates for the dashboard. Supabase Realtime first; when the channel errors or closes, poll GET /events?after= every 2 s.
// Contract: apps/web/README.md (b). Types come from ./schema.d.ts, which `just schema` generates from contracts.py.
// @ts-ignore - ESM build from a CDN URL has no type declarations; supabase-js is pinned to the version the research brief names
import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2.117.1/+esm';

/** @typedef {import('./schema').Event} DEvent */
/** @typedef {import('./schema').Policy} DPolicy */
/** @typedef {import('./schema').Run} DRun */
/** @typedef {import('./schema').Certificate} DCertificate */
/** @typedef {{table: 'events', type: 'INSERT' | 'BACKFILL' | 'POLL', row: DEvent}
 *  | {table: 'policies', type: 'INSERT' | 'UPDATE', row: DPolicy}
 *  | {table: 'certificates', type: 'INSERT' | 'UPDATE', row: DCertificate}
 *  | {table: 'resync', type: 'RESYNC', row: null}} Live  `resync` means: GET /state again, a gap may have dropped row changes */

/**
 * @param {string} supabaseUrl      https://<ref>.supabase.co
 * @param {string} publishableKey   sb_publishable_..., role anon (read-only on 5 tables)
 * @param {string} apiBase          '' when the API serves the page
 * @param {(m: Live) => void} onEvent
 * @returns {() => void} stop
 */
export function subscribe(supabaseUrl, publishableKey, apiBase, onEvent) {
  const sb = createClient(supabaseUrl, publishableKey, {
    realtime: { worker: true, heartbeatCallback: (/** @type {string} */ s) => { if (s === 'disconnected') sb.realtime.connect(); } },
  });
  const seen = new Set(); // event ids already handed to onEvent: backfill, poll and live inserts overlap
  let last = '';
  /** @type {ReturnType<typeof setInterval> | 0} */ let poll = 0; // browsers return a number, Node's types a Timeout: the type covers both
  /** @param {DEvent} e @param {'INSERT' | 'BACKFILL' | 'POLL'} type */
  const push = (e, type) => { if (!seen.has(e.id)) { seen.add(e.id); last = e.id; onEvent({ table: 'events', type, row: e }); } };
  /** @param {string} query @returns {Promise<DEvent[]>} oldest first */
  const get = async (query) => { try { const r = await fetch(`${apiBase}/events?${query}`); return r.ok ? await r.json() : []; } catch { return []; } };
  const backfill = async () => (await get('limit=200')).forEach((e) => push(e, 'BACKFILL'));
  const tick = async () => {
    const es = await get(last ? `after=${encodeURIComponent(last)}&limit=200` : 'limit=200');
    es.forEach((e) => push(e, 'POLL'));
    if (es.length) onEvent({ table: 'resync', type: 'RESYNC', row: null }); // /events carries no policy, run or certificate rows
  };
  const ch = sb.channel('dreaming', { config: { postgres_changes_options: { wait: true } } })
    .on('postgres_changes', { event: 'INSERT', schema: 'public', table: 'events' }, (/** @type {any} */ p) => push(p.new, 'INSERT'));
  // not runs: 100+ inserts/s in fast-forward would hit the Realtime cap; the page refetches /state on batch.progress
  for (const table of /** @type {const} */ (['policies', 'certificates']))
    ch.on('postgres_changes', { event: '*', schema: 'public', table }, (/** @type {any} */ p) => { if (p.eventType !== 'DELETE') onEvent({ table, type: p.eventType, row: p.new }); });
  ch.subscribe((/** @type {string} */ status) => {
    if (status === 'SUBSCRIBED') {
      clearInterval(poll); poll = 0; // Postgres Changes has no replay: after every (re)subscribe, re-read the gap and dedupe by id
      onEvent({ table: 'resync', type: 'RESYNC', row: null });
      backfill();
    } else if (['CHANNEL_ERROR', 'TIMED_OUT', 'CLOSED'].includes(status) && !poll) poll = setInterval(tick, 2000);
  });
  return () => { clearInterval(poll); sb.removeChannel(ch); };
}
