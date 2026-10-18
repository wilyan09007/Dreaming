// Screen 6 "build": the judge edits the workflow; each step snaps to the nearest measured site, the lanes redraw with
// the snapped steps, and "Run it" plays screen 2's race (Launch.race, from race.js) with totals composed from them.
(() => {
  const R = Launch.race, MAX = 12;
  const DEFAULTS = ['categorize the transaction', 'look up the vendor', 'match it to an invoice', 'anything odd?', 'post to the books'];
  const CHIPS = ['categorize', 'vendor lookup', 'match invoice', 'anything odd?', 'post to books', 'wake a human', 'summarize email'];
  // synonym table: words that point at a measured site (wake_a_human = human handoff, never dreamed)
  const SYN = {
    categorize: 'categorize categorise classify category tag label sort bucket expense type code',
    vendor_lookup: 'vendor lookup look up supplier merchant find enrich company',
    match_invoice: 'match invoice reconcile po purchase order bill receipt',
    anything_odd: 'anything odd anomaly fraud flag suspicious check weird unusual duplicate risk',
    post_to_books: 'post books ledger record journal quickbooks save book write entry',
    wake_a_human: 'wake human escalate approve approval ask manager handoff notify person review',
  };
  const STOP = new Set('a an the to it of and for with on in is if this that my our'.split(' '));
  const human = { id: 'wake_a_human', label: 'wake a human', kind: 'human', teacherMs: 0, oursMs: 0, teacherUsd1k: 0, oursUsd1k: 0, coverage: 0 };
  const llm = N.sites.filter((s) => s.kind === 'llm'), avg = (f) => llm.reduce((a, s) => a + s[f], 0) / llm.length;
  const site = (id) => (id === 'wake_a_human' ? human : N.byId[id]);

  const words = (s) => s.toLowerCase().match(/[a-z0-9]+/g)?.filter((w) => !STOP.has(w)) || [];
  const tri = (w) => { const p = ` ${w} `, out = new Set(); for (let i = 0; i < p.length - 2; i++) out.add(p.slice(i, i + 3)); return out; };
  const dice = (a, b) => { let n = 0; a.forEach((g) => b.has(g) && n++); return (2 * n) / (a.size + b.size); };
  const SYNW = Object.fromEntries(Object.entries(SYN).map(([id, s]) => [id, s.split(' ').map((w) => [w, tri(w)])]));
  // nearest site: exact synonym hits score high, trigram similarity catches typos and variants
  function snap(text) {
    const ws = words(text);
    if (!ws.length) return null;
    let best = { id: null, score: 0 };
    for (const id in SYNW) {
      const hits = ws.filter((w) => SYNW[id].some(([s]) => s === w)).length;
      let fuzzy = 0;
      for (const w of ws) { const t = tri(w); for (const [, st] of SYNW[id]) fuzzy = Math.max(fuzzy, dice(t, st)); }
      const score = Math.max(hits ? 0.6 + (0.4 * hits) / ws.length : 0, 0.85 * fuzzy);
      if (score > best.score) best = { id, score };
    }
    return best;
  }
  // smoke check of the snapper, runs once at load
  console.assert(snap('classify the expense').id === 'categorize' && snap('invoce matching').id === 'match_invoice' && snap('escalate to a manager').id === 'wake_a_human' && snap('summarize email').score < 0.35 && snap('   ') === null, 'build snap');

  function pill(m) {
    if (!m) return ['blank', '', 'type a step'];
    const s = site(m.id), pct = Math.round(m.score * 100);
    if (m.score < 0.35) return ['new', '', 'new step: the teacher answers until a stand-in certifies'];
    const bar = m.score < 0.6 ? `<i class="sim" style="--p:${pct}%"></i>` : '';
    const what = s.kind === 'human' ? 'human handoff, never dreamed' : s.coverage ? `${s.cand} serves ${Math.round(s.coverage * 100)}%` : s.cand;
    return [s.kind === 'human' ? 'human' : m.score < 0.6 ? 'near' : 'ok', bar, `${m.score < 0.6 ? `closest (${pct}%):` : '~'} ${s.label}, ${what}`];
  }
  function step(text, m) {
    if (m.score >= 0.35) return site(m.id);
    // unmatched: the teacher answers on both lanes until a stand-in certifies (cost/latency = mean measured LLM site)
    return { label: text.trim(), kind: 'llm', fallback: true, teacherMs: avg('teacherMs'), oursMs: avg('teacherMs'), teacherUsd1k: avg('teacherUsd1k'), oursUsd1k: avg('teacherUsd1k'), coverage: 0 };
  }

  let el, list, box, runBtn, addBtns, cur, pending;
  const rows = () => [...list.children];

  function row(text) {
    const r = document.createElement('li');
    r.innerHTML = `<span class="n num"></span><input type="text" maxlength="200" spellcheck="false" aria-label="step"><span class="pill"></span><button data-a="up" aria-label="move up">&uarr;</button><button data-a="down" aria-label="move down">&darr;</button><button data-a="del" aria-label="remove">&times;</button>`;
    r.querySelector('input').value = text;
    return r;
  }
  function add(text, after) {
    if (rows().length >= MAX) return;
    const r = row(text);
    list.insertBefore(r, after ? after.nextSibling : null);
    refresh();
    r.querySelector('input').focus();
  }

  // re-snap every step and redraw the lanes; typing only touches pills and the right column, never the inputs
  function refresh() {
    const rs = rows(), steps = [];
    rs.forEach((r, i) => {
      const text = r.querySelector('input').value, m = snap(text), [cls, bar, label] = pill(m);
      r.querySelector('.n').textContent = i + 1;
      const p = r.querySelector('.pill');
      p.className = 'pill ' + cls; p.innerHTML = bar + `<span></span>`; p.lastChild.textContent = label; p.title = label;
      if (m) steps.push(step(text, m));
    });
    addBtns.forEach((b) => (b.disabled = rs.length >= MAX));
    runBtn.disabled = !steps.length;
    el.classList.toggle('empty', !steps.length);
    if (steps.length) R.mount(box, steps); else box.replaceChildren();
  }
  const later = () => { clearTimeout(pending); pending = cur ? cur.after(150, refresh) : setTimeout(refresh, 150); };

  Launch.add({
    id: 'build',
    title: 'Build your own workflow',
    notes: 'Hand the keyboard to a judge: edit any step, add one, or tap a chip.\nEach step snaps to the nearest measured site; "wake a human" stays human, unknown steps stay on the teacher.\nHit Run it: the same race as screen 2, totals composed from their steps.',
    css: R.css('#s-build') + `
#s-build .head{display:flex;align-items:baseline;gap:24px;margin-bottom:14px}
#s-build .title{font-size:52px}
#s-build .grid{display:grid;grid-template-columns:640px 1fr;gap:24px;height:calc(100% - 80px)}
#s-build .ed{padding:14px 16px;display:flex;flex-direction:column;gap:10px;min-height:0}
#s-build ol{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:4px}
#s-build li{display:grid;grid-template-columns:24px 204px 1fr 30px 30px 30px;gap:6px;align-items:center;height:40px}
#s-build .n{font-family:var(--hand);font-weight:700;font-size:20px;color:var(--muted);text-align:right}
#s-build input{width:100%;min-width:0;height:36px;font:17px var(--body);padding:0 10px;border:2px solid var(--ink);border-radius:10px;background:#fff;color:var(--ink)}
#s-build input:focus{outline:3px solid var(--lav);outline-offset:1px}
#s-build .pill{display:flex;align-items:center;gap:6px;min-width:0;height:38px;padding:0 10px;border-radius:12px;border:2px solid transparent;font-size:16px;line-height:1.08;overflow:hidden}
#s-build .pill span{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;overflow-wrap:anywhere}
#s-build .pill.ok{background:var(--lilac);color:var(--purple)} #s-build .pill.near{background:var(--sky)}
#s-build .pill.new{background:var(--peach);border-color:var(--teacher);color:#9a4330} #s-build .pill.human{background:var(--sky);border:2px dashed var(--ink)}
#s-build .pill.blank{color:var(--faint);border:2px dashed var(--dot)}
#s-build .sim{flex:none;width:40px;height:8px;border-radius:4px;border:1.5px solid var(--ink);background:linear-gradient(90deg,var(--purple) var(--p),#fff 0)}
#s-build li button,#s-build .add button{border:2px solid var(--ink);border-radius:9px;background:#fff;cursor:pointer;font-size:17px;height:32px;padding:0}
#s-build li button:hover,#s-build .add button:hover{background:var(--yellow)}
#s-build .add{display:flex;flex-wrap:wrap;gap:6px;margin-top:auto}
#s-build .add button{padding:0 12px;border-radius:999px;font-size:16px}
#s-build .add button:first-child{background:var(--yellow)}
#s-build button:disabled{opacity:.4;cursor:default}
#s-build .rt{display:flex;flex-direction:column;gap:10px;min-width:0}
#s-build .bar{display:flex;align-items:center;gap:16px}
#s-build .run{font-family:var(--hand);font-weight:700;font-size:28px;padding:4px 26px;border:2.5px solid var(--ink);border-radius:14px;background:var(--ours);color:#fff;box-shadow:var(--lift);cursor:pointer}
#s-build .run:disabled{background:var(--faint);opacity:.6;cursor:default}
#s-build .bar span{font-size:17px;color:var(--muted)}
#s-build .rc{gap:10px}
#s-build .lane{grid-template-columns:1fr;padding:8px 18px 8px}
#s-build .ln,#s-build .ln b{font-size:23px}
#s-build .watch output{font-size:24px;min-width:80px} #s-build .dn{font-size:20px}
#s-build .track{height:112px}
#s-build .dot{width:46px;height:46px;font-size:21px}
#s-build .rail,#s-build .fill{top:30px}
#s-build .lab{font-size:16px} #s-build .lat{font-size:19px}
#s-build .pole{height:54px}
#s-build .stats{grid-column:1;grid-row:auto;width:auto;grid-template-columns:auto repeat(4,1fr);align-items:end;border-left:0;border-top:2px dashed var(--dot);padding:6px 0 0}
#s-build .stats em{grid-column:auto;font-size:20px} #s-build .stats b{font-size:28px}
#s-build .verdict{font-size:24px;padding:8px 22px} #s-build .verdict b{font-size:40px}
#s-build .none{display:none;flex:1;place-items:center;text-align:center;border:2.5px dashed var(--dot);border-radius:18px;font-family:var(--hand);font-size:30px;color:var(--muted);padding:24px}
#s-build.empty .none{display:grid}`,
    mount(e) {
      el = e;
      el.innerHTML = `<div class="head"><h2 class="title">Build your own workflow</h2><span class="kicker">type a step: it snaps to the nearest one we have measured</span></div>
<div class="grid"><div class="ed card"><ol></ol><div class="add"><button data-chip="">+ add step</button>${CHIPS.map((c) => `<button data-chip="${c}">${c}</button>`).join('')}</div></div>
<div class="rt"><div class="bar"><button class="run">Run it &#9654;</button><span>webhook trigger first, then your steps</span></div><div class="none">No steps yet.<br>Add one on the left, or tap a chip.</div><div class="box"></div></div></div>`;
      list = el.querySelector('ol'); box = el.querySelector('.box'); runBtn = el.querySelector('.run');
      addBtns = [...el.querySelectorAll('.add button')];
      list.addEventListener('input', later);
      list.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' && ev.target.matches('input')) add('', ev.target.parentNode); });
      list.addEventListener('click', (ev) => {
        const b = ev.target.closest('button'); if (!b) return;
        const r = b.parentNode, a = b.dataset.a;
        if (a === 'del') { const next = r.nextElementSibling || r.previousElementSibling; r.remove(); (next ? next.querySelector('input') : addBtns[0]).focus(); }
        else if (a === 'up' && r.previousElementSibling) list.insertBefore(r.previousElementSibling, r.nextElementSibling); // move the neighbour, never the focused row (moving it would blur the button)
        else if (a === 'down' && r.nextElementSibling) list.insertBefore(r.nextElementSibling, r);
        refresh();
      });
      el.querySelector('.add').addEventListener('click', (ev) => { const b = ev.target.closest('button'); if (b) add(b.dataset.chip); });
      runBtn.addEventListener('click', () => { if (!cur || runBtn.disabled) return; refresh(); R.play(cur, box, 150); });
    },
    enter(e, run) { cur = run; list.replaceChildren(...DEFAULTS.map(row)); refresh(); },
    leave() { cur = null; clearTimeout(pending); R.reset(box); },
    // Space on a focused build button presses it instead of advancing the deck
    key(ev) { if (ev.key === ' ' && ev.target.closest && ev.target.closest('#s-build button')) { ev.target.click(); return true; } },
  });
})();
