// Screen 5: the dream campaign. Every candidate x LLM site x seed is a cell; a fixed schedule prunes cells rung by
// rung (60 -> 2,000 rows) and stamps one certified winner per site. Deterministic on purpose: same show every time.
(() => {
  const sites = N.sites.filter((s) => s.kind === 'llm');
  const seeds = N.cells / (N.candidates.length * sites.length); // 45 / (5 x 3) = 3
  // per candidate (N.candidates order) x site: the rung each seed is pruned at, W = that site's certified winner.
  // ponytail: hand-written schedule; winners match N.sites[].cand (categorize/match = Psi, anything odd = linear head)
  const PLAN = [
    ['122', '011', 'W32'], // linear head
    ['W21', '2W3', '211'], // Psi
    ['101', '110', '011'], // fine-tuned encoder (GPU nondeterminism: replay is not byte-identical)
    ['000', '001', '100'], // hdc bundle
    ['000', '000', '010'], // exact cache (unseen merchants miss)
  ];
  const BOUND = [[9.8, 7.1, 5.6, 4.1], [12.4, 8.3, 6.4, 4.8], [11.1, 7.9, 6.0, 4.6]]; // best live bound per site per rung, %
  const why = (cand, r) => (cand === 'fine-tuned encoder' ? 'byte-unstable' : r >= 2 ? 'worse than spare' : 'error > budget');
  const REASONS = ['error > budget', 'byte-unstable', 'worse than spare'];
  const LAST = N.rungs.length - 1, COL = 390, SLOT = COL / N.rungs.length, CHIP = 80, PITCH = 30;
  const x = (slot) => slot * SLOT + (SLOT - CHIP) / 2;
  const fmtRows = (n) => n.toLocaleString();
  const winners = sites.length, total = N.cells;
  let cells = [], heads = [], ui = {};

  const h = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const place = (c, slot, dead) => { c.el.style.transform = `translate(${x(slot)}px,${dead ? 4 : 0}px) rotate(${dead ? -4 : 0}deg)`; };
  const setCounts = (pruned) => { ui.alive.textContent = total - pruned; ui.pruned.textContent = pruned; }; // one write: always sums to N.cells

  function reset(el) {
    el.classList.add('still'); // no transitions while snapping back
    el.classList.remove('play', 'done');
    delete el.dataset.rung;
    cells.forEach((c) => { c.el.classList.remove('dead', 'gone', 'win'); c.el.style.transitionDelay = ''; c.tag.classList.remove('on'); place(c, 0); c.stamp && c.stamp.classList.remove('on'); });
    heads.forEach((b) => { b.out.textContent = '--'; b.head.classList.remove('ok'); });
    setCounts(0);
    ui.tally.forEach((t) => (t.n.textContent = 0));
    ui.status.textContent = `all ${total} cells start on ${N.rungs[0]} rows`;
  }

  Launch.add({
    id: 'train',
    title: 'train: the dream campaign',
    notes: `${total} cells: ${N.candidates.length} candidates x ${sites.length} LLM sites x ${seeds} seeds, all start on ${N.rungs[0]} rows.\n` +
      'Point at the red chips: hard rules prune them on their own, each with a reason tag.\n' +
      `Survivors climb to ${N.rungs.slice(1).map(fmtRows).join(', ')} rows; watch each column's bound shrink.\n` +
      `One certified winner per site, under $${Math.ceil(N.gpuUsd)} of GPU for the whole campaign.`,
    css: `
#s-train .hd{position:absolute;left:72px;top:96px}
#s-train .title{margin-top:4px}
#s-train .ctr{position:absolute;right:72px;top:92px;display:flex;gap:18px;align-items:flex-end}
#s-train .ctr div{width:170px;padding:6px 16px 4px;text-align:center}
#s-train .ctr span{display:block;font-size:20px;color:var(--muted)}
#s-train .ctr output{display:block;font-size:64px;line-height:1}
#s-train .alive output{color:var(--ours)}
#s-train .pruned output{color:var(--red)}
#s-train .st{position:absolute;right:72px;top:196px;font-size:18px;color:var(--muted);text-align:right}
#s-train .grid{position:absolute;left:72px;top:228px;display:grid;grid-template-columns:208px repeat(${sites.length},${COL}px);column-gap:26px}
#s-train .head{display:flex;justify-content:space-between;align-items:baseline;border-bottom:2.5px solid var(--ink);padding:0 2px 2px}
#s-train .head b{font-size:28px}
#s-train .head span{font-size:20px;color:var(--muted)}
#s-train .head output{font-size:26px;color:var(--ours);display:inline-block;min-width:62px;text-align:right}
#s-train .head.ok output,#s-train .head.ok i{color:var(--green)}
#s-train .head i{font-style:normal;font-family:var(--hand);font-weight:700;font-size:22px;opacity:0;transition:opacity .3s}
#s-train .head.ok i{opacity:1}
#s-train .ruler{display:flex;height:30px;align-items:center}
#s-train .ruler em{flex:1;text-align:center;font-style:normal;font-size:16px;color:var(--faint);transition:color .25s,transform .25s}
#s-train .rl{font-size:16px;color:var(--faint);align-self:center}
${N.rungs.map((_, r) => `#s-train[data-rung="${r}"] .ruler em:nth-child(${r + 1}){color:var(--ours);font-weight:700;transform:scale(1.25)}`).join('\n')}
#s-train .cand{font-family:var(--hand);font-weight:700;font-size:25px;align-self:center;line-height:1}
#s-train .blk{position:relative;height:92px;border-bottom:2px dashed var(--dot);
  background:repeating-linear-gradient(90deg,transparent 0 ${SLOT / 2 - 1}px,rgba(220,216,228,.7) ${SLOT / 2 - 1}px ${SLOT / 2 + 1}px,transparent ${SLOT / 2 + 1}px ${SLOT}px)}
#s-train .chip{position:absolute;left:0;width:${CHIP}px;height:24px;border:2px solid var(--ink);border-radius:8px;background:var(--lilac);
  font-size:16px;line-height:20px;text-align:center;transition:transform .6s cubic-bezier(.5,0,.3,1),background-color .3s,color .3s,opacity .3s}
#s-train.play .chip{animation:tr-pop .45s cubic-bezier(.3,1.6,.5,1) backwards;animation-delay:var(--d)}
@keyframes tr-pop{from{opacity:0;scale:.3}}
#s-train .chip.dead{background:var(--red);color:#fff;opacity:.85;text-decoration:line-through}
#s-train .chip.gone{opacity:.4}
#s-train .chip.win{background:var(--mint);border-color:var(--green);color:var(--green);font-weight:700}
#s-train .tag{position:absolute;height:24px;padding:0 7px;border:1.5px solid var(--red);border-radius:8px;background:#fff;color:var(--red);
  font-size:16px;line-height:21px;white-space:nowrap;opacity:0;translate:0 4px;transition:opacity .25s,translate .25s}
#s-train .tag.on{opacity:1;translate:0 0}
#s-train .stamp{position:absolute;height:30px;padding:0 10px;border:2.5px solid var(--green);border-radius:9px;background:rgba(205,239,221,.95);
  color:var(--green);font-family:var(--hand);font-weight:700;font-size:21px;line-height:25px;white-space:nowrap;box-shadow:var(--lift);
  rotate:-3deg;opacity:0;scale:1.8;transition:opacity .25s,scale .4s cubic-bezier(.3,1.5,.5,1);z-index:2}
#s-train .stamp.on{opacity:1;scale:1}
#s-train.still .chip,#s-train.still .tag,#s-train.still .stamp,#s-train.still .head i{transition:none}
#s-train .ft{position:absolute;left:72px;right:560px;top:770px;font-size:18px;line-height:1.45;color:var(--muted)}
#s-train .ft b{color:var(--red);font-size:20px}
#s-train .gpu{position:absolute;right:72px;top:768px;width:470px;padding:8px 14px;rotate:1.5deg;font-family:var(--hand);font-weight:700;font-size:23px;line-height:1.1;text-align:center}
`,
    mount(el) {
      const hd = h('div', 'hd');
      hd.append(h('div', 'kicker', 'model training, managed: the dream campaign'), h('h2', 'title', `${total} cells enter. ${winners} get certified.`));
      const ctr = h('div', 'ctr');
      const box = (cls, label) => { const d = h('div', 'card ' + cls); const o = h('output', 'num'); d.append(h('span', null, label), o); ctr.append(d); return o; };
      ui.alive = box('alive', 'alive');
      ui.pruned = box('pruned', 'pruned');
      ui.status = h('p', 'st');

      const grid = h('div', 'grid');
      grid.append(h('div'));
      sites.forEach((s) => {
        const head = h('div', 'head'), sp = h('span', null, 'bound '), out = h('output', 'num', '--'), ok = h('i', null, ' certified');
        sp.append(out, ok); head.append(h('b', null, s.label), sp); grid.append(head);
        heads.push({ head, out });
      });
      grid.append(h('div', 'rl', 'rows per rung'));
      sites.forEach(() => { const r = h('div', 'ruler'); N.rungs.forEach((n) => r.append(h('em', 'num', fmtRows(n)))); grid.append(r); });
      N.candidates.forEach((cand, ci) => {
        grid.append(h('div', 'cand', cand));
        sites.forEach((s, si) => {
          const blk = h('div', 'blk');
          [...PLAN[ci][si]].forEach((d, lane) => {
            const die = d === 'W' ? N.rungs.length : +d;
            const c = { die, si, el: h('div', 'chip num', `seed ${lane + 1}`), tag: h('div', 'tag'), why: die <= LAST ? why(cand, die) : null };
            const top = 3 + lane * PITCH;
            c.el.style.top = c.tag.style.top = top + 'px';
            c.el.style.setProperty('--d', `${si * 120 + ci * 60 + lane * 20}ms`);
            if (c.why) {
              c.tag.textContent = c.why;
              // tag sits in the empty part of its lane: right of an early casualty, left of a late one
              if (die <= 1) c.tag.style.left = x(die) + CHIP + 8 + 'px'; else c.tag.style.right = COL - x(die) + 8 + 'px';
              blk.append(c.tag);
            } else {
              c.stamp = h('div', 'stamp', `certified ${BOUND[si][LAST].toFixed(1)}%`);
              c.stamp.style.top = top - 3 + 'px';
              c.stamp.style.right = COL - (x(LAST) + CHIP) - 4 + 'px';
              blk.append(c.stamp);
            }
            blk.append(c.el);
            cells.push(c);
          });
          grid.append(blk);
        });
      });

      const ft = h('div', 'ft');
      const tl = h('div');
      tl.append('pruned for: ');
      ui.tally = REASONS.map((r, i) => { const n = h('b', 'num', '0'); tl.append(i ? ' · ' : '', `${r} `, n); return { r, n }; });
      ft.append(tl, h('div', null, 'held-out slices: repeat traffic = the last 20% by time · unseen = whole merchants held out'));
      const gpu = h('div', 'sticky gpu', `${total} cells, pruned automatically, under $${Math.ceil(N.gpuUsd)} of GPU for the whole campaign`);
      el.append(hd, ctr, ui.status, grid, ft, gpu);
      reset(el);
    },
    enter(el, run) {
      reset(el);
      void el.offsetWidth; // one reflow so the snap-back is not animated and the pop-in restarts
      el.classList.remove('still');
      el.classList.add('play');
      let pruned = 0;
      const tally = Object.fromEntries(REASONS.map((r) => [r, 0]));
      (async () => {
        await run.wait(1200);
        for (let r = 0; r <= LAST; r++) {
          el.dataset.rung = r;
          ui.status.textContent = `rung ${r + 1} of ${N.rungs.length}: ${fmtRows(N.rungs[r])} rows per cell`;
          cells.forEach((c) => { c.tag.classList.remove('on'); c.el.classList.toggle('gone', c.die < r); }); // older casualties fade back
          heads.forEach((b, i) => run.count(b.out, BOUND[i][r], 650, (v) => v.toFixed(1) + '%', r ? BOUND[i][r - 1] : BOUND[i][0] + 8));
          await run.wait(750);
          const dying = cells.filter((c) => c.die === r);
          dying.forEach((c, k) => { c.el.style.transitionDelay = k * 22 + 'ms'; c.el.classList.add('dead'); place(c, r, true); c.tag.classList.add('on'); tally[c.why]++; });
          const p0 = pruned; pruned += dying.length;
          run.tween(600, (t) => setCounts(Math.round(p0 + (pruned - p0) * t)));
          ui.tally.forEach((t) => (t.n.textContent = tally[t.r]));
          await run.wait(800);
          if (r < LAST) {
            cells.forEach((c) => { c.el.style.transitionDelay = ''; if (c.die > r) place(c, r + 1); });
            await run.wait(650);
          }
        }
        cells.filter((c) => c.stamp).forEach((c, i) => run.after(i * 220, () => { c.el.classList.add('win'); c.stamp.classList.add('on'); heads[c.si].head.classList.add('ok'); }));
        await run.wait(700);
        ui.status.textContent = `${winners} certified, ${pruned} pruned, no human in the loop`;
        el.classList.add('done');
      })();
    },
    leave(el) { reset(el); },
  });
})();
