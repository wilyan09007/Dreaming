// 7 close: a monthly-volume dial (log, 100k..100M) drives yearly savings from N.per1k; three proof stickers; the live URL.
(() => {
  const P = N.per1k, MIN = 5, MAX = 8, STEPS = 300; // slider value v -> 10^(MIN + v/100) runs a month
  const runs = (v) => Math.pow(10, MIN + (v / STEPS) * (MAX - MIN));
  const short = (x) => { // 3 significant digits + K/M/B so the width barely moves
    const [d, u] = x >= 1e9 ? [1e9, 'B'] : x >= 1e6 ? [1e6, 'M'] : x >= 1e3 ? [1e3, 'K'] : [1, ''];
    return (+(x / d).toPrecision(3)).toString() + u;
  };
  const stats = [ // [label, per-1k delta, format]
    ['saved a year', P.teacherUsd - P.oursUsd, (x) => '$' + short(x)],
    ['LLM calls avoided a year', P.teacherCalls - P.oursCalls, short],
    ['hours of waiting removed a year', P.teacherHours - P.oursHours, (x) => short(x) + ' h'],
  ];
  const proofs = [ // [big, small, color, tilt]
    [`certified ≤${N.headline.bound}% error`, 'a bound on held-out traffic, or the stand-in is not served', 'var(--mint)', -2.5],
    [`replay: ${N.replay.identical}`, `byte-identical reruns, pinned by ${N.replay.hashes}`, 'var(--sky)', 1.5],
    [`1 in ${N.psi.audit} served calls audited`, 'against the real model, so drift is caught live', 'var(--lilac)', -1],
  ];
  const S = '#s-close';
  let els;
  function show(v) { // writes text only, so a drag repaints in the same frame with no layout work
    const m = runs(v), k = (m * 12) / 1000;
    els.runs.textContent = short(m);
    stats.forEach(([, d, f], i) => { els.out[i].textContent = f(k * d); });
    els.fill.style.transform = `scaleY(${0.12 + 0.88 * (v / STEPS)})`;
    els.range.value = v;
  }
  Launch.add({
    id: 'close',
    title: 'What it means at scale',
    notes: 'Drag the dial to your traffic: these are yearly savings from the measured per-1,000-invoice numbers.\nAt 10M runs a month that is real money, hundreds of millions of LLM calls and decades of waiting gone, every year.\nAnd it is safe: certified at 5% or less, replay byte-identical, 1 in 20 served calls audited.\nIt is live right now at dreaming-demo.fly.dev.',
    css: `
${S} .kicker{font-size:30px}
${S} .title{margin:4px 0 26px}
${S} .jar{display:grid;grid-template-columns:150px minmax(0,1fr);gap:40px;align-items:center;padding:28px 40px 30px 34px}
${S} .jar svg{width:150px;height:250px;overflow:visible}
${S} .fill{transform-origin:50% 100%;transform-box:fill-box;transition:transform .12s linear}
${S} .dial{display:grid;grid-template-columns:auto minmax(0,1fr);align-items:center;gap:8px 30px}
${S} .dial label{font-size:22px;color:var(--muted);grid-column:1/-1}
${S} .runs{width:280px;font-size:72px;line-height:1;color:var(--ours)}
${S} .runs b{display:inline-block;width:200px;text-align:right} /* proportional digits: a fixed box keeps /mo still */
${S} .runs small{font-size:26px;color:var(--muted);margin-left:6px}
${S} input{width:100%;margin:0;height:44px;background:none;cursor:pointer;-webkit-appearance:none;appearance:none}
${S} input::-webkit-slider-runnable-track{height:14px;border:2.5px solid var(--ink);border-radius:9px;background:linear-gradient(90deg,var(--lilac),var(--lav))}
${S} input::-webkit-slider-thumb{-webkit-appearance:none;width:40px;height:40px;margin-top:-15px;border-radius:50%;border:2.5px solid var(--ink);background:var(--yellow);box-shadow:2px 3px 0 var(--ink)}
${S} input::-moz-range-track{height:10px;border:2.5px solid var(--ink);border-radius:9px;background:var(--lilac)}
${S} input::-moz-range-thumb{width:36px;height:36px;border-radius:50%;border:2.5px solid var(--ink);background:var(--yellow)}
${S} .ticks{grid-column:2;display:flex;justify-content:space-between;font-size:18px;color:var(--muted);margin:-4px 6px 0}
${S} .stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:24px;margin-top:22px;padding-top:20px;border-top:2.5px dashed var(--dot)}
${S} output{display:block;font-size:76px;line-height:1;color:var(--green);white-space:nowrap;font-variant-numeric:tabular-nums}
${S} .stats span{font-size:21px;color:var(--muted)}
${S} .proofs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr)) 400px;gap:30px;margin-top:44px;align-items:center}
${S} .proof{padding:18px 22px;min-height:124px;opacity:0;transform:translateY(-40px) rotate(calc(var(--r) * 3deg)) scale(1.1);transition:opacity .25s,transform .5s cubic-bezier(.3,1.5,.6,1)}
${S} .proof.in,${S} .url.in{opacity:1;transform:rotate(calc(var(--r) * 1deg))}
${S} .proof b{display:block;font-size:31px;line-height:1.05;margin-bottom:6px}
${S} .proof p{font-size:18px;line-height:1.25}
${S} .url{--r:1;padding:16px 22px;text-align:center;opacity:0;transform:scale(.9);transition:opacity .3s,transform .4s cubic-bezier(.3,1.5,.6,1)}
${S} .url span{display:block;font-size:20px;color:var(--muted)}
${S} .url b{display:block;font-size:35px;color:var(--ours);line-height:1.1;white-space:nowrap}
`,
    mount(el) {
      el.innerHTML = `
<div class="kicker">what it means at scale</div>
<h1 class="title">Turn the dial to your traffic</h1>
<div class="card jar">
  <svg viewBox="0 0 150 250" aria-hidden="true" style="filter:url(#rough)">
    <path class="fill" d="M22 80 Q20 236 30 238 L120 238 Q130 236 128 80 Z" fill="var(--mint)"/>
    <path d="M30 40 L120 40 L120 62 Q132 70 130 90 L128 230 Q128 242 116 242 L34 242 Q22 242 22 230 L20 90 Q18 70 30 62 Z" fill="none" stroke="var(--ink)" stroke-width="3" stroke-linejoin="round"/>
    <rect x="24" y="22" width="102" height="20" rx="5" fill="var(--yellow)" stroke="var(--ink)" stroke-width="3"/>
    <text x="75" y="166" text-anchor="middle" font-family="Gaegu,cursive" font-weight="700" font-size="34" fill="var(--ink)">saved</text>
  </svg>
  <div>
    <div class="dial">
      <label for="close-range">agent runs a month (one run = one invoice through all ${N.sites.length} steps)</label>
      <div class="runs"><b class="num"></b><small>/mo</small></div>
      <input id="close-range" type="range" min="0" max="${STEPS}" step="1" aria-label="agent runs a month">
      <div class="ticks"><span>100K</span><span>1M</span><span>10M</span><span>100M</span></div>
    </div>
    <div class="stats">${stats.map(([l]) => `<div><output class="num"></output><span>${l}</span></div>`).join('')}</div>
  </div>
</div>
<div class="proofs">
  ${proofs.map(([b, p, c, r]) => `<div class="sticky proof" style="background:${c};--r:${r}"><b>${b}</b><p>${p}</p></div>`).join('')}
  <div class="card url"><span>live right now</span><b>dreaming-demo.fly.dev</b></div>
</div>`;
      els = { runs: el.querySelector('.runs b'), out: [...el.querySelectorAll('output')], fill: el.querySelector('.fill'), range: el.querySelector('input') };
      els.range.addEventListener('pointerdown', () => (els.grab = true)); // the arrival sweep yields to the judge's hand
      els.range.addEventListener('input', () => { els.grab = true; show(+els.range.value); });
      els.range.addEventListener('change', () => els.range.blur()); // give the arrow keys back to the deck after a drag
      show(0);
    },
    enter(el, run) {
      this.leave(el);
      els.grab = false;
      run.after(300, () => run.tween(2200, (t) => els.grab || show(Math.round(t * 200)))); // sweep 100k -> 10M a month
      el.querySelectorAll('.proof,.url').forEach((n, i) => run.after(2700 + i * 350, () => n.classList.add('in')));
    },
    leave(el) {
      show(0);
      el.querySelectorAll('.in').forEach((n) => n.classList.remove('in'));
    },
    key(e) { // up/down nudge the dial a tenth of a decade; left/right stay deck navigation
      if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return false;
      els.grab = true;
      show(Math.max(0, Math.min(STEPS, +els.range.value + (e.key === 'ArrowUp' ? 10 : -10))));
      return true;
    },
  });
})();
