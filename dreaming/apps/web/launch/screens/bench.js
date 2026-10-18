// Screen 3: the benchmark graphs. Three hand-drawn SVG bar charts; every figure comes from window.N.
(() => {
  const llm = N.sites.filter((s) => s.kind === 'llm');
  const W = 424; // chart width (svg units = stage px)
  let B, H; // baseline y and tallest bar of the chart being built
  const ms = (v) => (v >= 1000 ? (v / 1000).toFixed(1) + ' s' : Math.round(v) + ' ms');
  const saved = (s) => Math.floor(100 * (1 - s.oursUsd1k / s.teacherUsd1k)); // ponytail: floor, never rounds past the measured headline
  const easeOut = (t) => 1 - Math.pow(1 - t, 3);
  const T0 = 450, STEP = 600; // first bar after the cross-fade; each chart starts STEP ms after the previous one

  const bars = []; // filled in DOM order while the markup is built
  // a bar with a slightly wobbly rounded top, drawn at full height and grown by a scaleY about the baseline
  function bar(x, w, h, cls, at, fmt, inside) {
    bars.push({ h, at, fmt, inside, b: B });
    const t = B - h, r = Math.min(6, h / 2), d = bars.length % 2 ? 0.8 : -0.8;
    const p = `M${x} ${B}L${x + 0.6} ${t + r}Q${x + 1} ${t} ${x + r} ${t + d}L${x + w - r} ${t - d}Q${x + w} ${t} ${x + w - 0.5} ${t + r}L${x + w} ${B}`;
    return `<g class="bar"><path class="${cls}" d="${p}"/><text class="v ${cls}" x="${x + w / 2}" y="${inside ? B + 28 : B - 10}"></text></g>`;
  }
  const site = (cx, s, sub) => `<text class="site" x="${cx}" y="${B + 28}">${s.label}</text>` + (sub ? `<text class="cand" x="${cx}" y="${B + 50}">${sub}</text>` : '');
  const base = (x0) => `<path class="ax" d="M${x0 - 4} ${B}Q${(x0 + W) / 2} ${B + 1.5} ${W - 2} ${B - 0.5}"/>`;
  const svg = (h, body) => `<svg width="${W}" height="${h}" viewBox="0 0 ${W} ${h}" aria-hidden="true">${body}</svg>`;

  // (a) $ per 1,000 calls, teacher vs Dreaming, % saved above each pair
  function cost() {
    B = 340; H = 272;
    const x0 = 12, g = (W - x0) / 3, w = 54, max = Math.max(...llm.map((s) => s.teacherUsd1k)), at = T0;
    return svg(392, llm.map((s, i) => {
      const cx = x0 + g * i + g / 2, ht = (H * s.teacherUsd1k) / max, ho = (H * s.oursUsd1k) / max;
      return bar(cx - w - 5, w, ht, 't', at + i * 140, (t) => '$' + (s.teacherUsd1k * t).toFixed(2))
        + bar(cx + 5, w, ho, 'o', at + 70 + i * 140, (t) => '$' + (s.oursUsd1k * t).toFixed(2))
        + `<g data-at="${at + 1000 + i * 160}"><rect class="badge" x="${cx - 40}" y="${B - ht - 68}" width="80" height="32" rx="16"/><text class="save" x="${cx}" y="${B - ht - 45}">-${saved(s)}%</text></g>`
        + site(cx, s);
    }).join('') + base(x0));
  }

  // (b) p50 latency per call on a log axis, 1 ms .. 10 s
  function latency() {
    B = 340; H = 272;
    const x0 = 66, g = (W - x0) / 3, w = 48, at = T0 + STEP;
    const y = (v) => (H * Math.log10(v)) / 4;
    const ticks = [1, 10, 100, 1000, 10000].map((v) => `<path class="tk" d="M${x0 - 8} ${B - y(v)}h8"/><text class="tick" x="${x0 - 14}" y="${B - y(v) + 6}">${v < 1000 ? v + ' ms' : v / 1000 + ' s'}</text>`).join('');
    const grow = (v) => (t) => ms(Math.pow(10, Math.log10(v) * t)); // the label climbs the log axis with the bar
    return svg(392, llm.map((s, i) => {
      const cx = x0 + g * i + g / 2;
      return bar(cx - w - 4, w, y(s.teacherMs), 't', at + i * 140, grow(s.teacherMs))
        + bar(cx + 4, w, y(s.oursMs), 'o', at + 70 + i * 140, grow(s.oursMs))
        + site(cx, s);
    }).join('') + ticks + `<path class="ax" d="M${x0} ${B}L${x0 + 1} ${B - H - 8}"/>` + base(x0));
  }

  // (c) share of calls the certified stand-in serves at the error budget; the dashed box is 100% of calls
  function coverage() {
    B = 252; H = 190;
    const x0 = 12, g = (W - x0) / 3, w = 74, at = T0 + 2 * STEP;
    return svg(312, llm.map((s, i) => {
      const cx = x0 + g * i + g / 2;
      return `<rect class="ghost" x="${cx - w / 2}" y="${B - H}" width="${w}" height="${H}" rx="6"/>`
        + bar(cx - w / 2, w, H * s.coverage, 'c', at + i * 140, (t) => Math.round(100 * s.coverage * t) + '%', true)
        + site(cx, s, s.cand);
    }).join('') + base(x0));
  }

  // error ruler under (c): 0..10% error, the teacher's own noise floor and the budget we certify at
  function ruler() {
    const x = (p) => 14 + (p / 10) * (W - 28), f = N.selfDisagree, b = N.headline.bound;
    return svg(74, `<defs><pattern id="bench-hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><path d="M0 0V8" stroke="#9A2F2F" stroke-width="2.5" opacity=".35"/></pattern></defs>
      <rect x="${x(0)}" y="26" width="${x(f) - x(0)}" height="20" fill="url(#bench-hatch)"/>
      <rect x="${x(f)}" y="26" width="${x(b) - x(f)}" height="20" class="okz"/>
      <path class="ax" d="M${x(0)} 46Q${x(5)} 47.5 ${x(10)} 45.5"/>
      <text class="rl red" x="${x(f) - 6}" y="18" text-anchor="end">floor ${f}%</text>
      <g data-at="${T0 + 2 * STEP + 1100}"><path class="budget" d="M${x(b)} 8V54"/><text class="rl grn" x="${x(b) + 8}" y="18">budget &le;${b}%</text></g>
      <text class="tick st" x="${x(0)}" y="70">0% error</text><text class="tick" x="${x(10)}" y="70">10%</text>`);
  }

  const card = (h3, stat, cap, chart, foot) => `<div class="card"><h3>${h3}</h3><div class="stat"><b class="num" data-to="${stat}"></b><span>${cap}</span></div>${chart}<p class="foot">${foot}</p></div>`;
  const statFmt = [(v) => Math.round(v) + '%', (v) => '~' + Math.round(v) + 'x', (v) => Math.round(v) + '%'];
  let refs = [], stats = [];

  function grow(r, t) {
    r.p.setAttribute('transform', `matrix(1 0 0 ${t} 0 ${r.b * (1 - t)})`);
    r.v.setAttribute('transform', `translate(0 ${-r.h * t})`);
    r.v.style.opacity = r.inside ? Math.max(0, Math.min(1, t * 2.5 - 1.2)) : Math.min(1, t * 3);
    r.v.textContent = r.fmt(t);
  }
  function reset(el) {
    refs.forEach((r) => grow(r, 0));
    stats.forEach((b, i) => { b.textContent = statFmt[i](0); });
    el.querySelectorAll('[data-at]').forEach((n) => n.classList.remove('in'));
  }

  Launch.add({
    id: 'bench',
    title: 'Benchmarks: cost, latency, coverage',
    notes: 'Three measured charts: teacher in coral, Dreaming in purple.\nLeft: dollars per 1,000 calls on each LLM step, up to 87% saved.\nMiddle: log scale, seconds become milliseconds, about 500x faster per call.\nRight: share served by the certified stand-in at a 5% error budget; 3.5% is the teacher\'s own noise floor.',
    css: `
#s-bench .head{display:flex;justify-content:space-between;align-items:baseline;gap:24px}
#s-bench .title{font-size:60px;margin-top:6px;white-space:nowrap}
#s-bench .legend{display:flex;gap:26px;font-size:20px;white-space:nowrap}
#s-bench .legend i{display:inline-block;width:22px;height:22px;border:2.5px solid var(--ink);border-radius:6px;vertical-align:-4px;margin-right:8px}
#s-bench .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:24px;margin-top:26px}
#s-bench .card{padding:16px 20px 14px;display:flex;flex-direction:column}
#s-bench h3{font-size:30px;line-height:1.1}
#s-bench .stat{display:flex;align-items:baseline;gap:12px;margin:0 0 6px}
#s-bench .stat b{font-size:72px;line-height:1;color:var(--ours);min-width:132px}
#s-bench .card:nth-child(2) .stat b{min-width:206px} /* fixed widths: captions don't slide while the numbers count */
#s-bench .card:nth-child(3) .stat b{color:var(--green)}
#s-bench .stat span{font-size:19px;color:var(--muted)}
#s-bench .foot{font-size:17px;line-height:1.35;color:var(--muted);margin-top:auto;padding-top:6px}
#s-bench svg{display:block;overflow:visible}
#s-bench svg text{font-family:var(--body);font-size:17px;fill:var(--ink)}
#s-bench .t{fill:var(--teacher)} #s-bench .o{fill:var(--ours)} #s-bench .c{fill:#8EDDBE}
#s-bench .bar path{stroke:var(--ink);stroke-width:2.5;stroke-linejoin:round}
#s-bench text.v{font-family:var(--hand);font-weight:700;font-size:26px;text-anchor:middle}
#s-bench text.v.t{fill:#B9523A} #s-bench text.v.o{fill:var(--ours)} #s-bench text.v.c{fill:var(--ink)}
#s-bench .ax{stroke:var(--ink);stroke-width:2.5;fill:none;stroke-linecap:round;filter:url(#rough)}
#s-bench .tk{stroke:var(--ink);stroke-width:2}
#s-bench .tick{font-size:16px;fill:var(--muted);text-anchor:end} #s-bench .tick.st{text-anchor:start}
#s-bench .site{text-anchor:middle}
#s-bench .cand{text-anchor:middle;font-size:16px;fill:var(--muted)}
#s-bench .ghost{fill:none;stroke:var(--faint);stroke-width:2;stroke-dasharray:6 5}
#s-bench .badge{fill:var(--mint);stroke:var(--green);stroke-width:2.5}
#s-bench .save{font-family:var(--hand);font-weight:700;font-size:24px;fill:var(--green);text-anchor:middle}
#s-bench .okz{fill:var(--mint)}
#s-bench .budget{stroke:var(--green);stroke-width:3;stroke-dasharray:6 4}
#s-bench .rl{font-family:var(--hand);font-weight:700;font-size:21px}
#s-bench .red{fill:var(--red)} #s-bench .grn{fill:var(--green)}
#s-bench [data-at]{opacity:0;transform:translateY(10px) scale(.8);transform-box:fill-box;transform-origin:50% 100%}
#s-bench [data-at].in{opacity:1;transform:none;transition:opacity .3s ease,transform .5s cubic-bezier(.3,1.6,.5,1)}
`,
    mount(el) {
      const h = N.headline;
      el.innerHTML = `<div class="head"><div class="kicker">benchmarks, measured on the bookkeeper workflow</div>
        <div class="legend"><span><i style="background:var(--teacher)"></i>teacher (${N.teacher})</span><span><i style="background:var(--ours)"></i>Dreaming</span></div></div>
        <h2 class="title">The same calls: up to ${h.cheaper}% cheaper, ~${h.faster}x faster</h2>
        <div class="grid">
          ${card('$ per 1,000 calls', h.cheaper, 'cheaper, best step', cost(), 'Per LLM step, teacher vs Dreaming; green = % saved.')}
          ${card('p50 latency per call', h.faster, 'faster per call', latency(), 'Log scale: seconds become milliseconds.')}
          ${card(`coverage at &le;${h.bound}% error`, h.tokenFree, 'served, 0 LLM tokens', coverage() + ruler(),
            `We certify at &le;${h.bound}%. Below ${N.selfDisagree}% nothing is measurable: the teacher disagrees with itself ${N.selfDisagree}% of the time.`)}
        </div>`;
      refs = [...el.querySelectorAll('.bar')].map((g, i) => ({ ...bars[i], p: g.firstChild, v: g.lastChild }));
      stats = [...el.querySelectorAll('.stat b')];
      reset(el);
    },
    enter(el, run) {
      reset(el);
      refs.forEach((r) => run.after(r.at, () => run.tween(900, (t) => grow(r, t), easeOut)));
      el.querySelectorAll('[data-at]').forEach((n) => run.after(+n.dataset.at, () => n.classList.add('in')));
      stats.forEach((b, i) => run.after(T0 + i * STEP, () => run.count(b, +b.dataset.to, 1300, statFmt[i])));
    },
    leave: reset,
  });
})();
