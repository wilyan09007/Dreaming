// Screen 2 "race": one invoice through the bookkeeper workflow, the teacher on every step vs Dreaming's certified
// stand-ins, in real time, then x1,000. The lane component is exposed as Launch.race so the build screen reuses it.
(() => {
  const h = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const sec = (ms, d = 2) => `${+(ms / 1000).toFixed(d)} s`;
  const msf = (ms) => (ms < 1000 ? `${Math.round(ms)} ms` : sec(ms));
  const MAX_MS = 11600; // longer workflows (build screen) fast-forward so the teacher still lands in ~12 s

  // expected $ / LLM calls / hours over 1,000 runs: the same formula as numbers.js per1k
  function compose(steps) {
    return steps.reduce((a, s) => {
      a.teacherUsd += s.teacherUsd1k; a.oursUsd += s.oursUsd1k;
      if (s.kind === 'llm') { a.teacherCalls += 1000; a.oursCalls += Math.round(1000 * (1 - s.coverage) + (1000 * s.coverage) / N.psi.audit); }
      a.teacherHours += (1000 * s.teacherMs) / 3.6e6;
      a.oursHours += (1000 * (s.coverage * s.oursMs + (1 - s.coverage) * s.teacherMs)) / 3.6e6;
      return a;
    }, { teacherUsd: 0, oursUsd: 0, teacherCalls: 0, oursCalls: 0, teacherHours: 0, oursHours: 0 });
  }
  const trigger = { label: 'webhook', kind: 'trigger', teacherMs: 0, oursMs: 0, teacherUsd1k: 0, oursUsd1k: 0, coverage: 0 };

  function verdict(steps, k) {
    const tMs = steps.reduce((a, s) => a + s.teacherMs, 0), oMs = steps.reduce((a, s) => a + s.oursMs, 0);
    const cheaper = k.teacherUsd > 0 ? Math.round((1 - k.oursUsd / k.teacherUsd) * 100) : 0;
    const faster = oMs > 0 ? Math.floor(tMs / oMs) : 0;
    const parts = [];
    if (cheaper > 0) parts.push(`<b>${cheaper}%</b> cheaper per invoice`);
    if (faster > 1) parts.push(`<b>${faster}x</b> faster end to end`);
    return parts.length ? parts.join(', ') : 'no stand-in yet: the teacher answers every step until one certifies';
  }

  function lane(cls, name, steps, msOf, fmt, unit) {
    const el = h('div', 'lane ' + cls);
    const n = steps.length, x = (i) => (n > 1 ? 4 + (i * 80) / (n - 1) : 4), fx = 95;
    const w = `min(130px,calc(${80 / Math.max(n - 1, 1)}% - 6px))`; // label width follows the node spacing
    if (n > 8) el.classList.add('dense'); // many steps: numbered dots only (the numbers match the build list)
    el.innerHTML = `<div class="lh"><span class="ln">${name}</span><span class="watch num"><svg width="22" height="24" viewBox="0 0 22 24" aria-hidden="true"><circle cx="11" cy="14" r="8.5" fill="#fff" stroke="currentColor" stroke-width="2.2"/><path d="M11 14V9M9 2h4M11 2v3" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" fill="none"/></svg><span class="dn">done in</span><output>0</output></span></div>
<div class="track"><i class="rail"></i><i class="fill"></i>${steps.map((s, i) => `<div class="node k-${s.kind}${s.fallback && cls === 'o' ? ' fb' : ''}" style="left:${x(i)}%;width:${w}"><span class="dot">${i || '&#9889;'}</span><span class="lab" title="${esc(s.label)}">${esc(s.label)}</span><span class="lat">${s.kind === 'trigger' ? 'trigger' : s.kind === 'human' ? 'handoff' : fmt(msOf(s))}</span></div>`).join('')}
<div class="flag" style="left:${fx}%"><span class="pole"></span></div><div class="tok"><i></i></div></div>
<div class="stats"><em>&times; 1,000 invoices</em><div><span>invoices</span><b class="num">0</b></div><div><span>$ spent</span><b class="num">$0.00</b></div><div><span>LLM calls</span><b class="num">0</b></div><div><span>hours</span><b class="num">0.0 h</b></div></div>`;
    let t = 0; const ends = steps.map((s) => (t += msOf(s)));
    return { el, ends, total: t, xs: steps.map((_, i) => x(i)), fx, fmt: unit, nodes: [...el.querySelectorAll('.node')], watch: el.querySelector('.watch output'), tok: el.querySelector('.tok'), fill: el.querySelector('.fill'), stats: [...el.querySelectorAll('.stats b')], state: [], at: -1 };
  }

  // advance a lane to elapsed time e; returns true once it has finished
  function tick(L, e) {
    const t = Math.min(e, L.total);
    let at = L.nodes.length;
    L.nodes.forEach((nd, i) => {
      const s = t >= L.ends[i] ? 2 : t >= (L.ends[i - 1] || 0) ? 1 : 0;
      if (s < 2 && at > i) at = i;
      if (L.state[i] !== s) { L.state[i] = s; nd.classList.toggle('work', s === 1); nd.classList.toggle('done', s === 2); }
    });
    if (at !== L.at) { // token only moves forward: node by node, then to the finish flag
      L.at = at; const px = at < L.nodes.length ? L.xs[at] : L.fx;
      L.tok.style.transform = `translateX(${px}%)`; L.fill.style.transform = `scaleX(${px / 100})`;
    }
    L.watch.textContent = L.fmt(t);
    if (t >= L.total) L.el.classList.add('over');
    return e >= L.total;
  }

  function mount(c, sites) {
    const steps = [trigger, ...sites], k = compose(steps);
    const T = lane('t', `LLM agent: <b>${N.teacher}</b> on every step`, steps, (s) => s.teacherMs, (ms) => sec(ms), (ms) => sec(ms));
    const O = lane('o', 'Dreaming: <b>certified stand-ins</b>', steps, (s) => s.oursMs, msf, msf);
    T.to = [1000, k.teacherUsd, k.teacherCalls, k.teacherHours]; O.to = [1000, k.oursUsd, k.oursCalls, k.oursHours];
    c.replaceChildren(T.el, O.el, h('div', 'verdict sticky', verdict(steps, k)));
    c.classList.add('rc');
    c._race = { lanes: [T, O], k };
    reset(c);
  }

  function reset(c) {
    const st = c._race; if (!st) return;
    c.classList.remove('k', 'v');
    for (const L of st.lanes) {
      L.state = []; L.at = -1; L.el.classList.remove('over');
      L.nodes.forEach((nd) => nd.classList.remove('work', 'done'));
      L.tok.style.transform = L.fill.style.transform = '';
      L.watch.textContent = L.fmt(0);
      ['0', '$0.00', '0', '0.0 h'].forEach((v, i) => (L.stats[i].textContent = v));
    }
  }

  async function play(run, c, delay = 500) {
    const st = c._race; reset(c);
    await run.wait(delay);
    const scale = Math.min(1, MAX_MS / Math.max(st.lanes[0].total, 1)); // <1 = fast-forward, the stopwatch still reads true time
    const t0 = performance.now();
    const live = () => c._race === st; // a remount (build screen edit) abandons this play
    await new Promise((res) => run.frame((now) => {
      if (!live()) return false;
      const e = (now - t0) / scale;
      const done = st.lanes.map((L) => tick(L, e)).every(Boolean);
      if (done) { res(); return false; }
    }));
    await run.wait(400);
    if (!live()) return;
    c.classList.add('k');
    const fm = [(v) => Math.round(v).toLocaleString(), (v) => '$' + v.toFixed(2), (v) => Math.round(v).toLocaleString(), (v) => v.toFixed(1) + ' h'];
    await Promise.all(st.lanes.flatMap((L) => L.to.map((to, i) => run.count(L.stats[i], to, 1600, fm[i]))));
    if (live()) c.classList.add('v');
  }

  // component css, scoped per screen
  const css = (S) => `
${S} .rc{display:flex;flex-direction:column;gap:18px}
${S} .lane{position:relative;background:#fff;border:2.5px solid var(--ink);border-radius:18px;box-shadow:var(--lift);padding:12px 22px 10px;display:grid;grid-template-columns:1fr auto;grid-template-rows:auto 1fr;column-gap:24px}
${S} .lane.t{--c:var(--teacher);--soft:var(--peach)} ${S} .lane.o{--c:var(--ours);--soft:var(--lilac)}
${S} .lane::before{content:'';position:absolute;left:-2.5px;top:14px;bottom:14px;width:8px;border-radius:0 6px 6px 0;background:var(--c)}
${S} .lh{display:flex;justify-content:space-between;align-items:center;gap:16px;grid-column:1}
${S} .ln{font-family:var(--hand);font-size:28px;color:var(--c)} ${S} .ln b{font-size:28px}
${S} .watch{display:inline-flex;align-items:center;gap:8px;color:var(--c);background:var(--soft);border:2px solid var(--ink);border-radius:999px;padding:2px 14px 2px 8px}
${S} .watch output{font-size:30px;min-width:96px;text-align:right;color:var(--ink)}
${S} .track{position:relative;grid-column:1;height:132px}
${S} .rail,${S} .fill{position:absolute;left:0;right:0;top:36px;height:5px;border-radius:3px;background:var(--dot)}
${S} .fill{background:var(--c);transform:scaleX(0);transform-origin:0 50%;transition:transform .3s ease}
${S} .node{position:absolute;top:8px;transform:translateX(-50%);display:flex;flex-direction:column;align-items:center;text-align:center}
${S} .dot{position:relative;width:58px;height:58px;border-radius:50%;border:2.5px solid var(--ink);background:#fff;display:grid;place-items:center;font-family:var(--hand);font-weight:700;font-size:26px;transition:background .25s,box-shadow .25s}
${S} .k-trigger .dot{background:var(--yellow)} ${S} .k-human .dot{border-style:dashed;background:var(--sky)} ${S} .fb .dot{border-color:var(--teacher)}
${S} .lab{max-width:100%;font-size:17px;line-height:1.15;margin-top:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
${S} .lat{font-family:var(--hand);font-weight:700;font-size:23px;line-height:1;color:var(--green);opacity:0;transform:translateY(-6px);transition:opacity .25s,transform .25s}
${S} .work .dot{border-color:var(--c);background:var(--soft);animation:rcpulse .7s ease-in-out infinite}
${S} .done .dot{background:var(--mint);box-shadow:0 0 0 6px rgba(30,122,85,.22),0 0 22px 6px rgba(30,122,85,.35)}
${S} .done .dot::after{content:'\\2713';position:absolute;right:-9px;top:-9px;width:26px;height:26px;border-radius:50%;background:var(--green);color:#fff;font:700 17px/26px var(--body);text-align:center;border:2px solid #fff}
${S} .done .lat{opacity:1;transform:none}
${S} .dense .lab{display:none} ${S} .dense .lat{margin-top:6px;white-space:nowrap;font-size:17px}
@keyframes rcpulse{0%,100%{transform:scale(1);box-shadow:0 0 0 0 rgba(43,43,51,.0)}50%{transform:scale(1.12);box-shadow:0 0 0 9px rgba(43,43,51,.07)}}
${S} .flag{position:absolute;top:2px;transform:translateX(-50%);display:flex;flex-direction:column;align-items:center}
${S} .pole{width:26px;height:68px;background:repeating-conic-gradient(var(--ink) 0 25%,#fff 0 50%) 0 0/13px 13px;border:2px solid var(--ink);border-radius:3px}
${S} .dn{display:none;font-family:var(--hand);font-size:24px;color:var(--ink)} ${S} .over .dn{display:inline}
${S} .over .watch{background:var(--mint)}
${S} .lane.over .pole{box-shadow:0 0 0 5px var(--soft)}
${S} .tok{position:absolute;left:0;top:0;width:100%;height:0;pointer-events:none;transition:transform .3s cubic-bezier(.3,.7,.3,1)}
${S} .tok i{position:absolute;left:-15px;top:-8px;width:30px;height:26px;background:#fff;border:2.2px solid var(--ink);border-radius:4px;box-shadow:2px 2px 0 rgba(43,43,51,.2);background-image:linear-gradient(var(--c),var(--c)),linear-gradient(var(--faint),var(--faint));background-size:16px 3px,12px 2px;background-position:5px 6px,5px 14px;background-repeat:no-repeat}
${S} .stats{grid-column:2;grid-row:1/3;width:360px;display:grid;grid-template-columns:1fr 1fr;gap:4px 18px;align-content:center;border-left:2px dashed var(--dot);padding-left:22px;opacity:.28;transition:opacity .4s}
${S} .rc.k .stats{opacity:1}
${S} .stats em{grid-column:1/-1;font-style:normal;font-family:var(--hand);font-size:24px;color:var(--c)}
${S} .stats div{display:flex;flex-direction:column}
${S} .stats span{font-size:16px;color:var(--muted)}
${S} .stats b{font-size:40px;line-height:1}
${S} .verdict{align-self:center;font-family:var(--hand);font-size:34px;padding:12px 30px;transform:translateY(-40px) rotate(-6deg) scale(.7);opacity:0;transition:transform .55s cubic-bezier(.3,1.6,.5,1),opacity .3s}
${S} .verdict b{font-size:60px;color:var(--ours)}
${S} .rc.v .verdict{transform:rotate(-1.6deg);opacity:1}`;

  Launch.race = { mount, play, reset, compose, css };
  const k1 = compose([trigger, ...N.sites]); // self-check: this component composes the same per-1k totals as numbers.js
  console.assert(Math.abs(k1.oursUsd - N.per1k.oursUsd) < 1e-9 && k1.oursCalls === N.per1k.oursCalls, 'race compose != N.per1k');

  const p = N.per1k, cheaper = Math.round((1 - p.oursUsd / p.teacherUsd) * 100), faster = Math.floor(N.taskTeacherMs / N.taskOursMs);
  Launch.add({
    id: 'race',
    title: 'Teacher vs Dreaming, one invoice',
    notes: `Same bookkeeper workflow, same invoice, two lanes.\nPurple lane: done in ${N.taskOursMs} ms, every step a certified stand-in. The coral lane is ${N.teacher} on every step, ${sec(N.taskTeacherMs, 1)} in real time.\nThen x1,000 invoices: $${p.teacherUsd.toFixed(2)} vs $${p.oursUsd.toFixed(2)}, ${p.teacherCalls.toLocaleString()} vs ${p.oursCalls} LLM calls.\nVerdict: ${cheaper}% cheaper per invoice, ${faster}x faster end to end.`,
    css: css('#s-race') + `
#s-race .head{display:flex;align-items:baseline;gap:24px;margin-bottom:16px}
#s-race .title{font-size:58px}
#s-race .rc{height:calc(100% - 92px)}
#s-race .note{position:absolute;left:72px;bottom:30px;font-size:16px;color:var(--muted);max-width:820px}`,
    mount(el) {
      el.innerHTML = `<div class="head"><h2 class="title">One invoice, two ways</h2><span class="kicker">same workflow, teacher vs Dreaming</span></div><div class="box"></div><p class="note">x1,000 = expected values: misses fall back to the teacher, 1 in ${N.psi.audit} served calls is audited against it.</p>`;
      mount(el.querySelector('.box'), N.sites);
    },
    enter(el, run) { play(run, el.querySelector('.box')); },
    leave(el) { reset(el.querySelector('.box')); },
  });
})();
