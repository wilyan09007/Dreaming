// 1 hook: the pain (four sticky notes land) and the promise (four headline tiles count up from N.headline).
(() => {
  const H = N.headline;
  const pains = [ // [label, line, sticky color, tilt]
    ['cost', `$${N.per1k.teacherUsd.toFixed(2)} of model calls per 1,000 invoices, forever`, 'var(--yellow)', -3],
    ['latency', `${(N.taskTeacherMs / 1000).toFixed(1)} s of waiting on the model per invoice`, 'var(--peach)', 2.5],
    ['outages', 'the provider has a bad day, so your agent does too', 'var(--rose)', -1.5],
    ['audit', 'no proof the answer was right, only a log that it was given', 'var(--sky)', 3],
  ];
  const tiles = [ // [to, prefix, suffix, label, caption, color]
    [H.cheaper, '', '%', 'cheaper', 'same answers, a fraction of the model bill', 'var(--ours)'],
    [H.faster, '~', '×', 'faster per call', 'milliseconds per step instead of seconds', 'var(--ours)'],
    [H.tokenFree, '', '%', 'of calls, 0 LLM tokens', 'answered from memory, the big model never wakes', 'var(--ours)'],
    [H.bound, '≤', '%', 'error, certified', 'a stand-in is only served once it proves the bound', 'var(--green)'],
  ];
  const S = '#s-hook';
  Launch.add({
    id: 'hook',
    title: 'Your agent calls a big model on every step',
    notes: 'Every step of your agent is a call to a big model: you pay for it, wait for it, and trust it blindly.\nPoint at the four stickies: cost, latency, outages, audit.\nDreaming learns the repeated calls from your own traces and serves certified small stand-ins.\nRight side: up to 87% cheaper, ~500x faster per call, 89% of calls with zero LLM tokens, error certified at 5% or less.',
    css: `
${S}{display:grid;grid-template-columns:minmax(0,1fr) 700px;gap:56px;align-items:center}
${S} .kicker{font-size:30px}
${S} h1{font-size:92px;line-height:.95;margin:10px 0 34px;letter-spacing:-.5px}
${S} h1 em{font-style:normal;position:relative;color:var(--teacher);white-space:nowrap}
${S} h1 svg{position:absolute;left:-4px;right:-4px;bottom:-14px;width:calc(100% + 8px);height:22px;overflow:visible}
${S} h1 path{stroke-dasharray:420;stroke-dashoffset:420;transition:stroke-dashoffset .7s ease .35s}
${S}.go h1 path{stroke-dashoffset:0}
${S} .pains{display:grid;grid-template-columns:1fr 1fr;gap:26px 28px;padding-right:10px}
${S} .pain{min-height:136px;padding:16px 20px 18px;opacity:0;transform:translateY(-60px) rotate(calc(var(--r) * 3deg)) scale(1.12);transition:opacity .25s,transform .5s cubic-bezier(.3,1.5,.6,1)}
${S} .pain.in{opacity:1;transform:rotate(calc(var(--r) * 1deg))}
${S} .pain b{display:block;font-size:30px;line-height:1;margin-bottom:6px}
${S} .pain p{font-size:20px;line-height:1.25}
${S} .tiles{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:28px}
${S} .tile{--t:0deg;height:310px;padding:22px 24px 22px;display:flex;flex-direction:column;opacity:0;transform:translateY(18px) rotate(var(--t));transition:opacity .4s,transform .4s}
${S} .tile.in{opacity:1;transform:rotate(var(--t))}
${S} .tile:nth-child(2){--t:1deg} ${S} .tile:nth-child(3){--t:-1deg}
${S} output{display:block;font-size:100px;line-height:1;letter-spacing:-3px;white-space:nowrap;font-variant-numeric:tabular-nums}
${S} .tile b{font-size:34px;line-height:1.05;margin:4px 0 auto}
${S} .tile p{font-size:19px;line-height:1.3;color:var(--muted)}
${S} .tile.cert{box-shadow:6px 7px 0 var(--mint)}
`,
    mount(el) {
      el.innerHTML = `
<div>
  <div class="kicker">agents, but awake only when it matters</div>
  <h1>Your agent calls a big model on <em>every step<svg viewBox="0 0 400 22" preserveAspectRatio="none" aria-hidden="true"><path d="M4 14 Q100 4 200 12 T396 9" fill="none" stroke="var(--teacher)" stroke-width="6" stroke-linecap="round"/></svg></em></h1>
  <div class="pains">${pains.map(([b, p, c, r]) => `<div class="sticky pain" style="background:${c};--r:${r}"><b>${b}</b><p>${p}</p></div>`).join('')}</div>
</div>
<div class="tiles">${tiles.map(([, , , b, p, c], i) => `<div class="card tile${i === 3 ? ' cert' : ''}"><output class="num" style="color:${c}"></output><b>${b}</b><p>${p}</p></div>`).join('')}</div>`;
    },
    enter(el, run) {
      this.leave(el);
      run.after(60, () => el.classList.add('go'));
      el.querySelectorAll('.pain').forEach((p, i) => run.after(500 + i * 450, () => p.classList.add('in')));
      el.querySelectorAll('.tile').forEach((t, i) => run.after(300 + i * 180, () => {
        const [to, pre, suf] = tiles[i];
        t.classList.add('in');
        run.count(t.querySelector('output'), to, 2400, (v) => pre + Math.round(v) + suf); // last one lands at ~3.2 s
      }));
    },
    leave(el) {
      el.classList.remove('go');
      el.querySelectorAll('.in').forEach((n) => n.classList.remove('in'));
      el.querySelectorAll('.tile output').forEach((o, i) => { o.textContent = tiles[i][1] + '0' + tiles[i][2]; });
    },
  });
})();
