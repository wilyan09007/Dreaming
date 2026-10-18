// Screen 4: how Psi answers a call. A familiar call is served from memory; an unfamiliar one goes to the teacher
// and its answer is appended as a new dot. Nearest dots, weights, vote and confidence are all computed from the
// toy grid below, so the lit dots are exactly the ones the weights (and the vote) refer to.
(() => {
  const P = N.psi, site = N.byId.categorize;
  const CATS = [ // categories live in clusters of the memory, like real embeddings do
    { name: 'Office supplies', c: '#6A4FD8', at: [2, 2] },
    { name: 'Software', c: '#2F7FC1', at: [9, 2] },
    { name: 'Travel', c: '#C7861A', at: [5.5, 6] },
  ];
  const COLS = 12, ROWS = 7, GAP = 26, GX = 544, GY = 150; // dot grid, in diagram px
  const xy = ([c, r]) => [GX + c * GAP, GY + r * GAP];
  const d2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
  const dots = [];
  for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) {
    const p = [c, r];
    dots.push({ p, cat: CATS.reduce((b, k, i) => (d2(p, k.at) < d2(p, CATS[b].at) ? i : b), 0) });
  }
  // softmax attention over the 4 nearest dots (toy temperature), then a weighted vote per category
  function attend(q) {
    const near = dots.map((d, i) => ({ i, d: d2(d.p, q) })).sort((a, b) => a.d - b.d).slice(0, 4);
    const e = near.map((n) => Math.exp(-2 * n.d)), s = e.reduce((a, b) => a + b);
    near.forEach((n, k) => (n.w = e[k] / s));
    const vote = CATS.map((_, ci) => near.filter((n) => dots[n.i].cat === ci).reduce((a, n) => a + n.w, 0));
    const top = vote.indexOf(Math.max(...vote));
    return { q, near, vote, top, conf: vote[top] };
  }
  const Q = [
    { merchant: 'AMZN MKTP *WEB', amt: '$63.20', pill: 'call 1 · a kind it has seen before', vec: [.9, .25, .6, .35, .8, .15, .5, .95, .3, .7], ...attend([3.3, 3.4]) },
    { merchant: 'KOVA*ZEPHYR', amt: '$412.00', pill: 'call 2 · never seen anything like it', vec: [.3, .85, .2, .6, .4, .9, .15, .45, .75, .25], ...attend([5.6, 2.4]), teacher: 'Software' },
  ];
  const pct = (v) => Math.round(v * 100) + '%';

  // boxes: x, y, w, h in diagram px; the label + one-line why sit above each box
  const B = {
    call: [0, 110, 184, 120], enc: [222, 110, 146, 120], vec: [404, 95, 64, 150], mem: [510, 90, 340, 240],
    vote: [894, 90, 216, 200], gate: [1146, 110, 144, 120], out: [1326, 110, 130, 120],
  };
  const LAB = [
    ['call', 'a new call', 'what the agent wants categorized', 184],
    ['enc', 'encoder', `turns the words into ${P.dim} numbers`, 200],
    ['vec', 'query ξ', '', 80],
    ['mem', `memory: <span class="num cnt">${P.traces.toLocaleString('en-US')}</span> past traces`, 'one dot per call the teacher already answered', 340],
    ['vote', 'weighted vote', 'similar past calls vote; closer ones count more', 230],
    ['gate', 'certify', `serve only if ≤${N.headline.bound}% error is proven`, 160],
    ['out', 'answer out', 'no big model was called', 130],
  ];
  const box = (id, inner, cls = '') => { const [x, y, w, h] = B[id]; return `<div class="card box b-${id} ${cls}" style="left:${x}px;top:${y}px;width:${w}px;height:${h}px">${inner}</div>`; };
  const arrow = (x1, x2, y = 170) => `<path d="M${x1} ${y}H${x2 - 4}" marker-end="url(#psi-ah)"/>`;

  function mount(el) {
    el.innerHTML = `
      <div class="head"><div><div class="kicker">how a stand-in answers a call</div><h2 class="title">Psi: a workflow language model</h2></div><div class="pill"></div></div>
      <div class="dia">
        ${LAB.map(([id, b, why, w]) => { const [x, , bw] = B[id]; return `<div class="lab" style="left:${x + bw / 2 - w / 2}px;width:${w}px"><b>${b}</b>${why ? `<span>${why}</span>` : ''}</div>`; }).join('')}
        ${box('call', `<small>categorize</small><code class="mer"></code><b class="amt"></b>`)}
        ${box('enc', `<b class="big">${P.encoder}</b><small>${P.dim}-d vector</small>`)}
        ${box('vec', Q[0].vec.map(() => '<i></i>').join(''))}
        ${box('mem', `<em>softmax attention: closest dots win</em>`)}
        ${box('vote', CATS.map((k, i) => `<div class="vr"><span><i style="background:${k.c}"></i>${k.name}<em class="num v${i}">0%</em></span><u style="background:${k.c}"></u></div>`).join('') + '<div class="ans"></div>')}
        ${box('gate', `<small>Clopper-Pearson</small><b class="big">≤${N.headline.bound}% error</b><span class="stamp"></span>`)}
        ${box('out', `<i class="fill"></i><b class="big">served</b><b>0 LLM tokens</b><small>~${site.oursMs} ms</small>`)}
        <div class="card box b-teach" style="left:1150px;top:390px;width:306px;height:110px"><b class="big">teacher: ${N.teacher}</b><small>asked only when Psi is unsure · ~${site.teacherMs / 1000} s</small><b class="tans"></b></div>
        <div class="rule card"><small>the whole model, one line</small><div class="eq"><i>ξ</i><sub>new</sub> = X · softmax(<i>β</i> X<sup>T</sup><i>ξ</i>)</div><small>X = the memory · <i>ξ</i> = the call · <i>β</i> = ${P.beta}</small></div>
        <div class="sticky note"><b>the teacher's answer → a new dot</b><span>learn = append · forget = delete · no training step</span></div>
        <svg class="wires" width="1456" height="600" viewBox="0 0 1456 600" aria-hidden="true">
          <defs>
            <marker id="psi-ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M1 1 9 5 1 9" fill="none" stroke="#2B2B33" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></marker>
            <marker id="psi-ahc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M1 1 9 5 1 9" fill="none" stroke="#E0795F" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></marker>
          </defs>
          <g class="arrows">${arrow(184, 222)}${arrow(368, 404)}${arrow(468, 510)}${arrow(850, 894)}${arrow(1110, 1146)}${arrow(1290, 1326)}</g>
          <g class="br"><path d="M1218 240V384" marker-end="url(#psi-ahc)"/><text x="1230" y="318">not sure</text></g>
          <path class="br ret" d="M1150 445C960 445 700 440 690 338" marker-end="url(#psi-ahc)"/>
          <g class="dots">${dots.map((d) => { const [x, y] = xy(d.p); return `<circle cx="${x}" cy="${y}" r="5" fill="${CATS[d.cat].c}"/>`; }).join('')}</g>
          <g class="att"></g>
          <g class="mk"><circle r="11"/><text y="6">ξ</text></g>
          <circle class="nd" r="8" fill="${CATS[1].c}"/><text class="ndl"></text>
        </svg>
      </div>`;
  }

  const $ = (el, s) => el.querySelector(s);
  function clear(el) {
    el.querySelectorAll('.hot,.lit,.on').forEach((n) => n.classList.remove('hot', 'lit', 'on'));
    $(el, '.att').innerHTML = '';
    el.querySelectorAll('.vr u').forEach((u) => (u.style.transform = 'scaleX(0)'));
    el.querySelectorAll('.vr em').forEach((e) => (e.textContent = '0%'));
    $(el, '.ans').textContent = ''; $(el, '.tans').textContent = '';
    $(el, '.stamp').className = 'stamp';
    moveMk(el, [468, 170]);
  }
  const moveMk = (el, [x, y]) => ($(el, '.mk').style.transform = `translate(${x}px,${y}px)`);

  async function pass(el, run, k) {
    const q = Q[k], hot = (id) => $(el, '.b-' + id).classList.add('hot');
    $(el, '.pill').textContent = q.pill; $(el, '.pill').classList.add('on');
    $(el, '.mer').textContent = q.merchant; $(el, '.amt').textContent = q.amt;
    hot('call');
    await run.wait(300); hot('enc');
    await run.wait(250);
    el.querySelectorAll('.b-vec i').forEach((c, i) => (c.style.opacity = q.vec[i]));
    $(el, '.b-vec').classList.add('lit');
    await run.wait(250); hot('mem'); $(el, '.mk').classList.add('on'); moveMk(el, xy(q.q));
    await run.wait(450);
    // rays from the query to its nearest dots, each tagged with its softmax weight
    const [qx, qy] = xy(q.q);
    $(el, '.att').innerHTML = q.near.map((n) => {
      const [x, y] = xy(dots[n.i].p), dx = x - qx, dy = y - qy, l = Math.hypot(dx, dy) || 1;
      const tx = x + (dx / l) * 30, ty = y + (dy / l) * 26;
      return `<line x1="${qx}" y1="${qy}" x2="${x}" y2="${y}"/><circle class="hit" cx="${x}" cy="${y}" r="8" fill="${CATS[dots[n.i].cat].c}"/>` +
        `<g class="tag" transform="translate(${tx} ${ty})"><rect x="-23" y="-13" width="46" height="26" rx="9"/><text y="6">${n.w.toFixed(2)}</text></g>`;
    }).join('');
    run.after(30, () => $(el, '.att').classList.add('on'));
    await run.wait(500); hot('vote');
    el.querySelectorAll('.vr u').forEach((u, i) => (u.style.transform = `scaleX(${q.vote[i]})`));
    q.vote.forEach((v, i) => run.count($(el, '.v' + i), v * 100, 600, (x) => Math.round(x) + '%'));
    await run.wait(500);
    const ans = $(el, '.ans');
    ans.innerHTML = `${CATS[q.top].name} <span>${pct(q.conf)} sure</span>`;
    ans.style.color = q.teacher ? 'var(--muted)' : CATS[q.top].c;
    await run.wait(350); hot('gate');
    $(el, '.stamp').className = 'stamp on ' + (q.teacher ? 'no' : 'ok');
    $(el, '.stamp').textContent = q.teacher ? '✗' : '✓';
    await run.wait(350);
    if (!q.teacher) { hot('out'); $(el, '.b-out').classList.add('lit'); return; }
    // unsure: the dashed branch to the teacher, then its answer flies back into memory as a new dot
    el.querySelectorAll('.br').forEach((b) => b.classList.add('on')); hot('teach');
    await run.wait(700);
    $(el, '.tans').textContent = 'answers: ' + q.teacher;
    await run.wait(500);
    const ret = $(el, '.ret'), len = ret.getTotalLength(), nd = $(el, '.nd'), [ex, ey] = xy(q.q);
    nd.classList.add('fly');
    await run.tween(1100, (t) => {
      const p = t < 0.8 ? ret.getPointAtLength(len * t / 0.8) : { x: 690 + (ex - 690) * (t - 0.8) / 0.2, y: 338 + (ey - 338) * (t - 0.8) / 0.2 };
      nd.setAttribute('cx', p.x); nd.setAttribute('cy', p.y);
    });
    nd.classList.remove('fly'); nd.classList.add('on');
    $(el, '.mk').classList.remove('on'); $(el, '.att').classList.remove('on');
    const ndl = $(el, '.ndl'); ndl.setAttribute('x', ex + 14); ndl.setAttribute('y', ey - 10); ndl.textContent = 'new: ' + q.teacher; ndl.classList.add('on');
    $(el, '.note').classList.add('on');
    run.count($(el, '.cnt'), P.traces + 1, 500, (v) => Math.round(v).toLocaleString('en-US'), P.traces);
  }

  function reset(el) {
    clear(el);
    el.querySelectorAll('.b-vec i').forEach((c) => (c.style.opacity = 0.12));
    const nd = $(el, '.nd'); nd.classList.remove('on', 'fly');
    $(el, '.cnt').textContent = P.traces.toLocaleString('en-US');
  }

  Launch.add({
    id: 'psi',
    title: 'Psi: a workflow language model',
    notes: 'Every dot is a past call the teacher already answered: 8,219 of them, grouped by answer.\n' +
      'A familiar call lands inside a cluster, its nearest dots vote, the gate certifies it: served with 0 LLM tokens.\n' +
      'An unfamiliar call splits the vote, the gate says no, the teacher answers, and that answer becomes a new dot.\n' +
      'No training step: learning is an append, forgetting is a delete. The whole model is the one line bottom-left.',
    css: `
#s-psi .head{display:flex;justify-content:space-between;align-items:flex-start}
#s-psi .pill{margin-top:10px;font-family:var(--hand);font-size:26px;font-weight:700;padding:6px 18px;background:var(--lilac);border:2.5px solid var(--ink);border-radius:999px;box-shadow:var(--lift);transform:rotate(1.5deg);opacity:0;transition:opacity .3s}
#s-psi .pill.on{opacity:1}
#s-psi .dia{position:absolute;left:72px;top:222px;width:1456px;height:600px}
#s-psi .lab{position:absolute;top:0;text-align:center}
#s-psi .lab b{display:block;font-size:25px;line-height:1.1;white-space:nowrap}
#s-psi .lab>span{display:block;margin-top:3px;font-size:16px;line-height:1.25;color:var(--muted)}
#s-psi .box{position:absolute;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:4px;text-align:center;padding:10px}
#s-psi .box::after{content:'';position:absolute;inset:-9px;border:3px solid var(--lav);border-radius:25px;opacity:0;transition:opacity .3s}
#s-psi .box.hot::after{opacity:1}
#s-psi .box small{font-size:16px;color:var(--muted);white-space:nowrap}
#s-psi .b-gate{padding:8px}
#s-psi .box .big{font-size:27px;line-height:1}
#s-psi .mer{font-family:var(--mono);font-size:18px;font-weight:700;white-space:nowrap}
#s-psi .amt{font-size:28px}
#s-psi .b-vec{gap:3px;padding:8px}
#s-psi .b-vec i{display:block;width:100%;height:10px;border-radius:3px;background:var(--ours);opacity:.12;transition:opacity .35s}
#s-psi .b-mem{justify-content:flex-start;padding-top:12px}
#s-psi .b-mem em{font-size:16px;color:var(--purple);font-style:normal;font-weight:600}
#s-psi .b-vote{justify-content:flex-start;align-items:stretch;gap:0;padding:12px 16px}
#s-psi .vr{height:38px;display:flex;flex-direction:column;justify-content:center;gap:4px;text-align:left}
#s-psi .vr span{display:flex;align-items:center;gap:7px;font-size:16px}
#s-psi .vr span i{width:12px;height:12px;border-radius:50%}
#s-psi .vr em{margin-left:auto;font-family:var(--hand);font-weight:700;font-size:20px;font-style:normal}
#s-psi .vr u{display:block;height:8px;border-radius:4px;transform:scaleX(0);transform-origin:left;transition:transform .6s ease}
#s-psi .ans{margin-top:10px;padding-top:8px;border-top:2px dashed var(--dot);font-family:var(--hand);font-weight:700;font-size:26px;line-height:1.05;min-height:42px}
#s-psi .ans span{display:block;font-size:20px;color:var(--muted)}
#s-psi .stamp{position:absolute;right:-16px;top:-18px;width:44px;height:44px;border-radius:50%;border:2.5px solid var(--ink);display:flex;align-items:center;justify-content:center;font-size:26px;font-weight:700;color:#fff;transform:scale(0) rotate(-20deg);transition:transform .35s cubic-bezier(.3,1.6,.5,1)}
#s-psi .stamp.on{transform:scale(1) rotate(8deg)}
#s-psi .stamp.ok{background:var(--green)}
#s-psi .stamp.no{background:var(--red)}
#s-psi .b-out{border-color:var(--green);overflow:hidden}
#s-psi .b-out>*{position:relative;opacity:.35;transition:opacity .3s}
#s-psi .b-out .fill{position:absolute;inset:0;background:var(--mint);opacity:0}
#s-psi .b-out.lit>*{opacity:1}
#s-psi .b-out .big{color:var(--green)}
#s-psi .b-out b:not(.big){font-size:21px}
#s-psi .b-teach{border:2.5px dashed var(--teacher);opacity:.45;transition:opacity .3s}
#s-psi .b-teach.hot{opacity:1}
#s-psi .b-teach .big{color:var(--teacher);font-size:25px}
#s-psi .tans{font-size:24px;min-height:26px}
#s-psi .rule{position:absolute;left:0;top:330px;width:440px;padding:14px 20px;display:flex;flex-direction:column;gap:4px;transform:rotate(-1deg)}
#s-psi .rule small{font-size:16px;color:var(--muted)}
#s-psi .rule i{font-family:'Cambria Math',Cambria,Georgia,serif}
#s-psi .eq{font-family:'Cambria Math',Cambria,Georgia,serif;font-size:26px}
#s-psi .note{position:absolute;left:700px;top:478px;width:430px;display:flex;flex-direction:column;gap:4px;transform:rotate(-2deg);transition:transform .4s cubic-bezier(.3,1.6,.5,1)}
#s-psi .note b{font-size:24px;line-height:1.05}
#s-psi .note span{font-size:17px}
#s-psi .note.on{transform:rotate(-2deg) scale(1.02)}
#s-psi .wires{position:absolute;left:0;top:0;overflow:visible;pointer-events:none}
#s-psi .arrows path{fill:none;stroke:var(--ink);stroke-width:2.5;stroke-linecap:round}
#s-psi .br{opacity:.25;transition:opacity .4s}
#s-psi .br.on{opacity:1}
#s-psi .br path,#s-psi path.br{fill:none;stroke:var(--teacher);stroke-width:2.5;stroke-dasharray:8 7;stroke-linecap:round}
#s-psi .br text{font-family:var(--hand);font-weight:700;font-size:20px;fill:var(--teacher)}
#s-psi .dots circle{opacity:.38}
#s-psi .att{opacity:0;transition:opacity .35s}
#s-psi .att.on{opacity:1}
#s-psi .att line{stroke:var(--ours);stroke-width:2.5}
#s-psi .att .hit{stroke:var(--ink);stroke-width:2}
#s-psi .tag rect{fill:#fff;stroke:var(--ink);stroke-width:1.8}
#s-psi .tag text,#s-psi .mk text{font-family:var(--hand);font-weight:700;font-size:19px;text-anchor:middle;fill:var(--ink)}
#s-psi .mk{opacity:0;transition:transform .5s ease,opacity .25s}
#s-psi .mk.on{opacity:1}
#s-psi .mk circle{fill:#fff;stroke:var(--ours);stroke-width:2.5}
#s-psi .nd{opacity:0;stroke:var(--teacher);stroke-width:3;transform-box:fill-box;transform-origin:center}
#s-psi .nd.fly{opacity:1}
#s-psi .ndl{font-family:var(--hand);font-weight:700;font-size:20px;fill:var(--teacher);paint-order:stroke;stroke:#fff;stroke-width:5px;opacity:0;transition:opacity .3s}
#s-psi .ndl.on{opacity:1}
#s-psi .nd.on{opacity:1;animation:psi-pop .5s ease}
@keyframes psi-pop{50%{transform:scale(1.8)}}
`,
    mount,
    enter(el, run) {
      reset(el);
      (async () => {
        await run.wait(300);
        await pass(el, run, 0);
        await run.wait(2200); // let call 1 sink in: the served box stays lit long enough to point at
        clear(el);
        await run.wait(450);
        await pass(el, run, 1);
      })();
    },
    leave: reset,
  });
})();
