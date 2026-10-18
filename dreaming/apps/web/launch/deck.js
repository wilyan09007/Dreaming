// The launch deck: a fixed 1600x900 stage scaled to any window, one screen at a time, keyboard/clicker nav, and an
// optional presenter window (?presenter) that drives the display window over a BroadcastChannel.
// Screens register with Launch.add({ id, title, notes, css, mount(el), enter(el, run), leave(el), key(e, run) }).
// `run` is a per-visit scheduler: every timer/frame it starts is cancelled when the screen is left.
window.Launch = (() => {
  const screens = [];
  const W = 1600, H = 900;
  const stage = document.getElementById('stage');
  const dots = document.getElementById('dots');
  const presenter = new URLSearchParams(location.search).has('presenter');
  const bc = 'BroadcastChannel' in window ? new BroadcastChannel('dreaming-launch') : null;
  let cur = -1, run = null;

  const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  function scheduler() {
    const timers = new Set(), frames = new Set();
    let alive = true;
    const r = {
      alive: () => alive,
      after(ms, fn) { const id = setTimeout(() => { timers.delete(id); if (alive) fn(); }, ms); timers.add(id); return id; },
      every(ms, fn) { const id = setInterval(() => alive && fn(), ms); timers.add(id); return id; },
      wait: (ms) => new Promise((res) => r.after(ms, res)), // never resolves after leave, so awaiting chains just stop
      frame(fn) { // fn(now) each animation frame until it returns false
        const loop = (now) => { frames.delete(id); if (!alive || fn(now) === false) return; id = requestAnimationFrame(loop); frames.add(id); };
        let id = requestAnimationFrame(loop); frames.add(id);
      },
      tween(ms, fn, e = ease) { // fn(0..1); resolves at 1
        return new Promise((res) => { const t0 = performance.now(); r.frame((now) => { const t = Math.min(1, (now - t0) / ms); fn(e(t)); if (t === 1) { res(); return false; } }); });
      },
      count(el, to, ms, fmt = (v) => Math.round(v).toLocaleString(), from = 0) {
        return r.tween(ms, (t) => { el.textContent = fmt(from + (to - from) * t); }, (t) => 1 - Math.pow(1 - t, 3));
      },
      stop() { alive = false; timers.forEach((id) => { clearTimeout(id); clearInterval(id); }); frames.forEach(cancelAnimationFrame); },
    };
    return r;
  }

  function fit() {
    const h = presenter ? innerHeight * 0.66 : innerHeight; // the presenter window keeps its notes below the stage
    const s = Math.min(innerWidth / W, h / H);
    stage.style.transform = `translate(${(innerWidth - W * s) / 2}px,${(h - H * s) / 2}px) scale(${s})`;
  }

  function go(i, from = 'local') {
    i = Math.max(0, Math.min(screens.length - 1, i));
    if (i === cur) return;
    const prev = screens[cur];
    if (prev) { run && run.stop(); prev.el.classList.remove('on'); prev.leave && prev.leave(prev.el); }
    cur = i;
    const s = screens[i];
    s.el.classList.add('on');
    history.replaceState(null, '', location.search + '#' + s.id);
    [...dots.children].forEach((d, k) => d.setAttribute('aria-current', k === i ? 'step' : 'false'));
    run = scheduler();
    s.enter && s.enter(s.el, run);
    if (presenter) renderPresenter();
    if (from === 'local' && bc) bc.postMessage({ go: i });
  }
  function replay() { const s = screens[cur]; if (!s) return; run && run.stop(); s.leave && s.leave(s.el); run = scheduler(); s.enter && s.enter(s.el, run); if (bc) bc.postMessage({ replay: cur }); }

  function renderPresenter() {
    const s = screens[cur], n = screens[cur + 1];
    document.getElementById('p-now').textContent = `${cur + 1}/${screens.length} · ${s.title}`;
    document.getElementById('p-notes').textContent = s.notes || '';
    document.getElementById('p-next').textContent = n ? `next: ${n.title}` : 'last screen';
  }

  addEventListener('keydown', (e) => {
    if (e.target.closest('input,textarea,select,[contenteditable]')) return; // the build screen types freely
    const s = screens[cur];
    if (s && s.key && s.key(e, run)) { e.preventDefault(); return; }
    const k = e.key;
    if (k === 'ArrowRight' || k === 'PageDown' || k === ' ') go(cur + 1);
    else if (k === 'ArrowLeft' || k === 'PageUp') go(cur - 1);
    else if (k === 'Home') go(0);
    else if (k === 'End') go(screens.length - 1);
    else if (k === 'r' || k === 'R') replay();
    else if (k === 'f' || k === 'F') document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen().catch(() => {});
    else if (/^[1-9]$/.test(k)) go(+k - 1);
    else return;
    e.preventDefault();
  });
  if (bc) bc.onmessage = ({ data }) => { if ('go' in data) go(data.go, 'remote'); if ('replay' in data && data.replay === cur) { run && run.stop(); const s = screens[cur]; s.leave && s.leave(s.el); run = scheduler(); s.enter && s.enter(s.el, run); } };
  addEventListener('resize', fit);
  addEventListener('hashchange', () => { const i = screens.findIndex((s) => s.id === location.hash.slice(1)); if (i >= 0) go(i); });

  function add(def) {
    const el = document.createElement('section');
    el.className = 'screen';
    el.id = 's-' + def.id;
    el.setAttribute('aria-label', def.title);
    stage.appendChild(el);
    if (def.css) { const st = document.createElement('style'); st.textContent = def.css; document.head.appendChild(st); }
    def.mount && def.mount(el);
    const d = document.createElement('button');
    d.title = `${screens.length + 1} · ${def.title}`;
    d.setAttribute('aria-label', d.title);
    const i = screens.length;
    d.onclick = () => go(i);
    dots.appendChild(d);
    screens.push({ ...def, el });
  }

  function start() {
    document.body.classList.toggle('presenter', presenter);
    fit();
    const id = location.hash.slice(1);
    go(Math.max(0, screens.findIndex((s) => s.id === id)));
  }
  return { add, start, go, next: () => go(cur + 1), prev: () => go(cur - 1), W, H };
})();
