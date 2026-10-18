// Injected into every captured page: an animated cursor (eased moves, press + ripple) and the subtitle card.
// window.__ov.start(globalT, cues) starts the subtitle clock at final-timeline time globalT.
(() => {
  if (window.__ov) return;
  // the real clock, kept before warp (record.js) can speed up the page's own timers
  const now = performance.now.bind(performance), raf = requestAnimationFrame.bind(window), rst = setTimeout.bind(window);
  const css = `
  #__ov{position:fixed;inset:0;pointer-events:none;z-index:2147483647}
  #__cur{position:absolute;left:0;top:0;width:40px;height:40px;transform-origin:4px 3px;will-change:transform;
    filter:drop-shadow(2px 3px 0 rgba(43,43,51,.22))}
  #__cur svg{display:block}
  .__rip{position:absolute;width:20px;height:20px;margin:-10px 0 0 -10px;border-radius:50%;border:3px solid #6A4FD8;
    background:rgba(106,79,216,.16);animation:__rip .6s cubic-bezier(.2,.7,.3,1) forwards}
  @keyframes __rip{from{transform:scale(.4);opacity:1}to{transform:scale(3.4);opacity:0}}
  #__sub{position:absolute;left:50%;bottom:44px;transform:translateX(-50%);max-width:1700px;white-space:nowrap;
    font-family:'Bricolage Grotesque','Trebuchet MS',sans-serif;font-optical-sizing:auto;font-weight:560;font-size:40px;line-height:1.2;
    letter-spacing:-.005em;color:#2B2B33;background:#FFFEFB;border:2.5px solid #2B2B33;border-radius:18px;
    box-shadow:4px 5px 0 rgba(43,43,51,.16);padding:13px 30px 15px;opacity:0;transition:opacity .14s ease}
  #__sub.on{opacity:1}
  #__sub.top{bottom:auto;top:40px}
  #__sub em{font-style:normal;color:#6A4FD8;font-weight:700}`;
  const st = document.createElement('style'); st.textContent = css;
  const ov = document.createElement('div'); ov.id = '__ov';
  ov.innerHTML = `<div id="__sub"></div><div id="__cur"><svg width="40" height="40" viewBox="0 0 40 40">
    <path d="M5 3 L5 31 L12.2 24.6 L17 35.4 L22.4 33 L17.6 22.4 L27.4 22 Z" fill="#2B2B33" stroke="#fff" stroke-width="2.4" stroke-linejoin="round"/></svg></div>`;
  const mount = () => { document.head.appendChild(st); document.body.appendChild(ov); };
  document.body ? mount() : addEventListener('DOMContentLoaded', mount);

  const cur = () => document.getElementById('__cur');
  let x = 1500, y = 980, s = 1, hidden = false;
  const draw = () => { cur().style.transform = `translate(${x - 4}px,${y - 3}px) scale(${s})`; cur().style.opacity = hidden ? 0 : 1; };
  const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  let anim = 0;
  function move(tx, ty, ms) {
    const id = ++anim, fx = x, fy = y, t0 = now();
    // a slight arc reads as a hand, not a robot
    const dx = tx - fx, dy = ty - fy, bend = Math.min(80, Math.hypot(dx, dy) * 0.12);
    const nx = -dy / (Math.hypot(dx, dy) || 1), ny = dx / (Math.hypot(dx, dy) || 1);
    return new Promise((res) => {
      const step = () => {
        if (id !== anim) return res();
        const t = Math.min(1, (now() - t0) / ms), e = ease(t), arc = Math.sin(Math.PI * e) * bend;
        x = fx + dx * e + nx * arc; y = fy + dy * e + ny * arc; draw();
        t < 1 ? raf(step) : res();
      };
      raf(step);
    });
  }
  function press() {
    s = 0.86; draw(); rst(() => { s = 1; draw(); }, 140);
    const r = document.createElement('div'); r.className = '__rip'; r.style.left = x + 'px'; r.style.top = y + 'px';
    ov.appendChild(r); rst(() => r.remove(), 700);
  }
  function set(nx, ny) { x = nx; y = ny; draw(); }
  function hide(h) { hidden = h; draw(); }

  // subtitles
  let cues = [], g0 = 0, p0 = 0, last = -1, top = false;
  const EM = /(87%|2\.6 seconds|5 milliseconds|2,000|500 times|Psi\b|byte-identical|Dreams tab|scores|self-hosted|fully managed|Keep on dreaming\.)/g;
  const esc = (t) => t.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  function tick() {
    const t = g0 + (now() - p0) / 1000;
    const i = cues.findIndex((c) => t >= c.start && t < c.end);
    const el = document.getElementById('__sub');
    if (el) {
      if (i !== last) {
        last = i;
        if (i >= 0) { el.innerHTML = esc(cues[i].text).replace(EM, '<em>$1</em>'); el.classList.add('on'); }
        else el.classList.remove('on');
      }
      el.classList.toggle('top', top);
    }
    raf(tick);
  }
  function start(globalT, cueList) { g0 = globalT; p0 = now(); cues = cueList; last = -2; raf(tick); }
  function subTop(v) { top = v; }
  window.__ov = { move, press, set, hide, start, subTop, pos: () => [x, y] };
})();
