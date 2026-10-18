// Layout probe: screenshot every deck screen and live tab at 1920x1080 (frames go to ./probe/).
const { chromium } = require('playwright');
const path = require('path');
const DECK = 'file:///' + path.resolve(__dirname, '../../dreaming/apps/web/launch/index.html').replace(/\\/g, '/');
const OUT = process.argv[2] || 'probe';
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1920, height: 1080 } });
  for (const s of ['hook', 'race', 'bench', 'psi', 'train', 'build', 'close']) {
    await p.goto(DECK + '#' + s); await p.waitForTimeout(s === 'race' ? 13000 : 5000);
    await p.screenshot({ path: `${OUT}/deck-${s}.png` });
  }
  const fake = (r) => r.request().method() === 'POST' ? r.fulfill({ status: 200, body: '{}', contentType: 'application/json' }) : r.continue();
  await p.route('**/batch**', fake); await p.route('**/budget**', fake);
  for (const t of ['flows', 'dreams', 'replay', 'scores']) {
    await p.goto('https://dreaming-demo.fly.dev/#' + t); await p.waitForTimeout(t === 'scores' ? 9000 : 4000);
    await p.screenshot({ path: `${OUT}/live-${t}.png` });
  }
  await b.close();
})();
