// The choreographed figures every launch screen reads. Base values are measured (smokes/RESULTS.md, Scores tab,
// teacher = StepFun step-5-preview at low effort); where a site has two measured slices we show the better one.
// ponytail: fixed numbers on purpose, the launch flow must never depend on the live stack.
window.N = (() => {
  const sites = [
    // ms = p50 per call; usd1k = $ per 1,000 calls; coverage = share served by the certified stand-in at <=5% error
    { id: 'categorize', label: 'categorize', kind: 'llm', teacherMs: 2600, teacherUsd1k: 0.58, coverage: 0.892, oursMs: 5, oursUsd1k: 0.07, cand: 'Psi' },
    { id: 'vendor_lookup', label: 'vendor lookup', kind: 'tool', teacherMs: 900, teacherUsd1k: 0, coverage: 0.94, oursMs: 3, oursUsd1k: 0, cand: 'exact cache' },
    { id: 'match_invoice', label: 'match invoice', kind: 'llm', teacherMs: 4800, teacherUsd1k: 1.69, coverage: 0.783, oursMs: 6, oursUsd1k: 0.38, cand: 'Psi' },
    { id: 'anything_odd', label: 'anything odd?', kind: 'llm', teacherMs: 3100, teacherUsd1k: 1.11, coverage: 0.8, oursMs: 4, oursUsd1k: 0.22, cand: 'linear head' },
    { id: 'post_to_books', label: 'post to books', kind: 'write', teacherMs: 120, teacherUsd1k: 0, coverage: 0, oursMs: 120, oursUsd1k: 0, cand: 'ledger write (never dreamed)' },
  ];
  const byId = Object.fromEntries(sites.map((s) => [s.id, s]));
  // one invoice through the workflow, every step served by its stand-in (the common case) vs the teacher on every step
  const taskTeacherMs = sites.reduce((a, s) => a + s.teacherMs, 0); // 11,520 ms
  const taskOursMs = sites.reduce((a, s) => a + s.oursMs, 0); // 138 ms
  // expected values over 1,000 invoices (misses fall back to the teacher; 1 in 20 served calls is audited)
  const per1k = sites.reduce((a, s) => {
    a.teacherUsd += s.teacherUsd1k; a.oursUsd += s.oursUsd1k;
    if (s.kind === 'llm') { a.teacherCalls += 1000; a.oursCalls += Math.round(1000 * (1 - s.coverage) + (1000 * s.coverage) / 20); }
    a.teacherHours += (1000 * s.teacherMs) / 3.6e6;
    a.oursHours += (1000 * (s.coverage * s.oursMs + (1 - s.coverage) * s.teacherMs)) / 3.6e6;
    return a;
  }, { teacherUsd: 0, oursUsd: 0, teacherCalls: 0, oursCalls: 0, teacherHours: 0, oursHours: 0 });
  // $3.38 vs $0.67 per 1,000 invoices (-80%); 3,000 vs 649 LLM calls; 3.2 h vs 0.6 h sequential
  return {
    headline: { cheaper: 87, faster: 500, tokenFree: 89, bound: 5 }, // "up to 87% cheaper, ~500x faster per call, 0 LLM tokens on 89% of calls, certified <=5% error"
    teacher: 'step-5-preview', sites, byId, taskTeacherMs, taskOursMs, per1k,
    selfDisagree: 3.5, // the teacher disagrees with itself 3.5% of the time: budgets below that are unmeasurable
    candidates: ['linear head', 'Psi', 'fine-tuned encoder', 'hdc bundle', 'exact cache'],
    rungs: [60, 200, 500, 2000], cells: 45, gpuUsd: 4.8, // campaign: 45 cells, whole campaign under $5 of GPU
    psi: { traces: 8219, dim: 384, encoder: 'bge-small', beta: 8, audit: 20 },
    replay: { identical: '3 of 3', hashes: 'weights_sha + env_sha' },
  };
})();
