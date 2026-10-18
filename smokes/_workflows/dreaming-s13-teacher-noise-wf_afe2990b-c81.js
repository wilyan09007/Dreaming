export const meta = {
  name: 'dreaming-s13-teacher-noise',
  description: 'Measure the teacher noise floor (temperature-0 self-agreement) and frontier-step latency on OpenRouter, which bounds every certificate',
  phases: [{ title: 'Build' }, { title: 'Critique' }],
}
const ROOT = 'C:\\Users\\<user>\\Desktop\\personalprojects\\Dreaming'
const COMMON = `
ENVIRONMENT AND RULES:
- Project root: ${ROOT}. Python: "${ROOT}\\.venv\\Scripts\\python.exe" (openai, datasets, numpy, pandas installed). Set HF_HUB_DISABLE_SYMLINKS_WARNING=1. Helpers: import sys; sys.path.insert(0, r"${ROOT}\\smokes"); from common import load_banking77, stratified_subsample.
- OPENROUTERKEY in ${ROOT}\\.env (read with python/re; never print). openai client with base_url https://openrouter.ai/api/v1. ThreadPoolExecutor(8), retries with backoff, temperature=0, max_tokens 20. TRACK SPEND from usage and append "s13_teacher_noise,<usd>,<note>" to ${ROOT}\\smokes\\COSTS.md. Spend cap $4.
- Read ${ROOT}\\missions.md ("Certify", "Reality is the teacher") so you know why the teacher's own noise matters: a certificate promises error <= budget against the teacher; if the teacher flips its own answer x% of the time at temperature 0, budgets below x are not measurable.
- Never use mcp__claude-in-chrome tools. Keep code minimal (one run.py), log to log.txt, save every raw response to responses.jsonl so nothing is re-spent. Final return: 10-15 lines of key numbers.
`
const SPEC = `
SMOKE S13 — teacher noise floor and frontier-step latency. Folder: ${ROOT}\\smokes\\s13_teacher_noise.
Inputs: 100 Banking77 test queries (stratified_subsample(test labels, 100, seed=0)). Prompt as in a typed step: system = "You are a bank support intent classifier. Reply with exactly one label from the list, nothing else." + the 77 label names one per line; user = query. Parse to a label (exact after strip/lower, else nearest by difflib; count parse failures).
Models (check each exists on https://openrouter.ai/api/v1/models first; skip missing ones): anthropic/claude-haiku-5.5, anthropic/claude-sonnet-5.5, openai/gpt-5.4-mini (reasoning effort minimal), google/gemini-3.5-flash-lite (reasoning off if the API accepts it), qwen/qwen3-32b (reasoning off). Verify from usage that reasoning tokens are 0 where you turned it off; if a model refuses the setting, note it.
Design: for each model, 5 repeats of the same 100 prompts at temperature 0 = 500 calls per model: condition A default routing; condition B provider-pinned: extra_body={"provider": {"order": [<the provider the first call reported in response.provider or the model's primary endpoint from /models/{author}/{slug}/endpoints>], "allow_fallbacks": False}} plus seed=0, for the two most important models only (haiku-5.5 and gpt-5.4-mini) to stay under the cap. Estimate cost from 5 calls per model before the full run and shrink repeats to 3 if the projection exceeds $3.
Measure per model/condition: (1) flip rate = share of prompts whose 5 answers are not all identical; (2) pairwise disagreement rate between repeats; (3) majority-vote accuracy vs gold and single-run accuracy vs gold; (4) latency p50/p95 ms and tokens; (5) the provider(s) that served the calls (response headers/fields) and whether flips coincide with provider changes; (6) parse failures. Then the consequence: for each model, the smallest error budget that is measurable (greater than the pairwise disagreement rate) and the cost per 1000 teacher labels.
Write RESULTS.md: setup, a table per measure, interpretation (which teacher to use as the certification reference, whether provider pinning is required, what budget floor to recommend), caveats, time and cost.`
const CRITIC_SCHEMA = { type: 'object', properties: { material_issues: { type: 'array', items: { type: 'string' } }, minor_issues: { type: 'array', items: { type: 'string' } }, verdict: { type: 'string' } }, required: ['material_issues', 'minor_issues', 'verdict'] }
phase('Build')
const built = await agent(`${COMMON}\n${SPEC}`, { label: 'build:s13', phase: 'Build', model: 'sonnet' })
phase('Critique')
const crit = await agent(`${COMMON}
Skeptical reviewer: read ${ROOT}\\smokes\\s13_teacher_noise\\run.py, log.txt, RESULTS.md and sample responses.jsonl. Builder summary:
---
${built}
---
Spec:
${SPEC}
Refute: flips caused by parse differences rather than answer differences, reasoning not actually off, provider pinning not actually applied (check extra_body and the provider field in responses), numbers in RESULTS.md not matching responses.jsonl (recompute from the jsonl yourself with a short script, CPU only), spend not recorded. Material = changes a conclusion.`, { label: 'critique:s13', phase: 'Critique', model: 'sonnet', schema: CRITIC_SCHEMA })
return { built, crit }