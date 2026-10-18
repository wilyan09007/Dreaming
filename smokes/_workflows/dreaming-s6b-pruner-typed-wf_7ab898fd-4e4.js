export const meta = {
  name: 'dreaming-s6b-pruner-typed',
  description: 'Second pass on the context pruner: typed yes/no step, low-noise pinned teacher, majority-of-3 ablation labels',
  phases: [{ title: 'Build' }, { title: 'Critique' }, { title: 'Fix' }],
}
const ROOT = 'C:\\Users\\<user>\\Desktop\\personalprojects\\Dreaming'
const COMMON = `
ENVIRONMENT AND RULES:
- Project root: ${ROOT}. Python: "${ROOT}\\.venv\\Scripts\\python.exe" (torch-cpu, sentence-transformers, sklearn, datasets, openai, scipy). Set HF_HUB_DISABLE_SYMLINKS_WARNING=1. Helpers in ${ROOT}\\smokes\\common.py: embed(texts) cached bge-small embeddings; certify(conf, correct, budget, delta=0.05); apply_certificate; cp_upper(errors, n, delta). Import with sys.path.insert(0, r"${ROOT}\\smokes"); from common import *.
- OPENROUTERKEY in ${ROOT}\\.env (read with python/re; never print). openai client, base_url https://openrouter.ai/api/v1, ThreadPoolExecutor(8), retries with backoff. TRACK SPEND from usage (or usage.cost) and append "s6b_pruner_typed,<usd>,<note>" to ${ROOT}\\smokes\\COSTS.md. Spend cap $4.
- Read first: ${ROOT}\\missions.md section "### 5. Context pruner"; ${ROOT}\\smokes\\s6_context_pruner\\RESULTS.md (the first pass; do NOT modify that folder, another agent may be revising it) and ${ROOT}\\smokes\\s13_teacher_noise\\RESULTS.md (teacher noise: claude-haiku-5.5 with reasoning off has ~3.5% pairwise disagreement on labels; OpenAI endpoints ignore temperature; pin providers). You may reuse code from s6's run.py by copying it into the new folder.
- Work only in ${ROOT}\\smokes\\s6b_pruner_typed\\. Minimal code (one run.py + optional helper), log to log.txt, save every raw response to responses.jsonl, RESULTS.md with Setup / Results tables / Interpretation / Caveats / Time and cost. Never use mcp__claude-in-chrome tools. Final return: 10-15 lines of key numbers.
`
const SPEC = `
SMOKE S6b — context pruner, second pass. NOTE: an earlier attempt of this exact smoke was interrupted; the folder may already contain run.py, log.txt and responses.jsonl. Read them first and REUSE every cached response (key by model+prompt hash) instead of re-spending; only make the calls that are missing. The first pass (S6) failed because the free-text teacher (gpt-5.4-nano) disagreed with itself 36% of the time on identical prompts, which made single-call leave-one-out ablation labels noise and made agreement uncertifiable. Fix the two causes: a TYPED output and a low-noise, provider-pinned teacher with majority-of-3 labels.
Data: HotpotQA distractor validation, COMPARISON questions whose gold answer is exactly "yes" or "no" (filter type == "comparison" and answer.lower() in {"yes","no"}); take 200 with a fixed seed (120 for scorer training, 80 held out). Each has 10 paragraphs, 2 gold supporting ones. Prompt: system "Answer the question using only the context. Reply with exactly one word: yes or no."; user = numbered paragraphs (title + text) then the question. Parse to yes/no (anything else = parse failure, counted).
Teacher: anthropic/claude-haiku-5.5 with extra_body={"reasoning": {"enabled": false}}, max_tokens 5. Pin the provider: first make 3 probe calls, read the provider field from the response (response.provider or the model_extra / headers; log it), then pass extra_body={"provider": {"order": [<that provider slug>], "allow_fallbacks": false}, "reasoning": {"enabled": false}} for all remaining calls and verify every response came from that provider. Verify reasoning tokens are 0.
Labels: A_full = majority of 3 full-context calls (log the 3-call self-agreement rate: this is the noise floor). For each paragraph i: remove it, ask 3 times, majority vote; paragraph "matters" iff the majority answer differs from A_full. That is 200 questions × 11 prompts × 3 calls = 6,600 calls at ~1.3k tokens each ≈ 8.6M input tokens ≈ $0.9 + output; estimate from the first 5 questions and shrink to 150 questions if the projection exceeds $3.
Scorers (rank paragraphs per question): (1) learned: LogisticRegression on features [cosine(embed(question), embed(paragraph)), paragraph position, token length, question-word overlap, title-in-question, plus RELATIVE features: cosine rank within the question and cosine minus the question's max] trained on the 120 training questions' ablation labels; (2) zero-shot cross-encoder cross-encoder/ms-marco-MiniLM-L-6-v2 (sentence_transformers.CrossEncoder); (3) oracle = gold supporting paragraphs; (4) random.
Evaluation on the 80 held-out questions: for k in {2,3,4} keep top-k paragraphs and ask the pinned teacher 3 times (majority) → agreement with A_full, accuracy vs gold, tokens removed %, and the Clopper-Pearson 95% upper bound on disagreement (n=80). Report AUC of each scorer at predicting the majority ablation label (pooled and within-question), the fraction of paragraphs that matter (gold vs non-gold), the 3-call self-agreement of the teacher on full prompts, and the share of questions where a 5% / 10% / 15% disagreement budget would be certifiable for each scorer+k (use cp_upper on the 80 held-out rows; also state what n would be needed at the observed rates). Then write the interpretation: does a typed output + low-noise teacher make the ablation labels informative, and which scorer+k is deployable at what budget?`
const CRITIC_SCHEMA = { type: 'object', properties: { material_issues: { type: 'array', items: { type: 'string' } }, minor_issues: { type: 'array', items: { type: 'string' } }, verdict: { type: 'string' } }, required: ['material_issues', 'minor_issues', 'verdict'] }
phase('Build')
const built = await agent(`${COMMON}\n${SPEC}`, { label: 'build:s6b', phase: 'Build', model: 'sonnet' })
phase('Critique')
const crit = await agent(`${COMMON}
Skeptical reviewer: read ${ROOT}\\smokes\\s6b_pruner_typed\\run.py, log.txt, RESULTS.md and recompute the key numbers from responses.jsonl with a short CPU-only script (no API calls). Builder summary:
---
${built}
---
Spec:
${SPEC}
Refute: provider not actually pinned (check the provider field on every response), reasoning tokens not zero, parse failures counted as answers, majority votes computed wrong, training questions leaking into evaluation, numbers in RESULTS.md not matching the jsonl, spend not recorded, conclusions not supported by n=80 (check the CP bounds). Material = changes a conclusion or violates the spec.`, { label: 'critique:s6b', phase: 'Critique', model: 'sonnet', schema: CRITIC_SCHEMA })
phase('Fix')
const fix = (crit && crit.material_issues.length) ? await agent(`${COMMON}
Reviewer found material issues with ${ROOT}\\smokes\\s6b_pruner_typed:
${crit.material_issues.map((m, i) => `${i + 1}. ${m}`).join('\n')}
Minor: ${crit.minor_issues.join(' | ')}
Spec:
${SPEC}
Fix in place, reusing responses.jsonl wherever possible (respect the $4 cap), add "## Revision" to RESULTS.md, update COSTS.md if you spent more. Return revised key numbers.`, { label: 'fix:s6b', phase: 'Fix', model: 'sonnet' }) : null
return { built, crit, fix }