export const meta = {
  name: 'dreaming-followups-v2',
  description: 'Answer the user notes with evidence: real tool traces (S14), confidence signals vs external predictor (S15), draft_note pruner with a validated judge (S16), warm start incl. many-task hypernetwork (S17), plus a literature check',
  phases: [
    { title: 'Build', detail: 'S14, S15, S16, S17 builders + literature agent' },
    { title: 'Critique', detail: 'skeptic per smoke' },
    { title: 'Fix', detail: 'fix material issues from caches' },
  ],
}

const ROOT = 'C:\\Users\\<user>\\Desktop\\personalprojects\\Dreaming'
const COMMON = `
ENVIRONMENT AND RULES (apply to every step):
- Project root: ${ROOT}. Python: "${ROOT}\\.venv\\Scripts\\python.exe" (python 3.12; torch-cpu, sentence-transformers, scikit-learn, datasets, transformers, peft, openai, pandas, httpx, scipy, setfit). Install extra packages ONLY with: uv pip install --python "${ROOT}\\.venv\\Scripts\\python.exe" <pkg>. If something needs its own environment (e.g. tau2-bench), create a separate uv venv inside your smoke folder. Windows, CPU only; set HF_HUB_DISABLE_SYMLINKS_WARNING=1 and PYTHONIOENCODING=utf-8 when running python.
- Downloads go in their own new folder under ${ROOT}\\smokes\\.data\\<name>\\ ; never run code from inside a download folder; read downloaded files from scripts kept elsewhere, with python -I.
- Read first: ${ROOT}\\missions.md (product spec), ${ROOT}\\smokes\\FINDINGS.md (what is already known; do not redo it), ${ROOT}\\smokes\\common.py. Helpers: embed(texts) cached bge-small; load_banking77(); load_clinc150(); confidence(proba) top-2 gap; certify(conf, correct, budget, delta=0.05, grid=40) = fixed-grid Clopper-Pearson certificate -> dict(cutoff, coverage, emp_error, bound), cutoff=inf means nothing certifiable; apply_certificate(conf, correct, cutoff); cp_upper(errors, n, delta); stratified_subsample(labels, n); write_results(path, title, lines). Import with: import sys; sys.path.insert(0, r"${ROOT}\\smokes"); from common import *. Do NOT modify common.py.
- Teacher noise is known (${ROOT}\\smokes\\s13_teacher_noise\\RESULTS.md): OpenAI and Anthropic endpoints on OpenRouter ignore temperature; pairwise self-disagreement about 1-3% claude-sonnet-5.5, 3.5% claude-haiku-5.5 (reasoning off), 11% gpt-5.4-mini. Whenever you compare two configurations, also measure each against itself (same config twice) as the noise floor. Pin providers with extra_body={"provider": {"order": [<provider seen on a probe call>], "allow_fallbacks": False}} and log the serving provider per call.
- Keys in ${ROOT}\\.env: OPENROUTERKEY, ASI1KEY (ASI:One, OpenAI-compatible, base https://api.asi1.ai/v1, models asi1-mini / asi1 / asi1-ultra), RUNPODKEY. Read them with python/re; never print them or write them anywhere. OpenRouter: openai client with base_url https://openrouter.ai/api/v1. Use ThreadPoolExecutor(8), retries with backoff, and cache every raw response in responses.jsonl keyed by a hash of the request so reruns cost nothing. Track spend (OpenRouter usage.cost when present, else tokens x listed price from https://openrouter.ai/api/v1/models) and append one line "<smoke>,<usd>,<note>" to ${ROOT}\\smokes\\COSTS.md. Respect the spend cap; project cost from a 5-item pilot before any full run and shrink the run if needed.
- GPU only if truly needed: ${ROOT}\\smokes\\s8_gpu_student\\launch_runpod.py launches a self-terminating community-cloud RTX 4090 pod ($0.34/h). Read it first. From that folder: python launch_runpod.py --script <script in that folder> --args "<args>" --tag _<name> --max-min <minutes>; results come back as results_<name>.json and log_<name>.txt there. Copy your GPU script into that folder with your smoke id as prefix.
- Never use mcp__claude-in-chrome tools. WebFetch/WebSearch are fine.
- Code: minimal (one run.py plus at most one helper), fixed seeds, stdout in log.txt. Other agents run other smokes concurrently; work only in your own smoke folder (plus ${ROOT}\\smokes\\.data and the COSTS.md line).
- RESULTS.md (Write tool) must START with "## In plain words": 4-8 short sentences a non-expert can follow (what was tested, the answer, what it means for Dreaming). Then Setup, Results (markdown tables with real numbers), Interpretation for the mission, Caveats, Time and cost. Report failures as failures.
- Final text return: 10-20 lines of key numbers and blockers (data for the orchestrator).
`

const SMOKES = [
  { key: 's14_tool_traces', cap: 1, spec: `
SMOKE S14 - Mission 3 on REAL agent tool traces (no synthetic environment). Folder: ${ROOT}\\smokes\\s14_tool_traces. Spend cap: $1 (expected about $0).
Framing: Dreaming does not forecast the outside world. It learns one tool's behaviour from the agent's own log of past calls to that tool (a learned, certified cache), predicts outcome and latency of world-changing calls so a fallback is ready, and speculatively starts the next read-only call. S4 showed the mechanics on a generated world. Open question: in REAL agent traffic, how often are tool returns reusable or predictable from earlier calls, and how good is next-call speculation?
Data (no login; record sizes and licenses; if blocked, say why and move on):
 1. PRIMARY: tau-bench historical trajectories (MIT): https://github.com/sierra-research/tau-bench/tree/main/historical_trajectories (GPT-4o / Claude agents on retail and airline tasks, several trials per task, every tool call with its real return). Closest public analogue of "one workflow run many times".
 2. Optional: tau2-bench (MIT, https://github.com/sierra-research/tau2-bench) in a separate uv venv (python 3.12/3.13): replay each task's gold actions against the deterministic tools (no LLM) for banking_knowledge and retail to get (call, return, DB state) rows. Skip if not working within ~30 minutes.
 3. Toucan-1.5M (Apache-2.0, https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M): ONE parquet shard (check sizes with huggingface_hub; stay under ~1.5 GB): (tool, arguments, returned text).
 4. TraceLab (CC BY 4.0, https://github.com/uw-syfi/TraceLab releases): real tool latencies (args redacted). Latency only.
Analyses (split by TASK/EPISODE, never by row; chronological where timestamps exist):
 A. Reuse: per tool, share of calls whose canonicalized (tool, args) appeared earlier in OTHER episodes; among those, how often the earlier return equals the new one exactly (canonicalize only fields you can justify). (i) input-keyed cache; (ii) state-aware cache keyed on (tool, args, earlier write calls in the same episode) as in TVCache. Separate read vs write tools.
 B. Certified skipping: confidence = count and agreement of consistent prior observations; certify() on calibration episodes at 1% and 2%; on test episodes: share skipped and realized error above the cutoff, per tool and overall.
 C. Outage survival for the 2-3 most-called read tools: certified cache vs plain last-value cache vs circuit breaker (0%); error rate among answered calls.
 D. Write outcomes: predict success vs error return from (tool, argument features, earlier calls in the episode); AUC vs base rate.
 E. Speculation: predict the NEXT tool call (name; name+args) from the episode so far (n-gram over tool names + copy-argument heuristics, or a small classifier); top-1/top-3 hit rates vs the literature (Speculative Actions: 22-38% on tau-bench retail); share of next calls that are read-only AND correctly predicted.
 F. Latency (TraceLab): per-tool latency predictable better than per-tool median or EWMA? MAE and p90 error; share of total tool time in calls > 1 s.
 G. Toucan: share of tools whose return is a deterministic function of the arguments (identical repeated calls -> identical text) vs time-varying; examples.
Deliver a "transfer to Dreaming" table: for each tool case in missions.md (return = f(inputs); hidden state; call changes the world) the real-trace number that supports or undercuts it, and a one-line answer to "tool prediction good, needs more benching?".` },
  { key: 's15_confidence_signals', cap: 5, spec: `
SMOKE S15 - Which confidence signal should gate a call: the model's own logits, or a second model predicting whether the first was right? Folder: ${ROOT}\\smokes\\s15_confidence_signals. Spend cap: $5 on OpenRouter.
The user's hypothesis: "having another model predict whether the original model was going to be correct is useless; anything useful is already in the logits". Test it on both places Dreaming gates calls.
PART 1 - Student side (CPU, $0). Student = LogisticRegression on frozen bge-small embeddings, Banking77, trained on 770 and on 3080 rows (as in S1/S3). Signals for "this prediction is wrong": (a) top-2 gap from the student's own probabilities; (b) max prob; (c) entropy; (d) temperature-scaled gap (fit temperature on a calibration split); (e) kNN agreement: share of the 10 nearest training neighbours (cosine) that carry the predicted label; (f) an external head: LogisticRegression on the prompt embedding predicting error (like S3's LR feat+emb, but embedding only); (g) external head on (a)-(e) features; (h) combination (a)+(e). Report AUROC for error, and certified coverage at 2% and 5% budgets (certify on a 500-row calibration slice, apply on holdout, 5 random splits, mean and min). Also the same for CLINC150 with a 1500-row student. Conclusion: does anything beat the student's own gap by more than split noise?
PART 2 - Router side (the cheap frontier model vs the expensive one). Reuse ${ROOT}\\smokes\\s5_tier_router\\responses.jsonl: it holds E1 = openai/gpt-5.4 (effort low) answers for 200 Banking77 inputs (two runs, E1 and E1b) and cheap answers; read its run.py to see the key format and the exact prompt, and reuse the SAME 200 inputs and prompt. Do not re-run E1. Define the target as "cheap answer == E1 answer" (and also "== majority of E1/E1b where they agree"). New cheap calls (200 inputs each, pinned provider, reasoning off, temperature 0 where honoured):
  (i) openai/gpt-5.4-nano with logprobs=True, top_logprobs=5 (check /models supported_parameters first; if the OpenAI endpoint via OpenRouter refuses logprobs, try openai/gpt-4.1-nano or qwen/qwen3-32b which S6 showed exposes logprobs; say which you used);
  (ii) the same model a second time with temperature 0.7 (if honoured) or just a repeat, for self-consistency;
  (iii) the same model asked to also output a confidence 0-100 after the label (verbalized confidence), parsed;
  (iv) qwen/qwen3-32b (/no_think) with logprobs, as a second cheap model.
 Signals to compare for routing: (a) prompt-embedding predictor (S5's baseline, 5-fold); (b) the cheap model's own top-token logprob and the gap to the second-best label among the listed labels; (c) self-consistency (two cheap answers agree); (d) verbalized confidence; (e) the Banking77 student's top-2 gap on the same input (train the S1 student on the full train set; the student shadows every call site in Dreaming, so its confidence is free) and (f) student-agrees-with-cheap; (g) random mixing at equal routed share. For each: AUROC for cheap==E1, and certified coverage at 2%, 5%, 10% budgets (certify on a calibration half, apply on the other half, 5 rotations), with the blended cost per 1000 calls including any extra cheap call. Report everything next to the noise floor (E1 vs E1b self-agreement, cheap vs its own repeat).
Deliver a plain-language verdict on the hypothesis for the student side and for the router side, and name the single best gating signal for each.` },
  { key: 's16_pruner_note', cap: 5, spec: `
SMOKE S16 - Mission 5 (context pruner) on its own worked example, draft_note: a step inside a structured workflow whose output is 1-3 sentences of prose. Folder: ${ROOT}\\smokes\\s16_pruner_note. Spend cap: $5.
Read ${ROOT}\\smokes\\s6_context_pruner\\RESULTS.md and ${ROOT}\\smokes\\s6b_pruner_typed\\RESULTS.md first. The user's question: "so draft_note just needs a judge?" Answer it honestly: a judge adds its own noise, so measure the judge too.
Inputs: 300 draft_note prompts from the synthetic bookkeeping world in ${ROOT}\\smokes\\s4_tool_predictor\\env.py and ${ROOT}\\smokes\\s9_rollout\\run.py (import their generators; do not edit them). Each prompt = transaction (descriptor, amount, date), matched invoice or NONE, anomaly flag yes/no with reason, vendor history as 15-30 individual past-transaction records (one span each), and a bank-feed record. Include cases where history should matter (amount far above the vendor's typical, a duplicate of a recent charge, first purchase from the vendor). Inputs are synthetic; every output and label comes from a real frontier model, which is what Dreaming certifies against.
Teacher: anthropic/claude-haiku-5.5, reasoning off, provider pinned, max_tokens 120. System prompt: write a 1-3 sentence bookkeeping note for the reviewer covering the transaction, the invoice match, and why it was or was not flagged.
Judge: anthropic/claude-sonnet-5.5 (lowest reasoning allowed, pinned) answers SAME or DIFFERENT: do the two notes state the same facts and conclusion for a reviewer? Validate on 150 pairs with a second judge (openai/gpt-5.4-mini, reasoning none) and report Cohen's kappa and the judge's self-agreement on repeated pairs. Noise floor = two independent full-prompt notes judged against each other.
Labels: leave-one-out per span with the judge. To control cost, ablate the four fixed fields individually and the history records in blocks (say exactly how); majority-of-3 where affordable.
Scorers: learned span scorer (LogisticRegression on span type and record features: |amount - transaction amount|, same-day or duplicate, recency, cosine to the transaction) trained on 200 prompts; cross-encoder/ms-marco-MiniLM-L-6-v2 with the transaction as query; the rule "keep only transaction, invoice and flag" (missions.md's claim); oracle = spans whose ablation changed the note.
Evaluation on 100 held-out prompts: pruned vs full equivalence by the judge, tokens removed, the noise floor, Clopper-Pearson upper bound on disagreement, certified coverage at 5% and 10% when low-confidence prompts fall back to the full prompt (certify on a calibration split). Headline: share of input tokens removable under a certified 5% (and 10%) budget on this step, and whether the judge is reliable enough to be the measuring stick.` },
  { key: 's17_warm_start', cap: 3, spec: `
SMOKE S17 - Mission 2d reinterpreted as a warm start: make a new typed step learn from very few teacher-labelled rows, and test whether a hypernetwork still has a role when trained on MANY tasks. Folder: ${ROOT}\\smokes\\s17_warm_start. Spend cap: $1 of LLM calls (ASI:One asi1-mini first, OpenRouter openai/gpt-5.4-nano fallback, as in ${ROOT}\\smokes\\s2_hypernet\\run2.py) plus up to $2 of RunPod if needed.
Read ${ROOT}\\smokes\\s2_hypernet\\RESULTS.md, RESULTS2.md, run2.py (reuse its data code), ${ROOT}\\smokes\\research\\day-zero-student.md and ${ROOT}\\smokes\\s8_gpu_student\\RESULTS.md (phase B: a shared LoRA pretrained on other steps gave 84% vs 60% at 3 rows per label on a new step). The S2/S2b hypernetworks were trained on 105 labels from 7 domains and lost to label-embedding baselines on unseen tasks; the literature says Text-to-LoRA needed 64-479 training tasks. Settle whether task starvation was the problem.
Task pool for pretraining (all no-login, embed with common.embed, label names/descriptions via an LLM where missing, cached): CLINC150 train domains, Banking77 labels outside the evaluated sub-tasks, MASSIVE en-US (AmazonScience/massive via its parquet branch or the S3 tarball; 60 intents), SetFit/TREC-QC fine labels (50), SetFit/emotion (6), SetFit/ag_news (4), SetFit/20_newsgroups (20), plus any other ungated short-text classification sets you can load quickly. Build hundreds of distinct 5-15-way tasks by sampling label subsets from each dataset. Evaluation tasks (label sets DISJOINT from all pretraining tasks; say how you enforce it): the 3 held-out CLINC150 domains (15-way), five 12-label Banking77 sub-tasks, and DoDataThings/us-bank-transaction-categories-v2 (17 labels, raw bank descriptors; generate label descriptions). Rows per label k in {0,1,2,3,5,10,20}, 5 draws per k; a fixed 300-row calibration set per task (separate from training rows) for certify() at a 5% budget; report accuracy and certified coverage.
Arms (student = frozen bge-small + LogisticRegression unless stated):
 1. Baseline: LR on k real rows (k=0: label-description cosine).
 2. Synthetic prior: LLM-written examples per label (reuse ${ROOT}\\smokes\\.cache\\synth_examples.json; generate for new labels) mixed with the k real rows.
 3. Workflow-pretrained encoder: fine-tune bge-small on the pretraining pool (classification over the union of pool labels, head discarded, or supervised-contrastive), then frozen encoder + LR on the new task. CPU if it fits in ~20 min on a subsample, else RunPod via the launcher.
 4. Meta-learned starting point: Reptile / first-order MAML over episodes from the pool for a linear projection + cosine head on frozen embeddings; then fine-tune on the k rows.
 5. Hypernetwork, many-task version: the S2b set-transformer (or LINGEN) trained on 500+ tasks from the pool with early stopping on a held-out pool dataset; evaluated on the disjoint evaluation tasks at every k. This is the arm that answers "was it task starvation?".
 6. In-context: TabPFN (pip tabpfn; classifier on PCA-reduced embeddings; many-class extension if needed; if weights are gated or fail to download, say so and skip) and nearest-prototype.
 7. Active learning: the current student picks the most uncertain rows from an unlabelled pool for labelling (gold stands in for the teacher), same row budgets.
 8. Best combination (e.g. pretrained encoder + synthetic prior + active learning).
Deliver: accuracy and certified-coverage tables per arm, k and task family; "real rows needed to reach 50% certified coverage at 5%" per arm; and a plain-language recommendation: which accelerators to build, whether the hypernetwork earns a place once it has hundreds of tasks, and what to call the feature.` },
]

const CRITIC_SCHEMA = {
  type: 'object',
  properties: {
    material_issues: { type: 'array', items: { type: 'string' } },
    minor_issues: { type: 'array', items: { type: 'string' } },
    did_it_run: { type: 'boolean' },
    verdict: { type: 'string' },
  },
  required: ['material_issues', 'minor_issues', 'did_it_run', 'verdict'],
}

phase('Build')
const lit = agent(`${COMMON}
LITERATURE CHECK (WebSearch/WebFetch only, no code). Write ${ROOT}\\smokes\\research\\confidence-and-hypernets.md (under 180 lines) with two parts, each with a Summary of 4-6 bullets, a table of sources (claim | paper/repo | url | year), and Unverified.
Part A - "Is a second model predicting whether the first model is right useless, because it is all in the logits?" Cover: logit/logprob confidence vs trained error predictors vs self-consistency vs verbalized confidence for LLMs (Kadavath et al. 2022 "LMs (mostly) know what they know"; Tian et al. 2023 "Just ask for calibration"; semantic entropy, Kuhn 2023 / Farquhar Nature 2024; Xiong et al. 2024; 2025-2026 follow-ups on confidence estimation for routing/cascades and on "probes on hidden states beat logits"), selective classification (Geifman & El-Yaniv softmax response; "learning to defer" / confidence meta-models; SelectiveNet; Deep Gamblers; evidence on when an external head beats the max-softmax), and cascade routers that use the small model's own signal (FrugalGPT scorer, DART self-agreement, "Routoo"/"Cascade" 2025-26). State plainly where the literature says logits suffice and where an external signal demonstrably helps (e.g. hidden-state probes for hallucination, consistency for free text, API models without logprobs).
Part B - Can a hypernetwork still earn a place as a meta-learner or transfer learner for few-shot classification? Cover HyperTransformer (Zhmoginov et al. 2022), Meta-learning via hypernetworks (Zhao et al. 2020), HyperMAML, "hypernetworks for few-shot" surveys 2023-2026, conditional/FiLM-style task modulation, and the evidence that simple baselines (linear head on strong frozen features, prototypes) match meta-learners on text and under domain shift (Chen et al. 2019, Tian et al. 2020, and 2024-2026 updates). Conclude with the conditions under which a hypernetwork wins (small target models, many training tasks, no strong frozen features) and whether any apply to Dreaming's students.
Return 8-12 lines of key facts with URLs.`, { label: 'research:confidence-and-hypernets', phase: 'Build', model: 'sonnet' })

const results = await pipeline(
  SMOKES,
  s => agent(`${COMMON}\n${s.spec}`, { label: `build:${s.key}`, phase: 'Build', model: 'sonnet' }),
  (summary, s) => agent(`${COMMON}
You are a skeptical reviewer. Smoke folder: ${ROOT}\\smokes\\${s.key}. Read run.py (and any helper), log.txt, RESULTS.md and sample the raw data / responses.jsonl. The builder's summary was:
---
${summary}
---
Original task spec:
${s.spec}
Try to REFUTE the results. Check: leakage (episode/task splits, label sets overlapping pretraining, calibration rows reused for evaluation), the noise floor actually measured and used, providers actually pinned and reasoning actually off/on as claimed (usage and provider fields), parse failures silently counted as matches, numbers in RESULTS.md that differ from log.txt or from a recompute of the raw files (recompute key numbers yourself with a short CPU-only script; do not spend on APIs), spend recorded in COSTS.md, parts of the spec skipped without saying so, and an "In plain words" section that overstates the evidence. Material = would change a conclusion or violates the spec. Be concrete (file:line, what, why).`,
    { label: `critique:${s.key}`, phase: 'Critique', model: 'sonnet', schema: CRITIC_SCHEMA }).then(c => ({ summary, critique: c })),
  (r, s) => (r.critique && r.critique.material_issues.length)
    ? agent(`${COMMON}
Smoke folder: ${ROOT}\\smokes\\${s.key}. A reviewer found MATERIAL issues:
${r.critique.material_issues.map((m, i) => `${i + 1}. ${m}`).join('\n')}
Minor issues (fix if cheap): ${r.critique.minor_issues.join(' | ')}
Original spec:
${s.spec}
Fix the issues in place. Re-run only what is needed and reuse cached responses instead of re-spending (cap $${s.cap}). Update RESULTS.md including "## In plain words", add "## Revision" saying what changed and how the numbers moved, update COSTS.md if you spent more. Return the revised key numbers.`,
      { label: `fix:${s.key}`, phase: 'Fix', model: 'sonnet' }).then(fix => ({ ...r, fix }))
    : r,
)
const litResult = await lit
return { literature: litResult, smokes: results.map((r, i) => ({ smoke: SMOKES[i].key, ...(r || { summary: null, critique: null }) })) }