export const meta = {
  name: 'dreaming-followups',
  description: 'Follow-up smokes: tool prediction on real agent traces, router on yes/no + bookkeeping steps with better per-call signals, pruner on a draft_note step, and few-data training accelerators',
  phases: [
    { title: 'Build', detail: 'one builder per smoke: S14 real tool traces, S15 router steps, S16 draft_note pruner, S17 accelerators' },
    { title: 'Critique', detail: 'skeptic tries to refute each result' },
    { title: 'Fix', detail: 'builder fixes material issues and reruns from caches' },
  ],
}

const ROOT = 'C:\\Users\\<user>\\Desktop\\personalprojects\\Dreaming'
const COMMON = `
ENVIRONMENT AND RULES (apply to every step):
- Project root: ${ROOT}. Python: "${ROOT}\\.venv\\Scripts\\python.exe" (python 3.12; torch-cpu, sentence-transformers, scikit-learn, datasets, transformers, peft, openai, pandas, httpx, scipy). Install extra packages ONLY with: uv pip install --python "${ROOT}\\.venv\\Scripts\\python.exe" <pkg>. If something needs its own Python environment (e.g. tau2-bench), create a separate uv venv inside your smoke folder. Windows, CPU only; set HF_HUB_DISABLE_SYMLINKS_WARNING=1 and PYTHONIOENCODING=utf-8 when running python.
- Downloads go in their own new folder under ${ROOT}\\smokes\\.data\\<name>\\ ; never run code from inside a download folder; read downloaded files from scripts kept elsewhere, with python -I.
- Read first: ${ROOT}\\missions.md (product spec), ${ROOT}\\smokes\\FINDINGS.md (what is already known; do not redo it), ${ROOT}\\smokes\\common.py. Helpers: embed(texts) cached bge-small; load_banking77(); load_clinc150(); confidence(proba) top-2 gap; certify(conf, correct, budget, delta=0.05, grid=40) = FIXED-GRID Clopper-Pearson certificate -> dict(cutoff, coverage, emp_error, bound), cutoff=inf means nothing certifiable; apply_certificate(conf, correct, cutoff); cp_upper(errors, n, delta); stratified_subsample(labels, n); write_results(path, title, lines). Import with: import sys; sys.path.insert(0, r"${ROOT}\\smokes"); from common import *. Do NOT modify common.py.
- Teacher noise is known (${ROOT}\\smokes\\s13_teacher_noise\\RESULTS.md): OpenAI and Anthropic endpoints on OpenRouter ignore temperature; pairwise self-disagreement about 1-3% claude-sonnet-5.5, 3.5% claude-haiku-5.5 with reasoning off, 11% gpt-5.4-mini. Whenever you compare two configurations, also measure each against itself (same config twice) as the noise floor. Pin providers with extra_body={"provider": {"order": [<provider seen on a probe call>], "allow_fallbacks": False}} and log the serving provider per call.
- Keys in ${ROOT}\\.env: OPENROUTERKEY, ASI1KEY (ASI:One, OpenAI-compatible, base https://api.asi1.ai/v1, models asi1-mini / asi1 / asi1-ultra), RUNPODKEY. Read them with python/re; never print them or write them anywhere. OpenRouter: openai client with base_url https://openrouter.ai/api/v1. Use ThreadPoolExecutor(8), retries with backoff, and cache every raw response in responses.jsonl keyed by a hash of the request so reruns cost nothing. Track spend (OpenRouter usage.cost when present, else tokens x listed price from https://openrouter.ai/api/v1/models) and append one line "<smoke>,<usd>,<note>" to ${ROOT}\\smokes\\COSTS.md. Respect the spend cap in your task; project cost from a 5-item pilot before any full run and shrink the run if needed.
- GPU only if truly needed: ${ROOT}\\smokes\\s8_gpu_student\\launch_runpod.py launches a self-terminating community-cloud RTX 4090 pod ($0.34/h). Read it first. Usage from that folder: python launch_runpod.py --script <script in that folder> --args "<args>" --tag _<name> --max-min <minutes>; results come back as results_<name>.json and log_<name>.txt in that folder. Copy your GPU script into that folder (prefix its name with your smoke id).
- Never use mcp__claude-in-chrome tools. WebFetch/WebSearch are fine for docs and data discovery.
- Code: minimal (one run.py plus at most one helper), fixed seeds, keep stdout in log.txt. Other agents are running other smokes on this machine at the same time; work only in your own smoke folder (plus ${ROOT}\\smokes\\.data and the COSTS.md line).
- RESULTS.md (write it with the Write tool) must START with a section "## In plain words": 4-8 short sentences a non-expert can follow, saying what was tested, what the answer is, and what it means for Dreaming. Then: Setup, Results (markdown tables with the real numbers), Interpretation for the mission, Caveats, Time and cost. Report failures as failures; never round a failure into a success.
- Your final text return is data for the orchestrator: 10-20 lines of key numbers and blockers.
`

const SMOKES = [
  { key: 's14_tool_traces', cap: 1, spec: `
SMOKE S14 - Mission 3 on REAL agent tool traces (no synthetic environment). Folder: ${ROOT}\\smokes\\s14_tool_traces. Spend cap: $1 (expected about $0).
Framing (important): Dreaming never predicts the outside world in general. It learns one tool's behaviour from the agent's own log of past calls to that tool (a learned, certified cache), predicts the outcome and latency of calls that change the world so a fallback can be set up, and speculatively starts the next read-only call. Smoke S4 showed these mechanics on a world we generated. The open empirical question: in REAL agent traffic, how often are tool returns reusable or predictable from earlier calls, and how good is next-call speculation? Answer it on public traces.
Data (download what is reachable without login; record sizes and licenses; if something is blocked, say why and move on):
 1. PRIMARY: tau-bench historical trajectories (MIT): https://github.com/sierra-research/tau-bench/tree/main/historical_trajectories . Logged runs of GPT-4o / Claude agents on retail and airline customer-service tasks, several trials per task, every tool call with its real return. This is the closest public analogue of "one workflow run many times".
 2. Optional: tau2-bench (MIT, https://github.com/sierra-research/tau2-bench). If installable in a separate uv venv (python 3.12 or 3.13), replay each task's gold actions against the real deterministic tools (no LLM) for banking_knowledge and retail to get more (call, return) pairs with DB state. Skip if not working within about 30 minutes.
 3. Toucan-1.5M (Apache-2.0, https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M): real MCP server executions. Download ONE parquet shard (check shard sizes with huggingface_hub first; stay under about 1.5 GB) and extract (tool name, arguments, returned text).
 4. TraceLab (CC BY 4.0, https://github.com/uw-syfi/TraceLab releases): real tool latencies (arguments redacted). Latency question only.
Analyses. Split by TASK / EPISODE, never by row, so a cache never sees the episode it is tested on; chronological where timestamps exist.
 A. Reuse: per tool, share of calls whose canonicalized (tool, args) was seen earlier in OTHER episodes; among those, how often the earlier return equals the new one exactly (after canonicalizing only fields you can justify, e.g. generated ids). Do it for (i) an input-keyed cache (tool, args) and (ii) a state-aware cache keyed on (tool, args, the write calls made earlier in the same episode), as in TVCache. Separate read tools from write tools.
 B. Certified skipping: confidence = number of consistent prior observations and their agreement; calibrate on calibration episodes with certify() at budgets 1% and 2%; on test episodes report share of calls skipped and realized error above the cutoff, per tool and overall. This is "how much real tool traffic Dreaming could skip with a certificate".
 C. Outage survival: for the 2-3 most-called read tools, simulate the tool being down for the test episodes; share of its calls answered by the certified cache vs a plain uncertified last-value cache vs a circuit breaker (0%), and the error rate among answered calls.
 D. Write outcomes: for write tools, predict success vs an error return from (tool, argument features, earlier calls in the episode) with a simple model; AUC vs base rate.
 E. Speculation: predict the NEXT tool call (tool name; and name + args) from the episode so far, using an n-gram over tool names plus copy-argument heuristics from earlier returns (or a small classifier); top-1 and top-3 hit rates; compare with the literature (Speculative Actions reports 22-38% on tau-bench retail). Also the share of next calls that are read-only AND correctly predicted, since only those may be speculated.
 F. Latency (TraceLab): can per-tool latency be predicted better than a per-tool median or EWMA? MAE and p90 error; plus the share of total tool time spent in calls longer than 1 s (that is where skipping or speculation saves time).
 G. Toucan: share of tools whose return is a deterministic function of the arguments (repeated identical calls give identical text) vs time-varying (search, weather, prices); per tool category if available, with a few examples.
Deliver a "transfer to Dreaming" table: for each tool case in missions.md (return is a function of inputs; return depends on hidden state; call changes the world) the real-trace number that supports or undercuts it.` },
  { key: 's15_router_steps', cap: 8, spec: `
SMOKE S15 - Mission 1 (tier / thinking router), second pass on step types closer to the demo, with better per-call signals. Folder: ${ROOT}\\smokes\\s15_router_steps. Spend cap: $8 on OpenRouter.
Read ${ROOT}\\smokes\\s5_tier_router\\RESULTS.md first. On a 77-label step (Banking77) cheap tiers matched the expensive one on only 77-82% of inputs; thinking on vs off changed 16.5% of answers; prompt-embedding routers did no better than random mixing; teacher self-disagreement was 6-9% on OpenAI models. The mission's worked example claims thinking off changes only 3 of 200 answers on a yes/no step like flag_anomaly; that has never been tested.
Steps (200 inputs each, fixed seed, real text):
 S-bin: yes/no decision over a record: BoolQ (google/boolq or a parquet mirror; passage + question -> yes/no; gold available).
 S-cat: bookkeeping categorize: DoDataThings/us-bank-transaction-categories-v2 (17 labels, raw bank descriptors -> one label; gold available).
 S-ext: bookkeeping extraction: the same descriptors -> "the merchant name, normalized" (short string; match = lowercase, whitespace- and punctuation-normalized exact match; this is a cheap-vs-expensive comparison and needs no gold).
 Reuse S5's Banking77 numbers as the fourth step; do not re-run it.
Configurations (check each model id on https://openrouter.ai/api/v1/models and whether reasoning can be disabled via reasoning.mandatory before running; pin providers; verify reasoning tokens from usage):
 Thinking pairs: anthropic/claude-haiku-5.5 reasoning enabled (effort medium, or max_tokens 2000) vs disabled; openai/gpt-5.4-mini effort high vs none.
 Tier pairs: anthropic/claude-sonnet-5.5 vs claude-haiku-5.5 (reasoning off where allowed); openai/gpt-5.4 (effort low) vs gpt-5.4-mini (none) vs gpt-5.4-nano (none).
 Run EVERY configuration twice on every input; the repeat is the noise floor. Project cost from a 5-input pilot; drop the most expensive configuration if the projection exceeds the cap.
Measure per step: match rate of each cheap config vs its expensive partner next to each config's self-agreement, gold accuracy where gold exists, cost per 1000 calls, p50 latency. The key table: "share of answers that change when thinking is turned off" per step, against the noise floor.
Per-call routing, second attempt. For "will the cheap answer equal the expensive answer", compare these signals by AUC and by certified coverage at 2%, 5% and 10% budgets (certify on a calibration half, apply on the other half; also a 5-fold rotation with mean coverage): (a) prompt-embedding LogisticRegression (the S5 baseline), (b) cheap self-agreement: the two cheap repeats agree (DART-style), (c) cheap-model confidence from token logprobs where the API returns them (logprobs=true, top_logprobs=5 on OpenAI models; skip if unsupported), (d) random mixing at the same routed share. Report blended cost per 1000 calls at each budget vs always-expensive, counting the extra cheap repeat call in (b).
Interpretation in plain terms: per step type, is routing to the cheaper or no-thinking configuration admissible at a 2-5% budget, per step or per call, and what is the saving.` },
  { key: 's16_pruner_note', cap: 5, spec: `
SMOKE S16 - Mission 5 (context pruner) on its own worked example: a draft_note step, i.e. a step inside a structured workflow whose output is prose. Folder: ${ROOT}\\smokes\\s16_pruner_note. Spend cap: $5.
Read ${ROOT}\\smokes\\s6_context_pruner\\RESULTS.md and ${ROOT}\\smokes\\s6b_pruner_typed\\RESULTS.md first. S6 (HotpotQA short prose answers) was dominated by comparison noise; S6b (typed yes/no) worked. missions.md's example: draft_note receives the transaction, the matched invoice, the anomaly flag, the vendor's full transaction history and the bank-feed record; the note depends on the first three and rarely on the rest.
Inputs: build 300 realistic draft_note prompts from the synthetic bookkeeping world in ${ROOT}\\smokes\\s4_tool_predictor\\env.py and ${ROOT}\\smokes\\s9_rollout\\run.py (read and reuse their generators; import, do not edit them). Each prompt = transaction (descriptor, amount, date), matched invoice or NONE, anomaly flag yes/no with its reason, vendor history as 15-30 individual past-transaction records (one span per record), and a bank-feed record. Mix in cases where the history should matter (amount far above the vendor's typical amount, a duplicate of a recent charge, first purchase from the vendor). The inputs are synthetic; every output and label comes from a real frontier model, which is what Dreaming certifies against.
Teacher: anthropic/claude-haiku-5.5, reasoning off, provider pinned, max_tokens 120. System prompt: write a 1-3 sentence bookkeeping note for the reviewer covering the transaction, the invoice match, and why it was or was not flagged.
Equivalence: an LLM judge (anthropic/claude-sonnet-5.5 with the lowest reasoning allowed, pinned) answers SAME or DIFFERENT: do the two notes state the same facts and conclusion for a reviewer? Validate on 150 pairs with a second judge (openai/gpt-5.4-mini) and report Cohen's kappa. Noise floor = two independent full-prompt notes judged against each other.
Labels: leave-one-out per span with the judge. To control cost you may ablate the four fixed fields individually and the history records in blocks; say exactly what you did. Majority-of-3 where affordable.
Scorers: a learned span scorer (LogisticRegression on span type and record features such as |amount - transaction amount|, same-day or duplicate, recency, cosine to the transaction) trained on 200 prompts; cross-encoder baseline cross-encoder/ms-marco-MiniLM-L-6-v2 with the transaction as query; a rule baseline "keep only transaction, invoice and flag" (missions.md's claim); an oracle = the spans whose ablation changed the note.
Evaluation on 100 held-out prompts: pruned vs full equivalence by the judge, tokens removed, the noise floor, the Clopper-Pearson upper bound on disagreement, and certified coverage at 5% and 10% budgets when low-confidence prompts fall back to the full prompt (certify on a calibration split). Headline: what share of input tokens is removable under a certified 5% (and 10%) budget on this step.` },
  { key: 's17_accelerator', cap: 3, spec: `
SMOKE S17 - Mission 2d reinterpreted: not "write a student from the schema", but "make a new step learn from very few teacher-labelled rows". Folder: ${ROOT}\\smokes\\s17_accelerator. Spend cap: $1 of LLM calls (synthetic examples via ASI:One asi1-mini first, OpenRouter openai/gpt-5.4-nano as fallback, as in ${ROOT}\\smokes\\s2_hypernet\\run2.py) plus up to $2 of RunPod if needed.
Read ${ROOT}\\smokes\\s2_hypernet\\RESULTS.md, RESULTS2.md, ${ROOT}\\smokes\\research\\day-zero-student.md and ${ROOT}\\smokes\\s8_gpu_student\\RESULTS.md (phase B: a shared LoRA pretrained on other steps gave 84% vs 60% at 3 rows per label on a new step).
Question: which kind of model best ACCELERATES learning a new typed step when the user has little workflow data, measured as accuracy and certified coverage versus the number of real labelled rows?
New-step tasks (never used in any pretraining): the 3 held-out CLINC150 domains used in S2b (15-way each), five 12-label Banking77 sub-tasks, and the bookkeeping task DoDataThings/us-bank-transaction-categories-v2 (17 labels, raw bank descriptors). Pretraining pool for methods that need other tasks: the remaining CLINC150 domains plus the Banking77 labels not in the evaluated sub-task (label sets disjoint from each evaluated task; state how you enforce it).
Rows per label k in {0,1,2,3,5,10,20}; 5 random draws per k; a fixed 300-row calibration set per task, separate from training rows, for certify() at a 5% budget; report accuracy and certified coverage.
Accelerators (student = frozen bge-small embeddings + LogisticRegression unless stated):
 1. Baseline: LR on k real rows (k=0: label-description cosine).
 2. Synthetic prior: LLM-written examples per label from the label descriptions (reuse ${ROOT}\\smokes\\.cache\\synth_examples.json and label_descriptions.json; generate descriptions and examples for the DoDataThings labels) mixed with the k real rows.
 3. Workflow-pretrained encoder: fine-tune bge-small on the pretraining pool (supervised contrastive, SetFit-style pairs, or classification with the head discarded), then frozen encoder + LR on the new task. CPU if it fits in about 20 minutes, else RunPod.
 4. Meta-learned starting point: Reptile or first-order MAML over episodes from the pretraining pool for a small head (e.g. a linear projection + cosine classifier on frozen embeddings), then fine-tune on the k rows.
 5. In-context learner: classifies by reading the k examples, no training. Try TabPFN (pip tabpfn; v2 classifier on PCA-reduced embeddings, with the many-class extension if more than 10 classes; if weights are gated or the download fails, say so and skip) and a nearest-prototype classifier as the trivial in-context baseline.
 6. Active learning: instead of k random rows, the current student picks the most uncertain rows from an unlabelled pool to send to the teacher (the gold label stands in for the teacher); same row budgets.
 7. Best combination (e.g. pretrained encoder + synthetic prior + active learning).
Deliver: tables of accuracy and certified coverage per method, per k, per task family; "real rows needed to reach 50% certified coverage at a 5% budget" per method; and a plain-language recommendation of which accelerator(s) Dreaming should build and what to call it instead of "hypernetwork".` },
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
Try to REFUTE the results. Check: leakage (episode/task splits, label sets that overlap pretraining, calibration rows reused for evaluation), the noise floor actually measured and used, providers actually pinned and reasoning actually off/on as claimed (check usage and provider fields), parse failures silently counted as matches, numbers in RESULTS.md that differ from log.txt or from a recompute of the raw files (recompute key numbers yourself with a short CPU-only script; do not spend on APIs), spend recorded in COSTS.md, parts of the spec skipped without saying so, and an "In plain words" section that overstates the evidence. Material = would change a conclusion or violates the spec. Be concrete (file:line, what, why).`,
    { label: `critique:${s.key}`, phase: 'Critique', model: 'sonnet', schema: CRITIC_SCHEMA }).then(c => ({ summary, critique: c })),
  (r, s) => (r.critique && r.critique.material_issues.length)
    ? agent(`${COMMON}
Smoke folder: ${ROOT}\\smokes\\${s.key}. A reviewer found MATERIAL issues:
${r.critique.material_issues.map((m, i) => `${i + 1}. ${m}`).join('\n')}
Minor issues (fix if cheap): ${r.critique.minor_issues.join(' | ')}
Original spec:
${s.spec}
Fix the issues in place. Re-run only what is needed and reuse cached responses instead of re-spending (cap $${s.cap}). Update RESULTS.md, including "## In plain words", and add a "## Revision" section saying what changed and how the numbers moved. Update COSTS.md if you spent more. Return the revised key numbers.`,
      { label: `fix:${s.key}`, phase: 'Fix', model: 'sonnet' }).then(fix => ({ ...r, fix }))
    : r,
)
return results.map((r, i) => ({ smoke: SMOKES[i].key, ...(r || { summary: null, critique: null }) }))