export const meta = {
  name: 'dreaming-s22-architectures',
  description: 'Smoke data-efficient alternatives to a pretrained adapter (ReFT, adapter composition, unlabelled-stream adaptation, self-training) on a RunPod GPU, not the laptop',
  phases: [{ title: 'Build' }, { title: 'Critique' }, { title: 'Fix' }],
}
const ROOT = 'C:\\Users\\<user>\\Desktop\\personalprojects\\Dreaming'
const COMMON = `
ENVIRONMENT AND RULES:
- Project root: ${ROOT}. Local python "${ROOT}\\.venv\\Scripts\\python.exe" is for WRITING and syntax-checking code only. THE USER'S LAPTOP IS OVERLOADED: do not run any computation locally that takes more than ~60 seconds (no local embedding of datasets, no local training). A --tiny syntax/pipeline check of your GPU script on a handful of rows is allowed once. All real compute goes to RunPod through ${ROOT}\\smokes\\s8_gpu_student\\launch_runpod.py (read it first): from that folder, python launch_runpod.py --script <your script copied there, name prefixed s22_> --args "<args>" --tag _s22 --max-min <minutes>; it ships the script inline, serves live logs over the pod's HTTP proxy, downloads results_s22.json and log_s22.txt, terminates the pod, and appends the cost to COSTS.md. The pod image has torch 2.8 + CUDA; pip install what you need inside the script's job (transformers peft datasets scikit-learn scipy sentence-transformers pyreft etc. are installed by the job.sh "pip install" line in the launcher; add packages by editing a COPY of the launcher if needed, named launch_runpod_s22.py). Budget: at most $3 of RunPod (about 8 pod-hours at $0.34/h; aim for one 40-60 minute run) and $0.5 of OpenRouter/ASI:One.
- Read first: ${ROOT}\\missions.md (students, mission 2), ${ROOT}\\smokes\\FINDINGS.md (S1, S2/S2b, S8 phase B shared-LoRA result: 84% vs 60% at 3 rows/label on a new step; S17 warm-start arms are being run by another agent, do not duplicate them: synthetic prior, workflow-pretrained encoder, Reptile, TabPFN, active learning), ${ROOT}\\smokes\\s8_gpu_student\\train_student.py (reuse its data loading, stratified sampling, certify() copy, evaluate(), LoRA setup, HELD domains = ["travel", "work"]).
- Keys in ${ROOT}\\.env (RUNPODKEY, OPENROUTERKEY, ASI1KEY): read with python/re; never print or write them anywhere.
- Never use mcp__claude-in-chrome tools. Minimal code; fixed seeds; the GPU script must write results incrementally to results.json after every arm so a timeout still leaves data.
- RESULTS.md (Write tool) must START with "## In plain words" (4-8 short sentences a non-expert can follow), then Setup, Results tables, Interpretation, Caveats, Time and cost. Report failures as failures. Final text return: 10-20 lines of key numbers and blockers.
`
const SPEC = `
SMOKE S22 - research-grade alternatives to a pretrained adapter that learn from LESS labelled data at adapter-like cost. Folder: ${ROOT}\\smokes\\s22_architectures (write RESULTS.md and local scripts here; the GPU script lives in s8_gpu_student as s22_arch.py).
Protocol (same across arms): base model Qwen2.5-0.5B with a sequence-classification head (every label scored in one forward pass), bf16 autocast, bs 32, max_len 64, fixed step budget per run (600 steps or 10 epochs, whichever is smaller; same for every arm so cost is matched). New-step tasks = the two held-out CLINC150 domains "travel" and "work" (15 labels each; test = 450 rows per domain) and five 12-label Banking77 sub-tasks (seed 0 label subsets; test = the Banking77 test rows of those labels). Rows per label k in {1, 3, 10, 30}, 3 random draws each (seeds 0-2). Metrics per arm, task, k: accuracy; certified coverage at a 5% budget (certify on half of the test rows, apply on the other half, as in train_student.py); trainable parameters; train seconds. Pretraining pool for arms that need sibling steps: the 8 non-held-out CLINC domains (as in S8 phase B) and, for the Banking77 sub-tasks, the Banking77 labels outside the sub-task (disjoint label sets; state how enforced).
Arms:
 A. LoRA from base (r=16, q/k/v/o + head) = the adapter baseline.
 B. LoRA from the shared pool LoRA (S8 phase B recipe: one LoRA trained on the pool with the step name in the prompt, new head) = pretrained adapter.
 C. ReFT: LoReFT via pyreft (https://github.com/stanfordnlp/pyreft) on the same base, intervening on a few positions/layers (follow the repo's few-shot classification example; rank 4-8), head trained jointly; report its parameter count vs LoRA. If pyreft fails to install or run on this model within ~15 minutes of the pod's time, log the error, fall back to a hand-written LoReFT-style intervention (a low-rank learned edit of the last-token hidden state at one layer) and say so.
 D. Adapter composition (LoraHub-style): train one LoRA per pool step (8 CLINC domains; for Banking77 use 4 random disjoint label groups), then for a new step learn only (i) a new head and (ii) mixing coefficients over the frozen pool LoRAs (one scalar per adapter, softmax-normalised, init uniform) from the k rows; compare "composition only" with "composition then 100 steps of LoRA fine-tune". This is the cheapest way to reuse sibling steps.
 E. Unlabelled stream, no labels needed: (E1) SimCSE-style unsupervised contrastive adaptation of the base on the new step's UNLABELLED inputs (all training texts of the held-out domain minus the k labelled rows, no labels used; dropout positives; 300 steps), then arm A on the k rows; (E2) self-training: arm A on k rows, pseudo-label the unlabelled pool where the top-2 gap is above the 80th percentile, retrain on real + pseudo rows, one or two rounds. Report how much of arm B's gain the unlabelled stream recovers without any sibling steps.
 F. Frozen-embedding reference: bge-small + LogisticRegression on the k rows (from S1/S2b, recompute on the pod; it is the CPU-only incumbent).
Deliver: one table per task family (accuracy and certified coverage by arm and k), parameter counts and train seconds per arm, "rows per label needed to reach 50% certified coverage" per arm, and a plain-language recommendation: which mechanism is the best "learn-from-less" candidate for Dreaming's students, and whether any of them is a credible research-grade replacement for a pretrained adapter.`
const CRITIC_SCHEMA = { type: 'object', properties: { material_issues: { type: 'array', items: { type: 'string' } }, minor_issues: { type: 'array', items: { type: 'string' } }, did_it_run: { type: 'boolean' }, verdict: { type: 'string' } }, required: ['material_issues', 'minor_issues', 'did_it_run', 'verdict'] }
phase('Build')
const built = await agent(`${COMMON}\n${SPEC}`, { label: 'build:s22', phase: 'Build', model: 'sonnet' })
phase('Critique')
const crit = await agent(`${COMMON}
Skeptical reviewer (CPU-light: read files and results_s22.json / log_s22.txt only; no training, no pods). Folder ${ROOT}\\smokes\\s22_architectures plus ${ROOT}\\smokes\\s8_gpu_student\\s22_arch.py, results_s22.json, log_s22.txt. Builder summary:
---
${built}
---
Spec:
${SPEC}
Refute: label overlap between pool and new tasks, unlabelled pool containing test rows, unequal step budgets across arms, certify applied on rows used for calibration, pyreft silently replaced without saying so, numbers in RESULTS.md not matching results_s22.json (recompute with a short script), cost not in COSTS.md, "In plain words" overstating. Material = changes a conclusion or violates the spec.`, { label: 'critique:s22', phase: 'Critique', model: 'sonnet', schema: CRITIC_SCHEMA })
phase('Fix')
const fix = (crit && crit.material_issues.length) ? await agent(`${COMMON}
Folder ${ROOT}\\smokes\\s22_architectures. Reviewer found MATERIAL issues:
${crit.material_issues.map((m, i) => `${i + 1}. ${m}`).join('\n')}
Minor: ${crit.minor_issues.join(' | ')}
Spec:
${SPEC}
Fix in place. Prefer fixes that reuse results_s22.json; relaunch a pod only if a conclusion depends on it and the remaining budget allows (total RunPod for S22 under $3). Update RESULTS.md incl. "In plain words", add "## Revision" with what changed and how numbers moved. Return revised key numbers.`, { label: 'fix:s22', phase: 'Fix', model: 'sonnet' }) : null
return { built, crit, fix }