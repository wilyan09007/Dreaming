export const meta = {
  name: 'dreaming-finish-s25-s26-s16b',
  description: 'Finish the three unfinished smokes from their existing scripts and caches: S25 exotic families (4090), S26 similar-task pool transfer (H100/A100), S16b draft_note pruned re-ask (A5000); no critic cycle',
  phases: [{ title: 'Build' }],
}
const ROOT = 'C:\\Users\\<user>\\Desktop\\personalprojects\\Dreaming'
const SCRIPTS = 'C:\\Users\\<user>\\.claude\\projects\\C--Users-<user>-Desktop-personalprojects-Dreaming\\4bf0a8c4-ab7a-4fd7-a483-c5f84a5e6e6e\\workflows\\scripts\\'
const COMMON = `
ENVIRONMENT AND RULES:
- Project root: ${ROOT}. Laptop stays idle: local python ("${ROOT}\\.venv\\Scripts\\python.exe") only for writing code, syntax checks and ONE --tiny check under 60 s. All compute through ${ROOT}\\smokes\\pod.py (read it): python smokes/pod.py --tag <id> --bundle smokes/common.py smokes/<folder> [more] --cmd "python smokes/<folder>/run.py" [--pip "..."] --max-min <n> --gpu "<gpu>". pod.py tries COMMUNITY first and falls back to any cloud. The user authorised faster GPUs to finish sooner: use "NVIDIA H100 PCIe" or "NVIDIA A100 80GB PCIe" for anything that trains models, "NVIDIA GeForce RTX 4090" for mixed CPU/GPU work, "NVIDIA RTX A5000" for API-bound work. Pod boot is ~10 min regardless of GPU, so launch ONCE after the --tiny check; at most 2 launches. Assert torch.cuda.is_available() at start. Write results.json incrementally; most informative arms first.
- YOUR FOLDER ALREADY CONTAINS WORK from an interrupted attempt (run.py, caches, downloads, tiny_results.json, maybe salvaged_* files): read it first and finish from it rather than starting over. Keys in ${ROOT}\\.env (read with python/re; never print or write them). OpenRouter via openai client (base_url https://openrouter.ai/api/v1), pin providers, cache raw responses keyed by request hash, track usage.cost, append "<smoke>,<usd>,<note>" to ${ROOT}\\smokes\\COSTS.md.
- Read ${ROOT}\\smokes\\FINDINGS.md round-3 section first (S17, S22, S23, S24 verdicts) so your interpretation compares against them. Do NOT modify common.py or pod.py. Never use mcp__claude-in-chrome tools. Hold out by group, never by row.
- RESULTS.md (Write tool) must START with "## In plain words" (4-8 short sentences a non-expert can follow), then Setup, Results tables, Interpretation (compare with the round-3 verdicts), Caveats, Time and cost. Report failures as failures. Final text return: 10-15 lines of key numbers and blockers.
`
const fromFile = (file, key) => `Your full spec is the SMOKES entry (or SPEC constant) for '${key}' in ${SCRIPTS}${file}; open it with the Read tool and follow it. Folder: ${ROOT}\\smokes\\${key}.`
const SMOKES = [
  { key: 's25_exotic', gpu: 'NVIDIA GeForce RTX 4090', spec: fromFile('dreaming-s25-exotic-wf_e57f4916-718.js', 's25_exotic') + ' Use --max-min 55. If the SAE arm needs a gated model, skip it and say so.' },
  { key: 's26_pool_transfer', gpu: 'NVIDIA H100 PCIe', spec: fromFile('dreaming-s26-pool-and-s16b-wf_c5c40d10-841.js', 's26_pool_transfer') + ' Use "NVIDIA H100 PCIe" (fall back to "NVIDIA A100 80GB PCIe", then the 4090 if neither is available: pod.py prints the launch error), --max-min 60. Train the encoder and decoder pretraining arms first, then the hypernetwork, so the transfer question is answered even if time runs out.' },
  { key: 's16_pruner_note', gpu: 'NVIDIA RTX A5000', spec: fromFile('dreaming-s26-pool-and-s16b-wf_c5c40d10-841.js', 's16_pruner_note') + ' A salvaged_results.json from the interrupted pod run may be in the folder: reuse any cached judge calls it reflects. --max-min 45.' },
]
phase('Build')
const out = await Promise.all(SMOKES.map(s => agent(`${COMMON}\nGPU for this smoke: "${s.gpu}".\n${s.spec}`, { label: `finish:${s.key}`, phase: 'Build', model: 'sonnet' })))
return SMOKES.map((s, i) => ({ smoke: s.key, summary: out[i] }))