export const meta = {
  name: 'dreaming-research-sweep',
  description: 'Fan out web research on datasets, benchmarks, reusable components, pricing and precedents for each Dreaming mission; critic finds gaps; fill gaps',
  phases: [
    { title: 'Sweep', detail: '12 topic researchers, each writes smokes/research/<topic>.md' },
    { title: 'Critique', detail: 'one critic lists gaps and dubious claims' },
    { title: 'Fill', detail: 'gap-fill researchers' },
  ],
}

const CONTEXT = `
PROJECT CONTEXT (read fully; it is all the context you get):
"Dreaming" is a runtime for LLM agents: it logs every call an agent makes (frontier-model steps, tool/API calls, human reviews), trains a small predictor per call site, shadows it against reality, "certifies" it (finds a confidence cutoff where the error rate stays under an operator-set budget), then answers from the predictor and keeps the real call as fallback/audit sample. Hosted predictors must be byte-for-byte replayable. Demo domain: a bookkeeping agent with typed steps (categorize_transaction with ~12 labels, vendor_lookup tool, match_invoice, flag_anomaly yes/no, draft_note free text). Missions: (1) tier/thinking router: does a cheaper model config give the same answer; (2) students: small models fine-tuned per step / shared across steps / per-step LoRA adapters / a hypernetwork that writes student weights from a step's schema; (3) tool return + latency predictor and speculative next-step launch; (4) human-review deferral (predict which outputs a reviewer will change); (5) context pruner (which prompt spans the output depends on, labels by ablation); (6) connectors: OpenAI-compatible proxy + tool decorator + MCP proxy; (7) full rollout: chain predictors to imagine the whole episode. Also: determinism/hashing for replay, and a connector to Fetch.ai ASI:One/Agentverse.
Constraints: total budget under $100 (we have ~$20 OpenRouter credit, ~$20 RunPod, local CPU only, no AWS GPU quota). We are preparing feasibility smokes now, so we care about: what public datasets/benchmarks exist and can be downloaded WITHOUT a login (HuggingFace Hub, GitHub, direct URLs; Kaggle requires login so flag it), which existing open-source components we can reuse instead of building, concrete prices and throughput numbers, and anything that would significantly change plans or make something impossible.
RULES: Use WebSearch and WebFetch for research (do NOT use any mcp__claude-in-chrome tools, do not open a browser). Prefer primary sources (papers, repos, HF dataset cards, pricing pages). Give URLs for every claim. Say "unverified" when you could not confirm something. Today is 2026-10-17; prefer current information and note dates. Do not pad; precise facts over prose.
`

const TOPICS = [
  { key: 'datasets-typed-steps', q: `Public datasets usable as typed-step traces for distilling classification/extraction/decision students. Cover: Banking77, CLINC150 (clinc_oos), MASSIVE, TREC, and especially bank-transaction categorization datasets with raw merchant descriptor strings (search HuggingFace Hub and GitHub; Kaggle only as a fallback, flag login). Also multi-step agent trace datasets with tool calls AND tool returns: tau-bench / tau2-bench (its mock DBs and tool code), AppWorld, ToolBench, BFCL, API-Bank, Nemotron agentic/tool-use traces, AgentInstruct, ToolACE. For each: exact HF id or URL, size, label count, license, whether downloadable without login, and how it maps to one of our call-site types.` },
  { key: 'routing-benchmarks', q: `Tier/thinking routing. (a) RouterBench (Martian; HF withmartian/routerbench): what models/benchmarks it contains, whether per-sample responses and correctness for many models are included so we can compute cheap-vs-expensive agreement offline, download size. (b) RouteLLM: datasets, code, how routers are trained. (c) FrugalGPT cascades. (d) DART or similar draft-agreement / thinking-budget routers; papers measuring how often thinking-on vs thinking-off change the answer. (e) OpenRouter API: the "reasoning" parameter (effort / max_tokens / enabled:false) and which models honour it; current per-million-token prices for candidate pairs (e.g. gpt-4.1 vs gpt-4.1-mini/nano, claude sonnet vs haiku, gemini pro vs flash/flash-lite, deepseek, qwen3 thinking toggle, gpt-5 family and reasoning_effort). Give exact prices with dates.` },
  { key: 'student-models', q: `Small models and recipes for per-call-site students that score a fixed label set in one forward pass. Candidates: Qwen2.5-0.5B/1.5B, Qwen3-0.6B, SmolLM2-135M/360M, SmolLM3, ModernBERT-base, DeBERTa-v3-small/xsmall, embedding models (bge-small, e5-small, gte-small, all-MiniLM-L6-v2, nomic). Find: (1) reported accuracy on Banking77/CLINC150 for fine-tuned small encoders vs frozen embeddings + linear vs few-shot (SetFit numbers!), and how many rows per class were needed; (2) LoRA fine-tuning throughput/time for a 0.5B model on RTX 4090 / A100 (tokens per second, minutes per 10k short rows); (3) libraries to reuse: SetFit, PEFT, Unsloth, trl, sentence-transformers; (4) precedents: OpenPipe distillation, NVIDIA LLM-to-SLM conversion recipe (NeMo / "Data Flywheel"), BetterTogether (DSPy), and any report on how many logged teacher rows are needed before a student matches the teacher.` },
  { key: 'hypernetwork', q: `Hypernetworks that write a student's weights from a task description. Cover: Sakana AI Text-to-LoRA (paper, GitHub code, pretrained weights available?, training cost, results on unseen tasks), Drag-and-Drop LLMs (DnD), HyperCLIP, "write a classifier" / zero-shot classifier generation from text (Elhoseiny et al, Lei Ba et al), HyperNetworks for few-shot, "label-description embeddings as classifier weights" baselines, meta-learning of linear heads over frozen embeddings (e.g. MAML-style, prototypical nets). For each: is the code open, can we reuse a pretrained generator, what GPU/time did training take, what the gain over the obvious baseline was on unseen tasks. Conclude with the simplest version we could train on CPU in under an hour over frozen sentence embeddings, and the honest baselines it must beat (label-embedding zero-shot, k-shot prototypes, logistic regression).` },
  { key: 'determinism', q: `Byte-identical replay of small hosted models. Cover: Thinking Machines "Defeating Nondeterminism in LLM Inference" (batch-invariant kernels, Sept 2025) and follow-ups; vLLM / SGLang determinism options as of 2026; PyTorch torch.use_deterministic_algorithms and CUBLAS_WORKSPACE_CONFIG; CPU inference determinism (PyTorch CPU, ONNX Runtime, llama.cpp) across thread counts; whether batch-size-1 fixed-shape inference on one GPU is deterministic run-to-run; hashing model files (safetensors sha256) and environment fingerprints. Deliver: concrete recipe to get 1000 identical outputs under concurrent load, expected throughput penalty, and pitfalls (different GPU SKU, driver, thread count, fused kernels, bf16 accumulation).` },
  { key: 'compute-pricing', q: `Current (October 2026) GPU rental prices and APIs. RunPod: community vs secure cloud hourly prices for RTX 4090, RTX A5000, A40, L4, A100 80GB, H100; the GraphQL/REST API to create an on-demand pod with a custom docker image and startup command (fields like podFindAndDeployOnDemand, dockerArgs, env, volumeInGb, templateId) and to terminate it; whether a pod can run a one-shot script with no SSH (e.g. docker command override) and the runpod/pytorch image tags; RunPod serverless pricing per second. Alternatives: Vast.ai, Lambda, Modal (free credits?), Google Colab, Hugging Face Jobs/Spaces. Also: typical minutes and cost to LoRA-fine-tune a 0.5B model on 10k short rows, and hours to fine-tune a 110M encoder on 10k rows on CPU. Give URLs and dates for every price.` },
  { key: 'context-pruning', q: `Learned context pruning for LLM calls. Cover: SWE-Pruner, LaMR, LLMLingua / LLMLingua-2 (reusable HF models: microsoft/llmlingua-2-*), Selective Context, RECOMP, "prompt compression" surveys 2025-2026, token/span attribution for prompts, and any work that labels span importance by ablation (remove span, re-run, did output change). For each: open code/weights, reported compression ratio vs agreement, cost of building ablation labels. Benchmarks used (LongBench, MeetingBank, ZeroSCROLLS, etc). Deliver: what we can reuse directly for a span-level scorer and the cheapest honest way to build ablation labels for ~500 prompts with ~8 spans each.` },
  { key: 'certification-deferral', q: `Certified error bounds and learning to defer. Cover: selective classification with guaranteed risk (Geifman & El-Yaniv 2017 SGR algorithm, exact bound and its code), conformal risk control / conformal prediction libraries (MAPIE, TorchCP, crepes), calibration libraries (netcal, sklearn CalibratedClassifierCV), learning-to-defer (Mozannar & Sontag 2020, and later), confidence meta-models / "trust score". Also: datasets of human corrections of model outputs for text classification or extraction (e.g. annotator disagreement datasets, label-noise datasets with clean+noisy labels like Banking77 noisy variants, CIFAR-N style for text), and precedents for audit sampling to keep a certificate current (sequential testing, SPRT, control charts). Deliver: the exact bound formula we should implement for "above cutoff c, error <= budget with confidence 1-delta given n calibration rows", with a worked example n=500, budget=5%.` },
  { key: 'connectors', q: `OpenAI-compatible proxies and tool-call interception. Cover: LiteLLM proxy, Helicone, Portkey, OpenRouter itself, Langfuse/OpenLLMetry: which log every call, support custom headers passed through, and could be reused as our logging layer; how LangChain/LangGraph ChatOpenAI, the OpenAI Agents SDK, CrewAI, Pydantic AI, Vercel AI SDK set base_url and default headers (exact parameter names) so a call-site header can be attached per node; MCP proxies (sparfenyuk/mcp-proxy, FastMCP proxy/mount, MCP middleware) to intercept tool calls and returns; OpenAI Responses API vs Chat Completions compatibility issues when proxying. Deliver: minimal code sketch (python) of a proxy that logs and can answer from a local predictor, and the per-framework one-line change.` },
  { key: 'asi-one', q: `Fetch.ai ASI:One and Agentverse integration, as of 2026. Cover: uAgents python framework, Agentverse registration (mailbox vs hosted vs local agent), the Chat Protocol (uagents_core chat protocol) needed for ASI:One discoverability, required API keys/accounts (Agentverse API key, ASI:One API key) and whether creation requires a human sign-in, minimal working example code for an agent that answers chat messages, the "ASI Alliance"/Fetch hackathon track rules and judging criteria if findable, how to call an LLM from within the agent (ASI:One LLM API, OpenAI-compatible?), and known pitfalls (ports, almanac registration fees, funding on testnet). Estimate integration time for a Python runtime.` },
  { key: 'tool-prediction-speculation', q: `Predicting tool/API returns and speculative execution for agents. Cover: Speculative Actions (2025), PASTE, AOSpec, "world models for agents" (DreamGym, text world models, SimuAgent, WebDreamer, "Learning to simulate tools"), tool-return simulators used for training agents (e.g. ToolBench's simulated APIs, Nemotron tool simulators, tau-bench mock DBs), latency prediction (learned RTT estimators), and any public datasets of real tool-call logs with returns and latencies (e.g. API traces, Hugging Face datasets of function-calling with responses). Deliver: which of our three tool cases (return function of inputs; return depends on hidden state; call changes the world) has precedent, measured speculation hit rates in the literature, and the cheapest honest way to build a mock-but-realistic bookkeeping tool environment with latency and error injection.` },
  { key: 'bookkeeping-data', q: `Data for a bookkeeping demo agent: raw bank/credit-card transaction descriptor strings with merchant and category labels (HuggingFace Hub datasets, GitHub repos, Plaid/Teller sandbox data, Kaggle datasets — flag login), merchant-name normalization datasets, invoice datasets (e.g. for invoice-to-transaction matching: RVL-CDIP invoices, FUNDS, SROIE, CORD, Invoice2Data), anomaly/fraud transaction datasets (PaySim, IBM TabFormer credit card, Kaggle creditcard), and synthetic data generators (Faker providers, Plaid sandbox, open-source transaction generators). For each: URL, size, fields, license, no-login download? Deliver: the best no-login option for (a) categorize_transaction with real merchant strings, (b) vendor_lookup ground truth, (c) invoices, (d) anomaly labels; and a recommendation whether to synthesize instead with a frontier model and roughly what it costs.` },
]

const RESEARCH_SCHEMA = {
  type: 'object',
  properties: {
    file: { type: 'string' },
    key_facts: { type: 'array', items: { type: 'string' } },
    reusable: { type: 'array', items: { type: 'object', properties: { name: { type: 'string' }, url: { type: 'string' }, why: { type: 'string' } }, required: ['name', 'url', 'why'] } },
    blockers_or_plan_changes: { type: 'array', items: { type: 'string' } },
    cost_time_numbers: { type: 'array', items: { type: 'string' } },
    unverified: { type: 'array', items: { type: 'string' } },
  },
  required: ['file', 'key_facts', 'reusable', 'blockers_or_plan_changes', 'cost_time_numbers', 'unverified'],
}

phase('Sweep')
const sweep = await parallel(TOPICS.map(t => () => agent(
  `${CONTEXT}
YOUR TOPIC: ${t.key}
QUESTION: ${t.q}

Do 6-15 web searches/fetches as needed. Then write a markdown file at C:\\Users\\<user>\\Desktop\\personalprojects\\Dreaming\\smokes\\research\\${t.key}.md (use the Write tool; absolute Windows path) with sections: Summary (5-10 bullets), Findings (with URLs), Reusable components (table: name | url | license | why), Numbers (prices, sizes, throughput, with date and source), Blockers / plan changes, Unverified. Keep it under ~250 lines. Then return the structured summary.`,
  { label: `research:${t.key}`, phase: 'Sweep', model: 'sonnet', schema: RESEARCH_SCHEMA }
)))
const done = sweep.filter(Boolean)
log(`sweep: ${done.length}/${TOPICS.length} topics written`)

phase('Critique')
const CRITIC_SCHEMA = {
  type: 'object',
  properties: {
    gaps: { type: 'array', items: { type: 'object', properties: { topic: { type: 'string' }, question: { type: 'string' }, why_it_matters: { type: 'string' } }, required: ['topic', 'question', 'why_it_matters'] } },
    dubious_claims: { type: 'array', items: { type: 'object', properties: { file: { type: 'string' }, claim: { type: 'string' }, why: { type: 'string' } }, required: ['file', 'claim', 'why'] } },
    top_plan_changes: { type: 'array', items: { type: 'string' } },
  },
  required: ['gaps', 'dubious_claims', 'top_plan_changes'],
}
const critique = await agent(
  `${CONTEXT}
You are the completeness critic. Read every file in C:\\Users\\<user>\\Desktop\\personalprojects\\Dreaming\\smokes\\research\\ (use Glob + Read) and the mission list above. Identify: (1) GAPS — questions the feasibility study must answer that no file answers (e.g. a mission with no dataset, a price missing, an API detail missing, a precedent not checked); return at most 8, ordered by impact on the plan; (2) DUBIOUS CLAIMS — specific claims that look wrong, outdated, or unsupported by a URL, worth re-verifying (at most 8); (3) TOP PLAN CHANGES — the 5 most important things the research implies about what to build or not build. Be concrete; each gap must be a single researchable question.`,
  { label: 'critic', phase: 'Critique', model: 'opus', schema: CRITIC_SCHEMA }
)
log(`critic: ${critique ? critique.gaps.length : 0} gaps, ${critique ? critique.dubious_claims.length : 0} dubious claims`)

phase('Fill')
const fills = critique ? [
  ...critique.gaps.slice(0, 6).map((g, i) => ({ kind: 'gap', i, text: `GAP in topic "${g.topic}": ${g.question}\nWhy it matters: ${g.why_it_matters}` })),
  ...critique.dubious_claims.slice(0, 4).map((d, i) => ({ kind: 'verify', i, text: `VERIFY claim in ${d.file}: "${d.claim}"\nConcern: ${d.why}` })),
] : []
const filled = await parallel(fills.map(f => () => agent(
  `${CONTEXT}
TASK (${f.kind}): ${f.text}

Research this one item with WebSearch/WebFetch (3-8 calls). Then APPEND a section to the most relevant existing file in C:\\Users\\<user>\\Desktop\\personalprojects\\Dreaming\\smokes\\research\\ (Read it first, then use Edit to append a "## Addendum: <title>" section at the end; if no file fits, Write a new file smokes/research/addendum-${f.kind}-${f.i}.md). Return 3-8 bullet key facts with URLs, and say explicitly whether the original claim was confirmed, corrected, or unresolved.`,
  { label: `fill:${f.kind}-${f.i}`, phase: 'Fill', model: 'sonnet' }
)))

return {
  sweep: done,
  critique,
  fills: filled.filter(Boolean),
}