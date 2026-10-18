# Student models for per-call-site predictors (research, 2026-10-17)

Scope: small models that score a fixed label set in one forward pass (Qwen2.5-0.5B/Qwen3-0.6B/SmolLM, ModernBERT, DeBERTa-v3, embedding models), with recipes, numbers, libraries, precedents.
Conventions: "verified" = I read the number in the cited source this session. "derived" = my arithmetic. Anything else is in Unverified.

## Summary

- Cheapest viable student is NOT a decoder LLM. Frozen 33M-param embedding model + logistic regression gets 81.6-85.7% on Banking77 (77 classes) at 8 rows/class, trains in seconds on CPU. Fine-tuning gets to 88% at 10/class and ~94% at full data (~130 rows/class).
- Rows needed: on Banking77, ~10-20 rows/class (770-1,540 rows) lands within 3-6 pts of full-data accuracy (SetFit + all-mpnet-base-v2: 83.5 @5, 88.0 @10, 90.6 @15, 91.2 @20, 94.0 full). Plan the first student at roughly 10-20 clean rows per label per call site (about 120-240 rows for a 12-label step); this is an extrapolation from a 77-class task, so verify on the bookkeeping labels.
- Teacher-matching precedent: DistilRoBERTa student on ~5.5k GPT-5 labels got F1 0.821 vs teacher 0.846 on one concept (ClimatePolicyRadar, Jan 2026). NVIDIA's "10k-100k examples" is a rule of thumb with no experiment behind it, so use it as an upper bound.
- LoRA throughput for a 0.5B model: no public measured tokens/s found (searched twice). Derived estimate: ~10-30k tokens/s on a 4090, i.e. minutes per 10k short rows, cost cents at $0.34/hr. Measure it in the smoke; do not buy Unsloth-specific speedups for this size.
- RunPod (official page, updated 2026-10-08): RTX 4090 $0.34/hr community, A100 80GB PCIe $1.19/hr community. The $20 RunPod credit is roughly 58 4090-hours. Student training is not budget-limited; teacher labeling and eval are.
- LoRAX (the multi-LoRA server) was archived on 2026-10-12. For per-step adapters use vLLM multi-LoRA, which does support sequence-classification adapters (`score`/`classifier` head, `--max-lora-cls-labels`).
- Hypernetwork mission: the one public precedent (Sakana Text-to-LoRA) needs hundreds of training tasks (479 per secondary source); a bookkeeping agent has ~5 step types. See also hypernetwork.md in this folder. Treat as a research stretch, not a smoke.
- Replay determinism: forward passes are bitwise run-to-run deterministic at fixed batch shape; nondeterminism comes from batch-size-dependent kernels (Thinking Machines). Serve students at fixed batch/padding, or hash the argmax label plus a quantized score, not raw logits.

## Findings

### 1. Accuracy vs data (Banking77 = 77 intents, 10,003 train / 3,080 test; CLINC150 = 150 intents)

Frozen embeddings + logistic regression (MTEB Banking77Classification, test accuracy from model-card metadata):

| model | params/dim | Banking77 acc | source |
|---|---|---|---|
| bge-small-en-v1.5 | 384-d | 85.74 | https://huggingface.co/BAAI/bge-small-en-v1.5 (README model-index) |
| gte-small | 384-d | 84.08 | https://huggingface.co/thenlper/gte-small |
| e5-small-v2 | 384-d | 81.64 | https://huggingface.co/intfloat/e5-small-v2 |
| all-MiniLM-L6-v2 | 384-d | not in card (unverified) | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |

- MTEB default protocol (source code, main branch): `samples_per_label = 8`, `n_experiments = 10`, LogisticRegression `max_iter=100`. So these are 8-rows-per-class numbers: https://raw.githubusercontent.com/embeddings-benchmark/mteb/main/mteb/abstasks/classification.py . I did not check whether the Banking77 task overrides the defaults, nor that each model-index was produced with this protocol.

SetFit / fine-tuning on Banking77 (Loukas et al., FinNLP 2023, Table 4; micro-F1 = accuracy): https://arxiv.org/abs/2308.14634
- SetFit with all-mpnet-base-v2 (110M): 1-shot 57.4, 3-shot 76.3, 5-shot 83.5, 10-shot 88.0, 15-shot 90.6, 20-shot 91.2. Standard full-data fine-tune of the same model: 94.0 (paraphrase-mpnet-base-v2: 93.0; 10-shot 85.7).
- GPT-4 3-shot with human-picked examples 83.1 (74.2 random examples); GPT-3.5 1-shot 75.2. So a frontier teacher is itself only ~75-83% few-shot here; a student trained on ~1k gold rows (88%) can beat it. Teacher labels are not gold.
- Prior SOTA cited there: BERT full-data 93.66, ConveRT+USE 10-shot 85.19, Mehri & Eric 10-shot 85.95 / full 93.83. Label noise exists: a cleaned-label study reports 88.2 -> 92.4 acc for the same classifiers (cited in the same paper), so ~94% is near the ceiling.
- Older SetFit encoder (paraphrase-distilroberta, 82M) at 8 shots on Banking77: 77.0 (80.9 with domain-adaptive pretraining) from the AdaSent paper, Table 3: https://arxiv.org/abs/2311.00408 . Newer embedding models with a frozen LR head (bge-small 85.7) beat this older SetFit setup. Different protocols/seeds, so only directionally comparable.
- Fine-tuned RoBERTa-base + classifier, Banking77: 74.0 @5-shot, 84.3 @10-shot; CLINC150: 88.0 @5, 91.6 @10. Contrastive pretrain + finetune (CPFT): Banking77 80.9 / 87.2, CLINC150 92.3 / 94.2. https://arxiv.org/abs/2109.06349 (table row alignment reconstructed from pdftotext; the RoBERTa row matches a second source's quote).
- Full-data small encoders: ModernBERT-base fine-tune 93.5 acc (community model card, https://huggingface.co/DarioFrK/sift-banking77-modernbert); Phil Schmid's recipe reports F1 0.93 ModernBERT vs 0.90 BERT (article text/code are inconsistent about model and dataset: https://www.philschmid.de/fine-tune-modern-bert-in-2025).
- CLINC150 full-data BERT in-scope accuracy: 96.9 (Larson et al. 2019, Table "Full"): https://arxiv.org/abs/1909.02027 . SetFit-style 8-shot on CLINC150 ~85.9 for a 22M model per a search snippet only (unverified).
- SetFit original paper (N=8 and N=64 per class, small datasets, none are Banking77/CLINC): SetFit-MPNet 110M avg ~62.3 @N=8, comparable to T-Few 3B (63.4), 19x less inference compute; MiniLM-L3 (15M) 60.3, 123x. Training ~30 s on a V100 (~$0.025) per split for binary tasks: https://arxiv.org/abs/2209.11055 (read via ar5iv), https://huggingface.co/blog/setfit .

Takeaway for the plan (extrapolation, not measured): at ~12 labels, ~10-20 clean rows per label should reach the high-80s regime and ~130 per label the low-90s, if labels are consistent and classes are separable. Whether a decoder 0.5B beats an encoder here: no source found (unverified).

### 2. LoRA throughput / time for 0.5B

- No public measured tokens/s found for Qwen2.5-0.5B or Qwen3-0.6B LoRA on 4090 or A100 (two searches; results were inference-only estimates). Do not cite willitrunai-style numbers; they are decode estimates.
- Verified anchors only: (a) ModernBERT-base (149M) full fine-tune on Banking77, 10,003 rows x 5 epochs, bs 32, padded to 512, bf16, NVIDIA L4: 321 s (~156 rows/s, derived) - https://www.philschmid.de/fine-tune-modern-bert-in-2025 (inconsistencies noted above). (b) Unsloth vs HF: 2.74x on TinyLlama-1.1B/Alpaca, A100-40GB; 3.87x on a T4; 1.9x for 7B models - https://huggingface.co/blog/unsloth-trl (2023-12-era post; relative only).
- Derived estimate (not measured): LoRA fwd+bwd on a frozen 0.36B-non-embedding model ~ 4 x 0.36e9 ~ 1.4 GFLOP/token. At 20-30% utilisation of a 4090/A100 this is roughly 10-30k tokens/s, so 10k rows x ~64 tokens x 3 epochs = ~2M tokens = ~1-3 min; with ~300-token bookkeeping prompts, ~5-15 min. Cost at $0.34/hr: under $0.10. HF Trainer overhead on tiny models can halve this.
- Trainable params (derived, Qwen2.5-0.5B: hidden 896, 24 layers, MLP 4864, 2 KV heads): r=8 on q,v = ~0.54M; r=16 on all linear layers = ~8.8M (~18 MB bf16). Tiny adapters make per-step LoRA storage trivial.
- LoRA recipe guidance (Thinking Machines, 2025): apply LoRA to all layers incl. MLP (attention-only underperforms), LR ~10x the full-FT optimum (fitted 9.8x; ~15x for runs under ~100 steps), LoRA is less tolerant of large batches, capacity limits matter when data is large. The post does not cover classification: https://thinkingmachines.ai/blog/lora/ .
- PEFT supports `task_type=SEQ_CLS` for decoder LLMs via `AutoModelForSequenceClassification`; head (`score`) is trained fully via `modules_to_save` (code pattern from model memory plus the PEFT page redirect; verify before relying on it): https://huggingface.co/docs/peft/task_guides/seq-classification .
- Local machine is CPU-only Windows: Unsloth needs an NVIDIA GPU (PyPI classifiers "Environment :: GPU :: NVIDIA CUDA"), so train on RunPod. Encoders (MiniLM/bge-small, SetFit) are fine to train on CPU (SetFit blog: "a few minutes on CPU").

### 3. Libraries
See the Reusable components table. Key facts: SetFit 1.2.0 released 2026-09-04 (actively maintained, Apache-2.0); peft 0.21.2 (2026-10-01); sentence-transformers 6.1.0; unsloth 2026.10.6 (Apache-2.0 core, optional Studio UI is AGPL-3.0); trl 1.16.0.

### 4. Precedents and teacher-row counts

- NVIDIA "Small Language Models are the Future of Agentic AI" (arXiv 2506.02153, Sec. 6): 6-step conversion algorithm S1 log non-human-facing calls, S2 curate/filter, S3 cluster tasks, S4 pick SLM per task, S5 fine-tune (LoRA/QLoRA, full FT, or distillation from LLM), S6 iterate/retrain, including the router. Only data figure: "10k-100k examples being sufficient for fine-tuning of small models as a rule of thumb". Only cost figure: 7B SLM 10-30x cheaper to serve than 70-175B LLM (Sec. 3.2). https://arxiv.org/html/2506.02153v1 . This is the same architecture as Dreaming (log -> cluster by call site -> train student -> router); it is an opinion paper, not an experiment.
- NVIDIA Data Flywheel Blueprint (NeMo): collects OpenAI-format traffic, dedups, tags by `workload_id`, LoRA-SFT of Llama-3.2-1B, LLM-judge eval. Claim: 1B reaches 98% of Llama-3.3-70B tool-calling accuracy; 1 GPU vs 2. Config defaults `min_total_records: 50`, `eval_size: 20` are settings, not guidance. Dataset size and fine-tune time are not stated in the post; requires multiple A100/H100-class GPUs, so reuse the idea, not the stack. https://developer.nvidia.com/blog/build-efficient-ai-agents-through-model-distillation-with-nvidias-data-flywheel-blueprint . The Llama 3.2 1B on xLAM (~60k tool-calling rows) tutorial figure is from a search snippet only (unverified).
- Real distillation data point: ClimatePolicyRadar, ~5.5k GPT-5 (gpt-5-2025-08-07) labels -> DistilRoBERTa; F1 0.821 (0.828 with tuned threshold) vs GPT-5 0.846, validation support 446; no cost/latency or learning curve given. https://labs.climatepolicyradar.org/tech-blog/2026/01/28/our-first-distilled-classifier.html
- Choosing which rows to send to the teacher: M-RARU active learning claims up to 80% fewer samples than random for DistilBERT/SVM/etc. students (abstract only; absolute counts not given): https://arxiv.org/abs/2511.11574 . Relevant to Dreaming: only the uncertain/audit rows need real calls.
- OpenPipe: acquired by CoreWeave (announced 2025-09-03; ART, its open-source RL agent trainer, is the product now emphasised): https://techcrunch.com/2025/09/03/coreweave-acquires-agent-training-startup-openpipe/ . A "$23,000 (GPT-4) -> $19 for 2M recipes" claim is secondhand; I could not reach docs.openpipe.ai (DNS failure) and found no OpenPipe statement of rows needed.
- BetterTogether (DSPy): alternates prompt optimisation and weight fine-tuning of the same LM across program modules; up to 60% / 6% better than weights-only / prompt-only averaged over mistral-7b, llama-2-7b, llama-3-8b on multi-hop QA, math, classification. Student is a 7-8B generative LM, self-bootstrapped traces, no teacher. Exposed as `dspy.BetterTogether`. Good template for "per-module trace -> finetune", not for tiny label-scoring heads. https://arxiv.org/abs/2407.10930 , https://dspy.ai/api/optimizers/BetterTogether/
- Distilling Step-by-Step (rationales as extra supervision): 770M T5 beat few-shot PaLM-540B using 80% of the data on one benchmark (which one not confirmed). Only useful if you also keep the teacher's rationale text. https://research.google/blog/distilling-step-by-step-outperforming-larger-language-models-with-less-training-data-and-smaller-model-sizes/
- Hypernetwork-written adapters: Sakana Text-to-LoRA (arXiv 2506.06105, ICML 2025). Secondary sources: trained on 479 Super-NaturalInstructions tasks; variants 55M/34M/5M params; SFT-trained variant 66.3 vs 61.8 reconstruction on average; zero-shot on unseen tasks below task-specific adapters. Abstract itself mentions 9 pre-trained LoRAs. https://arxiv.org/abs/2506.06105 , https://sakana.ai/doc-to-lora/ . Cross-reference hypernetwork.md for the dedicated note.
- Serving many adapters: vLLM serves multiple LoRAs concurrently (`max_loras`, `max_lora_rank`), supports seq-classification adapters whose single-layer head is `score` or `classifier` saved via `modules_to_save`, `--max-lora-cls-labels` for differing label counts, dynamic load/unload with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` (docs: not for production unless isolated). https://docs.vllm.ai/en/latest/features/lora.html . LoRAX: Apache-2.0, "thousands of fine-tuned models on a single GPU" but archived 2026-10-12 and read-only: https://github.com/predibase/lorax
- Determinism: forward pass is run-to-run deterministic at fixed batch size; non-invariance across batch sizes comes from kernel reduction order. Batch-invariant kernels cost ~20% on matmul (Qwen3-8B, 1,000 sequences: 26 s default vs 55 s vs 42 s optimised). Qwen3-235B at temp 0 gave 80 unique outputs over 1,000 runs. https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ . For a hosted classifier, store label + int-quantized margin, and pin batch size/padding.

## Reusable components

| name | url | license | why |
|---|---|---|---|
| SetFit 1.2.0 (2026-09-04) | https://github.com/huggingface/setfit | Apache-2.0 | few-shot contrastive fine-tune of a sentence encoder + LR or differentiable head; 8-20 rows/class; CPU-trainable |
| sentence-transformers 6.1.0 | https://pypi.org/project/sentence-transformers/ | Apache-2.0 | encoder loading, embedding, ONNX/OpenVINO backends; frozen-embedding + LR baseline is ~10 lines |
| MTEB | https://github.com/embeddings-benchmark/mteb | Apache-2.0 (unverified) | `Banking77Classification` / CLINC tasks give a ready 8-shot harness for choosing the embedder |
| PEFT 0.21.2 (2026-10-01) | https://github.com/huggingface/peft | Apache-2.0 | LoRA with `SEQ_CLS` head on Qwen/SmolLM; adapters directly loadable by vLLM |
| TRL 1.16.0 | https://pypi.org/project/trl/ | Apache-2.0 | SFTTrainer if scoring label tokens via LM head instead of a classification head |
| Unsloth 2026.10.6 | https://pypi.org/project/unsloth/ | Apache-2.0 core; Studio UI AGPL-3.0 | 1.9-3.9x faster LoRA on NVIDIA; marginal value at 0.5B and needs a GPU box; seq-cls support unverified |
| Model2Vec (potion-base-8M, 7.5M params; 32M, 32.3M) | https://github.com/MinishLab/model2vec | MIT | static embeddings, "up to 500x faster on CPU", ~8-30 MB, `StaticModelForClassification`; trivially deterministic ultra-cheap student |
| vLLM multi-LoRA | https://docs.vllm.ai/en/latest/features/lora.html | Apache-2.0 (known, not re-checked) | one base model + many per-call-site adapters incl. classification heads |
| DSPy BetterTogether / BootstrapFinetune | https://dspy.ai/api/optimizers/BetterTogether/ | MIT (unverified) | reference design for per-module trace -> finetune loops |
| NVIDIA Data Flywheel Blueprint | https://github.com/NVIDIA-AI-Blueprints/data-flywheel (URL unverified) | unverified | design reference only; needs 2-6 A100/H100 class GPUs |
| bge-small-en-v1.5 / gte-small / e5-small-v2 | links in Findings 1 | MIT-class (gte/e5 license field unverified; bge verified only via README) | best frozen-embedding baselines at 33M params, 384-d |
| ModernBERT-base (149M, 8,192 ctx, Apache-2.0) | https://huggingface.co/answerdotai/ModernBERT-base | Apache-2.0 | long-context encoder student (context pruner, long review text) |
| DeBERTa-v3-small (44M backbone + 98M embeddings) | https://huggingface.co/microsoft/deberta-v3-small | MIT | strong small encoder for full fine-tune |
| Qwen2.5-0.5B (0.49B, 0.36B non-emb, 24 layers) | https://huggingface.co/Qwen/Qwen2.5-0.5B | Apache-2.0 | decoder student candidate |
| Qwen3-0.6B (0.6B, 0.44B non-emb, 28 layers, 32k ctx) | https://huggingface.co/Qwen/Qwen3-0.6B | Apache-2.0 | decoder student candidate |
| SmolLM2-135M/360M/1.7B | https://huggingface.co/HuggingFaceTB/SmolLM2-360M | Apache-2.0 | smallest decoder students (4T pretraining tokens for 360M) |
| SmolLM3-3B | https://huggingface.co/HuggingFaceTB/SmolLM3-3B | Apache-2.0 | too big for CPU-only serving; only if a 3B step model is needed |
| nomic-embed-text-v1.5 (~137M, 768-d, Matryoshka 512/256/128/64) | https://huggingface.co/nomic-ai/nomic-embed-text-v1.5 | Apache-2.0 | needs `trust_remote_code` today; 64-d truncation costs 6 MTEB points |

## Numbers

| item | value | date | source |
|---|---|---|---|
| RunPod RTX 4090 24GB | $0.34/hr community, $0.89/hr secure | page "Updated 2026-10-08" | https://www.runpod.io/pricing |
| RunPod A100 80GB PCIe / SXM | $1.19 / $1.39 community; $1.79 secure | 2026-10-08 | same |
| RunPod L40S / RTX 3090 / H100 PCIe | $0.79 / $0.22 / $1.99 community | 2026-10-08 | same |
| $20 RunPod credit | ~58 h of 4090 community, ~17 h of A100 PCIe community (derived) | | |
| SetFit train time (N=8, binary tasks) | ~30 s on V100, ~$0.025 | 2022-09 | https://huggingface.co/blog/setfit |
| ModernBERT-base full FT on Banking77 | 321 s for 5 epochs on L4 (~156 rows/s, derived, padded to 512) | 2025 | https://www.philschmid.de/fine-tune-modern-bert-in-2025 |
| Unsloth speedup | 2.74x (TinyLlama 1.1B, A100-40GB), 3.87x (T4) | 2023-12 | https://huggingface.co/blog/unsloth-trl |
| Banking77 SetFit all-mpnet-base-v2 | 83.5 @5, 88.0 @10, 91.2 @20, 94.0 full | 2023 | https://arxiv.org/abs/2308.14634 |
| Banking77 frozen bge-small + LR | 85.7 (8/class) | model card | BAAI/bge-small-en-v1.5 README |
| CLINC150 BERT full-data in-scope | 96.9 | 2019 | https://arxiv.org/abs/1909.02027 |
| Teacher rows for a working student | ~5.5k (DistilRoBERTa vs GPT-5, F1 -0.025); NVIDIA rule 10k-100k | 2026-01 / 2025-06 | links in Findings 4 |
| LoRA 0.5B throughput | 10-30k tok/s, ~1-3 min per 10k short rows x 3 epochs | derived, not measured | see Findings 2 |
| Banking77 GPT-4 few-shot | 83.1 (3-shot, curated), 74.2 (random) | 2023 | https://arxiv.org/abs/2308.14634 |
| Batch-invariant kernel overhead | ~20% matmul; 26 s -> 42-55 s for 1,000 seqs on Qwen3-8B | 2025-09 | Thinking Machines blog |

## Blockers / plan changes

1. Start students with frozen embeddings + LR (and kNN) on CPU. It is the baseline any heavier student must beat, it costs ~$0, it is deterministic, and it already hits 82-86% on a 77-class task at 8 rows/class. Add SetFit, then a fine-tuned encoder, then a LoRA decoder only if the certified coverage (fraction answered under the error budget) improves.
2. Do not plan on LoRAX; archived 2026-10-12. Use vLLM multi-LoRA for the "per-step LoRA adapters" mission, or skip serving entirely in the smoke (load adapters in-process).
3. Hypernetwork mission lacks training tasks: public precedent used ~479 tasks; Dreaming has ~5 step types. A smoke would have to pretrain on a pool of public classification datasets with varied label sets (Banking77, CLINC150, etc.) and test transfer to bookkeeping labels; it will likely underperform per-step adapters. Scope as a toy.
4. A teacher is not gold: GPT-4 few-shot gets 74-83% on Banking77. Keep a gold/human-reviewed slice per call site for certification, and measure student-vs-teacher agreement separately from student-vs-truth. Active selection of rows to label (M-RARU style) matters more than volume.
5. Replay: pin batch size/padding and CPU thread count (or use batch-invariant mode), store argmax + int-quantized margin. Raw float logits differing in the last bit across hosts is a real risk (not tested here).
6. Calibration is not covered by any source above; certification needs a confidence score whose ordering is reliable. Softmax of fine-tuned encoders is commonly overconfident; plan temperature scaling or conformal calibration on the shadow set (my inference, not sourced).
7. Banking77 and CLINC150 are intent datasets, not bookkeeping; they establish rows-per-class scaling only. The 12-label categorize_transaction step needs its own synthetic/teacher-labeled set.

## Unverified

- Any measured LoRA tokens/s for Qwen2.5-0.5B / Qwen3-0.6B on 4090 or A100; my 10-30k tok/s is a FLOPs-based estimate and 4090/A100 peak-FLOPs figures were not re-checked here.
- SetFit time for 77 classes x 8 rows on CPU (paper timing is for binary tasks).
- All-MiniLM-L6-v2 Banking77 score, nomic-embed Banking77 score, and whether the MTEB model-index numbers were produced with the 8-shot x 10-experiment protocol.
- SetFit numbers on CLINC150 (only a snippet: ~85.9 @8-shot for a 22M model); SetFit vs DeBERTa/ModernBERT under one protocol.
- Whether a decoder 0.5B with a classification head beats an encoder of similar size on short-text classification.
- Whether Unsloth supports sequence classification and Windows; PEFT SEQ_CLS code pattern was written from memory (the docs URL redirected).
- OpenPipe's rows-needed guidance, the $23,000 -> $19 claim, and what happened to its hosted fine-tuning product after the CoreWeave deal; OpenPipe ART license.
- NVIDIA Data Flywheel fine-tune dataset size, GPU-hours, and the repo URL; xLAM 60k figure.
- Text-to-LoRA base models, hypernetwork sizes and task counts beyond secondary sources.
- Licenses marked "unverified" in the table (MTEB, DSPy, gte/e5, vLLM re-check).
- Datasets: HF repo ids and no-login download status for Banking77 / CLINC150 (script-based vs parquet) were not checked in this note; license of both believed CC-BY but not confirmed.
- ONNX Runtime / sentence-transformers CPU throughput numbers (efficiency docs benchmarks section was truncated; no sentences/s found). Model2Vec "500x" is a README claim.
- Reported degradation when training on LLM-generated vs gold labels (seen in a search summary, source not identified).

## Addendum: SetFit CPU training time (verify of "a few minutes on CPU"), 2026-10-17

Verdict: CORRECTED. The quote exists, but it does not support the claim for a 12-17 label step with a 110M encoder on this laptop. Finding 1/line 52 should read "frozen-embedding + LR is CPU-trivial; SetFit with a 22M encoder is CPU-feasible with a capped pair count; SetFit with mpnet-base needs a GPU or ~1 h of CPU".

Verified facts
- The quote is real but unqualified: "you can even train SetFit on CPU in just a few minutes!" (https://huggingface.co/blog/setfit, published 2022-09). The same post gives "30 seconds ... $0.025" for 8 labeled examples on a V100 and uses paraphrase-mpnet-base-v2 with `num_iterations=20`, `num_epochs=1`. It states no dataset, class count, row count or CPU spec for the CPU claim and no multi-class scaling statement.
- The paper's 30 s figure: "SetFitMPNet takes approximately 30 seconds to train on a p3.2xlarge AWS instance (16GB GPU)" at N=8 per class, "for our datasets"; batch size 16, R=20, 1 epoch. The paper does not mention CPU training at all (https://arxiv.org/abs/2209.11055 via https://ar5iv.labs.arxiv.org/html/2209.11055). The "binary tasks" wording in line 40 / the concern is not in the paper: it does not name which datasets the timing covers, and its test set includes multiclass ones (SST-5, Emotion, AG News). Treat "binary" as unverified. Either way, 2-6 classes at 8 shots is 16-48 rows, versus 96-272 here.
- Pair count is set by row count N, not class count, and the library default is NOT num_iterations=20. In SetFit 1.x `TrainingArguments.num_iterations` defaults to None, `sampling_strategy` defaults to "oversampling", batch_size (16, 2), num_epochs (1, 16) (https://raw.githubusercontent.com/huggingface/setfit/main/src/setfit/training_args.py). With num_iterations set, pairs = 2 x num_iterations x N. With it unset ("oversampling"), the sampler builds all N(N+1)/2 index pairs (self-pairs included), then oversamples the smaller pool, so pairs = 2 x max(positives, negatives) (https://raw.githubusercontent.com/huggingface/setfit/main/src/setfit/sampler.py; summary by the fetch tool, not a line-by-line read). With K balanced classes the negatives dominate, so embedding-phase steps grow roughly as N^2, i.e. with classes x shots squared. Class count enters only through N and the positive/negative mix.
- Pair and step counts, my arithmetic from those formulas (batch 16, K classes x n shots/class):

| K x n | N | default (oversampling) pairs / steps | num_iterations=20 pairs / steps |
|---|---|---|---|
| 2 x 8 | 16 | 144 / 9 | 640 / 40 |
| 12 x 8 | 96 | 8,448 / 528 | 3,840 / 240 |
| 12 x 16 | 192 | 33,792 / 2,112 | 7,680 / 480 |
| 12 x 20 | 240 | 52,800 / 3,300 | 9,600 / 600 |
| 17 x 8 | 136 | 17,408 / 1,088 | 5,440 / 340 |
| 17 x 16 | 272 | 69,632 / 4,352 | 10,880 / 680 |
| 77 x 8 (Banking77) | 616 | 374,528 / 23,408 | 24,640 / 1,540 |

- CPU time, UNMEASURED (no torch installed on this machine; python import fails). Estimate = 6 x non-embedding params x tokens per step (16 pairs = 32 sentences x 16-32 tokens; transaction strings are short), at the 25-100 GFLOPS fp32 effective rate measured in compute-pricing.md (8-122 GFLOPS, numpy GEMM, so optimistic for PyTorch):
  - mpnet-base (~85M non-embedding): 2.6-21 s/step, central ~5-10 s. 12 x 8 default (528 steps): ~0.8-1.5 h. 12 x 8 with num_iterations=20 (240 steps): ~20-40 min. 12 x 16 default (2,112 steps): ~3-6 h.
  - MiniLM-L6 (~10.6M non-embedding, ~8x cheaper): 0.33-2.6 s/step, central ~0.65-1.3 s. 12 x 8 default: ~6-11 min. With num_iterations=20: ~3-5 min. 12 x 16 default: ~23-46 min.
  - The classifier-head phase (LR on frozen embeddings) and inference are forward-only and cheap in all cases.
- So "a few minutes on CPU" is plausible only for MiniLM-class models at 12 x 8 or with capped pairs; the concern (tens of minutes to hours for mpnet) is right in order of magnitude, with the default sampler making it worse than the num_iterations=20 tutorial path.
- Anecdote only: a forum thread estimates "over 700 hours" for ~3,600 training rows (GPU model unknown, PyTorch 1.10.1, batch 15, 12 epochs; no replies) (https://discuss.huggingface.co/t/how-can-i-speed-up-the-setfit-model/67751). It shows the unbounded-pairs failure mode, not a benchmark.
- No public CPU benchmark of SetFit multiclass training time was found (two searches, 2026-10-17). A GitHub issue (mirrored at https://lightrun.com/answers/huggingface-setfit-setfit-for-a-large-number-of-classes) says SetFit works on a ~90-class dataset; that is about accuracy, not speed.

Plan changes
1. Baseline stays frozen bge-small/MiniLM embeddings + LR (seconds on CPU, no pair generation).
2. Any SetFit smoke: set `num_iterations` explicitly (start at 5-20) or `max_steps` (e.g. 300) so runtime is bounded; do not rely on the default sampler beyond ~12 x 8 rows. Base model paraphrase-MiniLM-L3/L6 or all-MiniLM-L6-v2 on CPU; mpnet-base on a GPU (t4-small $0.40/h per compute-pricing.md; a 528-step job is minutes there).
3. Before quoting a CPU time in a plan, run one 20-step timing (`max_steps=20`) on this laptop and extrapolate; that is the cheapest way to turn the table above into a measurement. Needs `pip install torch setfit` (not installed).

## Addendum: Verification of "frozen embeddings + LR ... is deterministic" (Blockers #1) and "Model2Vec trivially deterministic" (2026-10-17)

Verdict: **CORRECTED.** True for the Model2Vec encode step. False as written for the whole "frozen embeddings + LR" student, because the head's BLAS matmul is batch- and thread-dependent, and the transformer embedders that produce the 82-86% numbers in the same bullet are not deterministic across batch/threads (see determinism.md). "Costs ~$0" was not re-checked here (it was never in doubt for CPU).

Measured locally (script `m2v_det.py`, not in the repo; same laptop and numpy 2.2.5 + MKL as determinism.md). The real `model2vec` and `sklearn` packages are not installed, so the script re-implements the ops read from the Model2Vec source (random float32 table 30000x256, 2048 random sentences of 3-40 tokens, head 256->256->12 with random weights). Treat as an emulation, not a run of the package.
- Encode step (length-bucketed `E[ids].mean(axis=1)` + L2 norm): all 2048 rows bit-identical when each sentence is encoded alone, in the full batch, shuffled, or in chunks of 64; one embedding-matrix hash (`9e7f70d4feba`) across default, `MKL_NUM_THREADS=1/4/14`, and `MKL_CBWR=AVX2,STRICT` with 1/14 threads. Reason: a gather plus a float mean, no BLAS, and each sentence's rows are reduced in a fixed order inside its own length bucket.
- Head step (`x @ W.T + b`, ReLU, `x @ W2.T + b2`): **0 of 2048 rows** bit-equal between batch size 1 and batch size 8, 64, 1024 or 2048, in every config (max abs logit diff 6.7e-8, about 1 ulp). Softmax outputs were bit-equal for only ~23% of rows (474-538 of 2048). Argmax agreed 2048/2048, but random weights have few near-ties, so this says nothing about rows near a certified cutoff.
- Thread dependence: at batch size 64, 1024 and 2048 the default, `MKL_NUM_THREADS=1` and `MKL_NUM_THREADS=4` runs give different logit hashes (bs64: `7449491f3b7d` / `aadd99f31890` / `bd30716e279d`). With `MKL_CBWR=AVX2,STRICT`, batch sizes 8-2048 and 1/14 threads all give one hash (`aadd99f31890`), but it is not equal to the batch-size-1 result.
- Batch size 1 (row-by-row) gave one logit hash (`86c93725b52a`) in all 6 configs. This is the cheapest exact path found: serve the head one row at a time, or pin one fixed padded shape plus `MKL_CBWR`. Tested on one AVX2-only CPU; other ISAs/CPUs and numpy/BLAS builds untested (MKL picks kernels per ISA unless CBWR pins it).

Source facts (read this session):
- Model2Vec encode: tokenizer `encode_batch_fast`, sentences grouped by token count (no padding), `out[indices] = self.embedding[id_matrix].mean(axis=1)`, then float32 L2 norm; batch_size 1024, joblib threads only above 10,000 sentences. https://raw.githubusercontent.com/MinishLab/model2vec/main/model2vec/model.py
- Model2Vec inference head is not sklearn: a numpy `MLPHead` loaded from `head.safetensors` (sklearn/skops only in a legacy-conversion path); `x @ self.weight.T + self.bias`, `np.maximum`, `np.exp` softmax. https://raw.githubusercontent.com/MinishLab/model2vec/main/model2vec/inference/model.py and https://raw.githubusercontent.com/MinishLab/model2vec/main/model2vec/inference/mlp.py . Dtype of the head weights is not determinable from the source.
- sklearn linear classifiers: `decision_function` is `safe_sparse_dot(X, coef_.T, dense_output=True) + intercept_`, so a dense LR head is a BLAS GEMM with the same exposure as the MLP head. https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/linear_model/_base.py (the definition of `safe_sparse_dot` itself was not shown).
- sklearn docs only say BLAS/LAPACK threads come from MKL/OpenBLAS/BLIS and can be pinned with `*_NUM_THREADS` or threadpoolctl; they make no bitwise-reproducibility statement. https://scikit-learn.org/stable/computing/parallelism.html
- NumPy docs: pairwise summation is used only along the fast axis, the order of additions is not specified, so the encode-step invariance is an implementation property of the bucketed axis=1 mean (pin the numpy version and keep a canary). https://numpy.org/doc/stable/reference/generated/numpy.sum.html
- Accuracy mismatch: the 82-86% Banking77 figures are for bge-small / gte-small / e5-small (transformer forward passes, determinism.md applies). No Banking77 score for any Model2Vec model was found; the potion-base-32M README table shows a "Class" column of 71.70 (unlabelled in the fetched text, so not confirmed as the MTEB classification average). https://huggingface.co/minishlab/potion-base-32M/raw/main/README.md

Plan change for Blockers #1: replace "it is deterministic" with "Model2Vec encode is deterministic; the head is deterministic only on a fixed code path". Concretely: (1) Model2Vec (or another static-embedding) student + head served at batch size 1, or at one fixed padded shape with `MKL_CBWR=AVX2,STRICT` and fixed thread count; (2) transformer-embedder students follow determinism.md Recipe B/C; (3) hash (label, int-quantized margin), not float probabilities. Quantizing helps but does not make a flip impossible: derived estimate, a ~1e-7 jitter against a 1/4096 grid moves a value across a grid line with probability ~3e-4 per value, so replay checks still need the fixed code path or a tolerance band near the cutoff.

Still unverified: the real `model2vec` package and real sklearn `LogisticRegression.predict_proba` were not run; SIMD dispatch differences in `np.exp` across CPUs (AVX2 vs AVX-512) were not tested; the per-sentence weighted/quantized-vocab encode path was not tested; training reproducibility (lbfgs, BLAS threads) was not tested (irrelevant for replay if the trained weights are hash-pinned, relevant if "retrain gives the same model" is promised); Model2Vec's `use_multiprocessing` path (>10,000 sentences) was not tested.
