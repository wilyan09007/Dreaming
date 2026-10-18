# Determinism / byte-identical replay of small hosted models

Researched 2026-10-17. "unverified" = not confirmed from a primary source this session. All URLs fetched or surfaced in search this session unless marked.

## Summary

- The real culprit is not atomics/concurrency, it is **batch non-invariance**: the same request gets different float reduction orders depending on how many other requests share the batch. A forward pass at a fixed shape on one GPU is run-to-run deterministic. (Thinking Machines, Sept 2025: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- TML result: Qwen3-235B, temp 0, 1000 completions x 1000 tokens gave **80 unique outputs** (first divergence at token 103); with batch-invariant kernels **1 unique**. Same code now ships in vLLM (`VLLM_BATCH_INVARIANT=1`, beta) and SGLang (`--enable-deterministic-inference`).
- Cost of batch invariance on GPU: **~24-62% slower** in 2025 measurements (SGLang avg +34% latency, TML original ~2.1x slower on Qwen3-8B). Oct 2026 vLLM blog shows 9-12% on big MoE/GB200 with tuned kernels, but no small-model/consumer-GPU numbers exist (unverified for our case).
- **Cross-SKU byte identity does not hold** with stock kernels: A100 vs H100 vs L40S differ (cuBLAS picks different split-K by arch/SM count). Same SKU, different physical cards: identical. H100-NVL/PCIe/H200 matched each other (same compute capability). Pin the SKU class and canary-check it at startup.
- Cheapest recipe for our workload (small dense model, mostly single-forward classification): **fixed-shape padded buckets, one stream, fixed batch size B, no dynamic batching** - batch invariance by construction. Or just use vLLM/SGLang batch-invariant mode on dense TP=1 models. Both need a 1000-request concurrent smoke (recipe below).
- CPU: I measured locally (numpy+MKL, Core Ultra 7 155U): fp32 matmul bits **change with thread count** unless `MKL_CBWR=AVX2,STRICT`, and a row computed alone != same row in a batch of 64 even with CBWR. So CPU is also not batch-invariant; use fixed shapes. ONNX Runtime / llama.cpp thread-count invariance: no documented guarantee found.
- Do not promise byte-identical **logits** across stacks. Promise byte-identical **(label, accept/defer decision)** from the audit log plus a replay check on the pinned stack; quantize confidence before hashing/thresholding.
- Hash everything pinned: weights (safetensors sha256, available from the HF API without download), engine/CUDA/driver versions, GPU SKU + SM count, env vars, plus a startup canary of ~16 fixed prompts with golden output hashes.

## Findings

### 1. Thinking Machines "Defeating Nondeterminism" (Sept 2025) and repo
Source: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ ; code https://github.com/thinking-machines-lab/batch_invariant_ops
- Atomics are rare in LLM forward passes; the forward pass is run-to-run deterministic. Problem = kernels change reduction strategy with batch size; server load changes batch size.
- Fixes: RMSNorm - one batch row per core (no split reduction). Matmul - one kernel config for all shapes, no split-K / no tensor-core instruction switching; ~20% slower than cuBLAS. Attention - write KV cache/page table before kernel; decode uses fixed split *size* not fixed number of splits.
- Numbers: Qwen3-8B 1 GPU, 1000 seqs x 90-110 tokens: vLLM default 26 s, deterministic unoptimized 55 s, improved attention 42 s. (Blog quotes seconds; one secondary source mislabels as tokens/s.)
- Repo: replaces `torch.mm`, `addmm`, `log_softmax`, `mean` via `with set_batch_invariant_mode():`. MIT. Hardware tested not stated. 7 commits (small, demo-grade). Its vLLM demo needed an upstream PR (now merged as the vLLM feature): without it 18 unique of 1000 length-100 completions, with it 1.

### 2. vLLM (2026)
- Docs (latest): https://docs.vllm.ai/en/latest/features/batch_invariance/ ; v0.26.0: https://docs.vllm.ai/en/v0.26.0/features/batch_invariance/
  - `VLLM_BATCH_INVARIANT=1` (set before importing vllm / before `vllm serve`). Status **beta**. "may impact performance"; no numbers given.
  - **Hardware: docs disagree by version.** v0.12-v0.22.1 say compute capability >= 9.0 (H100+); v0.26.0/latest say >= 8.0 NVIDIA, AMD Instinct (gfx90a/942/950), Intel XPU (Triton attn). So RTX 4090 (8.9), A40/A100/L4 need a recent vLLM (>= 0.26 per these pages).
  - Tested models (dense): Qwen3 0.6B/1.7B/8B (+AWQ 4B/8B), Qwen2.5 0.5B-32B, Llama 3.1/3.2, Mistral-7B-v0.3, Phi-3.5-mini; also DeepSeek, GPT-OSS, Qwen3 MoE. Qwen3-0.6B was added to the tested list by a Docs PR: https://github.com/vllm-project/vllm/pull/53403
  - TP: disables some custom all-reduce; TP>1 was **non-deterministic** until bug fixed Aug 25 2026 (fused all-reduce+RMSNorm): https://github.com/vllm-project/vllm/issues/51290 . Stay TP=1.
- Reproducibility page: https://docs.vllm.ai/en/latest/usage/reproducibility/ - not reproducible by default; offline: `VLLM_ENABLE_V1_MULTIPROCESSING=0` makes scheduling deterministic OR use batch invariance; online: only batch invariance. "Reproducible only on the same hardware and the same vLLM version." `seed` default 0 in V1 (per-worker random state).
- Async scheduling is on by default and (per third party, Sept 30 2026) can break invariance since batch membership depends on wall-clock: https://www.spheron.network/blog/vllm-batch-invariance-why-temperature-0-isnt-deterministic/ (unverified against vLLM issues; flag name to disable not confirmed). Fused-MoE kernels reported non-invariant (same source, unverified).
- vLLM+TorchTitan bitwise RL (Nov 2025): https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference . Oct 14 2026 follow-up: https://vllm.ai/blog/2026-10-14-train-inference-consistency - bitwise-consistent inference at 91.3% (DeepSeek-V4-Flash, GB200) and 87.5% (Nemotron-3.5 30B-A3B) of baseline throughput; not supported: TP>1, MTP, speculative decoding. No H100/consumer GPU data.

### 3. SGLang
- Blog: https://lmsys.org/blog/2025-09-22-sglang-deterministic/ ; docs https://docs.sglang.io/advanced_features/deterministic_inference.html
- `--enable-deterministic-inference`, backends FlashInfer / FA3 (default) / Triton (Triton also runs on AMD). Works with chunked prefill, CUDA graphs, radix cache (FA3, Triton; not FlashInfer), temp>0 via per-request `sampling_seed` (default 42; Gumbel noise from seeded hash, `multinomial_with_seed`).
- **Dense models only, TP1/TP2 only** (MoE and TP>2 listed as future work in the blog; docs silent). Verified over 50 trials: 1 unique output in single/mixed/prefix tests vs up to 18 unique in normal mode.
- Overhead (Qwen3-8B, TP1, H200, 256 req): end-to-end latency +24.4% to +55.1%; FlashInfer/FA3 avg +34.35% vs TML's 61.5%. CUDA graphs: 441->1245 tok/s (H100, 16 req 1024/1024), 2.8x. Authors: use "primarily for debugging and reproducibility".
- Built-in check: `python3 -m sglang.test.test_deterministic --test-mode single|prefix|radix_cache` -> expect `Unique samples: 1`.

### 4. Follow-up research (2026)
- LLM-42 (verify-rollback speculation): https://arxiv.org/abs/2601.17768 / html https://arxiv.org/html/2601.17768 . Llama-3.1-8B, 4xH100 PCIe. SGLang deterministic is 24% (in2048/out512) to 36% (ShareGPT) slower offline; mixing 1 deterministic request into 10 normal ones drops decode 931 -> ~415 tok/s (-56%). LLM-42 keeps overhead ~proportional to deterministic fraction. Preprint, code availability unverified. Only useful if you need determinism for a small fraction of traffic.
- Cross-architecture: https://arxiv.org/abs/2609.25624 (Sept 22 2026, v1 preprint). Fixed-config fused-upcast FP32-accumulate GEMM kernels; linear layers bitwise identical on A100 (SM80), L40S (SM89), H100 (SM90). Standard cuBLAS was bitwise identical across the three GPUs for **0 of 7** shapes probed. 11/12 model x benchmark cells had 0% cross-GPU divergence; residual 0.08% (1 problem) traced to unpinned attention on H100. Costs: ~2.4x slower than TF32 GEMM at decode shapes, 5.6x at prefill; claims 1.17-3.1x faster than an FP32-compute baseline (not vs batch_invariant_ops). Only linear layers pinned; invariance only for decode M <= 64. Also: pin the chat-template date string (it caused false divergence). Code release unverified.
- Same-SKU audit: https://arxiv.org/abs/2606.00279 (html v2 https://arxiv.org/html/2606.00279v2). Unmodified vLLM/HF, no flags: 10 repeats per GPU gave distance 0; A100-SXM4 vs A100-PCIe 0.000; A100 vs H100-NVL 0.526; A100 vs L40S 0.450; H100-NVL/PCIe/H200 mutually 0.000. Quantization: AWQ/GPTQ-marlin deterministic; **GPTQ exllama fallback diverged (atomicAdd in q_gemm.cu)**; Kimi-K2 INT4 dequant non-deterministic. Footnote: SM count is fixed per SKU, which is why same-SKU cards agree.
- TP-size invariance (ICML 2026 poster) https://arxiv.org/html/2511.17826v1 - irrelevant at TP1.

### 5. PyTorch / cuBLAS knobs
- cuBLAS bit-wise reproducibility guarantee: same bits every run **on GPUs with the same architecture and same number of SMs**, within one toolkit version; not guaranteed across toolkit versions; broken by multiple active streams unless you set `CUBLAS_WORKSPACE_CONFIG=:4096:8` (+~24 MiB) or `:16:8` (may limit perf) or per-stream workspaces; split-K itself is deterministic. https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
- PyTorch: https://docs.pytorch.org/docs/2.14/notes/randomness.html - "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms"; CPU vs GPU may differ. `torch.use_deterministic_algorithms(True)` errors on known nondeterministic ops; with `warn_only=True` fused SDPA backends (Flash/Efficient/cuDNN) keep default nondeterministic paths -> use `warn_only=False`. `cudnn.benchmark=False` (benchmark mode can pick different algos run to run). These flags make *run-to-run at fixed shape* deterministic; they do **not** give batch invariance (the CUBLAS_WORKSPACE_CONFIG text itself did not render in the PyTorch page I fetched; value comes from the NVIDIA doc).

### 6. CPU determinism
- My local test (script `blas_threads.py`, not in the repo; Core Ultra 7 155U, 14 threads, numpy 2.2.5 + MKL 2023.1, fp32 64x4096 @ 4096x4096, sha256 of result):
  - default: thread counts 1,2,4,8,14 -> 5 different hashes; repeats at 8 threads -> identical (run-to-run OK).
  - `MKL_CBWR=COMPATIBLE`: 1,2,4,8 same; 14 differs.
  - `MKL_CBWR=AVX2,STRICT`: threads 1,2,3,5,7,8,12,14 all **identical**.
  - Row 0 alone vs row 0 inside the 64-row batch: **not equal in every config** (not batch-invariant).
- Intel docs: reproducibility needs fixed thread count and alignment and same ISA path; strict CNR drops the thread-count requirement for a limited routine set: https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2023-0/obtaining-numerically-reproducible-results.html . Cross-CPU-model guarantee not stated there (unverified).
- llama.cpp: no documented thread-count invariance. PR #16016 (CUDA deterministic mode, still Draft/unmerged) says "CPU is already deterministic"; a contributor said he does not want to maintain bit-identical-across-batch-size guarantees and that disabling prompt cache + one server slot already gives bit-identical results: https://github.com/ggml-org/llama.cpp/pull/16016 . Issue #22956 (closed stale June 27 2026): `--threads 1` still nondeterministic when OpenBLAS+OpenMP ignores thread setting; workaround `OMP_NUM_THREADS=1`: https://github.com/ggml-org/llama.cpp/issues/22956
- ONNX Runtime: `SetDeterministicCompute` session option exists (C API, https://onnxruntime.ai/docs/api/c/struct_ort_api.html) but I could not confirm which EPs/ops it covers; a 2023 commit says some CUDA training kernels stay nondeterministic regardless. No doc on thread-count invariance. Unverified.

### 7. Hashing / fingerprints
- HF LFS object id = sha256 of the file; readable without download: `curl https://huggingface.co/api/models/Qwen/Qwen3-0.6B/tree/main` returned `model.safetensors` size 1503300328, lfs.oid `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` (checked 2026-10-17). `model_info(files_metadata=True)` also exposes LFS metadata (https://huggingface.co/docs/huggingface_hub/package_reference/hf_api). No hash->file reverse lookup exists (https://discuss.huggingface.co/t/model-file-lookup-by-sha256-hash/154293).
- Safetensors format: 8-byte length + JSON header + raw bytes (https://huggingface.co/docs/safetensors/index). File bytes depend on the writer (header key order/metadata), so hash the exact served file AND a canonical hash (sorted names, dtype, shape, raw tensor bytes) of the *loaded* state_dict after any dtype cast / LoRA merge / hypernet generation.
- Fingerprint to log with every prediction (recommendation, not from a source): GPU name, compute capability, SM count, driver, CUDA runtime + cuBLAS version, torch/vllm/triton versions, container image digest, dtype, attention backend, `VLLM_BATCH_INVARIANT`, `CUBLAS_WORKSPACE_CONFIG`, `MKL_CBWR`, thread counts, max_num_seqs, tokenizer + chat template hash, seed, engine git sha.

## Recipe: 1000 identical outputs under concurrent load

Recipe A (GPU, dense small model, off-the-shelf; try first):
1. Pod: one SKU class, Linux, TP=1. Candidates (RunPod 2026-10-08): RTX 4090 $0.34 community / $0.89 secure; A40 $0.35/$0.59; L4 $0.44/$0.59; L40S $0.79/$1.09. Needs vLLM >= ~0.26 for cc 8.x per docs, or SGLang with fa3/triton.
2. Pin container image digest + `vllm==<exact>` (or sglang), model by HF commit sha + safetensors sha256, explicit `--dtype` (bf16 or fp16, never `auto`), `--seed 0`.
3. vLLM: `VLLM_BATCH_INVARIANT=1`; disable async scheduling and (until proven) prefix caching; no spec decode; no MoE/FP8-MoE; no GPTQ-exllama; `temperature=0` (or per-request seed). SGLang: `--enable-deterministic-inference --attention-backend fa3` (per docs); per-request `sampling_seed`.
4. Pin chat template + any date/time in prompts.
5. Smoke: reference = 1000 prompts run serially; then run the same 1000 shuffled at concurrency 1/8/64 with 50% random distractor traffic; compare token ids **and** returned logprobs bitwise (exact float equality, not tolerance). Repeat after server restart and on a second pod of the same SKU. Pass = 1 unique output per prompt in all arms. Then deliberately run on a different SKU to document the divergence rate.
6. Startup canary: 16 fixed prompts, golden hash stored with the predictor; refuse to serve on mismatch.

Recipe B (our own server; likely better for classification heads, mission 1/4/5 predictors):
- Single forward pass (no autoregressive decode, no KV cache, no split-KV): read logits for the 12 labels. This removes the hardest kernels.
- Fixed batch size B (pad with dummy rows), fixed length bucket chosen only from the request's own length (power-of-two buckets, always same padding/position ids), one CUDA stream, one worker process per GPU, bf16/fp16 weights with fp32 accumulate.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True, warn_only=False)`, `cudnn.benchmark=False`, no `torch.compile(mode="max-autotune")` / Triton autotune (autotune picks configs by timing - my reasoning, unverified), optional `with set_batch_invariant_mode():` from batch_invariant_ops (does not cover attention).
- Reasoning (unverified empirically): with identical (B, L, kernel config) the reduction tree per output element is data-independent, so row results do not depend on co-batched content. Supported by cuBLAS doc (same arch+SM count -> same bits) and 2609.25624 ("invariance within decode bucket M<=64"). Test it with the same 1000-request smoke before relying on it.
- Output contract: store label + accept/defer decision at serve time; quantize confidence (e.g. int in 1/4096 steps) *before* comparing to the certified cutoff so near-threshold float jitter cannot flip decisions on replay.

Recipe C (CPU hosting, simplest guarantee): fp32, `MKL_CBWR=AVX2,STRICT` (or equivalent), fixed `OMP_NUM_THREADS`, bs=1 or fixed-shape, single process; verify on the target CPU model. Possible stronger option (my reasoning, unverified): int8 weights/activations with int32 accumulation are exactly associative, so integer GEMMs are thread-count/batch/ISA independent; remaining float ops (norm, softmax, scales) need fixed order.

## Reusable components

| name | url | license | why |
|---|---|---|---|
| batch_invariant_ops | https://github.com/thinking-machines-lab/batch_invariant_ops | MIT | drop-in batch-invariant mm/addmm/log_softmax/mean for own PyTorch server (Recipe B); no attention |
| vLLM batch invariance | https://docs.vllm.ai/en/latest/features/batch_invariance/ | Apache-2.0 (not re-checked) | `VLLM_BATCH_INVARIANT=1`, serves concurrent load; beta; use recent version |
| SGLang deterministic mode | https://docs.sglang.io/advanced_features/deterministic_inference.html | Apache-2.0 (not re-checked) | `--enable-deterministic-inference`, built-in unique-sample tests, per-request sampling_seed |
| LLM-42 | https://arxiv.org/abs/2601.17768 | paper; code unverified | deterministic only for flagged requests, if full invariance is too slow |
| fused-upcast cross-GPU GEMM (2609.25624) | https://arxiv.org/abs/2609.25624 | paper; code unverified | only known route to cross-SKU (Ampere/Ada/Hopper) linear-layer identity |
| Bit-exact audit repos (named in 2606.00279: `hardware_rounding_error_predictor`, `Floating_point_noise_GPU_verification`) | https://arxiv.org/abs/2606.00279 (repo URLs not captured) | unverified | rounding-error "signature" of HW/SW stack; fingerprint idea |
| cuBLAS reproducibility rules | https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility | n/a (docs) | authoritative statement of what is guaranteed |
| oneMKL CNR (`MKL_CBWR`) | https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2023-0/obtaining-numerically-reproducible-results.html | Intel (not re-checked) | CPU thread-count invariance, empirically confirmed locally |
| safetensors + HF Hub API | https://huggingface.co/docs/safetensors/index | Apache-2.0 (not re-checked) | sha256 of weights without download via tree API |

## Numbers (as of 2026-10-17)

| item | value | source |
|---|---|---|
| unique outputs, Qwen3-235B, 1000 x 1000 tok, temp 0, default | 80 (top one 78x) | TML blog |
| same with batch-invariant kernels | 1 | TML blog |
| Qwen3-8B, 1000 seqs ~100 tok | 26 s default / 55 s det. / 42 s improved attn | TML blog |
| SGLang det. mode latency overhead (H200, Qwen3-8B) | +24.4% to +55.1%; avg 34.35% FlashInfer/FA3 | LMSYS blog |
| SGLang CUDA graph speedup (H100) | 2.8x (441->1245 tok/s) | LMSYS blog |
| SGLang det. vs normal, Llama-3.1-8B 4xH100 PCIe | 24-36% slower offline | arXiv 2601.17768 |
| 1 det. request among 11 on SGLang | 931 -> ~415 tok/s | arXiv 2601.17768 |
| vLLM bitwise-consistent, GB200 MoE | 91.3% / 87.5% of baseline | vLLM blog 2026-10-14 |
| cross-GPU GEMM kernels | 2.4x (decode) / 5.6x (prefill) slower than TF32 GEMM; prefill ~0.85x cuBLAS | arXiv 2609.25624 |
| cuBLAS workspace cost | ~24 MiB for `:4096:8` | NVIDIA doc |
| matmul batch-invariant penalty | ~20% vs cuBLAS | TML blog |
| RunPod 2026-10-08 ($/hr secure / community) | 4090 0.89/0.34; A5000 0.27/0.16; L4 0.59/0.44; A40 0.59/0.35; L40S 1.09/0.79; A100 PCIe 1.79/1.19; H100 PCIe 2.89/1.99; H100 SXM 3.99/2.69 | https://www.runpod.io/gpu-cloud/pricing |
| $20 RunPod buys | ~59 h 4090 community or ~22 h 4090 secure; a 3-SKU determinism smoke (4090+A40+L4, ~1 h each) ~ $1.1-1.5 | computed |
| qwen3-0.6B weights | 1,503,300,328 bytes (bf16), sha256 f47f7117...874b | HF API, 2026-10-17 |
| local MKL test | see Findings 6 | own run |

## Blockers / plan changes

1. **Cross-SKU byte identity is not available off-the-shelf.** Hosted predictors must declare an `execution_profile` (GPU SKU class + stack digest + mode) as part of the replay key; replay on a different profile = "verify within tolerance", not byte-match. Pin one SKU class for the demo. Whether 4090 (128 SMs) and L40S (142 SMs), both Ada sm89, match is unverified and by the SM-count argument may not.
2. RunPod Community Cloud: you pick the SKU but not host/driver. Cuts cheaply but add the startup canary; driver-version effects on bits are not covered by any source I found (unverified).
3. vLLM docs conflict on minimum compute capability (9.0 in v0.12-0.22, 8.0 in v0.26+). Use a recent vLLM or SGLang; test on the actual 4090 first.
4. **Local Windows CPU box cannot reproduce GPU bytes**, and vLLM/SGLang are Linux/CUDA-targeted (Windows support not checked, unverified). Treat local runs as functional tests only; replay hashes are per backend.
5. Mission 2 consequences: a hypernetwork that writes student weights, and merged LoRA adapters, produce new weights - hash the *generated tensors* after cast, and run the hypernet in the same fixed-shape deterministic mode. Per-step LoRA served via batched multi-adapter kernels: batch invariance of those kernels is unverified; serve merged weights or one adapter per fixed-shape worker instead.
6. Avoid for the replayable path: MoE students (fused-MoE non-invariant), TP>1 (bug fixed Aug 2026, still limited), speculative decoding / MTP, GPTQ-exllama, `torch.compile` autotune, FlashInfer with radix cache in SGLang, `warn_only=True`.
7. Accept 25-60% throughput loss for generative steps (draft_note) on vLLM/SGLang deterministic mode; for single-forward classification steps use Recipe B and measure - expected much smaller (unverified).
8. Thresholding: certification cutoffs compared on raw float confidence will flip on replay jitter; certify on quantized confidence.

## Unverified

- Throughput penalty of batch-invariant mode for <=1.7B models on 4090/A40/L4 (no source; all numbers are 8B+ on H100/H200/GB200).
- Whether vLLM prefix caching, speculative decoding, FP8 dense, or async scheduling break invariance (docs silent; only a third-party blog flags async scheduling).
- Whether the exact flag to disable async scheduling / its default state in vLLM >= 0.26.
- Thread-count and batch invariance of PyTorch CPU, ONNX Runtime (MLAS, `SetDeterministicCompute` scope) and llama.cpp (ggml) - no primary source; my only data is numpy+MKL on one Intel CPU.
- Cross-CPU-model identity under `MKL_CBWR`; AMD CPUs.
- Effect of NVIDIA driver version on bits (cuBLAS doc only constrains toolkit version, arch, SM count, streams).
- Same-compute-capability, different-SM-count GPUs (4090 vs L40S, A40 vs A100?) - no source.
- Triton/torch.compile autotune as a nondeterminism source (reasoning only).
- Integer-GEMM (int8/int32) path giving full thread/ISA/batch invariance (reasoning only; ggml Q8_0 mixes float scales).
- Licenses marked "not re-checked"; code release status of LLM-42 and arXiv 2609.25624.
- Fixed-shape padded-bucket recipe (Recipe B) has not been run; it is the first thing the smoke should test.


## Addendum: CPU student (MiniLM / bge-small via torch-CPU and ONNX Runtime) replay determinism

Researched 2026-10-17 (gap 2). Verdict on the original claim "CPU frozen embeddings replay byte for byte across thread counts, batch composition, padding, Intel vs AMD": **NOT CONFIRMED, partly corrected, empirically unresolved.** No published test of bitwise identity for MiniLM or bge-small on torch-CPU or ORT-CPU was found in 5 web searches (only docs and source). Everything below is from docs and source reading, not a run. The "structural" statements are my inference from source and need the smoke at the end.

### Facts

1. **ORT `SetDeterministicCompute` is a GPU-only knob.** Header doc: "enable deterministic compute for GPU kernels where possible... most likely will have a performance cost" (since v1.17); config key `session.use_deterministic_compute`. It does nothing for the CPU EP. Corrects the "scope unconfirmed" note in section 6 above. https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_c_api.h (line ~5280 on main, 2026-10-17); key at https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h
2. **ORT CPU kernel choice is by CPUID feature bits, not vendor, so "Intel vs AMD" is really "which ISA class".** `platform.cpp` picks `MlasGemmFloatKernelAvx` / `...Fma3` / `...Avx512F` (and separate Avx512F variants of Exp, Erf, Gelu-Erf, Silu kernels) from AVX / AVX2+FMA3 / AVX512F bits. I found no vendor-string check there and no env var to force a lower ISA (the only getenv is a RISC-V one). So an AVX2-only part (Intel client 12th gen+/Core Ultra, AMD Zen 2/3) and an AVX512F part (Xeon, Zen 4+) run different kernels; MiniLM uses GELU(erf), so the activation kernel differs too. Whether the bits actually differ is untested. https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/mlas/lib/platform.cpp
3. **ORT fp32 SGEMM looks thread-count- and batch-invariant by construction (inference, untested), with exceptions.** `sgemm.cpp` partitions threads over output M and N only (never K). K is walked serially in blocks of `StrideK`, which depends on N and K (layer constants), not on M. Threads used = min(max threads, M*N*K/(64*1024)+1), so it varies with batch size but only moves tile boundaries. Exceptions: (a) `M == 1` (NoTrans, alpha 1) takes a different `KernelM1Routine`, so a single-row matmul differs from the same row in a bigger one; (b) attention softmax, LayerNorm and mean-pool reduce over sequence length and their lane grouping may change with padded length (not read; unverified). https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/mlas/lib/sgemm.cpp ; constants in https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/mlas/lib/mlasi.h
4. **torch-CPU (x86 wheels) fp32 matmul goes through MKL (my knowledge, not re-verified), so the MKL rules apply, and the strong mode is Intel-only.** Intel: bitwise run-to-run needs one executable and a constant thread count; processor dispatch, alignment, thread count and threaded algorithms all change operation order; strict CNR removes the thread-count condition but covers "a limited set of routines". https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2023-0/obtaining-numerically-reproducible-results.html . `MKL_CBWR` values: AUTO, COMPATIBLE, SSE2..AVX2, AVX512, AVX512_E1; `,STRICT` only for AVX2/AVX512 branches; per the fetched page "Settings other than AUTO or COMPATIBLE are available only for Intel processors", and an unsupported value is silently ignored and falls back to AUTO. https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2023-0/specifying-code-branches.html . Consequence: **on AMD the only pinning option is COMPATIBLE (SSE2, slow), and my local test (section 6) showed COMPATIBLE still changed bits at 14 threads**, so thread-count invariance for torch-CPU on AMD is not obtainable via CBWR (AMD behaviour itself untested). Corrects Recipe C, which implied `AVX2,STRICT` is portable.
5. **oneDNN's determinism promise is same-platform only.** "multiple executions of the same primitive on a given platform return the exact same result bitwise"; for most primitives this holds only if hardware and software environment are identical; `set_deterministic(true)` only removes atomics. Nothing about thread count or other CPUs. https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_deterministic.html (the Intel-hosted copy 301s to a 404 today).
6. **Integer path is not ISA-independent in practice.** PyTorch issue 59098: a quantized (FBGEMM) LSTM test gave different outputs on Cascade Lake because FBGEMM uses AVX512_VNNI when available; reporter said hardware-dependent, closed by skipping the test. The HF `sentence-transformers/all-MiniLM-L6-v2` repo ships ISA-specific quantized ONNX files (`model_qint8_avx512`, `model_qint8_avx512_vnni`, `model_quint8_avx2`, `model_qint8_arm64`, 23.0 MB each) plus fp32 `model.onnx` (90,405,214 B) and O1-O4 graph-optimized variants; `BAAI/bge-small-en-v1.5` has only fp32 `onnx/model.onnx` (133,093,490 B). Corrects Recipe C's "int8/int32 GEMMs are ISA independent" (AVX2 vs VNNI paths can differ; the saturation reason is my knowledge, unverified). https://github.com/pytorch/pytorch/issues/59098 ; https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main/onnx ; https://huggingface.co/BAAI/bge-small-en-v1.5/tree/main/onnx (sizes from HF tree API, 2026-10-17)
7. PyTorch itself disclaims cross-platform repro: "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms" (section 5 above). `torch.set_num_threads` overrides OMP/MKL env vars: https://docs.pytorch.org/2.3/notes/cpu_threading_torchscript_inference.html

### What guarantees what (best current answer)

| axis | torch-CPU (MKL) | ORT-CPU (MLAS) |
|---|---|---|
| run-to-run, fixed threads, fixed shape, same box | yes (Intel doc conditions) | expected yes; untested |
| thread count | only Intel + `MKL_CBWR=AVX2,STRICT` (local: confirmed for a raw GEMM); none on AMD | expected yes for GEMM (M/N-only partition), untested; no doc guarantee |
| batch composition / row alone vs in batch | no (local numpy+MKL: row differs even with CBWR) | expected yes except M==1 path and sequence-length reductions; untested |
| padding length | not guaranteed; mask makes it mathematically equal, not bitwise | same |
| across CPU models | no claim from PyTorch / Intel / oneDNN | different kernels per ISA class; no claim; no env var to pin ISA |

### Recommendation for the student design

- Do not put "byte-identical embeddings from raw text on arbitrary CPUs" in the contract. Treat the encoder as an approximately deterministic feature extractor and make the replay boundary the **logged feature vector**: store the embedding bytes (or an int8/fixed-point quantization) with each prediction. The LR head in float64 or int64 fixed-point with a fixed summation order is then exactly replayable anywhere (my reasoning, not a sourced claim).
- If re-execution from text must match, pin an `execution_profile`: ORT fp32 `model.onnx` (never the ISA-specific quantized files), fixed shape (pad to a constant length such as 128, fixed batch size B with dummy rows), fixed `intra_op_num_threads`, one process; put CPU ISA class + onnxruntime version + model sha256 in the replay key; run a 16-text canary hash at startup. Prefer ORT over torch for this because MLAS has no vendor library whose strict mode is Intel-only.
- Quantize confidence (e.g. 1/4096 steps) before the certified cutoff and store label + accept/defer as the replayed output, as in the main summary. Near-boundary cases can still flip, so also log a "within 2 quantization steps of cutoff" flag.

### Smoke to run (cheap; local CPU is enough for most of it)

Install `onnxruntime`, `numpy`, and CPU `torch`. For 200 diverse bookkeeping strings x {all-MiniLM-L6-v2, bge-small-en-v1.5} x {ORT fp32, torch fp32}, record sha256 of the float32 bytes and max abs diff / ULP between configs: threads {1,2,4,8,14}; row alone vs in batches of 8/64 with random neighbours and random order; pad-to-longest vs fixed 128/256; for torch `MKL_CBWR` unset / `AVX2,STRICT` / `COMPATIBLE`. Then repeat on a second CPU class (AVX512F Xeon or Zen 4, and an AMD Zen 2/3 or other AVX2-only box) and diff hashes against the local Core Ultra 7 155U (AVX2 only). Pass for the encoder-in-the-contract design: all hashes equal in the pinned fixed-shape arm. Pass for the logged-feature design: only the LR-head hashes must match.

### Unverified in this addendum

- Any actual measurement for torch-CPU or ORT-CPU embeddings; whether AVX2 vs AVX512F ORT kernels give different bits for MiniLM; whether ORT softmax/LayerNorm/pooling are padding-invariant; whether `MKL_ENABLE_INSTRUCTIONS` (Intel-documented, not fetched) can pin the ISA for MKL on AMD; whether current PyTorch x86 wheels route fp32 linear through MKL or oneDNN for these shapes; torch behaviour on AMD under MKL's non-Intel dispatch.
