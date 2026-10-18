# S8 — GPU students (RunPod RTX 4090, $0.34–0.89/h)

Device: NVIDIA GeForce RTX 4090. Model: Qwen/Qwen2.5-0.5B with a LoRA (r=16, q/k/v/o) and a fresh classification head; every label scored in one forward pass; bf16 autocast; bs 32; max_len 64; lr 2e-4 for everything (v1 recipe).

## A. Per-step student (2a) on Banking77 (77 labels), 3 epochs

| rows | steps | train s | samples/s | peak GB | acc | cert@5% coverage | holdout err above cutoff |
|---|---|---|---|---|---|---|---|
| 770 | 75 | 9.4 | 245.1 | 5.65 | 0.371 | 0.00 | - |
| 3080 | 291 | 33.9 | 272.8 | 7.63 | 0.849 | 0.63 | 0.0269 |
| 9993 | 939 | 110.1 | 272.2 | 7.63 | 0.916 | 0.83 | 0.0269 |

Batch-1 inference latency: 33.68 ms (bf16, python loop, no compile). For comparison the CPU frozen-bge-small + LR student (S1) reaches 0.821 / 0.910 / 0.933 at the same row counts, so the v1 LoRA recipe is under-trained at small n (75 steps) and still 1.7 points behind at full data; see v2 below.

## B. Shared workflow student (2b) vs per-step (2a) on a NEW step

Shared LoRA trained once on 8 CLINC150 domains (step name in the prompt, 750 steps, 88.1 s). New step = held-out domain (15 labels, 450 test rows), fine-tuned for 60 steps (~7 s) from the shared LoRA (new head) vs from the base model.

| step | rows (per label) | init | acc | cert@5% coverage | holdout err |
|---|---|---|---|---|---|
| travel | 15 (1) | base | 0.558 | 0.00 | - |
| travel | 15 (1) | shared | 0.553 | 0.00 | - |
| travel | 45 (3) | base | 0.604 | 0.00 | - |
| travel | 45 (3) | shared | 0.836 | 0.60 | 0.0073 |
| travel | 150 (10) | base | 0.967 | 0.92 | 0.0 |
| travel | 150 (10) | shared | 0.953 | 0.89 | 0.0 |
| travel | 450 (30) | base | 0.971 | 0.97 | 0.018 |
| travel | 450 (30) | shared | 0.991 | 1.00 | 0.0089 |
| travel | 1500 (100) | base | 0.998 | 1.00 | 0.0045 |
| travel | 1500 (100) | shared | 0.993 | 1.00 | 0.0045 |
| work | 15 (1) | base | 0.438 | 0.00 | - |
| work | 15 (1) | shared | 0.467 | 0.00 | - |
| work | 45 (3) | base | 0.651 | 0.00 | - |
| work | 45 (3) | shared | 0.780 | 0.00 | - |
| work | 150 (10) | base | 0.907 | 0.71 | 0.0135 |
| work | 150 (10) | shared | 0.918 | 0.69 | 0.0125 |
| work | 450 (30) | base | 0.947 | 0.84 | 0.0266 |
| work | 450 (30) | shared | 0.960 | 0.93 | 0.019 |
| work | 1500 (100) | base | 0.971 | 0.98 | 0.0183 |
| work | 1500 (100) | shared | 0.978 | 0.99 | 0.018 |

Reading: the shared student wins clearly at 3 rows per label (travel 0.84 vs 0.60, work 0.78 vs 0.65) and is the only one that certifies there; by 10 rows per label the two are within noise; neither hurt the other beyond noise at any size. Rows-to-certify a new step at a 5% budget: ~45 with the shared student vs ~150 from base.

## C. Determinism on the GPU student

- 1000 batch-1 forward passes from 8 threads on separate CUDA streams: **1 distinct** logits byte-string.
- Same input as row 0 of batches of random size 1–8 with dynamic padding: **12 distinct**.
- Same, but every batch padded to a fixed length of 64: **8 distinct** — batch size alone changes the kernel path even at fixed sequence length.
- Recipe that holds: batch size one, fixed shape, one SKU (see S7 for the CPU analogue and `research/determinism.md`).

## D. Student ladder (v2 run, same GPU)

Rungs: fine-tuned bge-small encoder (33M) + linear head (lr 5e-5 / head 1e-3, warmup+linear decay); SetFit contrastive fine-tune of bge-small; LoRA r=32 on all linear layers of Qwen2.5-0.5B with separate head LR (3e-4 / 2e-3) and a ≥600-step budget.

| method | rows | steps | train s | acc | cert@5% coverage | holdout err |
|---|---|---|---|---|---|---|
| finetune-bge-small | 770 | 300 | 17.2 | 0.825 | 0.59 | 0.0274 |
| finetune-bge-small | 3080 | 776 | 43.1 | 0.920 | 0.89 | 0.0331 |
| finetune-bge-small | 9993 | 2504 | 130.6 | 0.941 | 0.94 | 0.0288 |
| setfit-bge-small | 770 | - | 51.6 | 0.640 | 0.00 | - |
| setfit-bge-small | 3080 | - | 75.3 | 0.841 | 0.47 | 0.0275 |
| lora-qwen2.5-0.5b-tuned | 770 | 600 | 114.4 | 0.780 | 0.49 | 0.0419 |
| lora-qwen2.5-0.5b-tuned | 3080 | 600 | 114.5 | 0.911 | 0.83 | 0.0212 |
| lora-qwen2.5-0.5b-tuned | 9993 | 1878 | 351.9 | 0.936 | 0.93 | 0.0273 |

## Cost and time

v1 job: ~11 min wall including boot and pip install, $0.16 (secure-cloud 4090 at $0.89/h; the launcher now prefers community cloud at $0.34/h). Training itself: 9 s / 34 s / 110 s for 770 / 3080 / 9993 rows at 272 samples/s, 7.6 GB peak. Extrapolation: a 100k-row step is ~6 min; a 1M-row step ~1 h, i.e. $0.35–0.90.
