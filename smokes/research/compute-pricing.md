# compute-pricing (researched 2026-10-17)

## Summary

- RunPod pods (page "Updated October 8, 2026"): RTX 4090 is $0.34/hr Community vs $0.89/hr Secure; A5000 $0.16 vs $0.27; A40 $0.35 vs $0.59; L4 $0.44 vs $0.59; A100 80GB PCIe $1.19 vs $1.79; H100 PCIe $1.99 vs $2.89 (SXM $2.69 vs $3.99). Billed per second. $20 buys about 59 h of community 4090 or 22 h of secure 4090.
- PLAN CHANGE: RunPod REST API v1 (`rest.runpod.io/v1`) retires Nov 15, 2026, and the GraphQL API retires "early 2027". Both are about one month away or less. Write launcher code against REST v2 (`https://api.runpod.io/v2/pods`). v2 field names differ from v1.
- One-shot pod with no SSH works: set the start command via `cmd`/`args` (v2) or `dockerStartCmd` (v1) plus env vars. Every pod gets `RUNPOD_POD_ID` and a pod-scoped `RUNPOD_API_KEY`, so the script can `DELETE /v2/pods/{id}` itself at the end. Docs never confirm this self-delete or that the container exits and stops billing on its own, so test on a cheap pod first (the pod-scoped key and the pod's own `RUNPOD_POD_ID` are documented; the delete-from-inside call is not).
- Simplest alternative for one-shot GPU scripts: Hugging Face Jobs (`hf jobs run` / `run_job`). It has no SSH and no teardown, bills per minute only while Starting/Running, and has a 30 min default timeout. T4 $0.40/hr, L4 $0.80, A10G $1.00, A100 80GB $2.50. It needs a positive paid credit balance (no free tier).
- Modal Starter is free with $30/month compute credit. L4 is $0.000222/s (about $0.80/hr), A100 80GB $0.000694/s (about $2.50/hr). $30 buys about 37 h of L4 or about 12 h of A100 80GB. Startup credits (amount unstated) and up to $10k academic credits exist.
- Vast.ai and Lambda: Lambda 1x H100 PCIe $3.29/hr, A10 $1.29/hr, which is more expensive than RunPod and has no small cards. Vast.ai live prices could not be fetched (JS-rendered). Search-snippet medians put 4090 near $0.40/hr, unverified.
- Colab: no official compute-unit (CU) table. Third-party figures put Pro at $9.99/mo for 100 CU, T4 about 1.76 CU/hr and A100 about 15 CU/hr. H100 reported unavailable (GitHub issue, Aug 2026). Not suitable for scripted runs.
- Training time, my arithmetic and not benchmarked: LoRA on Qwen2.5-0.5B, 10k short rows, 3 epochs is about 3-6 min of GPU compute on a 4090 and costs about $0.05-0.15 including boot. Fine-tuning a 110M encoder on this laptop's CPU is about 0.75-2 h per epoch at 64 tokens/row, so use a GPU for any training; use CPU only for tiny models or frozen-embedding extraction.
- The $100 budget is not at risk from compute. The whole smoke campaign (dozens of short GPU runs) fits in a few dollars of RunPod or HF Jobs. The binding constraint is the OpenRouter credit for frontier-model calls, not GPU.

## Findings

### RunPod pod pricing (on-demand, per hour)
Source: https://www.runpod.io/pricing (visible "Updated October 8, 2026"; JSON-LD `dateModified` 2026-10-08).
The visible table shows Secure prices only. The Community prices come from the page's JSON-LD (structured data) block. Page metadata says "H100 80GB from $1.99/hr, RTX 4090 from $0.34/hr", which matches the community values.

| GPU | VRAM | Community | Secure |
|---|---|---|---|
| RTX 4090 | 24 GB | $0.34 | $0.89 |
| RTX A5000 | 24 GB | $0.16 | $0.27 |
| A40 | 48 GB | $0.35 | $0.59 |
| L4 | 24 GB | $0.44 | $0.59 |
| A100 80GB PCIe | 80 GB | $1.19 | $1.79 |
| A100 80GB SXM | 80 GB | $1.39 | $1.79 |
| H100 PCIe | 80 GB | $1.99 | $2.89 |
| H100 SXM | 80 GB | $2.69 | $3.99 |
| H100 NVL | 94 GB (unverified) | $2.59 | $3.19 |
| RTX 3090 / L40S / H200 | - | $0.22 / $0.79 / $3.59 | $0.50 / $1.09 / $5.29 |

VRAM and GPU ID strings: https://docs.runpod.io/references/gpu-types. IDs are e.g. `NVIDIA GeForce RTX 4090`, `NVIDIA RTX A5000`, `NVIDIA A40`, `NVIDIA L4`, `NVIDIA A100 80GB PCIe`, `NVIDIA A100-SXM4-80GB`, `NVIDIA H100 PCIe`, `NVIDIA H100 80GB HBM3`.

Billing and storage: https://docs.runpod.io/pods/pricing
- Compute and storage are billed per second. There are no ingress or egress fees.
- You need at least 1 hour of credit for your configuration to deploy.
- Container disk is $0.10/GB/month while running and free when stopped. Volume disk is $0.10/GB/month running and $0.20/GB/month stopped. Network volume is $0.07/GB/month (under 1 TB).
- Default spend limit is $80/hr. At a $0 balance, pods without a network volume are terminated and their data is unrecoverable.
- Cloud tiers: https://docs.runpod.io/pods/choose-a-pod says Secure runs in T3/T4 data centers with high reliability, and Community is peer-to-peer with variable reliability. "Runpod is no longer accepting new hosts for Community Cloud, but existing Community Cloud resources remain available." Expect Community supply to shrink, so keep a Secure fallback.

### RunPod serverless pricing
Source: https://www.runpod.io/pricing (2026-10-08). The page lists per-hour rates only, with no Flex/Active labels and no per-second figures. Per-second is my division by 3600:

| GPU | $/hr | $/s |
|---|---|---|
| RTX 4090 | 1.10 | 0.000306 |
| L4, A5000, 3090 (24 GB class) | 0.69 | 0.000192 |
| A6000, A40 (48 GB class) | 1.22 | 0.000339 |
| A100 80GB | 2.72 | 0.000756 |
| H100 80GB | 4.79 | 0.001331 |

- https://docs.runpod.io/serverless/pricing: billing runs "from when a worker starts until it fully stops, rounded up to the nearest second". The default idle timeout is 5 s. Image pulls while scaled down are not billed. Active workers get "discounts available through sales inquiry" (no number given).
- Serverless is 2-3x the pod rate, so use it only for the hosted-predictor demo (scale to zero), not for training.
- Changelog (https://docs.runpod.io/changelog): Batch Jobs for Serverless (Beta, Aug 19, 2026); global volumes on Serverless (Beta, Oct 8, 2026); Runpod SDK 2.0 "Flash" (Oct 15, 2026, queue endpoints and tasks on their own Pod; not read in detail).

### RunPod API: create / terminate a pod
Status (https://docs.runpod.io/changelog, Aug 18, 2026; https://docs.runpod.io/api-reference-v2/migrate-from-v1):
- REST API v2 is GA at `https://api.runpod.io/v2`.
- REST v1 retires Nov 15, 2026 (migration guide). A search-summary snippet said Dec 1, which I could not confirm; treat Nov 15 as the deadline.
- GraphQL retires "early 2027" (changelog only).

REST v2, from https://docs.runpod.io/api-reference-v2/pods/create-a-pod.md:
- `POST https://api.runpod.io/v2/pods` with header `Authorization: Bearer <api_key>`. Body limit is 102,400 bytes.
- Fields: `name` (required), `image` (required unless `templateId`), `cmd` (array) or `args` (string), `entrypoint` (array), `env` (object), `gpu: {id, count}`, `cloud` (`SECURE` default | `COMMUNITY`), `disk` (container GB, min 1), `mounts.persistent {size (min 10), path}`, `mounts.network`, `templateId` (body overrides template fields; `env` merges per key), `dataCenterIds`, `ports`, `startSsh`, `startJupyter`.
- The page shows no `interruptible` field in v2.
- Returns 201 with a pod object. Status goes PROVISIONING, STARTING, RUNNING, so poll.
- 400 can mean no capacity for that GPU and data center.
- Terminate: `DELETE https://api.runpod.io/v2/pods/{id}`, 204 on success (https://docs.runpod.io/api-reference-v2/pods/terminate-a-pod.md). Irreversible. A network volume is detached, not deleted.

REST v1 (retiring), from https://rest.runpod.io/v1/openapi.json:
- `POST https://rest.runpod.io/v1/pods` with fields `imageName`, `dockerStartCmd[]`, `dockerEntrypoint[]`, `env{}`, `gpuTypeIds[]`, `gpuCount`, `cloudType` (SECURE | COMMUNITY), `containerDiskInGb` (default 50), `volumeInGb` (default 20), `volumeMountPath` (default /workspace), `templateId`, `interruptible`, `ports`, `networkVolumeId`.
- `DELETE /pods/{podId}` returns 204.

GraphQL (retiring early 2027), from https://docs.runpod.io/sdks/graphql/manage-pods:
- `podFindAndDeployOnDemand(input: {cloudType, gpuCount, gpuTypeId, volumeInGb, containerDiskInGb, volumeMountPath, minVcpuCount, minMemoryInGb, name, imageName, dockerArgs, ports, env:[{key,value}], allowedCudaVersions})` returns `id imageName env machineId machine{podHostId}`.
- `dockerArgs` overrides the image CMD (https://docs.runpod.io/sdks/graphql/configurations).
- `templateId` on this mutation is NOT confirmed. The doc examples never pass it. `saveTemplate` creates templates.
- Stop: `podStop(input:{podId})`. Terminate is `podTerminate` (not shown in the fetched docs page, from memory; unverified).

### One-shot pod without SSH
- Override the start command with `cmd`/`args` (v2) or `dockerStartCmd` (v1). Docs: https://docs.runpod.io/pods/templates/overview lists "startup commands" but does not describe exit behavior.
- https://docs.runpod.io/pods/references/environment-variables says every pod has `RUNPOD_POD_ID` and `RUNPOD_API_KEY` ("Pod-scoped API key"). The page says nothing about whether that key may delete the pod itself.
- https://docs.runpod.io/pods/manage-pods shows `runpodctl pod stop $RUNPOD_POD_ID` in a delayed-stop example. https://docs.runpod.io/runpodctl/reference/runpodctl-remove-pod documents `runpodctl pod delete <id>`. Neither says whether it works from inside the pod.
- Stop keeps volume disk, which is billed ($0.20/GB/month stopped), and clears the container disk. Terminate removes everything. A pod with a network volume cannot be stopped, only terminated.
- Not confirmed in any doc: whether a pod goes EXITED (and stops compute billing) when its command finishes.
- Safe pattern, to test first: the script ends with `curl -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" https://api.runpod.io/v2/pods/$RUNPOD_POD_ID`. Also poll `GET /v2/pods/{id}` from the laptop and DELETE after results are uploaded (HF Hub or S3 upload from the pod). Set a laptop-side watchdog with a max wall-clock time.

### runpod/pytorch images
- Docker Hub API (https://hub.docker.com/v2/repositories/runpod/pytorch/tags, checked 2026-10-17) lists 4,929 tags. The newest are rc builds from 2026-10-16, e.g. `1.4.1-rc.175-cu1290-torch291-ubuntu2404` (15.5 GB), plus torch260/271/280/290/2130 variants, ubuntu2204 or 2404.
- Non-rc tags are `1.0.7-dev` / `1.0.9-dev` builds from July 2026, e.g. `1.0.9-dev-feat-TEM-11-container-hardening-cu1281-torch291-ubuntu2404` (11.9 GB). I could not identify a clean "stable" tag, and the old `2.x.x-py3.x-cuda...` tag scheme appears gone. Unverified.
- Recommendation: pin an exact tag you have smoke-tested, or use `pytorch/pytorch:<ver>-cuda<ver>-cudnn<ver>-devel` from Docker Hub. Images are about 12-16 GB, so pull time is a real part of cold start.

### Alternatives
- Hugging Face Jobs (pricing https://huggingface.co/docs/hub/jobs-pricing; usage https://huggingface.co/docs/huggingface_hub/guides/jobs; both live 2026-10-17):
  - Prices per hour: cpu-basic $0.01, cpu-upgrade $0.03, t4-small $0.40, t4-medium $0.60, l4x1 $0.80, l40sx1 $1.80, a10g-small $1.00, a10g-large $1.50, a100-large (80 GB) $2.50, h200 $5.00, rtx-pro-6000 $2.75. No RTX 4090 or H100.
  - Billed per minute, only while Starting/Running. A failing job is auto-suspended. Default timeout is 30 min, so always pass `timeout`.
  - Needs a positive credit balance; PRO/Team monthly credits also count.
  - Usage: `hf jobs run --flavor a10g-small --timeout 1h <image> <cmd>`, `run_job(image, command, flavor, timeout, env, secrets, volumes)`. `hf jobs uv run script.py` runs a PEP 723 script. `Volume(type="bucket"|"dataset"|"model")` mounts HF repos or buckets, so outputs can be written straight to a bucket.
  - Env vars in the container: `JOB_ID`, `ACCELERATOR`, `CPU_CORES`, `MEMORY`.
- HF GPU Spaces (https://huggingface.co/docs/hub/spaces-gpus): the same price table as Jobs, per-minute billing, H100 removed Dec 2025. Spaces are for serving a demo, not for batch training.
- Modal (https://modal.com/pricing, 2026-10-17):
  - GPU $/s: T4 0.000164, L4 0.000222, A10 0.000306, L40S 0.000542, A100 40GB 0.000583, A100 80GB 0.000694, H100 0.001097. Hourly figures are my x3600: L4 $0.80, A10 $1.10, A100-80 $2.50, H100 $3.95.
  - CPU $0.0000131/core/s, memory $0.00000222/GiB/s.
  - Starter plan: $30/month free credits. Team: $100/month credits. Startup credits (amount unspecified, apply at /startups) and academic credits up to $10k (/academics).
  - Secondary sources (https://www.beam.cloud/blog/modal-pricing-explained, costbench) mention multipliers for region pinning (1.5-1.75x) and non-preemptible (3x), not verified on Modal's page.
- Lambda (https://lambda.ai/pricing, no date on page): 1x H100 SXM $4.29, H100 PCIe $3.29, GH200 $2.29, A100 40GB $1.99, A10 24GB $1.29, A6000 $1.09 per GPU-hr. No free credits shown. No cheap small cards.
- Vast.ai (https://docs.vast.ai/api-reference/instances/create-instance): `PUT /api/v0/asks/{offer_id}/` with `image` (required), `disk`, `runtype` (`ssh` default unless `args` is given), `onstart` (4048-char cap; gzip+base64 for longer), `env` as a string of docker flags, `args`/`args_str`. Response key is `new_contract`. Destroy is `DELETE /api/v0/instances/{id}`. Interruptible instances are billed at the bid `price`, claimed "50%+ cheaper" (marketing). No in-container self-destroy documented. Live prices are JS-rendered and were not retrievable. Search snippets (https://vast.ai/pricing/gpu/A100-SXM4, https://vast.ai/oldpricing, undated) suggested medians of 4090 about $0.39-0.40, A100 SXM4 $1.27, H100 SXM $1.67-2.14. Unverified.
- Google Colab: signup page (https://colab.research.google.com/signup) is not machine-readable. Third-party (https://thundercompute.com/blog/colab-alternatives-for-cheap-deep-learning-in-2025, 2025 data): Pro $9.99/mo with 100 CU, Pro+ $49.99/mo (500 or 600 CU, sources disagree), T4 about 1.76 CU/hr, A100 about 15 CU/hr. Google does not guarantee a GPU type. Not scriptable, so skip for the pipeline.

### Training time and cost (my estimates, NOT measured on GPUs)
No public benchmark was found for Qwen2.5-0.5B LoRA on a 4090. The closest hit, a Qwen2.5-0.5B-Instruct SFT run showing 324 s for "9,308 samples" at 3.7 samples/s (https://huggingface.co/YangZhoumill/Qwen2.5-0.5B-Instruct/commit/b9b59cc0f7bec30d19e9b32445652b86c292fb9a), is internally inconsistent (3.7 samples/s x 324 s is about 1,200 samples), so I did not use it. A TokenSeek paper (https://arxiv.org/pdf/2601.19739) reports 0.43 A100-40GB GPU-hours for a full-token Qwen2.5-0.5B tuning baseline, but the dataset isn't specified.

Derivation (all assumptions explicit):
- LoRA 0.5B: Qwen2.5-0.5B has about 494M params. LoRA costs about 4 FLOPs/param/token (forward 2 + backward-to-activations 2, no weight grads), roughly 2e9 FLOPs/token. 10k rows x 128 tokens = 1.28M tokens/epoch, so about 2.5e15 FLOPs/epoch.
- On a 4090, assume 25-40 TFLOPS effective (15-25% of the roughly 165 TFLOPS bf16 dense peak, which is from my memory of the spec and unverified). That gives about 60-100 s/epoch, so 3 epochs is about 3-5 min of compute.
- Add 2-6 min for image pull, model download and load. Total pod time of 6-12 min is about $0.04-0.07 on Community ($0.34/hr) or $0.09-0.18 on Secure ($0.89/hr).
- On HF Jobs t4-small (no bf16, assume 5x slower), about 10-15 min of training for 3 epochs, so roughly $0.10 at $0.40/hr.
- Encoder on CPU (110M, BERT-base class, about 86M non-embedding params): 6 x 86M x 640k tokens (10k rows x 64) is about 3.3e14 FLOPs/epoch.
  - I measured this laptop: Intel Core Ultra 7 155U, 12 cores/14 threads, 31 GB RAM, Windows 11, numpy 2.2.5 fp32 matmul. Results ranged 8-122 GFLOPS across sizes and runs (noisy, possibly background load or power mode). No torch is installed here.
  - At 50-120 GFLOPS effective that is about 46 min to 1.8 h per epoch at 64 tokens/row, doubling at 128. 3 epochs is about 2.3-5.5 h. Real PyTorch CPU training usually achieves less than GEMM peak, so treat this as optimistic.
  - Forward-only embedding extraction (frozen encoder, 2 FLOPs/param/token) is about 3x cheaper: roughly 15-40 min for 10k rows.
  - A 22M MiniLM-L6-class model is about 8x cheaper than BERT-base.
  - The same 110M job on a 4090 is under 1 min/epoch. Rent a GPU for anything beyond a throwaway CPU smoke.

## Reusable components

| name | url | license | why |
|---|---|---|---|
| RunPod REST API v2 | https://docs.runpod.io/api-reference-v2/pods/create-a-pod | proprietary service | Supported create/terminate path; `cmd`/`args`, `env`, `templateId`, `cloud` |
| runpod Python SDK (2.0, "Flash") | https://docs.runpod.io/changelog | unverified (pip `runpod`) | Official client; SDK 2.0 announced Oct 15, 2026. Check it targets API v2 before relying on it |
| runpodctl | https://docs.runpod.io/runpodctl/reference/runpodctl-remove-pod | unverified | `runpodctl pod delete`/`stop`; whether it is in the image and works with the pod key is unconfirmed |
| Hugging Face Jobs (`hf jobs`, `huggingface_hub.run_job`) | https://huggingface.co/docs/huggingface_hub/guides/jobs | huggingface_hub is Apache-2.0 (from memory) | Simplest one-shot GPU runs: image + command + flavor + timeout; no teardown; bucket/dataset volumes |
| TRL `sft.py` via `hf jobs uv run` | https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py | Apache-2.0 (from memory) | Ready LoRA/SFT script; the HF guide shows running it directly on a10g-small |
| Modal | https://modal.com/pricing | proprietary service | $30/month free credit; Python-native functions with GPU per second; good for bursts |
| Vast.ai API/CLI | https://docs.vast.ai/api-reference/instances/create-instance | proprietary service | Cheapest marketplace; `runtype=args` plus `onstart`; self-destroy not documented |
| `pytorch/pytorch` or `runpod/pytorch` images | https://hub.docker.com/r/runpod/pytorch | unverified | Base image for pods; pin exact tag, 12-16 GB each |
| uv / PEP 723 inline-dependency scripts | https://docs.astral.sh/uv/guides/scripts/ | MIT/Apache-2.0 (from memory) | Makes scripts self-contained for `hf jobs uv run` and pod start commands |

## Numbers (all USD, fetched 2026-10-17 unless noted)

| item | value | source (date) |
|---|---|---|
| RunPod 4090 Community / Secure | $0.34 / $0.89 per hr | https://www.runpod.io/pricing (2026-10-08) |
| RunPod A5000 / A40 / L4 (Community) | $0.16 / $0.35 / $0.44 | same |
| RunPod A5000 / A40 / L4 (Secure) | $0.27 / $0.59 / $0.59 | same |
| RunPod A100 80GB PCIe (Community / Secure) | $1.19 / $1.79 | same |
| RunPod A100 80GB SXM (Community / Secure) | $1.39 / $1.79 | same |
| RunPod H100 PCIe (Community / Secure) | $1.99 / $2.89 | same |
| RunPod H100 SXM (Community / Secure) | $2.69 / $3.99 | same |
| RunPod serverless 4090 / A100 / H100 | $1.10 / $2.72 / $4.79 per hr | same |
| RunPod container disk / volume running / volume stopped / network vol | $0.10 / $0.10 / $0.20 / $0.07 per GB-month | https://docs.runpod.io/pods/pricing |
| RunPod default spend cap; deploy needs | $80/hr; 1 h of credit | same |
| RunPod image sizes | 11.9-17.5 GB per tag | Docker Hub API (2026-10-16/07-20) |
| HF Jobs t4-small / l4x1 / a10g-small / a100-large / h200 | $0.40 / $0.80 / $1.00 / $2.50 / $5.00 per hr | https://huggingface.co/docs/hub/jobs-pricing |
| HF Jobs billing; default timeout | per minute; 30 min | same |
| Modal L4 / A10 / A100-80 / H100 | $0.000222 / 0.000306 / 0.000694 / 0.001097 per s | https://modal.com/pricing |
| Modal free credit | $30/month Starter, $100/month Team | same |
| Lambda H100 SXM / PCIe / A10 | $4.29 / $3.29 / $1.29 per hr | https://lambda.ai/pricing (undated) |
| Colab Pro | $9.99/mo, 100 CU (third-party) | https://thundercompute.com/blog/colab-alternatives-for-cheap-deep-learning-in-2025 (2025) |
| $20 RunPod credit buys | about 59 h Community 4090, about 22 h Secure 4090, about 17 h Community A100 80GB | my arithmetic |
| LoRA 0.5B, 10k rows x 3 epochs | about 3-5 min compute; about $0.05-0.18 all-in | my estimate |
| 110M encoder, 10k rows, this laptop CPU | about 0.75-1.8 h per epoch at 64 tok/row | my estimate; measured GEMM 8-122 GFLOPS |
| Local machine | Core Ultra 7 155U, 12C/14T, 31 GB, no GPU, no torch installed | measured 2026-10-17 |

## Blockers / plan changes

1. RunPod REST v1 retires Nov 15, 2026 and GraphQL retires early 2027 (https://docs.runpod.io/changelog, Aug 18, 2026). Do not build the launcher on GraphQL `podFindAndDeployOnDemand` or v1. Use REST v2 (`https://api.runpod.io/v2/pods`) with `image`, `cmd`/`args`, `env`, `gpu.id`, `cloud`, `disk`, `mounts`. If the brief names v1 or GraphQL fields (`dockerArgs`, `volumeInGb`, `podFindAndDeployOnDemand`), translate them. Snippet in v2 body for the 4090: `"gpu": {"id": "NVIDIA GeForce RTX 4090", "count": 1}`.
2. Pod self-termination is unproven. Do not run an unattended multi-hour job until you have verified, on a $0.16/hr A5000 pod, that (a) the container exits or the script can DELETE its own pod using the injected key, and (b) billing stops. Add a laptop-side watchdog that polls and DELETEs after a max wall time, plus a RunPod spend limit.
3. Community Cloud is closed to new hosts, so availability can vanish for a given GPU. 400 "no capacity" is a documented error. Plan a Secure fallback (2.6x price on the 4090) and a second provider (HF Jobs l4x1/a10g-small).
4. HF Jobs requires a funded HF account (credit balance); there is no free tier. Add $5-10 there if you want it as the simple path. Jobs has no 4090/H100; the cheapest GPU is a T4 at $0.40/hr, and no bf16 on T4.
5. The Hosted-predictor / byte-for-byte replay requirement matters for compute choice. Different GPU models and driver stacks can give different floating-point results. Pin one GPU class (or CPU) for the replay-certified predictor and record the hardware, `torch`, and `cuda` versions in the manifest. Not a compute-pricing finding, but it affects which instance types are usable.
6. CPU-only fine-tuning of a 110M encoder is impractical on this laptop (about 1-2 h per epoch). Any encoder training smoke should run on a rented GPU, or use a tiny model (MiniLM-L6 class, 22M) on CPU.
7. Image pulls are 12-16 GB, so cold start can be minutes. Prefer a small custom image layered on a pinned tag, or install via `uv` at start, and keep pods short. Use a network volume ($0.07/GB-month) only if you reuse model weights across many runs; it also blocks stop (terminate only).

## Unverified

- Vast.ai live prices (page is JS-rendered). The 4090 about $0.40, A100 SXM4 $1.27, H100 SXM $1.67 to $2.14 values come from search-engine snippets of undated pages. Vast's one-shot behavior (`runtype=args`, container exit) and self-destroy were not confirmed in docs.
- Whether `RUNPOD_API_KEY` in a pod is allowed to DELETE the pod itself, whether `runpodctl` is preinstalled, and whether a pod stops billing compute when its command exits.
- Whether v1/v2 REST or GraphQL accepts `templateId` together with a custom start command in practice. v1 and v2 specs list `templateId`; GraphQL `podFindAndDeployOnDemand` docs do not.
- The REST v1 retirement date. The migration guide and changelog say Nov 15, 2026; one search-summary snippet mentioned Dec 1. The GraphQL early-2027 retirement comes only from the changelog line.
- A stable (non-rc) `runpod/pytorch` tag; only rc and `-dev` tags were visible. The 4090 bf16 peak (about 165 TFLOPS), effective-utilization assumptions, and token counts per row behind the training-time estimates.
- Colab compute-unit rates (third-party, 2025) and Pro+ allocation (500 vs 600 CU). Modal region/preemptible multipliers.
- H100 NVL VRAM (94 GB) is from memory. The Runpod `Flash` / SDK 2.0 details were not read.
- Kaggle was not researched (this topic's scope); it requires login.

## Addendum: OpenRouter `:batch` ids and the Batch API (gap 4, 2026-10-17)

Verdict on the earlier "unverified" semantics: CORRECTED. `:batch` ids are not a synchronous endpoint with delayed completion. They are catalog entries for a separate asynchronous Batch API. You submit the plain model slug to `POST /api/v1/batches`; there is no JSONL file upload.

- **How it is called** (official quickstart, fetched 2026-10-17, https://openrouter.ai/docs/batch-quickstart): `POST https://openrouter.ai/api/v1/batches` with `endpoint` (`/v1/chat/completions`, `/v1/responses`, `/v1/messages` or `/v1/embeddings`), `model` (plain slug, every example is `openai/gpt-4o`), and `requests: [{custom_id, body}]`. Optional `provider: {only: [...]}` and `completion_window` (only accepted value is `"24h"`). `endpoint`, `model`, `provider`, `completion_window` must come before `requests` or you get 400. 202 means queued, status `validating`; poll `GET /api/v1/batches/:id` through `in_progress` and `finalizing` to a terminal state (`completed`, `failed`, `expired`, `cancelled`). Results come back inline per `custom_id`; one bad row does not fail the batch. Also `GET /api/v1/batches` (list) and `DELETE` (terminal batches only). Announcement: https://openrouter.ai/blog/announcements/batch-api (post dated 2026-09-22; summarized at https://sessionwatcher.com/news/openrouter-batch-api-half-price-inference). A model with no `:batch` endpoint returns 400. Whether a sync `/chat/completions` call with `model: "x:batch"` works is unverified (no docs show it; I did not spend credit testing). The model page for `anthropic/claude-haiku-5.5:batch` documents only the plain slug at https://openrouter.ai/anthropic/claude-haiku-5.5:batch.
- **Turnaround** (same announcement, two-week beta, 230k+ batches): the window is 24 h and the provider chooses when to finish. Median 7 min, p90 1.0 h, p99 10.3 h. Single-request batches 5-11 min; batches of 1,000+ requests 12-21 min. Submitted 5am-noon Pacific is slower (p90 2-4.5 h); after 6pm Pacific p90 is under 50 min. These are observed beta timings, not guarantees. Plan for overnight or evening submissions and a 24 h worst case.
- **Price**: "typically billed at 50% of the model's standard per-token pricing"; `usage.cost` is the charged amount; web search bills at the standard rate; prompt-cache rates vary by model (quickstart above). Live `/api/v1/models` on 2026-10-17 (https://openrouter.ai/api/v1/models, 470 models): 78 `:batch` ids, not about 100 (39 openai, 15 anthropic, 12 google, 6 mistralai, 2 z-ai, 1 each deepseek, moonshotai, qwen, x-ai). 69 of the 78 are exactly 0.5x their base id. The other 9 are NOT half price against the base id: e.g. `deepseek/deepseek-v4.1-flash:batch` 0.05/0.15 vs base 0.15/0.60; `google/gemini-3.8-flash:batch` 0.375/1.875 vs base 0.825/4.125; `moonshotai/kimi-k3:batch` 2.28/11.40 vs base 0.304/11.94 (batch costlier on input); `z-ai/glm-5.3-flash:batch` and `openai/gpt-oss-20b:batch` also cost more than base on input. Read each `:batch` entry, do not assume half. Confirmed: `anthropic/claude-haiku-5.5` 0.10/0.50 -> `:batch` 0.05/0.25; `claude-sonnet-5.5` 2.00/10.00 -> 1.00/5.00. Haiku 5.5 `:batch` has a tier: prompts of 100k tokens or more bill 0.25/1.25 (irrelevant for short bookkeeping rows).
- **Tools / structured output**: the live `:batch` entries (all 78) advertise `tools`, `tool_choice`, `response_format` and `structured_outputs` in `supported_parameters`; none advertises `stream`. The quickstart says `stream: true`, `speed`, a max output token cap below 1, and Anthropic beta-gated features are rejected; audio and video input, the OpenRouter web-search plugin, router-managed `deferred_tools` and `:online` variants are unavailable; images/files must be public URLs; provider prefs other than `provider.only` / `allow_fallbacks` are rejected; unknown parameters are dropped silently. Docs say nothing explicit about generic function calling. The only structured-output rule: on Google models every request in a batch must use the same `response_format` (all omitted, all `json_object`, or all `json_schema` with the same schema). Whether Anthropic-routed batches honor `tools` / `json_schema` is advertised but untested; smoke it with 5 rows first.
- **Limits**: the docs state no max requests per batch, no file-size cap, no rate limit and no submit-time credit hold (only video inputs reserve balance). Each batch runs on one provider (cheapest batch endpoint by default). Inputs and results are kept 30 days (410 after that). BYOK batches bill OpenRouter only the BYOK fee. Whether the 5.5% credit-purchase fee still applies per batch is unverified; assume yes.
- **Conflicting third-party claim**: https://zentor.ai/blog/no-endpoints-found-for-openai-batch (redirected from moclaw.ai) says OpenRouter removed 35 OpenAI `:batch` ids on 2026-08-26 (422 -> 380 models), leaving 24 `:batch` ids, that a sync call to a stripped id returns 404 "No endpoints found", and that the Batch API path is `/api/beta/batches`. The live list shows 39 openai `:batch` ids, and the official docs use `/api/v1/batches`, so that post is stale or wrong in part. Use the official docs. The same post says `openai/flex` is a synchronous half-price route for the GPT-5.6 family; unverified, single source.
- **`:free` lever** (https://openrouter.ai/docs/api/reference/limits): 20 requests/min; 1,000 requests/day once total credits purchased reach 9 (50/day below that). A $20 top-up clears it. 16 `:free` ids live on 2026-10-17, including `nvidia/nemotron-3-super-120b-a12b:free`, `nvidia/nemotron-3-ultra-550b-a55b:free`, `google/gemma-4-31b-it:free`, `google/gemma-4-26b-a4b-it:free`, `thinkingmachines/inkling:free`. The limits page says nothing about prompt logging or training on free-tier data (unverified; assume prompts may be logged and use synthetic data only). Free models are not the frontier model being shadowed, so use them only for dev smokes, harness tests and cheap second-opinion labelers, not for certification rows.

Plan impact: for offline teacher labeling (shadow, calibration, leave-one-out rows) use the Batch API, which halves the per-row cost and lifts the row budget about 2x for the same credit: 20,000 rows of Haiku 5.5 labeling was about $0.38 (bookkeeping-data.md) and is about $0.19 batched; Sonnet 5.5 about $7.6 becomes about $3.8. Do not use it for the live logged agent episodes or anything needing streaming or interactive latency. A 24 h worst case plus 5am-noon Pacific slowness means that the smoke harness must be resumable and poll-based. Open items: sync `:batch` call behavior; tool-call / `json_schema` fidelity on Anthropic batch routes; whether the 5.5% fee applies; `:free` data policy.
