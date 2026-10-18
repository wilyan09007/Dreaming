"""Render RESULTS.md from results.json (v1: per-step LoRA, shared-vs-base, determinism) and results_v2.json (student ladder)."""
import json, pathlib
H = pathlib.Path(__file__).resolve().parent; L = []
def say(s=""): L.append(s)
R = json.load(open(H / "results.json"))
say("# S8 — GPU students (RunPod RTX 4090, $0.34–0.89/h)\n")
say(f"Device: {R['gpu']}. Model: {R['model']} with a LoRA (r=16, q/k/v/o) and a fresh classification head; every label scored in one forward pass; bf16 autocast; bs 32; max_len 64; lr 2e-4 for everything (v1 recipe).\n")
say("## A. Per-step student (2a) on Banking77 (77 labels), 3 epochs\n")
say("| rows | steps | train s | samples/s | peak GB | acc | cert@5% coverage | holdout err above cutoff |"); say("|---|---|---|---|---|---|---|---|")
for r in R["A_per_step"]:
    say(f"| {r['rows']} | {r['steps']} | {r['train_s']} | {r['samples_per_s']} | {r['peak_mem_gb']} | {r['acc']:.3f} | {r['cert5_coverage']:.2f} | {r['holdout_err_above'] if r['holdout_err_above'] is not None else '-'} |")
say(f"\nBatch-1 inference latency: {R['batch1_latency_ms']} ms (bf16, python loop, no compile). For comparison the CPU frozen-bge-small + LR student (S1) reaches 0.821 / 0.910 / 0.933 at the same row counts, so the v1 LoRA recipe is under-trained at small n (75 steps) and still 1.7 points behind at full data; see v2 below.\n")
say("## B. Shared workflow student (2b) vs per-step (2a) on a NEW step\n")
s = R["B_shared_train"]
say(f"Shared LoRA trained once on 8 CLINC150 domains (step name in the prompt, {s['steps']} steps, {s['train_s']} s). New step = held-out domain (15 labels, 450 test rows), fine-tuned for 60 steps (~7 s) from the shared LoRA (new head) vs from the base model.\n")
say("| step | rows (per label) | init | acc | cert@5% coverage | holdout err |"); say("|---|---|---|---|---|---|")
for r in R["B_new_step"]:
    say(f"| {r['step']} | {r['rows']} ({r['rows']//15}) | {r['init']} | {r['acc']:.3f} | {r['cert5_coverage']:.2f} | {r['holdout_err_above'] if r['holdout_err_above'] is not None else '-'} |")
say("\nReading: the shared student wins clearly at 3 rows per label (travel 0.84 vs 0.60, work 0.78 vs 0.65) and is the only one that certifies there; by 10 rows per label the two are within noise; neither hurt the other beyond noise at any size. Rows-to-certify a new step at a 5% budget: ~45 with the shared student vs ~150 from base.\n")
say("## C. Determinism on the GPU student\n")
say(f"- 1000 batch-1 forward passes from 8 threads on separate CUDA streams: **{R['det_batch1_concurrent_distinct']} distinct** logits byte-string.")
say(f"- Same input as row 0 of batches of random size 1–8 with dynamic padding: **{R['det_random_batch_distinct']} distinct**.")
say(f"- Same, but every batch padded to a fixed length of 64: **{R['det_random_batch_fixed_shape_distinct']} distinct** — batch size alone changes the kernel path even at fixed sequence length.")
say("- Recipe that holds: batch size one, fixed shape, one SKU (see S7 for the CPU analogue and `research/determinism.md`).\n")
v2 = H / "results_v2.json"
if v2.exists():
    R2 = json.load(open(v2)); say("## D. Student ladder (v2 run, same GPU)\n")
    say("Rungs: fine-tuned bge-small encoder (33M) + linear head (lr 5e-5 / head 1e-3, warmup+linear decay); SetFit contrastive fine-tune of bge-small; LoRA r=32 on all linear layers of Qwen2.5-0.5B with separate head LR (3e-4 / 2e-3) and a ≥600-step budget.\n")
    say("| method | rows | steps | train s | acc | cert@5% coverage | holdout err |"); say("|---|---|---|---|---|---|---|")
    for r in R2["runs"]:
        say(f"| {r['method']} | {r['rows']} | {r.get('steps','-')} | {r['train_s']} | {r['acc']:.3f} | {r['cert5_coverage']:.2f} | {r['holdout_err_above'] if r['holdout_err_above'] is not None else '-'} |")
    if "setfit_error" in R2: say(f"\nSetFit did not run: `{R2['setfit_error'][:200]}`")
say("\n## Cost and time\n")
say("v1 job: ~11 min wall including boot and pip install, $0.16 (secure-cloud 4090 at $0.89/h; the launcher now prefers community cloud at $0.34/h). Training itself: 9 s / 34 s / 110 s for 770 / 3080 / 9993 rows at 272 samples/s, 7.6 GB peak. Extrapolation: a 100k-row step is ~6 min; a 1M-row step ~1 h, i.e. $0.35–0.90.")
(H / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("wrote", H / "RESULTS.md")
