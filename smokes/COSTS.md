cost,usd,note
s2_descriptions,0.0048,label descriptions via openai/gpt-5.4-nano
s3_review_deferral,0.00,CPU only; no API calls
s13_teacher_noise,0.0118,probe: 1 call/model/candidate + 5 calls/model
s13_teacher_noise,0.0115,probe: 5 calls/model
s11_asi_one,0.025,ASI:One asi1-mini planner calls (130k tokens at $0.18/M in); hosted/local agents free tier
s13_teacher_noise,1.4015,full run, 5 reps; sum of usage.cost this invocation
s4_tool_predictor+s9_rollout,0.00,no API calls (CPU only)
s12_wrappers,0.0014,gpt-5.4-nano via OpenRouter; ~10 tiny live calls per T1/T1c run (est. 5 iterations) incl. recertify; T0/T2/T3 offline
s10,0.003,gpt-5.4-nano via local proxy; final run 28 upstream calls = $0.00042; total incl. 4 earlier dev/failed harness runs is an estimate (traces were reset each run)
s8_gpu,0.04,RunPod 4090 aborted run (controller crash on console encoding), ~6 min
s5,1.96,Part B live: 11 cfgs x 200 Banking77 calls on OpenRouter (8 configs + E1b/C1b/E2b repeats) + probes + temp/qwen tests; Part A free (RouterBench 0shot pkl)
s13_teacher_noise,1.1112,extra: sonnet max_tokens=300 rerun + cross-provider pins (C)
s6,0.94,gpt-5.4-nano HotpotQA: 2750 ablation + 450 replica/determinism + 1216 pruned re-asks = 4416 calls (cap $5)
s13_teacher_noise,0.0200,manual probing of reasoning/temperature/provider settings (not tracked per call; upper estimate)
s8_gpu,0.162,RunPod NVIDIA GeForce RTX 4090 11 min
s2b_synth,0.0007,10 synthetic examples per label via gpt-5.4-nano
s2b_synth,0.0010,10 synthetic examples per label via gpt-5.4-nano
s2b_synth,0.0007,10 synthetic examples per label via gpt-5.4-nano
s2b_synth,0.0046,10 synthetic examples per label via gpt-5.4-nano
s12_wrappers,0.0020,revision: final suite measured 0.00041 (T1 + T1c + T2c live rows + recertify re-executions); total incl. ~4 T1 and ~7 T2c dev/failed iterations is an estimate
s8_gpu,0.288,RunPod NVIDIA GeForce RTX 4090 19 min
s6b_pruner_typed,1.4842,claude-haiku-5.5 pinned (Claude Platform on AWS) reasoning off: 9709 calls = 7200 label-phase + ~2500 pruned re-asks + 4 probe/pin; all from the earlier attempt and re-read from cache (0 new calls in finalizing session); sum of usage.cost
s4_tool_predictor+s9_rollout,0.00,revision 2: no API calls (CPU only); S4 novelty-shift tables added, S9 RESULTS rewritten from log.txt
s6,1.99,revision (adds to the $0.94 original; s6 total about $2.93 of the $5 cap): nano replay+closed-book+gold-trained re-asks+judge $0.049; qwen3-32b T=0 teacher+judge $0.431; gemini-3.5-flash-lite T=0 teacher+judge $1.467; haiku-4.5 second-judge audit $0.036; endpoint/cost probes $0.008. Ledger tokens x listed price; checked vs usage.cost (ratio 1.000 on 8 calls/model)
s5_revision,0.1736,run.py evidence: 238 calls (temperature repeat, require_parameters 404, qwen reasoning switches, 30-input interleaved latency across 7 cfgs); Part A and analyze re-runs are CPU/offline; earlier s5 total 1.96 stands, new total ~2.13
s14_tool_traces,0.00,no API calls (CPU only; downloads: tau-bench 51 MB, TraceLab 101 MB, Toucan 2 shards 235 MB, tau2-bench clone 874 MB)
pod_podtest,0.187,RunPod NVIDIA GeForce RTX 4090 13 min ok=True
s15_confidence_signals,0.17,2200 cached cheap-model calls (nano/gpt-4.1-nano/qwen3-next-80b x 11 runs x 200) sum of usage.cost 0.159 + logprob/provider probes (qwen3-32b probe 2 calls) est 0.01; no E1 re-run; part 1 CPU only
s22_arch_s22d,0.019,RunPod NVIDIA GeForce RTX 3090 5 min
s19_token_economics,0.0036,38 max_tokens=1 probes (Claude token counts, cache_control pass-through, OpenAI auto-cache) made in the earlier attempt; 0 new calls in the finalizing session; rest is arithmetic + pricing-page fetches; no pod
pod_s14,0.040,RunPod NVIDIA GeForce RTX 3090 5 min ok=True
s22_arch_s22a,0.006,RunPod NVIDIA GeForce RTX 4090 1 min
pod_s18,0.049,RunPod NVIDIA GeForce RTX 3090 6 min ok=True
s22_arch_s22b,0.006,RunPod NVIDIA GeForce RTX 4090 1 min
s22_arch_s22d,0.006,RunPod NVIDIA GeForce RTX 4090 1 min
pod_s21,0.010,RunPod NVIDIA GeForce RTX 4090 2 min ok=True
pod_s14b,0.023,RunPod NVIDIA RTX A5000 5 min ok=True
s22_arch_s22c,0.075,RunPod NVIDIA GeForce RTX 4090 13 min
s14_tool_traces,0.00,final: no API calls; all compute on RunPod (pod_s14 RTX 3090 $0.040 = first full run; pod_s14b A5000 $0.023 = final run with F ablation); laptop used only for two tiny plumbing checks (<25 s each)
pod_s15,0.070,RunPod NVIDIA RTX A5000 15 min ok=True
s22_arch_s22d,0.012,RunPod NVIDIA GeForce RTX 4090 2 min
pod_s16,0.044,RunPod NVIDIA GeForce RTX 4090 8 min ok=True
pod_s18,0.088,RunPod NVIDIA GeForce RTX 4090 6 min ok=True
pod_s15rep,0.014,RunPod NVIDIA RTX A5000 3 min ok=True
s22_arch_diag,0.018,RunPod NVIDIA GeForce RTX 4090 3 min
s18_audit_certificate,0.00,no API calls (CPU analysis only); pod cost logged separately as pod_s18 (0.049 RTX 3090 run 1 + 0.088 RTX 4090 run 2 = 0.137)
pod_s21b,0.070,RunPod NVIDIA RTX A5000 16 min ok=False
pod_s23a3,0.031,RunPod NVIDIA GeForce RTX 4090 2 min ok=False
pod_s23a4,0.031,RunPod NVIDIA GeForce RTX 4090 2 min ok=False
pod_s23a2,0.009,RunPod NVIDIA RTX A5000 2 min ok=False
pod_s22lite,0.108,RunPod NVIDIA GeForce RTX 4090 7 min ok=False
pod_s17lite,0.139,RunPod NVIDIA GeForce RTX 4090 9 min ok=True
pod_s23lite,0.066,RunPod NVIDIA GeForce RTX 4090 12 min ok=True
pod_s17lite2,0.027,RunPod NVIDIA GeForce RTX 4090 5 min ok=True
s17_warm_start,0.0014,S17-lite: LLM label descriptions for 283 pool labels + synthetic examples for the 14 travel/work labels missing from the S2b cache (ASI:One asi1-mini); pod cost logged separately as pod_s17lite (0.139) + pod_s17lite2 (0.027)
pod_s16b,0.031,RunPod NVIDIA GeForce RTX 4090 5 min ok=True
s16c_judge_floor_offline,0.00,cache-only re-analysis
pod_s21c,0.031,RunPod NVIDIA GeForce RTX 4090 6 min ok=True
pod_s21d,0.017,RunPod NVIDIA RTX A5000 4 min ok=True
pod_s21d_boot_before_attach,0.080,RunPod NVIDIA RTX A5000 ~18 min boot/idle before the launcher was re-attached (launcher process had been killed by a shell timeout; pod kept)
s14b_copilot_traces,0.00,no API calls, no pods; local run of fetch/flatten/run.py (D E F S T X) on 4 CPUs, about 16 min compute
s2_descriptions,0.0047,label descriptions via openai/gpt-5.4-nano
pod_s26,0.468,RunPod NVIDIA A100 80GB PCIe 16 min ok=True
s26_pool_transfer,0.0065,16 ASI:One asi1-mini calls (label descriptions and synthetic examples for S26 pool and eval labels); pod cost logged separately as pod_s26 (0.468)
s27_hybrid_embedder_memory,0.00,CPU only
pod_s25,0.611,RunPod NVIDIA GeForce RTX 4090 41 min ok=False
pod_s25,0.000,laptop re-attach to pod v6afea12hinimh (20 min inside the 41-min window already billed on the row above; double count zeroed)
pod_s25b,0.03,RunPod RTX 4090 ~2 min, terminated (redundant: first S25 run recovered by the user)
s28_router_calibration,1.7096,929 E1 (gpt-5.4 low) + 1000 cheap (gpt-5.4-nano) calls; sum of usage.cost over responses.jsonl (E1 1.6117 + cheap 0.0979); 531 error rows at 0; 71 E1 calls not made (OpenRouter credits exhausted); analysis CPU only
pod_s29,0.056,RunPod NVIDIA GeForce RTX 4090 4 min ok=True
s29_memory_student,0.00,cancelled by the user before results; pod_s29 rows above are boot time only (one launch failed on a datasets column bug, the second was terminated at boot)
