# Day-zero student: schema -> classifier over frozen bge-small (research notes, 2026-10-17)

Setup being solved: given label names + one-line descriptions (+ optional k examples), get a 384-d-embedding classifier that works on unseen tasks.
Our reference points: hypernetwork 96% seen / 49% held-out CLINC domains / 27% B77; label-description cosine 71% / 66%; 10-shot LR 94% / 87%.

## Summary

- BEST BET: stop trying to out-learn cosine at k=0 and make the head a text-anchored linear probe. Initialise W at the label-description prototypes and fit only a small, penalised deviation from them as examples arrive (CLAP / LP++ recipe, CVPR 2024, vision-only so far). k=0 reproduces the cosine baseline exactly; k>0 degrades gracefully; there is nothing meta-trained to overfit. I found NO published text/intent version of this, so it is a cheap experiment, not a known win.
- The k=0 ceiling for off-the-shelf models on Banking77 is about 0.64-0.71 macro-F1 (BTZSC, Mar 2026: bge-base .64, bge-large .68, Qwen3-Embedding-0.6B .64, 8B .71, Qwen3-Reranker-8B .69). Our 66% bge-small cosine is already AT that plateau, so a stronger zero-shot encoder buys at most about +5 points.
- Skip GLiClass and NLI cross-encoders for fine-grained intents. Their B77 zero-shot F1 is .03-.57 (GLiClass-large .557; deberta-v3-large-zeroshot-v2.0 .567; most NLI models .1-.35), below plain embedding cosine, and they cost one forward pass per label.
- LLM-synthesised examples from descriptions alone roughly tie our baseline, not beat it: Generate-then-Refine on 5 unseen CLINC150 domains (75 intents) gets 70-77% synthetic-only vs 95.4% with real data. Synthetic data pays only mixed with a few real shots (Li et al. 2023/2024) or when a refiner trained on seen domains rewrites it (the one learned component with evidence: +3-7 points).
- Hypernetwork honesty: Text-to-LoRA needed 64-479 meta-training TASKS for modest zero-shot gains, and its few-task (reconstruction) training did not transfer (61.8 vs 66.3). We trained on 7 domains. No evidence found that a text->weights generator beats label-embedding + prototype baselines on unseen tasks; vision evidence says simple baselines match meta-learning under domain shift.
- Real-but-modest learned component: contrastive intent-aware encoder pretraining (PIE, EMNLP 2023) gained +2.5 (B77) to +6.8 (CLINC) zero-shot over paraphrase-mpnet label-name matching. That supports a learned metric projection, with small expectations.
- The cheap upgrade at k=0 is better label text, not better weights: description quality at test time moves accuracy more than anything done at train time (SIGDIAL 2024: +5-6 points from good test descriptions); LLM-refined definitions as prototypes gave +6-19 F1 (2026, but used 200 labelled/class to refine).
- If the call site already has an LLM, its own outputs on live traffic are the best "synthetic data" (real inputs, LLM labels). Caveat: models trained on LLM labels underperform gold-label ones (arXiv 2504.15432).

## Reusable models / methods

Numbers are as reported by the cited source; metric noted. "CPU" = my judgement unless the source measured it (none measured CPU).

| name | url | license | reported B77 / CLINC | CPU-feasible? |
|---|---|---|---|---|
| Embedder label-template cosine (bge/gte/e5/Qwen3-Emb), no instructions | https://arxiv.org/abs/2603.11991 | MIT/Apache (model cards, from memory) | k=0 B77 macro-F1: bge-base .64, bge-large .68, gte-base-v1.5 .66, e5-base-v2 .62, e5-mistral-7b .65, Qwen3-Emb-0.6B .64, Qwen3-Emb-8B .71, all-MiniLM-L6 .43. No CLINC. | <=0.6B yes; 7-8B no |
| Qwen3-Reranker 0.6B / 8B (pairwise) | https://arxiv.org/abs/2603.11991 | Apache-2.0 (unverified) | k=0 B77 F1 .63 / .69 | 0.6B slow (77 pairs/query); 8B no |
| GLiClass v3 (edge 32.7M ... large 439M) | https://huggingface.co/knowledgator/gliclass-modern-large-v3.0 ; paper https://arxiv.org/abs/2508.07662 | Apache-2.0 | k=0 "banking" F1: large .557, base .470, modern-large .368, modern-base .356, edge .027; 8-shot averages only (large .727 over 14 sets). No CLINC. | edge/base plausible; no CPU numbers published |
| deberta-v3-large-zeroshot-v2.0 (NLI-style) | https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 | MIT base; training-data licences vary ("-c" variants commercial-safe) | B77 (72 cls) F1 .567 zero-shot, .766 fine-tuned with <=500/class; bart-large-mnli .312 | per-label passes; poor |
| tasksource NLI (deberta-v3-base-tasksource-nli) | https://huggingface.co/sileod/deberta-v3-base-tasksource-nli | Apache-2.0 | none found for B77/CLINC; BTZSC NLI-family B77 F1 .01-.35 | per-label passes |
| ModernBERT NLI / zeroshot-v2.0 | https://cnb.cool/ai-models/MoritzLaurer/ModernBERT-large-zeroshot-v2.0 (mirror) | unverified | BTZSC modernbert-large-nli B77 F1 .21 | per-label passes |
| SetFit "zero-shot" (templated synthetic set, "The customer request is about {}") | https://docs.argilla.io/en/latest/tutorials/notebooks/labelling-textclassification-setfit-zeroshot.html | SetFit Apache-2.0 (memory) | only an 8-class coarse B77 variant, F1 ~0.86; no 77-class number. SetFit few-shot B77: 5-shot 77.3 (S)/81.7 (L), 10-shot 84.8/86.4 (FastFit Table 1) | training 12-50 min on GPU; no |
| PIE intent-aware encoder | https://arxiv.org/abs/2305.14827 | release unverified | k=0 label-name, N-way subsets: B77(N=27) 71.4 vs 68.9 mpnet-paraphrase; CLINC(N=50) 83.1 vs 76.3 | 110M, yes if weights exist |
| FLAN-T5-3B fine-tuned with intent descriptions, test on held-out CLINC domains | https://aclanthology.org/2024.sigdial-1.39.pdf | n/a | held-out 75 CLINC intents: 84-92% depending on description quality (their Table 1) | 3B: no |
| Generate-then-Refine (LLM synth + Flan-T5-large refiner) | https://arxiv.org/abs/2410.01953 | unverified | unseen-domain CLINC150 acc: ZeroGen 70-74, SuperGen 66-71, Refiner 76.0-76.9, real 95.4 | generation offline; classifier small: yes |
| FastFit (few-shot, many classes) | https://github.com/IBM/fastfit | Apache-2.0 | 5/10-shot B77: S(mpnet) 81.0/86.4, L(roberta-large) 85.2/88.8; CLINC 91.3/93.5 (S), 93.4/95.3 (L) | small ~minutes on GPU; CPU slow |
| DDR (label-description dual encoder) | https://github.com/luka-group/DDR | unverified | 5/10-shot B77 83.9/88.3; CLINC 92.7/94.6 | BERT-size: yes |
| Dynamic label-name refinement (retrieval ICL) | https://arxiv.org/abs/2412.15603 | n/a | 10-shot B77 87.3-88.0 (7-8B LLMs), 77.3 (Qwen2.5-1.5B) | LLM: no |
| CLAP / LP++ text-anchored probe | https://arxiv.org/abs/2312.12730 ; https://openaccess.thecvf.com/content/CVPR2024/html/Huang_LP_A_Surprisingly_Strong_Linear_Probe_for_Few-Shot_CLIP_CVPR_2024_paper.html | code licences unverified | vision only; no text numbers | trivially yes (a regularised LR) |

## Evidence notes

### 1. Pretrained universal zero-shot classifiers and embedders
- BTZSC (Mar 2026) is the only apples-to-apples benchmark found across NLI cross-encoders, embedders, rerankers, LLMs. Verbalised label templates, cosine for embedders, B77 macro-F1 in its Appendix D. Embedders .43-.71, NLI .01-.35, rerankers .24-.69, instruction LLMs .00-.42. Embedders give the best accuracy per unit compute; scaling 0.6B -> 8B helped Qwen3-Embedding only .64 -> .71. No instruction prefixes were used, so the "instruction-tuned embedder" upside is untested there. https://arxiv.org/abs/2603.11991
- GLiClass (Aug 2025): uni-encoder (labels concatenated with text), DeBERTa-v3 backbone beat ModernBERT, trained on 1.2M examples. B77 appears only as "banking": see table. DeBERTa-v3-base cross-encoder throughput collapses from 24.6 to 0.47 ex/s going 1 -> 128 labels on an A6000 while GLiClass-large holds 19 -> 17.6. https://arxiv.org/abs/2508.07662 , https://huggingface.co/knowledgator/gliclass-modern-large-v3.0
- Open-weight LLMs, label names only, 41 models (Jul 2026): B77 mean .388, best .708, none above 80%; CLINC mean .468, best .836. Qwen2.5-3B is best sub-4B (aggregate .632). https://arxiv.org/abs/2607.27421
- Adding LLM-written descriptions: Mistral-7B B77 macro-acc 81.6 -> 88.8 (no CoT). Descriptions were generated from a few training examples per class, and this conflicts with the 41-model ceiling (different prompt/eval), so treat as upper bound. https://arxiv.org/abs/2501.12332
- Label-name quality: LLM label renaming raised 10-shot B77 by +2.0 (Llama3-8B) to +4.0 (Qwen2.5-1.5B). https://arxiv.org/abs/2412.15603
- Description quality at test time beats train-time description diversity: 5 diverse descriptions vs 1 made no difference; cleansed test descriptions gave 85.3-85.8 vs 82.1-83.4 for raw ones (FLAN-T5-3B, held-out CLINC domains). https://aclanthology.org/2024.sigdial-1.39.pdf
- LLM-written definitions as cosine prototypes, refined iteratively with an LLM: +6 to +20 macro-F1 across 13 sub-1B embedders on news/web-category data (not intent; refinement used 200 labelled/class, test-set selection risk). https://arxiv.org/abs/2604.27335

### 2. LLM-synthesised training data
- Closest match to our protocol (train on seen domains, test on unseen CLINC domains, 5 splits): Generate-then-Refine. 100 utterances per unseen intent from Zephyr-7B / Llama3-8B; a Flan-T5-large refiner trained on seen domains rewrites them; diversity (distinct-1/2) rises to human level (0.200 vs 0.167 on CLINC). Accuracy 76-77 vs 70-74 plain ZeroGen vs 95.4 real. https://arxiv.org/abs/2410.01953
- Zero-shot vs few-shot generation, 10 tasks, GPT-3.5, 3000/label: average macro-F1 gap to real data 16.9 (zero-shot) and 6.7 (few-shot) points; zero-shot is worst on subjective tasks (-27.4), fine on topical ones (AG News -6.0). B77 is topical but fine-grained, so expect the smaller gap only if intents are separable. https://arxiv.org/abs/2310.07849
- 1000 synthetic samples/task is enough; adding only 100 real examples lifted most methods by 3-26%; topic-conditioned zero-shot generation is the most diverse; leakage of label words into synthetic text is a risk. No intent task. https://arxiv.org/abs/2407.12813
- Fine-grained confusion is the failure mode: DDAIR (Jan 2026) found LLM utterances drift into neighbouring intents; disambiguating by nearest-centroid check with regeneration gives only +0.5-0.8 F1 (B77 5-shot + 10 synthetic: 71-73 macro-F1, BERT-base). https://arxiv.org/abs/2601.11234
- Other diversity tricks with evidence: attribute-conditioned prompting (AttrPrompt, https://arxiv.org/abs/2306.15895); retrieval-seeded generation (SynthesizRR, https://arxiv.org/abs/2405.10040); sample 5 seed shots at random per call (DDAIR). Practical read: generate with attributes (tone, slot values, typos) and filter by nearest-centroid check in bge space.
- Caution on LLM-labelled real text: https://arxiv.org/pdf/2504.15432 (gold labels beat LLM labels, gap widens with task complexity).

### 3. Hypernetworks / meta-learning on few tasks
- Text-to-LoRA (ICML 2025): zero-shot SFT-trained T2L-L reaches 67.7 avg on 10 unseen benchmarks with 479 training tasks, 66.0 with 64; 73.9 vs 75.8 task-specific LoRA on the 8-task suite; base 60.0. Reconstruction-trained (few adapters) fails to generalise: 61.8 vs 66.3. Authors: still below task-specific adapters; quality depends on description quality. https://arxiv.org/html/2506.06105v2
- Doc-to-LoRA (Feb 2026) restates the lesson: diverse training data and an end-to-end task loss are what give generalisation; larger hypernetworks plus more data generalise better. https://pub.sakana.ai/doc-to-lora/
- Text2Model (EMNLP Findings 2024): hypernetwork from class descriptions to classifiers (vision/point-cloud/action); the stated generalisation lever is permutation-equivariance over the set of descriptions and the classification layer. https://arxiv.org/abs/2210.15182 . HyperCLIP is similar: https://arxiv.org/pdf/2412.16777
- Meta-learning vs strong pretrained baseline (images, but the closest hard evidence): baseline with distance classifier matches meta-learners and wins cross-domain (Chen et al.); simple fine-tuning beats meta-learning by 12.8% on CD-FSL (https://arxiv.org/abs/1912.07200); pretrained features + simple classifier far outperform meta-learners (https://arxiv.org/abs/2101.00562). I found no head-to-head text result using modern encoders.
- Prototype/retrieval families that do have text numbers: DDR (label-description thesaurus, dual encoder) 83.9 / 88.3 B77 at 5/10-shot; FastFit 85.2 / 88.8 (L). A plain classifier on a good encoder is close (87.4 at 10-shot, L), which matches our frozen-LR 87%, so frozen bge-small + LR is already at the few-shot frontier. https://arxiv.org/abs/2404.12365
- Zero-shot-initialised, text-anchored probing: CLAP and LP++ show the zero-shot init degrades when fitted naively, and fix it by penalising deviation from text prototypes (per-class penalty) or blending learnable per-class weights with fixed text embeddings. https://arxiv.org/abs/2312.12730

### 4. 2026 task-description-to-weights work
- Doc-to-LoRA (Sakana, Feb 2026): 309M-param Perceiver hypernetwork, rank-8 LoRA on Gemma-2-2b; sibling of Text-to-LoRA, not a classifier. https://pub.sakana.ai/doc-to-lora/
- MeGan (May 2026): 3.4M-param hypernetwork writes per-layer activation slopes of a frozen Llama-3.1-8B from a text condition, trained on 479 SNI tasks; self-reported to beat T2L on most tasks, weaker on code/math. Small generator, large backbone. https://arxiv.org/abs/2605.01973
- Style hypernetwork (Oct 2026): rank-8 LoRA from 3 reference abstracts on Qwen3-4B; seen authors 36.5% style accuracy vs unseen 31.4% (plain LoRA 37.5% on unseen), i.e. the generator did NOT beat LoRA on unseen conditions. https://arxiv.org/abs/2610.03163
- Drag-and-Drop LLMs (NeurIPS 2025): prompt-conditioned hyper-convolutional decoder to LoRA; no 2026 follow-up found. https://arxiv.org/abs/2506.16406
- No 2026 paper found that generates classifier heads / adapters from a label schema at small-encoder scale. Nothing at our scale to copy.

## Suggested experiments (ordered by cost)
1. k=0: score = cos(x, mean(label_name, description, 5-10 LLM pseudo-queries)) with prototype ensembling; measure vs 66/71.
2. k>0: LR with L2 penalty toward W0 = normalised prototypes (CLAP-style), sweep the penalty on seen domains only; report k = 0, 1, 3, 5, 10.
3. Add LLM-synthesised examples (attribute-conditioned, centroid-filtered) as pseudo-shots into step 2; compare 20 synthetic/label vs 5 real/label.
4. Only then: learned metric projection (PIE-style contrastive loss on seen domains, rank <= 32, weight-decay), evaluated strictly on held-out domains and B77.

## Unverified
- No published bge-small k=0 B77 number exists in sources I read (BTZSC starts at base size); the 0.64-0.71 plateau is extrapolated to small.
- Whether instruction prefixes (e5/gte/Qwen3 task instructions) help k=0 intent classification: BTZSC used none; no source tested it.
- GLiClass CLINC150 numbers: none published; GLiClass-edge "banking" F1 .027 looks like a failure mode, not a ranking.
- SetFit 77-class zero-shot (templated) accuracy: not found; the 0.86 F1 is a coarse 8-class variant.
- The 2501.12332 (88.8) vs 2607.27421 (70.8 best) gap on B77 for 7B LLMs is unreconciled.
- SIGDIAL 2024 Table 3 text extraction was garbled (CLINC 50-way zero-shot appears around 96-98, B77 27-way about 84-85); only Table 1 figures used above.
- Licences for Qwen3-Embedding, gte, bge, SetFit, DDR, CLAP/LP++ code, PIE weights, ModernBERT zeroshot-v2.0 were not verified from the sources; GLiClass (Apache-2.0), tasksource (Apache-2.0), FastFit repo (Apache-2.0) were.
- Table cells were pulled with a summarising fetch tool (small model); spot-check the BTZSC Appendix D and FastFit Table 1 numbers before quoting externally.
- Whether CLAP/LP++-style anchoring helps text embeddings: no evidence either way; vision-only.
