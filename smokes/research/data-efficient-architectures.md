# Learning from less labelled data at adapter-like cost: what exists after the hypernetwork (research, 2026-10-17)

Method: WebSearch plus WebFetch of arXiv abstract/HTML pages and GitHub READMEs, 2025-2026 first. "Fetched" numbers come from the page named; anything else is under Unverified. Our reference points (FINDINGS.md): frozen bge-small + LR 60% at 1 row/label and 82% at 10 rows/label on Banking77; S2b synthetic prior 72.6% at 0 rows; S8 phase B shared LoRA 84% vs 60% at 3 rows/label on a new 15-label step; hypernetwork lost to the label-embedding baseline (S2, S2b); S20 (TabICL head) and S17 (warm start) pending.

**Headline.** No new adapter architecture found that cuts labels by a large factor at 33M-0.5B scale. The verified gains come from (a) using the unlabelled stream (semi-supervised probes, domain adaptation), (b) sibling or public tasks as a prior (S8-B already shows it), and (c) one genuinely new sub-LoRA method (AHR, 2026) that is data-efficient but only shown on 1-8B decoders with 2-6 labels. ReFT, TTT/Titans/HOPE and TabPFN-style in-context heads have no evidence of fewer labels at our scale; TabPFN is in fact reported worse than LR at 5 shots. The hypernetwork idea does work at scale (PAW, 10M training examples) and its compiler weights are public, so it can be used rather than trained.

## Top 5 to smoke (highest expected data-efficiency gain per dollar first)
Common protocol: Banking77 (77-way) and CLINC-15 / Banking77-12 sub-tasks, k = 1, 3, 10 rows per label, baselines S1 LR and S2b prior, metric = accuracy and rows to a 5% certificate. Calibration rows must stay a random sample, never a selected or pseudo-labelled one.

**1. Text-anchored semi-supervised probe on the logged stream.** CPU, ~20 min, $0.
- Evidence: SS-Text-U (Mar 2026, vision-language, 12 medical sets): +10.9 / +7.1 / +2.7 / +1.3 / +0.3 ACA points over the best training-free probe at 1/2/4/8/16 shots, "50-75% labelling reduction", solver ~25 ms. TK-KNN (EMNLP Findings 2023) on Banking77 with BERT-base: 54.2 vs 34.7 supervised at 1% labels, 62.7 vs 47.5 at 2%, 76.7 vs 70.3 at 5%, 84.5 vs 80.8 at 10%. Both fetched.
- Sketch: cache bge-small embeddings of all B77 train text; reveal k labels per class, hide the rest as the unlabelled pool (24 x C rows as in the paper, then the full pool). Prototypes start at S2b synthetic-example centroids (text anchor). Variants: (a) SS-Text-U block-coordinate loop, T=3, Sinkhorn pseudo-labels with the class marginal fixed, closed-form W toward the anchor; (b) TK-KNN top-k balanced self-training around sklearn LR; (c) average class probabilities over the 10 nearest logged inputs at inference (TestNUC-S idea).
- Packages: https://github.com/jusiro/SS-Text-U (licence unchecked); https://github.com/ServiceNow/tk-knn (pins torch 2.1, transformers 4.34; licence not shown; only the idea is needed); `pip install scikit-learn sentence-transformers`.
- Blockers: no published text result for SS-Text-U; TK-KNN baselines are weak (fine-tuned BERT-base, far below frozen bge-small + LR at 1% labels), so the gain may shrink on a strong encoder; wrong pseudo-labels lock in on Banking77's confusable intents; expect gains only below ~4 rows/label.

**2. Sibling/public-task adapter library with routing (scale S8 phase B from 8 to 30-60 tasks).** 4090, ~25 min, ~$0.15.
- Evidence: LoRAuter (Jan 2026, Llama-2 7B): 88.4% of oracle on unseen tasks vs Arrow 82.0, LoraHub 67.8, SpectR 66.3; pool of 48 adapters, task embedding from <=200 validation queries. LoraHub: no gradients, "a few" examples, but does not beat ICL accuracy on BBH. S8-B in-house: 84 vs 60% at 45 rows (3/label).
- Sketch: tasks = 18 MASSIVE scenarios + 10 CLINC domains + Banking77-12 splits + TREC + other free sets; one bge-small LoRA (or head-only) per task (~7 s each). Leave-one-task-out at k = 1, 3, 10: (i) one multitask LoRA over the other tasks (S8-B scaled), (ii) uniform LoRA average via PEFT `add_weighted_adapter`, (iii) LoraHub coefficient search on the k rows (nevergrad), (iv) Arrow-style SVD-prototype routing, (v) LoRAuter-style task-embedding routing. Does adding tasks keep helping beyond 8?
- Packages: `pip install peft nevergrad`; https://github.com/sail-sg/lorahub; https://github.com/r-three/phatgoose; Arrow lives in the mttl codebase (repo URL unconfirmed).
- Blockers: all published libraries are 7B decoders with generative outputs; classification heads differ per label set, so use prototype/text-conditioned heads; no encoder-scale paper exists.

**3. Unsupervised domain adaptation of the encoder on the call stream (AdaSent-style).** 4090, ~10 min, ~$0.05.
- Evidence (fetched): AdaSent, EMNLP 2023, 17 few-shot sets, 8 shots: 71.5 vs 67.6 base; Banking77 80.6 vs 77.0; MLM-DAPT on the whole unlabelled train set for 2,344 steps, batch 256. DFT++ paper: MLM on too little text hurts, 50 utterances/label saturates.
- Sketch: continue bge-small with TSDAE or SimCSE (sentence-transformers) or MLM on the unlabelled B77/CLINC text, then LR at k = 1, 3, 10; compare with the untouched encoder; also try a 2-encoder co-training variant (bge-small + gte-small as views, each pseudo-labels for the other).
- Packages: `pip install sentence-transformers setfit`; https://github.com/UKPLab/AdaSent (Apache-2.0, DAPT/SEPT/SetFit scripts, DistilRoBERTa only tested).
- Blockers: bge-small is already contrastively trained; gain may be 0-2 points; S8 showed SetFit under-reproduces the literature.

**4. Use the public compiler as a hypernetwork prior (PAW, 2026).** 4090 (>=16 GB), ~30 min, ~$0.15.
- Evidence (fetched): 4B compiler trained on 10M-example FuzzyBench emits a rank-64 LoRA (~38.5M params, ~22 MB) for a frozen Qwen3-0.6B interpreter; 90.4 / 80.8 F1 / 95.8 / 90.6 on YouTube / SMS / Yelp / IMDB vs 93.6 / 89.0 / 98.1 / 94.6 for Qwen3-32B prompting. "Compile by Training" (EMNLP 2026 demo) then fine-tunes the interpreter on teacher-written examples: 83.6% on FuzzyBench-Hard, ~1 min compile.
- Sketch: compile specs (label list + descriptions + 3-6 examples) for CLINC-15 domains and Banking77-12 sub-tasks; score zero-shot; then start LoRA fine-tuning at k = 1, 3, 10 from the compiled adapter vs random init vs S8-B shared LoRA vs S2b.
- Packages: `pip install programasweights --extra-index-url https://pypi.programasweights.com/simple/`; weights `programasweights/paw-4b-qwen3-0.6b` on Hugging Face; https://github.com/programasweights/compiler (MIT); `compile-by-training` needs an OpenAI key.
- Blockers: single-step fuzzy functions only; a 0.6B decoder lost to the 33M encoder in S8 at every size; 77-way label lists are untested; checkpoint licence unstated.

**5. Attention-head reweighting (AHR, COLM 2026), the one sub-LoRA method with a low-data win.** 4090, ~25 min, ~$0.10.
- Evidence (fetched): one scalar per head (512-1,152 trainable params). Mean over GPT-2 XL, Llama-3.2-1B/3B, Qwen3-8B and 6 datasets with 2-6 classes: 74.95 vs LoRA 68.27 / AdaLoRA 71.22 / IA3 67.88 at 10 samples; 79.83 vs 77.32 / 77.47 / 77.83 at 30; 82.68 vs 83.05 at 100. Authors: other methods win at ~300+.
- Sketch: Qwen3-0.6B/1.7B, prompt lists the labels, train AHR vs LoRA r=8 vs frozen-bge-small LR on CLINC-15 domains and Banking77-12 at 10/20/50/100 rows; then try 30+ labels.
- Packages: https://github.com/tuomaso/attention-head-reweighting (Apache-2.0, `uv`, bundles a PEFT 0.20.0 fork; 2 commits, 1 star; only gpt2-xl documented).
- Blockers: decoders >=1B only, <=6 classes, no encoder; label lists in small-LM prompts collapse at 77 labels (4.0 macro-F1, Mistral-7B, 2501.12332); likely loses to the encoder + LR baseline.

Free add-on to any smoke: pick the k labelled rows by coverage (k-means medoids / TypiClust) instead of random. Evidence is image-only or abstract-only (see family 6), cost $0, but never use selected rows for the certificate.

## 1. Representation fine-tuning and sub-LoRA methods
| method | paper / repo | year | data-efficiency evidence | cost | code | prod maturity |
|---|---|---|---|---|---|---|
| LoReFT / DiReFT | https://arxiv.org/abs/2404.03592 | 2024 (NeurIPS) | 15x-65x fewer params than LoRA; abstract makes no low-data claim; benchmarks use full sets (commonsense, arithmetic, GLUE) | training ~ LoRA or less; negligible inference | https://github.com/stanfordnlp/pyreft, `pip install pyreft`, Apache-2.0, 1.6k stars; causal-LM examples only | research, no production use found |
| "ReFT" low-resource study | https://pmc.ncbi.nlm.nih.gov/articles/PMC12705377/ | 2025 | DistilBERT, 1,000 rows: F1 LoRA .891/.909 (AG News/Amazon), IA3 .881/.874, "ReFT" .881/.889, full FT .844/.333. Its "ReFT" is head + LayerNorm only (20k params), not LoReFT | T4, 5 epochs | none | caution: do not cite as ReFT evidence |
| AxBench ReFT-r1, RePS, HyperSteer | https://arxiv.org/abs/2501.17148 ; https://arxiv.org/abs/2505.20809 ; https://arxiv.org/abs/2506.03292 | 2025 | steering vectors; prompting still beats all of them; HyperSteer (hypernetwork writing steering vectors) ~ prompting on unseen prompts | tiny | AxBench repo | research; steering, not classification |
| LoFiT | https://arxiv.org/abs/2406.01563 | 2024 | 3-10% of heads, offset vectors, 20-200x fewer params than LoRA; strongest "low-data" claim is from a secondary review; truthfulness/reasoning tasks | small | https://github.com/fc2869/lo-fit | research |
| AHR | https://arxiv.org/abs/2607.13425 | 2026 (COLM) | +6.7 pts over LoRA at 10 samples, +2.5 at 30, parity at 100 (see Top 5 #5); decoders 1-8B, 2-6 classes | ~1k params | https://github.com/tuomaso/attention-head-reweighting | research, 1 star |
| VeRA / NoLA / RandLoRA | https://arxiv.org/abs/2502.00987 | 2025 | per RandLoRA: random-basis methods match LoRA in few-shot at far fewer params, lag as data grows | small | VeRA in `peft` | moderate |
| TinyLoRA | https://arxiv.org/abs/2602.04118 | 2026 | 13 params, 91% GSM8K on Qwen2.5-8B, but via RL; SFT needs 100-1000x larger updates | tiny | not checked | research; RL only, irrelevant to SFT classifiers |
| LoRA Without Regret | https://thinkingmachines.ai/blog/lora/ | 2025 | LoRA = full FT in sample efficiency on small/medium SFT; needs MLP layers and ~10x LR; rank 1 enough for RL | lower than FT | TRL guide | widely used |
| (IA)3 / T-Few, PETapter | https://arxiv.org/abs/2205.05638 ; https://arxiv.org/abs/2412.04975 | 2022 ; 2024-25 | few-shot PEFT beats ICL; PETapter (PEFT + PET heads) "comparable" to PET, numbers not read | small | T-Few repo unchecked | T-Few known, PETapter niche |

Judgement: these reduce bytes per task, not labels. Only AHR shows a label-efficiency gain, and only for few-class decoder prompts. For us a 33M encoder + LR is already a ~100k-parameter task cost.

## 2. Adapter libraries, composition, adapter generators
| method | paper / repo | year | data-efficiency evidence | cost | code | prod maturity |
|---|---|---|---|---|---|---|
| LoraHub | https://arxiv.org/abs/2307.13269 | 2023 (COLM 2024) | composes upstream LoRAs from "a few" examples, gradient-free; BBH: no better than ICL, far fewer tokens | ~40 optimiser steps, no backprop | https://github.com/sail-sg/lorahub | demo-grade |
| Arrow / MBC | https://arxiv.org/abs/2405.11157 | 2024 (ICML) | zero-shot per-input routing via adapter SVD; Phi-2, Mistral; no numbers on the page | no training | mttl (repo unconfirmed) | research |
| PHATGOOSE | https://arxiv.org/abs/2402.05859 | 2024 (ICML) | per-module sigmoid gates, ~100 steps; zero-shot, T5 family; matches or beats multitask training per secondary summaries | cheap | https://github.com/r-three/phatgoose | research |
| LoraRetriever | https://arxiv.org/abs/2402.09997 (ID from memory) | 2024 (ACL) | input-aware retrieval and mixture of LoRAs from a pool; 83.2% of oracle on unseen tasks at Llama-2 7B as reported inside the LoRAuter paper | no training | not checked | research |
| LoRAuter | https://arxiv.org/abs/2601.21795 | 2026 | 88.4% of oracle on unseen tasks (LoraRetriever 83.2, Arrow 82.0, LoraHub 67.8), 85.7% with 1,567 HF adapters; scales with tasks not adapters | validation-set embeddings only | none visible | research |
| Pretrained LoRA averaging | https://mlanthology.org/icmlw/2024/asadi2024icmlw-combining | 2024 | uniform/learned averaging of pretrained LoRAs beats FT and from-scratch LoRA in few-shot (workshop; numbers not read) | cheap | not checked | workshop |
| Text-to-LoRA / Drag-and-Drop / Doc-to-LoRA / Generative Adapter | https://arxiv.org/abs/2506.06105 ; https://arxiv.org/abs/2506.16406 ; https://arxiv.org/abs/2602.15902 ; https://arxiv.org/abs/2411.05877 | 2025-26 | generators need 64-479 tasks (T2L) or large self-supervised corpora (GA: 26 MetaICL tasks, 44.9 avg); still below task-specific LoRA | 1 forward pass | SakanaAI/text-to-lora (Apache-2.0 per aggregator), SakanaAI/Doc-to-LoRA | Doc-to-LoRA ICML 2026; 2-7B bases |
| Program-as-Weights (PAW) + Compile by Training | https://arxiv.org/abs/2607.02512 ; https://arxiv.org/abs/2609.04199 | 2026 | see Top 5 #4; the first hypernetwork result near a 32B prompt at 0.6B | seconds; ~1 min with training | https://github.com/programasweights (MIT), HF weights | public SDK, demo-scale |
| Task-vector / merge methods (TIES, AdaMerging) | https://arxiv.org/pdf/2310.02575 | 2023-25 | AdaMerging learns coefficients by entropy on unlabelled samples; no paper found showing merging beats FT on a new task with few labels | cheap | various | research |

Judgement: composition helps when a library already exists. Our library is the sibling steps; S8-B is the in-house proof, Top 5 #2 tests the scaling. Generators are worth using only if pre-trained by someone else (PAW).

## 3. Test-time training, fast weights, learning at inference
| method | paper / repo | year | data-efficiency evidence | cost | code | prod maturity |
|---|---|---|---|---|---|---|
| TTT layers | https://arxiv.org/abs/2407.04620 | 2024 | language modelling and long context only | linear | no repo on the page | research |
| TTT-E2E | https://arxiv.org/abs/2512.23675 | 2025 | 3B model, constant latency, 2.7x faster than full attention at 128K | meta-learned init | code on test-time-training.github.io (path unconfirmed); no weights seen | research |
| Titans / MIRAS / ATLAS | https://arxiv.org/abs/2501.00663 ; https://arxiv.org/abs/2504.13173 ; https://arxiv.org/abs/2505.23735 | 2024-25 | LM, recall, needle tasks (ATLAS +80% BABILong at 10M); no few-shot classification numbers | train-from-scratch | no official release found | none |
| Nested Learning / HOPE | https://arxiv.org/abs/2512.24695 | 2025 (NeurIPS) | abstract lists "few-shot generalization"; numbers sit past char 100k of the HTML, unread | scratch | none found | none |
| MesaNet | https://arxiv.org/abs/2506.05233 | 2025 (ICLR 2026) | locally optimal TTT layer, billion-scale, extra inference FLOPs; LM metrics only | higher | not stated | research |
| Gated DeltaNet | https://arxiv.org/abs/2412.06464 | 2024 (ICLR 2025) | retrieval and LM gains over Mamba2 | linear | in Qwen3.5 0.8B/2B/4B/9B (Mar 2026, Apache-2.0): open weights usable as a base | in production LLMs |
| TTT on in-context examples (ARC) | https://arxiv.org/abs/2411.07279 | 2024 | per-task update from demos: BBH 10-shot 50.5 -> 57.8, ARC up to 6x; 8B | per-task fine-tune | no repo on page | research |
| SEAL self-adapting LMs | https://arxiv.org/abs/2506.10943 | 2025 | model writes its own fine-tuning data; few-shot gains, no numbers on page | RL + SFT | https://jyopari.github.io/posts/seal | research |
| TTT on nearest neighbours | https://arxiv.org/abs/2305.18466 | 2024 (ICLR) | 20 neighbours, 1 step each helps LM loss; not classification | per-query | in paper | research |

Judgement: none of the Titans-family models has public weights or few-shot classification numbers; Qwen3.5-0.8B (Gated DeltaNet hybrid) is the only usable member and is just an LLM. Test-time updates add no label efficiency over retraining a 33M student in minutes. Skip.

## 4. In-context and prior-fitted learners, retrieval classifiers
| method | paper / repo | year | data-efficiency evidence | cost | code | prod maturity |
|---|---|---|---|---|---|---|
| TabPFN as head on frozen embeddings | https://arxiv.org/abs/2607.11007 | 2026 | 22,820 episodes; wins on NLL/ECE (-48-62% NLL) but at k=5 trails LR (.742 vs .787) and SVM/kNN; wins from k>=50 and only for d<=32; tested C in {2,5,10}, text = 20NG, AG News, Amazon, IMDB, emotion, SST-2 | 43 MB checkpoint, no gradients | https://github.com/Jingxiang-Zhang/tabpfn-multimodal-embeddings | tabular prod, not text |
| TabPFN-2.5 / TabPFN-3 | https://arxiv.org/pdf/2511.08667 ; https://docs.priorlabs.ai/capabilities/many-class | 2025-26 | class limit 10 (2.5, 2.6) and 160 (TabPFN-3); ECOC wrapper for more | forward pass | `pip install tabpfn`, `tabpfn-extensions[many_class]`; licence unchecked | tabular prod |
| TabICLv2 | https://arxiv.org/abs/2602.11139 | 2026 (ICML) | beats RealTabPFN-2.5 on TabArena/TALENT incl. >10-class benchmark; no text result | faster than TabPFN | https://github.com/soda-inria/tabicl (BSD-3 per earlier note) | research-to-prod |
| TabPFN text adapter, CoMET | https://arxiv.org/abs/2606.04876 ; https://arxiv.org/abs/2605.20674 | 2026 | adapter maps MiniLM-384 embeddings into TabPFN tokens (no numbers on page); CoMET = PCA + TFM, 2,000-class hierarchy | adapter-only training | not confirmed | research |
| TestNUC | https://arxiv.org/abs/2502.19163 | 2025 (ACL) | classify with predictions on K unlabelled neighbours: +2.9 to +6.4 pts over standard prompting (8 sets incl. Banking77, CLINC; K=10), plateau at ~8-10k unlabelled | $0.000028/sample with caching | https://github.com/HenryPengZou/TestNUC | research; LLM prompting, not a trained student |
| FastFit / DDR / SetFit | https://arxiv.org/abs/2404.12365 | 2024 | B77 5/10-shot 81.0/86.4 (S), 85.2/88.8 (L); see day-zero-student.md | GPU minutes | https://github.com/IBM/fastfit | used |
| ModernBERT-Large-Instruct | https://arxiv.org/abs/2502.03793 | 2025 | MLM-head generative classifier, 0.4B, strong zero-shot; no few-shot numbers | 0.4B | weights link not on page | niche |

Judgement: the new evidence lowers S20's odds on Banking77 (C=77, d=384). If run, use PCA to 32 dims and the b12/CLINC-15 sub-tasks, or TabPFN-3 / TabICLv2 many-class. Retrieval over the logged stream is already the LR/kNN student.

## 5. Using the unlabelled call stream
| method | paper / repo | year | data-efficiency evidence | cost | code | prod maturity |
|---|---|---|---|---|---|---|
| SS-Text-U (text-anchored semi-supervised probe) | https://arxiv.org/abs/2603.02959 | 2026 | see Top 5 #1 | 25 ms | https://github.com/jusiro/SS-Text-U | research, medical VLM |
| TK-KNN | https://arxiv.org/abs/2310.11607 | 2023 | see Top 5 #1; CLINC 53.7 vs 27.4 at 1%, HWU64 65.3 vs 48.9 | 30 self-training cycles | https://github.com/ServiceNow/tk-knn | research |
| AdaSent / DAPT | https://arxiv.org/abs/2311.00408 | 2023 (EMNLP) | see Top 5 #3 | one-time SEPT adapter 0.17 h; DAPT per task | https://github.com/UKPLab/AdaSent | research |
| Self-training under pretraining | https://arxiv.org/abs/2011.08626 | 2020 | pseudo-labelling best with small unlabelled sets, in-domain pretraining with large; combined best; IMDB 93.8% with 50 labels | one fine-tune | not checked | classic |
| UST (uncertainty-aware self-training) | https://arxiv.org/abs/2006.15315 | 2020 (NeurIPS) | 20-30 labels/class reach ~3 pts of full supervision (~91% aggregate, 5 sets) | several rounds | not shown | classic |
| Direct FT vs CPT, DFT++ | https://arxiv.org/abs/2306.05278 | 2023 | B77 5-shot: DFT 69.0 -> DFT++ 72.9 (BERT), 76.1 -> 78.9 (RoBERTa); 10-shot 78.9 -> 82.7 / 84.8 -> 86.1; gains need generated unlabelled text | generator | not checked | research |
| LLM-guided co-training | https://arxiv.org/abs/2509.16516 | 2025 | two encoders weight LLM pseudo-labels for each other; best on 4/5 sets of 14 methods; no numbers on page | two students | none | research |
| USB benchmark / semilearn | https://arxiv.org/abs/2208.07204 | 2022 | FixMatch/FlexMatch/UDA with pretrained Transformers incl. NLP; NLP numbers not read | 39 GPU-days all tasks | microsoft/Semi-supervised-learning | library |
| Test-time entropy minimisation | https://icml.cc/virtual/2026/poster/65581 | 2026 | FG-TTL: entropy minimisation collapses on LLM generation; irrelevant for one-token typed outputs; no text-encoder TTA paper found | cheap | n/a | n/a |

Judgement: this is the largest unspent resource; effects shrink fast beyond ~4-8 rows/label. Entropy-based test-time adaptation has no 2025-26 text-classifier evidence.

## 6. Label-efficient training signals
| method | paper / repo | year | data-efficiency evidence | cost | code | prod maturity |
|---|---|---|---|---|---|---|
| ActiveLLM | https://arxiv.org/abs/2405.10808 | 2024-26 (TACL 14) | LLM picks the labelled instances; beats classical AL for BERT, ADAPET, PERFECT, SetFit in few-shot (no numbers on page) | LLM calls | not linked | research |
| DoTCAL | https://arxiv.org/abs/2407.17284 | 2024 | MLM on all unlabelled text, then fine-tune on selected rows: up to +33% macro-F1, ~half the labels, 8 sets; BoW/LSI sometimes beat BERT at tiny budgets | MLM pass | not read | research |
| M-RARU active distillation | https://arxiv.org/abs/2511.11574 | 2025 | uncertainty + randomised accept/reject picks teacher queries; "up to 80%" fewer samples than random for SVM/RF/DistilBERT students | cheap | none | research |
| GCP, student-guided distillation | https://arxiv.org/abs/2602.03006 ; https://arxiv.org/abs/2610.02516 | 2026 | concept-graph active distillation, 8 sets (no numbers); student top-k + NLI teacher reranks, 77.5% agreement, preliminary; warns truncated top-k scores are not soft targets | modest | https://github.com/Ziyang-Yu/GCP | research |
| Cold-start AL (TypiClust, ProbCover) | https://arxiv.org/abs/2508.03441 | 2025 | TypiClust >= ProbCover > random in image benchmarks; no text benchmark found | k-means | various | research |
| LLM labels into BERT | https://arxiv.org/abs/2504.15432 | 2025 | RoBERTa-base on 5,000 rows: gap to gold 0.5 pts (IMDB) to 30 pts (E-commerce, 3B labeller); entropy filtering fails, 5-way consistency ensemble recovers 60-75% of the gap at ~25% extra inference | extra calls | none | caution |
| Small vs LLM break-even | https://arxiv.org/abs/2402.12819 | 2024 | ~100 labelled samples on average for a specialised small model to match a general LLM (8 tasks, 8 models); +100-200% with variance | FT | not read | reference |
| Weak supervision with LLMs | https://arxiv.org/abs/2502.12366 ; https://arxiv.org/abs/2505.23470 | 2025 | ScriptoriumWS: LLM-written labelling functions keep hand-written accuracy, much higher coverage; LF repair needs "surprisingly small" label sets (no numbers) | pennies | `pip install snorkel` | Snorkel in production; LLM LFs research |

Judgement: tool outputs as labelling functions fit Dreaming (vendor lookup -> category) and need no new model; active selection saves teacher dollars only where labels are expensive (human review, mission 4).

## 7. New in 2026 that fits adapter-like cost
| method | paper / repo | year | data-efficiency evidence | cost | code | prod maturity |
|---|---|---|---|---|---|---|
| Sparse memory finetuning | https://arxiv.org/abs/2510.15103 | 2025 | NQ F1 drop 11% vs 89% full FT, 71% LoRA after learning new facts; forgetting, not label efficiency | memory-layer models | not checked | research |
| Bayesian-LoRA, Laplace-LoRA | https://arxiv.org/abs/2601.21003 ; https://arxiv.org/abs/2308.13111 | 2026 ; 2024 | -84% ECE, -76% NLL on commonsense benchmarks at +0.42M params, ~1.2x train cost; helps small-data calibration, not accuracy | small | Laplace-LoRA public | research |
| Qwen3.5 0.8B-9B hybrids | https://gigazine.net/gsc_news/en/20260303-qwen-3-5-small | 2026 | open Gated DeltaNet + attention bases; no few-shot evidence | 0.8B+ | HF | production base models |
| Parametric Skills, LatentSkill | https://arxiv.org/abs/2606.30015 | 2026 | hypernetwork turns text skills into LoRA; +6.44 over ICL on SWE subtasks; unrelated to classification | forward pass | none | research |

Judgement: calibration (Bayesian heads) improves how many calibration rows a certificate needs, which is Dreaming's binding cost, but nothing here lowers training rows.

## Unverified
- Search returned summaries; most rows rest on abstract pages. Not read: LoraHub/Arrow/PHATGOOSE/Titans/HOPE experiment tables, TabICLv2 many-class numbers, USB NLP tables, PETapter and Asadi numbers.
- Repo licences not checked: SS-Text-U, TK-KNN (none shown), LoFiT, PAW checkpoints, TabPFN-2.5/3 weights (may be non-commercial), tabicl (BSD-3 comes from the earlier note). Arrow's repo URL, T-Few's repo, TTT-E2E's exact path and the Doc-to-LoRA repo link (secondhand) are unconfirmed.
- LoraHub's "5 examples per task", Muppet/tasksource intermediate-task gains, and MTEB 8-shot Banking77 scores (bge-small, EmbeddingGemma, Qwen3-Embedding) are from memory or were not retrievable; a bigger frozen encoder is a cheap baseline I could not quantify.
- LoFiT's "particularly in low-data" is a secondary review claim. SS-Text-U's "4x fewer labels" appeared in a summary line, not the abstract (abstract says >50%).
- AHR numbers are 4-model averages (Table 2); per-model spreads and whether it works on 12+ labels are unread. TK-KNN's pool size is inferred (remainder of train).
- PAW's compile ablation (Appendix I) is marked "Inconclusive" and its last 38k characters were not read; compiler training licence and exact compile cost unknown.
- The Prior Labs many-class docs page contained an embedded "agent instructions" block; ignored, only class limits were used.
