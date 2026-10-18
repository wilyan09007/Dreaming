# Hypernetworks that write a student from a task description

Researched 2026-10-17. All URLs fetched this session unless tagged (unverified).

## Summary

- Text-to-LoRA (Sakana, ICML 2025) is the only mature open implementation: code Apache-2.0, pretrained generators on HF for Mistral-7B, Llama-3.1-8B and Gemma-2-2B. It emits LoRAs for those LLMs only, so it cannot write a small student of ours. Reusable as a design reference, not as weights.
- Honest gain over the obvious baseline is small. T2L-L zero-shot on unseen benchmarks: 73.9 vs 71.9 for a multi-task LoRA trained on the same 479 tasks (8-task avg, Mistral), +2.0. On 10 tasks 67.7 vs 66.3 (+1.4). Llama 76.9 vs 76.5 (+0.4). Gemma-2B 66.4 vs 65.2 (+1.2). Base model 60.0 / 55.8. Oracle task-specific LoRAs 75.8. So "one shared student" gets ~90%+ of the hypernetwork's gain over base.
- T2L training cost is ambiguous: README says each SFT run "takes around 5 days on a single H100"; the paper says ~10 h wall-clock per model. At RunPod H100 $1.99-3.99/h that is $20-40 (10 h) or $239-479 (120 h). Only the 10 h figure fits our budget, and only barely.
- Drag-and-Drop LLMs (DnD, NeurIPS 2025): code Apache-2.0 and checkpoints on HF, but gains are vs the average of training LoRAs, a weak baseline. Compute for generator training not reported. Needs a library of many same-architecture LoRA checkpoints as training data.
- HyperCLIP (2024): hypernetwork turns the set of class-name text embeddings into the norm scale/bias of a small image encoder. +1 to +5.6 pts zero-shot over a SigLIP baseline, but trained on 4x A6000 over ~100M image-text pairs. No code link found. Closest analog to "class-set text -> small student params", but not CPU-feasible.
- Classic "write a classifier from text" (Elhoseiny 2013, Ba 2015) shows text -> classifier weights beats naive regression/domain-adaptation baselines (AUC 0.80-0.85 vs 0.59-0.66 on CUB) but absolute zero-shot top-1 was ~10%. Pre-CLIP/pre-sentence-embedding; historical precedent only.
- Meta-learned/hypernetwork heads over good frozen features have a documented weakness: simple baselines (linear/logistic head on frozen embedding) match or beat meta-learners (Chen et al. ICLR 2019; Tian et al. ECCV 2020). Expect the hypernetwork to win only at 0-2 shots.
- Simplest feasible version: a ~100-300k-param MLP that maps (label-text embedding, optional k-shot prototype) to a class weight vector, trained episodically on CPU over cached all-MiniLM-L6-v2 embeddings of Banking77 + CLINC150. Must beat label-embedding zero-shot, k-shot prototypes, and logistic regression (see last section).
- A generated student is still a model to certify: T2L loses ~9 pts with random descriptions (72.2 -> 63.5) and ~21 pts with another task's description (-> 51.4). The shadow/certify loop must gate it like any other predictor.

## Findings

### Text-to-LoRA (Sakana AI)
- Paper: https://arxiv.org/abs/2506.06105 (Charakorn, Cetin, Tang, Lange; ICML 2025; paper CC BY-NC-SA 4.0). HTML with tables: https://arxiv.org/html/2506.06105v2
- Code: https://github.com/SakanaAI/text-to-lora (Apache-2.0). Pretrained: https://huggingface.co/SakanaAI/text-to-lora (dirs `trained_t2l/{gemma_2b_t2l,llama_8b_t2l,mistral_7b_t2l,t2l_demo}`, repo total 1.72 GB, no login visible). Download: `huggingface-cli download SakanaAI/text-to-lora --local-dir . --include "trained_t2l/*"`.
- Demo needs ">16GB GPU"; README suggests gemma-2-2b-it may fit a smaller GPU.
- Generator sizes: T2L-L 55.25M, M 34.28M, S 4.92M params (one rank-8 q/v LoRA = 3.41M). Input: task description embedded by gte-large-en-v1.5 (Mistral-7B-Instruct embeddings are comparable, Table 4).
- Training data: 479 Super-Natural-Instructions tasks (500 minus 11 val minus 10 contaminated), 128 generated descriptions per task. Eval on 10 unseen benchmarks (ArcC, ArcE, BoolQ, GSM8K, HellaSwag, OBQA, PIQA, WinoGrande, HumanEval, MBPP).
- Two training modes. Reconstruction (distill pre-trained LoRAs): ~10 min for 9 LoRAs, ~10 h for 479 LoRAs on one H100, but "fails to generalize to unseen tasks" (zero-shot 61.8 avg vs SFT 66.3, Table 6). SFT (end-to-end through the LLM): needs far more memory; paper says ~10 h wall-clock per model; README says ~5 days per H100 run (discrepancy, unresolved).
- Zero-shot results, Mistral-7B-Instruct (8-task / 10-task avg): base 60.0/55.8; prepend description 65.8/60.6; 3-shot ICL 66.5/61.0; multi-task LoRA 71.9/66.3; Hyperdecoders 73.6/67.3; T2L-L SFT 73.9/67.7; task-specific LoRA oracle 75.8/n.a.
- Number of training tasks (Table 3, L): 479 -> 67.7, 256 -> 67.4, 128 -> 65.8, 64 -> 66.0. Weak dependence, S variant does not benefit from more tasks.
- Failure modes: aligned descriptions 73.3 / eval descriptions 72.2 / random strings 63.5 / other-task descriptions 51.4 (reconstruction-trained L). Paper: adapter-weight similarity does not correlate with description similarity (App. D).
- Follow-up: Doc-to-LoRA (Feb 2026), https://sakana.ai/doc-to-lora/ , https://arxiv.org/abs/2602.15902 . Blog says code released, "sub-second latency"; no training cost, base models or weights stated on the page.

### Drag-and-Drop LLMs (DnD)
- Paper: https://arxiv.org/abs/2506.16406 (NeurIPS 2025). Project: https://jerryliang24.github.io/DnD . Code: https://github.com/jerryliang24/Drag-and-Drop-LLMs (Apache-2.0; camera-ready code fixes 2025-10-23). Checkpoints/logs: https://huggingface.co/datasets/Jerrylz/DnD-checkpoints-and-logs ; README says you must agree to a license on the website before downloading the released LoRA adapters (login-like friction, unverified how enforced).
- Architecture: frozen Sentence-BERT (all-MiniLM-L6-v2) over unlabeled prompts of the target dataset -> cascaded hyper-convolutional decoder -> full LoRA tensors. Condition = prompts only; adding answers hurt (ARC-c 51.6 -> 27.0).
- Base models: Qwen2.5-0.5B (common sense), 1.5B and 7B (math/code), VL-3B. Generation time on one A100-80GB: 0.11 s (0.5B) to 0.73 s (7B code).
- Reported gains are vs training LoRAs (the page I read says average, abstract says "strongest"; unresolved): common-sense avg +21.0, HumanEval 17.6 -> 32.7, GSM8K 42.9 -> 66.3 (base 64.4). No comparison to a multi-task LoRA that I could see. With only 2 training datasets / 5 test datasets the delta was -1.4. Tested only up to 7B.
- Generator parameter count and training GPU-hours: not reported.

### HyperCLIP
- https://arxiv.org/abs/2412.16777 (Akinwande et al., CMU/Bosch). Transformer hypernetwork reads the set of class-text embeddings and outputs only normalization scale/bias of a small image encoder (8.8K-67.6K adapted params). Trained on ~100M filtered DataComp pairs, 128M samples, 4x RTX A6000, batch 1500; hours not reported.
- Gain vs same-recipe SigLIP: IN-1K +1.0 to +3.3, CIFAR-100 -0.9 to +5.6. Removing the transformer from the hypernetwork costs ~1.7 pts. Throughput overhead up to 47.7% (EdgeNext). No code link in the paper text.

### Text -> classifier weights, pre-LLM
- Elhoseiny, Saleh, Elgammal, "Write a Classifier", ICCV 2013: https://openaccess.thecvf.com/content_iccv_2013/html/Elhoseiny_Write_a_Classifier_2013_ICCV_paper.html ; journal version https://ar5iv.labs.arxiv.org/html/1601.00025 (arXiv 1601.00025). TF-IDF of Wikipedia text -> classifier params. Flowers AUC 0.68 vs domain-transfer 0.62; Birds 0.62 vs 0.59. Kernel variant MAU 9.1% flowers / 3.4-5.4% birds. Authors say code and splits are on their project pages (URL not captured).
- Ba, Swersky, Fidler, Salakhutdinov, ICCV 2015: https://arxiv.org/abs/1506.00511 . Text -> fc and conv weights of a CNN. Unseen-class AUC CUB-2010 0.82 (fc) vs 0.59-0.66 for DA baselines; Flowers 0.71 vs 0.62-0.68. Top-1 on CUB ~10%. Training 1-2 h (fc) to ~4 h (fc+conv) on a GTX Titan. No code release mentioned.

### Hypernetworks for few-shot (support set -> weights)
- HyperNetworks (Ha et al. 2016): https://arxiv.org/abs/1609.09106 (concept only).
- HyperTransformer (Zhmoginov et al., ICML 2022): https://arxiv.org/abs/2201.04182 . Generating only the last layer is "competitive or better" for larger targets. Code (TF, image datasets Omniglot/miniImageNet/tieredImageNet): https://github.com/google-research/google-research/tree/master/hypertransformer ; license not shown on the page.
- HyperShot https://arxiv.org/pdf/2203.11378 and HyperMAML https://arxiv.org/pdf/2205.15745 : image few-shot; numbers not fetched.
- LEOPARD (Bansal, Jha, McCallum, COLING 2020): https://arxiv.org/abs/1911.03863 . Transformer + parameter generator that predicts output-layer weights from per-class support data; 17 NLP classification tasks; +14.5% average relative gain at 4 shots over baselines. No official repo found.
- Hyperdecoders (Ivison & Peters, EMNLP Findings 2022): https://arxiv.org/pdf/2203.08304 , code github.com/allenai/hyperdecoders (stated in paper per search result, repo not opened). Per-instance adapters; 73.6 vs T2L 73.9 in T2L's table.
- HyperFast (tabular): https://github.com/AI-sandbox/HyperFast . Pretrained meta-model generates an MLP classifier in one forward pass, sklearn-style API, CPU fallback, weights auto-download. License CC BY-NC 4.0. Successor iLTM (Nov 2025) mentioned. Could in principle be fed 384-d sentence embeddings as tabular features (untested).

### Baselines that the literature says are hard to beat
- Chen et al., "A Closer Look at Few-shot Classification" (ICLR 2019): https://arxiv.org/abs/1904.04232 . Baseline / Baseline++ (cosine classifier) competitive with meta-learners, esp. with deeper backbones.
- Tian et al., "Rethinking Few-Shot ... Good Embedding Is All You Need?" (ECCV 2020): https://arxiv.org/abs/2003.11539 . Linear (logistic) classifier on frozen embedding matches/beats meta-learning; L2-normalizing features makes LR ~ nearest-neighbour.
- Label-name use for intent detection: NLI-FSL https://arxiv.org/abs/2112.07434 (class-label names as NLI hypotheses). Banking77 macro-F1, novel-class FSID: 74.31 (1-shot), 79.79 (5-shot); CLINC150 79.20 / 86.55. Useful as an external reference for "label names help".
- Similarity-based unsupervised text classification (embedding vs label): Schopf et al. https://arxiv.org/abs/2211.16285 . Similarity-based beat zero-shot NLI in most cases; SBERT/SimCSE embeddings best. Per-dataset numbers not fetched.
- SetFit (few-shot, fine-tunes a sentence transformer then LR head): https://arxiv.org/abs/2209.11055 , https://github.com/huggingface/setfit . Reference point, heavier than head-only.

### Public data usable without login
- Banking77: https://huggingface.co/datasets/PolyAI/banking77 , CC-BY-4.0, 10,003 train / 3,080 test, 77 intents. Repo still uses a loading script (viewer disabled), so with recent `datasets` versions load the auto-converted parquet (`refs/convert/parquet`) (unverified that it exists).
- CLINC150: https://huggingface.co/datasets/clinc/clinc_oos , CC-BY-3.0, 150 intents / 10 domains + OOS; configs small/imbalanced/plus (7.6k-15.25k train, 3.1k val, 5.5k test).
- Super-NaturalInstructions: https://github.com/allenai/natural-instructions , Apache-2.0 for the repo; per-task instance licenses vary; 1,600+ tasks as JSON; public, no login.
- CrossFit / NLP Few-shot Gym (160 tasks): https://github.com/INK-USC/CrossFit . No license stated in README; build is `datasets==1.4.0`, ~3 h, some data via Google Drive (quota errors). Friction high; only if we want broad task diversity.
- Kaggle: nothing here needs it.

## Reusable components

| name | url | license | why |
|---|---|---|---|
| all-MiniLM-L6-v2 (22.7M params, 384-d) | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | Apache-2.0 | Frozen encoder for the CPU smoke; same family DnD uses as its condition encoder |
| Text-to-LoRA code | https://github.com/SakanaAI/text-to-lora | Apache-2.0 | Reference for description generation (128 per task), SFT-vs-reconstruction, hypernet layout; weights not reusable for our students |
| T2L pretrained generators | https://huggingface.co/SakanaAI/text-to-lora | not stated on page checked | Only if we want a demo of LoRAs for Gemma-2-2B / Llama-8B / Mistral-7B |
| DnD code + checkpoint dataset | https://github.com/jerryliang24/Drag-and-Drop-LLMs | Apache-2.0 (code); adapters need website license agreement | Reference for "prompts -> weights" decoder; Qwen2.5-0.5B setting is smallest |
| SetFit | https://github.com/huggingface/setfit | Apache-2.0 (unverified on page; repo standard) | Few-shot baseline and ready sentence-embedding + LR pipeline |
| HyperFast | https://github.com/AI-sandbox/HyperFast | CC BY-NC 4.0 | Pretrained "support set -> classifier" generator with CPU path; NC license limits product use |
| HyperTransformer | https://github.com/google-research/google-research/tree/master/hypertransformer | not shown | Reference implementation of support-set -> last-layer weights; TF + images |
| scikit-learn LogisticRegression / numpy | n/a | BSD | The required baselines; no new dependency |
| Banking77, CLINC150, Natural-Instructions | see Public data | CC-BY-4.0 / CC-BY-3.0 / Apache-2.0 | Meta-train/meta-test class pools for episodic smoke |

## Numbers

| item | value | date | source |
|---|---|---|---|
| RunPod H100 PCIe 80GB | $1.99/h community, $2.89/h secure | page "Updated 2026-10-08" | https://www.runpod.io/pricing (community/secure labels inferred by fetcher) |
| RunPod H100 SXM | $2.69 / $3.99 per h | same | same |
| RunPod A100 80GB PCIe / SXM | $1.19 / $1.39 community; $1.79 secure | same | same |
| RunPod RTX 4090 24GB | $0.34 community / $0.89 secure | same | same |
| RunPod RTX A6000 48GB | $0.33 / $0.59 | same | same |
| RunPod L40S 48GB | $0.79 / $1.09 | same | same |
| T2L SFT, paper figure | ~10 h on 1 H100 = $20-27 (community PCIe/SXM) | 2025 | https://arxiv.org/html/2506.06105v2 |
| T2L SFT, README figure | ~5 days on 1 H100 = 120 h = $239-323 community, $347-479 secure | n/a | https://github.com/SakanaAI/text-to-lora |
| T2L reconstruction, 479 LoRAs | ~10 h; 9 LoRAs ~10 min (1 H100) | 2025 | arXiv HTML App. H |
| T2L checkpoint repo size | 1.72 GB total (4 dirs) | 2026-10-17 | HF tree page |
| T2L hypernet params | L 55.25M / M 34.28M / S 4.92M | 2025 | arXiv HTML App. F |
| DnD weight generation | 0.11-0.73 s per adapter on A100-80GB | 2025 | https://arxiv.org/abs/2506.16406 |
| HyperCLIP training | 4x RTX A6000, 128M samples, batch 1500 | 2024 | https://arxiv.org/abs/2412.16777 |
| Ba 2015 training | 1-2 h (fc) / ~4 h (fc+conv), GTX Titan | 2015 | https://ar5iv.labs.arxiv.org/html/1506.00511 |
| MiniLM-L6 | 22.7M params, 384-d, 256-token default truncation | 2026-10-17 | HF model card |
| Banking77 / CLINC150 sizes | 13,083 utterances / 59,275 rows (all configs incl. OOS) | 2026-10-17 | HF cards |

## Blockers / plan changes

1. No reusable generator for our setting. T2L/DnD/Doc-to-LoRA write LoRAs for specific 0.5B-8B LLMs. Mission 2's "hypernetwork writes student weights" should be scoped to head-only generation over frozen embeddings on CPU. Full T2L-style SFT is at best $20-40 on an H100 (10 h) and $239+ if the README's 5 days is right; do not plan on it with ~$20 RunPod.
2. The obvious baseline is nearly as good. T2L beats a same-data multi-task LoRA by 0.4-2.0 pts. For the roadmap, treat "shared student conditioned on step id / schema" as the default and the hypernetwork as an optional extra, not a gate.
3. Too few tasks. Our demo has ~5 step types; hypernetworks need tens to hundreds of training tasks (T2L: 64 tasks -> 66.0, 479 -> 67.7). Manufacture episodes from public multi-class data (random class subsets of Banking77/CLINC150, plus Natural-Instructions classification tasks) rather than from our own steps; hold out whole classes/domains for meta-test.
4. Description sensitivity. Wrong or vague descriptions cost 9-21 pts in T2L. Any generated student must go through shadow + certification, and the description (or label set) must be part of the logged/hashed input.
5. Replay/hashing: generated weights are a deterministic function of (generator hash, description, embedder hash) only if float ops are pinned; store and hash the generated head as an artifact rather than regenerating at replay time. (My inference; not from a source.)
6. Licenses: HyperFast CC BY-NC 4.0; TabPFN-2.5 weights non-commercial (search result via https://docs.priorlabs.ai/models; TabPFN-2 described as commercial-friendly, unverified); T2L paper CC BY-NC-SA but code Apache-2.0; DnD adapter download requires accepting terms on its website.
7. Dataset loaders: PolyAI/banking77 uses a deprecated loading script; CrossFit pins `datasets==1.4.0`. Budget half an hour of data plumbing.

## Unverified

- No source found for label-embedding zero-shot accuracy of MiniLM/bge on Banking77/CLINC150 with natural-language label names. A search snippet showed "ZS-DDN-SBERT 39.06 / 30.03" for Banking77 (Euclidean to label name) but I could not open the table to confirm what the columns mean. This is smoke #1 to measure ourselves.
- CPU embedding throughput and total episodic training time are my estimates (tens of thousands of short utterances embedded in minutes; MLP generator trains in minutes), not measured here.
- Whether the expected result holds (hypernetwork gains concentrated at 0-2 shots, parity at >=5) is an inference from Chen 2019 / Tian 2020, not a measured result for text.
- T2L 5-days (README) vs ~10 h (paper) discrepancy; per-file checkpoint sizes; whether pretrained hypermod.pt run acceptably on CPU with gemma-2-2b-it.
- DnD: whether the baseline is the strongest or the average LoRA; license agreement mechanics; generator size and training time.
- HyperCLIP, Ba 2015, LEOPARD: no code found. Elhoseiny code URL not captured. HyperTransformer license not shown. HyperShot/HyperMAML/Schopf numbers not fetched.
- HWU64 availability/license not checked. HyperFast/TabPFN on sentence embeddings untested.
- Doc-to-LoRA weights and training cost not stated on the pages read.

## Smallest CPU version (< 1 hour) and the baselines it must beat

Setup (all on cached embeddings, no GPU):
1. Embed once with all-MiniLM-L6-v2: all Banking77 + CLINC150 utterances, and the label names / one-sentence label descriptions. L2-normalize.
2. Split classes, not rows: e.g. CLINC150 domains 7 train / 3 test; Banking77 50 train / 27 test classes. Report on held-out classes only.
3. Episode: sample N classes (5-20), K support utterances per class (K in {0,1,2,5,10}), a query set.
4. Generator: w_c = normalize(a*e_c + b*p_c + MLP([e_c, p_c, e_c*p_c])) with e_c = label-text embedding, p_c = support prototype (zero vector when K=0), logits = s * cos(x, w_c) with learnable temperature s. ~100-300k params, so mathematically it contains the three baselines below as special cases and must learn something beyond them.
5. Train with Adam on thousands of episodes; minutes on CPU. Optionally add a second head variant conditioned only on step schema text, to mirror the Dreaming step types.

Baselines (same embeddings, same episodes, many seeds, report mean +/- CI):
- B0 label-embedding zero-shot: w_c = e_c. The K=0 bar.
- B1 k-shot prototypes: w_c = mean of support embeddings (cosine / nearest-centroid).
- B2 label-embedding + prototype blend with a single tuned scalar (the strongest "obvious" non-learned baseline).
- B3 logistic regression on the K-shot support (L2-normalized features, C tuned on a held-out episode set), optionally initialized/regularized toward e_c.
- B4 kNN, and full-data LR on the held-out classes as the ceiling.
- Optional heavier reference: SetFit on the same shots.

Pass criteria for the smoke: beats B2 and B3 on held-out classes at K in {0,1,2} by more than the seed CI; at K>=5 report parity honestly. If it only ties B2, drop the hypernetwork and ship "label-embedding init + LR" as the student-writing mechanism, which needs no meta-training.
