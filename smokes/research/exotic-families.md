# Exotic model families for few-label typed steps (research, 2026-10-17)

Method: WebSearch + WebFetch only, 2024-2026 first. "Fetched" = numbers read on the page named; "snippet" = search-result text only (see Unverified). Reference points from this repo: frozen bge-small + LR on S23 tasks (5 tasks, 12-15 labels) = 0.802 / 0.925 / 0.968 accuracy and 0.14 / 0.60 / 0.91 certified coverage at k = 1 / 3 / 10 rows per label; S2b synthetic prior (arm 2) = 0.914 / 0.943 / 0.969 and 0.80 / 0.87 / 0.95; Banking77 (77-way) LR 60% at 1 row/label, 82% at 10; teacher flip floor 2-3% (S13); logits already rank errors at AUROC 0.88-0.90 (S15).

**Headline.** I found no exotic family with a published number showing fewer labels than frozen embeddings + LR on text classification. The only direct positive few-label text claims are StructCBM (ICLR 2026, abstract only) and LLM-written-data tricks already in S2b. The families are worth smoking for the *other* requirements: certificate scores, byte-exact replay, auditability. Six cheap smokes at the end; all but one run on CPU or under $0.15.

## What "Cocomo" most likely is
1. **CoCoMix, "LLM Pretraining with Continuous Concepts"** (Meta FAIR/KAIST/UCSD, arXiv 2502.08524, ICLR 2026). Fetched: a pretrained TopK SAE (GPT-2 124M, 32,768 concepts, 32 active) gives concept labels; the model predicts the top-4 attributed concepts per token and mixes a concept vector back into its hidden state; the 1.38B model matches next-token training with 21.5% fewer tokens (69M / 386M / 1.38B tested). Code: github.com/facebookresearch/RAM/tree/main/projects/cocomix (from the paper, not opened). Pretraining only; no classification or fine-tuning result.
2. Other spellings, in order of plausibility: CoCoMo (Chang 2023, arXiv 2304.02438, "Computational Consciousness Modeling", prompt-engineering framework, irrelevant); Coconut (Meta, 2412.06769, COLM 2025, chain of continuous thought, reasoning not classification); LoCoMo (arXiv 2402.17753, ACL 2024, long-term conversational memory benchmark, fits "structural memory" but is an agent-memory benchmark); COCOMO (Boehm software cost model, unrelated). Ask the user which; CoCoMix is the only one that is a data-efficiency method with SAE concepts.
3. "Structural memory": no paper by that name found. Closest: StructMem (2604.21748, ACL 2026, structured agent memory on LoCoMo), BDH's synapse graph (below), structural plasticity in sparse SNNs (2510.19764). For a typed classifier, a structured memory of logged rows is the kNN/Hopfield family already covered.

## Families (mechanism / best 2024-26 text evidence / cost vs LR head / replay / native uncertainty / code / smoke today)

### BDH, Baby Dragon Hatchling (Pathway)
- Mechanism: scale-free graph of "neuron particles" with sparse positive activations; working memory at inference is Hebbian synapse strengthening (attention as state-space with plastic synapses). arXiv 2509.26507.
- Evidence: language and translation at GPT-2 parity, 10M-1B (abstract; no classification, no few-shot). Follow-up BDH-CQ (2608.09888, fetched): 150M model, 29.5% pass@2 on ARC-AGI-1 at $0.0007/task via in-context demonstrations updating recurrent memory; grid puzzles, not text labels. Text: none.
- Cost: train from scratch (repo trains tiny Shakespeare; no official checkpoint found, community 25M Nepali/Shakespeare models only), orders of magnitude above an LR head. Replay: float GPU, same as any net. Uncertainty: none native. Code: github.com/pathwaycom/bdh (MIT, no pip).
- Smoke today: no (nothing pretrained to probe). Revisit if Pathway ships a checkpoint; the interesting test would be demonstrations-as-training (k rows in context, zero gradient steps), which TabICL-style heads already approximate.

### Spiking neural networks and SNN adapters
- Mechanism: binary spike activations over T time steps, energy savings on neuromorphic hardware.
- Evidence (fetched): SpikingLM (ICML 2026, github.com/hamings1/SpikingLM): BERT-base-size, 109.5M params, T=4, GLUE average 77.1 (SST-2 90.1, MNLI 75.8/76.0, RTE 61.7), needs full MLM pre-training and full fine-tuning; no few-shot claim. SpikeBERT (2308.15122): six-dataset average 80.20. SpikeLoRA (SSRN, Jan 2026, snippet): -35% power, -0.08% GLUE vs LoRA. Label efficiency: none.
- Cost: more than LR head (full fine-tune plus T-step unrolling; GPU gives no energy win). Replay: surrogate-gradient float training, spikingjelly needs CuPy; no determinism notes. Uncertainty: spike counts, not calibrated. Smoke: no, it is an energy argument and we pay for neither joules nor neuromorphic chips.

### JEPA for text (LLM-JEPA, T-JEPA, VL-JEPA)
- Mechanism: add a loss d(Pred(Enc(view A)), Enc(view B)) (cosine) to the usual token loss so embeddings of two views of the same item predict each other; no reconstruction in token space.
- Evidence (fetched, arXiv 2509.14252, ICLR 2026): needs paired views (text/regex, text/SQL; paraphrases for RottenTomatoes). Accuracy NTP vs LLM-JEPA: RottenTomatoes 56.57 vs 57.76, NL-RX-SYNTH Llama-3.2-1B 57.29 vs 71.46, GSM8K 32.36 vs 36.36, Spider 47.52 vs 50.55; about 2x training compute, no inference cost; no explicit low-data claim. T-JEPA (2410.05016) is tabular; VL-JEPA (2512.10942) is vision-language; Pantagruel (2601.05911) is French encoders. Text few-label: none.
- Cost: 2x a fine-tune; replay as any fine-tune; no uncertainty. Code: LLM-JEPA repo (URL differs by source, see Unverified).
- Smoke: fold into the teacher-side-information smoke (item 3 below): views = (input, teacher rationale or label description). Standalone: no.

### CoCoMix and SAE-feature / concept-bottleneck classifiers
- Mechanism: SAE (Gemma Scope 2, sae-lens) turns hidden states into sparse named features; a sparse LR on binarized codes is the classifier. Concept-bottleneck models route the decision through named concepts scored by an LLM or by similarity.
- Evidence: Kantamneni et al., ICML 2025 (2502.16681, fetched): Gemma-2-9B layer 20 with Gemma Scope SAEs vs LR/PCA/kNN/XGBoost/MLP baselines at n = 2..1024; "SAEs simply underperform baselines in these settings"; adding SAEs to the probe set gave no average gain. Gallifant et al. (2502.11367, abstract): SAE features beat hidden-state and BoW baselines at macro F1 > 0.8 on safety tasks, binarization helps; no low-data experiment. StructCBM (ICLR 2026, abstract only): similarity-only few-shot text CBM, "with as few as 10 training samples" beats prior CBMs and is "comparable to LLMs"; repo github.com/alexiszlf/StructCBM has a bare README, no licence, DeepSeek key file. CB-LLM (ICLR 2025): competitive on full data.
- Cost: SAE route needs a decoder forward (Gemma-3-270M is about 8x bge-small) plus SAE encode, then LR; CBM route needs one LLM call per step to write concepts, then cosine features. Replay: same as the encoder; deterministic after extraction. Uncertainty: from the LR head; concept contributions are an audit trail. Code: `pip install sae-lens`; Gemma Scope 2 exists for Gemma 3 at 270M-27B (cc-by-4.0, widths 16k-1M, L0 small/medium/large).
- Smoke: yes, SAE route as the extra output of the unrun S21 hidden-state probe (item 5); CBM route as item 6.

### Hyperdimensional computing / vector-symbolic architectures
- Mechanism: map features to 10k-dim bipolar or integer hypervectors, bundle per class (centroid), optionally retrain by perceptron updates (OnlineHD, AdaptHD, IntRVFL).
- Evidence: HD/VSA benchmark (Artificial Intelligence Review 2025, 10.1007/s10462-025-11181-2, 7 datasets + 121 UCI, snippet): gains "often matched by off-the-shelf methods". Ayar et al. (abstract): HDC heads on BERT/DistilBERT/GPT-2 reach up to 83% on IMDb at about 9 KB. No Banking77/CLINC/few-shot text result found.
- Cost: at or below an LR head (one pass plus a few refinement epochs, D x C multiply at inference). Replay: strongest of the exotic families, scores are integer sums, so the head is bit-exact across hardware given identical input bytes; the float embedding still is not. Uncertainty: cosine margin, not calibrated. Code: `pip install torch-hd` (MIT, 390 stars; classifiers Vanilla, AdaptHD, OnlineHD, NeuralHD, DistHD, CompHD, SparseHD, QuantHD, LeHDC, IntRVFL).
- Smoke: yes, CPU, 15 min (item 4).

### Tsetlin machines
- Mechanism: teams of automata learn conjunctive clauses over Boolean features; class score is an integer vote sum.
- Evidence: 2606.19815 (cluster pre-trained TM, 5 unnamed datasets, "competitive with BERT", no numbers in abstract) and 2604.12223 (ACL Findings 2026, LLM-written sub-intents give synthetic data, "comparable to BERT", no numbers). No few-label numbers; TMs need Boolean features (BoW), so they cannot use the frozen embedding that makes LR strong.
- Cost: training slower than LR (clause epochs), integer-only inference. Replay: excellent (integers). Uncertainty: native vote margin; clauses are human-readable. Code: github.com/cair/tmu (licence not read; MSVC build on Windows).
- Smoke: possible on CPU in 20 min but expected to lose at k < 10; skip unless an interpretable integer-only student is wanted.

### Reservoir computing, extreme learning machines, random features
- Mechanism: fixed random expansion, closed-form ridge readout (kernel ridge in the limit).
- Evidence: none for 2024-26 few-shot text (searched). 2026 ridge-feature-transfer paper (2606.05626, snippet) shows ridge probes with random features help continual attribution; Wieting & Kiela 2019 random encoders are the old baseline.
- Cost: cheaper than LR (closed form). Replay: deterministic given seed. Uncertainty: exact leave-one-out residuals from the hat matrix, usable as free out-of-sample scores. Code: scikit-learn RidgeClassifierCV, torchhd IntRVFL. Smoke: yes, merged into item 1.

### Liquid / CfC networks
- Evidence: time-series papers only; no text few-label result found (an aggregator lists CfC IMDb 88.4, unverified). Skip.

### Product-key and sparse memory layers
- Mechanism: huge trainable key-value table with factored product-key lookup replaces FFNs at fixed FLOPs.
- Evidence: Memory Layers at Scale (2412.09764): beat dense models with 2x compute, gains largest on factual QA. Sparse Memory Finetuning (2510.15103): updating only highly activated slots cuts NaturalQuestions F1 loss to 11% vs 89% full fine-tune and 71% LoRA; this is about forgetting, not labels. Text classification few-label: none. Cost: pretraining-scale. Skip.

### Neural Turing / differentiable memory
- Evidence: Dynamic Memory Induction Networks (2020) and Induction Networks (2019) only; no 2024-26 text result found. Superseded by prototypes plus kNN already covered. Skip.

### Hebbian fast-weight learners
- Mechanism: outer-product fast weights bind label to feature in one shot (Munkhdalai & Trischler 2018).
- Evidence: 2026 ViT work (2605.02920) improved only the Swin variant, +0.3 / +0.5 at 1 / 5 shot on Omniglot, ViT/DeiT got worse; 2510.21908 Hebbian plasticity in transformers helps CIFAR-FS/Omniglot few-shot. No frozen-embedding text result. Math fact: outer-product memory with an inverse-covariance preconditioner is LDA/ridge, hence item 1.

### Compression-based classifiers (gzip NCD)
- Evidence (fetched): Opitz (2307.15002 v5): original gzip kNN used optimistic tie-breaking; realistic mean accuracy 0.730 not 0.820; a bag-of-words distance kNN scores 0.744 and wins at 5 shots (KirNews 0.451 vs 0.350) and 100 shots, at 7.9 s vs 244.9 s (AGNews) and 316 s vs 57,073 s (DBpedia). Neural NCD (2410.15280, fetched): RWKV-169M compressor, 5-100 shots, beats gzip on AGNews, loses on DBpedia, accuracy not predictable from compression rate, O(n^2) cost. Skip: slow at inference (per-query pass over training set), no gain over embedding kNN.

### NTK / infinite-width kernels
- Evidence: Malladi et al. ICML 2023 (2210.05643): prompt-based fine-tuning of RoBERTa-base at 16/64/512 shots per label often follows kernel dynamics; no 2024-26 text few-label result found. Features are per-parameter gradients times classes, heavy at 77 labels; GP heads already covered. Skip.

### Energy-based models and evidential deep learning (Dirichlet outputs)
- Mechanism: EDL heads output Dirichlet evidence, so total evidence is a native "I do not know"; energy = -logsumexp(logits).
- Evidence (fetched): EDL knowledge distillation (2507.18366): Mistral-7B LoRA Dirichlet students on Amazon, SST-2, Yahoo, YouTube; accuracy 0.958 / 0.954 / 0.610 / 0.900; ECE 0.011 / 0.017 / 0.042 / 0.097 vs teacher 0.021 / 0.029 / 0.194 / 0.031; OOD AUROC 0.96 total entropy; one forward pass, 11-36x faster than the prompt-ensemble teacher; code github.com/Harsha1969/BPE-KD. Critiques: NeurIPS 2024 (2402.06160) epistemic uncertainty is unreliable and EDL behaves like an energy-based OOD detector; ICML 2024 (2402.09056) same direction. Energy score is much worse than max-softmax for in-distribution selective classification (2403.14715, snippet).
- Cost: same as LR head (loss swap). Replay: same. Uncertainty: native, but ranking vs top-2 gap is the open question. Smoke: yes (item 2).

### Bayesian program learning, KAN, capsules, quantum-inspired tensor networks
- BPL: LLM hypothesis search reaches human-like rule induction (2509.01016) and neural-Bayesian programs exist for dialogue intents (2410.06190), but no text-classifier benchmark; symbolic rule students already covered. Skip.
- KAN heads: FourierKAN head beats MLP head by about 10 / 11 points acc / F1 over 7 frozen transformers on 4 tasks (NeurIPS 2024 workshop, snippet), Burmese study (2511.21081) competitive; no low-data ablation, more head parameters hurt at k < 10. Skip.
- Capsules: Induction Networks 2019 only. Tensor networks: MPS/TextTN papers, binary or SST-5, no few-label benchmark (2604.14287 survey unread). Skip.

## Other ideas the search suggested (not in the user's list)
- **Unlabelled-stream whitening plus closed-form generative head** (item 1): the stream gives the total covariance for free; class means need only k rows.
- **Teacher side-information** (item 3): the labelling call can also emit tags or a one-line reason; Distilling Step-by-Step (2305.02301, fetched) reports over 50% fewer examples on average and up to 85% less data (T5, generative); nobody tested an encoder classifier.
- **Dirichlet target from K teacher samples** (item 2 variant): S13 flips concentrate on hard prompts, so the empirical vote distribution is a free label-noise estimate; S13 holds only 100 prompts x 5 repeats, so a new run of about 1,000 rows x 5 calls is needed ($0.4 at haiku prices).

## Top 6 to smoke (expected information per dollar; protocol = S23: 5 tasks x 3 draws, k = 1/3/10 rows per label, baselines arm 1 and arm 2, accuracy plus certified coverage at 5%; reuse s23_new_architectures/lib.py)
1. **Hebbian / shrinkage-LDA / ridge head with unlabelled whitening.** CPU, 15 min, $0. Cache bge-small embeddings, center and ZCA-whiten on the whole unlabelled train text, then (a) nearest class mean, (b) sklearn LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), (c) RidgeClassifierCV with leave-one-out, (d) same with S2b synthetic centroids as class-mean prior. Pass: at least +3 points over arm 1 at k = 1-3, or equal accuracy with a certificate from 100 calibration rows using leave-one-out scores (do not use selected rows for the final certificate). Kill otherwise.
2. **Dirichlet / energy / Mahalanobis certificate score vs top-2 gap.** CPU, 20 min, $0 (+$0.4 for the K-sample variant). Train an EDL head (softplus evidence, KL-regularised loss) on the same embeddings; scores: top-2 gap, max softmax, energy, whitened Mahalanobis distance (from item 1), Dirichlet strength. Test on a split with 20% novel inputs (held-out CLINC intents or S4 novel merchants) and on in-distribution; metric = error-detection AUROC and certified coverage at 5%. Pass: beats top-2 gap by 5 coverage points when novel inputs are present without losing in-distribution.
3. **Teacher side-information as auxiliary supervision** (CoCoMix / step-by-step / JEPA-lite in one). GPU ~15 min or CPU, about $0.10 total. Only the k x C labelled rows need extras: Haiku-5.5 (reasoning off) writes label + 3 concept tags + a 15-word reason; fine-tune bge-small (S8 recipe, 17 s per 770 rows) with CE + multi-hot tag head + cosine loss to the reason embedding (lambda in {0.5, 1, 2}); drop aux heads at inference. Compare with S8 fine-tuned encoder and arm 2. Pass: +3 points at k = 1-3 over the plain fine-tuned encoder.
4. **HDC / IntRVFL integer head (torch-hd).** CPU, 15 min, $0. Quantise embeddings to sign or int8, D = 10,000, run Vanilla, OnlineHD, IntRVFL; also run the head on CPU vs GPU, 1 vs 8 threads, 1000 repeated calls, and check identical score bytes (S7 found float bytes move with thread count). Pass: within 2 points of LR at k >= 3 and bit-identical across profiles; value is replay, not labels.
5. **SAE codes vs dense hidden-state probe on Gemma-3-270M** (rides on the unrun S21 extractor). 4090 or the $0.27/h pod, 25 min, about $0.12. Mean-pool a mid-depth layer, Gemma Scope 2 resid_post 16k small-L0 via sae-lens, binarise codes (Gallifant), L1-LR; compare dense probe, SAE probe, bge-small LR at k = 1/3/10 and the certificate. Prior: loses (Kantamneni); worth it to close the door and to see whether named features give usable audit text. If the 270M SAE repo is absent, use gemma-scope-2-1b.
6. **Concept-bottleneck few-shot (home-made StructCBM).** CPU plus one LLM call, 20 min, about $0.01. LLM writes 50-100 concept phrases per step (same call pattern as S2b); features = cosine(bge-small(text), bge-small(concept)); sparse LR over concept scores + prototype branch. Expect about arm 2, but the decision is auditable by concept. Optional: run github.com/alexiszlf/StructCBM if the repo works (needs a DeepSeek key, 40 min).

## Skip, and why
- BDH: no pretrained checkpoint, no classification result, train-from-scratch cost; revisit if weights ship.
- Spiking nets and SpikeLoRA: energy argument only; SpikingLM is BERT-base size at GLUE 77.1 and needs MLM pre-training; no label-efficiency claim.
- Standalone JEPA / CoCoMix: need paired views or pretraining; the useful kernel is in item 3.
- Tsetlin: integer-exact and readable, but Boolean inputs forfeit the frozen embedding, no few-label numbers; reconsider only for an interpretable integer-only student.
- Liquid/CfC, capsules, NTM/MANN, tensor networks, BPL: no 2024-26 text few-label evidence.
- KAN heads: no low-data evidence, extra parameters hurt at k < 10.
- gzip/neural NCD: slower than embedding kNN, tie-break inflation (Opitz), BoW beats it.
- Memory layers: pretraining-scale; sparse memory finetuning is about forgetting.
- NTK: gradient-feature kernels too heavy at 77 labels; GP heads cover the idea.
- Energy score as a replacement for max-softmax: worse for in-distribution selective classification; only useful as OOD signal (kept inside item 2).

## Unverified
- All numbers labelled "snippet": HD/VSA AI Review 2025 ("often matched by off-the-shelf methods", Springer page redirected, paywalled); Ayar et al. HDC 83% IMDb at 9 KB (abstract; I did not check baselines); SpikeLoRA SSRN figures; FourierKAN +10% / +11%; CfC IMDb 88.4; energy-vs-MSP result from 2403.14715; 2606.05626 ridge/random-feature result.
- StructCBM: abstract only; no datasets, no accuracies; repo README is a template, licence absent; whether concepts come from LLM calls is inferred from the DeepSeek key file.
- LLM-JEPA repo: sources give both github.com/galilai-group/llm-jepa and github.com/rbalestr-lab/llm-jepa; the paper text gave contradictory forward-pass counts (one extra vs two instead of three) before settling on about 2x; RottenTomatoes accuracy of 57% looks low for a binary task, so I do not know its prompt format.
- CoCoMix: repo path taken from the paper, not opened; whether it fine-tunes on classification is unknown (pretraining only per abstract).
- Gemma Scope 2: the 4B card lists layers at 25 / 50 / 65 / 85% depth, widths 16k-1M, cc-by-4.0, but the card is internally inconsistent about a 270M repo; confirm `google/gemma-scope-2-270m-*` exists before the pod run. Exact release date not found.
- BDH-CQ: abstract-level only; weights availability unknown (repo pathwaycom/arc-task-gen not opened). No official BDH checkpoint found, absence not proven.
- Tsetlin: licence and Windows/CUDA build of tmu unconfirmed; the two 2026 TM papers give no numbers in their abstracts.
- torchhd: fit/predict API and last-commit date not read; `Vanilla` is the centroid classifier, no class named Centroid.
- "Structural memory" and "Cocomo" are my best readings of the user's words, not confirmed.
- The "free leave-one-out calibration" idea (item 1) is my inference from ridge algebra; it has selection-bias risk if hyperparameters are tuned on the same rows, and no source in this note tests it.
- Evidential distillation results (2507.18366) are for 7B LoRA students on 2-10 class tasks, not 77-way or encoder students.
