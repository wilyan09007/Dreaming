# S6 - Context pruner (mission 5): HotpotQA distractor, ablation labels, paragraph scorers

Revised 2026-10-17 after review. The first version claimed "temperature 0" for a teacher that never ran at temperature 0; this file replaces it. `## Revision` at the end lists what changed and how the numbers moved.

## Setup

- **Data:** `datasets.load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")` (no mirror needed). 250 questions from a seed-0 permutation of the 7,405 validation items; 3 items with fewer than 10 paragraphs were skipped while sampling, 0 had a gold-paragraph count other than 2. First 150 = scorer training, last 100 = held-out (split by question). Bridge and comparison questions mixed as sampled (held-out: 79 bridge, 21 comparison).
- **Three teachers, same protocol** (full context -> `A_full`; drop each of 10 paragraphs -> 10 ablation answers; one replicate of the full prompt; pruned re-asks; closed-book control). Prompt: "Answer the question using only the context. Reply with the shortest answer span." + 10 paragraphs ("Title: sentences") + question; `max_tokens=40`.

| tag | model (OpenRouter) | temperature actually applied | routing | mean answer tokens |
|---|---|---|---|---|
| `nano_t1` | `openai/gpt-5.4-nano`, reasoning off (0 reasoning tokens over 4,631 calls) | **default (not 0).** `temperature=0` was sent but the model has no endpoint that supports it: OpenRouter's endpoint metadata for the OpenAI and Azure endpoints lists `seed` and `reasoning`, not `temperature`; with `provider.require_parameters=true`, `temperature=0` returns 404 "No endpoints found that can handle the requested parameters" for reasoning disabled, default and effort `none` (probed 2026-10-17). Without the flag it is silently dropped. | OpenAI (recorded for the 215 later calls; the original 4,416 did not store the provider) | 9.5 |
| `qwen_t0` | `qwen/qwen3-32b`, `/no_think` suffix | 0, enforced with `require_parameters=true` (a dropped parameter is a 404, not a silent default) | DeepInfra 4,477 calls, Tenstorrent 24, SiliconFlow 17 (not pinned) | 13.3 |
| `gemini_t0` | `google/gemini-3.5-flash-lite` (0 reasoning tokens) | 0, enforced with `require_parameters=true` (only the Google AI Studio endpoints support it) | Google AI Studio 4,428 of 4,428 | 4.4 |

  The spec asked for gpt-5.4-nano at temperature 0, which OpenRouter cannot provide. `nano_t1` is the original run (the spec model at the provider's default temperature); the two T=0 teachers were added after the noise showed up, so they are post-hoc substitutes, not the spec model. Mean full prompt 1,327 / 1,417 / 1,384 tokens.
- **Ablation labels:** "matters" iff the answer to the pruned prompt differs from `A_full`. Two definitions: **strict** = SQuAD-normalized exact match (lowercase, strip punctuation and articles; the spec definition) and **judge** = strict-different pairs are sent to `google/gemini-3.5-flash-lite` (temperature 0, `require_parameters`, prompt: same answer? extra words/formatting do not matter, different entity/value/claim or "unknown" vs a real answer is NO). Judge outputs that do not start with YES or NO count as NO. The earlier F1/containment "relaxed" rule is dropped (audit under Caveats).
- **Scorers (held-out 100):** (1) **learned** (spec): StandardScaler + LogisticRegression on 6 per-paragraph features (cosine of bge-small embeddings, position, word length, question content-word overlap, title in question, title-token overlap), trained on the 150 training questions' strict ablation labels of that teacher (1,500 rows); (1b) **learned (gold-trained)** (control, new): same model trained on the dataset's supporting-fact labels, so it uses information a real deployment does not have; (1c) **learned+rel** (extra): learned plus within-question relative cosine/overlap and rank; (2) zero-shot `cross-encoder/ms-marco-MiniLM-L-6-v2` on (question, "Title: text"); (3) **oracle** = the 2 gold paragraphs, remaining slots filled by seeded random; (4) **random** (seed 0). Top-k (k = 2, 3, 4) kept in original order and re-asked.
- **Metrics:** agreement = pruned answer vs `A_full` of the same question (strict, or judge). "Noise floor" = the second full-context call vs `A_full`. Tokens removed = 1 - sum(pruned prompt tokens)/sum(full prompt tokens) from API usage. "UB" = one-sided 95% Clopper-Pearson upper bound on the disagreement rate, n = 100. McNemar = exact paired test on per-question agreement (a-only, b-only discordant counts). All tests are uncorrected for multiplicity.
- **Reproduce:** `python run.py nano_t1|qwen_t0|gemini_t0` (seed 0; resumable from `responses*.jsonl` and `judge_*.jsonl`, which are also the spend ledger; a warm run takes about 20 s and writes `results_<tag>.json` and `pairs_<tag>.json`; stdout is saved as `log_<tag>.txt`). `python judge_audit.py` runs the second-judge check (`judge_audit.json`, `judge_audit_log.txt`). `run_original.py`, `results_original.json`, `log_original.txt` are the superseded first version (relaxed-rule numbers live there); `log_attempt1.txt` is its first, crashed invocation. `responses.jsonl` (nano) has 4,631 lines = 4,416 original + 100 closed-book + 115 pruned re-asks for the gold-trained scorer; qwen has 4,518 and gemini 4,428.

## Results

### 1. Teacher noise and ablation-label quality (250 questions unless stated)

| metric | nano (default T) | qwen3-32b (T=0) | gemini-3.5-flash-lite (T=0) |
|---|---|---|---|
| same full prompt twice: answers differ, strict / judge | 0.360 / 0.148 | 0.108 / 0.020 | 0.072 / 0.048 |
| 5 identical calls (first 40 q): mean distinct answers; all 5 identical, strict / judge-equivalent | 2.33; 0.45 / 0.68 | 1.27; 0.75 / 0.90 | 1.30; 0.80 / 0.85 |
| 2 calls with `seed=0` (40 q): identical, strict / judge | 0.675 / 0.875 | 0.800 / 0.925 | 1.000 / 1.000 |
| `A_full` vs gold, all 250: EM / contains / token-F1 | 0.484 / 0.668 / 0.680 | 0.476 / 0.708 / 0.650 | 0.616 / 0.716 / 0.766 |
| `A_full` vs gold, held-out 100: EM / contains | 0.48 / 0.62 | 0.45 / 0.68 | 0.56 / 0.67 |
| closed book (no context), held-out: EM / contains | 0.10 / 0.23 | 0.17 / 0.24 | 0.31 / 0.32 |
| paragraphs that "matter", strict: all / gold / non-gold | 0.389 / 0.554 / 0.347 | 0.364 / 0.602 / 0.304 | 0.218 / 0.564 / 0.132 |
| paragraphs that "matter", judge: all / gold / non-gold | 0.206 / 0.436 / 0.148 | 0.164 / 0.418 / 0.100 | 0.162 / 0.510 / 0.075 |
| questions where no paragraph matters, strict / judge | 0.116 / 0.240 | 0.140 / 0.272 | 0.160 / 0.200 |
| ablation precision / recall vs gold paragraphs, strict | 0.285 / 0.554 | 0.331 / 0.602 | 0.516 / 0.564 |
| ablation precision / recall vs gold paragraphs, judge | 0.424 / 0.436 | 0.511 / 0.418 | 0.628 / 0.510 |
| calls / answers cut at `max_tokens` | 4,631 / 9 | 4,518 / 245 | 4,428 / 92 |

### 2. AUC of each scorer at predicting the ablation label (held-out, 1,000 paragraph rows)

Within-question AUC (what top-k selection uses); pooled AUC in parentheses. The last column is vs the gold label (identical for all teachers).

| scorer | nano strict | qwen strict | gemini strict | nano judge | qwen judge | gemini judge | vs gold |
|---|---|---|---|---|---|---|---|
| learned (spec) | 0.533 (0.522) | 0.726 (0.646) | 0.770 (0.681) | 0.578 (0.562) | 0.787 (0.741) | 0.800 (0.710) | 0.807 (0.819) |
| learned (gold-trained) | 0.605 (0.569) | 0.730 (0.633) | 0.774 (0.678) | 0.698 (0.614) | 0.789 (0.737) | 0.800 (0.709) | 0.908 (0.899) |
| learned+rel (extra) | 0.593 (0.511) | 0.726 (0.580) | 0.764 (0.677) | 0.637 (0.580) | 0.794 (0.667) | 0.796 (0.716) | 0.849 (0.819) |
| cross-encoder | 0.568 (0.530) | 0.698 (0.588) | 0.738 (0.641) | 0.641 (0.563) | 0.734 (0.665) | 0.759 (0.662) | 0.851 (0.828) |
| oracle (gold paragraphs) | 0.659 (0.567) | 0.775 (0.631) | 0.824 (0.695) | 0.783 (0.649) | 0.862 (0.751) | 0.866 (0.773) | 1.000 (1.000) |
| random | 0.526 (0.517) | 0.493 (0.509) | 0.518 (0.499) | 0.516 (0.523) | 0.516 (0.518) | 0.499 (0.497) | 0.509 (0.513) |

Standardized coefficients of the ablation-trained LR (cos, position, n_words, q_overlap, title_in_q, title_tok_overlap): nano +0.062, +0.062, -0.021, +0.020, +0.218, -0.121; qwen +0.360, -0.070, -0.083, +0.024, +0.029, -0.119; gemini +0.697, -0.049, -0.059, -0.052, +0.388, -0.400. Gold-trained: +1.889, +0.023, -0.766, -0.038, +1.147, -0.809.

### 3. Pruned re-ask on the 100 held-out questions

"Both gold kept" = both supporting paragraphs survived pruning. "Contains gold" = normalized gold answer is a word-bounded substring of the answer.

**nano (default temperature).** Full context on these 100: EM 0.48, contains 0.62.

| scorer | k | agree strict | agree judge | EM gold | contains gold | tokens removed | both gold kept | UB disagree strict | UB judge |
|---|---|---|---|---|---|---|---|---|---|
| learned | 2 | 0.39 | 0.51 | 0.27 | 0.37 | 79.8% | 0.36 | 0.69 | 0.58 |
| learned | 3 | 0.43 | 0.58 | 0.33 | 0.41 | 70.5% | 0.47 | 0.65 | 0.51 |
| learned | 4 | 0.48 | 0.63 | 0.37 | 0.47 | 61.1% | 0.55 | 0.61 | 0.46 |
| learned (gold-trained) | 2 | 0.53 | 0.68 | 0.39 | 0.51 | 81.0% | 0.60 | 0.56 | 0.41 |
| learned (gold-trained) | 3 | 0.54 | 0.74 | 0.42 | 0.58 | 73.3% | 0.73 | 0.55 | 0.34 |
| learned (gold-trained) | 4 | 0.60 | 0.77 | 0.46 | 0.65 | 65.5% | 0.82 | 0.49 | 0.31 |
| learned+rel (extra) | 2 | 0.41 | 0.54 | 0.28 | 0.35 | 80.1% | 0.40 | 0.67 | 0.55 |
| learned+rel (extra) | 3 | 0.50 | 0.65 | 0.37 | 0.47 | 71.2% | 0.56 | 0.59 | 0.44 |
| learned+rel (extra) | 4 | 0.53 | 0.69 | 0.37 | 0.50 | 61.6% | 0.68 | 0.56 | 0.39 |
| cross-encoder | 2 | 0.47 | 0.63 | 0.34 | 0.46 | 79.8% | 0.48 | 0.62 | 0.46 |
| cross-encoder | 3 | 0.49 | 0.66 | 0.41 | 0.50 | 71.3% | 0.64 | 0.60 | 0.43 |
| cross-encoder | 4 | 0.60 | 0.75 | 0.41 | 0.55 | 62.2% | 0.73 | 0.49 | 0.33 |
| oracle (gold) | 2 | 0.64 | 0.82 | 0.51 | 0.68 | 80.3% | 1.00 | 0.45 | 0.26 |
| oracle (gold) | 3 | 0.65 | 0.80 | 0.53 | 0.69 | 70.6% | 1.00 | 0.44 | 0.28 |
| oracle (gold) | 4 | 0.65 | 0.80 | 0.52 | 0.66 | 60.1% | 1.00 | 0.44 | 0.28 |
| random | 2 | 0.31 | 0.38 | 0.20 | 0.27 | 78.1% | 0.01 | 0.77 | 0.70 |
| random | 3 | 0.34 | 0.48 | 0.23 | 0.29 | 68.6% | 0.04 | 0.74 | 0.61 |
| random | 4 | 0.42 | 0.53 | 0.32 | 0.38 | 58.9% | 0.10 | 0.66 | 0.56 |
| **full context, 2nd call (noise floor)** | 10 | 0.72 | 0.87 | 0.47 | 0.58 | 0% | 1.00 | 0.36 | 0.20 |

**qwen3-32b (T=0).** Full context on these 100: EM 0.45, contains 0.68.

| scorer | k | agree strict | agree judge | EM gold | contains gold | tokens removed | both gold kept | UB disagree strict | UB judge |
|---|---|---|---|---|---|---|---|---|---|
| learned | 2 | 0.53 | 0.76 | 0.43 | 0.55 | 81.0% | 0.42 | 0.56 | 0.32 |
| learned | 3 | 0.63 | 0.87 | 0.47 | 0.63 | 72.8% | 0.64 | 0.46 | 0.20 |
| learned | 4 | 0.66 | 0.89 | 0.47 | 0.65 | 64.2% | 0.74 | 0.43 | 0.18 |
| learned (gold-trained) | 2 | 0.57 | 0.75 | 0.44 | 0.54 | 80.8% | 0.60 | 0.52 | 0.33 |
| learned (gold-trained) | 3 | 0.66 | 0.85 | 0.47 | 0.62 | 73.1% | 0.73 | 0.43 | 0.22 |
| learned (gold-trained) | 4 | 0.71 | 0.90 | 0.50 | 0.67 | 65.3% | 0.82 | 0.37 | 0.16 |
| learned+rel (extra) | 2 | 0.54 | 0.77 | 0.43 | 0.58 | 81.0% | 0.43 | 0.55 | 0.31 |
| learned+rel (extra) | 3 | 0.60 | 0.82 | 0.44 | 0.61 | 72.7% | 0.57 | 0.49 | 0.26 |
| learned+rel (extra) | 4 | 0.60 | 0.83 | 0.42 | 0.58 | 63.9% | 0.67 | 0.49 | 0.24 |
| cross-encoder | 2 | 0.51 | 0.68 | 0.40 | 0.52 | 79.5% | 0.48 | 0.58 | 0.41 |
| cross-encoder | 3 | 0.55 | 0.75 | 0.40 | 0.54 | 71.1% | 0.64 | 0.54 | 0.33 |
| cross-encoder | 4 | 0.63 | 0.79 | 0.40 | 0.57 | 62.0% | 0.73 | 0.46 | 0.29 |
| oracle (gold) | 2 | 0.69 | 0.91 | 0.61 | 0.75 | 80.0% | 1.00 | 0.39 | 0.15 |
| oracle (gold) | 3 | 0.68 | 0.91 | 0.50 | 0.72 | 70.4% | 1.00 | 0.41 | 0.15 |
| oracle (gold) | 4 | 0.67 | 0.94 | 0.52 | 0.72 | 60.0% | 1.00 | 0.42 | 0.11 |
| random | 2 | 0.25 | 0.38 | 0.19 | 0.26 | 77.9% | 0.01 | 0.82 | 0.70 |
| random | 3 | 0.25 | 0.38 | 0.25 | 0.33 | 68.4% | 0.04 | 0.82 | 0.70 |
| random | 4 | 0.35 | 0.49 | 0.28 | 0.37 | 58.8% | 0.10 | 0.73 | 0.60 |
| **full context, 2nd call (noise floor)** | 10 | 0.89 | 0.99 | 0.48 | 0.68 | 0% | 1.00 | 0.18 | 0.05 |

**gemini-3.5-flash-lite (T=0).** Full context on these 100: EM 0.56, contains 0.67.

| scorer | k | agree strict | agree judge | EM gold | contains gold | tokens removed | both gold kept | UB disagree strict | UB judge |
|---|---|---|---|---|---|---|---|---|---|
| learned | 2 | 0.62 | 0.71 | 0.44 | 0.52 | 80.6% | 0.58 | 0.47 | 0.37 |
| learned | 3 | 0.72 | 0.78 | 0.49 | 0.58 | 71.7% | 0.74 | 0.36 | 0.30 |
| learned | 4 | 0.70 | 0.79 | 0.49 | 0.60 | 62.4% | 0.77 | 0.38 | 0.29 |
| learned (gold-trained) | 2 | 0.63 | 0.69 | 0.43 | 0.52 | 81.2% | 0.60 | 0.46 | 0.39 |
| learned (gold-trained) | 3 | 0.74 | 0.81 | 0.49 | 0.58 | 73.5% | 0.73 | 0.34 | 0.27 |
| learned (gold-trained) | 4 | 0.74 | 0.82 | 0.50 | 0.61 | 65.6% | 0.82 | 0.34 | 0.26 |
| learned+rel (extra) | 2 | 0.62 | 0.72 | 0.44 | 0.52 | 80.6% | 0.59 | 0.47 | 0.36 |
| learned+rel (extra) | 3 | 0.71 | 0.77 | 0.46 | 0.55 | 71.9% | 0.70 | 0.37 | 0.31 |
| learned+rel (extra) | 4 | 0.67 | 0.78 | 0.46 | 0.57 | 62.8% | 0.77 | 0.42 | 0.30 |
| cross-encoder | 2 | 0.58 | 0.69 | 0.40 | 0.47 | 79.9% | 0.48 | 0.51 | 0.39 |
| cross-encoder | 3 | 0.65 | 0.75 | 0.46 | 0.53 | 71.5% | 0.64 | 0.44 | 0.33 |
| cross-encoder | 4 | 0.73 | 0.82 | 0.50 | 0.58 | 62.3% | 0.73 | 0.35 | 0.26 |
| oracle (gold) | 2 | 0.81 | 0.88 | 0.60 | 0.74 | 80.4% | 1.00 | 0.27 | 0.19 |
| oracle (gold) | 3 | 0.84 | 0.89 | 0.58 | 0.73 | 70.7% | 1.00 | 0.23 | 0.18 |
| oracle (gold) | 4 | 0.85 | 0.90 | 0.58 | 0.72 | 60.3% | 1.00 | 0.22 | 0.16 |
| random | 2 | 0.21 | 0.22 | 0.16 | 0.22 | 78.4% | 0.01 | 0.85 | 0.85 |
| random | 3 | 0.25 | 0.27 | 0.18 | 0.24 | 68.8% | 0.04 | 0.82 | 0.80 |
| random | 4 | 0.35 | 0.38 | 0.26 | 0.32 | 59.1% | 0.10 | 0.73 | 0.70 |
| **full context, 2nd call (noise floor)** | 10 | 0.93 | 0.98 | 0.58 | 0.68 | 0% | 1.00 | 0.13 | 0.06 |

Strict agreement split by whether both gold paragraphs survived (k = 2/3/4; full table in the logs): learned, both kept vs not: nano 0.67/0.64/0.65 vs 0.23/0.25/0.27; qwen 0.74/0.73/0.74 vs 0.38/0.44/0.42; gemini 0.84/0.85/0.82 vs 0.31/0.35/0.30. The oracle, which always keeps both: nano 0.64/0.65/0.65, qwen 0.69/0.68/0.67, gemini 0.81/0.84/0.85.

### 4. Paired tests (exact McNemar on per-question agreement, n = 100, uncorrected)

Learned (ablation-trained) vs cross-encoder; cell = CE-only agrees / learned-only agrees, p:

| teacher, metric | k=2 | k=3 | k=4 |
|---|---|---|---|
| nano strict | 13 / 5, p=0.096 | 16 / 10, p=0.33 | 22 / 10, p=0.050 |
| nano judge | 19 / 7, p=0.029 | 19 / 11, p=0.20 | 19 / 7, p=0.029 |
| qwen strict | 7 / 9, p=0.80 | 5 / 13, p=0.096 | 10 / 13, p=0.68 |
| qwen judge | 4 / 12, p=0.077 | 2 / 14, p=0.004 | 4 / 14, p=0.031 |
| gemini strict | 8 / 12, p=0.50 | 9 / 16, p=0.23 | 10 / 7, p=0.63 |
| gemini judge | 10 / 12, p=0.83 | 11 / 14, p=0.69 | 9 / 6, p=0.61 |

Gold-trained vs ablation-trained learned; cell = gold-trained-only / ablation-trained-only agrees, p:

| teacher, metric | k=2 | k=3 | k=4 |
|---|---|---|---|
| nano strict | 17 / 3, p=0.003 | 19 / 8, p=0.052 | 21 / 9, p=0.043 |
| nano judge | 21 / 4, p=0.0009 | 23 / 7, p=0.005 | 19 / 5, p=0.007 |
| qwen strict | 8 / 4, p=0.39 | 5 / 2, p=0.45 | 8 / 3, p=0.23 |
| qwen judge | 7 / 8, p=1.0 | 2 / 4, p=0.69 | 5 / 4, p=1.0 |
| gemini strict | 3 / 2, p=1.0 | 3 / 1, p=0.63 | 7 / 3, p=0.34 |
| gemini judge | 2 / 4, p=0.69 | 5 / 2, p=0.45 | 6 / 3, p=0.51 |

Learned+rel (extra) vs learned, strict (learned+rel-only / learned-only agrees; k=2/3/4): nano 4/2 (p=0.69), 13/6 (0.17), 12/7 (0.36); qwen 6/5 (1.0), 5/8 (0.58), 3/9 (0.15) in favour of learned; gemini 2/2 (1.0), 2/3 (1.0), 1/4 (0.38). Judge: all p >= 0.11. No cell is significant.

Oracle vs noise floor (full-context re-ask); cell = oracle-only / floor-only agrees, p:

| teacher, metric | k=2 | k=3 | k=4 |
|---|---|---|---|
| nano strict | 9 / 17, p=0.17 | 8 / 15, p=0.21 | 8 / 15, p=0.21 |
| nano judge | 6 / 11, p=0.33 | 4 / 11, p=0.12 | 4 / 11, p=0.12 |
| qwen strict | 2 / 22, p=4e-5 | 2 / 23, p=2e-5 | 2 / 24, p=1e-5 |
| qwen judge | 1 / 9, p=0.021 | 1 / 9, p=0.021 | 1 / 6, p=0.125 |
| gemini strict | 0 / 12, p=5e-4 | 1 / 10, p=0.012 | 1 / 9, p=0.021 |
| gemini judge | 0 / 10, p=0.002 | 0 / 9, p=0.004 | 1 / 9, p=0.021 |

Deployable scorers (learned, gold-trained, cross-encoder) vs the same floor: for qwen and gemini all 36 cells (3 scorers x 3 k x 2 metrics each) have p < 0.01 (largest: qwen judge 0.006, qwen strict 1e-4); for nano all are p < 0.05 except cross-encoder k=4 strict (p=0.065) and gold-trained k=4 judge (p=0.052).

Oracle vs full-context gold accuracy (oracle-only / full-only correct, p): nano EM k=2/3/4 12/9 (0.66), 11/6 (0.33), 10/6 (0.45), contains 12/6 (0.24), 13/6 (0.17), 9/5 (0.42); qwen EM 16/0 (3e-5), 6/1 (0.13), 10/3 (0.09), contains 8/1 (0.039), 5/1 (0.22), 5/1 (0.22); gemini EM 7/3 (0.34), 6/4 (0.75), 5/3 (0.73), contains 8/1 (0.039), 8/2 (0.11), 7/2 (0.18).

## Interpretation for the mission

- **Feasible as a mechanism, not as a promotable pruner on this task.** Ablation labels from a near-deterministic teacher train a scorer that ranks paragraphs well (within-question AUC vs the label 0.73 qwen / 0.77 gemini, oracle 0.78 / 0.82) and is statistically indistinguishable from the same model trained on gold labels (6 of 6 strict and 6 of 6 judge comparisons p >= 0.23). With the sampled nano teacher the same scorer is near random (0.53) and significantly worse than the gold-trained control (k=2 strict p=0.003). The first run's failure was therefore label noise from the teacher, not the ablation idea. The first version's "learned worse than cross-encoder at every k" does not survive: it holds for nano at best marginally (strict p=0.096 / 0.33 / 0.050) and reverses or vanishes with T=0 teachers (qwen judge k=3: learned better, p=0.004; gemini: no difference, p >= 0.23).
- **Accuracy/coverage: no deployable cell reaches the noise floor.** At 62-73% of tokens removed (k=4 and k=3), ablation-trained learned agrees with `A_full` on 0.63-0.66 strict / 0.87-0.89 judge for qwen (floor 0.89 / 0.99) and 0.70-0.72 strict / 0.78-0.79 judge for gemini (floor 0.93 / 0.98); the gold-trained control is at most 0.71 / 0.90 (qwen k=4) and 0.74 / 0.82 (gemini k=3-4). At 80% removed (k=2) learned falls to 0.53 / 0.76 (qwen) and 0.62 / 0.71 (gemini). Contains-gold accuracy at k=4 is 0.65 (qwen) and 0.60 (gemini) vs 0.68 and 0.67 for full context. Every learned, gold-trained and cross-encoder cell is significantly below the floor for both T=0 teachers (all p < 0.01).
- **What breaks is the retrieval step, plus a smaller residual that perfect retrieval does not remove.** Ablation-trained learned keeps both gold paragraphs in 0.42-0.77 of held-out questions with the T=0 teachers' labels (0.36-0.55 with nano's); across all non-oracle, non-random scorers it is 0.36-0.82, i.e. a gold paragraph is lost in 18-64% of questions. Given both kept, learned agreement is 0.84 / 0.85 / 0.82 (gemini) and 0.74 / 0.73 / 0.74 (qwen), similar to the oracle on all questions (0.81-0.85, 0.67-0.69; the both-kept subset is easier, so this is not a like-for-like gap); when one is lost it is 0.30-0.35 (gemini) and 0.38-0.44 (qwen). The oracle itself stays below the floor with T=0 teachers (strict 0.81-0.85 vs 0.93 gemini, 0.67-0.69 vs 0.89 qwen; judge 0.88-0.90 vs 0.98 and 0.91-0.94 vs 0.99), so removing distractors is not free even with perfect retrieval. For qwen most of the strict gap is surface form (strict 0.67-0.69 vs judge 0.91-0.94). The first version's "the oracle loses nothing detectable" was an artifact of a noisy floor (nano 0.72) and n = 100 (discordant 9 vs 17, p=0.17).
- **Ablation labels are weak "needed" labels even when the teacher is stable.** Dropping one of the two gold paragraphs changes the judged answer only 42-51% of the time (strict 55-60%), so ablation recall vs gold is 0.42-0.51 (judge); precision is 0.42-0.63 (judge). With T=0 teachers, dropping a distractor changes the answer more often than a replicate does (qwen 30.4% vs 10.8% strict, 10.0% vs 2.0% judge; gemini 13.2% vs 7.2% strict, 7.5% vs 4.8% judge), so the label also reflects prompt perturbation, not only dependence (I did not separate the two). Only the sampled nano teacher has distractor "matters" equal to its replicate disagreement (34.7% vs 36.0%). Closed-book contains-gold is 0.23 / 0.24 / 0.32 (nano / qwen / gemini), the same as random-context k=2 (0.27 / 0.26 / 0.22), so the model's memory accounts for random-context accuracy; the 21 comparison questions are near chance by construction (closed-book 0.52-0.57).
- **Certification is out of reach at n = 100.** Best 95% upper bound on disagreement: strict 0.22 (gemini oracle k=4), deployable strict 0.34 (gemini gold-trained k=3/4); judge 0.11 (qwen oracle k=4), deployable judge 0.16 (qwen gold-trained k=4) and 0.18 (qwen learned k=4). Even the full-context re-ask only bounds at 0.13 (gemini strict) and 0.05 (qwen judge). A budget of 5% needs far more than 100 held-out rows (not computed here), and a 15-20% budget is reachable only under the lenient judge criterion, per-cell, without correction for choosing among 57 cells. The scorer emits no calibrated confidence, so `certify()` (coverage at a budget) was not applicable.
- **What would be informative next (not run):** a confidence-gated fallback (send the full prompt when the scorer's margin is small) to trade coverage for agreement; typed outputs so the floor is exact; and a held-out set large enough for a 5-10% certificate.

## Caveats

- **Spec deviation, post-hoc teachers.** gpt-5.4-nano at temperature 0 is not obtainable on OpenRouter, so the spec teacher ran at the default temperature. qwen3-32b and gemini-3.5-flash-lite were chosen after the nano noise was seen. They are not deterministic either (replicate disagreement 10.8% / 7.2% strict). Qwen routing was not pinned (41 of 4,518 calls went to other providers). Qwen `/no_think` emits an empty think block (4,534 reasoning tokens over 4,518 calls, 1.0 per call), no substantive reasoning.
- **Truncation.** `max_tokens=40` cut 245 qwen answers (5.4%; 68 of 1,800 held-out cells), 92 gemini answers (2.1%; 46 of 1,800) and 9 nano answers (1 of 1,800). They are verbose sentences ("Based on the provided context, there is no mention ..."), not short spans, and are compared as truncated. Direction of the bias is unknown; the instruction "shortest answer span" is not followed by these models on every call.
- **Judge validity.** One judge (`gemini-3.5-flash-lite`, also the gemini teacher). A second-judge check (`anthropic/claude-haiku-4.5`, same prompt, 80 strict-different pairs per teacher balanced on the primary verdict, `judge_audit_log.txt`) agrees on 0.89 (nano), 0.875 (qwen), 0.875 (gemini) of pairs, Cohen's kappa 0.78 / 0.75 / 0.75; the 29 disagreements go both ways and some are primary-judge errors (for example "Yes." vs "No." for the same yes/no question judged the same). Treat judge agreement as accurate to about +/-0.05 and strict as the spec metric. Unparsed judge outputs (do not start with YES/NO) counted as NO: nano 1 (a held-out pair), qwen 7 and gemini 1 (label pairs; for qwen mostly hedges about truncated answers); counting them as YES moves held-out `agree judge` by at most 0.01 (nano) and 0 (qwen, gemini).
- **The old "relaxed" rule is withdrawn.** F1 >= 0.5 or containment, defined after seeing the noise, disagrees with the judge: of 703 strict-different nano pairs, containment-only pairs (190) were judged the same in 139, F1-only pairs (136) in 106, and pairs the rule calls different (377) were judged the same in 69; qwen: 177 -> 139, 164 -> 97, 246 -> 33 of 587; gemini: 61 -> 33, 35 -> 24, 288 -> 17 of 384.
- **Statistics.** n = 100 per cell, one scorer-training run on 150 questions, one seed, one replicate per question; about 60 McNemar tests per teacher and 57 bounds (19 cells x 3 teachers), none corrected. "No significant difference" at 10-25 discordant pairs is not evidence of equality. `agree_stable`/`agree_unstable` in the JSON condition on the replicate and are not used.
- **Synthetic vs real.** HotpotQA is a public QA benchmark with 2 gold of 10 Wikipedia paragraphs (so at most about 80% of tokens are removable), and the six features are question-paragraph relevance features that suit it. A real call site has no gold labels, no question/paragraph structure, longer and mostly irrelevant context (easier to prune), and typed outputs where agreement is exact. The gold-trained scorer is a control, not a deployable option. Wikipedia content is partly memorized (closed-book control), so some answers do not depend on the context.
- **Leakage:** none found. Scorers are fit on the first 150 questions only, evaluation is on the last 100, the cross-encoder is zero-shot, `certify()` is not used so no calibration rows are reused. `common.py`: no bug found.
- **Cost accounting** is tokens x listed price from the response ledgers. Checked against OpenRouter's `usage.cost` on 8 real prompts per model: ratio 1.000 for nano, qwen and gemini. It excludes failed requests and the first builder's ad-hoc probes.

## Time and cost

- **Cost:** about $2.93 of the $5 cap. Original run $0.938 (nano, 4,416 calls). Revision $1.99: nano replay additions $0.049 (215 new teacher calls + 1,031 judge calls), qwen3-32b teacher + judge $0.431 (4,518 + 895 calls), gemini teacher + judge $1.467 (4,428 + 630 calls), second-judge audit $0.036 (240 haiku calls), endpoint/cost probes $0.008. Logged in `smokes/COSTS.md`.
- **Wall time:** original about 2,360 s over 5 invocations (about 39 min, over the 25 min target; the embedding of 2,500 paragraphs took 697 s on a shared CPU). Revision cold runs as logged by the builder: 251 s (nano), 282 s (qwen), 162 s (gemini), API-bound with caches for embeddings and cross-encoder scores; the final logs come from warm re-runs (about 20 s each, tables identical to the cold runs). A from-scratch run of all three teachers does not fit the 25 min target.

## Revision

Why: the review found that the spec's `temperature=0` was never applied to the nano teacher, and that RESULTS.md was written before the builder's own T=0 reruns and contradicted them.

**What changed**
1. **Teacher temperature.** Verified: gpt-5.4-nano has no OpenRouter endpoint with `temperature` (endpoint metadata; 404 under `require_parameters` for reasoning disabled, default and effort `none`). All "temperature 0" claims for nano are withdrawn; its run is labelled default-temperature. Added two T=0 teachers where `require_parameters` makes a dropped parameter an error (qwen3-32b on DeepInfra, gemini-3.5-flash-lite on Google AI Studio), re-ran the full protocol for each (about $1.9, reuse of all nano responses; nano only got the new closed-book and gold-trained re-asks).
2. **Equivalence metric.** The F1/containment "relaxed" columns are replaced by a T=0 LLM-judge column, validated against a second judge (kappa 0.75-0.78) and with the unparsed-output sensitivity above.
3. **New controls:** closed-book (no context), gold-trained scorer, vs-noise-floor tests for deployable scorers, McNemar for every claimed difference, judge-unparsed and truncation counts, and cost verified against `usage.cost`.
4. **Files and ledger.** RESULTS.md cites the files that exist (`log_<tag>.txt`, `results_<tag>.json`, `responses*.jsonl`), the run command takes the tag, and `COSTS.md` has the revision spend. `run.py` additionally writes `pairs_<tag>.json` and the extra counts.

**How the numbers moved**

| claim in the first version | status now |
|---|---|
| "Same prompt twice differs 36% at temperature 0; dominant cause is teacher noise" | 36% was default-temperature sampling. At T=0: 10.8% (qwen), 7.2% (gemini); judge 14.8% / 2.0% / 4.8% (nano / qwen / gemini). Noise explains the nano scorer failure, not the shortfall against the floor |
| Learned scorer within-question AUC vs ablation label 0.533 ("near random") | 0.533 (nano), 0.726 (qwen), 0.770 (gemini); oracle 0.659 / 0.775 / 0.824 |
| "Cross-encoder beats a scorer trained on ablation labels; learned worse at every k" | nano: CE ahead, p=0.096 / 0.33 / 0.050 (strict); T=0: learned equal or ahead (gemini strict 0.62/0.72/0.70 vs 0.58/0.65/0.73; qwen 0.53/0.63/0.66 vs 0.51/0.55/0.63; p from 0.004 in favour of learned to 0.80) |
| "Oracle loses nothing detectable; reaches the noise-floor agreement" | holds only for nano (0.64-0.65 vs 0.72, p=0.17-0.21). Qwen 0.67-0.69 vs 0.89 (2 vs 22, p=4e-5); gemini 0.81-0.85 vs 0.93 (0 vs 12 at k=2, p=5e-4) |
| "Oracle has higher gold accuracy than full prompt (contains 0.68 vs 0.62, EM 0.51 vs 0.48)" | not significant on nano (contains p=0.24 / 0.17 / 0.42, EM p=0.66 / 0.33 / 0.45); qwen EM k=2 16 vs 0 (p=3e-5) looks like a format effect (shorter answers with less context); gemini contains k=2 p=0.039, uncorrected. Not claimed |
| "Learned+rel closes part of the gap" | within noise: learned+rel vs learned nano strict p=0.69 / 0.17 / 0.36, no significant cell for any teacher or metric |
| "Position coefficient is spurious" (asserted) | untested. The ablation-trained coefficient changes sign across teachers (+0.062 nano, -0.070 qwen, -0.049 gemini; gold-trained +0.023), consistent with no stable effect; no interval was computed |
| "Random k=2 gets contains-gold 0.27 from parametric knowledge" | now supported by the closed-book control: 0.23 / 0.24 / 0.32 |
| Scorers lose a gold paragraph in 27-64% of questions | 18-64% across all non-oracle scorers (gold-trained k=4 loses 18%, learned nano k=2 loses 64%) |
| Best UB on disagreement 0.32 (relaxed, CE k=4); 0.19 oracle | strict 0.22 (gemini oracle k=4), deployable strict 0.34; judge 0.11 oracle, deployable 0.16. Still nothing certifiable at n=100 |
| Spend $0.94 | about $2.93 total (revision $1.99 added) |

What survives unchanged: the nano strict tables (identical cached responses; every cell matches `log_original.txt`), the dataset sampling, the finding that nothing is certifiable at n = 100, and the retrieval step (losing a gold paragraph) as the main source of pruning disagreement.
