# Training plan: the student candidates, per call site

Written 2026-10-18 from the round-4 smokes (`smokes/FINDINGS.md`) and the demo stack in `dreaming/` (`DESIGN.md`, `db/migrations/0001_init.sql`, `packages/dreaming`). Decision it implements: **every candidate student is trained per call site in parallel, each gets its own certificate, the certificate picks what is promoted. The Hopfield memory is priority 1.**

Numbers quoted below are measured unless marked "est." Smoke ids (S1, S24, ...) point at `smokes/sNN_*/RESULTS.md`.

## 1. Priority order

| # | candidate | what it is | "training" is | runs on | code to port from |
|---|---|---|---|---|---|
| 1 | **Hopfield memory** over the trace rows | softmax(β · cosine) weighted vote over stored (embedding, label) rows, seeded with LLM-written examples from the schema | append a row; pick β on calibration rows at each checkpoint | CPU, in-process | `smokes/s27_hybrid_embedder_memory/run.py` `hopfield_head`; `s24_other_families/arms.py` `hopfield`; `s29_memory_student/run.py` (scale, imbalance, deletion, determinism; written, unrun) |
| 2 | frozen encoder + linear head | bge-small (33M, 384-d) frozen, multinomial logistic head per site, ~5k weights | one L-BFGS fit, 1–3 s | CPU, Modal job or in-process | already built: `packages/dreaming/predictors/student.py` `fit_head`, `predictors/train.py` `fit` |
| 3 | online head | the same linear head updated one row at a time (delta rule / recursive least squares) | one weight update per certified teacher answer | CPU, in-process | `s27/run.py` `delta_head`, `ridge_head` |
| 4 | HDC integer bundle | each row → 10,000-bit code (fixed random projection), one summed bundle per class | add or subtract a row's code | CPU, in-process | `s25_exotic/arms.py` `HDC`; `s29/run.py` `hdc_fit` |
| 5 | fine-tuned bge-small encoder + head | unfreeze the encoder on the site's rows | GPU, 17–131 s per retrain | Modal T4 or RunPod | `s8_gpu_student/train_student_v2.py` `ft_encoder` |
| 6 | 0.5B decoder + per-site LoRA | Qwen2.5-0.5B with a LoRA and a label head | GPU, 7–110 s; GPU to serve | not planned | `s8_gpu_student/train_student.py` |

Why the memory is first, in plain words: it has no training step, so a site has a student the moment its schema and seed rows exist; adding a trace row is learning and deleting one is forgetting, which is the only clean answer to a deletion request; every answer can show the rows that voted for it; and it fits a vector index a customer may already run. It is also the least proven at scale (S29 was written to test that and never ran), so building it first surfaces its risks earliest.

What the numbers say, honestly (5-task protocol, k = real rows per label, 5% budget):

| arm | acc k=1 / 3 / 10 | certified coverage k=1 / 3 / 10 | source |
|---|---|---|---|
| LR, k rows only | 0.80 / 0.93 / 0.97 | 0.14 / 0.60 / 0.91 | S24 baseline |
| Hopfield, k rows only | 0.79 / 0.92 / 0.96 | 0.10 / 0.61 / 0.88 | S24 1b |
| **Hopfield + LLM-written rows** | **0.89 / 0.93 / 0.96** | **0.50 / 0.63 / 0.88** | S24 1c |
| LR + LLM-written rows | ≈0.89 / – / – | **0.80 / 0.87 / 0.95** | S17 arm 2 |
| LR on bge-base (109M) | 0.83 / 0.93 / 0.97 | 0.18 / 0.78 / 0.91 | S27 |

So with seed rows the linear head certifies more traffic than the memory at every k (0.80 vs 0.50 at k=1). The memory's case is operational (no fit, forget by deletion, evidence rows, hot spare), not coverage. Running both in parallel costs nothing, and the certificate decides per site.

Rejected with numbers, do not build: hypernetworks (S2, S17, S26), hybrid SSM embedder (S27: within noise, 30× slower), Bayesian heads (S24: identical coverage to the softmax gap), kernel heads (S25: tie), spiking, gzip-NCD kNN, product-key memory, LLM-written rules (S24: 24–33% accuracy), TSDAE, SetFit (S8: 0.64 at 770 rows), encoder pretraining on sibling tasks (S26: +0.04 / +0.03 / +0.01, gone once seed rows are added).

## 2. Data

### 2.1 The row

One `trace_rows` row per call (schema of record: `dreaming/db/migrations/0001_init.sql`). The fields the trainers use:

| field | used for |
|---|---|
| `call_site`, `kind` | which site's candidates; `kind` ∈ model step, tool, router |
| `request` (rendered text), `request_sha` | the input; embedded once per text, cached by `request_sha` |
| `teacher.label`, `teacher.provider`, `teacher.cost` | the training and calibration label; provider is pinned and logged (S13) |
| `phase` at call time | only `watching` / `testing` rows calibrate (rows after promotion are biased) |
| `prediction.{label, confidence, scores, hashes}` | shadow predictions; `hashes` = prompt, weights, env |
| `agree` | prediction == teacher label; the certificate's "correct" |
| `decision` | called, predicted, audited, fallback, replayed |

One addition the memory needs: the embedding itself, per row. Today the student caches vectors by prompt sha in process; the memory needs them durable and queryable.

```
memory_rows (
  call_site    text,
  row_id       text,            -- trace_rows.id, or "seed:<label>:<i>" for LLM-written rows
  label        text,
  source       text,            -- seed | shadow | audit | correction
  encoder_sha  text,            -- rows from a different encoder are invalid, same rule as Head.encoder_sha
  embedding    float4[384],     -- unit norm; pgvector `vector(384)` when the extension exists
  ts           timestamptz,
  primary key (call_site, row_id)
)
```

1.5 KB per row as float32 (0.4 KB as int8, 1.25 KB per *class* as an HDC bundle). Supabase Free is 500 MB, so about 100k memory rows fit beside the trace rows; beyond that compaction (candidate 4) or a separate store.

### 2.2 Where rows come from, in the order they appear

1. **Seed rows** from the schema: 10 LLM-written example inputs per label, written by `asi1-mini` from the label names and descriptions (S2b, S17). Cost $0.005 per site. Cached for the smoke labels in `smokes/.cache/synth_examples.json`. Tagged `source=seed`, so a certificate can be computed with and without them.
2. **Shadow rows**: real calls answered by the pinned teacher while the site is awake, watching or testing. The candidates predict in shadow and `agree` is recorded. These are the only calibration rows.
3. **Audit rows** after promotion: 1 in 20 confident answers still goes to the teacher. They feed the audit certificate and may be appended to the memory and to the next head fit, but never calibrate (biased by the cutoff).
4. **Human corrections** (mission 4) carry a separate label; the deferral head certifies against them, not against the teacher.
5. **Self-labelled rows** during shadow (TK-KNN / Sinkhorn probes over the unlabelled call stream, S23 / S25): raise coverage at k=1 from 0.13 to 0.35–0.56, nothing from k=3 on. Tagged `source=self`, never used for calibration.

### 2.3 Labels and their noise

The label is the pinned teacher's answer. Teachers disagree with themselves on identical inputs, which floors every certificate:

| teacher | self-disagreement | $ per 1,000 labels | p50 latency |
|---|---|---|---|
| claude-sonnet-5.5 | 1–3% | $1.83 | 1.1 s |
| claude-haiku-5.5 | 3.5% | $0.08 | 0.8 s |
| gemini-3.5-flash-lite | 4–5% | $0.18 | 0.6 s |
| qwen3-32b | 5% | $0.04 | 0.4 s |
| gpt-5.4-mini | 11% | $0.37 | 0.8 s |

(S13, Banking77.) A budget below the floor is unmeasurable; the operator's "error budget" means disagreement with that pinned reference. Pin `provider.order` with `allow_fallbacks: false` and log the served provider on every row.

### 2.4 Splits

- **Hold out by group, never by row**: merchant, customer, session, whatever repeats. Row-level splits inflated coverage by 30 points and let 4.2% of novel inputs through as confident wrong answers (S4).
- **Calibration rows are shadow rows the weights never saw** (phase watching / testing). Existing rule in `learning.py`.
- **Selection and certification never share rows.** Either certify every candidate at δ/N on the calibration slice (N = number of candidates; the grid certifier already spends δ/40 per coverage level), or split the shadow rows into a selection slice and a certification slice by group. Default: δ/N on one slice, because it needs no second slice and the coverage cost is small; measure it once on S1's ladder.
- The calibration slice must contain novel inputs at the deployment's rate (S4), so unseen merchants are low-confidence by construction.

### 2.5 How many rows

| purpose | rows per site | evidence |
|---|---|---|
| day-zero student | 0 real rows + 10 seed rows per label | S2b: 87% on unseen CLINC domains, 29% certified coverage on 77-way Banking77 with no real rows |
| certificate at 5% certifies most of a good student's coverage | 200 calibration rows (70–93%); 500 saturates | S1, S28 |
| router budgets 5 / 10 / 15 / 20% | 1,000 / 300 / 200 / 200 calibration rows | S28 |
| human-review deferral at 5% | ≥1,000 corrections | S3 |
| 2% budget on a 77-label step | ≥6k training rows for half the traffic | S1 |
| linear head on Banking77, 5% budget | 770 rows → 57% coverage, 3,080 → 85%, 9,993 → 88% | S1 |

## 3. Benchmarks

### 3.1 Offline protocol (every candidate runs this before it enters the parallel trainer)

The S23 / S25 / S27 protocol, `smokes/s25_exotic/lib.py` (`build`, `draw`, `score`):

- 5 tasks: CLINC150 `travel` and `work` (15 labels each), three disjoint 12-label Banking77 sub-tasks.
- 300 train rows per task set aside as calibration; the official test split is the evaluation set (450 / 450 / 480 / 480 / 479 rows).
- k = 1, 3, 10 real rows per label, 3 draws (seeds 0–2), the same rows for every arm so arms are paired.
- Metrics: accuracy on test rows; certified coverage at a 5% budget (`common.certify`, fixed 40-point grid, δ 0.05, confidence = top-2 gap of the arm's class scores, fit on the calibration rows, applied to test rows); realized error above the cutoff; fit seconds; parameters or stored bytes; query latency.
- Report paired differences to the LR baseline with the standard error over the 15 (task, draw) cells. Differences under about 0.1 coverage at k=1 or k=3 are noise.

Acceptance for a candidate: within 2 s.e. of LR on coverage at k=3 and k=10, realized error under budget in every certified cell, and (label, quantized confidence) byte-stable across two runs on the pinned CPU profile.

### 3.2 Scale ladder

Banking77 (77 labels) at 77 / 385 / 770 / 3,080 / 9,993 rows and CLINC150 (150 labels) at 1,500 / 3,000 / 15,000 rows, 500 calibration rows, budgets 2 / 5 / 10% (S1 protocol; S29 section A extends it to 15k rows for the memory). LR reference: 60 / 69 / 82 / 91 / 93.3% accuracy, 0 / 17 / 57 / 85 / 88% coverage at 5%.

### 3.3 Live benchmark

The deployed demo stack (`dreaming`, Fly API + Modal trainer + Supabase): a 2,000-run batch of the bookkeeping workflow over four call sites. Measured today with the linear head and the rule teacher, 500 calibration rows, 5% budget: categorize 40% coverage (bound 4.4%), vendor_lookup 67% (1.1%), match_invoice 60% (4.2%), anything_odd 100% (0.6%); all four sites dreaming by 116 s. The parallel trainer is judged on: certified coverage per site at the operator's budget (per candidate), seconds to the first certificate (seed rows only vs with shadow rows), ms per predicted call, replay `same_bytes` rate, and runs per second after promotion.

### 3.4 Real traces

Copilot traces (S14b, `.data/ghcp*`), tau2-bench banking tools (deterministic Python tools with real-shaped returns), tau-bench trajectories; MCPMark trajectory logs are the open item for a repeat-rate measurement. None has a typed bookkeeping step, so the synthesized ledger (seeded generator, hold out merchants) stays the demo's benchmark.

## 4. Training, candidate by candidate

Shared by all: the encoder is bge-small at hub revision `5c38ec7c`, hashed by file (`model_sha`), batch 1, 1 thread, max length 128, unit-norm float32, vectors cached by prompt sha. A certificate names `weights_sha`, `env_sha` and `teacher_sha`; if any differs at startup the site returns to testing.

### 4.1 Hopfield memory (priority 1)

- **State**: the `memory_rows` for the site (seed + shadow + audit rows with their embeddings) and one scalar β.
- **Read**: `s = softmax(β · E q)` over stored rows, class score = sum of s per label, prediction = argmax, confidence = top-2 gap rounded to 4 decimals. In numpy this is one (rows × 384) matmul.
- **β**: grid {5, 10, 20, 40, 80, 160, 320, 640}, picked by negative log-likelihood on the calibration rows (S24). Refit at each checkpoint, never per call.
- **weights_sha** = sha of the sorted row-id list + β + encoder_sha. Appending a row therefore makes new weights; promotion still happens at checkpoints (see §5), and between checkpoints new rows serve the shadow copy only.
- **Delete** = remove the row, recompute weights_sha, re-gate on the calibration rows (S29 section C was written to measure how often this revokes).
- **Imbalance**: a frequent class with many rows dominates the softmax sum. Cap rows per class at query time (take the top-m by similarity per class) or weight by 1/count; S29 section B was written to pick between them. Decide from that run, not by guess.
- **Query cost**: O(rows × 384). Est. sub-millisecond at 10k rows, ~10 ms at 100k on one core, index needed around 1M (hnswlib top-50 approximates the read; an approximate read changes bytes, so replay must store the exact row set and recompute exactly). S29 section D was written to measure 1k to 1M.
- **Determinism**: a float sum over rows depends on order; fix the row order (sorted row id), accumulate in float64, quantize confidence before hashing. S29 section E was written to check CPU-1-thread vs GPU bytes.
- **Compute**: in-process CPU, no job. Seed rows cost one `asi1-mini` call per 15 labels.
- **Measured so far**: table in §1. Two zero-coverage cells at k=10 (CLINC work, S27), to understand before shipping.
- **First thing to run**: S29 sections A, B, C, E on one RTX A5000 pod via `smokes/pod.py` (est. < $1, 20–30 min). The script is written; this is the de-risk the memory now needs as priority 1.

### 4.2 Frozen encoder + linear head (built)

- `fit_head`: multinomial logistic regression, L-BFGS, l2 = 0.1 (sklearn C = 10), 20% holdout for a sanity accuracy. 1–3 s warm on Modal CPU, seconds on the laptop at 10k rows.
- Add the seed rows to the fit (S17: coverage at k=1 0.14 → 0.80) and the self-labelled rows during shadow.
- bge-base (109M) is the one upgrade that helps (+0.14 coverage at k=3) at ~3× embedding time; choose per site by call volume, and note the encoder swap invalidates every candidate's rows and certificates for that site.
- Artifact: `.npz` of W, b, labels, encoder_sha, kilobytes, in the `weights` table as bytea.

### 4.3 Online head (a mode of 4.2)

- `W += lr · (onehot − W x) xᵀ` per certified teacher answer, bias via a constant feature. Fixed lr 0.1, one pass per new row in production (the S27 grid picked 0.03–0.1 and 30 epochs on tiny k; at hundreds of rows one pass is the point).
- Ties LR on accuracy (0.80 / 0.91 / 0.96), coverage +0.01 / −0.12 / −0.04 (S27). The ridge closed form ties LR exactly; use it as the periodic resync so the online weights never drift from the batch solution.
- Every update is a new weights_sha, so it serves in shadow between checkpoints exactly like the memory; what it removes is the training job, not the promotion cadence.

### 4.4 HDC bundle (compaction under the memory)

- Projection: fixed random ±1 matrix 384 × 10,000 from seed 0 (regenerable, never stored); code = sign(x · P) as int8; class bundle = integer sum of its rows' codes; score = bundle · code; confidence = top-2 gap of the scores.
- Add or subtract a row in O(10,000) integer ops. Byte-exact on any hardware (the encoder upstream still jitters, S7).
- Measured: ties LR at k=1 and 3, −0.1 coverage at k=10, ~1 accuracy point below (S25). Use it when a site's memory passes ~100k rows: keep the bundle as the certified student and the raw rows in cold storage for evidence and deletion.

### 4.5 Fine-tuned encoder (GPU, high-volume sites only)

- bge-small unfrozen + head, lr 5e-5 encoder / 1e-3 head, warmup + linear decay, 300 / 776 / 2,504 steps at 770 / 3,080 / 9,993 rows: 17 / 43 / 131 s on an RTX 4090 (S8 v2). Accuracy 0.825 / 0.920 / 0.941 vs frozen LR 0.821 / 0.910 / 0.933; coverage 0.59 / 0.89 / 0.94 vs 0.57 / 0.85 / 0.88.
- Modal `gpu="T4"` (est. 2–3× the 4090 time, ≈ $0.02–0.05 per retrain) or a RunPod pod. 133 MB artifact to the Modal Volume, `weights.uri`, never Supabase Storage (50 MB cap).
- Serve on CPU, batch 1, pinned threads, same as today's encoder, so replay stays byte-identical. CUDA training is not bitwise deterministic; certify the CPU inference path.
- Gate: only sites above ~1,000 rows where the frozen head's certified coverage is the binding constraint. A point or two of coverage is the honest expectation.

### 4.6 LoRA decoder (not planned)

Below the frozen encoder at every row count on typed steps (0.849 vs 0.910 at 3,080 rows, S8). The one thing it does that nothing else here does: a shared LoRA pretrained on sibling steps lets a *new* step certify from ~45 rows instead of ~150 (S8 phase B, S26 +0.19 on travel). Revisit only if a decoder is needed anyway (free-text students via on-policy distillation, untested).

## 5. Selection and certification: the parallel trainer

Per site, at every checkpoint (`calibrate_after` calibration rows, and on a budget change):

1. Gather rows; tag seed / shadow / audit; group-split.
2. Fit the CPU candidates side by side in the `Learner`'s thread pool (memory: append + β; head: L-BFGS; online: resync; HDC: rebundle). The GPU candidate is a Modal job that lands on a later tick.
3. Every candidate predicts in shadow on every call. One embedding is shared; each head is microseconds. The row's `prediction` JSON gains `candidates: {name: {label, confidence, weights_sha}}` and `agree` per candidate.
4. Certify each candidate on the calibration slice at δ/N (or on the certification slice if two slices are used). One `certificates` row per candidate, with a new `candidate` column.
5. Promote the candidate with the largest certified coverage at the operator's budget. Tie-break by replay strength and cost: memory > head > HDC > online > fine-tuned. `policies.cert_id` and `weights_sha` point at the winner; a new `policies.shadow` JSON lists the others' latest certificate ids and coverages.
6. Audits continue at 1 in 20. When the audit bound revokes the winner, the next-best certified candidate is promoted without retraining. This hot spare is the practical gain of training everything.
7. The budget slider re-gates every candidate; the winner may change.

The rule to keep: a candidate chosen for its coverage on a slice must not quote that slice's bound without the δ/N correction. With N = 4 and δ = 0.05 the grid certifier spends δ/160 per level; est. cost is a few points of coverage at 200 rows and nothing at 500. Measure it once on the S1 ladder before shipping.

## 6. Telemetry

Per call, inside `trace_rows.prediction` (data, never streamed):
- which candidate answered, each candidate's label and confidence, `agree` per candidate when a teacher answer exists;
- `embed_ms`, `head_ms` per candidate, memory rows at query time.

Per site, in `policies` and the live `events` feed (Realtime cap 100 msg/s on Supabase Free, so per checkpoint, never per call):
- phase, promoted candidate, certified coverage per candidate at the current budget, calibration rows, memory rows and bytes, β, last refit, audits and audit errors, outage flag.
- New event kinds: `candidate.certified`, `candidate.promoted`, `candidate.revoked`, `memory.appended` (count), `memory.deleted` (with the recertification result), `train.finished` per candidate.

Dashboard additions: one row per candidate under each site (coverage bar at the current budget, realized audit error, rows, ms per call), the budget slider re-gating all of them, and a "show the rows that voted" panel for the memory's answers.

Charts worth keeping: cost saved per 1,000 calls (teacher dollars avoided minus embedding compute); coverage per candidate over time; seconds to first certificate per site; replay `same_bytes` rate; the teacher's self-disagreement measured live from the audit stream (the noise floor, S13).

Logs: JSON to stdout (Fly keeps ~7 days), optional Logfire with the scrubber kept off `error`, `session_id`, `call_site`; Modal jobs call `force_flush()` before exit.

## 7. Compute and cost

| candidate | where | time and cost per site | per-call latency (CPU) |
|---|---|---|---|
| memory | in-process | $0; append is free; β refit ms | est. < 1 ms at 10k rows, ~10 ms at 100k |
| linear head | Modal CPU or in-process | 1–3 s warm, ≈ $0.001 | microseconds |
| online head | in-process | $0 | microseconds |
| HDC bundle | in-process | $0 | microseconds |
| fine-tuned encoder | Modal T4 | 1–5 min, ≈ $0.02–0.05 | same as the frozen encoder |
| encoder (shared by all) | in-process | – | ~5 ms on a free core; ~300 ms on Fly's shared CPU, which is why the batch slows to 1 run/s after promotion. Fill the cache with `just record` or embed on a second core |

The binding cost is teacher labels for calibration, not compute: 200–1,000 rows per site at $0.04–1.83 per 1,000. Balances as of 2026-10-17 23:40 UTC: OpenRouter prepaid credits exhausted (top up before any live teacher run), RunPod ≈ $12, Modal $1 of the $30 credit unlocked until a payment method is added, campaign total ≈ $27 of the $100 cap.

## 8. Build order

1. **Memory predictor** (`predictors/memory_student.py`, port `hopfield_head`), `memory_rows` storage, seed rows from the schema (port S2b's writer), wired into `train.fit` as a second candidate and into the router as a shadow. Run S29 A / B / C / E on a pod (< $1) and record the results in `smokes/s29_memory_student/RESULTS.md`.
2. **Parallel candidates in `learning.py`**: per-candidate shadow predictions, δ/N certificates, `candidate` column, `policies.shadow`, promotion and hot-spare rules, new events. Offline test: the whole loop with fakes, two candidates, a revocation that fails over.
3. **Online mode** flag on the head; **HDC** compaction behind a row-count threshold. Both ports from S27 / S25.
4. **Fine-tuned encoder** job on Modal T4 (port `ft_encoder`), enabled per site by a row threshold and a GPU flag.
5. **Live benchmark**: the 2,000-run batch with all candidates in shadow; record which candidate wins per site and the seconds to first certificate with seed rows only.

## 9. Open risks

- The memory is untested beyond 180 stored rows per task: scale, class imbalance, deletion, 1M-row query cost and byte-determinism are exactly S29's sections (step 1).
- With seed rows the linear head certifies 0.80 at k=1 and the memory 0.50; if that holds on the live sites the memory will be promoted mainly at day zero and as the hot spare.
- Budgets at or below the teacher floor (5% with gpt-5.4-mini, 2% with anything) are unmeasurable; the dashboard should refuse them.
- Free-text steps have no candidate in this plan (judge floor 2.2% on core facts, S16c); the pruner is their lever.
- Supabase Free: 500 MB of Postgres caps memory rows around 100k across all sites; HDC compaction or an external vector store after that.
- δ/N selection correction is reasoned, not yet measured.
