# Training plan: candidates, data, the parallel campaign, the monitor, the numbers we will defend

Written 2026-10-17 (evening) for the night before the 10:00 AM 2026-10-18 cut-off. It fixes the interfaces (what a candidate is,
what a training run writes, what the monitor reads) and the shape of the campaign; it does not fix hyperparameters. Numbers are
measured unless marked est.; smoke ids (S1, S24, …) point at `../../smokes/sNN_*/RESULTS.md`; the per-candidate internals from
the smokes are in `../../training-plan.md` and summarized in §1.

Decision it implements: **every candidate trains per call site in parallel, every candidate is certified, the certificate
picks what is promoted, and everything that could be asked later is stored so it can be computed in retrospect.**

## 1. Candidates (the model slot, `../DESIGN.md` §5)

A candidate is one module with two functions, `fit(site, rows, encoder) -> Fitted` (the bytes plus n_train, holdout accuracy
and seconds) and `load(site, bytes, encoder) -> Predictor`, registered in `predictors.CANDIDATES`. `CallSite.candidates` (the
`candidates=` argument of `@step` / `@tool`) says which to train; the default by kind today is model → `head`, tool → `exact`,
router → `config`, and `memory` joins the model default once it is ported. Every candidate must be byte-stable in `(label,
quantized confidence)` at batch 1 with pinned threads; the monitor checks it (§4). Live sites list only candidates that share
their encoder (`../DESIGN.md` §5); `finetune` and `head@base` run in the campaign only.

| name | what it is | training is | runs on | status | port from |
|---|---|---|---|---|---|
| `head` | bge-small (33M, 384-d) frozen + multinomial logistic head, ~5k weights | one L-BFGS fit, 1–3 s | CPU; in-process or a job | built (`predictors/student.py`) | – |
| `memory` | Hopfield read over stored (embedding, label) rows: softmax(β·cosine) vote; seeded with LLM-written examples | append a row; pick β on calibration rows at checkpoints | CPU, in-process | to port | `smokes/s27_hybrid_embedder_memory/run.py` `hopfield_head`; `s29_memory_student/run.py` (scale, imbalance, deletion, determinism; written, unrun) |
| `online` | the same linear head updated one row at a time (delta rule, ridge resync) | one update per certified teacher answer | CPU, in-process | to port | `s27/run.py` `delta_head`, `ridge_head` |
| `hdc` | 10,000-bit codes from a fixed random projection, one integer bundle per class | add or subtract a row's code | CPU, in-process | to port (Devin's module, `tracks.md`) | `s25_exotic/arms.py` `HDC`; `s29/run.py` `hdc_fit` |
| `finetune` | bge-small unfrozen + head | 17 / 43 / 131 s on a 4090 at 770 / 3,080 / 9,993 rows (S8 v2); est. 2–3× on a T4 | GPU: Modal T4, veri.studio, RunPod | to port | `s8_gpu_student/train_student_v2.py` `ft_encoder` |
| `head@base` | `head` on bge-base (109M) | as `head`, ~3× the embedding time | CPU | a config, not code | – |
| `exact` | exact memory of (input → return) for tools | nothing | CPU | built (`predictors/tool.py`) | – |
| `config` | the cheaper frontier configuration (router sites) | nothing | a real call | built (`predictors/router.py`) | – |

What the smokes say, 5-task protocol, k real rows per label, 5% budget, certified coverage at k = 1 / 3 / 10: `head` with seed
rows 0.80 / 0.87 / 0.95 (S17); `memory` with seed rows 0.50 / 0.63 / 0.88 (S24); `online` ties `head` on accuracy, −0.01 / −0.12 /
−0.04 coverage (S27); `hdc` ties at k ≤ 3, −0.1 at k = 10 (S25); `finetune` +0.02 / +0.04 / +0.06 coverage over `head` at 770 /
3,080 / 9,993 rows (S8 v2); `head@base` +0.14 at k = 3 (S27). Rejected with numbers, not in the matrix: hypernetworks, SSM
embedders, Bayesian and kernel heads, LoRA decoders for typed steps, SetFit, sibling-task pretraining.

Why train them all anyway: `memory` has no fit step, forgets by deletion and shows the rows that voted; `online` removes the
training job; `hdc` is compaction and bit-exact; `finetune` is the sponsor's platform and the top of the ladder; the certificate
decides per site, and the losers are hot spares when an audit revokes the winner.

## 2. Data

### 2.1 Rows

One `trace_rows` row per call (schema: `db/migrations/0001_init.sql`); the fields the trainers use: `call_site`, `kind`,
`request` and `request_sha` (embedded once per text, cached by sha), `teacher.label` / `provider` / `cost`, `phase` at call time
(only `watching` / `testing` rows calibrate), `prediction` and `shadow` (every candidate's label and confidence), `agree`.
Rows are tagged by source: `seed` (LLM-written from the schema, 10 per label, ~$0.005 per site, S2b), `shadow`, `audit`,
`correction`, `self` (self-labelled during shadow, never calibrates). The `memory` candidate also needs each row's embedding
durable: `memory_rows(call_site, row_id, label, source, encoder_sha, embedding float4[384])`, ~1.5 KB per row.

### 2.2 Two workflows, six call sites

**Bookkeeper** (the live demo; synthetic seeded ledger, 120 merchants, 8% novel-merchant rate, 2% planted anomalies):
`categorize` (10 labels), `vendor_lookup` (tool, exact), `match_invoice` (3 labels); `anything_odd` is a router site and is
measured, not trained. Teacher: the pinned frontier model (StepFun `step-5-preview` for the record; the rule teacher offline).

**QuickBooks** (the benchmark; Intuit's `QuickBooks-Hackathon-2026`: 100 synthetic companies, one year each, 2,302 YAML files,
401 MB unpacked; `index.json` lists every company with entity counts). Inspected 2026-10-17: bills carry `vendor_id`,
`expense_kind`, `cost_category` and either `item_lines` or `account_lines[].account_id`; bank lines carry `memo`, `kind` and the
`source_document_id` they settle; invoices carry `due_sim_day` and `customer_payments` say when each was paid. The feed is clean
by construction (the README says so and asks that any injected mess be disclosed). Proposed sites, each a typed decision a
bookkeeping agent makes per document, with reality as the teacher (the books themselves), so no frontier labels are needed and
the LLM is a baseline row:

| site | input (rendered) | labels | teacher | trivial baseline, measured 2026-10-17 over all 100 companies, time-ordered |
|---|---|---|---|---|
| `invoice_paid` | customer, terms, amount, day of year, the customer's recent payment history | `on_time` / `late` / `never` (8.6% / 82% / 9.4% of 84,451 invoices) | `customer_payments` vs `due_sim_day` | "this customer's usual outcome so far": median 0.917 per company, min 0.375 |
| `bill_account` | vendor name, memo, amount, expense kind, line description | the posted account (16 used of the 36-account chart) or `cost_category` (28 values) | the posted account | "this vendor's usual account so far": median 1.000 per company (min 0.719); companies have 3–17 vendors (median 8) and 36–5,461 bills (median 234) |
| `bank_match` | bank memo, amount, date, open documents | the document id it settles (tool-shaped) | `source_document_id` | the feed is clean: memo → `kind` is 100% by memo alone in every company, and amount + date finds the document; only worth running with injected, disclosed mess (amount noise, memo truncation, duplicates) |

`invoice_paid` is the headline: imbalanced, per-customer, a rule gets 92% and a frontier model has nothing to read, and a
calibrated probability per invoice is exactly what a collections workflow consumes (System One shape). `bill_account` is the
Intuit-shaped `categorize`; the vendor rule already solves repeat vendors, so its value is the first bill from each vendor
(about 8 of 234 per company: the rule is undefined there, a seeded student is not) and day zero, and the table says so rather
than claiming a win over the rule. `bank_match` is the tool case and the first to drop tonight.

### 2.3 Splits (never by row)

- **By group**: merchant for the bookkeeper; company for QuickBooks: **one pooled student per site** is trained on the 70
  training companies' rows (time-ordered within each company) and **certified and tested separately on each of the 15 held-out
  companies** (15 more calibrate), so the table shows a certificate per company, which is what a per-customer promise looks like
  from the outside. That is the headline and the claim in `tracks.md`.
- **Forward in time within one company** (learn on the first months, serve on the later ones) is reported only for the
  companies with ≥ 500 rows at a site (a minority: bills median 234, invoices median 188 per company); the others never reach
  the rungs.
- Calibration rows are shadow rows the weights never saw. Selection and certification never share a bound: N candidates on one
  slice are certified at δ/N. The calibration slice carries novel inputs at the deployment's rate (S4).

### 2.4 How many rows

Day-zero: 0 real + 10 seed rows per label (S2b). Certificate at 5%: 200 calibration rows certify most of a good student's
coverage, 500 saturate (S1, S28). Router budgets 5 / 10 / 15 / 20%: 1,000 / 300 / 200 / 200 (S28). A 2% budget on a 77-label
step needs ≥ 6k training rows for half the traffic (S1). The campaign's rungs are therefore 60, 200, 500, 2,000 rows seen.

### 2.5 Label noise

The teacher's self-disagreement on identical inputs floors every certificate: Sonnet-5.5 1–3%, Haiku-5.5 3.5%,
gemini-flash-lite 4–5%, qwen3-32b 5%, gpt-5.4-mini 11% (S13). StepFun's is unmeasured and is the first 200 calls on its key.
A budget below the floor is unmeasurable and the page refuses it. For the QuickBooks sites the teacher is the books: zero noise,
so the certificate is against the truth and the floor is zero.

## 3. The campaign: many runs in parallel

One run = one cell of the matrix, one process, writes checkpoints as it goes, never touches a `Policy`.

| dimension | tier 1 (tonight, must) | tier 2 (only if ported by 01:00) |
|---|---|---|
| site | model sites `categorize`, `match_invoice`, `invoice_paid`, `bill_account`; tool sites `vendor_lookup`, `bank_match` | same |
| candidate | `head`, `memory` (CPU, model sites); `exact` (tool sites); `finetune` (T4, model sites; one more cell on veri) | `hdc` (Devin's port), `online` |
| data variant | `seed+real` (the production case), `seed` (day zero: one checkpoint) | same |
| seed | 0, 1, 2 for CPU cells (row order, the holdout draw, the head's init); 0 for GPU cells | same |
| rung (a checkpoint inside the run, not a cell) | 60, 200, 500, 2,000 rows seen; `finetune` 500 and 2,000 only | same |
| cells | 4 × 2 × 2 × 3 = 48 CPU + 2 × 3 = 6 `exact` + 4 GPU (+1 veri) = 58 | +48 CPU |

`real` (no seed rows) and `head@base` are not in the matrix: S17 and S27 already answered them, and a second encoder needs a
second image. The router site `anything_odd` is measured, not trained. CPU cells take 5–30 s each (the embedding of ≤ 2,000
texts dominates and is done once per site by `prepare_site`), GPU cells 1–5 min.

**Interfaces pinned for the build (2026-10-17 23:00), so the pieces slot together:** `quickbooks.data.download(root: Path) -> Path`
(idempotent, runs inside Modal) and `quickbooks.data.rows(site_name, root) -> list[{rendered, label, group, ts}]` (group = company
id, ts = ISO date; the same render function serves the live `@step`); the campaign app exposes two more Modal functions the API
calls by name, `launch_cells(campaign, sites, limit) -> list[TrainRun]` (writes the queued rows, spawns by name, stores the fc
id in `TrainRun.config`) and `progress(campaign) -> the monitor's table`, behind `POST /campaign/run` and `GET /campaign/{name}`;
the eval split per site is one pure function: sort by (ts, rendered), the last 20% by time is `forward`, then `sha(group) mod 10`
sends buckets 0–1 to `calib` and 2–3 to `test`, the rest is the training pool (paired across candidates, held out by group).

Where a cell's data comes from (`campaign.py`): `prepare_site(site)` runs once per site on Modal and writes
`/weights/campaign/<site>/rows.jsonl` (rendered input, label, group, timestamp) and `embeddings.npy` to the Volume. For the
bookkeeper sites it exports `trace_rows` from Postgres; the label source tonight is the rule teacher (`fly.toml` runs
`TEACHER=rule`), and the sites are re-prepared and their cells re-run once the StepFun record pass lands. For the QuickBooks
sites it downloads `CL251215/QuickBooks-Hackathon-2026` inside Modal (one 26 MB tarball; the laptop cannot write Postgres) and
derives the three sites' rows with `agents/quickbooks/data.py`. Cells read only the Volume.

How it runs (`devtools.md` §Modal): the campaign is its own deployed Modal app (`dreaming-campaign`; the trainer's `train` has
`max_containers=2`) with `train_cell` (CPU, `max_containers=12`), `train_cell_gpu` (T4, 8), `prepare_site`, and a `monitor`
on a cron every 10 minutes, so nothing depends on a laptop staying awake. `just deploy-campaign` once, then
`just campaign night-1017 categorize 3` spawns three cells by name as the smoke and `just campaign night-1017` the rest; the fc
ids are printed. Starter allows 100 containers and 10 concurrent GPUs, so GPU cells queue. A cell opens its Postgres
connection only at a checkpoint, through the session pooler the `dreaming-db` secret already holds, and closes it; 12 CPU
containers keep the free project's pool happy. Cells never call a teacher (labels come from rows; StepFun's voucher tier allows
5 calls in flight). `FunctionCall.from_id(...).cancel()` is how the monitor prunes a GPU cell. Results are kept 7 days and logs
1 day, so everything lands on the Volume and in Postgres.

The three objects are `TrainRun`, `Checkpoint` and `EvalSet` in `packages/dreaming/src/dreaming/contracts.py` (fields there,
mirrored by `db/migrations/0002_campaign.sql`; not copied here so they cannot drift). `Checkpoint.scores_uri` is one `.npz` on
the Modal Volume, `modal://dreaming-weights/campaign/<run>/<rung>.npz`: eval row index, per-row score per label as float16, the
quantized confidence, the predicted label, and a second forward pass of the same rows (`Checkpoint.stable`). 500 eval rows ×
36 labels × 2 bytes ≈ 36 KB per checkpoint; the whole campaign is under 100 MB. Weights land in the `weights` table by sha as
today (bytea; `finetune` on the Volume with `uri`).

Variability is a first-class dimension, not noise to average away: three seeds per cell, the same rows for every candidate so
comparisons are paired, group holdouts, the novel-input rate held at the deployment's, and the determinism check on every
checkpoint. GPU training is not bitwise deterministic; what is certified is the CPU inference path of the downloaded weights.

The veri.studio cell: the S8 v2 fine-tune recipe as a standalone `train.py` in a `veri run-script` job (`devtools.md` §veri;
their image is Python 3.10, so it does not import the `dreaming` package), writing the same artifact the `finetune` candidate
loads: `encoder/` in Hugging Face layout plus `head.npz` in `Head.to_bytes` layout. The download is hashed and written to
`weights` like any Modal cell, `TrainRun.where = "veri"`. One cell is the sponsor requirement; the matrix does not depend on it.

## 4. Wide telemetry, and the monitor that prunes

**Stored, so anything can be asked later** (per checkpoint): the raw scores file, the eval set sha (row ids, labels, groups), the
weights sha, the config, timing per call (`embed_ms`, `head_ms`), bytes, the rung. From these, after the fact: coverage at any
budget and δ, the certificate at δ/N, realized error above any cutoff, calibration (ECE), confusion, per-label recall, per-group
error (per merchant, per company), the gain over the trivial rule, the cost per 1,000 calls at any machine price, learning curves
per candidate, seed variance. Nothing in the summary is irreplaceable; the summary exists so the monitor is one query.

**The monitor** (`apps/trainer/monitor.py`, run by the `monitor` cron of the deployed campaign app every 10 minutes, and once
on demand with `just monitor night-1017`): it reads `train_runs` and `checkpoints`, prints one table per site (candidate ×
variant: coverage at 5% and 20% at the latest rung, realized error, seconds per cell, bytes, byte-stable yes/no, status), writes
the same table as one `campaign.progress` event, and applies the pruning rule. It lives on Modal because the laptop cannot reach
Postgres and because a cron keeps pruning while everyone sleeps. The dashboard can show the event as a campaign tab later.

**Pruning rule** (successive halving along the rungs, inside each site):
1. Hard fails, at any rung: a checkpoint that is not byte-stable; realized error above its own cutoff over the budget on the
   test slice; a cell that raised. Status → `pruned` / `failed`, reason in `note`.
2. At rungs 500 and 2,000: within a site and variant, a candidate whose best seed's certified coverage at 5% is below the median
   of the candidates by more than 0.10 (two s.e. in the smokes) is pruned for the next rung; its checkpoints stay.
3. GPU cells only: a pruned `finetune` cell is cancelled (`FunctionCall.cancel()`); CPU cells finish on their own (seconds).
4. Never prune a whole family on one seed; never prune on the calibration slice (selection and certification stay apart).

What "obviously bad" looks like tonight and gets pruned by rule 1 before anyone reads a table: a head that is not byte-stable
across two passes (a thread setting leaked), a memory that certifies 0 coverage at k = 10 (two such cells in S27, unexplained),
a fine-tune whose downloaded weights do not reproduce the job's own accuracy on CPU.

## 5. The numbers we will defend, and to whom

The audience is software engineers and people who run agents on orchestration platforms. They will ask four things: how much
does it save, how often is it wrong, how do you know, and isn't this just a cache. The table answers them, per call site and
per provider, from `train_runs` / `checkpoints` and the live `trace_rows`:

| number | definition | why it holds up |
|---|---|---|
| $ per 1,000 calls, before and after | teacher dollars from provider usage (or the pinned price list) vs predictor cost (CPU ms × machine price) plus the audit sample | both are measured on the same rows; the audit fraction is in the after number |
| p50 / p95 latency, before and after | wall-clock per call from the trace rows (live) or the virtual clock (fast-forward, labelled) | the page shows which; `speculation.md` says what parallelism adds |
| error above the cutoff, with its bound | disagreement with the pinned teacher on held-out-by-group rows, Clopper-Pearson upper bound at δ (δ/N when selected) | the budget is an operator knob and the realized error is reported next to the bound, on a later time slice than the calibration |
| coverage at 2 / 5 / 10 / 20% | share of calls the certified candidate answers | computed from stored scores, so a judge can pick a budget |
| the teacher's self-disagreement | S13 protocol, 200 identical calls | printed next to every bound: a 3% bound on a 3.5%-noise teacher means nothing, and we say so |
| baselines on the same rows | the vendor / customer rule, exact-match cache, the LLM alone, Jev (QuickBooks sites) | "isn't this a cache" is answered by the cache row; "a rule does this" by the rule row; "the LLM is better" by the LLM row with its cost |
| outage completion | share of runs completed with the tool down, vs a circuit breaker | the demo's unplug moment, measured in fast-forward |
| replay | share of predicted rows byte-identical on replay | the one claim no provider can make |

Report format: paired differences to the `head` baseline with standard errors over (group, seed) cells; differences under ~0.1
coverage at small k are noise and are called noise. Synthetic data is labelled synthetic on every chart; the QuickBooks data is
synthetic by Intuit's own README and any injected mess is listed.

## 6. Cost and time, from the smokes and today's spikes

| item | time | cost | source |
|---|---|---|---|
| embedding 2,000 texts, bge-small, CPU batch 1 | ~10 s on a free core (5 ms each); ~300 ms each on Fly's shared CPU | – | S7, 2026-10-17 batch |
| `head` fit | 1–3 s warm | est. $0.001 on Modal CPU | 2026-10-17 live (1.3 s on modal:cpu) |
| `memory`, `online`, `hdc` fit | sub-second | ~$0 | S24, S25, S27 |
| `finetune` | 17 / 43 / 131 s on a 4090 at 770 / 3,080 / 9,993 rows; est. 1–5 min on a T4 | T4 $0.59/h: est. $0.02–0.05 per cell | S8 v2; modal.com/pricing |
| Modal fan-out | 8 concurrent cold starts done in 21 s; warm calls 0.6 s; Starter cap 100 containers, 10 GPUs | CPU $0.047 per core-hour | 2026-10-17 spike; pricing page |
| the tier-1 CPU cells (54) | est. 5–10 min wall-clock at 12 containers; tier 2 (+48) the same again | est. under $0.50 | arithmetic |
| the GPU cells (4, one per model site) | est. 5–15 min wall-clock | est. under $0.50 | arithmetic |
| the veri cell (`finetune` on one site) | est. 5 min boot + 1–5 min | L4 $1.05/h: est. $0.20 | docs.veri.studio rates |
| StepFun record of the ledger, one seed | 6,000 typed calls at ≈ 47 calls/min (measured throughput at 5-way concurrency; latency binds, not the 100 RPM cap): ≈ 2 h | est. $2–5 at low effort ($1.00 in / $2.70 out per M, reasoning billed as output; 20–220 output tokens per call measured) | 2026-10-17 measurement, `devtools.md` |
| StepFun self-disagreement | 200 identical calls: **3.5%** (193 / 2 / 5 unparsed); p50 1.7 s sequential, 6.4 s at 5-way concurrency | $0.10 | 2026-10-17, `smokes/RESULTS.md` |
| seed rows for six sites | one call per 15 labels | cents | S2b |

Balances 2026-10-17 evening: OpenRouter credits exhausted (top-up needed for any Anthropic-pinned row); ASI:One key at its request
quota; Modal $1 of the $30 credit until a card is added (the single blocker for the matrix); RunPod ≈ $12 (backup for GPU cells).
Campaign total so far ≈ $27 of the $100 cap.

## 7. The night, in order (times PDT)

"agent" is the Claude Code session, "human" is whoever holds the accounts, "Devin" is Cognition's agent.

| when (PDT) | what | who |
|---|---|---|
| 22:30 | the design docs frozen; `just check` green; commit | agent |
| 22:30–23:00 | step 0: `modal run smokes/modal_db.py` (migration 0002), `just deploy-trainer`, `just deploy-api`; the candidate loop in `learning.py` and the `memory` port with the two-candidate failover test | agent |
| by 23:00 | Modal payment method; StepFun voucher + key; veri credits (DM @daniung) + `veri login`; Sentry DSN; OpenRouter top-up; ASI:One Pro code; the CREAO account via the hackathon link (after the 21:00–22:00 workshop) | human |
| 23:00–00:00 | `campaign.py` bodies (`prepare_site`, `train_cell`, `train_cell_gpu`, `monitor`) and `monitor.py`; `just deploy-campaign`; smoke `just campaign night-1017 categorize 3`: seconds and $ per cell, byte-stability, the monitor's table | agent |
| 00:00–00:30 | `agents/quickbooks/data.py` (runs inside `prepare_site`), eval sets by company; launch tier 1 | agent |
| 00:00–00:30 | StepFun: parse check and floor are done (22:05, 3.5%); start `just record 2000` seed 0 (≈ 2 h at ≈ 47 calls/min) | agent |
| 00:30–01:30 | `/campaign/run`, `/campaign/{id}` and `/v1/systemone` routes in `apps/api`; redeploy | agent |
| 00:30–01:30 | the CREAO agent built against those routes (Secrets, Create Agent, `capi_` key, one shared run); the Devin session on `hdc` started (failing test committed first) | human; Devin |
| 01:30–02:30 | the veri cell (`train.py`, `veri run-script`, download, hash); the Sentry sink in the API and the first caught-log rows; re-prepare the bookkeeper sites with the StepFun labels and relaunch their cells; flip `TEACHER=stepfun` in `fly.toml` | agent |
| 02:30–08:00 | the cron prunes; reruns of cells pruned by bugs; tier 2 if `hdc` / `online` landed; the QuickBooks baselines (rule, exact cache; Jev only with OpenRouter credit); the CREAO run against `/campaign` | agent, as needed |
| 08:00–09:30 | the tables in `smokes/RESULTS.md`: which candidate won per site and why, the certificates per held-out company, cost and latency before/after | agent |
| 09:30–11:30 | the 3:00 video, the Devpost sections per track, the README blocks (Fetch agent + badges, CREAO id + run, Devin PR, StepFun call pointer), the Fetch Submission Agent, Paper frames; `demo/index.html` rewire only if everything else is done | human + agent |
| 12:00 | submit; nothing after this counts | human |

## 8. Build order (code), smallest first

Done 2026-10-17 evening: `predictors/candidates.py` (`Candidate`, `CANDIDATES`, `for_site`, the three built ones), the contract
fields and migration 0002, `TrainRequest.candidate` / `TrainResult.candidate` through both trainers and `Loader.load`, the
`PROVIDERS` teacher table with `stepfun` and `gemini`, `smokes/modal_db.py` applying every migration, `campaign.py` as a
deployable app with raising bodies. Left, matching `../DESIGN.md` §12:

0. Migrate the live database and redeploy (the contracts now have columns the live schema lacks).
1. `learning.py`: one `TrainRequest` per candidate in `for_site(site)`; shadow predictions for every loaded candidate into
   `TraceRow.shadow`; certify each at δ/N; promote the best; the spare rule on revoke. Test: the whole loop offline with `head`
   and a fake second candidate, a revocation that fails over. `memory` ported (S27 `hopfield_head`, β grid, sorted-row float64
   accumulation; the S29 script is `smokes/s29_memory_student/run.py`, written and unrun).
2. `campaign.py` bodies and `monitor.py` (`table`, `prune`); `finetune` as a candidate (S8 v2 `ft_encoder`) for `train_cell_gpu`.
3. `agents/quickbooks/data.py` + the three sites + the rule and exact-cache baselines; `/v1/systemone` and `/campaign/*` routes.
4. Ports only if time: `hdc` (Devin, from `s25_exotic/arms.py`), `online` (S27 delta + ridge resync).

## 9. Open risks

- The matrix is gated on Modal's payment method; without it, RunPod runs the GPU cells and the laptop runs the CPU cells
  (slowly, and the laptop was asked to stay idle).
- `memory` beyond 180 stored rows per task is untested (S29 written, unrun): scale, imbalance, deletion, determinism are the
  monitor's first hard-fail candidates.
- δ/N selection cost is reasoned, not measured; the campaign's stored scores make it a query afterwards.
- StepFun's self-disagreement is 3.5% (measured), so the demo's budget slider starts at 5% and refuses 2%. Its thinking cannot be
  switched off, so the router site's cheap configuration is the `step-3.5-flash` tier, not "thinking off"; the certificate
  for that site is a tier-match certificate (mission 1 as first written).
- **The record pass, measured 2026-10-17 22:38–22:55 PDT:** `just record 2000 --seeds 0` sequential against step-5-preview
  produced 232 answers in 17 minutes (≈ 14/min; per call p50 2.9 s, p90 10.1 s, max 17.5 s: StepFun's tail is long at night),
  so a 6,000-call seed is ≈ 7 h sequential, not 2. Design consequence: the recorder must run the three typed sites' calls 5-wide
  (the voucher tier's concurrency; the ledger's rows are independent, only `anything_odd` reads history) and the demo's stream
  should be cut to the rows actually recorded; the 232 answers on disk (`data/recorded.jsonl`) are a valid partial file that a
  rerun resumes. Stopped at 22:58 by request.
- The provider benchmark (`tracks.md` §4) has, tonight, the StepFun row on the bookkeeper sites (the record pass is its labelling
  pass) and the QuickBooks sites, where the books are the teacher and no provider is needed. Every other row needs its own
  labelling pass (`just record` with `TEACHER=<provider>` on that site's rows, ≈ 6,000 calls) and a key with quota: Gemini if
  its key lands, OpenRouter rows and the Jev baseline only after a top-up, ASI:One only with the Pro code. Say which rows ran.
- `TeacherSpec.floor` exists but nothing refuses a budget below it yet; `POST /budget` returning 422 under the floor is a one-line
  change once the 200-call measurement is in.
- Free-text steps have no candidate; the pruner (S6b) is their lever and is out of scope tonight.
- Supabase Free (500 MB) caps `memory_rows` around 100k across sites; the campaign's artifacts stay on the Volume.
