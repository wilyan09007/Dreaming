# Build plan (24-hour event)

Companion to [`DESIGN.md`](DESIGN.md). Source: [`../research/00-unified-runtime-brief.md`](../research/00-unified-runtime-brief.md) sections 4, 5, 10.

The brief's rule: **build order is also the fallback plan.** Each stage below is independently demoable. Never start a later stage while an earlier one is not demoable.

## 1. Stages and gates

| Stage | Scope (brief priority) | Demoable result | Gate to proceed |
|---|---|---|---|
| S0 | Harness, mocks, baseline, traces | Frontier-only workflow runs N times; cost, latency, failure rate recorded | Baseline numbers exist and are reproducible from a seed |
| S1 | Distill narrow steps (P1): `categorize`, `flag_anomaly` | Cost per run falls as steps are promoted; held-out agreement reported | At least one call type promoted with a stated bound; end-to-end accuracy within margin |
| S2 | Tool-outcome model + router + distilled recovery (P2) | Failure line falls; kill-a-tool moment works | Failure rate falls on the flaky mock; kill moment rehearsed with zero frontier recovery calls |
| S3 | Thinking-budget router (P3) | Accuracy with vs without router | S1 and S2 stable. Otherwise skip |
| S4 | Environment rollouts / GRPO (P4) | Stretch only | S1 to S3 done and slack remains. Default: do not build |

**If only S1 ships**, it is still the original top idea. **S1 + S2 is the demo as written.** S3 and S4 are upside.

## 2. Workstreams (four people)

| # | Workstream | Owns | Depends on |
|---|---|---|---|
| A | **Harness and trace store** | SDK decorators, gateway skeleton, trace schema and emitter, store, Sentry and Tiger Data wiring, cost accounting | Contracts (`CONTRACTS.md`) |
| B | **Training pipeline** | Dreamer loop, predictor ladder, calibration, promotion test, audit sampler, demotion; Veri training jobs (distillation SFT) | A's trace store |
| C | **Tool model and recovery** | Health posterior, GBDT failure prior, router, `recover_from_tool_failure` call type, mock flakiness, **kill-a-tool moment** | A's gateway, B's promotion machinery |
| D | **Evaluation, dashboard, demo, submission** | Eval harness, dashboard (`UI/`), replay mode, demo video, Devpost page, sponsor registrations (Agentverse, Devin, Intuit dataset) | Everyone's event stream |

Reserve **the last 4 hours for the demo** (video, rehearsal, submission). The brief notes every 2026 winner report says the video took longer than expected.

## 3. Timeline

Hours are from kickoff. Adjust to the real schedule.

| Hours | A: Harness | B: Training | C: Tools | D: Eval/UI/Demo |
|---|---|---|---|---|
| 0 to 2 | Freeze `CONTRACTS.md`. Repo skeleton | Offline predictability check on sample data (see `04-evaluation-methodology.md` section 7) | Mock services skeleton, flakiness spec | Get dataset; labels check; register sponsors |
| 2 to 6 | Gateway, decorators, traces to store, cost accounting | Tier 0 to 2 predictors, comparator, shadow mode | Health posterior, router over alternates | Baseline harness; first baseline runs; dashboard shell with fake data |
| 6 to 10 | Audit hooks, Sentry export | Calibration, `tau_serve`, promotion test | GBDT failure prior; recovery call type traced | Live event stream into dashboard; cost and failure charts |
| 10 to 14 | Hardening, registry hot-swap | Promotion and demotion live; audit decay | Distilled recovery shadowed and promoted | Held-out eval with tiers 0/1 off; first full end-to-end run |
| 14 to 17 | Replay mode support | Veri SFT job (tier 3) if time; drift gate | Kill-a-tool flow end to end; probes optional | Ablations; multi-seed runs |
| 17 to 20 | Bug bash | Bug bash | Kill moment rehearsal x3 | Rehearse run of show; finalize charts |
| **20 to 24** | **Freeze code** | **Freeze code** | **Freeze code** | **Video, Devpost, final rehearsal, submit** |

## 4. Definition of done for the demo

- [ ] 300-run schedule completes in about 3 minutes live, or via replay with the same shape.
- [ ] Cost-per-run line falls, with visible promotion steps, to at most 25% of baseline (target, not promise).
- [ ] Failure-rate line falls with the tool model on.
- [ ] Kill-a-tool: mid-run failure handled with 0 frontier calls; later runs avoid the dead endpoint.
- [ ] Held-out agreement reported with tiers 0/1 disabled.
- [ ] Baseline vs Dreaming table: latency, cost, accuracy.
- [ ] Rehearsed three times clean, including the replay fallback.
- [ ] README states the Devin attribution; Devpost page and video submitted.

## 5. Mapping to the longer milestone list

`DESIGN.md` section 14 lists milestones M0 to M7 for the full project. For the event: M0 to M1 = S0; M2 to M3 = S1; M4 = S2; M5 = dashboard and replay (workstream D); M6 and M7 are post-event.

## 6. Risks to the plan

| Risk | Early signal | Response |
|---|---|---|
| Steps are not predictable on held-out rows | Offline check in hours 0 to 2 | Change which steps are distill targets; widen features; lower expectations on the cost line |
| Too little ships (the brief's flagged execution risk) | S1 gate not met by hour 10 | Cut S2 scope to the router and mock failures only; drop S3 and S4 |
| Kitchen-sink pitch | Slide lists mechanisms | Lead with the single principle; show two lines and one moment |
| Training job latency on Veri | First job not back in an hour | Keep local tiers 0 to 2 as the primary path; Veri is additive |
| Network failure on stage | Flaky venue Wi-Fi | Replay mode; local mocks; prebuilt cache of frontier outputs |
| Video takes longer than expected | Not started by hour 20 | Script and shot list prepared earlier (`video/README.md`) |
