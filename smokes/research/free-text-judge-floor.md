# Free-text judge floor: does draft_note "just need a judge"?

Researched 2026-10-17. Scope: LLM-as-judge for equivalence of short business prose, the agreement floor between two temperature-0 generations, canonicalization and typing as ways to lower it, and certifying against a noisy judge. "unverified" = not confirmed from a primary source. **Caveat on method:** WebFetch could not resolve arxiv.org, aclanthology.org, alphaxiv.org (DNS error), so every external number below comes from WebSearch result summaries, not from reading the papers; all are marked "(search summary)". In-repo numbers come from `smokes/s16_pruner_note/results.json` and `smokes/FINDINGS.md`.

## Bottom line

1. **No, a judge alone does not fix draft_note.** S16 already used a judge and still measured a 13-16% "different" rate between two independent full-prompt notes (n=100, upper 95% bounds 19.9% / 23.3%). A judge changes what counts as "different"; it does not make two samples of a prose step agree. Only the rubric and the generation side can move the floor.
2. **The rubric may matter a lot, but the S16 data do not show how much of the floor it explains.** The primary judge (sonnet-5.5) agreed with itself 99.3% (kappa 0.985, n=150), but flipped on order swap (86.7%, kappa 0.71, n=30) and disagreed with gpt-5.4-mini (86%, kappa 0.68). The strict vs "core facts" gap (33% vs 10% DIFFERENT, kappa 0.36) was measured on `sample[::5]`, a mixed 30 pairs (about 10 floor, 10 ablation, 10 pruned-policy; core 3/30, 95% CI 2.1-26.5%), not on the floor. The equivalence definition must be frozen before anything is certified, and both rubrics must be run on all 100 floor pairs first.
3. **S16b recipe:** (a) canonicalize both notes deterministically (extract vendor, amount, date, invoice id/amount, flag; normalize numbers and dates) and call DIFFERENT without a judge if the fact sets conflict; (b) judge only the residual (flag reason, follow-up) in both presentation orders, sonnet-5.5 pinned, second family as adjudicator; (c) report strict and core rubrics both, on all 100 floor pairs, and pick the certificate rubric and the aggregation rule BEFORE seeing the results. Caveat: S16's 13% "confirmed" figure is already a majority over 3 reference notes (`core.py` `verdict`), and "both orders must say DIFFERENT" plus a judge-free path that fires only on conflicts are further SAME-favouring rules, so each lowers the rate by construction. Whatever is kept must be paired with its false-SAME rate on the 50 ablation pairs (judge DIFFERENT there: sonnet 54%, mini 36%).
4. **Measured state on free prose: no certificate below 20%.** S16's pruned re-ask did run (results.json, log.txt; FINDINGS l.74 says it did not, see Open questions): learned+reason A removed 28.6% of tokens at 14% raw / 12% confirmed DIFFERENT vs a 16% / 13% floor (n=100, p=0.82); no certificate at 5/10/15%; only the confirmed metric certified at 20% (pooled n=180, coverage 0.50, bound 0.185). "8-10%" is a hypothesis for S16b, not an estimate from data. 5% needs a floor under ~2-3%, which only typed fields plausibly give: priors are S6b 1.7% (yes/no field) and S13 3.5% (pinned haiku-5.5, 77 labels). A multi-field record could exceed 5%, so measure record-level disagreement.
5. **Calibration rows:** about 200 human-adjudicated pairs, but allocate by the TRUE (human) label as in Lee et al. (https://arxiv.org/abs/2511.21140), or reweight by the judge's marginal rate; stratifying by judge label only gives PPV/NPV, not the Se/Sp that Rogan-Gladen needs. Also stratify by scenario. Prompts: prompts.json has 300 but only 140 are labelled, and the S16 floor used n=100. Add a fixed human anchor subset re-scored by the judge as the standing drift check (arXiv 2606.15474, repo S18). Report a corrected estimate (Rogan-Gladen or PPI) plus the raw judge rate; the certificate is "disagreement under judge J with measured error (Se, Sp)", not an exact-match claim.
6. **Typing is the standard workaround in practice but I found no paper that measures it** for run-to-run agreement (see section 4). S16b should include a typed arm (JSON fields via pinned endpoint, template render) so we measure it ourselves.
7. **The floor is scenario-dependent** (log.txt l.54-58): normal 5.9% (2/34), first 12.5%, dup 16.7%, noinv 19.0% raw / 9.5% confirmed, spike 35.3% / 29.4% (n=17). Disagreement sits in judgment-heavy cases (spike reasoning, missing invoice), which a vendor/amount/date canonicalizer does not touch. Report and calibrate per scenario; typing only spike/noinv reasoning may beat a global budget.
8. **Provider pin:** the sonnet judge was served by "Claude Platform on AWS" 3,122 times but "Anthropic" 3 times, so the pin leaked. If only AWS-hosted endpoints are reachable or reliably pinnable, route judge calls through an AWS machine acting as the proxy so the served provider is fixed (user suggestion, not tested). 11 judge outputs were also unparsed.
9. Do not spend votes re-asking the same judge in the same order: it was 99% self-consistent. Spend them on order swaps and a second family.

## 1. LLM-as-judge for equivalence of free text (2025-2026)

| Source | URL | Date | What it shows | Numbers |
|---|---|---|---|---|
| Survey on LLM-as-a-Judge | https://arxiv.org/html/2411.15594v6 | 2025 (v6) | Pairwise judging is reported as more reliable than pointwise scoring (search summary) | qualitative |
| Judging the Judges: position bias (IJCNLP-AACL 2025) | https://aclanthology.org/2025.ijcnlp-long.18/ | 2025 | 15 judges on MTBench/DevBench; position bias is not chance, varies by judge and task, grows as the quality gap shrinks (search summary) | no figures retrieved |
| Reliability without Validity (21 judges, ~541k judgments) | https://arxiv.org/html/2606.19544v1 | Jun 2026 | High test-retest reliability can hide severe position bias; exact-match agreement exceeds kappa for every judge ("kappa deflation" 33.8-41.2 points (the note originally said 41.3)); judge rankings shift up to 14 positions across benchmarks (search summary) | per-judge position bias 0.002 vs 0.192 (not confirmed on re-check) |
| Matter to Mechanism benchmark | https://arxiv.org/pdf/2606.02258 | 2026 | Pairwise judging by Gemini 2.5 judges on materials-science outputs is informative but order-sensitive; weak transfer to short bookkeeping notes (search summary) | mean order-flip rate 0.236; judge-to-judge agreement 0.32 |
| Judge's Verdict | https://arxiv.org/pdf/2510.09738 | Oct 2025 | Compares judge kappa with a human-human baseline and tests whether the pattern is human-like (search summary) | human-human kappa 0.801 |
| Weak judges, strong panel (vendor blog, unverified) | https://orq.ai/blog/llm-juries-in-practice | 2026 | Suggests kappa ~0.65 as a working target for pairwise comparisons; self-reported | kappa ~0.65 |
| Replacing Judges with Juries (PoLL) | https://arxiv.org/pdf/2404.18796 | 2024 | Panel of diverse smaller models; includes a "reference semantically contained in answer" task (search summary) | none retrieved |
| TrustJudge | https://arxiv.org/pdf/2509.21117 | 2025 | Documents non-transitive equivalence (A=B, B=C, A!=C) in pairwise judging (search summary) | qualitative |
| JudgeSense | https://arxiv.org/html/2604.23478v1 | 2026 | Several judges return one label on every pair, so perfect consistency can be anchoring, not robustness (search summary) | 494 paraphrase pairs |
| xVerify / CompassVerifier (answer equivalence for math/QA) | https://arxiv.org/html/2504.10481 ; https://aclanthology.org/2025.emnlp-main.1698.pdf | 2025 | Dedicated small verifiers beat general judges on formula equivalence; these are benchmark F1 against their own labels, not human agreement (search summary) | CompassJudger-1-32B below 85% on math equivalence |
| Calibrate, Don't Curate | https://arxiv.org/pdf/2605.09702 | May 2026 | Calibrated full panel beats keeping only the top-k accurate judges; optional split-conformal on top (search summary) | RewardBench2 NLL 0.006 (all judges) vs 0.013 (top-5) |
| In repo: S16 judge validation | smokes/s16_pruner_note/results.json | 2026-10 | sonnet self-agree 0.993 (kappa 0.985, n=150); sonnet-vs-mini agree 0.86 (kappa 0.68); swap agree 0.867 (kappa 0.71, n=30); strict-vs-core rubric kappa 0.36 (n=30); on floor pairs sonnet DIFFERENT 14% vs mini 14% (n=50), on ablation pairs 54% vs 36% | as listed |

Reading: no source gives a human-agreement number for "same facts and conclusion" on short bookkeeping prose. What transfers: pairwise beats pointwise, order swapping is mandatory, self-consistency is not validity, and panels of different families help. Our own data say the same thing at small n.

## 2. Floor between two temperature-0 generations; do structure, pinning, canonicalization lower it?

| Source | URL | Date | What it shows | Numbers |
|---|---|---|---|---|
| Atil et al., non-determinism of "deterministic" settings (Eval4NLP 2025; 2024 preprint https://arxiv.org/abs/2408.04667, GPT-3.5/4o-era models) | https://aclanthology.org/2025.eval4nlp-1.12.pdf | Aug 2024 / 2025 | 5 API LLMs, 8 tasks, 10 runs at "deterministic" settings: accuracy varies, raw outputs rarely string-identical (search summary) | accuracy variation up to 15%, best-worst gap up to 70%; a secondary site quotes GPT-4o median raw-string agreement 3% (unverified) |
| Thinking Machines, Defeating Nondeterminism | https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ | Sep 2025 | Cause is batch non-invariance; batch-invariant kernels give 1 unique output of 1000 (see determinism.md) | 80 unique outputs without, 1 with |
| In repo: S13 teacher noise | smokes/FINDINGS.md (S13, l.56) | 2026-10 | Two identical T=0 calls on a 77-label classifier disagree 1-3% (sonnet, default routing with temperature dropped, "mostly empty-answer artifacts"), 3.5% (pinned haiku-5.5, reasoning off), 11% (gpt-5.4-mini); 4 of 5 teachers drop `temperature=0` under default OpenRouter routing; different endpoints of one model disagree more than one endpoint with itself | see FINDINGS |
| In repo: S16 floor | smokes/s16_pruner_note/results.json | 2026-10 | Two independent full-prompt haiku-5.5 notes (pinned endpoint, reasoning off), strict judge: 16% raw / 13% confirmed DIFFERENT, n=100 | UB95 23.3% / 19.9% |
| In repo: S16 floor by scenario | smokes/s16_pruner_note/log.txt l.54-58 | 2026-10 | Raw/confirmed DIFFERENT: normal 5.9/5.9 (n=34, CI 0.7-19.7), first 12.5/12.5 (16), dup 16.7/16.7 (12), noinv 19.0/9.5 (21), spike 35.3/29.4 (17) | as listed |
| In repo: S16 pruned re-ask | smokes/s16_pruner_note/results.json, log.txt | 2026-10 | learned+reason A: 28.6% tokens removed, 14%/12% vs floor 16%/13% (p=0.82); no certificate at 5/10/15%; confirmed@20% passes (pooled n=180, coverage 0.50, bound 0.185), raw@20% fails | as listed |
| In repo: S6/S6b | smokes/FINDINGS.md | 2026-10 | HotpotQA free text: 7-11% strict self-disagreement even with temperature honoured, 2-5% under a T=0 judge; typed step with pinned teacher fixed it | see FINDINGS |
| SLOT, structured output post-processing (EMNLP Industry 2025) | https://aclanthology.org/2025.emnlp-industry.32.pdf | 2025 | Constrained decoding guarantees format, not content; a fine-tuned 1B post-processor reaches schema/content fidelity (search summary) | 88.9% schema accuracy, 81.7% content similarity (self-reported) |
| JSONSchemaBench | https://arxiv.org/html/2501.10868v1 | 2025 | Benchmarks six constrained-decoding frameworks on 10K schemas for coverage, speed, quality (search summary) | no variance numbers |
| Constrained decoding in small LLMs | https://arxiv.org/pdf/2609.23742 | 2026 | Eliminates structural failures but leaves a scale-dependent semantic gap (search summary) | none retrieved |

Findings:
- Typical agreement of raw prose at "temperature 0" is low as strings and moderate as meaning; the literature reports mostly string-level or accuracy-level variation, not meaning-level agreement on short prose. Our 13-16% is the only meaning-level measurement for this task type.
- Provider pinning and seeds help but do not remove the floor (S13: even honoured T=0 is not deterministic; Anthropic endpoints lack `seed`). Batch-invariant serving would, but only on self-hosted models.
- Constrained decoding fixes shape, not content. I found no study showing that JSON-schema generation lowers run-to-run content disagreement; the expected mechanism (fewer free choices, closed vocabularies for flag reason) is plausible but unmeasured. Treat as hypothesis.
- Canonicalization before judging: I found no literature on it for judge pipelines (search result for numeric normalization before pairwise comparison came back empty). Mechanically it removes format-only differences ("$48.20" vs "48.2 USD", date formats) and lets a deterministic check decide the fact conflicts, leaving the judge a smaller question. Expected effect is on the strict-rubric floor; size unknown.

## 3. Certifying against a noisy judge

| Source | URL | Date | What it shows | Numbers |
|---|---|---|---|---|
| How to Correctly Report LLM-as-a-Judge Evaluations (ICML 2026) | https://arxiv.org/abs/2511.21140 | Nov 2025, v4 | Rogan-Gladen correction with sensitivity and specificity from a human calibration set; CI includes both test and calibration variance; unbiased under test/calibration shift; allocating calibration examples between correct and incorrect matters more than total judgments (search summary) | none retrieved |
| Efficient Inference for Noisy LLM-as-a-Judge Evaluation (ICML 2026) | https://arxiv.org/abs/2601.05420 | Jan 2026 | Compares Rogan-Gladen with prediction-powered inference; EIF estimator beats PPI, both beat RG on finite-sample bias and CI efficiency; open-source package (search summary; orthogonal-methods.md says RG intervals were very wide on their data) | none retrieved |
| Noisy but Valid (ICLR 2026) | https://arxiv.org/abs/2601.20913 | Jan 2026 | Estimate judge TPR/FPR on a small human set, variance-corrected threshold, finite-sample Type-I control (search summary) | none retrieved |
| Bias and Uncertainty in LLM-as-a-Judge Estimation | https://arxiv.org/abs/2605.06939 | May 2026 | Corrections fail when judge quality (Youden J = Se + Sp - 1) is low or calibration is unstable across systems; MMLU-Pro case with a sign reversal (search summary) | none retrieved |
| Trust or Escalate (ICLR 2025) | https://arxiv.org/abs/2407.18370 | 2025 | Exact binomial bound on judge-human disagreement with fixed-sequence testing (from certification-deferral.md) | see that note |
| Einbinder et al., conformal under label noise (JMLR 2024) | https://arxiv.org/abs/2209.14295 | 2024 | Conformal stays conservative under dispersive label noise (from orthogonal-methods.md) | qualitative |

Findings:
- The literature's answer to "certify with a noisy judge" is to estimate the judge's Se/Sp on a human-labelled slice and correct or bound the downstream rate (Rogan-Gladen, PPI, Noisy-but-Valid). PPI-style estimators are more efficient than RG in the binary case. For our Clopper-Pearson machinery, the practical route is to bound the judge's DIFFERENT rate (what S16 does) and then widen by the judge's measured false-SAME rate; or use PPI on the audit stream.
- Defining the certificate against the judge rather than exact match is the only workable choice for prose (exact match would be near 100% disagreement), but it must be stated as "agreement under judge J, rubric R, pinned versions", with the judge's Se/Sp reported. A judge change is a certificate change.
- Corrections degrade as judge quality falls (claimed by arXiv 2605.06939, which I could not verify; the 'Fiedler' attribution is unsourced); with kappa 0.68 between our two judges, expect wide intervals unless the rubric is tightened first.
- Drift: a fixed human-labelled anchor set re-scored by the judge attributes score drift to judge or system (Who Drifted, https://arxiv.org/pdf/2606.15474, Jun 2026; flagged a silent judge version bump in 60/60 runs, self-reported). Judge upgrades are not interchangeable and repeated same-judge samples add little when errors correlate (https://arxiv.org/abs/2607.08535); a 32-judge panel is worth about 2-6 humans (https://www.alphaxiv.org/abs/2609.21277). Repo S18 already recommends an anytime-valid audit bound. All search summaries.
- Conformal on prose is not established: I found no paper that applies conformal risk control with a noisy binary judge for step-replacement certificates; only the generic conformal-under-noise result and the judge-calibration add-on.

## 4. Is "typing the step" the standard workaround?

- Practice: extract fields with schema-constrained output, then render with a template, is the common engineering pattern for repeatable reports (practitioner guides, e.g. https://dev.to/pockit_tools/llm-structured-output-in-2026-stop-parsing-json-with-regex-and-do-it-right-34pk; unverified, blog).
- Academic evidence that it lowers run-to-run content disagreement: not found. SLOT and JSONSchemaBench measure schema validity and fidelity, not repeat agreement. A guide at https://ofox.ai/blog/llm-benchmarks-text-extraction-summarization-2026/ notes a schema constrains shape, not truth (blog).
- In our repo the only evidence is indirect: S6b's typed step with pinned teacher fixed the pruner problem, and S13 shows closed-label calls floor at 3.5% on pinned haiku-5.5 (77 labels; sonnet's 1-3% was under default routing) and S6b's yes/no field at 1.7%. Multi-field records may exceed 5%.
- Cost of typing: the note text is no longer produced by the model, so a user whose step is genuinely free prose cannot be typed without changing the product behavior. Typing is a recommendation for call sites that are really "fields to prose", which draft_note is (S16: the note depends on transaction 63%, invoice 78%, flag 95% spans).

## What this changes in the plan

- FINDINGS mission table row 5 and the S16 line: replace "a judge fixes prose" with "a judge defines the metric; the floor is set by the scenario, the rubric and sampling; typed rendering is the hypothesised route to a 5% budget". Also correct FINDINGS l.74 if the re-ask did run. The user's "we should have benched that" is answered by S16: it was benched and returned a floor, not a pass.
- S16b spec gains three arms on the cached 140 prompts plus the new vendor-grouped split: (A) free note, strict rubric (replicates 13-16%); (B) free note, canonicalize plus core-facts rubric, order-swapped, two families; (C) typed: pinned haiku-5.5 emits JSON fields, template renders the note; floor measured as field-level disagreement. Report floor with Clopper-Pearson upper bound for each arm.
- Fix the rubric text and the aggregation rule before collecting calibration labels or seeing floor numbers; do not tune either to lower the floor. Freeze judge model, endpoint and prompt hash in every trace row, as with teachers.
- Certificate wording for prose steps: "disagreement rate with the teacher's note under judge J (Se, Sp measured on n human labels), upper 95% bound", with the judge version pinned.
- Budget statement in the product: measured, free prose certified nothing below 20% in S16; 8-10% is an S16b target for the canonicalized and typed arms, and 5% only if typed record-level disagreement supports it.

## Open questions

- What is the human-judged floor? We have no human labels for S16; judge-vs-human agreement on this rubric is unknown (kappa 0.68 sonnet vs mini is judge-vs-judge only).
- How much of the 13-16% is canonicalizable (format-only) versus real content difference (omitted evidence, different follow-up)? The core-rubric figure of 10% is 3/30 on a mixed sample, not the floor.
- Does JSON-then-template rendering reduce disagreement on extracted fields below 3%? Not measured anywhere I could find.
- Which estimator for the audit stream: Rogan-Gladen (simple, wide intervals) or PPI with the open-source package? Needs a sim on S16 data (sonnet-DIFFERENT 34.7% vs mini 28.7% on the 150 validation pairs).
- Primary-source numbers for sections 1 and 3 were not read (fetch blocked); re-verify the cited papers before quoting externally.

---

## Critic review (2026-10-17)

Method: re-read `smokes/s16_pruner_note/{results.json,log.txt,run.py,core.py}` and `smokes/FINDINGS.md`. I tried to spot-check by WebFetch, but this container cannot resolve arxiv.org, export.arxiv.org, aclanthology.org, alphaxiv.org, openreview.net, api.semanticscholar.org, thinkingmachines.ai or orq.ai (ENOTFOUND), so external checks used WebSearch result summaries. Those checks are abstract-level only and are marked "(search summary)".

### Bottom line

1. **Most severe: the note's key S16 rubric number is not a floor number.** The "strict 33% vs core 10% DIFFERENT on the same 30 pairs" comes from `small = sample[::5]` of the 150 validation pairs (`run.py` l.216-222). That is about 10 floor, 10 ablation and 10 pruned-policy pairs. Core 3/30 has a 95% CI of 2.1-26.5%. So "most of the floor is rubric strictness" (bottom-line item 2) and "the core-rubric figure of 10%" (open questions) are not measured on the floor. S16b must run the core rubric on all 100 floor pairs before choosing it.
2. **The S16b recipe lowers the rate by construction, and the note says nothing about it.** S16's "confirmed" floor (13%) is already a majority over 3 reference notes (`core.py` `verdict`), and the note never explains this. Adding "DIFFERENT only if both orders say DIFFERENT", plus a judge-free path that only fires on conflicts, stacks two more SAME-favouring rules. Each one cuts the false-DIFFERENT rate and raises the false-SAME rate. The note also picks the core rubric *because* it gave 10% against 33%, which breaks its own rule, "do not tune the rubric after seeing the floor". Fix: freeze the rubric and aggregation rule, then measure sensitivity to real differences on the ablation pairs. The judge rated 54% of those DIFFERENT under sonnet and 36% under mini, which gives a ready positive set.
3. **The note misses that S16 already ran the pruned re-ask and certified nothing below 20%.** FINDINGS l.74 says the re-ask "was not run", but `results.json` and `log.txt` contain it. learned+reason A removed 28.6% of tokens at 14% raw / 12% confirmed DIFFERENT, against a 16% / 13% floor (n=100, discordant 8/10, p=0.82). Every certificate at 5/10/15% had cutoff = inf. In results.json (pooled n=180) only the confirmed metric certified at 20% (coverage 0.50, bound 0.185); raw certified nothing. A later log.txt run (n=220) shows raw@20% at coverage 0.45, bound 0.197. That contradicts the "8-10% achievable on free prose" estimate in bottom-line item 4, and the note does not say so. Report 15-20% as the measured state and treat 8-10% as a hypothesis for S16b. Also flag the FINDINGS discrepancy to the owner (not edited here).
4. **The floor is scenario-dependent, which the note ignores.** From the floor by scenario (`log.txt` l.54-58): normal 5.9% (2/34, CI 0.7-19.7%), first 12.5%, dup 16.7%, noinv 19.0% raw / 9.5% confirmed, spike 35.3% raw / 29.4% confirmed (n=17). The disagreement sits in judgment-heavy scenarios: spike reasoning, missing invoice. It does not sit in vendor/amount/date formatting, which is what the proposed canonicalizer normalizes. S16b should report the floor per scenario and stratify calibration rows by scenario. A per-scenario budget, or typing only the spike/noinv reasoning, may beat a global 8-10%.
5. **The calibration design does not estimate Se/Sp directly.** Stratifying 100 judge-SAME and 100 judge-DIFFERENT pairs estimates PPV and NPV. Rogan-Gladen needs P(judge | human), so you have to reweight by the judge's marginal rate, or follow Lee et al.'s adaptive allocation over truly-correct and truly-incorrect items. Also, "300 prompts for the noise floor" does not match S16. prompts.json has 300 prompts, but only 140 are labelled and the floor used n=100.
6. **S13 is mis-cited.** "S13 shows 1-3% for pinned closed-label calls" is wrong. Sonnet's 1-3% came under default routing, where temperature was dropped, and FINDINGS calls it "mostly empty-answer artifacts". Pinned haiku-5.5 with reasoning off was 3.5% on 77 labels. The proposed typed arm uses haiku-5.5. Its relevant priors are S6b's 1.7% on a yes/no field and S13's 3.5% on 77 labels, so a multi-field record could exceed 5%. Measure record-level disagreement, not just per-field.
7. **External citations: 5 of 6 spot-checks match at abstract level, 1 is unverified, 1 is dated.**
   - Matches: Lee et al. 2511.21140, Chen et al. 2601.05420, Noisy-but-Valid (ICLR 2026), Judge's Verdict 0.801, and Matter-to-Mechanism 0.236 / 0.32.
   - Minor: Reliability-without-Validity gives 33.8-41.2, not 41.3.
   - Unverified: "Bias and Uncertainty in LLM-as-a-Judge Estimation" (2605.06939) and its "Fiedler" attribution did not surface in my search.
   - Dated: Atil et al. is a 2024 preprint (2408.04667, GPT-3.5/4o-era models) re-published at Eval4NLP 2025, so it is not a 2025-2026 result.
8. **Missed 2026 work that bears on the "a judge change is a certificate change" claim.**
   - "Who Drifted" (2606.15474): a human anchor set plus an e-process attributes drift to the judge or the system.
   - "When the Judge Changes, So Does the Measurement" (2607.08535): judge upgrades are not interchangeable, and repeated-sample juries add little when errors are correlated.
   - "How Many Humans Are 32 LLM Judges Worth?" (2609.21277): panels equal about 2-6 humans.
   - The repo's own S18 anytime-valid audit (FINDINGS l.72) is also missing. Together these support a fixed human anchor set re-scored by the judge as the standing check, not a one-off 200-row calibration.

### Evidence: in-repo checks

| Source | Path | What it shows | Numbers |
|---|---|---|---|
| S16 validation sampler | smokes/s16_pruner_note/run.py l.216-222 | `small = sample[::5]` over 50 floor + 50 abl + 50 pol pairs, so the swap/core-rubric subset is mixed, not floor | n_small 30, about 10 floor |
| S16 judge block | smokes/s16_pruner_note/results.json `judge` | core 0.10 vs strict 0.333 on the mixed 30; swap agree 0.867 (4/30 flips); 11 unparsed judge outputs; sonnet D / mini S 15 vs S/D 6 (sonnet stricter) | core 3/30, 95% CI 2.1-26.5% (computed) |
| S16 "confirmed" definition | smokes/s16_pruner_note/core.py `verdict` | cons = DIFFERENT only if a majority of 3 reference notes say DIFFERENT, stopping at the first SAME | already SAME-favouring |
| S16 floor by scenario | smokes/s16_pruner_note/log.txt l.54-58 | normal 0.059, first 0.125, dup 0.167, noinv 0.190/0.095, spike 0.353/0.294 (raw/confirmed) | n 34/16/12/21/17 |
| S16 pruned re-ask | results.json `policies`, `cert`; log.txt l.97, 171-186 | learned+reason A 28.6% tokens removed, 0.14/0.12 vs floor 0.16/0.13, p=0.82; no certificate at 5/10/15%; results.json pooled n=180: cons@20% passes (coverage 0.50, bound 0.185), raw@20% fails; later log.txt run n=220: raw@20% coverage 0.45, bound 0.197 | n=100 held-out, 120-180 cal |
| S16 providers | log.txt last line | sonnet judge served by "Claude Platform on AWS" 3,122 times and "Anthropic" 3 times, so the pin was not airtight | 3 leaks |
| S13 | FINDINGS.md l.56 | sonnet 1-3% "weakly determined, mostly empty-answer artifacts", temperature dropped; haiku-5.5 reasoning off 3.5%; gpt-5.4-mini 11% | 77-label step |
| S6b | FINDINGS.md l.64 | pinned haiku-5.5 typed yes/no step: 1.7% pairwise | 200 questions |
| S18 | FINDINGS.md l.72 | anytime-valid audit bound as the standing certificate; slow (about 5,300 calls for a 1% budget at 10% audits) | 192 configs |

### Evidence: external spot-checks (WebFetch blocked; all from WebSearch summaries)

| Source | URL | Date | Note's claim | Check result |
|---|---|---|---|---|
| Lee, Zeng, Jeong, Sohn, Lee, How to Correctly Report LLM-as-a-Judge Evaluations | https://arxiv.org/abs/2511.21140 ; code https://github.com/UW-Madison-Lee-Lab/LLM-judge-reporting | Nov 2025, ICML 2026 poster (https://icml.cc/virtual/2026/poster/63607) | RG correction, CI over test plus calibration, allocation matters | Matches (search summary). Allocation is adaptive between Se and Sp estimates, i.e. by true label, not by judge label |
| Chen, Lu, Li, Guo, Li, Efficient Inference for Noisy LLM-as-a-Judge Evaluation | https://arxiv.org/abs/2601.05420 ; https://icml.cc/virtual/2026/poster/61131 | Jan 2026, ICML 2026 | EIF beats PPI, both beat RG; open-source package | Matches (search summary); EIF is asymptotically equal to PPI++ for binary outcomes |
| Feng et al., Noisy but Valid | https://arxiv.org/abs/2601.20913 ; https://iclr.cc/virtual/2026/poster/10008079 | Jan 2026, ICLR 2026 | TPR/FPR from a small human set, finite-sample Type-I control | Matches (search summary) |
| Norman, Rivera et al., Reliability without Validity | https://arxiv.org/html/2606.19544v1 | Jun 2026 | kappa deflation 33.8-41.3 points, 21 judges, about 541k judgments, ranks shift up to 14 | Mostly matches; the summary gives 33.8-41.2. The per-judge position-bias numbers (0.002 / 0.192) were not confirmed (search summary) |
| Han et al., Judge's Verdict | https://arxiv.org/abs/2510.09738 | Oct 2025 (under review, ICLR 2026) | human-human kappa 0.801 | Matches (search summary); RAG/agentic QA, 54 judges |
| Matter to Mechanism | https://arxiv.org/pdf/2606.02258 | 2026 | flip rate 0.236, judge-judge agreement 0.32 | Matches (search summary). Judges are Gemini 2.5 Pro/Flash on materials-science research outputs, so transfer to short notes is weak |
| Atil et al., Non-determinism of "deterministic" settings | https://arxiv.org/abs/2408.04667 ; https://aclanthology.org/2025.eval4nlp-1.12/ | Aug 2024 preprint, Eval4NLP 2025 | up to 15% accuracy variation, 70% best-worst gap | Numbers match (search summary), but the work is 2024-era (GPT-3.5/4o-class models). Presenting it as 2025 overstates how current it is |
| Bias and Uncertainty in LLM-as-a-Judge Estimation ("Fiedler") | https://arxiv.org/abs/2605.06939 | 2026? | corrections fail at low Youden J; MMLU-Pro sign reversal | **Unverified**: a title search did not surface it, and the author name has no source in the note |
| Calibrate, Don't Curate | https://arxiv.org/pdf/2605.09702 | May 2026 | NLL 0.006 vs 0.013 on RewardBench2 | Matches (search summary) |

### Evidence: missed 2025-2026 work

| Source | URL | Date | What it shows | Numbers |
|---|---|---|---|---|
| Li, Who Drifted: the System or the Judge? | https://arxiv.org/pdf/2606.15474 | Jun 2026 | A fixed human anchor set re-scored by the judge, with a second e-process on the judge-human gap, attributes drift to the judge or the system | silent judge version bump flagged in 60/60 runs; strict-prompt change attributed 110/120; rolling z-test false-alarmed on 75% of drift-free streams (self-reported, search summary) |
| Yang, Hou, Yang, When the Judge Changes, So Does the Measurement | https://arxiv.org/abs/2607.08535 | Jul 2026 | Judge upgrades are not interchangeable; stronger judges reduce but do not remove position bias; repeated-sample juries add little when errors are correlated | qualitative (search summary) |
| How Many Humans Are 32 LLM Judges Worth? | https://www.alphaxiv.org/abs/2609.21277 | Sep 2026 | Effective human-equivalent size of a 32-judge panel on ChaosNLI | about 2.3-3.75 humans by distributional error, 2.0-2.2 effective votes in the binary case (search summary) |
| Systematic-review extraction with constrained JSON | https://arxiv.org/pdf/2608.26150 | Aug 2026 | Claims fixed-category schemas limit output variance, but ran no replicates | none measured (search summary). Supports the note's "unmeasured" status for typing |

### What this changes in the plan

- S16b first step: rerun strict and core rubrics on **all 100 floor pairs**, in both orders and with both judge families, before choosing a rubric. Then report the floor per scenario.
- Pre-register the aggregation rule. If a SAME-favouring rule (cons, both-orders, judge-free on conflict) is kept, report its false-SAME rate on the 50 ablation pairs plus the human set.
- Change the budget line to: "free prose measured: no certificate below 20% (S16); 8-10% is a target for S16b's canonicalized and typed arms."
- Human calibration: stratify by scenario and by human label (Lee et al. allocation), and keep a fixed anchor subset for drift checks (Who Drifted, S18).
- Typed arm: measure record-level disagreement over all fields with pinned haiku-5.5. Compare it with S6b's 1.7% and S13's 3.5% before promising 5%.

### Open questions

- Does the S16 floor in spike/noinv come from the reference notes or from real ambiguity in the prompt (for example, which spike threshold applies)? Reading the 13 confirmed-DIFFERENT floor pairs would answer this for free.
- FINDINGS l.74 says the pruned re-ask was not run, but results.json has it. Which is authoritative? Possibly the salvaged pod run (salvaged_results.json is byte-identical to results.json).
- Which 2605.06939 paper was meant, and who wrote it?

---

## Revision (2026-10-17)

Applied the critic review to the body above (the review section is kept for the record):
- Bottom line 2 and Open questions: the 33% vs 10% rubric figure is now labelled as a mixed 30-pair sample (about 10 floor), not a floor measurement; S16b runs both rubrics on all 100 floor pairs.
- Bottom line 3: added that the confirmed metric, both-orders rule and judge-free path are SAME-favouring; freeze rubric and aggregation first and report false-SAME on the ablation pairs.
- Bottom line 4: replaced the "8-10% achievable" estimate with the measured S16 pruned re-ask (no certificate below 20%) and relabelled 8-10% as a hypothesis; flagged the FINDINGS l.74 conflict.
- New bottom line 7 and table rows: per-scenario floor; S16 re-ask row.
- Bottom line 5: calibration allocated by true label (Lee et al.), stratified by scenario, 140 labelled of 300 prompts, plus a fixed anchor set for drift.
- S13 citation corrected (pinned haiku-5.5 3.5%; sonnet 1-3% was a routing artifact); typed-arm priors 1.7% / 3.5%, record-level measurement.
- Citation fixes: 41.2 not 41.3; per-judge position-bias figures unconfirmed; Atil dated 2024 preprint; 2605.06939 and "Fiedler" marked unverified; Matter to Mechanism transfer caveat; added drift literature (2606.15474, 2607.08535, 2609.21277).
- New bottom line 8 (judge provider pin leaked 3 times; per the user's request, if only AWS is available use an AWS machine as the proxy to fix the served provider; untested) and the 11 unparsed judge outputs.
