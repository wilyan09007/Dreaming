# Certification and deferral: certified error bounds, learning to defer, audit sampling

Research date: 2026-10-17. Tools: web search/fetch only plus local scipy for numbers. "(sim)" or "(calc)" = computed by me locally; everything else is cited. Anything not confirmed is in Unverified.

## Summary

- The bound to implement is the exact one-sided binomial (Clopper-Pearson) upper confidence bound on the error rate among answered rows. It is Lemma 3.1 of Geifman & El-Yaniv (2017) and is also what Learn-then-Test uses for selective classification: `U = scipy.stats.beta.ppf(1-delta, k+1, m-k)` (1.0 if k==m), with m = rows above cutoff c and k = errors among them. Certify c if U <= budget.
- Worked example, n=500 calibration rows, budget 5%, delta 5%: answering all 500 is certifiable only if k <= 16 errors (3.2% empirical). With 380 rows above c you may have at most 11 errors (2.9%). Rule-of-thumb table below.
- Multiple cutoffs are tried, so you pay for multiplicity. Use LTT fixed-sequence testing (descending cutoffs, stop at first failure, no delta split) or Bonferroni over K pre-registered cutoffs. SGR's delta/ceil(log2 m) split is the same idea.
- Hoeffding is useless at this size: with n=500 and a 5% budget even zero errors gives a 5.47% bound (calc). Clopper-Pearson needs 59 accepted rows with zero errors; Hoeffding needs 600.
- Plain fixed-sequence from the top cutoff never certifies, because the highest cutoffs accept very few rows and fail on step 1 (sim). Start the sequence at a cutoff with enough accepted rows (see Blockers).
- 500 rows is thin. In my simulation (perfectly calibrated scores, 5%/5%), n=500 certifies roughly one third of the oracle coverage and finds any cutoff in only 25-70% of draws; n=2000 gives about 70% of oracle coverage (table in Numbers).
- Naive "pick the cutoff where empirical error <= 5%" violates the budget 42-49% of the time in my sims, matching a June 2026 audit paper (49-73% at n=200). A certificate is mandatory, not optional.
- Certificates assume exchangeability between calibration and deployment. A June 2026 paper reports certified rules overrunning in 9-30% of trials under grouped shift. Audit sampling must be designed as a monitor for this. Anytime-valid sequential tests (SPRT, WSR betting sequences, conformal test martingales) are the precedent.
- Learning-to-defer (Mozannar & Sontag) is not needed for our case: with a single cheap fallback and a known cost, the post-hoc rule "defer when P(error) > cost" reduces to thresholding a good error score. Per-call-site ranking quality (AUROC of error vs correct) matters more than calibration. Also, monotone recalibration does not change which rows pass a rank-based cutoff.
- Closest public datasets with human corrections: AlleNoise (502k texts, 74k expert-corrected, GitHub, no login), ChaosNLI (100 annotations per item, direct Dropbox zip), cleanlab/labelerrors.com test-set corrections. There is no public clean-label Banking77 (only a paper flagging ~14% suspect training labels). Kaggle is not required for any of these.

## Findings

### 1. SGR (Geifman & El-Yaniv, NeurIPS 2017), exact content
Source: https://arxiv.org/pdf/1705.08500 (text extracted locally with pdftotext).
- Lemma 3.1 (Gascuel & Caraux 1992): for fixed classifier with true error R, sample of m i.i.d. points with empirical error r_hat, B(r_hat, delta, m) is the solution b of `sum_{j=0}^{m*r_hat} C(m,j) b^j (1-b)^(m-j) = delta`. Then `Pr{R > B} < delta`. The paper says this numerical bound is the tightest possible and that Hoeffding-type bounds approximate it with slack.
- That equation is the Clopper-Pearson one-sided upper limit. I verified numerically that `beta.ppf(1-d, k+1, m-k)` satisfies `binom.cdf(k, m, b) = d`, and that `binomtest(k,m).proportion_ci(confidence_level=1-2*d, method='exact').high` gives the same value (calc).
- SGR (Algorithm 1): sort calibration set by confidence kappa; binary search over index z, theta = kappa(x_z); g_i = accept if kappa >= theta; r_hat_i = empirical selective risk on accepted rows; b_i = B(r_hat_i, delta/ceil(log2 m), |g_i(S_m)|); if b_i < target r then z_max = z else z_min = z; after k = ceil(log2 m) rounds output (f, g_k) and bound b_k.
- Theorem 3.2: `Pr_{S_m}{ exists i: R(f|P_{g_i}) > B(r_hat_i, delta/k, g_i(S_m)) } < delta`, with k = ceil(log2 m). R(f|P_g) is the error among accepted points (selective risk), conditional on acceptance. The proof conditions on the random accepted count m_i.
- Headline: 2% top-5 ImageNet error guaranteed with probability 99.9% at about 60% coverage (https://alphaxiv.org/abs/1705.08500).
- Code: I found no official SGR repository (paper and ar5iv contain no code link). It is about 15 lines of scipy. My own observation (not from a source): the proof sketch treats each g_i as fixed while theta is chosen from the data. In my sims SGR never violated (0.1-1.0% vs 5% allowed), but LTT fixed-sequence is the cleanly proven version.

### 2. Learn-then-Test (LTT), the rigorous multi-cutoff version
Source: https://arxiv.org/abs/2110.01052 (read via https://ar5iv.labs.arxiv.org/html/2110.01052).
- Treat each candidate cutoff lambda_j as a hypothesis "risk(lambda_j) > alpha" with a valid p-value. Theorem 1: with probability >= 1-delta every lambda in the returned set controls risk.
- Bounded-loss p-value (Hoeffding-Bentkus): `p = min( exp(-n*h1(min(R_hat,alpha), alpha)), e * P(Bin(n,alpha) <= ceil(n*R_hat)) )`. For selective classification (their Section 3.2, "Selective Classification with Accuracy Control") they use the exact binomial tail without the factor e, on accepted rows only. Their experiment: ImageNet ResNet, 5k calibration, alpha=0.15, delta=0.1, grid step 0.001, cutoffs accepting fewer than 25 points clipped.
- Fixed-sequence testing (Prop. 3): fix an order (high to low cutoff), test each at level delta, stop at the first non-rejection. FWER <= delta with no Bonferroni penalty. Multi-start variant tests each start at delta/|starts|. Bonferroni: reject if p <= delta/|grid|.
- MAPIE 1.x ships LTT as `mapie.risk_control.BinaryClassificationController` (see Reusable). Its docs do not say which bound it uses and do not mention selective classification (https://mapie.readthedocs.io/en/v1.1.0/generated/mapie.risk_control.BinaryClassificationController.html), so verify before relying on it.
- Conformal Risk Control (Angelopoulos et al., ICLR 2024, https://arxiv.org/abs/2208.02814, code https://github.com/aangelopoulos/conformal-risk, MIT) bounds E[loss] for a monotone loss (expected, marginal over test point). Note this is a different quantity from "error among answered rows": if loss = error*1[answered], it controls errors per incoming call, not per answered call. Our certificate wording ("answered calls are wrong at most 5% of the time") needs the conditional binomial bound above.

### 3. Small-n and shift evidence (2026 preprints, not peer reviewed)
- Audit of selective prediction, arXiv 2606.15153, 13 Jun 2026 (https://arxiv.org/abs/2606.15153): naive empirical thresholding exceeds its declared budget in 49-73% of synthetic trials (n=200) and up to 68% of real splits; Clopper-Pearson and WSR (betting) certify useful coverage where Hoeffding certifies none; under grouped deployment (unseen machine types/generators) certified rules overrun in 9-30% of trials; a conservative per-group threshold restores validity at a large coverage cost.
- Bound ablation, arXiv 2603.08907, 9 Mar 2026 (https://arxiv.org/abs/2603.08907), single author: compares Hoeffding, empirical Bernstein, Clopper-Pearson, DRO, CVaR, union bound, LTT fixed-sequence and WSR betting on Banking77 (n=13K), CLINC-150 (22.5K), MASSIVE (1,102) and a 280-row set. On the 280-row set Hoeffding-family bounds are infeasible below alpha=0.20, and a transfer-warm-started betting bound gives 5.4x the LTT+Hoeffding coverage at alpha=0.10. Applies it to agentic response caching ("progressive trust"), which is close to our "answer from predictor" idea. No code link found.
- MAPIE docs on exchangeability testing (permutation and martingale tests, online monitoring): https://mapie.readthedocs.io/en/latest/content/exchangeability-testing/. crepes has `crepes.martingales` (conformal test martingales, `SimpleJumper`; P(martingale exceeds c) <= 1/c under exchangeability): https://github.com/henrikbostrom/crepes.

### 4. Learning to defer and cascades
- Mozannar & Sontag, ICML 2020, https://arxiv.org/abs/2006.01862 (PMLR https://proceedings.mlr.press/v119/mozannar20b.html): learn classifier plus rejector from samples of the expert's past decisions; reduction to cost-sensitive learning; consistent surrogate generalizing cross-entropy. No public code found (Unverified).
- Verma & Nalisnick, ICML 2022, https://arxiv.org/abs/2202.03673: the Mozannar-Sontag softmax surrogate is not calibrated for expert correctness; one-vs-all surrogate fixes it. Cao et al. 2023, https://arxiv.org/abs/2311.01106, argue the cause is the symmetric loss, not softmax, and propose an asymmetric softmax loss.
- Narasimhan et al., NeurIPS 2022, https://proceedings.neurips.cc/paper_files/paper/2022/hash/bc8f76d9caadd48f77025b1c889d2e2d-Abstract.html: post-hoc deferral on top of a trained base model; defer when base-model error probability exceeds expert cost. This is the form that fits us.
- Trust or Escalate (Jung, Brahman, Choi, ICLR 2025, https://arxiv.org/abs/2407.18370, code github.com/jaehunjung1/cascaded-selective-evaluation): fixed-sequence testing with exact binomial UCB on judge-human disagreement, descending lambda from 0.999, chosen lambda = smallest such that all lambda' >= lambda pass; cascade of judges each tested at delta/|M|; calibration set 500 examples (392 for one dataset), delta=0.1. Closest published recipe to our "tier router with certificate". Note they ran the sequence from 0.999 successfully because their judge confidence is high on many rows, so check the start-point issue below.
- Conformal Cascade (arXiv 2607.25018v3, 31 Jul 2026, https://www.alphaxiv.org/abs/2607.25018.md): accept when conformal set is a singleton; guarantee is marginal coverage >= 1-K*alpha (per-tier union bound), not selective error. Fine for classification, weaker than what we want.
- Gatekeeper (NeurIPS 2025, https://neurips.cc/virtual/2025/poster/116238): fine-tune the small model so it is confident only when right. Optional later.

### 5. Confidence scores and calibration libraries
- Trust score (Jiang et al., NeurIPS 2018, https://arxiv.org/abs/1805.11783): ratio of distance to nearest other-class vs predicted-class neighborhood. Code https://github.com/google/TrustScore (Apache-2.0, archived read-only 27 May 2024, 6 commits). Use as a feature for an error meta-model, not as a guarantee.
- scikit-learn calibration docs (https://scikit-learn.org/stable/modules/calibration.html): isotonic beats sigmoid only with more than about 1000 samples; sigmoid for small samples; `FrozenEstimator` for an already-fit model; `method="temperature"` is documented (one parameter, accuracy unchanged). Calibration data must be disjoint from classifier training data.
- netcal: Apache-2.0 (1.0 was MPL-2.0), PyPI 1.4.0 per aggregator; depends on torch (https://pypi.org/project/netcal/). Heavy for what we need.
- Design implication (my reasoning): the certificate needs a held-out set the cutoff is chosen on. If calibration (a monotone map) and cutoff selection both use it, the certificate is no longer valid, so split or use cross-fitting. A monotone calibration map does not change the accepted set for a given coverage, so skip it per call site and certify the raw score. It only matters for sharing one threshold across call sites.

### 6. Human-correction / disagreement / noisy-label text datasets (no login unless noted)
| Dataset | Size | What it gives | Access | License |
|---|---|---|---|---|
| AlleNoise, https://github.com/allegro/AlleNoise (AISTATS 2025, https://arxiv.org/abs/2407.10992) | 502,310 product titles, 5,692 classes; 74,094 noisy (about 14.7%), 428,216 clean | seller category (noisy) vs expert-chosen category; experts fixed items from customer complaints and catalog clean-ups; titles machine-translated to English | `allenoise/full_dataset.csv` in repo, no login | repo MIT; data license not stated |
| ChaosNLI, https://github.com/easonnie/ChaosNLI (EMNLP 2020, https://arxiv.org/abs/2010.03532) | 4,645 items x 100 annotations (SNLI 1,514, MNLI-m 1,599, aNLI 1,532) | full human label distribution; old gold label vs new majority; disagreement in 10/20/31% of aNLI/SNLI/MNLI | direct zip https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip, JSONL | CC BY-NC 4.0 |
| cleanlab label errors, https://labelerrors.com/ (https://arxiv.org/abs/2103.14749) | test sets of 20news, IMDB, Amazon Reviews among 10 datasets | confident-learning candidates validated by crowd; about 3.3% average error, about half of candidates confirmed | download path not verified | cleanlab code Apache-2.0 (https://github.com/cleanlab/cleanlab) |
| NoisyAG-News (https://arxiv.org/abs/2407.06579) | 50,000 AG-News items, 3 crowd labels each | noise 10/20/38% (Best/Med/Worst variants) | paper gives no download link | paper CC BY-NC-SA 4.0; dataset unknown |
| Banking77 label-error study (https://aclanthology.org/2022.insights-1.19) | 10,003 train utterances | about 14% flagged as suspect; no corrected release found | n/a | n/a |
- Closest analog to a bookkeeper correcting a categorization: AlleNoise (hierarchical category, human fix). Closest analog to "which outputs a reviewer would change": ChaosNLI (entropy of human labels predicts model failure; models near chance on low-agreement items per the paper).
- No public bookkeeping or invoice dataset with reviewer edits was found in this search. We will need to synthesize reviewer edits (e.g., flip labels with an annotator-confusion model) for the demo.

### 7. Audit sampling to keep the certificate current
- Fixed audit sample, classical precedent: attribute / acceptance sampling. Zero-deviation plan at 95% confidence and 5% tolerable rate needs 59 samples (0.95^59 = 0.0485, calc). Minitab and a reliability forum quote the same 59; a HUD OIG audit guide table lists 64-65 for the same inputs because of different software (https://www.hud.gov/sites/documents/20004appoigh.pdf).
- SPRT (Wald): test H0 error p0 vs H1 p1 sample by sample, stop at boundaries. Needs both p0 and p1 fixed in advance. gsDesign binomial SPRT vignette: https://cran.r-project.org/web/packages/gsDesign/vignettes/binomialSPRTExample.html. My Wald approximations (alpha=beta=0.05) of average audit sample number: p0=3% vs p1=8%: 121 audits if fine, 89 if drifted to 8%; p0=3% vs p1=10%: 73 / 50; p0=3% vs p1=5%: 543 / 461 (calc).
- Anytime-valid alternative: betting confidence sequences (Waudby-Smith & Ramdas, JRSSB 2024, https://arxiv.org/abs/2010.09686); package `confseq` (https://pypi.org/project/confseq/, MIT classifier, C++14 and Boost build needed, "early-stage").
- Fixed-n comparison (calc): detect a rise from 5% to 10% with one-sided binomial test alpha=0.05, power 0.9 needs n=233 (reject at >=18 errors); 5% to 8% needs 572 (>=38); 5% to 15% needs 77 (>=8).
- Re-certifying in windows with time-uniform delta spending (delta_t = delta*6/(pi^2 t^2)): zero-error rows needed at 5% budget grow 69, 96, 131, 158 for windows 1, 2, 5, 10 (calc). Alternative is an anytime-valid sequence that needs no spending.
- Control charts (Shewhart p-chart, CUSUM) are the classical cheap drift alarm but give no finite-sample certificate; I found no source tying them to ML certificates (Unverified). Use them only as a trigger for re-certification.

## Reusable components

| name | url | license | why |
|---|---|---|---|
| scipy.stats.beta / binomtest | https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html | BSD-3 | one-line exact bound; no new dependency needed |
| MAPIE 1.5.0 (5 Aug 2026) | https://github.com/scikit-learn-contrib/MAPIE | BSD-3-Clause | LTT `BinaryClassificationController` (confidence_level default 0.9), conformal classification, exchangeability tests; bound type undocumented |
| conformal-risk | https://github.com/aangelopoulos/conformal-risk | MIT | reference for CRC `get_lhat`; different guarantee (expected loss) |
| crepes 0.9.1 | https://github.com/henrikbostrom/crepes | BSD-3-Clause | Mondrian (class-conditional) conformal classifiers, `crepes.martingales` drift monitor |
| TorchCP | https://github.com/ml-stat-Sustech/TorchCP | LGPL-3.0 | LAC/APS/SAPS/RAPS scores; torch-model oriented, LGPL, so skip |
| cascaded-selective-evaluation | https://github.com/jaehunjung1/cascaded-selective-evaluation | not checked | reference code for fixed-sequence exact binomial cascade |
| google/TrustScore | https://github.com/google/TrustScore | Apache-2.0 | kNN trust score feature; archived |
| scikit-learn CalibratedClassifierCV | https://scikit-learn.org/stable/modules/calibration.html | BSD-3 | isotonic/sigmoid/temperature, FrozenEstimator |
| netcal | https://pypi.org/project/netcal/ | Apache-2.0 | ECE metrics and calibrators; torch dependency |
| confseq | https://pypi.org/project/confseq/ | MIT (classifier only) | betting confidence sequences for audit monitoring |
| cleanlab | https://github.com/cleanlab/cleanlab | Apache-2.0 | find label issues; sanity-check our own reference labels |
| Reference implementation of the bound | in this file below | n/a | copy and use |

```python
from scipy.stats import beta
def ucb(k, m, delta):            # exact one-sided (1-delta) upper bound on error rate
    return 1.0 if m == 0 or k >= m else beta.ppf(1 - delta, k + 1, m - k)
def certify(scores, err, alpha=0.05, delta=0.05, m_min=59):
    # scores: confidence per calibration row; err: 1 if predictor != reference
    # fixed-sequence, descending cutoffs; start where >= m_min rows accepted
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    best, k = None, 0
    for r, i in enumerate(order, 1):        # accept top r rows
        k += err[i]
        if r < m_min: continue
        if r < len(order) and scores[order[r]] == scores[i]: continue  # no ties split
        if ucb(k, r, delta) <= alpha: best = scores[i]
        else: break
    return best                              # cutoff c, or None
```
(Walking the rows in score order is a fixed-sequence over the data's own score values; rigorous fixed-sequence needs a pre-specified grid, see Blockers. Tested on 40 synthetic seeds, n=2000, Beta(10,1) scores: m_min=59 returned 0 accepted rows in at least one seed because a single error in the first 59-63 rows stops the sequence; m_min=150 had min 353, median 1225 accepted. Treat m_min as a tuning knob, larger is more robust.)

## Numbers (calc unless noted; 2026-10-17)

Max errors allowed among the m accepted rows so that U <= 5%:
| m accepted | delta=0.05 | delta=0.10 | K=10 cutoffs Bonferroni (delta/10 = 0.005) | SGR split (delta/9) |
|---|---|---|---|---|
| 59 | 0 | 0 | none | none |
| 100 | 1 | 1 | none | none |
| 200 | 4 | 5 | 2 | 2 |
| 300 | 8 | 9 | 5 | 5 |
| 380 | 11 | 13 | n/a | n/a |
| 400 | 12 | 14 | 9 | 9 |
| 500 | 16 | 18 | 12 | 13 |
Other deltas at m=500: K=5 (0.01): 13; K=20 (0.0025): 11; 1000-point grid Bonferroni: 7.
- Zero errors, min accepted rows for 5% budget: 59 (delta 0.05), 45 (0.10), 90 (0.01). Hoeffding: 600.
- Bound at n=500 all accepted: k=0 gives 0.60% (CP) vs 5.47% (Hoeffding); k=5: 2.09% vs 6.47%; k=10: 3.37% vs 7.47%; k=15: 4.58% vs 8.47%.
- Per-class certificates for 12 labels, zero errors: 107 accepted rows per class at delta/12 (calc).

Worked example, n=500, budget alpha=5%, delta=5%, rows sorted by confidence:
1. Pre-register descending cutoffs. For each c compute m(c) = rows with score >= c and k(c) = their errors.
2. U(c) = beta.ppf(0.95, k+1, m-k). Example: c with m=380, k=11: U = 4.746% <= 5%, certified. k=12: U = 5.066%, fail.
3. All 500 answered: k=16 gives U=4.820% (pass), k=17 gives U=5.056% (fail).
4. Certificate statement: with probability >= 95% over the calibration draw, the true error rate among rows with score >= c is <= 5%.
5. Per-call-site delta: if you certify S call sites simultaneously and want the whole system at 95%, use delta/S per site.

Simulation (sim; scores s~Beta(A,1), error~Bernoulli(1-s), so perfectly calibrated; 5%/5%; grid step 0.001; script at C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Desktop-personalprojects-Dreaming\4bf0a8c4-ab7a-4fd7-a483-c5f84a5e6e6e\scratchpad\sim3.py and sim4.py):
| scenario | method | P(true risk>5%) | P(cutoff found) | mean answered coverage overall |
|---|---|---|---|---|
| A=10 (acc 91%), n=500, oracle cov 73% | naive | 0.464 | 0.999 | 0.69 |
| | SGR (delta/9) | 0.002 | 0.251 | 0.12 |
| | fixed-seq, start at m>=59 | 0.019 | 0.687 | 0.22 |
| | plain fixed-seq from 0.999 | 0 | 0 | 0 |
| A=10, n=1000 | SGR / fixed-seq skip59 | 0.005 / 0.030 | 0.81 / 0.80 | 0.36 / 0.36 |
| A=10, n=2000 | SGR / fixed-seq skip59 | 0.010 / 0.045 | 0.997 / 0.885 | 0.53 / 0.51 |
| A=4 (acc 80%), n=500, oracle cov 36% | fixed-seq skip59 | 0.011 | 0.40 | 0.07 |
| A=4, n=2000 | SGR / skip59 | 0.002 / 0.025 | 0.71 / 0.79 | 0.14 / 0.15 |
Trials: 1500 (n=500), 600 (n=1000, 2000). Binomial noise on violation rates is about +/-1 point. Skip59 at n=2000 sits near the 5% allowance, so it is not safe to treat as rigorous.
- Compute cost: building a 500x500 beta.ppf table takes well under a second; the whole sim of 1500 trials ran in about 1-2 minutes on local CPU. No GPU, no money.
- Library facts: MAPIE 1.5.0 released 5 Aug 2026, BSD-3, Python >=3.10 per PyPI (https://pypi.org/project/MAPIE/). crepes 0.9.1, BSD-3. TorchCP 479 GitHub stars, LGPL-3.0. cleanlab 11.7k stars, Apache-2.0.

## Blockers / plan changes

- Calibration size: n=500 is only enough to certify a small slice. Plan for 1000-2000 labeled rows per call site for a 5% budget at delta 0.05, or relax to delta 0.10 or a 10% budget. Smoke budget: the demo's shadow period should log at least 1500 reference-labeled rows per step before attempting a certificate.
- Start-point problem: fixed-sequence needs a start where enough rows are accepted. Rigorous options (all with cost): (a) pre-register K cutoffs and use delta/K (K=10 costs about 4 allowed errors at m=500); (b) multi-start LTT; (c) choose the start on a separate small split; (d) skip cutoffs with m < 59, which my sim shows is near-nominal (1-4.5% violation) but which the theorem does not cover.
- The label is "agrees with the real call," not truth. If the real call is stochastic (LLM at temperature > 0), the reference has its own variance; define the reference as a temperature-0 or majority output and certify against that. Say so in the certificate.
- Free-text steps (draft_note): the binomial bound needs a 0/1 loss. Use a thresholded similarity or judge agreement as the 0/1 event, or use HB p-values with a [0,1] loss (LTT paper). Unverified how loose that gets.
- Per-class guarantees (12 categorize labels) multiply data needs; certify the pooled step first, then class-wise only for large classes.
- Shift: certified rules overrun 9-30% under grouped shift (2606.15153). Audit sampling must be stratified over time and over groups (vendor, account). Plan: audit rate of a few percent of answered calls, SPRT or WSR sequence over audit errors, re-certify (re-run the bound on fresh audit rows) when the alarm fires or the window changes.
- Do not use CRC or conformal set-size cascades if the promise is "error among answered calls"; they guarantee expected per-call loss or marginal coverage.
- Learning-to-defer papers are not worth implementing for the smoke. Use the post-hoc rule: error score = student confidence (plus optional meta-model), then certify the cutoff.
- No public bookkeeping dataset with reviewer edits was found; mission 4 (reviewer-change prediction) needs a synthetic edit process for the smoke, with ChaosNLI and AlleNoise as sanity benchmarks.

## Unverified

- Whether MAPIE `BinaryClassificationController` uses an exact binomial bound or Hoeffding-Bentkus, and whether it can express "error among answered rows" (docs fetched for v1.1.0 do not say; v1.4.0 and latest doc URLs returned 404 on fetch).
- Any official SGR or Mozannar-Sontag code; the Trust or Escalate repo license; the TrustScore license beyond the GitHub page (Apache-2.0 shown).
- labelerrors.com download method and per-dataset text error counts (the Amazon Reviews figure of about 390k errors / 4% came from a news summary, not the paper).
- NoisyAG-News public availability and license; AlleNoise data license (repo MIT; dataset license not stated); AlleNoise file size.
- The two 2026 preprints (2606.15153, 2603.08907) are single-source and not peer reviewed; I read only their arXiv abstract pages.
- Banking77 license (I believe CC BY 4.0 on HF `PolyAI/banking77`, from memory, not fetched).
- Wald ASN figures are approximations from my own formulas; no simulation of the SPRT.
- Whether the "skip cutoffs with m<59" start rule is valid in theory (empirically near-nominal in my sims only).
- crepes/MAPIE exchangeability martingale power for detecting our specific drifts.

## Addendum: Noise floor of the "real call" label (OpenRouter, T=0, provider pinning)

Researched 2026-10-17. Status of the original claim ("teacher self-disagreement is unmeasured; could be 2-5%"): **unresolved for our models, qualitatively confirmed, with one plan-changing correction** (temperature 0 may not be settable on the best teachers). No published per-model self-disagreement rate exists for current OpenRouter slugs; we must measure it.

### What the literature gives

- Atil et al., "Non-Determinism of Deterministic LLM Settings" (https://arxiv.org/abs/2408.04667; v1 2024-08-06, v5 2025-04-02; HTML https://arxiv.org/html/2408.04667v5). Setup: GPT-3.5 Turbo, GPT-4o, Llama-3-70B/8B-Instruct, Mixtral-8x7B-Instruct, "all hosted behind APIs we don't control"; T=0, top-p=1, fixed seed; 10 runs; 8 tasks (4 BBH + 4 MMLU), zero/few-shot, no chain-of-thought. Confirmed: accuracy varies "up to 15%" across natural runs, best-vs-worst gap "up to 70%"; worst cited case Mixtral on MMLU college math 75% best vs 3% worst. No model/task produced identical raw strings in all 10 runs (no TARr@10 of 100% anywhere). Llama-3-8B on the authors' own GPU was deterministic (control).
- Their agreement metric is not a pairwise flip rate. TARa@10 = share of questions whose parsed answer is identical in all 10 runs, so 1-TARa@10 = "any disagreement in 10 runs" and over-states the 2-run flip rate. Few-shot TARa@10 (%), GPT-4o / GPT-3.5 / Llama-70B: navigation 99.6/96.4/98.4; logical deduction 99.6/84.4/97.2; public relations 92.7/87.3/81.8; ruin names 95.2/66.0/94.4; professional accounting 66.7/91.1/69.5. So for easy short-answer items the any-of-10 instability is roughly 0.4-8%, for hard/ambiguous items 15-35%+. TARr@10 (raw string) is far lower (often under 50%): irrelevant for categorical labels but fatal for exact-match on draft_note.
- These are 2024 models on exam-style tasks. They give no rate for bookkeeping categorization. My inference (not measured): instability concentrates on items where the top-2 labels are close, which the 12-label categorize_transaction step will have, so the 2-5% guess is plausible but unsupported by data.
- Thinking Machines (https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/): Qwen3-235B-A22B-Instruct-2507, 1000 completions at T=0 of 1000 tokens, 80 unique outputs, first divergence at token 103 (992 vs 8 split at that branch); all 1000 identical with batch-invariant kernels. Cause is batch-size-dependent numerics (server load), not sampling. Implication (inference): divergence is a tail event over long generations; a 1-5 token label diverges far less often than raw-string stats suggest, but a near-tie logit can still flip it.
- Backend alone: "The Silent Hyperparameter" (https://arxiv.org/pdf/2605.19537, 2026-05-20, preprint): with weights, decoding, hardware fixed, switching among 5 engines (vLLM, SGLang, llama.cpp...) shifts benchmark scores up to 16.6 pp with high output disagreement. Abstract only, via search snippet. Related, unread: https://arxiv.org/pdf/2608.04714

### OpenRouter-specific facts (primary docs)

- Provider routing doc (https://openrouter.ai/docs/features/provider-routing, fetched 2026-10-17): default is load balancing "prioritizing price"; among providers without outages in the last 30 s, selection is "weighted by inverse square of the price" (a $1 provider is 9x likelier than $3). So one slug is routinely served by different providers across consecutive calls. `order` = "List of provider slugs to try in order"; `allow_fallbacks` defaults true, false limits routing to the chosen providers; setting `sort` or `order` disables load balancing; `require_parameters`, `only`, `ignore`, `quantizations` (e.g. ["int4","int8"]) exist. The doc says quantized models "may exhibit degraded performance for certain prompts" but gives no cross-provider quality data. Pinning therefore = `order:[<slug>]` + `allow_fallbacks:false` + `quantizations`; use the full endpoint slug (e.g. "deepinfra/turbo", not "deepinfra") to hit one variant (search snippet of OpenRouter docs, not re-fetched).
- `seed` (https://openrouter.ai/docs/api-reference/parameters): "repeated requests with the same seed and parameters should return the same result" but "Determinism is not guaranteed for some models." The routing doc does not mention seed. The pages I fetched did not say whether responses carry a `provider` field: check a real response and log it on every call (unverified).
- Third-party field report (2026-07-23, update 07-28, https://www.greaterwrong.com/posts/KsyoSAyBRXtwzSugg/not-pinning-your-openrouter-provider-might-invalidate-your): unpinned OpenRouter runs can mix providers within one experiment (QwQ-32B 90/10 vs 85/15 Nebius/DeepInfra across runs); Llama-3.3-70B endpoints differ fp8 vs bf16 with price not tracking quality; recommends pin endpoint + quantization, record serving provider, `allow_fallbacks:false`, `require_parameters:true`, short time window, and "treat `seed` as non-deterministic". Single-author and partly anecdotal; its "97% of repos unsafe" audit was done by Claude and is disputed in comments.
- OpenRouter's own view (https://openrouter.ai/blog/announcements/auto-exacto/ and https://openrouter.ai/blog/announcements/provider-variance-introducing-exacto/, search snippets only, not fetched): provider quality differences are real, mostly tool-call parser/engine issues rather than quantization; Exacto routes to better-measured providers. Declared quantization is provider-reported and not verified by OpenRouter.

### Plan-changing correction: T=0 may not be available on the best teachers

- Anthropic: from Claude Opus 4.7 on, non-default `temperature`/`top_p`/`top_k` return HTTP 400. Verified for Sonnet 5.5 in https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide ("Claude Sonnet 4.6 and earlier models and Claude Haiku 4.5 accept temperature, top_p, and top_k. On Claude Sonnet 5.5, a non-default value returns a 400 error. Remove them."). Opus 4.7/4.8 per search snippet of https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-7 (not fetched). Whether claude-sonnet-5 and claude-haiku-5.5 accept temperature: unverified. Release notes (search snippet, 2026-08-20) say Python SDK v1.0 removes the parameters from Messages methods.
- Mandatory-reasoning slugs (gpt-5/-mini/-nano, gemini-2.5-pro, claude-sonnet-5.5, per routing-benchmarks.md) generally do not take T=0 either; the OpenAI/Gemini restriction is from memory, unverified here.
- Consequence: for those teachers the "real call" is a sample from a T~1 distribution, so self-disagreement is real sampling noise, not a small numerical term, and could be well above 5% on 12-way labels. Models where T=0 works (gpt-4.1 family, claude-haiku-4.5, deepseek-v3.2, qwen3 non-thinking, gemini-2.5-flash) are the clean noise-floor arm.

### What this means for certification and the other missions

- Irreducible floor: if the teacher disagrees with an independent redraw of itself at pairwise rate s, even a perfect student agrees with a single stored real-call label only ~1-s of the time. A 5% budget with s=3% leaves ~2% for genuine student error. Report s in every certificate; define the reference as the stored call (what was logged), not "the" teacher output.
- Cheaper noise control: label calibration/audit rows with majority-of-3 (3x cost on those rows only) and certify against the majority; keep single-sample labels for the live audit stream and flag s. context-pruning.md already plans a 3x base-vs-base floor; router agreement and reviewer-change targets need the same.
- Real-call fallbacks cannot be replayed bit-for-bit through OpenRouter (unpinned, seed not guaranteed, provider and model version can change). Store response, serving provider, model id / fingerprint if returned, and request params; replay = read from log only. This differs from hosted predictors, which determinism.md says can be byte-replayable.

### Smoke to run (measure it, about $1-5)

- 200 prompts per step (categorize_transaction + flag_anomaly), 5 repeats. Arms: (A) defaults; (B) `order:[X]`, `allow_fallbacks:false`, `require_parameters:true`, `seed:42`, `temperature:0` where allowed; optional (C) B plus `quantizations`. Models: gpt-4.1-mini, gpt-4.1, claude-haiku-4.5, deepseek-v3.2, plus one T-locked model (claude-sonnet-5.5) as the sampled-noise reference. Log serving provider per call.
- Metrics: pairwise label flip rate (mean fraction of disagreeing repeat pairs per prompt), TARa@5, distinct providers per slug in arm A, prompts that flip in A but not B.
- Cost at list prices from routing-benchmarks.md (800 in / about 20 out tokens; my arithmetic): gpt-4.1-mini about $0.00035/call, gpt-4.1 about $0.0018/call; 200 x 5 x 2 arms = 2000 calls is about $0.7 and $3.5 respectively.

### Still unverified

- Any published per-model T=0 repeat-disagreement rate for Claude, GPT-4.1/5, Gemini, DeepSeek, or Qwen served via OpenRouter (searched, none found).
- Whether OpenRouter responses expose the serving provider, and whether each provider forwards `seed`.
- OpenAI/Gemini temperature restrictions on reasoning models; claude-sonnet-5 / haiku-5.5 temperature support.
- Anthropic's "not fully deterministic at temperature 0" wording (seen only on secondary sites).

## Addendum: vCache (verified semantic caching), prior art for the certify loop and a free testbed

Research date: 2026-10-17. "(sample)" = my 700-row sample pulled from the HF datasets-server rows API (7 pages x 100 at offsets 0, 6000, ..., 37700; script and JSON are in my scratchpad, not in the repo). "(calc)" = my arithmetic. PDF text was extracted with pdftotext, so Greek symbols are dropped and the Table 3 layout is garbled; I name the quantities in words.

### Verdict on the original claims

- vCache is published, code-released prior art for "answer from the cheap path under a user-set error bound": CONFIRMED. Paper: https://arxiv.org/abs/2502.03771 (v5, 21 Feb 2026; PDF header says "Published as a conference paper at ICLR 2026"; 10 authors incl. Zaharia and Gonzalez). Abstract: "up to 12.5x higher cache hit and 26x lower error rates" vs static-threshold and fine-tuned-embedding baselines. Code: https://github.com/vcache-project/vCache (the older vcache-lgs URL returns 404).
- "Its datasets are a ready testbed with per-prompt LLM responses and correctness labels": CORRECTED (partly true, partly not). Only SemBenchmarkClassification has two model tiers (Llama-3 8B and 70B) per prompt; none of the four has ground-truth labels or a stored "correct" flag; SemBenchmarkSearchQueries has no real LLM responses at all. Details in the dataset table below.
- "Guarantee is marginal, sigmoid-model-based, learned online": CONFIRMED, and the marginal part is stronger than assumed. It is marginal over ALL requests, including the ones that fall back to the real LLM (next section).
- "12.5x / 26x": CONFIRMED as quoted, but these are best-case ratios on SemCacheLMArena against GPTCache static thresholds (paper Sec. 5: "On SemCacheLMArena, it achieves up to 26x lower error, 12.5x higher hit rates"). On Classification vCache wins only for error bounds above 1.5%; below that it is "more conservative" than static baselines (same section).

### What vCache actually guarantees (paper text, ICLR version)

- Setup: embed the prompt, find the nearest cached prompt nn(x), similarity s(x). Correctness label c(x) = 1 if the cached response equals the fresh LLM response r(x) = LLM(x) (Eq. 4). So the label is "agrees with the real call", the same reference definition Dreaming uses, not truth.
- Policy: per cached embedding, fit a sigmoid P(c=1 | s) = 1 / (1 + exp(-gamma (s - t))) by MLE on that embedding's own observed (s, c) pairs (Eq. 9-10), take a pessimistic t from a confidence band, then explore (call the LLM) with probability epsilon = [(1 - delta) - Pc] / [1 - Pc] computed from the pessimistic Pc (Eq. 8, 11), else return the cached answer. Labels come for free from explores. The paper's fit examples use the embeddings with the most observations, 42 and 33 (Appendix D), so typical per-embedding fits rest on tens of labels.
- Guarantee (Def. 4.1, Eq. 6-7, Thm. 4.1): "marginal probability of vCache returning the correct answer, given any arbitrary prompt x, is lower bounded by (1 - delta)", where an explore counts as correct "by design". Assumptions (Sec. 4.3 and Sec. 6): prompts i.i.d. and the sigmoid family correctly models P(c=1 | s) for each embedding; "If these assumptions are violated, the analysis may not hold."
- Proof detail that matters (Appendix C): the confidence band on t is derived "by assuming a uniform prior on t and" the steepness (Eq. 23), i.e. a Bayesian posterior argument, and "in our experiments, we only use confidence intervals for t", with a point estimate for the steepness. My reading (reasoning, not stated by the authors): the implemented policy is not exactly the one the theorem covers, and the guarantee is not a frequentist finite-sample bound over the calibration draw.
- Evaluation metrics (Sec. 5): error rate = FP / n where n is ALL prompts; hit rate = (TP + FP) / n. So the plotted "observed error under delta" is error over all requests, not error among answered ones. 95% CIs are Wallis binomial bounds on that stream. In the extracted text I found no repeated-trial violation probability (Unverified: I did not inspect every appendix figure).
- Observed conservatism: Table 3 (SemCacheLMArena, E5-large-v2, GPT-4o-mini), vCache rows as I read the garbled layout: delta 0.02 -> 0.5% error / 41% hits; 0.03 -> 1.1% / 46%; 0.05 -> 2.0% / 54%. Error among hits would be 0.5/41 = 1.2%, 1.1/46 = 2.4%, 2.0/54 = 3.7% (calc), well under delta in practice.

### Marginal vs selective risk (the main argument for our certificate)

- vCache bounds P(wrong) per incoming request. Our certificate bounds P(wrong | answered from predictor). With hit rate h these differ by up to 1/h: selective error = marginal error / h (calc).
- Concrete case straight from Eq. 8 (calc): delta = 0.05 and a prompt whose true P(c=1) is 0.90. Required explore probability = (0.95 - 0.90) / (1 - 0.90) = 0.5. So vCache answers from cache on half of such prompts, those answers are wrong 10% of the time (2x delta), and the marginal error is exactly 5%. In general the selective error can reach delta/h (0.05/0.54 = 9.3% at the Table 3 hit rate, calc).
- Other differences relevant to us:

| | vCache | Dreaming certificate (this file) |
|---|---|---|
| Quantity bounded | P(returned answer = real call) over all requests; fallbacks count as correct | P(predictor wrong given answered), per call site |
| Probability over | exploration coin plus a posterior-style band; no statement over the calibration draw | calibration draw, exact binomial (Clopper-Pearson) |
| Assumptions | i.i.d. plus well-specified sigmoid per embedding | exchangeability only; any monotone score works |
| Data need | tens of labels per cached neighbor, labels bought online by exploring | >= 59 accepted rows at zero errors for 5%/5%; plan 1000-2000 labeled rows per site (see Numbers) |
| Drift | adapts continuously, claims OOD robustness (Appendix H.1, not independently checked) | certificate expires; needs audit sampling and re-certification |
| Replay | decision uses a Uniform(0,1) draw, so non-replayable unless seeded | deterministic cutoff on a hashed score |
| Cost of fallback | permanent exploration rate | audit sample rate |
| License | code CC BY-NC-ND 3.0 (see Plan changes); paper CC BY-NC-ND 4.0 per PDF metadata | n/a |

- Two cheap borrowings (my suggestion): (a) seed the explore coin from a hash of the call id so vCache-style randomization stays byte-replayable; (b) treat vCache-style explores as the audit stream, since an explore is exactly an audit sample.

### The four HF datasets

All four are Apache-2.0 on their cards. I confirmed `gated: false` and Apache-2.0 through the HF API for Classification and LmArena; the SearchQueries and Combo cards loaded anonymously. URLs: https://huggingface.co/datasets/vCache/SemBenchmarkClassification, https://huggingface.co/datasets/vCache/SemBenchmarkLmArena, https://huggingface.co/datasets/vCache/SemBenchmarkSearchQueries, https://huggingface.co/datasets/vCache/SemBenchmarkCombo. Each is a single `train.parquet` plus `dataset.py`; embeddings are stored as strings, hence the large files.

| Dataset | Rows (viewer; paper says) | Size | Real LLM responses | Correctness definition | Fit for us |
|---|---|---|---|---|---|
| Classification | 37,836 (paper/card: 45,000, gap unexplained) | 3.92 GB | `response_llama_3_8b`, `response_llama_3_70b`, `response_llama_3_70b_vllm` plus latencies; 5 embeddings (gte, gte_ft, e5_mistral_7b, e5_large_v2, e5_large_v2_ft) | no truth label and no class id; paper Sec. 6: string match of responses | Best. y = (8B answer == 70B answer) gives a real tier-router / "cheap agrees with expensive" set for mission 1 and a free certify smoke |
| LmArena | 51,147 (paper/card: 60,000) | 6.6 GB | GPT-4.1-nano and GPT-4o-mini responses (mean about 290 words) plus embeddings (OpenAI 3-large/3-small, gte, e5) | `ID_Set`: a hit is correct iff same class; classes = 3,500 LM-Arena prompts expanded with 1-23 GPT-4.1-nano paraphrases | Paraphrase-equivalence labels only. Free text, no ready agreement label between the two LLMs (would need a judge) |
| SearchQueries | 150,000 | 6.32 GB | NO: `response_llama_3_8b` holds one placeholder string ("Not required for the benchmark because of the id_set") | `id_set` from a GPT-4.1-nano judge plus union-find over k-means clusters of ORCAS queries | Embeddings and equivalence classes only. Not a real-LLM eval |
| Combo | 27,500 (matches) | 436 MB | Llama-3-8B responses only, gte embedding | `ID_Set`; no correct/incorrect column | Smallest file; fine for a download check, too thin for tiers |

- Real-data check on Classification (sample, 700 rows): 8B == 70B (case and trailing-period normalized) in 589/700 = 84.1%. By source: amazon_instant_video (yes/no "is the review friendly") 239/290 = 82.4%; ecommerce_dataset (4 categories) 298/334 = 89.2%; commonsense_qa (the task is "main subject of the question", one word from a set) 52/76 = 68.4%. So a 5% budget is not met by the 8B alone and needs a score plus a cutoff, which is the point of the smoke. 70B vs 70B-vLLM outputs were identical in 700/700 rows (sample; short outputs, decoding settings not documented): a small positive sign for replaying hosted outputs, not a determinism result.
- Domain gap: label spaces of 2, 4 and about 5 words, none bookkeeping-like (our categorize step has about 12 labels), and the "commonsense_qa" part is not the original CommonsenseQA task. Treat it as a mechanics test of the certify loop, not as evidence about the bookkeeping agent.
- Access cost: the rows API returned about 14.3 MB per 100 rows (embeddings dominate), so paging all of Classification through it is about 5 GB; download the parquet instead. Whether parquet column projection over HTTP avoids pulling embeddings was not tested (Unverified; pyarrow is not installed in this environment).

### Smoke this unlocks ($0, local CPU)

1. Download Classification `train.parquet` (3.9 GB, no login). Build y = 1 if the 8B answer differs from the 70B answer.
2. Score = logistic regression or kNN on `emb_gte` predicting "8B is wrong"; split train / calibration / test, with at least 1500 calibration rows (see Blockers above).
3. Certify a cutoff with the Clopper-Pearson fixed-sequence code in "Reusable components" at alpha = 5%, delta = 5%; report coverage and selective error on held-out rows. Repeat over seeds to measure the violation rate (the vCache paper reports none).
4. Baseline: run vCache's policy on the same stream. Nearest-neighbor reuse is a different cheap path from "the 8B answers", so this compares guarantee mechanics, not the same predictor. Report BOTH metrics: error over all requests (vCache's) and error among answered (ours). Expect our selective-risk certificate to hold at the stricter number by construction and vCache to be looser there (calc above).

### Plan changes

- Cite vCache (arXiv 2502.03771) in the certify section as closest prior art and as the weaker-guarantee baseline; do not claim novelty for "answer from cheap path under a user error bound". The defensible novelty is the distribution-free, selective-risk, finite-sample certificate plus byte-replayable decisions.
- The vCache repo license is CC BY-NC-ND 3.0: the raw `LICENSE` on the `master` branch begins "Attribution-NonCommercial-NoDerivs 3.0 Unported", and the README badge says the same. No derivatives and non-commercial. Reimplement the policy from the paper for a baseline (per-neighbor logistic fit plus a few lines of exploration logic) rather than forking; running the unmodified package is the only reuse I would assume is safe (not legal advice). The `main` branch URL for LICENSE returns 404, so the branch layout is unclear.
- Do not plan on SearchQueries for any real-LLM eval.

### Unverified

- Why Classification and LmArena have fewer rows than the paper's 45,000 / 60,000.
- Which LLM pair the paper's Classification experiments used: Sec. 5 says Llama-3-8B-Instruct and GPT-4o-mini, while the HF columns are Llama-3 8B and 70B.
- Repo stars and last commit (README fetch showed 82 stars, 48 commits; the GitHub API was rate-limited), PyPI availability of the package, and the README claim that the default `VerifiedDecisionPolicy` max failure rate is 2% (single fetch).
- Whether the implemented policy's confidence band matches the proof (my reading of Appendix C only); any independent reproduction of the 12.5x / 26x numbers.
- Whether any other paper has already compared vCache with a Clopper-Pearson or LTT selective-risk certificate (not searched).

## Addendum: `certify(..., m_min=59)` is a heuristic, not a certificate (verify, 2026-10-17)

Verdict: the concern is CONFIRMED and the "Reference implementation of the bound ... copy and use" label is CORRECTED. `ucb()` (exact Clopper-Pearson one-sided bound) is fine to copy. `certify()` as written must not be called a certificate and the table row above should read "heuristic, not LTT-valid; use `certify_grid` below".

Why (sources):
- LTT's fixed-sequence test (Section 2.3.1, Algorithm 1, Prop. 3) is "designed for settings where we know a-priori which hypotheses are more or less likely to control the risk"; the ordering is an "extra design-choice: selecting the ordering without looking at the calibration data"; multi-start is allowed "provided we adjust the significance level accordingly" (delta / number of starts). Learning the ordering from data uses a separate split (their Appendix D). https://arxiv.org/abs/2110.01052 (read via https://ar5iv.labs.arxiv.org/html/2110.01052; the first 100k of 175k characters only, Appendix D itself not read).
- Theorem 1 guarantee is P(all selected lambda have risk <= alpha) >= 1 - delta, conditional on a FWER-controlling procedure over a fixed hypothesis family. `certify()` walks the sample's own score values and starts at the 59th-highest score, so both the hypothesis set and the start are data-chosen, which is outside that theorem. Same requirement in general fixed-sequence theory: order given in advance (https://arxiv.org/pdf/1611.03146). Data-driven ordering literature controls FDR, not FWER, so it does not rescue the claim. I found no source proving validity for a data-chosen grid or start.
- The same LTT-style recipe is used with an explicit pre-specified sequence in recent work: retention sequence 0.80 down to 0.20 fixed in advance (https://arxiv.org/pdf/2609.27981); the proof logic is that an error needs the first true null in the sequence to be rejected, probability <= delta (https://arxiv.org/pdf/2607.13221). Both read only through search snippets.

New evidence (own sim, `sim5.py` in the scratchpad; scores ~ Beta(10,1), error ~ Bernoulli(1-s), 5% budget, delta 0.05, 1000 trials, SE about 0.4-0.7 points, true risk of the returned cutoff computed in closed form):
| n | skip m_min=59: P(true risk > 5%) / P(found) | skip m_min=150 | pre-registered 10-cutoff grid 0.70-0.99, Bonferroni delta/10 |
|---|---|---|---|
| 500 | 1.7% / 0.71 | 2.2% / 0.54 | 0.1% / 0.19 |
| 2000 | 3.1% / 0.91 | 4.1% / 0.98 | 0.0% / 0.985 |
| 8000 | 4.9% / 0.98 | 3.9% / 1.00 | 0.0% / 1.00 |
| 32000 | 4.2% / 0.99 | 5.3% / 1.00 | not run |
- The violation rate of the data-walk climbs toward the 5% allowance as n grows (1.7 -> 3.1 -> 4.9% for m_min=59) and m_min=150 reaches 5.3% at n=32000. No run is significantly above 5%, so this is not a demonstrated violation, but there is no margin and no theorem. The earlier 4.5% at n=2000 (600 trials) is within noise of that. This sim uses perfectly calibrated scores only; miscalibrated scores were not tested (unverified).
- Failure mode for m_min=59: P(found) is only 0.71 at n=500 because one error in the first 59 rows ends the walk (59 is the zero-error minimum, so it has no slack). The pre-registered grid is the most conservative at n=500 (found 0.19) but is valid and finds a cutoff 98.5-100% of the time from n=2000.

Replacement (tested: asserts that the returned cutoff passes the bound at delta, and that all-error data returns None; the Bonferroni variant is the same loop with `d = delta/len(grid)` and no stop-at-first-failure):
```python
import numpy as np
from scipy.stats import beta
def ucb(k, m, delta): return 1.0 if m == 0 or k >= m else beta.ppf(1 - delta, k + 1, m - k)
def certify_grid(scores, err, grid, starts, alpha=0.05, delta=0.05):
    # grid: PRE-REGISTERED descending cutoffs, fixed before seeing data (never taken from the sample).
    # starts: PRE-REGISTERED indices into grid. LTT multi-start fixed-sequence, level delta/len(starts) each.
    s, e = np.asarray(scores), np.asarray(err); d = delta / len(starts); best = None
    for j in starts:
        for c in grid[j:]:
            a = s >= c
            if ucb(int(e[a].sum()), int(a.sum()), d) > alpha: break   # stop at first failure
            best = c if best is None or c < best else best
    return best   # lowest cutoff certified, or None
# e.g. grid = [0.99 - 0.01*i for i in range(30)], starts=[0, 10, 20]
```
Rules for the smoke: set grid and starts in config before the shadow period ends; if the grid must be tuned, tune it on a separate split (LTT Appendix D); per-call-site delta/S still applies on top; report the certificate as "LTT fixed-sequence, grid G, starts J, delta, alpha" so it is reproducible. Do not use the string "certificate" for `certify()` output; call it "empirical cutoff (uncertified)".

Unresolved: whether the conditional-on-scores view (treat scores as fixed, only labels random) makes the data-walk valid for a different estimand; that is my own reasoning, not checked against any source. MAPIE's `BinaryClassificationController` still has an undocumented bound and was not tested here.

## Addendum: real reviewer-edit data for Mission 4 (MLQE-PE, LangMark, WMT APE)

Research date: 2026-10-17. Verdict on the claim "mission 4 must use simulated reviewer edits": CORRECTED. Public, login-free datasets of model output paired with human post-edits exist and I downloaded two of them. They are translation, not bookkeeping, so the bookkeeping demo still needs a simulator, but the "predict whether a reviewer changes this output, then certify the cutoff" pipeline can be validated on real human edits. "(calc)" = computed by me locally from the downloaded files (scripts: C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Desktop-personalprojects-Dreaming\4bf0a8c4-ab7a-4fd7-a483-c5f84a5e6e6e\scratchpad\mlqe_stats.py and lm_stats.py; data in sibling dirs mlqe_data and langmark_data, not in the repo).

### Candidates

| Dataset | Content | Access (checked) | License | Fit |
|---|---|---|---|---|
| LangMark (Welocalize + Duke, arXiv 2511.17153, https://arxiv.org/html/2511.17153v1) | 206,983 (source, NMT output, human post-edit) triplets, EN to de/es/fr/it/ja/pt-BR/ru; one proprietary in-domain NMT system; Smartsheet content; columns `source, pre_translation, human_post_edit, source_language, target_language, content_specialty, mqm_error` | Zenodo https://zenodo.org/records/15553365, `langmark.csv` 68.3 MB, "Open" access, no request gate. I curl-downloaded it (`.../files/langmark.csv?download=1`) and the md5 matched the record (f186444fd22776fedb538981ee84aca8) | CC BY 4.0 (Zenodo record) | Best fit. Production-style post-editing by expert linguists, with an unchanged-vs-edited outcome per segment |
| MLQE-PE (arXiv 2010.04480, LREC 2022, https://arxiv.org/abs/2010.04480; repo https://github.com/sheffieldnlp/mlqe-pe) | 11 language pairs, up to 10K sentences per pair, each with MT, post-edit, sentence HTER, word-level OK/BAD tags, plus 6 direct-assessment scores per segment. Post-editors: paid Unbabel community editors for En-De/En-Zh, professionals subcontracted by Facebook for other pairs; editors had no access to DA scores | GitHub repo public, `data/post-editing/{train,dev,test}`. HF mirror of the WMT20 version (En-De and En-Zh only): https://huggingface.co/datasets/wmt/wmt20_mlqe_task2, `gated:false`, direct parquet at `https://huggingface.co/api/datasets/wmt/wmt20_mlqe_task2/parquet/en-de/train/0.parquet` (checked, 200, no login) | Repo: CC0-1.0 (as shown on the repo page). HF card says "unknown" | Good for HTER regression and thresholded classification. Not a natural accept/modify decision (see below) |
| WMT APE 2019 task data (https://www.statmt.org/wmt19/ape-task.html) | En-De IT-domain triplets (src, NMT mt, human pe), data from QT21: train/dev at http://hdl.handle.net/11372/LRT-2613, test at lindat.mff.cuni.cz; En-Ru from Microsoft at hltdistributor.fbk.eu | Page mentions no login or registration. I did not download these | No license stated on the page; asks to cite the findings paper | Older, smaller. Triplet counts not on the page (unverified). eSCAPE listed there is synthetic (machine-made pseudo post-edits), so it is NOT real reviewer data |

### Numbers computed from the downloads (calc)

- MLQE `wmt20_mlqe_task2` (HF parquet), sentence HTER:
  - En-De train n=7,000: mean 0.318, median 0.300, HTER=0 in 6.6%, HTER<=0.1 in 15.3%. Validation n=1,000: HTER=0 in 7.1%.
  - En-Zh train n=7,000: mean 0.628, median 0.643, HTER=0 in 0.2%.
  - The `test` split has `hter = -10000.0` for every row (labels hidden), so usable labeled rows are 8,000 per pair (train + validation).
  - The paper's test21 averages are different (En-De 0.18, En-Zh 0.23, Ne-En 0.66; Table 2 of arXiv 2010.04480), so the WMT20 HF version and the test21 data are different draws. Do not mix them.
- LangMark, byte-identical after strip, `pre_translation == human_post_edit`: 27.9% of 206,983 rows left unchanged. By target language: ru 17.2% (n=9,266), fr 21.0%, ja 21.3%, de 22.2%, es-419 31.4%, pt-BR 36.6%, it 37.2%. By `content_specialty`: Marketing 26.4% (n=191,315) vs Product 45.8% (n=15,668). `mqm_error` is empty in 29.2% of rows, close to the unchanged rate (I did not check that these are the same rows). Per-language counts in the CSV differ slightly from the paper's Table 1 (e.g. de-DE 34,621 vs 33,308 in the paper); total matches.
- LangMark paper does not report the unchanged fraction or HTER; it gives only baseline TER of the NMT output vs post-edits (EN-DE about 31, EN-JP about 74), per the fetched HTML.

### What this changes for mission 4

- Real data is available for the key question "can a small model predict, from (input, output), which outputs a human editor changes, and can we certify a cutoff for it". LangMark gives 28% unchanged / 72% changed with 30K+ rows per language, so it supports the 1,500+ calibration rows per call site recommended earlier in this file, with room to split train/calibrate/test. MLQE En-De has 8,000 labeled rows.
- The binary label "changed at all" is the wrong target on MLQE: 93% of En-De and 99.8% of En-Zh outputs were edited. Use HTER thresholded near the median (about 0.3 En-De) or regression. LangMark's changed/unchanged is a usable binary target (28/72).
- Certificate mapping: treat "editor changes the output" as the error event on rows we would auto-accept. At a 5% budget the auto-accept slice must have under 5% changed rows, while the base rate of unchanged is only 28% (LangMark) or 7% (MLQE En-De). Expect small certified coverage on these corpora. This is a property of translation post-editing, not of the method; I did not run a predictor.
- Validity caveats: (a) translators post-edit every segment as a paid job, so this is forced review, not a reviewer who can approve at a glance; (b) "changed" includes style and terminology preference edits (LangMark `mqm_error` is dominated by Terminology and Linguistic Conventions); (c) the model is a single fixed NMT system, not a frontier LLM; (d) the domain is translation, so category/amount/vendor features do not transfer. Treat results as pipeline validation, not as bookkeeping evidence.
- ChaosNLI and AlleNoise (section 6 above) remain indirect proxies; LangMark and MLQE-PE are direct (output, human-edited output) pairs from real human editors and should be listed first for mission 4.

### Not found / unverified

- LLM-draft edit logs: two searches (accept/edit logs of LLM drafts; LLM outputs with human post-edits) found no public dataset of frontier-LLM drafts with reviewer accept/modify decisions. HED-IT (machine-generated dialogues vs human post-edits, arXiv 2406.07288, https://arxiv.org/pdf/2406.07288) surfaced in search; I did not check its size, license or download. No public bookkeeping or invoice reviewer-edit data found.
- WMT APE triplet counts, exact download behavior of the LINDAT/FBK hosts, and licenses (page states none).
- MLQE-PE per-pair HTER=0 rates for pairs other than En-De and En-Zh (GitHub `data/post-editing` folder exists with train/dev/test; I did not download it). The HF card's license is "unknown" while the GitHub repo shows CC0-1.0; confirm before redistributing.
- The "ACL 2025" venue for LangMark comes from a search-result summary, not the paper text I read.
- Number of editors per segment in LangMark (not stated in the paper text fetched) and in MLQE-PE post-editing (single post-edit per segment appears to be the design; not confirmed).
