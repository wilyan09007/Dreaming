"""S16b: finish the draft_note pruner evaluation from the cached S16 labels, with a VENDOR-grouped split.
Usage: `python run_b.py` (pod) or `python run_b.py --tiny` (local wiring check, a handful of prompts).
Pool = the 140 labelled prompts (ids 0-79, 200-259; labels, 3 full-prompt reference notes and judge verdicts are cached in labels.json / responses.jsonl).
Split = GroupKFold by transaction vendor (5 folds, no vendor in two folds). For fold f the span scorer is trained on the other folds only, and the
certificate cutoff is calibrated on the other folds only, so scorer-training, calibration and held-out rows never share a vendor.
Fixed fields (tx, inv, flag, bank) are always kept; a scorer only ranks history records and we keep the top m (m in 0, 2, 5). Extra arm: rule = tx+inv+flag only.
Every API call is cached in responses.jsonl, so a rerun only pays for cells that did not finish."""
import sys, json, math, time, hashlib
from pathlib import Path
from collections import Counter
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
TINY = "--tiny" in sys.argv; SUF = "_b_tiny" if TINY else "_b"
if not TINY:
    import torch; assert torch.cuda.is_available(), "pod came up without CUDA"
import core, llm
from core import *
from common import embed, cp_upper, certify
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest

BASE, BUD = 4.56006, 1.6          # OpenRouter ledger (deduplicated) when S16b started. S16b total cap is $2.5, but the interrupted first pod run spent ~$0.78-0.9 whose responses were lost with the pod,
                                  # so this run may add at most $1.6 incl. the 0.052 already cached and the local --tiny check
llm.CAP = BASE + BUD
EST = 0.0017                      # $ per prompt that needs a fresh note + judging (observed 0.0016-0.0021)
left = lambda: llm.CAP - llm.SPENT[0]
T0 = time.time(); LOG = open(HERE / f"log{SUF}.txt", "a", encoding="utf-8")
def say(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def spend(tag): say(f"[spend] {tag}: S16b has spent ${llm.SPENT[0]-BASE:.3f} of ${BUD} ({llm.NEW[0]} new calls this run, {time.time()-T0:.0f}s)")
RES = {}
def save(): json.dump(RES, open(HERE / f"results{SUF}.json", "w"), indent=1, default=str)
dif = lambda w: w != "SAME"
ub = lambda k, n: cp_upper(int(k), int(n), 0.05)

# ------------------------------------------------------------------ pool, labels, folds
labs = {int(k): v for k, v in json.load(open(HERE / "labels.json")).items()}
LAB = [0, 1, 2, 3, 4, 5, 200, 201, 202, 203, 204, 205] if TINY else sorted(labs)
REF = {i: labs[i]["refs"] for i in LAB}; TOK = {i: labs[i]["tok_full"] for i in LAB}; SL = {i: span_labels(P[i], labs[i]) for i in LAB}
K = 2 if TINY else 5
FOLD = {}
for f, (_, te) in enumerate(GroupKFold(K).split(LAB, groups=[P[i]["vendor"] for i in LAB])):
    for a in te: FOLD[LAB[a]] = f
vf = {}
for i in LAB: vf.setdefault(P[i]["vendor"], set()).add(FOLD[i])
assert all(len(s) == 1 for s in vf.values()), "a vendor landed in two folds"
say(f"=== S16b {'TINY ' if TINY else ''}pool {len(LAB)} labelled prompts, {len(vf)} vendors, {K} vendor-grouped folds sizes {[sum(FOLD[i]==f for i in LAB) for f in range(K)]}; S16 ledger {llm.SPENT[0]-BASE:+.3f} vs base")
old_tr = set(P[i]["vendor"] for i in range(80)); old_ho = [i for i in range(200, 260)]
RES["setup"] = dict(n=len(LAB), vendors=len(vf), folds=K, fold_sizes=[sum(FOLD[i] == f for i in LAB) for f in range(K)],
                    old_split_heldout_prompts_with_vendor_in_train=sum(P[i]["vendor"] in old_tr for i in old_ho), old_split_heldout_n=len(old_ho),
                    by_scn=dict(Counter(P[i]["scn"] for i in LAB)), max_prompts_per_vendor=max(Counter(P[i]["vendor"] for i in LAB).values()))
say(f"[old split] cached S16 split was by prompt id: {RES['setup']['old_split_heldout_prompts_with_vendor_in_train']}/{len(old_ho)} labelled held-out prompts had their vendor in the scorer-training prompts")

# ------------------------------------------------------------------ scorers (out-of-fold by vendor)
ET_ = embed([P[i]["tx"] for i in LAB] + [h["text"] for i in LAB for h in P[i]["hist"]])
ET = {i: ET_[k] for k, i in enumerate(LAB)}; off = len(LAB); EH = {}
for i in LAB: n = len(P[i]["hist"]); EH[i] = ET_[off:off + n]; off += n
NF = 10
def feats(i):   # rows: 4 fixed fields then history records; cols [tx inv flag bank is_hist rel_amt dup same_day log_recency cos | reason one-hot x5]
    p = P[i]; n = len(p["hist"]); fx = np.zeros((4, NF)); fx[range(4), range(4)] = 1; hs = np.zeros((n, NF)); hs[:, 4] = 1
    for j, h in enumerate(p["hist"]):
        rel = abs(h["amt"] - p["tx_amt"]) / p["tx_amt"]
        hs[j, 5] = min(rel, 5); hs[j, 6] = float(rel < .01 and h["days"] <= 7); hs[j, 7] = float(h["days"] == 0); hs[j, 8] = math.log1p(h["days"]); hs[j, 9] = float(EH[i][j] @ ET[i])
    f = p["flag"]; c = [("duplicate" in f), ("first purchase" in f), ("unusually high" in f), ("no matching invoice" in f)]; ctx = np.array(c + [not any(c)], float)
    return np.c_[fx, np.zeros((4, 5))], np.c_[hs, np.tile(ctx, (n, 1))]
XS = {i: feats(i) for i in LAB}
def stack(ids):
    X, y = [], []
    for i in ids: fx, hs = XS[i]; X += [fx, hs]; y += [[SL[i]["fixed"][f] for f in FIELDS], SL[i]["hist"]]
    return np.vstack(X), np.concatenate(y)
PF, PH = {}, {}
for f in range(K):
    m_ = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(*stack([i for i in LAB if FOLD[i] != f]))
    for i in LAB:
        if FOLD[i] == f: PF[i], PH[i] = m_.predict_proba(XS[i][0])[:, 1], m_.predict_proba(XS[i][1])[:, 1]
from sentence_transformers import CrossEncoder
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu" if TINY else "cuda")
CE = {i: ce.predict([(P[i]["tx"], h["text"]) for h in P[i]["hist"]], batch_size=64, show_progress_bar=False) for i in LAB}
rng = np.random.default_rng(16); RS, RC = {}, {}
for i in LAB: RS[i] = rng.random(len(P[i]["hist"])); RC[i] = float(rng.random())
dys = lambda i: np.array([h["days"] for h in P[i]["hist"]], float)
SCORE = {"learned": lambda i: PH[i], "cross-encoder": lambda i: CE[i], "oracle": lambda i: 1000.0 * np.array(SL[i]["hist"]) - dys(i), "random": lambda i: RS[i], "recency": lambda i: -dys(i)}
topm = lambda sc, i, m: tuple(sorted(int(j) for j in np.argsort(-SCORE[sc](i), kind="stable")[:m]))
def conf(sc, i, kept):   # higher = safer to prune; learned/CE: how relevant is the most relevant DROPPED record; oracle: share of label-positive records kept; rule: learned p over bank + all history
    n = len(P[i]["hist"]); D = [j for j in range(n) if j not in kept]
    if not D and sc in ("learned", "cross-encoder"): return 1e9   # nothing dropped: safest
    if sc == "learned": return float(1 - PH[i][D].max())
    if sc == "cross-encoder": return float(-CE[i][D].max())
    if sc == "oracle": pos = [j for j in range(n) if SL[i]["hist"][j]]; return sum(j in kept for j in pos) / len(pos) if pos else 1.0
    if sc == "random": return RC[i]
    if sc == "rule": return float(1 - max(PF[i][3], PH[i].max()))
def keepfn(sc, m):
    def f(i):
        k = keep_all(P[i])
        if sc == "rule": k["bank"] = 0; k["hist"] = ()
        else: k["hist"] = topm(sc, i, m)
        return k
    return f
# free offline diagnostics: do the scorers rank the label-positive history records first (vendor-out-of-fold)?
D = {}
for sc in ("learned", "cross-encoder", "recency", "random"):
    mixed = [i for i in LAB if 0 < sum(SL[i]["hist"]) < len(SL[i]["hist"])]; withpos = [i for i in LAB if sum(SL[i]["hist"]) > 0]
    d = dict(within_auc=float(np.mean([roc_auc_score(SL[i]["hist"], SCORE[sc](i)) for i in mixed])) if mixed else None, n_mixed=len(mixed))
    for m in (2, 5):
        kp = {i: set(topm(sc, i, m)) for i in withpos}
        d[f"recall@{m}"] = float(np.mean([sum(SL[i]["hist"][j] for j in kp[i]) / sum(SL[i]["hist"]) for i in withpos]))
        d[f"all_needed_kept@{m}"] = float(np.mean([all(SL[i]["hist"][j] == 0 or j in kp[i] for j in range(len(P[i]["hist"]))) for i in withpos]))
    D[sc] = d; say(f"[scorer vs history labels, vendor-OOF] {sc:14s} " + " ".join(f"{k} {round(v,3) if isinstance(v,float) else v}" for k, v in d.items()) + f" (prompts with a needed record: {len(withpos)})")
RES["scorer_diag"] = D; save()

# ------------------------------------------------------------------ cell evaluation
def note_cached(p, k, rep=0):
    ex = dict(EX_T); ex["usage"] = {"include": True}
    return hashlib.sha1(json.dumps([HAIKU, SYS_T, render(p, k), 200, ex, rep], sort_keys=True).encode()).hexdigest() in llm.R
def vboot(vals, ids, B=2000):   # vendor-cluster bootstrap 95% CI of a mean
    g = {}
    for v, i in zip(vals, ids): g.setdefault(P[i]["vendor"], []).append(float(v))
    s = np.array([sum(x) for x in g.values()]); c = np.array([len(x) for x in g.values()]); d = np.random.default_rng(0).integers(0, len(s), (B, len(s))); e = s[d].sum(1) / c[d].sum(1)
    return [float(np.percentile(e, 2.5)), float(np.percentile(e, 97.5))]
FL = {}
def floor_cell(ids):
    def one(i): p = P[i]; n = note(p, keep_all(p), 3); v = verdict(REF[i], n["text"], src="floor"); return dict(i=i, raw=v["raw"], cons=v["cons"], tok=n["tok"])
    for r in pmap(one, ids): FL[r["i"]] = r
def run_cell(name, kf, ids):
    new = sum(not note_cached(P[i], kf(i)) for i in ids)
    if new * EST > left() - 0.03: say(f"[cell] SKIPPED {name}: needs ~${new*EST:.2f} for {new} uncached notes, ${left():.2f} left"); return None
    def one(i):
        p = P[i]; k = kf(i); n = note(p, k, 0); v = verdict(REF[i], n["text"], src="b:" + name)
        return dict(i=i, raw=v["raw"], cons=v["cons"], tok=n["tok"], trunc=n["trunc"], kept_h=len(k["hist"]), kept_f=sum(k[f] for f in FIELDS), nh=len(p["hist"]))
    try: rows = pmap(one, ids)
    except Exception as e: say(f"[cell] FAILED {name}: {type(e).__name__} {str(e)[:200]}"); return None
    spend(f"{name} ({new} new notes)"); return rows
def summ(name, rows):
    ids = [r["i"] for r in rows]; n = len(rows); c = np.array([dif(r["cons"]) for r in rows]); w = np.array([dif(r["raw"]) for r in rows])
    o = dict(name=name, n=n, cons_dis=float(c.mean()), cons_ub=ub(c.sum(), n), cons_ci_vendor=vboot(c, ids), raw_dis=float(w.mean()), raw_ub=ub(w.sum(), n),
             tokens_removed=1 - sum(r["tok"] for r in rows) / sum(TOK[i] for i in ids), hist_kept=float(np.mean([r["kept_h"] for r in rows])), hist_total=float(np.mean([r["nh"] for r in rows])),
             fixed_dropped=float(np.mean([4 - r["kept_f"] for r in rows])), trunc=float(np.mean([r["trunc"] for r in rows])), by_scn={})
    for s in sorted({P[i]["scn"] for i in ids}):
        xs = [r for r in rows if P[r["i"]]["scn"] == s]; o["by_scn"][s] = dict(n=len(xs), cons_dis=float(np.mean([dif(r["cons"]) for r in xs])), tokens_removed=1 - sum(r["tok"] for r in xs) / sum(TOK[r["i"]] for r in xs))
    if all(i in FL for i in ids):
        fc = np.array([dif(FL[i]["cons"]) for i in ids]); b = int((c & ~fc).sum()); cc = int((~c & fc).sum())
        o["floor_cons"] = float(fc.mean()); o["vs_floor"] = dict(pruned_only_diff=b, floor_only_diff=cc, p=float(binomtest(b, b + cc, .5).pvalue) if b + cc else 1.0)
    say(f"[cell] {name:22s} n={n:3d} tokens removed {o['tokens_removed']*100:5.1f}% | hist kept {o['hist_kept']:.1f}/{o['hist_total']:.1f} | equivalent {100*(1-o['cons_dis']):.1f}% (disagree {o['cons_dis']:.3f}, CP UB {o['cons_ub']:.3f}, vendor-bootstrap {o['cons_ci_vendor'][0]:.3f}-{o['cons_ci_vendor'][1]:.3f}; raw {o['raw_dis']:.3f}) "
        f"| floor same rows {o.get('floor_cons', float('nan')):.3f} | discordant vs floor {o.get('vs_floor',{}).get('pruned_only_diff')}/{o.get('vs_floor',{}).get('floor_only_diff')} p={o.get('vs_floor',{}).get('p', float('nan')):.3f}")
    for s, d in o["by_scn"].items(): say(f"      {s:7s} n={d['n']:3d} tokens removed {d['tokens_removed']*100:5.1f}% disagree {d['cons_dis']:.3f}")
    return o
def cert(rows, cf, budget, excess=False):
    """cross-fitted certificate: for each vendor fold, cutoff from the OTHER folds (certify: fixed grid, delta 0.05), applied to that fold. Low-confidence prompts fall back to the full prompt."""
    ids = [r["i"] for r in rows]
    def ok(r): d = dif(r["cons"]) and (not excess or not dif(FL[r["i"]]["cons"])); return not d
    cov = err = saved = 0; cuts = []
    for f in range(K):
        te = [r for r in rows if FOLD[r["i"]] == f]; cal = [r for r in rows if FOLD[r["i"]] != f]
        if not te or not cal: continue
        c = certify([cf[r["i"]] for r in cal], [ok(r) for r in cal], budget); cuts.append(c["cutoff"])
        if np.isfinite(c["cutoff"]):
            sel = [r for r in te if cf[r["i"]] >= c["cutoff"]]; cov += len(sel); err += sum(not ok(r) for r in sel); saved += sum(TOK[r["i"]] - r["tok"] for r in sel)
    pc = certify([cf[i] for i in ids], [ok(r) for r in rows], budget)
    pooled_saved = sum(TOK[r["i"]] - r["tok"] for r in rows if cf[r["i"]] >= pc["cutoff"]) / sum(TOK[i] for i in ids) if np.isfinite(pc["cutoff"]) else 0.0
    return dict(budget=budget, covered=cov, coverage=cov / len(rows), errors=err, sel_error=err / cov if cov else None, ub=ub(err, cov) if cov else None,
                tokens_saved=saved / sum(TOK[i] for i in ids), folds_with_cutoff=int(sum(np.isfinite(x) for x in cuts)),
                pooled_in_sample=dict(coverage=pc["coverage"], emp_error=pc["emp_error"], bound=pc["bound"], tokens_saved=float(pooled_saved)))
CELLS, CERT = {}, {}
def certify_cell(name, rows, sc, kept_of):
    cf = {r["i"]: conf(sc, r["i"], kept_of(r["i"])) for r in rows}; out = {}
    for ex in ([False, True] if all(r["i"] in FL for r in rows) else [False]):
        for b in (0.05, 0.10, 0.15):
            c = cert(rows, cf, b, ex); out[("excess@" if ex else "") + f"{int(b*100)}%"] = c
            say(f"[certificate] {name:22s} {'(vs floor) ' if ex else '           '}budget {int(b*100):2d}%: covered {c['covered']:3d}/{len(rows)} ({c['coverage']*100:4.1f}%) error among covered {c['sel_error'] if c['sel_error'] is None else round(c['sel_error'],3)} "
                f"CP UB {c['ub'] if c['ub'] is None else round(c['ub'],3)} | input tokens saved {c['tokens_saved']*100:4.1f}% | folds with a cutoff {c['folds_with_cutoff']}/{K} | in-sample (all rows as calibration) coverage {c['pooled_in_sample']['coverage']*100:4.1f}% bound {c['pooled_in_sample']['bound'] if c['pooled_in_sample']['bound'] is None else round(c['pooled_in_sample']['bound'],3)} tokens {c['pooled_in_sample']['tokens_saved']*100:4.1f}%")
    CERT[name] = out; RES["cert"] = CERT; save()
def do(name, kf, sc):
    rows = run_cell(name, kf, LAB)
    if rows is None: return
    CELLS[name] = summ(name, rows); RES["cells"] = CELLS; save()
    certify_cell(name, rows, sc, lambda i: set(kf(i)["hist"]))

# m = 0 (free: the H0 ablation notes and verdicts are cached) is the same prompt for every scorer; only the confidence differs
rows0 = run_cell("m=0 (4 fixed fields)", keepfn("learned", 0), LAB)
floor_cell(LAB)   # ids 200-259 are cached from S16; 0-79 are new
fr = np.array([dif(FL[i]["cons"]) for i in LAB]); fw = np.array([dif(FL[i]["raw"]) for i in LAB])
RES["floor"] = dict(n=len(LAB), cons=float(fr.mean()), cons_ub=ub(fr.sum(), len(LAB)), cons_ci_vendor=vboot(fr, LAB), raw=float(fw.mean()), raw_ub=ub(fw.sum(), len(LAB)),
                    by_scn={s: float(np.mean([dif(FL[i]["cons"]) for i in LAB if P[i]["scn"] == s])) for s in sorted({P[i]["scn"] for i in LAB})})
say(f"[floor] two independent full-prompt notes on the same {len(LAB)} prompts: confirmed DIFFERENT {fr.mean():.3f} (CP UB {RES['floor']['cons_ub']:.3f}, vendor-bootstrap {RES['floor']['cons_ci_vendor'][0]:.3f}-{RES['floor']['cons_ci_vendor'][1]:.3f}); raw {fw.mean():.3f}; by scenario {RES['floor']['by_scn']}"); spend("floor"); save()
if rows0 is not None:
    CELLS["m=0"] = summ("m=0 (4 fixed fields)", rows0); RES["cells"] = CELLS; save()
    for sc in ("learned", "cross-encoder", "oracle", "random"): certify_cell(f"{sc} m=0", rows0, sc, lambda i: ())
# priority order: main scorer, strongest baseline, the rule, then the oracle ceiling and the random control
ORDER = [("learned", 5), ("learned", 2), ("cross-encoder", 5), ("cross-encoder", 2), ("rule", None), ("oracle", 5), ("random", 5), ("oracle", 2), ("random", 2)]
if TINY: ORDER = [("learned", 2), ("cross-encoder", 2), ("rule", None)]
for sc, m in ORDER: do(f"{sc} m={m}" if m else "rule (tx+inv+flag)", keepfn(sc, m), sc)

# ------------------------------------------------------------------ headline
say("=== headline: share of draft_note input tokens removable at each budget (cross-fitted, vendor-disjoint calibration/held-out, low-confidence prompts fall back to the full prompt)")
HL = {}
for key in ("5%", "10%", "15%", "excess@5%", "excess@10%", "excess@15%"):
    best = sorted(((v[key]["tokens_saved"], k, v[key]) for k, v in CERT.items() if key in v), key=lambda t: -t[0])[:3]; HL[key] = [(k, round(t, 4), c["coverage"], c["sel_error"], c["ub"]) for t, k, c in best]
    say(f"  budget {key:10s} " + " | ".join(f"{k}: tokens saved {t*100:.1f}% coverage {c['coverage']*100:.0f}% error {c['sel_error'] if c['sel_error'] is None else round(c['sel_error'],3)} (UB {c['ub'] if c['ub'] is None else round(c['ub'],3)})" for t, k, c in best) if best else f"  budget {key}: nothing certified")
RES["headline"] = HL
RES["cost"] = dict(s16b_spend=llm.SPENT[0] - BASE, ledger_total=llm.SPENT[0], new_calls=llm.NEW[0], seconds=time.time() - T0); save()
spend("end"); say(f"providers seen: {Counter((r['resp'].get('provider'), r['model']) for r in llm.R.values()).most_common()}")
