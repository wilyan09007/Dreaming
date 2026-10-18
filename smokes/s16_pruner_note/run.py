"""S16 run: context pruner on draft_note. `python run.py` (pod) or `python run.py --tiny` (local wiring check on a handful of prompts).
Every API call is cached in responses.jsonl (key = hash of the request), so reruns are free. Stages run in priority order under a spend guard:
labels (cached) -> floor -> scorers -> spec policies -> judge validation -> calibration + certificate -> extras (only if budget is left)."""
import sys, os, json, math, time
from pathlib import Path
from collections import Counter
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import core, llm
from core import *
from common import embed, cp_upper, certify, apply_certificate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from scipy.stats import binomtest

TINY = "--tiny" in sys.argv; SUF = "_tiny" if TINY else ""; RERUN = "--rerun" in sys.argv   # rerun: everything is cached, only new stages cost money, so skip the stage reserves
llm.CAP = 4.88   # ledger cap (earlier attempt already spent ~$3.0 of the $5 smoke cap)
if TINY: TRAIN, HELD, N_TR, N_OR = list(range(16)), list(range(200, 206)), 12, 6
else: TRAIN, HELD, N_TR, N_OR = list(range(200)), list(range(200, 300)), 80, 60
LAB_TR, LAB_HO = TRAIN[:N_TR], HELD[:N_OR]; UNL = TRAIN[N_TR:]
T0 = time.time(); LOG = open(HERE / f"log{SUF}.txt", "a", encoding="utf-8")
def say(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def spend(tag): say(f"[spend] {tag}: ${llm.SPENT[0]:.3f} ledger, {llm.NEW[0]} new calls this run, {time.time()-T0:.0f}s")
left = lambda: llm.CAP - llm.SPENT[0]
EST = 0.0024   # conservative $ per evaluated prompt (note + 1.2-1.5 sonnet judgments)
can = lambda n, reserve=0.0: left() >= (0.02 if RERUN else n * EST + reserve)
RES = {}
def save(): json.dump(RES, open(HERE / f"results{SUF}.json", "w"), indent=1, default=str)
ub = lambda k, n: cp_upper(int(k), int(n), 0.05)
dif = lambda w: w != "SAME"

# ------------------------------------------------------------------ labels (cached from the earlier attempt)
say(f"=== S16 run {'TINY ' if TINY else ''}N_TR={N_TR} N_OR={N_OR} ledger at start ${llm.SPENT[0]:.3f}")
labs = {}
def do_labels(ids):
    todo = [i for i in ids if i not in labs]
    for k in range(0, len(todo), 24):
        for l in pmap(label_prompt, todo[k:k + 24]): labs[l["i"]] = l
        say(f"[labels] {len(labs)} prompts labelled; ${llm.SPENT[0]:.3f}")
do_labels(LAB_HO + LAB_TR)
if not TINY: json.dump({str(k): v for k, v in labs.items()}, open(HERE / "labels.json", "w"))
REF, TOK = {}, {}
def need_refs(ids):   # three independent full-prompt haiku notes per prompt (A0 is the reference); tokens of the full prompt
    for i in ids:
        if i in labs and i not in REF: REF[i], TOK[i] = labs[i]["refs"], labs[i]["tok_full"]
    todo = [i for i in ids if i not in REF]
    for i, r in zip(todo, pmap(refs_of, todo)): REF[i], TOK[i] = [x["text"] for x in r], r[0]["tok"]
SL = {i: span_labels(P[i], labs[i]) for i in labs}
spend("labels")
def rate(ids, f): v = [f(i) for i in ids]; return (sum(v) / len(v)) if v else None
L = {}
for nm, ids in (("train", LAB_TR), ("held-out subset", LAB_HO), ("all labelled", LAB_TR + LAB_HO)):
    L[nm] = dict(n=len(ids), H0_cons=rate(ids, lambda i: dif(labs[i]["var"]["H0"]["cons"])), H0_raw=rate(ids, lambda i: dif(labs[i]["var"]["H0"]["raw"])),
                 dist=rate(ids, lambda i: SL[i]["dist"]), **{f: rate(ids, lambda i, f=f: SL[i]["fixed"][f]) for f in FIELDS},
                 hist_pos_share=float(np.mean([x for i in ids for x in SL[i]["hist"]])), hist_lit_share=float(np.mean([x for i in ids for x in SL[i]["hist_lit"]])),
                 trunc=float(np.mean([v["trunc"] for i in ids for v in labs[i]["var"].values()])),
                 calls_per_variant=float(np.mean([v["n"] for i in ids for v in labs[i]["var"].values()])), variants_per_prompt=float(np.mean([len(labs[i]["var"]) for i in ids])))
    say(f"[label stats] {nm}: " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in L[nm].items()}))
by = {}
for i in LAB_TR + LAB_HO: by.setdefault(P[i]["scn"], []).append(i)
for s, ids in sorted(by.items()):
    say(f"[label stats by scenario] {s:7s} n={len(ids):3d} H0 changes note {np.mean([dif(labs[i]['var']['H0']['cons']) for i in ids]):.2f} | "
        f"tx {np.mean([SL[i]['fixed']['tx'] for i in ids]):.2f} inv {np.mean([SL[i]['fixed']['inv'] for i in ids]):.2f} flag {np.mean([SL[i]['fixed']['flag'] for i in ids]):.2f} bank {np.mean([SL[i]['fixed']['bank'] for i in ids]):.2f} | "
        f"dist {np.mean([SL[i]['dist'] for i in ids]):.2f} | hist spans needed {np.mean([np.mean(SL[i]['hist']) for i in ids]):.2f}")
RES["label_stats"] = L; save()

# ------------------------------------------------------------------ noise floor on the held-out prompts (two independent full-prompt notes, judged)
need_refs(HELD)
def floor_one(i):
    p = P[i]; n = note(p, keep_all(p), 3); v = verdict(REF[i], n["text"], src="floor"); return dict(i=i, raw=v["raw"], cons=v["cons"], tok=n["tok"])
FL = {r["i"]: r for r in pmap(floor_one, HELD)}
fr = np.mean([dif(FL[i]["raw"]) for i in HELD]); fc = np.mean([dif(FL[i]["cons"]) for i in HELD])
say(f"[floor] two independent full-prompt notes judged against each other (held-out n={len(HELD)}): raw DIFFERENT {fr:.3f} (UB95 {ub(fr*len(HELD),len(HELD)):.3f}); confirmed (majority of 3 refs) DIFFERENT {fc:.3f} (UB95 {ub(fc*len(HELD),len(HELD)):.3f})")
for s in sorted({P[i]["scn"] for i in HELD}):
    ids = [i for i in HELD if P[i]["scn"] == s]; say(f"[floor by scenario] {s:7s} n={len(ids):3d} raw {np.mean([dif(FL[i]['raw']) for i in ids]):.3f} confirmed {np.mean([dif(FL[i]['cons']) for i in ids]):.3f}")
RES["floor"] = dict(raw=float(fr), cons=float(fc), n=len(HELD), ub_raw=ub(fr * len(HELD), len(HELD)), ub_cons=ub(fc * len(HELD), len(HELD)))
save(); spend("floor")

# ------------------------------------------------------------------ scorers (embeddings + learned LR, cross-encoder)
allq = TRAIN + HELD
ET_ = embed([P[i]["tx"] for i in allq] + [h["text"] for i in allq for h in P[i]["hist"]])
ET = {i: ET_[k] for k, i in enumerate(allq)}; off = len(allq); EH = {}
for i in allq: n = len(P[i]["hist"]); EH[i] = ET_[off:off + n]; off += n
NF = 10
def X_of(i):   # rows: 4 fixed fields (one-hot) then history records: [tx inv flag bank is_hist rel_amt dup same_day log_recency cos]
    p = P[i]; n = len(p["hist"]); fx = np.zeros((4, NF)); fx[range(4), range(4)] = 1; hs = np.zeros((n, NF)); hs[:, 4] = 1
    for j, h in enumerate(p["hist"]):
        rel = abs(h["amt"] - p["tx_amt"]) / p["tx_amt"]
        hs[j, 5] = min(rel, 5); hs[j, 6] = float(rel < .01 and h["days"] <= 7); hs[j, 7] = float(h["days"] == 0); hs[j, 8] = math.log1p(h["days"]); hs[j, 9] = float(EH[i][j] @ ET[i])
    return fx, hs
XS = {i: X_of(i) for i in allq}
def ctx_of(i):  # flag-reason code read from the flag line (part of the prompt): duplicate / first purchase / unusually high / no invoice / none
    f = P[i]["flag"]; c = [("duplicate" in f), ("first purchase" in f), ("unusually high" in f), ("no matching invoice" in f)]; return np.array(c + [not any(c)], float)
def feats(i, ctx):
    fx, hs = XS[i]
    if ctx: hs = np.c_[hs, np.tile(ctx_of(i), (len(hs), 1))]; fx = np.c_[fx, np.zeros((4, 5))]
    return fx, hs
def stack(ids, ctx=False):
    X, y = [], []
    for i in ids: fx, hs = feats(i, ctx); s = SL[i]; X += [fx, hs]; y += [[s["fixed"][f] for f in FIELDS], s["hist"]]
    return np.vstack(X), np.concatenate(y)
def mk(): return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
def predict(m, i, ctx=False): fx, hs = feats(i, ctx); return m.predict_proba(fx)[:, 1], m.predict_proba(hs)[:, 1]
PRB, PRB_CTX, MODELS = {}, {}, {}
for ctx, store in ((False, PRB), (True, PRB_CTX)):
    for tr, te in KFold(min(5, len(LAB_TR)), shuffle=True, random_state=0).split(LAB_TR):
        m = mk().fit(*stack([LAB_TR[a] for a in tr], ctx))
        for b in te: store[LAB_TR[b]] = predict(m, LAB_TR[b], ctx)
    m = mk().fit(*stack(LAB_TR, ctx)); MODELS[ctx] = m
    for i in UNL + HELD: store[i] = predict(m, i, ctx)
names = ["tx", "inv", "flag", "bank", "hist", "rel_amt", "dup", "same_day", "log_recency", "cos"]
say(f"[learned coef (standardized)] base {dict(zip(names, MODELS[False][-1].coef_[0].round(2).tolist()))}")
say(f"[learned coef (standardized)] +reason {dict(zip(names + ['r_dup', 'r_first', 'r_high', 'r_noinv', 'r_none'], MODELS[True][-1].coef_[0].round(2).tolist()))}")
from sentence_transformers import CrossEncoder
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
CE = {i: ce.predict([(P[i]["tx"], h["text"]) for h in P[i]["hist"]], batch_size=64, show_progress_bar=False) for i in HELD}
def auc_block(ids, score_of, name):
    y = np.concatenate([SL[i]["hist"] for i in ids]); s = np.concatenate([score_of(i) for i in ids])
    wa = [roc_auc_score(SL[i]["hist"], score_of(i)) for i in ids if 0 < sum(SL[i]["hist"]) < len(SL[i]["hist"])]
    out = dict(pooled=float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None, within=float(np.mean(wa)) if wa else None, n_within=len(wa), pos=float(y.mean()))
    say(f"[scorer AUC vs history-span label] {name:34s} pooled {out['pooled'] if out['pooled'] is None else round(out['pooled'],3)} within-prompt {out['within'] if out['within'] is None else round(out['within'],3)} (n_prompts {len(wa)}; positive rate {out['pos']:.2f})")
    return out
rng = np.random.default_rng(0); RAND = {i: rng.random(len(P[i]["hist"])) for i in allq}
rec = lambda i: -np.array([h["days"] for h in P[i]["hist"]], float)
A = {"learned_oof_train": auc_block(LAB_TR, lambda i: PRB[i][1], "learned (OOF, train)"), "learned_reason_oof_train": auc_block(LAB_TR, lambda i: PRB_CTX[i][1], "learned+reason (OOF, train)")}
for nm, f in (("learned", lambda i: PRB[i][1]), ("learned+reason", lambda i: PRB_CTX[i][1]), ("cross-encoder", lambda i: CE[i]), ("recency", rec), ("amount closeness", lambda i: -XS[i][1][:, 5]), ("random", lambda i: RAND[i])):
    A[nm + "_heldout"] = auc_block(LAB_HO, f, nm + " (held-out labelled)")
RES["auc"] = A
for f_i, f in enumerate(FIELDS):
    y = [SL[i]["fixed"][f] for i in LAB_HO]; s = [PRB[i][0][f_i] for i in LAB_HO]
    say(f"[fixed-field {f}] label rate held-out {np.mean(y):.2f}; mean learned p {np.mean(s):.2f}")
grid = [0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
def safe_share(th, P_):   # share of OOF train prompts where every label-positive span has p >= th (so it would be kept)
    return np.mean([all(P_[i][0][k] >= th for k, f in enumerate(FIELDS) if SL[i]["fixed"][f]) and all(P_[i][1][j] >= th for j in np.flatnonzero(SL[i]["hist"])) for i in LAB_TR])
def drop_share(th, P_):   # share of prompt characters below threshold
    d = t = 0
    for i in LAB_TR:
        p = P[i]; ch = np.array([len(p[f]) for f in FIELDS] + [len(h["text"]) for h in p["hist"]]); pr = np.r_[P_[i][0], P_[i][1]]; d += ch[pr < th].sum(); t += ch.sum()
    return d / t
def pick(q, P_): c = [th for th in grid if safe_share(th, P_) >= q]; return max(c) if c else grid[0]
for nm, P_ in (("base", PRB), ("+reason", PRB_CTX)):
    for th in grid: say(f"[theta grid {nm}] {th:.2f}: OOF prompts with every needed span kept {safe_share(th, P_):.2f}, chars dropped {drop_share(th, P_):.2f}")
TH = {"base": (pick(0.90, PRB), PRB), "reason": (pick(0.90, PRB_CTX), PRB_CTX), "base97": (pick(0.97, PRB), PRB), "reason97": (pick(0.97, PRB_CTX), PRB_CTX)}
say(f"[theta] A (>=90% of OOF prompts keep every needed span): base {TH['base'][0]}, +reason {TH['reason'][0]}; B (>=97%): base {TH['base97'][0]}, +reason {TH['reason97'][0]}")
RES["theta"] = {k: v[0] for k, v in TH.items()}; save()

# ------------------------------------------------------------------ policies
def keep_rule(i): k = keep_all(P[i]); k["bank"] = 0; k["hist"] = (); return k
def keep_topk(i, kk): k = keep_rule(i); k["hist"] = tuple(sorted(int(j) for j in np.argsort(-CE[i], kind="stable")[:kk])); return k
def keep_recent(i, kk): k = keep_rule(i); k["hist"] = tuple(range(min(kk, len(P[i]["hist"])))); return k
def keep_prob(i, key):
    th, P_ = TH[key]; pf, ph = P_[i]; k = {f: int(pf[a] >= th) for a, f in enumerate(FIELDS)}; k["hist"] = tuple(int(j) for j in np.flatnonzero(ph >= th))
    if not any(k[f] for f in FIELDS): k["flag"] = 1
    return k
def keep_oracle(i, plus):
    s = SL[i]; k = {f: s["fixed"][f] for f in FIELDS}; k["hist"] = tuple(j for j, x in enumerate(s["hist" if plus else "hist_lit"]) if x)
    if not any(k[f] for f in FIELDS): k["flag"] = 1
    return k
def conf_prob(i, key):   # 1 - largest probability among DROPPED spans; -1 when nothing is dropped (not a pruned prompt, never counted as covered)
    th, P_ = TH[key]; pf, ph = P_[i]; d = [x for x in list(pf) + list(ph) if x < th]; return 1 - max(d) if d else -1.0
def eval_policy(name, keepfn, ids, extra=None):
    need_refs(ids)
    def one(i):
        p = P[i]; k = keepfn(i); full = all(k[f] for f in FIELDS) and len(k["hist"]) == len(p["hist"])
        n = note(p, k, 4 if full else 0); v = verdict(REF[i], n["text"], src="pol:" + name)
        return dict(i=i, raw=v["raw"], cons=v["cons"], tok=n["tok"], trunc=n["trunc"], kept_h=len(k["hist"]), kept_f=sum(k[f] for f in FIELDS), nh=len(p["hist"]), full=bool(full), **(extra(i) if extra else {}))
    return pmap(one, ids)
def summ(res, name):
    n = len(res); r = np.array([dif(x["raw"]) for x in res]); c = np.array([dif(x["cons"]) for x in res]); tk = sum(x["tok"] for x in res); tf = sum(TOK[x["i"]] for x in res)
    o = dict(name=name, n=n, raw_dis=float(r.mean()), raw_ub=ub(r.sum(), n), cons_dis=float(c.mean()), cons_ub=ub(c.sum(), n), tokens_removed=1 - tk / tf, mean_hist_kept=float(np.mean([x["kept_h"] for x in res])),
             mean_hist=float(np.mean([x["nh"] for x in res])), fixed_dropped=float(np.mean([4 - x["kept_f"] for x in res])), trunc=float(np.mean([x["trunc"] for x in res])), frac_full=float(np.mean([x["full"] for x in res])), by_scn={})
    for s in sorted({P[x["i"]]["scn"] for x in res}):
        xs = [x for x in res if P[x["i"]]["scn"] == s]; o["by_scn"][s] = dict(n=len(xs), tok_removed=1 - sum(x["tok"] for x in xs) / sum(TOK[x["i"]] for x in xs), cons_dis=float(np.mean([dif(x["cons"]) for x in xs])), raw_dis=float(np.mean([dif(x["raw"]) for x in xs])))
    if all(x["i"] in FL for x in res):   # paired against the full-vs-full floor on the same prompts
        for m in ("raw", "cons"):
            b = sum(dif(x[m]) and not dif(FL[x["i"]][m]) for x in res); cc = sum((not dif(x[m])) and dif(FL[x["i"]][m]) for x in res)
            o["excess_" + m] = dict(policy_only_diff=b, floor_only_diff=cc, p=float(binomtest(b, b + cc, .5).pvalue) if b + cc else 1.0, floor_rate=float(np.mean([dif(FL[x["i"]][m]) for x in res])))
    e = o.get("excess_cons", {}); er = o.get("excess_raw", {})
    say(f"[policy] {name:22s} n={n:3d} tokens removed {o['tokens_removed']*100:5.1f}% | hist kept {o['mean_hist_kept']:.1f}/{o['mean_hist']:.1f} | fixed dropped {o['fixed_dropped']:.2f} | disagree raw {o['raw_dis']:.3f} (UB {o['raw_ub']:.3f}) "
        f"confirmed {o['cons_dis']:.3f} (UB {o['cons_ub']:.3f}) | floor on same prompts raw {er.get('floor_rate', float('nan')):.3f} conf {e.get('floor_rate', float('nan')):.3f} | discordant vs floor raw {er.get('policy_only_diff')}/{er.get('floor_only_diff')} p={er.get('p', float('nan')):.3f}, conf {e.get('policy_only_diff')}/{e.get('floor_only_diff')} p={e.get('p', float('nan')):.3f} | trunc {o['trunc']:.3f}")
    for s, d in o["by_scn"].items(): say(f"    by scenario {s:7s} n={d['n']:3d} tokens removed {d['tok_removed']*100:5.1f}% confirmed-disagree {d['cons_dis']:.3f} raw {d['raw_dis']:.3f}")
    return o
POL, PR = {}, {}
def char_drop(keepfn, ids):   # share of prompt characters the policy removes (free, no API)
    d = t = 0
    for i in ids:
        p = P[i]; k = keepfn(i); ch = {f: len(p[f]) for f in FIELDS}; kh = set(k["hist"]); t += sum(ch.values()) + sum(len(h["text"]) for h in p["hist"])
        d += sum(ch[f] for f in FIELDS if not k[f]) + sum(len(h["text"]) for j, h in enumerate(p["hist"]) if j not in kh)
    return d / t
def run_policy(name, keepfn, ids, conffn=None, reserve=0.0, min_drop=0.0):
    if name in PR: return
    cd = char_drop(keepfn, ids)
    if cd < min_drop: say(f"[policy] NOT EVALUATED {name}: removes only {cd*100:.1f}% of characters on {len(ids)} prompts, so it is the full prompt for practical purposes"); POL[name] = dict(name=name, chars_removed=cd, evaluated=False); return
    if not can(len(ids), reserve): say(f"[policy] SKIPPED {name}: ledger ${llm.SPENT[0]:.3f}, needs ~${len(ids)*EST:.2f} + reserve ${reserve:.2f}"); return
    try:
        PR[name] = eval_policy(name, keepfn, ids, extra=(lambda i: dict(conf=conffn(i))) if conffn else None)
        POL[name] = summ(PR[name], name); POL[name]["chars_removed"] = cd; RES["policies"] = POL; save(); spend(name)
    except Exception as e: say(f"[policy] FAILED {name}: {type(e).__name__} {str(e)[:200]}")
R_VAL, R_CAL = 0.38, 0.20   # budget reserved for judge validation and for calibration of the main learned scorer
LA, LR_ = "learned A", "learned+reason A"
for nm, kf, ids, ck, md in ((LR_, lambda i: keep_prob(i, "reason"), HELD, lambda i: conf_prob(i, "reason"), 0), ("rule(tx,inv,flag)", keep_rule, HELD, None, 0), ("cross-enc top-3", lambda i: keep_topk(i, 3), HELD, None, 0),
                            ("oracle (literal)", lambda i: keep_oracle(i, False), LAB_HO, None, 0), (LA, lambda i: keep_prob(i, "base"), HELD, lambda i: conf_prob(i, "base"), 0.05)):
    run_policy(nm, kf, ids, ck, R_VAL + R_CAL, md)
sub = set(LAB_HO)
for nm in list(PR):
    if nm in PR and nm != "oracle (literal)" and not nm.endswith("[oracle subset]") and len(PR[nm]) > len(sub): POL[nm + " [oracle subset]"] = summ([x for x in PR[nm] if x["i"] in sub], nm + " [oracle subset]")
fs = [FL[i] for i in LAB_HO]; say(f"[floor on oracle subset] raw {np.mean([dif(x['raw']) for x in fs]):.3f} confirmed {np.mean([dif(x['cons']) for x in fs]):.3f}")
RES["policies"] = POL; save()

# ------------------------------------------------------------------ judge validation: 150 pairs, repeat, second judge, order swap, core-facts rubric
try:
    rs = np.random.default_rng(16); pool = {"floor": [], "abl": [], "pol": []}; seen = set()
    for a, b, src, v in core.PAIRS:
        if a == b or (a, b) in seen: continue
        seen.add((a, b)); pool["floor" if src == "floor" else "abl" if src.startswith("abl") else "pol"].append((a, b, v))
    NS = 3 if TINY else 50; sample = []
    for k in pool: idx = rs.permutation(len(pool[k]))[:NS]; sample += [(k,) + pool[k][j] for j in idx]
    small = sample[::5]
    need = len(sample) * 0.0016 + 2 * len(small) * 0.0016 + len(sample) * 0.0003
    say(f"[validation] pool sizes { {k: len(v) for k, v in pool.items()} }; sample {len(sample)} (swap/core-rubric on {len(small)}); est ${need:.2f}, left ${left():.2f}")
    if left() < need and not RERUN: sample = [sample[j] for j in rs.permutation(len(sample))[: max(30, int((left() - 0.05) / 0.0035))]]; small = sample[::5]; say(f"[validation] shrunk to {len(sample)} pairs for budget")
    v1 = [t[3] for t in sample]
    v1b = pmap(lambda t: J(t[1], t[2], rep=1), sample); v_mini = pmap(lambda t: J(t[1], t[2], model=MINI), sample); v_mini2 = pmap(lambda t: J(t[1], t[2], model=MINI, rep=1), sample)
    v_sw = pmap(lambda t: J(t[1], t[2], swap=True), small); v_core = pmap(lambda t: J(t[1], t[2], system=JSYS_CORE), small)
    b = lambda xs: [dif(x) for x in xs]
    def kap(x, y): return float(cohen_kappa_score(b(x), b(y)))
    agr = lambda x, y: float(np.mean([dif(p) == dif(q) for p, q in zip(x, y)]))
    def flips(x, y): d = [dif(q) for p, q in zip(x, y) if dif(p)]; s = [dif(q) for p, q in zip(x, y) if not dif(p)]; return dict(first_D_n=len(d), D_stays_D=float(np.mean(d)) if d else None, first_S_n=len(s), S_becomes_D=float(np.mean(s)) if s else None)
    sm = lambda xs: [xs[k] for k in range(0, len(sample), 5)]
    V = dict(n=len(sample), sonnet_diff_rate=float(np.mean(b(v1))), mini_diff_rate=float(np.mean(b(v_mini))),
             kappa_sonnet_vs_mini=kap(v1, v_mini), agree_sonnet_vs_mini=agr(v1, v_mini), sonnet_self_agree=agr(v1, v1b), sonnet_self_kappa=kap(v1, v1b), sonnet_self_flips=flips(v1, v1b),
             mini_self_agree=agr(v_mini, v_mini2), mini_self_kappa=kap(v_mini, v_mini2), kappa_sonnet_rep_vs_mini=kap(v1b, v_mini),
             n_small=len(small), sonnet_diff_rate_small=float(np.mean(b(sm(v1)))), sonnet_swap_agree=agr(sm(v1), v_sw), sonnet_swap_kappa=kap(sm(v1), v_sw), core_diff_rate_small=float(np.mean(b(v_core))),
             agree_strict_vs_core_rubric=agr(sm(v1), v_core), kappa_strict_vs_core_rubric=kap(sm(v1), v_core))
    for src in pool:
        ix = [k for k, t in enumerate(sample) if t[0] == src]
        if ix: V["by_" + src] = dict(n=len(ix), sonnet_diff=float(np.mean([dif(v1[k]) for k in ix])), self_agree=float(np.mean([dif(v1[k]) == dif(v1b[k]) for k in ix])), mini_agree=float(np.mean([dif(v1[k]) == dif(v_mini[k]) for k in ix])), mini_diff=float(np.mean([dif(v_mini[k]) for k in ix])))
    cm = Counter((("D" if dif(p) else "S"), ("D" if dif(q) else "S")) for p, q in zip(v1, v_mini)); V["confusion_sonnet_mini"] = {f"sonnet {a} / mini {c}": n for (a, c), n in sorted(cm.items())}
    V["unparsed"] = sum(1 for x in v1 + v1b + v_mini + v_mini2 + v_sw + v_core if x == "FAIL")
    for k, v in V.items(): say(f"[judge] {k}: {v if not isinstance(v, float) else round(v, 3)}")
    RES["judge"] = V; RES["validation_sample"] = [dict(src=t[0], a=t[1], b=t[2], v1=t[3], v1b=x, mini=y) for t, x, y in zip(sample, v1b, v_mini)]; save(); spend("validation")
except Exception as e: say(f"[FAILED] judge validation: {type(e).__name__} {str(e)[:300]}")

# ------------------------------------------------------------------ calibration + certificate for the main learned scorer; pooled in-sample bound; n needed
def safe(f, *a):
    try: return f(*a)
    except Exception as e: say(f"[FAILED] {f.__name__}: {type(e).__name__} {str(e)[:300]}")
def n_needed(rate, budget, delta=0.05 / 40):
    for n in range(20, 20001, 10):
        if cp_upper(int(round(rate * n)), n, delta) <= budget: return n
    return None
def certificate(name, keepfn, conffn, cal_ids):
    cal = eval_policy(name + " cal", keepfn, cal_ids, extra=lambda i: dict(conf=conffn(i)))
    spend("calibration " + name); held = PR[name]; out = {}
    csum = summ(cal, name + " on calibration (train OOF" + (" + unlabelled train)" if len(cal_ids) > len(LAB_TR) else ")"))
    for metric in ("raw", "cons"):
        cc = np.array([x["conf"] for x in cal]); ok = np.array([not dif(x[metric]) for x in cal]); hc = np.array([x["conf"] for x in held]); hok = np.array([not dif(x[metric]) for x in held])
        pc = np.r_[cc, hc]; pok = np.r_[ok, hok]; ptok_s = np.array([TOK[x["i"]] - x["tok"] for x in cal + held]); ptok_f = np.array([TOK[x["i"]] for x in cal + held])
        for budget in (0.05, 0.10, 0.15, 0.20):
            c = certify(cc, ok, budget); r = dict(cutoff=c["cutoff"], cal_n=len(cal), cal_coverage=c["coverage"], cal_emp_error=c["emp_error"], cal_bound=c["bound"])
            if np.isfinite(c["cutoff"]):
                h = apply_certificate(hc, hok, c["cutoff"]); cov = hc >= c["cutoff"]
                r.update(test_coverage=h["coverage"], test_error=h["error"], test_n_above=h["n_above"], tokens_saved_test=sum(TOK[x["i"]] - x["tok"] for x, m in zip(held, cov) if m) / sum(TOK[x["i"]] for x in held))
            else: r.update(test_coverage=0.0, test_error=None, test_n_above=0, tokens_saved_test=0.0)
            pcert = certify(pc, pok, budget)   # in-sample: calibration + held-out as ONE calibration set (what these rows could certify at best)
            r["pooled"] = dict(n=len(pc), cutoff=pcert["cutoff"], coverage=pcert["coverage"], emp_error=pcert["emp_error"], bound=pcert["bound"],
                               tokens_saved=float(ptok_s[pc >= pcert["cutoff"]].sum() / ptok_f.sum()) if np.isfinite(pcert["cutoff"]) else 0.0)
            out[f"{metric}@{int(budget*100)}%"] = r
            say(f"[certificate {name}] metric={metric:4s} budget {budget:.2f}: cal n={len(cal)} cutoff {r['cutoff']:.3f} cov {r['cal_coverage']:.3f} err {r['cal_emp_error']} bound {r['cal_bound']} | held-out coverage {r['test_coverage']:.3f} error {r['test_error']} (n above {r['test_n_above']}) "
                f"input tokens saved {r['tokens_saved_test']*100:.1f}% | pooled n={len(pc)} cov {pcert['coverage']:.3f} err {pcert['emp_error']} bound {pcert['bound']} tokens saved {r['pooled']['tokens_saved']*100:.1f}%")
        e = float(np.mean(~pok)); top = pc >= np.median(pc[pc > -1]) if (pc > -1).any() else pc > -1
        out[f"{metric}_pooled_summary"] = dict(n=len(pc), overall_dis=e, pruned_share=float(np.mean(pc > -1)), top_half_conf_dis=float(np.mean(~pok[top])) if top.any() else None, n_needed_5=n_needed(e, 0.05), n_needed_10=n_needed(e, 0.10),
                                               floor_rate=RES["floor"][metric])
        say(f"[calibration {name}] pooled {metric} disagreement {e:.3f} on n={len(pc)} (floor on held-out {RES['floor'][metric]:.3f}); calibration rows needed to certify 5%/10% at that rate with zero slack: {out[f'{metric}_pooled_summary']['n_needed_5']}/{out[f'{metric}_pooled_summary']['n_needed_10']}")
    return dict(cal_summary=csum, cert=out)
RES["cert"] = {}
if LR_ in PR and can(len(LAB_TR)): RES["cert"][LR_] = safe(certificate, LR_, lambda i: keep_prob(i, "reason"), lambda i: conf_prob(i, "reason"), LAB_TR); save()
# ------------------------------------------------------------------ extras, only with leftover budget
if LR_ in PR and can(len(UNL[:40]), 0.03) and not TINY: RES["cert"][LR_ + " +40 unlabelled cal"] = safe(certificate, LR_, lambda i: keep_prob(i, "reason"), lambda i: conf_prob(i, "reason"), LAB_TR + UNL[:40]); save()
# reason-gated policy: the flag reason (read from the prompt) says whether history can matter; keys and rates come from LAB_TR labels only
RK = ["dup", "first", "high", "noinv"]
def rkey(i): return "+".join(n for n, x in zip(RK, ctx_of(i)[:4]) if x) or "none"
GK = {}
for k in sorted({rkey(i) for i in LAB_TR}):
    ids = [i for i in LAB_TR if rkey(i) == k]; h = sum(dif(labs[i]["var"]["H0"]["cons"]) for i in ids); bk = sum(SL[i]["fixed"]["bank"] for i in ids); ho = [i for i in LAB_HO if rkey(i) == k]
    GK[k] = dict(n=len(ids), h0=h, bank=bk, drop_hist=len(ids) >= 5 and h / len(ids) <= 0.15, drop_bank=len(ids) >= 5 and bk / len(ids) <= 0.15, conf=1 - (h + 1) / (len(ids) + 2),
                 heldout_n=len(ho), heldout_h0=sum(dif(labs[i]["var"]["H0"]["cons"]) for i in ho), heldout_bank=sum(SL[i]["fixed"]["bank"] for i in ho))
    say(f"[gate key] {k:11s} train n={GK[k]['n']:2d} history needed {h}/{len(ids)} bank needed {bk}/{len(ids)} -> drop history {GK[k]['drop_hist']} drop bank {GK[k]['drop_bank']} | held-out labelled n={len(ho)} history needed {GK[k]['heldout_h0']} bank needed {GK[k]['heldout_bank']}")
def keep_gate(i):
    k = keep_all(P[i]); g = GK.get(rkey(i))
    if g and g["drop_hist"]: k["hist"] = (); k["bank"] = 0 if g["drop_bank"] else 1
    return k
def conf_gate(i): g = GK.get(rkey(i)); return g["conf"] if g and g["drop_hist"] else -1.0
GATE = "reason-gated"
RES["gate_keys"] = GK
run_policy(GATE, keep_gate, HELD, conf_gate, 0.0, 0.0)
if GATE in PR and can(len(LAB_TR + UNL[:40])): RES["cert"][GATE] = safe(certificate, GATE, keep_gate, conf_gate, UNL[:40] if TINY else LAB_TR + UNL[:40]); save()
for nm, kf, ids, cf in (("oracle+ (spread->all)", lambda i: keep_oracle(i, True), LAB_HO, None), ("recent-3", lambda i: keep_recent(i, 3), HELD, None), ("learned+reason B", lambda i: keep_prob(i, "reason97"), HELD, lambda i: conf_prob(i, "reason97"))):
    run_policy(nm, kf, ids, cf, 0.0, 0.0)
RES["cost"] = dict(ledger_total=llm.SPENT[0], new_calls_this_run=llm.NEW[0], seconds=time.time() - T0); save()
spend("end"); say(f"providers seen: {Counter((r['resp'].get('provider'), r['model']) for r in llm.R.values()).most_common()}")
