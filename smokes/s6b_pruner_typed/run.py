"""S6b context pruner, second pass: typed yes/no output, pinned low-noise teacher (haiku-5.5, reasoning off), majority-of-3 labels.
usage: python run.py   (resumable: every raw response is appended to responses.jsonl and re-read on start)
ponytail: one file; responses.jsonl is cache + spend ledger."""
import sys, re, json, time, threading, string
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, r"C:\Users\<user>\Desktop\personalprojects\Dreaming\smokes")
from common import embed, cp_upper, certify
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import openai

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
MODEL, CAP, SEED, NQ, NTE, KS, REPS = "anthropic/claude-haiku-5.5", 4.0, 0, 200, 80, (2, 3, 4), 3
SYSTEM = "Answer the question using only the context. Reply with exactly one word: yes or no."
KEY = re.search(r"OPENROUTERKEY\s*=\s*(\S+)", (ROOT / ".env").read_text()).group(1).strip("\"'")
client = openai.OpenAI(api_key=KEY, base_url="https://openrouter.ai/api/v1", max_retries=0, timeout=90)
RESP, LOG, lock, T0 = HERE / "responses.jsonl", HERE / "log.txt", threading.Lock(), time.time()
R = {}
if RESP.exists():
    for l in RESP.open(encoding="utf-8"): r = json.loads(l); R[r["key"]] = r
SPENT = [sum(r["resp"]["usage"].get("cost") or 0 for r in R.values())]
PIN, NCALL = [None], [0]

def say(s=""):
    print(s, flush=True)
    with lock, LOG.open("a", encoding="utf-8") as f: f.write(s + "\n")

def call(key, user):
    if key in R: return R[key]
    if SPENT[0] > CAP: raise RuntimeError("spend cap hit")
    eb = {"reasoning": {"enabled": False}, "usage": {"include": True}}
    if PIN[0]: eb["provider"] = {"order": [PIN[0]], "allow_fallbacks": False}
    for a in range(6):
        try:
            t = time.time()
            d = client.chat.completions.create(model=MODEL, max_tokens=5, extra_body=eb,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]).model_dump()
            d["_ms"] = (time.time() - t) * 1000; d["_pin"] = PIN[0]
            break
        except (openai.BadRequestError, openai.NotFoundError): raise
        except Exception:
            if a == 5: raise
            time.sleep(2 ** a)
    rec = dict(key=key, resp=d)
    with lock:
        R[key] = rec; SPENT[0] += d["usage"].get("cost") or 0; NCALL[0] += 1
        with RESP.open("a", encoding="utf-8") as f: f.write(json.dumps(rec) + "\n")
        if NCALL[0] % 500 == 0: print(f"[calls] {NCALL[0]} new, spent ${SPENT[0]:.3f}, {time.time()-T0:.0f}s", flush=True)
    return rec

def label(rec):  # typed output: yes/no, anything else = parse failure
    s = (rec["resp"]["choices"][0]["message"]["content"] or "").strip().lower().strip(string.punctuation + " \n")
    return s if s in ("yes", "no") else "fail"

def maj(ls):  # majority of 3; three-way split (yes/no/fail) -> fail
    top, n = Counter(ls).most_common(1)[0]
    return top if n >= 2 else "fail"

# ---------- data: HotpotQA distractor val, comparison questions with yes/no gold ----------
import datasets
ds = datasets.load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
types, answers = ds["type"], ds["answer"]
cand = [i for i in range(len(ds)) if types[i] == "comparison" and answers[i].lower() in ("yes", "no")]
say(f"[data] {len(cand)} comparison yes/no questions in validation")
Q = []
for i in np.random.default_rng(SEED).permutation(cand):
    x = ds[int(i)]; Tl = x["context"]["title"]
    gold = np.array([t in set(x["supporting_facts"]["title"]) for t in Tl])
    if len(Tl) != 10 or gold.sum() != 2: continue  # keep the 10-paragraph / 2-gold protocol
    Q.append(dict(id=x["id"], q=x["question"], a=x["answer"].lower(), titles=Tl, gold=gold,
                  paras=[f"{t}: {' '.join(s.strip() for s in ss)}" for t, ss in zip(Tl, x["context"]["sentences"])]))
    if len(Q) == NQ: break
ALL = list(range(10))
def user(q, kept): return "Context:\n" + "\n\n".join(f"[{n+1}] {q['paras'][i]}" for n, i in enumerate(kept)) + f"\n\nQuestion: {q['q']}"
kk = lambda q, kept, rep: f"{q['id']}|{','.join(map(str, kept))}|{rep}"
def run(jobs):  # jobs: [(q, kept, rep)] ; dedup by key
    jobs = list({kk(*j): j for j in jobs}.values())
    with ThreadPoolExecutor(8) as ex: list(ex.map(lambda j: call(kk(*j), user(j[0], j[1])), jobs))
labs = lambda q, kept, reps=range(REPS): [label(R[kk(q, kept, r)]) for r in reps]
usage_pt = lambda q, kept: R[kk(q, kept, 0)]["resp"]["usage"]["prompt_tokens"]
say(f"[data] sampled {len(Q)} questions; yes share {np.mean([q['a']=='yes' for q in Q]):.3f}")

# ---------- probe + pin provider ----------
provs = [call(f"probe|{i}", user(Q[i], ALL))["resp"].get("provider") for i in range(3)]
say(f"[probe] 3 unpinned calls, provider field: {provs}")
for c in (provs[0], provs[0].lower().replace(" ", "-")):
    PIN[0] = c
    try:
        rec = call(f"pintest|{c}", user(Q[0], ALL)); ok = rec["resp"].get("provider") == provs[0]
        say(f"[pin] order=[{c!r}] allow_fallbacks=false -> served by {rec['resp'].get('provider')!r}")
        if ok: break
    except Exception as e: say(f"[pin] order=[{c!r}] failed: {type(e).__name__} {str(e)[:150]}")
else: raise SystemExit("could not pin provider")

# ---------- pilot cost projection on 5 questions, shrink to 150 if > $3 ----------
abl = lambda i: [j for j in ALL if j != i]
run([(q, kept, r) for q in Q[:5] for kept in [ALL] + [abl(i) for i in ALL] for r in range(REPS)])
pc = [R[kk(q, kept, r)]["resp"]["usage"]["cost"] for q in Q[:5] for kept in [ALL] + [abl(i) for i in ALL] for r in range(REPS)]
proj = np.mean(pc) * (NQ * (11 * REPS + REPS) + 3200)  # labels + 3 extra full reps + ~3.2k pruned re-asks (smaller prompts, so an overestimate)
say(f"[pilot] {len(pc)} calls, ${np.mean(pc)*1e3:.4f} per 1k calls -> projected total ${proj:.2f} for NQ={NQ}")
if proj > 3.0: NQ = 150; Q = Q[:NQ]; say(f"[pilot] over $3, shrinking to NQ={NQ}")
TR, TE = Q[:NQ - NTE], Q[NQ - NTE:]
say(f"[split] NQ={NQ} train={len(TR)} heldout={len(TE)}")

# ---------- ablation labels ----------
run([(q, ALL, r) for q in Q for r in range(2 * REPS)])           # full context: reps 0-2 = A_full, reps 3-5 = independent 2nd majority
run([(q, abl(i), r) for q in Q for i in ALL for r in range(REPS)])
say(f"[labels] done, spent ${SPENT[0]:.3f}, {time.time()-T0:.0f}s")
A = [maj(labs(q, ALL)) for q in Q]; A2 = [maj(labs(q, ALL, range(3, 6))) for q in Q]
Y = np.array([[maj(labs(q, abl(i))) != a for i in ALL] for q, a in zip(Q, A)])
G = np.array([q["gold"] for q in Q]); gans = [q["a"] for q in Q]
recs = [R[kk(q, kept, r)] for q in Q for kept in [ALL] + [abl(i) for i in ALL] for r in range(REPS)] + [R[kk(q, ALL, r)] for q in Q for r in range(3, 6)]
prov_ct = Counter(r["resp"].get("provider") for r in recs); rt = [r["resp"]["usage"]["completion_tokens_details"]["reasoning_tokens"] or 0 for r in recs]
say(f"[verify] pinned-phase provider counts {dict(prov_ct)}; reasoning tokens max {max(rt)} sum {sum(rt)}; finish reasons {dict(Counter(r['resp']['choices'][0]['finish_reason'] for r in recs))}")
fails = sum(label(r) == "fail" for r in recs); say(f"[parse] failures {fails}/{len(recs)}; maj label 'fail': full {sum(a=='fail' for a in A)}, ablation {int(sum(maj(labs(q, abl(i))) == 'fail' for q in Q for i in ALL))}")
# self-agreement of the teacher on identical full prompts (6 calls per question: reps 0-5)
six = [labs(q, ALL, range(6)) for q in Q]
pair3 = np.mean([np.mean([a != b for a, b in [(l[0], l[1]), (l[0], l[2]), (l[1], l[2])]]) for l in six])
pair6 = np.mean([np.mean([a != b for x, a in enumerate(l) for b in l[x + 1:]]) for l in six])
unan3 = np.mean([len(set(l[:3])) == 1 for l in six]); unan6 = np.mean([len(set(l)) == 1 for l in six])
floor = np.mean([a != b for a, b in zip(A, A2)])
say(f"[noise] full prompt, 3 calls: all-identical {unan3:.3f}, pairwise disagreement {pair3:.4f}; 6 calls: all-identical {unan6:.3f}, pairwise {pair6:.4f}; maj3 vs independent maj3 disagree {floor:.4f} (n={NQ})")
acc_full = np.mean([a == g for a, g in zip(A, gans)])
say(f"[labels] A_full accuracy vs gold {acc_full:.3f}; A_full yes share {np.mean([a=='yes' for a in A]):.3f}; matters: all {Y.mean():.3f}, gold {Y[G].mean():.3f}, non-gold {Y[~G].mean():.3f}, mean/question {Y.sum(1).mean():.2f}, none-matter questions {np.mean(Y.sum(1)==0):.3f}")
# single-call flip rate for ablations (what the old single-call labels would have shown)
sc = np.array([[labs(q, abl(i), [0])[0] != labs(q, ALL, [0])[0] for i in ALL] for q in Q])
say(f"[labels] single-call (rep 0) matters: all {sc.mean():.3f}, gold {sc[G].mean():.3f}, non-gold {sc[~G].mean():.3f}")

Yr = [np.array([[labs(q, abl(i), [r])[0] != a for i in ALL] for q, a in zip(Q, A)]) for r in range(REPS)]  # flips vs A_full (maj3)
pairs = [(0, 1), (0, 2), (1, 2)]
jac = float(np.mean([(Yr[a] & Yr[b]).sum() / max(1, (Yr[a] | Yr[b]).sum()) for a, b in pairs]))
ceil_pool = float(np.mean([roc_auc_score(Yr[b].ravel(), Yr[a].ravel()) for a, b in pairs + [(b, a) for a, b in pairs]]))
p_flip_pos, p_flip_neg = float(np.mean([Yr[r][Y].mean() for r in range(REPS)])), float(np.mean([Yr[r][~Y].mean() for r in range(REPS)]))
say(f"[reliability] single-call ablation flips vs A_full: pairwise Jaccard of positive sets {jac:.3f}; AUC of one rep's flips as a 'scorer' for another rep's flips (label-noise ceiling) {ceil_pool:.3f}; P(single flip | maj matters) {p_flip_pos:.3f}, P(single flip | maj does not) {p_flip_neg:.4f}")
nfl = (Y.sum(1) > 0); gans_a = np.array(gans); afull_ok = np.array([a == g for a, g in zip(A, gans)])
say(f"[labels] questions with any matter-paragraph: {nfl.sum()}/{NQ}; A_full right on them {afull_ok[nfl].mean():.3f} vs {afull_ok[~nfl].mean():.3f} on the rest; matter-paragraphs on A_full-wrong questions {Y[~afull_ok].sum(1).mean():.2f}/q vs {Y[afull_ok].sum(1).mean():.2f}/q on right ones")

# ---------- scorers ----------
content = lambda s: {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in ENGLISH_STOP_WORDS}
clean_title = lambda t: re.sub(r"\s*\(.*\)$", "", t).lower()
E = embed([q["q"] for q in Q] + [p for q in Q for p in q["paras"]])
EQ, EP = E[:NQ], E[NQ:].reshape(NQ, 10, -1)
cos = np.einsum("qd,qpd->qp", EQ, EP)
def feats(qi):
    q = Q[qi]; qw = content(q["q"]); c = cos[qi]; rank = np.argsort(np.argsort(-c))
    return np.array([[c[i], i, len(q["paras"][i].split()), len(qw & content(q["paras"][i])) / max(1, len(qw)),
                      float(clean_title(q["titles"][i]) in q["q"].lower()), rank[i], c[i] - c.max()] for i in ALL])
F = np.stack([feats(qi) for qi in range(NQ)])
tr, te = np.arange(len(TR)), np.arange(len(TR), NQ)
def fit_pred(y):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(F[tr].reshape(-1, 7), y[tr].ravel())
    return m.predict_proba(F.reshape(-1, 7))[:, 1].reshape(NQ, 10), m
S = {}
S["learned"], m = fit_pred(Y)
say("[learned] standardized coefs " + ", ".join(f"{n} {c:+.3f}" for n, c in zip("cos pos ntok qover titleinq cosrank cosminusmax".split(), m[-1].coef_[0])))
S["learned_gold (extra)"], _ = fit_pred(G)
XF = HERE / "xenc_scores.json"
if XF.exists() and len(json.loads(XF.read_text())) == NQ: X = np.array(json.loads(XF.read_text()))
else:
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    X = ce.predict([(q["q"], p) for q in Q for p in q["paras"]], batch_size=64, show_progress_bar=False).reshape(NQ, 10)
    XF.write_text(json.dumps(X.tolist()))
S["cross-encoder"] = X
rng = np.random.default_rng(SEED)
S["oracle"] = G + 0.01 * rng.random((NQ, 10))   # gold first, remaining slots random
S["random"] = rng.random((NQ, 10))
say(f"[scorers] done {time.time()-T0:.0f}s")

# ---------- AUC (held-out) ----------
def auc(s, y):
    pooled = roc_auc_score(y.ravel(), s.ravel())
    w = [roc_auc_score(y[i], s[i]) for i in range(len(y)) if 0 < y[i].sum() < 10]
    return pooled, float(np.mean(w)), len(w)
res = dict(NQ=NQ, noise=dict(unan3=unan3, pair3=pair3, unan6=unan6, pair6=pair6, floor=floor), acc_full=acc_full)
res["auc"] = {n: dict(zip(("abl_pooled", "abl_within", "abl_nq", "gold_pooled", "gold_within", "gold_nq"), auc(s[te], Y[te]) + auc(s[te], G[te]))) for n, s in S.items()}
brng = np.random.default_rng(1); B = [brng.integers(0, len(te), len(te)) for _ in range(1000)]
for n, s_ in S.items():
    bs = [roc_auc_score(Y[te][b].ravel(), s_[te][b].ravel()) for b in B if 0 < Y[te][b].sum() < Y[te][b].size]
    res["auc"][n]["abl_pooled_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
bd = [roc_auc_score(Y[te][b].ravel(), S["learned"][te][b].ravel()) - roc_auc_score(Y[te][b].ravel(), S["cross-encoder"][te][b].ravel()) for b in B]
res["auc_diff_learned_minus_xenc_pooled_ci"] = [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]
res["reliability"] = dict(jaccard=jac, noise_ceiling_auc=ceil_pool, p_flip_given_pos=p_flip_pos, p_flip_given_neg=p_flip_neg)
say(f"[auc] bootstrap 95% CI of learned minus cross-encoder pooled ablation AUC: {res['auc_diff_learned_minus_xenc_pooled_ci']}")
for n, a in res["auc"].items(): say(f"[auc] {n:22s} ablation pooled {a['abl_pooled']:.3f} CI95 [{a['abl_pooled_ci'][0]:.3f}, {a['abl_pooled_ci'][1]:.3f}] within {a['abl_within']:.3f} (n_q={a['abl_nq']}) | gold pooled {a['gold_pooled']:.3f} within {a['gold_within']:.3f}")
say(f"[split-stats] train positives {int(Y[tr].sum())}/{Y[tr].size}, heldout positives {int(Y[te].sum())}/{Y[te].size}; heldout A_full accuracy {np.mean([A[i]==gans[i] for i in te]):.3f} (wrong on {sum(A[i]!=gans[i] for i in te)}/{len(te)}); heldout yes share of A_full {np.mean([A[i]=='yes' for i in te]):.3f}; mean full prompt tokens {np.mean([usage_pt(q, ALL) for q in Q]):.0f}")
say(f"[auc] heldout matters rate {Y[te].mean():.3f}, gold {Y[te][G[te]].mean():.3f}, non-gold {Y[te][~G[te]].mean():.3f}")

# ---------- pruned re-asks on held-out ----------
kept_of = lambda n, qi, k: sorted(np.argsort(-S[n][qi], kind="stable")[:k].tolist())
run([(Q[qi], kept_of(n, qi, k), r) for n in S for k in KS for qi in te for r in range(REPS)])
allrec = [r for k_, r in R.items() if not k_.startswith("probe")]
say(f"[verify-all] non-probe calls {len(allrec)}: providers {dict(Counter(r['resp'].get('provider') for r in allrec))}; reasoning tokens sum {sum(r['resp']['usage']['completion_tokens_details']['reasoning_tokens'] or 0 for r in allrec)}; probe providers {dict(Counter(r['resp'].get('provider') for k_, r in R.items() if k_.startswith('probe')))}")
say(f"[eval] pruned re-asks done, spent ${SPENT[0]:.3f}, {time.time()-T0:.0f}s")
full_pt = sum(usage_pt(Q[qi], ALL) for qi in te)
def row(name, k, kl, ls):  # kl: kept lists per held-out question, ls: 3 labels per question
    pred = [maj(l) for l in ls]; n = len(te); err = int(sum(p != A[qi] for p, qi in zip(pred, te)))
    return dict(scorer=name, k=k, agree=1 - err / n, errors=err, ub=cp_upper(err, n, 0.05),
                percall=float(np.mean([np.mean([x == A[qi] for x in l]) for l, qi in zip(ls, te)])),
                acc=float(np.mean([p == gans[qi] for p, qi in zip(pred, te)])),
                tok_removed=1 - sum(usage_pt(Q[qi], kp) for qi, kp in zip(te, kl)) / full_pt,
                gold_kept=float(np.mean([G[qi][kp].sum() == 2 for qi, kp in zip(te, kl)])), fails=int(sum(p == "fail" for p in pred)),
                err_afull_wrong=int(sum(p != A[qi] and A[qi] != gans[qi] for p, qi in zip(pred, te))),
                err_gold_lost=int(sum(p != A[qi] and G[qi][kp].sum() < 2 for p, qi, kp in zip(pred, te, kl))),
                err_pruned_right=int(sum(p != A[qi] and p == gans[qi] for p, qi in zip(pred, te))),
                correct=[p == A[qi] for p, qi in zip(pred, te)])
rows = [row("full context, 2nd independent majority", 10, [ALL] * len(te), [labs(Q[qi], ALL, range(3, 6)) for qi in te])]
for n in S:
    for k in KS:
        kl = [kept_of(n, qi, k) for qi in te]
        rows.append(row(n, k, kl, [labs(Q[qi], kp) for qi, kp in zip(te, kl)]))
        rows[-1]["conf"] = [float(np.sort(S[n][qi])[::-1][k - 1] - np.sort(S[n][qi])[::-1][k]) for qi in te]  # score gap k-th vs (k+1)-th, SGR confidence
def need_n(p, b):  # smallest n whose CP upper bound at observed rate p is <= b
    if p >= b: return None
    for n in range(10, 20001):
        if cp_upper(int(round(p * n)), n, 0.05) <= b: return n
BUD = (0.05, 0.10, 0.15)
for r in rows:
    r["cert_all"] = {b: r["ub"] <= b for b in BUD}; r["need_n"] = {b: need_n(1 - r["agree"], b) for b in BUD}
    if "conf" in r and r["scorer"] not in ("oracle", "random"):
        c = {b: certify(r["conf"], r["correct"], b) for b in BUD}
        r["sgr_cov"] = {b: c[b]["coverage"] for b in BUD}; r["sgr_eff_tok"] = {b: c[b]["coverage"] * r["tok_removed"] for b in BUD}; r["sgr_emp"] = {b: c[b]["emp_error"] for b in BUD}; r["sgr_err"] = {b: c[b]["emp_error"] for b in BUD}
res["rows"] = [{k: v for k, v in r.items() if k not in ("conf", "correct")} for r in rows]
res["labels"] = dict(matters=float(Y.mean()), gold=float(Y[G].mean()), nongold=float(Y[~G].mean()), perq=float(Y.sum(1).mean()), none=float(np.mean(Y.sum(1) == 0)),
                     single_call=dict(all=float(sc.mean()), gold=float(sc[G].mean()), nongold=float(sc[~G].mean())),
                     heldout=dict(all=float(Y[te].mean()), gold=float(Y[te][G[te]].mean()), nongold=float(Y[te][~G[te]].mean())))
res["spent"] = SPENT[0]; res["prov"] = dict(prov_ct); res["pin"] = PIN[0]
(HERE / "results.json").write_text(json.dumps(res, indent=1, default=str))
f = lambda x: "-" if x is None else f"{x:.3f}"
say("\n| scorer | k | agree (maj3) | disagree n/80 | UB95 | per-call agree | acc gold | tokens removed | both gold kept | parse fails |\n|---|---|---|---|---|---|---|---|---|---|")
for r in res["rows"]: say(f"| {r['scorer']} | {r['k']} | {r['agree']:.3f} | {r['errors']}/{len(te)} | {r['ub']:.3f} | {r['percall']:.3f} | {r['acc']:.3f} | {r['tok_removed']*100:.1f}% | {r['gold_kept']:.2f} | {r['fails']} |")
say("\n| scorer | k | whole-set certifiable 5/10/15% | n needed at observed rate 5/10/15% | SGR coverage 5/10/15% | SGR emp. error 5/10/15% | SGR effective tokens removed 5/10/15% |\n|---|---|---|---|---|---|---|---|")
for r in res["rows"]: say(f"| {r['scorer']} | {r['k']} | {'/'.join('Y' if r['cert_all'][b] else 'N' for b in BUD)} | {'/'.join(str(r['need_n'][b] or 'never') for b in BUD)} | {'/'.join(f(r['sgr_cov'][b]) for b in BUD) if 'sgr_cov' in r else '-'} | {'/'.join(f(r['sgr_emp'][b]) for b in BUD) if 'sgr_cov' in r else '-'} | {'/'.join(f(r['sgr_eff_tok'][b]) for b in BUD) if 'sgr_cov' in r else '-'} |")
say(f"\n[done] spent ${SPENT[0]:.4f}, {time.time()-T0:.0f}s")
