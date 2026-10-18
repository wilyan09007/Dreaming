"""S6 context pruner (mission 5): HotpotQA distractor. Ablation labels from gpt-5.4-nano, 4 paragraph scorers, pruned re-asks.
ponytail: one file, jsonl response cache doubles as resume + spend ledger."""
import sys, re, json, string, time, threading
sys.path.insert(0, r"C:\Users\<user>\Desktop\personalprojects\Dreaming\smokes")
from common import *
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
MODEL, PIN, POUT, CAP, SEED, NQ, KS = "openai/gpt-5.4-nano", 0.2, 1.25, 5.0, 0, 250, (2, 3, 4)
PROMPT = "Answer the question using only the context. Reply with the shortest answer span.\n\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer:"
t0 = time.time()

# ---------- LLM with jsonl cache + spend ledger ----------
from openai import OpenAI
key = next(l.split("=", 1)[1].strip().strip("'\"") for l in open(ROOT.parent / ".env") if l.startswith("OPENROUTERKEY"))
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, timeout=60)
RESP, lock, R = HERE / "responses.jsonl", threading.Lock(), {}
if RESP.exists():
    for l in RESP.open(encoding="utf-8"):
        r = json.loads(l); R[r["key"]] = r
cost = lambda recs: sum(r["pt"] * PIN + r["ct"] * POUT for r in recs) / 1e6

def ask(k, q, paras, seed=None):
    if k in R: return R[k]
    if cost(R.values()) > CAP: raise RuntimeError("spend cap hit")
    msg = PROMPT.format(ctx="\n\n".join(paras), q=q)
    for a in range(6):
        try:
            r = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": msg}], temperature=0,
                                               max_tokens=40, extra_body={"reasoning": {"enabled": False}}, **({} if seed is None else {"seed": seed}))
            u, d = r.usage, r.usage.completion_tokens_details
            rec = dict(key=k, model=MODEL, answer=(r.choices[0].message.content or "").strip(), pt=u.prompt_tokens,
                       ct=u.completion_tokens, rt=(d.reasoning_tokens or 0) if d else 0, fin=r.choices[0].finish_reason)
            break
        except Exception as e:
            if a == 5: raise
            time.sleep(2 ** a)
    with lock:
        R[k] = rec
        with RESP.open("a", encoding="utf-8") as f: f.write(json.dumps(rec) + "\n")
    return rec

def run(jobs):  # jobs: [(key, question_dict, kept_idx)]
    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(lambda j: ask(j[0], j[1]["q"], [j[1]["paras"][i] for i in j[2]]), jobs))

kkey = lambda q, kept: f"{q['id']}|{','.join(map(str, kept))}"
ALL = list(range(10))
full_jobs = lambda qs: [(kkey(q, ALL), q, ALL) for q in qs]
abl_jobs = lambda qs: [(kkey(q, [j for j in ALL if j != i]), q, [j for j in ALL if j != i]) for q in qs for i in ALL]
rep_jobs = lambda qs: [(q["id"] + "|full_rep", q, ALL) for q in qs]
ans = lambda q, kept: R[kkey(q, kept)]["answer"]

# ---------- data ----------
def norm(s):  # SQuAD normalization
    s = "".join(c for c in s.lower() if c not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())

def f1(p, g):
    p, g = norm(p).split(), norm(g).split()
    c = sum((Counter(p) & Counter(g)).values())
    if not p or not g: return float(p == g)
    return 0.0 if c == 0 else 2 * (c / len(p)) * (c / len(g)) / ((c / len(p)) + (c / len(g)))

def same(a, b):  # relaxed equivalence: normalized word-containment either way, or token-F1 >= 0.5
    a, b = norm(a), norm(b)
    return a == b or f" {a} " in f" {b} " or f" {b} " in f" {a} " or f1(a, b) >= 0.5
has = lambda ans_, gold: f" {norm(gold)} " in f" {norm(ans_)} "

import datasets
ds = datasets.load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
Q, skipped = [], 0
for i in np.random.default_rng(SEED).permutation(len(ds)):
    x = ds[int(i)]; T = x["context"]["title"]
    gold = np.array([t in set(x["supporting_facts"]["title"]) for t in T])
    if len(T) != 10 or gold.sum() != 2: skipped += 1; continue  # a few val items have <10 paragraphs; keep the 10-paragraph protocol
    Q.append(dict(id=x["id"], q=x["question"], a=x["answer"], titles=T, type=x["type"], gold=gold,
                  paras=[f"{t}: {' '.join(s.strip() for s in ss)}" for t, ss in zip(T, x["context"]["sentences"])]))
    if len(Q) == NQ: break
print(f"[data] {len(Q)} questions, skipped {skipped} non-10-paragraph items while sampling")

# ---------- cost probe on 5 questions, shrink N if projection > $3 ----------
run(full_jobs(Q[:5]) + abl_jobs(Q[:5]))
cq = cost([R[j[0]] for j in full_jobs(Q[:5]) + abl_jobs(Q[:5])]) / 5  # USD per question for 11 calls
proj = NQ * cq * (1 + 1 / 11) + 100 * 12 * (cq / 11) * 0.45
print(f"[probe] ${cq:.5f}/question (11 calls), mean full prompt tokens {np.mean([R[kkey(q, ALL)]['pt'] for q in Q[:5]]):.0f}, projected total ${proj:.2f}")
if proj > 3:
    NQ = int(NQ * 3 / proj); print(f"[probe] projection > $3, shrinking to N={NQ}")
Q = Q[:NQ]; NTR = round(NQ * 0.6); TR, TE = Q[:NTR], Q[NTR:]
print(f"[split] N={NQ} train={len(TR)} heldout={len(TE)}")

# ---------- ablation labels ----------
run(full_jobs(Q) + abl_jobs(Q) + rep_jobs(Q))
A = [R[kkey(q, ALL)]["answer"] for q in Q]
Y = np.array([[norm(ans(q, [j for j in ALL if j != i])) != norm(a) for i in ALL] for q, a in zip(Q, A)])  # (N,10) "matters"
G = np.array([q["gold"] for q in Q])
rep_dis = np.mean([norm(R[q["id"] + "|full_rep"]["answer"]) != norm(a) for q, a in zip(Q, A)])

Yr = np.array([[not same(ans(q, [j for j in ALL if j != i]), a) for i in ALL] for q, a in zip(Q, A)])  # relaxed "matters"
rep_dis_r = np.mean([not same(R[q["id"] + "|full_rep"]["answer"], a) for q, a in zip(Q, A)])

# determinism probe on 40 questions: 5 default-sampling replicas (incl. the 2 above) + 2 calls with seed=0
tags = [("full_rep2", None), ("full_rep3", None), ("full_rep4", None), ("seedA", 0), ("seedB", 0)]
with ThreadPoolExecutor(8) as ex:
    list(ex.map(lambda z: ask(f"{z[0]['id']}|{z[1][0]}", z[0]["q"], z[0]["paras"], z[1][1]), [(q, t) for q in Q[:40] for t in tags]))
g = lambda q, t: R[f"{q['id']}|{t}"]["answer"]
S5 = [[R[kkey(q, ALL)]["answer"], g(q, "full_rep"), g(q, "full_rep2"), g(q, "full_rep3"), g(q, "full_rep4")] for q in Q[:40]]
det = dict(n=40, mean_distinct_of5=float(np.mean([len({norm(x) for x in s5}) for s5 in S5])),
           all5_identical=float(np.mean([len({norm(x) for x in s5}) == 1 for s5 in S5])),
           all5_equivalent_relaxed=float(np.mean([all(same(x, s5[0]) for x in s5) for s5 in S5])),
           seed0_pair_identical=float(np.mean([norm(g(q, "seedA")) == norm(g(q, "seedB")) for q in Q[:40]])),
           seed0_pair_equivalent_relaxed=float(np.mean([same(g(q, "seedA"), g(q, "seedB")) for q in Q[:40]])))
print("[determinism]", det)
print(f"[spend after labels] ${cost(R.values()):.3f}, reasoning tokens total {sum(r['rt'] for r in R.values())}, "
      f"finish!=stop {sum(r['fin'] != 'stop' for r in R.values())}")

def pr(Yb, Gb):
    tp = (Yb & Gb).sum(); return tp / max(Yb.sum(), 1), tp / Gb.sum()
em_full = np.array([norm(a) == norm(q["a"]) for q, a in zip(Q, A)]); f1_full = np.mean([f1(a, q["a"]) for q, a in zip(Q, A)])
has_full = np.array([has(a, q["a"]) for q, a in zip(Q, A)])
sl = {"all": slice(None), "train": slice(0, NTR), "heldout": slice(NTR, None)}
labstats = lambda Yx: {s: dict(n=int(Yx[v].shape[0]), frac_matter=float(Yx[v].mean()), frac_matter_gold=float(Yx[v][G[v]].mean()),
               frac_matter_nongold=float(Yx[v][~G[v]].mean()), q_zero_matter=float((Yx[v].sum(1) == 0).mean()),
               precision=float(pr(Yx[v], G[v])[0]), recall=float(pr(Yx[v], G[v])[1]), mean_matter_per_q=float(Yx[v].sum(1).mean()))
               for s, v in sl.items()}
lab = dict(strict=labstats(Y), relaxed=labstats(Yr))
gold_acc = {s: dict(em=float(em_full[v].mean()), contains=float(has_full[v].mean())) for s, v in sl.items()}
by_type = {t: float(em_full[[q["type"] == t for q in Q]].mean()) for t in {q["type"] for q in Q}}

# ---------- scorers ----------
STOP = set("the a an of in on at to for and or is was are were who what which where when how did does do by with from as that this his her its their it be been has have had than both same".split())
toks = lambda s: re.findall(r"\w+", s.lower())
qE, pE = embed([q["q"] for q in Q]), embed([p for q in Q for p in q["paras"]]).reshape(NQ, 10, -1)
COS = (pE * qE[:, None]).sum(-1)

def feats(n, q):
    qt = {w for w in toks(q["q"]) if w not in STOP}; ql = q["q"].lower(); rows = []
    for i, (t, p) in enumerate(zip(q["titles"], q["paras"])):
        ts = re.sub(r"\s*\(.*?\)", "", t).lower(); tt = toks(ts)
        rows.append([COS[n, i], i, len(toks(p)), len(qt & set(toks(p))) / max(len(qt), 1),
                     float(ts in ql), sum(w in ql for w in tt) / max(len(tt), 1)])
    return np.array(rows)

X = np.stack([feats(n, q) for n, q in enumerate(Q)])  # (N,10,6)
def rel(X):  # extra variant: within-question relative features (pointwise LR can't see the other 9 paragraphs otherwise)
    r = X[:, :, [0, 3]]; rank = r.argsort(1).argsort(1)
    return np.concatenate([X, r - r.max(1, keepdims=True), rank], -1)
def fit(Xa):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
    m.fit(Xa[:NTR].reshape(-1, Xa.shape[-1]), Y[:NTR].ravel())
    return lambda v: m.predict_proba(v.reshape(-1, Xa.shape[-1]))[:, 1].reshape(-1, 10), m
Xr = rel(X)
f_l, m_l = fit(X); f_r, _ = fit(Xr)
from sentence_transformers import CrossEncoder
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
cef = CACHE / f"ce_{NQ}.npy"
if cef.exists(): CE = np.load(cef)
else:
    CE = np.array(ce.predict([(q["q"], p) for q in Q for p in q["paras"]], batch_size=64, show_progress_bar=False)).reshape(NQ, 10); np.save(cef, CE)
rng = np.random.default_rng(SEED)
RND, TIE = rng.random((NQ, 10)), rng.random((NQ, 10)) * 1e-3
SC = {"learned": f_l(X), "learned+rel (extra)": f_r(Xr), "cross-encoder": CE, "oracle (gold paras)": G + TIE, "random": RND}
coef = dict(zip(["cos", "position", "n_words", "q_overlap", "title_in_q", "title_tok_overlap"], m_l[-1].coef_[0].round(3).tolist()))

def auc(S, L, v):  # pooled AUC, and mean within-question AUC
    s, l = S[v], L[v]; ok = (l.sum(1) > 0) & (l.sum(1) < 10)
    return roc_auc_score(l.ravel(), s.ravel()), float(np.mean([roc_auc_score(l[n], s[n]) for n in np.where(ok)[0]]))
auc_tab = {n: {"heldout_vs_ablation": auc(S, Y, sl["heldout"]), "heldout_vs_gold": auc(S, G, sl["heldout"]),
               "train_vs_ablation": auc(S, Y, sl["train"]), "heldout_vs_relaxed": auc(S, Yr, sl["heldout"])} for n, S in SC.items()}

# ---------- pruned re-asks on held-out ----------
def kept(S, n, k): return sorted(np.argsort(-S[n])[:k].tolist())
jobs = [(kkey(q, kept(S, NTR + n, k)), q, kept(S, NTR + n, k)) for S in SC.values() for k in KS for n, q in enumerate(TE)]
run(jobs)
res = {}
for name, S in SC.items():
    for k in KS:
        ks = [kept(S, NTR + n, k) for n in range(len(TE))]
        rec = [R[kkey(q, kk)] for q, kk in zip(TE, ks)]; full = [R[kkey(q, ALL)] for q in TE]
        ag = np.array([norm(r["answer"]) == norm(A[NTR + n]) for n, r in enumerate(rec)])
        em = np.array([norm(r["answer"]) == norm(q["a"]) for r, q in zip(rec, TE)])
        both = np.array([G[NTR + n][kk].sum() == 2 for n, kk in enumerate(ks)])
        agr = np.array([same(r["answer"], A[NTR + n]) for n, r in enumerate(rec)]); hs = np.array([has(r["answer"], q["a"]) for r, q in zip(rec, TE)])
        res[f"{name}|{k}"] = dict(agree=float(ag.mean()), agree_relaxed=float(agr.mean()), em_gold=float(em.mean()), contains_gold=float(hs.mean()), tok_removed=1 - sum(r["pt"] for r in rec) / sum(r["pt"] for r in full),
                                  both_gold_kept=float(both.mean()), agree_given_both=float(ag[both].mean()) if both.any() else None,
                                  agree_given_not_both=float(ag[~both].mean()) if (~both).any() else None,
                                  disagree_ub95=cp_upper(int((~ag).sum()), len(ag), 0.05), disagree_relaxed_ub95=cp_upper(int((~agr).sum()), len(agr), 0.05))

# noise-floor row: re-ask the full context a second time on the same held-out questions
rr = [R[q["id"] + "|full_rep"] for q in TE]
ag = np.array([norm(r["answer"]) == norm(A[NTR + n]) for n, r in enumerate(rr)]); agr = np.array([same(r["answer"], A[NTR + n]) for n, r in enumerate(rr)])
res["full-context re-ask (noise floor)|10"] = dict(agree=float(ag.mean()), agree_relaxed=float(agr.mean()), em_gold=float(np.mean([norm(r["answer"]) == norm(q["a"]) for r, q in zip(rr, TE)])),
    contains_gold=float(np.mean([has(r["answer"], q["a"]) for r, q in zip(rr, TE)])), tok_removed=0.0, both_gold_kept=1.0, agree_given_both=float(ag.mean()), agree_given_not_both=None,
    disagree_ub95=cp_upper(int((~ag).sum()), len(ag), 0.05), disagree_relaxed_ub95=cp_upper(int((~agr).sum()), len(agr), 0.05))

# ---------- report ----------
spend = cost(R.values()); wall = time.time() - t0
out = dict(N=NQ, ntrain=NTR, nheld=len(TE), labels=lab, gold_acc_full=gold_acc, em_full_by_type=by_type, f1_full=float(f1_full), replicate_disagree_strict=float(rep_dis),
           replicate_disagree_relaxed=float(rep_dis_r), determinism=det, auc=auc_tab, coef=coef, results=res, spend_usd=spend, wall_s=wall, n_calls=len(R),
           mean_full_prompt_tokens=float(np.mean([R[kkey(q, ALL)]["pt"] for q in Q])), mean_full_answer_tokens=float(np.mean([R[kkey(q, ALL)]["ct"] for q in Q])),
           reasoning_tokens=sum(r["rt"] for r in R.values()), nonstop_finish=sum(r["fin"] != "stop" for r in R.values()))
(HERE / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k not in ("results", "auc")}, indent=1))
print("\n| scorer | AUC pooled (strict label) | within-q (strict) | pooled (relaxed) | within-q (relaxed) | AUC pooled (gold) | within-q (gold) | train within-q (strict) |\n|---|---|---|---|---|---|---|---|")
for n, a in auc_tab.items():
    h, r_, g_ = a["heldout_vs_ablation"], a["heldout_vs_relaxed"], a["heldout_vs_gold"]
    print(f"| {n} | {h[0]:.3f} | {h[1]:.3f} | {r_[0]:.3f} | {r_[1]:.3f} | {g_[0]:.3f} | {g_[1]:.3f} | {a['train_vs_ablation'][1]:.3f} |")
print("\n| scorer | k | agree strict | agree relaxed | EM gold | contains gold | tokens removed | both gold kept | strict agree if both kept | if not | 95% UB disagree strict | 95% UB disagree relaxed |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
for key_, r in res.items():
    s_, k = key_.split("|"); f = lambda v: "-" if v is None else f"{v:.2f}"
    print(f"| {s_} | {k} | {r['agree']:.2f} | {r['agree_relaxed']:.2f} | {r['em_gold']:.2f} | {r['contains_gold']:.2f} | {r['tok_removed']:.1%} | {r['both_gold_kept']:.2f} | {f(r['agree_given_both'])} | {f(r['agree_given_not_both'])} | {r['disagree_ub95']:.2f} | {r['disagree_relaxed_ub95']:.2f} |")
print(f"\nwall {wall:.0f}s; spend ${spend:.3f}")
