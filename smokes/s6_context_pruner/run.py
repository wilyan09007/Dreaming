"""S6 context pruner (mission 5), revised. HotpotQA distractor. Ablation labels from a teacher, scorers, pruned re-asks.
usage: python run.py <tag>   tag in TEACH. nano_t1 replays the original run; gpt-5.4-nano has NO endpoint with `temperature` on OpenRouter
(require_parameters + temperature=0 -> 404 for every reasoning setting), so its temperature=0 was silently dropped and it ran at the default temperature;
qwen_t0 / gemini_t0 send temperature=0 with provider.require_parameters=true, so a dropped parameter is a 404, not a silent T~1.
ponytail: one file; per-teacher jsonl response cache doubles as resume + spend ledger."""
import sys, re, json, string, time, threading, hashlib
sys.path.insert(0, r"C:\Users\<user>\Desktop\personalprojects\Dreaming\smokes")
from common import *
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest
import openai

HERE = Path(__file__).parent
TEACH = {  # t0: send provider.require_parameters (fail loudly if temperature=0 would be dropped)
    "nano_t1": dict(model="openai/gpt-5.4-nano", pin=0.2, pout=1.25, eb={"reasoning": {"enabled": False}}, t0=False, suffix="", file="responses.jsonl"),
    "qwen_t0": dict(model="qwen/qwen3-32b", pin=0.08, pout=0.28, eb={}, t0=True, suffix="\n/no_think", file="responses_qwen_t0.jsonl"),
    "gemini_t0": dict(model="google/gemini-3.5-flash-lite", pin=0.3, pout=2.5, eb={}, t0=True, suffix="", file="responses_gemini_t0.jsonl"),
}
TAG = sys.argv[1]; T = TEACH[TAG]; MODEL = T["model"]
JUDGE, JPIN, JPOUT = "google/gemini-3.5-flash-lite", 0.3, 2.5
CAP, SEED, NQ, KS = 2.5, 0, 250, (2, 3, 4)  # per-invocation cap; smoke cap $5 incl. the original $0.94
PROMPT = "Answer the question using only the context. Reply with the shortest answer span.\n\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer:"
PROMPT0 = "Answer the question. Reply with the shortest answer span.\n\nQuestion: {q}\nAnswer:"  # closed-book control
JPROMPT = ("You compare two answers to one question.\nQuestion: {q}\nAnswer A: {a}\nAnswer B: {b}\n\nSame answer? Extra words, punctuation and formatting do not matter. "
           "Different entities, values or claims, or 'unknown/none' versus a real answer, mean NO.\nOutput only the single word YES or NO, with no explanation.")
t0 = time.time()

# ---------- LLM with jsonl caches + spend ledger ----------
key = next(l.split("=", 1)[1].strip().strip("'\"") for l in open(ROOT.parent / ".env") if l.startswith("OPENROUTERKEY"))
client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, timeout=90)
lock = threading.Lock()
def load(f):
    d = {}
    if f.exists():
        for l in f.open(encoding="utf-8"): r = json.loads(l); d[r["key"]] = r
    return d
RESP, R = HERE / T["file"], load(HERE / T["file"])
JF = HERE / f"judge_{TAG}.jsonl"; J = load(JF)
cost = lambda recs, pi=T["pin"], po=T["pout"]: sum(r["pt"] * pi + r["ct"] * po for r in recs) / 1e6
spent = lambda: cost(R.values()) + cost(J.values(), JPIN, JPOUT)

def chat(model, msg, mt, eb):  # one call with retries; 400/404 (e.g. require_parameters unsatisfiable) raise immediately
    for a in range(6):
        try:
            return client.chat.completions.create(model=model, messages=[{"role": "user", "content": msg}], temperature=0, max_tokens=mt, extra_body=eb)
        except (openai.BadRequestError, openai.NotFoundError): raise
        except Exception:
            if a == 5: raise
            time.sleep(2 ** a)

def record(k, r, store, f):
    u, d = r.usage, r.usage.completion_tokens_details
    rec = dict(key=k, answer=(r.choices[0].message.content or "").strip(), pt=u.prompt_tokens, ct=u.completion_tokens,
               rt=getattr(d, "reasoning_tokens", None) if d else None, fin=r.choices[0].finish_reason, prov=(r.model_extra or {}).get("provider"))
    with lock:
        store[k] = rec
        with f.open("a", encoding="utf-8") as fh: fh.write(json.dumps(rec) + "\n")
    return rec

def ask(k, q, paras, seed=None, prompt=PROMPT):
    if k in R: return R[k]
    if spent() > CAP: raise RuntimeError("spend cap hit")
    eb = {**T["eb"], **({"provider": {"require_parameters": True}} if T["t0"] else {})}
    if seed is not None: eb["seed"] = seed
    return record(k, chat(MODEL, prompt.format(ctx="\n\n".join(paras), q=q) + T["suffix"], 40, eb), R, RESP)

def run(jobs):  # jobs: [(key, question_dict, kept_idx)]
    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(lambda j: ask(j[0], j[1]["q"], [j[1]["paras"][i] for i in j[2]]), jobs))

kkey = lambda q, kept: f"{q['id']}|{','.join(map(str, kept))}"
ALL = list(range(10))
full_jobs = lambda qs: [(kkey(q, ALL), q, ALL) for q in qs]
abl_jobs = lambda qs: [(kkey(q, [j for j in ALL if j != i]), q, [j for j in ALL if j != i]) for q in qs for i in ALL]
rep_jobs = lambda qs: [(q["id"] + "|full_rep", q, ALL) for q in qs]
ans = lambda q, kept: R[kkey(q, kept)]["answer"]

# ---------- answer equivalence ----------
def norm(s):  # SQuAD normalization
    s = "".join(c for c in s.lower() if c not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())

def f1(p, g):
    p, g = norm(p).split(), norm(g).split()
    c = sum((Counter(p) & Counter(g)).values())
    if not p or not g: return float(p == g)
    return 0.0 if c == 0 else 2 * (c / len(p)) * (c / len(g)) / ((c / len(p)) + (c / len(g)))

contain = lambda a, b: f" {norm(a)} " in f" {norm(b)} " or f" {norm(b)} " in f" {norm(a)} "
legacy = lambda a, b: norm(a) == norm(b) or contain(a, b) or f1(a, b) >= 0.5  # the original "relaxed" rule, kept only to audit it
has = lambda ans_, gold: f" {norm(gold)} " in f" {norm(ans_)} "

jk = lambda qid, a, b: f"{qid}|" + "|".join(sorted((norm(a), norm(b))))
def judge_all(items):  # items: [(q, a, b)]; one judge call per unique strict-different pair (T=0, require_parameters)
    todo = {jk(q["id"], a, b): (q, a, b) for q, a, b in items if norm(a) != norm(b) and jk(q["id"], a, b) not in J}
    def one(kv):
        k, (q, a, b) = kv
        if spent() > CAP: raise RuntimeError("spend cap hit")
        return record(k, chat(JUDGE, JPROMPT.format(q=q["q"], a=a, b=b), 64, {"provider": {"require_parameters": True}}), J, JF)
    with ThreadPoolExecutor(8) as ex: list(ex.map(one, todo.items()))
    print(f"[judge] {len(todo)} new pair calls, {len(J)} cached total, unparsed {sum(not r['answer'].upper().startswith(('YES', 'NO')) for r in J.values())}")
def same(q, a, b):  # judge-based equivalence (missions.md: free text -> a judge decides); strict-equal short-circuits
    return norm(a) == norm(b) or J[jk(q["id"], a, b)]["answer"].upper().startswith("YES")
def same_hi(q, a, b):  # sensitivity: judge outputs that do not start with YES/NO (hedges, truncations) counted as YES instead of NO
    return norm(a) == norm(b) or not J[jk(q["id"], a, b)]["answer"].upper().startswith("NO")

# ---------- data ----------
import datasets
ds = datasets.load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
Q, skipped = [], 0
for i in np.random.default_rng(SEED).permutation(len(ds)):
    x = ds[int(i)]; Tl = x["context"]["title"]
    gold = np.array([t in set(x["supporting_facts"]["title"]) for t in Tl])
    if len(Tl) != 10 or gold.sum() != 2: skipped += 1; continue  # a few val items have <10 paragraphs; keep the 10-paragraph protocol
    Q.append(dict(id=x["id"], q=x["question"], a=x["answer"], titles=Tl, type=x["type"], gold=gold,
                  paras=[f"{t}: {' '.join(s.strip() for s in ss)}" for t, ss in zip(Tl, x["context"]["sentences"])]))
    if len(Q) == NQ: break
print(f"[teacher] {TAG}: {MODEL}, require_parameters={T['t0']}")
print(f"[data] {len(Q)} questions, skipped {skipped} non-10-paragraph items while sampling")
NTR = round(NQ * 0.6); TR, TE = Q[:NTR], Q[NTR:]
print(f"[split] N={NQ} train={len(TR)} heldout={len(TE)}")

# ---------- ablation labels ----------
run(full_jobs(Q) + abl_jobs(Q) + rep_jobs(Q))
A = [R[kkey(q, ALL)]["answer"] for q in Q]
# determinism probe on 40 questions: 5 same-prompt calls (incl. full + full_rep) + 2 calls with seed=0
tags = [("full_rep2", None), ("full_rep3", None), ("full_rep4", None), ("seedA", 0), ("seedB", 0)]
with ThreadPoolExecutor(8) as ex:
    list(ex.map(lambda z: ask(f"{z[0]['id']}|{z[1][0]}", z[0]["q"], z[0]["paras"], z[1][1]), [(q, t) for q in Q[:40] for t in tags]))
g = lambda q, t: R[f"{q['id']}|{t}"]["answer"]
S5 = [[R[kkey(q, ALL)]["answer"], g(q, "full_rep"), g(q, "full_rep2"), g(q, "full_rep3"), g(q, "full_rep4")] for q in Q[:40]]

pairs = [(q, a, ans(q, [j for j in ALL if j != i])) for q, a in zip(Q, A) for i in ALL] + [(q, a, R[q["id"] + "|full_rep"]["answer"]) for q, a in zip(Q, A)]
pairs += [(q, s5[0], x) for q, s5 in zip(Q, S5) for x in s5[1:]] + [(q, g(q, "seedA"), g(q, "seedB")) for q in Q[:40]]
judge_all(pairs)
Y = np.array([[norm(ans(q, [j for j in ALL if j != i])) != norm(a) for i in ALL] for q, a in zip(Q, A)])  # (N,10) strict "matters"
Yj = np.array([[not same(q, ans(q, [j for j in ALL if j != i]), a) for i in ALL] for q, a in zip(Q, A)])  # judge "matters"
G = np.array([q["gold"] for q in Q])
rep_dis = np.mean([norm(R[q["id"] + "|full_rep"]["answer"]) != norm(a) for q, a in zip(Q, A)])
rep_dis_j = np.mean([not same(q, R[q["id"] + "|full_rep"]["answer"], a) for q, a in zip(Q, A)])
det = dict(n=40, mean_distinct_of5=float(np.mean([len({norm(x) for x in s5}) for s5 in S5])),
           all5_identical=float(np.mean([len({norm(x) for x in s5}) == 1 for s5 in S5])),
           all5_equivalent_judge=float(np.mean([all(same(q, x, s5[0]) for x in s5) for q, s5 in zip(Q, S5)])),
           seed0_pair_identical=float(np.mean([norm(g(q, "seedA")) == norm(g(q, "seedB")) for q in Q[:40]])),
           seed0_pair_equivalent_judge=float(np.mean([same(q, g(q, "seedA"), g(q, "seedB")) for q in Q[:40]])))
print("[determinism]", det)
# audit of the original F1/containment "relaxed" rule against the judge, on every strict-different pair above
seen = {jk(q["id"], a, b): (q, a, b) for q, a, b in pairs if norm(a) != norm(b)}
aud = lambda sel: [same(q, a, b) for q, a, b in seen.values() if sel(a, b)]
legacy_check = dict(n_pairs=len(seen), judge_yes=sum(aud(lambda a, b: True)),
                    containment_only=len(aud(lambda a, b: contain(a, b))), containment_judge_yes=sum(aud(lambda a, b: contain(a, b))),
                    f1_only=len(aud(lambda a, b: not contain(a, b) and f1(a, b) >= 0.5)), f1_only_judge_yes=sum(aud(lambda a, b: not contain(a, b) and f1(a, b) >= 0.5)),
                    neither=len(aud(lambda a, b: not legacy(a, b))), neither_judge_yes=sum(aud(lambda a, b: not legacy(a, b))))
print("[legacy relaxed audit]", legacy_check)
rts = [r["rt"] for r in R.values()]
print(f"[spend after labels] ${spent():.3f}, reasoning tokens total {sum(x or 0 for x in rts)} (missing in {sum(x is None for x in rts)}), "
      f"finish!=stop {sum(r['fin'] != 'stop' for r in R.values())}, empty answers {sum(r['answer'] == '' for r in R.values())}, providers {dict(Counter(r.get('prov') for r in R.values()))}")

def pr(Yb, Gb):
    tp = (Yb & Gb).sum(); return tp / max(Yb.sum(), 1), tp / Gb.sum()
em_full = np.array([norm(a) == norm(q["a"]) for q, a in zip(Q, A)]); f1_full = np.mean([f1(a, q["a"]) for q, a in zip(Q, A)])
has_full = np.array([has(a, q["a"]) for q, a in zip(Q, A)])
sl = {"all": slice(None), "train": slice(0, NTR), "heldout": slice(NTR, None)}
labstats = lambda Yx: {s: dict(n=int(Yx[v].shape[0]), frac_matter=float(Yx[v].mean()), frac_matter_gold=float(Yx[v][G[v]].mean()),
               frac_matter_nongold=float(Yx[v][~G[v]].mean()), q_zero_matter=float((Yx[v].sum(1) == 0).mean()),
               precision=float(pr(Yx[v], G[v])[0]), recall=float(pr(Yx[v], G[v])[1]), mean_matter_per_q=float(Yx[v].sum(1).mean()))
               for s, v in sl.items()}
lab = dict(strict=labstats(Y), judge=labstats(Yj))
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
def fit(Xa, tgt):  # same 6-feature LR, trained on tgt = ablation label (spec) or gold-support label (control)
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
    m.fit(Xa[:NTR].reshape(-1, Xa.shape[-1]), tgt[:NTR].ravel())
    return m.predict_proba(Xa.reshape(-1, Xa.shape[-1]))[:, 1].reshape(-1, 10), m
Xr = rel(X)
s_l, m_l = fit(X, Y); s_lg, m_lg = fit(X, G); s_r, _ = fit(Xr, Y)
from sentence_transformers import CrossEncoder
ce_pairs = [(q["q"], p) for q in Q for p in q["paras"]]
cef = CACHE / ("ce_" + hashlib.sha1("\x00".join(a + "\x01" + b for a, b in ce_pairs).encode()).hexdigest()[:12] + ".npy")  # keyed on content, not just N
if cef.exists(): CE = np.load(cef)
else:
    CE = np.array(CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu").predict(ce_pairs, batch_size=64, show_progress_bar=False)).reshape(NQ, 10); np.save(cef, CE)
rng = np.random.default_rng(SEED)
RND, TIE = rng.random((NQ, 10)), rng.random((NQ, 10)) * 1e-3
SC = {"learned": s_l, "learned (gold-trained)": s_lg, "learned+rel (extra)": s_r, "cross-encoder": CE, "oracle (gold paras)": G + TIE, "random": RND}
fn = ["cos", "position", "n_words", "q_overlap", "title_in_q", "title_tok_overlap"]
coef = dict(ablation=dict(zip(fn, m_l[-1].coef_[0].round(3).tolist())), gold=dict(zip(fn, m_lg[-1].coef_[0].round(3).tolist())))

def auc(S, L, v):  # pooled AUC, and mean within-question AUC
    s, l = S[v], L[v]; ok = (l.sum(1) > 0) & (l.sum(1) < 10)
    return roc_auc_score(l.ravel(), s.ravel()), float(np.mean([roc_auc_score(l[n], s[n]) for n in np.where(ok)[0]]))
auc_tab = {n: {"heldout_vs_ablation": auc(S, Y, sl["heldout"]), "heldout_vs_gold": auc(S, G, sl["heldout"]),
               "train_vs_ablation": auc(S, Y, sl["train"]), "heldout_vs_judge": auc(S, Yj, sl["heldout"])} for n, S in SC.items()}

# ---------- pruned re-asks on held-out ----------
def kept(S, n, k): return sorted(np.argsort(-S[n])[:k].tolist())
jobs = [(kkey(q, kept(S, NTR + n, k)), q, kept(S, NTR + n, k)) for S in SC.values() for k in KS for n, q in enumerate(TE)]
run(jobs)
cb = [ask(f"{q['id']}|closed", q["q"], [], prompt=PROMPT0) for q in TE]  # closed-book control: no context at all
rr = [R[q["id"] + "|full_rep"] for q in TE]
ev_pairs = [(q, R[kkey(q, kept(S, NTR + n, k))]["answer"], A[NTR + n]) for S in SC.values() for k in KS for n, q in enumerate(TE)]
judge_all(ev_pairs)
stable = np.array([norm(r["answer"]) == norm(A[NTR + n]) for n, r in enumerate(rr)])  # A_full agrees with its own replicate (conditions on the replicate: indicative only)
tp = np.array([q["type"] for q in TE])
bt = lambda v: {t: (float(v[tp == t].mean()), int((tp == t).sum())) for t in sorted(set(tp))}
res, AG, AGJ, GOLD = {}, {}, {}, {}
for name, S in SC.items():
    for k in KS:
        ks = [kept(S, NTR + n, k) for n in range(len(TE))]
        rec = [R[kkey(q, kk)] for q, kk in zip(TE, ks)]; full = [R[kkey(q, ALL)] for q in TE]
        ag = np.array([norm(r["answer"]) == norm(A[NTR + n]) for n, r in enumerate(rec)])
        em = np.array([norm(r["answer"]) == norm(q["a"]) for r, q in zip(rec, TE)])
        both = np.array([G[NTR + n][kk].sum() == 2 for n, kk in enumerate(ks)])
        agj = np.array([same(q, r["answer"], A[NTR + n]) for n, (r, q) in enumerate(zip(rec, TE))]); hs = np.array([has(r["answer"], q["a"]) for r, q in zip(rec, TE)])
        AG[name, k], AGJ[name, k], GOLD[name, k] = ag, agj, (em, hs)
        res[f"{name}|{k}"] = dict(agree=float(ag.mean()), agree_judge=float(agj.mean()),
                                  agree_judge_hi=float(np.mean([same_hi(q, r["answer"], A[NTR + n]) for n, (r, q) in enumerate(zip(rec, TE))])), n_truncated=sum(r["fin"] != "stop" for r in rec), em_gold=float(em.mean()), contains_gold=float(hs.mean()), tok_removed=1 - sum(r["pt"] for r in rec) / sum(r["pt"] for r in full),
                                  both_gold_kept=float(both.mean()), agree_given_both=float(ag[both].mean()) if both.any() else None,
                                  agree_given_not_both=float(ag[~both].mean()) if (~both).any() else None,
                                  agree_stable=float(ag[stable].mean()), agree_unstable=float(ag[~stable].mean()) if (~stable).any() else None,
                                  contains_gold_by_type=bt(hs),
                                  disagree_ub95=cp_upper(int((~ag).sum()), len(ag), 0.05), disagree_judge_ub95=cp_upper(int((~agj).sum()), len(agj), 0.05))
# noise-floor row: re-ask the full context a second time on the same held-out questions
ag = np.array([norm(r["answer"]) == norm(A[NTR + n]) for n, r in enumerate(rr)]); agj = np.array([same(q, r["answer"], A[NTR + n]) for n, (r, q) in enumerate(zip(rr, TE))])
FLOOR = "full-context re-ask (noise floor)"; AG[FLOOR, 10], AGJ[FLOOR, 10] = ag, agj
res[FLOOR + "|10"] = dict(agree=float(ag.mean()), agree_judge=float(agj.mean()), em_gold=float(np.mean([norm(r["answer"]) == norm(q["a"]) for r, q in zip(rr, TE)])),
    contains_gold=float(np.mean([has(r["answer"], q["a"]) for r, q in zip(rr, TE)])), tok_removed=0.0, both_gold_kept=1.0, agree_given_both=float(ag.mean()), agree_given_not_both=None,
    agree_stable=float(ag[stable].mean()), agree_unstable=None, contains_gold_by_type=None,
    disagree_ub95=cp_upper(int((~ag).sum()), len(ag), 0.05), disagree_judge_ub95=cp_upper(int((~agj).sum()), len(agj), 0.05))
cbh = np.array([has(r["answer"], q["a"]) for r, q in zip(cb, TE)])
closed_book = dict(contains_gold=float(cbh.mean()), em_gold=float(np.mean([norm(r["answer"]) == norm(q["a"]) for r, q in zip(cb, TE)])), contains_gold_by_type=bt(cbh))
full_te = dict(contains_gold=float(has_full[NTR:].mean()), contains_gold_by_type=bt(has_full[NTR:]), stable_frac=float(stable.mean()))

# paired exact McNemar tests on held-out per-question agreement (b/c = discordant counts a-only / b-only)
def mcn(a, b):
    x, y = int((a & ~b).sum()), int((~a & b).sum())
    return [x, y, 1.0 if x + y == 0 else float(binomtest(min(x, y), x + y, 0.5).pvalue)]
CMP = [("cross-encoder", "learned"), ("cross-encoder", "learned (gold-trained)"), ("learned (gold-trained)", "learned"), ("cross-encoder", "learned+rel (extra)"), ("learned+rel (extra)", "learned"), ("oracle (gold paras)", "cross-encoder")]
mc = {f"{a} vs {b}|{k}|{m}": mcn(D[a, k], D[b, k]) for a, b in CMP for k in KS for m, D in (("strict", AG), ("judge", AGJ))}
mc.update({f"{nm} vs noise floor|{k}|{m}": mcn(D[nm, k], D[FLOOR, 10]) for nm in ("oracle (gold paras)", "learned", "learned (gold-trained)", "cross-encoder") for k in KS for m, D in (("strict", AG), ("judge", AGJ))})
mc.update({f"oracle vs full, {w}|{k}": mcn(GOLD["oracle (gold paras)", k][i], (em_full, has_full)[i][NTR:]) for k in KS for i, w in ((0, "EM gold"), (1, "contains gold"))})

# ---------- report ----------
spend, wall = spent(), time.time() - t0
out = dict(tag=TAG, model=MODEL, require_parameters=T["t0"], N=NQ, ntrain=NTR, nheld=len(TE), labels=lab, gold_acc_full=gold_acc, em_full_by_type=by_type, f1_full=float(f1_full),
           replicate_disagree_strict=float(rep_dis), replicate_disagree_judge=float(rep_dis_j), determinism=det, legacy_check=legacy_check, auc=auc_tab, coef=coef, results=res, mcnemar=mc,
           closed_book=closed_book, full_te=full_te, spend_usd=spend, teacher_usd=cost(R.values()), judge_usd=cost(J.values(), JPIN, JPOUT), wall_s=wall, n_calls=len(R), n_judge=len(J),
           providers=dict(Counter(str(r.get("prov")) for r in R.values())), mean_full_prompt_tokens=float(np.mean([R[kkey(q, ALL)]["pt"] for q in Q])),
           mean_full_answer_tokens=float(np.mean([R[kkey(q, ALL)]["ct"] for q in Q])), reasoning_tokens=sum(x or 0 for x in rts), reasoning_missing=sum(x is None for x in rts),
           nonstop_finish=sum(r["fin"] != "stop" for r in R.values()), empty_answers=sum(r["answer"] == "" for r in R.values()),
           judge_unparsed=sum(not r["answer"].upper().startswith(("YES", "NO")) for r in J.values()), judge_nonstop=sum(r["fin"] != "stop" for r in J.values()),
           mean_reasoning_tokens_per_call=float(np.mean([x or 0 for x in rts])), n_closed_nonstop=sum(r["fin"] != "stop" for r in cb))
allp = {jk(q["id"], a, b): dict(q=q["q"], a=a, b=b, judge=J[jk(q["id"], a, b)]["answer"]) for q, a, b in pairs + ev_pairs if norm(a) != norm(b)}
(HERE / f"pairs_{TAG}.json").write_text(json.dumps(list(allp.values())), encoding="utf-8")  # strict-different pairs + the judge's verdict, for judge_audit.py
(HERE / f"results_{TAG}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k not in ("results", "auc", "mcnemar")}, indent=1))
print("\n| scorer | AUC pooled (strict label) | within-q (strict) | pooled (judge) | within-q (judge) | AUC pooled (gold) | within-q (gold) | train within-q (strict) |\n|---|---|---|---|---|---|---|---|")
for n, a in auc_tab.items():
    h, r_, g_ = a["heldout_vs_ablation"], a["heldout_vs_judge"], a["heldout_vs_gold"]
    print(f"| {n} | {h[0]:.3f} | {h[1]:.3f} | {r_[0]:.3f} | {r_[1]:.3f} | {g_[0]:.3f} | {g_[1]:.3f} | {a['train_vs_ablation'][1]:.3f} |")
print("\n| scorer | k | agree strict | agree judge | EM gold | contains gold | tokens removed | both gold kept | strict agree if both kept | if not | strict agree stable / unstable q | 95% UB disagree strict | 95% UB disagree judge |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for key_, r in res.items():
    s_, k = key_.split("|"); f = lambda v: "-" if v is None else f"{v:.2f}"
    print(f"| {s_} | {k} | {r['agree']:.2f} | {r['agree_judge']:.2f} | {r['em_gold']:.2f} | {r['contains_gold']:.2f} | {r['tok_removed']:.1%} | {r['both_gold_kept']:.2f} | {f(r['agree_given_both'])} | {f(r['agree_given_not_both'])} | {f(r['agree_stable'])} / {f(r['agree_unstable'])} | {r['disagree_ub95']:.2f} | {r['disagree_judge_ub95']:.2f} |")
print("\nMcNemar exact (a-only, b-only, p):"); [print(" ", k, v) for k, v in mc.items()]
print("closed book:", closed_book, "\nfull context (held-out):", full_te)
print(f"\nwall {wall:.0f}s; spend ${spend:.3f} (teacher ${out['teacher_usd']:.3f}, judge ${out['judge_usd']:.3f})")
