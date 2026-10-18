"""S14: Mission 3 on real agent tool traces (tau-bench historical trajectories). Analyses A-E.
Run: python -I run.py [A|B|C|D|E ...]   (extra.py does F latency + G Toucan)
ponytail: plain dicts + sklearn; every number printed to stdout (log.txt) and dumped to results_tau.json."""
import sys, os, json, re, collections, hashlib, random, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import *  # certify, apply_certificate, cp_upper, ...
import numpy as np

TINY = "--tiny" in sys.argv  # plumbing check only: every 4th episode, 2 seeds, outputs to $S14_OUT
OUT = Path(os.environ.get("S14_OUT", HERE))
DATA = Path(os.environ.get("S14_DATA", HERE.parent / ".data")) / "tau_bench"  # pod: /workspace/job/data (fetch.py)
WRITE = {"cancel_pending_order", "exchange_delivered_order_items", "modify_pending_order_address", "modify_pending_order_items",
         "modify_pending_order_payment", "modify_user_address", "return_delivered_order_items", "transfer_to_human_agents",
         "book_reservation", "cancel_reservation", "send_certificate", "update_reservation_baggages",
         "update_reservation_flights", "update_reservation_passengers"}
PURE = {"calculate", "think"}
KIND = lambda t: "write" if t in WRITE else "pure" if t in PURE else "read"
R = {}  # results dump


def canon(tool, a):
    if tool == "calculate":  # whitespace is irrelevant to an arithmetic expression; the only canonicalization besides key order
        a = {k: re.sub(r"\s+", "", v) if isinstance(v, str) else v for k, v in a.items()}
    return json.dumps(a, sort_keys=True, separators=(",", ":"))


def atoms(a):
    out = set()
    def w(v):
        if isinstance(v, dict): [w(x) for x in v.values()]
        elif isinstance(v, list): [w(x) for x in v]
        elif isinstance(v, (str, int, float)) and not isinstance(v, bool): out.add(str(v))
    w(a); return out


def load():
    eps = []
    for f in sorted(DATA.glob("*.json")):
        agent = "gpt-4o" if f.name.startswith("gpt") else "sonnet-3.5"
        domain = "airline" if "airline" in f.name else "retail"
        for x in json.load(open(f, encoding="utf-8"))[::4 if TINY else 1]:
            calls, last, ut = [], None, 0
            for m in x["traj"]:
                if m["role"] == "user": ut += 1
                elif m["role"] == "assistant" and m.get("tool_calls"):
                    fn = m["tool_calls"][0]["function"]; a = fn["arguments"]
                    last = (fn["name"], json.loads(a) if isinstance(a, str) else a, ut > 0 or not calls); ut = 0
                elif m["role"] == "tool":
                    t, a, gap = last
                    ret = m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
                    calls.append(dict(tool=t, args=a, akey=canon(t, a), ret=ret, err=ret.startswith("Error"), kind=KIND(t), gap=gap))
            wh, wh_sc, wr = [], [], []  # successful write history (state-aware key); wr = (atoms, hash)
            for c in calls:
                h = hashlib.md5((c["tool"] + c["akey"]).encode()).hexdigest()[:8]
                if c["kind"] == "read":
                    at = atoms(c["args"])
                    rel = tuple(hh for a2, hh in wr if a2 & at)
                    c["k_in"] = (c["tool"], c["akey"]); c["k_st"] = (c["tool"], c["akey"], tuple(wh)); c["k_sc"] = (c["tool"], c["akey"], rel)
                    c["nw"] = len(wh)
                else:
                    c["k_in"] = c["k_st"] = c["k_sc"] = (c["tool"], c["akey"]); c["nw"] = len(wh)
                if c["kind"] == "write" and not c["err"]:
                    wh.append(h); wr.append((atoms(c["args"]), h))
            eps.append(dict(agent=agent, domain=domain, task=x["task_id"], trial=x["trial"], reward=x["reward"], calls=calls))
    return eps


def pct(a, b): return 100.0 * a / b if b else float("nan")
def f1(x): return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.1f}"
def table(head, rows): return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


# ---------------------------------------------------------------- A reuse
def reuse_rows(eps, key):
    """Chronological (trial-major) pass: for each read/pure call, was `key` seen in an EARLIER episode, and did the earlier return match?"""
    seen = collections.defaultdict(list)  # key -> [(task, agent, ret)]
    rows = []
    for e in sorted(eps, key=lambda e: (e["trial"], e["agent"], e["task"])):
        new = []; inep = set()
        for c in e["calls"]:
            k = c[key]; h = seen.get(k, [])
            same = [r for t, ag, r in h if t == e["task"]]; other = [r for t, ag, r in h if t != e["task"]]
            rows.append(dict(tool=c["tool"], kind=c["kind"], trial=e["trial"], nw=c["nw"], dom=e["domain"], agent=e["agent"],
                             seen=bool(h), same=bool(same), other=bool(other), eq_last=(h[-1][2] == c["ret"]) if h else None,
                             eq_same=(same[-1] == c["ret"]) if same else None, eq_other=(other[-1] == c["ret"]) if other else None,
                             in_ep=k in inep))
            inep.add(k); new.append((k, e["task"], e["agent"], c["ret"]))
        for k, t, ag, r in new: seen[k].append((t, ag, r))
    return rows


def analysis_A(eps):
    out = []
    for dom in ("retail", "airline"):
        for agent in ("gpt-4o", "sonnet-3.5", "pooled"):
            sub = [e for e in eps if e["domain"] == dom and (agent == "pooled" or e["agent"] == agent)]
            res = {k: reuse_rows(sub, k) for k in ("k_in", "k_st", "k_sc")}
            ncalls = sum(len(e["calls"]) for e in sub); ntr = len({e["trial"] for e in sub})
            out.append(f"\n**{dom} / {agent}** ({len(sub)} episodes, {ncalls} tool calls, {ntr} trials; chronological = trial-major, so trial 0 is cold start)")
            # per tool table
            tools = sorted({r["tool"] for r in res["k_in"]}, key=lambda t: (KIND(t), -sum(r["tool"] == t for r in res["k_in"])))
            rows = []
            for t in tools:
                ri = [r for r in res["k_in"] if r["tool"] == t]; rs = [r for r in res["k_st"] if r["tool"] == t]; rc = [r for r in res["k_sc"] if r["tool"] == t]
                n = len(ri); si = sum(r["seen"] for r in ri); ss = sum(r["seen"] for r in rs); sc = sum(r["seen"] for r in rc)
                ei = sum(r["eq_last"] for r in ri if r["seen"]); es = sum(r["eq_last"] for r in rs if r["seen"]); ec = sum(r["eq_last"] for r in rc if r["seen"])
                oo = [r for r in ri if r["other"] and not r["same"]]
                oth = pct(sum(r["eq_other"] for r in oo), len(oo)) if oo else None
                rows.append([t, KIND(t), n, f1(pct(si, n)), f1(pct(ei, si)), f1(pct(ss, n)), f1(pct(es, ss)), f1(pct(sc, n)), f1(pct(ec, sc)),
                             f1(pct(sum(r["other"] and not r["same"] for r in ri), n))])
            out += table(["tool", "kind", "calls", "(i) seen %", "(i) equal %", "(ii) seen %", "(ii) equal %", "(iii) seen %", "(iii) equal %", "seen only in other tasks %"], rows)
            # headline by kind
            for kd in ("read", "pure", "write"):
                ri = [r for r in res["k_in"] if r["kind"] == kd]; rs = [r for r in res["k_st"] if r["kind"] == kd]; rc = [r for r in res["k_sc"] if r["kind"] == kd]
                n = len(ri)
                R.setdefault("A", {})[f"{dom}/{agent}/{kd}"] = dict(
                    n=n, in_seen=pct(sum(r["seen"] for r in ri), n), in_safe=pct(sum(bool(r["eq_last"]) for r in ri), n),
                    st_seen=pct(sum(r["seen"] for r in rs), n), st_safe=pct(sum(bool(r["eq_last"]) for r in rs), n),
                    sc_seen=pct(sum(r["seen"] for r in rc), n), sc_safe=pct(sum(bool(r["eq_last"]) for r in rc), n),
                    in_ep_repeat=pct(sum(r["in_ep"] for r in ri), n),
                    same_task=pct(sum(r["same"] for r in rs), n), other_only=pct(sum(r["other"] and not r["same"] for r in rs), n),
                    st_eq_same=pct(sum(r["eq_same"] for r in rs if r["eq_same"] is not None), sum(r["eq_same"] is not None for r in rs)),
                    st_eq_other=pct(sum(r["eq_other"] for r in rs if r["eq_other"] is not None and not r["same"]), sum(r["eq_other"] is not None and not r["same"] for r in rs)))
            # input-key errors split by whether a successful write already happened in the episode
            ri = [r for r in res["k_in"] if r["kind"] == "read" and r["seen"]]
            for lab, flt in (("before any write", lambda r: r["nw"] == 0), ("after >=1 write", lambda r: r["nw"] > 0)):
                s = [r for r in ri if flt(r)]
                R["A"][f"{dom}/{agent}/in_err_{lab}"] = dict(n=len(s), err=pct(sum(not r["eq_last"] for r in s), len(s)))
            # learning curve by trial (state-aware key, reads)
            tr = sorted({r["trial"] for r in res["k_st"]})
            cur = []
            for t in tr:
                s = [r for r in res["k_st"] if r["kind"] == "read" and r["trial"] == t]
                cur.append(f"t{t}: {pct(sum(r['seen'] for r in s), len(s)):.0f}%")
            out.append(f"\nRead calls reusable (state-aware key) by trial index: " + ", ".join(cur))
    return out


# ---------------------------------------------------------------- splits, predictor
def splits(eps, regime, seed):
    rng = random.Random(seed)
    tasks = sorted({e["task"] for e in eps}); rng.shuffle(tasks)
    if regime == "W":  # same-workflow reruns: earlier half of the trials = memory; later trials of disjoint task halves = calibration / test
        trials = sorted({e["trial"] for e in eps}); half = len(trials) // 2
        A, B = set(tasks[: len(tasks) // 2]), set(tasks[len(tasks) // 2:])
        M = [e for e in eps if e["trial"] < trials[half]]
        C = [e for e in eps if e["trial"] >= trials[half] and e["task"] in A]; T = [e for e in eps if e["trial"] >= trials[half] and e["task"] in B]
    else:  # X: memory / calibration / test are disjoint TASKS (all trials)
        a, b = int(len(tasks) * .4), int(len(tasks) * .7)
        sM, sC, sT = set(tasks[:a]), set(tasks[a:b]), set(tasks[b:])
        M = [e for e in eps if e["task"] in sM]; C = [e for e in eps if e["task"] in sC]; T = [e for e in eps if e["task"] in sT]
    return M, C, T


def build_mem(M, key):
    mem = collections.defaultdict(collections.Counter)
    for e in M:
        for c in e["calls"]:
            if c["kind"] == "read": mem[c[key]][c["ret"]] += 1
    return mem


def predict(mem, c, key):
    """pred, conf: majority prior return; conf = k/(n+1) with k = votes for the majority, n = prior observations (count and agreement)."""
    cnt = mem.get(c[key])
    if not cnt: return None, -1.0, 0
    n = sum(cnt.values()); ret, k = cnt.most_common(1)[0]
    return ret, k / (n + 1), n


def score_rows(eps, mem, key):
    rows = []
    for e in eps:
        for c in e["calls"]:
            if c["kind"] != "read": continue
            p, conf, n = predict(mem, c, key)
            rows.append(dict(tool=c["tool"], conf=conf, ok=(p is not None and p == c["ret"]), seen=p is not None, n=n, task=e["task"], ep=id(e)))
    return rows


def certify_eval(Crows, Trows, budget):
    """Certificate over ANSWERABLE rows only (key seen in memory). An unseen key has no prediction and always goes to the real tool,
    so it is routed by construction, not by confidence. Coverage fields below are shares of ALL rows."""
    Cs = [r for r in Crows if r["seen"]]; Ts = [r for r in Trows if r["seen"]]
    cert = certify([r["conf"] for r in Cs], [r["ok"] for r in Cs], budget) if Cs else dict(cutoff=float("inf"), coverage=0.0, emp_error=None, bound=None)
    cert["coverage_all"] = cert["coverage"] * len(Cs) / max(1, len(Crows))
    if cert["cutoff"] == float("inf") or not Ts: return cert, dict(coverage=0.0, error=None, n_above=0, n=len(Trows))
    ap = apply_certificate([r["conf"] for r in Ts], [r["ok"] for r in Ts], cert["cutoff"])
    ap["n"] = len(Trows); return cert, ap


def analysis_B(eps, seeds=range(10)):
    out = []; budgets = (0.01, 0.02, 0.05)
    for regime, rname in (("W", "W: later trials of held-out tasks, memory = earlier trials (workflow re-run many times)"),
                          ("X", "X: all trials of held-out tasks, memory = other tasks only (no same-task history)")):
        out.append(f"\n### Regime {rname}")
        for dom in ("retail", "airline"):
            for agent in ("gpt-4o", "sonnet-3.5", "pooled"):
                sub = [e for e in eps if e["domain"] == dom and (agent == "pooled" or e["agent"] == agent)]
                rows = []
                for key, kname in (("k_in", "input-keyed"), ("k_st", "state-aware"), ("k_sc", "entity-scoped")):
                    for b in budgets:
                        acc = collections.defaultdict(float); feas = viol = 0; per_tool = collections.defaultdict(lambda: [0, 0, 0]); ntest = 0
                        ceil_ = []
                        for s in seeds:
                            M, C, T = splits(sub, regime, s); mem = build_mem(M, key)
                            Cr, Tr = score_rows(C, mem, key), score_rows(T, mem, key)
                            ceil_.append(pct(sum(r["ok"] for r in Tr), len(Tr)))
                            cert, ap = certify_eval(Cr, Tr, b); ntest += len(Tr)
                            if cert["cutoff"] != float("inf"):
                                feas += 1; acc["cal_cov"] += cert["coverage_all"]; skipped = ap["n_above"]; err = 0 if ap["error"] is None else ap["error"] * skipped
                                acc["skipped"] += skipped; acc["errors"] += err
                                if ap["error"] is not None and ap["error"] > b: viol += 1
                                for r in Tr:
                                    if r["conf"] >= cert["cutoff"]:
                                        pt = per_tool[r["tool"]]; pt[0] += 1; pt[1] += (not r["ok"])
                            for r in Tr: per_tool[r["tool"]][2] += 1
                        sk = acc["skipped"]
                        rows.append([kname, f"{int(b*100)}%", f"{feas}/{len(list(seeds))}", f1(100 * acc["cal_cov"] / feas) if feas else "-",
                                     f1(pct(sk, ntest)) if feas else "0.0", f1(pct(acc["errors"], sk)) if sk else "-", f"{viol}", f1(np.mean(ceil_))])
                        R.setdefault("B", {})[f"{regime}/{dom}/{agent}/{key}/{b}"] = dict(
                            feasible=feas, skipped_pct_of_reads=pct(sk, ntest) if feas else 0.0, realized_err_pct=pct(acc["errors"], sk) if sk else None,
                            violations=viol, per_tool={t: dict(skipped=v[0], errors=v[1], n=v[2]) for t, v in per_tool.items()}, oracle_reusable_pct=float(np.mean(ceil_)))
                out.append(f"\n**{dom} / {agent}** (read calls only; {len(list(seeds))} random task splits, pooled test counts)")
                out += table(["key", "budget", "certified (seeds)", "cal. coverage %", "test: reads skipped %", "test: realized error % of skipped", "seeds with err > budget", "oracle reusable %"], rows)
                # per-tool breakdown for the state-aware 2% certificate
                d = R["B"][f"{regime}/{dom}/{agent}/k_st/0.02"]
                if d["feasible"]:
                    out.append(f"  per tool (state-aware, 2%): " + "; ".join(f"{t}: skip {pct(v['skipped'], v['n']):.0f}% of {v['n']} err {v['errors']:.0f}" for t, v in sorted(d["per_tool"].items(), key=lambda kv: -kv[1]["n"])))
    return out


# ---------------------------------------------------------------- C outage
def analysis_C(eps, seeds=range(10)):
    out = ["Regime W splits, 10 seeds; the outage hits one read tool for every test episode. Policies answer each call to the dead tool from the log. "
           "'episodes served' = all calls to the dead tool answered; 'served and correct' = additionally every answer equal to the real return."]
    for dom in ("retail", "airline"):
        for agent in ("gpt-4o", "sonnet-3.5", "pooled"):
            sub = [e for e in eps if e["domain"] == dom and (agent == "pooled" or e["agent"] == agent)]
            cnt = collections.Counter(c["tool"] for e in sub for c in e["calls"] if c["kind"] == "read")
            top = [t for t, _ in cnt.most_common(3)]
            rows = []
            for X in top:
                pol = collections.defaultdict(lambda: dict(calls=0, ans=0, wrong=0, eps=0, served=0, correct=0))
                for s in seeds:
                    M, C, T = splits(sub, "W", s)
                    mem_in, mem_st = build_mem(M, "k_in"), build_mem(M, "k_st")
                    certs = {b: certify_eval(score_rows(C, mem_st, "k_st"), [], b)[0]["cutoff"] for b in (0.01, 0.02)}
                    certs_in = {b: certify_eval(score_rows(C, mem_in, "k_in"), [], b)[0]["cutoff"] for b in (0.01, 0.02)}
                    for e in T:
                        cs = [c for c in e["calls"] if c["tool"] == X]
                        if not cs: continue
                        # policies: name -> list of (answered, correct)
                        res = collections.defaultdict(list)
                        for c in cs:
                            res["circuit breaker"].append((False, False))
                            p, conf, n = predict(mem_in, c, "k_in"); res["plain last-value cache (input key)"].append((p is not None, p == c["ret"]))
                            p2, conf2, n2 = predict(mem_st, c, "k_st"); res["state-aware cache, uncertified"].append((p2 is not None, p2 == c["ret"]))
                            for b in (0.01, 0.02):
                                res[f"certified state-aware cache @{int(b*100)}%"].append((conf2 >= certs[b] and p2 is not None, p2 == c["ret"]))
                                res[f"certified input-keyed cache @{int(b*100)}%"].append((conf >= certs_in[b] and p is not None, p == c["ret"]))
                        for name, lst in res.items():
                            d = pol[name]; d["calls"] += len(lst); d["ans"] += sum(a for a, _ in lst); d["wrong"] += sum(a and not ok for a, ok in lst)
                            d["eps"] += 1; d["served"] += all(a for a, _ in lst); d["correct"] += all(a and ok for a, ok in lst)
                for name, d in pol.items():
                    rows.append([X, name, d["calls"], f1(pct(d["ans"], d["calls"])), f1(pct(d["wrong"], d["ans"])) if d["ans"] else "-", d["eps"], f1(pct(d["served"], d["eps"])), f1(pct(d["correct"], d["eps"]))])
                    R.setdefault("C", {})[f"{dom}/{agent}/{X}/{name}"] = dict(calls=d["calls"], answered_pct=pct(d["ans"], d["calls"]), err_pct=pct(d["wrong"], d["ans"]) if d["ans"] else None,
                                                                              eps=d["eps"], served_pct=pct(d["served"], d["eps"]), served_correct_pct=pct(d["correct"], d["eps"]))
            out.append(f"\n**{dom} / {agent}**, outage tools = {top}")
            out += table(["dead tool", "policy", "test calls", "calls answered %", "error % of answered", "episodes calling it", "episodes served %", "served and all correct %"], rows)
    return out


# ---------------------------------------------------------------- D write outcomes
CAT = {"cabin", "flight_type", "insurance", "reason", "payment_method_id", "new_payment_method_id", "payment_id", "country", "state", "province"}


def arg_feats(a, f):
    """Generic argument features: numbers as-is, list lengths, and the *type* of categorical / payment arguments (digits masked).
    Entity ids (order_id, reservation_id, item ids...) are deliberately not features, so the model cannot memorize entities."""
    for k, v in a.items():
        if isinstance(v, (bool, int, float)): f[k] = float(v)
        elif isinstance(v, str):
            if k in CAT: f[f"{k}~{re.sub(r'[0-9]+', '#', v)}"] = 1.0
        elif isinstance(v, list):
            f[f"{k}__len"] = float(len(v))
            for x in v[:3]:
                if isinstance(x, str) and k in CAT: f[f"{k}~{re.sub(r'[0-9]+', '#', x)}"] = 1.0
                elif isinstance(x, dict): arg_feats(x, f)
        elif isinstance(v, dict): arg_feats(v, f)


def write_feats(e, i):
    c = e["calls"][i]; prev = e["calls"][:i]; f = {f"tool={c['tool']}": 1.0, f"agent={e['agent']}": 1.0}
    f["n_prior"] = float(len(prev)); f["n_prior_reads"] = float(sum(p["kind"] == "read" for p in prev))
    f["n_prior_writes_ok"] = float(sum(p["kind"] == "write" and not p["err"] for p in prev)); f["n_prior_writes_err"] = float(sum(p["kind"] == "write" and p["err"] for p in prev))
    same = [p for p in prev if p["tool"] == c["tool"]]; f["same_tool_prior"] = float(len(same)); f["same_tool_prior_err"] = float(sum(p["err"] for p in same))
    f["prev_call_err"] = float(bool(prev) and prev[-1]["err"]); f["user_gap"] = float(c["gap"])
    arg_feats(c["args"], f)
    at = atoms(c["args"]); idat = {a for a in at if re.match(r"^(#W|[A-Z0-9]{6}$|.*_\d{4,})", a)}
    f["id_seen_in_prior_args"] = float(sum(any(a in atoms(p["args"]) for p in prev) for a in idat)) if idat else 0.0
    f["id_seen_frac"] = f["id_seen_in_prior_args"] / len(idat) if idat else 0.0
    # state learned from the log: latest earlier read of the same entity -> a few scalar fields of its return
    for p in reversed(prev):
        if p["kind"] == "read" and (atoms(p["args"]) & at):
            try: j = json.loads(p["ret"])
            except Exception: j = None
            if isinstance(j, dict):
                for fld in ("status", "cabin", "insurance", "flight_type"):
                    if isinstance(j.get(fld), (str, bool)): f[f"ret.{fld}={j[fld]}"] = 1.0
                for fld in ("items", "flights", "passengers", "payment_history"):
                    if isinstance(j.get(fld), list): f[f"ret.{fld}__len"] = float(len(j[fld]))
                f["ret.found"] = 1.0
            break
    return f


def analysis_D(eps):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.model_selection import GroupKFold
    out = []; rows = []
    for dom in ("retail", "airline"):
        X, y, g, tools = [], [], [], []
        for e in eps:
            if e["domain"] != dom: continue
            for i, c in enumerate(e["calls"]):
                if c["kind"] == "write" and c["tool"] != "transfer_to_human_agents":
                    X.append(write_feats(e, i)); y.append(int(c["err"])); g.append(e["task"]); tools.append(c["tool"])
        y = np.array(y); g = np.array(g); n = len(y)
        dv = DictVectorizer(sparse=False); Xa = dv.fit_transform(X); names = dv.feature_names_
        tool_cols = [j for j, nm in enumerate(names) if nm.startswith("tool=")]
        oof = {k: np.zeros(n) for k in ("tool-only LR", "all-context LR", "all-context GBM", "per-tool base rate")}
        for tr, te in GroupKFold(5).split(Xa, y, g):
            mu, sd = Xa[tr].mean(0), Xa[tr].std(0) + 1e-6
            Z = (Xa - mu) / sd
            oof["tool-only LR"][te] = LogisticRegression(C=1.0, max_iter=2000).fit(Xa[tr][:, tool_cols], y[tr]).predict_proba(Xa[te][:, tool_cols])[:, 1]
            oof["all-context LR"][te] = LogisticRegression(C=0.05, max_iter=3000).fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
            oof["all-context GBM"][te] = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, random_state=0).fit(Xa[tr], y[tr]).predict_proba(Xa[te])[:, 1]
            tt = np.array(tools)
            for j in te:
                m = tt[tr] == tt[j]; oof["per-tool base rate"][j] = (y[tr][m].sum() + 0.5) / (m.sum() + 1)
        base = y.mean()
        per_tool = collections.Counter((t, int(v)) for t, v in zip(tools, y))
        out.append(f"\n**{dom}**: {n} write calls (transfer excluded), {int(y.sum())} error returns (base rate {100*base:.1f}%); error rate per tool: " +
                   "; ".join(f"{t} {pct(per_tool[(t,1)], per_tool[(t,0)]+per_tool[(t,1)]):.0f}% of {per_tool[(t,0)]+per_tool[(t,1)]}" for t in sorted(set(tools))))
        for name, p in oof.items():
            auc = roc_auc_score(y, p); ap = average_precision_score(y, p)
            # fallback-first rule: flag top-q% by predicted P(error)
            order = np.argsort(-p); k = int(round(0.2 * n)); prec = y[order[:k]].mean(); rec = y[order[:k]].sum() / max(1, y.sum())
            # bootstrap AUC CI over tasks
            rng = np.random.default_rng(0); ut = np.unique(g); aucs = []
            for _ in range(200):
                bt = rng.choice(ut, len(ut)); idx = np.concatenate([np.where(g == t)[0] for t in bt])
                if 0 < y[idx].sum() < len(idx): aucs.append(roc_auc_score(y[idx], p[idx]))
            rows.append([dom, name, f"{auc:.3f}", f"[{np.percentile(aucs,2.5):.3f}, {np.percentile(aucs,97.5):.3f}]", f"{ap:.3f}", f"{100*base:.1f}%", f"{100*prec:.0f}%", f"{100*rec:.0f}%"])
            R.setdefault("D", {})[f"{dom}/{name}"] = dict(auc=auc, ap=ap, base=base, n=n, errors=int(y.sum()))
        # top features of the LR
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
        lr = LogisticRegression(C=0.05, max_iter=3000).fit((Xa - mu) / sd, y); co = lr.coef_[0]
        top = np.argsort(-np.abs(co))[:8]
        out.append("  strongest LR features (std. coef): " + ", ".join(f"{names[j]} {co[j]:+.2f}" for j in top))
    out += table(["domain", "model", "AUC (grouped 5-fold by task)", "AUC 95% CI (task bootstrap)", "avg precision", "base rate", "precision @ top-20%", "recall @ top-20%"], rows)
    return out


# ---------------------------------------------------------------- E next-call speculation
def tok(c): return c["tool"] + ("!" if c["err"] else "")


class NameModel:
    def __init__(self, M, order=3, min_support=2):
        self.o, self.ms = order, min_support
        self.cnt = [collections.defaultdict(collections.Counter) for _ in range(order + 1)]
        for e in M:
            cs = [c for c in e["calls"] if c["tool"] != "think"]
            for i in range(len(cs) - 1):
                for o in range(order + 1):
                    if i + 1 >= o: self.cnt[o][tuple(tok(c) for c in cs[i + 1 - o: i + 1])][cs[i + 1]["tool"]] += 1

    def dist(self, ctx_calls):
        for o in range(self.o, -1, -1):
            if len(ctx_calls) < o: continue
            d = self.cnt[o].get(tuple(tok(c) for c in ctx_calls[len(ctx_calls) - o:]))
            if d and sum(d.values()) >= self.ms: tot = sum(d.values()); return [(t, k / tot) for t, k in d.most_common()], o
        return [], -1


class PrefixModel:
    """Trajectory cache: next (tool, args) after the longest matching suffix of exact (tool, args) calls seen in memory."""
    def __init__(self, M, order=3):
        self.o = order; self.cnt = [collections.defaultdict(collections.Counter) for _ in range(order + 1)]
        for e in M:
            cs = [c for c in e["calls"] if c["tool"] != "think"]
            for i in range(len(cs) - 1):
                for o in range(1, order + 1):
                    if i + 1 >= o: self.cnt[o][tuple((c["tool"], c["akey"]) for c in cs[i + 1 - o: i + 1])][(cs[i + 1]["tool"], cs[i + 1]["akey"])] += 1

    def pred(self, ctx):
        for o in range(self.o, 0, -1):
            if len(ctx) < o: continue
            d = self.cnt[o].get(tuple((c["tool"], c["akey"]) for c in ctx[len(ctx) - o:]))
            if d:
                n = sum(d.values()); (t, ak), k = d.most_common(1)[0]; return t, ak, k / (n + 1), o
        return None


def walk(j, key, outl):
    """Collect values of field `key` (or the plural list `key[:-3]+'s'`, `key[:-4]+'s'`) anywhere in a parsed return."""
    pl = {key + "s", key.replace("_id", "s"), key.replace("_id", "_ids")}
    if isinstance(j, dict):
        for k, v in j.items():
            if k == key and isinstance(v, (str, int)): outl.append(v)
            elif k in pl and isinstance(v, list): outl.extend(x for x in v if isinstance(x, (str, int)))
            elif k in pl and isinstance(v, dict): outl.extend(v.keys())
            walk(v, key, outl)
    elif isinstance(j, list):
        for x in j: walk(x, key, outl)


class ArgModel:
    def __init__(self, M):
        self.keys = collections.defaultdict(collections.Counter); self.vals = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
        for e in M:
            for c in e["calls"]:
                self.keys[c["tool"]][tuple(sorted(c["args"]))] += 1
                for k, v in c["args"].items(): self.vals[c["tool"]][k][json.dumps(v)] += 1

    def build(self, tool, ctx):
        ks = self.keys.get(tool)
        if not ks: return None
        keys = ks.most_common(1)[0][0]; a = {}
        for k in keys:
            cand = None
            for c in reversed(ctx):  # (1) same-named arg in the most recent earlier call
                if k in c["args"]: cand = c["args"][k]; break
            if cand is None:  # (2) same-named field (or plural list) in an earlier return, newest return first
                for c in reversed(ctx):
                    if c["tool"].startswith("find_user_id") and k == "user_id" and not c["err"]: cand = c["ret"]; break
                    try: j = json.loads(c["ret"])
                    except Exception: continue
                    o = []; walk(j, k, o)
                    if o: cand = o[0]; break
            if cand is None:  # (3) most common value seen for this tool/arg in memory
                vv = self.vals[tool][k]
                if vv: cand = json.loads(vv.most_common(1)[0][0])
            if cand is None: return None
            a[k] = cand
        return canon(tool, a)


def eval_E(sub, regime, seeds):
    acc = collections.defaultdict(float); hit_by = collections.defaultdict(lambda: [0, 0]); gate = {th: [0, 0, 0] for th in (0.0, 0.3, 0.5, 0.7, 0.9)}
    for s in seeds:
        M, _, T = splits(sub, regime, s)
        if regime == "W":  # E has no calibration step, so use calibration tasks as extra test tasks
            M2, C2, T2 = splits(sub, regime, s); T = C2 + T2
        nm, pm, am = NameModel(M), PrefixModel(M), ArgModel(M)
        uni = collections.Counter(c["tool"] for e in M for c in e["calls"] if c["tool"] != "think")
        for e in T:
            cs = [c for c in e["calls"] if c["tool"] != "think"]
            for i in range(len(cs) - 1):
                ctx, nxt = cs[: i + 1], cs[i + 1]; acc["n"] += 1; ro = nxt["kind"] == "read"; acc["next_read"] += ro
                d, o = nm.dist(ctx); names = [t for t, _ in d][:3] or [t for t, _ in uni.most_common(3)]
                p1 = d[0][1] if d else 0.0
                acc["uni_top1"] += nxt["tool"] == uni.most_common(1)[0][0]
                acc["name_top1"] += nxt["tool"] == names[0]; acc["name_top3"] += nxt["tool"] in names
                # name + args (heuristic copy args) top-1 / top-3
                cands = []
                for t in names:
                    ak = am.build(t, ctx)
                    if ak: cands.append((t, ak))
                acc["heur_top1"] += bool(cands) and cands[0] == (nxt["tool"], nxt["akey"]); acc["heur_top3"] += (nxt["tool"], nxt["akey"]) in cands
                # arg-free tools
                pr = pm.pred(ctx)
                comb = ([(pr[0], pr[1])] if pr else []) + [c for c in cands if not pr or c != (pr[0], pr[1])]
                acc["comb_top1"] += bool(comb) and comb[0] == (nxt["tool"], nxt["akey"]); acc["comb_top3"] += (nxt["tool"], nxt["akey"]) in comb[:3]
                acc["prefix_cov"] += pr is not None
                if pr: acc["prefix_hit"] += (pr[0], pr[1]) == (nxt["tool"], nxt["akey"])
                # speculation: launch the top-1 combined prediction only if it is a read-only call
                if comb:
                    sp = KIND(comb[0][0]) == "read"; ok = comb[0] == (nxt["tool"], nxt["akey"])
                    acc["spec_launch"] += sp; acc["spec_hit_read"] += sp and ok
                    score = pr[2] if (pr and comb[0] == (pr[0], pr[1])) else p1 * 0.5  # prefix-cache agreement, else name prob discounted for unknown args
                    for th in gate:
                        if sp and score >= th: gate[th][0] += 1; gate[th][1] += ok
                    for th in gate: gate[th][2] += 1
                gk = "after user turn" if nxt["gap"] else "immediate chain"
                hit_by[gk][1] += 1; hit_by[gk][0] += bool(comb) and comb[0] == (nxt["tool"], nxt["akey"])
                hn = "after user turn" if nxt["gap"] else "immediate chain"
                hit_by["name:" + hn][1] += 1; hit_by["name:" + hn][0] += nxt["tool"] == names[0]
                rk = "read" if ro else "write"
                hit_by["name+args top1, next is " + rk][1] += 1; hit_by["name+args top1, next is " + rk][0] += bool(comb) and comb[0] == (nxt["tool"], nxt["akey"])
                hit_by["name top1, next is " + rk][1] += 1; hit_by["name top1, next is " + rk][0] += nxt["tool"] == names[0]
    return acc, hit_by, gate


def analysis_E(eps, seeds=range(5)):
    out = ["Positions = every tool call that has a following tool call in the same episode (think calls dropped). Prediction is made when the call is issued, before its return."]
    rows = []
    for regime in ("W", "X"):
        for dom in ("retail", "airline"):
            for agent in ("gpt-4o", "sonnet-3.5"):
                sub = [e for e in eps if e["domain"] == dom and e["agent"] == agent]
                acc, hit_by, gate = eval_E(sub, regime, seeds); n = acc["n"]
                rows.append([regime, dom, agent, int(n / len(list(seeds))), f1(pct(acc["next_read"], n)), f1(pct(acc["uni_top1"], n)), f1(pct(acc["name_top1"], n)), f1(pct(acc["name_top3"], n)),
                             f1(pct(acc["heur_top1"], n)), f1(pct(acc["heur_top3"], n)), f1(pct(acc["comb_top1"], n)), f1(pct(acc["comb_top3"], n)),
                             f1(pct(acc["prefix_cov"], n)), f1(pct(acc["prefix_hit"], acc["prefix_cov"])), f1(pct(acc["spec_hit_read"], n)), f1(pct(acc["spec_launch"], n))])
                R.setdefault("E", {})[f"{regime}/{dom}/{agent}"] = dict(n=n / len(list(seeds)), name_top1=pct(acc["name_top1"], n), name_top3=pct(acc["name_top3"], n), heur_top1=pct(acc["heur_top1"], n),
                    heur_top3=pct(acc["heur_top3"], n), comb_top1=pct(acc["comb_top1"], n), comb_top3=pct(acc["comb_top3"], n), spec_useful=pct(acc["spec_hit_read"], n), spec_launch=pct(acc["spec_launch"], n),
                    prefix_cov=pct(acc["prefix_cov"], n), prefix_prec=pct(acc["prefix_hit"], acc["prefix_cov"]), next_read=pct(acc["next_read"], n),
                    hit_by={k: dict(rate=pct(v[0], v[1]), share=pct(v[1], n)) for k, v in hit_by.items()}, gate={str(th): dict(launched=pct(v[0], v[2]), precision=pct(v[1], v[0])) for th, v in gate.items()})
    out += table(["regime", "domain", "agent", "positions/seed", "next is read %", "unigram top-1 %", "name n-gram top-1 %", "name top-3 %", "name+args (copy heuristic) top-1 %", "name+args top-3 %",
                  "+ trajectory cache top-1 %", "+ trajectory cache top-3 %", "traj. cache matches %", "traj. cache precision %", "useful speculation (read-only AND exactly right) % of next calls", "speculations launched %"], rows)
    out.append("\nBreakdown (combined predictor, name+args top-1 / name top-1; seeds pooled):")
    for k, v in R["E"].items():
        out.append(f"- {k}: " + ", ".join(f"{a}: {b['rate']:.1f}% (share {b['share']:.0f}%)" for a, b in sorted(v["hit_by"].items())))
    out.append("\nSpeculation gate (launch only when the top-1 call is read-only and score >= tau; score = trajectory-cache agreement, else 0.5 x name probability): launched % of positions / precision %")
    for k, v in R["E"].items():
        out.append(f"- {k}: " + "; ".join(f"tau {th}: {g['launched']:.1f}% / {g['precision']:.0f}%" for th, g in v["gate"].items()))
    return out


if __name__ == "__main__":
    want = {a for a in sys.argv[1:] if not a.startswith("--")} or set("ABCDE")
    eps = load()
    print(f"[load] {len(eps)} episodes, {sum(len(e['calls']) for e in eps)} tool calls")
    sections = {}
    for name, fn in (("A", analysis_A), ("B", analysis_B), ("C", analysis_C), ("D", analysis_D), ("E", analysis_E)):
        if name in want:
            lines = fn(eps, seeds=range(2)) if TINY and name in "BCE" else fn(eps); sections[name] = lines; print(f"\n######## {name}"); print("\n".join(lines), flush=True)
    prev = {}
    p = OUT / "results_tau.json"
    if p.exists(): prev = json.loads(p.read_text(encoding="utf-8"))
    prev.update(R); p.write_text(json.dumps(prev, indent=1, default=float), encoding="utf-8")
    for k, v in sections.items(): (OUT / f"tables_{k}.md").write_text("\n".join(v), encoding="utf-8")
