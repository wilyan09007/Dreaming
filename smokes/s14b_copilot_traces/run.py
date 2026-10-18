"""S14b: Mission 3 on the Microsoft GitHub Copilot coding-agent traces (day 1 = train, day 2 = chronological test). Metadata only: tool NAMES, status, durations.
Stages: F (latency), X (cross-dataset latency vs TraceLab), E (next-call speculation on tool names), D (failure prediction), S (structure stats), T (time shares + speculation bound).
Run: python -I run.py [F X E D S T ...]   (flatten.py must have been run for both days; fetch.py downloads the data). $0, CPU only.
GBM = sklearn HistGradientBoosting, fitted on a fixed 300k-row sample of the training rows (as in S14). Every number goes to stdout (log.txt) and results.json."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common14b import *
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

NFIT = 300_000
TOPK_TOOLS = 200  # categorical tool code: top-200 tools of day 1, rest = 'other'


def prep():
    """Load both days; add the shared codes. Returns (d1, d2, info)."""
    d1, d2 = load_calls(DAYS[0]), load_calls(DAYS[1])
    d2["sid"] = "b" + d2["sid"]; d1["sid"] = "a" + d1["sid"]
    allc = pd.concat([d1, d2], ignore_index=True)
    allc["sidc"] = pd.factorize(allc["sid"])[0]
    allc["tfull"] = pd.factorize(allc["tool"])[0]
    top = d1["tool"].value_counts().index[:TOPK_TOOLS]; tmap = {t: i for i, t in enumerate(top)}
    allc["tcat"] = allc["tool"].map(tmap).fillna(TOPK_TOOLS).astype(int)
    mods = sorted(allc["model"].dropna().unique()); mm = {m: i for i, m in enumerate(mods)}
    allc["mcat"] = allc["model"].map(mm).fillna(-1).astype(int)
    allc["fail"] = (allc["status"] == 2).astype(int)
    allc["ll"] = np.log1p(allc["dur"].clip(lower=0))
    allc["issue"] = allc["bstart"]
    return allc


def slow_stats(y):
    return dict(n=len(y), median=float(np.median(y)), p90=float(np.percentile(y, 90)), s1=float((y > 1000).mean()), s10=float((y > 10000).mean()),
                t1=float(y[y > 1000].sum() / y.sum()), t10=float(y[y > 10000].sum() / y.sum()))


CALL_TIME = ["tcat", "mcat", "gpos", "bidx", "k", "bsize", "hour", "turn", "prompt"]
HIST = ["l1", "l2", "l3", "s_ew", "st_ew", "st_cnt", "nfin"]


def hgb_reg(Xtr, ytr, cats):
    return HistGradientBoostingRegressor(loss="absolute_error", max_iter=250, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=50, categorical_features=cats, random_state=0).fit(Xtr, ytr)


def hgb_clf(Xtr, ytr, cats, it=250):
    return HistGradientBoostingClassifier(max_iter=it, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=50, categorical_features=cats, random_state=0).fit(Xtr, ytr)


def lat_metrics(y, p):
    e = np.abs(p - y)
    return dict(mae=float(e.mean()), mae_clip=float(np.abs(np.minimum(p, 3.6e6) - np.minimum(y, 3.6e6)).mean()), p90=float(np.percentile(e, 90)), med=float(np.median(e)),
                logerr=float(np.abs(np.log1p(p) - np.log1p(np.maximum(y, 0))).mean()))



def add_hist(A):
    """Causal history features for both days (cached). Per-tool stream EWMA is seeded from day-1 per-tool medians and restarts on day 2."""
    tr = A["day"] == DAYS[0]
    med_tool = A.loc[tr].groupby("tfull")["dur"].median(); glob = float(A.loc[tr, "dur"].median())
    if "l1" in A.columns: return A, glob, med_tool
    cache = FLAT / "hist.parquet"
    if cache.exists(): H = pd.read_parquet(cache)
    else:
        init = np.full(int(A.tfull.max()) + 1, np.log1p(glob)); init[med_tool.index.values] = np.log1p(med_tool.values)
        feats = []
        for msk in (tr, ~tr):
            d = A[msk]; feats.append(pd.DataFrame(causal_pass(d.sidc.values, d.tfull.values, d.issue.values, d.dur.values.clip(min=0), d.fail.values, init), index=d.index))
        H = pd.concat(feats).loc[A.index]; H.to_parquet(cache)
    return A.join(H), glob, med_tool

# ------------------------------------------------------------------ F  latency
def stage_F(A):
    out = []; R = {}
    tr = A["day"] == DAYS[0]; te = ~tr
    # call-time sanity: are the timestamps completion times?
    L = pd.read_parquet(FLAT / f"{DAYS[0]}_llm.parquet", columns=["sid", "turn", "lidx", "end", "dur"]); B = pd.read_parquet(FLAT / f"{DAYS[0]}_batches.parquet")
    m = B[B.lidx >= 0].merge(L, on=["sid", "turn", "lidx"], suffixes=("", "_l"))
    nxt = B.merge(L.assign(lidx=L.lidx - 1), on=["sid", "turn", "lidx"], suffixes=("", "_n"))
    R["ts_check"] = dict(batch_start_after_issuing_llm_end=float((m.bstart >= m.end - 5).mean()),
                         next_llm_start_after_batch_end_end_hyp=float(((nxt.end - nxt.dur) >= (nxt.bstart + nxt.bdur) - 5).mean()),
                         next_llm_end_after_batch_end_start_hyp=float((nxt.end >= nxt.bstart + nxt.bdur).mean()),
                         median_gap_ms_batch_end_to_next_llm_start=float(np.median((nxt.end - nxt.dur) - (nxt.bstart + nxt.bdur))))
    out.append("Timestamp convention check (day 1): a batch's start (timestamp - duration) is no earlier than the issuing LLM call's timestamp in "
               f"{pc(R['ts_check']['batch_start_after_issuing_llm_end'])} of pairs, and the next LLM call's start (timestamp - duration) is no earlier than the batch end in "
               f"{pc(R['ts_check']['next_llm_start_after_batch_end_end_hyp'])} (median gap {R['ts_check']['median_gap_ms_batch_end_to_next_llm_start']:.0f} ms). So timestamps are completion times. "
               "Each tool call is issued at its batch start and finishes at batch start + its own duration.")
    A, glob, med_tool = add_hist(A); say("F: causal history features ready")
    tr = A["day"] == DAYS[0]; te = ~tr; Tr, Te = A[tr], A[te]
    y = Te["dur"].values.clip(min=0)
    # --- descriptive
    s_all = {d: slow_stats(A.loc[A.day == d, "dur"].clip(lower=0).values) for d in DAYS}; s_all["pooled"] = slow_stats(A["dur"].clip(lower=0).values)
    R["lat_stats"] = s_all
    bat = pd.concat([pd.read_parquet(FLAT / f"{d}_batches.parquet", columns=["bdur"]) for d in DAYS])["bdur"].clip(lower=0).values
    R["batch_lat"] = slow_stats(bat)
    out.append(f"Tool calls: day 1 {tr.sum()} (sessions {Tr.sid.nunique()}), day 2 {te.sum()} (sessions {Te.sid.nunique()}). Calls with duration > 1 s: " +
               ", ".join(f"{d} {pc(v['s1'])} of calls / {pc(v['t1'])} of summed tool time" for d, v in s_all.items()) + "; > 10 s: " +
               ", ".join(f"{d} {pc(v['s10'],2)} of calls / {pc(v['t10'])} of time" for d, v in s_all.items()) +
               f". Summed over parallel calls (not elapsed time). Per batch (wall time of the whole batch, {len(bat)} batches): > 1 s {pc(R['batch_lat']['s1'])} of batches / {pc(R['batch_lat']['t1'])} of batch time, > 10 s {pc(R['batch_lat']['s10'],2)} / {pc(R['batch_lat']['t10'])}. "
               f"Duration quantiles pooled: median {s_all['pooled']['median']:.0f} ms, p90 {s_all['pooled']['p90']:.0f} ms, max {A['dur'].max():.3g} ms.")
    # --- per-tool table
    top = A["tool"].value_counts().index[:15]; rows = []; R["per_tool"] = {}
    for t in top:
        d = A.loc[A.tool == t, "dur"].clip(lower=0).values; tt = d.sum()
        rows.append([t, len(d), f"{np.median(d):.0f}", f"{np.percentile(d,90):.0f}", f"{100*(d>1000).mean():.1f}", f"{100*(d>10000).mean():.2f}", f"{100*d[d>1000].sum()/tt:.1f}", f"{100*tt/A['dur'].clip(lower=0).sum():.1f}"])
        R["per_tool"][t] = rows[-1][1:]
    out += ["", "Per-tool latency, top 15 tools by calls (both days):"] + table(["tool", "calls", "median ms", "p90 ms", "% calls > 1 s", "% calls > 10 s", "% of its time in calls > 1 s", "% of all tool time"], rows)
    # --- ladder
    P = {}
    P["global median"] = np.full(len(Te), glob)
    pt = Te["tfull"].map(med_tool).fillna(glob).values; P["per-tool median (day 1)"] = pt
    P["per-tool EWMA over the stream (alpha 0.1)"] = np.expm1(Te["tool_ew"].fillna(np.log1p(glob)).values)
    P["per-session same-tool EWMA (0.5)"] = np.expm1(Te["st_ew"].where(Te["st_ew"].notna(), np.log1p(pt)).values)
    rng = np.random.default_rng(0); fi = rng.choice(len(Tr), min(NFIT, len(Tr)), replace=False); Xs = Tr.iloc[fi]
    cats = [0, 1]
    g1 = hgb_reg(Xs[CALL_TIME].values, Xs["ll"].values, cats); P["GBM, call-time features only"] = np.expm1(g1.predict(Te[CALL_TIME].values))
    F2 = CALL_TIME + HIST
    g2 = hgb_reg(Xs[F2].values, Xs["ll"].values, cats); P["GBM + session history"] = np.expm1(g2.predict(Te[F2].values))
    P = {k: np.maximum(v, 0) for k, v in P.items()}
    R["ladder"] = {k: lat_metrics(y, v) for k, v in P.items()}
    slow = (y > 1000).astype(int); k_ = int(slow.sum()); R["slow_base_rate"] = float(slow.mean())
    sr = Tr.assign(s=(Tr.dur > 1000)).groupby("tfull")["s"].mean(); sc = {"per-tool slow rate": Te["tfull"].map(sr).fillna(float((Tr.dur > 1000).mean())).values}
    c1 = hgb_clf(Xs[CALL_TIME].values, (Xs.dur > 1000).astype(int).values, cats); sc["GBM call-time features"] = c1.predict_proba(Te[CALL_TIME].values)[:, 1]
    c2 = hgb_clf(Xs[F2].values, (Xs.dur > 1000).astype(int).values, cats); sc["GBM + session history"] = c2.predict_proba(Te[F2].values)[:, 1]
    # session-only score for reference: last-finished latency
    R["slow_clf"] = {}
    for k, s in sc.items():
        o = np.lexsort((rng.random(len(s)), -s))[:k_]; R["slow_clf"][k] = dict(auc=float(roc_auc_score(slow, s)), prec=float(slow[o].mean()))
    # --- bootstrap CI on key log-error contrasts (per-tool median vs GBM+history) by session
    ab = {k: np.abs(np.log1p(v) - np.log1p(y)) for k, v in P.items()}
    gcode = pd.factorize(Te["sid"])[0]; cnt = np.bincount(gcode); bsd = []
    for w in boot_weights(gcode, 200, 1):
        bsd.append(((ab["per-tool median (day 1)"] - ab["GBM + session history"]) * w).sum() / w.sum())
    R["logerr_gain_hist_vs_tool_median"] = dict(mean=float(np.mean(ab["per-tool median (day 1)"] - ab["GBM + session history"])), ci=[float(np.percentile(bsd, 2.5)), float(np.percentile(bsd, 97.5))])
    out += ["", f"Test = day 2 ({len(Te)} calls, {Te.sid.nunique()} sessions never seen in training); train = day 1; GBMs fitted on {len(fi)} random day-1 calls. Errors in ms; `MAE clip` clips true and predicted values at 1 h (a few multi-hour calls dominate raw MAE).",
            ""] + table(["predictor", "MAE ms", "MAE clip 1 h", "median abs err", "p90 abs err ms", "mean abs log1p-ratio"],
                        [[k, f"{v['mae']:.0f}", f"{v['mae_clip']:.0f}", f"{v['med']:.0f}", f"{v['p90']:.0f}", f"{v['logerr']:.2f}"] for k, v in R["ladder"].items()])
    out += ["", f"Log-error gain of GBM + history over per-tool median: {R['logerr_gain_hist_vs_tool_median']['mean']:.3f} (95% session-bootstrap CI {R['logerr_gain_hist_vs_tool_median']['ci'][0]:.3f} to {R['logerr_gain_hist_vs_tool_median']['ci'][1]:.3f}).",
            "", f"Will this call take > 1 s? Test base rate {pc(R['slow_base_rate'])} of calls. AUC, and precision when flagging as many calls as there are slow ones (k = {k_}):"] + \
        table(["predictor", "AUC", "precision at top-k"], [[k, f"{v['auc']:.3f}", pc(v["prec"], 0)] for k, v in R["slow_clf"].items()])
    # --- drift: random session hold-out inside day 1
    ss = Tr.sid.unique(); rs = np.random.default_rng(5); held = set(rs.choice(ss, len(ss) // 5, replace=False))
    hm = Tr.sid.isin(held).values; a, b = Tr[~hm], Tr[hm]
    sub = a.iloc[rs.choice(len(a), min(NFIT, len(a)), replace=False)]
    gd = hgb_reg(sub[CALL_TIME].values, sub["ll"].values, cats); pm = a.groupby("tfull")["dur"].median(); yb = b.dur.values.clip(min=0)
    gh = hgb_reg(sub[F2].values, sub["ll"].values, cats)
    R["drift_day1_holdout"] = dict(tool_median=lat_metrics(yb, b.tfull.map(pm).fillna(glob).values), gbm_static=lat_metrics(yb, np.expm1(gd.predict(b[CALL_TIME].values))), gbm_hist=lat_metrics(yb, np.expm1(gh.predict(b[F2].values))))
    out += ["", f"Same-day random hold-out of 20% of day-1 sessions ({len(b)} calls) for comparison, mean abs log1p-ratio: per-tool median {R['drift_day1_holdout']['tool_median']['logerr']:.2f}, GBM call-time {R['drift_day1_holdout']['gbm_static']['logerr']:.2f}, GBM + history {R['drift_day1_holdout']['gbm_hist']['logerr']:.2f} (day-1 to day-2 values are in the table above)."]
    # --- by model (top 5 by test volume): per-tool median vs GBM+hist
    tm = Te["model"].value_counts().index[:5]; rows = []
    for mname in tm:
        mk = (Te["model"] == mname).values
        rows.append([mname, mk.sum(), f"{np.median(y[mk]):.0f}", f"{ab['per-tool median (day 1)'][mk].mean():.2f}", f"{ab['GBM, call-time features only'][mk].mean():.2f}", f"{ab['GBM + session history'][mk].mean():.2f}"])
    out += ["", "By model (top 5 by test calls), mean abs log1p-ratio:"] + table(["model", "test calls", "median ms", "per-tool median", "GBM call-time", "GBM + history"], rows)
    write_md("F", out); save("F", R); say("F done")
    return A


# ------------------------------------------------------------------ E  next-call speculation on tool names
NE = 25  # classes: top-24 tools of day 1 + OTHER (index 24); index 25 = 'no previous call' padding in contexts


def ngram_tables(ctx, y, nctx):
    C = np.zeros((nctx, NE)); np.add.at(C, (ctx, y), 1); return C


def ngram_pred(C, ctx, min_n):
    """Counts -> (top1, p1, top3 (m,3), n_ctx). OTHER is never predicted (its mass stays in the denominator)."""
    n = C.sum(1); Cp = C.copy(); Cp[:, NE - 1] = -1
    o = np.argsort(-Cp, axis=1)[:, :3]; top1 = o[:, 0]; p1 = np.where(n > 0, C[np.arange(len(C)), top1] / np.maximum(n, 1), 0)
    return top1[ctx], p1[ctx], o[ctx], n[ctx]


def acc13(y, top3): return float((top3[:, 0] == y).mean()), float((top3 == y[:, None]).any(1).mean())


def stage_E(A):
    out = []; R = {}
    d1n = A.loc[A.day == DAYS[0], "tool"].value_counts().index[:NE - 1]; vm = {t: i for i, t in enumerate(d1n)}
    e = A["tool"].map(vm).fillna(NE - 1).astype(int).values; sc = A["sidc"].values; n = len(A)
    same1 = np.r_[False, sc[1:] == sc[:-1]]; same2 = np.r_[False, False, (sc[2:] == sc[:-2])]
    prev1 = np.where(same1, np.r_[0, e[:-1]], NE); prev2 = np.where(same2, np.r_[0, 0, e[:-2]], NE)
    hasnext = np.r_[sc[1:] == sc[:-1], False]; nxt = np.r_[e[1:], 0]
    turn = A["turn"].values; bidx = A["bidx"].values; nturn = np.r_[turn[1:], -1]; nbidx = np.r_[bidx[1:], -1]
    new_turn = hasnext & (nturn != turn); same_batch = hasnext & ~new_turn & (nbidx == bidx)
    ptype = np.where(new_turn, 0, np.where(bidx == 0, 1, 2))  # 0 next call opens a new turn (after a user message); 1 inside a turn, current call in its first batch; 2 later batch
    tr = (A["day"] == DAYS[0]).values & hasnext; te = (A["day"] == DAYS[1]).values & hasnext
    R["n_train_pos"] = int(tr.sum()); R["n_test_pos"] = int(te.sum())
    tail = float((nxt[te] == NE - 1).mean()); R["tail_share"] = tail
    yte = nxt[te]
    # (a) global most common
    g = np.bincount(nxt[tr], minlength=NE).astype(float); gt = g.copy(); gt[NE - 1] = -1; g3 = np.argsort(-gt)[:3]
    pg = np.tile(g3, (te.sum(), 1)); preds = {"(a) most common next tool overall": (pg, np.full(te.sum(), g[g3[0]] / g.sum()))}
    # (b) bigram / trigram with backoff
    C1 = ngram_tables(e[tr], nxt[tr], NE + 1); C2 = ngram_tables((prev1 * (NE + 1) + e)[tr], nxt[tr], (NE + 1) ** 2)
    t1, p1, o1, n1 = ngram_pred(C1, e[te], 1); t2, p2, o2, n2 = ngram_pred(C2, (prev1 * (NE + 1) + e)[te], 1)
    use3 = n2 >= 5; tri_o = np.where(use3[:, None], o2, o1); tri_p = np.where(use3, p2, p1)
    preds["(b) bigram (last tool)"] = (o1, p1); preds["(b) trigram (last 2 tools, backoff to bigram if < 5 seen)"] = (tri_o, tri_p)
    # (b') 4-gram
    ctx4 = (prev2 * (NE + 1) + prev1) * (NE + 1) + e; C3 = ngram_tables(ctx4[tr], nxt[tr], (NE + 1) ** 3)
    t3, p3, o3, n3 = ngram_pred(C3, ctx4[te], 1); use4 = n3 >= 5
    preds["(b) 4-gram (last 3 tools, backoff trigram then bigram)"] = (np.where(use4[:, None], o3, tri_o), np.where(use4, p3, tri_p))
    # (c) GBM on last 3 tool names + status + position + turn + model + batch size
    X = np.c_[e, prev1, prev2, A["status"].values, A["gpos"].values, turn, A["mcat"].values, A["bsize"].values, A["k"].values, bidx].astype(float)
    rng = np.random.default_rng(0); itr = np.flatnonzero(tr); fi = rng.choice(itr, min(200_000, len(itr)), replace=False)
    say("E: fitting multiclass GBM"); t_ = time.time()
    gb = HistGradientBoostingClassifier(max_iter=60, learning_rate=0.15, max_leaf_nodes=31, min_samples_leaf=50, categorical_features=[0, 1, 2, 6], random_state=0).fit(X[fi], nxt[fi])
    Pg = gb.predict_proba(X[te]); cl = gb.classes_; R["gbm_fit_s"] = time.time() - t_
    full = np.zeros((te.sum(), NE)); full[:, cl] = Pg; Pm = full.copy(); Pm[:, NE - 1] = -1
    og = np.argsort(-Pm, axis=1)[:, :3]; preds["(c) GBM (last 3 tools, status, position, turn, model, batch size)"] = (og, full[np.arange(te.sum()), og[:, 0]])
    R["train_sample_gbm"] = len(fi)
    # --- main table
    names = list(preds); rows = []; R["acc"] = {}
    def sub(mask):
        res = {}
        for k in names:
            o, p = preds[k]; a1, a3 = acc13(yte[mask], o[mask]); res[k] = (a1, a3)
        return res
    allm = np.ones(te.sum(), bool); pt = ptype[te]; mod = A["model"].values[te]
    groups = [("all positions", allm), ("next call opens a new turn (after a user message)", pt == 0), ("inside a turn, current call in the first batch", pt == 1), ("inside a turn, later batch", pt == 2),
              ("next call in the same batch (parallel)", same_batch[te])]
    for m_ in pd.Series(mod).value_counts().index[:5]: groups.append((f"model {m_}", mod == m_))
    for gname, mk in groups:
        r = sub(mk); R["acc"][gname] = dict(n=int(mk.sum()), **{k: v for k, v in r.items()})
    out += [f"Positions: every tool call that has a following call in the same session. Train {R['n_train_pos']} positions (day 1), test {R['n_test_pos']} (day 2). Target = next call's tool name, vocabulary = 24 most common day-1 tools + OTHER ({pc(tail)} of test targets are OTHER and count as misses; OTHER is never predicted). "
            f"There are 855 distinct names on day 1 but the top 24 carry {pc(1-float((nxt[tr]==NE-1).mean()))} of next calls. The GBM is fitted on {len(fi)} random day-1 positions."]
    for gname, mk in groups:
        r = R["acc"][gname]
        rows.append([gname, r["n"]] + [f"{100*r[k][0]:.1f} / {100*r[k][1]:.1f}" for k in names])
    out += ["", "Top-1 / top-3 accuracy (%):"] + table(["positions", "n"] + ["(a) most common tool", "(b) bigram", "(b) trigram", "(b) 4-gram", "(c) GBM"], rows)
    # --- gate
    rows = []; R["gate"] = {}
    for k in names[1:]:
        o, p = preds[k]; hit = o[:, 0] == yte
        for tau in (0.5, 0.7, 0.9):
            L = p >= tau; R["gate"][f"{k}|{tau}"] = dict(launched=float(L.mean()), precision=float(hit[L].mean()) if L.any() else None, hits_per_position=float((hit & L).mean()))
            rows.append([k.split(") ")[1] if ") " in k else k, tau, pc(L.mean()), pc(hit[L].mean()) if L.any() else "n/a", pc((hit & L).mean())])
    out += ["", "Gate on the predicted probability of the top-1 tool (launch the speculation only if p >= tau). n-gram p = count share of the top tool in its context; GBM p = predicted probability. `correct launches / all positions` is the share of all positions where a correct speculation was launched:"] + \
        table(["predictor", "tau", "launched share", "precision of launched", "correct launches / all positions"], rows)
    for gname, mk in groups[:4]:
        o, p = preds[names[2]]; L = (p >= 0.7) & mk; hit = o[:, 0] == yte
        R["gate"][f"trigram0.7|{gname}"] = dict(launched=float(L.sum() / mk.sum()), precision=float(hit[L].mean()) if L.any() else None)
    out += ["", "Trigram gate at tau 0.7 by position type: " + "; ".join(f"{g_}: launched {pc(R['gate'][f'trigram0.7|{g_}']['launched'])}, precision {pc(R['gate'][f'trigram0.7|{g_}']['precision'])}" if R['gate'][f'trigram0.7|{g_}']['precision'] is not None else f"{g_}: none launched" for g_, _ in groups[:4])]
    # --- batch level
    df = pd.DataFrame(dict(sidc=sc, turn=turn, bidx=bidx, k=A["k"].values, e=e, day=A["day"].values, bsize=A["bsize"].values))
    df["bid"] = df.groupby(["sidc", "turn", "bidx"], sort=False).ngroup()
    df["e0"] = df.groupby("bid")["e"].transform("first"); df["eprev"] = df.groupby("bid")["e"].shift(1)
    m = df[df.k >= 1].copy(); mtr, mte = m[m.day == DAYS[0]], m[m.day == DAYS[1]]; kk = lambda s: np.minimum(s, 4)
    Bn = {}
    gu = np.bincount(mtr.e, minlength=NE).astype(float); gu[NE - 1] = -1; u3 = np.argsort(-gu)[:3]; Bn["unconditional (mode of non-first slots)"] = np.tile(u3, (len(mte), 1))
    Cb = ngram_tables(mtr.eprev.astype(int).values, mtr.e.values, NE + 1); Bn["P(tool given previous tool in the batch)"] = ngram_pred(Cb, mte.eprev.astype(int).values, 1)[2]
    cx = lambda d: d.e0.values * 5 + kk(d.k.values) - 1
    Cf = ngram_tables(cx(mtr), mtr.e.values, NE * 5); Bn["P(tool given first tool and slot index)"] = ngram_pred(Cf, cx(mte), 1)[2]
    Bn["rest identical to the first tool"] = np.c_[mte.e0.values, np.full(len(mte), -1), np.full(len(mte), -1)]
    ym = mte.e.values; cls = np.where(mte.bsize.values == 2, "batch of 2", np.where(mte.bsize.values <= 4, "batch of 3-4", "batch of 5+"))
    rows = []; R["batch"] = {}
    for k_, o in Bn.items():
        a1, a3 = acc13(ym, o); row = [k_, f"{100*a1:.1f} / {100*a3:.1f}"]
        for c_ in ("batch of 2", "batch of 3-4", "batch of 5+"):
            mk = cls == c_; b1, b3 = acc13(ym[mk], o[mk]); row.append(f"{100*b1:.1f} / {100*b3:.1f}")
        rows.append(row); R["batch"][k_] = row[1:]
    nb = df.drop_duplicates("bid"); nbte = nb[(nb.day == DAYS[1])]
    R["batch_frac_multi"] = float((nbte.bsize >= 2).mean()); mt = df[(df.day == DAYS[1]) & (df.bsize >= 2)]
    homog = mt.groupby("bid")["e"].nunique().eq(1).mean(); R["batch_homog"] = float(homog)
    # exact whole-rest match given first tool + batch size (argmax per slot, order-sensitive and as multiset)
    pred_slot = Bn["P(tool given first tool and slot index)"][:, 0]; mte2 = mte.assign(pr=pred_slot, ok=(pred_slot == ym))
    ex = mte2.groupby("bid")["ok"].all(); R["batch_exact_rest"] = float(ex.mean())
    base = (mte2.e0 == mte2.e).groupby(mte2.bid).all(); R["batch_exact_rest_same_as_first"] = float(base.mean())
    out += ["", f"Batch level (day 2; {pc(R['batch_frac_multi'])} of {len(nbte)} batches have 2+ tools; among those, {pc(R['batch_homog'])} contain a single distinct tool). Given the first tool of a batch (and the fact that the batch has a slot there, i.e. its size is at least the slot index; the size is also used to split the columns), predict each of the other slots. Top-1 / top-3 accuracy (%) over all non-first slots ({len(mte)}):"] + \
        table(["predictor", "all non-first slots", "batch of 2", "batch of 3-4", "batch of 5+"], rows)
    out += [f"", f"All non-first slots of a batch right (argmax of P(tool given first tool and slot index)): {pc(R['batch_exact_rest'])} of multi-tool batches; baseline 'rest identical to the first tool': {pc(R['batch_exact_rest_same_as_first'])}."]
    write_md("E", out); save("E", R); say("E done")


# ------------------------------------------------------------------ D  failure prediction
CTX = ["nfail", "nfin", "lastfail", "st_nfail", "st_cnt", "l1", "frate"]


def design_lr(d, tools_top, mods):
    from scipy import sparse
    oh_t = sparse.csr_matrix((np.ones(len(d)), (np.arange(len(d)), d["tcat"].values)), shape=(len(d), TOPK_TOOLS + 1))
    oh_m = sparse.csr_matrix((np.ones(len(d)), (np.arange(len(d)), d["mcat"].values.clip(min=0))), shape=(len(d), mods))
    num = np.c_[np.log1p(d.gpos.values), np.log1p(d.bsize.values), d.k.values / 4.0, np.log1p(d.bidx.values), np.log1p(d.turn.values), np.nan_to_num(np.log1p(d.prompt.values), nan=0.0) / 10,
                np.sin(2 * np.pi * d.hour.values / 24), np.cos(2 * np.pi * d.hour.values / 24)]
    return sparse.hstack([oh_t, oh_m, sparse.csr_matrix(num)]).tocsr()


def design_lr_ctx(d, mods):
    from scipy import sparse
    c = np.c_[np.log1p(d.nfail.values), np.log1p(d.nfin.values), (d.lastfail.values == 1), (d.lastfail.values == 0), np.log1p(d.st_nfail.values), np.log1p(d.st_cnt.values), np.nan_to_num(d.l1.values, nan=0.0) / 5, d.frate.fillna(0).values]
    return sparse.hstack([design_lr(d, None, mods), sparse.csr_matrix(c.astype(float))]).tocsr()


def stage_D(A):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    out = []; R = {}
    A, glob, med_tool = add_hist(A); A["frate"] = A["nfail"] / A["nfin"].replace(0, np.nan)
    s = A["status"].value_counts(normalize=True).to_dict(); R["status_share"] = {str(k): float(v) for k, v in s.items()}
    D = A[A["status"].isin([1, 2])].reset_index(drop=True); nmod = int(A["mcat"].max()) + 1
    d1 = D[D.day == DAYS[0]].reset_index(drop=True); d2 = D[D.day == DAYS[1]].reset_index(drop=True)
    R["fail_rate"] = {DAYS[0]: float(d1.fail.mean()), DAYS[1]: float(d2.fail.mean())}
    out.append(f"Status shares over all tool calls: success (1) {pc(s.get(1,0))}, failure (2) {pc(s.get(2,0))}, status 0 {pc(s.get(0,0))} (not described in the schema; its durations are short, median 32 ms; concentrated in run_command_in_terminal and run_build). "
               f"Status-0 calls are dropped from every D' label set but stay in the history features (as non-failures). "
               f"Failure = status 2 among status 1/2 calls: day 1 {pc(R['fail_rate'][DAYS[0]],2)} of {len(d1)}, day 2 {pc(R['fail_rate'][DAYS[1]],2)} of {len(d2)}.")
    fs = d1.groupby("sid")["fail"].sum().sort_values(ascending=False); R["fail_conc_top5pct_sessions"] = float(fs.head(int(0.05 * len(fs))).sum() / fs.sum())
    lf = d1.groupby(d1.lastfail)["fail"].agg(["mean", "size"]); R["fail_by_lastfail"] = {str(k): [float(v["mean"]), int(v["size"])] for k, v in lf.iterrows()}
    out.append(f"Failures are clustered: the top 5% of day-1 sessions by failure count hold {pc(R['fail_conc_top5pct_sessions'])} of all failures. Failure rate of a call by status of the latest finished call in its session (day 1): " +
               ", ".join(f"{ {'-1.0':'none yet','0.0':'success','1.0':'failure'}.get(k,k)} {pc(v[0],2)} (n={v[1]})" for k, v in R["fail_by_lastfail"].items()) + ".")
    cats = [0, 1]; F1 = CALL_TIME; F2 = CALL_TIME + CTX
    def fit_predict(trn, tst, seed):
        rng = np.random.default_rng(seed); sub = trn.iloc[rng.choice(len(trn), min(NFIT, len(trn)), replace=False)]
        P = {}
        pr = trn["fail"].mean(); g = trn.groupby("tfull")["fail"].agg(["sum", "count"]); rate = (g["sum"] + 5 * pr) / (g["count"] + 5)
        P["per-tool base rate"] = tst["tfull"].map(rate).fillna(pr).values
        lr1 = LogisticRegression(C=1.0, max_iter=300).fit(design_lr(sub, None, nmod), sub.fail.values); P["LR call-time features"] = lr1.predict_proba(design_lr(tst, None, nmod))[:, 1]
        lr2 = LogisticRegression(C=1.0, max_iter=300).fit(design_lr_ctx(sub, nmod), sub.fail.values); P["LR + session context"] = lr2.predict_proba(design_lr_ctx(tst, nmod))[:, 1]
        c1 = hgb_clf(sub[F1].values, sub.fail.values, cats, 150); P["GBM call-time features"] = c1.predict_proba(tst[F1].values)[:, 1]
        c2 = hgb_clf(sub[F2].values, sub.fail.values, cats, 150); P["GBM + session context"] = c2.predict_proba(tst[F2].values)[:, 1]
        return P
    # (1) session-grouped 5-fold on day 1
    oof = {}; gk = GroupKFold(n_splits=5); gcode = pd.factorize(d1["sid"])[0]
    for f, (a, b) in enumerate(gk.split(d1, groups=gcode)):
        P = fit_predict(d1.iloc[a], d1.iloc[b], f)
        for k, v in P.items(): oof.setdefault(k, np.zeros(len(d1)))[b] = v
        say(f"D: fold {f} done")
    r1 = boot_binary(d1.fail.values, oof, gcode, "per-tool base rate", 0.2, 200, 0)
    # (2) chronological day 1 -> day 2
    P2 = fit_predict(d1, d2, 99); g2 = pd.factorize(d2["sid"])[0]
    r2 = boot_binary(d2.fail.values, P2, g2, "per-tool base rate", 0.2, 200, 1)
    R["cv5_day1"] = r1; R["chrono_day2"] = r2
    f = lambda r, k: f"{r[k]['auc']:.3f} ({r[k]['auc_ci'][0]:.3f}-{r[k]['auc_ci'][1]:.3f})"
    dd = lambda r, k: "-" if r[k]["dauc_ci"] is None else f"{r[k]['dauc']:+.3f} ({r[k]['dauc_ci'][0]:+.3f} to {r[k]['dauc_ci'][1]:+.3f})"
    pr = lambda r, k: f"{pc(r[k]['prec'])} ({pc(r[k]['prec_ci'][0])}-{pc(r[k]['prec_ci'][1])}) / {pc(r[k]['rec'])} ({pc(r[k]['rec_ci'][0])}-{pc(r[k]['rec_ci'][1])})"
    for title, r, d in (("Session-grouped 5-fold on day 1", r1, d1), ("Chronological: train day 1, test day 2", r2, d2)):
        out += ["", f"{title} ({len(d)} calls, {d.sid.nunique()} sessions, failure base rate {pc(d.fail.mean(),2)}); AUC with 95% session-bootstrap CI (200 resamples), AUC difference to the per-tool base rate (paired), precision / recall when the top 20% of calls by score are flagged (random ties):"] + \
            table(["predictor", "AUC", "AUC gain vs per-tool rate", "precision / recall at top 20%"], [[k, f(r, k), dd(r, k), pr(r, k)] for k in r])
    # per-tool
    rows = []; R["per_tool"] = {}
    for t in D["tool"].value_counts().index[:15]:
        x = D[D.tool == t]; xs = A[(A.tool == t)]
        rows.append([t, len(x), pc(x.fail.mean(), 2), int(x.fail.sum()), f"{x.loc[x.fail == 0, 'dur'].median():.0f}", f"{x.loc[x.fail == 1, 'dur'].median():.0f}" if x.fail.sum() else "-", pc((xs.status == 0).mean(), 1)])
        R["per_tool"][t] = rows[-1][1:]
    out += ["", "Per-tool failure rate (status 2 among status 1/2 calls), top 15 tools by calls (both days):"] + table(["tool", "calls", "failure rate", "failures", "median ms (success)", "median ms (failure)", "% status 0"], rows)
    write_md("D", out); save("D", R); say("D done")


# ------------------------------------------------------------------ S  structure stats
def stage_S(A):
    out = []; R = {}
    L = pd.concat([pd.read_parquet(FLAT / f"{d}_llm.parquet").assign(day=d) for d in DAYS], ignore_index=True)
    Bt = pd.concat([pd.read_parquet(FLAT / f"{d}_batches.parquet").assign(day=d) for d in DAYS], ignore_index=True)
    for d in list(DAYS) + ["pooled"]:
        a = A if d == "pooled" else A[A.day == d]; l = L if d == "pooled" else L[L.day == d]; b = Bt if d == "pooled" else Bt[Bt.day == d]
        r = {}; sc = a.sidc.values; tf = a.tfull.values
        has = np.r_[False, sc[1:] == sc[:-1]]; eq = np.r_[False, tf[1:] == tf[:-1]]
        r["repeat_prev"] = float(eq[has].mean())
        sb = np.r_[False, (a.bidx.values[1:] == a.bidx.values[:-1]) & (a.turn.values[1:] == a.turn.values[:-1])] & has
        r["repeat_prev_within_batch"] = float(eq[sb].mean()); r["repeat_prev_across_batch"] = float(eq[has & ~sb].mean()); r["share_pairs_in_same_batch"] = float(sb[has].mean())
        run_id = np.cumsum(~(has & eq)); rl = np.bincount(run_id)[1:]; rl = rl[rl > 0]
        r["run_len_mean"] = float(rl.mean()); r["run_len_median"] = float(np.median(rl)); r["run_len_p90"] = float(np.percentile(rl, 90)); r["run_len_max"] = int(rl.max())
        r["calls_in_runs_ge3"] = float(rl[rl >= 3].sum() / rl.sum()); r["calls_in_runs_ge10"] = float(rl[rl >= 10].sum() / rl.sum())
        bs = a.drop_duplicates(["sidc", "turn", "bidx"])["bsize"]; r["batches"] = int(len(bs)); r["batch_size_mean"] = float(bs.mean())
        r["batch_size_dist"] = {str(k): float((bs == k).mean()) if k < 5 else float((bs >= 5).mean()) for k in range(1, 6)}
        r["calls_per_session_mean"] = float(a.groupby("sidc").size().mean()); r["calls_per_session_median"] = float(a.groupby("sidc").size().median()); r["calls_per_session_p90"] = float(a.groupby("sidc").size().quantile(.9))
        cpt = a.groupby(["sidc", "turn"]).size(); r["calls_per_turn_mean"] = float(cpt.mean()); r["calls_per_turn_median"] = float(cpt.median()); r["calls_per_turn_p90"] = float(cpt.quantile(.9))
        lt = l.groupby(["sid", "turn"]).size(); r["turns"] = int(len(lt)); r["llm_calls_per_turn_mean"] = float(lt.mean()); r["llm_calls_per_turn_median"] = float(lt.median())
        r["turns_with_zero_tool_calls"] = float(1 - len(cpt) / len(lt)) if d != "pooled" else None
        r["turns_per_session_mean"] = float(l.groupby("sid").turn.nunique().mean())
        r["llm_calls"] = int(len(l)); r["llm_result"] = {k: float(v) for k, v in l.result.value_counts(normalize=True).items()}
        ok = l[l.prompt.notna() & (l.prompt > 0)]; r["token_rows_share"] = float(len(ok) / len(l)); r["cache_hit_token_weighted"] = float(ok.cached.sum() / ok.prompt.sum())
        ag = ok[ok.init == "agent"]; us = ok[ok.init == "user"]
        r["cache_hit_agent_calls"] = float(ag.cached.sum() / ag.prompt.sum()); r["cache_hit_user_calls"] = float(us.cached.sum() / us.prompt.sum())
        r["calls_with_any_cache_hit_agent"] = float((ag.cached > 0).mean()); r["prompt_tokens_median"] = float(ok.prompt.median()); r["completion_tokens_median"] = float(ok.completion.median())
        r["agent_initiated_llm_share"] = float((l.init == "agent").mean())
        mm = a.model.value_counts(normalize=True); r["n_models"] = int(a.model.nunique()); r["model_mix_top"] = {k: float(v) for k, v in mm.head(8).items()}; r["model_top5_share"] = float(mm.head(5).sum())
        dd = a.dur.clip(lower=0); by = dd.groupby(a.tool).sum().sort_values(ascending=False); r["top3_time_tools"] = list(by.index[:3]); r["top3_time_share"] = float(by.head(3).sum() / by.sum())
        bc = a.tool.value_counts(); r["top3_call_tools"] = list(bc.index[:3]); r["top3_call_share"] = float(bc.head(3).sum() / bc.sum()); r["top10_call_share"] = float(bc.head(10).sum() / bc.sum())
        r["n_tools"] = int(a.tool.nunique()); r["sessions_with_calls"] = int(a.sidc.nunique()); r["calls"] = int(len(a))
        R[d] = r
    p, r1, r2 = R["pooled"], R[DAYS[0]], R[DAYS[1]]
    rows = [
        ["tool calls / sessions with a tool call / distinct tool names", *[f"{x['calls']} / {x['sessions_with_calls']} / {x['n_tools']}" for x in (r1, r2)]],
        ["LLM calls / turns", *[f"{x['llm_calls']} / {x['turns']}" for x in (r1, r2)]],
        ["calls per session: mean / median / p90", *[f"{x['calls_per_session_mean']:.1f} / {x['calls_per_session_median']:.0f} / {x['calls_per_session_p90']:.0f}" for x in (r1, r2)]],
        ["turns per session (mean)", *[f"{x['turns_per_session_mean']:.2f}" for x in (r1, r2)]],
        ["tool calls per turn (turns with >= 1 call): mean / median / p90", *[f"{x['calls_per_turn_mean']:.1f} / {x['calls_per_turn_median']:.0f} / {x['calls_per_turn_p90']:.0f}" for x in (r1, r2)]],
        ["LLM calls per turn: mean / median", *[f"{x['llm_calls_per_turn_mean']:.1f} / {x['llm_calls_per_turn_median']:.0f}" for x in (r1, r2)]],
        ["turns with no tool call", *[pc(x["turns_with_zero_tool_calls"]) for x in (r1, r2)]],
        ["batch size: 1 / 2 / 3 / 4 / 5+ (share of batches)", *[" / ".join(f"{100*x['batch_size_dist'][str(k)]:.1f}" for k in range(1, 6)) for x in (r1, r2)]],
        ["mean batch size", *[f"{x['batch_size_mean']:.2f}" for x in (r1, r2)]],
        ["call has the same tool as the previous call in the session", *[pc(x["repeat_prev"]) for x in (r1, r2)]],
        ["  ... when the previous call is in the same batch / in an earlier batch", *[f"{pc(x['repeat_prev_within_batch'])} / {pc(x['repeat_prev_across_batch'])}" for x in (r1, r2)]],
        ["  share of consecutive pairs that are inside one batch", *[pc(x["share_pairs_in_same_batch"]) for x in (r1, r2)]],
        ["runs of the same tool: mean / median / p90 / max length", *[f"{x['run_len_mean']:.2f} / {x['run_len_median']:.0f} / {x['run_len_p90']:.0f} / {x['run_len_max']}" for x in (r1, r2)]],
        ["share of calls inside runs of length >= 3 / >= 10", *[f"{pc(x['calls_in_runs_ge3'])} / {pc(x['calls_in_runs_ge10'])}" for x in (r1, r2)]],
        ["distinct models; top-5 share of tool calls", *[f"{x['n_models']}; {pc(x['model_top5_share'])}" for x in (r1, r2)]],
        ["agent-initiated share of LLM calls", *[pc(x["agent_initiated_llm_share"]) for x in (r1, r2)]],
        ["LLM calls with token counts", *[pc(x["token_rows_share"]) for x in (r1, r2)]],
        ["cache hit share of prompt tokens: all / agent calls / user-turn calls", *[f"{pc(x['cache_hit_token_weighted'])} / {pc(x['cache_hit_agent_calls'])} / {pc(x['cache_hit_user_calls'])}" for x in (r1, r2)]],
        ["agent LLM calls with any cached token", *[pc(x["calls_with_any_cache_hit_agent"]) for x in (r1, r2)]],
        ["median prompt / completion tokens per LLM call", *[f"{x['prompt_tokens_median']:.0f} / {x['completion_tokens_median']:.0f}" for x in (r1, r2)]],
        ["top-3 tools by calls; share of calls (top 10 share)", *[f"{', '.join(x['top3_call_tools'])}; {pc(x['top3_call_share'])} ({pc(x['top10_call_share'])})" for x in (r1, r2)]],
        ["top-3 tools by summed tool time; share of tool time", *[f"{', '.join(x['top3_time_tools'])}; {pc(x['top3_time_share'])}" for x in (r1, r2)]],
        ["LLM call results", *[", ".join(f"{k} {pc(v,2)}" for k, v in x["llm_result"].items()) for x in (r1, r2)]],
    ]
    out += ["Structure of the corpus (day 1 = 2026-06-01, day 2 = 2026-06-02; numbers for the pooled two days are in results.json):"] + table(["quantity", "day 1", "day 2"], rows)
    out += ["", "Model mix (share of tool calls), day 1: " + ", ".join(f"{k} {pc(v)}" for k, v in r1["model_mix_top"].items()) + ". Day 2: " + ", ".join(f"{k} {pc(v)}" for k, v in r2["model_mix_top"].items()) + "."]
    write_md("S", out); save("S", R); say("S done")


# ------------------------------------------------------------------ T  time shares + speculation upper bound
def union_len(s, e):
    if len(s) == 0: return 0.0
    o = np.argsort(s, kind="stable"); s, e = s[o], e[o]; pm = np.r_[-np.inf, np.maximum.accumulate(e)[:-1]]
    return float(np.maximum(0.0, e - np.maximum(s, pm)).sum())


def stage_T(A):
    out = []; R = {}
    for d in DAYS:
        L = pd.read_parquet(FLAT / f"{d}_llm.parquet"); Bt = pd.read_parquet(FLAT / f"{d}_batches.parquet")
        L["s"] = L["end"] - L["dur"]; Bt["e"] = Bt["bstart"] + Bt["bdur"]
        # next LLM call after the one that issued each batch (same turn)
        nx = Bt[Bt.lidx >= 0].merge(L.assign(lidx=L.lidx - 1)[["sid", "turn", "lidx", "s", "dur", "end"]].rename(columns={"s": "ns", "dur": "nd", "end": "ne"}), on=["sid", "turn", "lidx"])
        nx["save19"] = np.minimum(nx.bdur, nx.nd).clip(lower=0); nx["save_loose"] = (nx.ns - nx.bstart).clip(lower=0)
        nx["save_ceiling"] = nx[["save19", "save_loose"]].min(axis=1) if False else nx["save19"]
        # per turn busy time
        L = L.sort_values(["sid", "turn", "s"]); Bt = Bt.sort_values(["sid", "turn", "bstart"])
        gl = L.groupby(["sid", "turn"], sort=False); gb = Bt.groupby(["sid", "turn"], sort=False)
        lb = {k: (v["s"].values, v["end"].values) for k, v in gl}; bb = {k: (v["bstart"].values, v["e"].values) for k, v in gb}
        rows = []
        for k, (ls, le) in lb.items():
            bs, be = bb.get(k, (np.empty(0), np.empty(0)))
            lo = min(ls.min(), bs.min() if len(bs) else np.inf); hi = max(le.max(), be.max() if len(be) else -np.inf)
            ul = union_len(ls, le); ub = union_len(bs, be); ua = union_len(np.r_[ls, bs], np.r_[le, be])
            rows.append((k[0], k[1], hi - lo, ul, ub, ua, lo, hi, float((be - bs).max()) if len(bs) else 0.0))
        T = pd.DataFrame(rows, columns=["sid", "turn", "span", "llm", "tool", "union", "lo", "hi", "bmax"])
        T = T[T.span > 0]
        T = T.merge(nx.groupby(["sid", "turn"])[["save19", "save_loose"]].sum().reset_index(), on=["sid", "turn"], how="left").fillna({"save19": 0, "save_loose": 0})
        T["both"] = T.llm + T.tool - T.union; T["gap"] = (T.span - T.union).clip(lower=0)
        S = T.groupby("sid").agg(span=("span", "sum"), llm=("llm", "sum"), tool=("tool", "sum"), both=("both", "sum"), gap=("gap", "sum"), lo=("lo", "min"), hi=("hi", "max"), turns=("turn", "size"))
        S["wall"] = S.hi - S.lo; S["idle"] = (S.wall - S.span).clip(lower=0)
        sv = nx.groupby("sid")[["save19", "save_loose"]].sum().reindex(S.index).fillna(0)
        S = S.join(sv)
        r = dict(turns=int(len(T)), sessions=int(len(S)))
        tot = S.span.sum(); r["active_time_h"] = float(tot / 3.6e6)
        r["shares_of_active_turn_time"] = dict(llm=float(S.llm.sum() / tot), tool_batches=float(S.tool.sum() / tot), overlap_llm_tool=float(S.both.sum() / tot), gap_neither=float(S.gap.sum() / tot))
        r["exclusive"] = dict(llm_only=float((S.llm.sum() - S.both.sum()) / tot), tool_only=float((S.tool.sum() - S.both.sum()) / tot))
        r["session_wall_incl_user_idle_h"] = float(S.wall.sum() / 3.6e6); r["idle_between_turns_share_of_session_wall"] = float(S.idle.sum() / S.wall.sum())
        r["per_session_share_median"] = dict(llm=float((S.llm / S.span).median()), tool=float((S.tool / S.span).median()), gap=float((S.gap / S.span).median()))
        r["per_session_share_p10_p90_tool"] = [float((S.tool / S.span).quantile(.1)), float((S.tool / S.span).quantile(.9))]
        r["upper_bound_save_S19_rule_of_active_time"] = float(S.save19.sum() / tot); r["upper_bound_save_S19_rule_of_session_wall"] = float(S.save19.sum() / S.wall.sum())
        r["upper_bound_save_loose_of_active_time"] = float(S.save_loose.sum() / tot)
        ps = (S.save19 / S.span); r["upper_bound_per_session_S19_median_p90"] = [float(ps.median()), float(ps.quantile(.9))]
        r["batches_with_next_llm"] = int(len(nx))
        # capped: batches over 1 h removed (outliers such as a call that sat for days)
        nz = nx[nx.bdur < 3.6e6]; r["upper_bound_save_S19_batches_under_1h_of_active_time"] = float(nz.save19.sum() / tot)
        Rb = T[T.bmax < 3.6e6]; tb = Rb.span.sum()
        r["robust_excl_turns_with_batch_over_1h"] = dict(turns_kept=float(len(Rb) / len(T)), time_kept=float(tb / T.span.sum()), llm=float(Rb.llm.sum() / tb), tool=float(Rb.tool.sum() / tb), gap=float(Rb.gap.sum() / tb),
                                                         save19=float(Rb.save19.sum() / tb), save_loose=float(Rb.save_loose.sum() / tb))
        R[d] = r
        # by model of the issuing call is not stored here; skip
        say(f"T {d}: {r['shares_of_active_turn_time']} bound {r['upper_bound_save_S19_rule_of_active_time']:.4f}")
    rows = []
    for nm, f in (("LLM calls (union of intervals)", lambda r: r["shares_of_active_turn_time"]["llm"]), ("tool batches (union)", lambda r: r["shares_of_active_turn_time"]["tool_batches"]),
                  ("both at once (overlap)", lambda r: r["shares_of_active_turn_time"]["overlap_llm_tool"]), ("neither (gaps inside a turn: client work, waits)", lambda r: r["shares_of_active_turn_time"]["gap_neither"]),
                  ("only LLM", lambda r: r["exclusive"]["llm_only"]), ("only tool", lambda r: r["exclusive"]["tool_only"]),
                  ("per-session median: LLM / tool / gap", lambda r: " / ".join(pc(r["per_session_share_median"][k]) for k in ("llm", "tool", "gap")))):
        rows.append([nm] + [pc(f(R[d])) if not isinstance(f(R[d]), str) else f(R[d]) for d in DAYS])
    out += ["Time split inside turns (a turn runs from its first LLM call's start to its last event's end; between-turn user idle time is excluded here). Shares are of summed active turn time; per-session rows use each session's own turn time:"] + table(["share of active turn time", "day 1", "day 2"], rows)
    rows = [["active turn time (hours) / sessions / turns"] + [f"{R[d]['active_time_h']:.0f} / {R[d]['sessions']} / {R[d]['turns']}" for d in DAYS],
            ["between-turn user idle, share of session wall clock"] + [pc(R[d]["idle_between_turns_share_of_session_wall"]) for d in DAYS],
            ["per-session tool share of turn time, p10 - p90"] + [f"{pc(R[d]['per_session_share_p10_p90_tool'][0])} - {pc(R[d]['per_session_share_p10_p90_tool'][1])}" for d in DAYS],
            ["upper bound on time saved, perfect prediction, S19 rule min(tool batch, next LLM step): share of active turn time"] + [pc(R[d]["upper_bound_save_S19_rule_of_active_time"], 2) for d in DAYS],
            ["  same, share of session wall clock incl. user idle"] + [pc(R[d]["upper_bound_save_S19_rule_of_session_wall"], 2) for d in DAYS],
            ["  same, per session median / p90"] + [f"{pc(R[d]['upper_bound_per_session_S19_median_p90'][0],2)} / {pc(R[d]['upper_bound_per_session_S19_median_p90'][1],2)}" for d in DAYS],
            ["  same, batches under 1 h only"] + [pc(R[d]["upper_bound_save_S19_batches_under_1h_of_active_time"], 2) for d in DAYS],
            ["robust variant: drop turns that contain a batch over 1 h (share of active time kept) -> LLM / tool / gap shares"] + [f"{pc(R[d]['robust_excl_turns_with_batch_over_1h']['time_kept'])} kept -> {pc(R[d]['robust_excl_turns_with_batch_over_1h']['llm'])} / {pc(R[d]['robust_excl_turns_with_batch_over_1h']['tool'])} / {pc(R[d]['robust_excl_turns_with_batch_over_1h']['gap'])}" for d in DAYS],
            ["  same robust subset: upper bound on time saved (S19 rule)"] + [pc(R[d]["robust_excl_turns_with_batch_over_1h"]["save19"], 2) for d in DAYS],
            ["looser bound: next LLM call starts at batch issue (saves its whole wait), share of active turn time"] + [pc(R[d]["upper_bound_save_loose_of_active_time"], 2) for d in DAYS]]
    out += ["", "Upper bound on what next-step speculation can hide, assuming a perfect predictor (next LLM step starts when the tool batch is issued and is committed when the tool finishes):"] + table(["quantity", "day 1", "day 2"], rows)
    write_md("T", out); save("T", R); say("T done")


# ------------------------------------------------------------------ X  cross-dataset latency check (TraceLab <-> Copilot)
def load_tracelab():
    import gzip
    f = ROOT / "tracelab" / "syfi_coding_trace_v0.0.2.jsonl.gz"; rows = []
    with gzip.open(f, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            for t in r.get("tools") or []: rows.append((r["provider"], r["session_id"], t["tool_name"], t["emitted_at"], t.get("tool_wall_latency_ms"), bool(t.get("is_error"))))
    df = pd.DataFrame(rows, columns=["provider", "session", "tool", "t", "wall", "is_err"])
    df["t"] = pd.to_datetime(df["t"], utc=True, format="ISO8601"); return df


FH = ["l1", "l2", "l3", "s_ew", "lnfin"]


def stage_X(A):
    out = []; R = {}
    A, glob, med_tool = add_hist(A); A["lnfin"] = np.log1p(A["nfin"])
    df = load_tracelab(); n0 = len(df); df = df[df.wall.notna()].copy(); df["dur"] = df["wall"].astype(float).clip(lower=0)
    df = df.sort_values(["session", "t"], kind="stable").reset_index(drop=True)
    df["issue"] = df["t"].dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]").astype("int64") / 1e6
    df["sidc"] = pd.factorize(df.session)[0]; df["tfull"] = pd.factorize(df.tool)[0]; df["fail"] = df.is_err.astype(int)
    H = pd.DataFrame(causal_pass(df.sidc.values, df.tfull.values, df.issue.values, df.dur.values, df.fail.values, None), index=df.index); df = df.join(H)
    df["lnfin"] = np.log1p(df["nfin"]); df["ll"] = np.log1p(df["dur"])
    start = df.groupby("session")["issue"].min().sort_values(); k = int(len(start) * 0.7); trs = set(start.index[:k]); df["train"] = df.session.isin(trs)
    Ttr, Tte = df[df.train], df[~df.train]
    Ctr, Cte = A[A.day == DAYS[0]], A[A.day == DAYS[1]]
    R["n"] = dict(tracelab_calls=n0, tracelab_with_latency=len(df), tracelab_train=len(Ttr), tracelab_test=len(Tte), copilot_train=len(Ctr), copilot_test=len(Cte))
    rng = np.random.default_rng(0)
    def samp(d): return d.iloc[rng.choice(len(d), min(NFIT, len(d)), replace=False)]
    sc, st = samp(Ctr), samp(Ttr)
    m_c = hgb_reg(sc[FH].values, sc["ll"].values, None); m_t = hgb_reg(st[FH].values, st["ll"].values, None)
    rows = []; R["table"] = {}
    for tname, train_own, test, own_model, other_model, oname in (("Copilot day 2", Ctr, Cte, m_c, m_t, "TraceLab"), ("TraceLab test sessions", Ttr, Tte, m_t, m_c, "Copilot day 1")):
        y = test["dur"].values.clip(min=0); gm = float(train_own["dur"].median()); mt = train_own.groupby("tfull" if tname.startswith("Copilot") else "tool")["dur"].median()
        key = test["tfull"] if tname.startswith("Copilot") else test["tool"]
        P = {"own global median": np.full(len(test), gm), "own per-tool median": key.map(mt).fillna(gm).values,
             "history-only GBM, trained on the same dataset": np.expm1(own_model.predict(test[FH].values)), f"history-only GBM, trained on {oname}": np.expm1(other_model.predict(test[FH].values))}
        slow = (y > 1000).astype(int); grp = pd.factorize(test["sid"] if tname.startswith("Copilot") else test["session"])[0]
        ab = {k_: np.abs(np.log1p(np.maximum(v, 0)) - np.log1p(y)) for k_, v in P.items()}
        R["table"][tname] = {}
        for k_, v in P.items():
            mtr = lat_metrics(y, np.maximum(v, 0)); auc = float(roc_auc_score(slow, v)) if k_ != "own global median" else 0.5
            R["table"][tname][k_] = dict(mtr, auc=auc)
        for k_ in list(P)[2:]:
            dif = ab["own per-tool median"] - ab[k_]; bs = [(dif * w).sum() / w.sum() for w in boot_weights(grp, 100, 2)]
            R["table"][tname][k_]["logerr_gain_vs_own_tool_median"] = float(dif.mean()); R["table"][tname][k_]["gain_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        R["table"][tname]["base_rate_slow"] = float(slow.mean())
        for k_, r in R["table"][tname].items():
            if k_ == "base_rate_slow": continue
            rows.append([tname, k_, f"{r['mae']:.0f}", f"{r['p90']:.0f}", f"{r['logerr']:.2f}", f"{r['auc']:.3f}", "-" if "gain_ci" not in r else f"{r['logerr_gain_vs_own_tool_median']:+.2f} ({r['gain_ci'][0]:+.2f} to {r['gain_ci'][1]:+.2f})"])
    out += [f"TraceLab v0.0.2: {n0} tool records, {len(df)} with wall latency; chronological split by session start as in S14 F: {len(Ttr)} train / {len(Tte)} test calls. Copilot: train day 1 ({len(Ctr)} calls), test day 2 ({len(Cte)} calls). "
            f"History-only model = HistGradientBoosting (absolute-error loss on log1p latency, {NFIT} training rows) on tool-agnostic causal features only: latencies of the last 3 finished calls of the session, session EWMA (0.5), log number of finished calls. No tool name, no model, no time. "
            f"Slow base rate (> 1 s): Copilot day 2 {pc(R['table']['Copilot day 2']['base_rate_slow'])}, TraceLab test {pc(R['table']['TraceLab test sessions']['base_rate_slow'])}. AUC scores the > 1 s question with the predicted latency itself.", ""] + \
        table(["test set", "predictor", "MAE ms", "p90 abs err ms", "mean abs log1p-ratio", "AUC > 1 s", "log-error gain vs own per-tool median (95% CI)"], rows)
    write_md("X", out); save("X", R); say("X done")


if __name__ == "__main__":
    stages = [s for s in sys.argv[1:] if not s.startswith("-")] or list("FXEDST")
    A = prep(); say(f"loaded {len(A)} calls")
    if "F" in stages: A = stage_F(A)
    if "E" in stages: stage_E(A)
    if "D" in stages: stage_D(A)
    if "S" in stages: stage_S(A)
    if "T" in stages: stage_T(A)
    if "X" in stages: stage_X(A)
