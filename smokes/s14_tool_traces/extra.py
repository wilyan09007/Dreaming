"""S14 extra: F (TraceLab tool latency) and G (Toucan: are real MCP tool returns a function of the arguments?).
Run: python -I -X utf8 extra.py [F|G]   -- reads downloads under smokes/.data read-only; writes results_extra.json + tables_F.md / tables_G.md."""
import sys, os, json, re, gzip, ast, collections, math, random, time, traceback
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np, pandas as pd
import run  # tau-bench helpers: certify_eval, table, pct, f1
from run import table, pct, f1, certify_eval

TINY = "--tiny" in sys.argv  # plumbing check only: first 4000 TraceLab rounds / first Toucan batch; outputs to $S14_OUT
OUT = Path(os.environ.get("S14_OUT", HERE))
DATA = Path(os.environ.get("S14_DATA", HERE.parent / ".data"))  # pod: /workspace/job/data (fetch.py)
R = {}


# ---------------------------------------------------------------- F latency
def load_tracelab():
    f = DATA / "tracelab" / "syfi_coding_trace_v0.0.2.jsonl.gz"; rows = []
    with gzip.open(f, "rt", encoding="utf-8") as fh:
        for ln, line in enumerate(fh):
            if TINY and ln >= 4000: break
            r = json.loads(line)
            for t in r.get("tools") or []:
                ex = t.get("executables") or []
                rows.append((r["provider"], r["session_id"], r["user"], t["tool_name"], t["emitted_at"], t.get("input_chars"), t.get("tool_wall_latency_ms"),
                             t.get("tool_internal_latency_ms"), bool(t.get("is_error")), ex[0] if ex else "", len(ex)))
    df = pd.DataFrame(rows, columns=["provider", "session", "user", "tool", "t", "in_chars", "wall", "internal", "is_err", "exe", "n_exe"])
    df["t"] = pd.to_datetime(df["t"], utc=True, format="ISO8601")
    return df


def causal_ewma(key, t, lat, alpha, init=None):
    """EWMA of log1p(latency) per key *as known when each call is emitted*: a call only counts once it has finished (t + latency < t_now),
    so parallel calls cannot see each other's latency. alpha=1 gives 'latency of the latest finished call with this key'.
    key: int codes; t: int64 ns of emission; lat: ms; init: per-row prior (NaN = no prior). Returns the pre-call value per row (NaN if nothing known)."""
    n = len(t); when = np.concatenate([t, t + (lat * 1e6).astype(np.int64)]); kind = np.repeat(np.array([0, 1], np.int8), n)  # tie: query before update
    ll = np.log1p(lat).tolist(); key = key.tolist(); init = [math.nan] * n if init is None else list(init); out = [math.nan] * n; cur = {}
    for j in np.lexsort((kind, when)).tolist():
        if j < n:
            v = cur.get(key[j]); out[j] = init[j] if v is None else v
        else:
            i = j - n; v = cur.get(key[i])
            if v is None: v = init[i]
            cur[key[i]] = ll[i] if v != v else (1 - alpha) * v + alpha * ll[i]
    return np.array(out)


def analysis_F():
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    T0 = time.time(); say = lambda m: print(f"[F {time.time()-T0:.0f}s] {m}", flush=True)
    out = []; df = load_tracelab(); n0 = len(df); say(f"loaded {n0} tool records")
    out.append(f"TraceLab v0.0.2 (CC BY 4.0): {n0} tool records, {df['session'].nunique()} sessions, {df['user'].nunique()} users, providers {df['provider'].value_counts().to_dict()}; "
               f"records with wall latency {df['wall'].notna().mean()*100:.1f}%, with tool-reported internal latency {df['internal'].notna().mean()*100:.1f}%.")
    df = df[df["wall"].notna()].copy(); df["lat"] = df["wall"].astype(float).clip(lower=0)
    df = df.sort_values(["session", "t"], kind="stable").reset_index(drop=True)
    tn = df["t"].dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]").astype("int64").values
    lat = df["lat"].values
    # history features, causal at emission time (only calls that already finished)
    sid = pd.factorize(df["session"])[0]; stid = pd.factorize(df["session"] + "|" + df["tool"])[0]
    df["pos"] = df.groupby("session").cumcount()
    df["prev_lat"] = causal_ewma(sid, tn, lat, 1.0); df["prev_same"] = causal_ewma(stid, tn, lat, 1.0); df["ew_sess_tool"] = causal_ewma(stid, tn, lat, 0.5)
    df["hour"] = df["t"].dt.hour; say("history features done")
    # chronological split by session start
    start = df.groupby("session")["t"].min().sort_values(); k = int(len(start) * 0.7)
    tr_s = set(start.index[:k]); df["train"] = df["session"].isin(tr_s)
    tr, te = df[df["train"]].copy(), df[~df["train"]].copy()
    out.append(f"Chronological split by session start: {len(tr_s)} train sessions ({len(tr)} calls, up to {start.iloc[k-1]:%Y-%m-%d}) / {len(start)-len(tr_s)} test sessions ({len(te)} calls, from {start.iloc[k]:%Y-%m-%d}). "
               f"History features (previous latencies, session EWMA) only use calls that had already finished when the new call was emitted.")
    # --- how much time is in slow calls
    tot = df["lat"].sum(); slow = df["lat"] > 1000
    out.append(f"All calls: median latency {df['lat'].median():.0f} ms, mean {df['lat'].mean():.0f} ms, p90 {df['lat'].quantile(.9):.0f} ms; calls > 1 s = {slow.mean()*100:.1f}% of calls but {df.loc[slow,'lat'].sum()/tot*100:.1f}% of summed tool time "
               f"(summed over parallel calls, so not elapsed time); calls > 10 s = {(df['lat']>10000).mean()*100:.2f}% of calls, {df.loc[df['lat']>10000,'lat'].sum()/tot*100:.1f}% of time.")
    rows = []
    for tname, d in df.groupby("tool"):
        if len(d) < (3000 if not TINY else 30): continue
        rows.append([tname, len(d), f"{d['lat'].median():.0f}", f"{d['lat'].quantile(.9):.0f}", f1(100 * (d['lat'] > 1000).mean()), f1(100 * d.loc[d['lat'] > 1000, 'lat'].sum() / d['lat'].sum())])
    rows.sort(key=lambda r: -r[1])
    out += table(["tool", "calls", "median ms", "p90 ms", "% calls > 1 s", "% of tool time in calls > 1 s"], rows[:16])
    R["F_overview"] = dict(n=len(df), slow_call_pct=float(slow.mean() * 100), slow_time_pct=float(df.loc[slow, "lat"].sum() / tot * 100))
    # --- predictors on test
    y = te["lat"].values
    med_g = tr["lat"].median(); med_tool = tr.groupby("tool")["lat"].median()
    p_global = np.full(len(te), med_g)
    p_tool = te["tool"].map(med_tool).fillna(med_g).values
    ke = tr.groupby(["tool", "exe"])["lat"].agg(["median", "size"]); ke = ke[ke["size"] >= (20 if not TINY else 2)]["median"]
    p_exe = np.array([ke.get(kk, np.nan) for kk in zip(te["tool"], te["exe"])]); p_exe = np.where(np.isnan(p_exe), p_tool, p_exe)
    # per-tool EWMA over the whole test stream (all sessions, log domain, alpha 0.1, seeded from the train median), causal at emission time
    tcode = pd.factorize(te["tool"])[0]; tte = te["t"].dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]").astype("int64").values
    p_ew = np.expm1(causal_ewma(tcode, tte, y, 0.1, np.log1p(p_tool)))
    # per-session same-tool EWMA (alpha 0.5), falling back to the (tool, executable) median when the session has no finished call of this tool
    p_sess = np.where(te["ew_sess_tool"].notna(), np.expm1(te["ew_sess_tool"].fillna(0).values), p_exe)
    say("baselines done")
    # gradient boosting on log latency, absolute-error loss (median); fitted on a fixed 300k-row sample of the training calls
    trs = tr.sample(min(len(tr), 300_000), random_state=0)
    top_tool = tr["tool"].value_counts().head(80).index; top_exe = tr["exe"].value_counts().head(60).index
    def X(d):
        return pd.DataFrame(dict(tool=pd.Categorical(d["tool"].where(d["tool"].isin(top_tool), "other"), categories=list(top_tool) + ["other"]).codes,
                                 exe=pd.Categorical(d["exe"].where(d["exe"].isin(top_exe), "other"), categories=list(top_exe) + ["other"]).codes,
                                 n_exe=d["n_exe"], provider=(d["provider"] == "claude").astype(int), in_chars=np.log1p(d["in_chars"].fillna(0)),
                                 prev_lat=d["prev_lat"], prev_same=d["prev_same"], ew=d["ew_sess_tool"], pos=np.log1p(d["pos"]), hour=d["hour"]))
    Xtr, Xte = X(trs), X(te); ytr = np.log1p(trs["lat"].values); static = ["tool", "exe", "n_exe", "provider", "in_chars", "pos", "hour"]
    mk = lambda: HistGradientBoostingRegressor(loss="absolute_error", max_iter=200, learning_rate=0.1, max_leaf_nodes=31, categorical_features=[0, 1], random_state=0)
    p_gbm = np.expm1(mk().fit(Xtr, ytr).predict(Xte)).clip(min=0); say("GBM with history done")
    p_gbm_static = np.expm1(mk().fit(Xtr[static], ytr).predict(Xte[static])).clip(min=0); say("GBM static done")
    rows = []
    for name, p in (("global median", p_global), ("per-tool median", p_tool), ("per-(tool, executable) median", p_exe), ("per-tool EWMA over the stream (alpha 0.1)", p_ew),
                    ("per-session same-tool EWMA (0.5)", p_sess), ("GBM, call-time features only (tool, executable, input size, hour)", p_gbm_static), ("GBM + session history (previous latencies)", p_gbm)):
        ae = np.abs(p - y); lr = np.abs(np.log1p(p) - np.log1p(y))
        rows.append([name, f"{ae.mean():.0f}", f"{np.median(ae):.0f}", f"{np.percentile(ae, 90):.0f}", f"{lr.mean():.2f}", f"{np.median(lr):.2f}"])
        R.setdefault("F_models", {})[name] = dict(mae=float(ae.mean()), medae=float(np.median(ae)), p90ae=float(np.percentile(ae, 90)), logerr=float(lr.mean()))
    out.append("\nTest-set error of point predictions (ms). MAE is dominated by the heavy tail (subagent, wait and long shell calls); the log error is the scale-free view.")
    out += table(["predictor", "MAE ms", "median abs err ms", "p90 abs err ms", "mean |log1p ratio|", "median |log1p ratio|"], rows)
    # excluding the very long tools
    longtools = {"Agent", "wait", "write_stdin", "Task", "TaskOutput", "ScheduleWakeup", "Skill", "AskUserQuestion", "ExitPlanMode"}  # sub-agents, waits, human-in-the-loop prompts
    m = ~te["tool"].isin(longtools).values
    rows = []
    for name, p in (("per-tool median", p_tool), ("per-tool EWMA", p_ew), ("per-session same-tool EWMA", p_sess), ("GBM, call-time features only", p_gbm_static), ("GBM + session history", p_gbm)):
        ae = np.abs(p - y)[m]; rows.append([name, f"{ae.mean():.0f}", f"{np.percentile(ae, 90):.0f}", f"{np.abs(np.log1p(p) - np.log1p(y))[m].mean():.2f}"])
        R.setdefault("F_models_nolong", {})[name] = dict(mae=float(ae.mean()), p90ae=float(np.percentile(ae, 90)))
    out.append(f"\nSame, excluding the long-wait tools {sorted(longtools)} ({m.mean()*100:.0f}% of test calls kept):")
    out += table(["predictor", "MAE ms", "p90 abs err ms", "mean |log1p ratio|"], rows)
    # slow-call classification
    ys = (y > 1000).astype(int); ytrs = (trs["lat"].values > 1000).astype(int)
    mc = lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, categorical_features=[0, 1], random_state=0)
    pr = mc().fit(Xtr, ytrs).predict_proba(Xte)[:, 1]; pr2 = mc().fit(Xtr[static], ytrs).predict_proba(Xte[static])[:, 1]
    rate_tool = (tr["lat"] > 1000).groupby(tr["tool"]).mean(); pb = te["tool"].map(rate_tool).fillna(0.1).values
    rows = []
    for name, p in (("per-tool slow rate", pb), ("GBM call-time features", pr2), ("GBM + session history", pr)):
        o = np.argsort(-p); kk = int(ys.sum()); prec = ys[o[:kk]].mean()
        rows.append([name, f"{roc_auc_score(ys, p):.3f}", f"{prec*100:.0f}%"])
    out.append(f"\nWill this call take > 1 s? Test base rate {ys.mean()*100:.1f}% of calls ({y[ys==1].sum()/y.sum()*100:.0f}% of test tool time). AUC and precision when flagging as many calls as there are slow ones:")
    out += table(["predictor", "AUC", "precision at top-k (k = #slow)"], rows)
    R["F_slow"] = {r[0]: r[1] for r in rows}
    # per-tool: does the executable matter? Bash only
    b = te["tool"].isin(["Bash", "exec_command"]).values
    if b.any():
        out.append(f"\nShell tools only (Bash/exec_command, {b.sum()} test calls): MAE per-tool median {np.abs(p_tool-y)[b].mean():.0f} ms, per-(tool, executable) median {np.abs(p_exe-y)[b].mean():.0f} ms, GBM call-time {np.abs(p_gbm_static-y)[b].mean():.0f} ms, GBM+history {np.abs(p_gbm-y)[b].mean():.0f} ms; "
                   f"p90 abs err {np.percentile(np.abs(p_tool-y)[b],90):.0f} / {np.percentile(np.abs(p_exe-y)[b],90):.0f} / {np.percentile(np.abs(p_gbm_static-y)[b],90):.0f} / {np.percentile(np.abs(p_gbm-y)[b],90):.0f} ms.")
    # ablation + drift check: why does the call-time GBM not beat the per-tool median on later sessions?
    rows = []
    for name, cols in (("tool, executable, provider", ["tool", "exe", "provider"]), ("+ input size", ["tool", "exe", "provider", "in_chars"]), ("all call-time features (+ n_exe, position in session, hour)", static)):
        pg = np.expm1(mk().fit(Xtr[cols], ytr).predict(Xte[cols])).clip(min=0)
        rows.append([name, f"{np.abs(pg - y).mean():.0f}", f"{np.abs(np.log1p(pg) - np.log1p(y)).mean():.2f}"])
    rows.append(["(reference) per-tool median", f"{np.abs(p_tool - y).mean():.0f}", f"{np.abs(np.log1p(p_tool) - np.log1p(y)).mean():.2f}"])
    out.append("\nAblation of the call-time GBM on the later (test) sessions:")
    out += table(["GBM features", "MAE ms", "mean |log1p ratio|"], rows)
    R["F_ablation"] = {r[0]: r[2] for r in rows}
    rs = np.random.RandomState(0); hold = set(rs.choice(sorted(tr_s), len(tr_s) // 5, replace=False))
    a_, b_ = trs[~trs["session"].isin(hold)], tr[tr["session"].isin(hold)]
    pa = np.expm1(mk().fit(X(a_)[static], np.log1p(a_["lat"].values)).predict(X(b_)[static])).clip(min=0)
    pm = b_["tool"].map(a_.groupby("tool")["lat"].median()).fillna(a_["lat"].median()).values; yb = b_["lat"].values
    out.append(f"\nDrift check, same period (hold out a random 20% of TRAIN-period sessions, {len(b_)} calls): mean |log1p ratio| per-tool median {np.abs(np.log1p(pm) - np.log1p(yb)).mean():.2f}, "
               f"call-time GBM {np.abs(np.log1p(pa) - np.log1p(yb)).mean():.2f} (MAE {np.abs(pm - yb).mean():.0f} vs {np.abs(pa - yb).mean():.0f} ms). If the GBM wins here but not on later sessions, the gap is drift, not a bug.")
    R["F_drift"] = dict(per_tool_median=float(np.abs(np.log1p(pm) - np.log1p(yb)).mean()), gbm=float(np.abs(np.log1p(pa) - np.log1p(yb)).mean()))
    say("F done")
    return out


# ---------------------------------------------------------------- G Toucan
ERR_PAT = re.compile(r"error|failed|failure|exception|not found|invalid|unauthori|forbidden|requires|not available|api key|apikey|timed? ?out|unable to|cannot|can't|denied|traceback|rate limit|status code|not supported|missing", re.I)


def ret_text(content):
    try: j = json.loads(content)
    except Exception: return content
    if isinstance(j, dict) and "text" in j: return str(j["text"])
    if isinstance(j, list): return "\n".join(str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in j)
    return content


def norm_json(s):
    try: return json.dumps(json.loads(s), sort_keys=True, separators=(",", ":"))
    except Exception: return s.strip()


MASK = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?Z?|\d+(\.\d+)?")


def load_toucan():
    f = DATA / "toucan" / "OSS" / "train-00010-of-00047.parquet"
    if TINY:
        import pyarrow.parquet as pq
        df = next(pq.ParquetFile(f).iter_batches(batch_size=300, columns=["uuid", "messages", "metadata"])).to_pandas()
    else: df = pd.read_parquet(f, columns=["uuid", "messages", "metadata"])
    calls = []; srv_tools = {}
    def tools_of(srv):
        sid = srv["server_id"]
        if sid not in srv_tools:
            try:
                rr = srv["remote_server_response"]; rr = rr if isinstance(rr, dict) else ast.literal_eval(rr)
                srv_tools[sid] = {t["name"] for t in rr.get("tools", [])}
            except Exception: srv_tools[sid] = set()
        return srv_tools[sid]
    for ep, (u, ms, md) in enumerate(zip(df["uuid"], df["messages"], df["metadata"])):
        try: servers = json.loads(md).get("mcp_servers", [])
        except Exception: servers = []
        pend = None
        for m in json.loads(ms):
            if m.get("function_call"):
                try: fc = ast.literal_eval(m["function_call"]) if isinstance(m["function_call"], str) else m["function_call"]
                except Exception: pend = None; continue
                try: a = json.loads(fc["arguments"]) if isinstance(fc["arguments"], str) else fc["arguments"]
                except Exception: a = {"_raw": str(fc.get("arguments"))}
                pend = (fc["name"], a)
            elif m["role"] == "function" and pend is not None:
                t = ret_text(m["content"]); name, a = pend; pend = None
                own = [s["server_id"] for s in servers if name in tools_of(s)]
                calls.append(dict(ep=ep, server=own[0] if own else "?", tool=name, akey=json.dumps(a, sort_keys=True, separators=(",", ":")), ret=norm_json(t),
                                  is_err=bool(ERR_PAT.search(t[:300])), rlen=len(t)))
    return pd.DataFrame(calls), len(df)


def analysis_G():
    out = []; c, nrows = load_toucan(); n = len(c)
    c["key"] = list(zip(c["server"], c["tool"], c["akey"]))
    out.append(f"Toucan-1.5M (Apache-2.0), one shard OSS/train-00010-of-00047 (147 MB; all rows are subset single-turn-original; the first shard I tried, train-00027, was all 'irrelevant' prompts with no tool calls): "
               f"{nrows} prompts, {n} tool calls, {c['tool'].nunique()} distinct tool names, {c['server'].nunique()} distinct MCP servers (tool -> server via each prompt's server tool list; '?' when unresolved: {(c['server']=='?').mean()*100:.1f}% of calls), {c['key'].nunique()} distinct (server, tool, args). "
               f"Returns that look like errors / unavailable-tool messages (keyword heuristic on the first 300 chars): {c['is_err'].mean()*100:.1f}% of calls.")
    # --- A-style reuse inside the shard (different prompts calling the same tool with the same args)
    grp = c.groupby("key")
    sizes = grp.size(); rep_keys = sizes[sizes >= 2].index
    cr = c[c["key"].isin(rep_keys)]
    out.append(f"Calls whose exact (server, tool, args) occurs in at least one other call: {len(cr)} = {len(cr)/n*100:.1f}% of calls ({len(rep_keys)} keys). (Within the same prompt too, so see the cross-prompt number below.)")
    # identical-return share per repeated key (cross-prompt only: dedupe repeated calls inside one prompt)
    same_all = grp["ret"].nunique(); mask_nuni = c.assign(rm=c["ret"].map(lambda s: MASK.sub("#", s)[:2000])).groupby("key")["rm"].nunique()
    xp = c.groupby("key")["ep"].nunique(); xkeys = xp[xp >= 2].index
    a = (same_all[xkeys] == 1); b = (mask_nuni[xkeys] == 1)
    out.append(f"Keys seen in >= 2 different prompts: {len(xkeys)}; all returns byte-identical (after JSON key-order normalization) for {a.mean()*100:.1f}% of them; identical after masking numbers and timestamps for {b.mean()*100:.1f}%.")
    ce = c[c["key"].isin(xkeys)]
    errshare = ce.groupby("key")["is_err"].mean()
    out.append(f"Among those cross-prompt keys, {(errshare>0.5).mean()*100:.1f}% are mostly error-looking returns; restricted to keys with no error-looking return: identical for {a[errshare[xkeys]==0].mean()*100:.1f}% (n={int((errshare[xkeys]==0).sum())}).")
    R["G_keys"] = dict(calls=n, distinct_keys=int(c["key"].nunique()), cross_prompt_keys=int(len(xkeys)), identical_pct=float(a.mean() * 100), identical_masked_pct=float(b.mean() * 100),
                       identical_noerr_pct=float(a[errshare[xkeys] == 0].mean() * 100), n_noerr=int((errshare[xkeys] == 0).sum()))
    # --- per-tool classification
    rows = []; cls = collections.Counter(); per_tool = []
    for (srv, tool), d in c.groupby(["server", "tool"]):
        ks = d.groupby("key")["ep"].nunique(); rk = ks[ks >= 2].index
        if len(rk) == 0: per_tool.append((srv, tool, len(d), 0, np.nan, np.nan)); continue
        ident = np.mean([d[d["key"] == k]["ret"].nunique() == 1 for k in rk]); errs = d[d["key"].isin(rk)]["is_err"].mean()
        per_tool.append((srv, tool, len(d), len(rk), ident, errs))
    pt = pd.DataFrame(per_tool, columns=["server", "tool", "calls", "repeated_keys", "identical_share", "err_share"])
    ntools = len(pt); unk = (pt["repeated_keys"] < 2).sum(); pt2 = pt[pt["repeated_keys"] >= 2]
    det = (pt2["identical_share"] >= 0.95).sum(); varying = (pt2["identical_share"] < 0.5).sum(); mixed = len(pt2) - det - varying
    out.append(f"\nTools (MCP server, tool name): {ntools}. {unk} ({unk/ntools*100:.0f}%) have fewer than 2 repeated argument keys in this shard, so their behaviour is unobservable here. Of the {len(pt2)} with >= 2 repeated keys: "
               f"{det} ({det/len(pt2)*100:.0f}%) deterministic (>= 95% of repeated keys return identical text), {mixed} mixed, {varying} ({varying/len(pt2)*100:.0f}%) time- or state-varying (< 50% of repeated keys identical). "
               f"Weighted by calls: deterministic tools carry {pt2.loc[pt2['identical_share']>=0.95,'calls'].sum()/pt2['calls'].sum()*100:.0f}% of the calls of these {len(pt2)} tools.")
    R["G_tools"] = dict(n_tools=ntools, unobservable=int(unk), observable=len(pt2), deterministic=int(det), mixed=int(mixed), varying=int(varying))
    # error-dominated deterministic tools
    detd = pt2[pt2["identical_share"] >= 0.95]
    out.append(f"Of the {len(detd)} deterministic tools, {(detd['err_share']>0.5).sum()} return mostly error-looking text (constant failure messages, e.g. missing credentials or a Windows-only tool in a Linux sandbox), so 'deterministic' there means 'deterministically broken'.")
    rows = []
    for _, r in pt2.sort_values("calls", ascending=False).head(14).iterrows():
        rows.append([r["tool"], r["calls"], int(r["repeated_keys"]), f"{r['identical_share']*100:.0f}%", f"{r['err_share']*100:.0f}%"])
    out.append("\nMost-called observable tools:")
    out += table(["tool", "calls", "repeated arg keys", "% repeated keys with identical return", "% of those calls error-looking"], rows)
    # examples of varying
    out.append("\nExamples of tools whose identical calls returned different text (excluding error-looking returns):")
    ex = 0
    for _, r in pt2[(pt2["identical_share"] < 0.5)].sort_values("calls", ascending=False).iterrows():
        d = c[(c["tool"] == r["tool"]) & (c["server"] == r["server"]) & (~c["is_err"])]
        for k, dd in d.groupby("key"):
            if dd["ret"].nunique() >= 2 and dd["ep"].nunique() >= 2:
                u = dd["ret"].unique()[:2]; out.append(f"- `{r['tool']}` args `{k[2][:90]}`: {len(dd)} calls, {dd['ret'].nunique()} distinct returns, e.g. `{u[0][:110]!r}` vs `{u[1][:110]!r}`"); ex += 1; break
        if ex >= 6: break
    out.append("\nExamples of deterministic tools with a large repeated-key set:")
    ex = 0
    for _, r in detd[detd["err_share"] < 0.2].sort_values("repeated_keys", ascending=False).head(4).iterrows():
        d = c[(c["tool"] == r["tool"]) & (c["server"] == r["server"])]; k = d.groupby("key")["ep"].nunique().sort_values(ascending=False).index[0]
        out.append(f"- `{r['tool']}`: {r['calls']} calls, {int(r['repeated_keys'])} repeated keys, e.g. args `{k[2][:90]}` -> `{d[d['key']==k]['ret'].iloc[0][:120]!r}`")
    # --- certified skipping on shard, split by prompt (episode)
    out.append("\nCertified skipping on Toucan (episodes = prompts; memory 50% / calibration 25% / test 25% of prompts, 10 random splits; confidence = k/(n+1) as for tau-bench; every tool treated as read-only):")
    rows = []
    for budget in (0.01, 0.02, 0.05):
        sk = er = tot = feas = 0; per = collections.defaultdict(lambda: [0, 0, 0]); oracle = []
        for seed in range(10):
            rng = random.Random(seed); eps = list(range(nrows)); rng.shuffle(eps); M, C, T = set(eps[: nrows // 2]), set(eps[nrows // 2: 3 * nrows // 4]), set(eps[3 * nrows // 4:])
            mem = collections.defaultdict(collections.Counter)
            for k, ep, r in zip(c["key"], c["ep"], c["ret"]):
                if ep in M: mem[k][r] += 1
            def rows_of(S):
                o = []
                for k, ep, r, tl in zip(c["key"], c["ep"], c["ret"], c["tool"]):
                    if ep not in S: continue
                    cnt = mem.get(k)
                    if not cnt: o.append(dict(conf=-1.0, ok=False, seen=False, tool=tl)); continue
                    nn = sum(cnt.values()); rr, kk = cnt.most_common(1)[0]; o.append(dict(conf=kk / (nn + 1), ok=rr == r, seen=True, tool=tl))
                return o
            Cr, Tr = rows_of(C), rows_of(T); tot += len(Tr); oracle.append(np.mean([r["ok"] for r in Tr]) * 100)
            cert, ap = certify_eval(Cr, Tr, budget)
            if cert["cutoff"] != float("inf"):
                feas += 1; sk += ap["n_above"]; er += (0 if ap["error"] is None else ap["error"] * ap["n_above"])
        rows.append([f"{int(budget*100)}%", f"{feas}/10", f1(pct(sk, tot)) if feas else "0.0", f1(pct(er, sk)) if sk else "-", f1(np.mean(oracle))])
        R.setdefault("G_cert", {})[str(budget)] = dict(feasible=feas, skipped_pct=pct(sk, tot), err_pct=pct(er, sk) if sk else None, oracle_pct=float(np.mean(oracle)))
    out += table(["budget", "certified (splits)", "test calls skipped %", "realized error % of skipped", "oracle: seen in memory and majority correct %"], rows)
    return out


# ---------------------------------------------------------------- T tau2-bench banking_knowledge gold replay
def analysis_T():
    rows = [json.loads(l) for l in open(HERE / "tau2_calls.jsonl", encoding="utf-8")]; out = []
    calls = []
    for r in rows:
        nm, a = r["name"], r["args"]
        if nm in ("call_discoverable_agent_tool", "call_discoverable_user_tool"):
            nm = a.get("agent_tool_name") or a.get("discoverable_tool_name") or nm; args = a.get("arguments", {})
            try: args = json.loads(args) if isinstance(args, str) else args
            except Exception: pass
        else: args = a
        read = nm.startswith("get_") and r["before"] == r["after"]
        calls.append(dict(task=r["task"], tool=nm, akey=json.dumps(args, sort_keys=True), ret=r["ret"], before=r["before"], read=read, mut=r["before"] != r["after"]))
    reads = [c for c in calls if c["read"]]
    out.append(f"tau2-bench banking_knowledge (MIT): replayed the gold actions of {len({c['task'] for c in calls})} tasks with the deterministic tools, no LLM (retrieval variant no_knowledge): {len(calls)} calls, "
               f"{sum(c['mut'] for c in calls)} change the database or user database, {len(reads)} are read-only get_* calls. One scripted trajectory per task, so this measures the tools, not an agent.")
    seen_in, seen_st = collections.defaultdict(list), collections.defaultdict(list); rws = []
    tasks = []
    for c in calls:
        if c["task"] not in tasks: tasks.append(c["task"])
    for t in tasks:
        new_in, new_st = [], []
        for c in [c for c in calls if c["task"] == t]:
            if not c["read"]: continue
            k = (c["tool"], c["akey"]); ks = (c["tool"], c["akey"], c["before"]); h, hs = seen_in.get(k, []), seen_st.get(ks, [])
            rws.append(dict(tool=c["tool"], seen=bool(h), eq=h[-1] == c["ret"] if h else None, seen_s=bool(hs), eq_s=hs[-1] == c["ret"] if hs else None))
            new_in.append((k, c["ret"])); new_st.append((ks, c["ret"]))
        for k, r in new_in: seen_in[k].append(r)
        for k, r in new_st: seen_st[k].append(r)
    n = len(rws); si = sum(r["seen"] for r in rws); ss = sum(r["seen_s"] for r in rws)
    out.append(f"\nAcross tasks (chronological by task order), read calls seen in an earlier task: input-keyed {si}/{n} = {pct(si,n):.1f}%, of which the earlier return equals the new one {sum(r['eq'] for r in rws if r['seen'])}/{si}; "
               f"keyed on (tool, args, full database hash before the call) {ss}/{n} = {pct(ss,n):.1f}%, equal {sum(r['eq_s'] for r in rws if r['seen_s'])}/{ss}.")
    cnt = collections.Counter(r["tool"] for r in rws)
    out += table(["read tool", "calls", "seen in earlier task %", "earlier return equal %"], [[t, v, f1(pct(sum(r["seen"] for r in rws if r["tool"] == t), v)),
                 f1(pct(sum(r["eq"] for r in rws if r["tool"] == t and r["seen"]), max(1, sum(r["seen"] for r in rws if r["tool"] == t))))] for t, v in cnt.most_common(8)])
    R["T_banking"] = dict(calls=len(calls), reads=n, seen_pct=pct(si, n), seen_state_pct=pct(ss, n), mutating=int(sum(c["mut"] for c in calls)))
    return out


if __name__ == "__main__":
    want = {a for a in sys.argv[1:] if not a.startswith("--")} or {"F", "G"}
    p = OUT / "results_extra.json"
    for name, fn in (("F", analysis_F), ("G", analysis_G), ("T", analysis_T)):
        if name not in want: continue
        try:  # one failing analysis must not lose the others: tables and json are written as each finishes
            lines = fn(); print(f"\n######## {name}"); print("\n".join(lines), flush=True)
            (OUT / f"tables_{name}.md").write_text("\n".join(lines), encoding="utf-8")
            prev = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
            prev.update(R); p.write_text(json.dumps(prev, indent=1, default=float), encoding="utf-8")
        except Exception:
            print(f"######## {name} FAILED"); traceback.print_exc(); sys.stdout.flush()
