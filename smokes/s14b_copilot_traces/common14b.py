"""Shared helpers for S14b (loading, causal history pass, bootstrap metrics)."""
import os, sys, json, math, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / ".data"
FLAT = ROOT / "ghcp_flat"
DAYS = ("2026-06-01", "2026-06-02")
T0 = time.time()
def say(m): print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)
def table(head, rows): return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
def pc(x, d=1): return f"{100*x:.{d}f}%"

RES = HERE / "results.json"
def save(key, val):
    r = json.loads(RES.read_text()) if RES.exists() else {}
    r[key] = val; RES.write_text(json.dumps(r, indent=1, default=float))
def write_md(name, lines): (HERE / f"tables_{name}.md").write_text("\n".join(lines) + "\n")


def load_calls(day):
    c = pd.read_parquet(FLAT / f"{day}_calls.parquet")
    c = c.sort_values(["sid", "bstart", "k"], kind="stable").reset_index(drop=True)
    c["gpos"] = c.groupby("sid").cumcount()
    c["hour"] = ((c["bstart"] // 3.6e6) % 24).astype(int)
    c["day"] = day
    return c


def causal_pass(sid, tcode, t, lat, fail, tool_init=None):
    """History features at the moment each call is ISSUED (t, ms), using only calls that had already FINISHED (t' + lat' <= t; a tie counts as not finished).
    sid, tcode: int codes; t, lat: ms; fail: 0/1. Events are processed in time order across the whole stream (query before update on ties).
    Returns dict of arrays: l1,l2,l3 (log1p latency of the 1st/2nd/3rd latest finished call in the session), s_ew (session EWMA 0.5 of log1p latency),
    st_ew / st_cnt / st_nfail (same session & tool: EWMA 0.5, number finished, number failed), nfin / nfail (session: finished, failed), lastfail (1/0/-1 status of
    latest finished call), tool_ew (per-tool EWMA 0.1 over the whole stream of finished calls, seeded by tool_init (log1p))."""
    n = len(t); T = int(tcode.max()) + 1; S = int(sid.max()) + 1; nan = math.nan
    when = np.concatenate([t, t + lat]); kind = np.concatenate([np.zeros(n, np.int8), np.ones(n, np.int8)])
    order = np.lexsort((kind, when)).tolist()
    ll = np.log1p(lat).tolist(); sl = sid.tolist(); tl = tcode.tolist(); fl = fail.tolist()
    s_ew = [nan] * S; a1 = [nan] * S; a2 = [nan] * S; a3 = [nan] * S; nfin = [0] * S; nfl = [0] * S; lastf = [-1] * S
    st = {}; tew = list(tool_init) if tool_init is not None else [nan] * T
    O = {k: [nan] * n for k in ("l1", "l2", "l3", "s_ew", "st_ew", "st_cnt", "st_nfail", "nfin", "nfail", "lastfail", "tool_ew")}
    o_l1, o_l2, o_l3, o_se, o_te, o_tc, o_tf, o_nf, o_nl, o_lf, o_tw = (O[k] for k in ("l1", "l2", "l3", "s_ew", "st_ew", "st_cnt", "st_nfail", "nfin", "nfail", "lastfail", "tool_ew"))
    for j in order:
        if j < n:
            s = sl[j]; o_l1[j] = a1[s]; o_l2[j] = a2[s]; o_l3[j] = a3[s]; o_se[j] = s_ew[s]; o_nf[j] = nfin[s]; o_nl[j] = nfl[s]; o_lf[j] = lastf[s]
            r = st.get(s * T + tl[j])
            if r is None: o_te[j] = nan; o_tc[j] = 0; o_tf[j] = 0
            else: o_te[j] = r[0]; o_tc[j] = r[1]; o_tf[j] = r[2]
            o_tw[j] = tew[tl[j]]
        else:
            i = j - n; s = sl[i]; x = ll[i]; tt = tl[i]
            a3[s] = a2[s]; a2[s] = a1[s]; a1[s] = x
            v = s_ew[s]; s_ew[s] = x if v != v else 0.5 * v + 0.5 * x
            nfin[s] += 1; nfl[s] += fl[i]; lastf[s] = fl[i]
            key = s * T + tt; r = st.get(key)
            if r is None: st[key] = [x, 1, fl[i]]
            else: r[0] = 0.5 * r[0] + 0.5 * x; r[1] += 1; r[2] += fl[i]
            v = tew[tt]; tew[tt] = x if v != v else 0.9 * v + 0.1 * x
    return {k: np.array(v, dtype=float) for k, v in O.items()}


def boot_weights(grp, B, seed=0):
    G = int(grp.max()) + 1; rng = np.random.default_rng(seed)
    for _ in range(B): yield rng.multinomial(G, np.full(G, 1.0 / G))[grp].astype(float)


def auc_w(y, gid, ng, w):
    pos = np.bincount(gid, weights=w * y, minlength=ng); neg = np.bincount(gid, weights=w * (1 - y), minlength=ng)
    cn = np.cumsum(neg) - neg
    return float((pos * (cn + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))


def boot_binary(y, scores, grp, base, frac=0.2, B=200, seed=0):
    """AUC, and precision/recall at the top `frac` flagged, with session-bootstrap 95% CI; also the paired AUC difference to scores[base].
    scores: dict name -> score array. Ties are broken by a fixed random jitter for the top-fraction flags."""
    y = y.astype(float); n = len(y); rng = np.random.default_rng(1); jit = rng.random(n)
    G = {}; flags = {}
    for k, s in scores.items():
        u, gid = np.unique(s, return_inverse=True); G[k] = (gid, len(u))
        o = np.lexsort((jit, -s)); f = np.zeros(n, bool); f[o[: int(round(frac * n))]] = True; flags[k] = f
    pt = {}; bs = {k: {"auc": [], "prec": [], "rec": []} for k in scores}; dif = {k: [] for k in scores}
    def one(w):
        out = {}
        for k in scores:
            a = auc_w(y, *G[k], w); f = flags[k]
            out[k] = (a, float((w * y * f).sum() / (w * f).sum()), float((w * y * f).sum() / (w * y).sum()))
        return out
    p0 = one(np.ones(n))
    for w in boot_weights(grp, B, seed):
        o = one(w)
        for k in scores:
            bs[k]["auc"].append(o[k][0]); bs[k]["prec"].append(o[k][1]); bs[k]["rec"].append(o[k][2]); dif[k].append(o[k][0] - o[base][0])
    res = {}
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    for k in scores:
        res[k] = dict(auc=p0[k][0], auc_ci=ci(bs[k]["auc"]), prec=p0[k][1], prec_ci=ci(bs[k]["prec"]), rec=p0[k][2], rec_ci=ci(bs[k]["rec"]),
                      dauc=p0[k][0] - p0[base][0], dauc_ci=ci(dif[k]) if k != base else None)
    return res
