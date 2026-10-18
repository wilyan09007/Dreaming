"""Replay, recertify, backtest. Everything Dreaming predicted can be re-executed exactly. Certificates are issued in learning.py.

    replay(run)        serve every teacher answer and tool return from the log, re-execute every prediction on its
                       logged input with the logged weights, report same_bytes per row
    recertify(cert)    re-run the certificate's calibration rows through the same weights; the cutoff and error count reproduce
    backtest(site)     a candidate predictor's certificate on a window of logged inputs, before it touches live traffic

"Same bytes" means the same (label, quantized confidence) and output sha under the same weights and env hashes
(determinism.md: promise the decision, not raw logits across hardware).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from .contracts import Certificate, Decision, Kind, Phase, ReplayReport, ReplayStep, TraceRow
from .gate import certify
from .hashing import canonical, sha
from .learning import calibration, issue, prediction_of
from .predictors import Predictor
from .store import Store


def rendered_of(row: TraceRow) -> str:
    """What the predictor read: steps log the rendered text itself, tools log their arguments."""
    return row.request if isinstance(row.request, str) else canonical(row.request).decode()


def output_sha(label: str, confidence: float) -> str:
    return sha([label, confidence])


def replay(store: Store, run_id: str, predictors: Mapping[str, Predictor], load: Callable[[str, str], Predictor] | None = None) -> ReplayReport:
    """`load(site_id, weights_sha)` fetches the exact weights a row names when the site has since retrained (Runtime.predictor_for)."""
    steps: list[ReplayStep] = []
    for row in store.run_rows(run_id):
        if row.kept is False:  # speculative work discarded on a mismatch: never served, nothing to re-execute
            continue
        if row.decision != Decision.predicted or row.prediction is None or row.kind == Kind.router:
            # a router's "prediction" is a frontier call: replay reads it from the log like any other frontier answer
            steps.append(ReplayStep(i=row.i, call_site=row.call_site, decision=row.decision, re_executed=False))
            continue
        want = row.prediction.hashes.weights or ""
        p = predictors.get(row.call_site)
        if (p is None or p.weights_sha != want) and load is not None:
            p = load(row.call_site, want)
        if p is None or p.weights_sha != want or p.env_sha != row.prediction.hashes.env:
            raise LookupError(f"{row.call_site}: replay needs weights {want[:8]} in env {(row.prediction.hashes.env or '')[:8]}")
        again = p.predict(rendered_of(row), fresh=True)  # no embedding cache: the forward pass really runs again
        logged = output_sha(row.prediction.label, row.prediction.confidence)
        now = output_sha(again.label, again.confidence)
        steps.append(ReplayStep(i=row.i, call_site=row.call_site, decision=row.decision, re_executed=True, same_bytes=logged == now, output_sha=now))
    re = [s for s in steps if s.re_executed]
    return ReplayReport(run_id=run_id, steps=steps, re_executed=len(re), same=sum(1 for s in re if s.same_bytes), from_log=len(steps) - len(re))


def recertify(store: Store, cert_id: str, predictor: Predictor) -> Certificate:
    """Re-run the calibration rows through the same weights. Raises if the rows or the weights changed."""
    cert = store.certificate(cert_id)
    rows = [r for r in store.rows(cert.call_site, shadow_only=True) if prediction_of(r, cert.weights_sha) is not None]  # promoted or spare
    if any(r.kind == Kind.router for r in rows):
        raise ValueError(f"{cert_id}: a router's predictor is a frontier call; its certificate is re-checked by audits, not re-executed")
    conf, correct, calib_sha = calibration(rows)
    if calib_sha != cert.calib_sha:
        raise LookupError(f"{cert_id}: calibration rows changed ({calib_sha[:8]} != {cert.calib_sha[:8]})")
    for r in rows:
        again = predictor.predict(rendered_of(r), fresh=True)
        if r.prediction and again.label != r.prediction.label:
            raise AssertionError(f"{cert_id}: row {r.id} predicted {again.label!r}, logged {r.prediction.label!r}")
    g = certify(conf, correct, cert.budget, cert.delta)
    return cert.model_copy(update={"cutoff": g.cutoff, "coverage": g.coverage, "emp_error": g.emp_error, "bound": g.bound, "n_errors": g.errors})


def backtest(rows: Sequence[TraceRow], predictor: Predictor, budget: float, teacher_sha: str, delta: float = 0.05) -> Certificate | None:
    """A candidate predictor's certificate on logged inputs with teacher answers, before it goes live."""
    scored = [
        r.model_copy(update={"prediction": predictor.predict(rendered_of(r)), "agree": None}) for r in rows if r.teacher and r.teacher.label is not None
    ]
    scored = [r.model_copy(update={"agree": r.prediction.label == (r.teacher.label if r.teacher else None)}) for r in scored if r.prediction]
    scored = [r.model_copy(update={"phase": Phase.testing}) for r in scored]  # a backtest is a calibration by construction
    return issue(rows[0].call_site, scored, budget, delta, predictor.weights_sha, predictor.env_sha, teacher_sha) if scored else None
