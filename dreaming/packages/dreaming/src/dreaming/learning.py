"""The per-site state machine: when to train, when to certify, when to revoke. Everything that changes a Policy.

    awake ──rows >= watch_after──> watching ──shadow_rows >= calibrate_after──> testing ──gate passes──> dreaming
            (one job per candidate;               (calibration = the shadow rows          │ audits fail ──> a spare takes over, or testing
             the first weights back shadow,        collected while watching/testing)        │ budget change ──> gate again
             the rest join as spares)

The candidate loop (DESIGN.md §4-§5): every candidate in `candidates.for_site(site)` is trained on the same rows, every loaded
candidate predicts every call (the promoted one into `TraceRow.prediction`, the spares into `TraceRow.shadow`), each is certified
at δ/N on the same calibration slice, and the one with the most certified coverage answers; the others stay as hot spares
(`Policy.shadow`). A spare is promoted when the winner's audits fail only if the spare's own shadow predictions on those audited
rows pass the gate at the budget: unbiased rows the spare never chose.

Rules the smokes forced:
- Calibration rows are the shadow rows the weights never trained on, collected before promotion (TraceRow.phase in
  watching/testing). Rows after promotion are biased (the predictor answered the confident ones) and never count.
- The gate runs at checkpoints (every `calibrate_after` calibration rows) and on a budget change, not on every tick:
  peeking at a growing set is multiple testing (S18).
- A certificate names weights, environment and teacher. Any of them changing at bind() sends the site back to testing.
- Audits are the standing check: after promotion, 1 in N confident answers still goes to the teacher; when their
  Clopper-Pearson bound exceeds the budget (>= 50 audits) the certificate is revoked.
  shortcut: a one-shot bound on the audit stream, not S18's anytime-valid e-process; upgrade before a long deployment.
- Outages are detected, never declared: three consecutive failures of the real call set Policy.outage; while out, one
  call in twenty probes the real call and a success clears it.
- Fail open: a trainer or store failure is one component.error event; the site stays where it was.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

from .context import new_id
from .contracts import (
    Attempt,
    CallSite,
    Certificate,
    Decision,
    EventKind,
    Phase,
    Policy,
    Prediction,
    Spare,
    TraceRow,
    TrainRequest,
    TrainResult,
)
from .gate import audits_needed, certify, cp_upper, smallest_budget
from .hashing import sha
from .predictors import Predictor
from .predictors.candidates import for_site
from .store import CALIBRATION
from .telemetry import boundary, emit

if TYPE_CHECKING:
    from .runtime import Runtime

OUTAGE_AFTER = 3  # consecutive failures of the real call
AUDIT_MIN = 50  # audits before their bound may revoke (and never fewer than audits_needed(budget, delta): zero errors cannot revoke)


def prediction_of(row: TraceRow, weights_sha: str) -> Prediction | None:
    """What these weights predicted on this row, wherever it was logged: as the promoted candidate (`prediction`) or a spare (`shadow`)."""
    if row.prediction is not None and row.prediction.hashes.weights == weights_sha:
        return row.prediction
    return next((p for p in row.shadow.values() if p.hashes.weights == weights_sha), None)


def scored(rows: Sequence[TraceRow], weights_sha: str) -> list[tuple[str, float, bool]]:
    """(row id, confidence, correct) for every row where these weights predicted and the teacher answered."""
    out: list[tuple[str, float, bool]] = []
    for r in rows:
        p = prediction_of(r, weights_sha)
        if p is not None and r.teacher is not None:
            out.append((r.id, p.confidence, p.label == r.teacher.label))
    return out


def calibration(rows: Sequence[TraceRow], weights_sha: str | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    """(confidence, correct, calib_sha) over calibration rows: prediction + teacher answer, collected before promotion
    (phase watching or testing). Rows after promotion are biased (the predictor answered the confident ones) and never count.
    With `weights_sha`, the rows are read for that candidate's predictions (promoted or spare); without, the promoted one's."""
    calib = [r for r in rows if r.phase in CALIBRATION]
    if weights_sha is None:
        picked = [(r.id, r.prediction.confidence, bool(r.agree)) for r in calib if r.prediction is not None and r.agree is not None]
    else:
        picked = scored(calib, weights_sha)
    conf = np.array([c for _, c, _ in picked], dtype=np.float64)
    correct = np.array([ok for _, _, ok in picked], dtype=bool)
    return conf, correct, sha(sorted(i for i, _, _ in picked))


def issue(call_site: str, rows: Sequence[TraceRow], budget: float, delta: float, weights_sha: str, env_sha: str, teacher_sha: str,
          candidate: str = "") -> Certificate | None:
    """The one place certificates are made: from stored calibration rows, by the gate. None when nothing certifies."""
    conf, correct, calib_sha = calibration(rows, weights_sha)
    return _issue(call_site, conf, correct, calib_sha, budget, delta, weights_sha, env_sha, teacher_sha, candidate)


def _issue(call_site: str, conf: np.ndarray, correct: np.ndarray, calib_sha: str, budget: float, delta: float, weights_sha: str,
           env_sha: str, teacher_sha: str, candidate: str) -> Certificate | None:
    g = certify(conf, correct, budget, delta)
    if not g.certified:
        return None
    return Certificate(
        id=new_id("cert"), call_site=call_site, candidate=candidate, weights_sha=weights_sha, env_sha=env_sha, teacher_sha=teacher_sha,
        budget=budget, delta=delta, cutoff=g.cutoff, coverage=g.coverage, emp_error=g.emp_error, bound=g.bound, n_calib=g.total, n_above=g.n,
        n_errors=g.errors, calib_sha=calib_sha, created_at=datetime.now(UTC),
    )


class Learner:
    def __init__(self, rt: Runtime) -> None:
        self.rt = rt
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="train")  # sites train side by side; a tick waits for the longest, not the sum
        self._pending: dict[tuple[str, str], Future[TrainResult]] = {}  # (site, candidate) -> the job
        self._trained_at: dict[str, int] = {}  # site -> rows at the last training attempt
        self._checked_at: dict[str, int] = {}  # site -> shadow rows at the last gate run
        self._failures: dict[str, int] = {}  # site -> consecutive real-call failures

    # ------------------------------------------------------------------ bind
    def check_certificates(self) -> None:
        """At startup: a certificate whose weights, environment or teacher no longer match goes back through the gate."""
        for site in self.rt.sites.values():
            p, pred = self.rt.policies[site.id], self.rt.predictors.get(site.id)
            if not p.cert_id:
                continue
            cert = self.rt.store.certificate(p.cert_id)
            same = pred is not None and (cert.weights_sha, cert.env_sha, cert.teacher_sha) == (pred.weights_sha, pred.env_sha, sha(site.teacher.model_dump()))
            if not same:
                self._revoke(site, "weights, environment or teacher changed since the certificate was issued")

    # ------------------------------------------------------------------ per call
    def observe(self, row: TraceRow, p: Policy) -> None:
        """Every Policy counter, and outage detection, from one logged row. The runtime calls this for every row."""
        site = self.rt.sites[row.call_site]
        p.rows += 1
        if row.agree is not None and row.phase in CALIBRATION:
            p.shadow_rows += 1
        if row.decision == Decision.audited and row.agree is not None:
            p.audit_rows += 1
            p.audit_errors += not row.agree
        if row.decision == Decision.predicted:
            p.predicted_rows += 1
        self.rt.dirty.add(site.id)
        if row.error is None and row.decision in (Decision.called, Decision.audited):
            self._failures.pop(site.id, None)
            if p.outage:  # any successful real call (a probe) clears a detected outage
                p.outage = False
                emit(self.rt.store, self.rt.component, EventKind.tool_recovered, f"{site.name} is back", call_site=site.id)
        elif row.error is not None:
            n = self._failures[site.id] = self._failures.get(site.id, 0) + 1
            if n == OUTAGE_AFTER and not p.outage:
                p.outage = True
                what = {Phase.dreaming: "the dream and the fallback keep answering", Phase.awake: "nothing has learned it yet"}.get(
                    p.phase, "its dream is not certified yet, so the fallback answers")
                emit(self.rt.store, self.rt.component, EventKind.tool_outage, f"{site.name} is down: {what}", call_site=site.id,
                     error=row.error)

    # ------------------------------------------------------------------ the tick
    def tick(self) -> None:
        self._apply_finished_training()
        for site in self.rt.sites.values():
            p = self.rt.policies[site.id]
            if site.kind.value == "effect" or (site.labels is None and site.kind.value != "tool"):
                continue  # effects and undeclared connector sites are logged only
            t = site.thresholds
            training = any(s == site.id for s, _ in self._pending)
            if p.phase == Phase.awake and not training and p.rows >= self._trained_at.get(site.id, 0) + t.watch_after:
                self._trained_at[site.id] = p.rows  # a failed attempt waits for watch_after more rows, not the next tick
                self._train(site)
            elif p.phase in (Phase.watching, Phase.testing) and p.shadow_rows >= self._checked_at.get(site.id, 0) + t.calibrate_after:
                self._checked_at[site.id] = p.shadow_rows
                if p.phase == Phase.watching:
                    self._set_phase(site, Phase.testing, f"checking the {site.name} dream against {p.shadow_rows} real answers")
                self._certify(site)
            elif (p.phase == Phase.dreaming and p.audit_rows >= max(AUDIT_MIN, audits_needed(self.rt.budget, t.delta))
                  and cp_upper(p.audit_errors, p.audit_rows, t.delta) > self.rt.budget):
                self._revoke(site, f"audits disagree too often: {p.audit_errors} of {p.audit_rows}", failover=True)

    def set_budget(self, budget: float) -> None:
        """Re-run the gate on every site's stored calibration rows. Pure math, no retraining."""
        self.rt.budget = budget
        emit(self.rt.store, self.rt.component, EventKind.budget_changed, f"mistakes allowed: {budget:.0%}", data={"budget": budget})
        for site in self.rt.sites.values():
            if self.rt.policies[site.id].phase in (Phase.testing, Phase.dreaming):
                self._certify(site)

    # ------------------------------------------------------------------ training (one job per candidate)
    def _train(self, site: CallSite) -> None:
        rows = self.rt.store.rows(site.id)
        ids = sorted(r.id for r in rows)
        for c in for_site(site):
            if c.needs_gpu:
                continue  # shortcut: GPU candidates (finetune) train in the campaign only; their weights do not fit the weights table
            req = TrainRequest(call_site=site.id, row_ids=ids, rows_sha=sha(ids), candidate=c.name)
            emit(self.rt.store, self.rt.component, EventKind.train_started, f"training a {c.name} dream for {site.name} on {len(rows)} rows",
                 call_site=site.id, data={"request": req.model_dump()})
            self._pending[(site.id, c.name)] = self._pool.submit(self.rt.trainer.train, req, site, rows)  # overlaps the next runs; applied at the next tick

    def _apply_finished_training(self) -> None:
        """Training overlaps the runs between two ticks; at the tick we wait for it, so every tick sees a settled state.
        The first weights back take the shadow slot; later candidates join as spares and shadow the same traffic from then on.
        shortcut: a slow job (a Modal cold start) pauses fast-forward for its duration; the narration says what it waits for."""
        for (site_id, name), fut in list(self._pending.items()):
            del self._pending[(site_id, name)]
            site, p = self.rt.sites[site_id], self.rt.policies[site_id]
            try:
                result = fut.result()
                predictor = self.rt.loader.load(site, self.rt.store.weights(result.weights_sha), result.candidate)
            except Exception as e:  # noqa: BLE001 - fail open: the site stays where it is, the dashboard says why
                emit(self.rt.store, self.rt.component, EventKind.train_failed, f"training {name} for {site.name} failed", call_site=site.id,
                     data={"dependency": "trainer", "where": type(self.rt.trainer).__name__, "candidate": name}, error=f"{type(e).__name__}: {e}")
                continue
            emit(self.rt.store, self.rt.component, EventKind.train_finished,
                 f"{site.name}: {name} weights {result.weights_sha[:6]} in {result.seconds:.1f}s on {result.where}", call_site=site.id,
                 data={"result": result.model_dump()})
            if site_id in self.rt.predictors:
                self.rt.spares.setdefault(site_id, {})[name] = predictor
                p.shadow[name] = Spare(weights_sha=predictor.weights_sha)
                self.rt.persist(p)
                continue
            self.rt.predictors[site_id] = predictor
            p.weights_sha, p.candidate = predictor.weights_sha, name
            p.shadow_rows, p.predicted_rows, p.audit_rows, p.audit_errors, p.cert_id, p.cutoff = 0, 0, 0, 0, None, None
            self._checked_at.pop(site_id, None)
            self.rt.persist(p)
            self._set_phase(site, Phase.watching, f"a dream is shadowing {site.name}: same input, answers compared")

    # ------------------------------------------------------------------ certification (the gate, every candidate at delta/N)
    def _candidates(self, site: CallSite) -> dict[str, Predictor]:
        p, pred = self.rt.policies[site.id], self.rt.predictors.get(site.id)
        return {} if pred is None else {p.candidate: pred, **self.rt.spares.get(site.id, {})}

    def _certify(self, site: CallSite) -> None:
        p, cands = self.rt.policies[site.id], self._candidates(site)
        if not cands:
            return
        delta = site.thresholds.delta / len(cands)  # selection and certification never share a bound: delta/N per candidate (DESIGN §4)
        rows = self.rt.store.rows(site.id, shadow_only=True)
        teacher_sha = sha(site.teacher.model_dump())
        certs = {name: issue(site.id, rows, self.rt.budget, delta, c.weights_sha, c.env_sha, teacher_sha, candidate=name) for name, c in cands.items()}
        best = max((c for c in certs.values() if c is not None), key=lambda c: (c.coverage, c.candidate == p.candidate), default=None)
        if best is None:
            conf, correct, _ = calibration(rows, cands[p.candidate].weights_sha)
            n = len(conf)
            p.attempt = Attempt(budget=self.rt.budget, n=n, emp_error=float((~correct).mean()) if n else 0.0,
                                bound=cp_upper(int((~correct).sum()), n, delta) if n else 1.0, min_budget=smallest_budget(conf, correct, delta))
            self.rt.persist(p)
            if p.cert_id:
                self._revoke(site, f"no promise at {self.rt.budget:.0%}: disagreed on {p.attempt.emp_error:.1%}, could be {p.attempt.bound:.1%}")
            return
        if best.candidate != p.candidate:
            self._swap(site, best.candidate)
        self._install(site, best, {name: c for name, c in certs.items() if c is not None and name != best.candidate},
                      f"the most certified coverage of {len(cands)} candidates at delta {delta:.3f}")
        emit(self.rt.store, self.rt.component, EventKind.cert_issued,
             f"{site.name} passed its check: wrong {best.emp_error:.1%} (bound {best.bound:.1%}), {best.budget:.0%} allowed",
             call_site=site.id, data={"certificate": best.model_dump(mode="json")})
        self._set_phase(site, Phase.dreaming, f"{site.name} is dreaming: answers {best.coverage:.0%} of calls")

    def _install(self, site: CallSite, cert: Certificate, spare_certs: dict[str, Certificate], why: str) -> None:
        """Store the new certificates (the old ones become superseded), point the Policy at the winner, list the spares."""
        p = self.rt.policies[site.id]
        with boundary(self.rt.store, self.rt.component, "store.put_certificate"):
            for old in [p.cert_id, *(s.cert_id for s in p.shadow.values())]:
                if old:  # a re-certification: the old promise is superseded, never silently overwritten
                    self.rt.store.put_certificate(self.rt.store.certificate(old).model_copy(update={"status": "superseded"}))
            for c in [cert, *spare_certs.values()]:
                self.rt.store.put_certificate(c)
        p.cert_id, p.cutoff, p.attempt, p.audit_rows, p.audit_errors = cert.id, cert.cutoff, None, 0, 0
        p.shadow = {name: Spare(weights_sha=pr.weights_sha, cert_id=(c := spare_certs.get(name)) and c.id, coverage=c.coverage if c else None)
                    for name, pr in self.rt.spares.get(site.id, {}).items()}
        self.rt.persist(p)
        if p.shadow:  # which candidate answers now, and why; the page's candidate rows read this
            spares = ", ".join(f"{n} ({s.coverage:.0%})" if s.coverage is not None else f"{n} (uncertified)" for n, s in sorted(p.shadow.items()))
            emit(self.rt.store, self.rt.component, EventKind.candidate_promoted, f"{site.name}: {p.candidate} answers: {why}; spares: {spares}",
                 call_site=site.id, data={"candidate": p.candidate, "cert_id": cert.id, "spares": sorted(p.shadow)})

    def _swap(self, site: CallSite, name: str) -> None:
        """The spare `name` answers from now on; the old winner becomes a spare. Weights stay loaded, nothing retrains."""
        p = self.rt.policies[site.id]
        spares = self.rt.spares.setdefault(site.id, {})
        old_name, old = p.candidate, self.rt.predictors[site.id]
        self.rt.predictors[site.id] = spares.pop(name)
        spares[old_name] = old
        p.candidate, p.weights_sha = name, self.rt.predictors[site.id].weights_sha

    # ------------------------------------------------------------------ revocation, and the spare rule
    def _revoke(self, site: CallSite, why: str, *, failover: bool = False) -> None:
        p = self.rt.policies[site.id]
        emit(self.rt.store, self.rt.component, EventKind.cert_revoked, f"{site.name}: {why}", call_site=site.id, data={"cert_id": p.cert_id or ""})
        revoked = p.cert_id
        if revoked:
            with boundary(self.rt.store, self.rt.component, "store.put_certificate", swallow=True):
                self.rt.store.put_certificate(self.rt.store.certificate(revoked).model_copy(update={"status": "revoked"}))
        p.cert_id, p.cutoff = None, None
        self.rt.persist(p)
        if failover and revoked and self._failover(site, revoked):
            return
        self._set_phase(site, Phase.testing, f"{site.name} keeps the real call")

    def _failover(self, site: CallSite, revoked: str) -> bool:
        """DESIGN §4: a spare takes over only if its own shadow predictions on the revoked certificate's audited rows pass the gate
        at the budget. Those rows are unbiased for the spare (it never chose which calls to audit); its old certificate is not.
        Arithmetic: the grid gate spends delta/40 per level, so at a 5% budget it needs ~130 error-free rows; a revocation at the
        59-audit minimum therefore hands over only when the audits ran longer. Otherwise the site goes to testing and re-certifies
        every candidate on fresh calibration rows at the next checkpoint (one `calibrate_after` later), which is the safe path."""
        audited = [r for r in self.rt.store.rows(site.id) if r.decision == Decision.audited and r.prediction is not None and r.prediction.cert_id == revoked]
        teacher_sha = sha(site.teacher.model_dump())
        for name, spare in sorted(self.rt.spares.get(site.id, {}).items()):
            picked = scored(audited, spare.weights_sha)
            conf = np.array([c for _, c, _ in picked], dtype=np.float64)
            correct = np.array([ok for _, _, ok in picked], dtype=bool)
            cert = _issue(site.id, conf, correct, sha(sorted(i for i, _, _ in picked)), self.rt.budget, site.thresholds.delta, spare.weights_sha,
                          spare.env_sha, teacher_sha, name)
            if cert is None:
                continue
            self._swap(site, name)
            self._install(site, cert, {}, f"its shadow answers on the {len(picked)} audited rows pass the gate at {self.rt.budget:.0%}")
            emit(self.rt.store, self.rt.component, EventKind.cert_issued,
                 f"{site.name} passed its check on the audited rows: wrong {cert.emp_error:.1%} (bound {cert.bound:.1%}), {cert.budget:.0%} allowed",
                 call_site=site.id, data={"certificate": cert.model_dump(mode="json")})
            return True
        return False

    def _set_phase(self, site: CallSite, phase: Phase, message: str) -> None:
        p = self.rt.policies[site.id]
        if p.phase == phase:
            return
        old, p.phase, p.updated_at = p.phase, phase, datetime.now(UTC)
        self.rt.persist(p)
        emit(self.rt.store, self.rt.component, EventKind.site_phase, message, call_site=site.id, data={"from": old, "to": phase})
