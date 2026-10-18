"""Everything that crosses a boundary: process, HTTP, database, browser. (Named contracts, not types, so it never shadows the stdlib.)

Four objects carry the whole system:

  CallSite     one declared place a call is made (static: comes from the registry)
  TraceRow     one row per call: who answered, what it cost, and the hashes that make it replayable
  Certificate  one promise per promotion: above this cutoff, this predictor is wrong at most this often
  Event        one line per thing worth seeing: phase changes, certificates, jobs, outages, errors

Policy is the per-site state the router reads. Run, Metrics, TrainRequest/TrainResult, ReplayReport and
Health are the views the apps exchange. TrainRun, Checkpoint and EvalSet are the offline campaign's rows
(docs/training-plan.md §3). Nothing else in the library is a type a reader needs to know.
`just schema` exports these as JSON Schema for the dashboard, so the browser and Python agree by construction.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Json = dict[str, Any] | list[Any] | str | int | float | bool | None


class Kind(StrEnum):
    """What a call site calls. The kind picks the predictor family and what 'skip' means."""

    model = "model"  # frontier-model step with a typed output: a student can replace it
    tool = "tool"  # tool or API call: a tool-return predictor can replace it
    router = "router"  # frontier-model step with a cheaper configuration on trial (tier, thinking off)
    a2a = "a2a"  # agent-to-agent request: predicted like a tool
    effect = "effect"  # changes the world: logged only, never predicted


class Phase(StrEnum):
    """The state machine of one call site. The dashboard's five stickers are exactly these."""

    awake = "awake"  # the real call answers; no predictor yet
    watching = "watching"  # a predictor runs in shadow; agreement is logged, never used
    testing = "testing"  # enough shadow rows: certifying
    dreaming = "dreaming"  # promoted: the predictor answers above the cutoff; 1 in N confident answers still audited
    down = "down"  # tools only, shown not stored: Policy.outage is set; the predictor (if certified) and the fallback answer


class Decision(StrEnum):
    """Who answered one call."""

    called = "called"  # the real call (no predictor, unsure, or shadowing)
    predicted = "predicted"  # the predictor; the real call was skipped
    audited = "audited"  # the predictor was confident but the real call ran anyway (audit sample); both logged
    fallback = "fallback"  # the real call failed or is down; the registered fallback answered


class Component(StrEnum):
    """Who emitted an event or log line. The first field to read when something breaks."""

    lib = "lib"
    api = "api"
    agent = "agent"
    trainer = "trainer"


class EventKind(StrEnum):
    batch_started = "batch.started"
    batch_progress = "batch.progress"  # data: Metrics; every 20 runs during fast-forward
    batch_finished = "batch.finished"
    site_phase = "site.phase"  # data: {"from": Phase, "to": Phase}
    cert_issued = "cert.issued"  # data: Certificate
    cert_revoked = "cert.revoked"
    train_started = "train.started"  # data: TrainRequest
    train_finished = "train.finished"  # data: TrainResult
    train_failed = "train.failed"  # error set
    tool_outage = "tool.outage"
    tool_recovered = "tool.recovered"
    budget_changed = "budget.changed"  # data: {"budget": float}
    replay_finished = "replay.finished"  # data: ReplayReport
    candidate_promoted = "candidate.promoted"  # data: {"candidate": str, "cert_id": str, "spares": [str]}: which candidate answers now, and why
    campaign_progress = "campaign.progress"  # data: the monitor's table per site (docs/training-plan.md §4); every 10 minutes
    component_error = "component.error"  # anything caught at a boundary; error set


# ---------------------------------------------------------------- declarations


class TeacherSpec(BaseModel):
    """The real call, pinned. Provider pinning is part of what a certificate means (S13)."""

    provider: str = "openrouter"  # openrouter | asi1 | tool | human
    model: str = ""  # e.g. "anthropic/claude-haiku-5.5"
    params: dict[str, Json] = Field(default_factory=dict)  # temperature, reasoning, provider.order: sent as-is
    price_in: float = 0.0  # $ per million input tokens, used when the provider reports no cost
    price_out: float = 0.0
    price_call: float = 0.0  # $ per call, for tools and APIs that bill per request
    floor: float | None = None  # the teacher's measured self-disagreement on identical inputs (S13): no budget below it is measurable


class Thresholds(BaseModel):
    watch_after: int = 60  # rows before a predictor starts shadowing (the day-zero student trains here)
    calibrate_after: int = 500  # shadow rows before certifying (S1: 200 certify most coverage, 500 saturate)
    audit_rate: float = 0.05  # share of confident answers still sent to the teacher
    delta: float = 0.05  # the certificate holds with probability 1 - delta
    speculate_at: float | None = None  # confidence at which a dependent may start on this site's guess (docs/speculation.md); None: never


class CallSite(BaseModel):
    """One place in a workflow where a call is made. The unit everything is keyed on."""

    id: str  # "<workflow>.<name>", e.g. "bookkeeper.categorize"
    workflow: str
    name: str
    kind: Kind
    labels: list[str] | None = None  # allowed outputs for typed steps; None: tool returns are learned from traces
    descriptions: dict[str, str] = Field(default_factory=dict)  # one line per label; renders the day-zero prompt
    teacher: TeacherSpec = Field(default_factory=TeacherSpec)
    cheap: TeacherSpec | None = None  # router sites: the cheaper configuration on trial (its "weights" are this spec)
    fallback: str | None = None  # name of the registered fallback (snapshot, cache, queue); tools only
    replayable: bool = True  # False for routers: a cheaper frontier configuration is still a frontier call
    thresholds: Thresholds = Field(default_factory=Thresholds)
    candidates: list[str] = Field(default_factory=list)  # predictor families to train in parallel (predictors.CANDIDATES); [] = the kind's default


# ---------------------------------------------------------------- one row per call


class Hashes(BaseModel):
    """What makes a prediction re-executable byte for byte (missions.md › Hashing for replayability)."""

    prompt: str  # sha256 of the rendered input, never the template
    weights: str | None = None  # sha256 of the weights file (adapters merged first)
    env: str | None = None  # sha256 of the execution profile: image digest, library versions, threads, batch size, SKU


class Evidence(BaseModel):
    """One stored row's share of a memory read's vote (predictors/memory.py). Derived at read time: not in the hashes or the certified output."""

    row_id: str  # a trace row id, or "seed:<label>:<i>" for an LLM-written seed row
    label: str
    weight: float  # this row's share of the whole vote, rounded to 3 decimals; a label's rows sum to its score


class Prediction(BaseModel):
    label: str
    confidence: float  # top-2 gap, rounded to 4 decimals before thresholding or hashing (S7)
    scores: dict[str, float] = Field(default_factory=dict)
    hashes: Hashes
    cert_id: str | None = None  # the certificate in force when this answer was used; None while shadowing
    cost_usd: float = 0.0  # what producing this prediction cost: ~0 for a student, a real call for a router's cheap config
    latency_ms: int = 0
    candidate: str = ""  # which family produced it (predictors.CANDIDATES key); "" on rows written before migration 0002
    evidence: list[Evidence] | None = None  # the memory candidate's top 3 voting rows; None for every other family


class TeacherAnswer(BaseModel):
    text: str  # raw text; `label` is the parsed typed output
    label: str | None = None
    provider: str  # what actually served: OpenRouter reports it per response; log it in every row
    model: str
    cost_usd: float = 0.0
    latency_ms: int = 0
    usage: dict[str, int] = Field(default_factory=dict)
    recorded: bool = False  # served from the pre-recorded cache in fast-forward mode; the UI says so


class TraceRow(BaseModel):
    """The record of one call. Replay serves a run's rows in `i` order."""

    id: str
    run_id: str
    i: int  # 1-based position of this call in its run
    call_site: str
    kind: Kind
    phase: Phase = Phase.awake  # the site's phase when this call was made; calibration rows are the watching/testing ones
    request: Json
    request_sha: str  # canonical-JSON sha256 of request; the predictor key covers model, tools and format too
    response: Json  # what the caller received
    decision: Decision
    prediction: Prediction | None = None  # the promoted (or only) candidate's prediction, logged whenever it ran, even when it did not answer
    shadow: dict[str, Prediction] = Field(default_factory=dict)  # every other candidate's prediction on the same input, by candidate name
    teacher: TeacherAnswer | None = None  # logged whenever the real call ran
    agree: bool | None = None  # prediction.label == teacher.label when both exist: free calibration data (per-candidate agreement is recomputed from `shadow`)
    speculated: bool = False  # the call rendered on a guessed input (docs/speculation.md); never true for effects
    kept: bool | None = None  # speculated rows only: the guesses were verified equal (True) or the work was discarded (False)
    latency_ms: int
    cost_usd: float
    error: str | None = None
    ts: datetime


# ---------------------------------------------------------------- the promise


class Certificate(BaseModel):
    """Above `cutoff`, this predictor's error rate against the teacher stays under `budget`,
    with probability 1 - delta, computed on exactly the rows `calib_sha` names (fixed-grid Clopper-Pearson, gate.py)."""

    id: str
    call_site: str
    candidate: str = ""  # the family this certifies; "" before migration 0002
    weights_sha: str
    env_sha: str
    teacher_sha: str  # sha256 of the pinned TeacherSpec: the reference the error is measured against (S13)
    budget: float
    delta: float
    cutoff: float
    coverage: float  # share of calibration rows at or above the cutoff
    emp_error: float  # observed error above the cutoff
    bound: float  # upper confidence bound on that error; bound <= budget is what "certified" means
    n_calib: int  # calibration rows the gate looked at (shadow rows collected before promotion)
    n_above: int  # of which at or above the cutoff
    n_errors: int  # errors among those
    calib_sha: str  # sha256 of the sorted calibration row ids: recertify re-runs exactly these
    created_at: datetime
    status: Literal["active", "superseded", "revoked"] = "active"


class Attempt(BaseModel):
    """What the gate found when nothing certified at the current budget: the page's "no promise at 5%" numbers."""

    budget: float
    n: int  # shadow rows the gate looked at
    emp_error: float  # disagreement over all of them
    bound: float  # upper bound on it at full coverage
    min_budget: float | None  # smallest budget on a 1% grid at which something certifies; None up to 20%


class Spare(BaseModel):
    """A candidate that is not promoted: its latest weights and, if it certified, its certificate and coverage. The hot spare."""

    weights_sha: str
    cert_id: str | None = None
    coverage: float | None = None


class Policy(BaseModel):
    """Per-site state the router reads on every call. One row per call site; every change emits site.phase."""

    call_site: str
    phase: Phase = Phase.awake
    cert_id: str | None = None
    cutoff: float | None = None  # None: nothing certified, the predictor never answers
    weights_sha: str | None = None  # the promoted candidate's weights
    candidate: str = ""  # the promoted candidate; "" before migration 0002
    shadow: dict[str, Spare] = Field(default_factory=dict)  # every other candidate, by name: promoted next if an audit revokes the winner
    rows: int = 0  # calls logged at this site
    shadow_rows: int = 0  # calibration rows: prediction + teacher answer, under the current weights, before promotion
    predicted_rows: int = 0  # calls the predictor answered
    audit_rows: int = 0  # confident answers the teacher double-checked after promotion
    audit_errors: int = 0  # of which disagreed: the standing check on the certificate
    attempt: Attempt | None = None  # set while testing fails; cleared when a certificate is issued
    outage: bool = False  # detected: three consecutive failures of the real call; cleared by a successful probe
    updated_at: datetime


# ---------------------------------------------------------------- views the apps exchange


class Run(BaseModel):
    id: str
    n: int = 0  # ordinal within the workflow: "run 1,250"
    workflow: str
    mode: Literal["live", "fast_forward"]
    input: Json  # the transaction: the receipt on the replay tab
    started_at: datetime
    finished_at: datetime | None = None
    calls: int = 0
    decisions: int = 0  # calls at sites that can be predicted (everything but effects)
    cost_usd: float = 0.0
    latency_ms: int = 0
    failed: bool = False
    stuck_at: str | None = None  # call site where a run could not continue (outage, nothing learned)
    replayable: int = 0  # decisions a hosted predictor answered: these re-execute byte for byte
    down: list[str] = Field(default_factory=list)  # call sites that were unplugged while this run ran
    critical_ms: int = 0  # latency along the longest dependency chain; equals latency_ms until nodes run in parallel (docs/speculation.md)


class Metrics(BaseModel):
    """One point of the 'the more it runs' chart: `runs` is the x (total so far), the rest averages over the last window."""

    runs: int
    cost_per_run: float
    sec_per_run: float
    failed_pct: float
    replayable_pct: float
    saved_usd: float  # cumulative, against the all-awake baseline


class Event(BaseModel):
    id: str
    ts: datetime
    component: Component
    kind: EventKind
    message: str  # one human sentence; the dashboard narrates these
    run_id: str | None = None
    call_site: str | None = None
    data: dict[str, Json] = Field(default_factory=dict)
    error: str | None = None  # set => something broke; component + kind + error name the root cause
    trace_id: str | None = None  # the W3C traceparent trace id of the HTTP request this happened under (telemetry.TRACE)


class Batch(BaseModel):
    """Fast-forward status, for the run button and the counter."""

    running: bool = False
    paused: bool = False
    target: int = 0
    done: int = 0
    recorded: bool = False  # the teacher serves pre-recorded answers; the page says so
    live_calls: int = 0  # teacher answers in this batch that were NOT pre-recorded (misses went live): the page shows the count


class SiteView(BaseModel):
    """One call site as the page shows it: the declaration, its state, its promise, and what a call costs awake vs dreaming."""

    site: CallSite
    policy: Policy
    certificate: Certificate | None = None
    cost_awake_usd: float = 0.0  # mean cost of rows the real call answered
    cost_dream_usd: float = 0.0  # a call under the promise in force: covered share at the prediction's cost (1 in N audited), the rest awake
    sec_awake: float = 0.0
    sec_dream: float = 0.0


class State(BaseModel):
    """GET /state: everything the page needs on load. Live changes arrive through Realtime (events, policies, certificates)."""

    sites: list[SiteView]
    budget: float
    metrics: Metrics
    series: list[Metrics]  # batch.progress snapshots, oldest first
    runs: list[Run]  # newest last; refetch on batch.progress (the runs table is not in the Realtime publication)
    first_run: Run | None = None  # run 1, kept on the receipt list for contrast
    batch: Batch
    config: dict[str, str] = Field(default_factory=dict)  # supabase_url, supabase_publishable_key: what Realtime needs
    spare_certificates: dict[str, Certificate] = Field(default_factory=dict)  # Policy.shadow[name].cert_id -> its certificate: the journal prints the spares' bounds
    unplugged: list[str] = Field(default_factory=list)  # tool sites whose fault switch is off (POST /tools/{site}/outage): the plug pill and, with Policy.outage, the rain cloud


class RunView(BaseModel):
    run: Run
    rows: list[TraceRow]
    evidence_texts: dict[str, str] = Field(default_factory=dict)  # GET /runs/{id}: rendered input of each evidence row found in trace_rows


class TrainRequest(BaseModel):
    """The trainer is a pure function of this: same request, same rows => same weights_sha."""

    call_site: str
    row_ids: list[str]  # exactly the rows the job may read; rows_sha names them
    rows_sha: str  # sha256 of the sorted row_ids
    encoder: str = "BAAI/bge-small-en-v1.5"
    candidate: str = "head"  # which family to fit (predictors.CANDIDATES); the learner submits one request per candidate
    finetune: bool = False  # kept for rows written before `candidate` existed; `candidate="finetune"` is the same thing


class TrainResult(BaseModel):
    """Weights, never a certificate: certificates come only from shadow rows the weights never trained on (replay.issue)."""

    call_site: str
    candidate: str = "head"  # which family the weights are (predictors.CANDIDATES); the learner loads them with it
    weights_sha: str
    env_sha: str
    n_train: int
    holdout_accuracy: float  # on a held-out slice of the training rows, for the record
    seconds: float
    where: str  # "modal:T4" | "modal:cpu" | "local:cpu" | "veri"


# ---------------------------------------------------------------- the offline campaign (docs/training-plan.md §3)


class TrainRun(BaseModel):
    """One cell of the matrix: one site, one candidate, one data variant, one seed. Written by the cell itself; never a Policy."""

    id: str
    campaign: str  # a name for the whole matrix, e.g. "night-1017"
    call_site: str
    candidate: str
    variant: Literal["seed+real", "real", "seed"]  # docs/training-plan.md §3
    seed: int
    config: dict[str, Json] = Field(default_factory=dict)  # everything else the cell was given, so it can be re-run
    where: str  # "modal:cpu" | "modal:T4" | "veri" | "local:cpu"
    status: Literal["queued", "running", "done", "failed", "pruned"] = "queued"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    seconds: float = 0.0
    cost_usd: float = 0.0
    note: str | None = None  # why it failed or was pruned (the rule that fired)


class Checkpoint(BaseModel):
    """What a cell writes at each rung. `scores_uri` is the raw material: per eval row, a score per label, twice (determinism)."""

    run_id: str
    rung: int  # rows seen: 60, 200, 500, 2000
    weights_sha: str
    eval_sha: str  # EvalSet.sha: which rows these scores are over
    scores_uri: str  # one .npz on the Volume: row index, scores float16, quantized confidence, label, second-pass scores
    seconds: float  # fit time at this rung
    embed_ms: float  # per call, measured on the cell
    head_ms: float
    bytes: int  # serialized weights
    stable: bool  # second forward pass byte-equal in (label, quantized confidence): the hard-fail check
    summary: dict[str, float] = Field(default_factory=dict)  # acc, coverage_02/05/10/20, bound, emp_error, ece, n_eval: one query for the monitor
    ts: datetime


class EvalSet(BaseModel):
    """A frozen slice of rows, by sha, so every candidate is scored on exactly the same rows (paired comparisons)."""

    sha: str  # sha256 of the sorted row_ids
    call_site: str
    split: Literal["calib", "test", "forward"]  # held out by group; "forward" = later in time than every training row
    row_ids: list[str]
    labels: list[str]  # aligned with row_ids
    groups: list[str]  # aligned: merchant or company, what the holdout was drawn by


class ReplayStep(BaseModel):
    i: int
    call_site: str
    decision: Decision
    re_executed: bool  # True for predicted rows: the predictor ran again on the logged prompt
    same_bytes: bool | None = None  # None when served from the log instead
    output_sha: str | None = None


class ReplayReport(BaseModel):
    run_id: str
    steps: list[ReplayStep]
    re_executed: int
    same: int
    from_log: int


class Health(BaseModel):
    """What /healthz returns. `checks` names each dependency and its state, so a red dashboard says who is red."""

    model_config = ConfigDict(extra="forbid")

    component: Component
    ok: bool
    checks: dict[str, str]  # "db": "ok", "trainer": "unreachable: ...", "teacher": "ok (openrouter)"
    version: str
    env_sha: str
    profile: dict[str, str] = Field(default_factory=dict)  # the execution profile behind env_sha: image, threads, batch, device
