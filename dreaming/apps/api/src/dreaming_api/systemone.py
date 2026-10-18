"""TypeSafe System One wire format over our students: any trained call site answers `choice` questions in Intuit's API shape.

    systemone_router(rt)   the APIRouter create_app includes; one route:
    POST /v1/systemone     {"model", "state", "questions": {id: {"type": "choice", "instructions", "criteria": {option: description}}}}
                           -> {"model", "answers": {id: {"type": "choice", "choice", "probabilities", "confidence", "certified", "certificate"}}, "usage"}

A question id is a call site's name or id; `criteria` must list exactly the site's labels (a student has a fixed label set).
`state` is the site's rendered input: the string itself, else its canonical JSON. 422: unknown id, label mismatch, or a
question type other than "choice" (score and noul are not served). 409: the site has no trained predictor yet.

This serves the student's answer whether or not a certificate covers it (the benchmark path measures the raw student), and says
which: `certified` is true when the site's policy holds a certificate and the answer's confidence clears its cutoff, `certificate`
is that certificate's id. The SDK's answer models ignore unknown fields, so typesafe-sdk clients parse the reply unchanged
(tests/test_routes.py drives the route through `typesafe_sdk.TypeSafeClient`). `model` names what answered,
"dreaming/<site>@<weights sha[:12]>" per question, not the model the client asked for. Bodies must be uncompressed JSON: the SDK
sends zstd only when backports-zstd is installed, so a client with it passes `compression=False`.
"""

from __future__ import annotations

from typing import Literal

from dreaming.contracts import Json
from dreaming.gate import CONF_DECIMALS
from dreaming.hashing import canonical
from dreaming.runtime import Runtime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


class Question(BaseModel):
    type: str
    instructions: str = ""
    criteria: Json = None  # choice: {option: description}; score (a list) is rejected below


class SystemOneIn(BaseModel):
    model: str
    state: Json
    questions: dict[str, Question]


class Answer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float
    certified: bool = False
    certificate: str | None = None


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class SystemOneOut(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage


def systemone_router(rt: Runtime) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/systemone")
    def systemone(body: SystemOneIn) -> SystemOneOut:
        rendered = body.state if isinstance(body.state, str) else canonical(body.state).decode()
        answers: dict[str, Answer] = {}
        models: list[str] = []
        for qid, q in body.questions.items():
            if q.type != "choice":
                raise HTTPException(422, f"question {qid}: only \"choice\" questions are served, not {q.type!r}")
            site = rt.sites.get(qid) or next((s for s in rt.sites.values() if s.name == qid), None)
            if site is None:
                raise HTTPException(422, f"question {qid}: no call site named {qid}")
            labels = site.labels or []
            if not labels or not isinstance(q.criteria, dict) or set(q.criteria) != set(labels):
                raise HTTPException(422, f"question {qid}: criteria must be exactly the options of {site.id}: {labels}")
            predictor = rt.predictors.get(site.id)
            if predictor is None:
                raise HTTPException(409, f"nothing trained for {site.id} yet")
            pred = predictor.predict(rendered)
            policy = rt.policies.get(site.id)
            cut = policy.cutoff if policy and policy.cert_id else None
            models.append(f"dreaming/{site.id}@{predictor.weights_sha[:12]}")
            probs = {lab: pred.scores.get(lab, 0.0) for lab in labels} if pred.scores else {lab: float(lab == pred.label) for lab in labels}
            top = sorted(probs.values(), reverse=True)
            answers[qid] = Answer(choice=pred.label, probabilities=probs, confidence=round(top[0] - (top[1] if len(top) > 1 else 0.0), CONF_DECIMALS),
                                  certified=cut is not None and pred.confidence >= cut, certificate=policy.cert_id if policy else None)
        # shortcut: whitespace token counts, indicative only; the encoder's tokenizer count is the upgrade if a client bills on usage
        return SystemOneOut(model=",".join(models), answers=answers, usage=Usage(
            input_tokens=len(rendered.split()) * len(answers), output_tokens=sum(len(a.choice.split()) for a in answers.values())))

    return router
