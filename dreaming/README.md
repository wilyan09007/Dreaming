# Dreaming (mock repo)

A runtime that learns a model of every call your agent makes, then stops making the ones it can predict. Product spec:
`../missions.md`. This folder pins the demo's architecture (`DESIGN.md`), the interface contracts
(`packages/dreaming/src/dreaming/contracts.py`, `db/migrations/0001_init.sql`), and the layout (`TREE.md`).
What is real and what is a stub: every stub says `shortcut:` in its docstring and raises `NotImplementedError`.

## Run it

```sh
uv sync                 # one venv for the library, the API and the agents (python >= 3.12)
just check              # ruff + pyright strict + pytest: the whole loop runs offline in under a second
just fast-forward 2000  # log → shadow → certify → promote → budget → outage, no accounts, no network
just dev                # the API on :8080 with the memory store; the page at /
```

With accounts (see `infra/README.md` for the one-time sign-ins): `just migrate`, `just deploy-trainer`, `just deploy-api`, `just smoke`.

## Where to look

| Question | File |
|---|---|
| What does the demo do and which service does what | `DESIGN.md` |
| What can change (predictor, teacher, trainer, encoder) and how it slots in | `DESIGN.md` §5 |
| Which component serves which track, and what we can claim | `docs/tracks.md` |
| What trains tonight, in parallel, and how it is monitored and pruned | `docs/training-plan.md` |
| Can nodes run in parallel on predicted values | `docs/speculation.md` |
| How to use a sponsor's tool during development | `docs/devtools.md` |
| What are the objects that cross a boundary | `packages/dreaming/src/dreaming/contracts.py` |
| How does one call get routed (the decision table) | `packages/dreaming/src/dreaming/router.py` |
| How does a call site go awake → dreaming | `packages/dreaming/src/dreaming/runtime.py` |
| What does "certified" mean, exactly | `packages/dreaming/src/dreaming/gate.py` |
| Why is a thing red | `Event.component` + `Event.kind` + `Event.error`, and `GET /healthz` |
| Service facts, prices, gotchas | `infra/SERVICES.md` |
