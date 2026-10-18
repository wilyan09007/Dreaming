# Dreaming — Missions

## What Dreaming is

A runtime that learns a model of every call your agent makes, then stops making the ones it can predict.

Every expensive thing an agent does is a call: a call to a frontier model to perform a step, a call to a tool or API, a decision to spend reasoning on a step, a request for a human to review an output. Dreaming logs every call, learns to predict what the call would have returned, proves the prediction is good enough, and then answers from the prediction instead of making the call. The real call stays as the fallback and as the source of truth that keeps the predictions honest.

The name comes from the Dreamer line of work in reinforcement learning, where an agent learns a model of its world and plans inside that model. Dreaming does the same for an agent's calls: it dreams each call before making it.

## Theme

**One mechanism, applied everywhere.**

Dreaming is not a collection of optimizations. It is one loop that applies to every kind of call:

1. **Log.** Every call writes one row: where it came from, what went in, what came out, how long it took, what it cost.
2. **Predict.** For each place a call is made, train a predictor of what the call returns.
3. **Shadow.** Run the predictor alongside the real call. Record whether they agree.
4. **Certify.** Once enough rows exist, compute the predictor's error rate against reality, with a confidence cutoff chosen so the error stays under a budget the operator sets.
5. **Promote.** The predictor answers first. The real call runs when the predictor is unsure, and on a sampled fraction of confident answers to keep the certificate current.
6. **Replay.** Anything Dreaming predicted can be re-executed exactly, because Dreaming controls the weights and the environment it runs them in.

Every mission below is this loop pointed at a different kind of call. The question each mission answers is: *for this kind of call, how far can the loop go?*

## Goal

Make agents cheaper, faster, and more reliable the more they run, and make every saving provable.

Concretely, for any agent workflow running on Dreaming, we want to show:

- **Cost per run falls** as predictors are promoted.
- **Latency falls** as calls are skipped or overlapped.
- **Failure rate falls** as the runtime learns what its tools will do and plans around them.
- **The share of decisions that can be re-executed exactly rises** as calls move from the frontier model, which can only be recorded, to predictors we host, which can be replayed.
- **Every promoted predictor carries a certificate**: a stated error bound, measured against reality, that the operator chose.

The demonstration is one agent workflow run many times with a chart of those lines moving, plus live moments where a judge changes the error budget and watches the system re-certify, picks a past decision and watches it replay, and kills a tool and watches the agent keep working because it already learned what that tool returns.

## Product synthesis

Dreaming fuses two ideas that are usually built separately.

**Distillation.** Teacher outputs become student labels. A small model learns to give the same answer as the frontier model for a narrow, repeated step, and takes that step over. This is how an agent gets cheaper.

**Environment modeling.** The agent learns what its environment, its tools and APIs, will do before it acts, and plans accordingly. This is how an agent gets more reliable.

They share everything that matters: the same data (traces), the same gate (shadow until agreement with reality clears a threshold), the same promotion logic (answer from the prediction, keep the real call as fallback). In Dreaming, a tool is a call site exactly like a model step. Predicting what `vendor_lookup` returns is the same job as predicting what `categorize_transaction` returns; the only difference is who the teacher is.

Two properties fall out of hosting the predictors ourselves:

- **Certification.** A frontier API gives you an answer and no confidence. Our predictors give us scores for every possible answer, so we can measure exactly how often we are wrong and at what confidence, and promise an error bound.
- **Replayability.** A frontier API at temperature zero can return different outputs on different runs; the arithmetic order inside the provider's servers depends on load, and rounding makes order matter. Our predictors run with fixed weights, fixed environment, and fixed input shapes, so the same input gives the same output byte for byte. Every promotion moves a decision from "can only be recorded" to "can be re-run."

The pitch in one line: **every saving comes with a certified error bound, and every decision can be replayed.**

---

## Missions, in experimentation priority

Each mission states what it predicts, how it works, what it buys, and what success looks like in the experiment.

### 1. Tier / thinking router

**Predicts:** whether a cheaper configuration of the frontier model gives the same answer as the expensive one for this call. A configuration is a tier (a smaller model from the same provider) or a reasoning setting (thinking on or off, or a smaller thinking budget).

**Method.**
1. For a call site, sample inputs from the trace log.
2. Run each input through the expensive configuration and the cheap one.
3. Count how often the answers match. For typed outputs (a label, a yes/no, an ID), match means equal. For free text, a judge model decides whether the two answers are equivalent.
4. If the match rate clears the error budget, route the call site to the cheap configuration. Keep sampling a fraction through the expensive one to hold the certificate.
5. Where the match rate is high for some inputs and low for others, train a small classifier on the inputs that predicts which configuration this particular call needs.

**Benefit.** Savings on day zero with nothing to train. Every step benefits, including free-text steps that no student can replace. It also tells us, for each step, how much the expensive configuration is actually contributing, which is the information every other mission needs.

**Worked example.** `flag_anomaly` runs with thinking on. Dreaming re-runs two hundred logged inputs with thinking off. The answer changed on three of them. Dreaming turns thinking off for that step and keeps re-checking one call in twenty. The step now returns in a fraction of the time with the same flags.

**Success looks like:** a per-step table of match rates between configurations, and a routing policy that holds the error budget while cutting tokens.

**Precedent:** cascade and routing systems (FrugalGPT, RouteLLM) and draft-agreement thinking routers (DART).

### 2. Students: per-step, shared workflow, adapters, weight generator

**Predicts:** the label the frontier model would return for a typed step: classify, extract, decide.

This mission has four variations. Each is an experiment in how to package the student. They share one rule: the student picks from a fixed set of outputs in a single forward pass. It never writes free text and it never calls tools. The runner calls tools; the student decides.

**Method, common to all variations.**
1. Build the student's own compact prompt from the step's schema: the input fields and the list of allowed outputs with one-line descriptions.
2. Run the student. It produces a score for every allowed output. The top output is the answer; the gap between the top two scores is the confidence.
3. Shadow against the frontier model. Log agreement.
4. Certify: sort shadow rows by confidence, find the cutoff where the error rate above it stays under budget, and promote with that cutoff.
5. Above the cutoff, the student answers. Below it, the frontier model answers and the pair becomes a new training row.

**Variation 2a: per-step student.** One small model fine-tuned per call site on that site's rows. Simplest to reason about. Training one step cannot change another.

**Variation 2b: shared workflow student.** One small model trained on every step's rows, each tagged with its step name, still called once per step. What it learns about the domain on one step is available on every other. New steps start from a model that already knows the workflow's vocabulary. Each retrain re-certifies every step by replaying logged inputs through the new weights.

**Variation 2c: per-step adapters on a shared base.** One base model loaded once; a small adapter (a few megabytes of extra weights) per step, selected by the step name. Training a step cannot change another step, and the base is shared in memory. Adapters are merged into the base before hashing so the served weights are one file.

**Variation 2d: weight generator (hypernetwork).** A model whose output is another model's weights. Input: a step's schema, the label descriptions, and a handful of example inputs if any exist. Output: the student's weights. Trained across many tasks: for each task, generate the student, score it on that task's examples, adjust the generator. Once trained, a new step gets a student from its schema alone, before any trace exists. The experiment starts with a generator that writes the weights of a linear classifier over text embeddings, measured against the obvious baseline (using the label descriptions' embeddings directly as the classifier), on tasks the generator never saw. The stretch is a generator that writes adapter weights for a small language model.

**Benefit.** The frontier call disappears for the step. Cost and latency on that step drop by the ratio of a small hosted model to a frontier API call. With 2d, a new step has a student the moment it is declared, and traces make it better from there.

**Worked example.** Step `categorize_transaction`, twelve categories. The runner renders:

```
Pick exactly one label.
  Office supplies — paper, toner, desk items
  Software        — subscriptions, licenses, app stores
  Meals           — restaurants, coffee, food delivery
  ...
Transaction: ADOBE *CREATIVE CLD, $54.99
Label:
```

The student scores all twelve labels in one pass: Software 0.71, Office supplies 0.22, the rest share the remainder. Answer: Software. Confidence: 0.49. The certified cutoff for this step is 0.40, so the answer is used and the row is logged with the student's weights hash, the environment hash, and the prompt hash.

**Success looks like:** for each step, an agreement curve over rows seen, a certificate, and the share of calls the student handles under the error budget. For 2b versus 2a: rows needed to certify a new step. For 2d: generated students beating the embedding baseline on unseen tasks, and fewer rows to certify when fine-tuning starts from generated weights.

**Precedent:** distillation of logged calls into specialists (OpenPipe; NVIDIA's LLM-to-SLM conversion recipe); fine-tuning one model on traces from every module of a pipeline (BetterTogether); generating adapters from task descriptions (Text-to-LoRA, Drag-and-Drop LLMs).

### 3. Tool return predictor + next-step speculation

**Predicts:** what a tool would return. Error is one of the possible return values. Latency is predicted alongside.

Tools fall into three cases, and the predictor does something different in each:

- **The return is a function of inputs we can see.** Example: a vendor lookup keyed on the merchant string. The predictor learns the mapping from traces. When certified, Dreaming answers from the prediction and skips the call. New inputs show up as low confidence and go to the real tool.
- **The return depends on state we cannot see.** Example: a refund's status at the bank. The predictor gives a distribution over returns. Dreaming does not skip the call; it **speculates**: it starts the next read-only step on the most likely return while the real call runs, keeps the work if the real return matches, discards it if not.
- **The call changes the world.** Example: issuing a refund. Nothing is skipped or speculated. The predictor's job is the outcome: will this call succeed, which error is likely, how long will it take. The runner uses that to set up the fallback before calling.

**Method.**
1. Every tool call logs the tool, its arguments, the time, the recent outcomes for that tool, the return, and the latency. Labels are free and exact: reality provides them on every call.
2. For each tool call site, train a predictor of the return distribution and latency. Text-shaped inputs use a small model like the students'; numeric and categorical inputs use a small tabular model.
3. Shadow, certify, and promote exactly as for students.
4. The runner's rule per call: certified top return → skip; confident but uncertified → speculate; error likely → fallback first; otherwise call normally.
5. Tool predictions also feed the students' gates: when a student's input came from a fallback source instead of the live tool, the gate requires more confidence.

**Benefit.** Paid and slow tool calls disappear where the tool is predictable. Latency overlaps where it is not. And the agent keeps working through an outage of a tool it has learned: transactions whose tool return is certified proceed as if the tool were up.

**Worked example.** A nightly import runs while the vendor API is down. For each transaction, the runner asks the vendor-return predictor first. Merchants seen before come back certified; those transactions proceed with no call. Unseen merchants come back low-confidence; the runner uses the registered fallback (a cached snapshot) and tags the row so downstream gates are more careful. The import finishes. A circuit breaker, by comparison, would have stopped every lookup.

**Speculation example.** `lookup_order` is running. The predictor says the order's status is `refunded` with high probability, and the workflow's next step on `refunded` is `get_refund_status`, which is read-only. The runner launches it immediately. When `lookup_order` returns `refunded`, the next step's result is already waiting. If it had returned `pending`, the speculative result is thrown away and nothing the user sees changes.

**Success looks like:** share of tool calls skipped at a certified error rate; speculation hit rate and latency saved; task completion during an injected outage versus a circuit-breaker baseline.

**Precedent:** value-equivalent world models that predict only decision-relevant quantities (MuZero); learned latency estimation in networks (TCP round-trip estimation); speculative pre-launch of predicted next calls (Speculative Actions, PASTE, AOSpec).

### 4. Human review deferral

**Predicts:** which outputs a human reviewer will change.

**Method.**
1. Every human correction is logged as a label on the row it corrected.
2. Train the gate's confidence head on those labels instead of on teacher agreement: given the student's scores, the tool flags, and the input, predict the probability the reviewer changes this output.
3. The operator sets a review budget: the share of outputs they are willing to look at, or the share of changes they must catch.
4. Dreaming queues the outputs most likely to be changed, up to the budget, and passes the rest through.
5. Reviewer decisions on the queued outputs are new labels. The head retrains on them.

**Benefit.** Reviewer time goes where it changes outcomes. Human labels are the strongest signal in the loop; a certificate against human corrections is the one that matters most to the operator.

**Worked example.** A bookkeeper reviews every categorization the agent produces. Dreaming trains the head on three months of her corrections. She sets the budget to a tenth of the volume. The queue now holds the transactions the head is least sure she would accept; she reviews those, and her decisions tighten the head.

**Success looks like:** a curve of changes caught versus share reviewed, and the review budget the operator can set from it.

**Precedent:** learning to defer; confidence meta-models that trust a prediction above a threshold and escalate below it.

### 5. Context pruner

**Predicts:** which parts of a prompt the output depends on.

**Method.**
1. For a call site, take logged inputs and split each prompt into spans: a record, a paragraph, a tool result.
2. Build labels by ablation: remove one span, re-run the call, mark whether the output changed.
3. Train a small scorer that reads the prompt and predicts, per span, whether it matters.
4. Shadow: run the pruned prompt and the full prompt side by side, measure agreement (exact for typed outputs, judge-based for free text), certify.
5. Promote: the call site runs on the pruned prompt.

**Benefit.** Input tokens on steps no student can replace. A free-text step like drafting a note cannot be distilled, but most of its prompt is history the note does not depend on.

**Worked example.** `draft_note` receives the transaction, the matched invoice, the anomaly flag, the full transaction history for the vendor, and the bank-feed record. The scorer learns that the note depends on the first three and rarely on the rest. The pruned prompt carries the first three plus whatever the scorer flags from the history. The note reads the same; the call carries a fraction of the tokens.

**Success looks like:** tokens removed per step at a certified agreement rate.

**Precedent:** learned context pruning for agents (SWE-Pruner, LaMR); token attribution for prompt compression.

### 6. Connectors to other orchestration platforms

**Goal:** any agent built on any framework gets Dreaming by wrapping the calls it already makes. Nobody rewrites their workflow on Dreaming.

**Principle.** Dreaming never takes over orchestration. The framework keeps deciding step order, branching, and retries. Dreaming wraps the four places where an agent already makes a call:
- **Model calls:** an in-process wrapper around the OpenAI-compatible client, or the HTTP proxy (`base_url` pointed at Dreaming) when the client cannot be wrapped. A header names the call site; untagged calls are logged only. The wrapper is preferred because it sees the request before it is a flat string. The proxy sees only strings, so proxied sites get tier routing and logging, and get students once someone declares the site's label set.
- **Tool calls:** a decorator around tool functions, and an MCP proxy for tools exposed over MCP. The wrapper consults the tool predictor, applies the runner's rule, and logs the outcome.
- **Agent-to-agent requests:** a wrapper around the framework's request/response primitive (in uAgents, `ctx.send_and_receive`). Target address plus message type is the call site; it gets the same tool predictor and fallback. One-way sends are effects and are logged only.
- **The handler as run boundary:** a decorator on the framework's entry point (a graph invocation, a message handler). It opens a run id, carries it in a context variable so every call inside lands under it, and closes the run on reply.

**What wrapping buys, and what it does not.** With the wrappers, an outside agent gets per-call prediction, certification, tool skip and outage survival, and replay of each predicted call. Whole-run replay is record-and-replay: in replay mode the wrappers serve logged responses in order; if the agent's own code is deterministic between calls the run replays exactly, and if the call sequence diverges replay reports the first divergence. Speculation needs to know the next step, so it is opt-in through a hint on the tool wrapper. Rollout across steps needs a declared step graph (see the runtime below).

**Worked example.** A LangGraph agent points its OpenAI client at Dreaming's proxy and tags its classification node with a call-site header. Nothing else changes. After enough runs, that node's calls are answered by a certified student and the agent's bill drops.

**Success looks like:** two agents on two different frameworks (for example LangGraph and uAgents), both running through the same wrappers, both showing promoted call sites.

### 7. Full rollout

**Predicts:** the whole episode before acting: imagine the plan, then execute it.

**Method.** Chain the predictors from missions 2 and 3. Given the task, predict the first step's output; given that, predict the tool's return; given that, predict the next step's output; and so on to the end. The imagined trajectory tells the runner where the plan will fail, which fallbacks to stage, and whether to route the task to a human before spending anything on it.

**Benefit.** Decisions about a run are made before the run. Trial and error becomes imagination and verification.

**Worked example.** A transaction arrives with an unfamiliar merchant and an amount far above the vendor's average. The rollout imagines: category uncertain → vendor lookup low confidence → invoice match likely `NONE` → anomaly flag `yes`. The runner routes the transaction to the review queue immediately, with the imagined trajectory attached as the reason.

**Success looks like:** how many steps ahead the imagined trajectory matches the real one, and the decisions it changes for the better.

**Precedent:** Dreamer, DreamGym, and the text-world-model line of work.

---

## Separate but necessary

### Hashing for replayability

Every prediction Dreaming makes must be re-executable byte for byte. For every hosted model, each trace row records:

- **Weights hash:** the exact model file that answered. Adapters are merged into the base first so there is one file to hash.
- **Environment hash:** hardware class, precision, inference library version, batch size.
- **Rendered prompt hash:** the exact text the model read, with everything resolved: retrieved examples, timestamps, IDs. We hash the rendered text, never the template.
- **Certificate ID:** which certificate was in force when the answer was used.

Predictors run with fixed input shapes at batch size one, so the arithmetic path does not depend on other traffic. Before any predictor is called deterministic, it runs the same input a thousand times under concurrent load and must produce exactly one distinct output.

`replay <run_id>` loads a run's rows in order, serves every frontier answer and tool return from the log, re-executes every prediction on its logged prompt, and reports whether each matched. `recertify <cert_id>` re-runs a certificate's calibration set and reproduces its error count. `backtest <model> <window>` replays a window of logged inputs through a candidate model and produces its certificate before it touches live traffic. Any change to weights, environment, or prompt template sends a call site back through the gate.

### Our own orchestration runtime

Whole-run replay, backtesting, speculation, and rollout all require that the workflow makes the same calls in the same order whenever it gets the same responses. That is guaranteed when Dreaming runs the steps, and holds for a wrapped agent whose code is deterministic between calls. The runtime is built to the demo, not to scale: it must work, it does not need to survive heavy load or every race.

The runtime is not a requirement for using Dreaming. Its core is the three things the connectors already use: the run-boundary decorator, the call wrappers, and the replay mode. The step registry with declared order is what you add when writing a new workflow, or when you want speculation and rollout, which need the step graph. Connectors and runtime share one routing function.

What it does:
- **Step registry.** `@step("categorize_transaction")` declares a call site, its typed output, and its fallback. The student's day-zero prompt is rendered from this declaration.
- **Deterministic step order.** Steps run in the declared order, with the declared dependencies.
- **Routing.** For every call: consult the relevant predictor, apply the certified cutoff, route to student, frontier, or fallback.
- **Trace logging** with the hashes above.
- **Replay, recertify, backtest** as commands.
- **Fast-forward mode.** Run a workflow many times quickly against cached tool responses, so learning is visible in minutes.

The proxy and tool wrapper from mission 6 are part of the runtime, so outside agents plug into the same loop.

### Connector to ASI:One / Agentverse (Fetch)

uAgents is a framework like any other under mission 6, not a separate integration. An existing uAgent keeps its handlers, tools, and agent-to-agent messages and adds the four wrappers: the model client wrapper (ASI:One's model API is OpenAI-compatible, so the same wrapper or proxy applies), the tool decorator, the `send_and_receive` wrapper, and the run-boundary decorator on the chat handler. Nothing is rewritten on Dreaming.

Two deployment cases:
- **Agent run locally with a mailbox to Agentverse:** the full in-process library runs alongside it (encoder, predictors, trace store). This is the demo case.
- **Agent hosted on Agentverse:** hosted agents run in a restricted Python sandbox, so the library cannot run there. Model calls go through the proxy, and tool decisions are asked over a small HTTP endpoint ("what would this tool return").

The demo agent is itself a uAgent wrapped this way and reachable from ASI:One through the chat protocol. A user in ASI:One submits work; the handler runs with every call going through the same log, gate, and predictors; the result comes back with a summary of which calls were predicted and under which certificates. Tool recovery under injected failures runs there as it does everywhere else. The ASI:One connector and the mission 6 connector are the same code.

---

Tracks, sponsor prizes, and how we use each are in `tracks.md`.

---

## How we work

- **Experiments over opinions.** Each mission ends in a measurement. Where two variations compete, we run both and let the numbers decide.
- **One mechanism.** If a mission needs a second gate, a second log, or a second kind of certificate, it is not yet understood. Bring it back to the loop.
- **Reality is the teacher.** Certificates are computed against real returns and real corrections, never against synthetic data.
- **Nothing shown is faked.** Pre-recorded material is labeled as pre-recorded. Live moments are live.
- **Replayable or not promoted.** A predictor that cannot pass the determinism check does not leave shadow mode.

## Vocabulary

- **Call site:** one place in a workflow where a call is made. The unit everything is keyed on.
- **Trace row:** the record of one call: where, input, output, timing, cost, hashes.
- **Teacher:** whatever produces the true answer for a call site. A frontier model, a tool, or a human.
- **Student / predictor:** the model that learns to answer for a call site.
- **Shadow:** running the predictor alongside the real call and recording agreement, without using the prediction.
- **Confidence:** the gap between the predictor's top two scores, or a learned score built from it.
- **Certificate:** a statement that, above a given confidence cutoff, a predictor's error rate against reality stays under a budget. It records the weights, environment, calibration set, cutoff, and error count.
- **Promotion:** the point at which a predictor answers first.
- **Audit sample:** the fraction of confident answers still sent to the teacher to keep the certificate current.
- **Fallback:** the registered alternative when a tool is predicted to fail: a snapshot, a cache, a queue.
- **Speculation:** starting the next step on a predicted return before the real one arrives, keeping the work only if the prediction matches.
- **Replay:** re-executing a run or a certificate from its logged rows and confirming byte-identical results.
- **Run boundary:** the wrapped entry point (a handler, a graph invocation) that opens and closes a run id; every call made inside it lands under that run.
