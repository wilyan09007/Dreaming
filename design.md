# Dreaming — System Design (full-scale, minimal)

Five ways to place the system, plus how the uAgents / ASI:One connector fits. They differ on three questions: **where the predict-or-call decision runs**, **where customer data lives**, and **how much backend storage is needed**.

## A. Hosted gateway

Two halves: the **request path**, which sits inside every call the customer's agent makes, and **everything off the request path**, which trains, certifies, audits, and replays.

```mermaid
flowchart TD
    Agent["Customer's agent<br/>SDK decorators · OpenAI client pointed at us · MCP proxy"]

    subgraph hot["Request path"]
        LB[Load balancer]
        GW["Gateway (stateless, many copies)<br/>policy cache · CPU predictors in-process<br/>decide: predict / forward / fallback · emit one trace event"]
    end

    Provider["Frontier provider<br/>OpenAI · Anthropic · OpenRouter"]
    GPU["GPU predictor pool<br/>LoRA students only, pinned image, batch size 1"]

    subgraph cold["Off the request path"]
        Q[Event queue]
        Writer[Trace writer]
        Audit["Audit monitor<br/>recompute error bounds · revoke"]
        Obj["Object storage<br/>prompt/output bodies · weights, by hash"]
        Analytics["Analytics store<br/>one row per call · by site and time"]
        Trainer["Trainer jobs<br/>train · backtest · certify · determinism check"]
        Meta["Metadata DB (Postgres)<br/>customers · call sites · budgets · certificates · corrections"]
        Publisher[Policy publisher]
        Replay["Replay / recertify / backtest jobs<br/>one pinned image per environment hash"]
        Review[Review queue + UI]
        Ops[Operator API + dashboard]
    end

    Agent --> LB --> GW
    GW -- forward --> Provider
    GW -- large students --> GPU
    GW -- one event per call --> Q
    Q --> Writer
    Q --> Audit
    Writer --> Obj
    Writer --> Analytics
    Analytics --> Trainer
    Obj --> Trainer
    Trainer -- weights --> Obj
    Trainer -- certificates --> Meta
    Audit -- revoke --> Meta
    Meta --> Publisher -- cutoffs · weights hash --> GW
    Obj -. weights by hash .-> GW
    Meta --> Replay
    Obj --> Replay
    Review --> Meta
    Ops --> Meta
```

Rules the gateway must keep: fail open (any internal error becomes a plain forward), never wait on storage, add milliseconds not tens.

## B. Library in the agent's process (recommended core)

The decision runs inside the customer's process, wrapping the four places an agent already makes calls. The backend shrinks to Postgres, object storage, and a trainer. The proxy is a thin adapter for agents nobody can edit and runs the same routing code.

```mermaid
flowchart LR
    subgraph proc["Customer's agent process"]
        Code["Agent code<br/>any framework"]
        Lib["Dreaming library<br/>model wrapper · tool decorator<br/>agent-request wrapper · run boundary<br/>CPU predictors · routing"]
        Code --> Lib
    end

    Other["Agent nobody can edit"] --> Proxy["Thin proxy<br/>same routing code"]

    Provider["Frontier provider"]
    Tools["Tools · MCP servers · other agents"]

    subgraph backend["Backend"]
        DB[("Postgres<br/>traces · certificates · budgets")]
        Obj[("Object storage<br/>bodies · weights by hash")]
        Trainer["Trainer worker<br/>train · certify"]
    end

    Lib -->|when unsure| Provider
    Lib -->|when unsure| Tools
    Proxy -->|when unsure| Provider
    Lib -->|trace batches| DB
    Proxy -->|trace batches| DB
    DB --> Trainer
    Trainer --> Obj
    Trainer -->|certificates| DB
    DB -.->|poll certificates| Lib
    Obj -.->|weights by hash| Lib
```

Add the queue, columnar store, and policy publisher from A only when Postgres struggles.

## C. Customer-hosted container

Same parts as B, shipped as one pinned container image that runs in the customer's cloud. Prompts never leave their account. The image digest doubles as the environment hash.

```mermaid
flowchart LR
    subgraph cust["Customer's cloud account"]
        Agents["Their agents<br/>library or proxy"]
        subgraph img["Dreaming container image"]
            GW["Proxy + trace endpoint"]
            Trainer[Trainer]
            Store[("Postgres + object storage")]
        end
        Agents --> GW
        GW --> Store
        Store --> Trainer
        Trainer --> Store
    end

    GW -->|only unpredicted calls| Provider["Frontier provider"]
    Store -.->|certificate summaries · savings numbers| Control["Our control plane<br/>dashboard · image releases"]
    Control -.->|new image versions| GW
```

## D. Local daemon

A small process next to the agent on each machine. Language-agnostic like the proxy, almost no network hop like the library. The customer deploys and upgrades it everywhere the agent runs.

```mermaid
flowchart LR
    subgraph host["Each machine running the agent"]
        Agent["Agent, any language"] -->|localhost| Daemon["Dreaming daemon<br/>proxy · tool endpoint · CPU predictors"]
    end
    Daemon -->|when unsure| Provider["Frontier provider"]
    Daemon -->|trace batches| Backend[("Backend<br/>Postgres · object storage · trainer")]
    Backend -.->|certificates · weights| Daemon
```

## E. Offline advisor

Nothing in the live path. Dreaming imports traces the customer already collects, then hands back a report, a routing config, or a certified student for them to deploy. No tool interception, no live replay.

```mermaid
flowchart LR
    Agent["Agent, unchanged"] --> Provider["Frontier provider"]
    Agent --> Logs["Their existing tracing<br/>OpenTelemetry · log exports"]
    Logs -->|import| Offline["Dreaming offline<br/>train · certify · backtest"]
    Offline --> Out["Report · routing config<br/>certified student"]
    Out -.->|customer applies| Agent
```

## Connector: uAgents and ASI:One

uAgents is one more framework under B. An existing uAgent keeps its handlers and adds the wrappers. Agents run locally with a mailbox get the full library. Agents hosted on Agentverse run in a sandbox, so they use the proxy and a tool endpoint instead.

```mermaid
flowchart LR
    User["ASI:One user"] -->|chat protocol| AV[Agentverse]

    subgraph local["Run locally with mailbox: full library"]
        UA["uAgent chat handler<br/>wrapped as run boundary"] --> W["Dreaming wrappers<br/>model · tools · send_and_receive"]
    end

    subgraph hosted["Hosted on Agentverse: sandbox, no library"]
        HA["Hosted uAgent"]
    end

    AV -->|mailbox| UA
    AV --> HA
    W -->|when unsure| ASI["ASI:One model API<br/>OpenAI-compatible"]
    W -->|when unsure| Peer["Other agents"]
    HA -->|model calls| Proxy["Dreaming proxy"]
    Proxy -->|when unsure| ASI
    HA -->|tool predictions| ToolAPI["Dreaming tool endpoint"]
```

## Choosing

| Option | Decision runs | Prompts leave the customer | Customer effort | Tools and whole-run replay |
|---|---|---|---|---|
| A. Hosted gateway | Our servers | Yes | Change base URL | No |
| B. Library | Their process | Traces only, unless self-hosted | Code change | Yes |
| C. Customer-hosted | Their cloud | No | Run our image | Yes, with the library |
| D. Local daemon | Their machines | Traces only | Deploy a process | Tools yes, replay no |
| E. Offline advisor | Nowhere live | Trace export | Export logs | No |

Default: B as the core, the proxy as its thin adapter, and C as the deployment for privacy-sensitive clients.
