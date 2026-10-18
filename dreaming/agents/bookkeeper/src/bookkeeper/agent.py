"""The uAgent: the ASI:One front door. Runs on Fly (or a laptop) with an Agentverse mailbox, outbound polling only.

    python -m bookkeeper.agent        needs AGENT_SEED (the address derives from it), DREAMING_API (the API's base URL) and
                                      AGENTVERSEKEY: at startup the agent registers its own mailbox by POSTing the key to its
                                      local /connect endpoint, which is all the Agentverse inspector's Connect button does (S11)

One runtime owner: the API holds the policies and certificates and runs every workflow call. This process only relays:
a chat message becomes `POST /run {"input": …}` on the API (a live run through the same library, same store), and the
reply quotes the API's answer: what was posted and which steps a dream answered, under which certificate. The agent needs
no database and no keys beyond its seed, and it can never hold a stale policy. Keep a turn under ~25 s and send one
reply (S11: a slower reply is dropped, and the planner takes any first message as the whole answer). The chat protocol itself is dreaming.connectors.uagents;
this module only knows the bookkeeper's words: a bank-feed line in, a plain-language receipt out.
"""

from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path
from uuid import uuid4

from dreaming.contracts import Decision, Json, Kind, RunView, TraceRow
from dreaming.hashing import short

from . import workflow

DESCRIPTION = ("Books a bank-feed line (descriptor + amount): categorize, vendor lookup, invoice match, ledger post; "
               "the receipt names which steps a certified trained model answered.")
USAGE = 'Send one bank-feed line, the descriptor then the amount, like "ADOBE *WEB 54.99".'
LINE = re.compile(r"(?P<descriptor>.+?)\s+\$?(?P<amount>-?\d[\d,]*(?:\.\d+)?)")
WHO = {Decision.predicted: "a dream answered", Decision.called: "the real call answered", Decision.audited: "the real call answered (audit)",
       Decision.fallback: "the fallback answered (the real call failed)"}


def _name(row: TraceRow) -> str:
    return row.call_site.split(".")[-1]


def to_input(text: str) -> Json:
    """The chat text as the transaction `run_transaction` expects. Only descriptor and amount are known: the vendor lookup finds the merchant."""
    line = text.rsplit(":", 1)[-1].strip().strip("\"'`")  # the planner rephrases: "Process this bank-feed line: NOTION *WEB 96.00"
    m = LINE.fullmatch(" ".join(line.split()))  # one space between words: the vendor lookup splits on " *"
    if m is None:
        raise ValueError(USAGE)
    return {"id": f"chat-{uuid4().hex[:8]}", "date": date.today().isoformat(), "merchant": "", "descriptor": m["descriptor"],
            "amount": float(m["amount"].replace(",", ""))}


def to_reply(view: RunView) -> str:
    """The run as a receipt: what was posted, who answered each step (a dream names its weights and certificate), cost, latency.
    Latency sums the calls' latencies: a live Run.latency_ms is wall-clock and leaves out the mock vendor API's charged time."""
    steps = [r for r in view.rows if r.kind != Kind.effect]
    done = [r for r in view.rows if r.kind == Kind.effect]
    outcome = f"{done[-1].response} ({_name(done[-1])})" if done else f"the run failed at {view.run.stuck_at or 'a step'}, nothing was posted"
    lines = [f"Result: {outcome}."]
    for r in steps:
        p = r.prediction
        proof = f", weights {short(p.hashes.weights)}, certificate {p.cert_id}" if r.decision == Decision.predicted and p else ""
        lines.append(f"- {_name(r)}: {WHO[r.decision]}: {r.response}{proof}")
    dreamed = sum(r.decision == Decision.predicted for r in steps)
    lines.append(f"{dreamed} of {len(steps)} steps were answered by a dream. "
                 f"Cost ${view.run.cost_usd:.4f}, latency {sum(r.latency_ms for r in view.rows) / 1000:.1f} s.")
    return "\n".join(lines)


def main() -> None:
    from dreaming.connectors.uagents import chat_handler
    from uagents import Agent, Context

    api = os.environ.get("DREAMING_API", "http://127.0.0.1:8080")
    # network="testnet" is what S11 proved; uagents defaults to mainnet. The address derives from the seed alone.
    # readme_path/description ride on the /connect registration: without them each restart publishes an empty profile, and
    # ASI:One's planner did not find the agent with an empty README ("no results for this specific identifier"), and did once one was set (2026-10-18)
    agent = Agent(name="dreaming-bookkeeper", seed=os.environ["AGENT_SEED"], mailbox=True, port=8000, network="testnet",
                  readme_path=str(Path(__file__).with_name("README.md")), description=DESCRIPTION)
    agent.include(chat_handler(api, workflow.W, to_input, to_reply), publish_manifest=True)

    @agent.on_event("startup")
    async def up(ctx: Context) -> None:
        ctx.logger.info(f"address={agent.address} relaying chats to {api}/run")
        key = os.environ.get("AGENTVERSEKEY")
        if not key:
            ctx.logger.warning("AGENTVERSEKEY unset: no mailbox, the agent is unreachable from ASI:One")
            return
        import httpx  # the inspector's Connect button is this one POST to the agent's own server (S11)

        try:
            async with httpx.AsyncClient(timeout=30) as http:
                r = await http.post("http://127.0.0.1:8000/connect", json={"user_token": key, "agent_type": "mailbox"})
            ctx.logger.info(f"mailbox connect: HTTP {r.status_code} {r.text[:120]}")
        except Exception as e:  # noqa: BLE001 - the agent keeps running; the log says why ASI:One cannot reach it
            ctx.logger.error(f"mailbox connect failed: {type(e).__name__}: {e}")

    agent.run()


if __name__ == "__main__":
    main()
