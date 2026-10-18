"""T2 shared: models (both processes must define identical Model classes), seeds/ports, a local-only Agent factory (no almanac, no agentverse)."""
import logging
from uagents import Agent, Model
from uagents.crypto import Identity
from uagents.resolver import RulesBasedResolver
from uagents_core.registration import AgentRegistrationPolicy

class VendorLookup(Model): vendor: str
class VendorInfo(Model): vendor: str; terms: str
class Ask(Model): text: str
class Answer(Model): text: str

SEEDS = {"responder": "s12 responder seed", "requester": "s12 requester seed", "driver": "s12 driver seed", "chat": "s12 chat seed"}
PORTS = {"responder": 8101, "requester": 8102, "driver": 8103, "chat": 8104}
ADDR = {n: Identity.from_seed(s, 0).address for n, s in SEEDS.items()}

class NoReg(AgentRegistrationPolicy):  # skip Almanac registration
    async def register(self, *a, **kw): pass
async def _noop(self, active): pass
Agent._update_agent_status = _noop  # ponytail: private-API patch; startup/shutdown otherwise POSTs a status to the Almanac API

def make(name):
    peers = {ADDR[n]: f"http://127.0.0.1:{p}/submit" for n, p in PORTS.items() if n != name}
    return Agent(name=name, seed=SEEDS[name], port=PORTS[name], endpoint=[f"http://127.0.0.1:{PORTS[name]}/submit"], network="testnet",
                 resolve=RulesBasedResolver(peers), registration_policy=NoReg(), enable_agent_inspector=False, publish_agent_details=False,
                 report_events=False, log_level=logging.WARNING)
