"""Replay tau2-bench gold actions (no LLM) against the deterministic tools; write one row per call: (domain, task, idx, tool, args, return, db hash before/after).
Run with the separate venv:  smokes/.data/s14_tau2_venv/Scripts/python.exe -X utf8 tau2_replay.py   (writes tau2_calls.jsonl).
Gold actions are one scripted trajectory per task, not an agent's own behaviour: this measures the tools, not an agent."""
import json, sys, hashlib
from pathlib import Path
SRC = Path(r"C:\Users\<user>\Desktop\personalprojects\Dreaming\smokes\.data\tau2_bench_src")
OUT = Path(__file__).parent / "tau2_calls.jsonl"
from tau2.domains.banking_knowledge.environment import get_environment as banking_env
from tau2.data_model.tasks import Task


def run(domain, get_env, kwargs):
    tasks = json.load(open(SRC / "data/tau2/domains" / domain / "tasks.json", encoding="utf-8"))
    n = 0
    with OUT.open("a", encoding="utf-8") as fh:
        for t in tasks:
            task = Task.model_validate(t)
            try:
                env = get_env(task=task, **kwargs)
                st = task.initial_state
                env.set_state(initialization_data=st.initialization_data if st else None, initialization_actions=st.initialization_actions if st else None,
                              message_history=list(st.message_history) if st and st.message_history else [])
            except Exception as e:
                print("skip task", t["id"], repr(e)[:200]); continue
            for i, a in enumerate((task.evaluation_criteria.actions if task.evaluation_criteria else None) or []):
                before = env.get_db_hash()
                try:
                    r = env.make_tool_call(tool_name=a.name, requestor=a.requestor, **a.arguments); ret = env.to_json_str(r); err = False
                except Exception as e:
                    ret = f"Error: {e}"; err = True
                after = env.get_db_hash()
                fh.write(json.dumps(dict(domain=domain, task=t["id"], idx=i, requestor=a.requestor, name=a.name, args=a.arguments, ret=ret, err=err, before=before, after=after)) + "\n"); n += 1
    print(domain, "calls", n)


if __name__ == "__main__":
    OUT.write_text("")
    run("banking_knowledge", banking_env, dict(retrieval_variant="no_knowledge"))
