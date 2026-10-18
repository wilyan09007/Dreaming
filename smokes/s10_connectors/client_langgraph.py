"""LangGraph 2-node agent through the Dreaming proxy. The change: base_url + a call-site header per node."""
import json
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

BASE, MODEL = "http://127.0.0.1:8765/v1", "openai/gpt-5.4-nano"
# way 1: one ChatOpenAI instance per call site (header fixed at construction)
cat_llm = ChatOpenAI(model=MODEL, base_url=BASE, api_key="x", temperature=0, max_tokens=60, default_headers={"X-Dreaming-Call-Site": "categorize"})
# way 2: one shared instance, header set per call with .bind (extra_headers is passed through to the openai client)
base_llm = ChatOpenAI(model=MODEL, base_url=BASE, api_key="x", temperature=0, max_tokens=60)
sum_llm = base_llm.bind(extra_headers={"X-Dreaming-Call-Site": "summarize"})

class S(TypedDict):
    txn: str; category: str; summary: str

def categorize(s: S):
    r = cat_llm.invoke([("system", "Reply with exactly one word: Software, Meals, Travel or Office."), ("human", f"Transaction: {s['txn']}")])
    return {"category": r.content.strip()}

def summarize(s: S):
    r = sum_llm.invoke([("system", "Write one short sentence (max 12 words) for a ledger entry."), ("human", f"{s['txn']} categorized as {s['category']}")])
    return {"summary": r.content.strip()}

g = StateGraph(S); g.add_node("categorize", categorize); g.add_node("summarize", summarize)
g.add_edge(START, "categorize"); g.add_edge("categorize", "summarize"); g.add_edge("summarize", END)
print(json.dumps(g.compile().invoke({"txn": "ADOBE *CREATIVE CLD $54.99"})))
