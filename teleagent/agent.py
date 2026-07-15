"""
TeleAgent AI - Agentic RAG core.

Built with LangGraph. The graph:

    START -> classify_intent -> route -> retrieve_faq (vector RAG) -> confidence_gate -> {escalate | skip} -> generate_answer -> END
                                       -> recommend_plan (tool)                                             -> generate_answer -> END

- classify_intent: Gemini labels the query (billing/plan/wifi/entertainment/wallet/device/general)
- retrieve_faq: vector similarity search over data/telecom_faq.json using Gemini embeddings
  (falls back to TF-IDF automatically if the embeddings API call fails, e.g. offline)
- recommend_plan: rule-based tool that scores data/plans.json against the user's stated usage
- escalate_to_human: agentic tool that raises a support ticket when retrieval confidence is low
- generate_answer: Gemini synthesizes a final, grounded answer from whatever context was gathered

`agent_trace` on the returned state is a step-by-step log of what the agent did, so the UI
can show its reasoning path instead of just the final answer.
"""

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import TypedDict, Optional, List, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

DATA_DIR = Path(__file__).parent / "data"

with open(DATA_DIR / "telecom_faq.json") as f:
    FAQ = json.load(f)

with open(DATA_DIR / "plans.json") as f:
    PLANS = json.load(f)

_faq_texts = [f"{item['question']} {item['answer']}" for item in FAQ]

# TF-IDF fallback index, built once at import time (no API calls needed)
_vectorizer = TfidfVectorizer(stop_words="english")
_faq_tfidf_matrix = _vectorizer.fit_transform(_faq_texts)

# Vector (Gemini embeddings) index, built lazily on first successful call and cached
_faq_embedding_matrix = None
_embedding_client_cache: Dict[str, GoogleGenerativeAIEmbeddings] = {}

CONFIDENCE_THRESHOLD = 0.45  # below this, the agent escalates to a human ticket


def _get_embedding_client(api_key: str) -> GoogleGenerativeAIEmbeddings:
    if api_key not in _embedding_client_cache:
        _embedding_client_cache[api_key] = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", google_api_key=api_key
        )
    return _embedding_client_cache[api_key]


def _ensure_faq_embeddings(api_key: str) -> Optional[np.ndarray]:
    """Builds (and caches) Gemini embeddings for the FAQ corpus. Returns None if the API call fails."""
    global _faq_embedding_matrix
    if _faq_embedding_matrix is not None:
        return _faq_embedding_matrix
    try:
        client = _get_embedding_client(api_key)
        vectors = client.embed_documents(_faq_texts)
        _faq_embedding_matrix = np.array(vectors)
        return _faq_embedding_matrix
    except Exception:
        return None


def retrieve_top_faqs(query: str, api_key: str, k: int = 3) -> Dict:
    """
    RAG retrieval step. Tries Gemini vector embeddings first (semantic match);
    falls back to TF-IDF keyword similarity if the embeddings call fails.
    Returns {"results": [...], "method": "vector"|"tfidf", "top_score": float}
    """
    faq_matrix = _ensure_faq_embeddings(api_key)
    scores = None
    method = "tfidf"
    if faq_matrix is not None:
        try:
            client = _get_embedding_client(api_key)
            q_vec = np.array(client.embed_query(query)).reshape(1, -1)
            scores = sk_cosine_similarity(q_vec, faq_matrix).flatten()
            method = "vector"
        except Exception:
            scores = None

    if scores is None:
        q_vec = _vectorizer.transform([query])
        scores = sk_cosine_similarity(q_vec, _faq_tfidf_matrix).flatten()
        method = "tfidf"

    top_idx = scores.argsort()[::-1][:k]
    results = [{**FAQ[i], "score": float(scores[i])} for i in top_idx if scores[i] > 0.05]
    top_score = float(scores[top_idx[0]]) if len(top_idx) else 0.0
    return {"results": results, "method": method, "top_score": top_score}


def recommend_plan(daily_data_gb: float, num_lines: int = 1) -> Dict:
    """Rule-based recommender tool: picks the cheapest plan that covers stated usage."""
    candidates = [p for p in PLANS if p["data_gb_per_day"] >= daily_data_gb]
    if num_lines > 1:
        candidates = [p for p in candidates if "Family" in p["name"] or "Bundle" in p["name"]]
        if not candidates:
            candidates = [p for p in PLANS if "Family" in p["name"]]
    if not candidates:
        candidates = sorted(PLANS, key=lambda p: -p["data_gb_per_day"])[:1]
    return min(candidates, key=lambda p: p["price_inr"])


def raise_ticket(query: str, intent: str) -> Dict:
    """Agentic escalation tool: simulates raising a support ticket for a human agent."""
    return {
        "ticket_id": f"TT-{uuid.uuid4().hex[:8].upper()}",
        "category": intent,
        "query": query,
        "status": "Open — routed to human agent",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


class AgentState(TypedDict, total=False):
    query: str
    api_key: str
    intent: str
    faq_context: List[Dict]
    retrieval_method: str
    top_score: float
    plan_context: Optional[Dict]
    escalate: bool
    ticket: Optional[Dict]
    answer: str
    agent_trace: List[str]


def build_graph(api_key: str):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        max_output_tokens=600,
        temperature=0.3,
    )

    def classify_intent(state: AgentState) -> AgentState:
        prompt = (
            "Classify this telecom customer query into exactly one label: "
            "billing, plan, wifi, entertainment, wallet, device, or general. "
            "Reply with only the single label word.\n\n"
            f"Query: {state['query']}"
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        label = str(resp.content).strip().lower()
        valid = {"billing", "plan", "wifi", "entertainment", "wallet", "device", "general"}
        state["intent"] = label if label in valid else "general"
        state.setdefault("agent_trace", []).append(f"Classified intent as **{state['intent']}**")
        return state

    def route(state: AgentState) -> str:
        wants_plan_advice = state["intent"] == "plan" and (
            "gb" in state["query"].lower() or "data" in state["query"].lower() or "recommend" in state["query"].lower()
        )
        return "recommend_plan" if wants_plan_advice else "retrieve_faq"

    def retrieve_faq_node(state: AgentState) -> AgentState:
        rag = retrieve_top_faqs(state["query"], state["api_key"])
        state["faq_context"] = rag["results"]
        state["retrieval_method"] = rag["method"]
        state["top_score"] = rag["top_score"]
        state["escalate"] = rag["top_score"] < CONFIDENCE_THRESHOLD
        method_label = "semantic vector search" if rag["method"] == "vector" else "keyword (TF-IDF) search"
        state.setdefault("agent_trace", []).append(
            f"Retrieved {len(rag['results'])} FAQ match(es) via {method_label} (top confidence {rag['top_score']:.2f})"
        )
        return state

    def confidence_gate(state: AgentState) -> str:
        return "escalate_to_human" if state.get("escalate") else "generate_answer"

    def escalate_to_human(state: AgentState) -> AgentState:
        ticket = raise_ticket(state["query"], state["intent"])
        state["ticket"] = ticket
        state.setdefault("agent_trace", []).append(
            f"Low retrieval confidence — raised support ticket **{ticket['ticket_id']}** for a human agent"
        )
        return state

    def recommend_plan_node(state: AgentState) -> AgentState:
        nums = re.findall(r"(\d+(?:\.\d+)?)\s*gb", state["query"].lower())
        daily_gb = float(nums[0]) if nums else 1.5
        lines_match = re.findall(r"(\d+)\s*(?:lines|members|people)", state["query"].lower())
        num_lines = int(lines_match[0]) if lines_match else 1
        plan = recommend_plan(daily_gb, num_lines)
        state["plan_context"] = plan
        rag = retrieve_top_faqs(state["query"], state["api_key"], k=1)
        state["faq_context"] = rag["results"]
        state["retrieval_method"] = rag["method"]
        state.setdefault("agent_trace", []).append(
            f"Ran plan-recommender tool for {daily_gb}GB/day, {num_lines} line(s) → **{plan['name']}**"
        )
        return state

    def generate_answer(state: AgentState) -> AgentState:
        context_parts = []
        for item in state.get("faq_context", []):
            context_parts.append(f"Q: {item['question']}\nA: {item['answer']}")
        if state.get("plan_context"):
            p = state["plan_context"]
            context_parts.append(
                f"Recommended plan: {p['name']} at ₹{p['price_inr']}/month, "
                f"{p['data_gb_per_day']}GB/day, {p['calls']} calls, OTT: {p['ott']}. Best for: {p['best_for']}."
            )
        if state.get("ticket"):
            t = state["ticket"]
            context_parts.append(
                f"No confident answer was found in the knowledge base. A support ticket {t['ticket_id']} "
                f"has been raised and routed to a human agent."
            )
        context = "\n\n".join(context_parts) if context_parts else "No matching knowledge base entry found."

        system = (
            "You are TeleAgent, a helpful telecom customer support assistant. "
            "Answer using ONLY the provided context. Be concise (3-5 sentences), warm, and specific. "
            "If a support ticket was raised, mention the ticket ID and reassure the customer a human will follow up."
        )
        user_msg = f"Context:\n{context}\n\nCustomer query: {state['query']}"
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user_msg)])
        state["answer"] = str(resp.content).strip()
        state.setdefault("agent_trace", []).append("Generated final grounded answer")
        return state

    graph = StateGraph(AgentState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_faq", retrieve_faq_node)
    graph.add_node("recommend_plan", recommend_plan_node)
    graph.add_node("escalate_to_human", escalate_to_human)
    graph.add_node("generate_answer", generate_answer)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent", route, {"retrieve_faq": "retrieve_faq", "recommend_plan": "recommend_plan"}
    )
    graph.add_conditional_edges(
        "retrieve_faq", confidence_gate, {"escalate_to_human": "escalate_to_human", "generate_answer": "generate_answer"}
    )
    graph.add_edge("escalate_to_human", "generate_answer")
    graph.add_edge("recommend_plan", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()


def run_agent(query: str, api_key: str) -> AgentState:
    app = build_graph(api_key)
    result = app.invoke({"query": query, "api_key": api_key, "agent_trace": []})
    return result
