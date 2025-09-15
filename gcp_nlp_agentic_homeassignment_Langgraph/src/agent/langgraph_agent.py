"""
LangGraph-based agent that mirrors the simple pipeline:
- START -> retrieve -> analyze -> synthesize -> END

Falls back gracefully if langgraph/langchain are unavailable.
"""
from typing import Dict, Any, List, TypedDict, Optional, Callable
import re
import pandas as pd

from ..gcp_nlp import gcp_entities, gcp_sentiment
from ..vertex_summarize import summarize_text
from ..config import SETTINGS


class AgentState(TypedDict, total=False):
    query: str
    text_col: str
    candidates: List[Dict[str, Any]]  # {text, row_index}
    analyses: List[Dict[str, Any]]    # {text, entities, sentiment, summary}
    answer: str


def _retrieve(df: pd.DataFrame, query: str, text_col: str, k: int = 5) -> List[Dict[str, Any]]:
    pat = re.compile("|".join(re.escape(tok) for tok in query.lower().split()), re.I)
    scores = df[text_col].str.lower().apply(lambda t: len(pat.findall(t)))
    top = df.assign(_score=scores).sort_values("_score", ascending=False).head(k)
    return [{"text": row[text_col], "row_index": int(idx)} for idx, row in top.iterrows()]


def build_graph(df: pd.DataFrame, text_col: str, *, faiss_retrieve: Optional[Callable[[str,int], List[Dict[str,Any]]]] = None):
    try:
        from langgraph.graph import StateGraph, START, END
    except Exception as e:
        raise ImportError("langgraph is not installed; install requirements to use this mode") from e

    graph = StateGraph(AgentState)

    def node_retrieve(state: AgentState) -> AgentState:
        cands = _retrieve(df, state["query"], text_col)
        # Optionally blend FAISS memory results
        if faiss_retrieve is not None:
            extra = faiss_retrieve(state["query"], 5)
            for item in extra:
                cands.append({"text": item.get("text", ""), "row_index": item.get("row_index", -1)})
            # Deduplicate by text
            seen = set()
            dedup = []
            for it in cands:
                t = it.get("text", "")
                if t and t not in seen:
                    seen.add(t)
                    dedup.append(it)
            cands = dedup[:5]
        return {"candidates": cands, "text_col": text_col}

    def node_analyze(state: AgentState) -> AgentState:
        analyses: List[Dict[str, Any]] = []
        for item in state.get("candidates", []):
            text = item["text"]
            try:
                ents = gcp_entities(text)
            except Exception as e:
                ents = {"error": str(e)}
            try:
                sent = gcp_sentiment(text)
            except Exception as e:
                sent = {"error": str(e)}
            try:
                summ = summarize_text(text, context=f"User query: {state['query']}")
            except Exception as e:
                summ = f"[Summary error] {e}"
            analyses.append({"text": text, "entities": ents, "sentiment": sent, "summary": summ})
        return {"analyses": analyses}

    def node_synthesize(state: AgentState) -> AgentState:
        joined = " ".join(item.get("summary", "") for item in state.get("analyses", []))
        try:
            # Prefer LangChain LLM if available; else fallback to local summarizer
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                from langchain_core.prompts import ChatPromptTemplate

                llm = ChatGoogleGenerativeAI(model=SETTINGS.gemini_model, api_key=SETTINGS.google_api_key or None)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful analyst. Provide a concise, faithful answer."),
                    ("human", "Question: {q}\nContext summaries: {ctx}\nAnswer succinctly in 3-5 sentences."),
                ])
                chain = prompt | llm
                resp = chain.invoke({"q": state["query"], "ctx": joined})
                final = getattr(resp, "content", None) or str(resp)
            except Exception:
                final = summarize_text(joined, context=f"Answer the user query: {state['query']}")
        except Exception as e:
            final = f"[Summary error] {e}"
        return {"answer": final}

    graph.add_node("retrieve", node_retrieve)
    graph.add_node("analyze", node_analyze)
    graph.add_node("synthesize", node_synthesize)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "analyze")
    graph.add_edge("analyze", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


def run_agent_langgraph(df: pd.DataFrame, query: str, text_col: str, *, faiss=None, bq_logger=None) -> Dict[str, Any]:
    try:
        app = build_graph(df, text_col, faiss_retrieve=(lambda q,k: faiss.retrieve(q,k)) if faiss else None)
    except Exception as e:
        raise ImportError("LangGraph/LangChain not available; install deps to use --agent-mode langgraph") from e
    result: AgentState = app.invoke({"query": query})
    out = {"query": query, "answer": result.get("answer", ""), "support": result.get("analyses", [])}
    # Persist: upsert into FAISS; log to BigQuery
    try:
        if faiss is not None:
            texts = [item.get("text", "") for item in out["support"]]
            metas = [{"source": "agent_support", "query": query}] * len(texts)
            faiss.upsert_texts(texts, metas)
    except Exception:
        pass
    try:
        if bq_logger is not None:
            bq_logger.log_run(query, out["answer"], out["support"])  
    except Exception:
        pass
    return out
