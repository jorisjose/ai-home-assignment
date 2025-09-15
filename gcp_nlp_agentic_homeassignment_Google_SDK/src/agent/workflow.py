"""A slim agent that chains:
1) retrieval over the small in-memory dataset (keyword match)
2) entity & sentiment extraction
3) summarization
"""
import re
from typing import Dict, Any, List
import pandas as pd
from ..gcp_nlp import gcp_entities, gcp_sentiment
from ..vertex_summarize import summarize_text

def _retrieve(df: pd.DataFrame, query: str, text_col: str, k: int = 5) -> pd.DataFrame:
    pat = re.compile("|".join(re.escape(tok) for tok in query.lower().split()), re.I)
    scores = df[text_col].str.lower().apply(lambda t: len(pat.findall(t)))
    return df.assign(_score=scores).sort_values("_score", ascending=False).head(k)

def run_agent(df: pd.DataFrame, query: str, text_col: str) -> Dict[str, Any]:
    top = _retrieve(df, query, text_col)
    analyses: List[Dict[str, Any]] = []
    for _, row in top.iterrows():
        text = row[text_col]
        entities = gcp_entities(text)
        sentiment = gcp_sentiment(text)
        summary = summarize_text(text, context=f"User query: {query}")
        analyses.append({"text": text, "entities": entities, "sentiment": sentiment, "summary": summary})

    # Final answer: summarize the summaries + mention recurring entities
    joined = " ".join(item["summary"] for item in analyses)
    final = summarize_text(joined, context=f"Answer the user query: {query}")
    return {"query": query, "answer": final, "support": analyses}
