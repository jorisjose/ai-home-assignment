from typing import List, Tuple, Dict, Any

def _get_language_module():
    """Import google.cloud.language_v2 lazily to avoid hard dependency at import time."""
    try:
        from google.cloud import language_v2 as language  # type: ignore
        return language
    except Exception as e:
        # Propagate a clear error for callers to handle
        raise ImportError("google-cloud-language is not available or misconfigured") from e

def gcp_entities(text: str) -> List[Tuple[str, str, float]]:
    language = _get_language_module()
    client = language.LanguageServiceClient()
    doc = {"content": text, "type_": language.Document.Type.PLAIN_TEXT}
    resp = client.analyze_entities(document=doc)
    out: List[Tuple[str, str, float]] = []
    for e in resp.entities:
        try:
            sal = getattr(e, "salience", 0.0)
            sal_f = round(float(sal), 3) if isinstance(sal, (int, float)) else 0.0
        except Exception:
            sal_f = 0.0
        try:
            etype = language.Entity.Type(getattr(e, "type_", 0)).name
        except Exception:
            etype = "UNKNOWN"
        out.append((getattr(e, "name", ""), etype, sal_f))
    return out

def gcp_sentiment(text: str) -> Dict[str, Any]:
    language = _get_language_module()
    client = language.LanguageServiceClient()
    doc = {"content": text, "type_": language.Document.Type.PLAIN_TEXT}
    resp = client.analyze_sentiment(document=doc)
    overall = {"score": round(resp.document_sentiment.score, 3), "magnitude": round(resp.document_sentiment.magnitude, 3)}
    return overall
