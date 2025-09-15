from typing import List, Tuple, Dict, Any
from google.cloud import language_v2 as language

def gcp_entities(text: str) -> List[Tuple[str, str, float]]:
    client = language.LanguageServiceClient()
    doc = {"content": text, "type_": language.Document.Type.PLAIN_TEXT}
    resp = client.analyze_entities(document=doc)
    return [(e.name, language.Entity.Type(e.type_).name, round(e.salience, 3)) for e in resp.entities]

def gcp_sentiment(text: str) -> Dict[str, Any]:
    client = language.LanguageServiceClient()
    doc = {"content": text, "type_": language.Document.Type.PLAIN_TEXT}
    resp = client.analyze_sentiment(document=doc)
    overall = {"score": round(resp.document_sentiment.score, 3), "magnitude": round(resp.document_sentiment.magnitude, 3)}
    return overall
