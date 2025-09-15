from typing import Optional
from .config import SETTINGS

# Vertex AI (Gemini) path with graceful fallback
def summarize_text(text: str, context: Optional[str] = None) -> str:
    if not SETTINGS.use_vertex_summary:
        return _textrank_fallback(text)

    try:
        # Lazy import to avoid requiring Vertex in environments without it
        from vertexai.generative_models import GenerativeModel
        import vertexai
        vertexai.init(project=SETTINGS.project_id, location=SETTINGS.region)
        model = GenerativeModel(SETTINGS.gemini_model)
        prompt = f"Summarize the following text in 2-3 sentences. Be precise and faithful.\nContext: {context or ''}\nText: {text}"
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception:
        # Fallback: TextRank
        return _textrank_fallback(text)

def _textrank_fallback(text: str) -> str:
    # Very small extractive summarizer using sumy TextRank
    from sumy.parsers.plaintext import PlainTextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer
    parser = PlainTextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    sentences = summarizer(parser.document, 3)
    return " ".join(str(s) for s in sentences).strip() or text[:240]
