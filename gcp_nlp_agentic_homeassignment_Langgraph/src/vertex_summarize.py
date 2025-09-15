from typing import Optional, List
import re
from .config import SETTINGS

def summarize_text(text: str, context: Optional[str] = None, max_words: int = 10) -> str:
    # 1) Prefer direct Gemini API if API key present
    if getattr(SETTINGS, "google_api_key", ""):
        try:
            import google.generativeai as genai
            genai.configure(api_key=SETTINGS.google_api_key)
            model = genai.GenerativeModel(SETTINGS.gemini_model)
            prompt = _format_prompt(text, context, max_words)
            resp = model.generate_content(prompt)
            out = (resp.text or "").strip() or _simple_fallback(text)
            return _truncate_words(out, max_words)
        except Exception:
            pass  # Fall through to Vertex or simple fallback

    # 2) Try Vertex AI if enabled
    if SETTINGS.use_vertex_summary:
        try:
            from vertexai.generative_models import GenerativeModel
            import vertexai
            vertexai.init(project=SETTINGS.project_id, location=SETTINGS.region)
            model = GenerativeModel(SETTINGS.gemini_model)
            prompt = _format_prompt(text, context, max_words)
            resp = model.generate_content(prompt)
            out = resp.text.strip()
            return _truncate_words(out, max_words)
        except Exception:
            return _truncate_words(_simple_fallback(text), max_words)

    # 3) If Vertex disabled entirely, use fallback
    return _truncate_words(_simple_fallback(text), max_words)

def _format_prompt(text: str, context: Optional[str], max_words: int) -> str:
    return (
        "Summarize the text faithfully in at most "
        f"{max_words} words. Use one sentence. Do not exceed the word limit."
        " Do not use ellipses.\n"
        f"Context: {context or ''}\nText: {text}"
    )

def _simple_fallback(text: str) -> str:
    # Minimal, dependency-free extractive summarizer.
    # 1) Split into sentences
    sentences = _split_sentences(text)
    if not sentences:
        return text[:240]
    # 2) Score sentences by word frequency (ignore common stopwords)
    scores = _score_sentences(sentences)
    # 3) Take top 3 by score, keep original order
    k = min(3, len(sentences))
    top_idx = sorted(sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:k])
    summary = " ".join(sentences[i] for i in top_idx).strip()
    return summary or (text[:240])

def _split_sentences(text: str) -> List[str]:
    raw = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    # Fallback if no punctuation
    if len(raw) <= 1:
        # chunk by ~200 chars
        chunk_size = 200
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size) if text[i:i+chunk_size].strip()]
    return raw

def _score_sentences(sentences: List[str]) -> List[float]:
    stop = {
        "the","a","an","and","or","but","if","to","of","in","on","for","with","as","by","at","is","it","this","that","was","were","are","be","have","has","had","i","you","he","she","they","we","my","our","your","their"
    }
    # Build word frequencies
    freqs = {}
    for s in sentences:
        for w in re.findall(r"\w+", s.lower()):
            if w in stop or w.isdigit():
                continue
            freqs[w] = freqs.get(w, 0) + 1
    if not freqs:
        return [0.0] * len(sentences)
    # Normalize frequencies
    max_f = max(freqs.values()) or 1
    for k in list(freqs.keys()):
        freqs[k] = freqs[k] / max_f
    scores: List[float] = []
    for s in sentences:
        score = 0.0
        for w in re.findall(r"\w+", s.lower()):
            score += freqs.get(w, 0.0)
        # favor shorter, information-dense sentences slightly
        length = max(len(s.split()), 1)
        scores.append(score / (length ** 0.25))
    return scores

def _truncate_words(text: str, max_words: int) -> str:
    """Return text capped to `max_words` tokens, with no ellipsis."""
    words = text.strip().split()
    if max_words <= 0:
        return ""
    if len(words) <= max_words:
        return " ".join(words)
    clipped = words[:max_words]
    # Remove trailing ellipsis-like tokens if any model added them
    if clipped[-1] in {"...", "..", "…"}:
        clipped = clipped[:-1]
    # Ensure no ellipsis characters remain
    clipped[-1] = clipped[-1].rstrip("…")
    return " ".join(clipped)
