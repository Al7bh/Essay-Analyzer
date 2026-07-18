"""
relevance.py
------------
Checks whether an essay actually addresses its given prompt/topic — a
real, well-documented gap in automated essay scoring: a model that only
looks at grammar/structure/vocabulary can give a fluent, well-organized
essay a high score even if it completely ignores the question asked.

This is OPTIONAL: the prompt is only checked if the user actually
provides one (essays don't always have a known prompt). If no prompt is
given, this check is simply skipped — see app.py.

Approach: TF-IDF cosine similarity is the standard technique for this in
the literature, but it's hard for a student to eyeball ("is 0.18 good or
bad?"). So we pair it with keyword coverage — which of the prompt's
meaningful (non-stopword) terms actually show up in the essay — since
that's directly explainable in a viva ("6 of 8 key terms from the prompt
appear in the essay") in a way a raw cosine similarity number isn't.
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# A short, hand-picked stopword list -- avoids pulling in nltk's corpus
# download (same reasoning as features.py: no internet access needed at
# runtime). Not exhaustive, but sufficient for picking out meaningful
# keywords from a typical essay prompt.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "of", "to",
    "in", "on", "at", "for", "with", "about", "as", "is", "are", "was",
    "were", "be", "been", "being", "this", "that", "these", "those",
    "you", "your", "yours", "it", "its", "do", "does", "did", "how",
    "what", "why", "which", "who", "whom", "should", "would", "could",
    "can", "will", "shall", "may", "might", "must", "not", "no", "yes",
    "write", "essay", "discuss", "explain", "describe", "consider",
}


def _keywords(text: str, min_len: int = 4):
    words = re.findall(r"[A-Za-z']+", text.lower())
    return [w for w in words if len(w) >= min_len and w not in _STOPWORDS]


def compute_relevance(essay: str, prompt: str) -> dict:
    """
    Returns:
        similarity: float 0-1, TF-IDF cosine similarity between prompt and essay
        band: 'good' | 'warn' -- thresholded on similarity (calibrated below)
        matched_keywords / missing_keywords: for a human-readable explanation
    """
    if not prompt or not prompt.strip():
        return None  # no prompt given -- caller should skip this check entirely

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf = vectorizer.fit_transform([prompt, essay])
        similarity = float(cosine_similarity(tfidf[0], tfidf[1])[0][0])
    except ValueError:
        # Happens if prompt/essay are entirely stopwords after TF-IDF's own
        # filtering (e.g. a one-word prompt) -- fail safe rather than crash.
        similarity = 0.0

    prompt_keywords = set(_keywords(prompt))
    essay_words = set(_keywords(essay))
    matched = sorted(prompt_keywords & essay_words)
    missing = sorted(prompt_keywords - essay_words)

    # Calibrated empirically (see notes in evaluate_relevance_thresholds.py
    # if you want to re-check this against more examples) -- TF-IDF cosine
    # similarity between a short prompt and a long essay is naturally low
    # even when genuinely on-topic, so thresholds are much lower than raw
    # intuition suggests.
    band = "good" if similarity >= 0.12 else "warn"

    return {
        "similarity": round(similarity, 3),
        "band": band,
        "matched_keywords": matched,
        "missing_keywords": missing,
    }
