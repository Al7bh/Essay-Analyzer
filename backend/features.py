"""
features.py
-----------
Turns raw essay text into a fixed set of numeric features that the
baseline ML model (Scikit-learn) can learn from.

This is intentionally simple and dependency-light — no large corpus
downloads, no GPU, nothing that needs internet access at runtime.
Every feature here is something a human grader would also notice:
length, sentence structure, vocabulary variety, and spelling.

IMPORTANT: train.py and app.py BOTH import extract_features() from
here, so the model always sees features computed the exact same way
whether it's being trained or making a live prediction.
"""

import re
from spellchecker import SpellChecker

# Loaded once at import time — pyspellchecker ships its own offline
# word-frequency dictionary, so this works with no internet access.
_spell = SpellChecker()

# A tiny hand-picked list of overused/low-value words. This is a
# placeholder heuristic for "vocabulary repetitiveness" — good enough
# for a baseline model, worth expanding later if time allows.
_WEAK_WORDS = {
    "good", "bad", "nice", "important", "very", "really", "thing",
    "things", "stuff", "big", "small", "great", "interesting"
}

# Real per-word alternatives, used for actual replacement suggestions --
# NOT a single generic list. This fixes a real bug where every essay's
# Vocabulary feedback suggested the same three words ("beneficial,
# crucial, effective") no matter which weak words were actually found.
_WEAK_WORD_SUGGESTIONS = {
    "good": ["beneficial", "effective", "valuable"],
    "bad": ["detrimental", "problematic", "harmful"],
    "nice": ["pleasant", "agreeable", "enjoyable"],
    "important": ["crucial", "significant", "essential"],
    "very": ["remarkably", "particularly", "notably"],
    "really": ["genuinely", "significantly", "considerably"],
    "thing": ["aspect", "factor", "element"],
    "things": ["aspects", "factors", "elements"],
    "stuff": ["material", "content", "items"],
    "big": ["substantial", "considerable", "significant"],
    "small": ["minor", "modest", "limited"],
    "great": ["excellent", "impressive", "outstanding"],
    "interesting": ["compelling", "noteworthy", "engaging"],
}


def _split_sentences(text: str):
    # Simple sentence splitter on . ! ? — not perfect (doesn't handle
    # abbreviations like "e.g."), but sufficient for a baseline model.
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def _split_words(text: str):
    return re.findall(r"[A-Za-z']+", text)


def extract_features(text: str) -> dict:
    """
    Returns a dict of numeric features for one essay, plus specific suggestions.
    """
    text = text or ""
    words = _split_words(text)
    sentences = _split_sentences(text)

    word_count = len(words)
    sentence_count = max(len(sentences), 1)  # avoid divide-by-zero
    avg_sentence_length = word_count / sentence_count
    avg_word_length = (
        sum(len(w) for w in words) / word_count if word_count else 0
    )

    unique_words = {w.lower() for w in words}
    vocab_richness = (len(unique_words) / word_count) if word_count else 0
    long_word_count = sum(1 for w in words if len(w) >= 7)
    long_word_ratio = (long_word_count / word_count) if word_count else 0

   # Spelling: check a sample of words and get suggestions
    sample = words[:400]
    misspelled = _spell.unknown([w.lower() for w in sample]) if sample else set()
    misspelled_count = len(misspelled)
    misspelled_ratio = (misspelled_count / len(sample)) if sample else 0
    
    # FIX: Generate specific spelling suggestions (keep all flagged words)
    spelling_suggestions = {}
    for word in misspelled:
        correction = _spell.correction(word)
        if correction and correction != word:
            spelling_suggestions[word] = correction
        else:
            spelling_suggestions[word] = None  # Flagged, but no suggestion found

    # NEW: Identify overused weak words for suggestions
    found_weak_words = [w.lower() for w in words if w.lower() in _WEAK_WORDS]
    weak_word_count = len(found_weak_words)
    weak_word_suggestions = {
        w: _WEAK_WORD_SUGGESTIONS.get(w, []) for w in set(found_weak_words)
    }

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": max(len([p for p in text.split("\n\n") if p.strip()]), 1),
        "avg_sentence_length": avg_sentence_length,
        "avg_word_length": avg_word_length,
        "vocab_richness": vocab_richness,
        "long_word_ratio": long_word_ratio,
        "misspelled_count": misspelled_count,
        "misspelled_ratio": misspelled_ratio,
        "weak_word_count": weak_word_count,
        "spelling_suggestions": spelling_suggestions, # Passed to app.py, ignored by model
        "found_weak_words": list(set(found_weak_words)), # Passed to app.py, ignored by model
        "weak_word_suggestions": weak_word_suggestions, # Passed to app.py, ignored by model
    }

# Fixed, ordered list of feature names — used to build the numeric
# vector fed into the Scikit-learn model (dict order isn't guaranteed
# stable across Python versions/processes, so we pin it explicitly).
FEATURE_ORDER = [
    "word_count",
    "sentence_count",
    "paragraph_count",
    "avg_sentence_length",
    "avg_word_length",
    "vocab_richness",
    "long_word_ratio",
    "misspelled_count",
    "misspelled_ratio",
    "weak_word_count",
]


def features_to_vector(features: dict):
    return [features[name] for name in FEATURE_ORDER]