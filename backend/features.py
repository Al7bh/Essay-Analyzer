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

# ---------------------------------------------------------------------
# Reducing false positives: pyspellchecker's bundled dictionary is a
# general-purpose English word-frequency list. It has NO knowledge of
# proper nouns, brand names, or contractions, so words like "Instagram"
# or "don't" get flagged as misspelled even though they're perfectly
# correct. Two things fix most of this:
#
# 1. A whitelist of common words the dictionary is known to miss --
#    loaded directly into the spellchecker so it treats them as known.
# 2. Heuristics applied at check-time: skip words that are ALL-CAPS
#    (likely acronyms: NASA, GPA) and skip capitalized words that
#    aren't the first word of their sentence (likely proper nouns:
#    a name, a brand, a place).
#
# Neither of these makes spelling detection perfect -- a genuinely
# misspelled capitalized word (e.g. "Instagraam") would now slip
# through uncaught. That's a real, known trade-off: fewer false
# "this is wrong" flags on correct words, at the cost of occasionally
# missing a real mistake. For an essay-feedback tool, false positives
# are worse (they erode trust in every OTHER correct flag), so this
# trade-off is the right one.
# ---------------------------------------------------------------------
_WHITELIST = {
    # Social media / tech brands (essays about this topic are common)
    "instagram", "tiktok", "facebook", "snapchat", "whatsapp", "youtube",
    "twitter", "google", "netflix", "spotify", "amazon", "apple",
    "microsoft", "iphone", "ipad", "android", "reddit", "linkedin",
    "discord", "chatgpt", "openai", "wifi", "smartphone", "smartphones",
    "internet", "online", "offline", "cyberbullying", "multitasking",
    "wellbeing", "gamification", "livestream", "podcast", "hashtag",
    "selfie", "selfies", "influencer", "influencers", "app", "apps",
    # Common contractions -- the regex `[A-Za-z']+` keeps the apostrophe,
    # but pyspellchecker's dictionary often doesn't include these forms.
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't", "won't", "wouldn't", "can't", "couldn't",
    "shouldn't", "mustn't", "it's", "that's", "there's", "here's",
    "what's", "who's", "they're", "we're", "you're", "i'm", "he's",
    "she's", "let's", "i've", "we've", "they've", "you've", "i'll",
    "we'll", "they'll", "you'll", "i'd", "we'd", "they'd", "you'd",
}
_spell.word_frequency.load_words(_WHITELIST)

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


def _words_with_sentence_position(text: str):
    """
    Like _split_words, but also flags whether each word is the first
    word of its sentence -- needed for the "skip capitalized words that
    aren't sentence-initial" false-positive heuristic below.
    Returns a list of (word, is_sentence_start) tuples.
    """
    sentences = _split_sentences(text)
    result = []
    for sentence in sentences:
        sentence_words = re.findall(r"[A-Za-z']+", sentence)
        for i, w in enumerate(sentence_words):
            result.append((w, i == 0))
    return result


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

   # Spelling: check a sample of words and get suggestions.
    # Words skipped here (ALL-CAPS acronyms, capitalized mid-sentence
    # words that look like proper nouns) are never even passed to the
    # spellchecker -- see the false-positive-reduction note above.
    positioned = _words_with_sentence_position(text)[:400]
    checkable = []
    for word, is_sentence_start in positioned:
        if len(word) >= 2 and word.isupper():
            continue  # likely an acronym (NASA, GPA, USA)
        if word[0].isupper() and not is_sentence_start and word.lower() not in _WEAK_WORDS:
            continue  # likely a proper noun (a name, brand, or place)
        checkable.append(word.lower())

    misspelled = _spell.unknown(checkable) if checkable else set()
    misspelled_count = len(misspelled)
    misspelled_ratio = (misspelled_count / len(checkable)) if checkable else 0
    
    # Multiple ranked candidates instead of pyspellchecker's single
    # ".correction()" pick -- that method just returns whichever
    # candidate has the highest raw word frequency, which isn't always
    # the contextually correct fix (e.g. "adress" -> "dress" outranks
    # "address" by frequency alone, despite "address" being the obvious
    # intended word). Offering the top few candidates, like the
    # vocabulary suggestions already do, lets the user pick the right
    # one instead of being stuck with the model's single best guess.
    spelling_suggestions = {}
    for word in misspelled:
        candidates = _spell.candidates(word) or set()
        candidates.discard(word)
        ranked = sorted(candidates, key=lambda w: -_spell.word_frequency[w])
        spelling_suggestions[word] = ranked[:3]

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