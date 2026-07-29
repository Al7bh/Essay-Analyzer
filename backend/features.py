"""
features.py
-----------
Turns raw essay text into a fixed set of numeric features that the
baseline ML model (Scikit-learn) can learn from.

VERSION 2: 17 features, up from 10. Two dead-weight features from v1
(paragraph_count had coefficient +0.000, avg_sentence_length was -0.025)
were replaced with 9 new linguistically motivated features:
  sentence_length_variance  -- detects monotonous same-length sentences
  avg_paragraph_length      -- how developed each paragraph is
  type_token_ratio_100      -- vocab richness on first 100 words (length-normalized)
  stopword_ratio            -- function-word density (low = more content-rich)
  commas_per_sentence       -- correlates with complex multi-clause sentences
  semicolons_per_sentence   -- strong marker of sophisticated writing
  question_count            -- flags essays with no nuance
  exclamation_count         -- flags informal/over-emphatic tone

IMPORTANT: FEATURE_ORDER is the contract between this file and the
trained model.pkl. Any change to FEATURE_ORDER requires retraining.
"""

import re
import math
from spellchecker import SpellChecker

_spell = SpellChecker()

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "of", "to",
    "in", "on", "at", "for", "with", "about", "as", "is", "are", "was",
    "were", "be", "been", "being", "this", "that", "these", "those",
    "you", "your", "it", "its", "do", "does", "did", "not", "no",
    "i", "we", "they", "he", "she", "my", "our", "their", "his", "her",
    "have", "has", "had", "will", "would", "could", "should", "can", "may",
    "from", "by", "up", "out", "there", "here", "all", "also", "just",
}

_WHITELIST = {
    "instagram", "tiktok", "facebook", "snapchat", "whatsapp", "youtube",
    "twitter", "google", "netflix", "spotify", "amazon", "apple",
    "microsoft", "iphone", "ipad", "android", "reddit", "linkedin",
    "discord", "chatgpt", "openai", "wifi", "smartphone", "smartphones",
    "internet", "online", "offline", "cyberbullying", "multitasking",
    "wellbeing", "gamification", "livestream", "podcast", "hashtag",
    "selfie", "selfies", "influencer", "influencers", "app", "apps",
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't", "won't", "wouldn't", "can't", "couldn't",
    "shouldn't", "mustn't", "it's", "that's", "there's", "here's",
    "what's", "who's", "they're", "we're", "you're", "i'm", "he's",
    "she's", "let's", "i've", "we've", "they've", "you've", "i'll",
    "we'll", "they'll", "you'll", "i'd", "we'd", "they'd", "you'd",
}
_spell.word_frequency.load_words(_WHITELIST)

_WEAK_WORDS = {
    "good", "bad", "nice", "important", "very", "really", "thing",
    "things", "stuff", "big", "small", "great", "interesting"
}

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
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def _split_words(text: str):
    return re.findall(r"[A-Za-z']+", text)


def _words_with_sentence_position(text: str):
    sentences = _split_sentences(text)
    result = []
    for sentence in sentences:
        sentence_words = re.findall(r"[A-Za-z']+", sentence)
        for i, w in enumerate(sentence_words):
            result.append((w, i == 0))
    return result


def _count_paragraphs(text: str) -> int:
    parts = re.split(r"\n+", text)
    return max(len([p for p in parts if p.strip()]), 1)


def extract_features(text: str, fast_mode: bool = False) -> dict:
    text = text or ""
    words = _split_words(text)
    sentences = _split_sentences(text)

    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    paragraph_count = _count_paragraphs(text)

    avg_word_length = sum(len(w) for w in words) / word_count if word_count else 0

    sentence_lengths = [len(_split_words(s)) for s in sentences]
    avg_sentence_length = word_count / sentence_count
    avg_paragraph_length = word_count / paragraph_count if paragraph_count else 0

    if len(sentence_lengths) > 1:
        mean_sl = sum(sentence_lengths) / len(sentence_lengths)
        variance = sum((l - mean_sl) ** 2 for l in sentence_lengths) / len(sentence_lengths)
        sentence_length_variance = math.sqrt(variance)
    else:
        sentence_length_variance = 0.0

    words_lower = [w.lower() for w in words]
    unique_words = set(words_lower)
    vocab_richness = len(unique_words) / word_count if word_count else 0

    first_100 = words_lower[:100]
    type_token_ratio_100 = len(set(first_100)) / len(first_100) if first_100 else 0

    long_word_count = sum(1 for w in words if len(w) >= 7)
    long_word_ratio = long_word_count / word_count if word_count else 0

    stopword_count = sum(1 for w in words_lower if w in _STOPWORDS)
    stopword_ratio = stopword_count / word_count if word_count else 0

    comma_count = text.count(",")
    semicolon_count = text.count(";")
    question_count = text.count("?")
    exclamation_count = text.count("!")
    commas_per_sentence = comma_count / sentence_count
    semicolons_per_sentence = semicolon_count / sentence_count

    positioned = _words_with_sentence_position(text)[:400]
    checkable = []
    for word, is_sentence_start in positioned:
        if len(word) >= 2 and word.isupper():
            continue
        if word[0].isupper() and not is_sentence_start and word.lower() not in _WEAK_WORDS:
            continue
        checkable.append(word.lower())

    misspelled = _spell.unknown(checkable) if checkable else set()
    misspelled_count = len(misspelled)
    misspelled_ratio = misspelled_count / len(checkable) if checkable else 0

    # ML model needs the weak word count, so calculate it outside the fast_mode check
    found_weak_words = [w for w in words_lower if w in _WEAK_WORDS]
    weak_word_count = len(found_weak_words)

    spelling_suggestions = {}
    weak_word_suggestions = {}

    # Skip heavy Levenshtein calculations during bulk training
    if not fast_mode:
        for word in misspelled:
            candidates = _spell.candidates(word) or set()
            candidates.discard(word)
            ranked = sorted(candidates, key=lambda w: -_spell.word_frequency[w])
            spelling_suggestions[word] = ranked[:3]

        weak_word_suggestions = {
            w: _WEAK_WORD_SUGGESTIONS.get(w, []) for w in set(found_weak_words)
        }

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "sentence_length_variance": sentence_length_variance,
        "avg_word_length": avg_word_length,
        "avg_paragraph_length": avg_paragraph_length,
        "vocab_richness": vocab_richness,
        "type_token_ratio_100": type_token_ratio_100,
        "long_word_ratio": long_word_ratio,
        "stopword_ratio": stopword_ratio,
        "commas_per_sentence": commas_per_sentence,
        "semicolons_per_sentence": semicolons_per_sentence,
        "question_count": question_count,
        "exclamation_count": exclamation_count,
        "misspelled_count": misspelled_count,
        "misspelled_ratio": misspelled_ratio,
        "weak_word_count": weak_word_count,
        # UI feedback only -- NOT in FEATURE_ORDER
        "spelling_suggestions": spelling_suggestions,
        "found_weak_words": list(set(found_weak_words)),
        "weak_word_suggestions": weak_word_suggestions,
        "paragraph_count": paragraph_count,
    }


FEATURE_ORDER = [
    "word_count",
    "sentence_count",
    "avg_sentence_length",
    "sentence_length_variance",
    "avg_word_length",
    "avg_paragraph_length",
    "vocab_richness",
    "type_token_ratio_100",
    "long_word_ratio",
    "stopword_ratio",
    "commas_per_sentence",
    "semicolons_per_sentence",
    "question_count",
    "exclamation_count",
    "misspelled_count",
    "misspelled_ratio",
    "weak_word_count",
]


def features_to_vector(features: dict):
    return [features[name] for name in FEATURE_ORDER]