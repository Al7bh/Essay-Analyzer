"""
coherence.py
------------
Checks whether an essay stays on one connected train of thought.

DESIGN HISTORY -- worth reading before changing this file. Three
approaches were tried, in order, each one only kept after being tested
against real coherent vs. deliberately disjointed essays:

1. TF-IDF similarity between ADJACENT sentences. Failed: good writing
   uses pronouns/synonyms instead of repeating nouns, so a coherent
   essay and a disjointed one both scored ~0.0 and were indistinguishable.

2. TF-IDF similarity of each sentence to the essay's overall topic
   vector. Also failed, for the same underlying reason (bag-of-words
   can't see that "the platform" and "Instagram" mean the same thing) --
   a disjointed essay scored 0.447, nearly identical to a coherent
   essay's 0.473.

3. SEMANTIC sentence embeddings (SBERT, all-MiniLM-L6-v2) compared to
   the essay's centroid -- THIS ONE WORKS. Tested on the same
   calibration essays: coherent (0.697) clearly outscored disjointed
   (0.515), and an essay with one deliberately off-track sentence
   inserted (0.595) correctly landed in between. This is what's used
   below. Note: the same test also tried using the SINGLE least-similar
   sentence as an "outlier detector" -- that part did NOT work (it was
   backwards -- disjointed scored higher than coherent on it), so only
   the AVERAGE similarity is used, not a per-sentence outlier score.

This is still a proxy, not true coherence understanding: it measures
semantic topical consistency, not whether an argument logically follows.
Combined with transition-word detection (a real, deterministic signal
of intentional structural connectors) for a more complete, explainable
note -- rather than relying on one signal alone.
"""

import re
from sentence_transformers import SentenceTransformer
import numpy as np

# Loaded once at import time. ~80MB, downloaded from Hugging Face on
# first run and cached locally after that -- needs normal internet
# access the first time the server starts.
_model = SentenceTransformer('all-MiniLM-L6-v2')

_TRANSITION_WORDS = {
    "however", "although", "though", "nevertheless", "nonetheless",
    "on the other hand", "in contrast", "conversely", "yet", "whereas",
    "furthermore", "moreover", "additionally", "in addition", "also",
    "besides",
    "therefore", "thus", "consequently", "as a result", "because of this",
    "hence",
    "for example", "for instance", "specifically", "such as",
    "first", "second", "third", "finally", "overall", "in conclusion",
    "to summarize", "in summary",
}


def _split_sentences(text: str):
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def compute_coherence(essay: str) -> dict:
    """
    Returns:
        avg_similarity: float, average semantic similarity of each
                        sentence to the essay's overall topic centroid
        band: 'good' | 'warn' -- threshold calibrated against real
              coherent/disjointed/mixed test essays (see docstring above)
        transitions_found: transition words also found, as supporting
                            evidence in the feedback note
    """
    sentences = _split_sentences(essay)
    if len(sentences) < 3:
        return None  # too short for a meaningful coherence check

    embeddings = _model.encode(sentences)
    centroid = np.mean(embeddings, axis=0)

    similarities = [
        float(np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid)))
        for emb in embeddings
    ]
    avg_similarity = sum(similarities) / len(similarities)

    # Threshold set between the tested "mixed" essay (0.595, has one
    # off-track sentence -- should warn) and "weak but on-topic" (0.682,
    # should still pass) -- so an essay with a genuine off-track idea
    # gets flagged, while a simply plain/choppy-but-on-topic essay doesn't.
    band = "good" if avg_similarity >= 0.60 else "warn"

    text_lower = essay.lower()
    transitions_found = sorted({
        tw for tw in _TRANSITION_WORDS
        if re.search(r"\b" + re.escape(tw) + r"\b", text_lower)
    })

    return {
        "avg_similarity": round(avg_similarity, 3),
        "band": band,
        "transitions_found": transitions_found,
    }