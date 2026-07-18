"""
test_coherence_sbert.py
------------------------
Run this LOCALLY (not in a sandboxed environment) to test whether
semantic sentence embeddings actually fix the coherence detection
problem that TF-IDF failed at.

Setup:
    pip install sentence-transformers

Run:
    python test_coherence_sbert.py

This downloads a small pretrained model (~80MB, first run only) from
Hugging Face -- needs normal internet access, which is why this has to
be run on your machine rather than in the sandboxed environment I was
working in (that environment's network is restricted to a small
allowlist that doesn't include huggingface.co).

WHAT TO LOOK FOR: the same test essays that broke TF-IDF (a coherent
essay and a deliberately disjointed one scored nearly identically). If
SBERT gives clear separation between them here, it's a real fix and
worth building into coherence.py. If it still doesn't separate them
well, that's useful evidence FOR moving to actual model training
instead of chasing a better off-the-shelf embedding.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import re


def split_sentences(text):
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def compute_coherence_sbert(essay, model):
    sentences = split_sentences(essay)
    if len(sentences) < 3:
        return None

    embeddings = model.encode(sentences)

    # Same "distance to document centroid" approach as the TF-IDF version,
    # just with semantic embeddings instead of bag-of-words vectors.
    centroid = np.mean(embeddings, axis=0)

    similarities = []
    for emb in embeddings:
        sim = np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid))
        similarities.append(float(sim))

    return {
        "avg_similarity": round(sum(similarities) / len(similarities), 3),
        "min_similarity": round(min(similarities), 3),
        "per_sentence": [round(s, 3) for s in similarities],
    }


if __name__ == "__main__":
    print("Loading model (first run downloads ~80MB from Hugging Face)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Loaded.\n")

    examples = {
        "COHERENT (natural style, pronouns/synonyms, no repeated nouns)": """Social media has had a very big impact on teenage mental health in recent years.
Many teenagers now spend several hours a day on platforms like Instagram and TikTok, which can be both
good and bad for their wellbeing. On one hand, social media allows teens to communicate with friends and
find communities that share their interests, which is nice for kids who feel isolated in their local area.
However, this constant connection also has a bad side. Studies have shown that heavy use is linked to
increased anxiety and depression, especially when teenagers compare their own lives to the carefully
edited posts of others.""",

        "DISJOINTED (random unrelated topics)": """Social media has changed how teenagers communicate. My favorite food is pizza
with extra cheese. The weather has been quite unpredictable this year. Basketball is a popular
sport played by millions. I enjoy reading mystery novels on weekends.""",

        "WEAK BUT ON-TOPIC (short, choppy, still coherent)": """i think social media is bad and good. some people like it
some people dont. it can help you talk to friends but also it can waste time. i dont know what to say more
about this topic because its hard.""",

        "MOSTLY COHERENT WITH ONE OFF-TRACK SENTENCE": """Social media has had a big impact on teenage mental health.
Many teenagers spend hours daily on Instagram and TikTok. This constant use is linked to anxiety and depression
in many studies. My favorite pizza topping is pepperoni and mushrooms. Schools are now trying to teach digital
literacy to address these mental health concerns among students.""",
    }

    print(f"{'Example':<55} {'avg_sim':>8} {'min_sim':>8}")
    print("-" * 75)
    for name, essay in examples.items():
        r = compute_coherence_sbert(essay, model)
        print(f"{name[:54]:<55} {r['avg_similarity']:>8} {r['min_similarity']:>8}")

    print("\nIf COHERENT scores meaningfully higher than DISJOINTED here")
    print("(unlike the TF-IDF attempt, where they were nearly identical),")
    print("this approach is worth building into coherence.py for real.")
