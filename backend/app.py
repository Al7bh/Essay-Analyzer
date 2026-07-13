"""
app.py
------
Flask API for the essay analyzer. Exposes POST /analyze, which the
frontend's "Analyze essay" button calls.

Run locally:
    python3 app.py
Then it's available at http://127.0.0.1:5000

Frontend integration: replace the mock click handler in index.html's
<script> with a real fetch('http://127.0.0.1:5000/analyze', {...}) call.
See README.md for the exact snippet to swap in.
"""

import os
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

from features import extract_features, features_to_vector

app = Flask(__name__)
CORS(app)  # allows the frontend (served from a different origin/file) to call this API

# Same fix as train.py: resolve relative to this file's location, not
# whatever folder the app happens to be launched from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model.pkl")
_bundle = None  # loaded lazily on first request, cached after that


def get_model_bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                f"{MODEL_PATH} not found. Run `python3 train.py` first "
                f"to train and save the model."
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def score_to_band(score: float) -> str:
    if score >= 85:
        return "Excellent — well above average"
    if score >= 70:
        return "Solid draft, a few things to tighten"
    if score >= 50:
        return "Developing — needs revision in several areas"
    return "Needs significant revision"


def build_feedback(feat: dict) -> list:
    """
    Turns raw numeric features into the same kind of margin-style
    annotations the frontend already displays (Grammar/Structure/
    Vocabulary). Thresholds here are simple starting heuristics —
    reasonable to tune once you see real essays come through.
    """
    feedback = []

    # --- Grammar / spelling ---
    misspelled = feat["misspelled_count"]
    if misspelled == 0:
        feedback.append({
            "category": "Grammar",
            "status": "good",
            "label": "Clean",
            "note": "No spelling issues detected in this draft.",
        })
    else:
        feedback.append({
            "category": "Grammar",
            "status": "warn",
            "label": f"{misspelled} issue{'s' if misspelled != 1 else ''}",
            "note": f"Found {misspelled} possibly misspelled word(s). Worth a careful proofread.",
        })

    # --- Structure (using sentence/paragraph counts as a proxy) ---
    if feat["sentence_count"] >= 5 and feat["paragraph_count"] >= 2:
        feedback.append({
            "category": "Structure",
            "status": "good",
            "label": "Strong",
            "note": "Multiple paragraphs and sentences suggest a developed structure.",
        })
    else:
        feedback.append({
            "category": "Structure",
            "status": "warn",
            "label": "Underdeveloped",
            "note": "Consider breaking your ideas into more paragraphs with clear topic sentences.",
        })

    # --- Vocabulary ---
    if feat["vocab_richness"] >= 0.6 and feat["weak_word_count"] <= 2:
        feedback.append({
            "category": "Vocabulary",
            "status": "good",
            "label": "Varied",
            "note": "Good variety of word choice throughout.",
        })
    else:
        feedback.append({
            "category": "Vocabulary",
            "status": "warn",
            "label": "Repetitive",
            "note": "Try varying word choice — some words (e.g. 'good', 'important') appear often and could be replaced with more specific language.",
        })

    return feedback


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    essay = data.get("essay", "")

    if not essay or len(essay.strip().split()) < 30:
        return jsonify({"error": "Essay must be at least 30 words."}), 400

    bundle = get_model_bundle()
    model = bundle["model"]
    scaler = bundle["scaler"]

    feat = extract_features(essay)
    vector = [features_to_vector(feat)]
    vector_scaled = scaler.transform(vector)

    raw_score = model.predict(vector_scaled)[0]
    score = max(0, min(100, round(raw_score)))  # clamp to a sane 0-100 range

    return jsonify({
        "score": score,
        "summary": score_to_band(score),
        "feedback": build_feedback(feat),
        "stats": {
            "word_count": feat["word_count"],
            "sentence_count": feat["sentence_count"],
        },
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
