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
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from features import extract_features, features_to_vector
from file_parser import extract_text_from_upload, FileParseError
from relevance import compute_relevance

# --- PATH DEFINITIONS ---
# Resolve relative to this file's location, not whatever folder the app happens to be launched from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSFORMER_DIR = os.path.join(SCRIPT_DIR, "transformer_model")
MODEL_PATH = os.path.join(SCRIPT_DIR, "model.pkl")

_transformer_bundle = None
_bundle = None  # loaded lazily on first request, cached after that

def get_transformer_bundle():
    global _transformer_bundle
    if _transformer_bundle is None and os.path.isdir(TRANSFORMER_DIR):
        try:
            tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
            model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_DIR)
            model.eval()
            _transformer_bundle = {"tokenizer": tokenizer, "model": model}
        except OSError:
            # The folder exists but is missing weight files (e.g. model.safetensors
            # was intentionally left out of a git upload to keep the repo small) —
            # this was previously an unhandled crash on every /analyze request.
            # Fall back to the baseline model instead, exactly as intended.
            app.logger.warning(
                f"{TRANSFORMER_DIR} exists but model weights couldn't be loaded. "
                f"falling back to the baseline model. Copy model.safetensors back "
                f"in to use the transformer."
            )
            _transformer_bundle = False  # sentinel: "checked, not usable" (not None, so we don't retry every request)
    return _transformer_bundle or None

def predict_with_transformer(essay: str) -> float:
    bundle = get_transformer_bundle()
    inputs = bundle["tokenizer"](
        essay, truncation=True, padding="max_length",
        max_length=512, return_tensors="pt"
    )
    with torch.no_grad():
        output = bundle["model"](**inputs)
    raw = output.logits.item()
    return max(0, min(100, round(raw * 100)))

app = Flask(__name__)
CORS(app)  # allows the frontend (served from a different origin/file) to call this API


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
        return "Excellent! Well above average"
    if score >= 70:
        return "Solid draft, a few things to tighten"
    if score >= 50:
        return "Developing. Needs revision in several areas"
    return "Needs significant revision"


def build_feedback(feat: dict) -> list:
    feedback = []

    # --- Grammar / spelling with Word-style corrections ---
    misspelled = feat["misspelled_count"]
    suggestions = feat.get("spelling_suggestions", {})
    
    if misspelled == 0:
        feedback.append({
            "category": "Grammar",
            "status": "good",
            "label": "Clean",
            "note": "No spelling issues detected in this draft.",
        })
    else:
        note_text = f"Found {misspelled} possibly misspelled word(s)."
        if suggestions:
            sugg_list = []
            for w, c in list(suggestions.items())[:4]:
                if c: # If a valid correction was found
                    sugg_list.append(f"Change <span class='error-word'>{w}</span> to <span class='sugg-word'>{c}</span>")
                else: # No correction found (e.g., proper nouns like "Instagram")
                    sugg_list.append(f"Unknown word: <span class='error-word'>{w}</span>")
                    
            note_text += f"<br><br><strong>Spelling Issues:</strong><br>• " + "<br>• ".join(sugg_list)
            if len(suggestions) > 4:
                note_text += "<br><em>...and others.</em>"

        feedback.append({
            "category": "Grammar",
            "status": "warn",
            "label": f"{misspelled} issue{'s' if misspelled != 1 else ''}",
            "note": note_text,
        })

        
    # --- Structure & Content Additions ---
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
            "note": "<strong>What to add:</strong> Consider breaking your ideas into 2-3 distinct paragraphs. Add a clear topic sentence to start each new idea, and support it with a specific example.",
        })

    # --- Vocabulary with specific, per-word replacements (not a fixed generic list) ---
    weak_words = feat.get("found_weak_words", [])
    weak_suggestions = feat.get("weak_word_suggestions", {})
    if feat["vocab_richness"] >= 0.6 and feat["weak_word_count"] <= 2:
        feedback.append({
            "category": "Vocabulary",
            "status": "good",
            "label": "Varied",
            "note": "Good variety of word choice throughout.",
        })
    else:
        note_text = "Try varying your word choice to sound more academic."
        if weak_words:
            # Build one line per ACTUAL weak word found, with ITS OWN real
            # alternatives -- not a single fixed list reused for every essay.
            lines = []
            for w in weak_words[:4]:
                alts = weak_suggestions.get(w, [])
                if alts:
                    alt_html = ", ".join(f"<span class='upgrade-word'>{a}</span>" for a in alts)
                    lines.append(f"Instead of <span class='weak-word'>{w}</span>, try {alt_html}")
            if lines:
                note_text += "<br><br><strong>Suggested Upgrades:</strong><br>" + "<br>".join(lines)
            if len(weak_words) > 4:
                note_text += "<br><em>...and others.</em>"

        feedback.append({
            "category": "Vocabulary",
            "status": "warn",
            "label": "Repetitive",
            "note": note_text,
        })

    return feedback

@app.route("/extract-text", methods=["POST"])
def extract_text():
    """
    Accepts a file upload (multipart/form-data, field name 'file') and
    returns extracted plain text, so the user can review/edit it in the
    textarea before analyzing — same downstream pipeline as pasted text,
    nothing else needs to change.
    """
    file_storage = request.files.get("file")
    try:
        result = extract_text_from_upload(file_storage)
        return jsonify({"essay": result["text"], "warnings": result["warnings"]})
    except FileParseError as e:
        # Expected, user-facing errors (bad file type, corrupted file,
        # password-protected PDF, etc.) — a clean 400, not a crash.
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Anything truly unexpected still gets caught here so the server
        # never 500s on a weird file — worth logging server-side if this
        # ever fires, since it means a case file_parser.py didn't predict.
        app.logger.exception("Unexpected error during file extraction")
        return jsonify({"error": "Something went wrong reading this file."}), 500


def build_relevance_feedback(essay: str, prompt: str):
    """
    Returns a Relevance feedback card, or None if no prompt was given
    (this check is entirely optional -- essays don't always have a known
    prompt, and we shouldn't fabricate a category when there's nothing
    to compare against).
    """
    result = compute_relevance(essay, prompt)
    if result is None:
        return None

    if result["band"] == "good":
        matched_preview = ", ".join(result["matched_keywords"][:5])
        note = f"This essay engages with key terms from the prompt ({matched_preview})."
        return {
            "category": "Relevance",
            "status": "good",
            "label": "On-topic",
            "note": note,
        }
    else:
        missing_preview = ", ".join(result["missing_keywords"][:5])
        note = (
            "Low overlap with the prompt's key terms. This may mean the essay "
            "drifts off-topic, or simply uses different wording for the same "
            "ideas (this check only catches shared vocabulary, not meaning)."
        )
        if missing_preview:
            note += f" Terms from the prompt not found in the essay: {missing_preview}."
        return {
            "category": "Relevance",
            "status": "warn",
            "label": "Low keyword overlap",
            "note": note,
        }


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    essay = data.get("essay", "")
    prompt = data.get("prompt", "")  # optional -- see build_relevance_feedback

    if not essay or len(essay.strip().split()) < 30:
        return jsonify({"error": "Essay must be at least 30 words."}), 400

    feat = extract_features(essay)  # still used for the feedback cards either way

    if get_transformer_bundle() is not None:
        score = predict_with_transformer(essay)
    else:
        bundle = get_model_bundle()
        vector = [features_to_vector(feat)]
        vector_scaled = bundle["scaler"].transform(vector)
        score = max(0, min(100, round(bundle["model"].predict(vector_scaled)[0])))

    feedback = build_feedback(feat)
    relevance_card = build_relevance_feedback(essay, prompt)
    if relevance_card is not None:
        feedback.append(relevance_card)

    return jsonify({
        "score": score,
        "summary": score_to_band(score),
        "feedback": feedback,
        "stats": {
            "word_count": feat["word_count"],
            "sentence_count": feat["sentence_count"],
        },
        # Structured per-word data for the interactive click-to-fix editor:
        # each entry carries its own real suggestion(s), not just a bare word.
        "issues": {
            "spelling": [
                {"word": w, "suggestion": s}
                for w, s in feat.get("spelling_suggestions", {}).items()
            ],
            "weak_words": [
                {"word": w, "suggestions": feat.get("weak_word_suggestions", {}).get(w, [])}
                for w in feat.get("found_weak_words", [])
            ],
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)