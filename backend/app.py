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
from coherence import compute_coherence
import db

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
db.init_db()  # creates history.db and the history table if they don't exist yet


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
            for w, candidates in list(suggestions.items())[:4]:
                if candidates:  # non-empty list of ranked candidates
                    shown = " or ".join(f"<span class='sugg-word'>{c}</span>" for c in candidates[:2])
                    sugg_list.append(f"Change <span class='error-word'>{w}</span> to {shown}")
                else:  # no candidates found at all (e.g. a very unusual typo)
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

        
    # --- Length -- separate from Structure on purpose. word_count is BY
    # FAR the single most influential feature in the actual scoring model
    # (Ridge coefficient +10.07 -- roughly 4-5x larger than every other
    # feature), yet nothing in the other feedback categories says
    # anything about it. That's a real gap: an essay can have clean
    # grammar, good structure, varied vocabulary, and solid coherence,
    # and still score moderately just because it's short -- which looks
    # like a contradiction unless length is called out explicitly.
    # Thresholds below are from the REAL training data, not guessed:
    # high-scoring essays (80+) had a median of 475 words; low-scoring
    # essays (<50) had a median of just 171 words.
    wc = feat["word_count"]
    if wc >= 400:
        feedback.append({
            "category": "Length",
            "status": "good",
            "label": "Well-developed",
            "note": f"At {wc} words, this essay is in the range associated with higher-scoring essays in the training data (which averaged around 475 words).",
        })
    elif wc >= 250:
        feedback.append({
            "category": "Length",
            "status": "good",
            "label": "Reasonable",
            "note": f"At {wc} words, this essay is a reasonable length. Higher-scoring essays in the training data averaged closer to 475 words -- a bit more elaboration could help.",
        })
    else:
        feedback.append({
            "category": "Length",
            "status": "warn",
            "label": "Short",
            "note": (
                f"At {wc} words, this essay is notably shorter than what tends to score well: "
                f"in the training data, essays scoring 80+ averaged around 475 words, while essays "
                f"scoring below 50 averaged around 171 words. This matters more to the score than any "
                f"single grammar or vocabulary issue. Consider adding another paragraph with a "
                f"specific example or counterargument, rather than only polishing what's already here."
            ),
        })

    # --- Structure -- based on ACTUAL sentence/paragraph counts, not a
    # fixed pair of strings. Previously this only had two possible notes
    # total, regardless of the essay, which is why most single-paragraph
    # essays always showed the identical "What to add" text -- same bug
    # class as the Vocabulary fix earlier, just not yet applied here.
    sc = feat["sentence_count"]
    pc = feat["paragraph_count"]

    if pc >= 3 and sc >= 8:
        feedback.append({
            "category": "Structure",
            "status": "good",
            "label": "Strong",
            "note": f"This essay has {sc} sentences across {pc} paragraphs, suggesting a well-developed structure with room to explore multiple ideas.",
        })
    elif pc >= 2 and sc >= 5:
        feedback.append({
            "category": "Structure",
            "status": "good",
            "label": "Developing",
            "note": f"This essay has {sc} sentences across {pc} paragraphs -- a reasonable structure. Adding one more paragraph could let you develop your points further.",
        })
    elif pc == 1 and sc >= 5:
        feedback.append({
            "category": "Structure",
            "status": "warn",
            "label": "Single paragraph",
            "note": f"<strong>What to add:</strong> This essay's {sc} sentences are all in a single paragraph. Consider breaking it into 2-3 paragraphs, each with a clear topic sentence and a specific supporting example.",
        })
    else:
        feedback.append({
            "category": "Structure",
            "status": "warn",
            "label": "Minimal",
            "note": f"<strong>What to add:</strong> This essay only has {sc} sentence{'s' if sc != 1 else ''} across {pc} paragraph{'s' if pc != 1 else ''} -- too short to show much structure yet. Aim for at least 2-3 paragraphs with several sentences each.",
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


def build_coherence_feedback(essay: str):
    """
    Returns a Coherence feedback card, or None if the essay is too short
    for a meaningful check. Primary signal is semantic sentence-embedding
    similarity to the essay's overall topic (tested and validated against
    real coherent/disjointed essays -- see coherence.py's docstring for
    the full story of what was tried before this and why). Transition
    word usage is included as supporting, concrete evidence alongside it.
    """
    result = compute_coherence(essay)
    if result is None:
        return None

    transitions_note = ""
    if result["transitions_found"]:
        preview = ", ".join(result["transitions_found"][:4])
        transitions_note = f" It also uses transition words ({preview}) to guide the reader."

    if result["band"] == "good":
        return {
            "category": "Coherence",
            "status": "good",
            "label": "Well-connected",
            "note": f"This essay's ideas stay consistently connected to its overall topic.{transitions_note}",
        }
    else:
        return {
            "category": "Coherence",
            "status": "warn",
            "label": "Some drift",
            "note": (
                "Parts of this essay seem to drift from its main topic -- one or more "
                "sentences may be less connected to the overall argument than the rest. "
                "Consider checking whether every sentence clearly supports your main point."
                + transitions_note
            ),
        }


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    essay = data.get("essay", "")
    prompt = data.get("prompt", "")  # optional -- see build_relevance_feedback

    # Customizable evaluation criteria: which feedback categories the user
    # actually wants to see. Defaults to all five if not specified, so
    # existing behavior is unchanged for anyone not using this. The
    # underlying SCORE is unaffected either way -- it comes from the ML
    # model, not from which feedback cards are displayed -- this only
    # customizes which feedback the user is shown, matching what the
    # synopsis calls "customizable evaluation criteria."
    ALL_CATEGORIES = ["Grammar", "Structure", "Length", "Vocabulary", "Coherence", "Relevance"]
    enabled_categories = data.get("enabled_categories", ALL_CATEGORIES)

    if not essay or len(essay.strip().split()) < 30:
        return jsonify({"error": "Essay must be at least 30 words."}), 400

    feat = extract_features(essay)  # still used for the feedback cards either way

    bundle = get_model_bundle()
    vector = [features_to_vector(feat)]
    vector_scaled = bundle["scaler"].transform(vector)
    baseline_score = max(0, min(100, round(bundle["model"].predict(vector_scaled)[0])))

    transformer_score = None
    if get_transformer_bundle() is not None:
        transformer_score = predict_with_transformer(essay)

    score = transformer_score if transformer_score is not None else baseline_score

    feedback = build_feedback(feat)
    coherence_card = build_coherence_feedback(essay)
    if coherence_card is not None:
        feedback.append(coherence_card)
    relevance_card = build_relevance_feedback(essay, prompt)
    if relevance_card is not None:
        feedback.append(relevance_card)

    feedback = [f for f in feedback if f["category"] in enabled_categories]

    return jsonify({
        "score": score,
        "baseline_score": baseline_score,
        "transformer_score": transformer_score,
        "summary": score_to_band(score),
        "feedback": feedback,
        "stats": {
            "word_count": feat["word_count"],
            "sentence_count": feat["sentence_count"],
        },
        # Structured per-word data for the interactive click-to-fix editor:
        # each entry carries its own real suggestion(s), not just a bare word.
        # NOTE: "suggestions" is a LIST for both spelling and weak_words now
        # (previously spelling used a single "suggestion" string) -- this
        # unification is what lets the frontend use one popup code path for
        # both, and lets spelling offer multiple candidates instead of being
        # stuck with pyspellchecker's single frequency-biased top pick.
        "issues": {
            "spelling": [
                {"word": w, "suggestions": s}
                for w, s in feat.get("spelling_suggestions", {}).items()
            ],
            "weak_words": [
                {"word": w, "suggestions": feat.get("weak_word_suggestions", {}).get(w, [])}
                for w in feat.get("found_weak_words", [])
            ],
        }
    })


@app.route("/history", methods=["GET"])
def get_history():
    entries = db.list_entries(limit=10)
    return jsonify({"history": entries})


@app.route("/history", methods=["POST"])
def add_history():
    """
    Saves a completed analysis to the database. Called by the frontend
    right after a successful /analyze -- takes the SAME data /analyze
    already returned, so no re-computation happens here, just storage.
    """
    data = request.get_json(silent=True) or {}
    required = ["essay", "score", "summary", "word_count", "feedback"]
    if not all(k in data for k in required):
        return jsonify({"error": "Missing required history fields."}), 400

    entry_id = db.save_entry(
        essay=data["essay"],
        score=data["score"],
        baseline_score=data.get("baseline_score"),
        transformer_score=data.get("transformer_score"),
        summary=data["summary"],
        word_count=data["word_count"],
        feedback=data["feedback"],
    )
    return jsonify({"id": entry_id})


@app.route("/history/<int:entry_id>", methods=["DELETE"])
def delete_history(entry_id):
    db.delete_entry(entry_id)
    return jsonify({"deleted": entry_id})


@app.route("/history", methods=["DELETE"])
def clear_history():
    db.clear_all()
    return jsonify({"cleared": True})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)