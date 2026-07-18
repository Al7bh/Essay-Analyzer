# Wiring the fine-tuned transformer into `app.py`

Once you've run the notebook and downloaded `essay_scorer_distilbert.zip`:

1. Unzip it into `backend/transformer_model/`, so you have:
   ```
   backend/transformer_model/
     config.json
     model.safetensors  (or pytorch_model.bin)
     tokenizer.json
     tokenizer_config.json
     vocab.txt
     ...
   ```

2. Install the extra dependency:
   ```bash
   pip install transformers torch --break-system-packages
   ```

3. In `app.py`, add a second prediction path that uses the transformer
   instead of the Scikit-learn model. **Keep the baseline as a fallback**
   — if the transformer model folder isn't there (e.g. you're demoing on
   a machine where you didn't copy it over), the app should still work:

   ```python
   import os
   import torch
   from transformers import AutoTokenizer, AutoModelForSequenceClassification

   TRANSFORMER_DIR = os.path.join(SCRIPT_DIR, "transformer_model")
   _transformer_bundle = None

   def get_transformer_bundle():
       global _transformer_bundle
       if _transformer_bundle is None and os.path.isdir(TRANSFORMER_DIR):
           tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
           model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_DIR)
           model.eval()
           _transformer_bundle = {"tokenizer": tokenizer, "model": model}
       return _transformer_bundle

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
   ```

4. In the `/analyze` route, try the transformer first and fall back to
   the baseline if it's not available:

   ```python
   @app.route("/analyze", methods=["POST"])
   def analyze():
       data = request.get_json(silent=True) or {}
       essay = data.get("essay", "")

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

       return jsonify({
           "score": score,
           "summary": score_to_band(score),
           "feedback": build_feedback(feat),
           "stats": {
               "word_count": feat["word_count"],
               "sentence_count": feat["sentence_count"],
           },
       })
   ```

   Notice `build_feedback()` still uses the same feature-based logic as
   before (grammar/structure/vocabulary cards) — only the *score itself*
   comes from the transformer. You don't need the transformer to also
   generate feedback text; the existing heuristics are still useful for
   that, and rewriting them would be extra work for no real benefit.

## Why keep both models in the repo

For your FYP report/defense, this is actually a good story: "we built
and validated an end-to-end baseline first, then improved the scoring
engine with a fine-tuned transformer, while keeping the baseline as a
fallback." That's a legitimate, defensible engineering decision, not
a compromise — mention it explicitly if asked why both exist.
