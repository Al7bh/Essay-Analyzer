# Wiring the fine-tuned transformer into `app.py`

**This describes what's already implemented in `app.py`** — this
document originally described the planned integration before it was
built; the code now matches what's below, kept here as the reference
explanation for your report.

## Setup

Once you've run the notebook (`finetune_essay_scorer.ipynb`) and
downloaded `essay_scorer_distilbert.zip`:

1. Unzip it into `backend/transformer_model/`, so you have:
   ```
   backend/transformer_model/
     config.json
     model.safetensors  (or pytorch_model.bin)
     tokenizer.json
     tokenizer_config.json
     ...
   ```
2. `pip install transformers torch --break-system-packages` (already in
   `requirements.txt`).

## How `app.py` actually uses this

Unlike the original plan (transformer OR baseline, whichever's
available), **the current code always computes both scores and returns
both** — the frontend shows the transformer's score as primary when
available, with the baseline shown alongside for comparison, per your
"show both scores" request.

```python
TRANSFORMER_DIR = os.path.join(SCRIPT_DIR, "transformer_model")
_transformer_bundle = None

def get_transformer_bundle():
    global _transformer_bundle
    if _transformer_bundle is None and os.path.isdir(TRANSFORMER_DIR):
        try:
            tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
            model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_DIR)
            model.eval()
            _transformer_bundle = {"tokenizer": tokenizer, "model": model}
        except OSError:
            # Folder exists but weight files are missing (e.g. left out of
            # a git upload to keep the repo small) -- fall back to
            # baseline-only instead of crashing every /analyze request.
            # This exact bug existed and crashed the app before it was
            # caught and fixed -- see conversation history.
            app.logger.warning(f"{TRANSFORMER_DIR} exists but weights couldn't be loaded -- falling back to baseline.")
            _transformer_bundle = False  # sentinel: "checked, not usable" -- avoids retrying every request
    return _transformer_bundle or None


def predict_with_transformer(essay: str) -> float:
    bundle = get_transformer_bundle()
    inputs = bundle["tokenizer"](essay, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    with torch.no_grad():
        output = bundle["model"](**inputs)
    raw = output.logits.item()
    return max(0, min(100, round(raw * 100)))
```

In `/analyze`:

```python
bundle = get_model_bundle()
vector = [features_to_vector(feat)]
vector_scaled = bundle["scaler"].transform(vector)
baseline_score = max(0, min(100, round(bundle["model"].predict(vector_scaled)[0])))

transformer_score = None
if get_transformer_bundle() is not None:
    transformer_score = predict_with_transformer(essay)

score = transformer_score if transformer_score is not None else baseline_score
```

Both scores are returned in the response (`baseline_score`,
`transformer_score`), alongside the single `score` field used as the
"headline" number.

`build_feedback()` — Grammar/Structure/Vocabulary — and the separate
Coherence/Relevance checks all run on `feat`/the raw essay text
directly, independent of which scoring model is used. **Only the score
itself comes from the transformer**; the feedback categories are the
same rule-based logic either way. See the main `README.md`'s section
on what's genuinely ML vs. rule-based for why this matters in your defense.

## Why keep both models, and show both scores

This is a legitimate, defensible engineering decision worth stating
explicitly if asked: "we built and validated an end-to-end baseline
first, measured it honestly (MAE 6.49), then improved the scoring
engine with a fine-tuned transformer — and we show both scores so the
comparison is visible, not just claimed." That's a stronger, more
honest story than only ever showing one number and asserting it's better.