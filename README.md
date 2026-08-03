# Marginal

Backend Flask API that scores an essay and returns structured feedback, using either a baseline Scikit-learn model or a fine-tuned DistilBERT transformer (both are computed and returned together).

## 1. Setup
```bash
cd backend
pip install -r requirements.txt --break-system-packages

```

*(Drop `--break-system-packages` if you are using a virtual environment.)*

**First run note:** `sentence-transformers` (used for the Coherence check) downloads a small pretrained model (~80MB) from Hugging Face the first time the server starts, and caches it locally after that. Needs normal internet access on that first run only.

## 2. The Two Models

Both models are pre-trained and included. Nothing to do before running the app.

* **Baseline (`model.pkl`)**: A Scikit-learn model trained on 17 hand-crafted NLP features. The training script automatically tests multiple algorithms (RidgeCV, GradientBoosting, SVR) and saves the best performer. The current iteration uses Gradient Boosting and achieved a highly competitive **MAE of ~5.80** and a **QWK of ~0.86** on Essay Set 1.
* **Transformer (`transformer_model/`)**: Fine-tuned `distilbert-base-uncased`. Initially tested on Set 1, but deliberately expanded to train across the **entire Kaggle ASAP dataset (~13,000 essays across 8 different writing prompts)**. Achieved a generalized **MAE of ~9.97** across all rubrics.

`/analyze` computes **both** scores every time and returns both (`baseline_score`, `transformer_score`). The frontend shows the transformer's score as primary when available, with the baseline alongside it via a toggle button. It falls back to baseline-only if the transformer model folder is missing or weight files were not copied in.

*Note: This fallback was a real bug found during development. The original code checked that the folder existed but never caught the `OSError` when weight files were absent, causing every request to crash with a 500 error. Fixed by wrapping model loading in `try/except` with a logged warning.*

**To retrain the baseline:** `python train.py sample_data/asap_set1_rescaled.csv`
**To retrain the transformer:** Run `Train_ASAP_Transformer.ipynb` via Jupyter/Colab.

## 3. Baseline Model Evolution (features.py v1 to v2)

**v1 (original, 10 features, Ridge regression, MAE 6.49):**
word_count, sentence_count, paragraph_count, avg_sentence_length, avg_word_length, vocab_richness, long_word_ratio, misspelled_count, misspelled_ratio, weak_word_count.
*After inspecting the trained Ridge model's actual coefficients, two features had near-zero weight: `paragraph_count` (+0.000) and `avg_sentence_length` (-0.025). These were wasting model capacity.*

**v2 (current, 17 features, Gradient Boosting, MAE 5.80):**
Dropped the two dead-weight features. Added 9 new, linguistically motivated features:

* `sentence_length_variance` -- detects monotonous same-length sentences
* `avg_paragraph_length` -- how developed each paragraph is
* `type_token_ratio_100` -- vocab richness on first 100 words (length-normalized)
* `stopword_ratio` -- function-word density (low = more content-rich writing)
* `commas_per_sentence` -- proxy for multi-clause sentence complexity
* `semicolons_per_sentence` -- strong marker of sophisticated writing
* `question_count` -- flags essays that are entirely assertions
* `exclamation_count` -- flags informal/over-emphatic tone
*(Note: `avg_word_length` was kept but repositioned)*

**train.py v2 (multi-model comparison):**
Instead of always using Ridge(alpha=1.0), the training script tries three models and saves whichever has the lowest MAE on the held-out test:

1. `RidgeCV` (cross-validates alpha automatically)
2. `GradientBoostingRegressor` (300 trees, max_depth=4, lr=0.05)
3. `SVR` with RBF kernel (C=50, epsilon=2.0)

## 4. Run the API

```bash
python app.py

```

Runs at `http://127.0.0.1:5000`. On first run, this also creates `history.db` (SQLite) automatically if it does not exist yet.

Check which model is loaded and its MAE: `GET http://127.0.0.1:5000/health`
Returns:

```json
{
  "status": "ok",
  "baseline_model": "GradientBoosting",
  "baseline_mae": 5.80,
  "feature_count": 17,
  "transformer_loaded": true
}

```

## 5. Endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/analyze` | POST | Main endpoint. See request/response shape below. |
| `/extract-text` | POST | Upload `.pdf`/`.docx`/`.txt` (multipart, field `file`). Returns extracted plain text. Never crashes on bad input. |
| `/history` | GET | Returns the 10 most recent saved analyses from the database. |
| `/history` | POST | Saves a completed analysis. |
| `/history/<id>` | DELETE | Deletes one saved entry. |
| `/history` | DELETE | Clears all history. |
| `/health` | GET | Liveness check with model info. |

### `/analyze` Request Body

```json
{
  "essay": "required, at least 30 words",
  "prompt": "optional -- enables Relevance feedback if provided",
  "enabled_categories": ["Grammar", "Structure", "Vocabulary", "Coherence", "Relevance"]
}

```

`enabled_categories` defaults to all five if not specified. This is the "customizable evaluation criteria" feature. Importantly, disabling a category does NOT just hide its feedback card -- it also zeroes out that category's underlying features in the baseline model's scoring via feature-mean substitution (`StandardScaler.mean_`). This is a legitimate ablation technique for a linear/tree model. The transformer score is unaffected by this since it has no equivalent named-feature structure.

### `/analyze` Response Shape

```json
{
  "score": 78,
  "baseline_score": 72,
  "transformer_score": 78,
  "baseline_summary": "Solid draft, a few things to tighten",
  "transformer_summary": "Solid draft, a few things to tighten",
  "summary": "Solid draft, a few things to tighten",
  "feedback": [
    {"category": "Grammar", "status": "good|warn", "label": "...", "note": "..."}
  ],
  "stats": {"word_count": 217, "sentence_count": 10},
  "issues": {
    "spelling": [{"word": "adress", "suggestions": ["dress", "address"]}],
    "weak_words": [{"word": "good", "suggestions": ["beneficial", "effective", "valuable"]}]
  }
}

```

`issues` feeds the click-to-fix editor. `suggestions` is a list for both spelling and vocabulary. Spelling used to return a single forced best-guess via `.correction()`, which was wrong for "adress" (returned "dress" over "address" because "dress" has higher raw corpus frequency). Fixed by offering the top 3 ranked candidates instead.

## 6. File Overview

| File | Purpose |
| --- | --- |
| `app.py` | Flask API, all routes, feedback-card assembly, feature ablation. |
| `features.py` | Essay text to 17 numeric features. Includes a `fast_mode` parameter to skip heavy spellcheck heuristics during bulk model training. |
| `train.py` | Trains and compares RidgeCV/GradientBoosting/SVR, uses `fast_mode=True` to extract features, saves best model as `model.pkl`. |
| `relevance.py` | TF-IDF + keyword overlap between essay and prompt. Optional. |
| `coherence.py` | SBERT-based semantic sentence coherence check. |
| `db.py` | SQLite essay history logic (`history.db`, auto-created on first run). |
| `file_parser.py` | Extracts text from PDF/DOCX/TXT. Crash-proof against corrupted files, passwords, embedded images. |
| `evaluate_models.py` | Compares baseline vs. transformer on held-out essays. Computes MAE, RMSE, and Quadratic Weighted Kappa (QWK). Generates a markdown report, CSV, and scatter plots. |
| `Train_ASAP_Transformer.ipynb` | Colab notebook that scales DistilBERT training across all 8 Kaggle ASAP datasets (~13k essays) using min-max normalization. |
| `sample_data/` | `sample_essays.csv` (12 test essays) and `asap_set1_rescaled.csv` (1,783 real ASAP essays). |

## 7. Trials and Errors Worth Knowing About

These are real things that went wrong during development. Each is documented here so you can explain them in your defense rather than being caught off guard.

**T1 -- Spellchecker flagging brand names as misspelled.**
`pyspellchecker`'s bundled dictionary has no knowledge of brand names, proper nouns, or contractions. "Instagram," "TikTok," "don't" were all being flagged as errors. Fixed with: (a) a custom whitelist of ~60 commonly-missed words loaded at import time, and (b) two heuristics applied before checking -- skip ALL-CAPS words (likely acronyms) and skip capitalized mid-sentence words (likely proper nouns). *Trade-off:* a genuinely misspelled name like "Instagraam" now slips through uncaught. This is a deliberate choice to prevent false positives from eroding user trust.

**T2 -- Spelling popup showing only one suggestion ("dress" for "adress").**
`pyspellchecker`'s `.correction()` returns the single highest raw-frequency candidate. "dress" (92,448 occurrences) outranked "address" (70,429) purely on corpus frequency. Fixed by switching to `.candidates()` and offering the top 3 ranked candidates.

**T3 -- Vocabulary popup silently broken (worked for spelling, not vocab).**
The `escapeHtml()` function did not escape double-quote characters. When a suggestion array was JSON.stringify'd and embedded in a `data-suggestions` HTML attribute, the embedded quote characters prematurely closed the attribute, corrupting the tag. `JSON.parse()` then threw silently. Fixed by adding quote escaping to `escapeHtml()`.

**T4 -- Vocabulary feedback showing the same three words every time.**
The feedback card always suggested "beneficial, crucial, effective" regardless of which weak words were actually in the essay. Root cause: a fixed hardcoded list instead of a per-word synonym lookup. Fixed with a real dictionary mapping each weak word to its own specific alternatives.

**T5 -- Transformer fallback silently crashing every request.**
The fallback logic checked that the `transformer_model` folder existed, but did not catch `OSError` when weight files were absent (e.g., `.safetensors` files ignored in Git). Result: every `/analyze` request threw an unhandled 500. Fixed by wrapping model loading in `try/except OSError`.

**T6 -- Paragraph counting always returning 1.**
The paragraph split used `\n\n` (blank line). In a `<textarea>`, visual line-wrapping does not insert real newline characters -- only actual Enter keypresses do. A single Enter between paragraphs was not being counted. Fixed by splitting on any real newline (`\n+`).

**T7 -- Coherence detection: two failed approaches before one that worked.**
*Attempt 1:* TF-IDF similarity between adjacent sentences. Failed: good writing uses pronouns and synonyms instead of repeating nouns. A coherent essay and a deliberately disjointed one both scored ~0.0.
*Attempt 2:* TF-IDF similarity of each sentence to the essay's overall topic vector. Failed: at 5-10 sentences, TF-IDF overlap still isn't a strong enough signal.
*Attempt 3:* SBERT semantic embeddings compared to essay centroid. This worked beautifully, providing clear numerical separation between coherent, mixed, and disjointed essays.

**T8 -- Baseline model had two dead-weight features.**
After training the v1 Ridge model, inspecting its coefficients revealed `paragraph_count` (+0.000) and `avg_sentence_length` (-0.025) had negligible effects. Both were replaced in `features.py` v2 with variance metrics.

**T9 -- `train.py` only ever tried one model type.**
The original script always used `Ridge(alpha=1.0)` with no hyperparameter search. GradientBoosting and SVR are known to outperform linear models on tabular feature sets. `train.py` v2 tries all three and saves whichever produces the lowest held-out MAE.

**T10 -- Spellchecker crashing the bulk model training (Infinite Hang).**
During multi-model training over 1,783 essays, the terminal would hang completely. Root cause: the system was spending massive compute power running Levenshtein distance math to generate typo suggestions (`_spell.candidates`) for every single typo in the corpus. Fixed by adding a `fast_mode=True` parameter to `extract_features()`. `train.py` uses this to skip suggestion generation (since the ML model only needs the numerical ratio of misspellings), reducing extraction time to ~10.5 seconds. The web API still runs the full check for the UI.

## 8. What is genuinely ML vs. Rule-Based

Be precise about this in your defense. Examiners do ask.

* **The score** is Machine Learning: the baseline Scikit-learn tabular model or the fine-tuned DistilBERT deep learning architecture.
* **The feedback** (Grammar, Structure, Vocabulary, Coherence, Relevance) is deterministic, rule-based, and dictionary-driven (Spellchecker, weak-word lists, TF-IDF cosine similarity, SBERT embeddings). **This is a deliberate design choice:** rule-based feedback is transparent, fast, and does not hallucinate.
