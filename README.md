# Marginal — Backend

Flask API that scores an essay and returns structured feedback, using
either a baseline Scikit-learn model or a fine-tuned DistilBERT
transformer (both are computed and returned together).

---

## 1. Setup

```bash
cd backend
pip install -r requirements.txt --break-system-packages
```
Drop `--break-system-packages` if you are using a virtual environment.

**First run note:** `sentence-transformers` (used for the Coherence
check) downloads a small pretrained model (~80MB) from Hugging Face
the first time the server starts, and caches it locally after that.
Needs normal internet access on that first run only.

---

## 2. The two models

Both are already trained and included. Nothing to do before running the app.

- **Baseline** (`model.pkl`): A Scikit-learn model trained on 17
  hand-crafted features over 1,783 real ASAP essays. The model type
  and MAE depend on which model won during training (see Section 3).
  Original Ridge baseline MAE was 6.49 (0-100 scale); retraining with
  the expanded feature set and multi-model comparison should improve this.
- **Transformer** (`transformer_model/`): Fine-tuned DistilBERT,
  trained via the Colab notebook (`../colab/finetune_essay_scorer.ipynb`).

`/analyze` computes **both** scores every time and returns both
(`baseline_score`, `transformer_score`). The frontend shows the
transformer's score as primary when available, with the baseline
alongside it via a toggle button. Falls back to baseline-only if the
transformer model folder is missing or weight files were not copied in.
This fallback was a real bug found during development: the original code
checked that the folder existed but never caught the OSError when weight
files were absent, causing every request to crash with a 500 error.
Fixed by wrapping model loading in try/except with a logged warning.

To retrain the baseline: `python train.py sample_data/asap_set1_rescaled.csv`
To retrain the transformer: see `../colab/finetune_essay_scorer.ipynb`

---

## 3. Baseline model evolution (features.py v1 to v2)

**v1 (original, 10 features, Ridge regression, MAE 6.49):**
word_count, sentence_count, paragraph_count, avg_sentence_length,
avg_word_length, vocab_richness, long_word_ratio, misspelled_count,
misspelled_ratio, weak_word_count.

After inspecting the trained Ridge model's actual coefficients, two
features had near-zero weight: `paragraph_count` (+0.000) and
`avg_sentence_length` (-0.025). These were wasting model capacity.

**v2 (current, 17 features):**
Dropped the two dead-weight features. Added 9 new ones:
- `sentence_length_variance` -- detects monotonous same-length sentences
- `avg_paragraph_length` -- how developed each paragraph is
- `type_token_ratio_100` -- vocab richness on first 100 words (length-normalized)
- `stopword_ratio` -- function-word density (low = more content-rich writing)
- `commas_per_sentence` -- proxy for multi-clause sentence complexity
- `semicolons_per_sentence` -- strong marker of sophisticated writing
- `question_count` -- flags essays that are entirely assertions
- `exclamation_count` -- flags informal/over-emphatic tone
- `avg_word_length` was kept but repositioned

**train.py v2 (multi-model comparison):**
Instead of always using Ridge(alpha=1.0), the new training script tries
three models and saves whichever has the lowest MAE on the held-out test:
1. RidgeCV (cross-validates alpha automatically)
2. GradientBoostingRegressor (300 trees, max_depth=4, lr=0.05)
3. SVR with RBF kernel (C=50, epsilon=2.0)

Run `python train.py sample_data/asap_set1_rescaled.csv` to retrain.
The saved `model.pkl` now includes `model_name` and `mae` fields so
`/health` can tell you exactly which model is loaded and what its MAE is.

---

## 4. Run the API

```bash
python app.py
```
Runs at `http://127.0.0.1:5000`. On first run this also creates
`history.db` (SQLite) automatically if it does not exist yet.

Check which model is loaded and its MAE:
```
GET http://127.0.0.1:5000/health
```
Returns:
```json
{
  "status": "ok",
  "baseline_model": "GradientBoosting",
  "baseline_mae": 5.xx,
  "feature_count": 17,
  "transformer_loaded": false
}
```

---

## 5. Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/analyze` | POST | Main endpoint. See request/response shape below. |
| `/extract-text` | POST | Upload `.pdf`/`.docx`/`.txt` (multipart, field `file`). Returns extracted plain text. Never crashes on bad input. |
| `/history` | GET | Returns the 10 most recent saved analyses from the database. |
| `/history` | POST | Saves a completed analysis. |
| `/history/<id>` | DELETE | Deletes one saved entry. |
| `/history` | DELETE | Clears all history. |
| `/health` | GET | Liveness check with model info. |

### `/analyze` request body
```json
{
  "essay": "required, at least 30 words",
  "prompt": "optional -- enables Relevance feedback if provided",
  "enabled_categories": ["Grammar", "Structure", "Vocabulary", "Coherence", "Relevance"]
}
```

`enabled_categories` defaults to all five if not specified. This is the
"customizable evaluation criteria" feature. Importantly, disabling a
category does NOT just hide its feedback card -- it also zeroes out that
category's underlying features in the baseline model's scoring via
feature-mean substitution (StandardScaler.mean_). This is a legitimate
ablation technique for a linear/tree model. The transformer score is
unaffected by this since it has no equivalent named-feature structure.

### `/analyze` response shape
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

`issues` feeds the click-to-fix editor. `suggestions` is a list for
both spelling and vocabulary. Spelling used to return a single forced
best-guess via `.correction()`, which was wrong for "adress" (returned
"dress" over "address" because "dress" has higher raw corpus frequency:
92,448 vs 70,429). Fixed by offering the top 3 ranked candidates instead.

---

## 6. File overview

| File | Purpose |
|---|---|
| `app.py` | Flask API, all routes, feedback-card assembly, feature ablation. |
| `features.py` | Essay text to 17 numeric features. v2 -- see Section 3 above. |
| `train.py` | Trains and compares RidgeCV/GradientBoosting/SVR, saves best as model.pkl. |
| `relevance.py` | TF-IDF + keyword overlap between essay and prompt. Optional. |
| `coherence.py` | SBERT-based sentence coherence check. Read its docstring -- it documents two failed TF-IDF attempts before the current approach. |
| `db.py` | SQLite essay history (history.db, auto-created on first run). |
| `file_parser.py` | Extracts text from PDF/DOCX/TXT. Crash-proof -- tested against corrupted files, password-protected PDFs, embedded images. |
| `evaluate_models.py` | Compares baseline vs. transformer on held-out essays. Generates evaluation_report/ with CSV, charts, and a markdown report. |
| `test_coherence_sbert.py` | Standalone validation script for the SBERT coherence approach. Kept as a record of the experiment. Run it locally to reproduce the calibration numbers (needs internet to download the SBERT model). |
| `sample_data/` | sample_essays.csv (12 test essays) and asap_set1_rescaled.csv (1,783 real ASAP essays). |
| `history.db` | Auto-created SQLite database. Not in git -- add to .gitignore. |

---

## 7. Trials and errors worth knowing about

These are real things that went wrong during development. Each is
documented here so you can explain them in your defense rather than
being caught off guard.

**T1 -- Spellchecker flagging brand names as misspelled.**
pyspellchecker's bundled dictionary has no knowledge of brand names,
proper nouns, or contractions. "Instagram," "TikTok," "don't" were all
being flagged as errors. Fixed with: (a) a custom whitelist of ~60
commonly-missed words loaded into the spellchecker at import time, and
(b) two heuristics applied before checking -- skip ALL-CAPS words
(likely acronyms) and skip capitalized mid-sentence words (likely proper
nouns). Trade-off: a genuinely misspelled name like "Instagraam" now
slips through uncaught. That is a deliberate choice -- false positives
on correct words erode user trust in every other correct flag.

**T2 -- Spelling popup showing only one suggestion ("dress" for "adress").**
pyspellchecker's .correction() returns the single highest raw-frequency
candidate. "dress" (92,448 occurrences) outranked "address" (70,429)
purely on corpus frequency, even though "address" is the obviously
intended word. Fixed by switching to .candidates() and offering the top
3 ranked candidates instead of one forced pick.

**T3 -- Vocabulary popup silently broken (worked for spelling, not vocab).**
The escapeHtml() function did not escape double-quote characters. When
a suggestion array was JSON.stringify'd and embedded in a data-suggestions
HTML attribute, the embedded quote characters prematurely closed the
attribute, corrupting the tag. JSON.parse() then threw silently -- no
visible error, popup just did nothing. Fixed by adding quote escaping
to escapeHtml(). Confirmed by reproducing the exact truncation before
and after the fix.

**T4 -- Vocabulary feedback showing the same three words every time.**
The feedback card always suggested "beneficial, crucial, effective"
regardless of which weak words were actually in the essay. Root cause:
a fixed hardcoded list instead of a per-word synonym lookup. Fixed with
a real dictionary mapping each weak word to its own specific alternatives.

**T5 -- Transformer fallback silently crashing every request.**
The fallback logic checked that the transformer_model folder existed,
but did not catch OSError when weight files were absent (e.g. after
uploading to GitHub without model.safetensors to keep file size down).
Result: every /analyze request threw an unhandled 500. Fixed by wrapping
model loading in try/except OSError, logging a warning, and setting a
False sentinel to avoid retrying on every request.

**T6 -- Paragraph counting always returning 1.**
The paragraph split used "\n\n" (blank line). In a textarea, visual
line-wrapping does not insert real newline characters -- only actual
Enter keypresses do. A single Enter between paragraphs was not being
counted. A 3-paragraph essay was consistently reported as 1 paragraph.
Fixed by splitting on any real newline (\n+) instead of requiring a
blank line.

**T7 -- Coherence detection: two failed approaches before one that worked.**
Attempt 1 -- TF-IDF similarity between adjacent sentences. Failed:
good writing uses pronouns and synonyms instead of repeating nouns, so
a coherent essay and a deliberately disjointed one (random unrelated
sentences) both scored ~0.0 and were indistinguishable.
Attempt 2 -- TF-IDF similarity of each sentence to the essay's overall
topic vector. Also failed: at 5-10 sentences, TF-IDF overlap still
isn't a strong enough signal. Disjointed scored 0.447, coherent scored
0.473 -- nearly identical.
Attempt 3 -- SBERT semantic embeddings compared to essay centroid.
This one worked. Tested on the same calibration essays:
  Coherent: 0.697
  Weak but on-topic: 0.682
  One off-track sentence inserted: 0.595
  Fully disjointed: 0.515
One finding from the same test: using the single least-similar sentence
as an outlier detector was backwards (disjointed scored higher than
coherent on that metric). Only the average similarity is used.

**T8 -- baseline model had two dead-weight features.**
After training the v1 Ridge model, inspecting its coefficients revealed:
paragraph_count coefficient = +0.000 (zero effect)
avg_sentence_length coefficient = -0.025 (negligible)
These features were contributing nothing but were consuming model
capacity. Both were replaced in features.py v2.

**T9 -- train.py only ever tried one model type.**
The original train.py always used Ridge(alpha=1.0) with no
hyperparameter search and no comparison to other model types.
GradientBoosting and SVR are known to outperform linear models on
tabular feature sets like this one. train.py v2 tries all three and
saves whichever produces the lowest held-out MAE.

---

## 8. What is genuinely ML vs. rule-based

Be precise about this in your defense. Examiners do ask.

- **The score** is ML: the baseline Scikit-learn model or fine-tuned DistilBERT.
- **The feedback** (Grammar, Structure, Vocabulary, Coherence, Relevance)
  is rule-based and dictionary-driven. Spellchecker, weak-word lists,
  TF-IDF, sentence embeddings compared by cosine similarity. Not ML.
  This is a deliberate design choice: rule-based feedback is transparent,
  fast, and cannot hallucinate. Say this plainly if asked.