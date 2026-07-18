# Marginal — Backend

Flask API that scores an essay and returns structured feedback, using
either a baseline Scikit-learn model or a fine-tuned DistilBERT
transformer (both are computed and returned together — see below).

## 1. Setup

```bash
cd backend
pip install -r requirements.txt --break-system-packages
```
(Drop `--break-system-packages` if you're using a virtual environment.)

**First run note:** `sentence-transformers` (used for the Coherence
check) downloads a small pretrained model (~80MB) from Hugging Face
the first time the server starts, and caches it locally after that —
needs normal internet access on that first run.

## 2. The two models

Both are already trained and included — nothing to do before running
the app.

- **Baseline** (`model.pkl`): Ridge regression over hand-crafted
  features (length, spelling, structure, vocabulary). Trained on 1,783
  real ASAP essays. Real held-out MAE: **6.49** (0–100 scale).
- **Transformer** (`transformer_model/`): fine-tuned DistilBERT,
  trained via the Colab notebook (`../colab/finetune_essay_scorer.ipynb`).

`/analyze` computes **both** scores every time and returns both
(`baseline_score`, `transformer_score`) — the frontend shows the
transformer's score as primary when it's available, with the baseline
alongside it, and falls back to baseline-only if the transformer model
folder is missing or its weight files weren't copied in (e.g. left out
of a git upload to keep the repo size down — this is handled gracefully,
not a crash, with a warning logged server-side).

To retrain the baseline: `python3 train.py sample_data/asap_set1_rescaled.csv`
To retrain the transformer: see `../colab/finetune_essay_scorer.ipynb`
and `../colab/TRANSFORMER_INTEGRATION.md`.

## 3. Run the API

```bash
python3 app.py
```
Runs at `http://127.0.0.1:5000`. On first run this also creates
`history.db` (SQLite) automatically if it doesn't exist yet.

## 4. Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/analyze` | POST | Main endpoint. Takes `{"essay": "...", "prompt": "...", "enabled_categories": [...]}` (only `essay` is required). Returns score, both model scores, and feedback cards. |
| `/extract-text` | POST | Upload a `.pdf`/`.docx`/`.txt` file (`multipart/form-data`, field `file`), get back extracted plain text. Handles corrupted files, password-protected PDFs, and embedded images gracefully — never crashes, always returns a clear error message on bad input. |
| `/history` | GET | Returns the 10 most recent saved analyses from the database. |
| `/history` | POST | Saves a completed analysis (called by the frontend right after a successful `/analyze` — reuses that response, no recomputation). |
| `/history/<id>` | DELETE | Deletes one saved entry. |
| `/history` | DELETE | Clears all history. |
| `/health` | GET | Simple liveness check. |

### `/analyze` request body
```json
{
  "essay": "required, at least 30 words",
  "prompt": "optional -- enables the Relevance feedback category if provided",
  "enabled_categories": ["Grammar", "Structure", "Vocabulary", "Coherence", "Relevance"]
}
```
`enabled_categories` is optional and defaults to all five — this is
the "customizable evaluation criteria" feature: it only filters which
feedback cards come back, it does **not** change the score itself
(the score always comes from the ML model regardless of which
categories are enabled).

### `/analyze` response shape
```json
{
  "score": 78,
  "baseline_score": 72,
  "transformer_score": 78,
  "summary": "Solid draft, a few things to tighten",
  "feedback": [
    {"category": "Grammar", "status": "good|warn", "label": "...", "note": "..."},
    ...
  ],
  "stats": {"word_count": 217, "sentence_count": 10},
  "issues": {
    "spelling": [{"word": "adress", "suggestions": ["dress", "address"]}],
    "weak_words": [{"word": "good", "suggestions": ["beneficial", "effective", "valuable"]}]
  }
}
```
`issues` feeds the click-to-fix editor in the frontend (clicking a
flagged word in the essay shows real suggestion chips, sourced from
this same data). Note `suggestions` is a **list** for both spelling and
vocabulary now — spelling used to return a single forced "best guess"
(`.correction()`), which was sometimes wrong (e.g. "adress" → "dress"
outranked the correct "address" on raw word frequency); offering
multiple ranked candidates fixed this.

## 5. File overview

| File | Purpose |
|---|---|
| `app.py` | Flask API — all routes, feedback-card assembly. |
| `features.py` | Essay text → numeric features (length, spelling, vocabulary, structure). Includes a whitelist + heuristics to reduce spellcheck false positives on brand names/proper nouns/contractions. |
| `relevance.py` | Checks essay-vs-prompt topical overlap (TF-IDF + keyword coverage). Optional — only runs if a prompt is provided. |
| `coherence.py` | Checks whether the essay stays on-topic sentence-to-sentence, using semantic sentence embeddings (SBERT). **Read this file's docstring** — it documents two earlier approaches (TF-IDF based) that were tried and failed real testing before this one. |
| `db.py` | SQLite storage for essay history (`history.db`, auto-created). |
| `file_parser.py` | Extracts text from uploaded `.pdf`/`.docx`/`.txt`, with real crash-prevention (tested against corrupted files, embedded images, password-protected PDFs). |
| `train.py` | Trains the baseline Ridge regression model, saves `model.pkl`. |
| `evaluate_models.py` | Compares baseline vs. transformer on the same held-out essays — generates `evaluation_report/` (CSV + charts + a markdown report) for your FYP write-up. |
| `test_coherence_sbert.py` | Standalone script used to validate the SBERT coherence approach before it was built into `coherence.py` — kept as a record of that experiment. |
| `sample_data/` | `sample_essays.csv` (12 tiny test essays) and `asap_set1_rescaled.csv` (1,783 real ASAP essays, rescaled to 0–100). |
| `history.db` | SQLite database, auto-created on first run. Not included in the repo (it's your local data) — add it to `.gitignore` if it isn't already. |

## 6. What's genuinely ML vs. rule-based — be precise about this in your defense

- **The score** is ML: baseline Ridge regression, or fine-tuned DistilBERT.
- **The feedback** (Grammar, Structure, Vocabulary, Coherence, Relevance)
  is **not** ML — it's rule-based/dictionary-driven (spellchecker,
  weak-word lists, TF-IDF, sentence embeddings compared by cosine
  similarity). This is a deliberate, defensible design choice — rule-based
  feedback is transparent, fast, and doesn't hallucinate, unlike asking
  a model to generate free-text feedback. Say this plainly if asked;
  don't let the polished UI imply more ML than is actually there.