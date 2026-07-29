# Project Summary for Thesis Writing

## Automated Essay Scoring System Using Machine Learning (Grader's Assistant)

**Project Context:** A Web Based Application developed as a Final Year Project for the BCS program at COMSATS University Islamabad, Abbottabad Campus (Co-developed with Daud Afzal).

This document organizes the full project into sections that map onto typical thesis chapters. Everything here is drawn directly from what was actually built and tested. No invented numbers, no unverified claims. Where a figure is exact (e.g., an MAE or QWK score), it's from a real test run; where something is a design decision, the reasoning behind it is included so it can be explained in a viva without hesitation.

---

## 1. Introduction / Problem Statement Material

**Problem being addressed:** Manual essay grading is slow, inconsistent across graders, and doesn't scale. This project builds a Web Based Application that combines trained ML scoring models with transparent, rule-based feedback, giving both a numeric score and specific, actionable reasons for that score.

**Two scoring models, not one.** A deliberate design choice worth stating explicitly in the introduction:
- A **baseline model** (Gradient Boosting over 17 hand-crafted features), built first, fully validated, and kept in production as a highly stable, structurally grounded automatic fallback.
- A **fine-tuned transformer** (DistilBERT), built as an upgrade to process semantic meaning natively, once the baseline was proven to work end-to-end.

This "simple-first, then upgrade" sequencing (rooted in Agile/Scrum methodology) is itself a methodological point worth making: it de-risked the project by guaranteeing a working system existed before attempting the harder, less certain deep-learning component. Both scores are shown to the user side by side, with a switchable toggle, rather than presenting only one number as if it were the single truth. That transparency is a deliberate design choice too: see Section 6a for why the two models actually behave differently under customization.

---

## 2. Dataset & Preprocessing

**Source:** The ASAP-AES dataset (Automated Student Assessment Prize), the standard benchmark dataset in this field, released by the Hewlett Foundation via Kaggle in 2012.

**A real obstacle worth documenting in the methodology section:**
Kaggle's official competition page requires "joining" the competition to download the data. Since the competition closed years ago, Kaggle blocks new joins with a "late submission" error, blocking the download. This was resolved by sourcing the identical dataset from a verified public GitHub mirror (`hanshaoling/AES_app`), **verified authentic** by checking:
- Row count: 12,978 essays
- Column count: 28
- Per-essay-set counts matched the official published statistics exactly (1,783 / 1,800 / 1,726 / 1,772 / 1,805 / 1,800 / 1,569 / 723 across the 8 essay sets).

**Min-Max Normalization (Crucial Methodological Step):**
ASAP's 8 essay sets each have a wildly different prompt and original score range (e.g., 2–12, 1–6, 0–60). They cannot be naively combined without first rescaling each set independently. Combining them without this step would force the model to learn a meaningless mixed scale. A preprocessing pipeline independently applied min-max normalization to each set, projecting all scores onto a universal **0 to 100 scale** before training.

---

## 3. Baseline Model (Methodology + Results)

**Algorithm:** Gradient Boosting Regressor (selected automatically via a training script that cross-validates RidgeCV, GradientBoosting, and SVR algorithms to find the lowest error).

**Features engineered (17 total, in `features.py`):**
word count, sentence count, paragraph count, average sentence length, average word length, vocabulary richness, long-word ratio, misspelled word count, misspelled word ratio, weak-word count, sentence length variance, average paragraph length, type-token ratio (first 100 words), stopword ratio, commas per sentence, semicolons per sentence, question count, and exclamation count.

**Training setup:** 80/20 train/test split, `random_state=42`. Features standardized via `StandardScaler` before fitting.

**Evaluation Metrics:**
In Automated Essay Scoring (AES) literature, Mean Absolute Error (MAE) alone is insufficient. The models were evaluated using **MAE**, **RMSE**, and **Quadratic Weighted Kappa (QWK)**, which measures inter-rater agreement between the automated system and human ground-truth scores.

**Result:** On the benchmark Set 1 test, the Baseline achieved a **Mean Absolute Error (MAE) of 5.80 points** and an excellent **QWK of 0.8629**.

**Feature importance analysis:**
While the v2 model utilizes non-linear tree ensembles instead of simple linear coefficients, extracting feature importances corroborates that `word_count` remains the dominant predictive factor. Essays scoring 80+ in the training set had a median length of **475 words**; essays scoring below 50 had a median of just **171 words**. This raises the real question of whether AES models proxy "quality" for "effort/elaboration"—a well-known critique in literature. 

**How this finding shaped the product:** Rather than leaving this as a passive observation, a live length hint above the essay input box (see Section 6b) surfaces these exact thresholds to the user *while* they're writing. Second, word count intentionally does NOT appear as a toggleable evaluation category, as it is treated as an always-on part of scoring.

---

## 4. Transformer Model (Methodology)

**Model:** `distilbert-base-uncased`, fine-tuned with a regression head (`num_labels=1`, `problem_type="regression"`).

**Training environment:** Google Colab, free-tier T4 GPU.

**Data:** Scaled across **all 8 ASAP essay sets (~13,000 essays)** to teach the model how to grade persuasive, narrative, and source-dependent formats simultaneously.

**Score handling:** Labels rescaled to 0 to 1 for training stability, then rescaled back to 0 to 100 at prediction time.

**Results:** On Set 1 specifically, the Transformer achieved a 5.80 MAE and a slightly superior **QWK of 0.8645**, proving its semantic understanding aligns closely with human grading patterns. Across the entire 8-prompt generalized dataset, it achieved an MAE of ~9.97, demonstrating true cross-prompt generalization without overfitting to a single prompt type.

**Integration:** Both models run on every request. If the heavy `.safetensors` model files aren't present (e.g., excluded from a git upload), the system catches the `OSError`, logs a warning, and **falls back to the baseline automatically**. 

---

## 5. Feedback System, Six Categories

**Important architectural point for the defense:** the *score* comes from the ML models. The *feedback categories* are **rule-based / dictionary-driven, not ML**. This is a deliberate, defensible design choice. Rule-based feedback is transparent, fast, requires no additional training data, and cannot hallucinate.

| Category | Technique | What it checks |
|---|---|---|
| **Grammar** | `pyspellchecker` plus custom whitelist and heuristics | Spelling errors, providing the top-3 ranked Levenshtein distance candidates. |
| **Structure** | Sentence/paragraph counting | Whether the essay is developed across multiple paragraphs |
| **Vocabulary** | Weak-word dictionary plus per-word synonym map | Overused low-value words (e.g. "good", "very"), each with 3 specific alternatives |
| **Coherence** | Semantic sentence embeddings (SBERT) | Whether the essay's sentences stay topically consistent throughout |
| **Relevance** | TF-IDF cosine similarity plus keyword overlap | Whether the essay addresses a given prompt/topic (optional) |

**Customizable evaluation criteria:** users can toggle which of these five categories are returned per request. 

---

## 6a. Customization That Actually Changes the Score: Feature Ablation

An earlier version of the customization feature only changed which feedback cards were displayed, while the score itself stayed identical. This was a real inconsistency.

**Fixing this required a genuine, tested technique:**
When a category is disabled, its underlying features are replaced with their **training-set mean** before the baseline model scores the essay. `StandardScaler` conveniently stores these means directly (`scaler.mean_`), so no retraining was needed. Because a standardized feature sitting exactly at the training mean contributes approximately zero to the prediction, that feature's influence is mathematically cancelled out of the score.

**An honest, important limitation:** This only works for the Baseline model. The Transformer is a black box operating directly on essay text tokens. There is no equivalent "zero out this named feature" operation available. The transformer's score stays based on the full, unmodified essay regardless of which categories are toggled. This is a real, defensible distinction between the two architectures.

---

## 6b. Length: From Feedback Card to Live Writing Hint

Length was initially implemented as a sixth feedback category. It was later deliberately removed from the feedback-card system and replaced with a different UI treatment: a small hint directly above the essay input box that updates live, on every keystroke, with no API call required.

**Why this is a legitimate design decision:** Length isn't a "quality" category in the same sense as grammar or vocabulary; it's a structural property the user can act on immediately while writing. Surfacing it as a live, always-visible hint matches how the information is actually useful: as guidance during writing, not as a post-submission verdict.

---

## 7. The Coherence Detection Journey

This section demonstrates genuine iterative, evidence-based engineering rather than a single lucky guess. **Three approaches were tried empirically:**

**Attempt 1: TF-IDF similarity between adjacent sentences.**
**Result: failed.** Well-written text deliberately uses pronouns and synonyms instead of repeating nouns, so a genuinely coherent essay and a deliberately disjointed one both scored close to 0.0 and were indistinguishable.

**Attempt 2: TF-IDF similarity of each sentence to the essay's overall topic vector.**
**Result: also failed.** At typical essay length, TF-IDF word-overlap still isn't a strong enough signal. The disjointed test essay scored 0.447, nearly identical to the coherent essay's 0.473.

**Attempt 3: Semantic sentence embeddings (SBERT, `all-MiniLM-L6-v2`), compared to the essay's centroid.**
**Result: worked.** Tested on calibration essays:
*   Coherent: 0.697
*   Weak but on-topic: 0.682
*   Mixed (one off-track sentence inserted): 0.595
*   Disjointed (unrelated topics): 0.515

This produced a sensible, monotonic ordering because semantic embeddings capture *meaning*, not just literal word overlap. Threshold set at 0.60.

---

## 8. Known Limitations

Presenting limitations honestly is a sign of a well-understood system:
- **Coherence and Relevance are proxies.** Coherence measures semantic topic-consistency; Relevance measures keyword/topic overlap. Neither evaluates whether an argument is *logically* sound.
- **Relevance checking false-negatives:** A genuinely on-topic essay that paraphrases the prompt's wording (e.g. using "adolescents" instead of "teenagers") can score as low-overlap, since TF-IDF matches vocabulary, not meaning. 
- **The spelling checker's proper-noun heuristic trades precision for recall.** To avoid flagging brand names, capitalized non-sentence-initial words are skipped. A genuinely misspelled name (e.g. "Instagraam") would slip through uncaught—a deliberate choice to preserve user trust over annoying false positives.
- **Transformer Ablation:** Category-based score customization only applies to the baseline model (see Section 6a). 

---

## 9. Debugging Narratives Worth Including

These are genuine bugs found and fixed during development, each with a root cause worth explaining:

- **Infinite Training Bottleneck (Fast Mode):** During bulk multi-model training over thousands of essays, the `pyspellchecker` candidate generation caused the terminal to hang indefinitely due to heavy Levenshtein distance computations. Fixed by implementing a `fast_mode=True` bypass flag in the extraction pipeline, dropping feature extraction time for 1,783 essays to just 10.5 seconds without impacting the final UI.
- **Silent request failures:** The transformer fallback logic checked that the model folder existed, but never caught the case where the folder exists but weight files are missing, causing every `/analyze` request to crash with an unhandled 500 error. Fixed by wrapping model loading in a `try/except OSError` and falling back to the baseline.
- **A hardcoded, non-varying feedback bug:** The Vocabulary feedback card always suggested the same three words ("beneficial, crucial, effective") regardless of which weak words were actually present. Fixed with a real per-word synonym dictionary.
- **A single-suggestion spelling limitation:** The original `.correction()` method returned only its single highest-frequency guess, heavily penalizing context (e.g. suggesting "dress" for "adress" over "address"). Fixed by extracting the top 3 ranked candidates.
- **A silently broken click-to-fix editor:** The interactive popup for fixing flagged words broke on vocabulary suggestions. Root cause: an HTML-escaping function didn't escape quote characters, so a JSON array embedded in an HTML attribute prematurely closed the tag. `JSON.parse()` then threw silently. Fixed by escaping quotes properly.
- **A paragraph-counting bug:** Essay structure feedback split text on a blank line (`\n\n`), but web `<textarea>` inputs often only insert a single `\n`. A 3-paragraph essay was reported as 1 paragraph. Fixed by splitting on any real newline (`\n+`).

---

## 10. System Architecture

**Frontend:** HTML/CSS/JavaScript (Vanilla). Essay input with a live-highlighting overlay for flagged words, active visual linking between feedback cards, a switchable score toggle, and an interactive `Ctrl+Enter` execution shortcut.

**Backend:** Flask (Python), REST API with these key endpoints:
- `POST /analyze`: main scoring and feedback endpoint
- `POST /extract-text`: PDF/DOCX/TXT upload and text extraction
- `GET/POST/DELETE /history`: essay history, backed by a real database

**Database:** SQLite (`db.py`), storing essay text, both model scores, structured feedback JSON, and timestamps.

**File upload handling:** `pypdf` and `python-docx`, built with explicit crash-prevention. Tested against corrupted files, password-protected PDFs, and documents containing embedded images (images are silently skipped with a user-facing notice).

---

## 11. Suggested Future Work

- Trait-level scoring (separate scores for coherence, argumentation, grammar, etc., rather than one overall number).
- A supervised or self-supervised coherence model (e.g. training a classifier to distinguish original sentence order from randomly shuffled order, bootstrapped from existing essay data) as a more rigorous alternative to the current embedding-similarity proxy.
- An equivalent ablation-style customization mechanism for the transformer, likely requiring a fundamentally different technique than the baseline's feature-mean substitution.
- Full LTI-based LMS integration to allow the Web Based Application to plug directly into university ecosystems like Canvas or Moodle.

---

## Quick-Reference Fact Sheet (For Defense Preparation)

- **Dataset:** Kaggle ASAP-AES (~13,000 real student essays, 8 prompts).
- **Baseline Model:** Gradient Boosting Regressor, 17 hand-crafted NLP features, MAE: 5.80, QWK: 0.8629.
- **Transformer Model:** DistilBERT, Set 1 MAE: 5.80, QWK: 0.8645. Generalized MAE: 9.97 (All 8 sets).
- **Dominant Baseline Feature:** `word_count`
- **Coherence Detection:** 3 iterations tested; SBERT-based approach adopted after TF-IDF failed twice.
- **Feedback Generation:** Rule-based / Dictionary-driven (Not ML).
- **Customization:** Category exclusion genuinely changes the baseline score via feature-mean ablation; transformer score is unaffected by category toggles (architectural limitation).