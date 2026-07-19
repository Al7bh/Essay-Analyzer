# Project Summary for Thesis Writing
## Automated Essay Scoring System Using Machine Learning ("Marginal")

This document organizes the full project into sections that map onto
typical thesis chapters. Everything here is drawn directly from what
was actually built and tested. No invented numbers, no unverified
claims. Where a figure is exact (e.g. an MAE score), it's from a real
test run; where something is a design decision, the reasoning behind
it is included so you can explain it in a viva without hesitation.

---

## 1. Introduction / Problem Statement material

**Problem being addressed:** Manual essay grading is slow, inconsistent
across graders, and doesn't scale. This project builds an automated
essay scoring system that combines a trained ML scoring model with
transparent, rule-based feedback, giving both a numeric score and
specific, actionable reasons for that score.

**Two scoring models, not one.** A deliberate design choice worth
stating explicitly in your introduction:
- A **baseline model** (Ridge regression over hand-crafted features),
  built first, fully validated, and kept in production as an automatic
  fallback.
- A **fine-tuned transformer** (DistilBERT), built as an upgrade once
  the baseline was proven to work end-to-end.

This "simple-first, then upgrade" sequencing is itself a methodological
point worth making: it de-risked the project by guaranteeing a working
system existed before attempting the harder, less certain deep-learning
component. Both scores are shown to the user side by side, with a
switchable toggle, rather than presenting only one number as if it were
the single truth. That transparency is a deliberate design choice too:
see Section 6a for why the two models actually behave differently under
customization.

---

## 2. Dataset

**Source:** The ASAP-AES dataset (Automated Student Assessment Prize,
Automated Essay Scoring), the standard benchmark dataset in this field,
released by the Hewlett Foundation via Kaggle in 2012.

**A real obstacle worth documenting in your methodology section:**
Kaggle's official competition page requires "joining" the competition
to download the data. Since the competition closed years ago, Kaggle
sometimes blocks new joins with a "late submission" error, which also
blocks the download. This was resolved by sourcing the identical
dataset from a public GitHub mirror (`hanshaoling/AES_app`), **verified
authentic** by checking:
- Row count: 12,978 essays
- Column count: 28
- Per-essay-set counts matched the official published statistics exactly
  (1,783 / 1,800 / 1,726 / 1,772 / 1,805 / 1,800 / 1,569 / 723 across
  the 8 essay sets)

**Subset used:** Essay Set 1, 1,783 real student essays, persuasive
writing. Scores were originally on a 2 to 12 scale and were **rescaled
to 0 to 100** (min-max normalization) to give a consistent,
interpretable scale across the whole system.

**Note on ASAP's structure, worth mentioning as a methodological
awareness point:** ASAP's 8 essay sets each have a different prompt and
a different original score range. They cannot be naively combined
without first rescaling each set independently. Combining them without
this step would make the model learn a meaningless mixed scale.

---

## 3. Baseline Model (Methodology + Results)

**Algorithm:** Ridge regression (L2-regularized linear regression),
via Scikit-learn.

**Features engineered (10 total, in `features.py`):**
word count, sentence count, paragraph count, average sentence length,
average word length, vocabulary richness (unique words divided by total
words), long-word ratio, misspelled word count, misspelled word ratio,
overused/weak-word count.

**Training setup:** 80/20 train/test split, `random_state=42` for
reproducibility. Features standardized via `StandardScaler` before
fitting.

**Result: Mean Absolute Error (MAE) of 6.49 points** on a 0 to 100
scale, on the held-out test set. This is the honest, reproducible
baseline number for your results chapter.

**Feature importance analysis (a genuine finding worth its own
subsection):** Inspecting the trained Ridge model's coefficients reveals
which features actually drive the score:

| Feature | Coefficient | Interpretation |
|---|---|---|
| word_count | **+10.068** | By far the dominant factor |
| misspelled_ratio | -2.521 | |
| long_word_ratio | +2.315 | |
| misspelled_count | +2.206 | |
| sentence_count | +1.611 | |
| weak_word_count | -0.730 | |
| avg_word_length | +0.639 | |
| vocab_richness | +0.538 | |
| avg_sentence_length | -0.025 | negligible |
| paragraph_count | +0.000 | no measurable effect |

**Corroborating evidence from the raw data itself:** essays scoring
80+ in the training set had a median length of **475 words**; essays
scoring below 50 had a median of just **171 words**. This is a genuine,
data-grounded finding: essay length is the single strongest predictor
of score in this model, considerably stronger than grammar or
vocabulary features. This is worth an honest discussion in your results
chapter. It raises the real question of whether the model is measuring
"quality" or largely proxying for "effort/elaboration," which is a
legitimate, well-known critique in the AES literature generally, not
unique to this implementation.

**How this finding directly shaped the product, not just the report:**
rather than leaving this as a passive observation, it was used twice
in the actual system. First, a live length hint above the essay input
box (see Section 6b) surfaces these exact thresholds to the user while
they're writing, not as a verdict after the fact. Second, it's why word
count intentionally does NOT appear as a toggleable evaluation category
(see Section 6a): it isn't tied to any single "quality" category the
way grammar or vocabulary is, so it's treated as an always-on part of
scoring, with its own dedicated communication channel instead.

---

## 4. Transformer Model (Methodology)

**Model:** `distilbert-base-uncased`, fine-tuned with a regression head
(`num_labels=1`, `problem_type="regression"`).

**Training environment:** Google Colab, free-tier T4 GPU.

**Data:** Same essay set 1 data as the baseline (1,783 essays, same
80/20 split, `random_state=42`), deliberately kept identical to the
baseline's data and split so the two models' MAE scores are a fair,
apples-to-apples comparison, not confounded by different training data.

**Score handling:** Labels rescaled to 0 to 1 for training stability,
then rescaled back to 0 to 100 at prediction time.

**Integration:** Both models run on every request; the API returns
`baseline_score` and `transformer_score` together, with the transformer
used as the "headline" score when available. If the transformer's model
files aren't present or fail to load (e.g. weight files intentionally
excluded from a size-constrained git upload), the system logs a warning
and **falls back to the baseline automatically**. This fallback
behavior was itself a bug that was found and fixed during development
(see Section 9, Debugging Narratives).

---

## 5. Feedback System, Six Categories

**Important architectural point for your defense:** the *score* comes
from the ML models above. The *feedback categories* below are
**rule-based / dictionary-driven, not ML**, a deliberate, defensible
design choice, not a shortcut. Rule-based feedback is transparent, fast,
requires no additional training data, and cannot hallucinate, all
properties that matter for a tool giving students actionable writing
advice. Be precise about this distinction if asked in your viva.

| Category | Technique | What it checks |
|---|---|---|
| **Grammar** | `pyspellchecker` plus custom whitelist and heuristics | Spelling errors, with up to 3 ranked correction candidates per word |
| **Structure** | Sentence/paragraph counting | Whether the essay is developed across multiple paragraphs |
| **Vocabulary** | Weak-word dictionary plus per-word synonym map | Overused low-value words (e.g. "good", "very"), each with 3 specific alternatives |
| **Coherence** | Semantic sentence embeddings (SBERT) | Whether the essay's sentences stay topically consistent throughout |
| **Relevance** | TF-IDF cosine similarity plus keyword overlap | Whether the essay addresses a given prompt/topic (optional, only runs if a prompt is supplied) |

Note that Length is deliberately not in this table. See Section 6b for
why it was removed from the feedback-card system entirely and replaced
with a different kind of UI element.

**Customizable evaluation criteria:** users can toggle which of these
five categories are returned per request. This satisfies a
"customizable evaluation criteria" requirement without needing a full
rubric-builder.

---

## 6a. Customization That Actually Changes the Score: Feature Ablation

An earlier version of the customization feature only changed which
feedback cards were displayed, while the score itself stayed identical
regardless of what was toggled off. This was identified as a real
inconsistency: if a user turns off "Vocabulary," they reasonably expect
the score to no longer reflect vocabulary at all, not just to stop
seeing a card about it.

**Fixing this required a genuine, tested technique, not a UI trick.**
When a category is disabled, its underlying features are replaced with
their **training-set mean** before the baseline model scores the essay,
rather than their real values. `StandardScaler` (already used to
normalize features before training) conveniently stores these means
directly (`scaler.mean_`), so no retraining was needed. This is a
legitimate ablation technique for a linear model: a standardized
feature sitting exactly at the training mean contributes approximately
zero to the prediction, since `scaled_value * coefficient` becomes
`0 * coefficient`. Effectively, that feature's influence is
mathematically cancelled out of the score.

Category-to-feature mapping used:

| Category | Features ablated when disabled |
|---|---|
| Grammar | misspelled_count, misspelled_ratio |
| Structure | sentence_count, paragraph_count, avg_sentence_length |
| Vocabulary | vocab_richness, long_word_ratio, weak_word_count, avg_word_length |

**Verified, not assumed:** on a deliberately weak-vocabulary test essay,
the baseline score moved from 31 to 35 when Vocabulary was excluded,
confirmed both in isolated testing and through the live API route.

**An honest, important limitation for your discussion chapter: this
only works for the baseline model.** The transformer is a black box
operating directly on essay text tokens. There is no equivalent
"zero out this named feature" operation available for it the way there
is for an interpretable linear model. So the transformer's score stays
based on the full, unmodified essay regardless of which categories are
toggled. This is a real, defensible distinction between the two models'
architectures, not a bug, and it's part of why both scores are shown
side by side via a switchable toggle in the UI (Transformer / Baseline
buttons) rather than only one number being surfaced: it makes this
architectural difference visible and honest, instead of hidden behind
a single blended number.

---

## 6b. Length: From Feedback Card to Live Writing Hint

Length was initially implemented as a sixth feedback category, using
the same real thresholds described in Section 3 (400+ words: well
developed; 250-399: reasonable; under 250: short). It was later
deliberately removed from the feedback-card and customization system
and replaced with a different UI treatment: a small hint directly above
the essay input box that updates live, on every keystroke, with no
API call required (word count is already computed client-side).

**Why this change, and why it's a legitimate design decision to
describe in your thesis, not just a cosmetic tweak:** length isn't a
"quality" category in the same sense as grammar or vocabulary; it's a
structural property the user can act on immediately while writing,
rather than a verdict delivered after submission. Surfacing it as a
live, always-visible hint (rather than a post-hoc card, and rather than
a toggleable evaluation criterion the way Grammar or Vocabulary are)
matches how the information is actually useful: as guidance during
writing, not as one score component among several to be turned on or
off. word_count remains part of the baseline model's actual scoring at
all times; only how this fact is communicated to the user changed.

---

## 7. The Coherence Detection Journey, a strong methodology narrative

This is worth writing up as its own subsection in your methodology
chapter, since it demonstrates genuine iterative, evidence-based
engineering rather than a single lucky guess. **Three approaches were
tried, in order, each one tested empirically before being kept or
discarded:**

**Attempt 1: TF-IDF similarity between adjacent sentences.**
Hypothesis: coherent writing has high word-overlap between consecutive
sentences. **Result: failed.** Well-written text deliberately uses
pronouns and synonyms instead of repeating nouns, so a genuinely
coherent essay and a deliberately disjointed one (unrelated sentences
about social media, pizza, weather, basketball) both scored close to
0.0 and were indistinguishable.

**Attempt 2: TF-IDF similarity of each sentence to the essay's overall
topic vector.** Hypothesis: an off-topic sentence would stand out
against the whole essay's vocabulary, even if adjacent-sentence
similarity fails. **Result: also failed.** At typical essay length
(5 to 10 sentences), TF-IDF word-overlap still isn't a strong enough
signal. The disjointed test essay scored 0.447, nearly identical to
the coherent essay's 0.473.

**Attempt 3: Semantic sentence embeddings (SBERT, `all-MiniLM-L6-v2`),
compared to the essay's centroid.** **Result: worked.** Tested on the
same calibration essays:

| Essay type | avg. similarity |
|---|---|
| Coherent | 0.697 |
| Weak but on-topic | 0.682 |
| Mixed (one off-track sentence inserted) | 0.595 |
| Disjointed (unrelated topics) | 0.515 |

This produced a sensible, monotonic ordering that neither TF-IDF
attempt could achieve, because semantic embeddings capture *meaning*,
not just literal word overlap. Threshold set at 0.60, chosen to sit
between the "mixed" and "weak-but-on-topic" scores, so a genuinely
off-track sentence gets flagged while merely simple/plain writing does
not. One caveat found in the same testing: using the single
*least-similar* sentence as an outlier detector did **not** work; it
was backwards, with the disjointed essay scoring higher than the
coherent one on that specific metric. So only the *average* similarity
is used, not a per-sentence outlier score.

**Why this narrative matters for your thesis:** it's a legitimate,
citable example of hypothesis, test, reject, iterate, exactly the kind
of rigor a methodology chapter should demonstrate. A system that worked
on the first try tells the reader less about your process than one
where you can show what didn't work and why.

---

## 8. Known Limitations, be upfront about these in your discussion chapter

Presenting limitations honestly is a sign of a well-understood system,
not a weak one. These are real, specific, and each has a reason:

- **Coherence and Relevance are proxies, not true understanding.**
  Coherence measures semantic topic-consistency; Relevance measures
  keyword/topic overlap with a prompt. Neither evaluates whether an
  argument is *logically* sound.
- **Relevance checking has a known false-negative case:** a genuinely
  on-topic essay that paraphrases the prompt's wording (e.g. using
  "adolescents" instead of "teenagers") can score as low-overlap, since
  TF-IDF matches vocabulary, not meaning. Tested and documented directly.
- **The spelling checker's proper-noun heuristic trades precision for
  recall.** To avoid flagging brand names and proper nouns as
  misspelled, capitalized non-sentence-initial words are skipped from
  spellchecking entirely, meaning a genuinely misspelled name (e.g.
  "Instagraam") would now slip through uncaught. This was a deliberate
  choice: false "this is wrong" flags on correct words erode trust in
  every other, correct, flag more than occasionally missing one real
  mistake.
- **The baseline model may be substantially rewarding essay length**
  over other qualities (see Section 3's feature-importance findings), a
  legitimate, known critique applicable to many AES systems, not unique
  to this one.
- **Category-based score customization only applies to the baseline
  model** (see Section 6a). The transformer's score cannot be cleanly
  adjusted per category, since it has no equivalent named-feature
  structure to ablate.
- **Training data is a single ASAP essay prompt (1,783 essays).** A
  larger, multi-prompt dataset would likely generalize better and is
  the most direct lever for improving both models further.

---

## 9. Debugging Narratives Worth Including

A thesis that only shows the final, working system misses a chance to
demonstrate real engineering process. These are genuine bugs found and
fixed during development, each with a root cause worth explaining:

- **Silent request failures:** the transformer fallback logic checked
  that the model *folder* existed, but never caught the case where the
  folder exists but weight files are missing, causing every `/analyze`
  request to crash with an unhandled 500 error. Fixed by wrapping model
  loading in a try/except and falling back to the baseline with a
  logged warning.
- **A hardcoded, non-varying feedback bug:** the Vocabulary feedback
  card always suggested the same three words ("beneficial, crucial,
  effective") regardless of which weak words were actually present in
  the essay. Root cause: a fixed list was hardcoded instead of a
  per-word suggestion lookup. Fixed with a real per-word synonym
  dictionary.
- **A single-suggestion spelling limitation:** the spellchecker's
  `.correction()` method returns only its single highest-frequency
  guess, which isn't always contextually correct (e.g. "adress" to
  "dress" outranked the correct "address" purely on raw corpus
  frequency: 92,448 vs. 70,429 occurrences). Fixed by offering the top
  3 ranked candidates instead of one forced pick.
- **A silently broken click-to-fix editor:** an interactive popup for
  fixing flagged words worked for spelling but not vocabulary
  suggestions. Root cause: an HTML-escaping function didn't escape
  quote characters, so a JSON array embedded in an HTML attribute had
  its quotes prematurely close the attribute, corrupting the tag.
  `JSON.parse()` then threw silently with no visible error. Fixed by
  escaping quotes properly, confirmed with a reproducible before/after
  test.
- **A paragraph-counting bug:** essay structure feedback split text on
  a *blank line* (double newline) to detect paragraph breaks, but a
  `<textarea>` doesn't insert real newline characters for visual
  line-wrapping; only actual Enter keypresses do. A single Enter
  between paragraphs (very common) wasn't being counted, so a genuinely
  3-paragraph essay was reported as 1 paragraph. Fixed by splitting on
  any real newline instead of requiring two.
- **A misleading, contradictory-looking result:** an essay could show
  every feedback category as "good" (clean grammar, strong structure,
  varied vocabulary, well-connected) while still receiving a moderate
  overall score, which looked like a bug. Investigation traced this to
  a real, legitimate cause rather than an error: the essay was short
  (193 words) relative to what the model strongly associates with
  higher scores (see Section 3), but nothing in the feedback system
  communicated this at the time. This directly motivated the live
  length hint described in Section 6b.

---

## 10. System Architecture (for your design chapter)

**Frontend:** HTML/CSS/JavaScript (no framework). Essay input with a
live-highlighting overlay for flagged words, a results panel with a
switchable score toggle (Transformer / Baseline), essay history, file
upload, and a click-to-fix interactive editor.

**Backend:** Flask (Python), REST API with these key endpoints:
- `POST /analyze`: main scoring and feedback endpoint
- `POST /extract-text`: PDF/DOCX/TXT upload and text extraction
- `GET/POST/DELETE /history`: essay history, backed by a real database

**Database:** SQLite (`db.py`), storing essay text, both model scores,
feedback, and timestamps: a genuine persistence layer, not
browser-only storage.

**File upload handling:** `pypdf` (PDF) and `python-docx` (Word),
built with explicit crash-prevention, tested against corrupted files,
password-protected PDFs, and documents containing embedded images
(images are silently skipped from text extraction, with a user-facing
notice of how many were skipped, rather than causing an error).

---

## 11. Suggested Future Work section

- Trait-level scoring (separate scores for coherence, argumentation,
  grammar, etc., rather than one overall number)
- Training on the full ASAP dataset (all 8 sets, properly rescaled) or
  a second dataset entirely, to test generalization beyond one prompt
- A supervised or self-supervised coherence model (e.g. training a
  classifier to distinguish original sentence order from randomly
  shuffled order, bootstrapped from existing essay data) as a more
  rigorous alternative to the current embedding-similarity proxy
- An equivalent ablation-style customization mechanism for the
  transformer, likely requiring a fundamentally different technique
  than the baseline's feature-mean substitution, since the transformer
  has no named-feature structure to work with
- Full LTI-based LMS integration (explicitly scoped out of this project
  due to its scale; LTI is a substantial protocol implementation, not
  a short add-on)

---

## Quick-reference fact sheet (for figures/tables in your thesis)

- Dataset: ASAP-AES, essay set 1, 1,783 real student essays
- Baseline MAE: 6.49 (0 to 100 scale), Ridge regression, 10 hand-crafted features
- Dominant scoring feature: word_count (coefficient +10.07, roughly 4 to 5x any other feature)
- Coherence detection: 3 iterations tested; SBERT-based approach adopted after TF-IDF failed twice
- Feedback categories: 5 toggleable (Grammar, Structure, Vocabulary, Coherence, Relevance), plus Length as a separate live hint
- Score computation: ML-based (2 models). Feedback: rule-based (not ML)
- Customization: category exclusion genuinely changes the baseline score via feature-mean ablation; transformer score is unaffected by category toggles (architectural limitation, not a bug)