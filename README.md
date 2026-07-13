# Marginal — Backend (Phase 2)

Flask API that scores an essay and returns feedback, using a baseline
Scikit-learn model (Ridge regression over hand-crafted text features).
No AI/transformer model yet — that's the Phase 3 upgrade.

## 1. Setup

```bash
cd backend
pip install -r requirements.txt --break-system-packages
```

(Drop `--break-system-packages` if you're using a virtual environment,
which is generally the cleaner approach — see below.)

**Recommended: use a virtual environment** so this doesn't clash with
other Python projects on your machine:
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Train the model

The model is **already trained on real ASAP data** (1,783 real student
essays from essay set 1) and included as `model.pkl` — you can skip
straight to step 4 and it'll work immediately.

To retrain it yourself:
```bash
python3 train.py sample_data/asap_set1_rescaled.csv
```
Real result: **Mean Absolute Error of 6.49 points** on a 0–100 scale,
using a proper 80/20 train/test split. That's the honest baseline
number to quote in your report — a simple Ridge regression using only
hand-crafted features (length, spelling, structure, vocabulary), no
deep learning.

`sample_data/sample_essays.csv` (12 tiny hand-written essays) is still
included too, purely as a fast smoke-test for the pipeline.

## 3. Getting more of the ASAP dataset

**Kaggle's competition page requires "joining" the competition to
download** — and since it's long closed, Kaggle sometimes blocks new
joins with a "late submission" message, which stops the download too.

**Working alternative — no login needed:**
```
https://raw.githubusercontent.com/hanshaoling/AES_app/main/training_set_rel3.xls
```
This is the *exact* original dataset (verified: 12,978 essays, 28
columns, matches the official per-set counts exactly), just mirrored
on GitHub by another student project. Download it directly — no
account, no competition join.

Once downloaded, convert it to what `train.py` expects (this is
exactly what produced `asap_set1_rescaled.csv`):
```python
import pandas as pd

df = pd.read_excel("training_set_rel3.xls")
df = df[df["essay_set"] == 1][["essay_id", "essay", "domain1_score"]].dropna()
df = df.rename(columns={"domain1_score": "score"})

min_s, max_s = df["score"].min(), df["score"].max()
df["score"] = ((df["score"] - min_s) / (max_s - min_s)) * 100

df[["essay", "score"]].to_csv("asap_set1_rescaled.csv", index=False)
```

To use a different essay set, or combine several, change the
`essay_set == 1` filter — just remember each set has its own score
range, so rescale each set separately *before* combining them, or the
model will learn nonsense (a "9" means something different in set 1
vs. set 7).

**Note:** `xls` reading needs the `xlrd` package —
`pip install xlrd --break-system-packages` if you don't already have it.


## 4. Run the API

```bash
python3 app.py
```

Server runs at `http://127.0.0.1:5000`. Test it:
```bash
curl -X POST http://127.0.0.1:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"essay": "your essay text here, at least 30 words..."}'
```

## 5. Connect the frontend

In `index.html`, replace the mock click handler with a real fetch call:

```javascript
analyzeBtn.addEventListener('click', async () => {
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'Analyzing…';
  try {
    const response = await fetch('http://127.0.0.1:5000/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ essay: textarea.value })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Analysis failed');

    // Update the DOM with real results instead of the hardcoded mock values
    document.getElementById('score-num').textContent = data.score;
    // ...update feedback cards from data.feedback here...

    workspace.classList.add('has-results');
  } catch (err) {
    alert('Something went wrong: ' + err.message);
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze essay';
  }
});
```

The full dynamic version (building the feedback cards from `data.feedback`
instead of hardcoded HTML) is the next step — happy to build that
once you're ready to connect frontend and backend for real.

## File overview

| File | Purpose |
|---|---|
| `features.py` | Turns essay text into numeric features (word count, sentence structure, vocabulary richness, spelling). Used by both training and the live API — keep them in sync. |
| `train.py` | Trains the Ridge regression model on a CSV of essays+scores, saves `model.pkl`. |
| `app.py` | Flask API. `POST /analyze` takes `{"essay": "..."}`, returns score + feedback. |
| `sample_data/sample_essays.csv` | 12 example essays for testing the pipeline before the real dataset is in. |
| `requirements.txt` | Exact package versions used — all free, all pip-installable, no paid services. |

## Notes on the model itself

This is deliberately the **simple baseline**, not the transformer. It
scores essays using surface-level features (length, spelling,
sentence/paragraph structure, vocabulary variety) rather than deep
language understanding. That's expected and fine for Phase 2 — it
proves the full pipeline (extract → train → predict → serve → display)
works, which is the harder engineering problem. The transformer
upgrade in Phase 3 swaps out `model.pkl`'s prediction step for a
fine-tuned DistilBERT call, without needing to change the API contract
(`/analyze` still takes essay text, still returns score + feedback) —
so the frontend won't need to change at all.
