# Transformer Model Integration & Fine-Tuning Guide

This guide details the fine-tuning, export, and integration of the deep learning Transformer model (`distilbert-base-uncased`) into the **Marginal / Essay Analyzer** backend.

---

## 1. Overview & Architectural Role

The system utilizes a **dual-model architecture**:
- **Baseline Model (`model.pkl`)**: A Scikit-learn tabular model (Gradient Boosting / RidgeCV) operating on 17 hand-crafted NLP features.
- **Transformer Model (`transformer_model/`)**: A fine-tuned `distilbert-base-uncased` neural network evaluating full semantic representations.

When a request hits `POST /analyze`, `app.py` computes **both** scores simultaneously. The Transformer score is presented as the primary headline score in the UI, while allowing users to toggle between both scores to compare structural mechanics versus semantic evaluation.

---

## 2. Dataset Scaling & Per-Prompt Normalization

### The Challenge of Multi-Prompt Training
The Kaggle ASAP-AES dataset contains **12,978 student essays across 8 distinct essay prompts**. Each prompt originally used a different score scale (e.g., Set 1: 2–12, Set 2: 1–6, Set 5: 0–4, Set 7: 0–30, Set 8: 0–60). 

Combining these raw scores directly would force the model to learn incompatible grading scales.

### The Solution: Min-Max Normalization
Inside `Train_ASAP_Transformer.ipynb`, all 8 essay sets are processed independently prior to concatenation:

$$\text{score}_{\text{norm}} = \left( \frac{\text{raw\_score} - \text{min\_score}_i}{\text{max\_score}_i - \text{min\_score}_i} \right) \times 100$$

1. **Prompt-Level Rescaling**: Each essay set $i$ is rescaled to a universal **0 to 100 scale**.
2. **Training Stability Rescaling**: For loss calculation stability in PyTorch, the target labels are divided by 100 (`label = score / 100.0`) during training, placing targets between `0.0` and `1.0`.
3. **Prediction Rescaling**: At prediction time, model logits are multiplied back by 100 and rounded (`round(raw * 100)`) to yield a 0–100 integer score.

### Single-Prompt vs. Multi-Prompt Trade-off
- **Single-Prompt Fine-Tuning (Set 1 only, 1,783 essays)**: Achieves a lower MAE (~5.80–5.85) on Set 1, but overfits to persuasive letter prompts.
- **Multi-Prompt Fine-Tuning (All 8 Sets, 12,978 essays)**: Achieves a generalized MAE of **~9.97** across persuasive, narrative, and source-based expository essays. This demonstrates true cross-prompt generalization without overfitting to a single topic.

---

## 3. Fine-Tuning Pipeline (`Train_ASAP_Transformer.ipynb`)

Fine-tuning is performed using Google Colab (free T4 GPU runtime recommended).

### Key Training Hyperparameters
```python
MODEL_NAME = "distilbert-base-uncased"

# Model Head Initialization
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=1, 
    problem_type="regression"
)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="mae_0_100",
    greater_is_better=False,
    logging_steps=20,
    report_to="none"
)

```

---

## 4. Local Setup & Folder Structure

Once training completes in Colab, save and download the model artifacts (`essay_scorer_distilbert.zip`). Extract the files into `backend/transformer_model/`.

The folder structure **must** look like this:

```text
backend/
├── app.py
├── features.py
├── train.py
├── evaluate_models.py
├── model.pkl                     <-- Baseline model
└── transformer_model/             <-- Transformer directory
    ├── config.json
    ├── model.safetensors          <-- Heavy neural network weights (~268 MB)
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    └── vocab.txt

```

> **Git Repository Management Note:**
> Because `model.safetensors` is ~268 MB, it should be excluded from Git uploads to keep repository sizes small. Add the following line to your `.gitignore`:
> ```gitignore
> backend/transformer_model/model.safetensors
> 
> ```
> 
> 

---

## 5. Backend Integration (`app.py`)

In `app.py`, the Transformer bundle is loaded lazily on the first incoming request and cached in memory.

### Loading & Inference Logic

```python
def get_transformer_bundle():
    global _transformer_bundle
    if _transformer_bundle is None and os.path.isdir(TRANSFORMER_DIR):
        try:
            tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
            model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_DIR)
            model.eval()
            _transformer_bundle = {"tokenizer": tokenizer, "model": model}
        except OSError:
            # Catches missing model.safetensors without crashing the server
            app.logger.warning(
                f"{TRANSFORMER_DIR} exists but model weights couldn't be loaded. "
                f"Falling back to the baseline model."
            )
            _transformer_bundle = False  # Sentinel to prevent repeated retry attempts
    return _transformer_bundle or None

def predict_with_transformer(essay: str) -> float:
    bundle = get_transformer_bundle()
    inputs = bundle["tokenizer"](
        essay, 
        truncation=True, 
        padding="max_length",
        max_length=512, 
        return_tensors="pt"
    )
    with torch.no_grad():
        output = bundle["model"](**inputs)
    raw = output.logits.item()
    return max(0, min(100, round(raw * 100)))

```

---

## 6. Fail-Safe Fallback Mechanism

A critical bug found during early testing occurred when the `transformer_model/` folder was committed to GitHub without `model.safetensors`. The backend detected the folder, attempted to load missing weights, and crashed with an unhandled `500 Internal Server Error` on every `/analyze` call.

### The Fix

1. **`try / except OSError` Block**: Catches weight-loading errors gracefully.
2. **Sentinel Flag (`_transformer_bundle = False`)**: Marks the Transformer as "checked but unavailable" so the server doesn't waste CPU cycles re-attempting to load missing files on every request.
3. **Seamless Baseline Fallback**: `app.py` automatically falls back to returning the Baseline score (`baseline_score`) as the primary score.
4. **Health Check Endpoint**: `GET /health` reports `"transformer_loaded": false` or `true` so you can verify status instantly.

---

## 7. Model Evaluation (`evaluate_models.py`)

To evaluate the Transformer alongside the Baseline on held-out test essays:

```bash
python evaluate_models.py

```

This generates `backend/evaluation_report/` containing:

* **`REPORT.md`**: Markdown summary table comparing MAE, RMSE, and Quadratic Weighted Kappa (QWK).
* **`comparison.csv`**: Per-essay raw predictions and error distances.
* **`scatter_comparison.png`**: Predicted vs. Actual score scatter plots with regression trendlines.
* **`error_by_length.png`**: Plot analyzing whether prediction errors correlate with essay word count.

### Sample Evaluation Metrics Output

```text
=== EVALUATION METRICS ===
Baseline      -> MAE: 5.80 | RMSE: 7.38 | QWK: 0.8629
Transformer   -> MAE: 5.80 | RMSE: 7.41 | QWK: 0.8645

```

*Note: A Quadratic Weighted Kappa (QWK) score above 0.80 represents excellent inter-rater agreement with official human markers.*
