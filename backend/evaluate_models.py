"""
evaluate_models.py
-------------------
Compares the baseline (Scikit-learn Ridge) and the fine-tuned transformer
(DistilBERT) on the SAME held-out essays, so the comparison is fair.

This is an offline analysis script for your report/defense — not something
the live app calls. Run it after both models exist:

    python3 evaluate_models.py [path/to/asap_set1_rescaled.csv]

Outputs (written to evaluation_report/):
    comparison.csv        — per-essay predictions from both models
    scatter_comparison.png — predicted vs. actual score, both models
    error_by_length.png    — does essay length affect either model's error?
    REPORT.md              — summary numbers + discussion, ready to paste
                              into your FYP report

Why the same held-out split as train.py: train.py uses
train_test_split(..., test_size=0.2, random_state=42) on the SAME csv,
so re-running that split here (same random_state) reproduces the exact
same held-out rows the baseline was already evaluated on — meaning
the baseline's MAE reported here should match train.py's printed MAE,
and the transformer is being tested on essays it never trained on either
(assuming you used the same CSV + random_state in the Colab notebook,
which the notebook does by default).
"""

import os
import sys
import joblib
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from features import extract_features, features_to_vector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(SCRIPT_DIR, "sample_data", "asap_set1_rescaled.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "model.pkl")
TRANSFORMER_DIR = os.path.join(SCRIPT_DIR, "transformer_model")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "evaluation_report")


def load_baseline():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"{MODEL_PATH} not found. Run train.py first.")
    return joblib.load(MODEL_PATH)


def load_transformer():
    if not os.path.isdir(TRANSFORMER_DIR):
        raise SystemExit(
            f"{TRANSFORMER_DIR} not found. Download and unzip the fine-tuned "
            f"model from Colab first (see TRANSFORMER_INTEGRATION.md)."
        )
    try:
        tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_DIR)
    except OSError as e:
        raise SystemExit(
            f"Found {TRANSFORMER_DIR} but couldn't load the model weights from it "
            f"(model.safetensors or pytorch_model.bin missing?). If you removed the "
            f"weight file to keep the repo small for upload, copy it back in before "
            f"running this script.\n\nOriginal error: {e}"
        )
    model.eval()
    return tokenizer, model


def predict_baseline(bundle, essay: str) -> float:
    feat = extract_features(essay)
    vector = [features_to_vector(feat)]
    vector_scaled = bundle["scaler"].transform(vector)
    return max(0, min(100, bundle["model"].predict(vector_scaled)[0]))


def predict_transformer(tokenizer, model, essay: str) -> float:
    inputs = tokenizer(essay, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)
    raw = output.logits.item()
    return max(0, min(100, raw * 100))


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_PATH
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading dataset: {data_path}")
    df = pd.read_csv(data_path).dropna(subset=["essay", "score"])

    # Same split as train.py -- these are essays NEITHER model was trained on,
    # assuming the transformer notebook used the same CSV + random_state.
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Evaluating on {len(test_df)} held-out essays\n")

    print("Loading baseline model...")
    baseline = load_baseline()
    print("Loading transformer model...")
    tokenizer, transformer_model = load_transformer()

    rows = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        essay, actual = row["essay"], row["score"]
        baseline_pred = predict_baseline(baseline, essay)
        transformer_pred = predict_transformer(tokenizer, transformer_model, essay)
        rows.append({
            "actual_score": actual,
            "baseline_pred": baseline_pred,
            "baseline_error": abs(baseline_pred - actual),
            "transformer_pred": transformer_pred,
            "transformer_error": abs(transformer_pred - actual),
            "word_count": len(essay.split()),
        })
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(test_df)} essays evaluated...")

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(OUTPUT_DIR, "comparison.csv"), index=False)

    baseline_mae = mean_absolute_error(results["actual_score"], results["baseline_pred"])
    baseline_rmse = mean_squared_error(results["actual_score"], results["baseline_pred"]) ** 0.5
    transformer_mae = mean_absolute_error(results["actual_score"], results["transformer_pred"])
    transformer_rmse = mean_squared_error(results["actual_score"], results["transformer_pred"]) ** 0.5

    improvement = ((baseline_mae - transformer_mae) / baseline_mae) * 100

    print("\n=== RESULTS ===")
    print(f"Baseline    — MAE: {baseline_mae:.2f}   RMSE: {baseline_rmse:.2f}")
    print(f"Transformer — MAE: {transformer_mae:.2f}   RMSE: {transformer_rmse:.2f}")
    print(f"Improvement: {improvement:+.1f}%")

    # --- Chart 1: predicted vs actual, both models ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True, sharex=True)
    for ax, col, title, mae in [
        (axes[0], "baseline_pred", "Baseline (Ridge)", baseline_mae),
        (axes[1], "transformer_pred", "Transformer (DistilBERT)", transformer_mae),
    ]:
        ax.scatter(results["actual_score"], results[col], alpha=0.4, s=18, color="#B5482F")
        ax.plot([0, 100], [0, 100], "--", color="#5C7A5C", linewidth=1.5, label="Perfect prediction")
        ax.set_xlabel("Actual score")
        ax.set_title(f"{title}\nMAE = {mae:.2f}")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Predicted score")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "scatter_comparison.png"), dpi=150)
    plt.close()

    # --- Chart 2: does essay length affect error? ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(results["word_count"], results["baseline_error"], alpha=0.4, s=16, label="Baseline", color="#B5482F")
    ax.scatter(results["word_count"], results["transformer_error"], alpha=0.4, s=16, label="Transformer", color="#5C7A5C")
    ax.set_xlabel("Essay word count")
    ax.set_ylabel("Absolute error (points)")
    ax.set_title("Prediction error vs. essay length")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "error_by_length.png"), dpi=150)
    plt.close()

    # --- Report ---
    better = "Transformer" if transformer_mae < baseline_mae else "Baseline"
    report = f"""# Model Comparison Report

Evaluated on {len(test_df)} held-out essays (never seen during training by
either model), from `{os.path.basename(data_path)}`.

## Results

| Model | MAE (0-100 scale) | RMSE |
|---|---|---|
| Baseline (Ridge regression) | {baseline_mae:.2f} | {baseline_rmse:.2f} |
| Transformer (fine-tuned DistilBERT) | {transformer_mae:.2f} | {transformer_rmse:.2f} |

**{better} model performed better**, with a {abs(improvement):.1f}% {'reduction' if improvement > 0 else 'increase'} in
mean absolute error compared to the baseline.

## What this means

- **MAE** (Mean Absolute Error) is the average number of points a model's
  prediction was off by, in either direction. Lower is better.
- The baseline model only sees hand-crafted features (word count, sentence
  structure, spelling, vocabulary richness) — no understanding of meaning,
  coherence, or argument quality.
- The transformer reads the full essay text and can, in principle, pick up
  on coherence and argument structure that the baseline's features can't
  capture — which is the theoretical reason to expect it to do better.

## Files in this folder

- `comparison.csv` — every held-out essay's actual score vs. both models' predictions
- `scatter_comparison.png` — visual accuracy comparison (closer to the diagonal line = more accurate)
- `error_by_length.png` — checks whether either model struggles more on very short or very long essays

## Suggested discussion points for your report

- If the transformer's improvement is modest, a fair explanation is that
  {len(test_df) + len(test_df) * 4} essays (train + test) is a small dataset for fine-tuning a
  66-million-parameter model — transformers typically need more data to
  fully outperform simpler models with strong hand-crafted features.
- Look at `error_by_length.png`: if one model's error grows with essay
  length, that's worth discussing (e.g. the baseline's `avg_sentence_length`
  feature may not scale well to very long essays; the transformer's
  512-token truncation may cut off the ends of long essays instead).
"""
    with open(os.path.join(OUTPUT_DIR, "REPORT.md"), "w") as f:
        f.write(report)

    print(f"\nSaved comparison.csv, scatter_comparison.png, error_by_length.png, and REPORT.md to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
