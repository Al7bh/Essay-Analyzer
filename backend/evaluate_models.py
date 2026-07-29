"""
evaluate_models.py
-------------------
Compares the baseline (Scikit-learn Ridge) and the fine-tuned transformer (DistilBERT)
on the SAME held-out essays, reporting MAE, RMSE, and Quadratic Weighted Kappa (QWK).

Run offline after both models are present:
    python3 evaluate_models.py [path/to/asap_set1_rescaled.csv]
"""

import os
import sys
import joblib
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, cohen_kappa_score
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
            f"{TRANSFORMER_DIR} not found. Ensure fine-tuned weights exist in "
            f"backend/transformer_model/."
        )
    try:
        tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_DIR)
    except OSError as e:
        raise SystemExit(
            f"Found {TRANSFORMER_DIR} but couldn't load model weights.\nOriginal error: {e}"
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
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(test_df)} essays evaluated...")
            
    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(OUTPUT_DIR, "comparison.csv"), index=False)
    
    # Compute MAE & RMSE
    baseline_mae = mean_absolute_error(results["actual_score"], results["baseline_pred"])
    baseline_rmse = mean_squared_error(results["actual_score"], results["baseline_pred"]) ** 0.5
    transformer_mae = mean_absolute_error(results["actual_score"], results["transformer_pred"])
    transformer_rmse = mean_squared_error(results["actual_score"], results["transformer_pred"]) ** 0.5
    
    # Compute Quadratic Weighted Kappa (QWK)
    y_true_discrete = np.round(results["actual_score"]).astype(int)
    y_base_discrete = np.round(results["baseline_pred"]).astype(int)
    y_trans_discrete = np.round(results["transformer_pred"]).astype(int)
    
    baseline_qwk = cohen_kappa_score(y_true_discrete, y_base_discrete, weights="quadratic")
    transformer_qwk = cohen_kappa_score(y_true_discrete, y_trans_discrete, weights="quadratic")
    
    mae_improvement = ((baseline_mae - transformer_mae) / baseline_mae) * 100
    
    print("\n=== EVALUATION METRICS ===")
    print(f"Baseline      -> MAE: {baseline_mae:.2f} | RMSE: {baseline_rmse:.2f} | QWK: {baseline_qwk:.4f}")
    print(f"Transformer   -> MAE: {transformer_mae:.2f} | RMSE: {transformer_rmse:.2f} | QWK: {transformer_qwk:.4f}")
    print(f"MAE Change: {mae_improvement:+.1f}%")
    
    # Chart 1: Predicted vs Actual with QWK in Title
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, sharex=True)
    for ax, col, title, mae, qwk in [
        (axes[0], "baseline_pred", "Baseline (Ridge)", baseline_mae, baseline_qwk),
        (axes[1], "transformer_pred", "Transformer (DistilBERT)", transformer_mae, transformer_qwk),
    ]:
        ax.scatter(results["actual_score"], results[col], alpha=0.4, s=18, color="#B5482F")
        ax.plot([0, 100], [0, 100], "--", color="#5C7A5C", linewidth=1.5, label="Ideal Fit")
        ax.set_xlabel("Actual Score")
        ax.set_title(f"{title}\nMAE: {mae:.2f} | QWK: {qwk:.3f}")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Predicted Score")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "scatter_comparison.png"), dpi=150)
    plt.close()
    
    # Chart 2: Error vs Length
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(results["word_count"], results["baseline_error"], alpha=0.4, s=16, label="Baseline", color="#B5482F")
    ax.scatter(results["word_count"], results["transformer_error"], alpha=0.4, s=16, label="Transformer", color="#5C7A5C")
    ax.set_xlabel("Essay Word Count")
    ax.set_ylabel("Absolute Error")
    ax.set_title("Prediction Error vs. Essay Length")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "error_by_length.png"), dpi=150)
    plt.close()
    
    # Generate REPORT.md
    report = f"""# Comprehensive Model Evaluation Report

Evaluated on {len(test_df)} held-out essays.

## Quantitative Comparison

| Metric | Baseline (Ridge) | Transformer (DistilBERT) | Interpretation |
|---|---|---|---|
| **MAE** (Lower is better) | {baseline_mae:.2f} | {transformer_mae:.2f} | Average points deviation from human score |
| **RMSE** (Lower is better) | {baseline_rmse:.2f} | {transformer_rmse:.2f} | Penalizes larger outlier errors more heavily |
| **QWK** (Higher is better) | **{baseline_qwk:.4f}** | **{transformer_qwk:.4f}** | Inter-rater agreement with human ground truth |

## Findings for Defense
- **Quadratic Weighted Kappa (QWK)** measures agreement with human evaluators on a -1 to +1 scale. Higher QWK reflects closer alignment with official rubrics.
- **Mean Absolute Error (MAE)** measures raw point distance on the 0–100 normalized scale.
"""
    with open(os.path.join(OUTPUT_DIR, "REPORT.md"), "w") as f:
        f.write(report)
        
    print(f"\nReport and visualizations saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()