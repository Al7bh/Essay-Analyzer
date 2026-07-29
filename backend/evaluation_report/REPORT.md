# Comprehensive Model Evaluation Report

Evaluated on 357 held-out essays.

## Quantitative Comparison

| Metric | Baseline (Ridge) | Transformer (DistilBERT) | Interpretation |
|---|---|---|---|
| **MAE** (Lower is better) | 5.80 | 5.80 | Average points deviation from human score |
| **RMSE** (Lower is better) | 7.38 | 7.41 | Penalizes larger outlier errors more heavily |
| **QWK** (Higher is better) | **0.8629** | **0.8645** | Inter-rater agreement with human ground truth |

## Findings for Defense
- **Quadratic Weighted Kappa (QWK)** measures agreement with human evaluators on a -1 to +1 scale. Higher QWK reflects closer alignment with official rubrics.
- **Mean Absolute Error (MAE)** measures raw point distance on the 0–100 normalized scale.
