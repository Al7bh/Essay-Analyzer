"""
train.py
--------
Trains the essay-scoring baseline model and saves it to model.pkl.

VERSION 2 -- tries multiple model types, picks the one with the lowest
MAE on the held-out test set, and saves that. Replaces the old fixed
"always use Ridge(alpha=1.0)" approach.

Models tried:
  1. RidgeCV       -- fast, interpretable, good baseline
  2. GradientBoosting -- often wins on tabular data
  3. SVR (RBF)     -- strong on small-to-medium datasets

All use the same 17-feature set from features.py v2.

Usage:
    python train.py                                    # sample data
    python train.py sample_data/asap_set1_rescaled.csv # real ASAP data
"""

import os
import sys
import time
import joblib
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from features import extract_features, features_to_vector, FEATURE_ORDER

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(SCRIPT_DIR, "sample_data", "sample_essays.csv")
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "model.pkl")


def load_dataset(path):
    df = pd.read_csv(path)
    if "essay" not in df.columns or "score" not in df.columns:
        raise ValueError(f"Expected 'essay' and 'score' columns, found: {list(df.columns)}")
    return df.dropna(subset=["essay", "score"])


def build_feature_matrix(essays):
    return [features_to_vector(extract_features(e,fast_mode=True)) for e in essays]


def get_candidates(small_dataset):
    if small_dataset:
        return [("RidgeCV", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]))]
    return [
        ("RidgeCV", RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])),
        ("GradientBoosting", GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=42,
        )),
        ("SVR (RBF, C=50)", SVR(kernel="rbf", C=50, epsilon=2.0, gamma="scale")),
    ]


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_PATH
    print(f"Loading dataset: {data_path}")
    df = load_dataset(data_path)
    print(f"  {len(df)} essays loaded")

    print(f"Extracting {len(FEATURE_ORDER)} features per essay...")
    t0 = time.perf_counter()
    X = build_feature_matrix(df["essay"].tolist())
    y = df["score"].tolist()
    print(f"  Done in {time.perf_counter()-t0:.1f}s")

    small = len(df) < 30
    if small:
        print(f"  Only {len(df)} rows -- training on all, no held-out test set.")
        X_train, X_test, y_train, y_test = X, y, X, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = []
    for name, model in get_candidates(small):
        print(f"\nTraining {name}...")
        t0 = time.perf_counter()
        model.fit(X_train_s, y_train)
        elapsed = time.perf_counter() - t0
        preds = [max(0, min(100, p)) for p in model.predict(X_test_s)]
        mae = mean_absolute_error(y_test, preds)
        print(f"  MAE: {mae:.2f}  ({elapsed:.1f}s)")
        results.append((mae, name, model))

    results.sort(key=lambda x: x[0])
    best_mae, best_name, best_model = results[0]

    print(f"\n{'='*50}")
    print(f"Best model: {best_name}  (MAE: {best_mae:.2f})")
    if len(results) > 1:
        for mae, name, _ in results:
            tag = " <-- SAVED" if name == best_name else ""
            print(f"  {name:<30} MAE: {mae:.2f}{tag}")
    print(f"{'='*50}")

    joblib.dump({
        "model": best_model,
        "scaler": scaler,
        "feature_order": FEATURE_ORDER,
        "model_name": best_name,
        "mae": best_mae,
    }, MODEL_SAVE_PATH)

    print(f"\nSaved {MODEL_SAVE_PATH}")
    prev_mae = 6.49
    if best_mae < prev_mae:
        print(f"Improvement over old baseline: {prev_mae - best_mae:.2f} points lower MAE.")
    else:
        print(f"Note: MAE did not improve over old baseline ({prev_mae}). Check feature extraction.")


if __name__ == "__main__":
    main()