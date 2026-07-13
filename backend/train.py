"""
train.py
--------
Trains the baseline essay-scoring model and saves it to model.pkl.

Usage:
    python3 train.py                      # uses sample_data/sample_essays.csv
    python3 train.py path/to/asap.csv     # use the real ASAP dataset instead

Expected CSV format: two columns, "essay" (text) and "score" (0-100 number).
If your ASAP CSV uses different column names or a different score scale
(ASAP's raw files vary by essay "set" and often score out of a different
max), rename/rescale columns to match this before running train.py —
see the README for the exact steps.
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from features import extract_features, features_to_vector, FEATURE_ORDER

# Resolve paths relative to THIS FILE's location, not whatever folder the
# script happens to be launched from. Without this, running `python
# backend/train.py` from one level up (a common case in VS Code / Code
# Runner) fails with "file not found" even though the file exists —
# because "sample_data/..." gets looked up relative to the wrong folder.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(SCRIPT_DIR, "sample_data", "sample_essays.csv")
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "model.pkl")


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "essay" not in df.columns or "score" not in df.columns:
        raise ValueError(
            f"Expected columns 'essay' and 'score' in {path}, "
            f"found: {list(df.columns)}"
        )
    df = df.dropna(subset=["essay", "score"])
    return df


def build_feature_matrix(essays):
    rows = [extract_features(e) for e in essays]
    return [features_to_vector(r) for r in rows]


def main():
    # A user-supplied path (2nd example in the usage docstring) is used
    # exactly as given — resolved relative to wherever the command was
    # run from, same as any normal command-line tool.
    data_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_PATH
    print(f"Loading dataset: {data_path}")
    df = load_dataset(data_path)
    print(f"  {len(df)} essays loaded")

    print("Extracting features...")
    X = build_feature_matrix(df["essay"].tolist())
    y = df["score"].tolist()

    # With a very small sample dataset, skip the train/test split (not
    # enough rows to hold any out) and just fit on everything so the
    # pipeline can be verified end-to-end. Once you're using the real
    # ASAP dataset (thousands of rows), this condition will be false
    # and a proper held-out test set will be used automatically.
    if len(df) < 30:
        print(
            f"  Only {len(df)} rows — training on all of them (no held-out "
            f"test set). Swap in the real ASAP dataset for a proper "
            f"train/test evaluation."
        )
        X_train, y_train = X, y
        X_test, y_test = X, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Ridge regression model...")
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    print(f"  Mean Absolute Error: {mae:.2f} points (on a 0-100 scale)")

    joblib.dump(
        {"model": model, "scaler": scaler, "feature_order": FEATURE_ORDER},
        MODEL_SAVE_PATH,
    )
    print(f"Saved {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
