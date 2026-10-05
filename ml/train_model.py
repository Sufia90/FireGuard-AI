"""Train the FireGuard AI burn-intensity classifier.

Upgraded trainer: compares RandomForest vs GradientBoosting with 5-fold
cross-validation and saves the best model. The saved bundle keeps the same
shape as before, so predict.py, evaluate_model.py and the dashboard work
unchanged — plus a "model_name" and per-model "all_scores" entry.

HONESTY NOTE (also stored in the bundle as "target_note"): the target is a
DERIVED proxy, not a NASA label — see data_processing/feature_engineering.py.
"""
import pickle
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from data_processing.data_cleaning import clean_experiments
from data_processing.nasa_data_loader import load_experiments

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "experiments" / "bass2_experimental_table.csv"
MODEL_PATH = ROOT / "models" / "fireguard_model.pkl"

NUMERIC_FEATURES = ["initial_o2", "fan_display", "air_display",
                    "flow_restrictor", "thickness_micron"]
CATEGORICAL_FEATURES = ["fuel_material"]
TARGET = "o2_depletion_band"

DERIVED_TARGET_NOTE = (
    "TARGET IS A DERIVED PROXY, NOT A NASA LABEL: the BASS-II table has no "
    "flame-behavior column, so the target is o2_depletion = initial O2 - final "
    "O2 (measured % by vol) binned into tertile bands (low / moderate / high "
    "depletion). Predictions are burn-intensity proxy estimates, never NASA "
    "findings."
)

# Candidate models compared on 5-fold CV; the winner is saved.
CANDIDATES = {
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
}


def load_training_data():
    return clean_experiments(load_experiments(DATA_PATH))


def encode_features(df, feature_columns=None):
    X = pd.get_dummies(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
                       columns=CATEGORICAL_FEATURES)
    if feature_columns is not None:
        X = X.reindex(columns=feature_columns, fill_value=0)
    return X


def compare_models(X, y):
    """5-fold CV accuracy for each candidate; returns {name: mean_score}."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}
    for name, model in CANDIDATES.items():
        fold_scores = cross_val_score(model, X, y, cv=cv)
        scores[name] = float(fold_scores.mean())
        print(f"  {name}: CV accuracy {scores[name]:.2%} "
              f"(folds: {np.round(fold_scores, 2).tolist()})")
    return scores


def train():
    df = load_training_data()
    X = encode_features(df)
    y = df[TARGET]
    feature_columns = list(X.columns)
    medians = {c: float(df[c].median()) for c in NUMERIC_FEATURES}

    print("Comparing candidate models (5-fold CV):")
    all_scores = compare_models(X, y)
    best_name = max(all_scores, key=all_scores.get)
    print(f"Winner: {best_name} ({all_scores[best_name]:.2%} CV accuracy)")

    model = CANDIDATES[best_name]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    accuracy = float(model.score(X_test, y_test))

    print(f"Trained on {len(X_train)} rows, tested on {len(X_test)} rows.")
    print(f"Holdout accuracy: {accuracy:.2%}")
    print(f"Classes: {list(model.classes_)}")
    print(DERIVED_TARGET_NOTE)

    return {
        "model": model,
        "model_name": best_name,
        "all_scores": all_scores,
        "feature_columns": feature_columns,
        "medians": medians,
        "accuracy": accuracy,
        "cv_accuracy": all_scores[best_name],
        "classes": list(model.classes_),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "target_note": DERIVED_TARGET_NOTE,
        "band_cutoffs": df.attrs.get("depletion_band_cutoffs"),
    }


def save_model(bundle, path=MODEL_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Model saved to {path} "
          f"(winner: {bundle.get('model_name', 'unknown')})")
    return path


def load_or_train_model(path=MODEL_PATH):
    path = Path(path)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    bundle = train()
    save_model(bundle, path)
    return bundle


def main():
    save_model(train())
    print("Done. Next: python ml/evaluate_model.py")


if __name__ == "__main__":
    main()
