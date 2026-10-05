"""Evaluate the trained FireGuard AI model on the holdout set.

Run from the repo root:
    python ml/evaluate_model.py

Prints a classification report, the confusion matrix, the winning model's
CV score, and the derived-target honesty note.
"""
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from ml.train_model import (
    TARGET,
    encode_features,
    load_or_train_model,
    load_training_data,
)


def main():
    bundle = load_or_train_model()
    model = bundle["model"]

    print(f"Evaluating model: {bundle.get('model_name', 'unknown')}")
    if bundle.get("all_scores"):
        print("Candidate CV scores:")
        for name, score in bundle["all_scores"].items():
            print(f"  {name}: {score:.2%}")

    df = load_training_data()
    X = encode_features(df, feature_columns=bundle["feature_columns"])
    y = df[TARGET]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    y_pred = model.predict(X_test)

    print("\n=== Classification report (holdout) ===")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("=== Confusion matrix (rows=true, cols=predicted) ===")
    print(confusion_matrix(y_test, y_pred, labels=bundle["classes"]))
    print(f"Labels order: {bundle['classes']}")
    print(f"\n5-fold CV accuracy: {bundle.get('cv_accuracy', 0):.2%}")
    print(f"\nTarget: {bundle['target']}")
    print(bundle.get("target_note", ""))


if __name__ == "__main__":
    main()
