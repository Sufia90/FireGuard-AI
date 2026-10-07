"""Single-prediction helper for the trained FireGuard AI model.

Used by the dashboard's ML Analysis tab and Fire-Safety Insights tab::

    from ml.predict import predict_behavior
    label, probabilities = predict_behavior({
        "initial_o2": 18.0,
        "fuel_material": "100 micron PMMA film   2 cm wide",
        "fan_display": 62.0,
        "air_display": 5.0,
        "flow_restrictor": 2.0,
        "thickness_micron": 100.0,
    })

Returns (predicted_band, {band: probability}). The band is the derived
O2-depletion proxy — never present it as a NASA finding.
"""
from pathlib import Path

import pandas as pd

from ml.train_model import encode_features, load_or_train_model

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "fireguard_model.pkl"

REQUIRED_FEATURES = {"initial_o2", "fuel_material", "fan_display",
                     "air_display", "flow_restrictor", "thickness_micron"}

_bundle = None

# my name anika
def _get_bundle():
    global _bundle
    if _bundle is None:
        _bundle = load_or_train_model(MODEL_PATH)
    return _bundle


def predict_behavior(features):
    """Predict the burn-intensity proxy band for one set of conditions."""
    missing = REQUIRED_FEATURES - set(features.keys())
    if missing:
        raise ValueError(f"Missing features for prediction: {sorted(missing)}")

    bundle = _get_bundle()
    model = bundle["model"]

    row = pd.DataFrame([{
        "initial_o2": float(features["initial_o2"]),
        "fuel_material": str(features["fuel_material"]),
        "fan_display": float(features["fan_display"]),
        "air_display": float(features["air_display"]),
        "flow_restrictor": float(features["flow_restrictor"]),
        "thickness_micron": float(features["thickness_micron"]),
    }])
    for column, median in bundle.get("medians", {}).items():
        row[column] = row[column].fillna(median)
    X = encode_features(row, feature_columns=bundle["feature_columns"])

    proba = model.predict_proba(X)[0]
    label = str(model.classes_[proba.argmax()])
    probabilities = {str(c): float(p) for c, p in zip(model.classes_, proba)}
    return label, probabilities
