"""Explainability helpers for the trained FireGuard AI model.

Honest scope: these are GLOBAL, impurity-based feature importances from the
tree model. They describe what the model relies on overall — they are not
per-prediction causal explanations (that would need SHAP or similar, which
is a documented extension, not a hidden claim).

Shown in the dashboard's ML Analysis tab as "What the model relies on".
"""
import pandas as pd


def feature_importance(bundle, top_n=15):
    """Sorted DataFrame of feature -> importance for the trained model."""
    model = bundle["model"]
    importance = pd.DataFrame({
        "feature": bundle["feature_columns"],
        "importance": model.feature_importances_,
    })
    return (importance.sort_values("importance", ascending=False)
            .head(top_n).reset_index(drop=True))


def top_features_text(bundle, top_n=5):
    """One-line-per-feature summary, handy for markdown display."""
    rows = feature_importance(bundle, top_n=top_n)
    return "\n".join(
        f"- `{r.feature}` — importance {r.importance:.3f}"
        for r in rows.itertuples()
    )
