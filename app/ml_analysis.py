"""ML Analysis tab: predict the burn-intensity proxy band for chosen conditions.

Includes the global feature-importance chart (from ml/model_explainability.py)
so the team can see what the winning model relies on overall.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ml import model_explainability
from ml import predict as predict_mod

DERIVED_LABEL_NOTE = (
    "🎯 **About the ML target:** the BASS-II table has no flame-behavior label, "
    "so the model predicts a **burn-intensity proxy** — O₂ depletion "
    "(initial O₂ − final O₂, both measured % by vol) binned into low / moderate / "
    "high tertile bands. This band is derived by us, **not** a NASA-provided label."
)


def render(df, bundle):
    st.header("🤖 ML Analysis")
    st.write("Choose experimental conditions — the trained model predicts the "
             "burn-intensity proxy band learned from BASS-II data.")
    _acc = bundle.get("cv_accuracy", bundle.get("accuracy", 0))
    st.caption(f"Model: **{bundle.get('model_name', 'unknown')}** — 5-fold "
               f"cross-validated accuracy on 129 BASS-II tests: {_acc:.1%}")
    st.warning(DERIVED_LABEL_NOTE)

    m1, m2, m3 = st.columns(3)
    initial_o2 = m1.slider("Initial O₂ %", 13.0, 23.0, 18.0, 0.1)
    fan_display = m2.number_input("Fan display", 0.0, 100000.0,
                                  float(df["fan_display"].median()))
    air_display = m3.number_input("Air display", 0.0, 20.0,
                                  float(df["air_display"].median()))
    m4, m5, m6 = st.columns(3)
    fuel_material = m4.selectbox("Fuel material",
                                 sorted(df["fuel_material"].unique()))
    flow_restrictor = m5.selectbox("Flow restrictor",
                                   sorted(df["flow_restrictor"].unique()))
    thickness_micron = m6.number_input("Thickness (micron)", 0.0, 10000.0,
                                       float(df["thickness_micron"].median()))

    if st.button("Predict band", type="primary"):
        label, probs = predict_mod.predict_behavior({
            "initial_o2": initial_o2,
            "fuel_material": fuel_material,
            "fan_display": fan_display,
            "air_display": air_display,
            "flow_restrictor": flow_restrictor,
            "thickness_micron": thickness_micron,
        })
        st.success(f"**Predicted burn-intensity band: {label}** "
                   f"(derived O₂-depletion proxy, not a NASA label)")

        prob_df = pd.DataFrame({"band": list(probs.keys()),
                                "probability": list(probs.values())})
        st.plotly_chart(px.bar(prob_df, x="band", y="probability",
                               title="Predicted band probabilities",
                               color="band",
                               color_discrete_sequence=px.colors.qualitative.Set2),
                        use_container_width=True)

        st.subheader("Most similar BASS-II tests")
        num_cols = ["initial_o2", "fan_display", "air_display",
                    "flow_restrictor", "thickness_micron"]
        num = df[num_cols].to_numpy(dtype=float)
        means, stds = num.mean(axis=0), num.std(axis=0) + 1e-9
        z = (num - means) / stds
        q = (np.array([initial_o2, fan_display, air_display,
                       flow_restrictor, thickness_micron]) - means) / stds
        dist = np.linalg.norm(z - q, axis=1)
        similar = df.assign(_dist=dist).sort_values("_dist").head(3)
        st.dataframe(similar[["test_id", "fuel_material", "initial_o2",
                              "o2_depletion", "o2_depletion_band"]],
                     use_container_width=True)
        st.caption("Similarity = Euclidean distance on standardized numeric "
                   "conditions.")

    st.subheader("What the model relies on (global feature importance)")
    imp = model_explainability.feature_importance(bundle)
    st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h",
                           title="Feature importances",
                           color="importance",
                           color_continuous_scale="Blues"),
                    use_container_width=True)
    st.caption("Global impurity-based importances — they describe the model "
               "overall, not a single prediction.")
