"""Fire-Safety Insights tab: evidence-grounded synthesis of ML + RAG.

Takes either a real BASS-II test or custom conditions, predicts the
burn-intensity band, lists supporting tests in the same band, and pulls
SRD_BASS-II.pdf excerpts as evidence. Everything is template-assembled —
no invented findings.
"""
import streamlit as st

from ml import predict as predict_mod
from rag import nasa_research_rag

DERIVED_LABEL_NOTE = (
    "🎯 **About the ML target:** the BASS-II table has no flame-behavior label, "
    "so the model predicts a **burn-intensity proxy** — O₂ depletion "
    "(initial O₂ − final O₂, both measured % by vol) binned into low / moderate / "
    "high tertile bands. This band is derived by us, **not** a NASA-provided label."
)


def render(df):
    st.header("🔥 Fire-Safety Insights")
    st.write("Evidence-grounded synthesis: ML pattern + retrieved BASS-II "
             "tests + document excerpts, assembled by a template.")

    choice = st.selectbox("Base the insight on…",
                          ["A BASS-II test", "Custom conditions"],
                          key="fsi_mode")
    if choice == "A BASS-II test":
        test_id = st.selectbox("Test", sorted(df["test_id"].unique()),
                               key="fsi_test")
        row = df[df["test_id"] == test_id].iloc[0]
        context = {
            "initial_o2": float(row["initial_o2"]),
            "fuel_material": row["fuel_material"],
            "fan_display": float(row["fan_display"]),
            "air_display": float(row["air_display"]),
            "flow_restrictor": float(row["flow_restrictor"]),
            "thickness_micron": float(row["thickness_micron"]),
        }
        observed_band = row["o2_depletion_band"]
    else:
        i1, i2, i3 = st.columns(3)
        context = {
            "initial_o2": i1.number_input("Initial O₂ %", 13.0, 23.0, 18.0,
                                          key="fsi_o2"),
            "fuel_material": st.selectbox(
                "Fuel material", sorted(df["fuel_material"].unique()),
                key="fsi_fuel"),
            "fan_display": i2.number_input("Fan display", 0.0, 100000.0,
                                            float(df["fan_display"].median()),
                                            key="fsi_fan"),
            "air_display": i3.number_input("Air display", 0.0, 20.0,
                                            float(df["air_display"].median()),
                                            key="fsi_air"),
            "flow_restrictor": float(sorted(df["flow_restrictor"].unique())[0]),
            "thickness_micron": float(df["thickness_micron"].median()),
        }
        observed_band = None

    if st.button("Generate insight", type="primary", key="fsi_go"):
        label, probs = predict_mod.predict_behavior(context)
        observed = (f"test {test_id} measured a **{observed_band}** band"
                    if observed_band else "no BASS-II test was selected")
        same = df[df["o2_depletion_band"] == label]

        st.subheader("🔥 Insight")
        st.markdown(f"""
**Observed pattern:** Under conditions of {context['initial_o2']}% initial O₂, \
{context['fuel_material']} fuel, {observed}, \
and the model predicts burn-intensity band **{label}** \
(confidence {probs[label]:.0%}).

**Relevant factors:**
- Initial oxygen concentration ({context['initial_o2']}%) — measured driver of O₂ depletion
- Fuel material ({context['fuel_material']}) and flow setup

**Supporting BASS-II tests in the same band:** {", ".join(same["test_id"].head(10))} \
({len(same)} tests total in band '{label}')
""")
        ev = nasa_research_rag.answer_with_evidence(
            f"{label} {context['fuel_material']} oxygen ignition")
        if ev["evidence"]:
            st.subheader("BASS-II evidence (document excerpts)")
            for e in ev["evidence"][:2]:
                st.markdown(f"**Source:** `{e['source']}`")
                st.caption(e["excerpt"] + "…")
        st.warning(DERIVED_LABEL_NOTE)
