"""Experiment Comparison tab: side-by-side comparison of two BASS-II tests.

Descriptive only — it aligns the two tests' measured fields and charts
their oxygen numbers. It does not assert scientific equivalence.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render(df):
    st.header("🔄 Experiment Comparison")
    st.write("Place two BASS-II tests side by side. The comparison is "
             "descriptive — it does not assert scientific equivalence.")

    ids = sorted(df["test_id"].unique())
    cc1, cc2 = st.columns(2)
    test_a = cc1.selectbox("Test A", ids, index=0)
    test_b = cc2.selectbox("Test B", ids, index=min(3, len(ids) - 1))
    row_a = df[df["test_id"] == test_a].iloc[0]
    row_b = df[df["test_id"] == test_b].iloc[0]

    comp = pd.DataFrame({
        "Field": ["Fuel material", "Initial O₂ %", "Final O₂ %",
                  "O₂ depletion (pp)", "Burn-intensity band", "Fan display",
                  "Air display", "Flow restrictor", "Thickness (micron)"],
        test_a: [row_a["fuel_material"], round(row_a["initial_o2"], 1),
                 round(row_a["final_o2"], 1), round(row_a["o2_depletion"], 1),
                 row_a["o2_depletion_band"], row_a["fan_display"],
                 row_a["air_display"], row_a["flow_restrictor"],
                 row_a["thickness_micron"]],
        test_b: [row_b["fuel_material"], round(row_b["initial_o2"], 1),
                 round(row_b["final_o2"], 1), round(row_b["o2_depletion"], 1),
                 row_b["o2_depletion_band"], row_b["fan_display"],
                 row_b["air_display"], row_b["flow_restrictor"],
                 row_b["thickness_micron"]],
    })
    st.table(comp.set_index("Field"))

    fig = go.Figure()
    for label, row in ((test_a, row_a), (test_b, row_b)):
        fig.add_bar(name=f"Test {label}",
                    x=["Initial O₂ %", "Final O₂ %", "O₂ depletion (pp)"],
                    y=[row["initial_o2"], row["final_o2"],
                       row["o2_depletion"]])
    fig.update_layout(barmode="group",
                      title=f"O₂ comparison: test {test_a} vs test {test_b}",
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#e8eefc"))
    st.plotly_chart(fig, use_container_width=True)

    diffs = []
    for field in ["fuel_material", "initial_o2", "final_o2",
                  "flow_restrictor", "o2_depletion_band"]:
        if str(row_a[field]) != str(row_b[field]):
            diffs.append(f"{field}: {row_a[field]} → {row_b[field]}")
    st.info("**Template summary of differences:** "
            + ("; ".join(diffs) if diffs else "identical on compared fields."))
