"""Experiment Explorer tab: filter the BASS-II table and chart it."""
import plotly.express as px
import streamlit as st


def render(df):
    st.header("📊 Experiment Explorer")
    st.write("Filter the 129 BASS-II tests by fuel, burn-intensity band, and "
             "oxygen range. Charts update with the filters.")

    f1, f2, f3 = st.columns(3)
    fuels = f1.multiselect("Fuel material",
                           sorted(df["fuel_material"].unique()),
                           default=sorted(df["fuel_material"].unique()))
    bands = f2.multiselect("Burn-intensity band",
                           sorted(df["o2_depletion_band"].unique()),
                           default=sorted(df["o2_depletion_band"].unique()))
    o2_min, o2_max = f3.slider("Initial O₂ % range",
                               float(df["initial_o2"].min()),
                               float(df["initial_o2"].max()),
                               (float(df["initial_o2"].min()),
                                float(df["initial_o2"].max())))

    view = df[(df["fuel_material"].isin(fuels)) &
              (df["o2_depletion_band"].isin(bands)) &
              (df["initial_o2"].between(o2_min, o2_max))]
    st.dataframe(view[["test_id", "fuel_material", "initial_o2", "final_o2",
                       "o2_depletion", "o2_depletion_band", "flow_restrictor",
                       "fan_display", "air_display", "thickness_micron"]],
                 use_container_width=True)
    st.caption(f"Showing {len(view)} of {len(df)} BASS-II tests.")

    g1, g2 = st.columns(2)
    with g1:
        counts = view["o2_depletion_band"].value_counts().reset_index()
        counts.columns = ["band", "count"]
        st.plotly_chart(px.bar(counts, x="band", y="count",
                               title="Burn-intensity proxy band distribution",
                               labels={"band": "Band", "count": "Tests"},
                               color="band",
                               color_discrete_sequence=px.colors.qualitative.Set2),
                        use_container_width=True)
    with g2:
        st.plotly_chart(px.scatter(view, x="initial_o2", y="o2_depletion",
                                   color="o2_depletion_band",
                                   hover_data=["test_id", "fuel_material"],
                                   title="Initial O₂ vs measured O₂ depletion",
                                   labels={"initial_o2": "Initial O₂ %",
                                           "o2_depletion": "O₂ depletion (pp)"}),
                        use_container_width=True)
