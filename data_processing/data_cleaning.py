"""Cleaning orchestration for the BASS-II experimental table.

Feature-building primitives live in feature_engineering.py; this module
wires them into the single clean_experiments() step used by the ML
pipeline, the dashboard, and the exploration notebook.
"""
from data_processing.feature_engineering import (
    add_depletion_proxy,
    coerce_numeric,
    parse_thickness_micron,
)


def clean_experiments(df):
    """Full cleaning pass: tidy columns, typed features, depletion proxy."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.drop_duplicates().reset_index(drop=True)

    df["test_id"] = df["Test #"].astype(str).str.strip()
    df["fuel_material"] = df["Fuel Sample Material"].astype(str).str.strip()

    df["initial_o2"] = coerce_numeric(df, "Calibrated  initial O2 % by vol")
    df["final_o2"] = coerce_numeric(df, "Calibrated final O2 % by vol")
    df["fan_display"] = coerce_numeric(df, "Fan display")
    df["air_display"] = coerce_numeric(df, "Air display")
    df["flow_restrictor"] = coerce_numeric(df, "Flow restrictor")
    df["thickness_micron"] = df["Fuel Sample Material"].map(
        parse_thickness_micron)

    df = add_depletion_proxy(df)
    cutoffs = df.attrs.get("depletion_band_cutoffs", [])

    for column in ["initial_o2", "fan_display", "air_display",
                   "flow_restrictor", "thickness_micron"]:
        df[column] = df[column].fillna(df[column].median())

    print(f"clean_experiments: {len(df)} rows, "
          f"depletion tertile cutoffs: {[round(c, 2) for c in cutoffs]}")
    return df
