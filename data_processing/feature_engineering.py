"""Feature engineering for the BASS-II experimental table.

Parsing helpers and the derived burn-intensity proxy live here so that
data_cleaning.py stays a thin orchestration layer and the ML pipeline
imports feature logic from one place.
"""
import re

import numpy as np
import pandas as pd

_MICRON_RE = re.compile(r"(\d+(?:\.\d+)?)\s*micron", re.IGNORECASE)
_MM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mm", re.IGNORECASE)

DEPLETION_BAND_LABELS = ["low depletion", "moderate depletion", "high depletion"]

DERIVED_TARGET_DESCRIPTION = (
    "o2_depletion_band is a derived proxy, NOT a NASA-provided label: "
    "o2_depletion = initial O2 - final O2 (both measured % by vol), binned "
    "into tertiles. The BASS-II table has no flame-behavior label column."
)


def coerce_numeric(df, column):
    """Return a column as float; an all-NaN series if the column is absent."""
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def parse_thickness_micron(fuel_text):
    """Thickness in microns, parsed from a fuel-sample description string.

    Examples: "100 micron PMMA film" -> 100.0 ; "2 mm rod" -> 2000.0 ;
    anything without a size -> NaN (median-imputed later).
    """
    text = str(fuel_text)
    micron = _MICRON_RE.search(text)
    if micron:
        return float(micron.group(1))
    mm = _MM_RE.search(text)
    if mm:
        return float(mm.group(1)) * 1000.0
    return np.nan


def add_depletion_proxy(df):
    """Add o2_depletion and its tertile band; record cutoffs in df.attrs."""
    df = df.copy()
    df["o2_depletion"] = df["initial_o2"] - df["final_o2"]
    df = df.dropna(subset=["test_id", "o2_depletion"]).reset_index(drop=True)
    df["o2_depletion_band"] = pd.qcut(
        df["o2_depletion"], q=3, labels=DEPLETION_BAND_LABELS
    )
    cutoffs = df["o2_depletion"].quantile([1 / 3, 2 / 3]).tolist()
    df.attrs["depletion_band_cutoffs"] = cutoffs
    df.attrs["derived_target_description"] = DERIVED_TARGET_DESCRIPTION
    return df
