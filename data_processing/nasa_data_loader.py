"""Loaders for the project's data files.

- load_experiments(): reads the real BASS-II experimental table CSV and
  validates that the columns the pipeline needs are present.
- load_data_inventory(): reads the team's research table
  (data/metadata/data_inventory.csv).
"""
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "Test #",
    "Fuel Sample Material",
    "Calibrated  initial O2 % by vol",
    "Calibrated final O2 % by vol",
}
OPTIONAL_COLUMNS = {
    "Flow restrictor",
    "Fan display",
    "Air display",
    "Initial CO2 % by vol",
    "Final CO2 % by vol",
    "Initial CO (ppm)",
    "Final CO (ppm)",
}

DEFAULT_EXPERIMENTS_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "experiments" / "bass2_experimental_table.csv"
)


def load_experiments(path: str | Path = DEFAULT_EXPERIMENTS_PATH) -> pd.DataFrame:
    """Load the BASS-II experimental table, validating required columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Experiment file not found: {path}\n"
            "Expected the real BASS-II table at "
            "data/experiments/bass2_experimental_table.csv"
        )

    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {sorted(missing)}\n"
            f"Columns found: {list(df.columns)}"
        )
    absent_optional = OPTIONAL_COLUMNS - set(df.columns)
    if absent_optional:
        print(f"Note: optional columns absent from {path.name}: "
              f"{sorted(absent_optional)} (filled with NaN downstream)")

    return df


def load_data_inventory(path: str | Path = "data/metadata/data_inventory.csv") -> pd.DataFrame:
    """Load the team's data-inventory table (Sadia's research task output)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data inventory not found: {path}")
    return pd.read_csv(path)
