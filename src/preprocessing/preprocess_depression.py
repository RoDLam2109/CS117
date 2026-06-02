"""Preprocessing pipeline for the Student Depression dataset used in module 2."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

DEPRESSION_TARGET_COLUMN = "Depression"
DEPRESSION_FEATURE_COLUMNS = [
    "Academic Pressure",
    "CGPA",
    "Study Satisfaction",
    "Sleep Duration",
    "Dietary Habits",
    "Have you ever had suicidal thoughts ?",
    "Work/Study Hours",
    "Financial Stress",
    "Family History of Mental Illness",
    "Age",
    "Gender",
]

SLEEP_DURATION_MAP = {
    "Less than 5 hours": 0,
    "5-6 hours": 1,
    "7-8 hours": 2,
    "More than 8 hours": 3,
}

DIETARY_HABITS_MAP = {
    "Unhealthy": 0,
    "Moderate": 1,
    "Healthy": 2,
}

YES_NO_MAP = {
    "yes": 1,
    "no": 0,
    "1": 1,
    "0": 0,
}


def _normalise_binary(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(YES_NO_MAP)
        .astype("Int64")
    )


def clean_depression_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned dataframe that keeps only the fields used by module 2."""
    missing_columns = [
        column
        for column in [*DEPRESSION_FEATURE_COLUMNS, DEPRESSION_TARGET_COLUMN]
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Dataset is missing required module 2 columns: "
            + ", ".join(missing_columns)
        )

    cleaned = df.loc[:, [*DEPRESSION_FEATURE_COLUMNS, DEPRESSION_TARGET_COLUMN]].copy()

    cleaned["Sleep Duration"] = cleaned["Sleep Duration"].map(SLEEP_DURATION_MAP)
    cleaned["Dietary Habits"] = cleaned["Dietary Habits"].map(DIETARY_HABITS_MAP)
    cleaned["Have you ever had suicidal thoughts ?"] = _normalise_binary(
        cleaned["Have you ever had suicidal thoughts ?"]
    )
    cleaned["Family History of Mental Illness"] = _normalise_binary(
        cleaned["Family History of Mental Illness"]
    )
    cleaned[DEPRESSION_TARGET_COLUMN] = pd.to_numeric(
        cleaned[DEPRESSION_TARGET_COLUMN], errors="coerce"
    )

    numeric_columns: Iterable[str] = [
        "Academic Pressure",
        "CGPA",
        "Study Satisfaction",
        "Sleep Duration",
        "Dietary Habits",
        "Have you ever had suicidal thoughts ?",
        "Work/Study Hours",
        "Financial Stress",
        "Family History of Mental Illness",
        "Age",
    ]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned["Gender"] = cleaned["Gender"].astype(str).str.strip().str.title()
    cleaned = cleaned.dropna(subset=[DEPRESSION_TARGET_COLUMN])
    cleaned[DEPRESSION_TARGET_COLUMN] = cleaned[DEPRESSION_TARGET_COLUMN].astype(int)

    return cleaned


def preprocess_depression(data: str | Path | pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Load and clean the Student Depression dataset for module 2."""
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.read_csv(Path(data))

    cleaned = clean_depression_dataframe(df)
    X = cleaned[DEPRESSION_FEATURE_COLUMNS].copy()
    y = cleaned[DEPRESSION_TARGET_COLUMN].copy()
    return X, y
