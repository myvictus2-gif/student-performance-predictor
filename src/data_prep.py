"""
data_prep.py - loading and cleaning
==================================

Cleaning steps (each one is counted in the cleaning report):
  1. Remove exact duplicate rows.
  2. Turn impossible values (e.g. attendance 120 %, -2 study hours) into
     missing values, because they are data-entry errors, not real values.
  3. Drop rows whose *target* (final marks) is missing: we cannot learn from them.
  4. Missing *input* values are filled later, inside the model pipeline, with the
     median of the training data (SimpleImputer). Doing it inside the pipeline
     means the test data never influences the fill values (no data leakage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load_raw() -> tuple[pd.DataFrame, str]:
    """Return the raw data and a description of where it came from.

    If the Kaggle file StudentPerformanceFactors.csv is in data/, it is used
    (its columns are renamed to this project's names). Otherwise the synthetic
    practice dataset is used.
    """
    if config.KAGGLE_FILE.exists():
        df = pd.read_csv(config.KAGGLE_FILE)
        df = df.rename(columns={
            "Hours_Studied": "study_hours", "Attendance": "attendance",
            "Sleep_Hours": "sleep_hours", "Previous_Scores": "previous_marks",
            "Exam_Score": "final_marks",
        })
        missing = [c for c in config.FEATURES + [config.TARGET] if c not in df.columns]
        if missing:
            raise ValueError(f"Kaggle file is missing expected columns: {missing}")
        source = ("Kaggle 'Student Performance Factors' (StudentPerformanceFactors.csv). "
                  "Note: in this file study hours are recorded as given by the source; "
                  "check the Kaggle page for their unit.")
        return df[config.FEATURES + [config.TARGET]], source

    df = pd.read_csv(config.SYNTHETIC_FILE)
    source = ("Synthetic practice dataset (data/generate_data.py): 1,000 students generated "
              "from documented rules, with deliberate errors added for the cleaning step.")
    return df[config.FEATURES + [config.TARGET]], source


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"rows_loaded": int(len(df)),
              "missing_before": {c: int(df[c].isna().sum()) for c in df.columns}}

    # 1. duplicates
    dup = df.duplicated()
    report["duplicates_removed"] = int(dup.sum())
    df = df[~dup].copy()

    # 2. impossible values -> missing
    invalid = {}
    for col, (lo, hi) in config.VALID_RANGES.items():
        bad = (df[col] < lo) | (df[col] > hi)
        invalid[col] = int(bad.sum())
        df.loc[bad, col] = np.nan
    report["impossible_values_set_missing"] = invalid

    # 3. rows without a target
    no_target = df[config.TARGET].isna()
    report["rows_without_target_removed"] = int(no_target.sum())
    df = df[~no_target].reset_index(drop=True)

    report["missing_after_cleaning"] = {c: int(df[c].isna().sum()) for c in config.FEATURES}
    report["rows_after_cleaning"] = int(len(df))
    return df, report


def outlier_table(df: pd.DataFrame) -> pd.DataFrame:
    """IQR outliers per column. They are valid values, so they are reported, not removed."""
    rows = []
    for col in config.FEATURES + [config.TARGET]:
        s = df[col].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((s < lo) | (s > hi)).sum())
        rows.append({"Column": config.LABELS[col], "Lower limit": round(lo, 1),
                     "Upper limit": round(hi, 1), "Outliers": n,
                     "Share": f"{100 * n / len(s):.1f}%"})
    return pd.DataFrame(rows)


def load_clean() -> tuple[pd.DataFrame, dict, str]:
    raw, source = load_raw()
    df, report = clean(raw)
    return df, report, source
