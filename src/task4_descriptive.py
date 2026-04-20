"""
Task 4 — Group descriptive statistics for cDP vs wDP cohorts.

Reads the per-visit and per-patient labels produced by task4_progression.py
and writes a numeric and categorical summary workbook.

Run with:    python task4_descriptive.py
Or via:      make task4
"""

from pathlib import Path

import numpy as np
import pandas as pd

from paths import TASK4_OUT, ensure_dirs

INPUT_FILE = TASK4_OUT / "task4_progression_outputs.xlsx"


def build_descriptive_stats():
    """Build the numeric descriptive statistics dataframe (cDP vs wDP)."""
    visit = pd.read_excel(INPUT_FILE, sheet_name="Visit_level_labels")
    patient = pd.read_excel(INPUT_FILE, sheet_name="Patient_level_labels")

    # first visit per subject = baseline row
    baseline = (
        visit.sort_values(["subject_id", "EDSSDateYYYYMM"])
             .drop_duplicates("subject_id", keep="first")
             .copy()
    )

    # merge final patient labels onto baseline data
    df = baseline.merge(
        patient[["subject_id", "patient_status", "n_edss_visits",
                 "time_to_confirmation_months"]],
        on="subject_id",
        how="left",
    )

    # follow-up duration based on EDSS dates
    visit["EDSSDateYYYYMM"] = pd.to_datetime(visit["EDSSDateYYYYMM"], errors="coerce")
    followup = (
        visit.groupby("subject_id")["EDSSDateYYYYMM"]
             .agg(["min", "max"])
             .reset_index()
    )
    followup["followup_months"] = (followup["max"] - followup["min"]).dt.days / 30.4375

    df = df.merge(followup[["subject_id", "followup_months"]],
                  on="subject_id", how="left")

    # numeric variables to summarize
    numeric_vars = [
        "age_years",
        "EDSSValue",
        "disease_duration_years",
        "n_edss_visits",
        "followup_months",
        "Brain (WM+GM) volume cm3",
        "Grey Matter (GM) volume cm3",
        "White Matter (WM) volume cm3",
        "Lateral ventricle total volume cm3",
        "lesionvolume",
        "lesioncount",
    ]

    rows = []
    for var in numeric_vars:
        if var not in df.columns:
            continue
        for grp in ["wDP", "cDP"]:
            s = pd.to_numeric(df.loc[df["patient_status"] == grp, var],
                              errors="coerce").dropna()
            rows.append({
                "group": grp,
                "variable": var,
                "n": int(s.size),
                "mean": float(s.mean()) if len(s) else np.nan,
                "sd": float(s.std()) if len(s) else np.nan,
                "median": float(s.median()) if len(s) else np.nan,
                "q1": float(s.quantile(0.25)) if len(s) else np.nan,
                "q3": float(s.quantile(0.75)) if len(s) else np.nan,
                "min": float(s.min()) if len(s) else np.nan,
                "max": float(s.max()) if len(s) else np.nan,
            })

    desc_numeric = pd.DataFrame(rows)

    # categorical summaries
    cat_vars = ["sex_id", "DiagnosisValue", "therapy_std", "location"]
    cat_tables = {}
    for var in cat_vars:
        if var in df.columns:
            tab = (
                df.groupby(["patient_status", var], dropna=False)
                  .size()
                  .reset_index(name="count")
            )
            tab["pct_within_group"] = tab.groupby("patient_status")["count"].transform(
                lambda x: 100 * x / x.sum()
            )
            cat_tables[var] = tab

    return df, desc_numeric, cat_tables


def main():
    ensure_dirs()
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Required input not found: {INPUT_FILE}\n"
            "Run task4_progression.py first (or `make task4`)."
        )

    df, desc_numeric, cat_tables = build_descriptive_stats()

    out = TASK4_OUT / "task4_group_descriptive_stats.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        desc_numeric.to_excel(writer, sheet_name="numeric_summary", index=False)
        for name, tab in cat_tables.items():
            tab.to_excel(writer, sheet_name=name[:31], index=False)

    print(f"Saved: {out}")
    print(df["patient_status"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
