from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================
from paths import TASK2_OUT, TASK4_OUT, ensure_dirs

# Default input: cleaned all-visits sheet produced by Task 2
INPUT_FILE = TASK2_OUT / "task2_eda_tables.xlsx"
INPUT_SHEET = "Cleaned_all_visits"

OUTPUT_DIR = TASK4_OUT

# Core columns
COL_SUBJECT = "subject_id"
COL_SCAN_DATE = "EDSSDateYYYYMM"
COL_EDSS = "EDSSValue"

# Optional columns to retain in outputs if present
OPTIONAL_KEEP_COLS = [
    "image_session_id",
    "sex_id",
    "age_years",
    "DiagnosisValue",
    "TherapyName",
    "therapy_std",
    "location",
]

# Plot colors
FIG_BG = "#f0f0f0"
AX_BG = "#fafafa"
BLUE = "#4c78a8"
GREEN = "#59a14f"
ORANGE = "#f28e2b"
RED = "#e15759"


# ============================================================
# Utility functions
# ============================================================
def parse_date(series: pd.Series) -> pd.Series:
    """
    Parse mixed date input safely.

    Handles:
    - YYYYMM numeric or string, e.g. 201402
    - already-datetime input
    - other parseable string formats
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    s = series.astype("string").str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)

    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    yyyymm_mask = s.str.fullmatch(r"\d{6}", na=False)
    parsed.loc[yyyymm_mask] = pd.to_datetime(s.loc[yyyymm_mask], format="%Y%m", errors="coerce")

    remaining_mask = ~yyyymm_mask
    if remaining_mask.any():
        parsed.loc[remaining_mask] = pd.to_datetime(s.loc[remaining_mask], errors="coerce")

    return parsed


def months_between(a: pd.Timestamp, b: pd.Timestamp) -> float:
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return (b - a).days / 30.4375


def threshold_from_baseline_edss(baseline_edss: float) -> float:
    """
    Task 4 EDSS worsening threshold:
      - baseline EDSS < 1.0  -> +1.5
      - baseline EDSS 1.0 to 5.5 -> +1.0
      - baseline EDSS > 5.5 -> +0.5
    """
    if baseline_edss < 1.0:
        return 1.5
    if baseline_edss <= 5.5:
        return 1.0
    return 0.5


def set_plot_style() -> None:
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.titlesize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })


def save_figure(fig: plt.Figure, filename: str, dpi: int = 300) -> Path:
    out = OUTPUT_DIR / filename
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


# ============================================================
# Loading and preparation
# ============================================================
def load_input(path: Path, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name)


def prepare_edss_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep a copy of excluded rows for transparency.
    Task 4 needs subject, date, and EDSS.
    """
    x = df.copy()

    x[COL_SCAN_DATE] = parse_date(x[COL_SCAN_DATE])
    x[COL_EDSS] = pd.to_numeric(x[COL_EDSS], errors="coerce")

    required_mask = (
        x[COL_SUBJECT].notna() &
        x[COL_SCAN_DATE].notna() &
        x[COL_EDSS].notna()
    )

    excluded = x.loc[~required_mask].copy()
    usable = x.loc[required_mask].copy()

    usable = usable.sort_values([COL_SUBJECT, COL_SCAN_DATE]).reset_index(drop=True)
    excluded = excluded.reset_index(drop=True)

    return usable, excluded


# ============================================================
# Core Task 4 labeling logic
# ============================================================
def label_one_patient(g: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    For one patient:
    - baseline = first available EDSS visit
    - candidate progression = EDSS increase vs baseline meets threshold
    - confirmed progression = candidate worsening remains present at a visit
      at least 6 months later

    Returns:
    - visit-level labeled dataframe
    - patient-level summary row
    """
    g = g.sort_values(COL_SCAN_DATE).copy().reset_index(drop=True)

    # Initialize output columns
    g["baseline_date"] = pd.NaT
    g["baseline_edss"] = np.nan
    g["dp_threshold"] = np.nan
    g["delta_edss_from_baseline"] = np.nan
    g["months_from_baseline"] = np.nan

    g["candidate_DP_visit"] = 0
    g["confirmed_cDP_onset_visit"] = 0
    g["confirmation_visit"] = 0
    g["cDP_visit_post_onset"] = 0
    g["cDP_visit_post_confirmation"] = 0

    patient_id = g[COL_SUBJECT].iloc[0]

    # baseline = first visit with valid EDSS after preprocessing
    baseline_date = g.loc[0, COL_SCAN_DATE]
    baseline_edss = float(g.loc[0, COL_EDSS])
    thr = threshold_from_baseline_edss(baseline_edss)

    g["baseline_date"] = baseline_date
    g["baseline_edss"] = baseline_edss
    g["dp_threshold"] = thr
    g["delta_edss_from_baseline"] = g[COL_EDSS] - baseline_edss
    g["months_from_baseline"] = g[COL_SCAN_DATE].apply(lambda d: months_between(baseline_date, d))

    # Need at least two EDSS visits to assess progression
    n_visits = len(g)
    if n_visits < 2:
        patient_row = {
            COL_SUBJECT: patient_id,
            "n_edss_visits": n_visits,
            "baseline_date": baseline_date,
            "baseline_edss": baseline_edss,
            "dp_threshold": thr,
            "has_candidate_DP": 0,
            "has_confirmed_cDP": 0,
            "patient_status": "wDP",
            "cDP_onset_date": pd.NaT,
            "confirmation_date": pd.NaT,
            "time_to_onset_months": np.nan,
            "time_to_confirmation_months": np.nan,
        }
        return g, patient_row

    # candidate progression visits: later visits meeting threshold vs baseline
    candidate_mask = (
        (g[COL_SCAN_DATE] > baseline_date) &
        (g["delta_edss_from_baseline"] >= thr)
    )
    g.loc[candidate_mask, "candidate_DP_visit"] = 1

    candidate_indices = g.index[candidate_mask].tolist()

    confirmed_onset_idx = None
    confirmation_idx = None

    # Find earliest candidate that has confirmation >= 6 months later
    for idx in candidate_indices:
        candidate_date = g.loc[idx, COL_SCAN_DATE]
        confirm_from_date = candidate_date + pd.DateOffset(months=6)

        later_confirm_mask = (
            (g[COL_SCAN_DATE] >= confirm_from_date) &
            (g["delta_edss_from_baseline"] >= thr)
        )

        confirm_indices = g.index[later_confirm_mask].tolist()
        if len(confirm_indices) > 0:
            confirmed_onset_idx = idx
            confirmation_idx = confirm_indices[0]
            break

    if confirmed_onset_idx is None:
        patient_row = {
            COL_SUBJECT: patient_id,
            "n_edss_visits": n_visits,
            "baseline_date": baseline_date,
            "baseline_edss": baseline_edss,
            "dp_threshold": thr,
            "has_candidate_DP": int(candidate_mask.any()),
            "has_confirmed_cDP": 0,
            "patient_status": "wDP",
            "cDP_onset_date": pd.NaT,
            "confirmation_date": pd.NaT,
            "time_to_onset_months": np.nan,
            "time_to_confirmation_months": np.nan,
        }
        return g, patient_row

    # Mark onset and confirmation
    onset_date = g.loc[confirmed_onset_idx, COL_SCAN_DATE]
    confirm_date = g.loc[confirmation_idx, COL_SCAN_DATE]

    g.loc[confirmed_onset_idx, "confirmed_cDP_onset_visit"] = 1
    g.loc[confirmation_idx, "confirmation_visit"] = 1

    # Visits at/after onset that still meet threshold
    mask_post_onset = (
        (g[COL_SCAN_DATE] >= onset_date) &
        (g["delta_edss_from_baseline"] >= thr)
    )
    g.loc[mask_post_onset, "cDP_visit_post_onset"] = 1

    # Visits at/after first confirmation that still meet threshold
    mask_post_confirm = (
        (g[COL_SCAN_DATE] >= confirm_date) &
        (g["delta_edss_from_baseline"] >= thr)
    )
    g.loc[mask_post_confirm, "cDP_visit_post_confirmation"] = 1

    patient_row = {
        COL_SUBJECT: patient_id,
        "n_edss_visits": n_visits,
        "baseline_date": baseline_date,
        "baseline_edss": baseline_edss,
        "dp_threshold": thr,
        "has_candidate_DP": int(candidate_mask.any()),
        "has_confirmed_cDP": 1,
        "patient_status": "cDP",
        "cDP_onset_date": onset_date,
        "confirmation_date": confirm_date,
        "time_to_onset_months": months_between(baseline_date, onset_date),
        "time_to_confirmation_months": months_between(baseline_date, confirm_date),
    }

    return g, patient_row


def run_task4_labeling(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    visit_parts: List[pd.DataFrame] = []
    patient_parts: List[Dict[str, object]] = []

    for _, g in df.groupby(COL_SUBJECT, sort=False):
        labeled_visits, patient_row = label_one_patient(g)
        visit_parts.append(labeled_visits)
        patient_parts.append(patient_row)

    visit_df = pd.concat(visit_parts, axis=0).reset_index(drop=True)
    patient_df = pd.DataFrame(patient_parts).sort_values(COL_SUBJECT).reset_index(drop=True)

    return visit_df, patient_df


# ============================================================
# Tables and summaries
# ============================================================
def build_summary_tables(visit_df: pd.DataFrame, patient_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    visit_counts = pd.DataFrame([
        {
            "metric": "Total evaluable EDSS visits",
            "value": int(len(visit_df)),
        },
        {
            "metric": "Candidate DP visits",
            "value": int(visit_df["candidate_DP_visit"].sum()),
        },
        {
            "metric": "Confirmed cDP onset visits",
            "value": int(visit_df["confirmed_cDP_onset_visit"].sum()),
        },
        {
            "metric": "Confirmation visits",
            "value": int(visit_df["confirmation_visit"].sum()),
        },
        {
            "metric": "cDP visits post onset",
            "value": int(visit_df["cDP_visit_post_onset"].sum()),
        },
        {
            "metric": "cDP visits post confirmation",
            "value": int(visit_df["cDP_visit_post_confirmation"].sum()),
        },
    ])

    patient_counts = pd.DataFrame([
        {
            "metric": "Total patients evaluated",
            "value": int(len(patient_df)),
        },
        {
            "metric": "Patients with candidate DP",
            "value": int(patient_df["has_candidate_DP"].sum()),
        },
        {
            "metric": "Patients with confirmed cDP",
            "value": int(patient_df["has_confirmed_cDP"].sum()),
        },
        {
            "metric": "Patients without confirmed DP (wDP)",
            "value": int((patient_df["patient_status"] == "wDP").sum()),
        },
    ])

    if len(patient_df) > 0:
        patient_counts["pct"] = np.where(
            patient_counts["metric"] == "Total patients evaluated",
            100.0,
            patient_counts["value"] / len(patient_df) * 100.0,
        )

    if len(visit_df) > 0:
        visit_counts["pct_of_visits"] = visit_counts["value"] / len(visit_df) * 100.0

    cdp_times = patient_df.loc[patient_df["has_confirmed_cDP"] == 1, "time_to_confirmation_months"].dropna()
    time_summary = pd.DataFrame([{
        "n_confirmed_cDP_patients": int(len(cdp_times)),
        "mean_time_to_confirmation_months": float(cdp_times.mean()) if len(cdp_times) else np.nan,
        "median_time_to_confirmation_months": float(cdp_times.median()) if len(cdp_times) else np.nan,
        "iqr_low": float(cdp_times.quantile(0.25)) if len(cdp_times) else np.nan,
        "iqr_high": float(cdp_times.quantile(0.75)) if len(cdp_times) else np.nan,
        "min": float(cdp_times.min()) if len(cdp_times) else np.nan,
        "max": float(cdp_times.max()) if len(cdp_times) else np.nan,
    }])

    return {
        "visit_label_summary": visit_counts,
        "patient_label_summary": patient_counts,
        "time_to_cDP_summary": time_summary,
    }


def build_summary_text(usable_df: pd.DataFrame, excluded_df: pd.DataFrame, visit_df: pd.DataFrame, patient_df: pd.DataFrame) -> str:
    n_patients = len(patient_df)
    n_visits = len(visit_df)

    n_candidate_patients = int(patient_df["has_candidate_DP"].sum())
    n_confirmed_patients = int(patient_df["has_confirmed_cDP"].sum())
    n_wdp_patients = int((patient_df["patient_status"] == "wDP").sum())

    n_candidate_visits = int(visit_df["candidate_DP_visit"].sum())
    n_confirmed_onset_visits = int(visit_df["confirmed_cDP_onset_visit"].sum())
    n_confirm_post_onset = int(visit_df["cDP_visit_post_onset"].sum())
    n_confirm_post_confirmation = int(visit_df["cDP_visit_post_confirmation"].sum())

    lines = [
        "Task 4 disability progression summary",
        "====================================",
        f"Input evaluable EDSS visits: {len(usable_df)}",
        f"Excluded rows without subject/date/EDSS: {len(excluded_df)}",
        "",
        f"Patients evaluated: {n_patients}",
        f"Visit-level rows evaluated: {n_visits}",
        "",
        "Patient-level results",
        f"- Candidate DP patients: {n_candidate_patients}/{n_patients} ({100 * n_candidate_patients / n_patients:.1f}%)" if n_patients else "- Candidate DP patients: 0",
        f"- Confirmed cDP patients: {n_confirmed_patients}/{n_patients} ({100 * n_confirmed_patients / n_patients:.1f}%)" if n_patients else "- Confirmed cDP patients: 0",
        f"- Without confirmed DP (wDP): {n_wdp_patients}/{n_patients} ({100 * n_wdp_patients / n_patients:.1f}%)" if n_patients else "- Without confirmed DP (wDP): 0",
        "",
        "Visit-level results",
        f"- Candidate DP visits: {n_candidate_visits}/{n_visits} ({100 * n_candidate_visits / n_visits:.1f}%)" if n_visits else "- Candidate DP visits: 0",
        f"- Confirmed cDP onset visits: {n_confirmed_onset_visits}/{n_visits} ({100 * n_confirmed_onset_visits / n_visits:.1f}%)" if n_visits else "- Confirmed cDP onset visits: 0",
        f"- cDP visits post onset: {n_confirm_post_onset}/{n_visits} ({100 * n_confirm_post_onset / n_visits:.1f}%)" if n_visits else "- cDP visits post onset: 0",
        f"- cDP visits post confirmation: {n_confirm_post_confirmation}/{n_visits} ({100 * n_confirm_post_confirmation / n_visits:.1f}%)" if n_visits else "- cDP visits post confirmation: 0",
    ]

    confirmed_times = patient_df.loc[patient_df["has_confirmed_cDP"] == 1, "time_to_confirmation_months"].dropna()
    if len(confirmed_times) > 0:
        lines.extend([
            "",
            "Time to confirmation",
            f"- Mean: {confirmed_times.mean():.1f} months",
            f"- Median: {confirmed_times.median():.1f} months",
            f"- IQR: {confirmed_times.quantile(0.25):.1f}-{confirmed_times.quantile(0.75):.1f} months",
        ])

    return "\n".join(lines)


# ============================================================
# Figures
# ============================================================
def plot_patient_status(patient_df: pd.DataFrame) -> Path:
    set_plot_style()

    counts = patient_df["patient_status"].value_counts().reindex(["wDP", "cDP"], fill_value=0)

    fig, ax = plt.subplots(figsize=(6.5, 5), facecolor=FIG_BG)
    ax.set_facecolor(AX_BG)

    ax.bar(counts.index, counts.values, color=[BLUE, ORANGE], edgecolor="white")
    ax.set_title("Figure 1. Patient-level disability progression labels")
    ax.set_xlabel("Patient label")
    ax.set_ylabel("Patients")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    for i, v in enumerate(counts.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=11)

    fig.tight_layout()
    return save_figure(fig, "Figure_1_patient_level_progression_labels.png")


def plot_visit_status(visit_df: pd.DataFrame) -> Path:
    set_plot_style()

    summary = pd.Series({
        "Candidate DP": int(visit_df["candidate_DP_visit"].sum()),
        "cDP onset": int(visit_df["confirmed_cDP_onset_visit"].sum()),
        "Post-onset cDP": int(visit_df["cDP_visit_post_onset"].sum()),
        "Post-confirm cDP": int(visit_df["cDP_visit_post_confirmation"].sum()),
    })

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=FIG_BG)
    ax.set_facecolor(AX_BG)

    ax.bar(summary.index, summary.values, color=[BLUE, ORANGE, GREEN, RED], edgecolor="white")
    ax.set_title("Figure 2. Visit-level progression labels")
    ax.set_ylabel("Visits")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    for i, v in enumerate(summary.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    return save_figure(fig, "Figure_2_visit_level_progression_labels.png")


def plot_time_to_confirmation(patient_df: pd.DataFrame) -> Path | None:
    set_plot_style()

    times = patient_df.loc[patient_df["has_confirmed_cDP"] == 1, "time_to_confirmation_months"].dropna()
    if len(times) == 0:
        return None

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=FIG_BG)
    ax.set_facecolor(AX_BG)

    ax.hist(times, bins=15, color=GREEN, edgecolor="white")
    ax.set_title("Figure 3. Time to confirmed disability progression")
    ax.set_xlabel("Months from baseline to first confirmation")
    ax.set_ylabel("Patients")
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return save_figure(fig, "Figure_3_time_to_confirmed_progression.png")


# ============================================================
# Main
# ============================================================
def main() -> None:
    ensure_dirs()
    raw_df = load_input(INPUT_FILE, INPUT_SHEET)
    usable_df, excluded_df = prepare_edss_data(raw_df)

    visit_df, patient_df = run_task4_labeling(usable_df)
    summary_tables = build_summary_tables(visit_df, patient_df)

    # Reorder visit-level output for readability
    preferred_cols = [
        COL_SUBJECT,
        COL_SCAN_DATE,
        COL_EDSS,
        "baseline_date",
        "baseline_edss",
        "dp_threshold",
        "delta_edss_from_baseline",
        "months_from_baseline",
        "candidate_DP_visit",
        "confirmed_cDP_onset_visit",
        "confirmation_visit",
        "cDP_visit_post_onset",
        "cDP_visit_post_confirmation",
    ]
    keep_cols = [c for c in preferred_cols if c in visit_df.columns]
    extra_cols = [c for c in OPTIONAL_KEEP_COLS if c in visit_df.columns and c not in keep_cols]
    other_cols = [c for c in visit_df.columns if c not in keep_cols + extra_cols]
    visit_df = visit_df[keep_cols + extra_cols + other_cols]

    # Figures
    fig_paths = [
        plot_patient_status(patient_df),
        plot_visit_status(visit_df),
    ]
    fig3 = plot_time_to_confirmation(patient_df)
    if fig3 is not None:
        fig_paths.append(fig3)

    # Save outputs
    workbook = OUTPUT_DIR / "task4_progression_outputs.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        visit_df.to_excel(writer, sheet_name="Visit_level_labels", index=False)
        patient_df.to_excel(writer, sheet_name="Patient_level_labels", index=False)
        excluded_df.to_excel(writer, sheet_name="Excluded_rows", index=False)

        for name, tbl in summary_tables.items():
            tbl.to_excel(writer, sheet_name=name[:31], index=False)

    summary_txt = build_summary_text(
        usable_df=usable_df,
        excluded_df=excluded_df,
        visit_df=visit_df,
        patient_df=patient_df,
    )
    summary_path = OUTPUT_DIR / "task4_progression_summary.txt"
    summary_path.write_text(summary_txt, encoding="utf-8")

    print(summary_txt)
    print("\nSaved outputs:")
    print(f"  - Workbook: {workbook}")
    print(f"  - Summary : {summary_path}")
    print("  - Figures :")
    for p in fig_paths:
        print(f"      {p}")


if __name__ == "__main__":
    main()