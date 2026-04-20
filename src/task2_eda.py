"""
MS Longitudinal Cohort — Exploratory Data Analysis Pipeline
============================================================
Improvements over original:
- Single configuration block (no mid-file re-assignments)
- Dates parsed once in a dedicated preprocessing step
- Column validation at load time with clear error messages
- month_diff uses day-accurate calculation consistent with disease_duration_years
- safe_skew emits a warning when returning NaN
- Table 1 includes categorical summaries (sex, diagnosis, therapy)
- Excel sheet name truncation is validated for uniqueness
- normalize_sex coding convention is documented
- resolve_duplicates no longer parses dates (done upstream)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import skew

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    format="%(levelname)s | %(funcName)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ============================================================
# Configuration  (edit only this block)
# ============================================================
from paths import DATA_PATH, TASK2_OUT, ensure_dirs, assert_data_exists

INPUT_FILE  = DATA_PATH
SHEET_NAME  = 0
OUTPUT_DIR  = TASK2_OUT

# Column names
COL_IMAGE_SESSION    = "image_session_id"
COL_SUBJECT          = "subject_id"
COL_SCAN_DATE        = "MRIDateYYYYMM"
COL_SEX              = "sex_id"
COL_AGE              = "age_years"
COL_EDSS             = "EDSSValue"
COL_LOCATION         = "location"
COL_EDUCATION        = "EducationYY"
COL_PROGRESSION_DATE = "ProgressionYYYYMM"
COL_DIAGNOSIS_DATE_00= "DiagnosisDateYYYYMM_00"
COL_DIAGNOSIS_DATE   = "DiagnosisDateYYYYMM"
COL_DIAGNOSIS        = "DiagnosisValue"
COL_THERAPY          = "TherapyName"
COL_BRAIN            = "Brain (WM+GM) volume cm3"
COL_GM               = "Grey Matter (GM) volume cm3"
COL_WM               = "White Matter (WM) volume cm3"
COL_VENT             = "Lateral ventricle total volume cm3"
COL_LESION_VOL       = "lesionvolume"
COL_LESION_COUNT     = "lesioncount"
COL_LOG_LESION_VOL   = "log_lesionvolume"

# Columns that must exist in the raw file
REQUIRED_COLUMNS: List[str] = [
    COL_IMAGE_SESSION, COL_SUBJECT, COL_SCAN_DATE, COL_SEX, COL_AGE,
    COL_EDSS, COL_BRAIN, COL_GM, COL_WM, COL_VENT,
    COL_LESION_VOL, COL_LESION_COUNT, COL_THERAPY, COL_DIAGNOSIS,
    COL_DIAGNOSIS_DATE_00, COL_DIAGNOSIS_DATE,
]

# Plot palette
FIG_BG = "#eaeaea"
AX_BG  = "#f5f5f5"
BLUE   = "#4c78a8"
GREEN  = "#59a14f"
PURPLE = "#9c6ade"
TEAL   = "#4e9f9d"
ORANGE = "#f28e2b"


# ============================================================
# Utility helpers
# ============================================================
def parse_date(series: pd.Series) -> pd.Series:
    """
    Parse mixed date input safely.

    Handles:
    - YYYYMM numeric or string (e.g. 201402)  → first day of that month
    - Existing datetime-like values
    - Other pandas-parseable date strings

    Returns a datetime64[ns] Series; unparseable values become NaT.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    s = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)

    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    yyyymm_mask = s.str.fullmatch(r"\d{6}", na=False)
    parsed.loc[yyyymm_mask] = pd.to_datetime(
        s.loc[yyyymm_mask], format="%Y%m", errors="coerce"
    )

    remaining = ~yyyymm_mask
    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(s.loc[remaining], errors="coerce")

    return parsed


def days_between(a: pd.Timestamp, b: pd.Timestamp) -> float:
    """
    Exact calendar-day difference between two timestamps.
    Returns NaN if either value is missing.
    """
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return (b - a).days


def month_diff(a: pd.Timestamp, b: pd.Timestamp) -> float:
    """
    Difference in months expressed as a float using day-accurate arithmetic
    (consistent with disease_duration_years which uses .dt.days / 365.25).

    Returns NaN if either value is missing.
    """
    d = days_between(a, b)
    return d / 30.4375 if not np.isnan(d) else np.nan  # 365.25 / 12


def normalize_sex(val: object) -> str:
    """
    Map raw sex codes to 'F', 'M', or 'Unknown'.

    Assumed coding convention:
        1  → Male
        2  → Female
    (common in many clinical databases; update mapping if yours differs)
    """
    if pd.isna(val):
        return "Unknown"
    s = str(val).strip().upper()
    if s in {"F", "FEMALE", "2"}:
        return "F"
    if s in {"M", "MALE", "1"}:
        return "M"
    return "Unknown"


def standardize_therapy(val: object) -> str | float:
    """Harmonize brand names, generics, and non-English labels to canonical INN names."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if not s:
        return np.nan
    s = " ".join(s.replace("_", " ").replace("-", " ").split())

    exact: Dict[str, str] = {
        "tecfidera":            "dimethyl fumarate",
        "dimethyl fumarate":    "dimethyl fumarate",
        "fingolimod":           "fingolimod",
        "gilenya":              "fingolimod",
        "tysabri":              "natalizumab",
        "natalizumab":          "natalizumab",
        "aubagio":              "teriflunomide",
        "teriflunomide":        "teriflunomide",
        "copaxone":             "glatiramer acetate",
        "glatiramer acetate":   "glatiramer acetate",
        "rebif":                "interferon beta",
        "avonex":               "interferon beta",
        "betaferon":            "interferon beta",
        "extavia":              "interferon beta",
        "interferon beta":      "interferon beta",
        "plegridy":             "peginterferon beta-1a",
        "peginterferon beta 1a":"peginterferon beta-1a",
        "ocop":                 "ocrelizumab",
        "ocrevus":              "ocrelizumab",
        "ocrelizumab":          "ocrelizumab",
        "rituximab":            "rituximab",
        "lemtrada":             "alemtuzumab",
        "alemtuzumab":          "alemtuzumab",
        "mavenclad":            "cladribine",
        "cladribine":           "cladribine",
        "no therapy":           "no therapy",
        "none":                 "no therapy",
        # Slovenian
        "brez terapije":        "no therapy",
        "brez terapij":         "no therapy",
        "brez tx":              "no therapy",
        "0":                    "no therapy",
    }
    if s in exact:
        return exact[s]

    substring_map: List[Tuple[Tuple[str, ...], str]] = [
        (("tecfidera", "dimethyl"),              "dimethyl fumarate"),
        (("gilenya", "fingolimod"),              "fingolimod"),
        (("tysabri", "natalizumab"),             "natalizumab"),
        (("aubagio", "teriflunomide"),           "teriflunomide"),
        (("copaxone", "glatiramer"),             "glatiramer acetate"),
        (("rebif", "avonex", "betaferon",
          "extavia", "interferon"),              "interferon beta"),
        (("plegridy", "peginterferon"),          "peginterferon beta-1a"),
        (("ocrevus", "ocrelizumab"),             "ocrelizumab"),
        (("ritux",),                             "rituximab"),
        (("lemtrada", "alemtuzumab"),            "alemtuzumab"),
        (("mavenclad", "cladribine"),            "cladribine"),
        (("no therap", "brez"),                  "no therapy"),
    ]
    for keywords, canonical in substring_map:
        if any(k in s for k in keywords):
            return canonical

    return s  # return as-is so unknown values are visible


def safe_skew(series: pd.Series, label: str = "") -> float:
    """
    Compute sample skewness.  Returns NaN (with a warning) when the series
    has fewer than 3 observations or is constant.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 3:
        warnings.warn(
            f"safe_skew: too few observations (n={len(s)}) for '{label}' — returning NaN",
            stacklevel=2,
        )
        return np.nan
    if s.nunique() < 2:
        warnings.warn(
            f"safe_skew: constant series for '{label}' — returning NaN",
            stacklevel=2,
        )
        return np.nan
    return float(skew(s, bias=False))


def unique_sheet_names(mapping: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Truncate sheet names to Excel's 31-character limit and assert uniqueness
    so we never silently overwrite a sheet.
    """
    truncated = {name[:31]: df for name, df in mapping.items()}
    if len(truncated) < len(mapping):
        originals = list(mapping.keys())
        shorts = [n[:31] for n in originals]
        dupes = [s for s in shorts if shorts.count(s) > 1]
        raise ValueError(
            f"Sheet name truncation produces duplicates: {dupes}. "
            "Rename these keys to avoid collisions after 31 characters."
        )
    return truncated


def set_plot_style() -> None:
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "figure.titlesize": 17,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })


def save_figure(fig: plt.Figure, filename: str, dpi: int = 300) -> Path:
    out = OUTPUT_DIR / filename
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


# ============================================================
# Data loading, validation, and preprocessing
# ============================================================
def load_data(path: Path, sheet_name=0) -> pd.DataFrame:
    log.info("Loading %s (sheet=%s)", path, sheet_name)
    return pd.read_excel(path, sheet_name=sheet_name)


def validate_columns(df: pd.DataFrame, required: List[str]) -> None:
    """Raise a descriptive ValueError if any required column is absent."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input file is missing {len(missing)} required column(s):\n"
            + "\n".join(f"  - {c}" for c in missing)
        )
    log.info("Column validation passed (%d required columns present)", len(required))


def preprocess_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse all date columns in one place before any other transformation.
    This is the single source of truth for date parsing — nothing else
    should call parse_date.
    """
    x = df.copy()
    for col in (COL_SCAN_DATE, COL_DIAGNOSIS_DATE_00, COL_DIAGNOSIS_DATE,
                COL_PROGRESSION_DATE):
        if col in x.columns:
            x[col] = parse_date(x[col])
    return x


def resolve_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Resolve duplicated image_session_id rows.

    Preference rule (descending priority):
        1. More lesion information (non-zero & non-null)
        2. More non-missing values overall

    NOTE: expects dates to have already been parsed by preprocess_dates().
    """
    x = df.copy()
    x["__nonmissing__"]   = x.notna().sum(axis=1)
    x["__lesion_score__"] = (
        x[COL_LESION_VOL].fillna(-1).gt(0).astype(int)
        + x[COL_LESION_COUNT].fillna(-1).gt(0).astype(int)
        + x[COL_LESION_VOL].notna().astype(int)
        + x[COL_LESION_COUNT].notna().astype(int)
    )

    duplicated_mask = x.duplicated(COL_IMAGE_SESSION, keep=False)
    duplicated = x.loc[duplicated_mask].drop(
        columns=["__nonmissing__", "__lesion_score__"]
    ).copy()

    n_dupes = duplicated[COL_IMAGE_SESSION].nunique()
    log.info(
        "Duplicate image_session_ids: %d rows across %d sessions",
        len(duplicated), n_dupes,
    )

    cleaned = (
        x.sort_values(
            [COL_IMAGE_SESSION, "__lesion_score__", "__nonmissing__"],
            ascending=[True, False, False],
        )
        .drop_duplicates(COL_IMAGE_SESSION, keep="first")
        .drop(columns=["__nonmissing__", "__lesion_score__"])
        .copy()
    )

    return cleaned, duplicated


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add standardized and computed columns. Mutates a copy; does not re-parse dates."""
    x = df.copy()

    x[COL_SEX]      = x[COL_SEX].map(normalize_sex)
    x["therapy_std"]= x[COL_THERAPY].map(standardize_therapy)

    x[COL_LESION_VOL]   = pd.to_numeric(x[COL_LESION_VOL],   errors="coerce")
    x[COL_LESION_COUNT] = pd.to_numeric(x[COL_LESION_COUNT], errors="coerce")
    x[COL_LOG_LESION_VOL] = np.log1p(x[COL_LESION_VOL].clip(lower=0))

    diag_date = x[COL_DIAGNOSIS_DATE_00].fillna(x[COL_DIAGNOSIS_DATE])
    x["disease_duration_years"] = (x[COL_SCAN_DATE] - diag_date).dt.days / 365.25
    x.loc[x["disease_duration_years"] < 0, "disease_duration_years"] = np.nan

    return x


def get_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Return the earliest scan per subject as the baseline visit."""
    return (
        df.sort_values([COL_SUBJECT, COL_SCAN_DATE])
        .drop_duplicates(COL_SUBJECT, keep="first")
        .copy()
    )


def longitudinal_summary(
    df: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute per-patient visit counts, total follow-up, and inter-visit gaps.
    All time differences use day-accurate arithmetic (month_diff → days/30.4375).
    """
    x = df.sort_values([COL_SUBJECT, COL_SCAN_DATE]).copy()

    visit_counts = x.groupby(COL_SUBJECT).size().rename("visits_per_patient")

    followup = x.groupby(COL_SUBJECT)[COL_SCAN_DATE].agg(["min", "max"])
    followup_months = pd.Series(
        [month_diff(a, b) for a, b in zip(followup["min"], followup["max"])],
        index=followup.index,
        name="followup_months",
    )

    gaps: List[float] = []
    for _, sub in x.groupby(COL_SUBJECT):
        dates = sub[COL_SCAN_DATE].dropna().sort_values().tolist()
        for i in range(len(dates) - 1):
            gaps.append(month_diff(dates[i], dates[i + 1]))

    gap_series = pd.Series(gaps, name="intervisit_gap_months", dtype="float64")
    return visit_counts, followup_months, gap_series


# ============================================================
# Summary tables
# ============================================================
def summarize_numeric(series: pd.Series, name: str) -> Dict[str, object]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "variable": name,
        "n": int(s.size),
        "mean": float(s.mean())            if s.size else np.nan,
        "sd":   float(s.std())             if s.size else np.nan,
        "median": float(s.median())        if s.size else np.nan,
        "iqr_low":  float(s.quantile(0.25)) if s.size else np.nan,
        "iqr_high": float(s.quantile(0.75)) if s.size else np.nan,
        "min":  float(s.min())             if s.size else np.nan,
        "max":  float(s.max())             if s.size else np.nan,
        "skew": safe_skew(s, label=name),
    }


def summarize_categorical(series: pd.Series, name: str) -> pd.DataFrame:
    """Return a count+percentage breakdown for a categorical variable."""
    counts = series.fillna("Missing").value_counts(dropna=False)
    df = counts.rename_axis("category").reset_index(name="n")
    df["pct"] = 100 * df["n"] / df["n"].sum()
    df.insert(0, "variable", name)
    return df


def make_table1(baseline: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Clinical Table 1 — baseline characteristics.
    Returns two DataFrames: numeric summaries and categorical summaries.
    """
    numeric_vars = {
        "Age at baseline (years)":              COL_AGE,
        "Baseline EDSS":                        COL_EDSS,
        "Disease duration at baseline (years)": "disease_duration_years",
        "Brain volume (cm3)":                   COL_BRAIN,
        "Grey matter volume (cm3)":             COL_GM,
        "White matter volume (cm3)":            COL_WM,
        "Lateral ventricle total volume (cm3)": COL_VENT,
        "Lesion volume":                        COL_LESION_VOL,
        "Lesion count":                         COL_LESION_COUNT,
    }
    categorical_vars = {
        "Sex":       COL_SEX,
        "Diagnosis": COL_DIAGNOSIS,
        "Therapy at baseline": "therapy_std",
    }

    numeric_rows = [summarize_numeric(baseline[col], label)
                    for label, col in numeric_vars.items()]

    cat_frames = [summarize_categorical(baseline[col], label)
                  for label, col in categorical_vars.items()]

    return {
        "numeric":     pd.DataFrame(numeric_rows),
        "categorical": pd.concat(cat_frames, ignore_index=True),
    }


def make_missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        COL_EDSS, COL_BRAIN, COL_GM, COL_WM, COL_VENT,
        COL_LESION_VOL, COL_LESION_COUNT, COL_EDUCATION,
        COL_PROGRESSION_DATE, "therapy_std", COL_DIAGNOSIS,
    ]
    present_cols = [c for c in cols if c in df.columns]
    rows = [
        {
            "variable":    col,
            "missing_n":   int(df[col].isna().sum()),
            "missing_pct": float(df[col].isna().mean() * 100),
        }
        for col in present_cols
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("missing_pct", ascending=False)
        .reset_index(drop=True)
    )


def make_cohort_tables(
    clean: pd.DataFrame,
    baseline: pd.DataFrame,
    visit_counts: pd.Series,
) -> Dict[str, pd.DataFrame]:
    sex_counts = (
        baseline[COL_SEX].value_counts(dropna=False)
        .rename_axis("sex").reset_index(name="count")
    )
    sex_counts["pct"] = 100 * sex_counts["count"] / len(baseline)

    diagnosis_counts = (
        baseline[COL_DIAGNOSIS].fillna("Missing").value_counts()
        .rename_axis("diagnosis").reset_index(name="count")
    )
    diagnosis_counts["pct"] = 100 * diagnosis_counts["count"] / len(baseline)

    therapy_counts = (
        baseline["therapy_std"].fillna("missing").value_counts()
        .rename_axis("therapy").reset_index(name="count")
    )
    therapy_counts["pct"] = 100 * therapy_counts["count"] / len(baseline)

    location_counts = (
        clean[COL_LOCATION].fillna("missing").value_counts()
        .rename_axis("location").reset_index(name="count")
    )
    location_counts["pct"] = 100 * location_counts["count"] / len(clean)

    visit_dist = (
        visit_counts.value_counts().sort_index()
        .rename_axis("visits").reset_index(name="patients")
    )
    visit_dist["pct"] = 100 * visit_dist["patients"] / visit_dist["patients"].sum()

    return {
        "sex_counts":              sex_counts,
        "diagnosis_counts":        diagnosis_counts,
        "baseline_therapy_counts": therapy_counts,
        "location_counts":         location_counts,
        "visit_count_dist":        visit_dist,
    }


def make_correlation_table(baseline: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "Age":              COL_AGE,
        "EDSS":             COL_EDSS,
        "Disease duration": "disease_duration_years",
        "Brain volume":     COL_BRAIN,
        "WM volume":        COL_WM,
        "GM volume":        COL_GM,
        "Lesion volume":    COL_LESION_VOL,
        "Lesion count":     COL_LESION_COUNT,
    }
    corr_df = baseline[list(cols.values())].copy()
    corr_df.columns = list(cols.keys())
    return corr_df.corr(method="spearman")


# ============================================================
# Plotting
# ============================================================
def plot_baseline_distributions(baseline: pd.DataFrame) -> Path:
    set_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=FIG_BG)
    fig.suptitle(
        "Figure 1. Baseline distributions of key demographic, clinical, and MRI variables",
        y=1.02,
    )

    vars_info = [
        (COL_AGE,             "Baseline age",              "Years"),
        (COL_EDSS,            "Baseline EDSS",             "EDSS"),
        ("disease_duration_years", "Baseline disease duration", "Years"),
        (COL_LOG_LESION_VOL,  "Baseline lesion volume",    "log(1 + lesion volume)"),
        (COL_LESION_COUNT,    "Baseline lesion count",     "Count"),
        (COL_BRAIN,           "Baseline brain volume",     "cm$^3$"),
    ]

    for ax, (col, title, xlabel) in zip(axes.ravel(), vars_info):
        ax.set_facecolor(AX_BG)
        s = pd.to_numeric(baseline[col], errors="coerce").dropna()
        ax.hist(s, bins=20, color=BLUE, alpha=0.95, edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.30)
        ax.set_axisbelow(True)

    fig.tight_layout()
    return save_figure(fig, "Figure_1_baseline_distributions.png")


def plot_baseline_cohort_characterisation(
    clean: pd.DataFrame,
    baseline: pd.DataFrame,
    visit_counts: pd.Series,
) -> Path:
    set_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), facecolor=FIG_BG)
    fig.suptitle(
        "Figure 2. Cohort characterisation: demographics and clinical variables",
        y=1.02,
    )

    for ax in axes.ravel():
        ax.set_facecolor(AX_BG)
        ax.grid(True, alpha=0.30)
        ax.set_axisbelow(True)

    axes[0, 0].hist(
        pd.to_numeric(baseline[COL_AGE], errors="coerce").dropna(),
        bins=20, color="#9eb6df", edgecolor="white",
    )
    axes[0, 0].set_title("Age at baseline")
    axes[0, 0].set_xlabel("Age (years)")
    axes[0, 0].set_ylabel("Count")

    sex_counts = baseline[COL_SEX].fillna("Unknown").value_counts()
    axes[0, 1].pie(
        sex_counts.values,
        labels=sex_counts.index,
        autopct="%.0f%%",
        startangle=90,
        colors=["#e07a6f", "#4c78c9", "#bab0ac"][: len(sex_counts)],
    )
    axes[0, 1].set_title("Sex distribution")
    axes[0, 1].grid(False)

    axes[0, 2].hist(
        pd.to_numeric(baseline[COL_EDSS], errors="coerce").dropna(),
        bins=20, color=GREEN, edgecolor="white",
    )
    axes[0, 2].set_title("Baseline EDSS distribution")
    axes[0, 2].set_xlabel("EDSS score")
    axes[0, 2].set_ylabel("Count")

    dx = baseline[COL_DIAGNOSIS].fillna("Missing").value_counts().sort_values()
    axes[1, 0].barh(dx.index.astype(str), dx.values, color=PURPLE)
    axes[1, 0].set_title("Diagnosis type at baseline")
    axes[1, 0].set_xlabel("Count")

    vc = visit_counts.value_counts().sort_index()
    axes[1, 1].bar(vc.index, vc.values, width=0.9, color=ORANGE)
    axes[1, 1].set_title("Visits per patient")
    axes[1, 1].set_xlabel("Number of visits")
    axes[1, 1].set_ylabel("Number of patients")
    axes[1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))

    tx = baseline["therapy_std"].fillna("missing").value_counts().head(8).sort_values()
    axes[1, 2].barh(tx.index.astype(str), tx.values, color=TEAL)
    axes[1, 2].set_title("Top baseline therapy categories")
    axes[1, 2].set_xlabel("Count")

    fig.tight_layout()
    return save_figure(fig, "Figure_2_baseline_cohort_characterisation.png")


def plot_baseline_mri_distributions(baseline: pd.DataFrame) -> Path:
    set_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor=FIG_BG)
    fig.suptitle("Figure 3. Baseline MRI feature distributions", y=1.02)

    info = [
        (COL_BRAIN,       "Brain volume (cm$^3$)"),
        (COL_GM,          "Grey matter volume (cm$^3$)"),
        (COL_WM,          "White matter volume (cm$^3$)"),
        (COL_LESION_VOL,  "Lesion volume"),
        (COL_LESION_COUNT,"Lesion count"),
        (COL_VENT,        "Lateral ventricle total volume (cm$^3$)"),
    ]

    for ax, (col, title) in zip(axes.ravel(), info):
        ax.set_facecolor(AX_BG)
        s = pd.to_numeric(baseline[col], errors="coerce").dropna()
        ax.hist(s, bins=22, color="#9eb6df", edgecolor="white")
        ax.set_title(title)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.30)
        ax.set_axisbelow(True)

        sk = safe_skew(s, label=col)
        if not np.isnan(sk):
            ax.text(
                0.98, 0.96, f"skew = {sk:.2f}",
                ha="right", va="top",
                transform=ax.transAxes,
                fontsize=11,
                color="dimgray",
            )

    fig.tight_layout()
    return save_figure(fig, "Figure_3_baseline_mri_distributions.png")


def plot_baseline_correlation(corr: pd.DataFrame) -> Path:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 10), facecolor=FIG_BG)
    ax.set_facecolor(AX_BG)

    im = ax.imshow(corr.values, cmap="viridis", vmin=-0.6, vmax=1.0)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    ax.set_title("Figure 4. Baseline correlation structure (Spearman's ρ)")

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(
                j, i, f"{corr.iloc[i, j]:.2f}",
                ha="center", va="center", color="black", fontsize=10,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Spearman's $\\rho$")

    fig.tight_layout()
    return save_figure(fig, "Figure_4_baseline_correlation_structure.png")


def plot_longitudinal_structure(
    visit_counts: pd.Series,
    followup_months: pd.Series,
    gaps: pd.Series,
) -> Path:
    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=FIG_BG)
    fig.suptitle("Figure 5. Longitudinal sampling structure", y=1.02)

    for ax in axes:
        ax.set_facecolor(AX_BG)
        ax.grid(True, alpha=0.30)
        ax.set_axisbelow(True)

    vc = visit_counts.value_counts().sort_index()
    axes[0].bar(vc.index, vc.values, width=0.9, color=BLUE)
    axes[0].set_title("Visits per patient")
    axes[0].set_xlabel("Number of visits")
    axes[0].set_ylabel("Patients")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    axes[1].hist(followup_months.dropna(), bins=16, color=BLUE, edgecolor="white")
    axes[1].set_title("Total follow-up per patient")
    axes[1].set_xlabel("Months")
    axes[1].set_ylabel("Patients")

    axes[2].hist(gaps.dropna(), bins=16, color=BLUE, edgecolor="white")
    axes[2].set_title("Inter-visit gaps")
    axes[2].set_xlabel("Months")
    axes[2].set_ylabel("Intervals")

    fig.tight_layout()
    return save_figure(fig, "Figure_5_longitudinal_sampling_structure.png")


def plot_missingness(clean: pd.DataFrame) -> Path:
    set_plot_style()
    cols = [
        COL_EDSS, COL_BRAIN, COL_GM, COL_WM, COL_VENT,
        COL_LESION_VOL, COL_LESION_COUNT, COL_EDUCATION,
        COL_PROGRESSION_DATE, "therapy_std", COL_DIAGNOSIS,
    ]
    present_cols = [c for c in cols if c in clean.columns]
    miss = clean[present_cols].isna().mean().sort_values(ascending=True) * 100

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=FIG_BG)
    ax.set_facecolor(AX_BG)
    ax.barh(miss.index.astype(str), miss.values, color=TEAL)
    ax.set_title("Figure 6. Missingness in key variables")
    ax.set_xlabel("Missing values (%)")
    ax.grid(True, alpha=0.30)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return save_figure(fig, "Figure_6_missingness.png")


# ============================================================
# Text report
# ============================================================
def build_summary_text(
    raw_df: pd.DataFrame,
    duplicated: pd.DataFrame,
    clean: pd.DataFrame,
    baseline: pd.DataFrame,
    visit_counts: pd.Series,
    followup_months: pd.Series,
    gaps: pd.Series,
    corr: pd.DataFrame,
) -> str:
    female  = int((baseline[COL_SEX] == "F").sum())
    male    = int((baseline[COL_SEX] == "M").sum())
    unknown = int((baseline[COL_SEX] == "Unknown").sum())
    n       = len(baseline)

    therapy_missing = int(baseline["therapy_std"].isna().sum())
    therapy_no      = int((baseline["therapy_std"] == "no therapy").sum())
    therapy_yes     = n - therapy_missing - therapy_no

    lines = [
        "Task 2 EDA Summary",
        "==================",
        f"Raw file            : {len(raw_df):,} rows × {raw_df.shape[1]} columns",
        f"Duplicate sessions  : {len(duplicated):,} rows across "
        f"{duplicated[COL_IMAGE_SESSION].nunique():,} session IDs",
        f"Cleaned dataset     : {len(clean):,} sessions from "
        f"{clean[COL_SUBJECT].nunique():,} patients",
        f"Scan date span      : {clean[COL_SCAN_DATE].min().date()} → "
        f"{clean[COL_SCAN_DATE].max().date()}",
        "",
        "Baseline cohort",
        f"  Patients  : {n}",
        f"  Female    : {female} ({100 * female / n:.1f}%)",
        f"  Male      : {male} ({100 * male / n:.1f}%)",
        f"  Unknown   : {unknown} ({100 * unknown / n:.1f}%)",
        f"  Age       : {baseline[COL_AGE].mean():.2f} ± {baseline[COL_AGE].std():.2f} years",
        f"  EDSS      : {baseline[COL_EDSS].mean():.2f} ± {baseline[COL_EDSS].std():.2f}",
        f"  Brain vol : {baseline[COL_BRAIN].median():.1f} cm3 (median)",
        f"  WM vol    : {baseline[COL_WM].median():.1f} cm3 (median)",
        f"  GM vol    : {baseline[COL_GM].median():.1f} cm3 (median)",
        "",
        "Longitudinal structure",
        f"  Visits/patient : mean {visit_counts.mean():.2f}, "
        f"median {visit_counts.median():.0f}, "
        f"range {visit_counts.min()}–{visit_counts.max()}",
        f"  Follow-up      : mean {followup_months.mean():.1f} months, "
        f"median {followup_months.median():.1f}, "
        f"max {followup_months.max():.1f}",
        f"  Inter-visit gap: mean {gaps.mean():.1f} months, "
        f"median {gaps.median():.1f}, "
        f"range {gaps.min():.1f}–{gaps.max():.1f}",
        "",
        "Therapy at baseline",
        f"  With therapy    : {therapy_yes} ({100 * therapy_yes / n:.1f}%)",
        f"  No therapy      : {therapy_no} ({100 * therapy_no / n:.1f}%)",
        f"  Missing         : {therapy_missing} ({100 * therapy_missing / n:.1f}%)",
        "",
        "Key baseline Spearman correlations",
        f"  Age       vs EDSS            : {corr.loc['Age', 'EDSS']:.2f}",
        f"  EDSS      vs Disease duration: {corr.loc['EDSS', 'Disease duration']:.2f}",
        f"  Brain vol vs EDSS            : {corr.loc['Brain volume', 'EDSS']:.2f}",
        f"  Brain vol vs Age             : {corr.loc['Brain volume', 'Age']:.2f}",
        f"  GM vol    vs Age             : {corr.loc['GM volume', 'Age']:.2f}",
    ]
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================
def main() -> None:
    # Verify environment is set up before doing any work
    ensure_dirs()
    assert_data_exists()

    # 1. Load & validate
    raw_df = load_data(INPUT_FILE, sheet_name=SHEET_NAME)
    validate_columns(raw_df, REQUIRED_COLUMNS)

    # 2. Parse dates once — upstream of everything else
    df = preprocess_dates(raw_df)

    # 3. Clean & enrich
    clean, duplicated = resolve_duplicates(df)
    clean    = add_derived_columns(clean)
    baseline = get_baseline(clean)

    # 4. Longitudinal stats
    visit_counts, followup_months, gaps = longitudinal_summary(clean)

    # 5. Tables
    table1      = make_table1(baseline)
    missingness = make_missingness_table(clean)
    cohort_tbls = make_cohort_tables(clean, baseline, visit_counts)
    corr        = make_correlation_table(baseline)

    # 6. Figures
    fig_paths = [
        plot_baseline_distributions(baseline),
        plot_baseline_cohort_characterisation(clean, baseline, visit_counts),
        plot_baseline_mri_distributions(baseline),
        plot_baseline_correlation(corr),
        plot_longitudinal_structure(visit_counts, followup_months, gaps),
        plot_missingness(clean),
    ]

    # 7. Save all tables into a single workbook
    sheet_map: Dict[str, pd.DataFrame] = {
        "Table1_numeric":          table1["numeric"],
        "Table1_categorical":      table1["categorical"],
        "Missingness":             missingness,
        "Baseline_correlation":    corr,
        "Baseline_cleaned":        baseline,
        "Cleaned_all_visits":      clean,
        "Duplicated_rows":         duplicated,
        **cohort_tbls,
    }
    sheet_map = unique_sheet_names(sheet_map)  # validates ≤31 chars + uniqueness

    workbook = OUTPUT_DIR / "task2_eda_tables.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for sheet_name, tbl in sheet_map.items():
            tbl.to_excel(writer, sheet_name=sheet_name, index=(sheet_name == "Baseline_correlation"))

    # 8. Save and print text summary
    summary_txt  = build_summary_text(
        raw_df=raw_df, duplicated=duplicated, clean=clean,
        baseline=baseline, visit_counts=visit_counts,
        followup_months=followup_months, gaps=gaps, corr=corr,
    )
    summary_path = OUTPUT_DIR / "task2_eda_summary.txt"
    summary_path.write_text(summary_txt, encoding="utf-8")

    print(summary_txt)
    log.info("Workbook : %s", workbook)
    log.info("Summary  : %s", summary_path)
    for p in fig_paths:
        log.info("Figure   : %s", p)


if __name__ == "__main__":
    main()