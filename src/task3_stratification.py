#!/usr/bin/env python3
"""
Reproduce Task 3 MRI stratification results and publication-style figure.

What this script does
---------------------
1. Loads the original Excel dataset.
2. Selects one baseline row per subject (earliest MRI visit).
3. Recomputes the four MRI stratification strategies:
   - Brain/ICV tertiles
   - Normative brain-score tertiles
   - Hippocampal asymmetry groups
   - Lesion-burden tertiles
4. Calculates:
   - group sizes
   - DP rates
   - EDSS medians
   - Kruskal-Wallis p-values
5. Saves:
   - a polished 2x4 figure
   - a CSV summary table
   - a text report with the verified numbers

Required columns in the Excel file
----------------------------------
subject_id
MRIDateYYYYMM
image_session_id
EDSSValue
DP
Brain (WM+GM) volume cm3
Intracranial Cavity (IC) volume cm3
Brain (WM+GM) volume % z-score
Hippocampus volume asymmetry
lesionvolume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from paths import DATA_PATH, TASK3_OUT, ensure_dirs, assert_data_exists
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal


def validate_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "The dataset is missing required columns:\n- " + "\n- ".join(missing)
        )


def tertile_groups(series: pd.Series, labels: Sequence[str]) -> tuple[pd.Series, float, float]:
    q1, q2 = series.quantile([1/3, 2/3]).tolist()
    grouped = pd.cut(
        series,
        bins=[-np.inf, q1, q2, np.inf],
        labels=labels,
        include_lowest=True
    )
    return grouped, float(q1), float(q2)


def kruskal_pvalue(df: pd.DataFrame, group_col: str, value_col: str, labels: Sequence[str]) -> float:
    arrays = [df.loc[df[group_col] == g, value_col].dropna() for g in labels]
    return float(kruskal(*arrays).pvalue)


def p_label(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def dp_rates(df: pd.DataFrame, group_col: str, labels: Sequence[str]) -> pd.Series:
    return (df.groupby(group_col, observed=False)["DP"].mean() * 100).reindex(labels)


def edss_medians(df: pd.DataFrame, group_col: str, labels: Sequence[str]) -> pd.Series:
    return df.groupby(group_col, observed=False)["EDSSValue"].median().reindex(labels)


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.01, 0.98, f"({label})",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12, fontweight="bold"
    )


def clean_boxplot(
    ax: plt.Axes,
    data_groups: Sequence[pd.Series],
    labels: Sequence[str],
    counts: Sequence[int],
    colors: Sequence[str],
    title: str,
    ptext: str,
    show_ylabel: bool = False,
) -> None:
    bp = ax.boxplot(
        data_groups,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops=dict(color="#2F2F2F", linewidth=1.8),
        whiskerprops=dict(color="#555555", linewidth=1.1),
        capprops=dict(color="#555555", linewidth=1.1),
        boxprops=dict(edgecolor="#555555", linewidth=1.1),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 6.8)
    ax.set_title(title, pad=6, fontweight="semibold")
    if show_ylabel:
        ax.set_ylabel("Baseline EDSS")
    style_axes(ax)

    ax.text(
        0.98, 0.95, ptext,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#FAFAFA", edgecolor="#C8C8C8", linewidth=0.7),
    )

    ymin, ymax = ax.get_ylim()
    y_text = ymin - 0.14 * (ymax - ymin)
    for x, n in enumerate(counts, start=1):
        ax.text(x, y_text, f"n={n}", ha="center", va="top", fontsize=9.3, color="#555555", clip_on=False)


def clean_bars(
    ax: plt.Axes,
    labels: Sequence[str],
    values: pd.Series,
    counts: Sequence[int],
    colors: Sequence[str],
    show_ylabel: bool = False,
    ylim: int = 50,
) -> None:
    x = np.arange(len(labels))
    bars = ax.bar(x, values.values, color=colors, edgecolor="#555555", linewidth=0.8, width=0.78)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, ylim)
    if show_ylabel:
        ax.set_ylabel("DP rate (%)")
    style_axes(ax)

    for bar, val, n in zip(bars, values.values, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.9,
            f"{val:.1f}%",
            ha="center", va="bottom",
            fontsize=10.5, color="#222222"
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            1.0,
            f"n={n}",
            ha="center", va="bottom",
            fontsize=9.2, color="#4F4F4F"
        )


def build_baseline(df: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        df.sort_values(["subject_id", "MRIDateYYYYMM", "image_session_id"])
          .groupby("subject_id", as_index=False)
          .first()
          .copy()
    )

    baseline["brain_icv"] = (
        baseline["Brain (WM+GM) volume cm3"] /
        baseline["Intracranial Cavity (IC) volume cm3"]
    )
    baseline["norm_score"] = baseline["Brain (WM+GM) volume % z-score"]
    baseline["ai"] = baseline["Hippocampus volume asymmetry"]
    baseline["lesion_cm3"] = baseline["lesionvolume"] / 1000.0

    return baseline


def make_outputs(excel_path: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(excel_path)

    required_columns = [
        "subject_id",
        "MRIDateYYYYMM",
        "image_session_id",
        "EDSSValue",
        "DP",
        "Brain (WM+GM) volume cm3",
        "Intracranial Cavity (IC) volume cm3",
        "Brain (WM+GM) volume % z-score",
        "Hippocampus volume asymmetry",
        "lesionvolume",
    ]
    validate_columns(df, required_columns)

    baseline = build_baseline(df)

    brain_labels = ["Atrophied", "Intermediate", "Preserved"]
    norm_labels = ["Lower", "Middle", "Upper"]
    ai_labels = ["Symmetric", "Asymmetric"]
    lesion_labels = ["Low", "Medium", "High"]

    baseline["brain_grp"], brain_q1, brain_q2 = tertile_groups(baseline["brain_icv"], brain_labels)
    baseline["norm_grp"], norm_q1, norm_q2 = tertile_groups(baseline["norm_score"], norm_labels)
    baseline["ai_grp"] = np.where((baseline["ai"] - 0.5).abs() <= 0.02, "Symmetric", "Asymmetric")

    lesion_df = baseline.dropna(subset=["lesion_cm3"]).copy()
    lesion_df["lesion_grp"], lesion_q1, lesion_q2 = tertile_groups(lesion_df["lesion_cm3"], lesion_labels)

    brain_counts = baseline["brain_grp"].value_counts().reindex(brain_labels).astype(int)
    norm_counts = baseline["norm_grp"].value_counts().reindex(norm_labels).astype(int)
    ai_counts = pd.Series(baseline["ai_grp"]).value_counts().reindex(ai_labels).astype(int)
    lesion_counts = lesion_df["lesion_grp"].value_counts().reindex(lesion_labels).astype(int)

    brain_dp = dp_rates(baseline, "brain_grp", brain_labels)
    norm_dp = dp_rates(baseline, "norm_grp", norm_labels)
    ai_dp = dp_rates(baseline, "ai_grp", ai_labels)
    lesion_dp = dp_rates(lesion_df, "lesion_grp", lesion_labels)

    brain_median = edss_medians(baseline, "brain_grp", brain_labels)
    norm_median = edss_medians(baseline, "norm_grp", norm_labels)
    ai_median = edss_medians(baseline, "ai_grp", ai_labels)
    lesion_median = edss_medians(lesion_df, "lesion_grp", lesion_labels)

    p_brain = kruskal_pvalue(baseline, "brain_grp", "EDSSValue", brain_labels)
    p_norm = kruskal_pvalue(baseline, "norm_grp", "EDSSValue", norm_labels)
    p_ai = kruskal_pvalue(baseline, "ai_grp", "EDSSValue", ai_labels)
    p_lesion = kruskal_pvalue(lesion_df, "lesion_grp", "EDSSValue", lesion_labels)

    summary_rows = []

    def add_rows(strategy: str, labels: Sequence[str], counts: pd.Series, medians: pd.Series, dps: pd.Series, p: float) -> None:
        for lab in labels:
            summary_rows.append({
                "strategy": strategy,
                "group": lab,
                "n": int(counts[lab]),
                "median_edss": float(medians[lab]) if pd.notna(medians[lab]) else np.nan,
                "dp_rate_percent": float(dps[lab]) if pd.notna(dps[lab]) else np.nan,
                "kruskal_pvalue": p,
            })

    add_rows("Brain/ICV tertiles", brain_labels, brain_counts, brain_median, brain_dp, p_brain)
    add_rows("Normative score tertiles", norm_labels, norm_counts, norm_median, norm_dp, p_norm)
    add_rows("Hippocampal asymmetry", ai_labels, ai_counts, ai_median, ai_dp, p_ai)
    add_rows("Lesion burden tertiles", lesion_labels, lesion_counts, lesion_median, lesion_dp, p_lesion)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "task3_mri_stratification_summary.csv", index=False)

    report = f"""Task 3 MRI Stratification Verified Results
========================================

Input file: {excel_path}
Baseline cohort size: {len(baseline)}
Baseline lesion-data cohort size: {len(lesion_df)}

1) Brain/ICV tertiles
---------------------
Cut points: q1 = {brain_q1:.5f}, q2 = {brain_q2:.5f}
Counts: {dict(brain_counts)}
DP rates (%): {brain_dp.round(1).to_dict()}
Median EDSS: {brain_median.to_dict()}
Kruskal-Wallis: {p_label(p_brain)}

2) Normative brain-score tertiles
---------------------------------
Cut points: q1 = {norm_q1:.5f}, q2 = {norm_q2:.5f}
Counts: {dict(norm_counts)}
DP rates (%): {norm_dp.round(1).to_dict()}
Median EDSS: {norm_median.to_dict()}
Kruskal-Wallis: {p_label(p_norm)}

3) Hippocampal asymmetry
------------------------
Definition: Symmetric if |AI - 0.5| <= 0.02, otherwise Asymmetric
Counts: {dict(ai_counts)}
DP rates (%): {ai_dp.round(1).to_dict()}
Median EDSS: {ai_median.to_dict()}
Kruskal-Wallis: {p_label(p_ai)}

4) Lesion burden tertiles
-------------------------
Cut points (cm^3): q1 = {lesion_q1:.3f}, q2 = {lesion_q2:.3f}
Counts: {dict(lesion_counts)}
DP rates (%): {lesion_dp.round(1).to_dict()}
Median EDSS: {lesion_median.to_dict()}
Kruskal-Wallis: {p_label(p_lesion)}
"""
    (outdir / "task3_mri_stratification_report.txt").write_text(report, encoding="utf-8")

    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#555555",
    })

    severity_3 = ["#C76D6D", "#D9B46C", "#7AA88A"]
    binary_2 = ["#7F9FC9", "#D28B6A"]
    lesion_colors = [severity_3[2], severity_3[1], severity_3[0]]
    top_titles = ["Brain/ICV", "Normative score", "Hippocampal asymmetry", "Lesion burden"]
    panel_letters = list("ABCDEFGH")

    fig, axes = plt.subplots(2, 4, figsize=(15.2, 6.9))
    fig.subplots_adjust(left=0.055, right=0.99, top=0.94, bottom=0.12, wspace=0.24, hspace=0.30)

    clean_boxplot(
        axes[0, 0],
        [baseline.loc[baseline["brain_grp"] == g, "EDSSValue"].dropna() for g in brain_labels],
        brain_labels, brain_counts.values, severity_3, top_titles[0], p_label(p_brain), show_ylabel=True
    )
    add_panel_label(axes[0, 0], panel_letters[0])

    clean_boxplot(
        axes[0, 1],
        [baseline.loc[baseline["norm_grp" ] == g, "EDSSValue"].dropna() for g in norm_labels],
        norm_labels, norm_counts.values, severity_3, top_titles[1], p_label(p_norm), show_ylabel=False
    )
    add_panel_label(axes[0, 1], panel_letters[1])

    clean_boxplot(
        axes[0, 2],
        [baseline.loc[baseline["ai_grp"] == g, "EDSSValue"].dropna() for g in ai_labels],
        ai_labels, ai_counts.values, binary_2, top_titles[2], p_label(p_ai), show_ylabel=False
    )
    add_panel_label(axes[0, 2], panel_letters[2])

    clean_boxplot(
        axes[0, 3],
        [lesion_df.loc[lesion_df["lesion_grp"] == g, "EDSSValue"].dropna() for g in lesion_labels],
        lesion_labels, lesion_counts.values, lesion_colors, top_titles[3], p_label(p_lesion), show_ylabel=False
    )
    add_panel_label(axes[0, 3], panel_letters[3])

    clean_bars(
        axes[1, 0], brain_labels, brain_dp, brain_counts.values, severity_3, show_ylabel=True, ylim=50
    )
    add_panel_label(axes[1, 0], panel_letters[4])

    clean_bars(
        axes[1, 1], norm_labels, norm_dp, norm_counts.values, severity_3, show_ylabel=False, ylim=50
    )
    add_panel_label(axes[1, 1], panel_letters[5])

    clean_bars(
        axes[1, 2], ai_labels, ai_dp, ai_counts.values, binary_2, show_ylabel=False, ylim=45
    )
    add_panel_label(axes[1, 2], panel_letters[6])

    clean_bars(
        axes[1, 3], lesion_labels, lesion_dp, lesion_counts.values, lesion_colors, show_ylabel=False, ylim=50
    )
    add_panel_label(axes[1, 3], panel_letters[7])

    fig.text(0.012, 0.72, "EDSS", rotation=90, va="center", ha="center", fontsize=12, fontweight="semibold")
    fig.text(0.012, 0.28, "DP", rotation=90, va="center", ha="center", fontsize=12, fontweight="semibold")

    fig.savefig(outdir / "mri_stratification_task3_final_polished.png", dpi=340, bbox_inches="tight")
    fig.savefig(outdir / "mri_stratification_task3_final_polished.pdf", dpi=340, bbox_inches="tight")
    fig.savefig(outdir / "mri_stratification_task3_final_polished.jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved files:")
    print(f"- {outdir / 'task3_mri_stratification_summary.csv'}")
    print(f"- {outdir / 'task3_mri_stratification_report.txt'}")
    print(f"- {outdir / 'mri_stratification_task3_final_polished.png'}")
    print(f"- {outdir / 'mri_stratification_task3_final_polished.pdf'}")
    print(f"- {outdir / 'mri_stratification_task3_final_polished.jpg'}")


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Reproduce Task 3 MRI stratification results and figure.")
    parser.add_argument("--input", type=str, default=str(DATA_PATH), help="Path to the Excel dataset.")
    parser.add_argument("--outdir", type=str, default=str(TASK3_OUT), help="Directory to save results and figures.")

    # Use sys.argv[1:] if calling from CLI, otherwise empty for Jupyter
    args = parser.parse_args(sys.argv[1:] if not hasattr(sys, 'ps1') else [])

    excel_path = Path(args.input)
    outdir = Path(args.outdir)

    if not excel_path.exists():
        raise FileNotFoundError(f"Input file not found: {excel_path}")

    make_outputs(excel_path, outdir)


if __name__ == "__main__":
    main()
