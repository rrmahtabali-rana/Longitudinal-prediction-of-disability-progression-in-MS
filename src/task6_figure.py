"""
Generate fig_rsf_importance.png — RSF permutation importance horizontal bar chart.

Reads the CSV produced by task6_survival.py and renders a publication-ready
figure (300 dpi).

Run with:    python task6_figure.py
Or via:      make task6
Requires:    task6_survival.py to have been run first (produces the input CSV).

Output: outputs/task6_outputs/fig_rsf_importance.png
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paths import TASK6_OUT, ensure_dirs

INPUT_FILE = TASK6_OUT / "task6_rsf_permutation_importance.csv"
OUTPUT_FILE = TASK6_OUT / "fig_rsf_importance.png"


def clean_name(raw: str) -> str:
    """
    Convert internal feature name to a readable label.

    Examples:
      num__current__EDSSValue                       -> Current: EDSS
      num__baseline__lesioncount                    -> Baseline: Lesion Count
      num__slope_per_year__Lateral ventricle ...    -> Slope: Lateral Ventricle Vol.
      num__delta_prev__EDSSValue                    -> Delta (prev): EDSS
      cat__current__sex_id_f                        -> Current: Female
      cat__current__TherapyName_fingolimod          -> Current: Fingolimod
    """
    s = raw
    for prefix in ["num__", "cat__"]:
        if s.startswith(prefix):
            s = s[len(prefix):]
            break

    if s.startswith("current__"):
        prefix_label = "Current"
        s = s[len("current__"):]
    elif s.startswith("baseline__"):
        prefix_label = "Baseline"
        s = s[len("baseline__"):]
    elif s.startswith("slope_per_year__"):
        prefix_label = "Slope"
        s = s[len("slope_per_year__"):]
    elif s.startswith("delta_prev__"):
        prefix_label = "\u0394 (prev)"
        s = s[len("delta_prev__"):]
    else:
        prefix_label = ""

    s = s.replace("__", " ").replace("_", " ").strip()

    replacements = {
        "EDSSValue": "EDSS",
        "lesioncount": "Lesion Count",
        "lesionvolume": "Lesion Volume",
        "Brain (WM+GM) volume cm3": "Brain Volume",
        "Grey Matter (GM) volume cm3": "GM Volume",
        "White Matter (WM) volume cm3": "WM Volume",
        "Lateral ventricle total volume cm3": "Lateral Ventricle Vol.",
        "time since baseline days": "Time Since Baseline",
        "num prior visits": "No. Prior Visits",
        "sex id f": "Female",
        "sex id m": "Male",
        "TherapyName fingolimod": "Fingolimod",
        "TherapyName copaxone": "Copaxone",
        "TherapyName rebif": "Rebif",
        "TherapyName no therapy": "No Therapy",
        "time since prev days": "Time Since Prev. Visit",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)

    s = s.strip()
    if s:
        s = s[0].upper() + s[1:]

    if prefix_label and s:
        return f"{prefix_label}: {s}"
    return s if s else raw


def get_colour(raw: str) -> str:
    """Colour-code feature labels by clinical domain."""
    r = raw.lower()
    if "edss" in r:
        return "#1565C0"  # deep blue
    if "lesion" in r:
        return "#E65100"  # orange
    if any(x in r for x in ["volume", "ventricle", "brain", "matter"]):
        return "#00695C"  # teal
    if "sex" in r:
        return "#6A1B9A"  # purple
    if "therapy" in r:
        return "#827717"  # olive
    return "#546E7A"      # grey: time / visits / other


def render_figure(df: pd.DataFrame, output_path) -> None:
    """Render the horizontal bar chart and save to disk."""
    df = df.head(15).copy()
    df["label"] = df["feature"].apply(clean_name)
    df["colour"] = df["feature"].apply(get_colour)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    y_pos = np.arange(len(df))

    ax.barh(
        y_pos,
        df["importance_mean"],
        xerr=df["importance_std"],
        color=df["colour"],
        edgecolor="white",
        linewidth=0.4,
        height=0.65,
        capsize=3,
        error_kw={"elinewidth": 1.0, "ecolor": "#333333", "capthick": 1.0},
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["label"], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.7, linestyle="-")
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, color="#BBBBBB")
    ax.set_axisbelow(True)
    ax.set_xlabel("Permutation Importance (C-index drop)", fontsize=10)
    ax.set_title(
        "RSF Permutation Importance — Test Set (Top 15 Features)",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_entries = [
        mpatches.Patch(color="#1565C0", label="EDSS"),
        mpatches.Patch(color="#E65100", label="Lesion"),
        mpatches.Patch(color="#00695C", label="MRI Volume"),
        mpatches.Patch(color="#6A1B9A", label="Sex"),
        mpatches.Patch(color="#827717", label="Therapy"),
        mpatches.Patch(color="#546E7A", label="Time / Visits"),
    ]
    ax.legend(
        handles=legend_entries,
        title="Feature domain",
        title_fontsize=8,
        fontsize=8,
        loc="lower right",
        framealpha=0.85,
        edgecolor="#CCCCCC",
    )

    ax.text(
        0.99, 0.01,
        "Error bars: SD over 10 permutation repeats",
        transform=ax.transAxes,
        fontsize=7,
        color="#555555",
        ha="right", va="bottom",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_dirs()
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Required input not found: {INPUT_FILE}\n"
            "Run task6_survival.py first (or `make task6`)."
        )

    df = pd.read_csv(INPUT_FILE)
    render_figure(df, OUTPUT_FILE)
    print(f"Saved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
