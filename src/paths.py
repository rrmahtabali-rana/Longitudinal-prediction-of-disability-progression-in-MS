"""
Centralised path configuration for the pipeline.

All scripts import from this module so that:
  - The dataset can live anywhere (controlled via MS_DATA_PATH env var).
  - All outputs are written under a single OUTPUTS_DIR root.
  - No script contains hardcoded /content/... Colab paths.

Usage
-----
1. Place data_original.xlsx anywhere on disk.
2. Set the environment variable before running any script:

       export MS_DATA_PATH=/full/path/to/data_original.xlsx

   Or, on Windows (PowerShell):

       $env:MS_DATA_PATH="C:\\path\\to\\data_original.xlsx"

3. (Optional) Override the outputs root:

       export MS_OUTPUTS_ROOT=/full/path/to/outputs

If MS_DATA_PATH is not set, the default falls back to ./data/data_original.xlsx
relative to the repository root, which matches the recommended layout.
"""

from __future__ import annotations

import os
from pathlib import Path

# ─── Repository root ─────────────────────────────────────────────────────────
# This file lives in <repo>/src/, so the repo root is the parent of its parent.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# ─── Input data ──────────────────────────────────────────────────────────────
# Either set MS_DATA_PATH or place data_original.xlsx under <repo>/data/.
DATA_PATH: Path = Path(
    os.environ.get("MS_DATA_PATH", REPO_ROOT / "data" / "data_original.xlsx")
)

# ─── Output roots ────────────────────────────────────────────────────────────
OUTPUTS_ROOT: Path = Path(
    os.environ.get("MS_OUTPUTS_ROOT", REPO_ROOT / "outputs")
)
FIGURES_ROOT: Path = Path(
    os.environ.get("MS_FIGURES_ROOT", REPO_ROOT / "figures")
)

# ─── Per-task output directories ─────────────────────────────────────────────
TASK2_OUT: Path = OUTPUTS_ROOT / "task2_outputs"
TASK3_OUT: Path = OUTPUTS_ROOT / "task3_outputs"
TASK4_OUT: Path = OUTPUTS_ROOT / "task4_outputs"
TASK5_OUT: Path = OUTPUTS_ROOT / "task5_outputs"
TASK6_OUT: Path = OUTPUTS_ROOT / "task6_outputs"


def ensure_dirs() -> None:
    """Create all required output directories if missing."""
    for d in (
        OUTPUTS_ROOT,
        FIGURES_ROOT,
        TASK2_OUT,
        TASK3_OUT,
        TASK4_OUT,
        TASK5_OUT,
        TASK6_OUT,
    ):
        d.mkdir(parents=True, exist_ok=True)


def assert_data_exists() -> None:
    """Verify the dataset file is present; fail early with a clear message."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}.\n"
            "Either:\n"
            "  • Place data_original.xlsx in <repo>/data/, OR\n"
            "  • Set the MS_DATA_PATH environment variable to its absolute path."
        )


if __name__ == "__main__":
    print(f"REPO_ROOT     = {REPO_ROOT}")
    print(f"DATA_PATH     = {DATA_PATH}  (exists={DATA_PATH.exists()})")
    print(f"OUTPUTS_ROOT  = {OUTPUTS_ROOT}")
    print(f"FIGURES_ROOT  = {FIGURES_ROOT}")
    print(f"TASK2_OUT     = {TASK2_OUT}")
    print(f"TASK3_OUT     = {TASK3_OUT}")
    print(f"TASK4_OUT     = {TASK4_OUT}")
    print(f"TASK5_OUT     = {TASK5_OUT}")
    print(f"TASK6_OUT     = {TASK6_OUT}")
