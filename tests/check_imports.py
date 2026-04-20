"""
Import smoke test — standalone runner.

Verifies that:
  1. All third-party dependencies are installed.
  2. All src/ modules can be imported without error (no module-level side effects).
  3. The paths module locates the dataset (or warns clearly).
  4. Key symbols that task6 imports from task5 actually exist.

Run with:
    python tests/check_imports.py
Or:
    make test

NOTE: This is *not* a pytest test. The filename intentionally avoids the
`test_` prefix so pytest will not auto-discover it. Running it under pytest
is harmless but pointless — pytest will simply not see the checks.

A non-zero exit code from direct `python` invocation means the environment
is not ready to run the pipeline.
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

# Make src/ importable regardless of cwd
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


# ─── Test definitions ───────────────────────────────────────────────────────

THIRD_PARTY = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "matplotlib",
    "openpyxl",
    "torch",
    "catboost",
    "sksurv",
]

SRC_MODULES = [
    "paths",
    "task2_eda",
    "task3_stratification",
    "task4_progression",
    "task4_descriptive",
    "task5_pipeline",
    "task6_survival",
    "task6_figure",
]

TASK5_EXPORTS = [
    "Config",
    "TabularPreprocessor",
    "build_tabular_features",
    "classification_metrics",
    "filter_df_by_patients",
    "load_longitudinal_data",
    "patient_level_split",
    "samples_to_tabular_df",
    "set_seed",
]


# ─── Reporting helpers ──────────────────────────────────────────────────────

PASS = "[ OK ]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def report(status: str, msg: str) -> None:
    print(f"  {status}  {msg}")


# ─── The actual checks ─────────────────────────────────────────────────────

def run_checks() -> int:
    """Execute all import checks; return 0 on success, 1 on failure."""
    failures: list[str] = []

    print("=" * 64)
    print("  Import smoke test")
    print("=" * 64)

    # 1. Third-party dependencies
    print("\n[1/4] Third-party dependencies")
    for mod in THIRD_PARTY:
        try:
            importlib.import_module(mod)
            report(PASS, mod)
        except Exception as e:
            report(FAIL, f"{mod}  ->  {type(e).__name__}: {e}")
            failures.append(f"third-party: {mod}")

    # 2. paths module + dataset
    print("\n[2/4] Dataset path resolution")
    try:
        import paths  # type: ignore

        print(f"  REPO_ROOT    = {paths.REPO_ROOT}")
        print(f"  DATA_PATH    = {paths.DATA_PATH}")
        print(f"  OUTPUTS_ROOT = {paths.OUTPUTS_ROOT}")
        if paths.DATA_PATH.exists():
            report(PASS, f"Dataset found at {paths.DATA_PATH}")
        else:
            report(WARN, "Dataset NOT found — pipeline will fail at runtime")
            report(WARN, "  (See data/README.md for placement instructions)")
    except Exception as e:
        report(FAIL, f"paths module  ->  {type(e).__name__}: {e}")
        failures.append("paths")

    # 3. Source modules
    print("\n[3/4] src/ modules")
    for mod in SRC_MODULES:
        try:
            importlib.import_module(mod)
            report(PASS, mod)
        except Exception as e:
            report(FAIL, f"{mod}  ->  {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
            failures.append(f"src: {mod}")

    # 4. Cross-module symbol presence (task6 -> task5)
    print("\n[4/4] task5_pipeline ⇒ task6_survival exports")
    try:
        task5 = importlib.import_module("task5_pipeline")
        for sym in TASK5_EXPORTS:
            if hasattr(task5, sym):
                report(PASS, f"task5_pipeline.{sym}")
            else:
                report(FAIL, f"task5_pipeline.{sym} is MISSING (task6 will fail)")
                failures.append(f"missing: task5_pipeline.{sym}")
    except Exception as e:
        report(FAIL, f"could not import task5_pipeline  ->  {e}")
        failures.append("task5_pipeline import")

    # ─── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    if failures:
        print(f"  RESULT: {len(failures)} failure(s)")
        for f in failures:
            print(f"    - {f}")
        print("=" * 64)
        return 1

    print("  RESULT: all imports succeeded.")
    print("=" * 64)
    return 0


# ─── Script entry point only — sys.exit is NEVER called at module load ─────

if __name__ == "__main__":
    sys.exit(run_checks())
