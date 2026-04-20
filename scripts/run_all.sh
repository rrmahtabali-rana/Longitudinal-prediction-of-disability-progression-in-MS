#!/usr/bin/env bash
# =============================================================================
#  run_all.sh — Run the full MS disability progression pipeline (Tasks 2-6)
#
#  Usage:    bash scripts/run_all.sh
#  Or:       ./scripts/run_all.sh    (after `chmod +x scripts/run_all.sh`)
#
#  Order of execution respects inter-task dependencies:
#    Task 2 → Task 4 (progression) → Task 4 (descriptive)
#    Task 3 (independent of 2/4)
#    Task 5 → Task 6 (survival) → Task 6 (figure)
# =============================================================================

set -euo pipefail   # fail fast on any error or undefined variable

# ─── Locate the repository root regardless of where this script is invoked ──
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
SRC_DIR="$REPO_ROOT/src"

cd "$REPO_ROOT"

# ─── Sanity checks ──────────────────────────────────────────────────────────
echo "================================================================"
echo "  MS Disability Progression Pipeline"
echo "  Repo root: $REPO_ROOT"
echo "================================================================"

# Verify Python is available
if ! command -v python &>/dev/null; then
    echo "ERROR: 'python' command not found. Activate your virtualenv first."
    exit 1
fi

# Verify the dataset is accessible
python "$SRC_DIR/paths.py" || {
    echo ""
    echo "ERROR: Dataset not found. See data/README.md for placement instructions."
    exit 1
}

# ─── Run each task with timing ──────────────────────────────────────────────
run_task() {
    local label="$1"; shift
    local script="$1"; shift
    echo ""
    echo "----------------------------------------------------------------"
    echo "  $label"
    echo "  → python $script"
    echo "----------------------------------------------------------------"
    local t0=$SECONDS
    ( cd "$SRC_DIR" && python "$script" )
    local elapsed=$(( SECONDS - t0 ))
    echo "  ✓ done in ${elapsed}s"
}

run_task "Task 2 — Exploratory data analysis"          task2_eda.py
run_task "Task 3 — MRI feature stratification"         task3_stratification.py
run_task "Task 4 — Disability progression labelling"   task4_progression.py
run_task "Task 4 — Descriptive statistics"             task4_descriptive.py
run_task "Task 5 — Predictive modelling"               task5_pipeline.py
run_task "Task 6 — Survival analysis"                  task6_survival.py
run_task "Task 6 — RSF importance figure"              task6_figure.py

echo ""
echo "================================================================"
echo "  All tasks completed successfully."
echo "  Outputs written under: $REPO_ROOT/outputs/"
echo "================================================================"
