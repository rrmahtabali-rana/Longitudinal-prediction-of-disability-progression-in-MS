# =============================================================================
#  run_all.ps1 — Run the full MS disability progression pipeline (Tasks 2-6)
#
#  Usage from the repository root:
#      .\scripts\run_all.ps1
#
#  If you get an execution-policy error, run once per shell:
#      Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# =============================================================================

$ErrorActionPreference = "Stop"

# ─── Locate repository root ─────────────────────────────────────────────────
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Resolve-Path (Join-Path $ScriptDir "..")
$SrcDir    = Join-Path $RepoRoot "src"

Set-Location $RepoRoot

Write-Host "================================================================"
Write-Host "  MS Disability Progression Pipeline"
Write-Host "  Repo root: $RepoRoot"
Write-Host "================================================================"

# ─── Sanity check: Python on PATH ──────────────────────────────────────────
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "ERROR: 'python' command not found. Activate your virtualenv first."
    exit 1
}

# ─── Sanity check: dataset present ─────────────────────────────────────────
& python (Join-Path $SrcDir "paths.py")
if ($LASTEXITCODE -ne 0) {
    Write-Error "ERROR: Dataset not found. See data/README.md for placement instructions."
    exit 1
}

# ─── Helper: run a single task with timing ─────────────────────────────────
function Invoke-Task {
    param(
        [string]$Label,
        [string]$Script
    )
    Write-Host ""
    Write-Host "----------------------------------------------------------------"
    Write-Host "  $Label"
    Write-Host "  -> python $Script"
    Write-Host "----------------------------------------------------------------"
    $t0 = Get-Date
    Push-Location $SrcDir
    try {
        & python $Script
        if ($LASTEXITCODE -ne 0) { throw "Task '$Label' failed (exit $LASTEXITCODE)" }
    }
    finally {
        Pop-Location
    }
    $elapsed = [int]((Get-Date) - $t0).TotalSeconds
    Write-Host "  [OK] done in ${elapsed}s"
}

# ─── Pipeline ──────────────────────────────────────────────────────────────
Invoke-Task "Task 2 — Exploratory data analysis"          "task2_eda.py"
Invoke-Task "Task 3 — MRI feature stratification"         "task3_stratification.py"
Invoke-Task "Task 4 — Disability progression labelling"   "task4_progression.py"
Invoke-Task "Task 4 — Descriptive statistics"             "task4_descriptive.py"
Invoke-Task "Task 5 — Predictive modelling"               "task5_pipeline.py"
Invoke-Task "Task 6 — Survival analysis"                  "task6_survival.py"
Invoke-Task "Task 6 — RSF importance figure"              "task6_figure.py"

Write-Host ""
Write-Host "================================================================"
Write-Host "  All tasks completed successfully."
Write-Host "  Outputs written under: $RepoRoot\outputs\"
Write-Host "================================================================"
