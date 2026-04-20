# MS Disability Progression Prediction

Longitudinal analysis and machine-learning prediction of disability progression in multiple sclerosis (MS), using MRI-derived volumetric features, Expanded Disability Status Scale (EDSS) assessments, and therapy records.

This repository accompanies the PhD candidate assessment task for the Young Researcher Position 2026 (advisor: prof. dr. Žiga Špiclin, University of Ljubljana, Faculty of Electrical Engineering).

> **Dataset is not included in this repository.** See [`data/README.md`](data/README.md) for instructions on placement.

---

## Table of contents

1. [Overview](#overview)
2. [Repository structure](#repository-structure)
3. [Quick start](#quick-start)
4. [Detailed setup](#detailed-setup)
5. [Running the pipeline](#running-the-pipeline)
6. [Expected outputs](#expected-outputs)
7. [Reproducibility notes](#reproducibility-notes)
8. [Troubleshooting](#troubleshooting)
9. [Citation](#citation)
10. [License](#license)

---

## Overview

The pipeline implements all nine tasks from the assessment brief:

| Task | Script | What it does |
|------|--------|--------------|
| 2 | `src/task2_eda.py` | Exploratory data analysis: cohort description, baseline summaries, longitudinal sampling structure, missingness audit |
| 3 | `src/task3_stratification.py` | MRI feature stratification: ICV-normalised brain volume, normative scores, hippocampal asymmetry, lesion burden tertiles |
| 4 | `src/task4_progression.py` | Disability progression labelling: 6-month sustained EDSS worsening criteria |
| 4 | `src/task4_descriptive.py` | Descriptive statistics for cDP vs wDP groups |
| 5 | `src/task5_pipeline.py` | Predictive modelling: Logistic Regression, TCN, CatBoost with isotonic calibration; landmark-based sampling, patient-level splits |
| 6 | `src/task6_survival.py` | Survival analysis: Cox PH and Random Survival Forest, time-dependent AUC, calibration |
| 6 | `src/task6_figure.py` | Generates RSF permutation importance figure |

Tasks 7, 8, and 9 (model evaluation, confounders, critical assessment) are reported in the manuscript using outputs produced by tasks 5 and 6.

---

## Repository structure

```
ms-progression-prediction/
├── README.md                      ← you are here
├── LICENSE                        ← MIT
├── requirements.txt               ← pip dependencies
├── environment.yml                ← conda alternative
├── .gitignore                     ← excludes data, outputs, caches
├── Makefile                       ← convenience targets (make all, make task5, ...)
│
├── src/                           ← all source code
│   ├── paths.py                   ← centralised path config (env-var driven)
│   ├── task2_eda.py
│   ├── task3_stratification.py
│   ├── task4_progression.py
│   ├── task4_descriptive.py
│   ├── task5_pipeline.py
│   ├── task6_survival.py
│   └── task6_figure.py
│
├── notebooks/
│   └── Task_2_9_clean.ipynb       ← canonical notebook (cells 15 + 21 + others)
│
├── data/
│   └── README.md                  ← explains where to place data_original.xlsx
│   └── data_original.xlsx         ← NOT COMMITTED (.gitignored)
│
├── outputs/                       ← created at runtime, .gitignored
│   ├── task2_outputs/
│   ├── task3_outputs/
│   ├── task4_outputs/
│   ├── task5_outputs/
│   └── task6_outputs/
│
├── figures/                       ← published figures (kept under version control)
│
├── scripts/
│   ├── run_all.sh                 ← Linux/macOS runner
│   └── run_all.ps1                ← Windows PowerShell runner
│
├── tests/
│   └── check_imports.py           ← smoke test: ensure modules import (NOT a pytest file)
│
└── docs/
    └── REPRODUCIBILITY.md         ← exact versions, hardware, runtime
```

---

## Quick start

For users who already have Python 3.10+ and `pip`:

```bash
# 1. Clone the repository
git clone <your-repo-url> ms-progression-prediction
cd ms-progression-prediction

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate                  # Linux/macOS
# .venv\Scripts\activate                   # Windows PowerShell

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Place the dataset
#    Copy data_original.xlsx into ./data/
#    (The repository does not include the dataset — see data/README.md)

# 5. Sanity check
python tests/check_imports.py

# 6. Run everything
bash scripts/run_all.sh                    # Linux/macOS
# .\scripts\run_all.ps1                    # Windows
```

Total runtime: approximately **8–15 minutes** on a modern laptop (no GPU required for any task; CatBoost and PyTorch run on CPU by default).

---

## Detailed setup

### Python version

Python **3.12 or later** is required. The pipeline was developed and tested on Python 3.12.

### Option A — pip + venv (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
```

### Option B — conda

```bash
conda env create -f environment.yml
conda activate ms-progression
```

### Dataset placement

The dataset (`data_original.xlsx`, ~16 MB) is not redistributed in this repository. After obtaining it from the original source, you have two options:

**Option 1 — default location (easier):**
Place the file at `data/data_original.xlsx` (relative to the repo root). All scripts find it automatically.

**Option 2 — environment variable (more flexible):**
```bash
export MS_DATA_PATH=/full/absolute/path/to/data_original.xlsx
```
On Windows PowerShell:
```powershell
$env:MS_DATA_PATH = "C:\full\path\to\data_original.xlsx"
```

You can also override the outputs directory with `MS_OUTPUTS_ROOT`. Defaults to `./outputs/`.

---

## Running the pipeline

### Run everything

```bash
bash scripts/run_all.sh                    # Linux/macOS
.\scripts\run_all.ps1                      # Windows PowerShell
make all                                   # any platform with GNU Make
```

### Run a single task

```bash
cd src
python task2_eda.py                        # Task 2: EDA
python task3_stratification.py             # Task 3: MRI stratification
python task4_progression.py                # Task 4: progression labelling
python task4_descriptive.py                # Task 4: descriptive stats
python task5_pipeline.py                   # Task 5: classification models
python task6_survival.py                   # Task 6: survival models
python task6_figure.py                     # Task 6: RSF importance figure
```

### Strict execution order

Some tasks depend on outputs of earlier tasks:

```
task2_eda.py    →  task4_progression.py  →  task4_descriptive.py
task5_pipeline.py  →  task6_survival.py  →  task6_figure.py
task3_stratification.py                       (independent of 2/4)
```

`scripts/run_all.sh` and the `Makefile` both respect this order.

---

## Expected outputs

After running everything, `outputs/` will contain ~50 files. Key outputs verified to match the manuscript:

| File | Content | Manuscript reference |
|---|---|---|
| `task2_outputs/task2_eda_tables.xlsx` | Cleaned dataset, baseline + visit-level summaries | Section 2.1, Figs. 1–4 |
| `task3_outputs/task3_mri_stratification_summary.csv` | Tertile groups, EDSS medians, DP rates per stratification | Section 2.2, Fig. 5 |
| `task4_outputs/task4_progression_outputs.xlsx` | Visit-level + patient-level DP labels | Section 2.3 |
| `task5_outputs/task5_metrics_patient_level_calibrated.csv` | CatBoost ROC-AUC 0.744, F1 0.619 | Table 2 |
| `task5_outputs/task5_test_ece_summary_calibrated.csv` | ECE reductions: LR 62.9%, TCN 77.6%, CatBoost 80.8% | Table 3 |
| `task6_outputs/task6_cindex.csv` | RSF C-index 0.665, Cox 0.618 | Table 4 |
| `task6_outputs/task6_time_dep_auc.csv` | RSF AUC@730d 0.704 | Table 4 |
| `task6_outputs/fig_rsf_importance.png` | RSF permutation importance | Fig. 7 |
| `task6_outputs/task6_calibration_combined.png` | Survival calibration plots | Fig. 8 |

---

## Reproducibility notes

- **Random seed**: `42` is set globally in all scripts that use stochastic methods (`set_seed()` in `task5_pipeline.py`).
- **Patient-level splits**: train/val/test = 70%/15%/15%, stratified on the ever-progressed label, with no patient appearing in more than one split.
- **Calibration**: isotonic regression fit on validation predictions only — never on test.
- **Bootstrap CIs**: 1000 resamples for classification, 500 for survival, all patient-clustered.
- **Numerical reproducibility**: results may differ in the 3rd decimal place across machines due to BLAS implementations and CatBoost / PyTorch CPU thread scheduling.
- **Hardware tested**: results in the manuscript were produced on Linux (Ubuntu 22.04) with Python 3.12, 16 GB RAM, no GPU.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for exact package versions and runtime estimates.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: Dataset not found at .../data_original.xlsx` | Place the dataset at `data/data_original.xlsx` or set `MS_DATA_PATH` |
| `ImportError: catboost is required` | Run `pip install -r requirements.txt`; on Apple Silicon you may need `pip install catboost --no-binary :all:` |
| `ImportError: scikit-survival` not found | `pip install scikit-survival` (on Windows requires Microsoft C++ Build Tools) |
| Task 6 fails with "all times must be within follow-up time" warning at t=180 | Expected — the script suppresses this warning; see manuscript Section 2.6 |
| `task6_survival.py` cannot import from `task5_pipeline` | Run from the `src/` directory, or add `src/` to `PYTHONPATH` |
| `pytest` errors with `INTERNALERROR` and `SystemExit: 1` on `tests/` | Don't run the smoke test via `pytest`. Use `python tests/check_imports.py` or `make test` directly. The smoke test is a standalone runner, not a pytest suite. |

---




---

## License

This code is released under the MIT License. See [`LICENSE`](LICENSE) for details.

The dataset is **not** included and is governed by its own terms of use, which are not affected by this license.

---

## Acknowledgements

- Reference framework for survival analysis: D'hondt et al. (2025), *Computer Methods and Programs in Biomedicine*, 263, 108624.
- Multi-centre MS cohort collected at three Slovenian sites: Ljubljana, Maribor, Celje.
