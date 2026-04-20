# Reproducibility notes

This document records the exact computational environment, runtime, and
expected numerical results for the MS disability progression pipeline.

## Computational environment

| Item | Value used in manuscript |
|------|--------------------------|
| Operating system | Ubuntu 22.04 LTS (also tested on Google Colab Linux) |
| Python | 3.12.x |
| RAM | 16 GB |
| CPU cores | 8 |
| GPU | not required (PyTorch CPU build is sufficient) |

## Pinned package versions

The `requirements.txt` file specifies version ranges that should produce
numerically equivalent results. The exact versions used in the manuscript:

| Package | Version |
|---------|---------|
| numpy | 1.26.x |
| pandas | 2.2.x |
| scipy | 1.13.x |
| scikit-learn | 1.5.x |
| catboost | 1.2.x |
| scikit-survival | 0.23.x |
| torch | 2.4.x (CPU) |
| matplotlib | 3.9.x |
| openpyxl | 3.1.x |

If you need exact replication, freeze your environment after install and
diff against this list:

```bash
pip freeze > my_environment.txt
```

## Runtime estimates

On the reference machine (8-core CPU, 16 GB RAM, no GPU):

| Task | Approx. runtime |
|------|-----------------|
| Task 2 — EDA | ~30 s |
| Task 3 — MRI stratification | ~10 s |
| Task 4 — Progression labelling | ~20 s |
| Task 4 — Descriptive stats | ~5 s |
| Task 5 — LR + TCN + CatBoost (with calibration + bootstrap) | ~5–8 min |
| Task 6 — Cox + RSF (with bootstrap) | ~3–5 min |
| Task 6 — Figure generation | ~2 s |
| **Total** | **~10–15 min** |


A single global seed `42` is used throughout. It is set via
`task5_pipeline.set_seed(42)` which seeds:

- Python's `random`
- NumPy
- PyTorch (CPU and CUDA if present)
- CatBoost
- scikit-learn (where it accepts `random_state`)

## Numerical reproducibility caveats

Even with identical seeds and pinned package versions, you may observe
**third-decimal-place differences** in metrics across runs due to:

- BLAS/LAPACK implementation differences (OpenBLAS vs MKL vs Apple
  Accelerate)
- Non-determinism in CatBoost's parallel histogram computation
- PyTorch CPU thread scheduling for small batch sizes
- Floating-point summation order in `numpy.mean` over large arrays

