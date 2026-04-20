"""
Task 6 — Survival analysis of time-to-progression in MS.

Reuses the Task 5 landmark construction but reframes labels as (time, event)
pairs for time-to-event modelling. Implements two models:

  • Cox proportional hazards regression (baseline, interpretable)
  • Random Survival Forest (D'hondt et al. 2025 framing)

Reports:
  • Harrell's concordance index (C-index)  — rank agreement
  • Integrated Brier Score over [30, 730] d  — calibration + discrimination
  • Cumulative/dynamic AUC at 365, 730 d   — horizon-specific discrimination
    (180 d excluded: min observed follow-up in test set is ~184 d)
  • Horizon classifiers with val-tuned thresholds (F1-optimised)
    to avoid degenerate F1=0 at fixed threshold=0.5 under low event rates
  • Cox coefficients (with hazard ratios) + RSF permutation importance
  • Patient-grouped bootstrap 95% CIs on test metrics

Cross-linking to Task 5:
  A fitted survival model yields P(T ≤ t | x) for any t. Evaluating at
  t ∈ {365, 730} days gives "free" binary classifiers derived from a
  single fit. The 730-day RSF classifier is directly comparable to Task 5's
  CatBoost classifier with the same horizon.

Prerequisites:
  pip install scikit-survival catboost torch scikit-learn pandas numpy openpyxl
  task5_pipeline.py must be in the same directory or on PYTHONPATH.

Dataset definitions (event / censoring / time origin):
  • Time origin = landmark date (each MRI visit of each patient)
  • Event      = 1 if ProgressionYYYYMM exists and lies strictly after the
                 landmark date; else 0 (right-censored)
  • Time       = min(progression_date, last_visit_date) − landmark_date in days
  • Inclusion  = follow-up ≥ 30 days (excludes pathologically short windows)

Fixes vs original version:
  1. 180-day horizon removed — data minimum follow-up in test is ~184 d,
     making t=180 evaluable in zero bootstrap resamples; warnings suppressed.
  2. Horizon classifiers now use validation-tuned F1 thresholds instead of
     the fixed 0.5 default, which produced degenerate F1=0 under the low
     event prevalence (~5% at 365 d, ~16% at 730 d).
  3. All threshold tuning is done on validation only; test set is never
     touched during tuning (no leakage).
  4. Cox ridge penalty increased from 0.01 → 1.0 to suppress the extreme
     hazard ratios caused by sparse therapy category cells (some therapies
     appear in only 1-2 patients), which drove train C-index to 0.78 while
     val C-index collapsed to 0.54 — a clear sign of overfitting.
  5. Survival calibration plots added — mean predicted S(t) vs observed
     Kaplan-Meier S(t) stratified by risk tertile, for both Cox and RSF on
     the test set. This directly addresses Task 7's calibration requirement.
"""

import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

# ── Suppress the repetitive 180-day AUC boundary warning ────────────────────
# This is a known data limitation (min follow-up in test ≈ 184 d) that is
# documented in the report. The warning itself is correct; we just do not need
# it printed thousands of times during bootstrapping.
warnings.filterwarnings(
    "ignore",
    message=".*all times must be within follow-up time.*",
    category=RuntimeWarning,
)

# Ensure task5_pipeline.py is importable when running as a script
_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if _here not in sys.path:
    sys.path.insert(0, _here)

from task5_pipeline import (  # noqa: E402
    Config,
    TabularPreprocessor,
    build_tabular_features,
    classification_metrics,
    filter_df_by_patients,
    load_longitudinal_data,
    patient_level_split,
    samples_to_tabular_df,
    set_seed,
)
from paths import TASK6_OUT, ensure_dirs  # noqa: E402

try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import (
        concordance_index_censored,
        cumulative_dynamic_auc,
        integrated_brier_score,
    )
except ImportError as exc:
    raise ImportError(
        "scikit-survival is required. Install with:\n"
        "    pip install scikit-survival"
    ) from exc


# =========================================
# 1. CONFIGURATION
# =========================================

@dataclass
class SurvivalConfig:
    """Task 6-specific knobs, independent of Task 5's Config."""

    min_followup_days: int = 30

    # Horizons (days) for time-dependent AUC and horizon classifiers.
    # NOTE: 180 d is intentionally excluded because the minimum observed
    # follow-up in the test set after splitting is ~184 d, making t=180
    # unevaluable. Only 365 d (12 mo) and 730 d (24 mo, matches Task 5)
    # are reported.
    eval_horizons_days: List[int] = field(
        default_factory=lambda: [365, 730]
    )

    # Integrated Brier score grid
    ibs_t_min_days: float = 30.0
    ibs_t_max_days: float = 730.0
    ibs_n_grid: int = 50

    # Cox PH ridge penalty. Increased from 0.01 to 1.0 to suppress extreme
    # hazard ratios from sparse therapy categories and reduce overfitting.
    cox_alpha: float = 1.0

    # Random Survival Forest
    rsf_n_estimators: int = 300
    rsf_min_samples_leaf: int = 15
    rsf_max_features: str = "sqrt"

    # Bootstrap
    n_boot: int = 500
    bootstrap_ci_level: float = 0.95

    # Horizon threshold tuning objective on validation set
    # Options: "f1" | "balanced_accuracy"
    threshold_objective: str = "f1"

    # Permutation importance
    perm_n_repeats: int = 10


# =========================================
# 2. SURVIVAL DATASET CONSTRUCTION
# =========================================

def build_survival_samples(
    df_split: pd.DataFrame,
    cfg: Config,
    scfg: SurvivalConfig,
) -> List[Dict]:
    """
    Build one sample per MRI visit with (time, event) survival labels.

    Uses the same feature engineering as Task 5's build_landmark_samples but
    a broader inclusion rule — Task 5's [180, 730]-day window is dropped
    because survival analysis handles short follow-up via right-censoring.

    A landmark at or after the patient's progression date is skipped (the
    event is already observed), as are landmarks with < min_followup_days
    of follow-up.
    """
    samples: List[Dict] = []

    for patient_id, pdf in df_split.groupby(cfg.patient_col):
        pdf = pdf.sort_values(cfg.date_col).reset_index(drop=True)
        progression_date = pdf["progression_date"].iloc[0]
        has_progression = pd.notna(progression_date)
        last_date = pdf[cfg.date_col].max()

        for i in range(len(pdf)):
            landmark_date = pdf.loc[i, cfg.date_col]

            # Skip landmarks at or after progression — event already occurred
            if has_progression and landmark_date >= progression_date:
                continue

            if has_progression:
                time_days = int((progression_date - landmark_date).days)
                event = 1
            else:
                time_days = int((last_date - landmark_date).days)
                event = 0

            if time_days < scfg.min_followup_days:
                continue

            history = pdf.loc[:i].copy()
            tabular = build_tabular_features(history, cfg)

            samples.append({
                "patient_id": patient_id,
                "landmark_date": landmark_date,
                "time": float(time_days),
                "event": int(event),
                "tabular": tabular,
            })

    return samples


def make_surv_y(samples: List[Dict]) -> np.ndarray:
    """
    Build the structured-array label format that scikit-survival expects.
    dtype = [("event", bool), ("time", float)]
    """
    y = np.zeros(len(samples), dtype=[("event", bool), ("time", float)])
    for i, s in enumerate(samples):
        y[i] = (bool(s["event"]), float(s["time"]))
    return y


# =========================================
# 3. MODELS
# =========================================

def fit_cox(X_train: np.ndarray, y_train: np.ndarray, scfg: SurvivalConfig):
    """Cox PH with a small ridge penalty for numerical stability."""
    cox = CoxPHSurvivalAnalysis(alpha=scfg.cox_alpha)
    cox.fit(X_train, y_train)
    return cox


def fit_rsf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: Config,
    scfg: SurvivalConfig,
):
    """Random Survival Forest — D'hondt et al. 2025 framing."""
    rsf = RandomSurvivalForest(
        n_estimators=scfg.rsf_n_estimators,
        min_samples_leaf=scfg.rsf_min_samples_leaf,
        max_features=scfg.rsf_max_features,
        n_jobs=-1,
        random_state=cfg.random_state,
    )
    rsf.fit(X_train, y_train)
    return rsf


# =========================================
# 4. SURVIVAL METRICS
# =========================================

def compute_cindex(model, X: np.ndarray, y: np.ndarray) -> float:
    """
    Harrell's concordance index. Higher predicted risk should correspond to
    earlier event times. Both Cox and RSF have .predict() returning a
    monotone-in-risk score, so no sign flip is needed.
    """
    risk = model.predict(X)
    c, _, _, _, _ = concordance_index_censored(
        event_indicator=y["event"],
        event_time=y["time"],
        estimate=risk,
    )
    return float(c)


def _surv_probs_at(
    model, X: np.ndarray, times: np.ndarray
) -> np.ndarray:
    """Return (n_samples, n_times) survival probability matrix S(t | x)."""
    surv_funcs = model.predict_survival_function(X)
    probs = np.asarray(
        [[float(fn(t)) for t in times] for fn in surv_funcs]
    )
    return probs


def compute_ibs(
    model,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    scfg: SurvivalConfig,
) -> float:
    """
    Integrated Brier Score on an evaluation set.
    The time grid is constrained so that it lies inside the range of
    observed event times in BOTH y_train (needed for the censoring
    distribution estimator) and y_eval.
    """
    t_lo = max(
        float(y_train["time"].min()),
        float(y_eval["time"].min()),
        scfg.ibs_t_min_days,
    ) + 1.0
    t_hi = min(
        scfg.ibs_t_max_days,
        float(y_train["time"].max()) - 1.0,
        float(y_eval["time"].max()) - 1.0,
    )
    if t_hi <= t_lo:
        return float("nan")

    times = np.linspace(t_lo, t_hi, scfg.ibs_n_grid)
    surv_probs = _surv_probs_at(model, X_eval, times)
    try:
        ibs = integrated_brier_score(y_train, y_eval, surv_probs, times)
        return float(ibs)
    except Exception as exc:
        warnings.warn(
            f"IBS failed: {exc}. Returning NaN.", RuntimeWarning, stacklevel=2
        )
        return float("nan")


def compute_time_dependent_auc(
    model,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    horizons: List[int],
) -> Dict[int, float]:
    """
    Cumulative/dynamic AUC at each horizon t (days).
    Returns NaN for any horizon outside the observed follow-up range.
    """
    risk = model.predict(X_eval)
    out: Dict[int, float] = {}
    for t in horizons:
        try:
            auc_t, _mean = cumulative_dynamic_auc(
                y_train, y_eval, risk, float(t)
            )
            out[int(t)] = float(auc_t[0])
        except Exception as exc:
            warnings.warn(
                f"Time-dependent AUC failed at t={t}: {exc}.",
                RuntimeWarning,
                stacklevel=2,
            )
            out[int(t)] = float("nan")
    return out


# =========================================
# 5. HORIZON CLASSIFIERS (cross-link to Task 5)
# =========================================

def horizon_classifier_probs(
    model, X: np.ndarray, horizon_days: int
) -> np.ndarray:
    """
    Convert a fitted survival model into a binary classifier at one horizon.
    For each sample: P(T ≤ horizon) = 1 − S(horizon).
    """
    surv_funcs = model.predict_survival_function(X)
    return np.asarray(
        [1.0 - float(fn(float(horizon_days))) for fn in surv_funcs]
    )


def horizon_binary_labels(
    samples: List[Dict], horizon_days: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (y_bin, keep_mask) for evaluating a horizon classifier.

    A sample contributes to the horizon-t classification task only if its
    status AT TIME t is known:
      • event=1  → status known at every t
                   (positive if time ≤ t, else negative)
      • event=0  → status at t known only if time ≥ t;
                   censored-before-t → drop
    """
    y_bin = np.array(
        [1 if (s["event"] == 1 and s["time"] <= horizon_days) else 0
         for s in samples],
        dtype=int,
    )
    keep = np.array(
        [s["event"] == 1 or s["time"] >= horizon_days for s in samples],
        dtype=bool,
    )
    return y_bin, keep


def tune_horizon_threshold(
    model,
    X_val: np.ndarray,
    val_samples: List[Dict],
    horizon_days: int,
    objective: str = "f1",
    thresholds: Optional[np.ndarray] = None,
) -> float:
    """
    Tune the classification threshold for a horizon classifier on validation
    data by maximising the chosen objective (default: F1-score).

    WHY THIS IS NEEDED:
    With a fixed threshold of 0.5, the horizon classifiers produce F1=0
    because the positive event rate is only ~5% at 365 d and ~16% at 730 d.
    The survival model correctly assigns low probabilities (well below 0.5)
    to most patients, so nothing is ever predicted positive.

    Tuning on validation and applying on test is the correct approach —
    it mirrors Task 5's find_best_threshold() and avoids test-set leakage.

    Fallback: if the validation fold has too few positives to tune
    meaningfully, the positive prevalence is used as the threshold
    (a sensible and commonly used baseline).
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 181)

    y_bin, keep = horizon_binary_labels(val_samples, horizon_days)

    # Cannot tune if no kept samples or only one class present
    if keep.sum() == 0 or len(np.unique(y_bin[keep])) < 2:
        prevalence = float(y_bin[keep].mean()) if keep.sum() > 0 else 0.5
        return prevalence

    p = horizon_classifier_probs(model, X_val[keep], horizon_days)
    y_true = y_bin[keep]

    best_thr = float(np.mean(y_true))   # safe fallback = prevalence
    best_score = -np.inf

    for thr in thresholds:
        y_pred = (p >= thr).astype(int)

        if objective == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)

        elif objective == "balanced_accuracy":
            tn = int(((y_true == 0) & (y_pred == 0)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            sens = recall_score(y_true, y_pred, zero_division=0)
            score = (sens + spec) / 2.0

        else:
            raise ValueError(
                f"Unknown threshold objective: {objective}. "
                "Use 'f1' or 'balanced_accuracy'."
            )

        if score > best_score:
            best_score = score
            best_thr = float(thr)

    return best_thr


# =========================================
# 6. PATIENT-GROUPED BOOTSTRAP CIs
# =========================================

def patient_bootstrap_survival(
    model,
    X: np.ndarray,
    y: np.ndarray,
    samples: List[Dict],
    y_train: np.ndarray,
    horizons: List[int],
    n_boot: int = 500,
    random_state: int = 42,
    ci: float = 0.95,
) -> pd.DataFrame:
    """
    Patient-grouped bootstrap CIs for C-index and time-dependent AUC at each
    horizon. Whole patients (not individual landmarks) are resampled with
    replacement so within-patient correlation is respected.
    """
    pids = np.array([s["patient_id"] for s in samples])
    unique_pids = np.unique(pids)
    pid_to_idx = {p: np.where(pids == p)[0] for p in unique_pids}
    rng = np.random.default_rng(random_state)

    cindex_boot: List[float] = []
    auc_boot: Dict[int, List[float]] = {t: [] for t in horizons}

    for _ in range(n_boot):
        sampled_pids = rng.choice(unique_pids, size=len(unique_pids), replace=True)
        idx = np.concatenate([pid_to_idx[p] for p in sampled_pids])
        X_b, y_b = X[idx], y[idx]

        # Resampled cohort might contain no events — skip gracefully
        if y_b["event"].sum() == 0:
            cindex_boot.append(np.nan)
            for t in horizons:
                auc_boot[t].append(np.nan)
            continue

        try:
            cindex_boot.append(compute_cindex(model, X_b, y_b))
        except Exception:
            cindex_boot.append(np.nan)

        aucs = compute_time_dependent_auc(model, y_train, X_b, y_b, horizons)
        for t in horizons:
            auc_boot[t].append(aucs.get(t, np.nan))

    lo_q = (1.0 - ci) / 2.0 * 100.0
    hi_q = (1.0 + ci) / 2.0 * 100.0

    def _summ(arr: List[float]) -> Tuple[float, float, int]:
        a = np.asarray([x for x in arr if np.isfinite(x)], dtype=float)
        if len(a) == 0:
            return np.nan, np.nan, 0
        return (
            float(np.percentile(a, lo_q)),
            float(np.percentile(a, hi_q)),
            int(len(a)),
        )

    point_cindex = compute_cindex(model, X, y)
    point_aucs = compute_time_dependent_auc(model, y_train, X, y, horizons)

    rows = []
    lo, hi, n = _summ(cindex_boot)
    rows.append({
        "metric": "cindex",
        "horizon_days": np.nan,
        "point": point_cindex,
        "ci_lo": lo,
        "ci_hi": hi,
        "n_boot_valid": n,
    })
    for t in horizons:
        lo, hi, n = _summ(auc_boot[t])
        rows.append({
            "metric": "time_dep_auc",
            "horizon_days": int(t),
            "point": point_aucs.get(t, np.nan),
            "ci_lo": lo,
            "ci_hi": hi,
            "n_boot_valid": n,
        })
    return pd.DataFrame(rows)


# =========================================
# 7. SURVIVAL CALIBRATION PLOTS
# =========================================

def plot_survival_calibration(
    model,
    model_name: str,
    X_test: np.ndarray,
    test_samples: List[Dict],
    y_train: np.ndarray,
    scfg: SurvivalConfig,
    n_tertiles: int = 3,
    save_path: Optional[str] = None,
) -> None:
    """
    Calibration plot: mean predicted S(t) vs Kaplan-Meier S(t) stratified
    by risk tertile on the test set.

    WHY THIS MATTERS:
    The Integrated Brier Score summarises calibration in a single number but
    does not show WHERE the model is miscalibrated (at low, medium, or high
    risk patients). This plot directly shows whether the model's predicted
    survival curves match observed event rates across the risk spectrum.

    HOW IT WORKS:
    1. Predict risk score for every test sample.
    2. Split samples into n_tertiles equal groups by risk score.
    3. For each group: compute mean predicted S(t) across the time grid
       AND fit a Kaplan-Meier curve to the observed (time, event) data.
    4. Plot both on the same axes — if the model is well calibrated, the
       predicted and observed curves should overlap within each tertile.

    A well-calibrated model shows predicted ≈ observed for all tertiles.
    Systematic over/under-prediction reveals miscalibration direction.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend for Colab
        import matplotlib.pyplot as plt
        from sksurv.nonparametric import kaplan_meier_estimator
    except ImportError:
        print("matplotlib not available — skipping calibration plot.")
        return

    # Time grid for predicted curves
    t_lo = max(float(y_train["time"].min()), scfg.ibs_t_min_days) + 1.0
    t_hi = min(scfg.ibs_t_max_days, float(y_train["time"].max()) - 1.0)
    if t_hi <= t_lo:
        print(f"Cannot plot calibration for {model_name}: invalid time range.")
        return
    time_grid = np.linspace(t_lo, t_hi, scfg.ibs_n_grid)

    # Risk scores → tertile assignments
    risk_scores = model.predict(X_test)
    tertile_edges = np.percentile(
        risk_scores, np.linspace(0, 100, n_tertiles + 1)
    )
    tertile_labels = [
        f"Low risk (T{i+1})" if i == 0
        else f"High risk (T{i+1})" if i == n_tertiles - 1
        else f"Mid risk (T{i+1})"
        for i in range(n_tertiles)
    ]

    # Assign each sample to a tertile
    tertile_idx = np.digitize(risk_scores, tertile_edges[1:-1])

    # Predicted survival probabilities for all test samples
    surv_probs = _surv_probs_at(model, X_test, time_grid)  # (n, n_times)

    # Build arrays of (time, event) from test samples
    times_arr  = np.array([s["time"]  for s in test_samples])
    events_arr = np.array([s["event"] for s in test_samples], dtype=bool)

    # Colour palette — one colour per tertile
    colours = ["#2196F3", "#FF9800", "#F44336"]   # blue, orange, red

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(
        f"Survival Calibration — {model_name.upper()} (Test Set)\n"
        f"Mean predicted S(t) vs Kaplan-Meier S(t) by risk tertile",
        fontsize=13, fontweight="bold",
    )

    for ax_idx, ax in enumerate(axes):
        for tert in range(n_tertiles):
            mask = tertile_idx == tert
            if mask.sum() < 5:
                continue

            colour = colours[tert % len(colours)]
            label  = tertile_labels[tert]
            n_grp  = int(mask.sum())
            n_ev   = int(events_arr[mask].sum())

            # ── Mean predicted S(t) ───────────────────────────────────
            mean_pred = surv_probs[mask].mean(axis=0)   # (n_times,)

            # ── Kaplan-Meier observed S(t) ────────────────────────────
            try:
                km_times, km_surv = kaplan_meier_estimator(
                    events_arr[mask], times_arr[mask]
                )
            except Exception:
                continue

            if ax_idx == 0:
                # Left panel: predicted curves
                ax.plot(
                    time_grid, mean_pred,
                    color=colour, lw=2,
                    label=f"{label} (n={n_grp}, ev={n_ev})",
                )
            else:
                # Right panel: overlay predicted (solid) vs KM (dashed)
                ax.plot(
                    time_grid, mean_pred,
                    color=colour, lw=2, linestyle="-",
                    label=f"{label} predicted",
                )
                ax.step(
                    km_times, km_surv,
                    color=colour, lw=2, linestyle="--",
                    label=f"{label} KM observed",
                    where="post",
                )

        ax.set_xlabel("Days from landmark", fontsize=11)
        ax.set_ylabel("Survival probability S(t)", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(t_lo, t_hi)
        ax.axhline(0.5, color="grey", lw=0.8, linestyle=":")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title(
            "Mean predicted S(t) per tertile" if ax_idx == 0
            else "Predicted (—) vs KM observed (--)",
            fontsize=11,
        )

    plt.tight_layout()

    out = save_path or TASK6_OUT / f"task6_calibration_{model_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved calibration plot → {out}")


def plot_calibration_both_models(
    cox,
    rsf,
    X_test: np.ndarray,
    test_samples: List[Dict],
    y_train: np.ndarray,
    scfg: SurvivalConfig,
) -> None:
    """
    Single combined figure: Cox vs RSF calibration side by side.
    Four panels — Cox predicted | Cox overlay | RSF predicted | RSF overlay.
    Saves as task6_calibration_combined.png.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sksurv.nonparametric import kaplan_meier_estimator
    except ImportError:
        print("matplotlib not available — skipping combined calibration plot.")
        return

    t_lo = max(float(y_train["time"].min()), scfg.ibs_t_min_days) + 1.0
    t_hi = min(scfg.ibs_t_max_days, float(y_train["time"].max()) - 1.0)
    if t_hi <= t_lo:
        return
    time_grid  = np.linspace(t_lo, t_hi, scfg.ibs_n_grid)
    times_arr  = np.array([s["time"]  for s in test_samples])
    events_arr = np.array([s["event"] for s in test_samples], dtype=bool)
    colours    = ["#2196F3", "#FF9800", "#F44336"]
    n_tertiles = 3

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    fig.suptitle(
        "Survival Model Calibration — Test Set\n"
        "Mean predicted S(t) vs Kaplan-Meier S(t) stratified by risk tertile",
        fontsize=13, fontweight="bold",
    )

    model_pairs = [("Cox PH", cox, axes[0]), ("RSF", rsf, axes[1])]

    for m_label, model, row_axes in model_pairs:
        risk_scores   = model.predict(X_test)
        tertile_edges = np.percentile(risk_scores, np.linspace(0, 100, n_tertiles + 1))
        tertile_idx   = np.digitize(risk_scores, tertile_edges[1:-1])
        surv_probs    = _surv_probs_at(model, X_test, time_grid)

        tertile_labels = ["Low risk (T1)", "Mid risk (T2)", "High risk (T3)"]

        for ax_idx, ax in enumerate(row_axes):
            for tert in range(n_tertiles):
                mask   = tertile_idx == tert
                if mask.sum() < 5:
                    continue
                colour = colours[tert % len(colours)]
                label  = tertile_labels[tert]
                n_grp  = int(mask.sum())
                n_ev   = int(events_arr[mask].sum())

                mean_pred = surv_probs[mask].mean(axis=0)

                try:
                    km_times, km_surv = kaplan_meier_estimator(
                        events_arr[mask], times_arr[mask]
                    )
                except Exception:
                    continue

                if ax_idx == 0:
                    ax.plot(
                        time_grid, mean_pred,
                        color=colour, lw=2,
                        label=f"{label} (n={n_grp}, ev={n_ev})",
                    )
                else:
                    ax.plot(
                        time_grid, mean_pred,
                        color=colour, lw=2, linestyle="-",
                        label=f"{label} pred",
                    )
                    ax.step(
                        km_times, km_surv,
                        color=colour, lw=2, linestyle="--",
                        label=f"{label} KM",
                        where="post",
                    )

            ax.set_xlabel("Days from landmark", fontsize=10)
            ax.set_ylabel("Survival probability S(t)", fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.set_xlim(t_lo, t_hi)
            ax.axhline(0.5, color="grey", lw=0.8, linestyle=":")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="upper right")

            title_prefix = f"{m_label} — "
            ax.set_title(
                title_prefix + ("Mean predicted S(t)" if ax_idx == 0
                                else "Predicted (—) vs KM (--) "),
                fontsize=10,
            )

    plt.tight_layout()
    out = TASK6_OUT / "task6_calibration_combined.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved combined calibration plot → {out}")


# =========================================
# 8. MAIN
# =========================================

def main():
    ensure_dirs()
    cfg = Config()
    scfg = SurvivalConfig()
    set_seed(cfg.random_state)

    # ── Data loading (reuses Task 5) ─────────────────────────────────────
    print("Loading data...")
    df = load_longitudinal_data(cfg)
    train_ids, val_ids, test_ids = patient_level_split(df, cfg)

    train_visits_df = filter_df_by_patients(df, train_ids, cfg.patient_col)
    val_visits_df   = filter_df_by_patients(df, val_ids,   cfg.patient_col)
    test_visits_df  = filter_df_by_patients(df, test_ids,  cfg.patient_col)

    # ── Build (time, event) samples ──────────────────────────────────────
    train_samples = build_survival_samples(train_visits_df, cfg, scfg)
    val_samples   = build_survival_samples(val_visits_df,   cfg, scfg)
    test_samples  = build_survival_samples(test_visits_df,  cfg, scfg)

    def _ev(s): return sum(x["event"] for x in s)

    print(
        f"\nSurvival samples — "
        f"Train: {len(train_samples)} ({_ev(train_samples)} events) | "
        f"Val: {len(val_samples)} ({_ev(val_samples)} events) | "
        f"Test: {len(test_samples)} ({_ev(test_samples)} events)"
    )
    print(
        f"Min follow-up: {scfg.min_followup_days} d | "
        f"Eval horizons: {scfg.eval_horizons_days} d\n"
        f"Note: 180 d horizon excluded — min observed follow-up in "
        f"test set is ~184 d (data limitation, documented in report)."
    )

    if min(len(train_samples), len(val_samples), len(test_samples)) == 0:
        raise RuntimeError("A split has zero survival samples.")

    # ── Feature preparation (reuses Task 5's TabularPreprocessor) ────────
    print("\nPreparing features...")
    tab_prep = TabularPreprocessor().fit(samples_to_tabular_df(train_samples))
    X_train  = tab_prep.transform(samples_to_tabular_df(train_samples))
    X_val    = tab_prep.transform(samples_to_tabular_df(val_samples))
    X_test   = tab_prep.transform(samples_to_tabular_df(test_samples))

    y_train = make_surv_y(train_samples)
    y_val   = make_surv_y(val_samples)
    y_test  = make_surv_y(test_samples)

    feat_names = (
        np.asarray(tab_prep.feature_names_)
        if tab_prep.feature_names_ is not None
        else np.array([f"f{i}" for i in range(X_train.shape[1])])
    )

    # ── Fit models ───────────────────────────────────────────────────────
    print("\nFitting Cox PH...")
    cox = fit_cox(X_train, y_train, scfg)

    print("Fitting Random Survival Forest...")
    rsf = fit_rsf(X_train, y_train, cfg, scfg)

    # ── C-index across all splits ────────────────────────────────────────
    cindex_rows = []
    for m_name, model in [("cox", cox), ("rsf", rsf)]:
        for split_name, X, y in [
            ("train", X_train, y_train),
            ("val",   X_val,   y_val),
            ("test",  X_test,  y_test),
        ]:
            cindex_rows.append({
                "model":     m_name,
                "split":     split_name,
                "n_samples": len(y),
                "n_events":  int(y["event"].sum()),
                "cindex":    compute_cindex(model, X, y),
            })
    cindex_df = pd.DataFrame(cindex_rows)
    print("\n=== Harrell's C-index ===")
    print(cindex_df.to_string(index=False))

    # ── Integrated Brier Score on val + test ─────────────────────────────
    ibs_rows = []
    for m_name, model in [("cox", cox), ("rsf", rsf)]:
        for split_name, X, y in [
            ("val",  X_val,  y_val),
            ("test", X_test, y_test),
        ]:
            ibs_rows.append({
                "model":       m_name,
                "split":       split_name,
                "ibs":         compute_ibs(model, y_train, X, y, scfg),
                "t_min_days":  scfg.ibs_t_min_days,
                "t_max_days":  scfg.ibs_t_max_days,
            })
    ibs_df = pd.DataFrame(ibs_rows)
    print("\n=== Integrated Brier Score ===")
    print(ibs_df.to_string(index=False))

    # ── Time-dependent AUC ───────────────────────────────────────────────
    tdauc_rows = []
    for m_name, model in [("cox", cox), ("rsf", rsf)]:
        for split_name, X, y in [
            ("val",  X_val,  y_val),
            ("test", X_test, y_test),
        ]:
            aucs = compute_time_dependent_auc(
                model, y_train, X, y, scfg.eval_horizons_days
            )
            for t, v in aucs.items():
                tdauc_rows.append({
                    "model":        m_name,
                    "split":        split_name,
                    "horizon_days": t,
                    "auc":          v,
                })
    tdauc_df = pd.DataFrame(tdauc_rows)
    print("\n=== Time-dependent cumulative/dynamic AUC ===")
    print(tdauc_df.to_string(index=False))

    # ── Horizon classifiers with val-tuned thresholds ────────────────────
    # Fixed threshold=0.5 produces F1=0 because positive rate is only
    # ~5% at 365 d and ~16% at 730 d. We tune on validation (F1-objective)
    # and apply the same threshold to test — no leakage.
    print(
        f"\nTuning horizon classifier thresholds on validation set "
        f"(objective = {scfg.threshold_objective})..."
    )

    horizon_thresholds: Dict[str, Dict[int, float]] = {}
    thr_info_rows = []

    for m_name, model in [("cox", cox), ("rsf", rsf)]:
        horizon_thresholds[m_name] = {}
        for t in scfg.eval_horizons_days:
            thr = tune_horizon_threshold(
                model, X_val, val_samples, t,
                objective=scfg.threshold_objective,
            )
            horizon_thresholds[m_name][t] = thr

            # Compute positive prevalence in val for logging
            y_bin_v, keep_v = horizon_binary_labels(val_samples, t)
            prev_val = (
                float(y_bin_v[keep_v].mean())
                if keep_v.sum() > 0 else float("nan")
            )
            print(
                f"  {m_name} @ {t}d → "
                f"tuned threshold = {thr:.3f}  "
                f"(val prevalence = {prev_val:.3f}, "
                f"n_pos = {int(y_bin_v[keep_v].sum())}, "
                f"n_keep = {int(keep_v.sum())})"
            )
            thr_info_rows.append({
                "model":           m_name,
                "horizon_days":    t,
                "tuned_threshold": thr,
                "val_prevalence":  prev_val,
                "val_n_pos":       int(y_bin_v[keep_v].sum()),
                "val_n_keep":      int(keep_v.sum()),
                "objective":       scfg.threshold_objective,
            })

    thr_info_df = pd.DataFrame(thr_info_rows)

    horizon_rows = []
    for t in scfg.eval_horizons_days:
        for m_name, model in [("cox", cox), ("rsf", rsf)]:
            for split_name, X, samples in [
                ("val",  X_val,  val_samples),
                ("test", X_test, test_samples),
            ]:
                y_bin, keep = horizon_binary_labels(samples, t)
                if keep.sum() == 0:
                    continue
                if len(np.unique(y_bin[keep])) < 2:
                    continue

                p = horizon_classifier_probs(model, X[keep], t)
                thr = horizon_thresholds[m_name][t]

                m = classification_metrics(y_bin[keep], p, threshold=thr)
                m["model"]               = m_name
                m["split"]               = split_name
                m["horizon_days"]        = t
                m["n_keep"]              = int(keep.sum())
                m["n_positive_at_horizon"] = int(y_bin[keep].sum())
                m["threshold_source"]    = "val_tuned_f1"
                horizon_rows.append(m)

    horizon_df = pd.DataFrame(horizon_rows)
    print("\n=== Horizon classifiers (survival → binary at t, val-tuned threshold) ===")
    cols_first = [
        "model", "split", "horizon_days",
        "n_keep", "n_positive_at_horizon", "threshold",
        "roc_auc", "pr_auc", "brier_score",
        "accuracy", "precision", "recall_sensitivity", "specificity", "f1",
    ]
    other_cols = [c for c in horizon_df.columns if c not in cols_first]
    print(horizon_df[cols_first + other_cols].to_string(index=False))

    # ── Cox coefficients (interpretability) ──────────────────────────────
    cox_coef_df = pd.DataFrame({
        "feature":      feat_names,
        "coef":         cox.coef_,
        "abs_coef":     np.abs(cox.coef_),
        "hazard_ratio": np.exp(cox.coef_),
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)

    print("\n=== Cox PH coefficients (top 20 by |coef|) ===")
    print(cox_coef_df.head(20).to_string(index=False))
    print(
        "\nNote: Extreme hazard ratios for therapy variables are due to sparse "
        "category cells (some therapies appear in only 1-2 patients). "
        "The Cox coefficients for therapy should be interpreted with caution. "
        "RSF permutation importance is more reliable for feature ranking."
    )

    # ── RSF permutation importance on test ───────────────────────────────
    print(
        f"\nComputing RSF permutation importance on test "
        f"(n_repeats={scfg.perm_n_repeats})..."
    )
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(
        rsf, X_test, y_test,
        n_repeats=scfg.perm_n_repeats,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    rsf_perm_df = pd.DataFrame({
        "feature":         feat_names,
        "importance_mean": perm.importances_mean,
        "importance_std":  perm.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    print("\n=== RSF permutation importance (top 20 on test) ===")
    print(rsf_perm_df.head(20).to_string(index=False))

    # ── Bootstrap CIs on test for RSF and Cox ────────────────────────────
    print(
        f"\nComputing patient-grouped bootstrap CIs "
        f"(n_boot={scfg.n_boot})..."
    )
    boot_frames = []
    for m_name, model in [("cox", cox), ("rsf", rsf)]:
        b = patient_bootstrap_survival(
            model, X_test, y_test, test_samples, y_train,
            horizons=scfg.eval_horizons_days,
            n_boot=scfg.n_boot,
            random_state=cfg.random_state,
            ci=scfg.bootstrap_ci_level,
        )
        b.insert(0, "model", m_name)
        b.insert(1, "split", "test")
        boot_frames.append(b)
    boot_df = pd.concat(boot_frames, ignore_index=True)

    print("\n=== TEST: bootstrap 95% CIs (patient-grouped) ===")
    pretty = boot_df.copy()
    pretty["ci_95"] = [
        f"[{lo:.3f}, {hi:.3f}]"
        if (np.isfinite(lo) and np.isfinite(hi))
        else "[nan, nan]"
        for lo, hi in zip(pretty["ci_lo"], pretty["ci_hi"])
    ]
    print(
        pretty[[
            "model", "metric", "horizon_days",
            "point", "ci_95", "n_boot_valid",
        ]].to_string(index=False)
    )

    # ── Save RSF survival curves for test patients (for figures) ─────────
    curve_grid = np.linspace(
        scfg.ibs_t_min_days, scfg.ibs_t_max_days, scfg.ibs_n_grid
    )
    rsf_surv_probs_test = _surv_probs_at(rsf, X_test, curve_grid)
    curves_rows = []
    for i, s in enumerate(test_samples):
        row = {
            "patient_id":    s["patient_id"],
            "landmark_date": s["landmark_date"],
            "time":          s["time"],
            "event":         s["event"],
        }
        for j, t in enumerate(curve_grid):
            row[f"S_at_{int(round(t))}d"] = float(rsf_surv_probs_test[i, j])
        curves_rows.append(row)
    curves_df = pd.DataFrame(curves_rows)

    # ── Per-sample risk scores for downstream analysis ───────────────────
    risk_rows = []
    for m_name, model in [("cox", cox), ("rsf", rsf)]:
        for split_name, X, samples in [
            ("train", X_train, train_samples),
            ("val",   X_val,   val_samples),
            ("test",  X_test,  test_samples),
        ]:
            r = model.predict(X)
            for s, score in zip(samples, r):
                risk_rows.append({
                    "model":          m_name,
                    "split":          split_name,
                    "patient_id":     s["patient_id"],
                    "landmark_date":  s["landmark_date"],
                    "time":           s["time"],
                    "event":          s["event"],
                    "risk_score":     float(score),
                })
    risk_df = pd.DataFrame(risk_rows)

    # ── Survival calibration plots ───────────────────────────────────────
    # Shows mean predicted S(t) vs Kaplan-Meier S(t) for low / mid / high
    # risk tertiles on the test set. Well-calibrated curves should overlap.
    # Addresses Task 7 calibration requirement.
    print("\nGenerating survival calibration plots...")
    plot_calibration_both_models(
        cox, rsf, X_test, test_samples, y_train, scfg
    )

    # ── Save all CSV outputs ─────────────────────────────────────────────
    cindex_df.to_csv(TASK6_OUT / "task6_cindex.csv", index=False)
    ibs_df.to_csv(TASK6_OUT / "task6_ibs.csv", index=False)
    tdauc_df.to_csv(TASK6_OUT / "task6_time_dep_auc.csv", index=False)
    horizon_df.to_csv(TASK6_OUT / "task6_horizon_classifiers.csv", index=False)
    thr_info_df.to_csv(TASK6_OUT / "task6_horizon_thresholds.csv", index=False)
    boot_df.to_csv(TASK6_OUT / "task6_test_bootstrap_ci.csv", index=False)
    cox_coef_df.to_csv(TASK6_OUT / "task6_cox_coefficients.csv", index=False)
    rsf_perm_df.to_csv(TASK6_OUT / "task6_rsf_permutation_importance.csv", index=False)
    curves_df.to_csv(TASK6_OUT / "task6_test_rsf_survival_curves.csv", index=False)
    risk_df.to_csv(TASK6_OUT / "task6_risk_scores.csv", index=False)

    print(
        "\nSaved 11 files:\n"
        "  task6_cindex.csv\n"
        "  task6_ibs.csv\n"
        "  task6_time_dep_auc.csv\n"
        "  task6_horizon_classifiers.csv\n"
        "  task6_horizon_thresholds.csv\n"
        "  task6_test_bootstrap_ci.csv\n"
        "  task6_cox_coefficients.csv\n"
        "  task6_rsf_permutation_importance.csv\n"
        "  task6_test_rsf_survival_curves.csv\n"
        "  task6_risk_scores.csv\n"
        "  task6_calibration_combined.png  ← new: calibration plot"
    )


if __name__ == "__main__":
    main()