"""
Task 5 — final cleaned pipeline for MS disability progression prediction.

What this version changes relative to the last calibrated script:
1) Normalizes TherapyName before any preprocessing.
2) Removes delta/slope feature generation for age_years and disease_duration_years globally.
3) Tunes thresholds on patient-level validation predictions (max over landmarks),
   then reuses those thresholds for both patient-level and sample-level reporting.

Kept from the latest strong version:
- 180–730 day label window
- robust YYYYMM date parsing
- patient-level train/val/test split
- TCN + CatBoost + trimmed LR baseline
- isotonic calibration on validation only
- bootstrap CIs and reliability / ECE tables
- no final meta-stack
"""

import random
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

try:
    from catboost import CatBoostClassifier
except ImportError as exc:
    raise ImportError("catboost is required. Install with: pip install catboost") from exc

# ─── Path config: route I/O through the repo's central paths module ─────────
from paths import DATA_PATH, TASK5_OUT, ensure_dirs, assert_data_exists


# =========================================
# 1. CONFIGURATION
# =========================================


@dataclass
class Config:
    csv_path: str = ""  # Defaults to paths.DATA_PATH via __post_init__

    patient_col: str = "subject_id"
    date_col: str = "MRIDateYYYYMM"
    dp_col: str = "DP"

    visit_numeric_cols: List[str] = field(default_factory=lambda: [
        "EDSSValue",
        "age_years",
        "disease_duration_years",
        "Brain (WM+GM) volume cm3",
        "Grey Matter (GM) volume cm3",
        "White Matter (WM) volume cm3",
        "Lateral ventricle total volume cm3",
        "lesionvolume",
        "lesioncount",
    ])
    visit_categorical_cols: List[str] = field(default_factory=lambda: [
        "TherapyName",
        "sex_id",
    ])

    tabular_only_numeric_cols: List[str] = field(default_factory=list)
    tabular_only_categorical_cols: List[str] = field(default_factory=list)

    therapy_col: Optional[str] = "TherapyName"
    progression_date_col: str = "ProgressionYYYYMM"

    min_gap_days: int = 180
    horizon_days: int = 730

    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42

    tcn_channels: Sequence[int] = (64, 64, 64)
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.2
    tcn_lr: float = 1e-3
    tcn_weight_decay: float = 1e-4
    tcn_batch_size: int = 64
    tcn_epochs: int = 60
    tcn_patience: int = 8

    catboost_params: Dict = field(default_factory=lambda: {
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "iterations": 300,
        "depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "l2_leaf_reg": 6.0,
        "random_strength": 1.0,
        "random_seed": 42,
        "thread_count": -1,
        "verbose": False,
        "allow_writing_files": False,
        "od_type": "Iter",
        "od_wait": 50,
        "use_best_model": True,
    })

    max_negative_landmarks_per_patient: Optional[int] = None
    bootstrap_n: int = 1000
    reliability_bins: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        # If user didn't override, use centralised path config
        if not self.csv_path:
            self.csv_path = str(DATA_PATH)


MODEL_NAMES = ["lr", "tcn", "cat"]
TREND_EXCLUDE = {"age_years", "disease_duration_years"}
THERAPY_ALIASES = {
    "tecfidera": "dimethyl_fumarate",
    "dimetilfumarat": "dimethyl_fumarate",
    "dimethyl fumarate": "dimethyl_fumarate",
    "dimethylfumarate": "dimethyl_fumarate",
    "fingolimod": "fingolimod",
    "gilenya": "fingolimod",
    "natalizumab": "natalizumab",
    "tysabri": "natalizumab",
    "aubagio": "teriflunomide",
    "teriflunomide": "teriflunomide",
    "ocopri": "ocrelizumab",
    "ocrelizumab": "ocrelizumab",
    "rituximab": "rituximab",
    "interferon beta": "interferon_beta",
    "interferon-beta": "interferon_beta",
    "interferon_beta": "interferon_beta",
    "glatiramer acetate": "glatiramer_acetate",
    "glatiramer_acetate": "glatiramer_acetate",
}


# =========================================
# 2. UTILITIES
# =========================================


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def _parse_yyyymm(val: object) -> pd.Timestamp:
    if val is None:
        return pd.NaT
    try:
        if pd.isna(val):
            return pd.NaT
    except (TypeError, ValueError):
        pass

    if isinstance(val, pd.Timestamp):
        return val
    if isinstance(val, np.datetime64):
        return pd.Timestamp(val)

    if isinstance(val, (int, float, np.integer, np.floating)):
        v = int(val)
        if 190001 <= v <= 210012 and 1 <= (v % 100) <= 12:
            return pd.Timestamp(year=v // 100, month=v % 100, day=1)
        return pd.to_datetime(val, errors="coerce")

    if isinstance(val, str):
        s = val.strip()
        if s.endswith(".0") and s[:-2].isdigit() and len(s[:-2]) == 6:
            s = s[:-2]
        if s.isdigit() and len(s) == 6:
            v = int(s)
            if 1 <= (v % 100) <= 12:
                return pd.Timestamp(year=v // 100, month=v % 100, day=1)
        return pd.to_datetime(s, errors="coerce")

    return pd.to_datetime(val, errors="coerce")



def normalize_therapy_name(val: object) -> object:
    if val is None:
        return np.nan
    try:
        if pd.isna(val):
            return np.nan
    except Exception:
        pass
    s = str(val).strip().lower()
    if not s:
        return np.nan
    s = " ".join(s.split())
    return THERAPY_ALIASES.get(s, s.replace(" ", "_"))



def normalize_general_categorical(val: object) -> object:
    if val is None:
        return np.nan
    try:
        if pd.isna(val):
            return np.nan
    except Exception:
        pass
    s = str(val).strip()
    return s.lower() if s else np.nan



def filter_df_by_patients(df: pd.DataFrame, patient_ids: Sequence, patient_col: str) -> pd.DataFrame:
    pid_set = set(patient_ids.tolist() if isinstance(patient_ids, np.ndarray) else list(patient_ids))
    return df[df[patient_col].isin(pid_set)].copy().reset_index(drop=True)



def samples_to_tabular_df(samples: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame([s["tabular"] for s in samples])


# =========================================
# 3. DATA LOADING AND PREPARATION
# =========================================


def load_longitudinal_data(cfg: Config) -> pd.DataFrame:
    if cfg.csv_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(cfg.csv_path)
    else:
        df = pd.read_csv(cfg.csv_path)

    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    if not pd.api.types.is_datetime64_any_dtype(df[cfg.date_col]):
        df[cfg.date_col] = df[cfg.date_col].apply(_parse_yyyymm)

    extra: Dict[str, pd.Series] = {}
    if "disease_duration_years" not in df.columns:
        diag_dates = pd.Series([_parse_yyyymm(v) for v in df["DiagnosisDateYYYYMM"]], index=df.index)
        duration_days = (df[cfg.date_col] - diag_dates).dt.days
        extra["disease_duration_years"] = (duration_days / 365.25).clip(lower=0)
    if "progression_date" not in df.columns:
        extra["progression_date"] = df[cfg.progression_date_col].apply(_parse_yyyymm)
    if extra:
        df = pd.concat([df, pd.DataFrame(extra, index=df.index)], axis=1)

    required_cols = list(dict.fromkeys(
        [cfg.patient_col, cfg.date_col, cfg.dp_col] +
        cfg.visit_numeric_cols + cfg.visit_categorical_cols +
        cfg.tabular_only_numeric_cols + cfg.tabular_only_categorical_cols
    ))
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[cfg.date_col].isna().all():
        raise ValueError(f"All values in {cfg.date_col} failed to parse.")

    # normalize categorical strings before feature generation
    if cfg.therapy_col and cfg.therapy_col in df.columns:
        df[cfg.therapy_col] = df[cfg.therapy_col].apply(normalize_therapy_name)
    for c in cfg.visit_categorical_cols + cfg.tabular_only_categorical_cols:
        if c == cfg.therapy_col:
            continue
        if c in df.columns:
            df[c] = df[c].apply(normalize_general_categorical)

    df = df.sort_values([cfg.patient_col, cfg.date_col]).reset_index(drop=True)
    df[cfg.dp_col] = df[cfg.dp_col].fillna(0).astype(int).clip(0, 1)
    df = add_time_features(df, cfg)
    return df



def add_time_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    df["time_since_prev_days"] = df.groupby(cfg.patient_col)[cfg.date_col].diff().dt.days.fillna(0)
    baseline_dates = df.groupby(cfg.patient_col)[cfg.date_col].transform("min")
    df["time_since_baseline_days"] = (df[cfg.date_col] - baseline_dates).dt.days
    missing_indicators = pd.concat({f"{c}__missing": df[c].isna().astype(int) for c in cfg.visit_numeric_cols}, axis=1)
    df = pd.concat([df, missing_indicators], axis=1)
    return df


# =========================================
# 4. SPLITTING AND LANDMARKING
# =========================================


def patient_level_split(df: pd.DataFrame, cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    patient_table = df.groupby(cfg.patient_col)[cfg.dp_col].max().rename("ever_dp").reset_index()
    pids = patient_table[cfg.patient_col].to_numpy()
    ys = patient_table["ever_dp"].to_numpy()

    train_ids, temp_ids, _, temp_y = train_test_split(
        pids, ys,
        test_size=(1.0 - cfg.train_size),
        stratify=ys if len(np.unique(ys)) > 1 else None,
        random_state=cfg.random_state,
    )
    rel_test = cfg.test_size / (cfg.val_size + cfg.test_size)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=rel_test,
        stratify=temp_y if len(np.unique(temp_y)) > 1 else None,
        random_state=cfg.random_state,
    )
    return np.array(train_ids), np.array(val_ids), np.array(test_ids)



def linear_slope_per_year(time_days: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(time_days) & np.isfinite(values)
    if valid.sum() < 2:
        return np.nan
    x = time_days[valid].astype(float)
    y = values[valid].astype(float)
    if np.allclose(x, x[0]):
        return np.nan
    return float(np.polyfit(x, y, deg=1)[0] * 365.25)



def _safe_get(series: pd.Series, col: str):
    return series[col] if col in series.index else np.nan



def build_tabular_features(history: pd.DataFrame, cfg: Config) -> Dict:
    current = history.iloc[-1]
    baseline = history.iloc[0]
    previous = history.iloc[-2] if len(history) >= 2 else None

    num_cols = list(dict.fromkeys(cfg.visit_numeric_cols + cfg.tabular_only_numeric_cols))
    cat_cols = list(dict.fromkeys(cfg.visit_categorical_cols + cfg.tabular_only_categorical_cols))
    feats: Dict[str, object] = {}

    for col in num_cols:
        cur_val = _safe_get(current, col)
        base_val = _safe_get(baseline, col)
        prev_val = _safe_get(previous, col) if previous is not None else np.nan

        feats[f"current__{col}"] = cur_val
        feats[f"baseline__{col}"] = base_val

        if col not in TREND_EXCLUDE:
            feats[f"delta_prev__{col}"] = (cur_val - prev_val if pd.notna(cur_val) and pd.notna(prev_val) else np.nan)
            feats[f"slope_per_year__{col}"] = linear_slope_per_year(
                history["time_since_baseline_days"].to_numpy(),
                history[col].to_numpy(),
            )

    feats["num_prior_visits"] = max(len(history) - 1, 0)
    feats["time_since_last_visit_days"] = float(_safe_get(current, "time_since_prev_days"))
    feats["time_since_baseline_days"] = float(_safe_get(current, "time_since_baseline_days"))

    for col in cat_cols:
        feats[f"current__{col}"] = _safe_get(current, col)
        feats[f"baseline__{col}"] = _safe_get(baseline, col)

    if cfg.therapy_col and cfg.therapy_col in history.columns:
        cur_tx = _safe_get(current, cfg.therapy_col)
        prev_tx = _safe_get(previous, cfg.therapy_col) if previous is not None else np.nan
        feats["therapy_changed"] = int(previous is not None and pd.notna(cur_tx) and pd.notna(prev_tx) and cur_tx != prev_tx)
    else:
        feats["therapy_changed"] = 0
    return feats



def build_landmark_samples(df_split: pd.DataFrame, cfg: Config) -> List[Dict]:
    samples: List[Dict] = []
    seq_numeric_cols = list(cfg.visit_numeric_cols) + ["time_since_prev_days", "time_since_baseline_days"] + [f"{c}__missing" for c in cfg.visit_numeric_cols]
    seq_categorical_cols = list(cfg.visit_categorical_cols)

    for patient_id, pdf in df_split.groupby(cfg.patient_col):
        pdf = pdf.sort_values(cfg.date_col).reset_index(drop=True)
        progression_date = pdf["progression_date"].iloc[0]
        has_progression = pd.notna(progression_date)
        last_date = pdf[cfg.date_col].max()
        negative_count = 0

        for i in range(len(pdf)):
            landmark_date = pdf.loc[i, cfg.date_col]
            history = pdf.loc[:i].copy()

            if has_progression:
                if landmark_date >= progression_date:
                    continue
                gap_days = int((progression_date - landmark_date).days)
                max_followup = int((last_date - landmark_date).days)
                if gap_days < cfg.min_gap_days:
                    continue
                if gap_days <= cfg.horizon_days:
                    label = 1
                else:
                    if max_followup >= cfg.horizon_days:
                        label = 0
                    else:
                        continue
            else:
                max_followup = int((last_date - landmark_date).days)
                if max_followup >= cfg.horizon_days:
                    label = 0
                else:
                    continue

            if label == 0 and cfg.max_negative_landmarks_per_patient is not None:
                if negative_count >= cfg.max_negative_landmarks_per_patient:
                    continue
                negative_count += 1

            seq_df = history[seq_numeric_cols + seq_categorical_cols].copy()
            tabular = build_tabular_features(history, cfg)
            samples.append({
                "patient_id": patient_id,
                "landmark_date": landmark_date,
                "label": int(label),
                "seq_df": seq_df,
                "tabular": tabular,
            })
    return samples


# =========================================
# 5. PREPROCESSORS
# =========================================


class SequencePreprocessor:
    def __init__(self, numeric_cols: List[str], categorical_cols: List[str]):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.numeric_imputer = SimpleImputer(strategy="median")
        self.numeric_scaler = StandardScaler()
        self.categorical_imputer = SimpleImputer(strategy="most_frequent")
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def fit(self, visits_df: pd.DataFrame) -> "SequencePreprocessor":
        X_num = visits_df[self.numeric_cols]
        X_num_imp = self.numeric_imputer.fit_transform(X_num)
        self.numeric_scaler.fit(X_num_imp)
        if self.categorical_cols:
            X_cat = visits_df[self.categorical_cols]
            X_cat_imp = self.categorical_imputer.fit_transform(X_cat)
            self.encoder.fit(X_cat_imp)
        return self

    def transform_sequence(self, seq_df: pd.DataFrame) -> np.ndarray:
        X_num = self.numeric_scaler.transform(self.numeric_imputer.transform(seq_df[self.numeric_cols]))
        if self.categorical_cols:
            X_cat = self.encoder.transform(self.categorical_imputer.transform(seq_df[self.categorical_cols]))
            X = np.concatenate([X_num, X_cat], axis=1)
        else:
            X = X_num
        return X.astype(np.float32)

    @property
    def output_dim(self) -> int:
        cat_dim = int(sum(len(cats) for cats in self.encoder.categories_)) if self.categorical_cols else 0
        return len(self.numeric_cols) + cat_dim


class TabularPreprocessor:
    def __init__(self):
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.transformer: Optional[ColumnTransformer] = None
        self.feature_names_: Optional[np.ndarray] = None

    def fit(self, tabular_df: pd.DataFrame) -> "TabularPreprocessor":
        self.numeric_cols = [c for c in tabular_df.columns if pd.api.types.is_numeric_dtype(tabular_df[c])]
        self.categorical_cols = [c for c in tabular_df.columns if c not in self.numeric_cols]

        transformers = []
        if self.numeric_cols:
            transformers.append(("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), self.numeric_cols))
        if self.categorical_cols:
            transformers.append(("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), self.categorical_cols))

        self.transformer = ColumnTransformer(transformers=transformers, sparse_threshold=0.0)
        self.transformer.fit(tabular_df)
        try:
            self.feature_names_ = self.transformer.get_feature_names_out()
        except Exception:
            self.feature_names_ = None
        return self

    def transform(self, tabular_df: pd.DataFrame) -> np.ndarray:
        if self.transformer is None:
            raise RuntimeError("TabularPreprocessor must be fitted before calling transform.")
        return self.transformer.transform(tabular_df).astype(np.float32)



def fit_preprocessors(visits_df_train: pd.DataFrame, train_samples: List[Dict], cfg: Config) -> Tuple[SequencePreprocessor, TabularPreprocessor]:
    seq_numeric_cols = list(cfg.visit_numeric_cols) + ["time_since_prev_days", "time_since_baseline_days"] + [f"{c}__missing" for c in cfg.visit_numeric_cols]
    seq_categorical_cols = list(cfg.visit_categorical_cols)
    seq_prep = SequencePreprocessor(seq_numeric_cols, seq_categorical_cols).fit(visits_df_train)
    tab_prep = TabularPreprocessor().fit(samples_to_tabular_df(train_samples))
    return seq_prep, tab_prep



def transform_samples(samples: List[Dict], seq_prep: SequencePreprocessor, tab_prep: TabularPreprocessor) -> Dict[str, np.ndarray]:
    if not samples:
        raise ValueError("No samples provided to transform.")
    seq_arrays = [seq_prep.transform_sequence(s["seq_df"]) for s in samples]
    lengths = np.array([len(x) for x in seq_arrays], dtype=np.int64)
    feat_dim = seq_prep.output_dim
    max_len = int(lengths.max())

    X_seq = np.zeros((len(samples), max_len, feat_dim), dtype=np.float32)
    for i, arr in enumerate(seq_arrays):
        X_seq[i, :len(arr), :] = arr

    X_tab = tab_prep.transform(samples_to_tabular_df(samples))
    y = np.array([s["label"] for s in samples], dtype=np.float32)
    groups = np.array([s["patient_id"] for s in samples])
    return {"X_seq": X_seq, "lengths": lengths, "X_tab": X_tab, "y": y, "groups": groups}


# =========================================
# 6. MODELS
# =========================================


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(out_c, out_c, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_c, out_c, kernel_size=1) if in_c != out_c else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNClassifier(nn.Module):
    def __init__(self, input_dim, channels=(64, 64, 64), kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for i, out_c in enumerate(channels):
            layers.append(TemporalBlock(prev, out_c, kernel_size, 2 ** i, dropout))
            prev = out_c
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(prev, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x, lengths):
        z = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        max_len = z.size(1)
        mask = (torch.arange(max_len, device=z.device)[None, :] < lengths[:, None]).float().unsqueeze(-1)
        valid_counts = mask.sum(dim=1).clamp_min(1.0)
        pooled = (z * mask).sum(dim=1) / valid_counts
        return self.head(pooled).squeeze(1)


class SequenceDataset(Dataset):
    def __init__(self, X_seq, lengths, y):
        self.X_seq = torch.from_numpy(X_seq).float()
        self.lengths = torch.from_numpy(lengths).long()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.lengths[idx], self.y[idx]



def compute_pos_weight(y: np.ndarray) -> torch.Tensor:
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return torch.tensor(1.0 if pos == 0 or neg == 0 else neg / pos, dtype=torch.float32)



def train_tcn(train_pack, valid_pack, cfg, verbose=False):
    model = TCNClassifier(
        input_dim=train_pack["X_seq"].shape[-1],
        channels=cfg.tcn_channels,
        kernel_size=cfg.tcn_kernel_size,
        dropout=cfg.tcn_dropout,
    ).to(cfg.device)

    train_loader = DataLoader(SequenceDataset(train_pack["X_seq"], train_pack["lengths"], train_pack["y"]), batch_size=cfg.tcn_batch_size, shuffle=True)
    valid_loader = DataLoader(SequenceDataset(valid_pack["X_seq"], valid_pack["lengths"], valid_pack["y"]), batch_size=cfg.tcn_batch_size, shuffle=False)

    criterion = nn.BCEWithLogitsLoss(pos_weight=compute_pos_weight(train_pack["y"]).to(cfg.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.tcn_lr, weight_decay=cfg.tcn_weight_decay)

    best_state = None
    best_score = -np.inf
    patience = 0
    for epoch in range(cfg.tcn_epochs):
        model.train()
        for Xb, Lb, yb in train_loader:
            Xb, Lb, yb = Xb.to(cfg.device), Lb.to(cfg.device), yb.to(cfg.device)
            optimizer.zero_grad()
            loss = criterion(model(Xb, Lb), yb)
            loss.backward()
            optimizer.step()

        probs, trues = [], []
        model.eval()
        with torch.no_grad():
            for Xb, Lb, yb in valid_loader:
                probs.append(torch.sigmoid(model(Xb.to(cfg.device), Lb.to(cfg.device))).cpu().numpy())
                trues.append(yb.numpy())
        probs = np.concatenate(probs)
        trues = np.concatenate(trues)
        score = average_precision_score(trues, probs) if len(np.unique(trues)) > 1 else -np.inf
        if verbose:
            print(f"Epoch {epoch+1:03d} | Valid AUPRC = {score:.5f}")
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.tcn_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_tcn(model, pack, cfg):
    loader = DataLoader(SequenceDataset(pack["X_seq"], pack["lengths"], pack["y"]), batch_size=cfg.tcn_batch_size, shuffle=False)
    probs = []
    model.eval()
    for Xb, Lb, _ in loader:
        probs.append(torch.sigmoid(model(Xb.to(cfg.device), Lb.to(cfg.device))).cpu().numpy())
    return np.concatenate(probs)



def train_catboost(X_train, y_train, X_valid, y_valid, cfg):
    pos = max((y_train == 1).sum(), 1)
    neg = max((y_train == 0).sum(), 1)
    params = dict(cfg.catboost_params)
    params["scale_pos_weight"] = float(neg / pos)
    if len(np.unique(y_valid)) < 2:
        warnings.warn("CatBoost eval fold has one class; using Logloss + use_best_model=False.", RuntimeWarning, stacklevel=2)
        params["eval_metric"] = "Logloss"
        params["use_best_model"] = False
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=False)
    return model



def _select_parsimonious_feature_mask(feature_names: np.ndarray) -> np.ndarray:
    keep = np.ones(len(feature_names), dtype=bool)
    for i, name in enumerate(feature_names):
        if "baseline__" in name or "delta_prev__" in name:
            keep[i] = False
    return keep



def train_logreg(X_train: np.ndarray, y_train: np.ndarray, cfg: Config, feature_names: Optional[np.ndarray] = None):
    if len(np.unique(y_train)) < 2:
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X_train, y_train)
        return model, None
    mask = None
    X_fit = X_train
    if feature_names is not None:
        mask = _select_parsimonious_feature_mask(feature_names)
        if mask.sum() >= 2:
            X_fit = X_train[:, mask]
    model = LogisticRegression(
        penalty="l2", C=0.1, solver="liblinear", class_weight="balanced", max_iter=2000, random_state=cfg.random_state
    )
    model.fit(X_fit, y_train)
    return model, mask



def _apply_lr_mask(X: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    return X if mask is None else X[:, mask]



def _lr_predict_proba(model, X):
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        classes = getattr(model, "classes_", np.array([0]))
        return proba[:, 0] if classes[0] == 1 else 1.0 - proba[:, 0]
    return proba[:, 1]


# =========================================
# 7. TRAIN / PREDICT
# =========================================


def fit_models(train_visits_df, train_samples, val_samples, test_samples, cfg: Config):
    seq_prep, tab_prep = fit_preprocessors(train_visits_df, train_samples, cfg)
    train_pack = transform_samples(train_samples, seq_prep, tab_prep)
    val_pack = transform_samples(val_samples, seq_prep, tab_prep)
    test_pack = transform_samples(test_samples, seq_prep, tab_prep)

    tcn = train_tcn(train_pack, val_pack, cfg, verbose=False)
    cat = train_catboost(train_pack["X_tab"], train_pack["y"], val_pack["X_tab"], val_pack["y"], cfg)
    lr, lr_mask = train_logreg(train_pack["X_tab"], train_pack["y"], cfg, feature_names=tab_prep.feature_names_)

    def predict_all(pack):
        p_tcn = predict_tcn(tcn, pack, cfg)
        p_cat = cat.predict_proba(pack["X_tab"])[:, 1]
        p_lr = _lr_predict_proba(lr, _apply_lr_mask(pack["X_tab"], lr_mask))
        return p_lr, p_tcn, p_cat

    tr_lr, tr_tcn, tr_cat = predict_all(train_pack)
    va_lr, va_tcn, va_cat = predict_all(val_pack)
    te_lr, te_tcn, te_cat = predict_all(test_pack)

    return {
        "preprocessors": {"sequence": seq_prep, "tabular": tab_prep},
        "models": {"lr": lr, "tcn": tcn, "cat": cat},
        "packs": {"train": train_pack, "val": val_pack, "test": test_pack},
        "predictions": {
            "train_lr": tr_lr, "train_tcn": tr_tcn, "train_cat": tr_cat,
            "val_lr": va_lr, "val_tcn": va_tcn, "val_cat": va_cat,
            "test_lr": te_lr, "test_tcn": te_tcn, "test_cat": te_cat,
        },
    }


# =========================================
# 8. METRICS / THRESHOLDS / CALIBRATION
# =========================================


def classification_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "n_samples": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan
    return out



def find_best_threshold(y_true, y_prob, objective="f1", thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 181)
    best = None
    for thr in thresholds:
        m = classification_metrics(y_true, y_prob, threshold=float(thr))
        if objective == "f1":
            score = m["f1"]
        elif objective == "balanced_accuracy":
            score = (m["recall_sensitivity"] + m["specificity"]) / 2.0 if np.isfinite(m["specificity"]) else -np.inf
        else:
            raise ValueError(f"Unsupported objective: {objective}")
        cand = (score, m["specificity"] if np.isfinite(m["specificity"]) else -np.inf, -abs(float(thr) - 0.5))
        if best is None or cand > best["cand"]:
            best = {"threshold": float(thr), "score": float(score), "metrics": m, "cand": cand}
    return {"threshold": best["threshold"], "score": best["score"], **best["metrics"]}



def aggregate_to_patient_level(samples, y_prob, agg="max"):
    df = pd.DataFrame({
        "pid": [s["patient_id"] for s in samples],
        "y": [s["label"] for s in samples],
        "p": y_prob,
    })
    grp = df.groupby("pid")
    if agg == "max":
        p_pat = grp["p"].max().to_numpy()
    elif agg == "mean":
        p_pat = grp["p"].mean().to_numpy()
    else:
        raise ValueError(f"Unknown agg: {agg}")
    y_pat = grp["y"].max().to_numpy()
    return y_pat, p_pat



def tune_thresholds_patient_level(artifacts: Dict, samples_by_split: Dict[str, List[Dict]], split: str = "val", agg: str = "max") -> Tuple[Dict[str, float], pd.DataFrame]:
    tuned: Dict[str, float] = {}
    rows = []
    samples = samples_by_split[split]
    for m_name in MODEL_NAMES:
        y_pat, p_pat = aggregate_to_patient_level(samples, artifacts["predictions"][f"{split}_{m_name}"], agg=agg)
        best = find_best_threshold(y_pat, p_pat, objective="f1")
        tuned[m_name] = float(best["threshold"])
        rows.append({
            "model": m_name,
            "objective": "patient_f1",
            "agg": agg,
            "best_threshold": float(best["threshold"]),
            "val_f1": float(best["f1"]),
            "val_roc_auc": float(best["roc_auc"]) if np.isfinite(best["roc_auc"]) else np.nan,
            "val_pr_auc": float(best["pr_auc"]) if np.isfinite(best["pr_auc"]) else np.nan,
            "val_sens": float(best["recall_sensitivity"]),
            "val_spec": float(best["specificity"]),
        })
    return tuned, pd.DataFrame(rows)



def metrics_table(artifacts: Dict, thresholds=None):
    thresholds = thresholds or {}
    rows = []
    for split in ["train", "val", "test"]:
        y_true = artifacts["packs"][split]["y"]
        for m_name in MODEL_NAMES:
            thr = float(thresholds.get(m_name, 0.5))
            row = classification_metrics(y_true, artifacts["predictions"][f"{split}_{m_name}"], threshold=thr)
            row["split"] = split
            row["model"] = m_name
            rows.append(row)
    cols = ["split", "model", "n_samples", "positive_rate", "threshold", "roc_auc", "pr_auc", "accuracy", "precision", "recall_sensitivity", "specificity", "f1", "brier_score"]
    return pd.DataFrame(rows)[cols]



def patient_level_metrics_table(artifacts: Dict, samples_by_split: Dict[str, List[Dict]], thresholds=None, agg="max"):
    thresholds = thresholds or {}
    rows = []
    for split, samples in samples_by_split.items():
        for m_name in MODEL_NAMES:
            y_pat, p_pat = aggregate_to_patient_level(samples, artifacts["predictions"][f"{split}_{m_name}"], agg=agg)
            thr = float(thresholds.get(m_name, 0.5))
            row = classification_metrics(y_pat, p_pat, threshold=thr)
            row["split"] = split
            row["model"] = m_name
            row["agg"] = agg
            rows.append(row)
    cols = ["split", "model", "agg", "n_samples", "positive_rate", "threshold", "roc_auc", "pr_auc", "accuracy", "precision", "recall_sensitivity", "specificity", "f1", "brier_score"]
    return pd.DataFrame(rows)[cols]



def fit_isotonic_calibrators(artifacts: Dict) -> Dict[str, Optional[IsotonicRegression]]:
    y_val = artifacts["packs"]["val"]["y"].astype(int)
    out: Dict[str, Optional[IsotonicRegression]] = {}
    for m_name in MODEL_NAMES:
        p_val = np.asarray(artifacts["predictions"][f"val_{m_name}"], dtype=float)
        if len(np.unique(y_val)) < 2:
            out[m_name] = None
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_val, y_val)
        out[m_name] = iso
    return out



def build_calibrated_artifacts(artifacts: Dict, calibrators: Dict[str, Optional[IsotonicRegression]]) -> Dict:
    cal_preds: Dict[str, np.ndarray] = {}
    for key, val in artifacts["predictions"].items():
        parts = key.split("_", 1)
        if len(parts) == 2 and parts[0] in {"train", "val", "test"} and parts[1] in MODEL_NAMES:
            iso = calibrators.get(parts[1])
            cal_preds[key] = np.asarray(val, dtype=float).copy() if iso is None else np.asarray(iso.predict(np.asarray(val, dtype=float)), dtype=float)
        else:
            cal_preds[key] = val
    return {**artifacts, "predictions": cal_preds}



def reliability_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    rows = []
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin": b, "bin_lo": float(bins[b]), "bin_hi": float(bins[b + 1]), "count": 0, "pred_mean": np.nan, "obs_rate": np.nan, "abs_gap": np.nan})
            continue
        pred_mean = float(y_prob[mask].mean())
        obs_rate = float(y_true[mask].mean())
        gap = abs(pred_mean - obs_rate)
        ece += (count / n) * gap
        rows.append({"bin": b, "bin_lo": float(bins[b]), "bin_hi": float(bins[b + 1]), "count": count, "pred_mean": pred_mean, "obs_rate": obs_rate, "abs_gap": gap})
    df = pd.DataFrame(rows)
    df.attrs["ece"] = float(ece)
    return df



def reliability_table_all_models(artifacts: Dict, split: str = "test", n_bins: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y_true = artifacts["packs"][split]["y"].astype(int)
    bin_rows, ece_rows = [], []
    for m_name in MODEL_NAMES:
        tbl = reliability_table(y_true, artifacts["predictions"][f"{split}_{m_name}"], n_bins=n_bins)
        ece_rows.append({"split": split, "model": m_name, "ece": float(tbl.attrs["ece"]), "n_bins": n_bins})
        tbl = tbl.copy()
        tbl.insert(0, "split", split)
        tbl.insert(1, "model", m_name)
        bin_rows.append(tbl)
    return pd.concat(bin_rows, ignore_index=True), pd.DataFrame(ece_rows)



def patient_bootstrap_ci(samples: List[Dict], y_prob: np.ndarray, threshold: float = 0.5, n_boot: int = 1000, random_state: int = 42, ci: float = 0.95) -> pd.DataFrame:
    y_true = np.array([s["label"] for s in samples], dtype=int)
    pids = np.array([s["patient_id"] for s in samples])
    unique_pids = np.unique(pids)
    n_pat = len(unique_pids)
    pid_to_idx = {p: np.where(pids == p)[0] for p in unique_pids}
    rng = np.random.default_rng(random_state)

    def _auc(yt, yp):
        return roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else np.nan

    def _prauc(yt, yp):
        return average_precision_score(yt, yp) if len(np.unique(yt)) > 1 else np.nan

    def _brier(yt, yp):
        return brier_score_loss(yt, yp)

    def _f1_t(yt, yp):
        return f1_score(yt, (yp >= threshold).astype(int), zero_division=0)

    def _prec_t(yt, yp):
        return precision_score(yt, (yp >= threshold).astype(int), zero_division=0)

    def _sens_t(yt, yp):
        return recall_score(yt, (yp >= threshold).astype(int), zero_division=0)

    def _spec_t(yt, yp):
        yp_b = (yp >= threshold).astype(int)
        tn = int(((yt == 0) & (yp_b == 0)).sum())
        fp = int(((yt == 0) & (yp_b == 1)).sum())
        return (tn / (tn + fp)) if (tn + fp) > 0 else np.nan

    def _acc_t(yt, yp):
        return accuracy_score(yt, (yp >= threshold).astype(int))

    metric_fns = {
        "roc_auc": _auc,
        "pr_auc": _prauc,
        "brier_score": _brier,
        "f1": _f1_t,
        "precision": _prec_t,
        "recall_sensitivity": _sens_t,
        "specificity": _spec_t,
        "accuracy": _acc_t,
    }

    point = {name: fn(y_true, y_prob) for name, fn in metric_fns.items()}
    boot = {name: [] for name in metric_fns}
    for _ in range(n_boot):
        sampled_pids = rng.choice(unique_pids, size=n_pat, replace=True)
        idxs = np.concatenate([pid_to_idx[p] for p in sampled_pids])
        yt_b = y_true[idxs]
        yp_b = y_prob[idxs]
        for name, fn in metric_fns.items():
            try:
                boot[name].append(fn(yt_b, yp_b))
            except Exception:
                boot[name].append(np.nan)

    lo_q = (1.0 - ci) / 2.0 * 100.0
    hi_q = (1.0 + ci) / 2.0 * 100.0
    rows = []
    for name, vals in boot.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[~np.isnan(arr)]
        lo = float(np.percentile(arr, lo_q)) if len(arr) else np.nan
        hi = float(np.percentile(arr, hi_q)) if len(arr) else np.nan
        rows.append({"metric": name, "point": float(point[name]) if np.isfinite(point[name]) else np.nan, "ci_lo": lo, "ci_hi": hi, "n_boot_valid": int(len(arr))})
    return pd.DataFrame(rows)



def bootstrap_ci_table(artifacts: Dict, samples_by_split: Dict[str, List[Dict]], thresholds: Dict[str, float], split: str = "test", n_boot: int = 1000, random_state: int = 42) -> pd.DataFrame:
    frames = []
    samples = samples_by_split[split]
    for m_name in MODEL_NAMES:
        df = patient_bootstrap_ci(samples, artifacts["predictions"][f"{split}_{m_name}"], threshold=float(thresholds.get(m_name, 0.5)), n_boot=n_boot, random_state=random_state)
        df.insert(0, "split", split)
        df.insert(1, "model", m_name)
        df.insert(2, "threshold", float(thresholds.get(m_name, 0.5)))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)



def catboost_feature_importance_table(artifacts: Dict, top_k=20):
    model = artifacts["models"]["cat"]
    tab_prep = artifacts["preprocessors"]["tabular"]
    imp = model.get_feature_importance()
    names = tab_prep.feature_names_ if tab_prep.feature_names_ is not None else np.array([f"feature_{i}" for i in range(len(imp))])
    return pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False).head(top_k).reset_index(drop=True)



def prediction_frame(samples, y_prob, model_name, threshold=0.5):
    return pd.DataFrame({
        "patient_id": [s["patient_id"] for s in samples],
        "landmark_date": [s["landmark_date"] for s in samples],
        "label": [s["label"] for s in samples],
        f"pred_{model_name}": y_prob,
        f"pred_label_{model_name}": (y_prob >= threshold).astype(int),
        f"threshold_{model_name}": float(threshold),
    })


# =========================================
# 9. MAIN
# =========================================


def main():
    # Verify environment is set up before doing any work
    ensure_dirs()
    assert_data_exists()

    cfg = Config()
    set_seed(cfg.random_state)

    df = load_longitudinal_data(cfg)
    train_ids, val_ids, test_ids = patient_level_split(df, cfg)
    train_visits_df = filter_df_by_patients(df, train_ids, cfg.patient_col)
    val_visits_df = filter_df_by_patients(df, val_ids, cfg.patient_col)
    test_visits_df = filter_df_by_patients(df, test_ids, cfg.patient_col)

    train_samples = build_landmark_samples(train_visits_df, cfg)
    val_samples = build_landmark_samples(val_visits_df, cfg)
    test_samples = build_landmark_samples(test_visits_df, cfg)
    samples_by_split = {"train": train_samples, "val": val_samples, "test": test_samples}

    def _positives(samples):
        return sum(s["label"] for s in samples)

    print(f"Train: {len(train_ids)} patients | {len(train_samples)} samples | {_positives(train_samples)} positives")
    print(f"Val:   {len(val_ids)} patients | {len(val_samples)} samples | {_positives(val_samples)} positives")
    print(f"Test:  {len(test_ids)} patients | {len(test_samples)} samples | {_positives(test_samples)} positives")
    print(f"Label window: [{cfg.min_gap_days}, {cfg.horizon_days}] days")

    if min(len(train_samples), len(val_samples), len(test_samples)) == 0:
        raise RuntimeError("A split has zero landmark samples.")

    artifacts = fit_models(train_visits_df, train_samples, val_samples, test_samples, cfg)

    # uncalibrated reporting with patient-tuned thresholds
    results_default = metrics_table(artifacts)
    tuned, thr_table = tune_thresholds_patient_level(artifacts, samples_by_split, split="val", agg="max")
    results_tuned = metrics_table(artifacts, thresholds=tuned)
    patient_results = patient_level_metrics_table(artifacts, samples_by_split, thresholds=tuned, agg="max")
    boot_test = bootstrap_ci_table(artifacts, samples_by_split, thresholds=tuned, split="test", n_boot=cfg.bootstrap_n, random_state=cfg.random_state)
    reliability_bins, ece_summary = reliability_table_all_models(artifacts, split="test", n_bins=cfg.reliability_bins)

    print("\n=== Sample-level metrics (threshold = 0.5) ===")
    print(results_default.to_string(index=False))
    print("\n=== Patient-level validation-tuned thresholds (objective = F1, agg = max) ===")
    print(thr_table.to_string(index=False))
    print("\n=== Sample-level metrics (patient-tuned thresholds) ===")
    print(results_tuned.to_string(index=False))
    print("\n=== Patient-level metrics (agg = max, patient-tuned thresholds) ===")
    print(patient_results.to_string(index=False))
    print("\n=== Top CatBoost Features ===")
    print(catboost_feature_importance_table(artifacts, top_k=20).to_string(index=False))
    print("\n=== TEST: Expected Calibration Error (uncalibrated) ===")
    print(ece_summary.to_string(index=False))

    # calibrated reporting
    calibrators = fit_isotonic_calibrators(artifacts)
    artifacts_cal = build_calibrated_artifacts(artifacts, calibrators)
    tuned_cal, thr_table_cal = tune_thresholds_patient_level(artifacts_cal, samples_by_split, split="val", agg="max")
    results_tuned_cal = metrics_table(artifacts_cal, thresholds=tuned_cal)
    patient_results_cal = patient_level_metrics_table(artifacts_cal, samples_by_split, thresholds=tuned_cal, agg="max")
    boot_test_cal = bootstrap_ci_table(artifacts_cal, samples_by_split, thresholds=tuned_cal, split="test", n_boot=cfg.bootstrap_n, random_state=cfg.random_state)
    reliability_bins_cal, ece_summary_cal = reliability_table_all_models(artifacts_cal, split="test", n_bins=cfg.reliability_bins)

    ece_comparison = pd.merge(
        ece_summary.rename(columns={"ece": "ece_uncalibrated"})[["model", "ece_uncalibrated"]],
        ece_summary_cal.rename(columns={"ece": "ece_calibrated"})[["model", "ece_calibrated"]],
        on="model",
    )
    ece_comparison["absolute_reduction"] = ece_comparison["ece_uncalibrated"] - ece_comparison["ece_calibrated"]
    denom = ece_comparison["ece_uncalibrated"].replace(0.0, np.nan)
    ece_comparison["relative_reduction_pct"] = 100.0 * ece_comparison["absolute_reduction"] / denom

    auc_uncal = results_tuned[results_tuned["split"] == "test"][["model", "roc_auc", "pr_auc"]].rename(columns={"roc_auc": "roc_auc_uncal", "pr_auc": "pr_auc_uncal"})
    auc_cal = results_tuned_cal[results_tuned_cal["split"] == "test"][["model", "roc_auc", "pr_auc"]].rename(columns={"roc_auc": "roc_auc_cal", "pr_auc": "pr_auc_cal"})
    auc_compare = pd.merge(auc_uncal, auc_cal, on="model")
    auc_compare["roc_auc_delta"] = auc_compare["roc_auc_cal"] - auc_compare["roc_auc_uncal"]
    auc_compare["pr_auc_delta"] = auc_compare["pr_auc_cal"] - auc_compare["pr_auc_uncal"]

    print("\n=== (Calibrated) Patient-level validation-tuned thresholds ===")
    print(thr_table_cal.to_string(index=False))
    print("\n=== (Calibrated) Sample-level metrics ===")
    print(results_tuned_cal.to_string(index=False))
    print("\n=== (Calibrated) Patient-level metrics ===")
    print(patient_results_cal.to_string(index=False))
    print("\n=== TEST: Expected Calibration Error (calibrated) ===")
    print(ece_summary_cal.to_string(index=False))
    print("\n=== TEST: ECE uncalibrated vs calibrated ===")
    print(ece_comparison.to_string(index=False))
    print("\n=== TEST: ranking metrics uncalibrated vs calibrated ===")
    print(auc_compare.to_string(index=False))

    # save (all paths go to centralised TASK5_OUT directory)
    results_default.to_csv(TASK5_OUT / "task5_metrics_default.csv", index=False)
    thr_table.to_csv(TASK5_OUT / "task5_thresholds.csv", index=False)
    results_tuned.to_csv(TASK5_OUT / "task5_metrics_tuned.csv", index=False)
    patient_results.to_csv(TASK5_OUT / "task5_metrics_patient_level.csv", index=False)
    catboost_feature_importance_table(artifacts, top_k=50).to_csv(TASK5_OUT / "task5_catboost_feature_importance.csv", index=False)
    boot_test.to_csv(TASK5_OUT / "task5_test_bootstrap_ci.csv", index=False)
    reliability_bins.to_csv(TASK5_OUT / "task5_test_reliability_bins.csv", index=False)
    ece_summary.to_csv(TASK5_OUT / "task5_test_ece_summary.csv", index=False)

    results_tuned_cal.to_csv(TASK5_OUT / "task5_metrics_tuned_calibrated.csv", index=False)
    patient_results_cal.to_csv(TASK5_OUT / "task5_metrics_patient_level_calibrated.csv", index=False)
    thr_table_cal.to_csv(TASK5_OUT / "task5_thresholds_calibrated.csv", index=False)
    boot_test_cal.to_csv(TASK5_OUT / "task5_test_bootstrap_ci_calibrated.csv", index=False)
    reliability_bins_cal.to_csv(TASK5_OUT / "task5_test_reliability_bins_calibrated.csv", index=False)
    ece_summary_cal.to_csv(TASK5_OUT / "task5_test_ece_summary_calibrated.csv", index=False)
    ece_comparison.to_csv(TASK5_OUT / "task5_ece_comparison.csv", index=False)
    auc_compare.to_csv(TASK5_OUT / "task5_auc_uncal_vs_cal.csv", index=False)

    for split, split_samples in samples_by_split.items():
        for m_name in MODEL_NAMES:
            prediction_frame(split_samples, artifacts["predictions"][f"{split}_{m_name}"], m_name, threshold=tuned[m_name]).to_csv(TASK5_OUT / f"task5_{split}_predictions_{m_name}.csv", index=False)
            prediction_frame(split_samples, artifacts_cal["predictions"][f"{split}_{m_name}"], m_name, threshold=tuned_cal[m_name]).to_csv(TASK5_OUT / f"task5_{split}_predictions_{m_name}_calibrated.csv", index=False)

    print(f"\nSaved cleaned pipeline outputs to {TASK5_OUT}")


if __name__ == "__main__":
    main()
