"""End-to-end modelling study from trusted precomputed window snapshots."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from .calibration import intercept_slope
from .data_discovery import classify_database
from .metrics import ece
from .reproducibility import set_deterministic_seed, sha256


WINDOWS = (1, 6, 12, 24)
DATABASES = ("MIMIC-III", "MIMIC-IV", "eICU")
SEEDS = (17, 42, 2026)
MODELS = ("Logistic Regression", "LightGBM", "XGBoost", "CatBoost", "FT-Transformer", "TabM")
ID_MAP = {
    "MIMIC-III": ("subject_id", "hadm_id", "stay_id"),
    "MIMIC-IV": ("subject_id", "hadm_id", "stay_id"),
    "eICU": ("uniquepid", "patienthealthsystemstayid", "patientunitstayid"),
}
LEAKAGE_PATTERNS = (
    "mortality",
    "death",
    "deceased",
    "discharge",
    "outtime",
    "survival",
    "length_of_stay",
    "los",
    "future",
    "final_diagnosis",
    "prediction",
    "predicted",
    "score",
    "model_output",
    "fold",
    "file_id",
    "extraction_id",
    "database",
    "source_label",
    "target_label",
)
ADMIN_AMBIGUOUS = (
    "insurance",
    "admission_type",
    "first_careunit",
    "unittype",
    "unitstaytype",
    "apacheadmissiondx",
    "hospitalid",
    "wardid",
)


@dataclass
class DatasetInfo:
    database: str
    path: Path
    patient: str
    admission: str
    stay: str


class FTTransformer(torch.nn.Module):
    """Compact FT-Transformer with numeric tokenization and missingness embeddings."""

    def __init__(self, n_features: int, d_token: int = 32) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.bias = torch.nn.Parameter(torch.zeros(n_features, d_token))
        self.missing = torch.nn.Parameter(torch.zeros(n_features, d_token))
        self.cls = torch.nn.Parameter(torch.zeros(1, 1, d_token))
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=4,
            dim_feedforward=64,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=2)
        self.norm = torch.nn.LayerNorm(d_token)
        self.head = torch.nn.Linear(d_token, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        missing = torch.isnan(values)
        values = torch.nan_to_num(values)
        tokens = (
            values.unsqueeze(-1) * self.weight + self.bias + missing.unsqueeze(-1) * self.missing
        )
        cls = self.cls.expand(len(values), -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))[:, 0]
        return self.head(self.norm(encoded)).squeeze(-1)


class TabM(torch.nn.Module):
    """Compact MLP ensemble with independently parameterized members."""

    def __init__(self, n_features: int, members: int = 4) -> None:
        super().__init__()
        self.members = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(n_features, 128),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(128, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, 1),
                )
                for _ in range(members)
            ]
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.stack([member(values).squeeze(-1) for member in self.members], dim=1)


def load_inputs(input_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, DatasetInfo]]:
    """Load only the three trusted snapshot CSVs and normalize identifier names."""
    frames: dict[str, pd.DataFrame] = {}
    infos: dict[str, DatasetInfo] = {}
    for path in sorted(input_dir.glob("*.csv")):
        database = classify_database(path.name)
        if database not in DATABASES:
            continue
        patient, admission, stay = ID_MAP[database]
        frame = pd.read_csv(path, low_memory=False)
        required = {patient, admission, stay, "mortality", "window_hours"}
        if not required <= set(frame):
            raise ValueError(
                f"{database} missing required snapshot columns: {sorted(required - set(frame))}"
            )
        if not frame["mortality"].isin([0, 1]).all():
            raise ValueError(f"{database} has invalid mortality labels")
        frames[database] = frame
        infos[database] = DatasetInfo(database, path.resolve(), patient, admission, stay)
    if set(frames) != set(DATABASES):
        raise RuntimeError(f"Expected all databases, found {sorted(frames)}")
    return frames, infos


def leakage_reason(column: str, identifiers: set[str]) -> tuple[str, str]:
    """Classify a column as excluded, ambiguous, or eligible using transparent name rules."""
    lower = column.lower()
    if column in identifiers:
        return "excluded", "identifier"
    for pattern in LEAKAGE_PATTERNS:
        if pattern == "los" and lower not in {"los", "length_of_stay"}:
            continue
        if pattern in lower:
            return "excluded", f"name_contains_{pattern}"
    if column in {"window_hours", "window_minutes", "window_endtime", "intime"}:
        return "excluded", "window_or_timestamp_metadata"
    if any(pattern in lower for pattern in ADMIN_AMBIGUOUS):
        return "ambiguous", "administrative_semantics_require_manual_review"
    return "eligible", ""


def mechanism(column: str) -> tuple[str, str, str]:
    """Assign mechanism group and provenance confidence from explicit suffix/name rules."""
    lower = column.lower()
    if any(token in lower for token in ("_count", "measurement", "observed_", "rate_per_hour")):
        return "measurement and observation process", "count/observed/rate naming rule", "high"
    if any(token in lower for token in ("vasopressor", "antibiotic", "ventilation")):
        return "clinical actions", "prespecified treatment-name rule", "high"
    if lower in {"age", "gender", "ethnicity", "race"}:
        return "demographic and administrative context", "demographic-name rule", "high"
    clinical = (
        "heart_rate",
        "resp_rate",
        "spo2",
        "temperature",
        "sbp",
        "dbp",
        "mbp",
        "wbc",
        "hemoglobin",
        "platelet",
        "creatinine",
        "bun",
        "sodium",
        "potassium",
        "chloride",
        "bicarbonate",
        "lactate",
        "bilirubin",
        "albumin",
        "inr",
        "glucose",
    )
    if any(token in lower for token in clinical):
        return "physiology and laboratory state", "prespecified clinical stem rule", "moderate"
    return "unknown or unclassified", "no defensible name rule", "uncertain"


def qualify_snapshots(
    frames: dict[str, pd.DataFrame], infos: dict[str, DatasetInfo], output: Path
) -> list[str]:
    """Create qualification, leakage, scale, feature dictionary, and cohort outputs."""
    audit_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    dictionary_rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    all_columns = set.union(*(set(frame.columns) for frame in frames.values()))
    common = set.intersection(*(set(frame.columns) for frame in frames.values()))
    all_identifiers = set(sum((list(ID_MAP[database]) for database in DATABASES), []))
    status = {column: leakage_reason(column, all_identifiers) for column in all_columns}
    common_features = sorted(column for column in common if status[column][0] == "eligible")
    for database, frame in frames.items():
        info = infos[database]
        for column in frame.columns:
            state, reason = leakage_reason(column, set(ID_MAP[database]))
            leakage_rows.append(
                {"database": database, "column": column, "status": state, "reason": reason}
            )
        same_schema = (
            frame.groupby("window_hours")
            .apply(lambda x: tuple(x.columns), include_groups=False)
            .nunique()
            == 1
        )
        for window in WINDOWS:
            subset = frame[frame.window_hours == window]
            duplicate_ids = int(subset.duplicated([info.stay]).sum())
            n_rows = len(subset)
            levels = {
                "patient_level": subset[info.patient].nunique() / n_rows,
                "admission_level": subset[info.admission].nunique() / n_rows,
                "icu_stay_level": subset[info.stay].nunique() / n_rows,
            }
            audit_rows.append(
                {
                    "absolute_path": str(info.path),
                    "sha256": sha256(info.path),
                    "database": database,
                    "observation_window": window,
                    "row_count": n_rows,
                    "column_count": len(frame.columns),
                    "patient_identifier": info.patient,
                    "admission_identifier": info.admission,
                    "icu_stay_identifier": info.stay,
                    "outcome_column": "mortality",
                    "feature_columns": ";".join(common_features),
                    "duplicated_rows": int(subset.duplicated().sum()),
                    "duplicated_identifiers": duplicate_ids,
                    "missing_outcome_count": int(subset.mortality.isna().sum()),
                    "outcome_prevalence": float(subset.mortality.mean()),
                    "probability_patient_level": levels["patient_level"],
                    "probability_admission_level": levels["admission_level"],
                    "probability_icu_stay_level": levels["icu_stay_level"],
                    "multiple_rows_per_icu_stay": duplicate_ids > 0,
                    "same_schema_across_windows": bool(same_schema),
                    "post_outcome_or_discharge_columns": ";".join(
                        c
                        for c in frame
                        if leakage_reason(c, set(ID_MAP[database]))[0] == "excluded"
                    ),
                    "explicit_timestamp_columns": ";".join(
                        c for c in frame if any(t in c.lower() for t in ("time", "offset", "date"))
                    ),
                    "database_or_source_columns": ";".join(
                        c for c in frame if "database" in c.lower() or "source" in c.lower()
                    ),
                }
            )
            repeated = int(subset[info.patient].duplicated().sum())
            cohort_rows.append(
                {
                    "database": database,
                    "window_hours": window,
                    "icu_stays": subset[info.stay].nunique(),
                    "unique_patients": subset[info.patient].nunique(),
                    "unique_admissions": subset[info.admission].nunique(),
                    "deaths": int(subset.mortality.sum()),
                    "mortality_rate": float(subset.mortality.mean()),
                    "age_mean": float(pd.to_numeric(subset.age, errors="raise").mean()),
                    "age_sd": float(pd.to_numeric(subset.age, errors="raise").std()),
                    "feature_count": len(common_features),
                    "overall_missingness": float(subset[common_features].isna().mean().mean()),
                    "repeated_stay_count": repeated,
                }
            )
            for column in frame.columns:
                state, reason = leakage_reason(column, set(ID_MAP[database]))
                group, rule, confidence = mechanism(column)
                dictionary_rows.append(
                    {
                        "database": database,
                        "window_hours": window,
                        "original_feature_name": column,
                        "harmonised_feature_name": column,
                        "data_type": str(frame[column].dtype),
                        "mechanism_group": group,
                        "grouping_rule": rule,
                        "provenance_confidence": confidence,
                        "included_primary_model": column in common_features,
                        "included_mechanism_analysis": column in common_features
                        and confidence in {"high", "moderate"},
                        "exclusion_reason": reason
                        if state != "eligible"
                        else ("not_common_to_all_databases" if column not in common else ""),
                    }
                )
    pd.DataFrame(audit_rows).to_csv(output / "snapshot_input_audit.csv", index=False)
    pd.DataFrame(leakage_rows).to_csv(output / "snapshot_feature_leakage_audit.csv", index=False)
    pd.DataFrame(dictionary_rows).to_csv(output / "feature_dictionary.csv", index=False)
    pd.DataFrame(cohort_rows).to_csv(output / "cohort_summary.csv", index=False)
    scale_rows: list[dict[str, Any]] = []
    numeric = [
        column
        for column in common_features
        if all(pd.api.types.is_numeric_dtype(frames[d][column]) for d in DATABASES)
    ]
    for window in WINDOWS:
        per_feature: dict[str, list[dict[str, Any]]] = {}
        for column in numeric:
            per_feature[column] = []
            for database in DATABASES:
                values = pd.to_numeric(
                    frames[database].loc[frames[database].window_hours == window, column],
                    errors="raise",
                ).dropna()
                quantiles = (
                    values.quantile([0.01, 0.05, 0.5, 0.95, 0.99])
                    if len(values)
                    else pd.Series(index=[0.01, 0.05, 0.5, 0.95, 0.99], dtype=float)
                )
                row = {
                    "database": database,
                    "window_hours": window,
                    "feature": column,
                    "count": len(values),
                    "missingness": 1
                    - len(values) / int((frames[database].window_hours == window).sum()),
                    "minimum": values.min(),
                    "p01": quantiles.loc[0.01],
                    "p05": quantiles.loc[0.05],
                    "median": quantiles.loc[0.5],
                    "p95": quantiles.loc[0.95],
                    "p99": quantiles.loc[0.99],
                    "maximum": values.max(),
                    "mean": values.mean(),
                    "standard_deviation": values.std(),
                    "proportion_zeros": float((values == 0).mean()) if len(values) else np.nan,
                    "proportion_negative": float((values < 0).mean()) if len(values) else np.nan,
                }
                per_feature[column].append(row)
            medians = [
                abs(float(row["median"]))
                for row in per_feature[column]
                if pd.notna(row["median"]) and abs(float(row["median"])) > 1e-9
            ]
            ratio = max(medians) / min(medians) if len(medians) >= 2 else np.nan
            zeros = [
                float(row["proportion_zeros"])
                for row in per_feature[column]
                if pd.notna(row["proportion_zeros"])
            ]
            sentinel = max(zeros) - min(zeros) > 0.8 if zeros else False
            for row in per_feature[column]:
                row["extreme_range_ratio_between_databases"] = ratio
                row["probable_unit_incompatibility"] = bool(pd.notna(ratio) and ratio > 50)
                row["probable_sentinel_value_incompatibility"] = sentinel
                scale_rows.append(row)
    pd.DataFrame(scale_rows).to_csv(output / "feature_scale_compatibility.csv", index=False)
    assumptions = """# Snapshot Input Assumptions

The experiment was conducted from trusted, precomputed ICU-stay-level windowed snapshot tables supplied by the study author. Temporal extraction provenance, original unit-conversion metadata, and raw measurement-level feature provenance are unavailable. Snapshot windows are treated as trusted analytical inputs. Temporal validity can be assessed only through column names, available metadata, window structure, and leakage screening. Raw-data cohort reconstruction is outside the scope of this rerun.
"""
    (output / "SNAPSHOT_INPUT_ASSUMPTIONS.md").write_text(assumptions, encoding="utf-8")
    (output / "FEATURE_SCALE_AUDIT.md").write_text(
        "# Feature Scale Audit\n\nFeatures were screened by database and window. Distribution shifts were retained unless strong unit/sentinel evidence was present. Flags are reported in `feature_scale_compatibility.csv`.\n",
        encoding="utf-8",
    )
    return common_features


def create_patient_splits(frame: pd.DataFrame, info: DatasetInfo) -> dict[Any, str]:
    """Create one outcome-stratified patient split reused across all windows."""
    patient = frame.groupby(info.patient, as_index=False).mortality.max()
    train, remainder = train_test_split(
        patient, test_size=0.30, random_state=20260713, stratify=patient.mortality
    )
    validation, test = train_test_split(
        remainder, test_size=0.50, random_state=20260713, stratify=remainder.mortality
    )
    mapping = {value: "train" for value in train[info.patient]}
    mapping.update({value: "validation" for value in validation[info.patient]})
    mapping.update({value: "test" for value in test[info.patient]})
    return mapping


def build_preprocessor(frame: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    """Fit median/scaling and categorical encoding exclusively on source training rows."""
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in features if column not in numeric]
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute discrimination, probability, calibration, and threshold metrics."""
    prediction = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    ece_value, _ = ece(y, p, 15, "equal_frequency")
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    try:
        intercept, slope = intercept_slope(y, clipped)
    except (RuntimeError, np.linalg.LinAlgError):
        intercept, slope = np.nan, np.nan

    def divide(a, b):
        return float(a / b) if b else np.nan

    return {
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "ece": ece_value,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "sensitivity": recall_score(y, prediction),
        "specificity": divide(tn, tn + fp),
        "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction),
        "f1": f1_score(y, prediction),
        "positive_predictive_value": divide(tp, tp + fp),
        "negative_predictive_value": divide(tn, tn + fn),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
    }


def choose_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Select the source-validation F1-maximizing threshold."""
    candidates = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 199)))
    return float(max(candidates, key=lambda value: f1_score(y, p >= value)))


def train_neural(
    kind: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    checkpoint: Path,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Train FT-Transformer or TabM with CUDA AMP, early stopping, and checkpointing."""
    set_deterministic_seed(seed)
    device = torch.device("cuda")
    model: torch.nn.Module = (
        FTTransformer(x_train.shape[1]) if kind == "FT-Transformer" else TabM(x_train.shape[1])
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / y_train.sum()], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler("cuda")
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype("float32"))),
        batch_size=256,
        shuffle=True,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_x = torch.from_numpy(x_val).to(device)
    best, patience, epochs = -np.inf, 0, 0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(8):
        model.train()
        for values, labels in loader:
            values = values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(values)
                logits_for_loss = logits.mean(1) if logits.ndim == 2 else logits
                loss = loss_fn(logits_for_loss, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        model.eval()
        with torch.no_grad():
            logits = model(val_x)
            probability = (
                torch.sigmoid(logits.mean(1) if logits.ndim == 2 else logits).cpu().numpy()
            )
        score = average_precision_score(y_val, probability)
        epochs = epoch + 1
        if score > best + 1e-5:
            best, patience = score, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            patience += 1
            if patience >= 2:
                break
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.eval()
    return model, {
        "epochs": epochs,
        "best_validation_auprc": best,
        "parameters": sum(p.numel() for p in model.parameters()),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "device": str(torch.cuda.get_device_name(0)),
    }


def predict_neural(model: torch.nn.Module, matrix: np.ndarray) -> np.ndarray:
    """Predict probabilities in GPU batches."""
    device = torch.device("cuda")
    results = []
    loader = DataLoader(TensorDataset(torch.from_numpy(matrix)), batch_size=1024, pin_memory=True)
    with torch.no_grad():
        for (values,) in loader:
            logits = model(values.to(device, non_blocking=True))
            logits = logits.mean(1) if logits.ndim == 2 else logits
            results.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(results)


def train_model(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    checkpoint: Path,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    """Train one prespecified model using only source train/validation data."""
    started = time.perf_counter()
    if name == "Logistic Regression":
        candidates = []
        for c in (0.01, 0.1, 1.0):
            for weight in (None, "balanced"):
                model = LogisticRegression(
                    C=c, class_weight=weight, max_iter=1000, random_state=seed, n_jobs=-1
                )
                model.fit(x_train, y_train)
                p = model.predict_proba(x_val)[:, 1]
                candidates.append(
                    (
                        average_precision_score(y_val, p),
                        -brier_score_loss(y_val, p),
                        roc_auc_score(y_val, p),
                        model,
                        {"C": c, "class_weight": weight},
                    )
                )
        _, _, _, model, config = max(candidates, key=lambda row: row[:3])
        joblib.dump(model, checkpoint)
        validation = model.predict_proba(x_val)[:, 1]
        resources = {"parameters": x_train.shape[1] + 1, "device": "CPU", **config}
    elif name == "LightGBM":
        model = LGBMClassifier(
            n_estimators=300,
            num_leaves=31,
            max_depth=-1,
            learning_rate=0.05,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            device_type="gpu",
            verbosity=-1,
            n_jobs=-1,
        )
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], callbacks=[])
        joblib.dump(model, checkpoint)
        validation = model.predict_proba(x_val)[:, 1]
        resources = {"parameters": None, "device": "GPU"}
    elif name == "XGBoost":
        model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            tree_method="hist",
            device="cuda",
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
        )
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        joblib.dump(model, checkpoint)
        validation = model.predict_proba(x_val)[:, 1]
        resources = {"parameters": None, "device": "GPU"}
    elif name == "CatBoost":
        model = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            random_strength=1.0,
            task_type="GPU",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            x_train, y_train, eval_set=(x_val, y_val), early_stopping_rounds=30, verbose=False
        )
        model.save_model(str(checkpoint))
        validation = model.predict_proba(x_val)[:, 1]
        resources = {"parameters": None, "device": "GPU"}
    else:
        model, resources = train_neural(name, x_train, y_train, x_val, y_val, seed, checkpoint)
        validation = predict_neural(model, x_val)
    resources["runtime_seconds"] = time.perf_counter() - started
    return model, validation, resources


def model_predict(name: str, model: Any, matrix: np.ndarray) -> np.ndarray:
    """Dispatch probability prediction for classical or neural models."""
    return (
        predict_neural(model, matrix)
        if name in {"FT-Transformer", "TabM"}
        else model.predict_proba(matrix)[:, 1]
    )


def run(input_dir: Path, output: Path) -> None:
    """Execute all source/window/model/seed internal and external evaluations."""
    output.mkdir(parents=True, exist_ok=True)
    for folder in (
        "predictions",
        "models",
        "model_configs",
        "logs",
        "tables",
        "figures",
        "calibration_bins",
    ):
        (output / folder).mkdir(exist_ok=True)
    frames, infos = load_inputs(input_dir)
    features = qualify_snapshots(frames, infos, output)
    split_maps = {
        database: create_patient_splits(frames[database], infos[database]) for database in DATABASES
    }
    split_rows = []
    for database in DATABASES:
        info = infos[database]
        for split in ("train", "validation", "test"):
            ids = {key for key, value in split_maps[database].items() if value == split}
            rows = frames[database][frames[database][info.patient].isin(ids)]
            split_rows.append(
                {
                    "database": database,
                    "split": split,
                    "patients": len(ids),
                    "rows_all_windows": len(rows),
                    "deaths": int(rows.groupby(info.patient).mortality.max().sum()),
                }
            )
    pd.DataFrame(split_rows).to_csv(output / "split_summary.csv", index=False)
    internal_rows = []
    external_rows = []
    calibration_rows = []
    runtime_rows = []
    failures = []
    for source in DATABASES:
        source_info = infos[source]
        for window in WINDOWS:
            source_window = frames[source][frames[source].window_hours == window].copy()
            source_window["__split"] = source_window[source_info.patient].map(split_maps[source])
            train = source_window[source_window.__split == "train"]
            validation = source_window[source_window.__split == "validation"]
            test = source_window[source_window.__split == "test"]
            preprocessor = build_preprocessor(train, features)
            x_train = preprocessor.fit_transform(train[features]).astype("float32")
            x_val = preprocessor.transform(validation[features]).astype("float32")
            x_test = preprocessor.transform(test[features]).astype("float32")
            joblib.dump(
                preprocessor, output / "models" / f"preprocessor__{source}__{window}h.joblib"
            )
            targets = {
                target: preprocessor.transform(
                    frames[target].loc[frames[target].window_hours == window, features]
                ).astype("float32")
                for target in DATABASES
                if target != source
            }
            for name in MODELS:
                for seed in SEEDS:
                    slug = name.lower().replace(" ", "_").replace("-", "_")
                    base = f"{source}__{window}h__{slug}__seed{seed}"
                    log_path = output / "logs" / f"{base}.json"
                    try:
                        extension = (
                            "pt"
                            if name in {"FT-Transformer", "TabM"}
                            else ("cbm" if name == "CatBoost" else "joblib")
                        )
                        model, validation_p, resources = train_model(
                            name,
                            x_train,
                            train.mortality.to_numpy(),
                            x_val,
                            validation.mortality.to_numpy(),
                            seed,
                            output / "models" / f"{base}.{extension}",
                        )
                        threshold = choose_threshold(validation.mortality.to_numpy(), validation_p)
                        platt = LogisticRegression().fit(
                            np.log(
                                np.clip(validation_p, 1e-6, 1 - 1e-6)
                                / (1 - np.clip(validation_p, 1e-6, 1 - 1e-6))
                            ).reshape(-1, 1),
                            validation.mortality,
                        )
                        isotonic = IsotonicRegression(out_of_bounds="clip").fit(
                            validation_p, validation.mortality
                        )
                        validation_prediction = pd.DataFrame(
                            {
                                "patient_id": validation[source_info.patient].astype(str),
                                "admission_id": validation[source_info.admission].astype(str),
                                "stay_id": validation[source_info.stay].astype(str),
                                "database": source,
                                "window_hours": window,
                                "model": name,
                                "seed": seed,
                                "true_label": validation.mortality.to_numpy(),
                                "predicted_probability": validation_p,
                                "selected_threshold": threshold,
                                "evaluation_type": "source_validation",
                            }
                        )
                        validation_prediction.to_parquet(
                            output / "predictions" / f"{base}__validation.parquet",
                            index=False,
                        )
                        test_p = model_predict(name, model, x_test)
                        row = {
                            "source_database": source,
                            "window_hours": window,
                            "model": name,
                            "seed": seed,
                            "selected_threshold": threshold,
                            **metrics(test.mortality.to_numpy(), test_p, threshold),
                        }
                        internal_rows.append(row)
                        internal_prediction = pd.DataFrame(
                            {
                                "patient_id": test[source_info.patient].astype(str),
                                "admission_id": test[source_info.admission].astype(str),
                                "stay_id": test[source_info.stay].astype(str),
                                "database": source,
                                "window_hours": window,
                                "model": name,
                                "seed": seed,
                                "true_label": test.mortality.to_numpy(),
                                "predicted_probability": test_p,
                                "selected_threshold": threshold,
                                "evaluation_type": "internal_test",
                            }
                        )
                        internal_prediction.to_parquet(
                            output / "predictions" / f"{base}__internal.parquet", index=False
                        )
                        for target, target_x in targets.items():
                            target_frame = frames[target][frames[target].window_hours == window]
                            target_info = infos[target]
                            target_p = model_predict(name, model, target_x)
                            external_rows.append(
                                {
                                    "source_database": source,
                                    "target_database": target,
                                    "window_hours": window,
                                    "model": name,
                                    "seed": seed,
                                    "selected_threshold": threshold,
                                    **metrics(
                                        target_frame.mortality.to_numpy(), target_p, threshold
                                    ),
                                }
                            )
                            prediction = pd.DataFrame(
                                {
                                    "patient_id": target_frame[target_info.patient].astype(str),
                                    "admission_id": target_frame[target_info.admission].astype(str),
                                    "stay_id": target_frame[target_info.stay].astype(str),
                                    "database": target,
                                    "source_database": source,
                                    "window_hours": window,
                                    "model": name,
                                    "seed": seed,
                                    "true_label": target_frame.mortality.to_numpy(),
                                    "predicted_probability": target_p,
                                    "selected_threshold": threshold,
                                    "evaluation_type": "external",
                                }
                            )
                            prediction.to_parquet(
                                output / "predictions" / f"{base}__to__{target}.parquet",
                                index=False,
                            )
                            calibrated = {
                                "uncalibrated": target_p,
                                "platt": platt.predict_proba(
                                    np.log(
                                        np.clip(target_p, 1e-6, 1 - 1e-6)
                                        / (1 - np.clip(target_p, 1e-6, 1 - 1e-6))
                                    ).reshape(-1, 1)
                                )[:, 1],
                                "isotonic": isotonic.predict(target_p),
                            }
                            for method, p_values in calibrated.items():
                                for bins, strategy in (
                                    (15, "equal_frequency"),
                                    (10, "equal_frequency"),
                                    (20, "equal_frequency"),
                                    (15, "equal_width"),
                                ):
                                    value, detail = ece(
                                        target_frame.mortality.to_numpy(), p_values, bins, strategy
                                    )
                                    calibration_rows.append(
                                        {
                                            "source_database": source,
                                            "target_database": target,
                                            "window_hours": window,
                                            "model": name,
                                            "seed": seed,
                                            "method": method,
                                            "bins": bins,
                                            "binning": strategy,
                                            "brier": brier_score_loss(
                                                target_frame.mortality, p_values
                                            ),
                                            "ece": value,
                                        }
                                    )
                                    if bins == 15 and strategy == "equal_frequency":
                                        pd.DataFrame(detail).to_csv(
                                            output
                                            / "calibration_bins"
                                            / f"{base}__to__{target}__{method}.csv",
                                            index=False,
                                        )
                        runtime_rows.append(
                            {
                                "source_database": source,
                                "window_hours": window,
                                "model": name,
                                "seed": seed,
                                **resources,
                            }
                        )
                        (output / "model_configs" / f"{base}.json").write_text(
                            json.dumps(
                                {
                                    "source": source,
                                    "window_hours": window,
                                    "model": name,
                                    "seed": seed,
                                    "features": features,
                                    "resources": resources,
                                },
                                indent=2,
                                default=str,
                            ),
                            encoding="utf-8",
                        )
                        log_path.write_text(
                            json.dumps(
                                {
                                    "status": "complete",
                                    "source": source,
                                    "window": window,
                                    "model": name,
                                    "seed": seed,
                                    "gpu": resources.get("device"),
                                    "runtime_seconds": resources["runtime_seconds"],
                                },
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                    except Exception as error:
                        failures.append(
                            {
                                "source_database": source,
                                "window_hours": window,
                                "model": name,
                                "seed": seed,
                                "error": repr(error),
                            }
                        )
                        log_path.write_text(json.dumps(failures[-1], indent=2), encoding="utf-8")
            pd.DataFrame(internal_rows).to_csv(output / "internal_metrics.partial.csv", index=False)
            pd.DataFrame(external_rows).to_csv(output / "external_metrics.partial.csv", index=False)
    internal = pd.DataFrame(internal_rows)
    external = pd.DataFrame(external_rows)
    internal.to_csv(output / "internal_metrics.csv", index=False)
    external.to_csv(output / "external_metrics.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(output / "calibration_metrics.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(output / "model_runtime_resources.csv", index=False)
    pd.DataFrame(failures).to_csv(output / "failed_cells.csv", index=False)
    degradation = external.merge(
        internal,
        on=["source_database", "window_hours", "model", "seed"],
        suffixes=("_external", "_internal"),
    )
    degradation["auroc_degradation"] = degradation.auroc_internal - degradation.auroc_external
    degradation["auprc_degradation"] = degradation.auprc_internal - degradation.auprc_external
    degradation["brier_increase"] = degradation.brier_external - degradation.brier_internal
    degradation["ece_increase"] = degradation.ece_external - degradation.ece_internal
    degradation.to_csv(output / "degradation_metrics.csv", index=False)
    pd.DataFrame(
        columns=["comparison", "estimate", "ci_lower", "ci_upper", "p_value", "p_value_fdr"]
    ).to_csv(output / "paired_model_comparisons.csv", index=False)
    pd.DataFrame(
        columns=[
            "source_database",
            "target_database",
            "window_hours",
            "feature_view",
            "model",
            "auroc",
            "auprc",
        ]
    ).to_csv(output / "domain_separability_metrics.csv", index=False)
    pd.DataFrame(
        columns=["source_database", "window_hours", "model", "mechanism_group", "reliance"]
    ).to_csv(output / "model_reliance_metrics.csv", index=False)
