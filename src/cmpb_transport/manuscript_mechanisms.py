"""Main-manuscript database separability and predictive reliance analyses."""

from __future__ import annotations

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .full_snapshot_experiment import (
    DATABASES,
    MODELS,
    SEEDS,
    WINDOWS,
    FTTransformer,
    TabM,
    create_patient_splits,
    load_inputs,
    model_predict,
)


def domain_preprocessor(frame: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in features if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers)


def run_domain_separability(input_dir: Path, output: Path) -> pd.DataFrame:
    """Run 3 pairs x 4 windows x 5 views x 2 classifiers x 3 seeds."""
    frames, infos = load_inputs(input_dir)
    dictionary = pd.read_csv(output / "feature_dictionary.csv")
    eligible = dictionary[
        dictionary.included_primary_model
        & dictionary.provenance_confidence.isin(["high", "moderate"])
    ]
    views = {
        group: sorted(set(part.harmonised_feature_name))
        for group, part in eligible.groupby("mechanism_group")
    }
    views["valid_full"] = sorted(
        set(dictionary.loc[dictionary.included_primary_model, "harmonised_feature_name"])
    )
    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(DATABASES):
        for right in DATABASES[left_index + 1 :]:
            for window in WINDOWS:
                left_frame = frames[left][frames[left].window_hours.eq(window)]
                right_frame = frames[right][frames[right].window_hours.eq(window)]
                n = min(len(left_frame), len(right_frame))
                for view, features in views.items():
                    for seed in SEEDS:
                        a = left_frame.sample(n=n, random_state=seed).copy()
                        b = right_frame.sample(n=n, random_state=seed).copy()
                        a["__domain"], b["__domain"] = 0, 1
                        a["__group"] = left + ":" + a[infos[left].patient].astype(str)
                        b["__group"] = right + ":" + b[infos[right].patient].astype(str)
                        combined = pd.concat([a, b], ignore_index=True)
                        groups = combined["__group"].drop_duplicates()
                        group_domains = combined.groupby("__group").__domain.max().reindex(groups)
                        train_groups, test_groups = train_test_split(
                            groups,
                            test_size=0.30,
                            random_state=seed,
                            stratify=group_domains,
                        )
                        train = combined[combined.__group.isin(set(train_groups))]
                        test = combined[combined.__group.isin(set(test_groups))]
                        preprocessor = domain_preprocessor(train, features)
                        x_train = preprocessor.fit_transform(train[features]).astype("float32")
                        x_test = preprocessor.transform(test[features]).astype("float32")
                        for model_name in ("Logistic Regression", "LightGBM"):
                            model = (
                                LogisticRegression(
                                    max_iter=1000,
                                    random_state=seed,
                                    class_weight="balanced",
                                    n_jobs=-1,
                                )
                                if model_name == "Logistic Regression"
                                else LGBMClassifier(
                                    n_estimators=150,
                                    learning_rate=0.05,
                                    num_leaves=31,
                                    random_state=seed,
                                    device_type="gpu",
                                    verbosity=-1,
                                )
                            )
                            model.fit(x_train, train.__domain)
                            probability = model.predict_proba(x_test)[:, 1]
                            rows.append(
                                {
                                    "database_a": left,
                                    "database_b": right,
                                    "window_hours": window,
                                    "feature_view": view,
                                    "model": model_name,
                                    "seed": seed,
                                    "n_train": len(train),
                                    "n_test": len(test),
                                    "class_prevalence_test": float(test.__domain.mean()),
                                    "auroc": float(roc_auc_score(test.__domain, probability)),
                                    "auprc": float(
                                        average_precision_score(test.__domain, probability)
                                    ),
                                    "device": "GPU" if model_name == "LightGBM" else "CPU",
                                }
                            )
    result = pd.DataFrame(rows)
    if len(result) != 360:
        raise RuntimeError(f"Expected 360 domain-separability cells, produced {len(result)}")
    result.to_csv(output / "domain_separability_metrics.csv", index=False)
    return result


def load_model(
    output: Path, source: str, window: int, name: str, seed: int, n_features: int
) -> Any:
    slug = name.lower().replace(" ", "_").replace("-", "_")
    base = f"{source}__{window}h__{slug}__seed{seed}"
    if name == "CatBoost":
        model = CatBoostClassifier()
        model.load_model(str(output / "models" / f"{base}.cbm"))
        return model
    if name == "FT-Transformer":
        model = FTTransformer(n_features).cuda()
        model.load_state_dict(torch.load(output / "models" / f"{base}.pt", weights_only=True))
        model.eval()
        return model
    if name == "TabM":
        model = TabM(n_features).cuda()
        model.load_state_dict(torch.load(output / "models" / f"{base}.pt", weights_only=True))
        model.eval()
        return model
    return joblib.load(output / "models" / f"{base}.joblib")


def run_model_reliance(input_dir: Path, output: Path) -> pd.DataFrame:
    """Compute four source-test group permutations for all 216 fitted mortality models."""
    frames, infos = load_inputs(input_dir)
    dictionary = pd.read_csv(output / "feature_dictionary.csv")
    eligible = dictionary[
        dictionary.included_primary_model
        & dictionary.provenance_confidence.isin(["high", "moderate"])
    ]
    groups = {
        group: sorted(set(part.harmonised_feature_name))
        for group, part in eligible.groupby("mechanism_group")
    }
    rows = []
    for source in DATABASES:
        split_map = create_patient_splits(frames[source], infos[source])
        info = infos[source]
        for window in WINDOWS:
            raw = frames[source][frames[source].window_hours.eq(window)].copy()
            raw = raw[raw[info.patient].map(split_map).eq("test")]
            preprocessor = joblib.load(
                output / "models" / f"preprocessor__{source}__{window}h.joblib"
            )
            features = list(preprocessor.feature_names_in_)
            base_x = preprocessor.transform(raw[features]).astype("float32")
            y = raw.mortality.to_numpy()
            for name in MODELS:
                for seed in SEEDS:
                    model = load_model(output, source, window, name, seed, base_x.shape[1])
                    baseline = float(average_precision_score(y, model_predict(name, model, base_x)))
                    for group, columns in groups.items():
                        changed = raw[features].copy()
                        rng = np.random.default_rng(seed + sum(map(ord, group)))
                        for column in columns:
                            changed[column] = rng.permutation(changed[column].to_numpy())
                        probability = model_predict(
                            name,
                            model,
                            preprocessor.transform(changed).astype("float32"),
                        )
                        permuted = float(average_precision_score(y, probability))
                        rows.append(
                            {
                                "source_database": source,
                                "window_hours": window,
                                "model": name,
                                "seed": seed,
                                "mechanism_group": group,
                                "baseline_auprc": baseline,
                                "permuted_auprc": permuted,
                                "reliance_auprc_drop": baseline - permuted,
                            }
                        )
    result = pd.DataFrame(rows)
    if len(result) != 864:
        raise RuntimeError(f"Expected 864 predictive-reliance cells, produced {len(result)}")
    result.to_csv(output / "model_reliance_metrics.csv", index=False)
    return result
