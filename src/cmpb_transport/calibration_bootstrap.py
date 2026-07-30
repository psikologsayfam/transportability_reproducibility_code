"""Paired bootstrap intervals for external calibration improvements."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from cmpb_transport.metrics import ece
from cmpb_transport.snapshot_bootstrap_analysis import (
    N_BOOTSTRAP,
    draw_counts,
    stable_seed,
)

SEEDS = (17, 42, 2026)


def logit(probability: np.ndarray) -> np.ndarray:
    """Return finite logits for probabilities clipped only at machine-stable bounds."""
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def prepare_calibrated_ensemble(
    output: Path, source: str, target: str, window: int, model: str
) -> pd.DataFrame:
    """Fit seed-specific source-validation calibrators and ensemble target outputs.

    Inputs identify one external transfer cell. The returned table contains aligned
    target labels and three probabilities. Platt and isotonic fits use source
    validation predictions only. Target labels are never passed to a fitter.
    Identifier or label disagreement across seeds raises an exception.
    """
    slug = model.lower().replace(" ", "_").replace("-", "_")
    frames: list[pd.DataFrame] = []
    platt_probabilities: list[np.ndarray] = []
    isotonic_probabilities: list[np.ndarray] = []
    for seed in SEEDS:
        base = f"{source}__{window}h__{slug}__seed{seed}"
        validation = pd.read_parquet(output / "predictions" / f"{base}__validation.parquet")
        external = pd.read_parquet(
            output / "predictions" / f"{base}__to__{target}.parquet"
        ).sort_values(["patient_id", "stay_id"]).reset_index(drop=True)
        platt = LogisticRegression().fit(
            logit(validation.predicted_probability.to_numpy()), validation.true_label
        )
        isotonic = IsotonicRegression(out_of_bounds="clip").fit(
            validation.predicted_probability.to_numpy(), validation.true_label.to_numpy()
        )
        frames.append(external)
        platt_probabilities.append(
            platt.predict_proba(logit(external.predicted_probability.to_numpy()))[:, 1]
        )
        isotonic_probabilities.append(
            isotonic.predict(external.predicted_probability.to_numpy())
        )
    base_frame = frames[0][["patient_id", "stay_id", "true_label"]].copy()
    for frame in frames[1:]:
        if not base_frame.equals(frame[["patient_id", "stay_id", "true_label"]]):
            raise RuntimeError("Calibration ensemble populations do not align across seeds")
    base_frame["uncalibrated"] = np.mean(
        [frame.predicted_probability.to_numpy() for frame in frames], axis=0
    )
    base_frame["platt"] = np.mean(platt_probabilities, axis=0)
    base_frame["isotonic"] = np.mean(isotonic_probabilities, axis=0)
    return base_frame


def calibration_from_counts(
    counts: np.ndarray,
    y: np.ndarray,
    probability: np.ndarray,
    order: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute exact weighted Brier score and 15-bin equal-frequency ECE.

    Inputs are bootstrap multiplicities, binary labels, and probabilities. Outputs
    contain one metric per replicate plus a one-class validity mask. Bins contain
    equal resampled-row counts, with exact boundary interpolation for multiplicity
    ties. Invalid probabilities or one-class point populations raise an exception.
    """
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Calibration population must contain both classes")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be finite and lie in [0, 1]")
    n_rows = len(y)
    positives = counts @ y
    valid = (positives > 0) & (positives < n_rows)
    brier = counts @ ((y - probability) ** 2) / n_rows
    if order is None:
        order = np.argsort(-probability, kind="stable")
    sorted_counts = counts[:, order]
    sorted_y = y[order]
    sorted_probability = probability[order]
    cumulative_count = np.cumsum(sorted_counts, axis=1)
    cumulative_y = np.cumsum(sorted_counts * sorted_y, axis=1)
    cumulative_probability = np.cumsum(sorted_counts * sorted_probability, axis=1)
    bin_sizes = np.array([len(group) for group in np.array_split(np.arange(n_rows), 15)])
    boundaries = np.cumsum(bin_sizes)
    prefix_y = np.empty((len(counts), len(boundaries)))
    prefix_probability = np.empty_like(prefix_y)
    row_ids = np.arange(len(counts))
    for column, boundary in enumerate(boundaries):
        indices = np.array(
            [np.searchsorted(cumulative_count[row], boundary, side="left") for row in row_ids]
        )
        prior = np.maximum(indices - 1, 0)
        before_count = np.where(indices > 0, cumulative_count[row_ids, prior], 0)
        before_y = np.where(indices > 0, cumulative_y[row_ids, prior], 0)
        before_probability = np.where(
            indices > 0, cumulative_probability[row_ids, prior], 0
        )
        remainder = boundary - before_count
        prefix_y[:, column] = before_y + remainder * sorted_y[indices]
        prefix_probability[:, column] = (
            before_probability + remainder * sorted_probability[indices]
        )
    bin_y = np.diff(np.column_stack([np.zeros(len(counts)), prefix_y]), axis=1)
    bin_probability = np.diff(
        np.column_stack([np.zeros(len(counts)), prefix_probability]), axis=1
    )
    ece_values = np.sum(np.abs(bin_y - bin_probability), axis=1) / n_rows
    return {"brier": brier, "ece": ece_values, "valid": valid}


def bootstrap_cell(output: Path, key: tuple[Any, ...]) -> list[dict[str, Any]]:
    """Return paired target-patient intervals for Platt and isotonic improvements."""
    source, target, window, model = key[1]
    frame = prepare_calibrated_ensemble(output, source, target, window, model)
    y = frame.true_label.to_numpy(dtype=np.int64)
    point = {
        method: {
            "brier": float(np.mean((y - frame[method].to_numpy()) ** 2)),
            "ece": ece(y, frame[method].to_numpy(), 15, "equal_frequency")[0],
        }
        for method in ("uncalibrated", "platt", "isotonic")
    }
    rng = np.random.default_rng(stable_seed(key))
    orders = {
        method: np.argsort(-frame[method].to_numpy(), kind="stable")
        for method in ("uncalibrated", "platt", "isotonic")
    }
    values: dict[tuple[str, str], list[float]] = {
        (method, metric): []
        for method in ("platt", "isotonic")
        for metric in ("brier", "ece")
    }
    skipped = 0
    remaining = N_BOOTSTRAP
    while remaining:
        batch = min(40, remaining)
        counts = draw_counts(rng, len(frame), batch)
        results = {
            method: calibration_from_counts(
                counts, y, frame[method].to_numpy(), orders[method]
            )
            for method in ("uncalibrated", "platt", "isotonic")
        }
        valid = results["uncalibrated"]["valid"]
        skipped += int((~valid).sum())
        for method in ("platt", "isotonic"):
            for metric in ("brier", "ece"):
                improvement = results["uncalibrated"][metric] - results[method][metric]
                values[(method, metric)].extend(improvement[valid].tolist())
        remaining -= batch
    rows: list[dict[str, Any]] = []
    for (method, metric), replicates in values.items():
        rows.append(
            {
                **dict(zip(key[0], key[1], strict=True)),
                "calibration_method": method,
                "metric": metric,
                "improvement_uncalibrated_minus_calibrated": (
                    point["uncalibrated"][metric] - point[method][metric]
                ),
                "ci_lower": float(np.quantile(replicates, 0.025)),
                "ci_upper": float(np.quantile(replicates, 0.975)),
                "bootstrap_replicates_requested": N_BOOTSTRAP,
                "bootstrap_replicates_valid": len(replicates),
                "bootstrap_replicates_skipped": skipped,
                "bootstrap_unit": "patient_id",
                "random_seed": stable_seed(key),
            }
        )
    return rows


def run(output: Path, n_jobs: int = 8) -> None:
    """Run 144 external calibration-improvement cells and save interval output."""
    metrics = pd.read_csv(output / "external_metrics.csv")
    tasks = []
    for row in (
        metrics[["source_database", "target_database", "window_hours", "model"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        key = (
            ("source_database", "target_database", "window_hours", "model"),
            (row.source_database, row.target_database, row.window_hours, row.model),
        )
        tasks.append(key)
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(bootstrap_cell)(output, key) for key in tasks
    )
    pd.DataFrame([row for group in results for row in group]).to_csv(
        output / "bootstrap_calibration_improvements.csv", index=False
    )
