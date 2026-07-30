"""Deterministic patient-cluster bootstrap for newly generated predictions."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

N_BOOTSTRAP = 2000
GLOBAL_SEED = 20260713


def stable_seed(key: tuple[Any, ...]) -> int:
    """Derive a process-independent seed from the configured global seed and cell key."""
    digest = hashlib.sha256((str(GLOBAL_SEED) + repr(key)).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def metric_vector(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute metrics with one stable probability sort, including exact tie handling."""
    order = np.argsort(-p, kind="stable")
    sorted_p = p[order]
    sorted_y = y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_p) != 0) + 1]
    group_sizes = np.diff(np.r_[starts, len(y)])
    group_positive = np.add.reduceat(sorted_y, starts)
    group_negative = group_sizes - group_positive
    total_positive = int(sorted_y.sum())
    total_negative = len(y) - total_positive
    cumulative_positive = np.cumsum(group_positive)
    cumulative_total = np.cumsum(group_sizes)
    group_precision = cumulative_positive / cumulative_total
    auprc_value = float(np.sum((group_positive / total_positive) * group_precision))
    negative_below = total_negative - np.cumsum(group_negative)
    auroc_value = float(
        np.sum(group_positive * (negative_below + 0.5 * group_negative))
        / (total_positive * total_negative)
    )
    ece_value = 0.0
    for group in np.array_split(np.arange(len(y)), 15):
        if len(group) == 0:
            continue
        ece_value += len(group) / len(y) * abs(sorted_y[group].mean() - sorted_p[group].mean())
    prediction = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()

    def divide(a, b):
        return float(a / b) if b else np.nan

    return {
        "auroc": auroc_value,
        "auprc": auprc_value,
        "brier": float(np.mean((y - p) ** 2)),
        "ece": ece_value,
        "sensitivity": recall_score(y, prediction),
        "specificity": divide(tn, tn + fp),
        "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction),
        "f1": f1_score(y, prediction),
        "positive_predictive_value": divide(tp, tp + fp),
        "negative_predictive_value": divide(tn, tn + fn),
    }


def ensemble(files: list[Path]) -> pd.DataFrame:
    """Average three seed predictions after strict identifier/label alignment."""
    frames = [
        pd.read_parquet(path).sort_values(["patient_id", "stay_id"]).reset_index(drop=True)
        for path in files
    ]
    base = frames[0].copy()
    for other in frames[1:]:
        if not base[["patient_id", "stay_id", "true_label"]].equals(
            other[["patient_id", "stay_id", "true_label"]]
        ):
            raise RuntimeError("Seed prediction populations do not align")
    base["predicted_probability"] = np.mean(
        [frame.predicted_probability.to_numpy() for frame in frames], axis=0
    )
    base["selected_threshold"] = float(
        np.mean([frame.selected_threshold.iloc[0] for frame in frames])
    )
    return base


def draw_cluster_indices(
    rng: np.random.Generator, groups: list[np.ndarray], n_rows: int
) -> np.ndarray:
    """Draw complete clusters, with an exact fast path when every patient has one row."""
    if len(groups) == n_rows and all(len(group) == 1 for group in groups):
        return rng.integers(0, n_rows, n_rows)
    chosen = rng.integers(0, len(groups), len(groups))
    return np.concatenate([groups[index] for index in chosen])


def draw_counts(rng: np.random.Generator, n_rows: int, batch: int) -> np.ndarray:
    """Draw exact ordinary-bootstrap multiplicities for singleton patient clusters."""
    counts = np.empty((batch, n_rows), dtype=np.int16)
    for row in range(batch):
        counts[row] = np.bincount(rng.integers(0, n_rows, n_rows), minlength=n_rows)
    return counts


def metrics_from_counts(
    counts: np.ndarray, y: np.ndarray, p: np.ndarray, threshold: float
) -> dict[str, np.ndarray]:
    """Compute exact weighted metrics for bootstrap multiplicities after one fixed sort."""
    n_rows = len(y)
    total_positive = counts @ y
    total_negative = n_rows - total_positive
    valid = (total_positive > 0) & (total_negative > 0)
    order = np.argsort(-p, kind="stable")
    sorted_p = p[order]
    sorted_y = y[order]
    sorted_counts = counts[:, order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_p) != 0) + 1]
    group_positive = np.add.reduceat(sorted_counts * sorted_y, starts, axis=1)
    group_total = np.add.reduceat(sorted_counts, starts, axis=1)
    group_negative = group_total - group_positive
    cumulative_positive = np.cumsum(group_positive, axis=1)
    cumulative_total = np.cumsum(group_total, axis=1)
    precision_at_group = np.divide(
        cumulative_positive,
        cumulative_total,
        out=np.zeros_like(cumulative_positive, dtype=float),
        where=cumulative_total > 0,
    )
    auprc_values = np.divide(
        np.sum(group_positive * precision_at_group, axis=1),
        total_positive,
        out=np.full(len(counts), np.nan),
        where=total_positive > 0,
    )
    negative_below = total_negative[:, None] - np.cumsum(group_negative, axis=1)
    auroc_values = np.divide(
        np.sum(group_positive * (negative_below + 0.5 * group_negative), axis=1),
        total_positive * total_negative,
        out=np.full(len(counts), np.nan),
        where=valid,
    )
    brier_values = counts @ ((y - p) ** 2) / n_rows
    cumulative_count = np.cumsum(sorted_counts, axis=1)
    cumulative_y = np.cumsum(sorted_counts * sorted_y, axis=1)
    cumulative_p = np.cumsum(sorted_counts * sorted_p, axis=1)
    bin_sizes = np.array([len(group) for group in np.array_split(np.arange(n_rows), 15)])
    boundaries = np.cumsum(bin_sizes)
    prefix_y = np.empty((len(counts), len(boundaries)))
    prefix_p = np.empty_like(prefix_y)
    row_ids = np.arange(len(counts))
    for column, boundary in enumerate(boundaries):
        indices = np.array(
            [np.searchsorted(cumulative_count[row], boundary, side="left") for row in row_ids]
        )
        before_count = np.where(
            indices > 0, cumulative_count[row_ids, np.maximum(indices - 1, 0)], 0
        )
        before_y = np.where(indices > 0, cumulative_y[row_ids, np.maximum(indices - 1, 0)], 0)
        before_p = np.where(indices > 0, cumulative_p[row_ids, np.maximum(indices - 1, 0)], 0)
        remainder = boundary - before_count
        prefix_y[:, column] = before_y + remainder * sorted_y[indices]
        prefix_p[:, column] = before_p + remainder * sorted_p[indices]
    bin_y = np.diff(np.column_stack([np.zeros(len(counts)), prefix_y]), axis=1)
    bin_p = np.diff(np.column_stack([np.zeros(len(counts)), prefix_p]), axis=1)
    ece_values = np.sum(np.abs(bin_y - bin_p), axis=1) / n_rows
    positive_prediction = p >= threshold
    tp = counts @ (y * positive_prediction)
    fp = counts @ ((1 - y) * positive_prediction)
    fn = total_positive - tp
    tn = total_negative - fp

    def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        return np.divide(
            numerator,
            denominator,
            out=np.full(len(counts), np.nan),
            where=denominator > 0,
        )

    sensitivity = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    precision = ratio(tp, tp + fp)
    f1 = ratio(2 * precision * sensitivity, precision + sensitivity)
    return {
        "auroc": auroc_values,
        "auprc": auprc_values,
        "brier": brier_values,
        "ece": ece_values,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "recall": sensitivity,
        "f1": f1,
        "positive_predictive_value": precision,
        "negative_predictive_value": ratio(tn, tn + fn),
        "valid": valid,
    }


def bootstrap_cell(key: tuple[Any, ...], frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Bootstrap complete patient clusters for one fixed analytical cell."""
    groups = [part.index.to_numpy() for _, part in frame.groupby("patient_id", sort=False)]
    rng = np.random.default_rng(stable_seed(key))
    point = metric_vector(
        frame.true_label.to_numpy(),
        frame.predicted_probability.to_numpy(),
        float(frame.selected_threshold.iloc[0]),
    )
    values: dict[str, list[float]] = {metric: [] for metric in point}
    skipped = 0
    if len(groups) == len(frame) and all(len(group) == 1 for group in groups):
        remaining = N_BOOTSTRAP
        while remaining:
            batch = min(20, remaining)
            batch_result = metrics_from_counts(
                draw_counts(rng, len(frame), batch),
                frame.true_label.to_numpy(),
                frame.predicted_probability.to_numpy(),
                float(frame.selected_threshold.iloc[0]),
            )
            valid = batch_result.pop("valid")
            skipped += int((~valid).sum())
            for metric, metric_values in batch_result.items():
                values[metric].extend(metric_values[valid].tolist())
            remaining -= batch
    else:
        for _ in range(N_BOOTSTRAP):
            indices = draw_cluster_indices(rng, groups, len(frame))
            sample = frame.loc[indices]
            if sample.true_label.nunique() != 2:
                skipped += 1
                continue
            sample_result = metric_vector(
                sample.true_label.to_numpy(),
                sample.predicted_probability.to_numpy(),
                float(frame.selected_threshold.iloc[0]),
            )
            for metric, value in sample_result.items():
                values[metric].append(value)
    rows: list[dict[str, Any]] = []
    for metric, replicates in values.items():
        rows.append(
            {
                **dict(zip(key[0], key[1], strict=True)),
                "metric": metric,
                "point_estimate": point[metric],
                "ci_lower": float(np.nanquantile(replicates, 0.025)),
                "ci_upper": float(np.nanquantile(replicates, 0.975)),
                "bootstrap_replicates_requested": N_BOOTSTRAP,
                "bootstrap_replicates_valid": len(replicates),
                "bootstrap_replicates_skipped": skipped,
                "bootstrap_unit": "patient_id",
                "random_seed": stable_seed(key),
            }
        )
    return rows


def discover_cells(output: Path) -> list[tuple[tuple[Any, ...], pd.DataFrame]]:
    """Discover internal and external seed-ensemble cells from new prediction files only."""
    metrics = pd.read_csv(output / "external_metrics.csv")
    cells = []
    for row in (
        metrics[["source_database", "target_database", "window_hours", "model"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        slug = row.model.lower().replace(" ", "_").replace("-", "_")
        files = [
            output
            / "predictions"
            / f"{row.source_database}__{row.window_hours}h__{slug}__seed{seed}__to__{row.target_database}.parquet"
            for seed in (17, 42, 2026)
        ]
        key = (
            ("evaluation_type", "source_database", "target_database", "window_hours", "model"),
            ("external", row.source_database, row.target_database, row.window_hours, row.model),
        )
        cells.append((key, ensemble(files)))
    internal = pd.read_csv(output / "internal_metrics.csv")
    for row in (
        internal[["source_database", "window_hours", "model"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        slug = row.model.lower().replace(" ", "_").replace("-", "_")
        files = [
            output
            / "predictions"
            / f"{row.source_database}__{row.window_hours}h__{slug}__seed{seed}__internal.parquet"
            for seed in (17, 42, 2026)
        ]
        key = (
            ("evaluation_type", "source_database", "target_database", "window_hours", "model"),
            ("internal", row.source_database, row.source_database, row.window_hours, row.model),
        )
        cells.append((key, ensemble(files)))
    return cells


def paired_comparison(
    key: tuple[Any, ...], reference: pd.DataFrame, candidate: pd.DataFrame
) -> list[dict[str, Any]]:
    """Paired patient bootstrap of candidate-minus-logistic metric differences."""
    if not reference[["patient_id", "stay_id", "true_label"]].equals(
        candidate[["patient_id", "stay_id", "true_label"]]
    ):
        raise RuntimeError("Paired model populations do not align")
    rng = np.random.default_rng(stable_seed(key))
    [part.index.to_numpy() for _, part in reference.groupby("patient_id", sort=False)]
    metrics = ("auroc", "auprc", "brier")
    point_ref = metric_vector(
        reference.true_label.to_numpy(),
        reference.predicted_probability.to_numpy(),
        float(reference.selected_threshold.iloc[0]),
    )
    point_can = metric_vector(
        candidate.true_label.to_numpy(),
        candidate.predicted_probability.to_numpy(),
        float(candidate.selected_threshold.iloc[0]),
    )
    values: dict[str, list[float]] = {metric: [] for metric in metrics}
    skipped = 0
    remaining = N_BOOTSTRAP
    while remaining:
        batch = min(20, remaining)
        counts = draw_counts(rng, len(reference), batch)
        a = metrics_from_counts(
            counts,
            reference.true_label.to_numpy(),
            reference.predicted_probability.to_numpy(),
            float(reference.selected_threshold.iloc[0]),
        )
        b = metrics_from_counts(
            counts,
            candidate.true_label.to_numpy(),
            candidate.predicted_probability.to_numpy(),
            float(candidate.selected_threshold.iloc[0]),
        )
        valid = a.pop("valid") & b.pop("valid")
        skipped += int((~valid).sum())
        for metric in metrics:
            values[metric].extend((b[metric][valid] - a[metric][valid]).tolist())
        remaining -= batch
    return [
        {
            **dict(zip(key[0], key[1], strict=True)),
            "metric": metric,
            "difference_candidate_minus_logistic": point_can[metric] - point_ref[metric],
            "ci_lower": float(np.quantile(value, 0.025)),
            "ci_upper": float(np.quantile(value, 0.975)),
            "bootstrap_replicates_valid": len(value),
            "bootstrap_replicates_skipped": skipped,
            "random_seed": stable_seed(key),
        }
        for metric, value in values.items()
    ]


def run(output: Path, n_jobs: int = 8) -> None:
    """Run 2,000-replicate cell CIs and paired external model comparisons."""
    cells = discover_cells(output)
    rows = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(bootstrap_cell)(key, frame) for key, frame in cells
    )
    pd.DataFrame([row for group in rows for row in group]).to_csv(
        output / "bootstrap_confidence_intervals.csv", index=False
    )
    external = {
        (key[1][1], key[1][2], key[1][3], key[1][4]): frame
        for key, frame in cells
        if key[1][0] == "external"
    }
    tasks = []
    for (source, target, window, model), candidate in external.items():
        if model == "Logistic Regression":
            continue
        reference = external[(source, target, window, "Logistic Regression")]
        key = (
            ("source_database", "target_database", "window_hours", "candidate_model"),
            (source, target, window, model),
        )
        tasks.append((key, reference, candidate))
    comparisons = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(paired_comparison)(*task) for task in tasks
    )
    pd.DataFrame([row for group in comparisons for row in group]).to_csv(
        output / "paired_model_comparisons.csv", index=False
    )
