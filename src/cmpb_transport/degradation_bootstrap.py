"""Independent source-internal and external-target degradation bootstrap."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from cmpb_transport.snapshot_bootstrap_analysis import (
    GLOBAL_SEED,
    N_BOOTSTRAP,
    discover_cells,
    draw_counts,
    metric_vector,
    stable_seed,
)


def discrimination_from_counts(
    counts: np.ndarray, y: np.ndarray, probability: np.ndarray
) -> dict[str, np.ndarray]:
    """Compute exact tied-score AUROC, average precision, and Brier for row weights.

    Inputs are bootstrap multiplicities, binary labels, and probabilities. Outputs
    are one value per replicate. Labels must contain both classes and probabilities
    must be finite in [0, 1]. One-class replicates are marked invalid. The formulas
    use weighted Mann-Whitney AUROC, stepwise average precision, and mean squared
    probability error; malformed arrays raise an exception.
    """
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Point-estimate population must contain both outcome classes")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be finite and lie in [0, 1]")
    n_rows = len(y)
    positives = counts @ y
    negatives = n_rows - positives
    valid = (positives > 0) & (negatives > 0)
    order = np.argsort(-probability, kind="stable")
    sorted_probability = probability[order]
    sorted_y = y[order]
    sorted_counts = counts[:, order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_probability) != 0) + 1]
    group_positive = np.add.reduceat(sorted_counts * sorted_y, starts, axis=1)
    group_total = np.add.reduceat(sorted_counts, starts, axis=1)
    group_negative = group_total - group_positive
    cumulative_positive = np.cumsum(group_positive, axis=1)
    cumulative_total = np.cumsum(group_total, axis=1)
    precision = np.divide(
        cumulative_positive,
        cumulative_total,
        out=np.zeros_like(cumulative_positive, dtype=float),
        where=cumulative_total > 0,
    )
    auprc = np.divide(
        np.sum(group_positive * precision, axis=1),
        positives,
        out=np.full(len(counts), np.nan),
        where=positives > 0,
    )
    negative_below = negatives[:, None] - np.cumsum(group_negative, axis=1)
    auroc = np.divide(
        np.sum(group_positive * (negative_below + 0.5 * group_negative), axis=1),
        positives * negatives,
        out=np.full(len(counts), np.nan),
        where=valid,
    )
    brier = counts @ ((y - probability) ** 2) / n_rows
    return {"auroc": auroc, "auprc": auprc, "brier": brier, "valid": valid}


def bootstrap_degradation(
    key: tuple[Any, ...], internal: pd.DataFrame, external: pd.DataFrame
) -> list[dict[str, Any]]:
    """Bootstrap metric degradation with independent patient-population draws.

    The inputs are aligned seed-ensemble prediction tables from distinct internal
    and external populations. Output rows contain percentile confidence intervals.
    Each snapshot has one row per patient, so ordinary multiplicity draws are exact
    patient-cluster draws. Internal and external draws are independent. Replicates
    containing one outcome class are skipped and counted. Invalid schemas fail.
    """
    rng_internal = np.random.default_rng(stable_seed(key + (("internal",),)))
    rng_external = np.random.default_rng(stable_seed(key + (("external",),)))
    internal_y = internal.true_label.to_numpy(dtype=np.int64)
    external_y = external.true_label.to_numpy(dtype=np.int64)
    internal_p = internal.predicted_probability.to_numpy(dtype=float)
    external_p = external.predicted_probability.to_numpy(dtype=float)
    point_internal = metric_vector(
        internal_y, internal_p, float(internal.selected_threshold.iloc[0])
    )
    point_external = metric_vector(
        external_y, external_p, float(external.selected_threshold.iloc[0])
    )
    values: dict[str, list[float]] = {metric: [] for metric in ("auroc", "auprc", "brier")}
    skipped = 0
    remaining = N_BOOTSTRAP
    while remaining:
        batch = min(40, remaining)
        internal_result = discrimination_from_counts(
            draw_counts(rng_internal, len(internal), batch), internal_y, internal_p
        )
        external_result = discrimination_from_counts(
            draw_counts(rng_external, len(external), batch), external_y, external_p
        )
        valid = internal_result.pop("valid") & external_result.pop("valid")
        skipped += int((~valid).sum())
        for metric in values:
            if metric == "brier":
                difference = external_result[metric] - internal_result[metric]
            else:
                difference = internal_result[metric] - external_result[metric]
            values[metric].extend(difference[valid].tolist())
        remaining -= batch
    rows: list[dict[str, Any]] = []
    for metric, replicates in values.items():
        point = (
            point_external[metric] - point_internal[metric]
            if metric == "brier"
            else point_internal[metric] - point_external[metric]
        )
        rows.append(
            {
                **dict(zip(key[0], key[1], strict=True)),
                "metric": metric,
                "point_estimate": point,
                "ci_lower": float(np.quantile(replicates, 0.025)),
                "ci_upper": float(np.quantile(replicates, 0.975)),
                "bootstrap_replicates_requested": N_BOOTSTRAP,
                "bootstrap_replicates_valid": len(replicates),
                "bootstrap_replicates_skipped": skipped,
                "internal_bootstrap_unit": "patient_id",
                "external_bootstrap_unit": "patient_id",
                "bootstrap_population_relation": "independent",
                "random_seed_base": GLOBAL_SEED,
            }
        )
    return rows


def run(output: Path, n_jobs: int = 8) -> None:
    """Run all 144 ordered-transfer degradation bootstrap cells and save CSV output."""
    cells = discover_cells(output)
    internal = {
        (key[1][1], key[1][3], key[1][4]): frame
        for key, frame in cells
        if key[1][0] == "internal"
    }
    tasks = []
    for key, frame in cells:
        if key[1][0] != "external":
            continue
        _, source, target, window, model = key[1]
        result_key = (
            ("source_database", "target_database", "window_hours", "model"),
            (source, target, window, model),
        )
        tasks.append((result_key, internal[(source, window, model)], frame))
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(bootstrap_degradation)(*task) for task in tasks
    )
    pd.DataFrame([row for group in results for row in group]).to_csv(
        output / "bootstrap_degradation_confidence_intervals.csv", index=False
    )
