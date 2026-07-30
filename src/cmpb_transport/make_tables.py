"""Build only the six main-manuscript tables from final analysis results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TABLE_FILES = (
    "table_1_cohort_characteristics.csv",
    "table_2_external_performance_by_model_family.csv",
    "table_3_performance_by_transfer_direction.csv",
    "table_4_performance_by_observation_window.csv",
    "table_5_raw_and_calibrated_probability_quality.csv",
    "table_6_database_separability_and_predictive_reliance.csv",
)


def _read(results: Path, name: str) -> pd.DataFrame:
    path = results / name
    if not path.is_file():
        raise FileNotFoundError(f"Required analysis result is missing: {path}")
    return pd.read_csv(path)


def build_manuscript_tables(results: Path) -> dict[str, pd.DataFrame]:
    cohort = _read(results, "cohort_summary.csv")
    external = _read(results, "external_metrics.csv")
    degradation = _read(results, "degradation_metrics.csv")
    calibration = _read(results, "calibration_metrics.csv")
    domain = _read(results, "domain_separability_metrics.csv")
    reliance = _read(results, "model_reliance_metrics.csv")

    first = cohort[cohort.window_hours.eq(1)].copy()
    last = cohort[cohort.window_hours.eq(24)][["database", "overall_missingness"]]
    table_1 = first.merge(last, on="database", suffixes=("_1h", "_24h"))[
        [
            "database",
            "icu_stays",
            "deaths",
            "mortality_rate",
            "age_mean",
            "age_sd",
            "overall_missingness_1h",
            "overall_missingness_24h",
        ]
    ]
    table_1[["mortality_rate", "overall_missingness_1h", "overall_missingness_24h"]] *= 100

    table_2 = (
        external.groupby("model", as_index=False)[["auroc", "auprc", "brier", "ece"]]
        .mean()
        .sort_values("auroc", ascending=False)
        .reset_index(drop=True)
    )
    table_3 = (
        degradation.groupby(["source_database", "target_database"], as_index=False)[
            ["auroc_degradation", "auprc_degradation", "brier_increase"]
        ]
        .mean()
        .sort_values(["source_database", "target_database"])
        .reset_index(drop=True)
    )
    table_4 = (
        external.groupby(["model", "window_hours"], as_index=False)[["auroc", "auprc"]]
        .mean()
        .sort_values(["model", "window_hours"])
        .reset_index(drop=True)
    )
    primary = calibration[
        calibration.bins.eq(15) & calibration.binning.eq("equal_frequency")
    ]
    table_5 = (
        primary.groupby(["model", "method"], as_index=False)[["brier", "ece"]]
        .mean()
        .sort_values(["model", "method"])
        .reset_index(drop=True)
    )
    domain_summary = (
        domain.assign(
            feature_view=domain.feature_view.replace({"valid_full": "complete predictor set"})
        )
        .groupby("feature_view", as_index=False)
        .auroc.mean()
        .rename(columns={"auroc": "domain_auroc"})
    )
    reliance_summary = (
        reliance.groupby("mechanism_group", as_index=False)
        .reliance_auprc_drop.mean()
        .rename(columns={
            "mechanism_group": "feature_view",
            "reliance_auprc_drop": "auprc_reliance_decrease",
        })
    )
    table_6 = (
        domain_summary.merge(reliance_summary, on="feature_view", how="left")
        .sort_values("domain_auroc", ascending=False)
        .reset_index(drop=True)
    )
    return dict(zip(TABLE_FILES, (table_1, table_2, table_3, table_4, table_5, table_6)))


def write_manuscript_tables(results: Path) -> list[Path]:
    directory = results / "manuscript_tables"
    directory.mkdir(exist_ok=True)
    paths = []
    for filename, frame in build_manuscript_tables(results).items():
        path = directory / filename
        frame.to_csv(path, index=False)
        paths.append(path)
    return paths


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Compatibility helper: mean primary metrics by manuscript evaluation cell."""
    groups = [
        column
        for column in ("source_database", "target_database", "window_hours", "model")
        if column in frame
    ]
    return frame.groupby(groups, as_index=False)[["auroc", "auprc", "brier", "ece"]].mean()
