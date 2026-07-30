"""Generate only the five figures present in the main manuscript."""

# ruff: noqa: E402

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_cache = Path(tempfile.gettempdir()) / "cmpb_matplotlib"
_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .make_tables import build_manuscript_tables


FIGURE_SOURCE_FILES = (
    "figure_2_external_auroc_distributions_source.csv",
    "figure_3_directional_auroc_degradation_source.csv",
    "figure_4_raw_platt_isotonic_ece_source.csv",
    "figure_5_database_separability_source.csv",
    "figure_6_predictive_reliance_source.csv",
)


def build_figure_source_data(results: Path) -> dict[str, pd.DataFrame]:
    external = pd.read_csv(results / "external_metrics.csv")
    tables = build_manuscript_tables(results)
    return {
        FIGURE_SOURCE_FILES[0]: external[
            ["source_database", "target_database", "window_hours", "model", "seed", "auroc"]
        ].sort_values(["model", "source_database", "target_database", "window_hours", "seed"]),
        FIGURE_SOURCE_FILES[1]: tables["table_3_performance_by_transfer_direction.csv"][
            ["source_database", "target_database", "auroc_degradation"]
        ],
        FIGURE_SOURCE_FILES[2]: tables["table_5_raw_and_calibrated_probability_quality.csv"][
            ["model", "method", "ece"]
        ],
        FIGURE_SOURCE_FILES[3]: tables[
            "table_6_database_separability_and_predictive_reliance.csv"
        ][["feature_view", "domain_auroc"]],
        FIGURE_SOURCE_FILES[4]: tables[
            "table_6_database_separability_and_predictive_reliance.csv"
        ][["feature_view", "auprc_reliance_decrease"]].dropna(),
    }


def _save(figure: plt.Figure, directory: Path, stem: str) -> None:
    figure.tight_layout()
    figure.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def generate_manuscript_figures(output: Path) -> list[Path]:
    directory = output / "manuscript_figures"
    source_directory = directory / "source_data"
    source_directory.mkdir(parents=True, exist_ok=True)
    sources = build_figure_source_data(output)
    for filename, frame in sources.items():
        frame.to_csv(source_directory / filename, index=False)

    auroc = sources[FIGURE_SOURCE_FILES[0]]
    models = auroc.groupby("model").auroc.mean().sort_values(ascending=False).index
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.boxplot(
        [auroc.loc[auroc.model.eq(model), "auroc"] for model in models],
        tick_labels=models,
    )
    axis.set(ylabel="External AUROC", xlabel="Model family")
    axis.tick_params(axis="x", rotation=25)
    axis.spines[["top", "right"]].set_visible(False)
    _save(figure, directory, "figure_2_external_auroc_distributions_by_model_family")

    directional = sources[FIGURE_SOURCE_FILES[1]]
    databases = ["MIMIC-III", "MIMIC-IV", "eICU"]
    matrix = directional.pivot(index="source_database", columns="target_database", values="auroc_degradation").reindex(index=databases, columns=databases)
    figure, axis = plt.subplots(figsize=(5.8, 4.8))
    image = axis.imshow(matrix.to_numpy(), cmap="Reds", vmin=0, vmax=np.nanmax(matrix.to_numpy()))
    axis.set_xticks(range(3), databases, rotation=25, ha="right")
    axis.set_yticks(range(3), databases)
    axis.set(xlabel="Target database", ylabel="Source database")
    for row in range(3):
        for column in range(3):
            value = matrix.iloc[row, column]
            axis.text(column, row, "-" if pd.isna(value) else f"{value:.4f}", ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Mean AUROC degradation")
    _save(figure, directory, "figure_3_directional_auroc_degradation_heatmap")

    ece = sources[FIGURE_SOURCE_FILES[2]]
    method_order = ["uncalibrated", "platt", "isotonic"]
    ece_pivot = ece.pivot(index="model", columns="method", values="ece").reindex(columns=method_order)
    figure, axis = plt.subplots(figsize=(9, 4.8))
    positions, width = np.arange(len(ece_pivot)), 0.24
    for index, method in enumerate(method_order):
        axis.bar(positions + (index - 1) * width, ece_pivot[method], width, label=method.title())
    axis.set_xticks(positions, ece_pivot.index, rotation=25, ha="right")
    axis.set(ylabel="Mean external expected calibration error", xlabel="Model family")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    _save(figure, directory, "figure_4_raw_platt_isotonic_ece_comparison")

    domain = sources[FIGURE_SOURCE_FILES[3]].sort_values("domain_auroc", ascending=False)
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.barh(domain.feature_view, domain.domain_auroc, color="#4477AA")
    axis.axvline(0.5, color="black", linestyle="--", linewidth=1)
    axis.invert_yaxis()
    axis.set(xlabel="Held-out domain AUROC", ylabel="Feature view", xlim=(0.5, 1.0))
    axis.spines[["top", "right"]].set_visible(False)
    _save(figure, directory, "figure_5_database_separability_by_feature_view")

    reliance = sources[FIGURE_SOURCE_FILES[4]].sort_values("auprc_reliance_decrease", ascending=False)
    figure, axis = plt.subplots(figsize=(8, 4.6))
    axis.barh(reliance.feature_view, reliance.auprc_reliance_decrease, color="#228833")
    axis.invert_yaxis()
    axis.set(xlabel="Mean AUPRC decrease after permutation", ylabel="Feature group")
    axis.spines[["top", "right"]].set_visible(False)
    _save(figure, directory, "figure_6_predictive_reliance_by_feature_group")
    return sorted(path for path in directory.rglob("*") if path.is_file())
