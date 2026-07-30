"""Fail-closed validation against the authoritative manuscript package."""

from __future__ import annotations

import json
import hashlib
import re
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from .make_figures import build_figure_source_data
from .make_tables import build_manuscript_tables


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


RAW_DATA_DIRECTORIES = {"data", "datasets", "raw", "raw_data", "raw_inputs"}
TEXT_SUFFIXES = {".py", ".md", ".json", ".toml", ".yaml", ".yml", ".txt"}


def validate_repository_data_scope(repository_root: Path) -> None:
    violations = []
    for path in repository_root.rglob("*"):
        relative = path.relative_to(repository_root)
        if any(part.lower() in {".git", ".venv", "__pycache__"} for part in relative.parts):
            continue
        if path.is_dir() and path.name.lower() in RAW_DATA_DIRECTORIES:
            violations.append(str(relative))
    if violations:
        raise RuntimeError(f"Raw clinical-data directories detected: {sorted(violations)}")


def validate_repository_privacy(repository_root: Path) -> None:
    patterns = {
        "private_key": re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
        "openai_token": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
        "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    }
    violations = []
    for path in repository_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(repository_root)
        if any(part.lower() in {".git", ".venv", "__pycache__"} for part in relative.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in patterns.items():
            if pattern.search(text):
                violations.append(f"{relative} ({label})")
    if violations:
        raise RuntimeError(f"Credential material detected: {sorted(violations)}")


@contextmanager
def _authoritative_root(package: Path):
    if package.is_dir():
        yield package
        return
    with tempfile.TemporaryDirectory(prefix="cmpb_authoritative_") as temporary:
        with zipfile.ZipFile(package) as archive:
            archive.extractall(temporary)
        candidates = list(Path(temporary).rglob("external_metrics.csv"))
        if len(candidates) != 1:
            raise RuntimeError("Authoritative archive must contain one external_metrics.csv")
        yield candidates[0].parent


def _compare_frames(
    actual: pd.DataFrame, expected: pd.DataFrame, label: str, tolerance: float = 1e-12
) -> dict[str, object]:
    actual = actual.reset_index(drop=True)
    expected = expected.reset_index(drop=True)
    if list(actual.columns) != list(expected.columns) or actual.shape != expected.shape:
        raise RuntimeError(
            f"{label}: schema/shape mismatch; actual={actual.shape}, expected={expected.shape}"
        )
    numeric = list(expected.select_dtypes(include=[np.number]).columns)
    text = [column for column in expected if column not in numeric]
    if text and not actual[text].fillna("<NA>").astype(str).equals(
        expected[text].fillna("<NA>").astype(str)
    ):
        raise RuntimeError(f"{label}: categorical cell mismatch")
    maximum = 0.0
    cells = 0
    for column in numeric:
        left = pd.to_numeric(actual[column], errors="coerce").to_numpy(float)
        right = pd.to_numeric(expected[column], errors="coerce").to_numpy(float)
        match = np.isclose(left, right, rtol=0, atol=tolerance, equal_nan=True)
        if not match.all():
            row = int(np.flatnonzero(~match)[0])
            raise RuntimeError(
                f"{label}: numeric mismatch at row {row}, column {column}: "
                f"{left[row]!r} != {right[row]!r} (atol={tolerance})"
            )
        finite = np.isfinite(left) & np.isfinite(right)
        if finite.any():
            maximum = max(maximum, float(np.max(np.abs(left[finite] - right[finite]))))
        cells += len(left)
    return {"artifact": label, "numeric_cells": cells, "tolerance": tolerance, "max_abs_diff": maximum}


def _coverage_checks(root: Path) -> dict[str, int]:
    expected = {
        "external_metrics.csv": 432,
        "internal_metrics.csv": 216,
        "calibration_metrics.csv": 5184,
        "domain_separability_metrics.csv": 360,
        "model_reliance_metrics.csv": 864,
        "paired_model_comparisons.csv": 360,
        "bootstrap_degradation_confidence_intervals.csv": 432,
    }
    observed = {}
    for filename, rows in expected.items():
        frame = pd.read_csv(root / filename)
        observed[filename] = len(frame)
        if len(frame) != rows:
            raise RuntimeError(f"Authoritative {filename}: expected {rows} rows, found {len(frame)}")
    for filename in (
        "paired_model_comparisons.csv",
        "bootstrap_degradation_confidence_intervals.csv",
    ):
        frame = pd.read_csv(root / filename)
        replicate_columns = [column for column in frame if "replicates" in column and "skipped" not in column]
        if not replicate_columns or not all((frame[column] == 2000).all() for column in replicate_columns):
            raise RuntimeError(f"Authoritative {filename}: 2,000-replicate contract is not satisfied")
    return observed


def validate_authoritative(output: Path, package: Path | None) -> dict[str, object]:
    report_path = output / "authoritative_numerical_validation.json"
    if package is None or not package.is_file() and not package.is_dir():
        report = {
            "status": "NOT_COMPLETED",
            "authoritative_numerical_validation_completed": False,
            "reason": "authoritative package not supplied or not found",
            "matched_tables": [],
            "matched_figure_source_data": [],
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    comparisons: list[dict[str, object]] = []
    try:
        with _authoritative_root(package) as root:
            coverage = _coverage_checks(root)
            expected_tables = build_manuscript_tables(root)
            expected_figures = build_figure_source_data(root)
            for filename, expected in expected_tables.items():
                actual = pd.read_csv(output / "manuscript_tables" / filename)
                comparisons.append(_compare_frames(actual, expected, f"table:{filename}"))
            for filename, expected in expected_figures.items():
                actual = pd.read_csv(output / "manuscript_figures" / "source_data" / filename)
                comparisons.append(_compare_frames(actual, expected, f"figure_source:{filename}"))
        report = {
            "status": "PASS",
            "authoritative_numerical_validation_completed": True,
            "authoritative_package": package.name,
            "authoritative_coverage": coverage,
            "matched_tables": [item["artifact"] for item in comparisons if str(item["artifact"]).startswith("table:")],
            "matched_figure_source_data": [item["artifact"] for item in comparisons if str(item["artifact"]).startswith("figure_source:")],
            "numeric_comparisons": comparisons,
            "input_sha256": file_sha256(package) if package.is_file() else None,
        }
    except Exception as error:
        report = {
            "status": "FAIL",
            "authoritative_numerical_validation_completed": True,
            "authoritative_package": package.name,
            "error": str(error),
            "matched_tables": [item["artifact"] for item in comparisons if str(item["artifact"]).startswith("table:")],
            "matched_figure_source_data": [item["artifact"] for item in comparisons if str(item["artifact"]).startswith("figure_source:")],
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise RuntimeError(f"Authoritative numerical validation failed: {error}") from error
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def validate_results(
    output: Path,
    input_paths: dict[str, Path] | None = None,
    authoritative_package: Path | None = None,
) -> dict[str, object]:
    """Verify input hashes when available, then perform external authoritative validation."""
    if input_paths and (output / "run_config.json").is_file():
        run = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
        for name, item in run.get("inputs", {}).items():
            if not input_paths[name].is_file() or file_sha256(input_paths[name]) != item["sha256"]:
                raise RuntimeError(f"Input provenance failure: {name}")
    return validate_authoritative(output, authoritative_package)
