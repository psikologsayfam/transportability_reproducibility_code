"""Discovery restricted to explicitly admitted raw-data roots."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd
from .reproducibility import sha256

IDENTIFIER_TOKENS = (
    "subject_id",
    "patient_id",
    "uniquepid",
    "hadm_id",
    "admission_id",
    "stay_id",
    "patientunitstayid",
    "patienthealthsystemstayid",
)
TIME_TOKENS = ("time", "offset", "date")
OUTCOME_TOKENS = ("mortality", "death", "deathtime", "hospital_expire")


def classify_database(name: str) -> str:
    """Classify a filename as MIMIC-III, MIMIC-IV, eICU, or unknown."""
    value = name.lower()
    if "eicu" in value:
        return "eICU"
    if "mimic 3" in value or "mimiciii" in value:
        return "MIMIC-III"
    if "mimic 4" in value or "mimiciv" in value:
        return "MIMIC-IV"
    return "unknown"


def inspect_file(path: Path) -> dict[str, Any]:
    """Inspect one admitted CSV without accessing any old result directory.

    Returns path, dimensions, hash, schema candidates, and temporal-provenance flags.
    Fails for unsupported files or unreadable headers.
    """
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Unsupported discovery format: {path}")
    frame = pd.read_csv(path, low_memory=False)
    columns = list(frame.columns)
    lower = {c: c.lower() for c in columns}
    identifiers = [c for c, v in lower.items() if v in IDENTIFIER_TOKENS or v.endswith("_id")]
    timestamps = [c for c, v in lower.items() if any(token in v for token in TIME_TOKENS)]
    outcomes = [c for c, v in lower.items() if any(token in v for token in OUTCOME_TOKENS)]
    metadata = {
        "absolute_path": str(path.resolve()),
        "database": classify_database(path.name),
        "table_name": path.stem,
        "file_format": "csv",
        "file_size_bytes": path.stat().st_size,
        "row_count": len(frame),
        "column_count": len(columns),
        "sha256": sha256(path),
        "identifier_columns": ";".join(identifiers),
        "timestamp_columns": ";".join(timestamps),
        "candidate_outcome_fields": ";".join(outcomes),
        "candidate_feature_fields": ";".join(
            c for c in columns if c not in identifiers + timestamps + outcomes
        ),
        "is_windowed_snapshot": "window_hours" in columns,
        "has_measurement_level_timestamps": any(
            c.lower()
            not in {"intime", "outtime", "window_endtime", "window_minutes", "unitdischargeoffset"}
            for c in timestamps
        ),
        "has_source_table_provenance": all(
            any(term in c.lower() for c in columns)
            for term in ("source_table", "source_column", "unit")
        ),
    }
    return metadata


def discover_files(roots: list[Path], ignored: set[str]) -> pd.DataFrame:
    """Discover only admitted roots, excluding prohibited old-output directories."""
    rows = []
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.rglob("*.csv")):
            if any(part.lower() in ignored for part in path.parts):
                continue
            rows.append(inspect_file(path))
    if not rows:
        raise RuntimeError("No admissible raw or minimally processed files found")
    return pd.DataFrame(rows)


def assess_readiness(inventory: pd.DataFrame) -> tuple[bool, list[str]]:
    """Assess whether all databases support raw temporal cohort reconstruction."""
    reasons = []
    expected = {"MIMIC-III", "MIMIC-IV", "eICU"}
    found = set(inventory.database)
    if expected - found:
        reasons.append(f"Missing databases: {sorted(expected - found)}")
    for _, row in inventory.iterrows():
        if bool(row.is_windowed_snapshot) and not bool(row.has_measurement_level_timestamps):
            reasons.append(f"{row.database}: windowed snapshots lack measurement-level timestamps")
        if not bool(row.has_source_table_provenance):
            reasons.append(f"{row.database}: feature source-table/unit provenance unavailable")
    return not reasons, reasons
