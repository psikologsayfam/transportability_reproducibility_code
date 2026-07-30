"""Strict binary prediction metrics."""

from __future__ import annotations
import numpy as np


def validate(y: object, p: object) -> tuple[np.ndarray, np.ndarray]:
    """Validate paired binary labels and finite probabilities; positive class is one."""
    a = np.asarray(y, dtype=int)
    b = np.asarray(p, dtype=float)
    if (
        a.ndim != 1
        or len(a) != len(b)
        or len(a) == 0
        or not np.isin(a, [0, 1]).all()
        or len(np.unique(a)) != 2
    ):
        raise ValueError("Invalid or one-class labels")
    if not np.isfinite(b).all() or ((b < 0) | (b > 1)).any():
        raise ValueError("Invalid probabilities")
    return a, b


def auroc(y: object, p: object) -> float:
    """Return pairwise rank AUROC with half-credit for ties."""
    a, b = validate(y, p)
    pos = b[a == 1]
    neg = b[a == 0]
    return float((pos[:, None] > neg).mean() + 0.5 * (pos[:, None] == neg).mean())


def auprc(y: object, p: object) -> float:
    """Return non-interpolated average precision for positive class one."""
    a, b = validate(y, p)
    ranked = a[np.argsort(-b, kind="stable")]
    precision = np.cumsum(ranked) / np.arange(1, len(a) + 1)
    return float(precision[ranked == 1].mean())


def brier(y: object, p: object) -> float:
    """Return mean squared probability error."""
    a, b = validate(y, p)
    return float(np.mean((a - b) ** 2))


def ece(
    y: object, p: object, bins: int = 15, strategy: str = "equal_frequency"
) -> tuple[float, list[dict[str, float]]]:
    """Return expected calibration error and per-bin details using explicit binning."""
    a, b = validate(y, p)
    if strategy == "equal_frequency":
        groups = np.array_split(np.argsort(b, kind="stable"), bins)
    elif strategy == "equal_width":
        groups = [
            np.where((b >= i / bins) & ((b < (i + 1) / bins) if i < bins - 1 else (b <= 1)))[0]
            for i in range(bins)
        ]
    else:
        raise ValueError("Unknown ECE strategy")
    rows = []
    score = 0.0
    for i, g in enumerate(groups):
        if not len(g):
            continue
        observed = float(a[g].mean())
        predicted = float(b[g].mean())
        weight = len(g) / len(a)
        score += weight * abs(observed - predicted)
        rows.append(
            {"bin": i, "n": len(g), "observed": observed, "predicted": predicted, "weight": weight}
        )
    return float(score), rows


def threshold_metrics(y: object, p: object, threshold: float) -> dict[str, float]:
    """Return sensitivity, specificity, precision/PPV, recall, F1, and NPV at a fixed source threshold."""
    a, b = validate(y, p)
    pred = b >= threshold
    tp = int((pred & (a == 1)).sum())
    tn = int((~pred & (a == 0)).sum())
    fp = int((pred & (a == 0)).sum())
    fn = int((~pred & (a == 1)).sum())
    def ratio(n, d):
        return float(n / d) if d else float("nan")
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    return {
        "sensitivity": recall,
        "specificity": ratio(tn, tn + fp),
        "precision": precision,
        "recall": recall,
        "f1": ratio(2 * precision * recall, precision + recall),
        "positive_predictive_value": precision,
        "negative_predictive_value": ratio(tn, tn + fn),
    }
