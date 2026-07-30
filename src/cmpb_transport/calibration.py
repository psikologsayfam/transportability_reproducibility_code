"""Calibration regression and ECE summaries."""

from __future__ import annotations
import numpy as np
from .metrics import validate, brier, ece


def intercept_slope(y: object, p: object) -> tuple[float, float]:
    """Fit y ~ intercept + slope*logit(p) using Newton-Raphson; fail at exact 0/1 probabilities."""
    a, b = validate(y, p)
    if ((b <= 0) | (b >= 1)).any():
        raise ValueError("Calibration regression requires probabilities in (0,1)")
    x = np.column_stack([np.ones(len(a)), np.log(b / (1 - b))])
    beta = np.zeros(2)
    for _ in range(100):
        fitted = 1 / (1 + np.exp(-np.clip(x @ beta, -35, 35)))
        weights = fitted * (1 - fitted)
        step = np.linalg.solve(x.T @ (weights[:, None] * x), x.T @ (a - fitted))
        beta += step
        if abs(step).max() < 1e-9:
            return float(beta[0]), float(beta[1])
    raise RuntimeError("Calibration model failed to converge")


def summarize(y: object, p: object) -> dict[str, object]:
    """Return primary 15-bin equal-frequency calibration metrics."""
    value, detail = ece(y, p)
    intercept, slope = intercept_slope(y, p)
    return {
        "brier": brier(y, p),
        "ece": value,
        "intercept": intercept,
        "slope": slope,
        "bins": detail,
    }
