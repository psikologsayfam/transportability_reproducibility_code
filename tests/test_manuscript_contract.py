import numpy as np
import pandas as pd
import pytest

from cmpb_transport.full_snapshot_experiment import mechanism
from cmpb_transport.snapshot_bootstrap_analysis import ensemble, metric_vector


def test_feature_group_definition_matches_manuscript():
    assert mechanism("antibiotic_flag")[0] == "clinical actions"
    assert mechanism("vasopressor_flag")[0] == "clinical actions"
    assert mechanism("antibiotic_count")[0] == "measurement and observation process"
    assert mechanism("heart_rate_count")[0] == "measurement and observation process"
    assert mechanism("heart_rate_mean")[0] == "physiology and laboratory state"
    assert mechanism("age")[0] == "demographic and administrative context"


def test_metric_vector_primary_metrics():
    y = np.array([0, 1, 0, 1, 0, 1])
    probability = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    result = metric_vector(y, probability, 0.5)
    assert result["auroc"] == pytest.approx(1.0)
    assert result["auprc"] == pytest.approx(1.0)
    assert result["brier"] == pytest.approx(np.mean((y - probability) ** 2))


def test_three_seed_ensemble_is_aligned_arithmetic_mean(tmp_path):
    files = []
    for seed, value in zip((17, 42, 2026), (0.2, 0.4, 0.6)):
        frame = pd.DataFrame(
            {
                "patient_id": ["a", "b"],
                "stay_id": ["a", "b"],
                "true_label": [0, 1],
                "predicted_probability": [value, value],
                "selected_threshold": [0.5, 0.5],
            }
        )
        path = tmp_path / f"seed{seed}.parquet"
        frame.to_parquet(path, index=False)
        files.append(path)
    result = ensemble(files)
    np.testing.assert_allclose(result.predicted_probability, 0.4)


def test_three_seed_ensemble_rejects_population_mismatch(tmp_path):
    files = []
    for index in range(3):
        frame = pd.DataFrame(
            {
                "patient_id": ["a" if index < 2 else "wrong", "b"],
                "stay_id": ["a", "b"],
                "true_label": [0, 1],
                "predicted_probability": [0.2, 0.8],
                "selected_threshold": [0.5, 0.5],
            }
        )
        path = tmp_path / f"seed{index}.parquet"
        frame.to_parquet(path, index=False)
        files.append(path)
    with pytest.raises(RuntimeError, match="do not align"):
        ensemble(files)
