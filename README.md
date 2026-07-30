## Included manuscript analyses

- MIMIC-III, MIMIC-IV, and eICU snapshots at 1, 6, 12, and 24 hours.
- Patient-grouped 70/15/15 source partitions with split seed 20260713.
- Logistic regression, LightGBM, XGBoost, CatBoost, FT-Transformer, and TabM with
  training seeds 17, 42, and 2026.
- Source-validation Platt and isotonic calibration, applied unchanged to source-test
  and external predictions.
- Three unordered database pairs, five feature views, logistic-regression and
  LightGBM domain classifiers, balanced membership, and patient-grouped 70/30 splits.
- Source-test grouped permutation reliance for all six mortality models.
- Patient bootstrap comparisons against logistic regression and independently
  resampled internal-to-target degradation intervals, with 2,000 percentile
  replicates.
- The recorded three-seed ensemble rule: strict patient/stay/label alignment followed
  by arithmetic averaging. Calibrated probabilities are averaged after seed-specific
  source-validation fitting and target application.
- Six main-manuscript tables and five main-manuscript figures only.
- Fail-closed cell-by-cell checking against the bundled authoritative output package.

## Data placement

The three prepared snapshot CSV files are not redistributed. Place them in a local
directory using the original filenames:

- `mimic 3 1h 24h windows.csv`
- `mimic 4 windows 1_24.csv`
- `eicu dataset 1h-24h windows.csv`

These files remain governed by their original MIMIC/eICU access and data-use terms.

## Environment

Python 3.12 and an NVIDIA CUDA environment matching the recorded experiment are
recommended for numerical reproduction. Install the package and pinned dependencies:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .
```

Install the CUDA-specific PyTorch wheel appropriate for the host separately if the
default wheel does not provide CUDA. The recorded experiment used PyTorch 2.11.0 with
CUDA 12.8. GPU execution is part of the archived model configuration; changing device
or library versions can change stochastic model results.
