# ATR2 chemical-factor validation

Run from the repository root with Python 3.9 or newer:

```bash
python analysis_scripts/validate_atr2_cf.py
```

Dependencies are NumPy, pandas, SciPy, and Matplotlib. TensorFlow, Keras and
MAST-ML are not needed for this analysis. The script reads the two CF functions
from the local `models.py` syntax tree without importing the whole model module.

The default inputs and output directory are repository-relative:

- Input: `data_files/atr2_cf_calibration_50_alloys.csv`.
- CF implementation: `models.py`, function `OWAY.atr2cf292`.
- Results and PNG/SVG figures: `analysis_outputs/atr2_cf_validation/`.

The input is a byte-for-byte copy of the 50-alloy calibration table used for the
verified November 11, 2026 analysis. Source details and its SHA256 are recorded
in the adjacent provenance JSON. Only measured `delta_sigma_y_MPa` and chemistry
are used in fitting. The legacy `ATR2 CF pred` column is retained in the source
table for traceability and is never used as a prediction or fitting target.

## Calculation

Full-fit results evaluate the current `OWAY.atr2cf292` implementation with all
existing coefficients unchanged. There is no full-dataset coefficient refit.
The Cu/P `max(0, ...)` rules and the code's Cu saturation value are retained.

CV uses five folds repeated five times, with seed 42: 25 fits, each trained on
40 alloys and tested on 10. All nine named coefficients are exposed to
`scipy.optimize.least_squares`, including the Ni reference and separate Mn and
Si coefficients. Every fold starts from the existing production coefficients.
Threshold constants remain fixed; no coefficient bounds, Mn/Si ratio constraint
or regularizer is imposed. This analysis does not change `models.py` or anchors.

The nine coefficients contain a redundant Ni-reference parameter. All nine are
optimized, while the predictions have eight independent coefficient
combinations. Each fit's training objective and test predictions are verified
against the global least-squares solution in the equivalent eight-term space.
That independent calculation is a check, not the CV estimator. The analytic
optimizer Jacobian is also checked against finite differences.

Both full-fit evaluation and CV use the nominal 292 C base CF condition.
Original CSV temperature labels are preserved, without an additional
temperature correction in this calibration validation. Production anchor
generation is a separate calculation that applies its query-temperature shift.

Each measured and predicted DSY value is separately converted with
`TTS = 0.00067*DSY**2 + 0.49*DSY`, before calculating TTS residuals. There is no
TTS regression, parity calibration, or propagation of DSY summary errors.

Residuals are prediction minus measurement. Residual SD uses `ddof=0`.
Primary CV statistics pool all 250 held-out predictions, five per alloy.
The bar chart's CV error bars are the population SD of the 25 split MAEs.
Mean split RMSE and pooled RMSE are separately exported and should not be mixed.
All parity figures place measured/true values on x and predictions on y.

## Verified reference results

With the supplied data, current production coefficients, and default seed:

| Evaluation | Quantity | ME | MAE | RMSE | Residual SD |
| --- | --- | ---: | ---: | ---: | ---: |
| Full fit | DSY, MPa | 0.266221 | 13.322338 | 15.754419 | 15.752169 |
| Full fit | TTS, C | -0.015962 | 10.899059 | 13.064195 | 13.064186 |
| Pooled CV | DSY, MPa | -0.789889 | 15.890882 | 19.063058 | 19.046686 |
| Pooled CV | TTS, C | -0.844369 | 13.017001 | 15.873190 | 15.850716 |

CV MAE across splits is 15.890882 +/- 3.306963 MPa and
13.017001 +/- 2.784173 C. These values supersede the older notebook CF
validation, which omitted the nonnegative Cu/P threshold expressions.

## Outputs and options

The output directory includes per-point full-fit and out-of-fold predictions,
all nine fitted coefficients per split, fold assignments, optimizer checks,
pooled and per-split errors, the executed parameterized function, source hashes,
and settings. Figure 2 combines full-fit TTS parity with full-fit/CV MAE bars.
Additional figures show DSY and TTS parity for both full fit and CV.

```bash
python analysis_scripts/validate_atr2_cf.py --output /path/to/results
python analysis_scripts/validate_atr2_cf.py --no-plots
python analysis_scripts/plot_atr2_cf_validation.py --output /path/to/results
```

`--data`, `--models-file`, and `--seed` can override the defaults. The plotted
data and numerical results are unchanged by re-running the plotting script.
The old notebooks remain historical records; use this script for the corrected
current-code CF validation.
