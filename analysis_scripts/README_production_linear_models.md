# Production OWAY, ML-OWAY, and Naive Linear

The production implementations in `models.py` now use the rules behind the
current embrittlement and life-extension figures. Complete earlier versions
are retained as commented reference blocks beside their replacements.

| Model | Production fit |
| --- | --- |
| `NaiveLinear` | Unconstrained TTS versus log10(fluence), using every supplied measured row. At least two distinct measured fluences are required. |
| `OWAY` | Unweighted TTS versus linear fluence, using all measured points strictly above 3e19 n/cm2 plus the ATR2 anchor at 1.38e20. Flux independent. |
| `MLOWAY` / `JOWAY` | Unweighted TTS versus linear fluence, using the Jacobs26 anchor at 4e19, flux-matched measured points strictly above 3e19, and ATR2 at 1.38e20. |

Production includes the highest measured point. Naive Linear has no fluence
threshold, slope constraint, clipping, anchors, or fallback. Pass
`fit_log_scale=False` only when explicitly requesting raw-fluence OLS.

OWAY returns unavailable/NaN when measured high-fluence points are missing;
it no longer silently invokes EONY. The CF coefficients, Cu/P clipping,
temperature adjustment, and quadratic DSY-to-TTS conversion are unchanged.

ML-OWAY uses the flux supplied in `flux_n_cm2_sec=` or in each query row. Without
either, it uses the highest-fluence database point's flux; an ambiguous flux
requires an explicit value. With no database measurements, query chemistry,
temperature and flux are required. The measured/query flux ratio must be within
[1/1.25, 1.25], inclusive. Raw and log flux/fluence features for the Jacobs26
anchor are updated together. Batch prediction evaluates the neural anchor once
for the query dataframe, rather than rebuilding the ensemble for every point.
The ATR2 CF endpoint depends on chemistry and temperature, without a flux term.

`JOWAY` remains compatible with existing imports and prediction column names.
`MLOWAY` supplies an explicit name for the updated model; both behave identically.
`InformedLinear` retains the older heuristic linear model separately.

```python
from RPV_model_benchmarking.models import OWAY, MLOWAY, NaiveLinear

# queries: requested chemistry, temperature, fluence, and flux; include alloy
# identity when querying a named database alloy.
oway_predictions, oway_details = OWAY().predict(queries)
ml_predictions, ml_details = MLOWAY().predict(queries, flux_n_cm2_sec=3.68e12)

# Reuse an already-loaded database with database=database_rows.
fit = OWAY().fit(one_alloy_rows)
predictions = fit.predict([6e19, 1e20])

naive_prediction, naive_details = NaiveLinear().predict(
    one_alloy_rows, alloy='HB2-W1', fluence=1e20)
```

## Evaluation is explicit

Use `remove_highest_fluence=True` to exclude every replicate at the maximum
measured fluence. `include_threshold=True` includes equality at 3e19 for the
measured-data selection. The displayed test-cohort cutoff is a separate choice
made by the evaluation driver; it does not automatically change the fitting
threshold. Naive Linear continues to fit all remaining measurements.

For ML-OWAY, measured-point holdout does not retrain the neural network. Supply
an appropriately trained `predictor=` for an evaluation requiring neural-network
training exclusions. This production update does not regenerate or reinterpret
the saved historical evaluation results; their saved anchors and recorded
measurement-selection protocol remain the source for those figures.

`fit()` returns coefficients, anchor values, selected measured rows, and their
selection audit. Batch `predict()` exposes its fit objects through `model.fits_`.
Identity matching prefers the supplied alloy name. Without a name, exact
Cu/Ni/Mn/P/Si matching is used; measurements from multiple alloy identities are
not combined into one fit.

Run the focused behavior checks without loading neural-network weights:

```bash
python analysis_scripts/test_production_linear.py
```

The installation was additionally checked against 1,248 saved production
scenarios (624 alloys at two fluxes), all 624 Naive Linear evaluation records,
and real Jacobs26 anchors for the four example alloys at both fluxes.
