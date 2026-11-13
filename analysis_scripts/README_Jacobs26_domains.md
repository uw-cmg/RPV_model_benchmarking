# Jacobs26 domain predictions

The downloaded Jacobs26 implementation is retained, including its MADML
domain model and Keras unpacking helper. The domain model is located relative
to the package directory rather than the shell's working directory. The
SciKeras unpacker is restored in a `finally` block after loading.

```python
from RPV_model_benchmarking.models import EnsembleNN_Jacobs26

predictions, details = EnsembleNN_Jacobs26().predict(
    queries, return_ebars=True, return_domains=True)
domain_d = details['Jacobs26 NN ensemble domain d']
```

The public return value remains `(predictions, dataframe)`. The dataframe also
contains the calibrated ensemble error bars in degrees C. Domain `d` and the
error bars are separate quantities. Prediction values are unchanged by
requesting domains. Python 3.11 or newer is needed for the supplied serialized
domain model. The saved domain file and neural-network weights are unchanged.

The current notebook includes a single-query domain example, production
predictions for the surveillance database, current embrittlement curves,
name-matched 81-alloy fleet forecasts, and the fixed-coefficient/full-nine-CV
ATR2 validation. Older replaced cell contents remain as comments. Optional
saved historical benchmarking plots are disabled by default and can be
enabled with `RUN_SAVED_BENCHMARK_PLOTS=True`.

Before installation, the combined production/domain code was run through
the complete `models.py` module with the actual TensorFlow ensemble. It was
checked against 330,538 saved production curve points, 243 fleet predictions,
1,872 independent Naive Linear production calculations, 5,632 direct Jacobs26
curve checks, and 19 previously saved domain/error-bar predictions.
