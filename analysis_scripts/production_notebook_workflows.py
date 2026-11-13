"""Current production and explicit holdout workflows shared by the notebook."""
import numpy as np
import pandas as pd


def prepare_query_features(rows):
    rows = rows.copy()
    for name in ['fluence_n_cm2', 'flux_n_cm2_sec']:
        if name not in rows:
            rows[name] = 10**pd.to_numeric(rows['log('+name+')'])
        rows[name] = pd.to_numeric(rows[name])
        if not np.isfinite(rows[name]).all() or not rows[name].gt(0).all():
            raise ValueError('Query fluence and flux must be finite and positive.')
    rows['log(fluence_n_cm2)'] = np.log10(rows.fluence_n_cm2)
    rows['log(flux_n_cm2_sec)'] = np.log10(rows.flux_n_cm2_sec)
    rows['Time'] = rows.fluence_n_cm2 / rows.flux_n_cm2_sec
    rows['log(Time)'] = np.log10(rows.Time)
    return rows


def forecast_fleet(database=None, flux=3e10, scenarios=None):
    """Forecast US surveillance alloys eligible for production OWAY, by name."""
    from RPV_model_benchmarking.data import DataLoader
    from RPV_model_benchmarking.models import OWAY, MLOWAY
    if database is None:
        database = DataLoader().load_rpv_data()
    if scenarios is None:
        scenarios = {'60-year': 2e19, '80-year': 2.7e19, 'max': 1e20}
    usa = database[database.datatype.eq('Plotter') & database.Nation.eq('USA')].copy()
    maximum = usa.groupby('alloy').fluence_n_cm2.max()
    eligible_names = maximum.index[maximum.gt(3e19)]
    representative = usa.groupby('alloy', sort=True).first().loc[eligible_names].copy()
    representative['temperature_C'] = usa.groupby('alloy').temperature_C.mean().reindex(representative.index)
    queries = []
    for scenario, fluence in scenarios.items():
        block = representative.reset_index()
        block['scenario'] = scenario
        block['fluence_n_cm2'] = fluence
        block['flux_n_cm2_sec'] = flux
        queries.append(block)
    queries = prepare_query_features(pd.concat(queries, ignore_index=True))
    oway, details = OWAY().predict(queries, database=database)
    ml, ml_details = MLOWAY().predict(queries, database=database)
    result = queries[['alloy', 'scenario', 'temperature_C', 'fluence_n_cm2', 'flux_n_cm2_sec']].copy()
    result['OWAY_TTS_degC'] = oway
    result['ML_OWAY_TTS_degC'] = ml
    result['ML_OWAY_minus_OWAY_degC'] = ml - oway
    result['OWAY_measured_fit_rows'] = details['OWAY measured fit rows'].to_numpy()
    result['ML_OWAY_measured_fit_rows'] = ml_details['ML-OWAY measured fit rows'].to_numpy()
    assert np.isfinite(result[['OWAY_TTS_degC', 'ML_OWAY_TTS_degC']]).all().all()
    return result


def summarize_fleet(result):
    records = []
    for scenario, group in result.groupby('scenario', sort=False):
        difference = group.ML_OWAY_minus_OWAY_degC.to_numpy()
        records.append(dict(scenario=scenario, alloys=group.alloy.nunique(),
            OWAY_mean_degC=float(group.OWAY_TTS_degC.mean()),
            OWAY_SD_degC=float(group.OWAY_TTS_degC.std(ddof=0)),
            ML_OWAY_mean_degC=float(group.ML_OWAY_TTS_degC.mean()),
            ML_OWAY_SD_degC=float(group.ML_OWAY_TTS_degC.std(ddof=0)),
            mean_ML_minus_OWAY_degC=float(difference.mean()),
            SD_ML_minus_OWAY_degC=float(difference.std())))
    return pd.DataFrame(records)


def plot_fleet_comparison(result):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8.4, 8.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=.05, wspace=.05)
    ax = fig.add_subplot(grid[1, 0])
    top = fig.add_subplot(grid[0, 0], sharex=ax)
    right = fig.add_subplot(grid[1, 1], sharey=ax)
    bins = np.arange(-30, 281, 5)
    for scenario, color in [('60-year', 'blue'), ('80-year', 'green'), ('max', 'red')]:
        group = result[result.scenario.eq(scenario)]
        x, y = group.ML_OWAY_TTS_degC, group.OWAY_TTS_degC
        ax.scatter(x, y, color=color, alpha=.5, s=25, label=scenario)
        top.hist(x, bins=bins, color=color, alpha=.5, edgecolor='black')
        right.hist(y, bins=bins, color=color, alpha=.5, edgecolor='black', orientation='horizontal')
    ax.plot([0, 260], [0, 260], 'k--')
    ax.set_xlabel('Predicted TTS (ML-OWAY) (°C)', fontsize=14)
    ax.set_ylabel('Predicted TTS (OWAY) (°C)', fontsize=14)
    ax.legend(fontsize=12)
    plt.setp(top.get_xticklabels(), visible=False)
    plt.setp(right.get_yticklabels(), visible=False)
    return fig


def plot_embrittlement_example(alloy='HB2-W1', database=None, points=100):
    """Production curves for one named alloy, at database and ATR2 flux."""
    import matplotlib.pyplot as plt
    from RPV_model_benchmarking.data import DataLoader
    from RPV_model_benchmarking.models import OWAY, MLOWAY, NaiveLinear, EnsembleNN_Jacobs26
    if database is None:
        database = DataLoader().load_rpv_data()
    rows = database[database.datatype.eq('Plotter') & database.alloy.eq(alloy)].copy()
    if rows.empty:
        raise ValueError('No surveillance measurements found for ' + str(alloy))
    highest = rows[rows.fluence_n_cm2.eq(rows.fluence_n_cm2.max())]
    if highest.flux_n_cm2_sec.nunique() != 1:
        raise ValueError('Highest-fluence measurements have different fluxes.')
    grid = np.linspace(1e17, max(2e20, float(rows.fluence_n_cm2.max())*1.05), points)
    queries = []
    for label, flux in [('database', float(highest.flux_n_cm2_sec.iloc[0])), ('high', 3.68e12)]:
        for _, profile in highest.iterrows():
            block = pd.DataFrame([profile.to_dict()]*points)
            block['fluence_n_cm2'] = grid
            block['flux_n_cm2_sec'] = flux
            block['scenario'] = label
            queries.append(block)
    queries = prepare_query_features(pd.concat(queries, ignore_index=True))
    queries['Jacobs26'], _ = EnsembleNN_Jacobs26().predict(queries.copy())
    queries['ML-OWAY'], _ = MLOWAY().predict(queries.copy(), database=database)
    queries['OWAY'], _ = OWAY().predict(queries.copy(), database=database)
    naive = np.array([NaiveLinear().predict(rows, alloy, f)[0] for f in grid])
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, style in [('database', '-'), ('high', '--')]:
        subset = queries[queries.scenario.eq(label)].groupby('fluence_n_cm2')[['Jacobs26', 'ML-OWAY', 'OWAY']].mean()
        for model, color in [('Jacobs26', 'red'), ('ML-OWAY', 'purple')]:
            ax.plot(subset.index/1e20, subset[model], style, color=color,
                    label=model + (' (database flux)' if label == 'database' else ' (3.68e12 flux)'))
        if label == 'database':
            ax.plot(subset.index/1e20, subset.OWAY, color='green', label='OWAY (flux independent)')
    ax.plot(grid/1e20, naive, color='blue', label='Naive Linear (log fluence)')
    ax.scatter(rows.fluence_n_cm2/1e20, rows['Measured DT41J  [C]'], color='black', label='Measured TTS', zorder=5)
    ax.set(xlabel='Fluence (10²⁰ n/cm²)', ylabel='TTS (°C)', title=alloy)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig, queries


def evaluate_remaining_points(model_name='NaiveLinear', database=None, predictor=None):
    """Hold out each alloy's maximum; fixed inclusive measured threshold is 3e19.

    Naive Linear uses every remaining measurement. ML-OWAY needs an explicitly
    supplied predictor with the intended training exclusions; this function
    neither silently substitutes full-fit anchors nor launches NN training.
    """
    from RPV_model_benchmarking.data import DataLoader
    from RPV_model_benchmarking.models import OWAY, MLOWAY, NaiveLinear
    if database is None:
        database = DataLoader().load_rpv_data()
    if model_name in ['ML-OWAY', 'MLOWAY', 'JOWAY'] and predictor is None:
        raise ValueError('Supply the evaluation predictor explicitly; full-fit anchors are production predictions.')
    if model_name not in ['NaiveLinear', 'OWAY', 'ML-OWAY', 'MLOWAY', 'JOWAY']:
        raise ValueError('Use NaiveLinear, OWAY, or ML-OWAY; NN retraining is a separate workflow.')
    records = []
    surveillance = database[database.datatype.eq('Plotter')]
    for name, rows in surveillance.groupby('alloy', sort=True):
        maximum = float(rows.fluence_n_cm2.max())
        tests = rows[rows.fluence_n_cm2.eq(maximum)]
        if model_name == 'NaiveLinear':
            prediction, _ = NaiveLinear().predict(rows, name, maximum, remove_highest_fluence=True)
        else:
            model = OWAY() if model_name == 'OWAY' else MLOWAY()
            options = dict(database=database, remove_highest_fluence=True,
                           fluence_threshold=3e19, include_threshold=True)
            if model_name != 'OWAY':
                options['predictor'] = predictor
            predictions, _ = model.predict(tests, **options)
            prediction = float(np.mean(predictions))
        records.append(dict(alloy=name, highest_fluence_n_cm2=maximum,
            measured_TTS_degC=float(tests['Measured DT41J  [C]'].mean()),
            predicted_TTS_degC=prediction, available=bool(np.isfinite(prediction))))
    return pd.DataFrame(records)
