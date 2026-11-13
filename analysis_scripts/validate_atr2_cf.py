#!/usr/bin/env python3
"""Current-code full-fit evaluation and literal nine-coefficient repeated CV.

Full-fit predictions execute OWAY.atr2cf292 unchanged. CV executes an AST copy
of that function with its nine coefficient values exposed as free optimizer
arguments. Cu/P thresholds remain fixed; all nine coefficients are optimized.
No seven-parameter Mn/Si constraint or fixed Ni reference is imposed in CV.
"""
from pathlib import Path
import argparse
import ast
import copy
import hashlib
import json
import platform
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import scipy

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data_files/atr2_cf_calibration_50_alloys.csv'
CODE = ROOT / 'models.py'
OUTPUT = ROOT / 'analysis_outputs/atr2_cf_validation'
NAMES = ['A_intercept', 'B_Cu', 'C_Cu_Ni', 'D_Ni', 'E_Ni_reference',
         'F_Mn', 'G_Si', 'H_P', 'I_Cu_P']
PARAM_ASSIGNMENTS = {'A2CF_A': 0, 'A2CF_B': 1, 'A2CF_C': 2,
                     'A2CF_D': 3, 'A2CF_E': 5, 'A2CF_F': 6,
                     'A2CF_G': 7, 'A2CF_H': 8}
SEED = 42


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_functions():
    tree = ast.parse(CODE.read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'OWAY')
    node = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == 'atr2cf292')
    original = copy.deepcopy(node)
    original.name = 'original_cf'
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[original], type_ignores=[])), str(CODE), 'exec'), namespace)
    initial = np.empty(9)
    for statement in node.body:
        if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name):
            name = statement.targets[0].id
            if name in PARAM_ASSIGNMENTS:
                initial[PARAM_ASSIGNMENTS[name]] = ast.literal_eval(statement.value)
    initial[4] = .75

    class ExposeCoefficients(ast.NodeTransformer):
        def visit_Assign(self, statement):
            if isinstance(statement.targets[0], ast.Name) and statement.targets[0].id in PARAM_ASSIGNMENTS:
                statement.value = ast.Subscript(value=ast.Name(id='parameters', ctx=ast.Load()),
                    slice=ast.Constant(PARAM_ASSIGNMENTS[statement.targets[0].id]), ctx=ast.Load())
            return self.generic_visit(statement)

        def visit_BinOp(self, expression):
            if (isinstance(expression.op, ast.Sub) and isinstance(expression.left, ast.Name)
                    and expression.left.id == 'ni' and isinstance(expression.right, ast.Constant)
                    and expression.right.value == .75):
                expression.right = ast.Subscript(value=ast.Name(id='parameters', ctx=ast.Load()),
                    slice=ast.Constant(4), ctx=ast.Load())
            return self.generic_visit(expression)

    parameterized = ExposeCoefficients().visit(copy.deepcopy(node))
    parameterized.name = 'parameterized_cf'
    parameterized.args.args.append(ast.arg(arg='parameters'))
    module = ast.fix_missing_locations(ast.Module(body=[parameterized], type_ignores=[]))
    exec(compile(module, str(CODE), 'exec'), namespace)
    return namespace['original_cf'], namespace['parameterized_cf'], initial, ast.unparse(module)


def convert(x):
    x = np.asarray(x, dtype=float)
    return .00067*x**2 + .49*x


def summarize(y, prediction):
    error = np.asarray(prediction)-np.asarray(y)
    result = dict(n=len(error), ME=float(error.mean()), MAE=float(abs(error).mean()),
        RMSE=float(np.sqrt(np.mean(error**2))), residual_SD=float(error.std(ddof=0)))
    np.testing.assert_allclose(result['RMSE']**2, result['ME']**2 + result['residual_SD']**2, atol=1e-10)
    return result


def frame(data, indices, prediction, method, **meta):
    measured = data.delta_sigma_y_MPa.to_numpy(float)[indices]
    out = pd.DataFrame(dict(source_row=indices, alloy=data.alloy.iloc[indices].to_numpy(),
        measured_dsy_MPa=measured, predicted_dsy_MPa=prediction,
        measured_TTS_degC=convert(measured), predicted_TTS_degC=convert(prediction),
        method=method, **meta))
    out['residual_dsy_MPa'] = prediction-measured
    out['residual_TTS_degC'] = out.predicted_TTS_degC-out.measured_TTS_degC
    return out


def frame_metrics(points):
    return [dict(quantity=q, unit=u, **summarize(points[m], points[p])) for q,u,m,p in [
        ('DSY', 'MPa', 'measured_dsy_MPa', 'predicted_dsy_MPa'),
        ('TTS', '°C', 'measured_TTS_degC', 'predicted_TTS_degC')]]


def main():
    global SOURCE, CODE, OUTPUT, SEED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=SOURCE,
                        help='CSV containing the 50 measured ATR2 calibration alloys.')
    parser.add_argument('--models-file', type=Path, default=CODE,
                        help='Production models.py whose OWAY.atr2cf292 function is evaluated.')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    SOURCE, CODE, OUTPUT, SEED = args.data.resolve(), args.models_file.resolve(), args.output.resolve(), args.seed
    OUTPUT.mkdir(parents=True, exist_ok=True)
    original_hash = sha(CODE)
    data = pd.read_csv(SOURCE)
    assert len(data) == data.alloy.nunique() == 50
    chemistry = data[['wt_percent_Cu','wt_percent_Ni','wt_percent_Mn','wt_percent_Si','wt_percent_P']].to_numpy(float)
    y = data.delta_sigma_y_MPa.to_numpy(float)
    assert np.isfinite(chemistry).all() and np.isfinite(y).all()
    original, parameterized, initial, parameterized_source = source_functions()

    def predict(parameters, rows):
        return np.array([parameterized(None, *row, parameters) for row in chemistry[rows]])

    all_rows = np.arange(50)
    fixed = np.array([original(None, *row) for row in chemistry])
    np.testing.assert_array_equal(fixed, predict(initial, all_rows))
    full = frame(data, all_rows, fixed, 'Full fit: unchanged code coefficients')

    cu, ni, mn, si, ph = chemistry.T
    q = np.maximum(0, np.minimum(cu, .239)-.04)
    p = np.maximum(0, ph-.004)

    def jacobian(parameters, rows):
        A,B,C,D,E,F,G,H,I = parameters
        return np.column_stack([np.ones(len(rows)), q[rows], q[rows]*(ni[rows]-E),
            ni[rows]-E, -(C*q[rows]+D), mn[rows], si[rows],
            (1-I*q[rows])*p[rows], -H*q[rows]*p[rows]])

    # Independent eight-term calculation is a verification only. It never
    # supplies initialization, coefficients or predictions to the nine-variable optimizer.
    independent_X = np.column_stack([np.ones(50), q, q*ni, ni, mn, si, p, q*p])
    random = np.random.RandomState(SEED)
    coefficients, split_metrics, oof_parts, assignments, checks = [], [], [], [], []
    for repeat in range(1, 6):
        permutation = np.arange(50)
        random.shuffle(permutation)
        for fold in range(1, 6):
            split = 5*(repeat-1)+fold
            test = np.sort(permutation[10*(fold-1):10*fold])
            train = np.setdiff1d(all_rows, test)
            assert len(train) == 40 and len(test) == 10 and not set(train)&set(test)
            result = least_squares(lambda parameters: predict(parameters, train)-y[train],
                initial.copy(), jac=lambda parameters: jacobian(parameters, train),
                method='trf', x_scale=[100,500,500,100,1,25,25,1500,4],
                ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=10000)
            if not result.success:
                raise RuntimeError(f'Optimizer failed in split {split}: {result.message}')
            predicted = predict(result.x, test)
            # Independent objective and held-out prediction check against the
            # global unbounded least-squares solution in the same function space.
            scale = independent_X[train,1:].std(axis=0)
            center = independent_X[train,1:].mean(axis=0)
            Z_train = np.column_stack([np.ones(40), (independent_X[train,1:]-center)/scale])
            Z_test = np.column_stack([np.ones(10), (independent_X[test,1:]-center)/scale])
            reference = np.linalg.lstsq(Z_train, y[train], rcond=None)[0]
            assert np.linalg.matrix_rank(Z_train) == 8
            max_difference = float(abs(predicted-Z_test@reference).max())
            np.testing.assert_allclose(predicted, Z_test@reference, rtol=0, atol=2e-5)
            np.testing.assert_allclose(result.fun@result.fun,
                np.sum((Z_train@reference-y[train])**2), rtol=1e-10, atol=1e-7)
            # Check the analytic derivative independently at the solution.
            numerical_jac = np.empty((40,9))
            for j in range(9):
                step = 1e-5*max(1, abs(result.x[j]))
                plus, minus = result.x.copy(), result.x.copy()
                plus[j] += step; minus[j] -= step
                numerical_jac[:,j] = (predict(plus,train)-predict(minus,train))/(2*step)
            np.testing.assert_allclose(jacobian(result.x,train), numerical_jac, rtol=1e-6, atol=1e-6)
            one = frame(data, test, predicted, 'Repeated 5-fold CV: nine coefficients optimized',
                split=split, repeat=repeat, fold=fold)
            oof_parts.append(one)
            split_metrics.extend([dict(split=split, repeat=repeat, fold=fold, **m) for m in frame_metrics(one)])
            coefficients.append(dict(split=split, repeat=repeat, fold=fold,
                **dict(zip(NAMES,map(float,result.x))),
                optimizer_success=bool(result.success), optimizer_status=int(result.status),
                function_evaluations=int(result.nfev), training_SSE=float(result.fun@result.fun),
                final_jacobian_rank=int(np.linalg.matrix_rank(jacobian(result.x,train)))))
            checks.append(dict(split=split, independent_prediction_max_difference_MPa=max_difference))
            assignments.extend([dict(split=split, repeat=repeat, fold=fold, source_row=int(i),
                alloy=data.alloy.iloc[i], role=role)
                for role, rows in [('train',train),('test',test)] for i in rows])

    oof = pd.concat(oof_parts, ignore_index=True)
    assert len(oof) == 250 and oof.groupby('alloy').size().eq(5).all()
    assert not oof.duplicated(['repeat','alloy']).any()
    stats = pd.DataFrame([dict(method=method, **m) for method, points in
        [('Full fit: unchanged code coefficients',full),
         ('Repeated five-fold CV (pooled)',oof)] for m in frame_metrics(points)])
    splits = pd.DataFrame(split_metrics)
    summaries = pd.DataFrame([dict(quantity=quantity, unit=group.unit.iloc[0], metric=metric,
        mean_over_25_splits=float(group[metric].mean()), SD_over_25_splits=float(group[metric].std(ddof=0)))
        for quantity, group in splits.groupby('quantity',sort=False)
        for metric in ['ME','MAE','RMSE','residual_SD']])
    bars = pd.DataFrame([dict(method='Full fit',quantity=r.quantity,MAE=r.MAE,SD_across_25_splits=0)
        for r in stats[stats.method.eq('Full fit: unchanged code coefficients')].itertuples()] +
        [dict(method='5-fold CV (5 repeats)',quantity=r.quantity,MAE=r.mean_over_25_splits,
              SD_across_25_splits=r.SD_over_25_splits) for r in summaries[summaries.metric.eq('MAE')].itertuples()])
    for name, table in [('calibration_input_50_alloys',data),('fullfit_predictions',full),
        ('cv_out_of_fold_predictions',oof),('error_statistics',stats),('cv_split_metrics',splits),
        ('cv_mean_and_SD_over_25_splits',summaries),('cv_fit_all_nine_coefficients',pd.DataFrame(coefficients)),
        ('cv_fold_assignments',pd.DataFrame(assignments)),('optimization_verification',pd.DataFrame(checks)),
        ('figure2_bar_values',bars)]:
        table.to_csv(OUTPUT/f'{name}.csv',index=False)
    assert sha(CODE) == original_hash
    settings = dict(source_file=str(SOURCE),source_SHA256=sha(SOURCE),current_code=str(CODE),
        current_code_SHA256=original_hash,function='OWAY.atr2cf292',
        full_fit='Execute unchanged code with every existing coefficient fixed; no optimization',
        coefficient_order=NAMES,unchanged_code_coefficients=dict(zip(NAMES,map(float,initial))),
        CV='Optimize all nine named coefficients independently in each training fold using scipy.optimize.least_squares',
        CV_starting_coefficients='Unchanged production code coefficients, same start in every split',
        CV_bounds='Unbounded',fixed_Cu_min=.04,fixed_Cu_max=.239,fixed_P_min=.004,
        Cu_rule='max(0,min(Cu,0.239)-0.04)',P_rule='max(0,P-0.004)',
        Ni_reference='Free in CV; fixed at unchanged value 0.75 for full fit',
        Mn_and_Si='Separate free CV coefficients; their ratio is not fixed',
        temperature='Base-condition CF at nominal 292 C; no extra temperature transformation',
        unique_alloys=50,n_splits=5,n_repeats=5,total_fits=25,train_size=40,test_size=10,
        random_state=SEED,partitions='Repeated shuffled five-fold CV; same seed-42 partitions as the verified analysis',out_of_fold_predictions=250,
        conversion='TTS=0.00067*DSY**2+0.49*DSY applied separately to every measured and predicted value',
        residual='Predicted minus measured',residual_SD_ddof=0,
        MAE_error_bars='Population SD of 25 split MAEs; not residual SD or standard error',
        coefficient_identifiability='All 9 are optimized, but Ni-reference redundancy leaves Jacobian rank 8; coefficients are not unique',
        software=dict(Python=platform.python_version(),NumPy=np.__version__,SciPy=scipy.__version__))
    settings['parity_plot_axes'] = 'Measured / true on x; predicted on y'
    (OUTPUT/'settings.json').write_text(json.dumps(settings,indent=2))
    (OUTPUT/'parameterized_current_code_function.py').write_text(parameterized_source+'\n')
    (OUTPUT/'README.txt').write_text('''ATR2 CF: unchanged current-code full fit and all-nine-coefficient CV

Full fit evaluates the current OWAY.atr2cf292 function without changing any
coefficient. There is no optimization of all 50 alloys. The original code file
is read only and its SHA256 is verified unchanged after execution.

CV repeats five-fold validation five times (25 fits, 40 train / 10 test each).
Every fit optimizes all nine named coefficients, including the Ni reference
and separate Mn and Si coefficients. Cu/P cutoffs stay fixed at current-code
values. The nine-variable nonlinear optimizer directly evaluates an AST copy
of the production function with its nine coefficients exposed as arguments.
No Mn:Si constraint, fixed Ni reference, parameter bounds or regularizer is
applied. All starts use the production coefficients and no test observations.

The Ni-reference parameter is algebraically redundant with other coefficients.
Consequently the nine fitted numbers are not uniquely identifiable (Jacobian
rank 8). Every split's fitted predictions and objective were checked against
an independent global OLS solution in the equivalent eight-term function
space. This independent fit is verification only, never the CV estimator.

Primary CV metrics pool 250 out-of-fold predictions; each alloy is tested once
in each repeat. CV MAE error bars show the population SD across 25 split MAEs.
Residual SD is separately calculated with ddof=0.

Measured and predicted DSY values are individually converted using
TTS=0.00067*DSY**2+0.49*DSY. TTS metrics are calculated from the resulting
individual residuals; no error propagation or TTS regression is performed.
Full-fit and CV both use base-condition CF at nominal 292 C, without an extra
temperature adjustment. Raw CSV temperature labels are retained in the input.

This run supersedes the seven-coefficient constrained analysis for Figure 2.
Full-fit coefficients and the production code remain unchanged. Numerical
inputs, nine fitted coefficients per fold, fold assignments, point predictions,
statistics, optimizer diagnostics and settings are exported for reproducibility.
''')
    print(stats.to_string(index=False))
    print(summaries[summaries.metric.eq('MAE')].to_string(index=False))
    print('Largest independent prediction check difference:',max(x['independent_prediction_max_difference_MPa'] for x in checks))
    if not args.no_plots:
        from plot_atr2_cf_validation import main as draw_figures
        draw_figures(OUTPUT)
    print(OUTPUT)


if __name__ == '__main__':
    main()
