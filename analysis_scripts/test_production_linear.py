"""Behavior checks for production linear models without loading neural networks.

Run: python analysis_scripts/test_production_linear.py
The exact four class bodies are loaded from models.py without its unrelated
TensorFlow/MAST-ML imports. A small injected predictor tests the ML-OWAY API.
"""
from pathlib import Path
import ast
import sys
import types
import unittest
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


class Predictor:
    def __init__(self):
        self.calls = 0
        self.inputs = []

    def predict(self, data):
        self.calls += 1
        self.inputs.append(data.copy())
        return (80 - 5*np.log10(data.flux_n_cm2_sec.to_numpy(float)/1e10)), data


class ProductionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        package = types.ModuleType('RPV_model_benchmarking')
        package.__path__ = [str(REPO)]
        sys.modules['RPV_model_benchmarking'] = package
        namespace = dict(np=np, pd=pd, __name__='RPV_model_benchmarking.models')
        tree = ast.parse((REPO/'models.py').read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.ClassDef)
                 and n.name in ['NaiveLinear', 'OWAY', 'JOWAY', 'MLOWAY']]
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                     str(REPO/'models.py'), 'exec'), namespace)
        cls.Naive, cls.OWAY, cls.ML = [namespace[n] for n in ['NaiveLinear', 'OWAY', 'MLOWAY']]

    def setUp(self):
        self.data = pd.DataFrame({
            'alloy': ['example']*5, 'datatype': ['Plotter']*5,
            'fluence_n_cm2': [1e19, 3e19, 6e19, 9e19, 9e19],
            'flux_n_cm2_sec': [1e10]*5, 'Measured DT41J  [C]': [10, 30, 60, 90, 100],
            'temperature_C': [290]*5, 'wt_percent_Cu': [.1]*5,
            'wt_percent_Ni': [.6]*5, 'wt_percent_Mn': [1.2]*5,
            'wt_percent_Si': [.2]*5, 'wt_percent_P': [.01]*5, 'wt_percent_C': [.15]*5})
        self.query = self.data.iloc[[0]].copy()
        self.query['fluence_n_cm2'] = 1e20

    def test_naive_log_default_uses_all_rows(self):
        model = self.Naive()
        prediction, _ = model.predict(self.data, 'example', 1e20)
        x = np.log10(self.data.fluence_n_cm2.to_numpy()/1e20)
        expected = np.linalg.lstsq(np.column_stack([x, np.ones(len(x))]),
                                  self.data['Measured DT41J  [C]'], rcond=None)[0][1]
        self.assertAlmostEqual(prediction, expected)
        self.assertEqual(model.training_row_count_, 5)

    def test_naive_holdout_removes_all_maximum_replicates(self):
        model = self.Naive()
        model.predict(self.data, 'example', 9e19, remove_highest_fluence=True)
        self.assertEqual(model.training_row_indices_, [0, 1, 2])

    def test_naive_no_fallback_or_slope_clipping(self):
        small = self.data.iloc[[0, 0]]
        self.assertTrue(np.isnan(self.Naive().predict(small, 'example', 1e20)[0]))
        negative = self.data.iloc[:3].copy()
        negative['Measured DT41J  [C]'] = [100, 50, 10]
        model = self.Naive()
        model.predict(negative, 'example', 1e20)
        self.assertLess(model.slope_degC_per_log10_fluence_, 0)
        negative.iloc[0, negative.columns.get_loc('fluence_n_cm2')] = 0
        with self.assertRaises(ValueError):
            model.predict(negative, 'example', 1e20)

    def test_oway_production_includes_highest_and_is_flux_independent(self):
        model = self.OWAY()
        _, info = model.predict(self.query, database=self.data)
        self.assertEqual(model.fits_[0].measured_fit_rows.index.tolist(), [2, 3, 4])
        self.assertFalse(info['OWAY EONY model used'].iloc[0])
        changed = self.data.copy()
        changed['flux_n_cm2_sec'] = 1e14
        other = model.fit(changed, query_profiles=self.query)
        self.assertAlmostEqual(other.predict(1e20), model.fits_[0].predict(1e20))

    def test_oway_inclusive_evaluation_and_unavailable(self):
        fit = self.OWAY().fit(self.data, query_profiles=self.query,
            remove_highest_fluence=True, include_threshold=True)
        self.assertEqual(fit.measured_fit_rows.index.tolist(), [1, 2])
        missing = self.query.copy()
        missing['alloy'] = 'unknown'
        pred, info = self.OWAY().predict(missing, database=self.data)
        self.assertTrue(np.isnan(pred[0]))
        self.assertFalse(info['OWAY available'].iloc[0])

    def test_ml_anchors_at_requested_flux_and_one_batch_call(self):
        predictor = Predictor()
        query = pd.concat([self.query]*3, ignore_index=True)
        model = self.ML()
        model.predict(query, database=self.data, flux_n_cm2_sec=3.68e12, predictor=predictor)
        self.assertEqual(predictor.calls, 1)
        self.assertTrue(predictor.inputs[0].fluence_n_cm2.eq(4e19).all())
        self.assertTrue(predictor.inputs[0].flux_n_cm2_sec.eq(3.68e12).all())
        for fit in model.fits_:
            self.assertEqual(len(fit.measured_fit_rows), 0)
            np.testing.assert_allclose(fit.predict([4e19, 1.38e20]),
                                      [fit.jacobs_anchor_tts_degC, fit.atr2_tts_degC], atol=1e-9)

    def test_ml_flux_boundary_and_default_database_flux(self):
        model = self.ML()
        fit = model.fit(self.data, flux_n_cm2_sec=1.25e10, predictor=Predictor())
        self.assertEqual(len(fit.measured_fit_rows), 3)
        fit = model.fit(self.data, flux_n_cm2_sec=1.25001e10, predictor=Predictor())
        self.assertEqual(len(fit.measured_fit_rows), 0)
        missing_flux = self.query.drop(columns=['flux_n_cm2_sec'])
        _, info = model.predict(missing_flux, database=self.data, predictor=Predictor())
        self.assertEqual(info['ML-OWAY flux source'].iloc[0], 'highest_fluence_database_point')


if __name__ == '__main__':
    unittest.main()
