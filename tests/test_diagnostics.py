# Distributed under the MIT License.
# See LICENSE.txt for details.

"""
Unit tests for diagnostics.py

Run with:
    python -m unittest test_diagnostics.py -v
"""

import os
import unittest

import matplotlib
import numpy as np
import pandas as pd

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data.csv")

# Use a non-interactive backend so the tests can run in CLI without plot outputs
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from SimulationSupport.gpr.diagnostics import (
    loo_crossval,
    plot_loo_crossval,
    plot_loo_residuals,
)


def make_df():
    """
    Load test data from a CSV file containing the first 25 rows of the q87d subset
    of the SXS catalog, so the tests use a realistic data input.
    """
    return pd.read_csv(TEST_DATA_PATH)


# Single class with one test function for each diagnostics function
class TestDiagnostics(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    # Test loo_crossval
    def test_loo_crossval(self):
        df = make_df()
        input_columns = [
            "initial_separation",
            "mass_ratio",
            "S1x",
            "S1y",
            "S1z",
            "S2x",
            "S2y",
            "S2z",
        ]
        X = df[input_columns].values
        Y = df["initial_orbital_frequency"].values

        preds, uncertainties = loo_crossval(X, Y, target_name="omega")

        # Test that the function returns (predictions, uncertainties)
        result = (preds, uncertainties)
        self.assertEqual(len(result), 2)

        # Test that the outputs are arrays
        self.assertIsInstance(preds, np.ndarray)
        self.assertIsInstance(uncertainties, np.ndarray)

        # Test that there is one prediction per data point
        self.assertEqual(preds.shape, Y.shape)

        # Test that the GP predictive uncertainties are non-negative
        self.assertTrue(np.all(uncertainties >= 0))

    # Test plot_loo_crossval
    def test_plot_loo_crossval(self):
        Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        preds = np.array([1.1, 1.9, 3.1, 3.8, 5.2])

        # Call plot_loo_crossval once and reuse the result for each test
        rmse, mae, r2 = plot_loo_crossval(Y, preds)

        # Test that the function returns (rmse, mae, r2)
        result = (rmse, mae, r2)
        self.assertEqual(len(result), 3)

        # Test that outputs are floats
        for scalar in result:
            self.assertIsInstance(float(scalar), float)

        # Test that R^2 is a squared correlation and lies in [0,1]
        self.assertGreaterEqual(r2, 0.0)
        self.assertLessEqual(r2, 1.0)

        # Test that perfect predictions give R^2 = 1, RMSE = 0 = MAE
        rmse_perf, mae_perf, r2_perf = plot_loo_crossval(Y, Y.copy())
        self.assertAlmostEqual(r2_perf, 1.0, places=5)
        self.assertAlmostEqual(rmse_perf, 0.0, places=10)
        self.assertAlmostEqual(mae_perf, 0.0, places=10)

    # Test plot_loo_residuals
    # No GPR calls; show = False is passed in all tests to suppress plots
    def test_plot_loo_residuals(self):
        Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        preds = np.array([1.1, 1.9, 3.1, 3.8, 5.2])

        # Test that residual = true - predicted for each point
        residuals = plot_loo_residuals(Y, preds, show=False)
        np.testing.assert_allclose(residuals, Y - preds, rtol=1e-10)

        # Test that the output shape matches the input arrays
        Y_shape = np.array([1.0, 2.0, 3.0])
        preds_shape = np.array([1.0, 2.0, 3.0])
        residuals_shape = plot_loo_residuals(Y_shape, preds_shape, show=False)
        self.assertEqual(residuals_shape.shape, Y_shape.shape)

        # Test that if the predictions match the truth exactly, all the residuals are 0
        Y_perf = np.array([1.0, 2.0, 3.0, 4.0])
        preds_perf = Y_perf.copy()
        residuals_perf = plot_loo_residuals(Y_perf, preds_perf, show=False)
        np.testing.assert_allclose(residuals_perf, 0.0, atol=1e-10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
