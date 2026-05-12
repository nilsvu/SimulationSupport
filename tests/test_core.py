# Distributed under the MIT License.
# See LICENSE.txt for details.

"""
Unit tests for core.py

Run with:
    python -m unittest test_core.py -v
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd
import torch

# Use a non-interactive backend so the tests can run in CLI without making the plots
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from SimulationSupport.gpr import (
    load_gpr_checkpoint,
    predict_with_gpr_model,
    run_gpr_pipeline,
    save_gpr_checkpoint,
    train_gpr_model,
)

# Load testing data made from the first 25 rows of the q87d subset of the SXS catalog
# so the GPR sees realistic inputs
data = pd.read_csv(Path(__file__).parent / "test_data.csv")


def make_df():
    """Construct and return a DataFrame built from the first 25 rows of the q87d subset of the SXS catalog
    so the tests see a realistic data input."""
    return pd.DataFrame(data)


# Training function that is reused throughout the tests
def _train_test_model():
    """Train a small GPR model on the testing dataframe."""
    df = make_df()
    features = [
        "initial_separation",
        "mass_ratio",
        "S1x",
        "S1y",
        "S1z",
        "S2x",
        "S2y",
        "S2z",
    ]
    X = df[features].values
    Y = df["initial_orbital_frequency"].values
    model, likelihood = train_gpr_model(X, Y)
    return model, likelihood, features, X, Y


# Test GPRegressionModel
class TestGPROn8DData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.likelihood, cls.features, cls.X, cls.Y = (
            _train_test_model()
        )
        cls.Y_pred, cls.Y_std = predict_with_gpr_model(
            cls.X, cls.model, cls.likelihood
        )

    def test_prediction_shape(self):
        # 8D input should produce one prediction per row
        self.assertEqual(self.Y_pred.shape, self.Y.shape)
        self.assertEqual(self.Y_std.shape, self.Y.shape)

    def test_stds_non_negative_8d(self):
        # Uncertainties should be non-negative and nontrivial
        self.assertTrue(np.all(self.Y_std >= 0))
        self.assertTrue(np.any(self.Y_std > 1e-10), "All uncertainties are 0")

    def test_pred_quality(self):
        """
        Test that the GP returns arrays of the right shape, and also learns
        the training data.
        """
        corr = np.corrcoef(self.Y, self.Y_pred)[0, 1]
        self.assertGreater(
            corr,
            0.9,
            f"GP failed to learn the training data (corr={corr:.3f}). Either"
            " training diverged or the model is returning the prior mean.",
        )

    def test_ard_length_scales(self):
        # ARD should produce one length scale per input dimension.
        # If this breaks, kernel construction has regressed to a single shared scale.
        rbf_lengthscale = self.model.rbf_kernel.lengthscale
        self.assertEqual(
            rbf_lengthscale.shape[-1],
            len(self.features),
            "Expected one RBF length scale per feature "
            f"({len(self.features)}), got shape {rbf_lengthscale.shape}",
        )

    def test_normalize_input(self):
        """
        Test that input_mean and input_std have one entry per feature and
        that applying them to X produces zero-mean columns.
        """
        exp_shape = (self.X.shape[1],)
        self.assertEqual(self.model.input_mean.shape, exp_shape)
        self.assertEqual(self.model.input_std.shape, exp_shape)

        # Recreate normalized_X using the stored statistics
        normalized_X = (self.X - self.model.input_mean) / self.model.input_std
        col_means = normalized_X.mean(axis=0)
        col_stds = normalized_X.std(axis=0)

        self.assertTrue(
            np.allclose(col_means, 0, atol=1e-1),
            f"Expected means ~0, got {col_means}",
        )
        self.assertTrue(
            np.allclose(col_stds, 1.0, atol=1e-1),
            f"Expected stddevs ~1, got {col_stds}",
        )

    def test_denormalize_output(self):
        """
        Test that stored output_mean and output_std are correct and
        that denormalize_output is correct (normalize then denormalize should
        get back to the original Y values).
        """
        self.assertIsInstance(self.model.output_mean, (float, np.floating))
        self.assertIsInstance(self.model.output_std, (float, np.floating))
        self.assertGreater(self.model.output_std, 0.0)

        # Small subset of Y
        Y_subset = self.Y[:5]
        Y_normalized = (
            Y_subset - self.model.output_mean
        ) / self.model.output_std
        Y_denormalized = self.model.denormalize_output(Y_normalized)

        np.testing.assert_allclose(
            Y_denormalized,
            Y_subset,
            atol=1e-8,
            err_msg="Denormalization did not recover original Y values",
        )

    def test_run_gpr_pipeline(self):
        """
        Test that run_gpr_pipeline correctly carries out train + predict
        and returns outputs of the right shape. The underlying functions are
        tested separately above with real computation; here we simply verify
        the wiring by mocking them.
        """
        mock_pred = np.ones_like(self.Y)
        mock_std = np.ones_like(self.Y) * 0.1

        with (
            patch(
                "SimulationSupport.gpr.train_gpr_model",
                return_value=(self.model, self.likelihood),
            ),
            patch(
                "SimulationSupport.gpr.predict_with_gpr_model",
                return_value=(mock_pred, mock_std),
            ),
        ):
            model, likelihood, Y_pred = run_gpr_pipeline(
                self.X, self.Y, target_name="test", plot=False, silent=True
            )
            self.assertEqual(Y_pred.shape, self.Y.shape)
            self.assertIsNotNone(model)
            self.assertIsNotNone(likelihood)


# Test save_gpr_checkpoint and load_gpr_checkpoint together
class TestSaveAndLoadGprCheckpoint(unittest.TestCase):

    def setUp(self):
        """
        Train a model, save a checkpoint, and load it back.
        Individual tests inspect different aspects of both the saved
        file and the loaded model.
        """
        # Temporary directory
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.ckpt_path = str(Path(self.tmp_dir.name) / "test_model.pt")

        self.model, self.likelihood, self.features, self.X, self.Y = (
            _train_test_model()
        )
        self.run_col = "initial_orbital_frequency"
        self.base_column = "spec_pn_guess_omega"

        # Save once - all tests below read this file
        save_gpr_checkpoint(
            model=self.model,
            likelihood=self.likelihood,
            features=self.features,
            output_name="omega",
            run_col=self.run_col,
            base_column=self.base_column,
            X=self.X,
            Y=self.Y,
            path=self.ckpt_path,
        )
        self.loaded_model, self.loaded_likelihood, self.meta = (
            load_gpr_checkpoint(self.ckpt_path)
        )

    def tearDown(self):
        # Temporary directory
        self.tmp_dir.cleanup()
        plt.close("all")

    def test_file_creation(self):
        # Test that save_gpr_checkpoint actually creates a file on disk
        self.assertTrue(os.path.exists(self.ckpt_path))

    def test_checkpoint(self):
        # Test that the saved checkpoint contains all keys
        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        for key in (
            "model_state_dict",
            "likelihood_state_dict",
            "metadata",
            "normalization",
        ):
            self.assertIn(key, ckpt, f"Missing key: {key}")

    def test_metadata(self):
        # Test that metadata stores the correct input features, output name, and column names
        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        meta = ckpt["metadata"]
        self.assertEqual(meta["input_features"], self.features)
        self.assertEqual(meta["output_name"], "omega")
        self.assertEqual(meta["base_column"], [self.base_column])
        self.assertEqual(meta["target_definition"]["run_col"], self.run_col)
        self.assertEqual(
            meta["target_definition"]["base_column"], self.base_column
        )

    def test_normalization_stats_saved(self):
        # Test that all normalization statistics are present
        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        norm = ckpt["normalization"]
        for key in ("input_mean", "input_std", "output_mean", "output_std"):
            self.assertIn(key, norm, f"Missing normalization key: {key}")
            self.assertIsNotNone(norm[key])

    def test_training_data_saved(self):
        # Test that the raw training data is saved by the checkpoint
        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        self.assertIn("training_data", ckpt)
        np.testing.assert_allclose(
            ckpt["training_data"]["X"].numpy(), self.X, rtol=1e-6
        )

    def test_returns_objects(self):
        # Test that function returns (model, likelihood, meta)
        self.assertEqual(
            len((self.loaded_model, self.loaded_likelihood, self.meta)), 3
        )

    def test_model_in_eval_mode(self):
        # Test that model.training is False
        self.assertFalse(self.loaded_model.training)

    def test_likelihood_in_eval_mode(self):
        # Test lihelihood.training is False
        self.assertFalse(self.loaded_likelihood.training)

    def test_metadata_contains_input_features(self):
        # Test that metadata stores the feature list so users know what columns to pass
        self.assertIn("input_features", self.meta)

    def test_normalization_stats_loaded(self):
        # Test that all four normalization stats are restored  after loading
        self.assertIsNotNone(self.loaded_model.input_mean)
        self.assertIsNotNone(self.loaded_model.input_std)
        self.assertIsNotNone(self.loaded_model.output_mean)
        self.assertIsNotNone(self.loaded_model.output_std)

    def test_model_predicts_after_loading(self):
        # Test that the loaded model produces valid predictions
        df = make_df()
        X_test = df[self.meta["input_features"]].values[:3]
        preds, stds = predict_with_gpr_model(
            X_test, self.loaded_model, self.loaded_likelihood
        )
        self.assertEqual(len(preds), 3)
        self.assertTrue(np.all(stds >= 0))

    def test_predictions_match_original(self):
        """
        Test that the parameters from the loaded model match those from
        the original one. Tests whether save/load preserves the model state. If every tensor
        in the state_dict matches, the loaded model is
        the original model (then identical inputs give identical outputs).
        """

        orig_model = self.model.state_dict()
        loaded_model = self.loaded_model.state_dict()
        self.assertEqual(
            set(orig_model.keys()),
            set(loaded_model.keys()),
            "Model state_dict keys differ after save and load",
        )
        for key in orig_model:
            self.assertTrue(
                torch.allclose(orig_model[key], loaded_model[key]),
                f"Model parameter '{key}' changed after save and load",
            )

        orig_likelihood = self.likelihood.state_dict()
        loaded_likelihood = self.loaded_likelihood.state_dict()
        self.assertEqual(
            set(orig_likelihood.keys()),
            set(loaded_likelihood.keys()),
            "Likelihood state_dict keys differ after save and load",
        )
        for key in orig_likelihood:
            self.assertTrue(
                torch.allclose(orig_likelihood[key], loaded_likelihood[key]),
                f"Likelihood parameter '{key}' changed after save and load",
            )

    def test_predictions_match_non_training_points(self):
        """
        The original and reloaded models should
        produce the same predictions on inputs that are not the training data.
        """
        X_heldout = (
            self.X + 0.01
        )  # inputs that differ slightly from the training data to avoid GPyTorch's cached path (part of how GPyTorch works)

        preds_before, stds_before = predict_with_gpr_model(
            X_heldout, self.model, self.likelihood
        )
        preds_after, stds_after = predict_with_gpr_model(
            X_heldout, self.loaded_model, self.loaded_likelihood
        )

        np.testing.assert_allclose(
            preds_after,
            preds_before,
            rtol=1e-5,
            err_msg="Predictions changed after save and load",
        )
        np.testing.assert_allclose(
            stds_after,
            stds_before,
            rtol=1e-5,
            err_msg="Uncertainties changed after save and load",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
