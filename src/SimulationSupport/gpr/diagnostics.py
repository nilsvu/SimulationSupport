# Distributed under the MIT License.
# See LICENSE.txt for details.

"""
Gaussian Process Regression machine learning diagnostic functions library.
Contains all the functions necessary to validate and plot the GPR model used
to predict better low-eccentricity orbital parameter initial guesses.
"""

import logging
import multiprocessing as mp
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from SimulationSupport.gpr import predict_with_gpr_model, train_gpr_model

logger = logging.getLogger(__name__)


# Leave-one-out parallelization set up
def _loo_single(i, X, Y):
    """
    Run a single LOO iteration for index i.

    Trains a GPR on all points except index i, then predicts the held-out point.
    Called in parallel by loo_crossval.

    Args:
        i           (int): Index of the held-out point
        X           (np.ndarray): Input features, with shape (N, D)
        Y           (np.ndarray): Target variable, with shape (N, )

    Returns:
        pred_mean   (float): Predicted mean for the held-out point
        pred_std    (float): Predicted std dev for the held-out point
    """
    N = len(Y)

    # Create train and test split
    # Boolean mask: all True except index i (held out point)
    train_mask = np.ones(N, dtype=bool)
    train_mask[i] = False

    X_train = X[train_mask]
    Y_train = Y[train_mask]
    # Slice preserves the 2D shape expected by the model
    X_test = X[i : i + 1]

    # Train and predict
    model_loo, likelihood_loo = train_gpr_model(X_train, Y_train)
    pred_mean, pred_std = predict_with_gpr_model(
        X_test, model_loo, likelihood_loo
    )

    return pred_mean[0], pred_std[0]


# Leave-one-out cross-validation
def loo_crossval(
    X: np.ndarray,
    Y: np.ndarray,
    target_name="Target",
    n_jobs=None,
):
    """
    Perform Leave-One-Out Cross-Validation for a GPR model.

    Trains N models (each omitting one point), predicts the held-out point,
    collect predictions and uncertainties, and then computes and plots summary metrics.
    This gives an unbiased estimate of generalization performance.

    Args:
        X                   (np.ndarray): Input features, with shape (N, D)
        Y                   (np.ndarray): Target variable, with shape (N, )
        target_name         (str): Label for plots and print outputs
        n_jobs              (int): Number of parallel workers (None uses all available cores)

    Returns:
        predictions_loo     (np.ndarray): LOO predicted values, with shape (N, )
        uncertainties_loo   (np.ndarray): LOO predicted std devs, with shape (N, )
    """
    N = len(Y)

    # Force single worker when GPU is used as parallel processes can't share a GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        n_jobs = 1

    logger.info(
        f"Processing {N} LOO iterations for {target_name} using"
        f" {n_jobs or os.cpu_count()} parallel workers..."
    )

    # Build argument tuples for each LOO fold
    # because pool.starmap needs one tuple of positional args per call
    args = [(i, X, Y) for i in range(N)]

    # Each worker is a new Python process and reimports the modules it needs
    # Avoids deadlock
    ctx = mp.get_context("spawn")
    if n_jobs == 1:
        # Run sequentially when there is a single worker
        results = [_loo_single(*a) for a in args]
    else:
        # Run LOO folds in parallel with multiple workers
        with ctx.Pool(processes=n_jobs) as pool:
            results = pool.starmap(_loo_single, args)

    # Unpack results back into the predictions and uncertainties arrays
    predictions_loo, uncertainties_loo = np.array(results).T

    return predictions_loo, uncertainties_loo


def plot_loo_crossval(Y, predictions_loo, target_name="Target", plot=False):
    """
    Compute summary statistics for Leave-One-Out Cross-Validation results, and optionally
    plot the correlation.

    Args:
        Y               (np.ndarray): True target values, with shape (N, )
        predictions_loo (np.ndarray): LOO predicted values, with shape (N, )
        target_name     (str): Label for plots and log outputs
        plot            (bool): Whether to produce a correlation plot. Default is False.

    Returns:
        rmse_loo        (float): Root mean squared error of the LOO predictions
        mae_loo         (float): Mean absolute error of the LOO predictions
        r_squared_loo   (float): R^2 computed from the Pearson correlation
    """

    Y_loo = Y  # Same as the original Y for the multi input case

    # Calculate metrics - always computed, regardless of whether plot is requested
    # R^2 is computed from the Pearson correlation coefficient -
    # equivalent to the coefficient of determination for a linear fit through the origin
    correlation = np.corrcoef(Y_loo, predictions_loo)[0, 1]
    r_squared_loo = correlation**2

    # Metrics with goal values
    rmse_loo = np.sqrt(np.mean((Y_loo - predictions_loo) ** 2))
    mae_loo = np.mean(np.abs(Y_loo - predictions_loo))
    y_range = Y_loo.max() - Y_loo.min()

    # Plot correlation
    if plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(Y_loo, predictions_loo, alpha=0.6, s=20)

        # y = x reference line: perfect predictions would lie exactly on this line
        min_val = min(Y_loo.min(), predictions_loo.min())
        max_val = max(Y_loo.max(), predictions_loo.max())
        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            lw=2,
            label="Perfect Correlation",
        )

        # Labels and formatting
        plt.xlabel(f"True Δ{target_name}", fontsize=12)
        plt.ylabel(f"LOO Predicted Δ{target_name}", fontsize=12)
        plt.title(f"LOO: GPR Predictions vs True ({target_name})", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Display R^2
        plt.text(
            0.95,
            0.95,
            f"R² = {r_squared_loo:.4f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            horizontalalignment="right",
        )
        plt.tight_layout()
        plt.show()

    # Log metrics with goal values for quick analysis
    logger.info(f"Leave-one-out Cross Validation Results ({target_name})")
    logger.info(
        f"RMSE: {rmse_loo:.6f} ({100 * rmse_loo / y_range:.6f}% of target"
        " range; goal: < 1 % of target range, lower is better)"
    )
    logger.info(
        f"MAE: {mae_loo:.6f} ({100 * mae_loo / y_range:.6f}% of target range;"
        " goal: < 1 % of target range, lower is better)"
    )
    logger.info(
        f"R²: {r_squared_loo:.4f} (goal: > 0.95 excellent, > 0.90 good, < 0.70"
        " poor)"
    )
    logger.info(f"\n Dataset size: {len(Y_loo)} points")
    logger.info(
        f"Each model is trained on {len(Y_loo)-1} points, and tested on 1 point"
    )
    logger.info("This provides an unbiased generalization estimate.")

    return rmse_loo, mae_loo, r_squared_loo


# Leave-one-out residual computation and plotting
def plot_loo_residuals(Y_loo, predictions_loo, target_name="Target", show=True):
    """
    Calculate LOO prediction residuals, plot a histogram, and print summary statistics.

    Args:
        Y_loo           (np.ndarray): True target values from LOO cross validation
        predictions_loo (np.ndarray): Predicted values from LOO cross validation
        target_name     (str): Name of target variable
        show            (bool): Whether to produce a residual plot. Default is True.

    Returns:
        residuals_loo   (np.ndarray): Per-point residuals (true - predicted), with shape (N,)

    """

    # Compute residuals: LOO prediction error
    # Residual = true - predicted
    residuals_loo = Y_loo - predictions_loo

    # Make histogram
    plt.figure(figsize=(8, 5))
    plt.hist(residuals_loo, bins=20, color="skyblue", edgecolor="k", alpha=0.8)
    # Vertical line at zero: residuals centered here indicate no systematic bias;
    # ideally the histogram is centered on this line; a shifted distribution suggests
    # the model is over- or under-predicting
    plt.axvline(0, color="r", linestyle="--", label="Zero Error")

    plt.title(f"LOO Residuals Histogram for {target_name}")
    plt.xlabel(" Residuals", fontsize=16)
    plt.ylabel("Count", fontsize=16)
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14)
    plt.tight_layout()
    if show:
        plt.show()

    # Print statistics
    logger.info(f"Residual statistics for {target_name}:")
    logger.info(f"Mean residual:          {np.mean(residuals_loo):.4e}")
    logger.info(f"Std of residuals:       {np.std (residuals_loo):.4e}")
    logger.info(f"Max residual:           {np.max (residuals_loo):.4e}")
    logger.info(f"Min residual:           {np.min (residuals_loo):.4e}")

    return residuals_loo
