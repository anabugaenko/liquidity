import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize, least_squares

from liquidity.response_functions.price_response_functions import compute_conditional_aggregate_impact
from typing import List, Callable, Union, Tuple

from liquidity.util.goodness_of_fit import loglikelihoods


def bin_data_into_quantiles(df, x_col="vol_imbalance", y_col="R", q=100, duplicates="raise"):
    """
    3/9/23 This is majorly used.
    """
    binned_x = pd.qcut(df[x_col], q=q, labels=False, retbins=True, duplicates=duplicates)
    binned_x = binned_x[0]
    df["x_bin"] = binned_x

    y_binned = df.groupby(["x_bin"])[y_col].mean()
    y_binned.index = y_binned.index.astype(int)

    x_binned = df.groupby(["x_bin"])[x_col].mean()
    x_binned.index = x_binned.index.astype(int)

    if "T" in df.columns:
        r_binned = df.groupby(["x_bin"])["T"].first()
        r_binned.index = r_binned.index.astype(int)
    else:
        r_binned = None

    return pd.concat([x_binned, r_binned, y_binned], axis=1).reset_index(drop=True)


def get_agg_features(df: pd.DataFrame, durations: List[int]) -> pd.DataFrame:
    df["event_timestamp"] = df["event_timestamp"].apply(lambda x: pd.Timestamp(x))
    df["date"] = df["event_timestamp"].apply(lambda x: x.date())
    results_ = []
    for i, T in enumerate(durations):
        lag_data = compute_conditional_aggregate_impact(df, T=T, normalise=True)
        lag_data["R"] = lag_data[f"R{T}"]
        lag_data = lag_data.drop(columns=f"R{T}")
        lag_data["T"] = T
        results_.append(lag_data)

    return pd.concat(results_)


def mle_fit(
    x_values: List[float], y_values: List[float], function: Callable
) -> Union[None, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Fits a function or curve to the data using the maximum likelihood estimation (MLE) method.

    Parameters:
    x_values (List[float]): The independent variable values.
    y_values (List[float]): The dependent variable values.
    function (Callable): The function to fit.

    Returns:
    np.ndarray: The residuals.
    np.ndarray: The optimized parameters.
    np.ndarray: The fitted values.
    """
    num_params = function.__code__.co_argcount - 1  # Exclude the 'x' parameter
    initial_guess = [0] * num_params  # Initialize all parameters with 0.1 for MLE

    # Compute negative loglikelihood
    def _negative_loglikelihood(params: np.ndarray, x_values: np.ndarray, y_values: np.ndarray) -> float:
        model_predictions = function(x_values, *params)
        residuals = y_values - model_predictions

        loglikelihood_values = loglikelihoods(residuals)

        return -np.sum(loglikelihood_values)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            result = minimize(_negative_loglikelihood, initial_guess, args=(x_values, y_values), method="Nelder-Mead")
            params = result.x

        fitted_values = function(x_values, *params)
        residuals = y_values - fitted_values
        return residuals, params, fitted_values
    except Exception as e:
        print(f"Failed to fit curve for function {function.__name__}. Error: {e}")
        return None


def least_squares_fit(
    x_values: List[float], y_values: List[float], function: Callable
) -> Union[None, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Fits a function or curve to the data using the least squares method.

    Parameters:
    x_values (List[float]): The independent variable values.
    y_values (List[float]): The dependent variable values.
    function (Callable): The function to fit.

    Returns:
    np.ndarray: The residuals.
    np.ndarray: The optimized parameters.
    np.ndarray: The fitted values.

    """
    num_params = function.__code__.co_argcount - 1  # Exclude the 'x' parameter
    initial_guess = [0.5] * num_params  # Initialize all parameters with 0.5 for least_squares

    def _residuals(params: np.ndarray, x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
        model_predictions = function(x_values, *params)
        return y_values - model_predictions

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            result = least_squares(_residuals, initial_guess, args=(x_values, y_values), loss="soft_l1")
            params = result.x

        fitted_values = function(x_values, *params)
        residuals = y_values - fitted_values
        return residuals, params, fitted_values
    except RuntimeError as e:
        print(f"Failed to fit curve for function {function.__name__}. Error: {e}")
        return None
