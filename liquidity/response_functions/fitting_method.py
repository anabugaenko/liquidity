import warnings
from typing import List, Callable, Union, Tuple

import numpy as np
from scipy.optimize import minimize, least_squares

from liquidity.util.goodness_of_fit import loglikelihoods


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
) -> Union[Tuple[None, None], Tuple[np.ndarray, float]]:
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
    initial_guess = [1] * num_params  # Initialize all parameters with 0.5 for least_squares

    def _residuals(params: np.ndarray, x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
        model_predictions = function(x_values, *params)
        return y_values - model_predictions

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            result = least_squares(_residuals, initial_guess, args=(x_values, y_values), loss="soft_l1")
            params = result.x

        fitted_values = function(x_values, *params)
        mape = np.mean(np.abs((y_values - fitted_values) / y_values)) * 100

        return params, mape
    except RuntimeError as e:
        print(f"Failed to fit curve for function {function.__name__}. Error: {e}")
        return None, None
