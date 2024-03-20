import warnings
import numpy as np
import pandas as pd
from typing import List, Optional, Callable, Union, Tuple

from scipy import stats
from scipy.optimize import least_squares


# TODO: add NN gradient based method as alternative to least squares estimator
def lse(x_values: List[float],
        y_values: List[float],
        function: Callable,
        reflect_y: bool,
        bounds: List[float],
) -> Union[None, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Fits a nonlinear function (curve) to the data using ny method of
    least squares  estimation (LSE) with with bounds on the variables.

    Parameters:
        x_values (List[float]): The independent variable values.
        y_values (List[float]): The dependent variable values.
        function (Callable): The function to fit.
        bounds (Any): 2-tuple of array_like or `Bounds`, optional.
        reflect_y (bool, optional): If True, reflects the scaling function along the x-axis.

    Returns:
        np.ndarray: The residuals.
        np.ndarray: The optimized parameters.
        np.ndarray: The model prediction values.

    Notes:
        When the random errors are assumed to have normal distributions with the same variance,
        LSE and MLE lead to the same optimal coefficients.
    """
    num_params = function.__code__.co_argcount - 1  # Exclude the 'x' parameter
    initial_guess = [0.1] * num_params

    def _residuals(params: np.ndarray, x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
        """Get residuals"""
        model_predictions = function(x_values, *params)
        return y_values - model_predictions

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            result = least_squares(
                _residuals,
                initial_guess,
                args=(x_values, y_values),
                bounds=bounds,
                loss="soft_l1",
            )

            params = result.x

        predicted_values = function(x_values, *params)
        residuals = y_values - predicted_values
        return residuals, params, predicted_values
    except RuntimeError as e:
        print(f"Failed to fit curve for function {function.__name__}. Error: {e}")
        return None