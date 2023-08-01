import warnings

from scipy.optimize import least_squares


def least_squares_fit(x_values, y_values, function):
    num_params = function.__code__.co_argcount - 1  # Exclude the 'x' parameter
    initial_guess = [0.5] * num_params  # Initialize all parameters with 0.5

    def _residuals_log(params, x_values, y_values):
        model_values = function(x_values, *params)
        return y_values - model_values

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            result = least_squares(_residuals_log, initial_guess, loss='soft_l1', args=(x_values, y_values))
            params = result.x

        fitted_values = function(x_values, *params)
        residuals = y_values - fitted_values
        return residuals, params, fitted_values
    except RuntimeError as e:
        print(f"Failed to fit curve for function {function.__name__}. Error: {e}")
        return None
