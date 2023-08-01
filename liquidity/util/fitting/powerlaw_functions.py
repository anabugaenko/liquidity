import numpy as np

from liquidity.util.fitting.functions import exponential_function, logarithmic_function, periodic_function, \
    linear_function
from liquidity.util.fitting.non_linear_fits import least_squares_fit


# pure power-law function
def pure_powerlaw(x, alpha, C):
    return C * x ** alpha

def powerlaw_with_exp_svf(x, alpha, beta, lambda_):
    return x ** (-alpha) * exponential_function(x, beta, lambda_)

def powerlaw_with_log_svf(x, alpha, beta, lambda_):
    return x ** (-alpha) * logarithmic_function(x, beta, lambda_)

def powerlaw_with_lin_svf(x, alpha, beta, lambda_):
    return x ** (-alpha) * linear_function(x, beta, lambda_)

def powerlaw_with_per_svf(x, alpha, a0, A, omega):
    return x ** (-alpha) * periodic_function(x, a0, A, omega)

def find_x_min_index(y_values, x_values, function):
    results_dict = {}
    for x_min in range(len(x_values) - 3):
        data = y_values[x_min:]
        if len(data) == 0:  # Skip when data becomes empty
            continue
        x_adjusted = x_values[:len(data)]  # Adjust lags to match the size of data
        residuals, params, _ = least_squares_fit(x_adjusted, data, function)
        sse = np.sum(residuals**2)
        bic = len(data) * np.log(sse / len(data)) + len(params) * np.log(len(data))
        results_dict[x_min] = bic
    min_bic = min(list(results_dict.values()))
    return list(results_dict.keys())[list(results_dict.values()).index(min_bic)]
