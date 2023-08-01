import numpy as np
import statsmodels.api as sm

# TODO: reconcile the three below

def linear_fit_ols(x_values, y_values):
    log_x_values = np.log(x_values)
    log_y_values = np.log(y_values)

    X = sm.add_constant(log_x_values)
    ols_model = sm.OLS(log_y_values, X)
    ols_results = ols_model.fit()

    params = ols_results.params  # const, slope
    fitted_values = log_x_values*params[1]+params[0]
    residuals = log_y_values - fitted_values
    return residuals, params, fitted_values


def linear_fit_robust_regression(x_values, y_values):
    log_x_values = np.log(x_values)
    log_y_values = np.log(y_values)

    X = sm.add_constant(log_x_values)
    huber_model = sm.RLM(log_y_values, X, M=sm.robust.norms.HuberT())
    huber_results = huber_model.fit()

    params = huber_results.params  # const, slope
    fitted_values = log_x_values * params[1] + params[0]
    residuals = log_y_values - fitted_values
    return residuals, params, fitted_values


def linear_fit_generalised_regression(x_values, y_values):
    log_x_values = np.log(x_values)
    log_y_values = np.log(y_values)

    X = sm.add_constant(log_x_values)
    gls_model = sm.GLS(log_y_values, X)
    gls_results = gls_model.fit()

    params = gls_results.params  # const, slope
    fitted_values = log_x_values * params[1] + params[0]
    residuals = log_y_values - fitted_values
    return residuals, params, fitted_values

