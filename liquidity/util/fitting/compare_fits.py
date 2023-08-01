import numpy as np
import pandas as pd
from scipy.optimize import least_squares, curve_fit
from scipy.stats import laplace, norm, t, expon, lognorm, powerlaw, gumbel_l, gumbel_r, kstest
from scipy.special import erfc
from liquidity.util.fitting.powerlaw_functions import powerlaw_with_log_svf, powerlaw_with_exp_svf, \
    powerlaw_with_per_svf, find_x_min_index
from liquidity.util.fitting.goodness_of_fit import compute_bic, compute_goodness_of_fit, compute_loglikelihood


def compare_robust_regression_fits(series, max_lag, stock_name):
    rr_results = {}

    def _residuals_log(params, acf_model, k_values, autocorr_values):
        gamma, a, b = params
        model_values = acf_model(k_values, gamma, a, b)
        return autocorr_values - model_values

    def _get_goodness_of_fit(residuals, y_values, num_params, n=1001):
        ssr = np.sum(residuals ** 2)
        sst = np.sum((y_values - np.mean(y_values)) ** 2)
        sigma2_est = ssr / n
        log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(sigma2_est) + 1)
        bic = -2 * log_likelihood + num_params * np.log(n)
        rsquared = 1 - ssr / sst

        return bic, rsquared

    # log linear slowly varying function
    x_min = find_x_min_index(series, max_lag, powerlaw_with_log_svf)
    y_values = series[x_min:]
    lags = np.arange(1, max_lag - x_min)
    initial_guess = [0.5, 1, 1]
    alpha, a_est, b_est = least_squares(_residuals_log, initial_guess, loss='soft_l1',
                                        args=(powerlaw_with_log_svf, lags, y_values)).x
    fitted_values = powerlaw_with_log_svf(lags, alpha, a_est, b_est)
    residuals = y_values - fitted_values
    bic, r2 = _get_goodness_of_fit(residuals, y_values, 2, n=1001)
    rr_results['log_Lx'] = (alpha, round(x_min, 0), round(bic, 2), round(r2, 2))

    result = pd.DataFrame(rr_results).T.reset_index()
    result.columns = ['slow varying func', 'alpha', 'xmin', 'BIC', 'R2']
    result['stock'] = stock_name
    result['fitting method'] = 'robust regression'

    return result


def compare_least_squares_fits(series, max_lag, stock_name, to_print=False):
    """
    for each slowly varying function
    - find best xmin
    - fit power law with slow func
    - print xmin, alpha, BIC and R2
    """

    least_squares_results = {}

    # log linear slowly varying function
    x_min = find_x_min_index(series, np.arange(1, max_lag + 1), powerlaw_with_log_svf)
    y_values = series[x_min:]
    lags = np.arange(1, max_lag - x_min)
    initial_guess = [0.5, 1, 1]
    params, _ = curve_fit(powerlaw_with_log_svf, lags, y_values, p0=initial_guess)
    bic, r2 = get_ic_for_model(powerlaw_with_log_svf, y_values, lags, params)
    least_squares_results['log_Lx'] = (params[0], round(x_min, 0), round(bic, 2), round(r2, 2))

    # exponential slowly varying function
    x_min = find_x_min_index(series, np.arange(1, max_lag + 1), powerlaw_with_exp_svf)
    y_values = series[x_min:]
    lags = np.arange(1, max_lag - x_min)
    initial_guess = [0.5, 1, 1]
    params, _ = curve_fit(powerlaw_with_exp_svf, lags, y_values, p0=initial_guess)
    bic, r2 = get_ic_for_model(powerlaw_with_exp_svf, y_values, lags, params)
    least_squares_results['exp_Lx'] = (params[0], round(x_min, 0), round(bic, 2), round(r2, 2))

    # period slowly varying function
    x_min = find_x_min_index(series, np.arange(1, max_lag + 1), powerlaw_with_per_svf)
    y_values = series[x_min:]
    lags = np.arange(1, max_lag - x_min)
    initial_guess = [0.5, 1, 1, 1]
    params, _ = curve_fit(powerlaw_with_per_svf, lags, y_values, p0=initial_guess)
    bic, r2 = get_ic_for_model(powerlaw_with_per_svf, y_values, lags, params)
    least_squares_results['per_Lx'] = (params[0], round(x_min, 0), round(bic, 2), round(r2, 2))

    # linear slowly varying function
    x_min = find_x_min_index(series, np.arange(1, max_lag + 1), powerlaw_with_exp_svf)
    y_values = series[x_min:]
    lags = np.arange(1, max_lag - x_min)
    initial_guess = [0.5, 1, 1]
    params, _ = curve_fit(powerlaw_with_exp_svf, lags, y_values, p0=initial_guess)
    bic, r2 = get_ic_for_model(powerlaw_with_exp_svf, y_values, lags, params)
    least_squares_results['lin_Lx'] = (params[0], round(x_min, 0), round(bic, 2), round(r2, 2))

    def _print_result(res):
        print(f"    xmin: {res[1]}")
        print(f"    alpha: {res[0][0]}")
        print(f"    BIC: {res[2]}")
        print(f'    Adjusted R2: {res[3]}')

    if to_print:
        print('Least squares fitting of power law with slowly varying function')
        print("Log")
        _print_result(least_squares_results['log_Lx'])
        print("Exponential")
        _print_result(least_squares_results['exp_Lx'])
        print("Period")
        _print_result(least_squares_results['per_Lx'])
        print("Linear")
        _print_result(least_squares_results['lin_Lx'])

    result = pd.DataFrame(least_squares_results).T.reset_index()
    result.columns = ['slow varying func', 'alpha', 'xmin', 'BIC', 'R2']
    result['stock'] = stock_name
    result['fitting method'] = 'least squares'

    return result


def get_ic_for_model(model, y_values, lags, params):
    fitted_values = model(lags, *params)
    residuals = y_values - fitted_values
    adjusted_rsquared, bic = compute_goodness_of_fit(residuals, y_values, params)

    return bic, adjusted_rsquared


def distribution_tests(series, function_name, test='all',
                       distributions=[laplace, norm, t, expon, lognorm, powerlaw, gumbel_l, gumbel_r]):
    tests = {
        'ks': 'Kolmogorov-Smirnov test:',
        'lr': 'Likelihood ratio test:'
    }

    if test == 'all':
        test = tests.keys()

    # Base case - always do Likelihood ratio test
    print('\n', tests['lr'])
    results = []
    for distribution in distributions:
        params = distribution.fit(series)
        log_likelihood = np.sum(distribution.logpdf(series, *params))
        results.append((distribution.name, log_likelihood))
    results.sort(key=lambda x: x[1], reverse=True)
    for result in results:
        print(result)
        print("")
    print(f'The most likely distribution for {function_name} according to Likelihood ratio test: {results[0][0]}')

    # The distribution with the highest log-likelihood is considered the most plausible distribution.
    best_likelihood = results[0][0]

    # KS test
    # A smaller KS statistic indicates a better fit to the data.
    # If the KS statistic is small, we do not reject the null hypothesis that the data follows the specified distribution.
    if 'ks' in test:
        print('\n', tests['ks'])
        results = []
        for dist in distributions:
            params = dist.fit(series)
            ks_result = kstest(series, dist.name, args=params)
            results.append((dist.name, ks_result[0]))  # Store (distribution name, KS statistic)

        # Sort by KS statistic in ascending order and print the most likely distribution
        results.sort(key=lambda x: x[1])
        for result in results:
            print(f"For {result[0]} distribution: KS statistic={result[1]}")
            print("\n")
        print(f"The most likely distribution for {function_name} according to KS-test: {results[0][0]}")
        print("\n")


    return best_likelihood


def get_residuals_loglikelihoods(first_series, fitting_func1, second_series, fitting_func2):
    x_values = np.arange(1, len(first_series) + 1)

    # Trim
    if len(first_series) != len(second_series):
        min_len = min(len(first_series), len(second_series))
        first_series = first_series[-min_len:]
        second_series = second_series[-min_len:]
        x_values = x_values[-min_len:]

    params_first_model = fitting_func1.fit(first_series)
    params_second_model = fitting_func2.fit(second_series)

    loglikelihoods1 = compute_loglikelihood(fitting_func1.pdf, params_first_model, x_values, first_series)
    loglikelihoods2 = compute_loglikelihood(fitting_func2.pdf, params_second_model, x_values, second_series)

    return loglikelihoods1, loglikelihoods2


