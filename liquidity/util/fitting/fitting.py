import sys
import numpy as np
import pandas as pd
from typing import Callable
from matplotlib import pyplot as plt
from numpy import asarray, isinf, isnan
from powerlaw import loglikelihood_ratio
from scipy.stats import laplace, norm, t, expon, lognorm, powerlaw, gumbel_l, gumbel_r

from liquidity.util.fitting.compare_fits import distribution_tests, get_residuals_loglikelihoods
from liquidity.util.fitting.goodness_of_fit import compute_goodness_of_fit # loglikelihood_ratio
from liquidity.util.fitting.non_linear_fits import least_squares_fit
from liquidity.util.fitting.linear_fits import linear_fit_ols, linear_fit_robust_regression, \
    linear_fit_generalised_regression
from liquidity.util.fitting.powerlaw_functions import powerlaw_with_log_svf, powerlaw_with_exp_svf, \
    powerlaw_with_per_svf, powerlaw_with_lin_svf, \
    pure_powerlaw, find_x_min_index
from liquidity.util.fitting.parameters import BetaLambdaParam, PeriodicFunctionParam, PowerlawParam, \
    PowerlawSVFAlphaBetaLambdaParam, PowerlawSVFPeriodicParam
from liquidity.util.fitting.functions import exponential_function, logarithmic_function, periodic_function, \
    linear_function
from liquidity.util.fitting.util import block_print, enable_print

FUNC_TO_PARAM = {
    logarithmic_function: BetaLambdaParam,
    exponential_function: BetaLambdaParam,
    periodic_function: PeriodicFunctionParam,
    linear_function: BetaLambdaParam,
    pure_powerlaw: PowerlawParam,
    powerlaw_with_exp_svf: PowerlawSVFAlphaBetaLambdaParam,
    powerlaw_with_log_svf: PowerlawSVFAlphaBetaLambdaParam,
    powerlaw_with_lin_svf: PowerlawSVFAlphaBetaLambdaParam,
    powerlaw_with_per_svf: PowerlawSVFPeriodicParam,
}

NAME_TO_DIST = {
    'laplace': laplace,
    'norm': norm,
    't': t,
    'expon': expon,
    'lognorm': lognorm,
    'powerlaw': powerlaw,
    'gumbel_l': gumbel_l,
    'gumbel_r': gumbel_r
}

NAME_TO_FUNC = {
    exponential_function.__name__: exponential_function,
    logarithmic_function.__name__: logarithmic_function,
    periodic_function.__name__: periodic_function,
    linear_function.__name__: linear_function
}

LINEAR_FITTING_METHODS = {
    'OLS': linear_fit_ols,
    'Robust regression': linear_fit_robust_regression,
    'Generalised regression': linear_fit_generalised_regression,
}

POWERLAW_FUNCTIONS = [pure_powerlaw, powerlaw_with_exp_svf, powerlaw_with_log_svf, powerlaw_with_lin_svf,
                      powerlaw_with_per_svf]

NON_POWERLAW_FUNCTIONS = [exponential_function, logarithmic_function, periodic_function, linear_function]

SUPPORTED_FUNCTIONS = POWERLAW_FUNCTIONS + NON_POWERLAW_FUNCTIONS + list(LINEAR_FITTING_METHODS.values())


class FitResult:
    def __init__(self, residuals, params, fitted_values, function, fitting_method, xmin_index, xmin, data):
        self.residuals = residuals
        self.fitted_values = fitted_values
        self._function = function
        self._fitting_method = fitting_method
        self.fitting_method = fitting_method.__name__
        self.fitted_function = function.__name__
        self.fitting_method = fitting_method.__name__
        self.xmin_index = xmin_index
        self.xmin = xmin
        self.data = data
        self.params = self._map_paramas(function, params)
        self.adjusted_rsquared, self.bic = compute_goodness_of_fit(self.residuals, self.data.xmin_y_values,
                                                                   self.params.__dict__)

    def _map_paramas(self, function, param):
        param_class = FUNC_TO_PARAM[function]
        return param_class(*param)

    def to_dictionary(self):
        return {
            'residuals': self.residuals,
            'params': self.params.__dict__,
            'function': self._function,
            'fitted_values': self.fitted_values,
            'fitting_method': self._fitting_method,
            'xmin_index': self.xmin_index,
            'xmin': self.xmin,
            'data': self.data,
            'bic': self.bic,
            'adjusted_rsquared': self.adjusted_rsquared,
        }

    def print_fitted_results(self):
        print("\n")
        print(f'For {self._function.__name__} fitted using {self._fitting_method.__name__}.')
        print("\n")
        print('Pre-fitting parameters:')
        print(f'xmin: {self.xmin}')

        print("\n")
        print('Fitting parameters:')
        params_dict = self.params.__dict__
        for param_name, param_value in params_dict.items():
            print(f'{param_name} = {param_value}')

        print("\n")
        print('Goodness of fit to data:')
        print('BIC =', self.bic)
        print('Adjusted R-squared =', self.adjusted_rsquared)
        print('\n')

    def _plot_data(self, scale='loglog', data_kwargs=None, fit_kwargs=None, figure_kwargs=None):
        # Assigning relevant values
        func_name = self._function.__name__

        # Create plot with figure_kwargs
        plt.figure(**(figure_kwargs if figure_kwargs else {}))

        # Plot raw data according to the specified scale
        plot_func = plt.loglog if scale == 'loglog' else plt.plot if scale == 'linear' else None
        if plot_func is None:
            raise ValueError(f"Invalid scale value: {scale}. Must be either 'loglog' or 'linear'.")

        # Plots raw data
        plot_func(self.data.xmin_x_values, self.data.xmin_y_values, label='Raw data',
                  **(data_kwargs if data_kwargs else {}))

        # Plots fitted function on top of raw data
        plot_func(self.data.xmin_x_values, self._function(self.data.xmin_x_values, *(self.params.get_values())),
                  label=f'Fitted function: {func_name}', **(fit_kwargs if fit_kwargs else {}))

        # update legend with fit_kwargs if provided
        plt.legend(**(fit_kwargs if fit_kwargs else {'frameon': False}))

        plt.grid(False)
        plt.show()

    def plot_fit(self, data_kwargs=None, fit_kwargs=None, figure_kwargs=None, scale='loglog'):
        self._plot_data(data_kwargs=data_kwargs, fit_kwargs=fit_kwargs, figure_kwargs=figure_kwargs, scale=scale)


class Fit:
    def __init__(self, data: pd.DataFrame, xmin=None, verbose=False):

        # Ensure the DataFrame has exactly is 2D (has two columns)
        if len(data.columns) != 2:
            raise ValueError("Input data must have exactly two columns", file=sys.stderr)

        self.x_values = asarray(data.iloc[:, 0], dtype='float')
        self.y_values = asarray(data.iloc[:, 1], dtype='float')
        self.fit_results_dict = {}
        self.verbose = verbose

        # Ensure xmin is properly assigned
        if xmin and type(xmin) != tuple and type(xmin) != list:
            self.xmin = float(xmin)
            if verbose: print(f'Set xmin to {self.xmin}')
        else:
            self.xmin = None

        # Check for 0
        if 0 in self.x_values or 0 in self.y_values:
            if verbose:
                print("Values less than or equal to 0 in data. Throwing away these values", file=sys.stderr)

            zero_x_indx = 0 in self.x_values
            zero_y_indx = 0 in self.y_values
            self.x_values = np.delete(self.x_values, zero_x_indx + zero_y_indx)
            self.y_values = np.delete(self.y_values, zero_x_indx + zero_y_indx)

        # Check for inf
        if any(isinf(self.x_values)) or any(isinf(self.y_values)):
            if verbose:
                print("Infinite values in data. Throwing away these values", file=sys.stderr)

            inf_x_indx = isinf(self.x_values)
            inf_y_indx = isinf(self.y_values)
            self.x_values = np.delete(self.x_values, inf_x_indx + inf_y_indx)
            self.y_values = np.delete(self.y_values, inf_x_indx + inf_y_indx)

        # Check for nan
        if any(isnan(self.x_values)) or any(isnan(self.y_values)):
            if verbose:
                print("NaN values in data. Throwing away these values", file=sys.stderr)

            nan_x_indx = isnan(self.x_values)
            nan_y_indx = isnan(self.y_values)
            self.x_values = np.delete(self.x_values, nan_x_indx + nan_y_indx)
            self.y_values = np.delete(self.y_values, nan_x_indx + nan_y_indx)

        # Check for negative values
        if any(self.x_values < 0) or any(self.y_values < 0):
            if verbose:
                print("Negative values in data. Throwing away these values", file=sys.stderr)

            neg_x_indx = self.x_values < 0
            neg_y_indx = self.y_values < 0
            self.x_values = np.delete(self.x_values, neg_x_indx+neg_y_indx)
            self.y_values = np.delete(self.y_values, neg_x_indx+neg_y_indx)


        # Fit power-laws
        self._fit_functions()

    def __getattr__(self, name):
        if name in self.fit_results_dict.keys():
            return self.fit_results_dict[name]
        else:
            # Default behaviour
            print(f'Trying to access {name} on the fit object but {name} does not exist yet.'
                  f'Consider fitting {name} first.')
            raise AttributeError

    def _fit_functions(self):
        original_stdout = block_print() if not self.verbose else None

        try:
            print('Using Nonlinear Least-squares fitting method to directly fit power law functions: \n')

            for function in POWERLAW_FUNCTIONS:
                xmin_indx = np.where(self.x_values == self.xmin)[0][0] if self.xmin is not None \
                    else find_x_min_index(self.y_values, self.x_values, function)
                result = self._process_function(function, xmin_indx)
                self.fit_results_dict[function.__name__] = result

            print('Using Linear fitting methods to fit linear function on Loglog scale: \n')

            for method_name, fitting_method in LINEAR_FITTING_METHODS.items():
                xmin_indx = np.where(self.x_values == self.xmin)[0][0] if self.xmin is not None \
                    else find_x_min_index(self.y_values, self.x_values, pure_powerlaw)
                result = self._process_linear_method(fitting_method, xmin_indx)
                self.fit_results_dict[fitting_method.__name__] = result

        except Exception as e:
            enable_print(original_stdout)
            print(e)
        finally:
            enable_print(original_stdout)

        return self.fit_results_dict

    def _process_function(self, function: Callable, xmin_index: float) -> FitResult:
        xmin_x_values = self.x_values[xmin_index:]
        xmin_y_values = self.y_values[xmin_index:]

        residuals, params, fitted_values = least_squares_fit(xmin_x_values, xmin_y_values, function)
        result = FitResult(residuals=residuals, params=params, fitted_values=fitted_values, function=function,
                           fitting_method=least_squares_fit, xmin_index=xmin_index,
                           xmin=self.x_values[xmin_index], data=pd.DataFrame({
                'xmin_x_values': xmin_x_values,
                'xmin_y_values': xmin_y_values
            }))

        result.print_fitted_results()

        return result

    def _process_linear_method(self, fitting_method: Callable, xmin_index: float) -> FitResult:
        xmin_x_values = self.x_values[xmin_index:]
        xmin_y_values = self.y_values[xmin_index:]

        residuals, params, fitted_values = fitting_method(xmin_x_values, xmin_y_values)

        powerlaw_params = [params[1], np.exp(params[0])]  # alpha, const
        powerlaw_fitted_values = pure_powerlaw(xmin_x_values, *powerlaw_params)
        powerlaw_residuals = xmin_y_values - powerlaw_fitted_values
        result = FitResult(residuals=powerlaw_residuals, params=powerlaw_params, fitted_values=powerlaw_fitted_values,
                           function=pure_powerlaw, fitting_method=fitting_method, xmin_index=xmin_index,
                           xmin=self.x_values[xmin_index], data=pd.DataFrame({
                'xmin_x_values': xmin_x_values,
                'xmin_y_values': xmin_y_values
            }))

        result.print_fitted_results()

        return result

    def plot_data(self, scale='loglog', kwargs=None):
        xmin_y_values = self.y_values[self.xmin:]
        xmin_x_values = self.x_values[self.xmin:]

        # Plot raw data according to the specified scale
        plot_func = plt.loglog if scale == 'loglog' else plt.plot if scale == 'linear' else None
        if plot_func is None:
            raise ValueError(f"Invalid scale value: {scale}. Must be either 'loglog' or 'linear'.")

        # Plot raw data
        plot_func(xmin_x_values, xmin_y_values, label='Raw data',
                  **(kwargs if kwargs else {}))

        plt.grid(False)
        plt.legend(frameon=False)
        plt.show()

    def return_all_fitting_results(self):
        results_list = []
        for func_name, results in self.fit_results_dict.items():
            result_dict = results.to_dictionary().copy()
            func = result_dict['function']
            xmin_index = result_dict['xmin_index']
            xmin = result_dict['xmin']
            params = result_dict['params']
            adjusted_rsquared = result_dict['adjusted_rsquared']
            bic = result_dict['bic']

            summary_dict = {
                'name': func_name,
                'fitted function': func,
                'xmin_index': xmin_index,
                'xmin': xmin,
                'fitting param': params,
                'adjusted r-squared': round(adjusted_rsquared, 4),
                'bic': round(bic, 4),
            }

            results_list.append(summary_dict)

        return pd.DataFrame(results_list)

    def function_compare(self, fitting_func1: str, fitting_func2: str, verbose=False, **kwargs):
        """
        Using residuals.
        """

        original_stdout = block_print() if not verbose else None

        try:

            if fitting_func1 not in [function.__name__ for function in SUPPORTED_FUNCTIONS]:
                print(f"{fitting_func1} function is not supported, please choose from {SUPPORTED_FUNCTIONS}")
                enable_print(original_stdout)
                return None, None

            if fitting_func2 not in [function.__name__ for function in SUPPORTED_FUNCTIONS]:
                print(f"{fitting_func2} function is not supported, please choose from {SUPPORTED_FUNCTIONS}")
                enable_print(original_stdout)
                return None, None

            # Get xmin for power law fit
            pl_results = self.fit_results_dict.get(fitting_func1, None)
            if pl_results is None:
                print(f'Fitting result does not exist for specified {fitting_func1}. Consider running fit_power_law()')
                enable_print(original_stdout)
                return None, None
            xmin_index = pl_results.xmin_index

            # Fit the other function and store the results in fit_results
            if fitting_func2 in NAME_TO_FUNC.keys():
                print(f'Fitting {fitting_func2}')
                other_function: Callable = NAME_TO_FUNC.get(fitting_func2, None)
                result = self._process_function(other_function, xmin_index)
                self.fit_results_dict[other_function.__name__] = result

            # Get the residuals
            power_law_residuals = self.fit_results_dict[fitting_func1].residuals
            other_residuals = self.fit_results_dict[fitting_func2].residuals

            # Evaluate the distribution of the residuals for each series

            plr_dist = distribution_tests(power_law_residuals, function_name=fitting_func1)
            other_dist = distribution_tests(other_residuals, function_name=fitting_func2)

            # Compare residuals fitting
            plt_dist_fnc = NAME_TO_DIST[plr_dist]
            other_dist_fnc = NAME_TO_DIST[other_dist]

            fitting_func1 = plt_dist_fnc
            fitting_func2 = other_dist_fnc

            # Compute loglikelihood from residuals
            loglikelihoods1, loglikelihoods2 = get_residuals_loglikelihoods(power_law_residuals, fitting_func1,
                                                                            other_residuals, fitting_func2)


            # Computer normalised loglikelihood ratio R and p-value
            R, p = loglikelihood_ratio(loglikelihoods1, loglikelihoods2, **kwargs)

        except Exception as e:
            enable_print(original_stdout)
            print(e)

            return e
        finally:
            enable_print(original_stdout)

        enable_print(original_stdout)

        return R, p
