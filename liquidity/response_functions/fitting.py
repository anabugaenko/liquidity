import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from liquidity.util.fitting_util import get_agg_features, bin_data_into_quantiles
from typing import List

def scaling_function(x: float, alpha: float, beta: float) -> float:
    """
    Define the sigmoidal function F(x) as mentioned in the paper.

    Parameters:
    - x (float): The value for which we are calculating the function.
    - alpha (float): Represents the small x growth power.
    - beta (float): Represents the large x growth power.

    Returns:
    - float: The result of the scale function for given x, alpha, and beta.
    """
    return x / (1 + abs(x) ** alpha) ** (beta / alpha)


def scaling_form(orderflow_imbalance, chi, kappa, alpha, beta, gamma):
    """
    Function used for optimization #1

    Parameters
    ----------
    x : np.array()
        x-axis data
    alpha: float
        small x growth power
    beta: float
        large x growth power
    chi: float
        scaling exponent of x, typically in [0.5, 1]
    kappa: float
        scaling exponent of y, typically in [0.5, 1]
    gamma: float
        if x and y properly normalized, gamma should be 1

    Returns: np.array() with the same shape as one column of x
    ----------
    """
    # Separate input array
    imbalance = orderflow_imbalance[0]
    T = orderflow_imbalance[1]
    normalised_imbalance = imbalance / np.power(T, kappa)
    return np.power(T, chi) * scaling_function(normalised_imbalance, alpha, beta) * gamma


def scaling_form_reflect(orderflow_imbalance, chi, kappa, alpha, beta, gamma):
    """
    Inverse (on y axis) sigmoid.
    """
    imbalance = orderflow_imbalance[0]
    T = orderflow_imbalance[1]
    normalised_imbalance = - imbalance / np.power(T, kappa)
    return np.power(T, chi) * scaling_function(normalised_imbalance, alpha, beta) * gamma


def fit_scaling_form(data_all, y_reflect=False, f_scale=0.2, verbose=False):
    fit_func = scaling_form if not y_reflect else scaling_form_reflect

    try:
        popt, pcov = curve_fit(fit_func, np.transpose(data_all.iloc[:, :2].to_numpy()),
                               data_all.iloc[:, 2].to_numpy(),
                               bounds=(0, np.inf), loss='soft_l1', f_scale=f_scale)
        if verbose:
            print(f'parameters found: {popt}')
            # https://stackoverflow.com/questions/52275542/how-to-calculate-the-standard-error-from-a-variance-covariance-matrix
            print(f'standard deviations: {np.sqrt(np.diag(pcov))} \n')

        return (popt, pcov, fit_func)
    except RuntimeError:
        print('Optimal parameters not found: The maximum number of function evaluations is exceeded')
        return None, None, None


def fit_scaling_function(df, x_col='vol_imbalance', y_col='R', y_reflect=False, verbose=False):
    x_values = -df[x_col].values if y_reflect else df[x_col].values
    popt, pcov = curve_fit(scaling_function, x_values,
                           df[y_col].values,
                           bounds=(0, np.inf), loss='soft_l1')

    if verbose:
        print(f'parameters found: {popt}')
        print(f'standard deviations: {np.sqrt(np.diag(pcov))} \n')

    return popt, pcov, scaling_function


def rescale_data(df: pd.DataFrame, popt, imbalance_col='vol_imbalance') -> pd.DataFrame:

    df['x_scaled'] = df[imbalance_col] / np.power(df['T'], popt[1])
    df['y_scaled'] = df['R'] / np.power(df['T'], popt[0])

    return df


def normalise_axis(df_: pd.DataFrame, imbalance_col='vol_imbalance') -> pd.DataFrame:
    df = df_.copy()
    if imbalance_col == 'vol_imbalance':
        df['vol_imbalance'] = df['vol_imbalance'] / df['daily_vol']
    else:
        df['sign_imbalance'] = df['sign_imbalance'] / df['daily_num']
    df['R'] = df['R'] / abs(df['daily_R1'])

    return df


def compute_scaling_exponents(df: pd.DataFrame, durations: List = [5, 10, 20, 50, 100]):
    data = get_agg_features(df, durations)
    data_norm = normalise_axis(data)
    binned_data = []

    for T in durations:
        result = data_norm[data_norm['T'] == T][['vol_imbalance', 'T', 'R']]
        binned_data.append(bin_data_into_quantiles(result))
    binned_result = pd.concat(binned_data)
    popt, pcov, fit_func = fit_scaling_form(binned_result)

    return popt, pcov, fit_func, data_norm


class FitResult:
    T: int
    params: List
    data: pd.DataFrame

    def __init__(self, T, params, data):
        self.T = T
        self.params = params
        self.data = data


def renormalise(df: pd.DataFrame, params, durations: List = [5, 10, 20, 50, 100]):
    chi, kappa, alpha, beta, gamma = params
    fit_param = {}
    for T in durations:
        result = df[df['T'] == T][['vol_imbalance', 'T', 'R']]

        result['vol_imbalance'] = result['vol_imbalance'] / T ** kappa
        result['R'] = result['R'] / T ** chi

        binned_result = bin_data_into_quantiles(result, q=100)
        param = fit_scaling_form(binned_result)
        if param[0] is None:
            # FIXME: add dynamic adjusting for less samples at higher durations
            print('re-trying')
            binned_result = bin_data_into_quantiles(result, q=31)
            param = fit_scaling_form(binned_result)

        if param[0] is not None:
            fit_param[T] = FitResult(T, param, binned_result)
        else:
            print(f'Failed to fit for lag {T}')

    return fit_param