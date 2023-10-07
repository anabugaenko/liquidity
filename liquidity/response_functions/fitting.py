import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares

from liquidity.util.utils import bin_data_into_quantiles, get_agg_features
from typing import List


class FitResult:
    T: int
    params: List
    data: pd.DataFrame

    def __init__(self, T, params, data):
        self.T = T
        self.params = params
        self.data = data


def scaling_function(x: float, alpha: float, beta: float) -> float:
    """
    Define the Sigmoidal function F(x) as mentioned in the paper.

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
    TODO: fix gamma
    """
    imbalance = orderflow_imbalance[0]
    T = orderflow_imbalance[1]
    normalised_imbalance = -imbalance / np.power(T, kappa)
    return np.power(T, chi) * scaling_function(normalised_imbalance, alpha, beta) * gamma


def fit_rescaled_form(x, y, known_alpha=None, know_beta=None):
    """
    Fits scaling form with known parameters from scaling function

    """
    if known_alpha and know_beta:
        def rescaled_form(Q: float, RN: float, QN: float) -> float:
            """
            This version treats RN and QN as constants to be found during optimisation.
            """

            return RN * scaling_function(Q / QN, known_alpha, know_beta)
    else:
        def rescaled_form(Q: float, RN: float, QN: float,
                             alpha: float, beta: float) -> float:
            """
            This version treats RN and QN as constants to be found during optimisation.
            """

            return RN * scaling_function(Q / QN, alpha, beta)

    def _residuals(params, x, y):
        return y - rescaled_form(x, *params)

    num_params = rescaled_form.__code__.co_argcount - 1
    initial_guess = [0.5] * num_params
    result = least_squares(_residuals, initial_guess, args=(x, y),loss='soft_l1')
    return result.x

def fit_scaling_form(data_all, y_reflect=False, f_scale=0.2, verbose=False):
    fit_func = scaling_form if not y_reflect else scaling_form_reflect

    try:
        initial_guess = [0.5, 0.5, 1.0, 1.0, 1.0]
        popt, pcov = curve_fit(
            fit_func,
            np.transpose(data_all.iloc[:, :2].to_numpy()),
            data_all.iloc[:, 2].to_numpy(),
            p0=initial_guess,
            bounds=(0, np.inf),
            loss="soft_l1",
            f_scale=f_scale,
        )
        if verbose:
            print(f"parameters found: {popt}")
            # https://stackoverflow.com/questions/52275542/how-to-calculate-the-standard-error-from-a-variance-covariance-matrix
            print(f"standard deviations: {np.sqrt(np.diag(pcov))} \n")

        return (popt, pcov, fit_func)
    except RuntimeError:
        print("Optimal parameters not found: The maximum number of function evaluations is exceeded")
        return None, None, None


def compute_fit_results(features_df, alpha, beta, MAX_LAG=1000):
    fit_results = {}
    for lag in range(1, MAX_LAG):
        result = features_df[features_df['T'] == lag][['vol_imbalance', 'R']]
        result.replace([np.inf, -np.inf], np.nan, inplace=True)
        result.dropna(inplace=True)
        binned_result = bin_data_into_quantiles(result, x_col='vol_imbalance', duplicates='drop', q=31)
        x = binned_result['vol_imbalance'].values
        y = binned_result['R'].values
        param = fit_rescaled_form(x, y, known_alpha=alpha, know_beta=beta)
        fit_results[lag] = {
            'param': param,
            'x': x,
            'y': y,
            'alpha': alpha,
            'beta': beta
        }

    return fit_results

def compute_RN_QN(fitting_result_dict):
    """
    Helper function to extract series of RN and QN
    from fit param for each N
    """
    pass


def fit_scaling_function(df, x_col="vol_imbalance", y_col="R", y_reflect=False, verbose=False):
    x_values = -df[x_col].values if y_reflect else df[x_col].values
    popt, pcov = curve_fit(scaling_function, x_values, df[y_col].values, bounds=(0, np.inf), loss="soft_l1")

    if verbose:
        print(f"parameters found: {popt}")
        print(f"standard deviations: {np.sqrt(np.diag(pcov))} \n")

    return popt, pcov, scaling_function


def rescale_data(df: pd.DataFrame, popt, imbalance_col="vol_imbalance") -> pd.DataFrame:
    df["x_scaled"] = df[imbalance_col] / np.power(df["T"], popt[1])
    df["y_scaled"] = df["R"] / np.power(df["T"], popt[0])

    return df


def compute_scaling_exponents(df: pd.DataFrame, durations: List = [5, 10, 20, 50, 100]):
    data_norm = get_agg_features(df, durations)
    popt, pcov, fit_func = fit_scaling_form(data_norm[["vol_imbalance", "T", "R"]])
    return popt, pcov, fit_func, data_norm


def renormalise(df: pd.DataFrame, params, durations: List = [5, 10, 20, 50, 100]):
    chi, kappa, alpha, beta, gamma = params
    fit_param = {}
    for T in durations:
        result = df[df["T"] == T][["vol_imbalance", "T", "R"]]

        result["vol_imbalance"] = result["vol_imbalance"] / T**kappa
        result["R"] = result["R"] / T**chi

        binned_result = bin_data_into_quantiles(result, q=100)
        param = fit_scaling_form(binned_result)
        if param[0] is None:
            # FIXME: add dynamic adjusting for less samples at higher durations
            print("re-trying")
            binned_result = bin_data_into_quantiles(result, q=31)
            param = fit_scaling_form(binned_result)

        if param[0] is not None:
            fit_param[T] = FitResult(T, param, binned_result)
        else:
            print(f"Failed to fit for lag {T}")

    return fit_param
