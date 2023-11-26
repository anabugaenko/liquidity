import warnings
import numpy as np
import pandas as pd
from typing import List
from scipy.optimize import curve_fit, least_squares

from powerlaw_function import Fit

from liquidity.util.utils import bin_data_into_quantiles
from liquidity.response_functions.functional_form import scaling_form, scaling_law


class ScalingFormFitResult:
    T: int
    param: List  # RN, QN
    alpha: float
    beta: float
    data: pd.DataFrame

    def __init__(self, T, param, alpha, beta, data):
        self.T = T
        self.param = param
        self.alpha = alpha
        self.beta = beta
        self.data = data


class FitResult:
    T: int
    params: List
    data: pd.DataFrame

    def __init__(self, T, params, data):
        self.T = T
        self.params = params
        self.data = data


def scaling_form_fit_results(features_df, alpha, beta):
    fit_results = {}
    Ts = features_df["T"].unique()
    for T in Ts:
        result = features_df[features_df["T"] == T][["vol_imbalance", "R", "T"]]
        result.replace([np.inf, -np.inf], np.nan, inplace=True)
        result.dropna(inplace=True)
        binned_result = bin_data_into_quantiles(result, duplicates="drop", q=1000) # should be size of data for per data point precision
        x = binned_result["vol_imbalance"].values
        y = binned_result["R"].values
        param = fit_known_scaling_form(binned_result, y, known_alpha=alpha, known_beta=beta)
        fit_results[T] = ScalingFormFitResult(T, param, alpha, beta, pd.DataFrame({"x": x, "y": y}))

    return fit_results


# def fit_scaling_function(df, x_col="vol_imbalance", y_col="R", y_reflect=False, verbose=False):
#
#     # Fit on non normalized data
#
#     try:
#         x_values = -df[x_col].values if y_reflect else df[x_col].values
#         y_values = df[y_col].values
#         initial_guess = [0.1, 0.1]
#         popt, pcov = curve_fit(
#             f=scaling_function,
#             xdata=x_values,
#             ydata=y_values,
#             p0=initial_guess,
#             bounds=(0, np.inf),
#             loss="soft_l1",
#             f_scale=0.2,
#         )
#
#         if verbose:
#             print(f"parameters found: {popt}")
#             print(f"standard deviations: {np.sqrt(np.diag(pcov))} \n")
#         return popt, pcov, scaling_function
#
#     except RuntimeError:
#         print("Optimal parameters not found: The maximum number of function evaluations is exceeded")
#         return None, None, None


def fit_rescaled_form(data_all, y_reflect=False, f_scale=0.2, verbose=False):
    fit_func = scaling_law # if not y_reflect else rescaled_form_reflect

    try:
        x_values = np.transpose(data_all.iloc[:, :2].to_numpy())
        y_values = data_all.iloc[:, 2].to_numpy()
        initial_guess = [0.1, 0.1, 0.1, 0.1, 0.1]
        popt, pcov = curve_fit(
            f=fit_func,
            xdata=x_values,
            ydata=y_values,
            p0=initial_guess,
            bounds=(0, np.inf),
            loss="soft_l1",
            f_scale=f_scale,
        )
        if verbose:
            print(f"parameters found: {popt}")
            # https://stackoverflow.com/questions/52275542/how-to-calculate-the-standard-error-from-a-variance-covariance-matrix
            print(f"standard deviations: {np.sqrt(np.diag(pcov))} \n")

        # fitted_values = rescaled_form(x_values, *popt)
        # mape = np.mean(np.abs((y_values - fitted_values) / y_values)) * 100

        return popt, pcov, fit_func
    except RuntimeError:
        print("Optimal parameters not found: The maximum number of function evaluations is exceeded")
        return None, None, None


def fit_known_scaling_form(data, y, known_alpha, known_beta):
    """
    Fits scaling form with known parameters from scaling function
    """

    def _known_scaling_form(data: pd.DataFrame, RN: float, QN: float) -> float:
        """
        This version treats RN and QN as constants to be found during optimisation.
        """
        return scaling_form(data, RN, QN, known_alpha, known_beta)

    def _residuals(params, data, y):
        return y - _known_scaling_form(data, *params)

    num_params = _known_scaling_form.__code__.co_argcount - 1
    initial_guess = [0.5, 1]

    result = least_squares(_residuals, initial_guess, args=(data, y), loss="soft_l1")
    return result.x

# def fit_scaling_form(x, y, known_alpha, known_beta):
#     """
#     Fits scaling form with known parameters from scaling function
#     """
#
#     def _rescaled_form(Q: float, RN: float, QN: float) -> float:
#         """
#         This version treats RN and QN as constants to be found during optimisation.
#         """
#         return scaling_form(Q, RN, QN, known_alpha, known_beta)
#
#     def _residuals(params, x, y):
#         return y - _rescaled_form(x, *params)
#
#     num_params = _rescaled_form.__code__.co_argcount - 1
#     initial_guess = [0.1] * num_params
#
#     result = least_squares(_residuals, initial_guess, args=(x, y), loss="soft_l1")
#     return result.x


# def fit_scaling_form_new(x, y, shape_param=None, scale_param=None):
#     """
#     Fits scaling form with known parameters from scaling function
#     """
#
#     if shape_param is not None:
#         def _scaling_form(Q: float, RN: float, QN: float) -> float:
#             """
#             This version treats RN and QN as constants to be found during optimisation.
#             """
#             return scaling_form(Q, RN, QN, *shape_param)
#     elif scale_param is not None:
#         def _scaling_form(Q: float, alpha: float, beta: float) -> float:
#             """
#             This version treats RN and QN as constants to be found during optimisation.
#             """
#             return scaling_form(Q, *scale_param, alpha, beta)
#     else:
#         def _scaling_form(Q: float, RN: float, QN: float, alpha:float, beta:float) -> float:
#             """
#             This version treats RN and QN as constants to be found during optimisation.
#             """
#             return scaling_form(Q, RN, QN, alpha, beta)
#
#     def _residuals(params, x, y):
#         return y - _scaling_form(x, *params)
#
#     num_params = _scaling_form.__code__.co_argcount - 1
#     initial_guess = [0.1] * num_params
#
#     result = least_squares(_residuals, initial_guess, args=(x, y), loss="soft_l1")
#     return result.x, _scaling_form


# TODO: move all the below to new python file called finite_size_scaling
def find_shape_parameters(normalised_aggregate_data: pd.DataFrame):
    """
    Computes shape parameters Alpha and Beta from known features
    """
    popt, pcov, fit_func = fit_rescaled_form(normalised_aggregate_data[["vol_imbalance", "T", "R"]])
    return popt, pcov, fit_func



def find_scaling_exponents(fitting_method: str, xy_values: pd.DataFrame, xmin_index=10) -> Fit:
    """Fits the data using the specified method and returns the fitting results."""
    if fitting_method == "MLE":
        return Fit(xy_values, xmin_distance="BIC", xmin_index=xmin_index)
    return Fit(xy_values, nonlinear_fit_method=fitting_method, xmin_distance="BIC")


def compute_scale_factors(conditional_aggregate_impact, alpha, beta, fitting_method="MLE", **kwargs):
    """
    Helper function to extract series of RN and QN
    from fit param for each N
    """

    # fit rescaled form at each N, returns dictionary of fitting results
    fit_results_per_lag = scaling_form_fit_results(conditional_aggregate_impact, alpha, beta)



    RN_series = []
    QN_series = []
    # FIXME: is this correct order for QN and RN series
    for lag, result in fit_results_per_lag.items():
        RN_series.append(result.param[0])
        QN_series.append(result.param[1])



    lags = list(fit_results_per_lag.keys())

    # Fit and return scaling exponents
    scaled_RN = [r * lag for r, lag in zip(RN_series, lags)]
    scaled_QN = [r * lag for r, lag in zip(QN_series, lags)]

    # Fit and return scaling exponents
    RN_df = pd.DataFrame({"x_values": lags, "y_values": scaled_RN})
    QN_df = pd.DataFrame({"x_values": lags, "y_values": scaled_QN})

    RN_fit_object = find_scaling_exponents(fitting_method, RN_df, **kwargs)
    QN_fit_object = find_scaling_exponents(fitting_method, QN_df, **kwargs)

    return RN_df, QN_df, RN_fit_object, QN_fit_object, fit_results_per_lag



def rescale_data(df: pd.DataFrame, popt, imbalance_col="vol_imbalance") -> pd.DataFrame:
    # popt[1] - kappa
    df["x_scaled"] = df[imbalance_col] / np.power(df["T"], popt[1])

    # popt[0] - chi
    df["y_scaled"] = df["R"] / np.power(df["T"], popt[0])

    return df


def renormalise(df: pd.DataFrame, params, durations, q=25):
    """
    Used for renormalisation and collapse at different scales.
    Hope returns similar params at different scales after renormalisation
    """
    chi, kappa, alpha, beta, gamma = params
    fit_param = {}
    for T in durations:
        result = df[df["T"] == T][["vol_imbalance", "T", "R"]]

        result["vol_imbalance"] = result["vol_imbalance"] / T**kappa
        result["R"] = result["R"] / T**chi
        binned_result = bin_data_into_quantiles(result, q=q, duplicates="drop")
        param = fit_rescaled_form(binned_result)

        if param[0] is not None:
            fit_param[T] = FitResult(T, param, binned_result)
        else:
            print(f"Failed to fit for lag {T}")

    return fit_param

def renormalise2(df: pd.DataFrame, params, durations, q=25):
    """
    Used for renormalisation and collapse at different scales.
    Hope returns similar params at different scales after renormalisation
    """
    chi, kappa, alpha, beta = params
    fit_param = {}
    for T in durations:
        result = df[df["T"] == T][["vol_imbalance", "T", "R"]]
        result.replace([np.inf, -np.inf], np.nan, inplace=True)
        result.dropna(inplace=True)

        result["vol_imbalance"] = result["vol_imbalance"] / T**kappa
        result["R"] = result["R"] / T**chi


        binned_result = bin_data_into_quantiles(result, q=q, duplicates="drop")
        x = binned_result["vol_imbalance"].values
        y = binned_result["R"].values

        param = fit_known_scaling_form(binned_result, y, known_alpha=alpha, known_beta=beta)

        if param[0] is not None:
            fit_param[T] = FitResult(T, param, binned_result)
        else:
            print(f"Failed to fit for lag {T}")

    return fit_param
