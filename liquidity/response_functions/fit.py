import numpy as np
import pandas as pd
from typing import List
from scipy.optimize import curve_fit, least_squares

from powerlaw_function import Fit

from liquidity.util.utils import bin_data_into_quantiles
from liquidity.response_functions.features import compute_aggregate_features
from liquidity.response_functions.functional_form import scaling_function, scaling_form, scaling_form_reflect


class RescaledFormFitResult:
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


def rescaled_form_fit_results(features_df, alpha, beta, MAX_LAG=1000):
    # Rename
    fit_results = {}
    for lag in range(1, MAX_LAG):
        result = features_df[features_df["T"] == lag][["vol_imbalance", "R"]]
        result.replace([np.inf, -np.inf], np.nan, inplace=True)
        result.dropna(inplace=True)
        binned_result = bin_data_into_quantiles(result, x_col="vol_imbalance", duplicates="drop", q=31)
        x = binned_result["vol_imbalance"].values
        y = binned_result["R"].values
        param = fit_rescaled_form(x, y, known_alpha=alpha, know_beta=beta)
        fit_results[lag] = RescaledFormFitResult(lag, param, alpha, beta, pd.DataFrame({"x": x, "y": y}))

    return fit_results


def fit_scaling_function(df, x_col="vol_imbalance", y_col="R", y_reflect=False, verbose=False):
    x_values = -df[x_col].values if y_reflect else df[x_col].values
    y_values = df[y_col].values
    initial_guess = [1.0, 1.0]
    popt, pcov = curve_fit(
        f=scaling_function,
        xdata=x_values,
        ydata=y_values,
        p0=initial_guess,
        bounds=(0, np.inf),
        loss="soft_l1",
        f_scale=0.2,
    )

    if verbose:
        print(f"parameters found: {popt}")
        print(f"standard deviations: {np.sqrt(np.diag(pcov))} \n")

    fitted_values = scaling_function(x_values, *popt)
    mape = np.mean(np.abs((y_values - fitted_values) / y_values)) * 100

    return popt, mape, scaling_function


def fit_scaling_form(data_all, y_reflect=False, f_scale=0.2, verbose=False):
    fit_func = scaling_form if not y_reflect else scaling_form_reflect

    try:
        x_values = np.transpose(data_all.iloc[:, :2].to_numpy())
        y_values = data_all.iloc[:, 2].to_numpy()
        initial_guess = [0.5, 0.5, 1.0, 1.0, 1.0]
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

        fitted_values = scaling_form(x_values, *popt)
        # mape = np.mean(np.abs((y_values - fitted_values) / y_values)) * 100

        return popt, pcov, fit_func  # popt, mape, fit_func
    except RuntimeError:
        print("Optimal parameters not found: The maximum number of function evaluations is exceeded")
        return None, None, None


def fit_rescaled_form(x, y, known_alpha=None, know_beta=None):
    """
    Fits scaling form with known parameters from scaling function
    """

    def _rescaled_form(Q: float, RN: float, QN: float) -> float:
        """
        This version treats RN and QN as constants to be found during optimisation.
        """

        return RN * scaling_function(Q / QN, known_alpha, know_beta)

    def _residuals(params, x, y):
        return y - _rescaled_form(x, *params)

    num_params = _rescaled_form.__code__.co_argcount - 1
    initial_guess = [0.5] * num_params
    result = least_squares(_residuals, initial_guess, args=(x, y), loss="soft_l1")
    return result.x


# FIXME: rename to _find_shape_parameters
def compute_shape_parameters(df: pd.DataFrame, durations: List = [5, 10, 20, 50, 100], **kwargs):
    """
    Computes shape parameters Alpha and Beta from known features
    """
    data_norm = compute_aggregate_features(df, durations, **kwargs)
    popt, pcov, fit_func = fit_scaling_form(data_norm[["vol_imbalance", "T", "R"]])
    return popt, pcov, fit_func, data_norm


def _find_scaling_exponents(fitting_method: str, xy_values: pd.DataFrame) -> Fit:
    """Fits the data using the specified method and returns the fitting results."""
    if fitting_method == "MLE":
        return Fit(xy_values, xmin_distance="BIC", xmin_index=10)
    return Fit(xy_values, nonlinear_fit_method=fitting_method, xmin_distance="BIC")


def compute_RN_QN(features_df, alpha, beta, fitting_method="MLE", **kwargs):
    """
    Helper function to extract series of RN and QN
    from fit param for each N
    """

    # fit rescaled form at each N, returns dictionary of fitting results
    fit_results_per_lag = rescaled_form_fit_results(features_df, alpha, beta, **kwargs)

    RN_series = []
    QN_series = []
    for lag, result in fit_results_per_lag.items():
        RN_series.append(result.param[0])
        QN_series.append(result.param[1])

    lags = list(fit_results_per_lag.keys())

    # Fit and return scaling exponents
    RN_df = pd.DataFrame({"x_values": lags, "y_values": RN_series})
    QN_df = pd.DataFrame({"x_values": lags, "y_values": QN_series})

    RN_df = RN_df[RN_df["y_values"] >= RN_df["y_values"].iloc[10]]
    QN_df = QN_df[QN_df["y_values"] >= QN_df["y_values"].iloc[10]]

    RN_fit_object = _find_scaling_exponents(fitting_method, RN_df)
    QN_fit_object = _find_scaling_exponents(fitting_method, QN_df)

    return RN_df, QN_df, RN_fit_object, QN_fit_object


# FIXME: used where?
def rescale_data(df: pd.DataFrame, popt, imbalance_col="vol_imbalance") -> pd.DataFrame:
    df["x_scaled"] = df[imbalance_col] / np.power(df["T"], popt[1])
    df["y_scaled"] = df["R"] / np.power(df["T"], popt[0])

    return df


def renormalise(df: pd.DataFrame, params, durations, q=31):
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
        param = fit_scaling_form(binned_result)

        if param[0] is not None:
            fit_param[T] = FitResult(T, param, binned_result)
        else:
            print(f"Failed to fit for lag {T}")

    return fit_param
