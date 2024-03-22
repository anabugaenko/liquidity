# -*- coding: utf-8 -*-
"""
Code for FSS by NN and LSE.

Copyright 2024 Anastasia Bugaenko
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

from powerlaw_function.powerlaw import Fit # from powerlaw_function import Fit
from liquidity.util.utils import _validate_imbalances
from liquidity.finite_scaling.fit import FitResult, fit_known_scaling_form


def _find_critical_exponents(
    xy_values: pd.DataFrame, fitting_method: str = "MLE"
) -> Fit:
    """
    Determine scaling behavior of the data by fitting power law and compare against alternative hypothesis.

    Parameters
    ----------
    xy_values : pd.DataFrame
        DataFrame containing x and y values to fit.
    fitting_method : str, optional
        Method used for fitting scaling exponents. Default is "MLE".

    Returns
    -------
    Fit
        Fit object representing the scaling behavior of the data.
    """
    if fitting_method == "MLE":
        return Fit(xy_values, xmin_distance="BIC", xmin_index=10)

    return Fit(xy_values, nonlinear_estimator=fitting_method, xmin_distance="BIC")


def find_scale_factors(
    aggregate_impact_data: pd.DataFrame,
    alpha: float,
    beta: float,
    reflect_y: bool = False,
    response_column: str = "R_cond",
    imbalance_column: str = "volume_imbalance",
) -> Dict[int, FitResult]:
    """
    Finds the scale factors RT and xT as a function of T by fitting the
    scaling form to the aggregate impact data for each T.

    Parameters
    ----------
    aggregate_impact_data : pd.DataFrame
        DataFrame containing the aggregate impact data.
    alpha : float
        Known alpha parameter from the scaling function.
    beta : float
        Known beta parameter from the scaling function.
    reflect_y : bool, optional
        If True, reflects the scaling function along the x-axis. Default is False.
    imbalance_column : str, optional
        Column name in the DataFrame for the order flow imbalance data. Default is "volume_imbalance".

    Returns
    -------
    Dict[int, FitResult]
        A dictionary mapping each bin size T to its corresponding FitResult.
    """
    _validate_imbalances(imbalance_column)
    aggregate_impact = aggregate_impact_data.copy()

    aggregate_impact.dropna(inplace=True)
    aggregate_impact.replace([np.inf, -np.inf], np.nan, inplace=False)

    # Map out scale factors
    scale_factors = {}
    binning_frequencies = aggregate_impact["T"].unique()
    for T in binning_frequencies:
        data = aggregate_impact[aggregate_impact["T"] == T][
            ["T", imbalance_column, response_column]
        ]
        R_values = data[response_column].values
        T_values = data["T"].values
        imbalance_values = data[imbalance_column].values

        param = fit_known_scaling_form(
            imbalance_values=imbalance_values,
            t_values=T_values,
            r_values=R_values,
            known_alpha=alpha,
            known_beta=beta,
            reflect_y=reflect_y,
        )
        if param[0] is not None:
            scale_factors[T] = FitResult(
                T, param, pd.DataFrame({"x": imbalance_values, "y": R_values})
            )
        else:
            print(f"Failed to fit for lag {T}")

    return scale_factors


def mapout_scale_factors(
    aggregate_impact_data: pd.DataFrame,
    alpha: float,
    beta: float,
    reflect_y: bool = False,
    fitting_method: str = "MLE",
    response_column: str = "R_cond",
    imbalance_column: str = "volume_imbalance",
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict, Dict[float, FitResult]]:
    """
    Find the rescaling exponents chi 𝛘 and kappa ϰ by fitting the scaling form to the shape of aggregate impact for
    each `T` while keeping values of the shape parameters alpha `α` and beta `β` the same (constant) for all `T`.

    Parameters
    ----------
    aggregate_impact_data : pd.DataFrame
        DataFrame containing the aggregate impact data.
    alpha : float
        Known alpha parameter from the scaling function.
    beta : float
        Known beta parameter from the scaling function.
    reflect_y : bool, optional
        If True, reflects the scaling function along the x-axis.
    fitting_method : str, optional
        Method used for fitting scaling exponents 𝛘 and ϰ.
    imbalance_column : str, optional
        Column name in the DataFrame for order flow imbalance data.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict, Dict[float, FitResult]]
        Contains DataFrames for scaled RT and xT values, fit objects representing the series RT and xT,
        fits of rescaling exponents 𝛘 and ϰ, and a dictionary of scale factors for each bin size T.

    Notes
    -----
    The preceding fit of the scaling function, yielding xT and RT for each T, doesn not impose any assumptions on their scaling.
    """
    _validate_imbalances(imbalance_column)
    data = aggregate_impact_data.copy()

    data.dropna(inplace=True)
    data.replace([np.inf, -np.inf], np.nan, inplace=False)

    # Map out scale factors RT and xT for each bin size T
    RT_series = []
    xT_series = []
    scale_factors = find_scale_factors(
        data,
        alpha=alpha,
        beta=beta,
        reflect_y=reflect_y,
        response_column=response_column,
        imbalance_column=imbalance_column,
    )

    for T, fit_result in scale_factors.items():
        RT_series.append(fit_result.params[0])
        xT_series.append(fit_result.params[1])

    # Perform unkown rescaling with respect to the corresponding bin size T
    bin_size = list(scale_factors.keys())
    scaled_RT = [R * T for R, T in zip(RT_series, bin_size)]
    scaled_xT = [x * T for x, T in zip(xT_series, bin_size)]

    # Determine the rescaling exponents chi 𝛘 and kappa ϰ
    RT = pd.DataFrame({"x_values": bin_size, "y_values": scaled_RT})
    xT = pd.DataFrame({"x_values": bin_size, "y_values": scaled_xT})
    RT_fit_object = _find_critical_exponents(RT, fitting_method, **kwargs)
    xT_fit_object = _find_critical_exponents(xT, fitting_method, **kwargs)

    return RT, xT, RT_fit_object, xT_fit_object


def transform(
    conditional_aggregate_impact: pd.DataFrame,
    rescaling_params: List[float],
    response_column: str = "R_cond",
    imbalance_column: str = "sign_imbalance",
) -> pd.DataFrame:
    """
    Transforms aggregate impact data at different scales, rescaling the data onto a single scaling function.

    Parameters
    ----------
    conditional_aggregate_impact : pd.DataFrame
        DataFrame containing conditional aggregate impact data.
    rescaling_params : List[float]
        Rescaling parameters for the master curve (chi, kappa, alpha, beta, CONST).
    imbalance_column : str, optional
        Column name for the order flow imbalance data. Default is "sign_imbalance".

    Returns
    -------
    DataFrame
        A DataFrame with rescaled aggregate impact data.

    Notes
    -----
    The data should return similar shape parameters for the different binning frequencies following renormalization.
    """
    _validate_imbalances(imbalance_column)
    original_data = conditional_aggregate_impact.copy()

    original_data.dropna(inplace=True)
    original_data.replace([np.inf, -np.inf], np.nan, inplace=False)

    # Do FSS analysis on original data using found rescaling exponents
    chi, kappa, alpha, beta, CONST = rescaling_params
    original_data[response_column] = original_data[response_column] / np.power(original_data["T"], chi)
    original_data[imbalance_column] = original_data[imbalance_column] / np.power(original_data["T"], kappa)

    rescaled_data = original_data

    return rescaled_data