import numpy as np
import pandas as pd
from typing import List, Dict

from liquidity.util.estimators import lse
from liquidity.finite_scaling.functional_form import scaling_form, scaling_law


class MapParams:
    """
    Dynamically maps parameters to their correpsonding values using Python's introspective capabilites.

    Parameters:
        function(callable): The function for which the parameters are being stored. It should be such that first
        argument in the functions siganture is the independent variable (commonly 'x'), followed by its parameters.
        params (list or tuple): The parameter values for the function in order as in the function definition.

    Attributes:
        param_names (list): The names of the parameters of the function, excluding the independent variable.

    Methods:
        get_values(): Returns the parameter values in the same order as `param_names`.
    """

    def __init__(self, function, params):
        self.param_names = list(signature(function).parameters.keys())[
            1:
        ]  # excluding independent variable 'x'
        for name, value in zip(self.param_names, params):
            setattr(self, name, value)

    def get_values(self):
        return [getattr(self, name) for name in self.param_names]


class FitResult:
    """
    Represents the result of a fitting procedure.

    Attributes:
        T (int): The bin size or event-time scale used in the fitting procedure.
        param (List[float]): The parameters resulting from the scaling form.
        data (pd.DataFrame): The data used for fitting.
    """

    T: int
    params: List
    data: pd.DataFrame

    def __init__(self, T, params, data):
        self.T = T
        self.params = params
        self.data = data


def fit_known_scaling_form(
    t_values: List[float],
    imbalance_values: List[float],
    r_values: List[float],
    known_alpha: float,
    known_beta: float,
    reflect_y: bool = False,
    bounds: List[float] = (0.1, np.inf),
) -> Dict:
    """
    Fits a scaling form with known parameters alpha `α` and beta `β` from the scaling function.

    Parameters
    ----------
    t_values : list of float
        List of binning frequencies or event-time scale values T.
    imbalance_values : list of float
        List of order flow imbalance values.
    r_values : list of float
        List of aggregate impact values R.
    known_alpha : float
        Known alpha parameter from the scaling function.
    known_beta : float
        Known beta parameter from the scaling function.
    reflect_y : bool, optional
        If True, reflects the scaling function along the x-axis. Default is False.
    bounds : list of float, optional
        Bounds for the optimization. Default is (0.1, np.inf).

    Returns
    -------
    dict
        A dictionary containing the optimized parameters.

    Notes
    -----
    If reflect=True, fits scaling form reflection where we invert the scaling function along the x-axis.
    """

    def _known_scaling_form(data: pd.DataFrame, RN: float, QN: float) -> float:
        """
        Defines a scaling form with known parameters.

        This version treats RN and QN as optimization parameters to
        be found whilst fixing alpha and beta as constants.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing order flow imbalance and binning frequencies.
        RN : float
            Parameter for the scaling form.
        QN : float
            Parameter for the scaling form.

        Returns
        -------
        float
            The value of the scaling function at the given point.
        """
        return scaling_form(data, RN, QN, known_alpha, known_beta)

    # Perform least squares optimization
    orderflow_imbalance = pd.DataFrame({"T": t_values, "imbalance": imbalance_values})
    residuals, params, model_predictions = lse(
        x_values=orderflow_imbalance,
        y_values=r_values,
        function=_known_scaling_form,
        reflect_y=reflect_y,
        bounds=bounds,
    )

    return params


def fit_scaling_form(
    t_values: List[float],
    imbalance_values: List[float],
    r_values: List[float],
    reflect_y: bool = False,
    bounds: List[float] = (0, np.inf),
) -> Dict:
    """
    Fit a scaling form to aggregate impact :math: `R(x, T)` data.

    Parameters
    ----------
    t_values : list of float
        List of binning frequencies or event-time scale values T.
    imbalance_values : list of float
        List of order flow imbalance values.
    r_values : list of float
        List of aggregate impact values R.
    reflect_y : bool, optional
        If True, reflects the scaling function along the x-axis.
    bounds : list of float, optional
        Bounds for the optimization.

    Returns
    -------
    dict
        A dictionary containing the optimized parameters.

    Notes
    -----
    The function uses a neural network or the method of least squares to find the optimal scaling form parameters.
    """
    orderflow_imbalance = pd.DataFrame({"T": t_values, "imbalance": imbalance_values})

    # Perform least squares optimization
    residuals, params, model_predictions = lse(
        x_values=orderflow_imbalance,
        y_values=r_values,
        function=scaling_form,
        reflect_y=reflect_y,
        bounds=bounds,
    )

    return params


def fit_scaling_law(
    t_values: List[float],
    imbalance_values: List[float],
    r_values: List[float],
    reflect_y: bool = False,
    bounds: List[float] = (0, np.inf),
) -> Dict:
    """
     Fit a scaling law to the renormalized aggregate impact `:math: R(x, T)` data.

     Parameters
    ----------
    T_values : list of float
        List of binning frequencies or event-time scale values T.
    imbalance_values : list of float
        List of order flow imbalance values.
    R_values : list of float
        List of conditional aggregate impact values R.
    reflect_y : bool, optional
        If True, reflects the scaling function along the x-axis.
    bounds : list of float, optional
        Bounds for the optimization.

    Returns
    -------
    Dict
        A dictionary containing the optimized parameters.

    Notes
    -----
    Assumes the conditional aggregate impact data ["T", "x_imbalance", "R"] has been renormalized.
    If reflect=True, fits scaling form reflection where we invert the scaling function along the x-axis.
    """
    orderflow_imbalance = pd.DataFrame({"T": t_values, "imbalance": imbalance_values})

    # Perform least squares optimization
    residuals, params, model_predictions = lse(
        x_values=orderflow_imbalance,
        y_values=r_values,
        function=scaling_law,
        reflect_y=reflect_y,
        bounds=bounds,
    )

    return params
