import numpy as np
import pandas as pd


# TODO: add empirical functional form of unconditional aggregate impact R(l)

def scaling_function(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Apply a scaling function 𝓕(x) to a given series.

    Parameters
    ----------
    x : np.ndarray
        An array of imbalances.
    alpha : float
        The alpha parameter representing the height (mean) of the scaling function.
    beta : float
         The beta parameter representing the width (std.) of the scaling function.

    Returns
    -------
    np.ndarray
        An array of scaled imbalances.

    Notes
    -----
    The scaling function 𝓕(x) is empirically a sigmoidal.
    """
    return x / np.power(1 + np.power(np.abs(x), alpha), beta / alpha)


def scaling_form(
    orderflow_imbalance: pd.DataFrame,
    RT: float,
    xT: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """
    Apply the scaling form :math: `R(x, T) = RT * 𝓕(xT)` to a series of order flow imbalances. See Patzelt (2018).

    Patzelt, Felix, and Jean-Philippe Bouchaud. "Universal scaling and
    nonlinearity of aggregate price impact in financial markets."
    Physical Review E 97, no. 1 (2018): 012304.

    Parameters
    ----------
    orderflow_imbalance : pd.DataFrame
        A DataFrame containing the columns 'imbalance' and 'T'.
    RT : float
        Scale factor for the returns scale.
    xT : float
        Scale factor for the imbalance scale.
    alpha : float
        The alpha parameter affecting the shape of the scaling function.
    beta : float
        The beta parameter affecting the shape of the scaling function.

    Returns
    -------
    np.ndarray
        An array of scaled order flow imbalances.
    """

    T = orderflow_imbalance["T"].values
    imbalance = orderflow_imbalance["imbalance"].values

    # Apply the scaling function to the normalized orderflow imbalance
    rescaled_imbalance = imbalance / (xT * T)
    scaled_imbalance = scaling_function(rescaled_imbalance, alpha, beta)

    return (RT * T) * scaled_imbalance


def scaling_law(
    orderflow_imbalance: pd.DataFrame,
    chi: float,
    kappa: float,
    alpha: float,
    beta: float,
    CONST: float,
) -> np.ndarray:
    """
    Apply the scaling law :math: `R(x, T) = R(1)T^chi * 𝓕(x/xT^kappa)`
    to series of order flow imbalances. See Bouchaud (2018).

    Bouchaud, Jean-Philippe, Julius Bonart, Jonathan Donier, and Martin Gould.
    Trades, quotes and prices: financial markets under the microscope.
    Cambridge University Press, 2018.

    Parameters
    ----------
    orderflow_imbalance : pd.DataFrame
        A DataFrame containing the columns 'imbalance' and 'T'.
    chi : float
        The chi rescaling exponent for returns in the scaling law.
    kappa : float
        The kappa rescaling exponent for x_imbalance in the scaling law.
    alpha : float
        The alpha parameter for the scaling function.
    beta : float
        The beta parameter for the scaling function.
    CONST : float
        A constant factor in the scaling law.

    Returns
    -------
    np.ndarray
        A series of scaled order flow imbalances according to the scaling law.
    """
    T = orderflow_imbalance["T"].values
    imbalance = orderflow_imbalance["imbalance"].values

    # Rescale imbalance according to kappa
    rescaled_imbalance = imbalance / np.power(T, kappa)

    # Apply the scaling function to rescaled imbalance
    scaled_imbalance = scaling_function(rescaled_imbalance, alpha, beta)

    return np.power(T, chi) * scaled_imbalance * CONST
