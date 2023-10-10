import numpy as np


def scaling_function(x: float, alpha: float, beta: float) -> float:
    """
    Define the Sigmoidal function F(x) as mentioned in the paper.

    Parameters:
    - x (float): The unormalised imbalance values for which we are calculating the function.
    - alpha (float): Represents the small x growth power.
    - beta (float): Represents the large x growth power.

    Returns:
    - float: The result of the scale function for given x, alpha, and beta.
    """
    return x / np.power(1 + np.power(abs(x), alpha), beta /alpha)


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
    normalised_imbalance = -imbalance / np.power(T, kappa)
    return np.power(T, chi) * scaling_function(normalised_imbalance, alpha, beta) * gamma


def rescaled_form(imbalance, RN, QN, alpha, beta):
    return RN * scaling_function(imbalance / QN, alpha, beta)
