import numpy as np

# TODO: add functional form of conditional and unconditional R(1, v) and R(l)


def scaling_function(x: float, alpha: float, beta: float) -> float:
    """
    Computes the value of the scaling function F(x), defined as a sigmoidal function.

    Parameters:
    - x (float): The normalized imbalance values for which we are calculating the function.
    - alpha (float): Represents the power controlling the function's behavior for small values of x.
    - beta (float): Represents the power controlling the function's behavior for large values of x.

    Returns:
    - float: The value of the scaling function F(x) for the given x, alpha, and beta.

    Notes:
    The function is described by the equation:

    .. math::
       F(x) = \frac{x}{(1 + |x|^\alpha)^{\frac{\beta}{\alpha}}}

    This function is used in the empirical relationship observed in financial markets
    to compute the rescaled impact based on the imbalance of orders.

    Pure math formula: F(x) = x / (1 + abs(x)^alpha)^(beta/alpha)
    """

    # Compute the value of the scaling function for the given parameters
    return x / np.power(1 + np.power(abs(x), alpha), beta / alpha)


def scaling_form(
    orderflow_imbalance: np.array, chi: float, kappa: float, alpha: float, beta: float, gamma: float
) -> np.array:
    """
    Computes the scaling form that characterizes the positive correlations between aggregate signed order flow
    and contemporaneous price returns over a coarse-grained timescale T.

    The function describes the empirical scaling law that holds for a variety of stocks and futures contracts.
    It represents the dependence of the aggregate impact on order-flow imbalance and the change in behavior
    of this dependence as T changes.

    Parameters:
    - orderflow_imbalance (np.array): A 2D array where:
        - orderflow_imbalance[0] is the order-flow imbalance, ΔV.
        - orderflow_imbalance[1] represents the time scale T.
    - chi (float): Scaling exponent of T, representing the empirical correlation between aggregate impact and time scale.
    - kappa (float): Scaling exponent for the normalization of the order-flow imbalance.
    - alpha (float): Small x growth power for the scaling function, obtained from empirical data.
    - beta (float): Large x growth power for the scaling function, also obtained from empirical data.
    - const (float): A constant, which should be close to 1 if order-flow imbalance and T are properly normalized.

    Returns:
    - np.array: The computed values of the scaling form for the given parameters.

    .. math::
       R(ΔV,T) \approx R(1) \times T^{\chi} \times F\left(\frac{ΔV}{T^{\kappa}}\right) \times \gamma
       where F(x) is the scaling function that is linear for small arguments and concave for large arguments.

    Pure math formula:  R(ΔV,T) = T^chi * F(ΔV/ T^kappa) * gamma
    """
    # Extracting the order-flow imbalance (ΔV) and timescale (T) from the input array
    imbalance = orderflow_imbalance[0]
    T = orderflow_imbalance[1]

    # Normalize the order-flow imbalance using the timescale and the scaling exponent kappa
    normalised_imbalance = imbalance / np.power(T, kappa)

    # Compute the scaling form value using the normalized imbalance, scaling function, and given parameters
    return np.power(T, chi) * scaling_function(normalised_imbalance, alpha, beta) * gamma  # replace gamma with const


def scaling_form_reflect(orderflow_imbalance, chi, kappa, alpha, beta, gamma):
    """
    The scaling form (sigmoid) where the scaling function is inverted on the y-axis).
    Used when fitting :math: L(t) := (LO, CA, QA)
    """
    imbalance = orderflow_imbalance[0]
    T = orderflow_imbalance[1]
    normalised_imbalance = -imbalance / np.power(T, kappa)
    return np.power(T, chi) * scaling_function(normalised_imbalance, alpha, beta) * gamma



def rescaled_form(imbalance: float, RN: float, QN: float, alpha: float, beta: float) -> float:
    """
    Computes the rescaled impact based on the empirical relationship observed in financial markets.

    Parameters:
    - imbalance (float): Signed volume imbalance ΔV of the aggregate-volume impact.
    - RN (float): N-dependent return scale.
    - QN (float): N-dependent volume scale.
    - alpha (float): Shape parameter describing the shape of the scaling function.
    - beta (float): Shape parameter describing the shape of the scaling function.

    Returns:
    - float: Rescaled impact value based on the given parameters.

    Notes:
    The function follows the equation:

    .. math::
       RN(ΔV) \approx RN \times F\left(\frac{ΔV}{QN}\right)

    where :math:`F(x)` is a sigmoidal function described in the reference paper.

    Pure math formula: RN(ΔV) ≈ RN * F(ΔV / QN)
    """

    # Compute the scaling function value based on the given parameters
    return RN * scaling_function(imbalance / QN, alpha, beta)
