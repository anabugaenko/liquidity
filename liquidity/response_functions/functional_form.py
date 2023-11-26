import numpy as np
import pandas as pd

# TODO: add functional form of conditional and unconditional R(1, v) and R(l)

def scaling_function(x: float, alpha: float, beta: float) -> float:
    return x / np.power(1 + np.power(abs(x), alpha), beta / alpha)

def scaling_form(orderflow_imbalance, RN, QN, alpha, beta):

    # Extract imbalance and T from the DataFrame
    imbalance = orderflow_imbalance["vol_imbalance"].values
    T = orderflow_imbalance["T"].values

    # Apply the scaling function to the rescaled imbalance
    rescaled_imbalance = imbalance / (QN * T)
    scaled_imbalance = scaling_function(rescaled_imbalance, alpha, beta)

    return (RN * T) * scaled_imbalance


def scaling_law(orderflow_imbalance, chi, kappa, alpha, beta, const):
    imbalance = orderflow_imbalance["vol_imbalance"].values
    T = orderflow_imbalance["T"].values

    rescaled_imbalance = imbalance / np.power(T, kappa)
    scaled_imbalance = scaling_function(rescaled_imbalance, alpha, beta)

    return np.power(T, chi) * scaled_imbalance * const