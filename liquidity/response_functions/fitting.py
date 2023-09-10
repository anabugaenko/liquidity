import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


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


