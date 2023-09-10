import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from liquidity.response_functions.price_response_functions import compute_conditional_aggregate_impact


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


def scaling_form(qT, chi, kappa, alpha, beta, gamma):
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
    q = qT[0]
    T = qT[1]
    x = q / np.power(T, kappa)
    return np.power(T, chi) * scaling_function(x, alpha, beta) * gamma


def scaling_form_reflect(qT, chi, kappa, alpha, beta, gamma):
    """
    Inverse (on y axis) sigmoid.
    """
    q = qT[0]
    T = qT[1]
    x = - q / np.power(T, kappa)
    return np.power(T, chi) * scaling_function(x, alpha, beta) * gamma


def transform_results_df(df_, T, imbalance_col='vol_imbalance'):
    data = df_.rename(columns={f'R{T}': 'R'})
    data['T'] = T
    cols = [imbalance_col, 'T', 'R']
    data = data[cols]
    data = data.groupby(imbalance_col).mean().reset_index()
    return data


def prepare_data_for_fitting(df_, durations, imbalance_col='vol_imbalance'):
    results_ = []
    for i in range(len(durations)):
        result = compute_conditional_aggregate_impact(df_, T=durations[i])
        results_.append(transform_results_df(result, durations[i], imbalance_col=imbalance_col))

    return pd.concat(results_)


def prepare_lo_data_for_fitting(df_, durations, imbalance_col='vol_imbalance'):
    results_ = []
    for i in range(len(durations)):
        result = compute_conditional_aggregate_impact(df_, T=durations[i])
        results_.append(transform_results_df(result, durations[i], imbalance_col=imbalance_col))

    return pd.concat(results_)


def get_fit_params(data_all, y_reflect=False, f_scale=0.2, verbose=False):
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


def fit_scale_function(df, x_col='vol_imbalance', y_col='R', y_reflect=False, verbose=False):
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


def bin_data_into_quantiles(df, x_col='vol_imbalance', y_col='R', q=100, duplicates='raise'):
    """
    3/9/23 This is majorly used.
    """
    binned_x = pd.qcut(df[x_col], q=q, labels=False, retbins=True, duplicates=duplicates)
    binned_x = binned_x[0]
    df['x_bin'] = binned_x

    y_binned = df.groupby(['x_bin'])[y_col].mean()
    y_binned.index = y_binned.index.astype(int)

    x_binned = df.groupby(['x_bin'])[x_col].mean()
    x_binned.index = x_binned.index.astype(int)

    if 'T' in df.columns:
        r_binned = df.groupby(['x_bin'])['T'].first()
        r_binned.index = r_binned.index.astype(int)
    else:
        r_binned = None

    return pd.concat([x_binned, r_binned, y_binned], axis=1).reset_index(drop=True)


def get_agg_features(df: pd.DataFrame, durations):
    results_ = []
    for i, T in enumerate(durations):
        lag_data = compute_conditional_aggregate_impact(df, T=T)
        lag_data['R'] = lag_data[f'R{T}']
        lag_data = lag_data.drop(columns=f'R{T}')
        lag_data['T']= T
        results_.append(lag_data)

    return pd.concat(results_)