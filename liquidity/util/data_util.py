import pandas as pd
import numpy as np
from scipy import stats


def remove_outliers(df_, columns=['norm_trade_volume', 'R1'], print_info=True):
    """
    TODO: reconcile
    """
    z = np.abs(stats.zscore(df_[columns]))
    if print_info:
        print(df_.shape)
    df_ = df_[(z < 2).all(axis=1)]
    if print_info:
        print(df_.shape)
    return df_


def _remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
       TODO: reconcile
    """
    """
    Remove observations where volume imbalance or price response
    value was beyond three standard deviations of it's mean.
    """

    def winsorize_queue(s: pd.Series, level: int = 3) -> pd.Series:
        upper_bound = level * s.std()
        return s.clip(upper=upper_bound)

    queue_columns = ['vol_imbalance', 'sign_imbalance']

    for name in queue_columns:
        s = df[name]
        df[name] = winsorize_queue(s)

    return df


def remove_midprice_orders(df_: pd.DataFrame) -> pd.DataFrame:
    mask = df_['price'] == df_['midprice']
    return df_[~mask]


def rename_columns(df_: pd.DataFrame) -> pd.DataFrame:
    df_columns = df_.columns

    if 'old_price' in df_columns and 'old_size' in df_columns:
        df_ = df_.drop(['price', 'size'], axis=1)
        df_ = df_.rename(columns={'old_price': 'price', 'old_size': 'size'})

    if 'R1_CA' in df_columns:
        df_ = df_.rename(columns={'R1_CA': 'R1'})

    if 'R1_LO' in df_columns:
        df_ = df_.rename(columns={'R1_LO': 'R1'})

    if 'execution_size' in df_columns:
        df_ = df_.rename(columns={'execution_size': 'size'})

    if 'trade_sign' in df_columns:
        df_ = df_.rename(columns={'trade_sign': 'sign'})

    return df_
