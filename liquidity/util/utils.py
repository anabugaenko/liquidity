import pandas as pd
import numpy as np
from scipy import stats


def smooth_outliers(df: pd.DataFrame, columns=['vol_imbalance', 'sign_imbalance'],
                    std_level=3, remove=False, verbose=False):
    """
        Clip or remove values at 3 standard deviations for each series.
    """

    if remove:
        z = np.abs(stats.zscore(df[columns]))
        original_shape = df.shape
        df = df[(z < std_level).all(axis=1)]
        new_shape = df.shape
        if verbose:
            print(f"Removed {original_shape[0] - new_shape[0]} rows")
    else:
        def winsorize_queue(s: pd.Series, level) -> pd.Series:
            upper_bound = level * s.std()
            if verbose:
                print(f"clipped at {upper_bound}")
            return s.clip(upper=upper_bound)

        for name in columns:
            s = df[name]
            if verbose:
                print(f'Series {name}')
            df[name] = winsorize_queue(s, level=std_level)

    return df


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


def add_order_signs(df_: pd.DataFrame) -> pd.DataFrame:
    def _ennumerate_sides(row):
        return 1 if row['side'] == 'ASK' else -1

    df_['sign'] = df_.apply(lambda row: _ennumerate_sides(row), axis=1)
    return df_


def add_returns(df, pct=True, remove_first=True):
    """
    To pd.DataFrame of order type time series add
    percentage returns and absolute normalised (by its volatility) returns.
    """
    if type(df['event_timestamp'].iloc[0]) != pd.Timestamp:
        df['event_timestamp'] = df['event_timestamp'].apply(lambda x: pd.Timestamp(x))
    if remove_first:
        df = remove_first_daily_prices(df)
    df['returns'] = df['midprice'].pct_change(1) if pct else df['midprice'].diff()
    df = df[~df['returns'].isin([np.nan, np.inf, -np.inf])]
    std = np.std(df['returns'])
    df['norm_returns'] = abs(df['returns'] / std)
    return df


def remove_midprice_orders(df_: pd.DataFrame) -> pd.DataFrame:
    mask = df_['price'] == df_['midprice']
    return df_[~mask]


def remove_first_daily_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    If the price deviated significantly during auction hours the first
    returns on the day would be considered outliers.
    """
    df_ = df.copy()
    df_['indx'] = df_.index
    df_ = df_.set_index('event_timestamp')
    first_days_indx = df_.groupby(pd.Grouper(freq='D')).first()['indx']
    first_days_indx = first_days_indx.dropna().astype(int)
    df_ = df_.loc[~df_['indx'].isin(first_days_indx)]
    return df_.drop(columns=['indx']).reset_index()


