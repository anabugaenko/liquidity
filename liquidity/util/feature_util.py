import numpy as np
import pandas as pd


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
