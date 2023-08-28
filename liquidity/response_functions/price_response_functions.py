import pandas as pd
import numpy as np


def add_daily_features(df_: pd.DataFrame, response_column: str = 'R1') -> pd.DataFrame:
    """
    From a given time series of transactions add daily means of
    lag one price response R1 and order size.
    """
    if type(df_['event_timestamp'].iloc[0]) != pd.Timestamp:
        df_['event_timestamp'] = df_['event_timestamp'].apply(lambda x: pd.Timestamp(x))
    df_['date'] = df_['event_timestamp'].apply(lambda x: x.date())

    daily_R1 = df_[[response_column, 'date']].groupby('date').agg(daily_R1=(response_column, 'mean'))
    daily_volume = df_[['size', 'date']].groupby('date').agg(daily_vol=('size', 'sum'))
    daily_num = df_[['size', 'date']].groupby('date').agg(daily_num=('size', 'count'))

    df_['daily_R1'] = daily_R1.reindex(index=df_['event_timestamp'], method='ffill').values
    df_['daily_vol'] = daily_volume.reindex(index=df_['event_timestamp'], method='ffill').values
    df_['daily_num'] = daily_num.reindex(index=df_['event_timestamp'], method='ffill').values

    return df_


def individual_response_function(df_: pd.DataFrame, response_column: str = 'R1') -> pd.DataFrame:
    """
    Lag one price response of market orders defined as
    difference in mid-price immediately before subsequent MO
    and the mid-price immediately before the current MO
    aligned by the original MO direction.
    """
    df_['midprice_change'] = df_['midprice'].diff().shift(-1).fillna(0)
    df_[response_column] = df_['midprice_change'] * df_.index.get_level_values('sign')
    return df_


def aggregate_response_function(df_: pd.DataFrame, T: int, response_column: str, log=False) -> pd. DataFrame:
    """
    From a given timeseries of transactions  compute many lag price response
    (T specifies number of lags).

    """

    if 'norm_size' in df_.columns:
        df_['signed_volume'] = df_['norm_size'] * df_['sign']
    elif 'norm_trade_volume' in df_.columns:
        df_['signed_volume'] = df_['norm_trade_volume'] * df_['sign']
    else:
        df_['signed_volume'] = df_['size']*df_['sign']

    df_agg = df_.groupby(df_.index // T).agg(
        event_timestamp=('event_timestamp', 'first'),
        midprice=('midprice', 'first'),
        vol_imbalance=('signed_volume', 'sum'),
        sign_imbalance=('sign', 'sum'),
        sign=('sign', 'first'),
        daily_R1=('daily_R1', 'first'),
        daily_vol=('daily_vol', 'first'),
        daily_num=('daily_num', 'first'),
        # price_changing=('price_changing', 'first')
    )
    if not log:
        df_agg[response_column] = df_agg['midprice'].diff().shift(-1).fillna(0)
    else:
        df_agg[response_column] = np.log(df_agg['midprice'].shift(-1).fillna(0)) - np.log(df_agg['midprice'])
    return df_agg
