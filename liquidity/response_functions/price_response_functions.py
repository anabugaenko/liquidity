import pandas as pd
import numpy as np


def unconditional_impact(df_: pd.DataFrame, response_column: str = 'R1') -> pd.DataFrame:
    """
    Lag one price response of market orders defined as
    difference in mid-price immediately before subsequent MO
    and the mid-price immediately before the current MO
    aligned by the original MO direction.
    """
    df_['midprice_change'] = df_['midprice'].diff().shift(-1).fillna(0)
    df_[response_column] = df_['midprice_change'] * df_.index.get_level_values('sign')
    return df_


# TODO:
"""
Add response function R(L), R(v, 1) (where we also condition on R(epsilon, 1), price changign and none price changing)
"""


def aggregate_impact(df_: pd.DataFrame, T: int, response_column: str, log=False) -> pd. DataFrame:
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
