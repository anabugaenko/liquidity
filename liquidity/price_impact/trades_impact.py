import pandas as pd
import numpy as np

from liquidity.price_impact.price_response import add_daily_features, get_aggregate_response, \
    _normalise_features
from liquidity.price_impact.load_data import load_l3_data, select_trading_hours, select_top_book, select_columns, \
    shift_prices
from scipy import stats

from liquidity.price_impact.util import numerate_side


def remove_midprice_trades(df_: pd.DataFrame) -> pd.DataFrame:
    mask = df_['execution_price'] == df_['midprice']
    return df_[~mask]


def select_executions(df_: pd.DataFrame) -> pd.DataFrame:
    mask = df_['order_executed']
    return df_[mask]


def add_order_sign(df_: pd.DataFrame) -> pd.DataFrame:
    df_['sign'] = df_.apply(lambda row: numerate_side(row), axis=1)
    return df_


def aggregate_same_ts_events(df_: pd.DataFrame) -> pd.DataFrame:
    """
    In an LOB one MO that matched several LOs is represented by multiple events
    so we merge these to reconstruct properties of the original MO.
    """
    df_ = df_.groupby(['event_timestamp', 'sign']).agg({
        'side': 'last',
        'lob_action': 'last',
        'order_executed': 'all',
        'execution_price': 'last',
        'execution_size': 'sum',
        'ask': 'last',
        'bid': 'last',
        'midprice': 'last',
        'ask_volume': 'first',
        'bid_volume': 'first',
        'price_changing': 'last',
    })
    return df_


def add_price_response(df_: pd.DataFrame, response_column: str = 'R1') -> pd.DataFrame:
    """
    Lag one price response of market orders defined as
    difference in mid-price immediately before subsequent MO
    and the mid-price immediately before the current MO
    aligned by the original MO direction.
    """
    df_['midprice_change'] = df_['midprice'].diff().shift(-1).fillna(0)
    df_[response_column] = df_['midprice_change'] * df_.index.get_level_values('sign')
    return df_


def normalise_trade_volume(df_: pd.DataFrame, lob_data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise trade size by the average volume on the same side best quote.
    """
    ask_mean_vol = lob_data['best_ask_size'].mean()
    bid_mean_vol = lob_data['best_bid_size'].mean()

    def _normalise(row):
        if row['side'] == 'ASK':
            return row['execution_size'] / ask_mean_vol
        else:
            return row['execution_size'] / bid_mean_vol

    df_['norm_trade_volume'] = df_.apply(_normalise, axis=1)
    return df_


def get_daily_trades_with_impact(filepath: str, date: str):
    data = load_l3_data(filepath)
    df = select_trading_hours(date, data)
    df = select_top_book(df)
    df = select_columns(df)
    df = shift_prices(df)
    df = remove_midprice_trades(df)
    df = add_order_sign(df)
    ddf = select_executions(df)
    ddf = aggregate_same_ts_events(ddf)
    ddf = add_price_response(ddf)
    ddf = normalise_trade_volume(ddf, data)
    return ddf


def _remove_outliers(df: pd.DataFrame, T: int) -> pd.DataFrame:
    """
    Remove observations where volume imbalance or price response
    value was beyond three standard deviations of it's mean.
    """
    # z = np.abs(stats.zscore(df[['vol_imbalance', f'R{T}']]))
    # df = df[(z < 3).all(axis=1)]

    def winsorize_queue(s: pd.Series, level: int = 3) -> pd.Series:
        upper_bound = level * s.std()
        return s.clip(upper=upper_bound)

    queue_columns = ['vol_imbalance', 'sign_imbalance']

    for name in queue_columns:
        s = df[name]
        df[name] = winsorize_queue(s)

    return df


def get_aggregate_trade_response_features(df_: pd.DataFrame,
                                          T: int,
                                          normalise: bool = True,
                                          remove_outliers: bool = False) -> pd.DataFrame:
    data = df_.copy()
    data = data.rename(columns={'execution_size': 'size', 'trade_sign': 'sign'})
    data = add_daily_features(data)
    data = get_aggregate_response(data, T=T, response_column=f'R{T}')
    if remove_outliers:
        data = _remove_outliers(data, T=T)
    if normalise:
        data = _normalise_features(data, response_column=f'R{T}')
    return data


def clean_lob_data(date: str, df_raw: pd.DataFrame) -> pd.DataFrame:
    df = select_trading_hours(date, df_raw)
    df = select_top_book(df)
    df = select_columns(df)
    df = shift_prices(df)
    return remove_midprice_trades(df)
