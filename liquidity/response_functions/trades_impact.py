import pandas as pd

from liquidity.response_functions.price_response_functions import add_daily_features, aggregate_response_function
from liquidity.util.data_util import normalise_imbalances, remove_midprice_orders
from liquidity.response_functions.lob_data import select_trading_hours, select_top_book, select_columns, \
    shift_prices

from liquidity.util.util import _remove_outliers


def select_executions(df_: pd.DataFrame) -> pd.DataFrame:
    mask = df_['order_executed']
    return df_[mask]


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


def get_aggregate_trade_response_features(df_: pd.DataFrame,
                                          T: int,
                                          normalise: bool = True,
                                          remove_outliers: bool = False,
                                          log=False) -> pd.DataFrame:
    data = df_.copy()
    data = data.rename(columns={'execution_size': 'size', 'trade_sign': 'sign'})
    data = add_daily_features(data)
    data = aggregate_response_function(data, T=T, response_column=f'R{T}', log=log)
    if remove_outliers:
        data = _remove_outliers(data)
    if normalise:
        data = normalise_imbalances(data)
    return data


def clean_lob_data(date: str, df_raw: pd.DataFrame) -> pd.DataFrame:
    df = select_trading_hours(date, df_raw)
    df = select_top_book(df)
    df = select_columns(df)
    df = shift_prices(df)
    return remove_midprice_orders(df)
