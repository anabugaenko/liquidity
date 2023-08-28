import pandas as pd

from liquidity.util.lob_data import load_l3_data, select_trading_hours, select_top_book, select_columns, \
    shift_prices
from liquidity.response_functions.price_response_functions import unconditional_impact
from liquidity.util.util import add_order_sign


def remove_midprice_trades(df_: pd.DataFrame) -> pd.DataFrame:
    mask = df_['execution_price'] == df_['midprice']
    return df_[~mask]


def get_trades_impact(filepath: str, date: str):
    data = load_l3_data(filepath)
    df = select_trading_hours(date, data)
    df = select_top_book(df)
    df = select_columns(df)
    df = shift_prices(df)
    df = remove_midprice_trades(df)
    df = add_order_sign(df)
    ddf = select_executions(df)
    ddf = aggregate_same_ts_events(ddf)
    ddf = unconditional_impact(ddf)
    ddf = normalise_trade_volume(ddf, data)
    return ddf


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
