import pandas as pd

from liquidity.util.orderbook import load_l3_data, select_trading_hours, select_top_book, select_columns, shift_prices
from liquidity.response_functions.features import order_signs
from liquidity.util.utils import add_mean_queue_lengths


def remove_midprice_trades(df_: pd.DataFrame) -> pd.DataFrame:
    mask = df_["execution_price"] == df_["midprice"]
    return df_[~mask]


def get_trades_data(filepath: str, date: str):
    data = load_l3_data(filepath)
    df = add_mean_queue_lengths(data)
    df = select_trading_hours(date, df)
    df = select_top_book(df)
    df = select_columns(df)
    df = shift_prices(df)
    df = remove_midprice_trades(df)
    df = order_signs(df)
    ddf = select_executions(df)
    ddf = aggregate_same_ts_events(ddf)
    ddf = ddf.reset_index()
    return ddf


def select_executions(df_: pd.DataFrame) -> pd.DataFrame:
    mask = df_["order_executed"]
    return df_[mask]


def aggregate_same_ts_events(df_: pd.DataFrame) -> pd.DataFrame:
    """
    In an LOB one MO that matched several LOs is represented by multiple events
    so we merge these to reconstruct properties of the original MO.
    """
    df_ = df_.groupby(["event_timestamp", "sign"]).agg(
        {
            "side": "last",
            "lob_action": "last",
            "order_executed": "all",
            "execution_price": "last",
            "execution_size": "sum",
            "ask": "last",
            "bid": "last",
            "midprice": "last",
            "ask_volume": "first",
            "bid_volume": "first",
            "price_changing": "last",
            "ask_queue_size_mean": "first",
            "bid_queue_size_mean": "first",
        }
    )
    return df_
