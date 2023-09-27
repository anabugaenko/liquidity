import pandas as pd

from liquidity.util.utils import remove_midprice_orders, rename_columns, add_order_signs
from liquidity.util.orderbook import (
    load_l3_data,
    select_trading_hours,
    select_top_book,
    select_columns,
    shift_prices,
    select_best_quotes,
)
from liquidity.response_functions.price_response_functions import add_price_response
from liquidity.util.trades_data_util import remove_midprice_trades


def get_lo_impact(filepath: str, date: str) -> pd.DataFrame:
    """
    Loads LOB events timeseries for a day from a file and
    returns a DataFrame of LO arrivals timeseries.
    :param filepath:
    :param date:
    :return:
    """
    data = load_l3_data(filepath)
    df = select_trading_hours(date, data)
    df = select_top_book(df)
    df = select_columns(df)
    df = shift_prices(df)
    df = remove_midprice_orders(df)
    df = add_order_signs(df)
    df = select_lo_inserts(df)
    df = add_price_response(df, response_column="R1_LO")
    df = normalise_lo_sizes(df)
    return df


def get_ca_impact(filepath: str, date: str) -> pd.DataFrame:
    data = load_l3_data(filepath)
    df = select_trading_hours(date, data)
    df = select_top_book(df)
    df = select_columns(df)
    df = shift_prices(df)
    df = remove_midprice_orders(df)
    df = add_order_signs(df)
    df = select_cancellations(df)
    df = df.drop(["price", "size"], axis=1)
    df = rename_columns(df)
    df = add_price_response(df, response_column="R1_CA")
    df = normalise_lo_sizes(df)
    return df


def get_qa_impact(raw_daily_df: pd.DataFrame, date: str) -> pd.DataFrame:
    df = select_trading_hours(date, raw_daily_df)
    df = select_best_quotes(df)
    df = select_columns(df)
    df = shift_prices(df)
    df = remove_midprice_orders(df)
    df = remove_midprice_trades(df)
    df = add_order_signs(df)
    df = df.groupby(["event_timestamp"]).last()
    df = df.reset_index()
    df = add_price_response(df)
    return df


def select_lo_inserts(df_: pd.DataFrame) -> pd.DataFrame:
    # check for updates that increased volume - hidden orders?
    m1 = df_["old_size"] < df_["size"]
    m2 = df_["lob_action"] == "UPDATE"
    if not df_[m1 & m2].shape[0] == 0:
        print("Found and removed order updates that increased size \n", df_[m1 & m2][["size", "old_size"]])
        df_ = df_[~(m1 & m2)]

    mask1 = df_["lob_action"] == "INSERT"
    mask2 = df_["lob_action"] == "UPDATE"
    mask2 = mask2 & (df_["price_changing"] == True)

    return df_[mask1 | mask2]


def normalise_lo_sizes(df_: pd.DataFrame) -> pd.DataFrame:
    ask_mean_size = df_[df_["side"] == "ASK"]["size"].mean()
    bid_mean_size = df_[df_["side"] == "BID"]["size"].mean()

    def _normalise(row):
        if row["side"] == "ASK":
            return row["size"] / ask_mean_size
        else:
            return row["size"] / bid_mean_size

    df_["norm_size"] = df_.apply(_normalise, axis=1)
    return df_


def select_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    mask1 = df["lob_action"] == "REMOVE"
    mask2 = df["order_executed"] == False
    mask3 = df["old_price_level"] == 1
    mask_complete_removals = mask1 & mask2 & mask3

    mask4 = df["lob_action"] == "UPDATE"
    mask5 = df["order_executed"] == False
    mask6 = df["old_price_level"] == 1
    mask7 = df["size"] < df["old_size"]
    mask_partial_removals = mask4 & mask5 & mask6 & mask7

    return df[mask_complete_removals | mask_partial_removals]
