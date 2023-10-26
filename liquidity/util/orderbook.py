import pandas as pd

from liquidity.util.utils import remove_midprice_orders

UCT_OFFSET = 2


def load_l3_data(filepath: str) -> pd.DataFrame:
    """
    Returns DataFrame of daily incremental Level 3 order book data.
    Example code below shows how to load the data on BMLL platform.
    """

    """
    This is how the data was retrieved in BMLL DataLab
    
    lid - str representing unique listing identifier on BMLL platform
    date - str 
    
    nsec = NormalisedSecurity.from_listing_id(lid, date)
    mkt_dat = nsec.market_data(feed='L3')
    return mkt_dat.incremental_book_L3()
    """

    return pd.read_csv(filepath, header=0, index_col=0)


def select_trading_hours(date: str, df_: pd.DataFrame, utc: bool = False) -> pd.DataFrame:
    """
    Select only non-auction trading hours: 10:30 am and 3:00 pm.
    """
    if utc:
        utc_offset = UCT_OFFSET
        dt = pd.Timedelta(utc_offset, unit="m")
    else:
        dt = pd.Timedelta(0, unit="m")
    ts1 = pd.Timestamp(date + " 10:30") - dt
    ts2 = pd.Timestamp(date + " 15:00") - dt
    df_["event_timestamp"] = df_["event_timestamp"].apply(lambda x: pd.Timestamp(x))
    mask = df_["event_timestamp"].between(ts1, ts2)
    df_ = df_[mask]
    df_["event_timestamp"] = df_["event_timestamp"] + dt
    return df_


def select_top_book(df_: pd.DataFrame) -> pd.DataFrame:
    """
    Select top level of order book events (level 1)
    or orders that arrive at a better price (level 0).
    Level 0:
    - for arriving LO: new best price
    - for MO: level no longer exists after trade
    - for CA: level no longer exists after removal
    """
    mask = (df_.price_level == 1) | (df_.price_level == 0)
    return df_[mask]


def select_columns(df_: pd.DataFrame) -> pd.DataFrame:
    """
    Select only relevant info about each event.
    """
    sub_columns_list = [
        "event_timestamp",
        "side",
        "lob_action",
        "order_executed",
        "execution_price",
        "execution_size",
        "best_ask_price",
        "best_ask_size",
        "best_bid_price",
        "best_bid_size",
        "is_new_best_price",
        "price",
        "price_level",
        "old_price",
        "old_price_level",
        "size",
        "old_size",
        "best_ask_num_orders",
        "best_bid_num_orders",
        "ask_queue_size_mean",
        "bid_queue_size_mean",
    ]

    df_ = df_[sub_columns_list]
    df_ = df_.rename(
        columns={
            "best_ask_price": "ask",
            "best_bid_price": "bid",
            "best_ask_size": "ask_volume",
            "best_bid_size": "bid_volume",
            "best_ask_num_orders": "ask_count",
            "best_bid_num_orders": "bid_count",
            "is_new_best_price": "price_changing",
        }
    )

    df_["midprice"] = (df_["ask"] + df_["bid"]) * 0.5
    return df_


def shift_prices(df_: pd.DataFrame) -> pd.DataFrame:
    """
    This transformation is specific to how BMLL offer orderbook view
    where each event is accompanied by respective price value after the event has taken place.
    We're interested in how the price changed so shifting it to get values of
    mid-price, bid and ask immediately before each event.
    """
    df_["midprice"] = df_["midprice"].shift().fillna(0)
    df_["ask"] = df_["ask"].shift().fillna(0)
    df_["bid"] = df_["bid"].shift().fillna(0)
    df_["ask_volume"] = df_["ask_volume"].shift().fillna(0)
    df_["bid_volume"] = df_["bid_volume"].shift().fillna(0)
    return df_


def select_best_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Need to carefully select only events that affected top book level.
    By definition selecting events at level 0 and 1 is not accurate for cancellation orders
    since the level is set to 0 if the removal caused a price level to no longer exist.
    """
    price_level_mask = (df.price_level == 1) | (df.price_level == 0)
    old_price_level_mask = (df.old_price_level == 1) | (df.old_price_level == 0)
    return df[price_level_mask & old_price_level_mask]


def clean_lob_data(date: str, df_raw: pd.DataFrame) -> pd.DataFrame:
    df = select_trading_hours(date, df_raw)
    df = select_top_book(df)
    df = select_columns(df)
    df = shift_prices(df)
    return remove_midprice_orders(df)


# def normalise_all_sizes(df_: pd.DataFrame):
#     """
#     if execution -> execution_size
#     if insert/LO -> size
#     if cancel/remove -> old size
#     """
#
#     def _select_size_for_order_type(row):
#         mask1 = row["lob_action"] == "INSERT"
#         mask2 = row["lob_action"] == "UPDATE"
#         mask2 = mask2 & (row["price_changing"] == True)
#         lo_mask = mask1 | mask2
#
#         if lo_mask:
#             return row["size"]
#
#         mo_mask = row["order_executed"]
#
#         if mo_mask:
#             return row["execution_size"]
#
#         mask1 = row["lob_action"] == "REMOVE"
#         mask2 = row["order_executed"] == False
#         mask3 = row["old_price_level"] == 1
#         mask_complete_removals = mask1 & mask2 & mask3
#
#         mask4 = row["lob_action"] == "UPDATE"
#         mask5 = row["order_executed"] == False
#         mask6 = row["old_price_level"] == 1
#         mask7 = row["size"] < row["old_size"]
#         mask_partial_removals = mask4 & mask5 & mask6 & mask7
#         ca_mask = mask_complete_removals | mask_partial_removals
#
#         if ca_mask:
#             return row["old_size"]
#
#         return 0
#
#     df_["new_size"] = df_.apply(lambda row: _select_size_for_order_type(row), axis=1)
#     df_ = df_[~(df_["new_size"] == 0)]
#
#     ask_mean_size = df_[df_["side"] == "ASK"]["size"].mean()
#     bid_mean_size = df_[df_["side"] == "BID"]["size"].mean()
#
#     def _normalise(row):
#         if row["side"] == "ASK":
#             return row["new_size"] / ask_mean_size
#         else:
#             return row["new_size"] / bid_mean_size
#
#     df_["norm_size"] = df_.apply(_normalise, axis=1)
#
#     return df_


def rename_orderbook_columns(df_: pd.DataFrame) -> pd.DataFrame:
    df_columns = df_.columns

    if "old_price" in df_columns and "old_size" in df_columns:
        df_ = df_.drop(["price", "size"], axis=1)
        df_ = df_.rename(columns={"old_price": "price", "old_size": "size"})

    if "R1_CA" in df_columns:
        df_ = df_.rename(columns={"R1_CA": "R1"})

    if "R1_LO" in df_columns:
        df_ = df_.rename(columns={"R1_LO": "R1"})

    if "execution_size" in df_columns:
        df_ = df_.rename(columns={"execution_size": "size"})

    if "trade_sign" in df_columns:
        df_ = df_.rename(columns={"trade_sign": "sign"})

    return df_
