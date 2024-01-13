import numpy as np
import pandas as pd
from typing import List

from liquidity.util.utils import (
    normalise_size,
    remove_first_daily_prices,
)
from market_impact.response_functions import price_response
from liquidity.util.orderbook import rename_orderbook_columns


def signed_volume(df_: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the signed volume of an order given raw L3 market data
    """
    df_["signed_volume"] = df_["size"] * df_["sign"]
    return df_


def order_signs(df_: pd.DataFrame) -> pd.DataFrame:
    """
    Adds numeric sign distinguish the side of the book an event occurred, for which we introduce
    another pair of signs, -1 and +1 for the bid-side and ask-side events repsectively.
    """

    def _ennumerate_sides(row):
        return 1 if row["side"] == "ASK" else -1

    df_["sign"] = df_.apply(lambda row: _ennumerate_sides(row), axis=1)
    return df_


def bid_ask_spead(df_: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the spread s(t) ..math::
        s(t) = a(t) - b(t),
    which can be measured in several ways:
       - At random instances in calendar time,
       - At random instances in event-time,
       - immediately before an event, e.g., execution, limit order placement etc.
    Here, we compute the spread in inter-event time, which corresponds to the latter case.
    """

    df_["spread"] = df_["ask"] - df_["bid"]

    return df_


def mean_queue_lengths(lob_data: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the mean number and volume profile at the best quotes.
    """

    def _get_best_num(row):
        if row["side"] == "ASK":
            return ask_num_mean
        else:
            return bid_num_mean

    def _get_best_vol(row):
        if row["side"] == "ASK":
            return ask_size_mean
        else:
            return bid_size_mean

    # Compute average queue lenght at best
    ask_num_mean = lob_data["best_ask_num_orders"].mean()
    bid_num_mean = lob_data["best_bid_num_orders"].mean()
    lob_data["ask_queue_number_mean"] = ask_num_mean
    lob_data["bid_queue_number_mean"] = bid_num_mean

    # Compute average volume profile at best quotes
    ask_size_mean = lob_data["best_ask_size"].mean()
    bid_size_mean = lob_data["best_bid_size"].mean()
    lob_data["ask_queue_size_mean"] = ask_size_mean
    lob_data["bid_queue_size_mean"] = bid_size_mean

    # Average number and volume profile at best quotes
    lob_data["average_num_at_best"] = lob_data.apply(_get_best_num, axis=1)
    lob_data["average_vol_at_best"] = lob_data.apply(_get_best_vol, axis=1)

    return lob_data


def daily_orderbook_states(
    df_: pd.DataFrame, response_column: str = "R1"
) -> pd.DataFrame:
    """
    From a given time series of transactions add daily means of lag one price response R1
    and order size (to be used as denominator in normalisation).
    """
    if type(df_["event_timestamp"].iloc[0]) != pd.Timestamp:
        df_["event_timestamp"] = df_["event_timestamp"].apply(lambda x: pd.Timestamp(x))
    df_["date"] = df_["event_timestamp"].apply(lambda x: x.date())

    daily_R1 = (
        df_[[response_column, "date"]]
        .groupby("date")
        .agg(daily_R1=(response_column, "mean"))
    )
    daily_volume = df_[["size", "date"]].groupby("date").agg(daily_vol=("size", "sum"))
    daily_num = df_[["size", "date"]].groupby("date").agg(daily_num=("size", "count"))

    df_["daily_R1"] = daily_R1.reindex(
        index=df_["event_timestamp"], method="ffill"
    ).values
    df_["daily_vol"] = daily_volume.reindex(
        index=df_["event_timestamp"], method="ffill"
    ).values
    df_["daily_num"] = daily_num.reindex(
        index=df_["event_timestamp"], method="ffill"
    ).values

    return df_


def compute_orderbook_states(raw_orderbook_df: pd.DataFrame):
    """
    Computes limit order book (LOB) states from raw or transformed time series of L3 order book data.
    """
    raw_orderbook_df = raw_orderbook_df.copy()

    if type(raw_orderbook_df["event_timestamp"].iloc[0]) != pd.Timestamp:
        raw_orderbook_df["event_timestamp"] = raw_orderbook_df["event_timestamp"].apply(
            lambda x: pd.Timestamp(x)
        )
    data = rename_orderbook_columns(raw_orderbook_df)
    data = price_response(data)
    data = bid_ask_spead(data)
    orderbook_states = daily_orderbook_states(data)
    orderbook_states = normalise_size(orderbook_states)
    orderbook_states = signed_volume(orderbook_states)

    return orderbook_states


def compute_intraday_features(
    orderbook_states_df_: pd.DataFrame, bin_size: int, normalize: bool = True
) -> pd.DataFrame:
    """
    Compute intra-day features for different order types using T sized bins.
    """
    data = orderbook_states_df_.copy()

    intraday_features = data.groupby(data.index // bin_size).agg(
        # Orderbook states
        event_timestamp=("event_timestamp", "first"),
        midprice=("midprice", "first"),
        sign=("sign", "first"),
        signed_volume=("signed_volume", "first"),
        price_changing=("price_changing", "first"),
        # Daily features
        daily_R1=("daily_R1", "first"),
        daily_num=("daily_num", "first"),
        daily_vol=("daily_vol", "first"),
        # Imbalances
        volume_imbalance=("signed_volume", "sum"),
        sign_imbalance=("sign", "sum"),
        price_change_imbalance=("price_changing", "sum"),
        # Average queue length and volume profile at best quotes
        average_num_at_best=("average_num_at_best", "first"),
        average_vol_at_best=("average_vol_at_best", "first"),
    )

    return intraday_features


def compute_aggregate_features(
    orderbook_states: pd.DataFrame, bin_frequencies: List[int]
) -> pd.DataFrame:
    """
    Compute aggregate features for different order types using T sized binning frequencies.
    """
    # TODO: add option to normalize column by column via kwargs

    data = orderbook_states.copy()

    data["event_timestamp"] = data["event_timestamp"].apply(lambda x: pd.Timestamp(x))
    data["date"] = data["event_timestamp"].apply(lambda x: x.date())
    results_ = []
    for i, bin_size in enumerate(bin_frequencies):
        lag_data = compute_intraday_features(data, bin_size=bin_size, normalize=True)
        lag_data["T"] = bin_size
        results_.append(lag_data)

    # Aggregate features
    aggregate_features = pd.concat(results_)

    return aggregate_features


def compute_returns(
    orderbook_states: pd.DataFrame, remove_first: bool = True
) -> pd.DataFrame:
    """
    Compute various representations of returns for a given order book states and prices.

    Parameters:
    - orderbook_States_df (pd.DataFrame): Input dataframe with a 'event_timestamp' and 'midprice' column.
    - remove_first (bool, optional): Flag to indicate whether to remove the first daily price. Defaults to True.

    Returns:
    - pd.DataFrame: DataFrame with added series of different representations of returns.
    """
    data = orderbook_states.copy()

    if type(data["event_timestamp"].iloc[0]) != pd.Timestamp:
        data.loc[:, "event_timestamp"] = data["event_timestamp"].apply(
            lambda x: pd.Timestamp(x)
        )

    # Fractional returns
    data["returns"] = data["midprice"].diff()

    # Drop first daily price
    if remove_first:
        data = remove_first_daily_prices(data)

    # Percentage (relative) returns
    # df["pct_returns"] = (df["midprice"] / df["midprice"].shift(1)) - 1 # using numpy's pct_change equivalent for robustness
    data["pct_returns"] = data["midprice"].pct_change()

    # Other representations of returns
    # Remove any NaN or infinite values from 'returns'
    data = data[~data["returns"].isin([np.nan, np.inf, -np.inf])]

    # Log returns
    data["log_returns"] = np.log(data["midprice"]) - np.log(data["midprice"].shift(1))

    # Volatility (return magnitudes)
    data["variance"] = (
        data["returns"] ** 2
    )  # time-varying standard deviation derived directly from returns
    data["volatility"] = np.sqrt(data["variance"])

    return data
