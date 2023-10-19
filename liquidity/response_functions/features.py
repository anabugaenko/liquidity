from typing import List

import numpy as np
import pandas as pd

from liquidity.util.orderbook import rename_orderbook_columns
from liquidity.util.utils import remove_first_daily_prices


def daily_orderbook_states(df_: pd.DataFrame, response_column: str = "R1") -> pd.DataFrame:
    """
    From a given time series of transactions add daily means of lag one price response R1
    and order size (to be used as denominator in normalisation).
    """
    if type(df_["event_timestamp"].iloc[0]) != pd.Timestamp:
        df_["event_timestamp"] = df_["event_timestamp"].apply(lambda x: pd.Timestamp(x))
    df_["date"] = df_["event_timestamp"].apply(lambda x: x.date())

    daily_R1 = df_[[response_column, "date"]].groupby("date").agg(daily_R1=(response_column, "mean"))
    daily_volume = df_[["size", "date"]].groupby("date").agg(daily_vol=("size", "sum"))
    daily_num = df_[["size", "date"]].groupby("date").agg(daily_num=("size", "count"))

    df_["daily_R1"] = daily_R1.reindex(index=df_["event_timestamp"], method="ffill").values
    df_["daily_vol"] = daily_volume.reindex(index=df_["event_timestamp"], method="ffill").values
    df_["daily_num"] = daily_num.reindex(index=df_["event_timestamp"], method="ffill").values

    return df_

def compute_orderbook_states(raw_orderbook_df: pd.DataFrame):

    if type(raw_orderbook_df["event_timestamp"].iloc[0]) != pd.Timestamp:
        raw_orderbook_df["event_timestamp"] = raw_orderbook_df["event_timestamp"].apply(lambda x: pd.Timestamp(x))
    data = rename_orderbook_columns(raw_orderbook_df)  # TODO: might not be needed anymore
    data = add_R1(data)
    data = add_spead(data)
    data = add_signed_volume(data)
    orderbook_states = daily_orderbook_states(data)
    return orderbook_states


def compute_intraday_features(df_: pd.DataFrame, T: int) -> pd.DataFrame:
    """
    From a given timeseries of transactions  aggregate different features
    using T sized bins.
    """
    #TODO: remove
    if "signed_volume" not in df_.columns:
        df_["signed_volume"] = df_["size"] * df_["sign"]

    df_agg = df_.groupby(df_.index // T).agg(
        event_timestamp=("event_timestamp", "first"),
        midprice=("midprice", "first"),
        sign=("sign", "first"),
        signed_volume=("signed_volume", "first"),
        price_changing=('price_changing', 'first'),

        # Market depth
        # queue_length=("size", "count"),
        # volume_profile=("size", "sum"),

        # Imbalances
        vol_imbalance=("signed_volume", "sum"),
        sign_imbalance=("sign", "sum"),
        price_changing_imbalance=('price_changing', 'sum'),

        # Daily features
        daily_R1=("daily_R1", "first"),
        daily_vol=("daily_vol", "first"),
        daily_num=("daily_num", "first"),

    )

    return df_agg


def compute_aggregate_features(df: pd.DataFrame, durations: List[int], **kwargs) -> pd.DataFrame:
    df["event_timestamp"] = df["event_timestamp"].apply(lambda x: pd.Timestamp(x))
    df["date"] = df["event_timestamp"].apply(lambda x: x.date())
    results_ = []
    for i, T in enumerate(durations):
        lag_data = compute_intraday_features(df, T=T)
        lag_data["T"] = T
        results_.append(lag_data)

    return pd.concat(results_)


def compute_returns(df: pd.DataFrame, remove_first: bool = True) -> pd.DataFrame:
    """
    Compute various representations of returns for a given DataFrame.

    Parameters:
    - df (pd.DataFrame): Input dataframe with a 'midprice' column and 'event_timestamp' column.
    - remove_first (bool, optional): Flag to indicate whether to remove the first daily price. Defaults to True.

    Returns:
    - pd.DataFrame: DataFrame with added columns for different return representations.
    """
    df = df.copy()

    if type(df["event_timestamp"].iloc[0]) != pd.Timestamp:
        df.loc[:, "event_timestamp"] = df["event_timestamp"].apply(lambda x: pd.Timestamp(x))

    if remove_first:
        df = remove_first_daily_prices(df)

    # Absolute returns
    df["returns"] = df["midprice"].diff()

    # Percentage (relative) returns
    # df["pct_returns"] = (df["midprice"] / df["midprice"].shift(1)) - 1 # using numpy's pct_change equivalent for robustness
    df["pct_returns"] = df["midprice"].pct_change()

    # Other representations of returns
    # Remove any NaN or infinite values from 'returns'
    df = df[~df["returns"].isin([np.nan, np.inf, -np.inf])]

    # Time-varying variance derived directly from returns
    df["variance"] = df["returns"] ** 2

    # Volatility (return magnitudes - time-varying standard deviation derived from variance)
    df["volatility"] = np.sqrt(df["variance"])

    # Log returns
    df["log_returns"] = np.log(df["midprice"]) - np.log(df["midprice"].shift(1))

    return df


def add_R1(df_: pd.DataFrame) -> pd.DataFrame:
    # R(1)
    df_["midprice_change"] = df_["midprice"].diff().shift(-1).fillna(0)
    df_["R1"] = df_["midprice_change"] * df_["sign"]
    return df_


def add_spead(df_: pd.DataFrame) -> pd.DataFrame:
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


def add_signed_volume(df_: pd.DataFrame) -> pd.DataFrame:
    df_["signed_volume"] = df_["size"] * df_["sign"]
    return df_


def add_order_signs(df_: pd.DataFrame) -> pd.DataFrame:
    def _ennumerate_sides(row):
        return 1 if row["side"] == "ASK" else -1

    df_["sign"] = df_.apply(lambda row: _ennumerate_sides(row), axis=1)
    return df_
