import numpy as np
import pandas as pd
from typing import List

from liquidity.util.orderbook import rename_orderbook_columns, add_daily_features
from liquidity.util.utils import add_R1, normalise_size, remove_first_daily_prices
from liquidity.response_functions.price_response_functions import compute_conditional_aggregate_impact


def compute_orderbook_states(raw_orderbook_df: pd.DataFrame):
    # R1_ordertype, spread

    if type(raw_orderbook_df["event_timestamp"].iloc[0]) != pd.Timestamp:
        raw_orderbook_df["event_timestamp"] = raw_orderbook_df["event_timestamp"].apply(lambda x: pd.Timestamp(x))
    data = rename_orderbook_columns(raw_orderbook_df)
    data = add_R1(data)
    data = add_daily_features(data)
    orderbook_states = normalise_size(data)
    return orderbook_states


def compute_aggregate_features(df: pd.DataFrame, durations: List[int], **kwargs) -> pd.DataFrame:
    df["event_timestamp"] = df["event_timestamp"].apply(lambda x: pd.Timestamp(x))
    df["date"] = df["event_timestamp"].apply(lambda x: x.date())
    results_ = []
    for i, T in enumerate(durations):
        lag_data = compute_conditional_aggregate_impact(df, T=T, **kwargs)
        lag_data["R"] = lag_data[f"R{T}"]
        lag_data = lag_data.drop(columns=f"R{T}")
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
