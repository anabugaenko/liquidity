import numpy as np
import pandas as pd

from liquidity.util.orderbook import rename_orderbook_columns, add_daily_features
from liquidity.util.utils import add_R1, normalise_size, remove_first_daily_prices


def compute_orderbook_states(raw_orderbook_df: pd.DataFrame):
    # R1_ordertype, spread, midprice

    if type(raw_orderbook_df["event_timestamp"].iloc[0]) != pd.Timestamp:
        raw_orderbook_df["event_timestamp"] = raw_orderbook_df["event_timestamp"].apply(lambda x: pd.Timestamp(x))
    data = rename_orderbook_columns(raw_orderbook_df)
    data = add_R1(data)
    data = add_daily_features(data)
    orderbook_states = normalise_size(data)
    return orderbook_states


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


def compute_aggregate_features(df_: pd.DataFrame, T: int) -> pd.DataFrame:
    """
    From a given timeseries of transactions  aggregate different features
    using T sized bins.
    """

    # queue length - sign=("sign", "sum"),
    # volume profile - volume=("norm_size", sum)

    if "norm_size" in df_.columns:
        df_["signed_volume"] = df_["norm_size"] * df_["sign"]
    elif "norm_trade_volume" in df_.columns:
        df_["signed_volume"] = df_["norm_trade_volume"] * df_["sign"]
    else:
        df_["signed_volume"] = df_["size"] * df_["sign"]

    df_agg = df_.groupby(df_.index // T).agg(
        event_timestamp=("event_timestamp", "first"),
        midprice=("midprice", "first"),
        sign=("sign", "first"),
        signed_volume=("signed_volume", "first"),
        vol_imbalance=("signed_volume", "sum"),
        sign_imbalance=("sign", "sum"),
        daily_R1=("daily_R1", "first"),
        daily_vol=("daily_vol", "first"),
        daily_num=("daily_num", "first"),
        # price_changing=('price_changing', 'first')
    )

    return df_agg
