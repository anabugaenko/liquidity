import pandas as pd
import numpy as np

from liquidity.response_functions.features import compute_aggregate_features
from liquidity.util.utils import smooth_outliers
from liquidity.util.orderbook import add_daily_features, rename_orderbook_columns


def _price_response_function(df_: pd.DataFrame, lag: int = 1, log_prices=False) -> pd.DataFrame:
    """
    : math :
    R(l):
        Lag one price response of market orders defined as difference in mid-price immediately before subsequent MO
        and the mid-price immediately before the current MO aligned by the original MO direction.
    """
    response_column = f"R{lag}"
    df_agg = df_.groupby(df_.index // lag).agg(
        midprice=("midprice", "first"),
        sign=("sign", "first"),
        daily_R1=("daily_R1", "first"),
    )
    if not log_prices:
        df_agg[response_column] = df_agg["midprice"].diff().shift(-1).fillna(0)
    else:
        df_agg[response_column] = np.log(df_agg["midprice"].shift(-1).fillna(0)) - np.log(df_agg["midprice"])
    return df_agg


def compute_price_response(
    df: pd.DataFrame, lag: int = 1, normalise: bool = True, remove_outliers: bool = True, log_prices=False
) -> pd.DataFrame:
    """
    R(l) interface
    """

    data = _price_response_function(df, lag=lag, log_prices=log_prices)
    if remove_outliers:
        data = smooth_outliers(data)
    if normalise:
        data[f"R{lag}"] = data[f"R{lag}"] / data["daily_R1"]
    return data


def compute_conditional_aggregate_impact(
    df: pd.DataFrame, T: int, normalise: bool = True, remove_outliers: bool = True, log_prices=False
) -> pd.DataFrame:
    """
    RN(ΔV, ΔƐ)
    At the moment assumes conditioning on sign dependent variable

    TODO: Implement price changing and none price changing flag for R(v, 1) and R(epsilon, 1)
    TODO: Implement compute_individual_impact: condition on volume and  sign of previous order
    """

    def _normalise_axis(df: pd.DataFrame) -> pd.DataFrame:
        if "vol_imbalance" in df.columns:
            df["vol_imbalance"] = df["vol_imbalance"] / df["daily_vol"]
        if "sign_imbalance" in df.columns:
            df["sign_imbalance"] = df["sign_imbalance"] / df["daily_num"]
        if "R" in df.columns:
            df["R"] = df["R"] / df["daily_R1"]
        return df


    data = df.copy()
    if type(data["event_timestamp"].iloc[0]) != pd.Timestamp:
        data["event_timestamp"] = data["event_timestamp"].apply(lambda x: pd.Timestamp(x))
    data = rename_orderbook_columns(data)
    data = add_daily_features(data)
    data_agg = compute_aggregate_features(data, T=T)

    response_column = f"R{T}"

    if not log_prices:
        data_agg[response_column] = data_agg["midprice"].diff().shift(-1).fillna(0)  # FIXME: excludes the sign atm
    else:
        data_agg[response_column] = np.log(data_agg["midprice"].shift(-1).fillna(0)) - np.log(data_agg["midprice"])

    if normalise:
        data_agg = _normalise_axis(data_agg)

    if remove_outliers:
        data_agg = smooth_outliers(data_agg, T=T)
    return data_agg
