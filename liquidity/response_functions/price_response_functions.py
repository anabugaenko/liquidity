import pandas as pd
import numpy as np

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


def _conditional_aggregate_impact(df_: pd.DataFrame, T: int, response_column: str, log_prices=False) -> pd.DataFrame:
    """
    RN(ΔV, ΔƐ)
    At the moment assumes conditioning on sign dependent variable

    From a given timeseries of transactions  compute many lag price response
    (T specifies number of lags).

    TODO: Implement price changing and none price changing flag for R(v, 1) and R(epsilon, 1),

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
    if not log_prices:
        df_agg[response_column] = df_agg["midprice"].diff().shift(-1).fillna(0)  # FIXME: excludes the sign atm
    else:
        df_agg[response_column] = np.log(df_agg["midprice"].shift(-1).fillna(0)) - np.log(df_agg["midprice"])
    return df_agg


def compute_conditional_aggregate_impact(
    df: pd.DataFrame, T: int, normalise: bool = True, remove_outliers: bool = True, log_prices=False
) -> pd.DataFrame:
    """
    Aggregate features
    RN(ΔV, ΔƐ)
    """

    def _normalise_axis(df: pd.DataFrame) -> pd.DataFrame:
        if "vol_imbalance" in df.columns:
            df["vol_imbalance"] = df["vol_imbalance"] / df["daily_vol"]
        if "sign_imbalance" in df.columns:
            df["sign_imbalance"] = df["sign_imbalance"] / df["daily_num"]
        if "R" in df.columns:
            df["R"] = df["R"] / df["daily_R1"]
        return df

    # TODO: Implement compute_individual_impact (condition on volume and  sign of previous order)
    data = df.copy()
    if type(data["event_timestamp"].iloc[0]) != pd.Timestamp:
        data["event_timestamp"] = data["event_timestamp"].apply(lambda x: pd.Timestamp(x))
    data = rename_orderbook_columns(data)
    data = add_daily_features(data)
    data = _conditional_aggregate_impact(data, T=T, response_column=f"R{T}", log_prices=log_prices)

    if normalise:
        data = _normalise_axis(data)

    if remove_outliers:
        data = smooth_outliers(data, T=T)
    return data
