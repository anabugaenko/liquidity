import pandas as pd
import numpy as np

from liquidity.util.utils import smooth_outliers, normalise_by_daily


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


# TODO: rename to conditional aggregate impact
def compute_conditional_aggregate_impact(df: pd.DataFrame, normalise: bool = True, log_prices=False) -> pd.DataFrame:
    """
    Aggregate features
    RN(ΔV, ΔƐ)
    Assumes conditioning on sign dependent variable hence
    do not include sign in price response computation
    """

    # TODO: Implement compute_individual_impact (condition on volume and  sign of previous order)
    data = df.copy()
    if type(data["event_timestamp"].iloc[0]) != pd.Timestamp:
        data["event_timestamp"] = data["event_timestamp"].apply(lambda x: pd.Timestamp(x))

    if not log_prices:
        data["R"] = data["midprice"].diff().shift(-1).fillna(0)
    else:
        data["R"] = np.log(data["midprice"].shift(-1).fillna(0)) - np.log(data["midprice"])

    if normalise:
        data = normalise_by_daily(data)

    data = data[["vol_imbalance", "T", "R"]]

    return data
