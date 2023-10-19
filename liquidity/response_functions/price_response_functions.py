import pandas as pd
import numpy as np

from liquidity.util.utils import normalise_imbalances, smooth_outliers


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
    data_agg: pd.DataFrame, normalise: bool = True, remove_outliers: bool = True, log_prices=False
) -> pd.DataFrame:
    """
    RN(ΔV, ΔƐ)
    At the moment assumes conditioning on sign dependent variable

    TODO: Implement price changing and none price changing flag for R(v, 1) and R(epsilon, 1)
    TODO: Implement compute_individual_impact: condition on volume and  sign of previous order
    """

    response_column = "R"
    if not log_prices:
        data_agg[response_column] = data_agg["midprice"].diff().shift(-1).fillna(0)
    else:
        data_agg[response_column] = np.log(data_agg["midprice"].shift(-1).fillna(0)) - np.log(data_agg["midprice"])

    if normalise:
        data_agg = normalise_imbalances(data_agg)

    if remove_outliers:
        data_agg = smooth_outliers(data_agg)
    return data_agg
