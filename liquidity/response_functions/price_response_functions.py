import pandas as pd
import numpy as np
from scipy import stats


# TODO: reconcile
def add_price_response(df_: pd.DataFrame, response_column: str = 'R1') -> pd.DataFrame:
    df_['midprice_change'] = df_['midprice'].diff().shift(-1).fillna(0)
    df_[response_column] = df_['midprice_change'] * df_['sign']
    return df_

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


def _conditional_aggregate_impact(df_: pd.DataFrame, T: int, response_column: str, log_prices=False) -> pd.DataFrame:
    """

    At the moment assumes conditioning on sign dependent variable

    From a given timeseries of transactions  compute many lag price response
    (T specifies number of lags).

    TODO: Implement price changing and none price changing flag for R(v, 1) and R(epsilon, 1),

    """

    if "norm_size" in df_.columns:
        df_["signed_volume"] = df_["norm_size"] * df_["sign"]
    elif "norm_trade_volume" in df_.columns:
        df_["signed_volume"] = df_["norm_trade_volume"] * df_["sign"]
    else:
        df_["signed_volume"] = df_["size"] * df_["sign"]

    df_agg = df_.groupby(df_.index // T).agg(
        event_timestamp=("event_timestamp", "first"),
        midprice=("midprice", "first"),
        vol_imbalance=("signed_volume", "sum"),
        sign_imbalance=("sign", "sum"),
        sign=("sign", "first"),
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


def add_daily_features(df_: pd.DataFrame, response_column: str = "R1") -> pd.DataFrame:
    """
    From a given time series of transactions add daily means of
    lag one price response R1 and order size.
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


def normalise_price_response(df: pd.DataFrame, response_column: str) -> pd.DataFrame:
    df[response_column] = df[response_column] / df["daily_R1"] * df["daily_R1"].mean()
    return df


def normalise_axis(df: pd.DataFrame) -> pd.DataFrame:
    if "vol_imbalance" in df.columns:
        df["vol_imbalance"] = df["vol_imbalance"] / df["daily_vol"]
    if "sign_imbalance" in df.columns:
        df["sign_imbalance"] = df["sign_imbalance"] / df["daily_num"]
    if "R" in df.columns:
        df["R"] = df["R"] / df["daily_R1"]
    return df


def compute_price_response(
    df: pd.DataFrame, lag: int = 1, normalise: bool = False, remove_outliers: bool = True, log_prices=False
) -> pd.DataFrame:
    """
    R(l) interface::
        Called when fitting.
    """
    data = df.copy()
    if type(data["event_timestamp"].iloc[0]) != pd.Timestamp:
        data["event_timestamp"] = data["event_timestamp"].apply(lambda x: pd.Timestamp(x))
    data = rename_columns(data)
    data = add_daily_features(data)
    data = _price_response_function(data, lag=lag, log_prices=log_prices)
    if remove_outliers:
        data = smooth_outliers(data)
    if normalise:
        data = normalise_price_response(data, f"R{lag}")
    return data


def compute_conditional_aggregate_impact(
    df: pd.DataFrame, T: int, normalise: bool = True, remove_outliers: bool = True, log_prices=False
) -> pd.DataFrame:
    """
    Called when fitting.
    """
    # TODO: Implement compute_individual_impact: condition on previous sign and volume
    data = df.copy()
    if type(data["event_timestamp"].iloc[0]) != pd.Timestamp:
        data["event_timestamp"] = data["event_timestamp"].apply(lambda x: pd.Timestamp(x))
    data = rename_columns(data)
    data = add_daily_features(data)
    data = _conditional_aggregate_impact(data, T=T, response_column=f"R{T}", log_prices=log_prices)

    if normalise:
        data = normalise_axis(data)

    if remove_outliers:
        data = smooth_outliers(data, T=T)
    return data


def smooth_outliers(
    df: pd.DataFrame,
    T=None,
    columns=["vol_imbalance", "sign_imbalance"],
    std_level=3,
    remove=False,
    verbose=False
):
    """
    Clip or remove values at 3 standard deviations for each series.
    """
    if T:
        columns_all = columns + [f"R{T}"]
    else:
        columns_all = columns

    columns_all = set(columns_all).intersection(df.columns)
    if len(columns_all) == 0:
        return df

    if remove:
        z = np.abs(stats.zscore(df[columns]))
        original_shape = df.shape
        df = df[(z < std_level).all(axis=1)]
        new_shape = df.shape
        if verbose:
            print(f"Removed {original_shape[0] - new_shape[0]} rows")
    else:

        def winsorize_queue(s: pd.Series, level) -> pd.Series:
            upper_bound = level * s.std()
            lower_bound = - level * s.std()
            if verbose:
                print(f"clipped at {upper_bound}")
            return s.clip(upper=upper_bound, lower=lower_bound)

        for name in columns_all:
            s = df[name]
            if verbose:
                print(f"Series {name}")
            df[name] = winsorize_queue(s, level=std_level)

    return df


def rename_columns(df_: pd.DataFrame) -> pd.DataFrame:
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
