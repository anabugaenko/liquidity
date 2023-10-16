import pandas as pd
import numpy as np
from scipy import stats


def add_R1(df_: pd.DataFrame) -> pd.DataFrame:
    # R(1)
    df_["midprice_change"] = df_["midprice"].diff().shift(-1).fillna(0)
    df_["R1"] = df_["midprice_change"] * df_["sign"]
    return df_


def add_spead(df_: pd.DataFrame) -> pd.DataFrame:
    """
    :math: s(t) = a(t) - b(t)

        The  spread can be measured in several ways:

       - At random instances in calendar time,
       - At random instances in event-time,
       - immediately before an event, e.g., execution, limit order placement etc.

    """
    pass
    return df_

# Placeholder for the remove_first_daily_prices function as it was not provided
def remove_first_daily_prices(df: pd.DataFrame) -> pd.DataFrame:
    # Your implementation here
    pass


def remove_midprice_orders(df_: pd.DataFrame) -> pd.DataFrame:
    mask = df_["price"] == df_["midprice"]
    return df_[~mask]


def add_order_signs(df_: pd.DataFrame) -> pd.DataFrame:
    def _ennumerate_sides(row):
        return 1 if row["side"] == "ASK" else -1

    df_["sign"] = df_.apply(lambda row: _ennumerate_sides(row), axis=1)
    return df_


def remove_first_daily_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    If the price deviated significantly during auction hours the first
    returns on the day would be considered outliers.
    """
    df_ = df.copy()
    df_["indx"] = df_.index
    df_ = df_.set_index("event_timestamp")
    first_days_indx = df_.groupby(pd.Grouper(freq="D")).first()["indx"]
    first_days_indx = first_days_indx.dropna().astype(int)
    df_ = df_.loc[~df_["indx"].isin(first_days_indx)]
    return df_.drop(columns=["indx"]).reset_index()


def normalise_size(df_: pd.DataFrame, size_col_name: str = "size") -> pd.DataFrame:
    """
    Normalise trade size by the average volume on the same side best quote.
    """
    ask_mean_vol = df_["ask_volume"].mean()
    bid_mean_vol = df_["bid_volume"].mean()

    def _normalise(row):
        if row["side"] == "ASK":
            return row[size_col_name] / ask_mean_vol
        else:
            return row[size_col_name] / bid_mean_vol

    df_["norm_size"] = df_.apply(_normalise, axis=1)
    return df_


def bin_data_into_quantiles(df, x_col="vol_imbalance", y_col="R", q=100, duplicates="raise"):
    """
    Returns binned series.
    """
    binned_x = pd.qcut(df[x_col], q=q, labels=False, retbins=True, duplicates=duplicates)
    binned_x = binned_x[0]
    df["x_bin"] = binned_x

    y_binned = df.groupby(["x_bin"])[y_col].mean()
    y_binned.index = y_binned.index.astype(int)

    x_binned = df.groupby(["x_bin"])[x_col].mean()
    x_binned.index = x_binned.index.astype(int)

    if "T" in df.columns:
        r_binned = df.groupby(["x_bin"])["T"].first()
        r_binned.index = r_binned.index.astype(int)
    else:
        r_binned = None

    return pd.concat([x_binned, r_binned, y_binned], axis=1).reset_index(drop=True)


def smooth_outliers(
    df: pd.DataFrame, T=None, columns=["vol_imbalance", "sign_imbalance"], std_level=2, remove=False, verbose=False
):
    # TODO: default columns to None
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
            lower_bound = -level * s.std()
            if verbose:
                print(f"clipped at {upper_bound}")
            return s.clip(upper=upper_bound, lower=lower_bound)

        for name in columns_all:
            s = df[name]
            if verbose:
                print(f"Series {name}")
            df[name] = winsorize_queue(s, level=std_level)

    return df



