import pandas as pd
import numpy as np
from scipy import stats


def add_order_signs(df_: pd.DataFrame) -> pd.DataFrame:
    def _ennumerate_sides(row):
        return 1 if row["side"] == "ASK" else -1

    df_["sign"] = df_.apply(lambda row: _ennumerate_sides(row), axis=1)
    return df_


# TODO: move to price_response_finctons
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


# Placeholder for the remove_first_daily_prices function as it was not provided
def remove_first_daily_prices(df: pd.DataFrame) -> pd.DataFrame:
    # Your implementation here
    pass


def remove_midprice_orders(df_: pd.DataFrame) -> pd.DataFrame:
    mask = df_["price"] == df_["midprice"]
    return df_[~mask]


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
