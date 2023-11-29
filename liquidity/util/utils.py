import pandas
import pandas as pd
import numpy as np
from scipy import stats


def remove_midprice_orders(df_: pd.DataFrame) -> pd.DataFrame:
    mask = df_["price"] == df_["midprice"]
    return df_[~mask]


def add_mean_queue_lengths(lob_data: pd.DataFrame) -> pd.DataFrame:
    lob_data["ask_queue_size_mean"] = lob_data["best_ask_size"].mean()
    lob_data["bid_queue_size_mean"] = lob_data["best_bid_size"].mean()

    return lob_data


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


# FIXME: move to market impact package?
def normalise_size(df_: pd.DataFrame, size_col_name: str = "size") -> pd.DataFrame:
    """
    Normalise trade size by the average volume on the same side best quote.
    """
    # ask_mean_vol = df_["ask_volume"].mean()
    # bid_mean_vol = df_["bid_volume"].mean()

    ask_mean_vol = df_["ask_queue_size_mean"].mean()
    bid_mean_vol = df_["bid_queue_size_mean"].mean()

    def _normalise(row):
        if row["side"] == "ASK":
            return row[size_col_name] / ask_mean_vol
        else:
            return row[size_col_name] / bid_mean_vol

    df_["norm_size"] = df_.apply(_normalise, axis=1)
    return df_
