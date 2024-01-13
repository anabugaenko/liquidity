import pandas as pd
import numpy as np


# TODO: move to features and depricate file
def calculate_v_over_vbest(row):
    if row["side"] == "ASK":
        return row["size"] / row["ask_volume"]
    else:
        return row["size"] / row["bid_volume"]


def remove_empty_queue(df):
    mask = df["ask_volume"] == 0.0
    mask = mask & (df["bid_volume"] == 0.0)
    return df[~mask]


def get_v_over_vbest_ratios(mo_df: pd.DataFrame):
    df_ = mo_df.copy()
    df_ = remove_empty_queue(df_)
    return df_.apply(calculate_v_over_vbest, axis=1)


def get_v_over_vbest_distribution(df, stock, n=100):
    series = get_v_over_vbest_ratios(df)
    bins, values = get_bins_and_values(series, n=n)

    df_ = pd.DataFrame(bins, values).reset_index()
    df_.columns = ["value", "bin"]
    df_["style"] = stock
    return df_


def get_bins_and_values(series, n=100):
    bins = np.linspace(0, 2, n)
    weightsa = np.ones_like(series) / float(len(series))
    hist = np.histogram(np.array(series), bins, weights=weightsa)

    bins = (hist[1][1:] + hist[1][:-1]) / 2
    values = hist[0]

    return bins, values
