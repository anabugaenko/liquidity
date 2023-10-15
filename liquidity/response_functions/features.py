import pandas as pd
from typing import List

from liquidity.response_functions.price_response_functions import compute_conditional_aggregate_impact, \
    compute_price_response
from liquidity.util.orderbook import rename_orderbook_columns, add_daily_features
from liquidity.util.utils import normalise_size, add_R1


# TODO: function that gets non aggregate features, e.g.,  size, normalised_volume etc
def add_orderbook_states(raw_orderbook_df: pd.DataFrame):
    # R1_ordertype, spread, midprice

    if type(raw_orderbook_df["event_timestamp"].iloc[0]) != pd.Timestamp:
        raw_orderbook_df["event_timestamp"] = raw_orderbook_df["event_timestamp"].apply(lambda x: pd.Timestamp(x))
    data = rename_orderbook_columns(raw_orderbook_df)
    data = add_R1(data)
    data = add_daily_features(data)
    orderbook_states = normalise_size(data)
    return orderbook_states

def add_aggregate_features(df: pd.DataFrame, durations: List[int], **kwargs) -> pd.DataFrame:
    df["event_timestamp"] = df["event_timestamp"].apply(lambda x: pd.Timestamp(x))
    df["date"] = df["event_timestamp"].apply(lambda x: x.date())
    results_ = []
    for i, T in enumerate(durations):
        lag_data = compute_conditional_aggregate_impact(df, T=T, **kwargs)
        lag_data["R"] = lag_data[f"R{T}"]
        lag_data = lag_data.drop(columns=f"R{T}")
        lag_data["T"] = T
        results_.append(lag_data)

    return pd.concat(results_)

