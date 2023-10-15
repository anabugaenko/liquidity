import pandas as pd
from typing import List

from liquidity.response_functions.price_response_functions import compute_conditional_aggregate_impact


# TODO: function that gets non aggregate features, e.g.,  size, normalised_volume etc
def get_orderbook_states(raw_orderbook_df: pd.DataFrame):
    # size_volume, normalized_volume, price_chng (is_new_best_price), R1_ordertype
    pass


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
