import pandas as pd


def remove_midprice_orders(df_: pd.DataFrame) -> pd.DataFrame:
    """
    Removes events that occured at the mid-price.
    """
    mask = df_["price"] == df_["midprice"]
    return df_[~mask]


def remove_first_daily_prices(orderbook_states: pd.DataFrame) -> pd.DataFrame:
    """
    If the price deviated significantly during auction hours the first
    returns on the day would be considered outliers.
    """
    data_ = orderbook_states.copy()
    data_["indx"] = data_.index
    data_ = data_.set_index("event_timestamp")
    first_days_indx = data_.groupby(pd.Grouper(freq="D")).first()["indx"]
    first_days_indx = first_days_indx.dropna().astype(int)
    data_ = data_.loc[~data_["indx"].isin(first_days_indx)]
    return data_.drop(columns=["indx"]).reset_index()


def normalise_size(
    orderbook_states: pd.DataFrame, column_name: str = "size"
) -> pd.DataFrame:
    """
    Normalise order size :math:`v_x` by the average volume on the same side best quote.
    .. math::
        v = v/\overline{V}_{\mathrm{best}}
    """
    data_ = orderbook_states.copy()

    # Compute avarage over average
    ask_mean_volume = data_[
        "ask_queue_size_mean"
    ].mean()  # ask_mean_vol = df_["ask_volume"].mean()
    bid_mean_volume = data_[
        "bid_queue_size_mean"
    ].mean()  # bid_mean_vol = df_["bid_volume"].mean()

    def _normalise(row):
        if row["side"] == "ASK":
            return row[column_name] / ask_mean_volume
        else:
            return row[column_name] / bid_mean_volume

    data_["norm_size"] = data_.apply(_normalise, axis=1)

    return data_
