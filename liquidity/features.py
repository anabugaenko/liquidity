import numpy as np
import pandas as pd
from typing import List

from liquidity.util.orderbook import rename_orderbook_columns
from liquidity.util.utils import normalise_size, normalize_impact, normalize_imbalances, remove_first_daily_prices


def signed_volume(df_: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the signed volume of an order given raw L3 market data
    """
    df_["signed_volume"] = df_["size"] * df_["sign"]
    return df_


def order_signs(df_: pd.DataFrame) -> pd.DataFrame:
    """
    Adds numeric sign distinguish the side of the book an event occurred,
    for which we introduce another pair of signs, -1 and +1 for the
    bid-side and ask-side events repsectively.
    """

    def _ennumerate_sides(row):
        return 1 if row["side"] == "ASK" else -1

    df_["sign"] = df_.apply(lambda row: _ennumerate_sides(row), axis=1)
    return df_


def bid_ask_spead(df_: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the spread s(t) ..math::
        s(t) = a(t) - b(t),
    which can be measured in several ways:
       - At random instances in calendar time,
       - At random instances in event-time,
       - immediately before an event, e.g., execution, limit order placement etc.
    Here, we compute the spread in inter-event time, which corresponds to the latter case.
    """

    df_["spread"] = df_["ask"] - df_["bid"]

    return df_


def mean_queue_lengths(lob_data: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the mean number and volume profile at the best quotes.
    """

    def _get_best_num(row):
        if row["side"] == "ASK":
            return ask_num_mean
        else:
            return bid_num_mean

    def _get_best_vol(row):
        if row["side"] == "ASK":
            return ask_size_mean
        else:
            return bid_size_mean

    # Compute average queue lenght at best
    ask_num_mean = lob_data["best_ask_num_orders"].mean()
    bid_num_mean = lob_data["best_bid_num_orders"].mean()
    lob_data["ask_queue_number_mean"] = ask_num_mean
    lob_data["bid_queue_number_mean"] = bid_num_mean

    # Compute average volume profile at best quotes
    ask_size_mean = lob_data["best_ask_size"].mean()
    bid_size_mean = lob_data["best_bid_size"].mean()
    lob_data["ask_queue_size_mean"] = ask_size_mean
    lob_data["bid_queue_size_mean"] = bid_size_mean

    # Average number and volume profile at best quotes
    lob_data["average_num_at_best"] = lob_data.apply(_get_best_num, axis=1)
    lob_data["average_vol_at_best"] = lob_data.apply(_get_best_vol, axis=1)

    return lob_data


def compute_returns(orderbook_states: pd.DataFrame, remove_first: bool = True) -> pd.DataFrame:
    """
    Compute various representations of returns for a given order book states and prices.

    Parameters:
    - orderbook_States_df (pd.DataFrame): Input dataframe with a 'event_timestamp' and 'midprice' column.
    - remove_first (bool, optional): Flag to indicate whether to remove the first daily price. Defaults to True.

    Returns:
    - pd.DataFrame: DataFrame with added series of different representations of returns.
    """
    lob_data = orderbook_states.copy()

    # Drop first daily price
    if remove_first:
        lob_data = remove_first_daily_prices(lob_data)

    # Fractional returns
    lob_data["returns"] = lob_data["midprice"].diff()

    # Remove any NaN or infinite values from 'returns'
    lob_data = lob_data[~lob_data["returns"].isin([np.nan, np.inf, -np.inf])]

    # Percentage (relative) returns
    lob_data["pct_returns"] = lob_data["midprice"].pct_change()
    # lob_data["pct_returns"] = (lob_data["midprice"] / lob_data["midprice"].shift(1)) - 1 # using numpy's pct_change equivalent for robustness

    # Other representations of returns
    # Log returns
    lob_data["log_returns"] = np.log(lob_data["midprice"]) - np.log(lob_data["midprice"].shift(1))

    # Volatility (return magnitudes)
    lob_data["variance"] = (
        lob_data["returns"] ** 2
    )  # time-varying standard deviation derived directly from returns
    lob_data["volatility"] = np.sqrt(lob_data["variance"])

    return lob_data


def r1(
    orderbook_states: pd.DataFrame,
    log_prices: bool = False,
    price_column: str = "midprice",
) -> pd.DataFrame:
    """
    Computes the price response `R1` of an order as the lag-dependent change in mid-price `m(t)` between time t and t + 1.

    Parameters
    ----------
    orderbook_states : pd.DataFrame
        DataFrame containing order book states.
    log_prices : bool, optional
        Compute log returns instead of fractional returns. Default is False.
    price_column : str, optional
        Column name for the price series data. Default is "midprice".

    Returns
    -------
    pd.DataFrame
        Orderbook states DataFrame updated with lag-1 unconditional impact R(1).
    """
    lob_data = orderbook_states.copy()

    # Compute log or fractional returns
    if log_prices:
        lob_data["price_change"] = np.log(lob_data[price_column].shift(-1)) - np.log(
            lob_data[price_column]
        )
    else:
        lob_data["price_change"] = lob_data[price_column].diff().shift(-1).fillna(0)

    # Compute conditional and unconditional impact of an order
    # data["R1"] = data["price_change"] * data["sign"]  # TODO: depricate R1
    lob_data["R1_cond"] = lob_data["price_change"]  # sign already accounted for in the conditioning variable
    lob_data["R1_uncond"] = lob_data["price_change"] * lob_data["sign"]  # when unconditioned, we explicitly account for the sign

    return lob_data


def daily_orderbook_states(
    orderbook_states: pd.DataFrame, response_column: str = "R1_uncond"
) -> pd.DataFrame:
    """
    From a given time series of transactions add daily means of lag one price response R1
    and order size (to be used as denominator in normalisation).
    """
    orderbook_states = orderbook_states.copy()

    if type(orderbook_states["event_timestamp"].iloc[0]) != pd.Timestamp:
        orderbook_states["event_timestamp"] = orderbook_states["event_timestamp"].apply(lambda x: pd.Timestamp(x))
    orderbook_states["date"] = orderbook_states["event_timestamp"].apply(lambda x: x.date())

    daily_R1 = (
        orderbook_states[[response_column, "date"]]
        .groupby("date")
        .agg(daily_R1=(response_column, "mean"))
    )
    daily_volume = orderbook_states[["size", "date"]].groupby("date").agg(daily_vol=("size", "sum"))
    daily_num = orderbook_states[["size", "date"]].groupby("date").agg(daily_num=("size", "count"))

    orderbook_states["daily_R1"] = daily_R1.reindex(
        index=orderbook_states["event_timestamp"], method="ffill"
    ).values
    orderbook_states["daily_vol"] = daily_volume.reindex(
        index=orderbook_states["event_timestamp"], method="ffill"
    ).values
    orderbook_states["daily_num"] = daily_num.reindex(
        index=orderbook_states["event_timestamp"], method="ffill"
    ).values

    return orderbook_states


def compute_orderbook_states(raw_orderbook_df: pd.DataFrame):
    """
    Computes limit order book (LOB) states from raw or transformed time series of L3 order book data.
    """
    raw_orderbook_df = raw_orderbook_df.copy()

    if type(raw_orderbook_df["event_timestamp"].iloc[0]) != pd.Timestamp:
        raw_orderbook_df["event_timestamp"] = raw_orderbook_df["event_timestamp"].apply(
            lambda x: pd.Timestamp(x)
        )
    data = rename_orderbook_columns(raw_orderbook_df)
    data = r1(data)
    data = bid_ask_spead(data)
    orderbook_states = daily_orderbook_states(data)
    orderbook_states = normalise_size(orderbook_states)
    orderbook_states = signed_volume(orderbook_states)

    return orderbook_states


def compute_intraday_features(
    orderbook_states_df_: pd.DataFrame,
    bin_size: int,
) -> pd.DataFrame:
    """
    Computes intraday features using T sized bins in event-time.
    """
    data = orderbook_states_df_.copy()

    if bin_size == 1:

        data["sign_imbalance"] = data["sign"]
        data["volume_imbalance"] = data["signed_volume"]
        data["price_change_imbalance"] = data["price_changing"]
        data["R_cond"] = data["R1_cond"]
        data["R_uncond"] = data["R1_uncond"]
        return data

    intraday_features = data.groupby(data.index // bin_size).agg(
        # Orderbook states
        event_timestamp=("event_timestamp", "first"),
        midprice=("midprice", "first"),
        sign=("sign", "first"),
        signed_volume=("signed_volume", "first"),
        price_changing=("price_changing", "first"),
        # Daily features
        daily_R1=("daily_R1", "first"),
        daily_num=("daily_num", "first"),
        daily_vol=("daily_vol", "first"),
        # Aggregate impact
        R_cond=("R1_cond", "sum"),
        R_uncond=("R1_uncond", "sum"),
        # Imbalances
        sign_imbalance=("sign", "sum"),
        volume_imbalance=("signed_volume", "sum"),
        price_change_imbalance=("price_changing", "sum"),
        # Average queue length and volume profile at the best quotes
        average_num_at_best=("average_num_at_best", "first"),
        average_vol_at_best=("average_vol_at_best", "first"),
    ).reset_index()

    return intraday_features.dropna()


def compute_aggregate_features(orderbook_states: pd.DataFrame,
                               bin_frequencies: List[int],
                               remove_first: bool = True) -> pd.DataFrame:
    """
    Computes aggregate intraday features for different order types given T sized binning frequencies.
    """
    data = orderbook_states.copy()

    # Drop first daily price
    if remove_first:
        data = remove_first_daily_prices(data)

    results_for_T = []
    all_dates = data["date"].unique()
    for T, bin_size in enumerate(bin_frequencies):

        results_for_dates = []
        for date in all_dates:

            daily_data = data[data["date"] == date]

            binned_data = compute_intraday_features(
                daily_data,
                bin_size=bin_size,
            )
            binned_data["T"] = bin_size
            results_for_dates.append(binned_data)

        results_for_T.append(pd.concat(results_for_dates))

    # Aggregate features
    aggregate_features = pd.concat(results_for_T)

    return aggregate_features
