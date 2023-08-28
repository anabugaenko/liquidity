import pandas as pd

from liquidity.response_functions.lob_data import load_l3_data, select_trading_hours, select_top_book, select_columns, \
    shift_prices
from liquidity.response_functions.price_response import add_daily_features, aggregate_response_function, individual_response_function
from liquidity.util.data_util import normalise_imbalances
from liquidity.util.util import numerate_side, _remove_outliers, add_order_sign


def remove_midprice_orders(df_: pd.DataFrame) -> pd.DataFrame:
    mask = df_['price'] == df_['midprice']
    return df_[~mask]


def select_lo_inserts(df_: pd.DataFrame) -> pd.DataFrame:
    # check for updates that increased volume - hidden orders?
    m1 = df_['old_size'] < df_['size']
    m2 = df_['lob_action'] == 'UPDATE'
    if not df_[m1 & m2].shape[0] == 0:
        print('Found and removed order updates that increased size \n', df_[m1 & m2][['size', 'old_size']])
        df_ = df_[~(m1 & m2)]

    mask1 = df_['lob_action'] == 'INSERT'
    mask2 = df_['lob_action'] == 'UPDATE'
    mask2 = mask2 & (df_['price_changing'] == True)

    return df_[mask1 | mask2]


def normalise_lo_sizes(df_: pd.DataFrame) -> pd.DataFrame:
    ask_mean_size = df_[df_['side'] == 'ASK']['size'].mean()
    bid_mean_size = df_[df_['side'] == 'BID']['size'].mean()

    def _normalise(row):
        if row['side'] == 'ASK':
            return row['size'] / ask_mean_size
        else:
            return row['size'] / bid_mean_size

    df_['norm_size'] = df_.apply(_normalise, axis=1)
    return df_


def get_daily_lo_arrivals(filepath: str, date: str) -> pd.DataFrame:
    """
    Loads LOB events timeseries for a day from a file and
    returns a DataFrame of LO arrivals timeseries.
    :param filepath:
    :param date:
    :return:
    """
    data = load_l3_data(filepath)
    df = select_trading_hours(date, data)
    df = select_top_book(df)
    df = select_columns(df)
    df = shift_prices(df)
    df = remove_midprice_orders(df)
    df = add_order_sign(df)
    df = select_lo_inserts(df)
    df = individual_response_function(df, response_column='R1_LO')
    df = normalise_lo_sizes(df)
    return df


def clean_lob_data(date: str, df_raw: pd.DataFrame) -> pd.DataFrame:
    df = select_trading_hours(date, df_raw)
    df = select_top_book(df)
    df = select_columns(df)
    df = shift_prices(df)
    return remove_midprice_orders(df)


def get_aggregate_lo_response_features(df_: pd.DataFrame,
                                       T: int,
                                       normalise: bool = True,
                                       remove_outliers: bool = False) -> pd.DataFrame:
    data = df_.copy()
    if type(data['event_timestamp'].iloc[0]) != pd.Timestamp:
        data['event_timestamp'] = data['event_timestamp'].apply(lambda x: pd.Timestamp(x))
    # data = remove_first_daily_prices(data)
    data = data.rename(columns={'R1_LO': 'R1'})
    data = add_daily_features(data)
    data = data.reset_index(drop=True)
    data = aggregate_response_function(data, T=T, response_column=f'R{T}')
    if remove_outliers:
        data = _remove_outliers(data, T=T)
    if normalise:
        data = normalise_imbalances(data)
    return data


def get_agg_features(df: pd.DataFrame, durations):
    results_ = []
    for i, T in enumerate(durations):
        lag_data = get_aggregate_lo_response_features(df, T=T)
        lag_data['R'] = lag_data[f'R{T}']
        lag_data = lag_data.drop(columns=f'R{T}')
        lag_data['T'] = T
        results_.append(lag_data)

    return pd.concat(results_)
