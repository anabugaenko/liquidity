import pandas as pd

from liquidity.price_impact.lo_impact import remove_midprice_orders, normalise_lo_sizes
from liquidity.price_impact.lob_data import select_trading_hours, load_l3_data, shift_prices, select_top_book, \
    select_columns
from liquidity.price_impact.price_response import add_daily_features, get_aggregate_response, _normalise_features
from liquidity.price_impact.util import numerate_side, _remove_outliers


def select_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    mask1 = df['lob_action'] == 'REMOVE'
    mask2 = df['order_executed'] == False
    mask3 = df['old_price_level'] == 1
    mask_complete_removals = mask1 & mask2 & mask3

    mask4 = df['lob_action'] == 'UPDATE'
    mask5 = df['order_executed'] == False
    mask6 = df['old_price_level'] == 1
    mask7 = df['size'] < df['old_size']
    mask_partial_removals = mask4 & mask5 & mask6 & mask7

    return df[mask_complete_removals | mask_partial_removals]


def add_price_response(df_: pd.DataFrame) -> pd.DataFrame:

    # numerate the side
    df_['sign'] = df_.apply(lambda row: numerate_side(row), axis=1)

    # compute directional response
    df_['midprice_change'] = df_['midprice'].diff().shift(-1).fillna(0)
    df_['R1_CA'] = df_['midprice_change'] * df_['sign']
    return df_


def rename_price_columns(df_: pd.DataFrame) -> pd.DataFrame:
    df_ = df_.drop(['price', 'size'], axis=1)
    return df_.rename(columns={'old_price': 'price', 'old_size': 'size'})


def get_daily_ca_arrivals(filepath: str, date: str) -> pd.DataFrame:
    data = load_l3_data(filepath)
    df = select_trading_hours(date, data)
    df = select_top_book(df)
    df = select_columns(df)
    df = shift_prices(df)
    df = remove_midprice_orders(df)
    df = select_cancellations(df)
    df = rename_price_columns(df)
    df = add_price_response(df)
    df = normalise_lo_sizes(df)
    return df


def get_aggregate_ca_response_features(df_: pd.DataFrame,
                                       T: int,
                                       normalise: bool = True,
                                       remove_outliers: bool = False) -> pd.DataFrame:
    data = df_.copy()
    data = data.rename(columns={'R1_CA': 'R1'})
    data = rename_price_columns(data)
    data = add_daily_features(data)
    data = get_aggregate_response(data, T=T, response_column=f'R{T}')
    if remove_outliers:
        data = _remove_outliers(data, T=T)
    if normalise:
        data = _normalise_features(data, response_column=f'R{T}')
    return data
