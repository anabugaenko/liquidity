import pandas as pd

from liquidity.response_functions.price_response_functions import add_daily_features, aggregate_response_function
from liquidity.util.data_util import normalise_imbalances
from liquidity.util.util import _remove_outliers


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


def rename_price_columns(df_: pd.DataFrame) -> pd.DataFrame:
    df_ = df_.drop(['price', 'size'], axis=1)
    return df_.rename(columns={'old_price': 'price', 'old_size': 'size'})


def get_aggregate_ca_response_features(df_: pd.DataFrame,
                                       T: int,
                                       normalise: bool = True,
                                       remove_outliers: bool = False) -> pd.DataFrame:
    data = df_.copy()
    data = data.rename(columns={'R1_CA': 'R1'})
    data = rename_price_columns(data)
    data = add_daily_features(data)
    data = aggregate_response_function(data, T=T, response_column=f'R{T}')
    if remove_outliers:
        data = _remove_outliers(data, T=T)
    if normalise:
        data = normalise_imbalances(data)
    return data
