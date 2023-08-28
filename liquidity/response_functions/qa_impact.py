import pandas as pd

from liquidity.response_functions.price_response_functions import add_daily_features, aggregate_response_function
from liquidity.util.data_util import normalise_imbalances
from liquidity.util.util import _remove_outliers


def select_top_book(df: pd.DataFrame) -> pd.DataFrame:
    """
    Need to carefully select only events that affected top book level.
    By definition selecting events at level 0 and 1 is not accurate for cancellation orders
    since the level is set to 0 if the removal caused a price level to no longer exist.
    """
    price_level_mask = (df.price_level == 1) | (df.price_level == 0)
    old_price_level_mask = (df.old_price_level == 1) | (df.old_price_level == 0)
    return df[price_level_mask & old_price_level_mask]


def normalise_all_sizes(df_: pd.DataFrame):
    """
    if execution -> execution_size
    if insert/LO -> size
    if cancel/remove -> old size
    """

    def _select_size_for_order_type(row):
        mask1 = row['lob_action'] == 'INSERT'
        mask2 = row['lob_action'] == 'UPDATE'
        mask2 = mask2 & (row['price_changing'] == True)
        lo_mask = mask1 | mask2

        if lo_mask:
            return row['size']

        mo_mask = row['order_executed']

        if mo_mask:
            return row['execution_size']

        mask1 = row['lob_action'] == 'REMOVE'
        mask2 = row['order_executed'] == False
        mask3 = row['old_price_level'] == 1
        mask_complete_removals = mask1 & mask2 & mask3

        mask4 = row['lob_action'] == 'UPDATE'
        mask5 = row['order_executed'] == False
        mask6 = row['old_price_level'] == 1
        mask7 = row['size'] < row['old_size']
        mask_partial_removals = mask4 & mask5 & mask6 & mask7
        ca_mask = mask_complete_removals | mask_partial_removals

        if ca_mask:
            return row['old_size']

        return 0

    df_['new_size'] = df_.apply(lambda row: _select_size_for_order_type(row), axis=1)
    df_ = df_[~(df_['new_size'] == 0)]

    ask_mean_size = df_[df_['side'] == 'ASK']['size'].mean()
    bid_mean_size = df_[df_['side'] == 'BID']['size'].mean()

    def _normalise(row):
        if row['side'] == 'ASK':
            return row['new_size'] / ask_mean_size
        else:
            return row['new_size'] / bid_mean_size

    df_['norm_size'] = df_.apply(_normalise, axis=1)

    return df_


def get_aggregate_qa_response_features(df_: pd.DataFrame,
                                       T: int,
                                       normalise: bool = True,
                                       remove_outliers: bool = False) -> pd.DataFrame:
    data = df_.copy()
    data = add_daily_features(data)
    data = aggregate_response_function(data, T=T, response_column=f'R{T}')
    if remove_outliers:
        data = _remove_outliers(data, T=T)
    if normalise:
        data = normalise_imbalances(data)
    return data
