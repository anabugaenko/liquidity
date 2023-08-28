import pandas as pd

from liquidity.response_functions.price_response_functions import add_daily_features, aggregate_impact
from liquidity.util.data_util import normalise_imbalances
from liquidity.util.util import _remove_outliers


def rename_price_columns(df_: pd.DataFrame) -> pd.DataFrame:
    df_ = df_.drop(['price', 'size'], axis=1)
    return df_.rename(columns={'old_price': 'price', 'old_size': 'size'})


def rename_columns(df_: pd.DataFrame) -> pd.DataFrame:
    df_columns = df_.columns

    if 'old_price' in df_columns and 'old_size' in df_columns:
        df_ = rename_price_columns(df_)

    if 'R1_CA' in df_columns:
        df_ = df_.rename(columns={'R1_CA': 'R1'})

    if 'R1_LO' in df_columns:
        df_ = df_.rename(columns={'R1_LO': 'R1'})

    if 'execution_size' in df_columns:
        df_ = df_.rename(columns={'execution_size': 'size'})

    if 'trade_sign' in df_columns:
        df_ = df_.rename(columns={'trade_sign': 'sign'})

    return df_


def get_aggregate_impact_series(df_: pd.DataFrame,
                                T: int,
                                normalise: bool = True,
                                remove_outliers: bool = False,
                                log=False) -> pd.DataFrame:
    data = df_.copy()
    if type(data['event_timestamp'].iloc[0]) != pd.Timestamp:
        data['event_timestamp'] = data['event_timestamp'].apply(lambda x: pd.Timestamp(x))
    data = rename_columns(data)
    data = add_daily_features(data)
    data = aggregate_impact(data, T=T, response_column=f'R{T}', log=log)
    if remove_outliers:
        data = _remove_outliers(data)
    if normalise:
        data = normalise_imbalances(data)
    return data