import pandas as pd

from liquidity.response_functions.price_response_functions import aggregate_impact
from liquidity.util.utils import smooth_outliers, rename_columns


def add_daily_features(df_: pd.DataFrame, response_column: str = 'R1') -> pd.DataFrame:
    """
    From a given time series of transactions add daily means of
    lag one price response R1 and order size.
    """
    if type(df_['event_timestamp'].iloc[0]) != pd.Timestamp:
        df_['event_timestamp'] = df_['event_timestamp'].apply(lambda x: pd.Timestamp(x))
    df_['date'] = df_['event_timestamp'].apply(lambda x: x.date())

    daily_R1 = df_[[response_column, 'date']].groupby('date').agg(daily_R1=(response_column, 'mean'))
    daily_volume = df_[['size', 'date']].groupby('date').agg(daily_vol=('size', 'sum'))
    daily_num = df_[['size', 'date']].groupby('date').agg(daily_num=('size', 'count'))

    df_['daily_R1'] = daily_R1.reindex(index=df_['event_timestamp'], method='ffill').values
    df_['daily_vol'] = daily_volume.reindex(index=df_['event_timestamp'], method='ffill').values
    df_['daily_num'] = daily_num.reindex(index=df_['event_timestamp'], method='ffill').values

    return df_


def normalise_imbalances(df_: pd.DataFrame) -> pd. DataFrame:
    """
    Normalise volume imbalance by mean daily order size relative to its average;
    sign imbalance by mean daily number of orders.
    """
    df_['vol_imbalance'] = df_['vol_imbalance'] / df_['daily_vol'] * df_['daily_vol'].mean()
    df_['sign_imbalance'] = df_['sign_imbalance'] / df_['daily_num'] * df_['daily_num'].mean()

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
        data = smooth_outliers(data)
    if normalise:
        data = normalise_imbalances(data)
    return data

# TODO: get individual impact.

