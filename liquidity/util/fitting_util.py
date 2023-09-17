import pandas as pd

from liquidity.response_functions.price_response_functions import compute_conditional_aggregate_impact
from typing import List


def bin_data_into_quantiles(df, x_col='vol_imbalance', y_col='R', q=100, duplicates='raise'):
    """
    3/9/23 This is majorly used.
    """
    binned_x = pd.qcut(df[x_col], q=q, labels=False, retbins=True, duplicates=duplicates)
    binned_x = binned_x[0]
    df['x_bin'] = binned_x

    y_binned = df.groupby(['x_bin'])[y_col].mean()
    y_binned.index = y_binned.index.astype(int)

    x_binned = df.groupby(['x_bin'])[x_col].mean()
    x_binned.index = x_binned.index.astype(int)

    if 'T' in df.columns:
        r_binned = df.groupby(['x_bin'])['T'].first()
        r_binned.index = r_binned.index.astype(int)
    else:
        r_binned = None

    return pd.concat([x_binned, r_binned, y_binned], axis=1).reset_index(drop=True)


def get_agg_features(df: pd.DataFrame, durations: List[int]) -> pd.DataFrame:
    df['event_timestamp'] = df['event_timestamp'].apply(lambda x: pd.Timestamp(x))
    df['date'] = df['event_timestamp'].apply(lambda x: x.date())
    results_ = []
    for i, T in enumerate(durations):
        lag_data = compute_conditional_aggregate_impact(df, T=T)
        lag_data['R'] = lag_data[f'R{T}']
        lag_data = lag_data.drop(columns=f'R{T}')
        lag_data['T'] = T
        results_.append(lag_data)

    return pd.concat(results_)
