import pandas as pd


def numerate_side(row):
    return 1 if row['side'] == 'ASK' else -1


def add_order_sign(df_: pd.DataFrame) -> pd.DataFrame:
    df_['sign'] = df_.apply(lambda row: numerate_side(row), axis=1)
    return df_


def _remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove observations where volume imbalance or price response
    value was beyond three standard deviations of it's mean.
    """

    def winsorize_queue(s: pd.Series, level: int = 3) -> pd.Series:
        upper_bound = level * s.std()
        return s.clip(upper=upper_bound)

    queue_columns = ['vol_imbalance', 'sign_imbalance']

    for name in queue_columns:
        s = df[name]
        df[name] = winsorize_queue(s)

    return df
