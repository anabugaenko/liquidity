import pandas as pd


def numerate_side(row):
    return 1 if row['side'] == 'ASK' else -1


def add_order_sign(df_: pd.DataFrame) -> pd.DataFrame:
    df_['sign'] = df_.apply(lambda row: numerate_side(row), axis=1)
    return df_


