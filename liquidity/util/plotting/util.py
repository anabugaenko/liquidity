import numpy as np
import pandas as pd


def get_data_for_plotting(qmax, durations):
    n = 1001

    # T matrix matching with x-axis
    T_array = np.array(durations)
    T_mat = np.transpose(np.tile(T_array, (n, 1)))

    # repeat x-axis sample for
    q = np.linspace(-qmax, qmax, n)

    return q, T_mat


def _custom_downsample_acf(acf: pd.DataFrame, max_lag: int, bin_size=30, q=100,
                           drop_negative=False):
    df_ = pd.DataFrame(range(max_lag), acf).reset_index()
    df_.columns = ['acf', 'lag']

    if drop_negative:
        df_ = df_[df_.acf > 0].reset_index(drop=True)

    max_lag = len(df_.acf)
    top = int(max_lag * 0.08 // 1)
    bottom = max_lag - top

    df_top = df_.head(top)
    df_bottom = df_.tail(bottom)

    df_bottom = df_bottom.groupby(df_bottom.index // bin_size).mean()
    # df_bottom['bin_qcut'] = pd.qcut(df_bottom.acf, q)
    # df_bottom = df_bottom.groupby(['bin_qcut']).max().reset_index()[['acf', 'lag']]
    # df_bottom = df_bottom.sort_values(by='lag')

    result = pd.concat([df_top, df_bottom]).reset_index(drop=True)

    return result
