import pandas as pd
import numpy as np
from typing import Optional, List
from scipy import stats
from scipy.stats import kendalltau, spearmanr

from liquidity.response_functions.lob_data import select_trading_hours, select_top_book, select_columns, \
    shift_prices
from liquidity.util.limit_orders_data_util import remove_midprice_orders


def set_row_groups(start: float, group_size: float, data: pd.DataFrame, column_name: str = 'norm_trade_volume') \
        -> pd.DataFrame:
    """
    Iteratively assign observations to a group.
    group_size is an axis interval in a variable (not count of observations)
    """
    threshold = start
    group_count = 0
    for i in range(0, data.shape[0]):
        if data.loc[i][column_name] < threshold:
            data.loc[i]['group'] = group_count
        else:
            threshold = threshold + group_size
            group_count += 1
            data.loc[i]['group'] = group_count
    return data


def custom_subsample(data: pd.DataFrame, start: float, split_point: float, group_size: float,
                     is_lower: bool = True, downsample_column: str = 'norm_trade_volume',
                     other_columns: Optional[List[str]] = ['R1']) -> pd.DataFrame:
    """
    Subsample/normalise data in the interval specified by the split and indication
    whether it's a lower part of the entire range of samples.
    Start if either a split point (for the second half) or a min of downsample_column
    range in a give data.
    group_size is an axis interval in a variable (not count of observations)
    """
    if is_lower:
        df = data[data[downsample_column] <= split_point].copy()
    else:
        df = data[data[downsample_column] > split_point].copy().reset_index(drop=True)
    df['group'] = np.nan
    df = df.sort_values(by=downsample_column)

    df = set_row_groups(start, group_size, df)

    main_col = df[[downsample_column, 'group']].astype(float).groupby('group').mean()
    other_cols = df[other_columns + ['group']].astype(float).groupby('group').mean()

    bars = pd.concat([main_col, other_cols], axis=1)
    bars.columns = [downsample_column] + other_columns

    return bars


def get_subsampled_series(data: pd.DataFrame, split_point: float, hd_group_size: float, ld_group_size: float) \
        -> pd.DataFrame:
    """
    Downsample/subsample/reduce observations space based on the custom specified
    split point and group density.
    Appropriate for datasets with irregular sample density.
    """
    high_density_bars = custom_subsample(data=data,
                                         start=0.,
                                         split_point=split_point,
                                         group_size=hd_group_size,
                                         is_lower=True)
    low_density_bars = custom_subsample(data=data,
                                        start=split_point,
                                        split_point=split_point,
                                        group_size=ld_group_size,
                                        is_lower=False)
    result = pd.concat([high_density_bars, low_density_bars], axis=0, join='outer', ignore_index=True)
    return result


def downsample_imbalance_response(df_: pd.DataFrame,
                                  T: int,
                                  quantile: bool = False,
                                  downsample_num: int = 50,
                                  bin_size: int = 31,
                                  qmax: float = 0.0005) -> pd. DataFrame:
    if quantile:
        df_ = df_[(df_['vol_imbalance'] > -qmax) & (df_['vol_imbalance'] < qmax)]
        df_['bin'] = pd.qcut(df_['vol_imbalance'], bin_size)
        return df_.groupby(['bin']).mean().reset_index()
    else:
        grouped = df_.groupby(['vol_imbalance']).agg({f'R{T}': 'mean'}).reset_index()
        grouped = grouped.sort_values(by=['vol_imbalance'])

        imbalance = grouped[['vol_imbalance']].astype(float).groupby(grouped.index // downsample_num).mean()
        aggregate = grouped[[f'R{T}']].astype(float).groupby(grouped.index // downsample_num).mean()

        result = pd.concat([imbalance, aggregate], axis=1)
        result.columns = ['vol_imbalance', f'R{T}']
        return result


def get_density_df(series, q=50):
    """
    Compute density statistics and downsample for visualisation.
    """
    bins = np.sort(series)
    values = 1 - (1. * np.arange(len(bins)) / (len(bins) - 1))

    df_ = pd.DataFrame(bins, values).reset_index()
    df_.columns = ['bin', 'value']

    try:
        df_['bin_qcut'] = pd.qcut(df_.value, q)
    except ValueError as e:
        print(e)
        df_['bin_qcut'] = pd.qcut(df_.value, q, duplicates='drop')
    df_ = df_.groupby(['bin_qcut']).max().reset_index()
    df_ = df_.iloc[1:]  # in case there are natural outliers remove very rare values

    return df_


def get_series_acf(series: pd.Series, lags: int) -> List:
    """
    Returns a list of autocorrelation values for each of the lags from 0 to `lags`
    """
    acl_ = []
    for i in range(lags):
        ac = series.autocorr(lag=i)
        acl_.append(ac)
    return acl_


class CustomSeries(pd.Series):
    def nonlinear_autocorr(self, lag: int = 1, method: str = 'spearman') -> float:
        """
        Compute the lag-N autocorrelation using Kendall's Tau or Spearman rank correlation.

        This method computes the selected correlation method between
        the Series and its shifted self.

        Parameters
        ----------
        lag : int, default 1
            Number of lags to apply before performing autocorrelation.
        method : str, default 'spearman'
            The correlation method to use. Either 'spearman' or 'kendall'.

        Returns
        -------
        float
            The selected correlation method between self and self.shift(lag).

        See Also
        --------
        Series.corr : Compute the correlation between two Series.
        Series.shift : Shift index by desired number of periods.
        DataFrame.corr : Compute pairwise correlation of columns.
        DataFrame.corrwith : Compute pairwise correlation between rows or
            columns of two DataFrame objects.

        Notes
        -----
        If the selected correlation method is not well defined return 'NaN'.
        """
        shifted_self = self.shift(lag)
        valid_indices = (~np.isnan(self) & ~np.isnan(shifted_self))

        if method.lower() == 'spearman':
            return spearmanr(self[valid_indices], shifted_self[valid_indices])[0]
        elif method.lower() == 'kendall':
            return kendalltau(self[valid_indices], shifted_self[valid_indices])[0]
        else:
            raise ValueError("Invalid method. Choose either 'spearman' or 'kendall'.")


def get_nonlinear_acf(series: pd.Series, lags: int, method: str) -> List:
        """
        Returns a list of autocorrelation values for each of the lags from 0 to `lags`
        """
        acl_ = []
        for i in range(lags):
            ac = CustomSeries(series).nonlinear_autocorr(lag=i, method=method)
            acl_.append(ac)
        return acl_


def remove_outliers(df_, columns=['norm_trade_volume', 'R1'], print_info=True):
    z = np.abs(stats.zscore(df_[columns]))
    if print_info:
        print(df_.shape)
    df_ = df_[(z < 2).all(axis=1)]
    if print_info:
        print(df_.shape)
    return df_


def normalise_imbalances(df_: pd.DataFrame) -> pd. DataFrame:
    """
    Normalise volume imbalance by mean daily order size relative to its average;
    sign imbalance by mean daily number of orders.
    """
    df_['vol_imbalance'] = df_['vol_imbalance'] / df_['daily_vol'] * df_['daily_vol'].mean()
    df_['sign_imbalance'] = df_['sign_imbalance'] / df_['daily_num'] * df_['daily_num'].mean()

    return df_


def clean_lob_data(date: str, df_raw: pd.DataFrame) -> pd.DataFrame:
    df = select_trading_hours(date, df_raw)
    df = select_top_book(df)
    df = select_columns(df)
    df = shift_prices(df)
    return remove_midprice_orders(df)


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
