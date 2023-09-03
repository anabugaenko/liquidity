from typing import List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


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


def get_series_acf(series: pd.Series, lags: int) -> List:
    """
    Returns a list of autocorrelation values for each of the lags from 0 to `lags`
    """
    acl_ = []
    for i in range(lags):
        ac = series.autocorr(lag=i)
        acl_.append(ac)
    return acl_


def get_nonlinear_acf(series: pd.Series, lags: int, method: str) -> List:
        """
        Returns a list of autocorrelation values for each of the lags from 0 to `lags`
        """
        acl_ = []
        for i in range(lags):
            ac = CustomSeries(series).nonlinear_autocorr(lag=i, method=method)
            acl_.append(ac)
        return acl_
