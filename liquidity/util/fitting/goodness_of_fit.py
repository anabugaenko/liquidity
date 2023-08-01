from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats
import pylab as py


def compute_bic(log_likelihood, num_params, num_samples):
    return np.log(num_samples) * num_params - 2 * log_likelihood

def compute_loglikelihood(func, params, x_values, y_values):
    residuals = y_values - func(x_values, *params)
    loglikelihood = -0.5 * np.log(2 * np.pi * residuals ** 2) - residuals ** 2 / 2
    return loglikelihood

def compute_goodness_of_fit(residuals, y_values, params):
    ssr = np.sum(residuals ** 2)  # sum of squared residuals
    sst = np.sum((y_values - np.mean(y_values)) ** 2)  # total sum of squares
    rsquared = 1 - ssr / sst
    n = len(y_values)
    p = len(params)  # number of predictors, assuming params is the parameter array from the fit
    adjusted_rsquared = 1 - (1 - rsquared) * (n - 1) / (n - p - 1)

    # log_likelihood = compute_log_likelihood(ssr, n)
    loglikelihood = -0.5 * np.log(2 * np.pi * residuals ** 2) - residuals ** 2 / 2
    loglikelihood = np.sum(loglikelihood)  # sum over all data points
    bic = compute_bic(loglikelihood, len(params), n)

    return adjusted_rsquared, bic

def _plot_distruibutions(data, **kwargs):
    plt.hist(data, bins=20, density=True, **kwargs)
    plt.xlabel('Residuals', **kwargs)
    plt.ylabel('Probability', **kwargs)
    plt.title('Distribution of residuals', **kwargs)

    # Overlay a normal distribution with the same mean and standard deviation as the residuals
    mu, std = stats.norm.fit(data)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)


def plot_residuals(residuals, **kwargs):
    plt.figure(figsize=(12, 5), **kwargs)

    plt.subplot(1, 2, 1)
    # plot distribution vs normal
    _plot_distruibutions(residuals)

    plt.subplot(1, 2, 2)
    # Q&Q plot
    stats.probplot(residuals, dist="norm", plot=py, **kwargs)
    py.show()

#Much of the rest of this function was inspired by Aaron Clauset powerlaw code, specifically around lines 1748-1822 of
# this version: https://github.com/jeffalstott/powerlaw/blob/master/powerlaw.py
def loglikelihood_ratio(loglikelihoods1, loglikelihoods2, nested=False, normalized_ratio=True):
    """
    This version of the function allows for both the use of normalized or unnormalized R based on the normalized_ratio
    parameter, and adjusts the calculation of p accordingly. Calculates a loglikelihood ratio and the p-value for testing
    which of two functions is more likely to have created a set of observations.

    Parameters
    ----------
    loglikelihoods1 : list or array
        The logarithms of the likelihoods of each observation, calculated from
        a particular functions .
    loglikelihoods2 : list or array
        The logarithms of the likelihoods of each observation, calculated from
        a particular function distribution.
    nested : bool, optional
        Whether one of the two functions that generated the
        likelihoods is a nested version of the other. False by default.
    normalized_ratio : bool, optional
        Whether to return the loglikelihood ratio, R, or the normalized
        ratio R/sqrt(n*variance). True by default.

    Returns
    -------
    R : float
        The loglikelihood ratio of the two sets of likelihoods. If positive,
        the first set of likelihoods is more likely (and so the function
        that produced them is a better fit to the data). If
        negative, the reverse is true. If normalized_ratio is True, this value
        is the normalized loglikelihood ratio, otherwise, it's the unnormalized ratio.
    p : float
        The significance of the sign of R. If below a critical value
        (typically 0.1 or .05 depending on the problem domain) the sign of R is taken to be significant. If above the
        critical value the sign of R is taken to be due to statistical
        fluctuations. p is always computed using the normalized loglikelihood ratio.
    """
    from numpy import sqrt, asarray, inf, log, mean, sum
    from scipy.special import erfc
    from scipy.stats import chi2
    from sys import float_info

    n = float(len(loglikelihoods1))

    if n == 0:
        R = 0
        p = 1
        return R, p

    loglikelihoods1 = asarray(loglikelihoods1)
    loglikelihoods2 = asarray(loglikelihoods2)

    # Clean for extreme values, if any
    min_val = log(10**float_info.min_10_exp)
    loglikelihoods1[loglikelihoods1 == -inf] = min_val
    loglikelihoods2[loglikelihoods2 == -inf] = min_val

    R = sum(loglikelihoods1 - loglikelihoods2)

    mean_diff = mean(loglikelihoods1) - mean(loglikelihoods2)
    variance = sum((loglikelihoods1 - loglikelihoods2 - mean_diff) ** 2) / n

    # Normalize R for calculating p
    R_norm = R / sqrt(n * variance)

    if nested:
        p = 1 - chi2.cdf(abs(2 * R_norm), 1)
    else:
        p = erfc(abs(R_norm) / sqrt(2))


    # Return normalized R only if normalized_ratio is True
    if normalized_ratio:
        R = R_norm

    return R, p
