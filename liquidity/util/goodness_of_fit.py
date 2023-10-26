import numpy as np
from scipy import stats
from typing import List, Tuple
from sklearn.metrics import mean_absolute_error


# TODO: remove bic from loglikelihoods, then we don't have to make assumptions about distribution of residuals
def loglikelihoods(data: List[float]) -> List[float]:
    """
    Compute the log likelihood of the data, hence  incorporates
    the variance of the data and assumes a certain distribution

    Parameters:
    data (List[float]): The data for which the log likelihood is to be computed.

    Returns:
    float: The log likelihoods of the data.

    """

    # Compute the standard deviation of the data as an initial parameter
    data_std = np.std(data)

    # Compute the log probability density function of the data
    loglikelihoods = stats.norm.logpdf(data, loc=0, scale=data_std)

    return loglikelihoods


def compute_bic_from_loglikelihood(log_likelihood: float, num_params: int, num_samples: int) -> float:
    """
    Compute the Bayesian Information Criterion (BIC) for a given model. Note, the BIC is computed as:

        BIC = log(n) * k - 2 * log(L)

    where n is the number of samples, k is the number of parameters, and L is the maximum likelihood
    that incorporates the variance of the data and assumes a certain distribution (usually normal).

    Parameters:
    - log_likelihood (float): The log-likelihood of the model.
    - num_params (int): The number of parameters in the model.
    - num_samples (int): The number of samples in the dataset.

    Returns:
    - float: The BIC for the model.
    """

    # Compute the BIC
    BIC = np.log(num_samples) * num_params - 2 * log_likelihood

    return BIC


def compute_bic_from_residuals(residuals: np.ndarray, num_parameters: int) -> float:
    """
    Compute the Bayesian Information Criterion (BIC) given the residuals using:

        BIC = n * log(RSS/n) + k * log(n)

    where n is the number of samples, k is the number of parameters and RSS is the residual sum of squares.
    In this way, it uses the residuals directly without making any assumptions about the
    distribution of the data (or residuals).


    Parameters:
    residuals (np.ndarray): The residuals (difference between actual and predicted values).
    num_parameters (int): The number of parameters in the model.

    Returns:
    float: The computed BIC value.

    """

    n = len(residuals)
    RSS = np.sum(residuals**2)
    BIC = n * np.log(RSS / n) + num_parameters * np.log(n)
    return BIC


def compute_rsquared(residuals, y_values, params):
    ssr = np.sum(residuals**2)
    sst = np.sum((y_values - np.mean(y_values)) ** 2)
    rsquared = 1 - ssr / sst

    # Compute the Adjusted R-squared value
    n = len(y_values)
    p = len(params)
    adjusted_rsquared = 1 - (1 - rsquared) * (n - 1) / (n - p - 1)

    return rsquared, adjusted_rsquared


def get_goodness_of_fit(
    residuals: List[float],
    y_values: List[float],
    params: List[float],
    model_predictions: List[float],
    bic_method="residuals",
    verbose=False,
) -> Tuple[float, float, float, float]:
    """
    Compute the goodness of fit of a model.

    Parameters:
    - residuals (List[float]): The residuals of the model.
    - y_values (List[float]): The actual observed values.
    - params (List[float]): The parameters of the model.
    - model_predictions (List[float]): The predicted values by the model.
    - bic_method (str): The method to use for BIC computation ('log_likelihood' or 'residuals'). Default is 'residuals'.

    Returns:
    - float: The Kolmogorov-Smirnov statistic, D.
    - float: The BIC.


    - float: MAPE metric.
    - float: The adjusted R-squared value.
    """
    # Compute the R-squared value
    rsquared, adjusted_rsquared = compute_rsquared(residuals, y_values, params)

    # MAPE metric is an error metric that is less sensitive to outliers than root Mean Squared Error (MAE).

    mae = mean_absolute_error(y_values, model_predictions)
    mape = np.mean(np.abs((y_values - model_predictions) / y_values)) * 100


    return mae, mape, adjusted_rsquared
