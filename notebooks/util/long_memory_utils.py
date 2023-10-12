import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, List, Union

from powerlaw_function import Fit
from multiprocessing import Pool, cpu_count

from hurst_exponent.acf import linear_acf, nonlinear_acf
from hurst_exponent.hurst_exponent import standard_hurst, generalized_hurst


# Helper functions

# Define a pool worker function
def pool_worker(args):
    stock, values, option, acf_range = args
    return compute_acf(stock, values, option=option, acf_range=acf_range)

def compute_acf(stock: str,
                data: Union[float, int],
                option: str = "linear",
                acf_range: int = 1001) -> Tuple[str, Union[float, int]]:
    """
    Computes the autocorrelation function (ACF) for a given stock's data based on the specified option.

    Parameters:
    - stock (str): Name of the stock.
    - data (Union[float, int]): Time series data.
    - option (str): 'linear' or 'nonlinear' to specify the ACF computation method.
    - acf_range (int): Range for the ACF computation.

    Returns:
    - Tuple containing the stock name and its ACF result.
    """
    result = None
    if option == "linear":
        result = linear_acf(data, acf_range)
    elif option == "nonlinear":
        result = nonlinear_acf(data, acf_range)
    return stock, result


def construct_xy(sample: pd.Series, name: str) -> pd.DataFrame:
    """
    Constructs a DataFrame with x and y values for plotting Autocorrelation Function (ACF) from a given sample series.

    The function creates x-values based on the index range of the sample, starting from 1.
    The y-values are directly taken from the provided sample series.

    :param sample: A pandas Series representing y-values (e.g., ACF values).
    :param name: A string representing the name of the sample (e.g., stock name) for error reporting.

    :return: A pandas DataFrame with columns 'x_values' and 'y_values' ready for plotting.

    :raises ValueError: If the size of the given sample does not match the constructed y-values.
    """

    y_values = list(sample)
    if len(sample) != len(y_values):
        raise ValueError(f"Sample sizes mismatch for {name}.")

    xy_df = pd.DataFrame({
        'x_values': range(1, len(y_values) + 1),
        'y_values': y_values
    })

    return xy_df


def compute_acfs(filename: str,
                 data: Dict[str, Union[float, int]],
                 option: str = "linear",
                 acf_range: int = 1001,
                 processes: int = 4) -> Dict[str, Union[float, int]]:
    """
    Computes and loads autocorrelation functions (ACF) for given returns based on the specified option.

    :param filename: File name to load/save data.
    :param data: A dictionary mapping strings (e.g., stock names) to their respective data series.
    :param option: 'linear' or 'nonlinear'.
    :param acf_range: Range for the ACF computation.
    :param processes: Number of processes for parallel processing.

    :return: Dictionary containing ACF results.
    """
    if not os.path.exists(filename):
        # Ensure we don't request more processes than available CPUs
        processes = min(processes, cpu_count())

        # Adjust data items to include the option and acf_range
        task_args = [(stock, values, option, acf_range) for stock, values in data.items()]

        # Use a context manager for the pool
        with Pool(processes=processes) as pool:
            results = pool.map(pool_worker, task_args)

        # Organize results
        acf_results = {stock: result for stock, result in results}

        # Save data for lazy loading
        with open(filename, "wb") as f:
            pickle.dump(acf_results, f)

    # Load data
    with open(filename, "rb") as f:
        acfs = pickle.load(f)

    return acfs


def compute_hurst_exponent(random_variate: str,
                           stock: str,
                           data: pd.Series,
                           method: str = 'standard', **kwargs) -> Tuple[Union[None, Dict[str, Any]], Any]:
    """
    Computes the Hurst exponent for a given stock's data using specified methods.

    Parameters:
    - stock (str): Name of the stock.
    - random_variate (str): Type of random variate considered for the analysis.
    - data (pd.Series): Time series data of stock returns.
    - method (str): Either 'standard' or 'generalized' to compute Hurst exponent.

    Returns:
    - Dictionary containing the fitted parameters, Hurst value, stock name, and random variate type.
    - Fitted power law object, or None.
    """
    # Computing the Hurst exponent based on the method
    if method == 'standard':
        hurst_val, fit = standard_hurst(data, **kwargs)
    elif method == 'generalized':
        hurst_val, fit = generalized_hurst(data, **kwargs)
    else:
        raise ValueError("Invalid method provided. Choose either 'standard' or 'generalized'.")

    fit_dict = fit.powerlaw.to_dictionary()

    # Update the dictionary with Hurst values, stock name, and random variate
    if fit_dict.get("function_name") == "powerlaw":
        fit_dict.update({
            f'{method}_hurst': hurst_val,
            'stock': stock,
            'random_variate': random_variate
        })
        return fit_dict, fit
    return None, None


def get_acf_params(stock, data, **kwargs) -> Tuple[Union[None, Dict[str, Any]], Any]:
    """
    Fits a power law to the autocorrelation function (ACF) of a stock's data
    to extract the long-memory parameter γ (gamma).

    A process is characterized as long-memory if, as \( x \to \infty \):

    .. math::
        C(l) \sim \frac{c_\infty}{l^\gamma}

    where \( 0 < \gamma < 1 \). The smaller γ, the longer the memory.

    Parameters:
    - stock_data_tuple: Tuple containing stock name and corresponding data.

    Returns:
    - Dictionary containing the fitted parameters and the stock name, or None.
    - Fitted power law object, or None.
    """

    fit = Fit(data,  **kwargs)  # Fitting powerlaw to the DataFrame passed here
    fit_dict = fit.powerlaw.to_dictionary()

    if fit_dict.get("function_name") == "powerlaw":
        gamma = fit.powerlaw.params.alpha  # γ is analogous to 'alpha' in this context
        fit_dict.update({"gamma": gamma, "stock": stock})  # rename alpha to gamma
        return fit_dict, fit
    return None, None


def plot_acf_difference(
        stock_name: Union[str, Dict[str, List[float]]],
        linear_acfs: Dict[str, List[float]],
        nonlinear_acfs: Dict[str, List[float]],
) -> None:
    """
    Plot and compare the difference between linear and nonlinear ACF for a specific stock
    or multiple stocks.

    :param stock_name: Name of the stock or a dictionary of multiple stock data.
    :param linear_acfs: Linear ACFs data.
    :param nonlinear_acfs: Nonlinear ACFs data.

    :return: None
    """
    if isinstance(stock_name, str):
        stocks = [stock_name]
    else:
        stocks = list(stock_name.keys())

    for stock in stocks:
        linear_acf, nonlinear_acf = linear_acfs[stock], nonlinear_acfs[stock]
        difference = [x1 - x2 for x1, x2 in zip(linear_acf, nonlinear_acf)]

        plt.figure(figsize=(14, 4))

        # Original scale
        plt.subplot(1, 3, 1)
        plt.plot(linear_acf, label="Linear")
        plt.plot(nonlinear_acf, label="Nonlinear", color="green")
        plt.legend(frameon=False)
        plt.grid(False)

        # Log scale
        plt.subplot(1, 3, 2)
        plt.plot(linear_acf, label="Linear")
        plt.plot(nonlinear_acf, label="Nonlinear", color="green")
        plt.loglog()
        plt.legend(frameon=False)
        plt.grid(False)

        # Difference
        plt.subplot(1, 3, 3)
        plt.plot(difference, label="Difference", color="red")
        plt.legend(frameon=False)
        plt.grid(False)

        plt.suptitle(f"Linear vs nonlinear ACF across lags for {stock}")
        plt.show()

        print(f"{stock} Max difference: {max(difference)}")