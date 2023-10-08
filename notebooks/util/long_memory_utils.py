import os
import pickle
import matplotlib.pyplot as plt
from typing import Dict, Union, List

from powerlaw_function import Fit

from hurst_exponent.acf import linear_acf, nonlinear_acf

from pathos.multiprocessing import ProcessingPool

# Helper functions


def compute_and_load_acf(
    filename: str,
    acf_series_dict: Dict[str, Union[float, int]],
    option: str = "linear",
    acf_range=1001,
    processes: int = 4,
) -> Dict[str, Union[float, int]]:
    """
    Computes and loads autocorrelation functions (ACF) for given returns based on the specified option.

    :param filename: File name to load/save data.
    :param acf_series_dict: A dictionary mapping stock names to their respective ACF data to series.
    :param option: 'linear' or 'nonlinear'.
    :param acf_range: Range for the ACF computation.
    :param processes: Number of processes for parallel processing.

    :return: Dictionary containing ACF results.
    """

    if not os.path.exists(filename):
        # Define a function that computes the specified ACF for each stock.
        def compute_acf(stock_series):
            stock, series = stock_series
            result = None
            if option == "linear":
                result = linear_acf(series, acf_range)
            elif option == "nonlinear":
                result = nonlinear_acf(series, acf_range)
            return stock, result

        pool = ProcessingPool(processes=processes)
        results = pool.map(compute_acf, acf_series_dict.items())
        pool.close()
        pool.join()

        # Organize results
        acf_results = {stock: result for stock, result in results}

        # Save data for lazy loading
        with open(filename, "wb") as f:
            pickle.dump(acf_results, f)

    # Load data
    with open(filename, "rb") as f:
        acfs = pickle.load(f)

    return acfs


def plot_acf_difference(
    stock_name: str, linear_acfs: Dict[str, List[float]], nonlinear_acfs: Dict[str, List[float]], acf_range: int = 1001
) -> None:
    """
    Plot and compare the difference between linear and nonlinear ACF for a specific stock.

    :param stock_name: Name of the stock.
    :param linear_acfs: Linear ACFs data.
    :param nonlinear_acfs: Nonlinear ACFs data.
    :param acf_range: Range of ACF computation.

    :return: None
    """

    linear_acf, nonlinear_acf = linear_acfs[stock_name], nonlinear_acfs[stock_name]
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

    plt.suptitle(f"Linear vs nonlinear ACF across lags for {stock_name} MO Returns")
    plt.show()

    print(f"{stock_name} Max difference: {max(difference)}")


from typing import Tuple, Dict, Any, List, Union
import pandas as pd
from powerlaw import Fit


def get_acf_params(stock_data_tuple: Tuple[str, List[float]]) -> Tuple[Union[None, Dict[str, Any]], Any]:
    """
    Fits a power law to the autocorrelation function (ACF) of a stock's data
    to extract the long-memory parameter γ (gamma/gramma).

    A process is characterized as long-memory if, as \( k \to \infty \):

    .. math::
        C(l) \sim \frac{c_\infty}{l^\gamma}

    where \( 0 < \gamma < 1 \). The smaller γ, the longer the memory.

    Parameters:
    - stock_data_tuple: Tuple containing stock name and corresponding data.

    Returns:
    - Dictionary containing the fitted parameters and the stock name, or None.
    - Fitted power law object, or None.
    """

    stock, data = stock_data_tuple
    fit = Fit(data)  # Fitting powerlaw to the DataFrame passed here
    fit_dict = fit.powerlaw.to_dictionary()

    if fit_dict.get("function_name") == "powerlaw":
        gamma = fit.powerlaw.params.alpha  # γ is analogous to 'alpha' in this context
        fit_dict.update({"gamma": gamma, "stock": stock})  # rename alpha to gamma
        return fit_dict, fit  # Return both dictionary and fit object
    return None, None


fit_results_list = []
fit_objects = {}  # Dictionary to store fit objects
