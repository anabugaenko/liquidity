import os
import pickle
import matplotlib.pyplot as plt
from typing import Dict, Union, Tuple

from hurst_exponent.acf import linear_acf, nonlinear_acf

from pathos.multiprocessing import ProcessingPool

# Helper functions

def compute_and_load_acf(filename: str,
                         acf_series_dict: Dict[str, Union[float, int]],
                         option: str = 'linear',
                         acf_range=1001,
                         processes: int = 4) -> Dict[str, Union[float, int]]:
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
            if option == 'linear':
                result = linear_acf(series, acf_range)
            elif option == 'nonlinear':
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


def plot_acf_difference(stock_name, linear_acfs, nonlinear_acfs, acf_range=1001):

    linear_acf = linear_acfs[stock_name]
    nonlinear_acf = nonlinear_acfs[stock_name]

    plt.figure(figsize=(14, 4))

    # Original scale
    plt.subplot(1, 3, 1)
    plt.plot(range(acf_range), linear_acf, label='Linear')
    plt.plot(range(acf_range), nonlinear_acf, label='Nonlinear', color='green')
    plt.grid(False)
    plt.legend(frameon=False)

    # log scale
    plt.subplot(1, 3, 2)
    plt.plot(range(acf_range), linear_acf, label='Linear')
    plt.plot(range(acf_range), nonlinear_acf, label='Nonlinear', color='green')
    plt.grid(False)
    plt.loglog()
    plt.legend(frameon=False)
    plt.suptitle(f'Linear vs nonlinear ACF across lags for {stock_name} MO Returns')

    # Difference
    plt.subplot(1, 3, 3)
    difference = [x1-x2 for x1, x2 in zip(linear_acf, nonlinear_acf)]
    plt.plot(range(acf_range), difference, label="Difference", color='red')
    plt.legend(frameon=False)
    plt.grid(False)
    plt.suptitle(f'Linear vs nonlinear ACF across lags for {stock_name} MO Returns')

    print(f'{stock_name} Max difference: {max(difference)}')

    plt.show()