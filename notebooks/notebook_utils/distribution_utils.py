import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from typing import Dict, Union, List, Tuple, Optional

import powerlaw
from powerlaw import Fit, plot_pdf, plot_cdf, plot_ccdf


# Helper functions
def fit_powerlaw(
    data_dict: Union[str, Dict[str, List[float]]],
    series: Union[List[float], None] = None,
    filename: Optional[str] = None,
) -> Dict:
    """
    Fits a power law to the provided data.

    Parameters:
    - data_dict (Union[str, Dict[str, List[float]]]): Either a string label (e.g. stock name) or a dictionary mapping
      labels to data series.
    - series (Union[List[float], None], optional): Data series if a single label is provided. Defaults to None.
    - filename (str, optional): Path to save serialized fit objects. If provided, saves the data. Defaults to None.

    Returns:
    - Dict: Dictionary mapping labels to fit results.

    Notes:
    - If providing a single label and series, use the `data_dict` as a string and provide the `series`.
    - If providing multiple labels and their series, use the `data_dict` as a dictionary mapping labels to series.
    """
    # Determine mapping from labels to series
    if isinstance(data_dict, str) and series is not None:
        labels_to_series = {data_dict: series}
    elif isinstance(data_dict, dict):
        labels_to_series = data_dict
    else:
        raise ValueError(
            "Invalid input. Please provide a valid label and series or a dictionary mapping labels to series."
        )

    # Fit the series for each label
    fit_results = {}
    for label, s in labels_to_series.items():
        fit = Fit(s.dropna(), discrete=False)
        fit_results[label] = fit

    # Serialize the fit objects
    serialized_fit_objects = {}
    for label, fit in fit_results.items():
        data = {"data": fit.data.tolist(), "xmin": fit.xmin, "xmax": fit.xmax}
        serialized_fit_objects[label] = data

    # Save to a file if filename is provided
    if filename:
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if not os.path.exists(filename):
            with open(filename, "wb") as f:
                pickle.dump(serialized_fit_objects, f)
        else:
            print(f"File {filename} already exists. Skipping.")

    return fit_results


def load_fit_objects(filename):
    """
    Loads serialized fit objects from a file and returns a dictionary of fit objects.

    :param filename: The name of the file where the serialized fit is saved.
    :return: A dictionary mapping stock names to their respective fit objects.
    """
    with open(filename, "rb") as f:
        loaded_data = pickle.load(f)

    fit_objects = {}
    for stock_name, data in loaded_data.items():
        fit = Fit(data["data"], xmin=data["xmin"], xmax=data["xmax"], discrete=False)
        fit_objects[stock_name] = fit

    return fit_objects


def get_fitting_params(
    fit_input: Union[Dict[str, Fit], Tuple[str, Fit]], distribution: str
) -> pd.DataFrame:
    """
    Retrieves fitting parameters for a specified distribution across stocks.

    :param fit_input: Either a tuple containing a single stock name and its Fit object or a dictionary mapping stock names to their respective Fit objects.
    :param distribution: Alternative distribution name for comparison. Supported distributions are:
    'power_law', 'lognormal', 'exponential','truncated_power_law', 'stretched_exponential',
    and 'lognormal_positive'.

    :return: DataFrame containing fitting parameters for the given distribution.
    """

    # Check input type and adjust accordingly
    if isinstance(fit_input, tuple) and len(fit_input) == 2:
        stock_name, fit = fit_input
        fit_objects = {stock_name: fit}
    elif isinstance(fit_input, dict):
        fit_objects = fit_input
    else:
        raise ValueError(
            "Invalid input. Please provide a valid stock name and fit or a dictionary mapping stock names to fits."
        )

    param_map = {
        "power_law": ["alpha"],
        "lognormal": ["mu", "sigma"],
        "exponential": ["Lambda"],
        "truncated_power_law": ["alpha", "Lambda", "xmin"],
        "stretched_exponential": ["Lambda", "beta"],
        "lognormal_positive": ["mu", "sigma"],
    }

    def get_params(fit, dist) -> List:
        """Utility to fetch distribution parameters and handle errors."""
        try:
            return [
                getattr(fit, dist).__getattribute__(param) for param in param_map[dist]
            ]
        except AttributeError:
            return [np.nan] * len(param_map[dist])

    results = []
    for stock_name, fit in fit_objects.items():
        base_result = {"Stock": stock_name, "Distribution": distribution}
        params = get_params(fit, distribution)
        base_result.update(zip(param_map[distribution], params))
        base_result.update(
            {
                "xmin": fit.xmin,
                "KS Distance": getattr(fit, distribution).D
                if hasattr(fit, distribution)
                else np.nan,
            }
        )
        results.append(base_result)

    return pd.DataFrame(results)


def distribution_compare(
    fit_input: Union[Dict[str, powerlaw.Fit], Tuple[str, powerlaw.Fit]], distribution: str
) -> pd.DataFrame:
    """
    Compares power law distribution to a specified alternative distribution across stocks.

    :param fit_input: Either a tuple containing a single stock name and its Fit object or a dictionary mapping stock names to their respective Fit objects.
    :param distribution: Alternative distribution name for comparison. Supported distributions are:
    'power_law', 'lognormal', 'exponential','truncated_power_law', 'stretched_exponential',
    and 'lognormal_positive'.

    :return: DataFrame containing comparison results.
    """

    # Check input type and adjust accordingly
    if isinstance(fit_input, tuple) and len(fit_input) == 2:
        stock_name, fit = fit_input
        fit_objects = {stock_name: fit}
    elif isinstance(fit_input, dict):
        fit_objects = fit_input
    else:
        raise ValueError(
            "Invalid input. Please provide a valid stock name and fit or a dictionary mapping stock names to fits."
        )

    param_map = {
        "power_law": ["alpha"],
        "lognormal": ["mu", "sigma"],
        "exponential": ["Lambda"],
        "truncated_power_law": ["alpha", "Lambda", "xmin"],
        "stretched_exponential": ["Lambda", "beta"],
        "lognormal_positive": ["mu", "sigma"],
    }

    def get_params(fit, dist) -> List:
        """Utility to fetch distribution parameters and handle errors."""
        try:
            return [
                getattr(fit, dist).__getattribute__(param) for param in param_map[dist]
            ]
        except AttributeError:
            return [np.nan] * len(param_map[dist])

    results = []
    for stock_name, fit in fit_objects.items():
        base_result = {"Stock": stock_name, "Alternative Distribution": distribution}
        params = get_params(fit, distribution)
        base_result.update(zip(param_map[distribution], params))
        base_result.update(
            {
                "xmin": fit.xmin,
                "Power Law Alpha": fit.power_law.alpha,
                "KS Distance (Power Law)": fit.power_law.D,
                "KS Distance (" + distribution + ")": getattr(fit, distribution).D
                if hasattr(fit, distribution)
                else np.nan,
                "Loglikelihood Ratio": fit.distribution_compare(
                    "power_law", distribution, normalized_ratio=True
                )[0],
                "p-value": fit.distribution_compare(
                    "power_law", distribution, normalized_ratio=True
                )[1],
            }
        )
        results.append(base_result)

    return pd.DataFrame(results)


def plot_distributions(stock_name, data):
    """
    Plots PDF, CDF, and CCDF for given data.

    Parameters:
    - data: Return data for the stock
    - stock_name: Name of the stock
    """

    # Drop NaN values and filter out non-positive values
    data_cleaned = data.dropna()
    data_cleaned = data_cleaned[data_cleaned > 0]

    # Check if data_cleaned is empty
    if len(data_cleaned) == 0:
        print(f"No valid data for {stock_name}. Skipping...")
        return

    plt.figure(figsize=(15, 4))
    gs = gridspec.GridSpec(1, 4)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[0, 2])

    plt.suptitle(f"{stock_name}")

    # Plot PDF
    plot_pdf(data_cleaned, color="b", linewidth=2, ax=ax1)
    ax1.set_title("PDF")
    ax1.grid(False)

    # Plot CDF
    plot_cdf(data_cleaned, color="r", linewidth=2, ax=ax2)
    ax2.set_title("CDF")
    ax2.grid(False)

    # Plot CCDF
    plot_ccdf(data_cleaned, color="g", linewidth=2, ax=ax3)
    ax3.set_title("CCDF")
    ax3.grid(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_fit_objects(
    fit_input: Union[Dict[str, powerlaw.Fit], Tuple[str, powerlaw.Fit]],
    distributions: List[str] = [
        "power_law",
        "truncated_power_law",
        "exponential",
        "lognormal",
    ],
) -> None:
    """
    Plot the Empirical CCDF, Power Law Fit, and comparison between Power Law
    and the specified distributions for the given stock fit objects.

    :param fit_input: Either a tuple containing a single stock name and its Fit
    object or a dictionary mapping stock names to their respective Fit objects.
    :param distributions: List of distributions to include in comparison. Supported are:
    'power_law', 'lognormal', 'exponential'.

    :return: None, but will display the plots.
    """
    # Handle input type and adjust accordingly
    if isinstance(fit_input, tuple) and len(fit_input) == 2:
        stock_name, fit = fit_input
        fit_objects = {stock_name: fit}
    elif isinstance(fit_input, dict):
        fit_objects = fit_input
    else:
        raise ValueError(
            "Invalid input. Please provide a valid stock name and fit or a dictionary mapping stock names to fits."
        )

    num_stocks = len(fit_objects)
    if num_stocks == 1:
        fig, axs = plt.subplots(3, 1, figsize=(6, 14))
        axs = np.expand_dims(axs, axis=1)
    else:
        fig, axs = plt.subplots(3, num_stocks, figsize=(18, 14))

    # Color map for different distributions
    color_map = {
        "power_law": "g",
        "truncated_power_law": "c",
        "lognormal": "y",
        "exponential": "r",
        "stretched_exponential": "m",
        "lognormal_positive": "b",
    }

    # Legend setup based on distributions
    legend_elements = [
        mlines.Line2D(
            [0],
            [0],
            color="b",
            marker=".",
            linestyle="None",
            markersize=10,
            label="Empirical Data",
        )
    ]
    for dist in distributions:
        if dist in color_map:
            legend_elements.append(
                mlines.Line2D(
                    [0],
                    [0],
                    color=color_map[dist],
                    linestyle="--",
                    markersize=10,
                    label=dist.replace("_", " ").capitalize(),
                )
            )

    for i, (stock_name, fit) in enumerate(fit_objects.items()):
        x = np.linspace(min(fit.data), max(fit.data), num=1000)

        # Empirical CCDF plotting
        empirical_data = fit.ccdf()
        axs[0, i].loglog(empirical_data[0], empirical_data[1], "b.")
        axs[0, i].set_title(f"{stock_name} - Empirical CCDF")
        axs[0, i].grid(False)

        # Power Law Fit plotting
        y_powerlaw = fit.power_law.ccdf(x)
        axs[1, i].loglog(empirical_data[0], empirical_data[1], "b.")
        axs[1, i].loglog(x, y_powerlaw, "g--")
        axs[1, i].set_title(f"{stock_name} - Power Law Fit")
        axs[1, i].grid(False)

        # Distributions comparison plotting
        axs[2, i].loglog(empirical_data[0], empirical_data[1], "b.")
        for dist in distributions:
            if dist == "power_law":
                y = fit.power_law.ccdf(x)
            elif dist == "lognormal":
                lognormal_fit = powerlaw.Fit(
                    fit.data, discrete=False, xmin=min(fit.data)
                ).lognormal
                y = lognormal_fit.ccdf(x)
            elif dist == "exponential":
                exponential_fit = powerlaw.Fit(
                    fit.data, discrete=False, xmin=min(fit.data)
                ).exponential
                y = exponential_fit.ccdf(x)
            axs[2, i].loglog(x, y, color_map[dist] + "--")
        axs[2, i].set_title(f"{stock_name} - Distributions Comparison")
        axs[2, i].grid(False)
        axs[2, i].legend(
            handles=legend_elements, loc="upper right", fontsize="small", frameon=False
        )

    plt.tight_layout()
    plt.show()
