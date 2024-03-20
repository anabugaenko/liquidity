import pylab
import pandas as pd
from typing import List

from matplotlib import rc
from matplotlib import pyplot as plt
from matplotlib import font_manager as font_manager

from liquidity.util.utils import bin_data_into_quantiles, smooth_outliers
from liquidity.finite_scaling.functional_form import scaling_function, scaling_form, scaling_law


# Set the backend

rc("text", usetex=True)
rc("mathtext", fontset="stix")
rc("axes", labelsize="large")
plt.rcParams["figure.dpi"] = 350
plt.rcParams["text.usetex"] = True
pylab.rcParams["xtick.major.pad"] = "8"
pylab.rcParams["ytick.major.pad"] = "8"
plt.rcParams["text.latex.preamble"] = r"\usepackage{lmodern} \usepackage{amssymb}"


# Plot Constants

ALPHA_LIST = [1, 0.6, 0.4, 0.8, 0.55]
MARKETS_LIST = ["P", "o", "s", "^", "D"]
FACE_COLORS = ["white", "white", "#0E1111", "white", "white"]
LINESTYLE_LIST = ["solid", "--", ":", "-.", "-", "dashed", "dashdot", "dotted"]


class ColorChoice:
    """
    Maps associated asset with respective colormap.
    """

    def __init__(
        self,
        security_name: str,
        dark_line_color: str,
        light_markers_color: str,
        dark_color: str,
        marker: str,
        alpha: float = None,
    ):
        self.security_name = security_name
        self.dark_line_color = dark_line_color
        self.light_markers_color = light_markers_color
        self.dark_color = dark_color
        self.marker = marker
        if alpha is not None:
            self.alpha = alpha


TSLA_COLORS = ColorChoice(
    security_name="TSLA",
    dark_line_color="#0E1111",
    light_markers_color="#F60E1A",  # 3FBAFF
    dark_color="#380404",  # 0E1111
    marker="^",
    alpha=0.65,
)


MSFT_COLORS = ColorChoice(
    security_name="MSFT",
    dark_line_color="#0E1111",
    light_markers_color="#009FF8",
    dark_color="#0E1111",
    marker="o",
    alpha=1,
)


AMZN_COLORS = ColorChoice(
    security_name="AMZN",
    dark_line_color="#0E1111",
    light_markers_color="#FCC400",
    dark_color="#0E1111",
    marker="D",
    alpha=0.6,
)


NFLX_COLORS = ColorChoice(
    security_name="NFLX",
    dark_line_color="#0E1111",
    light_markers_color="#FF8624",
    dark_color="#0E1111",
    marker="s",
    alpha=0.8,
)


EBAY_COLORS = ColorChoice(
    security_name="EBAY",
    dark_line_color="#0E1111",
    light_markers_color="#00FF0C",
    dark_color="#0E1111",
    marker="+",
    alpha=0.95,
)


# Helper functions

def _determine_label(x_col: str, plotting_func: str) -> tuple:
    """
    Determines the labels for the x and y axes based on the column name and plot type.

    Args:
        x_col (str): The name of the x-axis column.
        plotting_func (str): The type of plot ('form' for scaling form or 'law' for scaling law).

    Returns:
        tuple: A tuple containing the x and y axis labels.
    """

    # FIXME: account for normalization by average values at best
    # FIXME: account for general scaling function form (propagator, sigmoid, NN etc)
    if plotting_func == "scaling_form":
        if x_col == "volume_imbalance":
            return (
                r"$\Delta V/V_T$",
                r"$R(\Delta V, T)/R_T$",
            )
        elif x_col == "sign_imbalance":
            return (
                r"$\Delta \mathcal{E}/\mathcal{E}_T$",
                r"$R(\Delta \mathcal{E}, T)/R_T$",
            )
        else:
            return (x_col, r"$R$")

    elif plotting_func == "scaling_law":
        if x_col == "volume_imbalance":
            return (
                r"$\Delta V/V_{D}T^{\varkappa}$",
                r"$R(\Delta V, T)/\mathcal{R}(1)T^{\chi}$",
            )
        elif x_col == "sign_imbalance":
            return (
                r"$\Delta \mathcal{E}/\mathcal{E}_{D}T^{\varkappa}$",
                r"$R(\Delta \mathcal{E}, T)/\mathcal{R}(1)T^{\chi}$",
            )
        else:
            return (x_col, r"$R$")

    # Default return if plot_type is not recognized
    return (x_col, r"$R$")


# Plotting functions

def plot_scaling_function(
    aggregate_impact,
    scaling_params,
    # line_color,
    markers_color,
    ylim=None,
    xlim=None,
    master_curve=False,
    save=False,
    filename=None,
    qcut: List[int] = 25,
    plotting_func: str ="scaling_law",
    response_column: str = "R_cond",
    imbalance_column: str ="volume_imbalance",
    binning_frequencies: List[float] = [5, 10, 20, 50, 100],
):
    fig = plt.figure(figsize=(4.2, 4))
    ax = fig.gca()
    ax.grid(False)

    binned_data = []
    for T in binning_frequencies:
        aggregate_impact_data = aggregate_impact.copy()
        aggregate_impact_data = aggregate_impact_data[aggregate_impact_data["T"] == T][
            ["T", imbalance_column, response_column]
        ]
        smoothed_data = smooth_outliers(aggregate_impact_data, columns=[imbalance_column, response_column])
        all_binned_data = bin_data_into_quantiles(
            aggregate_impact_data,
            x_col=imbalance_column,
            y_col=response_column,
            q=qcut,
            duplicates="drop",
        )

        binned_data.append(all_binned_data)
        all_binned_data = pd.concat(binned_data, axis=0)

    for indx, T in enumerate(binning_frequencies):
        plotting_data = all_binned_data[all_binned_data["T"] == T][
            ["T", imbalance_column, response_column]
        ]

        t_values = plotting_data["T"].values
        r_values = plotting_data[response_column].values
        imbalance_values = plotting_data[imbalance_column].values
        orderflow_imbalance = pd.DataFrame(
            {"T": t_values, "imbalance": imbalance_values}
        )

        #  Extract model predictions
        if plotting_func == "scaling_law":
            y_hat = scaling_law(orderflow_imbalance, *scaling_params)
        else:
            y_hat = scaling_form(orderflow_imbalance, *scaling_params)

        # plt.plot(
        #     imbalance_values,
        #     y_hat,
        #     markersize=5.5,
        #     linewidth=0.8,
        #     markeredgewidth=0.8,
        #     linestyle=LINESTYLE_LIST[indx % len(LINESTYLE_LIST)],
        #     marker=MARKETS_LIST[indx % len(MARKETS_LIST)],
        #     markerfacecolor=markers_color,
        #     label="x",
        #     color=line_color,
        #     alpha=ALPHA_LIST[indx % len(ALPHA_LIST)],
        # )

        plt.scatter(imbalance_values, r_values)
        plt.plot(imbalance_values, y_hat, label=f"T = {T}")


    xlabel, ylabel = _determine_label(imbalance_column, plotting_func="scaling_form")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in", width=0.5, size=3
    )
    ax.minorticks_off()

    if ylim is not None:
        plt.ylim(-ylim, ylim)
    if xlim is not None:
        plt.xlim(-xlim, xlim)

    font = font_manager.FontProperties(size=14)
    legend = ax.legend(markerfirst=True, prop=font, labelspacing=0.05)

    for indx, T in enumerate(binning_frequencies):
        legend.get_texts()[indx].set_text(f"$T = {int(T)}$")
        legend.get_lines()[indx].set_markeredgecolor("#1A1919")
        legend.get_lines()[indx].set_markerfacecolor("white")
        legend.get_lines()[indx].set_color("#0E1111")
        legend.get_lines()[indx].set_alpha(alpha=0.9)
        legend.get_lines()[indx].set_linewidth(0.5)
        legend.get_lines()[indx].set_ms(4)

    legend.markerscale = 0.3
    # legend.get_frame().set_edgecolor('w')

    for spine in ax.spines.values():
        spine.set_edgecolor("#1A1919")
        spine.set_linewidth(0.5)

    plt.locator_params(axis="x", nbins=4)
    plt.locator_params(axis="y", nbins=10)
    ax.tick_params(labelsize=12)

    plt.tight_layout(pad=0.25)

    if save and filename is not None:
        plt.savefig(f"../../plots/model_collapse/{filename}.pdf")

    plt.show()


def plot_collapsed_scaling_function(
    aggregate_impact,
    scaling_params,
    #line_color,
    markers_color,
    ylim=None,
    xlim=None,
    save=False,
    filename=None,
    master_curve=None,
    qcut: List[int] = 31,
    response_column: str = "R_cond",
    imbalance_column="volume_imbalance",
    binning_frequencies: List[float] = [5, 10, 20, 50, 100],
):
    fig = plt.figure(figsize=(4.2, 4))
    ax = fig.gca()
    ax.grid(False)

    binned_data = []
    for T in binning_frequencies:
        aggregate_impact_data = aggregate_impact.copy()
        aggregate_impact_data = aggregate_impact_data[aggregate_impact_data["T"] == T][
            ["T", imbalance_column, response_column]
        ]

        smoothed_data = smooth_outliers(aggregate_impact_data,
                                        columns=[imbalance_column, response_column])
        all_binned_data = bin_data_into_quantiles(
            smoothed_data,
            x_col=imbalance_column,
            y_col=response_column,
            q=qcut,
            duplicates="drop",
        )

        binned_data.append(all_binned_data)
        all_binned_data = pd.concat(binned_data, axis=0)

    for indx, T in enumerate(binning_frequencies):
        plotting_data = all_binned_data[all_binned_data["T"] == T][
            ["T", imbalance_column, response_column]
        ]

        r_values = plotting_data[response_column].values
        imbalance_values = plotting_data[imbalance_column].values

        #  Scaling function master curve paramters
        chi, kappa, alpha, beta, CONST = scaling_params
        y_hat = scaling_function(x=imbalance_values, alpha=alpha, beta=beta) * CONST

        # plt.plot(
        #     imbalance_values,
        #     y_hat,
        #     linewidth=0,  # set linewidth to 0 to remove lines
        #     linestyle="",  # remove linestyle
        #     markersize=5.5,
        #     markeredgewidth=0.8,
        #     marker=MARKETS_LIST[indx % len(MARKETS_LIST)],
        #     markerfacecolor=markers_color,
        #     label="x",
        #     color=line_color,
        #     alpha=ALPHA_LIST[indx % len(MARKETS_LIST)],
        # )

        plt.scatter(imbalance_values, r_values, label=f"T = {T}")

    if master_curve is not None:
        plotting_data = all_binned_data[
            all_binned_data["T"] == min(binning_frequencies)
        ][["T", imbalance_column, response_column]]
        imbalance_values = plotting_data[imbalance_column].values

        chi, kappa, alpha, beta, CONST = scaling_params
        y_hat = scaling_function(x=imbalance_values, alpha=alpha, beta=beta) * CONST
        # plt.plot(
        #     imbalance_values,
        #     y_hat,
        #     linestyle="solid",
        #     linewidth=0.9,
        #     label=master_curve,
        #     color="black",
        # )
        plt.plot(imbalance_values, y_hat, label=f"T = {T}")

    xlabel, ylabel = _determine_label(imbalance_column, plotting_func="scaling_law")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in", width=0.5, size=3
    )
    ax.minorticks_off()

    if ylim is not None:
        plt.ylim(-ylim, ylim)
    if xlim is not None:
        plt.xlim(-xlim, xlim)

    font = font_manager.FontProperties(size=14)
    legend = ax.legend(markerfirst=True, prop=font, labelspacing=0.05)

    # for indx, T in enumerate(binning_frequencies):
    #     legend.get_texts()[indx].set_text(f"$T = {int(T)}$")
    #     legend.get_lines()[indx].set_markeredgecolor("#1A1919")
    #     legend.get_lines()[indx].set_markerfacecolor("white")
    #     legend.get_lines()[indx].set_color("#0E1111")
    #     legend.get_lines()[indx].set_alpha(alpha=0.9)
    #     legend.get_lines()[indx].set_linewidth(0.5)
        # legend.get_lines()[indx].set_ms(4)

    legend.markerscale = 0.3
    # legend.get_frame().set_edgecolor('w')

    for spine in ax.spines.values():
        spine.set_edgecolor("#1A1919")
        spine.set_linewidth(0.5)

    plt.locator_params(axis="x", nbins=4)
    plt.locator_params(axis="y", nbins=10)
    ax.tick_params(labelsize=12)

    plt.tight_layout(pad=0.25)

    if save and filename is not None:
        plt.savefig(f"../../plots/model_collapse/{filename}.pdf")

    plt.show()


def plot_model_predictions():
    pass