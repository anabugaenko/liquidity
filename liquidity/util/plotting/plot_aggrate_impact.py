import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import rc
import matplotlib.font_manager as font_manager
import pylab

from liquidity.response_functions.fitting import scaling_function
from liquidity.util.plotting.util import get_data_for_plotting


plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{lmodern} \usepackage{amssymb}"

X_AXIS_LABELS = {
    "vol": r"$\it{\Delta{V}/V_{D}\langle{V_D}\rangle}$",
    "vol_renorm": r"$\it{\Delta{V}/V_{D}T^{\xi_v}}$",
    "sign": r"$\it{\Delta{\mathcal{E}}/\mathcal{E}_{D}\langle{\mathcal{E}_{D}}\rangle}$",
    "sign_renorm": r"$\it{\mathcal{E}'/\mathcal{E}_{D}T^{\xi}}$",
}

MARKERS_LIST = ["o", "X", "s", "P", "D"]

COLORS_LIST = ["#FF00AC", "#FCC400", "#00D258", "#BC04FB", "#00FEE0"]


def _plot_impact_imb_curves(
    suptitle: str,
    data_all: pd.DataFrame,
    durations,
    xlim: float = None,
    ylim: float = None,
    imbalance_col="vol_imbalance",
):
    plt.rcParams["figure.dpi"] = 80
    fig = plt.figure(figsize=(14, 15))
    fig.suptitle(suptitle)
    for i in range(len(durations)):
        plt.subplot(3, 3, i + 1)
        df_ = data_all[data_all["T"] == durations[i]]
        _plot_response_vs_imbalance(fig, df_, durations[i], xlim, ylim, imbalance_col=imbalance_col)


def _plot_response_vs_imbalance(
    fig, df_, T, xlim=None, ylim=None, imbalance_col="vol_imbalance", num_q=31, marker="o", alpha=0.5, color=None
):
    ax = fig.gca()

    df_["bin"] = pd.qcut(df_[imbalance_col], num_q, duplicates="drop")
    df_ = df_.groupby(["bin"]).mean().reset_index()
    # plt.scatter(x=df_[imbalance_col], y=df_[f"R"], alpha=alpha, label=f'T={T}', marker=marker, s=15)
    plt.plot(
        df_[imbalance_col],
        df_[f"R"],
        label=f"T={T}",
        marker=marker,
        linewidth=0,
        markerfacecolor="white",
        markersize=5,
        markeredgecolor=color,
        alpha=0.5,
    )
    plt.legend(loc="best")

    xlabel = X_AXIS_LABELS["vol"] if imbalance_col == "vol_imbalance" else X_AXIS_LABELS["sign"]
    ylabel = r"$\it{R(\Delta{V},T)}$" if imbalance_col == "vol_imbalance" else r"$\it{R(\mathcal{E},T)}$"
    ax.set(xlabel=xlabel, ylabel=ylabel)

    plt.locator_params(axis="x", nbins=6)
    if ylim is not None:
        plt.ylim(-ylim, ylim)
    if xlim is not None:
        plt.xlim(-xlim, xlim)


def plot_fitted_model_with_data(
    data_binned: pd.DataFrame,
    durations,
    popt,
    xlim,
    ylim,
    color_fitted_line,
    color_markers_line,
    color_markers,
    title=None,
    alpha=0.7,
    imbalance_col="vol_imbalance",
    y_reflect=False,
    qmax=0.005,
    save=False,
    filename=None,
):
    if popt is None:
        print("Not solved")
        return

    pylab.rcParams["xtick.major.pad"] = "8"
    pylab.rcParams["ytick.major.pad"] = "8"
    rc("text", usetex=True)
    rc("mathtext", fontset="stix")
    rc("axes", labelsize="large")
    plt.rcParams["figure.dpi"] = 150

    durations = data_binned["T"].unique()

    q, _ = get_data_for_plotting(qmax, durations)

    fig = plt.figure(figsize=(5, 4.5))
    if title is not None:
        fig.suptitle(title)
    ax = sns.lineplot(
        x="x_scaled",
        y="y_scaled",
        data=data_binned,
        style="T",
        color=color_markers_line,
        markers=MARKERS_LIST,
        legend="full",
        markerfacecolor="none",
        markevery=0.000005,
        markeredgecolor=color_markers,
        ms=5,
        alpha=alpha,
        linewidth=1,
    )
    ax.grid(False)
    if not y_reflect:
        plt.plot(
            q,
            scaling_function(q, popt[2], popt[3]) * popt[-1],
            "--",
            color=color_fitted_line,
            linewidth=1,
            # label=r"$\mathbb{R}(\Delta V, T)$"
        )
    else:
        plt.plot(
            q,
            scaling_function(-q, popt[2], popt[3]) * popt[-1],
            "--",
            color=color_fitted_line,
            linewidth=1.9,
            # label=r"$\mathbb{R}(\Delta V, T)$"
        )

    xlabel = X_AXIS_LABELS["vol_renorm"] if imbalance_col == "vol_imbalance" else X_AXIS_LABELS["sign_renorm"]
    ylabel = (
        r"$\it{R(\Delta{V},T)/\mathcal{R}(1)T^{\chi_v}}$"
        if imbalance_col == "vol_imbalance"
        else r"$\it{R(\mathcal{E},T)/\mathcal{R}(1)T^{\chi}}$"
    )
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    ax.tick_params(bottom=True, top=True, left=True, right=True, direction="in", width=0.5, size=3)
    ax.minorticks_off()

    plt.xlim(-xlim, xlim)
    plt.ylim(-ylim, ylim)
    plt.locator_params(axis="x", nbins=6)
    plt.locator_params(axis="y", nbins=10)

    import matplotlib.font_manager as font_manager

    # font = font_manager.FontProperties(family='Times', size=14)
    legend = ax.legend(markerfirst=False)
    for indx, T in enumerate(durations):
        legend.get_texts()[indx].set_text(f"$T = {int(T)}$")
        legend.get_lines()[indx].set_markeredgecolor("#0E1111")
        legend.get_lines()[indx].set_markerfacecolor("#0E1111")
        legend.get_lines()[indx].set_color("#0E1111")
        # legend.get_lines()[indx].set_alpha(alpha)
        legend.get_lines()[indx].set_linewidth(1)
        legend.get_lines()[indx].set_ms(5)

    legend.markerscale = 0.3
    legend.get_frame().set_edgecolor("w")

    for spine in ax.spines.values():
        spine.set_edgecolor("#000000")
        spine.set_linewidth(1)

    plt.tight_layout(pad=0.25)

    if save and filename is not None:
        plt.savefig(f"../../plots/model_fitting/{filename}.pgf")

    plt.show()


def plot_renorm_fits(
    data_norm, durations, fit_param, ylim=None, xlim=None, imbalance_col="vol_imbalance", save=False, filename=None
):
    pylab.rcParams["xtick.major.pad"] = "8"
    pylab.rcParams["ytick.major.pad"] = "8"
    rc("text", usetex=False)
    rc("font", family="serif")
    rc("font", style="normal")
    rc("mathtext", fontset="stix")
    rc("axes", labelsize="large")
    plt.rcParams["font.serif"] = ["Verdana"] + plt.rcParams["font.serif"]
    plt.rcParams["figure.dpi"] = 150

    fig = plt.figure(figsize=(4, 4))
    ax = fig.gca()
    for indx, T in enumerate(durations):
        n = 1001
        qmax = 0.05 if xlim is None else xlim * 5
        q = np.linspace(-qmax, qmax, n)

        if T in fit_param.keys():
            popt = fit_param[T][0]

            _plot_response_vs_imbalance(
                fig,
                data_norm[data_norm["T"] == T],
                T=T,
                imbalance_col=imbalance_col,
                num_q=50,
                marker=MARKERS_LIST[indx],
                alpha=1,
                color=COLORS_LIST[indx],
            )

            xlabel = (
                r"$\it{\Delta{V}'}/V_D$"
                if imbalance_col == "vol_imbalance"
                else r"$\it{\Delta{\mathcal{E}}'}/\mathcal{E}_D$"
            )
            ylabel = (
                r"$\it{R(\Delta{V},T)}/\mathcal{R}(1)$"
                if imbalance_col == "vol_imbalance"
                else r"$\it{R(\Delta{\mathcal{E}},T)}/\mathcal{R}(1)$"
            )

            ax.set(xlabel=xlabel, ylabel=ylabel)

            if popt is not None:
                plt.plot(
                    q, fit_param[T][2](np.array([q, T]), *popt), "--", label="", color=COLORS_LIST[indx], alpha=0.5
                )
                if ylim is not None:
                    plt.ylim(-ylim, ylim)
                if xlim is not None:
                    plt.xlim(-xlim, xlim)

    ax.tick_params(bottom=True, top=True, left=True, right=True, direction="in", width=0.5, size=3)

    legend = ax.legend(markerfirst=False)
    for indx, T in enumerate(durations):
        legend.get_texts()[indx].set_text(f"$T = {int(T)}$")
        legend.get_lines()[indx].set_markerfacecolor("#0E1111")
        legend.get_lines()[indx].set_markeredgecolor("#0E1111")

    legend.markerscale = 0.3
    legend.get_frame().set_edgecolor("w")

    for spine in ax.spines.values():
        spine.set_edgecolor("#808080")
        spine.set_linewidth(0.4)

    if save and filename is not None:
        plt.savefig(f"../../plots/model_fitting/{filename}.pdf", bbox_inches="tight")

    plt.show()


def plot_collapsed_fitted_func(
    fit_param,
    line_color,
    markers_color,
    ylim=None,
    xlim=None,
    imbalance_col="vol_imbalance",
    renorm=True,
    durations=None,
    master_curve=None,
    save=False,
    filename=None,
):
    X_AXIS_LABELS = {
        "vol": r"$\it{\Delta V^\prime/V_{D}\langle{V_D}\rangle}$",
        "vol_renorm": r"$\it{\Delta V^\prime/V_{D}T^\varkappa}$",
        "sign": r"$\it{\Delta \mathcal{E}^\prime/\mathcal{E}_{D}\langle{\mathcal{E}_{D}}\rangle}$",
        "sign_renorm": r"$\it{\Delta \mathcal{E}^\prime/\mathcal{E}_{D}T^\varkappa}$",
    }

    MARKETS_LIST = ["P", "o", "s", "^", "D"]
    LINESTYLE_LIST = [(0, (5, 10)), "dashdot", (0, (5, 1)), "dashed", (5, (10, 3))]
    ALPHA_LIST = [1, 0.6, 0.4, 0.8, 0.55]

    pylab.rcParams["xtick.major.pad"] = "8"
    pylab.rcParams["ytick.major.pad"] = "8"
    plt.rcParams["figure.dpi"] = 200

    fig = plt.figure(figsize=(4.2, 4))
    ax = fig.gca()
    ax.grid(False)

    n = 1001
    qmax = 0.05 if xlim is None else xlim * 5
    q = np.linspace(-qmax, qmax, n)

    durations = list(fit_param.keys())
    for indx, T in enumerate(durations):
        if T in fit_param.keys():
            popt = fit_param[T][0]
            if popt is not None:
                plt.plot(
                    q,
                    fit_param[T][2](np.array([q, len(q) * [T]]), *popt),
                    # linestyle=LINESTYLE_LIST[indx],
                    # marker=MARKETS_LIST[indx], markerfacecolor=markers_color,
                    markevery=0.05,
                    ms=3.5,
                    label="x",
                    color=line_color,  # alpha=ALPHA_LIST[indx],
                    linewidth=0.5,
                    markeredgewidth=0.5,
                )

    if master_curve is not None:
        plt.plot(
            q,
            scaling_function(-q, master_curve[2], master_curve[3]) * master_curve[-1],
            "-",
            linewidth=1,
            color="black",
        )

    if renorm:
        xlabel = X_AXIS_LABELS["vol_renorm"] if imbalance_col == "vol_imbalance" else X_AXIS_LABELS["sign_renorm"]
        ylabel = (
            r"$\it{R(\Delta V^\prime,T)/\mathcal{R}(1)T^{\chi_v}}$"
            if imbalance_col == "vol_imbalance"
            else r"$\it{R(\Delta \mathcal{E}^\prime,T)/\mathcal{R}(1)T^\chi}$"
        )
    else:
        xlabel = X_AXIS_LABELS["vol"] if imbalance_col == "vol_imbalance" else X_AXIS_LABELS["sign"]
        ylabel = (
            r"$\it{R(\Delta V^\prime,T)}$"
            if imbalance_col == "vol_imbalance"
            else r"$\it{R(\Delta \mathcal{E}^\prime,T)}$"
        )
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    ax.tick_params(bottom=True, top=True, left=True, right=True, direction="in", width=0.5, size=3)
    ax.minorticks_off()

    if ylim is not None:
        plt.ylim(-ylim, ylim)
    if xlim is not None:
        plt.xlim(-xlim, xlim)

    font = font_manager.FontProperties(size=14)
    # legend = ax.legend(markerfirst=True, prop=font, labelspacing=0.05)

    # for indx, T in enumerate(durations):
    #     legend.get_texts()[indx].set_text(f"$T = {int(T)}$")
    #     legend.get_lines()[indx].set_markeredgecolor('#1A1919')
    #     legend.get_lines()[indx].set_markerfacecolor('white')
    #     legend.get_lines()[indx].set_color('#0E1111')
    #     legend.get_lines()[indx].set_alpha(alpha=0.9)
    #     legend.get_lines()[indx].set_linewidth(0.5)
    #     legend.get_lines()[indx].set_ms(4)

    # legend.markerscale = .3
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
