from matplotlib import pyplot as plt, pylab
import matplotlib.font_manager as font_manager
import numpy as np

from liquidity.util.plotting.constants import TSLA_COLORS, EBAY_COLORS, MSFT_COLORS, AMZN_COLORS, NFLX_COLORS

plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = [r"\usepackage{lmodern}"]


def plot_returns_cdf(tsla, ebay, msft, amzn, nflx, exponent=None):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca()
    ax.grid(False)

    plt.plot(
        tsla.bin,
        tsla.value,
        marker="^",
        markevery=2,
        label="TSLA",
        color=TSLA_COLORS.dark_line_color,
        markerfacecolor="white",
        linewidth=0.5,
    )
    plt.plot(
        ebay.bin,
        ebay.value,
        marker="*",
        markevery=2,
        label="EBAY",
        color=EBAY_COLORS.dark_line_color,
        markerfacecolor="white",
        linewidth=0.5,
    )
    plt.plot(
        msft.bin,
        msft.value,
        marker="*",
        markevery=2,
        label="MSFT",
        color=MSFT_COLORS.dark_line_color,
        markerfacecolor="white",
        linewidth=0.5,
    )
    plt.plot(
        amzn.bin,
        amzn.value,
        marker="s",
        markevery=2,
        label="AMZN",
        color=AMZN_COLORS.dark_line_color,
        markerfacecolor="white",
        linewidth=0.5,
    )
    plt.plot(
        nflx.bin,
        nflx.value,
        marker="o",
        markevery=2,
        label="NFLX",
        color=NFLX_COLORS.dark_line_color,
        markerfacecolor="white",
        linewidth=0.5,
    )

    if exponent is not None:
        plt.plot(
            np.linspace(1, 10000, 100),
            (np.linspace(1, 10000, 100) ** exponent) * 10,
            linewidth=0.5,
            color="grey",
            linestyle=(0, (5, 10)),
            label=f"$y \sim x^{exponent}$",
        )

    plt.legend(loc="best")

    plt.xlabel(r"Return %")
    plt.ylabel(r"Probability\ density")

    plt.yscale("log")
    plt.xscale("log")

    plt.ylim(0.005, 2)

    ax.tick_params(bottom=True, top=True, left=True, right=True, direction="in", width=0.5, size=3)
    ax.minorticks_on()


def plot_returns_cdf_order_types(
    lo_returns,
    mo_returns,
    ca_returns,
    qa_returns,
    liquidity_taking=True,
    liquidity_provision=False,
    xlim=None,
    ylim=None,
    exponent=None,
    exp_shift=100,
    exp_label=None,
    mark_every=0.02,
    save=False,
    filename=None,
):
    pylab.rcParams["xtick.major.pad"] = "8"
    pylab.rcParams["ytick.major.pad"] = "8"
    plt.rcParams["figure.dpi"] = 200

    fig = plt.figure(figsize=(4.2, 4))
    ax = fig.gca()
    ax.grid(False)

    ms = 3.5
    mew = 0.5
    alpha = 0.7

    if liquidity_taking:
        plt.plot(
            mo_returns.bin,
            mo_returns.value,
            marker="D",
            markevery=mark_every,
            label="$\mathrm{MO}$",
            color="black",
            markerfacecolor="white",
            linewidth=0.5,
            ms=ms,
            mew=mew,
            alpha=alpha,
        )

        plt.plot(
            ca_returns.bin,
            ca_returns.value,
            marker="o",
            markevery=mark_every,
            label="$\mathrm{CA}$",
            color="#BABBBA",
            markerfacecolor="white",
            linewidth=0.5,
            ms=ms,
            mew=mew,
            alpha=0.8,
        )

    if liquidity_provision:
        plt.plot(
            lo_returns.bin,
            lo_returns.value,
            marker="^",
            markevery=mark_every,
            label="$\mathrm{LO}$",
            color="#757575",
            markerfacecolor="white",
            linewidth=0.5,
            ms=ms,
            mew=mew,
            alpha=0.8,
        )

        plt.plot(
            qa_returns.bin,
            qa_returns.value,
            marker="s",
            markevery=mark_every,
            label="$\mathrm{QA}$",
            color="black",
            markerfacecolor="white",
            linewidth=0.5,
            ms=ms,
            mew=mew,
            alpha=alpha,
        )

    plt.legend(loc="best")

    plt.xlabel(r"$\mathrm{Return (\%)}$", fontsize=12)
    plt.ylabel(r"$\mathrm{Probability\ density}$", fontsize=12)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.yscale("log")
    plt.xscale("log")

    ax.minorticks_on()
    ax.tick_params(bottom=True, top=True, left=True, right=True, direction="in", which="minor", size=3, labelsize=14)
    ax.tick_params(bottom=True, top=True, left=True, right=True, direction="in", width=0.5, size=3, labelsize=14)

    if exponent is not None:
        plt.plot(
            np.linspace(0.1, 10000, 100),
            (np.linspace(0.1, 10000, 100) ** exponent) * exp_shift,
            linewidth=0.5,
            color="grey",
            linestyle=(0, (5, 10)),
            label=exp_label,
        )

    font = font_manager.FontProperties(size=12)
    legend = ax.legend(markerfirst=True, prop=font, labelspacing=0.05)
    legend.markerscale = 0.3
    legend.get_frame().set_edgecolor("w")

    for spine in ax.spines.values():
        spine.set_edgecolor("#1A1919")
        spine.set_linewidth(0.5)

    plt.tight_layout(pad=0.25)

    if save and filename is not None:
        plt.savefig(f"../../plots/ret_dist/{filename}.pdf")

    plt.show()


def plot_ccdf_with_fits(fit, lognormal=True, powerlaw_truncated=False, exponential=False):
    fig = fit.plot_ccdf(linewidth=3, label="Empirical data")
    fit.power_law.plot_ccdf(ax=fig, color="r", linestyle="--", label="Power law fit")
    if lognormal:
        fit.lognormal.plot_ccdf(ax=fig, color="g", linestyle="--", label="Lognormal fit")
    if exponential:
        fit.exponential.plot_ccdf(ax=fig, color="o", linestyle="--", label="Exponential fit")
    if powerlaw_truncated:
        fit.truncated_power_law.plot_ccdf(ax=fig, color="b", linestyle="--", label="Powerlaw truncated fit")
    plt.legend(loc="best")
