import pandas as pd
from typing import List

from liquidity.finite_scaling.fss import transform
from liquidity.finite_scaling.fit import fit_scaling_law


# FIXME: rename file to liquidity
# TODO: add price change and nu (tick-size) parameter
# TODO: add gradient based optimizer via NN estimator
# TODO: flag for price imapct and market impact to reflect scaling function along x-axis

def price_response(conditional_aggregate_impact: pd.DataFrame,
                   market_impact: bool = False,
                   estimator: str = "LSE",
                   response_column: str = "R_cond",
                   imbalance_column: str = "sign_imbalance"):
    """
    Computes critical exponents of scaling laws that govern the
    empirical scaling of  conditional aggregate price returns.

    Parameters
    ----------
    conditional_aggregate_impact : pd.DataFrame
        DataFrame containing conditional aggregate impact data.
    market_impact: bool
        Flag to indicate whether we are computing response for price impact of an MO or market impact of LO, CA, QA
    response_column : str, optional
        Column name for aggregate impact `R` data. Default is  conditional agggregate impact "R_cond".
    imbalance_column : str, optional
        Column name for the order flow imbalance `x_imbalance` data. Default is "sign_imbalance".

    Returns
    -------
    np.ndarray
        The optimized scaling exponents for the width `x_imbalance`, and heigt `R` of aggregate impact;
        chi `𝛘`, kappa `ϰ`, alpha `α`, beta `β` and `CONST`.
    DataFrame
        A DataFrame containing rescaled aggregate impact data.

    Notes
    -----
    Assumes the conditional aggregate impact pd.DataFrame[["T", "x_imbalance", "R"]] has been appropriately normalized.
    chi `𝛘` and kappa `ϰ` are critical exponents that govern the  scaling of the height `R` and width `x` of condition
    aggregate impact as the bin size T of the master curve is increased.
    """
    original_data = conditional_aggregate_impact.copy()

    # Extract data for susceptibility
    t_values = original_data['T'].values
    r_values = original_data[response_column].values
    imbalance_values = original_data[imbalance_column].values

    # Compute critical exponents
    rescaling_params = fit_scaling_law(t_values=t_values, imbalance_values=imbalance_values, r_values=r_values)

    # Transform original data to new (X, Y) coordinates
    # using critical/rescaling exponents
    rescaled_data = transform(original_data, rescaling_params=rescaling_params, response_column=response_column, imbalance_column=imbalance_column)

    return rescaled_data, rescaling_params


def liquidity(rescaling_params: List[float], operator: str = "rate_of_change"):
    """
    Computes (il-)liquidity parameters lambda `λ` using rescaling exponents chi `𝛘` and kappa `ϰ`.

    Whilst the height `R` and width 'x' of the aggregate impact curve scale independently with increments governed by
    chi `𝛘` and kappa `ϰ`, the slope of the linear region of the master curve `R(x, T)` decreases as power-law with T

    .. math::

        R(x, T) ~ T^-λ

    where the scaling by and large consistent with the Hurst-exponents H of the return and order-sign time series.
    This slope is known as Kyle's lambda. Its value is often considered a measure of a market’s (il-)liquidity.

    Returns
    –––––––––––
        ... rate_of_change `λ = ϰ - 𝛘` or magnitude_of_change `λ = ϰ + 𝛘` of conditional aggregate impact.
    """

    # Extract critical exponents
    chi, kappa, _, _, _ = rescaling_params

    # Compute liquidity parameter
    if operator == "rate_of_change":
        lambda_ = kappa - chi
    elif operator == "magnitude":
        lambda_ = kappa + chi
    else:
        raise ValueError('Invalid operator. Options are "change_rate" or "magnitude".')

    return  lambda_


if __name__ == "__main__":

    # Set backend
    import os
    from liquidity.impact import aggregate_impact
    from liquidity.util.plot_utils import plot_collapsed_scaling_function

    # Constants
    BINNING_FREQUENCIES = [10, 20, 50, 100]

    # Load sample data – AAPL trades data.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "..", "data/market_orders", "AAPL-2017-NEW.csv")
    sample_data = pd.read_csv(csv_path, header=0, index_col=0)

    # Compute conditional aggregate impact
    imbalance_column = "sign_imbalance"
    conditional_aggregate_impact = aggregate_impact(sample_data, bin_frequencies=BINNING_FREQUENCIES, imbalance_column=imbalance_column,)

    # Compute price response
    aapl_rescaled_data, aapl_rescaling_params = price_response(
        conditional_aggregate_impact, imbalance_column=imbalance_column)

    chi, kappa, alpha, beta, CONST = aapl_rescaling_params
    print(f'chi: {chi}')
    print(f'kappa: {kappa}')
    print(f'alpha: {alpha}')
    print(f'beta: {beta}')

    # Liquidity parameter
    aapl_liquidity = liquidity(rescaling_params=aapl_rescaling_params, operator="magnitude")
    print(f'lambda: {aapl_liquidity}')

    # plot master curve
    # plot_collapsed_scaling_function(
    #     aapl_rescaled_data,
    #     scaling_params=aapl_rescaling_params,
    #     # line_color=EBAY_COLORS.dark_color,
    #     markers_color="white",
    #     imbalance_column=imbalance_column,
    #     master_curve="Sigmoid",
    #     binning_frequencies=BINNING_FREQUENCIES)


