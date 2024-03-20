import pandas as pd
from typing import List

from liquidity.features import compute_aggregate_features
from liquidity.util.utils import normalize_impact, normalize_imbalances, remove_first_daily_prices


def aggregate_impact(
    orderbook_states: pd.DataFrame,
    bin_frequencies: List[int],
    conditional: bool = True,
    imbalance_column: str = "sign_imbalance",
    normalization_constant: str = "daily",
    **kwargs
    ) -> pd.DataFrame:
    """
     Compute the conditional `R(x, T)` or unconditional aggregate impact `R(T)` of an order,
     where the generalized impact is taken as the average price change for any lag T > 0.

    Notes
    -----
        The unconditional aggregate impact `R(T)` generalises the definition of the lag-1 price impact case `R(1)` to any T > 0.
        The conditional aggregate impact `R(x, T)` is is the price change over the interval `ℓ ∈ [t,t + T)`, conditioned
        to a certain order flow imbalance x.
    """
    if conditional:
        aggregate_features = compute_aggregate_features(orderbook_states=orderbook_states, bin_frequencies=bin_frequencies, **kwargs)

        # Normalize variables describing our system
        aggregate_features = normalize_impact(
            aggregate_features,
            impact_column="R_cond",
            normalization_constant=normalization_constant
        )
        aggregate_features = normalize_imbalances(
            aggregate_features,
            imbalance_column=imbalance_column,
            normalization_constant=normalization_constant
        )

        # Conditional aggregate impact
        #cond_aggregate_impact = aggregate_features[["event_timestamp", "T", imbalance_column, response_column]]

        return aggregate_features

    else:
        aggregate_features = compute_aggregate_features(orderbook_states=orderbook_states, bin_frequencies=bin_frequencies, **kwargs)

        aggregate_features = normalize_impact(
            aggregate_features,
            impact_column="R_uncond",
            normalization_constant=normalization_constant
        )
        aggregate_features = normalize_imbalances(
            aggregate_features,
            imbalance_column=imbalance_column,
            normalization_constant=normalization_constant
        )

        # Unconditional aggregate impact
        #uncond_aggregate_impact = aggregate_features[["event_timestamp", "T", imbalance_column, response_column]]

        return aggregate_features


def compute_impact_from_returns(
        orderbook_states: pd.DataFrame,
        returns: pd.DataFrame,
        bin_frequencies: List[int],
        normalize: bool = False,
        conditional: bool = True,
        imbalance_column: str = "sign_imbalance",
        normalization_constant: str = "daily",
):
    """
    Computes aggregate impact series for different bining frequenceis from a set of return series.
    """
    orderbook_states = remove_first_daily_prices(orderbook_states)
    assert all(returns["event_timestamp"] == orderbook_states["event_timestamp"]), "Data has different timestamp index."

    if conditional:
        # Replace (raw) lag-1 aggregate impact R, with returns computed from lag-1 prices
        masked_returns = orderbook_states.copy()
        masked_returns["R1_cond"] = returns["R1_cond"].values

        # Compute aggregate features for specified binning_frequences
        aggregate_features = compute_aggregate_features(masked_returns, bin_frequencies, remove_first=False).reset_index()

        # Normalize variables describing our system
        aggregate_features = normalize_imbalances(
            aggregate_features,
            imbalance_column=imbalance_column,
            normalization_constant=normalization_constant
        )
        if normalize:
            # Rescale returns if not already normalized
             aggregate_features = normalize_impact(
                aggregate_features,
                impact_column="R_cond",
                normalization_constant=normalization_constant
            )
    else:
        masked_returns = orderbook_states.copy()
        masked_returns["R1_uncond"] = returns["R1_uncond"].values

        aggregate_features = compute_aggregate_features(masked_returns, bin_frequencies, remove_first=False).reset_index()

        aggregate_features = normalize_imbalances(
            aggregate_features,
            imbalance_column=imbalance_column,
            normalization_constant=normalization_constant
        )
        if normalize:
            # Rescale returns if not already normalized
             aggregate_features = normalize_impact(
                aggregate_features,
                impact_column="R_uncond",
                normalization_constant=normalization_constant
            )

    return aggregate_features


def cross_impact():
    pass



