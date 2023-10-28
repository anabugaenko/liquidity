import os
import pandas as pd
from matplotlib import pyplot as plt
from liquidity.response_functions.functional_form import rescaled_form
from liquidity.util.utils import bin_data_into_quantiles, smooth_outliers
from liquidity.response_functions.fit import find_shape_parameters, compute_scale_factors
from liquidity.response_functions.features import compute_aggregate_features
from liquidity.response_functions.price_response_functions import compute_conditional_aggregate_impact

def Rl():
    pass


def conditional_impact(ordeerbook_states_df: pd.DataFrame, bin_size):

    """
    In order to determine the rescaling exponents ξ and ψ:

    - The shape of R(ΔV,T) is fitted for all T using the scaling form R(ΔV,T) = T^chi * F(ΔV/ T^kappa) * gamma
    with scaling function F(x) given by F(x) = x / (1 + abs(x)^alpha)^(beta/alpha).

    - Once shape parameters 𝛼 and β are fixed, we can map out scale factors QN and RN, by fitting the
    rescaled_form RN(ΔV) ≈ RN * F(ΔV / QN) for each T keeping constant the value of 𝛼 and β for all T.
    """
    df_ = ordeerbook_states_df.copy()
    aggregate_features = compute_aggregate_features(df_, durations=bin_size)
    conditional_aggregate_impact = compute_conditional_aggregate_impact(aggregate_features, normalise=True)

    # 1. Fix shape parameters 𝛼 and β by fitting scaling_form on entire aggregate features series

    # Remove outliers
    smoothed_aggregate_impact = smooth_outliers(conditional_aggregate_impact)
    popt, pcov, fit_func = find_shape_parameters(smoothed_aggregate_impact)
    _, _, alpha, beta, _ = popt

    # Once 𝛼 and β are fixed, use them to series to map out QN and RN as a functon of T.
    ALPHA = alpha
    BETA = beta

    RN_series, QN_series, RN_fit_object, QN_fit_object, fit_results_per_lag = compute_scale_factors(conditional_aggregate_impact, ALPHA, BETA)


    return RN_series, QN_series, RN_fit_object, QN_fit_object, conditional_aggregate_impact, ALPHA, BETA
