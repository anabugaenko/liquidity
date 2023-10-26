# order_type
# price_changing
# T
# for x, what is y, (e.g., for volume, what is impact or for imbalance, what is expected impact
#
import os
import pandas as pd
from matplotlib import pyplot as plt
from liquidity.response_functions.functional_form import rescaled_form
from liquidity.util.utils import bin_data_into_quantiles, smooth_outliers
from liquidity.response_functions.fit import find_shape_parameters, compute_RN_QN
from liquidity.response_functions.features import compute_aggregate_features
from liquidity.response_functions.price_response_functions import compute_conditional_aggregate_impact

# def Rl():
#     pass


def aggregate_impact(ordeerbook_states_df: pd.DataFrame, bin_size):

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

    RN_series, QN_series, RN_fit_object, QN_fit_object = compute_RN_QN(conditional_aggregate_impact, ALPHA, BETA)


    return RN_series, QN_series, RN_fit_object, QN_fit_object, conditional_aggregate_impact, ALPHA, BETA

if __name__ == '__main__':

    # Load data
    current_dir = os.path.abspath('')
    root_dir = os.path.join(current_dir, '', '..')
    data_dir = os.path.join(root_dir, 'data', 'market_orders')
    filename = "TSLA-2017-NEW.csv"
    stock_file_path = os.path.join(data_dir, filename)
    tsla_raw_df = pd.read_csv(stock_file_path)

    BIN_SIZE = list(range(1, 1001))
    OBSERVATION_WINDOWS = [10, 20, 50, 100, 150]

    RN_df, QN_df, RN_fit_object, QN_fit_object, conditional_aggregate_impact, ALPHA, BETA = aggregate_impact(tsla_raw_df, BIN_SIZE)


    OBSERVATION_WINDOWS = [x for x in OBSERVATION_WINDOWS if x in RN_df['x_values'].values]

    rn = RN_df[RN_df['x_values'].isin(OBSERVATION_WINDOWS)]['y_values']
    qn = QN_df[QN_df['x_values'].isin(OBSERVATION_WINDOWS)]['y_values']


    # Found rescaling exponents
    chi = RN_fit_object.powerlaw.params.alpha
    kappa = QN_fit_object.powerlaw.params.alpha

    # Series of scale factors RN and QN
    plt.scatter(RN_df['x_values'],RN_df['y_values'])
    plt.scatter(QN_df['x_values'],QN_df['y_values'])
    plt.loglog()

    # Powerlaw fits
    RN_fit_object.powerlaw.plot_fit()
    QN_fit_object.powerlaw.plot_fit()



    # Aggregate impact functions
    for lag in OBSERVATION_WINDOWS:
        data = conditional_aggregate_impact[conditional_aggregate_impact["T"] == lag][["vol_imbalance", "T", "R"]]
        data_binned = bin_data_into_quantiles(data, q=50)

        imbalance = data_binned["vol_imbalance"].values
        indx = OBSERVATION_WINDOWS.index(lag)
        response = rescaled_form(imbalance, rn.iloc[indx], qn.iloc[indx], ALPHA, BETA)
        plt.scatter(imbalance, response)
        plt.plot(imbalance, response)


    # Final rescaled functions
    OBSERVATION_WINDOWS = [10, 20, 50, 100, 150]
    OBSERVATION_WINDOWS = [x for x in OBSERVATION_WINDOWS if x in RN_df['x_values'].values]

    rn = RN_df[RN_df['x_values'].isin(OBSERVATION_WINDOWS)]['y_values']
    qn = QN_df[QN_df['x_values'].isin(OBSERVATION_WINDOWS)]['y_values']

    for T in OBSERVATION_WINDOWS:
        data = conditional_aggregate_impact[conditional_aggregate_impact['T']==T][["vol_imbalance", "T", "R"]]

        data["vol_imbalance"] = data["vol_imbalance"] / T**kappa
        data["R"] = data["R"] / T**chi

        data_binned = bin_data_into_quantiles(data, q=50)
        imbalance = data_binned["vol_imbalance"].values
        indx = OBSERVATION_WINDOWS.index(T)
        response = rescaled_form(imbalance, chi, kappa, ALPHA, BETA)
        plt.scatter(imbalance, response)
        plt.plot(imbalance, response)


