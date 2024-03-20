import pandas as pd
from typing import List, Optional, Callable, Union, Tuple


def _validate_responses(impact_column: str) -> None:
    """
    Validates whether given response variable is supported.

    Parameters
    ----------
    impact_column : str
        The column name representing the order flow imbalance.

    Raises
    ------
    ValueError
        If the imbalance column is not supported.
    """
    valid_response_columns = ["R1_cond", "R_cond", "R1_uncond", "R_uncond"]
    if impact_column not in valid_response_columns:
        raise ValueError(
            f"Unknown response column: {impact_column}. Expected one of {valid_response_columns}."
        )


def _validate_imbalances(imbalance_column: str) -> None:
    """
    Validates whether given imbalance variable is supported.

    Parameters
    ----------
    imbalance_column : str
        The column name representing the order flow imbalance.

    Raises
    ------
    ValueError
        If the imbalance column is not supported.
    """
    valid_imbalance_columns = ["sign_imbalance", "volume_imbalance"]
    if imbalance_column not in valid_imbalance_columns:
        raise ValueError(
            f"Unknown imbalance column: {imbalance_column}. Expected one of {valid_imbalance_columns}."
        )


def remove_midprice_orders(df_: pd.DataFrame) -> pd.DataFrame:
    """
    Removes events that occured at the mid-price.
    """
    mask = df_["price"] == df_["midprice"]
    return df_[~mask]


def remove_first_daily_prices(orderbook_states: pd.DataFrame) -> pd.DataFrame:
    """
    If the price deviated significantly during auction hours
    the first returns on the day are to be considered outliers.
    """
    data_ = orderbook_states.copy()

    if type(data_["event_timestamp"].iloc[0]) != pd.Timestamp:
        data_["event_timestamp"] = data_["event_timestamp"].apply(lambda x: pd.Timestamp(x))
        data_["date"] = data_["event_timestamp"].apply(lambda x: x.date())

    data_["indx"] = data_.index
    data_ = data_.set_index("event_timestamp")
    first_days_indx = data_.groupby(pd.Grouper(freq="D")).first()["indx"]
    first_days_indx = first_days_indx.dropna().astype(int)
    data_ = data_.loc[~data_["indx"].isin(first_days_indx)]
    return data_.drop(columns=["indx"]).reset_index()


def normalise_size(
    orderbook_states: pd.DataFrame, column_name: str = "size"
) -> pd.DataFrame:
    """
    Normalise order size :math:`v_x` by the average volume on the same side best quote.
    .. math::
        v = v/\overline{V}_{\mathrm{best}}
    """
    data_ = orderbook_states.copy()

    # Compute avarage over average
    ask_mean_volume = data_[
        "ask_queue_size_mean"
    ].mean()  # ask_mean_vol = df_["ask_volume"].mean()
    bid_mean_volume = data_[
        "bid_queue_size_mean"
    ].mean()  # bid_mean_vol = df_["bid_volume"].mean()

    def _normalise(row):
        if row["side"] == "ASK":
            return row[column_name] / ask_mean_volume
        else:
            return row[column_name] / bid_mean_volume

    data_["norm_size"] = data_.apply(_normalise, axis=1)

    return data_


def normalize_impact(
    aggregate_impact: pd.DataFrame,
    impact_column: str = "R_cond",
    normalization_constant: str = "daily",
) -> pd.DataFrame:
    """
    Normalize the aggregate impact 'R' by its corresponding daily value `daily_R1`.

    Parameters
    ----------
    aggregate_impact : pd.DataFrame
        DataFrame containing the aggregate impact data.
    impact_column : str, optional
        Column name for the value used for normalization. Default is "R".

    Returns
    -------
    pd.DataFrame
        DataFrame with the normalized 'R' values.

    Note
    ----
    We use 'abs()' to preserve the sign of `daily_R1` when the average order flow signs are negative.

    TODO: complete average for impact normalization
    """
    data = aggregate_impact.copy()
    _validate_responses(impact_column=impact_column)

    if normalization_constant == "daily":
        data[impact_column] = data[impact_column] / abs(data["daily_R1"])

    return data


def normalize_imbalances(
    aggregate_features_data: pd.DataFrame,
    imbalance_column: str = "volume_imbalance",
    normalization_constant: str = "daily",
) -> pd.DataFrame:
    """
    Normalize imbalances using either daily or average queue values at the best quotes.

    Parameters
    ----------
    aggregate_features_data : pd.DataFrame
        DataFrame containing aggregate features data.
    normalization_constant : str, optional
        The normalization mode - either 'daily' or 'average'. Defaults to "daily".
    imbalance_column : str, optional
        The dependent variable to normalize - 'volume_imbalance' or 'sign_imbalance'. Defaults to "volume_imbalance".

    Returns
    -------
    pd.DataFrame
        DataFrame with the normalized imbalance values.

    Raises
    ------
    ValueError
        If an unknown normalization constant is provided.
    """
    data = aggregate_features_data.copy()
    _validate_imbalances(imbalance_column=imbalance_column)

    if normalization_constant == "daily":
        # Normalize imbalance using daily values
        if (
            imbalance_column in data.columns
            and "daily_num" in data.columns
            and "daily_vol" in data.columns
        ):
            norm_column = (
                "daily_num" if imbalance_column == "sign_imbalance" else "daily_vol"
            )
            data[imbalance_column] = data[imbalance_column] / data[norm_column]
        else:
            raise KeyError(
                f'{"One or more required columns are missing for `daily` normalization."}'
            )
    elif normalization_constant == "average":
        # Normalize imbalance using average queue values
        if (
            imbalance_column in data.columns
            and "average_num_at_best" in data.columns
            and "average_vol_at_best" in data.columns
        ):
            norm_column = (
                "average_num_at_best"
                if imbalance_column == "sign_imbalance"
                else "average_vol_at_best"
            )
            data[imbalance_column] = data[imbalance_column] / data[norm_column]
        else:
            raise KeyError(
                f'{"One or more required columns are missing for `average` normalization."}'
            )
    else:
        raise ValueError(
            f"Unknown normalization constant: {normalization_constant}. Expected 'daily' or 'average'."
        )

    return data


def bin_data_into_quantiles(
    df: pd.DataFrame,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    q: int = 100,
    duplicates: str = "raise",
) -> pd.DataFrame:
    """
    Dynamically bin a DataFrame into quantiles based on a specified column.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be binned.
    x_col : Optional[str], optional
        The column in 'df' on which to base the quantile bins. Defaults to None.
    y_col : Optional[str], optional
        The column in 'df' for which the mean is calculated in each bin. Defaults to None.
    q : int, optional
        The number of quantiles to bin into. Defaults to 100.
    duplicates : str, optional
        Handling of duplicate edges (can be 'raise' or 'drop'). Defaults to "raise".

    Returns
    -------
    pd.DataFrame
        DataFrame containing the binned data.

    Notes
    -----
    This function will raise an error if 'duplicates' is set to 'raise' and duplicate bin edges are found.
    """
    # Bin 'x_col' into quantiles
    binned_x = pd.qcut(
        df[x_col], q=q, labels=False, retbins=True, duplicates=duplicates
    )
    df["x_bin"] = binned_x[0]

    # Calculate mean of 'y_col' for each bin
    y_binned = df.groupby(["x_bin"])[y_col].mean()
    y_binned.index = y_binned.index.astype(int)

    # Calculate mean of 'x_col' for each bin
    x_binned = df.groupby(["x_bin"])[x_col].mean()
    x_binned.index = x_binned.index.astype(int)

    # If 'T' column exists, include the first value of 'T' for each bin
    if "T" in df.columns:
        r_binned = df.groupby(["x_bin"])["T"].first()
        r_binned.index = r_binned.index.astype(int)
    else:
        r_binned = None

    # Concatenate the binned data into a single DataFrame
    return pd.concat([x_binned, r_binned, y_binned], axis=1).reset_index(drop=True)


def smooth_outliers(
    data: pd.DataFrame,
    columns: Optional[List[str]],
    T: Optional[int] = None,
    std_level: int = 2,
    remove: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Clip or remove values at standard deviations for each series in a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to process.
    columns : Optional[List[str]], optional
        List of column names to process. Defaults to None.
    T : Optional[int], optional
        An additional column indicator to append to each column in 'columns'. Defaults to None.
    std_level : int, optional
        The number of standard deviations to use as the clipping or removal threshold. Defaults to 2.
    remove : bool, optional
        If True, rows with outliers will be removed. If False, values will be clipped. Defaults to False.
    verbose : bool, optional
        If True, prints additional information. Defaults to False.

    Returns
    -------
    pd.DataFrame
        DataFrame with outliers smoothed.
    # TODO: default columns to None
    """
    if T:
        all_columns = columns + [f"R{T}"]
    else:
        all_columns = columns

    all_columns = set(all_columns).intersection(data.columns)
    if len(all_columns) == 0:
        return data

    if remove:
        # Remove rows with outliers
        z = np.abs(stats.zscore(data[columns]))
        original_shape = data.shape
        data = data[(z < std_level).all(axis=1)]
        new_shape = data.shape
        if verbose:
            print(f"Removed {original_shape[0] - new_shape[0]} rows")
    else:
        # Clip values in each column
        def winsorize_queue(s: pd.Series, level) -> pd.Series:
            upper_bound = level * s.std()
            lower_bound = -level * s.std()
            if verbose:
                print(f"clipped at {upper_bound}")
            return s.clip(upper=upper_bound, lower=lower_bound)

        for name in all_columns:
            s = data[name]
            if verbose:
                print(f"Series {name}")
            data[name] = winsorize_queue(s, level=std_level)

    return data