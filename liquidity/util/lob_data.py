import pandas as pd

UCT_OFFSET = 2


def load_l3_data(filepath: str) -> pd.DataFrame:
    """
    Returns DataFrame of daily incremental Level 3 order book data.
    Example code below in a comment show how to load the data on BMLL platform.
    """

    """
    This is how the data was retrieved in BMLL DataLab
    
    lid - str representing unique listing identifier on BMLL platform
    date - str 
    
    nsec = NormalisedSecurity.from_listing_id(lid, date)
    mkt_dat = nsec.market_data(feed='L3')
    return mkt_dat.incremental_book_L3()
    """

    return pd.read_csv(filepath, header=0, index_col=0)


def select_trading_hours(date: str, df_: pd.DataFrame, utc: bool = False) -> pd.DataFrame:
    """
    Select only non-auction trading hours: 10:30 am and 3:00 pm.
    """
    if utc:
        utc_offset = UCT_OFFSET
        dt = pd.Timedelta(utc_offset, unit='m')
    else:
        dt = pd.Timedelta(0, unit='m')
    ts1 = pd.Timestamp(date + ' 10:30') - dt
    ts2 = pd.Timestamp(date + ' 15:00') - dt
    df_['event_timestamp'] = df_['event_timestamp'].apply(lambda x: pd.Timestamp(x))
    mask = (df_['event_timestamp'].between(ts1, ts2))
    df_ = df_[mask]
    df_['event_timestamp'] = df_['event_timestamp'] + dt
    return df_


def select_top_book(df_: pd.DataFrame) -> pd.DataFrame:
    """
    Select top level of order book events (level 1)
    or orders that arrive at a better price (level 0).
    Level 0:
    - for arriving LO: new best price
    - for MO: level no longer exists after trade
    - for CA: level no longer exists after removal
    """
    mask = (df_.price_level == 1) | (df_.price_level == 0)
    return df_[mask]


def select_columns(df_: pd.DataFrame) -> pd.DataFrame:
    """
    Select only relevant info about each event.
    """
    sub_columns_list = [
        'event_timestamp', 'side', 'lob_action',
        'order_executed', 'execution_price', 'execution_size',
        'best_ask_price', 'best_ask_size',
        'best_bid_price', 'best_bid_size',
        'is_new_best_price', 'price', 'price_level',
        'old_price', 'old_price_level',
        'size', 'old_size',
        'best_ask_num_orders', 'best_bid_num_orders']

    df_ = df_[sub_columns_list]
    df_ = df_.rename(columns={'best_ask_price': 'ask', 'best_bid_price': 'bid',
                              'best_ask_size': 'ask_volume', 'best_bid_size': 'bid_volume',
                              'best_ask_num_orders': 'ask_count', 'best_bid_num_orders': 'bid_count',
                              'is_new_best_price': 'price_changing'})

    df_['midprice'] = (df_['ask'] + df_['bid']) * 0.5
    return df_


def shift_prices(df_: pd.DataFrame) -> pd.DataFrame:
    """
    This transformation is specific to how BMLL offer orderbook view
    where each event is accompanied by respective price value after the event has taken place.
    We're interested in how the price changed so shifting it to get values of
    mid-price, bid and ask immediately before each event.
    """
    df_['midprice'] = df_['midprice'].shift().fillna(0)
    df_['ask'] = df_['ask'].shift().fillna(0)
    df_['bid'] = df_['bid'].shift().fillna(0)
    df_['ask_volume'] = df_['ask_volume'].shift().fillna(0)
    df_['bid_volume'] = df_['bid_volume'].shift().fillna(0)
    return df_
