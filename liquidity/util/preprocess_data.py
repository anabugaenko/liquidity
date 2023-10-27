import os
import pandas as pd

from liquidity.util.trades_data_util import get_trades_data
from liquidity.util.limit_orders_data_util import get_lo_data, get_qa_data, get_ca_data
from liquidity.response_functions.features import compute_orderbook_states


# FIXME: make path google collab directory specific
YEAR = 2017
CURRENT_DIR = os.path.abspath('.')
ROOT_DIR = os.path.join(CURRENT_DIR, '..', '..')
DATA_DIR_PATH = '/Users/ana_bugaenko/data/BMLL/NASDAQ'


def get_daily_trades_data(file_name, date: str = "2016-09-30"):
    """
    Get market orders (MO) or trades data
    """
    raw_trades = get_trades_data(file_name, date)
    lob_states = compute_orderbook_states(raw_trades)
    return lob_states


def get_daily_orderbook_data(file_name, date: str = "2016-09-30", order_type: str="LO"):
    """
    Get orderbook data, limit orders (LO), queues of active orders (QA) and cancellations (CA)
    """

    orderbook_types = ['LO', 'QA', 'CA']

    if order_type in orderbook_types:
        if order_type == 'LO':
            raw_orders = get_lo_data(file_name, date)
            lob_states = compute_orderbook_states(raw_orders)
        elif order_type == 'CA':
            raw_orders = get_ca_data(file_name, date)
            lob_states = compute_orderbook_states(raw_orders)
        elif order_type == 'QA':
            raw_orders = get_qa_data(file_name, date)
            lob_states = compute_orderbook_states(raw_orders)
        else:
            raise ValueError(f"Unknown order type: {order_type}. Expected one of {orderbook_types}.")

    return lob_states


if __name__ == '__main__':

    all_stocks = os.listdir(DATA_DIR_PATH) # tsla = ["TSLA"]
    all_stocks = [stock for stock in all_stocks if not stock.startswith('.')]

    for stock in all_stocks:
        print(f"Computing data for {stock}")
        STOCK_PATH = f"{DATA_DIR_PATH}/{stock}/{YEAR}"
        SAVE_PATH = os.path.join(ROOT_DIR, 'data', 'market_orders') # limit_orders
        all_files = os.listdir(STOCK_PATH)

        daily_datas = []
        for filename in all_files:
            date = filename[7:][:10]
            print(f"Date: {date}")
            full_filename = f"{STOCK_PATH}/{filename}"

            # MO
            MO_orderbook_states = get_daily_trades_data(full_filename, date)
            daily_datas.append(MO_orderbook_states)

            # LO
            # LO_orderbook_states = get_daily_trades_data(full_filename, date)
            # daily_datas.append(LO_orderbook_states)

        data_all = pd.concat(daily_datas)
        print("Merged data")
        data_all = data_all.sort_values('event_timestamp')
        data_all.to_csv(f"{SAVE_PATH}/{stock}-{YEAR}-NEW.csv")
        print("Saved data")

