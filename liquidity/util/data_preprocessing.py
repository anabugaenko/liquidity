import pandas as pd
import os

from liquidity.response_functions.features import compute_orderbook_states

from liquidity.util.trades_data_util import get_trades_data
from liquidity.util.limit_orders_data_util import get_lo_data, get_ca_data, get_qa_data


DATA_DIR_PATH = '/Users/ana_bugaenko/data/BMLL/NASDAQ'
YEAR = 2016


def get_daily_trades_data(file_name, date: str = "2016-09-30"):
    raw_trades = get_trades_data(file_name, date)
    lob_states = compute_orderbook_states(raw_trades)
    return lob_states

def get_daily_orderbook_data(file_name, date: str = "2016-09-30", order_type: str="LO"):

    orderbook_types = ['LO', 'CA', 'QA']

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
    all_stocks = os.listdir(DATA_DIR_PATH)
    all_stocks = [stock for stock in all_stocks if not stock.startswith('.')]

    for stock in all_stocks:
        print(f"Computing data for {stock}")
        STOCK_PATH = f"{DATA_DIR_PATH}/{stock}/{YEAR}"
        all_files = os.listdir(STOCK_PATH)

        daily_datas = []
        for filename in all_files:
            date = filename[7:][:10]
            print(f"Date: {date}")
            full_filename = f"{STOCK_PATH}/{filename}"
            daily_orderbook_states = get_daily_trades_data(full_filename, date)
            daily_datas.append(daily_orderbook_states)

        data_all = pd.concat(daily_datas)
        print("Merged data")
        data_all = data_all.sort_values('event_timestamp')
        data_all.to_csv(f"{stock}-{YEAR}")
        print("Saved data")