import pandas as pd


def get_binned_data_for_plotting(data_all, durations, column='x_scaled', q_num=31):
    data_binned = pd.DataFrame()

    for T in durations:
        temp = data_all[data_all['T'] == T]

        qmax = temp[[column]].quantile(0.99)[0]
        temp = temp[(temp[column] > -qmax) & (temp[column] < qmax)]

        temp['bin'] = pd.qcut(temp[column], q_num)

        temp = temp.groupby(['bin']).mean().reset_index()
        data_binned = data_binned.append(temp.iloc[:, 1:], ignore_index=True)

    return data_binned
