import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from alpha.config.config import *


class GraphDataset(Dataset):
    def __init__(self,
                 srt_date,
                 end_date,
                 label_df,
                 look_back_window,
                 factor_list,
                 universe_version,
                 ):
        self.srt_date = srt_date
        self.end_date = end_date
        self.trade_dates = self.get_trade_dates(srt_date, end_date)
        self.label_df = label_df
        self.look_back_window = look_back_window
        self.factor_list = factor_list
        self.factor_data = self._load_data()
        self.universe_df = pd.read_hdf(os.path.join(DATA_PATH, "Ashare_data/basic_data/{}.h5".format(universe_version)),
                                       key="{}".format(universe_version))

    @staticmethod
    def get_trade_dates(srt_date, end_date):
        trade_dates = pd.read_hdf(os.path.join(DATA_PATH, "Ashare_data/basic_data/trade_dates.h5"), key="trade_dates")
        return trade_dates[(trade_dates >= srt_date) & (trade_dates <= end_date)]

    def __len__(self):
        return len(self.trade_dates)

    def _load_data(self):
        arr_list = []
        for factor in self.factor_list:
            # df = pd.read_hdf(os.path.join(DATA_PATH, "Ashare_data/factor_data/{}.h5".format(factor)), key="v")
            df = pd.read_hdf(os.path.join(DATA_PATH, "Ashare_data/factor_data/{}.h5".format(factor)), key=factor[-9:])
            df = df.fillna(method="ffill")
            # srt_idx = self.trade_dates.index[0] - self.look_back_window + 1
            # end_idx = self.trade_dates.index[-1] + 1
            for i in range(len(df.index)):
                if df.index[i] == self.trade_dates.iloc[0]:
                    srt_idx = i - self.look_back_window + 1
                elif df.index[i] == self.trade_dates.iloc[-1]:
                    end_idx = i + 1
                    break

            df = df.iloc[srt_idx:end_idx, :]
            arr_list.append(df.values)
        return np.stack(arr_list, axis=-1).transpose((1, 0, 2))

    def __getitem__(self, idx):
        data = self.factor_data[:, idx: idx + self.look_back_window, :]
        date = self.trade_dates.iloc[idx]
        graph = pd.read_hdf(
            os.path.join(DATA_PATH, "Ashare_data/graph_data/adjacent_matrix_{}.h5".format(str(date[:4]))), key="graph")
        stock_id = pd.read_hdf(os.path.join(DATA_PATH, "Ashare_data/basic_data/stock_id.h5"), key="stock_id")
        label = self.label_df.loc[date]

        is_universe = self.universe_df.loc[date].values
        is_nan_x = ~np.isnan(data).sum(-1).sum(-1).astype(bool)
        is_nan_y = ~np.isnan(label.values).astype(bool)
        mask = is_universe & is_nan_x & is_nan_y

        data = data[mask]
        stock_id = list(stock_id[mask])
        graph = graph.loc[stock_id, stock_id].replace(np.nan, 0).values
        label = label[mask].values
        return [data, label, graph, [date for _ in range(len(label))], stock_id]
        # return {
        #     "data": data,
        #     "label": label,
        #     "graph": graph,
        #     "date": [date for _ in range(len(label))],
        #     "stock_id": stock_id,
        # }
