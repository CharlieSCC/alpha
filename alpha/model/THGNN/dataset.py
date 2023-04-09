import os
import pandas as pd
from torch.utils.data import Dataset
from alpha.config.config import *


class AllGraphDataSampler(Dataset):
    def __init__(self,
                 srt_date,
                 end_date,
                 look_back_window,
                 factor_list
                 ):
        self.srt_date = srt_date
        self.end_date = end_date
        self.trade_dates = self.get_trade_dates(srt_date, end_date)
        self.look_back_window = look_back_window
        self.factor_list = factor_list
        self.factor_data = self._load_data()

    @staticmethod
    def get_trade_dates(srt_date, end_date):
        trade_dates = pd.read_hadf(os.path.join(DATA_PATH, "Ashare_data/basic_data/trade_dates.h5"), key="trade_dates")
        return trade_dates[(trade_dates >= srt_date) & (trade_dates <= end_date)]

    def __len__(self):
        return len(self.trade_dates)

    def _load_data(self):

        for factor in self.factor_list:
            df = pd.read_hadf(os.path.join(DATA_PATH, "Ashare_data/factor_data/{}.h5".format(factor)), key="v")
            srt_idx = self.trade_dates.index[0] - self.look_back_window + 1
            end_idx = self.trade_dates.index[-1] + 1
            df = df.iloc[srt_idx:end_idx, :]

    def __getitem__(self, idx):
        data = self.factor_data[idx - self.look_back_window + 1: idx + 1]
        graph =
        date = self.trade_dates[idx]
        stock_list =
        is_universe =
        is_nan =
        return {
            "data": data,
            "graph": graph,
            "date": date,
            "stock_list": stock_list,
            "is_universe": is_universe,
            "is_nan": is_nan,

        }