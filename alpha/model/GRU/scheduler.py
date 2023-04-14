import os
from tqdm import tqdm
import pandas as pd
import torch
import logging
import copy
import pickle
from torch.utils.data import DataLoader
from alpha.model.GRU.dataset import *
from alpha.model.GRU.model import *


def set_logger(logger):
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(filename=logger.name)

    logger.setLevel(logging.INFO)
    handler1.setLevel(logging.INFO)
    handler2.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)

    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


class gru_scheduler:
    def __init__(self,
                 name,
                 train_len,
                 valid_len,
                 look_back_window,
                 factor_list,
                 universe_version,
                 label_df,
                 batch_size,
                 hidden_size,
                 num_layers,
                 lr,
                 weight_decay,
                 epochs,
                 max_patience
                 ):
        super(gru_scheduler).__init__()
        self.name = name

        if not os.path.exists(os.path.join(DATA_PATH, name)):
            os.makedirs(os.path.join(DATA_PATH, name))
        self.logger = logging.getLogger(os.path.join(DATA_PATH, name, "task.log"))
        self.logger = set_logger(self.logger)
        self.train_len = train_len
        self.valid_len = valid_len
        self.look_back_window = look_back_window
        self.label_df = label_df
        self.factor_list = factor_list
        self.universe_version = universe_version
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.mum_layers = num_layers
        self.is_gpu = torch.cuda.is_available()
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.loss_fn = torch.nn.MSELoss()
        self.max_patience = max_patience

    @staticmethod
    def get_date(date, num):
        trade_dates = pd.read_hdf(os.path.join(DATA_PATH, "Ashare_data/basic_data/trade_dates.h5"), key="trade_dates")
        idx = max(trade_dates[trade_dates <= date].index)
        return trade_dates.iloc[idx-num]

    def train(self, srt_date, end_date):
        train_srt_date = self.get_date(srt_date, self.train_len)
        valid_srt_date = self.get_date(srt_date, self.valid_len)
        train_end_date = self.get_date(valid_srt_date, self.look_back_window)
        valid_end_date = self.get_date(srt_date, self.look_back_window)

        train_dataset = TabularDataset(train_srt_date,
                                       train_end_date,
                                       self.label_df,
                                       self.look_back_window,
                                       self.factor_list,
                                       self.universe_version)
        valid_dataset = TabularDataset(valid_srt_date,
                                       valid_end_date,
                                       self.label_df,
                                       self.look_back_window,
                                       self.factor_list,
                                       self.universe_version)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        model = gru(len(self.factor_list), self.hidden_size, self.mum_layers,)
        if self.is_gpu:
            model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_model = copy.deepcopy(model)
        best_metric = -np.inf
        num_patience = 0
        for i in range(self.epochs):
            train_loss, train_metric = self.train_epoch(model, train_dataloader, optimizer, "train")
            self.logger.info("EPOCH {}: LOSS {} | METRIC {:.3f}".format(i, train_loss, train_metric))
            valid_loss, valid_metric = self.train_epoch(model, valid_dataloader, optimizer, "valid")
            self.logger.info("EPOCH {}: LOSS {} | METRIC {:.3f}".format(i, valid_loss, valid_metric))
            if best_metric < valid_metric:
                num_patience = 0
                best_metric = valid_metric
                best_model = copy.deepcopy(model)
                self.logger.info("EPOCH {}: BEST METRIC {:.3f}".format(i, valid_metric))
            else:
                num_patience += 1
                self.logger.info("EPOCH {}: NUM PATIENCE {:.3f}".format(i, num_patience))
            if num_patience >= self.max_patience:
                break
        with open(os.path.join(DATA_PATH, self.name, "model_{}_{}").format(srt_date, end_date)) as f:
            pickle.dump(best_model, f)

    def train_epoch(self, model, loader, optimizer, mode):
        if mode == "train":
            model.train()
        elif mode == "valid":
            model.eval()
        total_loss = 0
        y_list = []
        y_pred_list = []
        stock_id_list = []
        date_list = []
        for x, y, date, stock_id in tqdm(loader):
            y = (y.squeeze() - torch.mean(y.squeeze()))/torch.std(y.squeeze())
            if self.is_gpu:
                x = x.squeeze().cuda()
                y = y.squeeze().cuda()
            y_pred, _ = model(x.float())
            loss = self.loss_fn(y.float(), y_pred)
            if mode == "train":
                loss.backward()
                optimizer.step()
            total_loss += loss.data
            y_list.extend(y.squeeze().detach().cpu().numpy().tolist())
            y_pred_list.extend(y_pred.squeeze().detach().cpu().numpy().tolist())
            stock_id_list.extend(stock_id)
            date_list.extend(date)
        info_df = pd.DataFrame({
            "date": date_list,
            "stock_id": stock_id_list,
            "y": y_list,
            "y_pred": y_pred_list
        })
        ic = info_df.groupby("date").apply(lambda dd: dd[["y", "y_pred"]].corr().loc["y", "y_pred"]).mean()
        return total_loss/len(loader), ic

    def predict(self, srt_date, end_date):
        with open(os.path.join(DATA_PATH, self.name, "model_{}_{}").format(srt_date, end_date)) as f:
            best_model = pickle.load(f)
        test_dataset = TabularDataset(srt_date,
                                      end_date,
                                      self.label_df,
                                      self.look_back_window,
                                      self.factor_list,
                                      self.universe_version)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        if self.is_gpu:
            best_model.cuda()
        best_model.eval()
        total_loss = 0
        y_list = []
        y_pred_list = []
        stock_id_list = []
        date_list = []
        for x, y, date, stock_id in tqdm(test_dataloader):
            y = (y.squeeze() - torch.mean(y.squeeze()))/torch.std(y.squeeze())
            if self.is_gpu:
                x = x.squeeze().cuda()
                y = y.squeeze().cuda()
            y_pred, _ = best_model(x.float())
            loss = self.loss_fn(y.float(), y_pred)
            total_loss += loss.data
            y_list.extend(y.squeeze().detach().cpu().numpy().tolist())
            y_pred_list.extend(y_pred.squeeze().detach().cpu().numpy().tolist())
            stock_id_list.extend(stock_id)
            date_list.extend(date)
        info_df = pd.DataFrame({
            "date": date_list,
            "stock_id": stock_id_list,
            "y": y_list,
            "y_pred": y_pred_list
        })
        ic = info_df.groupby("date").apply(lambda dd: dd[["y", "y_pred"]].corr().loc["y", "y_pred"]).mean()
        return total_loss/len(test_dataloader), ic, info_df


if __name__ == "__main__":
    opn = pd.read_hdf(os.path.join(DATA_PATH, "Ashare_data/1day_data/pv.h5"), key="open")
    opn_r = opn.pct_change()
    opn_r = opn_r.shift(-2)
    s = gru_scheduler(
                name="gru_0.0.1",
                train_len=252*5,
                valid_len=252,
                look_back_window=20,
                factor_list=["alphas_101_alpha_001", "alphas_101_alpha_003"],
                universe_version="zz800",
                label_df=opn_r,
                batch_size=1,
                hidden_size=8,
                num_layers=1,
                lr=0.001,
                weight_decay=0.0001,
                epochs=20,
                max_patience=5)
    s.train("20210101", "20211221")







