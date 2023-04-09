import os
from tqdm import tqdm
import pandas as pd
import torch
import logging
import copy
import pickle
from torch.utils.data import DataLoader
from alpha.model.THGNN.dataset import *
from alpha.model.THGNN.model import *


class THGNN_scheduler:
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
                 num_heads,
                 out_features,
                 num_layers,
                 lr,
                 weight_decay,
                 epochs,
                 max_patience
                 ):
        super(THGNN_scheduler).__init__()
        self.name = name

        if not os.path.exists(os.path.join(DATA_PATH, name)):
            os.makedirs(os.path.join(DATA_PATH, name))
        self.logger = logging.getLogger(os.path.join(DATA_PATH, name, "task.log"))
        self.train_len = train_len
        self.valid_len = valid_len
        self.look_back_window = look_back_window
        self.label_df = label_df
        self.factor_list = factor_list
        self.universe_version = universe_version
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.out_features = out_features
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

        train_dataset = GraphDataset(train_srt_date,
                                     train_end_date,
                                     self.label_df,
                                     self.look_back_window,
                                     self.factor_list,
                                     self.universe_version)
        valid_dataset = GraphDataset(valid_srt_date,
                                     valid_end_date,
                                     self.label_df,
                                     self.look_back_window,
                                     self.factor_list,
                                     self.universe_version)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        model = THGNN(len(self.factor_list), self.hidden_size, self.mum_layers, self.out_features, self.num_heads)
        if self.is_gpu:
            model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_model = copy.deepcopy(model)
        best_metric = -np.inf
        num_patience = 0
        for i in range(self.epochs):
            train_loss, train_metric = self.train_epoch(model, train_dataloader, optimizer, "train")
            self.logger.info("LOSS {} | METRIC {.3f}".format(train_loss, train_metric))
            valid_loss, valid_metric = self.train_epoch(model, valid_dataloader, optimizer, "valid")
            self.logger.info("LOSS {} | METRIC {.3f}".format(valid_loss, valid_metric))
            if best_metric < valid_metric:
                num_patience = 0
                best_metric = valid_metric
                best_model = copy.deepcopy(model)
                self.logger.info("EPOCH {}: BEST METRIC {.3f}".format(i, valid_metric))
            else:
                num_patience += 1
                self.logger.info("EPOCH {}: NUM PATIENCE {.3f}".format(i, num_patience))
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
        for x, y, graph, date, stock_id in tqdm(loader):
            y = (y.squeeze() - torch.mean(y.squeeze()))/torch.std(y.squeeze())
            upstream = copy.deepcopy(graph.squeeze())
            upstream[upstream <= 0] = 0
            upstream[upstream > 0] = 1
            downstream = copy.deepcopy(graph.squeeze())
            downstream[downstream >= 0] = 0
            downstream[downstream < 0] = 1
            if self.is_gpu:
                x = x.squeeze().cuda()
                y = y.squeeze().cuda()
                upstream = upstream.cuda()
                downstream = downstream.cuda()
            y_pred, _ = model(x.float(), upstream.float(), downstream.float(), True)
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


if __name__ == "__main__":
    opn = pd.read_hdf(os.path.join(DATA_PATH, "Ashare_data/1day_data/pv.h5"), key="open")
    opn_r = opn.pct_change()
    opn_r = opn_r.shift(-2)
    thgnn = THGNN_scheduler(
                name="THGNN_0.0.1",
                train_len=60,
                valid_len=30,
                look_back_window=5,
                factor_list=["alphas_101_alpha_001", "alphas_101_alpha_003"],
                universe_version="zz800",
                label_df=opn_r,
                batch_size=1,
                hidden_size=8,
                num_heads=4,
                out_features=8,
                num_layers=1,
                lr=0.001,
                weight_decay=0.0001,
                epochs=20,
                max_patience=5)
    thgnn.train("20200101", "20201221")







