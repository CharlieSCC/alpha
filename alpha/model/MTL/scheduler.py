import os
from tqdm import tqdm
import pandas as pd
import torch
import logging
import copy
import pickle
from torch.utils.data import DataLoader
from alpha.model.MTL.dataset import *
from alpha.model.MTL.model import *


class ICLoss(nn.Module):
    def __init__(self):
        super(ICLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=0, eps=1e-6)

    def forward(self, y_pred, y_true):
        mean_y_pred = torch.mean(y_pred, dim=0)
        mean_y_true = torch.mean(y_true, dim=0)
        
        diff_y_pred = y_pred - mean_y_pred
        diff_y_true = y_true - mean_y_true
        
        loss = self.cosine_similarity(diff_y_pred, diff_y_true)

        return -loss


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


class mtl_scheduler:
    def __init__(self,
                 name,
                 train_len,
                 valid_len,
                 look_back_window,
                 future_window,
                 encoder_opt_dim,
                 classifier_opt_dim,
                 pre_rnn_dim,
                 hid_dim,
                 encoder_rnn_layers,
                 decoder_rnn_layers,
                 factor_list,
                 universe_version,
                 label_df,
                 batch_size,
                 num_layers,
                 lr,
                 weight_decay,
                 dropout,
                 epochs,
                 max_patience
                 ):
        super(mtl_scheduler).__init__()
        self.name = name

        if not os.path.exists(os.path.join(DATA_PATH, name)):
            os.makedirs(os.path.join(DATA_PATH, name))
        self.logger = logging.getLogger(os.path.join(DATA_PATH, name, "task.log"))
        self.logger = set_logger(self.logger)
        self.train_len = train_len
        self.valid_len = valid_len
        self.look_back_window = look_back_window
        self.future_window = future_window
        self.encoder_opt_dim,=encoder_opt_dim,
        self.classifier_opt_dim,=classifier_opt_dim,
        self.pre_rnn_dim,=pre_rnn_dim,
        self.hid_dim,=hid_dim,
        self.encoder_rnn_layers,=encoder_rnn_layers,
        self.decoder_rnn_layers,=decoder_rnn_layers,
        self.label_df = label_df
        self.factor_list = factor_list
        self.universe_version = universe_version
        self.batch_size = batch_size
        self.mum_layers = num_layers
        self.is_gpu = torch.cuda.is_available()
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.epochs = epochs
        self.loss_fn1 = torch.nn.MSELoss()
        self.loss_fn2 = torch.nn.CrossEntropyLoss()
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
                                       self.future_window,
                                       self.factor_list,
                                       self.universe_version)
        valid_dataset = TabularDataset(valid_srt_date,
                                       valid_end_date,
                                       self.label_df,
                                       self.look_back_window,
                                       self.future_window,
                                       self.factor_list,
                                       self.universe_version)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
  
        encoder = Encoder(ipt_dim = len(self.factor_list),opt_dim = self.encoder_opt_dim,look_back_window = self.look_back_window, 
                          pre_rnn_dim = self.pre_rnn_dim, hid_dim = self.hid_dim, encoder_rnn_layers = self.encoder_rnn_layers,dropout= self.dropout)
        decoder=  Decoder(ipt_dim = len(self.factor_list),opt_dim = self.encoder_opt_dim,hid_dim = self.hid_dim, 
                          decoder_rnn_layers = self.decoder_rnn_layers ,dropout= self.dropout)
        classifier = Classifier(ipt_dim = self.hid_dim,opt_dim = self.classifier_opt_dim,hid_dim = self.hid_dim)

        model = StockNet(encoder,decoder,classifier,future_window=self.future_window,is_gpu = self.is_gpu)

        if self.is_gpu:
            model.cuda()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        best_model = copy.deepcopy(model)
        best_metric = -np.inf
        num_patience = 0
        for i in range(self.epochs):
            train_loss, train_metric = self.train_epoch(model, train_dataloader, optimizer, "train",i)
            self.logger.info("EPOCH {}: LOSS {:.6f} | METRIC {:.3f}".format(i, train_loss, train_metric))
            valid_loss, valid_metric = self.train_epoch(model, valid_dataloader, optimizer, "valid",i)
            self.logger.info("EPOCH {}: LOSS {:.6f} | METRIC {:.3f}".format(i, valid_loss, valid_metric))
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
        if not os.path.exists(os.path.join(DATA_PATH, self.name,)):
            os.makedirs(os.path.join(DATA_PATH, self.name,))
        with open(os.path.join(DATA_PATH, self.name, "model_{}_{}.pkl").format(srt_date, end_date),'wb') as f:
            pickle.dump(best_model, f)

    def train_epoch(self, model, loader, optimizer, mode,epoch):
        if mode == "train":
            model.train()
            if epoch < 8:
                for p in model.decoder.parameters():
                    p.requires_grad=False 

                for p in model.classifier.parameters():
                    p.requires_grad=False 

                for m in model.decoder.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

                for m in model.classifier.modules():
                    if isinstance(m, nn.BatchNorm1d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
            elif epoch< 15:
                for p in model.encoder.parameters():
                    p.requires_grad=False 

                for p in model.classifier.parameters():
                    p.requires_grad=False 

                for m in model.encoder.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

                for m in model.classifier.modules():
                    if isinstance(m, nn.BatchNorm1d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
            elif epoch < 20: 
                for p in model.encoder.parameters():
                    p.requires_grad=False 

                for p in model.decoder.parameters():
                    p.requires_grad=False 

                for m in model.encoder.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        
            if epoch == 8 or epoch == 15 or epoch == 20 :
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.lr

        elif mode == "valid":
            model.eval()
        total_loss = 0
        y_list = []
        y_pred_list = []
        stock_id_list = []
        date_list = []
        for x, y,f_y, date, stock_id in tqdm(loader):
            x = (x.squeeze() - torch.mean(x.squeeze(), dim=0, keepdim=True)) / (torch.std(x.squeeze(), dim=0, keepdim=True) + 1e-6)
            y = (y.squeeze() - torch.mean(y.squeeze(), dim=0, keepdim=True)) / (torch.std(y.squeeze(), dim=0, keepdim=True) + 1e-6)
            f_y = (f_y.squeeze() - torch.mean(f_y.squeeze(), dim=0, keepdim=True)) / (torch.std(f_y.squeeze(), dim=0, keepdim=True) + 1e-6)
            
            x = x.squeeze()
            y = y.squeeze()
            f_y = f_y.squeeze()

            past_ret = y.float()[:,-1]
            cl = torch.ones(past_ret.shape)
            cl[np.where(past_ret<=np.percentile(past_ret,30))] = 0
            cl[np.where(past_ret>=np.percentile(past_ret,60))] = 2

            if self.is_gpu:
                x = x.cuda()
                y = y.cuda()
                f_y = f_y.cuda()
                cl = cl.cuda()

            y_pred = model(x.float())
            lp = self.loss_fn1(y_pred['past_ret'][:,-1],y.float()[:,-1])
            lf = self.loss_fn1(y_pred['future_ret'],f_y.float())
            lc = self.loss_fn2(y_pred['pred_cl'],cl.long())
            if(epoch < 8):
                loss = lp
            elif(epoch < 15):
                loss = lf
            elif(epoch < 20):
                loss = lc
            else:
                loss = lp+lf+lc

            if mode == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.data
            y_list.extend(y[:,-1].squeeze().detach().cpu().numpy().tolist())
            y_pred_list.extend(y_pred['past_ret'][:,-1].squeeze().detach().cpu().numpy().tolist())
            stock_id_list.extend(stock_id)
            date_list.extend(date)

        for p in model.parameters():
            p.requires_grad=True

        info_df = pd.DataFrame({
            "date": date_list,
            "stock_id": stock_id_list,
            "y": y_list,
            "y_pred": y_pred_list
        })
        # def evaluation(target, predict):
        #     predict = np.squeeze(predict)
        #     target = np.squeeze(target)
            
        #     gap = (predict - target)**2
        #     numerator = np.sum(gap)
        #     denominator = np.sum((target)**2)
        #     return (1-(numerator/denominator))*100
        info_df["date"] = info_df["date"].astype(str).str[2:-3]
        info_df["stock_id"] = info_df["stock_id"].astype(str).str[2:-3]
        ic = info_df.groupby("date").apply(lambda dd: dd[["y", "y_pred"]].corr().loc["y", "y_pred"]).mean()
        # r2 = info_df.groupby("date").apply(lambda dd: evaluation(dd["y"], dd["y_pred"])).mean()
        return total_loss/len(loader), ic

    def predict(self, srt_date, end_date):
        with open(os.path.join(DATA_PATH, self.name, "model_{}_{}.pkl").format(srt_date, end_date),'rb') as f:
            best_model = pickle.load(f)
        test_dataset = TabularDataset(srt_date,
                                      end_date,
                                      self.label_df,
                                      self.look_back_window,
                                      self.future_window,
                                      self.factor_list,
                                      self.universe_version)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        if self.is_gpu:
            best_model.cuda()
        best_model.eval()
        total_loss = 0
        ret_list = []
        y_list = []
        y_pred_list = []
        stock_id_list = []
        date_list = []
        for x, y, date, stock_id in tqdm(test_dataloader):
            if x.shape[1] == 0:
                continue
            x = (x.squeeze() - torch.mean(x.squeeze(), dim=0, keepdim=True)) / (torch.std(x.squeeze(), dim=0, keepdim=True) + 1e-6)
            y_ = (y.squeeze() - torch.mean(y.squeeze(), dim=0, keepdim=True)) / (torch.std(y.squeeze(), dim=0, keepdim=True) + 1e-6)
            if self.is_gpu:
                x = x.squeeze().cuda()
                y_ = y_.squeeze().cuda()
            y_pred = best_model(x.float(),y_.float()[:,:-1])
            loss = self.loss_fn(y_.float()[:,-1], y_pred)
            total_loss += loss.data
            ret_list.extend(y.squeeze().cpu().numpy()[:,-1].tolist())
            y_list.extend(y_.squeeze().detach().cpu().numpy()[:,-1].tolist())
            y_pred_list.extend(y_pred.squeeze().detach().cpu().numpy().tolist())
            stock_id_list.extend(stock_id)
            date_list.extend(date)

        info_df = pd.DataFrame({
            "date": date_list,
            "stock_id": stock_id_list,
            "y": y_list,
            "y_pred": y_pred_list,
            "ret": ret_list
        })
        info_df["date"] = info_df["date"].astype(str).str[2:-3]
        info_df["stock_id"] = info_df["stock_id"].astype(str).str[2:-3]
        ic = info_df.groupby("date").apply(lambda dd: dd[["y", "y_pred"]].corr().loc["y", "y_pred"]).mean()
        info_df.to_csv(os.path.join(DATA_PATH, self.name, "info_{}_{}.csv").format(srt_date, end_date))
        return total_loss/len(test_dataloader), ic, info_df



if __name__ == "__main__":

    # filenames=os.listdir(os.path.join(DATA_PATH, "Ashare_data/factor_data"))
    # factor_list = [i[:-3] for i in filenames]
    opn = pd.read_hdf(os.path.join(DATA_PATH, "Ashare_data/1day_data/pv.h5"), key="open")
    opn_r = opn.pct_change()
    opn_r = opn_r.shift(-2)
    s = mtl_scheduler(
                name="mtl_0.0.1",
                train_len=252*5,
                valid_len=252,
                look_back_window=20,
                future_window = 3,
                encoder_opt_dim = 1,
                classifier_opt_dim = 3,
                pre_rnn_dim = 16,
                hid_dim = 16,
                encoder_rnn_layers=2,
                decoder_rnn_layers=2,
                factor_list=["alphas_101_alpha_001", "alphas_101_alpha_003"],
                universe_version="zz800",
                label_df=opn_r,
                batch_size=1,
                num_layers=1,
                lr=0.001,
                weight_decay=0.0001,
                dropout = 0.1,
                epochs=50,
                max_patience=5)
    s.train("20210101", "20211221")

    # print(s.predict("20210101", "20211221"))







