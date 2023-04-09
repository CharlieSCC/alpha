import os
import sys
import math
import torch
import pickle
import warnings
import torch.multiprocessing
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils import data
from pandas.core.frame import DataFrame
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.optim.lr_scheduler import StepLR


warnings.filterwarnings("ignore")
t_float = torch.float64
torch.multiprocessing.set_sharing_strategy('file_system')

class Args:
    def __init__(self, gpu=0, subtask="regression"):
        # device
        self.gpu = str(gpu)
        self.device = 'cuda:0'
        # data settings
        adj_threshold = 0.1
        self.adj_str = str(int(100*adj_threshold))
        self.pos_adj_dir = "pos_adj_" + self.adj_str
        self.neg_adj_dir = "neg_adj_" + self.adj_str
        self.feat_dir = "features"
        self.label_dir = "label"
        self.mask_dir = "mask"
        self.data_start = data_start
        self.data_middle = data_middle
        self.data_end = data_end
        self.pre_data = pre_data
        # epoch settings
        self.max_epochs = 50
        self.epochs_eval = 10
        # learning rate settings
        self.lr = 0.0003
        self.gamma = 0.3
        # model settings
        self.hidden_dim = 256
        self.num_heads = 8
        self.out_features = 32
        self.model_name = "StockHeteGAT"
        self.batch_size = 1024
        self.loss_fcn = mse_loss
        # save model settings
        self.save_path = os.path.join(os.path.abspath('../../../gnn_code'), "model_saved/")
        self.load_path = self.save_path
        self.save_name = self.model_name + "_hidden_" + str(self.hidden_dim) + "_head_" + str(self.num_heads) + \
                         "_outfeat_" + str(self.out_features) + "_batchsize_" + str(self.batch_size) + "_adjth_" + \
                         str(self.adj_str)
        self.epochs_save_by = 50
        self.sub_task = subtask
        eval("self.{}".format(self.sub_task))()

    def regression(self):
        self.save_name = self.save_name + "_reg_rank_"
        self.loss_fcn = mse_loss
        self.label_dir = self.label_dir + "_regression"
        self.mask_dir = self.mask_dir + "_regression"

    def regression_binary(self):
        self.save_name = self.save_name + "_reg_binary_"
        self.loss_fcn = mse_loss
        self.label_dir = self.label_dir + "_twoclass"
        self.mask_dir = self.mask_dir + "_twoclass"

    def classification_binary(self):
        self.save_name = self.save_name + "_clas_binary_"
        self.loss_fcn = bce_loss
        self.label_dir = self.label_dir + "_twoclass"
        self.mask_dir = self.mask_dir + "_twoclass"

    def classification_tertiary(self):
        self.save_name = self.save_name + "_clas_tertiary_"
        self.loss_fcn = bce_loss
        self.label_dir = self.label_dir + "_threeclass"
        self.mask_dir = self.mask_dir + "_threeclass"




def mse_loss(logits, targets):
    mse = nn.MSELoss()
    loss = mse(logits.squeeze(), targets)
    return loss


def bce_loss(logits, targets):
    bce = nn.BCELoss()
    loss = bce(logits.squeeze(), targets)
    return loss


def evaluate(model, features, adj_pos, adj_neg, labels, mask, loss_func=nn.MSELoss()):
    model.eval()
    with torch.no_grad():
        logits = model(features, adj_pos, adj_neg)
    loss = loss_func(logits[mask], labels[mask])
    return loss, logits


def extract_data(data_dict, device):
    pos_adj = data_dict['pos_adj'].to(device).squeeze()
    neg_adj = data_dict['neg_adj'].to(device).squeeze()
    features = data_dict['features'].to(device).squeeze()
    labels = data_dict['labels'].to(device).squeeze()
    mask = data_dict['mask']
    return pos_adj, neg_adj, features, labels, mask


def train_epoch(epoch, args, model, dataset_train, optimizer, scheduler, loss_fcn):
    model.train()
    loss_return = 0
    for batch_data in dataset_train:
        for batch_idx, data in enumerate(batch_data):
            model.zero_grad()
            pos_adj, neg_adj, features, labels, mask = extract_data(data, args.device)
            logits = model(features, pos_adj, neg_adj)
            loss = loss_fcn(logits[mask], labels[mask])
            loss.backward()
            optimizer.step()
            scheduler.step()
            if batch_idx == 0:
                loss_return += loss.data
    return loss_return/len(dataset_train)


def eval_epoch(args, model, dataset_eval, loss_fcn):
    loss = 0.
    logits = None
    for batch_idx, data in enumerate(dataset_eval):
        pos_adj, neg_adj, features, labels, mask = extract_data(data, args.device)
        loss, logits = evaluate(model, features, pos_adj, neg_adj, labels, mask, loss_func=loss_fcn)
        break
    return loss, logits


def fun_train_predict(data_start, data_middle, data_end, pre_data):
    args = Args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    dataset = AllGraphDataSampler(base_dir="data_train_predict/", data_start=data_start,
                                  data_middle=data_middle, data_end=data_end)
    val_dataset = AllGraphDataSampler(base_dir="data_train_predict/", mode="val", data_start=data_start,
                                      data_middle=data_middle, data_end=data_end)
    dataset_loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, collate_fn=lambda x: x)
    val_dataset_loader = DataLoader(val_dataset, batch_size=1, pin_memory=True)
    model = eval(args.model_name)(hidden_dim=args.hidden_dim, num_heads=args.num_heads,
                                  out_features=args.out_features).to(args.device)

    # train
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    cold_scheduler = StepLR(optimizer=optimizer, step_size=5000, gamma=0.9, last_epoch=-1)
    default_scheduler = cold_scheduler
    print('start training')
    for epoch in range(args.max_epochs):
        train_loss = train_epoch(epoch=epoch, args=args, model=model, dataset_train=dataset_loader,
                                 optimizer=optimizer, scheduler=default_scheduler, loss_fcn=mse_loss)
        if epoch % args.epochs_eval == 0:
            eval_loss, _ = eval_epoch(args=args, model=model, dataset_eval=val_dataset_loader, loss_fcn=mse_loss)
            print('Epoch: {}/{}, train loss: {:.6f}, val loss: {:.6f}'.format(epoch + 1, args.max_epochs, train_loss,
                                                                              eval_loss))
        else:
            print('Epoch: {}/{}, train loss: {:.6f}'.format(epoch + 1, args.max_epochs, train_loss))
        if (epoch + 1) % args.epochs_save_by == 0:
            print("save model!")
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}
            torch.save(state, os.path.join(args.save_path, pre_data + "_epoch_" + str(epoch + 1) + ".dat"))

    # predict
    checkpoint = torch.load(os.path.join(args.load_path, pre_data + "_epoch_" + str(epoch + 1) + ".dat"))
    model.load_state_dict(checkpoint['model'])
    data_kdcode = os.listdir('kdcode')
    data_kdcode = sorted(data_kdcode)
    data_kdcode_last = data_kdcode[data_middle:data_end]
    for i in tqdm(range(len(val_dataset))):
        df = pd.read_csv('kdcode/' + data_kdcode_last[i], dtype=object)
        tmp_data = val_dataset[i]
        pos_adj, neg_adj, features, labels, mask = extract_data(tmp_data, args.device)
        model.train()
        logits = model(features, pos_adj, neg_adj)
        result = logits.data.cpu().numpy().tolist()
        result_new = []
        for j in range(len(result)):
            result_new.append(result[j][0])
        res = {"score": result_new}
        res = DataFrame(res)
        df['score'] = res
        df.to_csv('prediction/' + data_kdcode_last[i], encoding='utf-8-sig', index=False)

data_start = 0
data_middle = 10
data_end = data_middle+5
pre_data = '2019-12-31'
fun_train_predict(data_start, data_middle, data_end, pre_data)